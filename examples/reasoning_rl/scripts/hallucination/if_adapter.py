# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Stage-4 instruction-following adapter (D31).

The hallucination stages (2/3) train abstention and calibration; stage 4 adds an
*instruction-following* domain so the calibrated policy does not lose the ability
to follow explicit formatting / content constraints.  The source is the same one
``scripts/to_parquet_if.py`` was written for -- ``nvidia/Nemotron-RL-instruction_following``
(WildChat prompts + Open-Instruct / IFBench verifiable constraints, up to 5 per
row) -- but this adapter is built for the **stage-4 mix**, not the stage-1 one:

* rows are emitted in the stage-1 parquet schema (``ability="if"``,
  ``domain="if"``, ``data_source="if_nemotron"``) so ``mix_halluc.py`` can mix
  them with the hallucination domain and the old-domain replay without a schema
  cast;
* the ground truth is ``{"constraints": [...], "grading_mode": "binary"|"fraction"}``.
  ``binary`` is the NeMo-Gym / IFBench convention (all constraints must pass) and
  is the default; ``fraction`` is the optional graded mode implemented in
  ``reward/compute_score.py::_if_score`` (passed / total) for experiments where
  the all-or-nothing signal is too sparse;
* rows carrying an instruction id the reward cannot evaluate are **dropped**, not
  emitted with a permanently-zero reward (``if_verifier.SUPPORTED_INSTRUCTION_IDS``
  now covers the whole published registry, so this is an assertion rather than a
  filter in practice -- the funnel reports it);
* ``--exclude-path`` drops prompts that already appeared in the previously
  trained artifacts (stage 1 / 2 / 3), which is how a rebuilt stage 4 avoids
  re-training the same WildChat request;
* the pool is deterministic for a given ``--seed`` and ``--quota``: exact dedup
  then a seeded shuffle then the cap.

Why no prompt template is appended
----------------------------------
The task *is* the user request: nothing the model needs is missing, and adding a
"think block" instruction would change the user's own formatting constraints
(e.g. ``length_constraints:number_words`` counts the final response, but a
template would also change what the user asked for).  The reward's format gate
still requires the Qwen3 thinking shape, which the chat template produces on its
own; constraints are checked on the visible (post-``</think>``) response only.

Usage::

    python if_adapter.py --out ~/data/reasoning_rl/halluc/built_stage4/if.parquet
    python if_adapter.py --if-path /data/nemotron_if/train.jsonl --quota 8000
    python if_adapter.py --exclude-path ~/data/reasoning_rl/final
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dedup import exact_dedup, minhash_near_dedup  # noqa: E402

import schema  # noqa: E402

_REASONING_RL_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REASONING_RL_DIR, "reward"))

IF_REPO = "nvidia/Nemotron-RL-instruction_following"
DATA_SOURCE = "if_nemotron"
ABILITY = "if"
DOMAIN = "if"
GRADING_MODES = ("binary", "fraction")

DEFAULT_RAW_DIR = os.path.expanduser("~/data/reasoning_rl/halluc/raw/if")
# Named DEFAULT_OUT like every other adapter, and named after *this* source.
# Stage 4 passes --out .../halluc/built_stage4/if.parquet; the default keeps the
# "every adapter writes into DEFAULT_ADAPTER_DIR" invariant of test_mix_halluc.
DEFAULT_OUT = os.path.expanduser("~/data/reasoning_rl/halluc/built/if.parquet")


def report_path_for(out: str) -> str:
    """The funnel report next to the artifact (``<out>_report.json``)."""
    return os.path.splitext(out)[0] + "_report.json"


# ---------------------------------------------------------------------------
# raw loading
# ---------------------------------------------------------------------------


def _load_rows(if_path: str | None, raw_dir: str) -> list[dict]:
    """Load the raw rows from an explicit path, the raw dir, or the HF repo."""
    import datasets

    path = if_path
    if path is None:
        for name in ("train.jsonl", "train.parquet", "train_if.parquet"):
            candidate = os.path.join(os.path.expanduser(raw_dir), name)
            if os.path.exists(candidate):
                path = candidate
                break

    if path is not None:
        path = os.path.expanduser(path)
        if os.path.isfile(path) and os.path.getsize(path) == 0:
            # An empty jsonl makes ``datasets`` raise StopIteration; report the
            # empty pool as an empty list so main() can fail cleanly.
            return []
        if path.endswith(".parquet"):
            ds = datasets.load_dataset("parquet", data_files=path, split="train")
        else:
            ds = datasets.load_dataset("json", data_files=path, split="train")
    else:
        ds = datasets.load_dataset(IF_REPO, split="train")
    return list(ds)


def _clean_prompt(raw) -> list[dict]:
    """Chat-message list with role/content strings, or [] when unusable."""
    if isinstance(raw, str):
        raw = [{"role": "user", "content": raw}]
    if not isinstance(raw, list):
        return []
    prompt = []
    for message in raw:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user")).strip() or "user"
        content = str(message.get("content", "")).strip()
        if content:
            prompt.append({"role": role, "content": content})
    return prompt


def _normalise_constraint(raw) -> dict | None:
    """Normalise one constraint entry to ``{id, kwargs}`` (None if malformed)."""
    if not isinstance(raw, dict):
        return None
    cid = raw.get("id") or raw.get("instruction_id")
    if not cid:
        return None
    kwargs = raw.get("kwargs")
    if kwargs is None:
        kwargs = {}
    if isinstance(kwargs, str):
        try:
            kwargs = json.loads(kwargs)
        except (json.JSONDecodeError, TypeError):
            kwargs = {}
    if not isinstance(kwargs, dict):
        kwargs = {}
    return {"id": str(cid), "kwargs": kwargs}


def extract_constraints(ex: dict) -> list[dict]:
    """Constraints from any of the source's three spellings.

    The repo tolerates (a) a list of constraint dicts, (b) parallel
    ``instruction_id_list`` + ``kwargs`` lists, and (c) ``kwargs`` stored as a
    JSON string; this is the same tolerance ``to_parquet_if.py`` implements, kept
    here so the two builders cannot drift.
    """
    raw_ids = ex.get("instruction_id_list")
    raw_kwargs = ex.get("kwargs")
    constraints: list[dict] = []
    if isinstance(raw_ids, list) and raw_ids:
        if isinstance(raw_ids[0], dict):
            constraints = [_normalise_constraint(item) for item in raw_ids]
            constraints = [item for item in constraints if item is not None]
        else:
            ids = [str(item) for item in raw_ids]
            kwargs_list = raw_kwargs if isinstance(raw_kwargs, list) else [None] * len(ids)
            if len(kwargs_list) < len(ids):
                kwargs_list.extend([None] * (len(ids) - len(kwargs_list)))
            for cid, kwargs in zip(ids, kwargs_list, strict=False):
                constraint = _normalise_constraint({"id": cid, "kwargs": kwargs})
                if constraint is not None:
                    constraints.append(constraint)
    return constraints


# ---------------------------------------------------------------------------
# dedup keys
# ---------------------------------------------------------------------------


def prompt_key(prompt: list[dict]) -> str:
    """Normalised text a row is deduplicated on.

    The whole conversation is joined (a WildChat prompt may carry a system turn)
    and whitespace/case-folded, mirroring ``mix_halluc._question_key`` so the two
    sides of the "was this already trained?" comparison agree.
    """
    text = " ".join(str(message.get("content", "")) for message in prompt or [])
    return " ".join(text.split()).casefold()


def read_excluded_keys(paths: list[str] | None) -> set[str]:
    """Prompt keys of the previously trained artifacts, for ``--exclude-path``."""
    if not paths:
        return set()
    from mix import read_parquet_rows

    keys: set[str] = set()
    for path in paths:
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            print(f"[if_adapter] WARNING: --exclude-path does not exist: {path}", file=sys.stderr)
            continue
        rows = read_parquet_rows(path)
        keys.update(prompt_key(row.get("prompt", [])) for row in rows)
        print(f"[if_adapter] excluded {len(rows)} previously trained rows from {path}")
    return keys


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def collect_candidates(
    raw_rows: list[dict],
    *,
    excluded: set[str] | None = None,
    minhash_threshold: float = 0.0,
) -> tuple[list[dict], dict]:
    """Normalise + filter the raw rows; returns ``(candidates, funnel)``.

    Filter order is exact-first (cheap) and every stage is counted, so a build
    that halves the pool names the stage that did it rather than just reporting a
    smaller number.
    """
    from if_verifier import unsupported_instruction_ids

    excluded = excluded or set()
    funnel: Counter[str] = Counter()
    candidates: list[dict] = []
    unsupported: Counter[str] = Counter()

    for idx, ex in enumerate(raw_rows):
        prompt = _clean_prompt(ex.get("prompt"))
        if not prompt:
            funnel["empty_prompt"] += 1
            continue
        constraints = extract_constraints(ex)
        if not constraints:
            funnel["no_constraints"] += 1
            continue
        missing = unsupported_instruction_ids(constraints)
        if missing:
            funnel["unsupported_instruction_id"] += 1
            for cid in missing:
                unsupported[cid] += 1
            continue
        task_id = str(ex.get("task_id") or ex.get("id") or f"if-{idx}")
        source = str(ex.get("source") or "nemotron_if")
        candidates.append(
            {
                "prompt": prompt,
                "constraints": constraints,
                "task_id": task_id,
                "source": source,
                "raw_index": idx,
            }
        )
        funnel["admissible"] += 1

    # Exact dedup on the prompt: one WildChat request is one training prompt even
    # when the pool carries several constraint draws for it (same convention as
    # scripts/to_parquet_if.py, so a stage-4 rebuild of the same raw file is
    # deterministic).
    kept = exact_dedup([prompt_key(c["prompt"]) for c in candidates])
    funnel["duplicate_prompt"] += len(candidates) - len(kept)
    candidates = [candidates[i] for i in kept]

    if excluded:
        before = len(candidates)
        candidates = [c for c in candidates if prompt_key(c["prompt"]) not in excluded]
        funnel["already_trained"] += before - len(candidates)

    if minhash_threshold and minhash_threshold > 0 and candidates:
        kept = minhash_near_dedup(
            [prompt_key(c["prompt"]) for c in candidates], threshold=minhash_threshold
        )
        funnel["near_duplicate_prompt"] += len(candidates) - len(kept)
        candidates = [candidates[i] for i in kept]

    funnel["candidates"] = len(candidates)
    details = {"unsupported_instruction_ids": dict(unsupported)}
    return candidates, {"funnel": dict(funnel), **details}


def build_rows(
    candidates: list[dict],
    *,
    quota: int | None = None,
    seed: int = 0,
    grading_mode: str = "binary",
) -> list[dict]:
    """Take ``quota`` candidates in a seeded order and render verl rows."""
    if quota is not None and quota < len(candidates):
        candidates = list(candidates)
        random.Random(seed).shuffle(candidates)
        candidates = candidates[:quota]
    rows = []
    for index, candidate in enumerate(candidates):
        rows.append(_build_row(candidate, index=index, seed=seed, grading_mode=grading_mode))
    return rows


def _build_row(candidate: dict, *, index: int, seed: int, grading_mode: str) -> dict:
    ground_truth = {
        "constraints": candidate["constraints"],
        "grading_mode": grading_mode,
    }
    info = dict(schema.EXTRA_INFO_DEFAULTS)
    info.update(
        {
            "split": "train",
            "index": index,
            "task_id": candidate["task_id"],
            "domain": DOMAIN,
            "source": candidate["source"],
            "difficulty": "",
            "prior_solve_rate": -1.0,
            "seed": seed,
        }
    )
    return {
        "data_source": DATA_SOURCE,
        "prompt": candidate["prompt"],
        "ability": ABILITY,
        "reward_model": {"style": "rule", "ground_truth": json.dumps(ground_truth, ensure_ascii=False)},
        "extra_info": info,
    }


def summarise(rows: list[dict], grading_mode: str) -> dict:
    """Per-id / per-count / per-source coverage of the written artifact."""
    by_id: Counter[str] = Counter()
    by_count: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    lengths = []
    for row in rows:
        payload = json.loads(row["reward_model"]["ground_truth"])
        constraints = payload.get("constraints", [])
        by_count[str(len(constraints))] += 1
        by_source[row["extra_info"]["source"]] += 1
        for constraint in constraints:
            by_id[str(constraint.get("id", ""))] += 1
        lengths.append(len(" ".join(message["content"] for message in row["prompt"])))
    lengths.sort()

    def percentile(fraction: float) -> int:
        return lengths[min(len(lengths) - 1, int(fraction * (len(lengths) - 1)))] if lengths else 0

    return {
        "grading_mode": grading_mode,
        "rows": len(rows),
        "constraints_per_row": dict(sorted(by_count.items())),
        "by_instruction_id": dict(sorted(by_id.items())),
        "by_source": dict(sorted(by_source.items())),
        "prompt_chars": {"p10": percentile(0.1), "p50": percentile(0.5), "p90": percentile(0.9), "max": percentile(1.0)},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR, help="local raw dir holding train.jsonl / train.parquet")
    parser.add_argument("--if-path", default=None, help="explicit jsonl/parquet override; else raw dir, else HF repo")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output parquet path")
    parser.add_argument("--quota", type=int, default=None, help="cap on emitted rows after dedup")
    parser.add_argument("--seed", type=int, default=44, help="shuffle seed for the quota draw")
    parser.add_argument(
        "--grading-mode",
        choices=GRADING_MODES,
        default="binary",
        help="binary = NeMo-Gym convention (all constraints must pass); fraction = passed / total",
    )
    parser.add_argument(
        "--exclude-path",
        action="append",
        default=None,
        help="previously trained parquet/file/dir; prompts seen there are dropped (repeatable)",
    )
    parser.add_argument(
        "--minhash-threshold",
        type=float,
        default=0.0,
        help="optional in-pool near-duplicate Jaccard threshold (0 = exact dedup only)",
    )
    args = parser.parse_args(argv)

    if not 0.0 <= args.minhash_threshold < 1.0:
        parser.error("--minhash-threshold must be in [0, 1)")

    raw_rows = _load_rows(args.if_path, args.raw_dir)
    print(f"[if_adapter] {len(raw_rows)} raw rows")

    excluded = read_excluded_keys(args.exclude_path)
    candidates, report = collect_candidates(
        raw_rows,
        excluded=excluded,
        minhash_threshold=args.minhash_threshold,
    )
    funnel = report["funnel"]
    print(f"[if_adapter] after filtering: {funnel}")

    rows = build_rows(candidates, quota=args.quota, seed=args.seed, grading_mode=args.grading_mode)
    if not rows:
        print("no rows survived filtering; nothing written", file=sys.stderr)
        return 1

    schema.normalise_extra_info(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    schema.write_rows_parquet(rows, args.out)

    summary = summarise(rows, args.grading_mode)
    report.update(
        {
            "config": {
                "if_path": args.if_path,
                "raw_dir": os.path.expanduser(args.raw_dir),
                "quota": args.quota,
                "seed": args.seed,
                "grading_mode": args.grading_mode,
                "minhash_threshold": args.minhash_threshold,
                "exclude_paths": args.exclude_path or [],
            },
            "summary": summary,
        }
    )
    report_path = report_path_for(args.out)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"rows written: {len(rows)} -> {args.out}")
    print(f"  constraints/row: {summary['constraints_per_row']}")
    print(f"  instruction ids: {len(summary['by_instruction_id'])} distinct")
    print(f"report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
