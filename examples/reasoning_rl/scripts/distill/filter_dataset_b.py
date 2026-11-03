#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Dataset-B post-filter: keep only rewrites whose final answer is unchanged and whose reward is still full.

``build_dataset_b.py`` checks both conditions while it writes a row, but its
generation-time check is weaker than the stored artifact needs:

* a rewrite may change or introduce a ``\\boxed{...}`` **inside** the think body.
  The generation check compares ``extract_final`` of the full response; when the
  tail carries a boxed answer (the normal contract) that is the last boxed, so a
  body answer can change while the check still passes.  Anyone reading the
  rewritten reasoning -- or a consumer that takes the first boxed -- sees a
  different final answer;
* a ``logic_*`` row is scored with ``extract_logic_answer``, which prefers
  ``<answer>`` tags anywhere in the response before the tail's ``\\boxed{}``; a
  body rewrite can therefore change the answer the reward reads while
  ``extract_final`` (boxed-only) sees no change;
* a rewrite may replace the ``</think>`` tail (the final-answer section) itself
  if the artifact was produced or edited by an older/mutated pipeline; the
  stored ``rewritten_score`` then describes a response that is no longer in the
  row;
* the stored ``rewritten_score`` can be stale even when the response was only
  edited after generation.

This script therefore re-derives every check from the stored row itself:

1. **label identity** -- the post-``</think>`` tail must be byte-identical to the
   stored ``original_response`` tail (disable with ``--allow_tail_change``), the
   per-domain extracted answer must be equal, and no new ``\\boxed{}`` value may
   appear in the think body (deleting duplicated ones is allowed);
2. **reward** -- the stored response is re-scored against its own ground truth
   with the same dispatcher the B builder uses, and must reach ``--min_score``;
3. **original sanity** -- the stored original response must be well formed; with
   ``--min_original_score`` its stored score must also reach that threshold, and
   with ``--check_original_reward`` it is re-scored as well;
4. **A cross-check** (optional ``--a``) -- ``sample_id`` must exist in A, the
   prompt, original response, ground truth and data source must agree.  This is
   exactly the join ``build_dpo_pairs.py`` performs, so a row that fails here
   would be dropped there anyway.

The reward is invoked exactly as ``build_dataset_b.py`` invokes it (no stage-4
length-penalty kwargs), so the re-score answers "would the B builder still keep
this row", not "would the stage-4 trainer score it".

Rows are written under ``--output_dir``::

    dataset_b_filtered.parquet   rows that passed every check (schema unchanged)
    dataset_b_rejected.parquet   rejected rows + b_filter_* audit columns
    b_filter_rejected.jsonl      compact per-row rejection report
    b_filter_stats.json          drop reasons, per-source counts, score/ratio stats

Usage::

    python examples/reasoning_rl/scripts/distill/filter_dataset_b.py \
        --b ~/data/reasoning_rl/dpo/stage_b/dataset_b.parquet \
        --a ~/data/reasoning_rl/dpo/stage_a/dataset_a.parquet \
        --output_dir ~/data/reasoning_rl/dpo/stage_b_filtered

Then rebuild the DPO pairs from the filtered artifact::

    python examples/reasoning_rl/scripts/distill/build_dpo_pairs.py \
        --a ~/data/reasoning_rl/dpo/stage_a/dataset_a.parquet \
        --b ~/data/reasoning_rl/dpo/stage_b_filtered/dataset_b_filtered.parquet \
        --output_dir ~/data/reasoning_rl/dpo

``--a`` is optional: every B row stores its own ``original_response``, which is
enough for the label checks.  Passing ``--a`` additionally proves that the B row
still belongs to the same A rollout (and makes A authoritative for the comparison
and for the reward's ground truth / ``extra_info``).

A rejection is fail-closed: a parser/reward crash, an unknown ``data_source``
that the reward refuses, or an unparseable ground truth drops the row instead of
letting it through.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from common import (
    ground_truth_of,
    load_reward,
    load_verifier_helpers,
    read_parquet_rows,
    score_response,
    split_think,
    summarise,
    write_parquet_rows,
)
from examples.reasoning_rl.reward.compute_score import _extract_all_boxed, extract_logic_answer

logger = logging.getLogger("filter_dataset_b")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_float(*values) -> float:
    """First numeric value, or -999.0 (the "not available" sentinel used in parquet)."""
    for value in values:
        number = _as_float(value)
        if number is not None:
            return number
    return -999.0


def _canonical_json(value) -> str:
    """Canonical form for comparing a prompt or a JSON ground truth."""
    try:
        return json.dumps(json.loads(value), ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return "" if value is None else str(value)


def _prompt_fingerprint(prompt) -> str:
    return json.dumps(prompt, ensure_ascii=False, sort_keys=True)


def _short(value, limit: int = 240) -> str:
    if isinstance(value, dict | list | tuple):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    text = " ".join(str("" if value is None else value).split())
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


def _row_info(row: dict) -> dict:
    """Audit fields shared by the kept/rejected/report paths."""
    stored = row.get("rewritten_score")
    if stored is None:
        stored = row.get("score")
    return {
        "sample_id": str(row.get("sample_id") or ""),
        "task_id": str(row.get("task_id") or ""),
        "data_source": str(row.get("data_source") or ""),
        "score": None,
        "original_score": None,
        "stored_score": _as_float(stored),
        "original_stored_score": _as_float(row.get("score")),
        "original_answer": "",
        "rewritten_answer": "",
        "tail_changed": False,
        "label_unverifiable": False,
        "compression_ratio": None,
    }


def _label_of(response: str, data_source: str, extract_final, extract_logic_answer) -> dict:
    """The answer label(s) a downstream consumer can read from this response.

    ``extract_final`` is the boxed/marker parser every hallucination row uses.
    ``logic_*`` additionally uses ``extract_logic_answer`` because the logic
    reward reads ``<answer>`` tags, ``Final Answer:`` lines and bare tails before
    it falls back to ``\\boxed{}`` -- a body rewrite can therefore change the
    logic label without touching the tail's boxed answer.  ``if_*`` / ``bbh_*``
    verifiers read the whole post-``</think>`` tail, so the tail itself is part of
    the label there.
    """
    status, answer, option = extract_final(response)
    label = {"boxed": [status, answer, option]}
    if str(data_source).startswith("logic"):
        try:
            label["logic"] = extract_logic_answer(response)
        except Exception as exc:  # noqa: BLE001 - a broken parser is a rejection, not a crash
            label["logic"] = f"<error:{type(exc).__name__}>"
    elif str(data_source).startswith(("if_", "bbh_")):
        # These verifiers read the post-</think> response itself, so the tail is
        # the label: with --allow_tail_change, a changed tail must still differ.
        parts = split_think(response)
        label["tail"] = parts[1] if parts is not None else None
    return label


def _body_answer_check(original_body: str, body: str) -> tuple[str, str]:
    """Reject a body that changes or introduces a ``\\boxed{}`` answer.

    Pure deletion is allowed (compression removes a duplicated conclusion).  Any
    boxed value that did not exist in the original body is a changed/added
    answer and is rejected -- including a changed *earlier* boxed, which the
    generation-time "last boxed of the full response" check cannot see.
    """
    try:
        before = _extract_all_boxed(original_body)
        after = _extract_all_boxed(body)
    except Exception:  # noqa: BLE001 - a malformed body cannot contribute an answer
        return "", ""
    if not after:
        return "", ""
    before_counts = Counter(item.strip() for item in before)
    after_counts = Counter(item.strip() for item in after)
    if not after_counts - before_counts:
        return "", ""
    reason = "body_answer_added" if not before else "body_answer_changed"
    return reason, f"body boxed {_short(sorted(before))} -> {_short(sorted(after))}"


# ---------------------------------------------------------------------------
# the per-row filter
# ---------------------------------------------------------------------------


def check_row(
    row: dict,
    *,
    reward_fn,
    format_ok,
    extract_final,
    extract_logic_answer,
    sandbox_url: str | None = None,
    min_score: float = 1.0,
    min_original_score: float | None = None,
    check_body_answer: bool = True,
    allow_tail_change: bool = False,
    allow_missing_original: bool = False,
    check_original_reward: bool = False,
    a_row: dict | None = None,
    require_a: bool = False,
) -> tuple[str, str, dict]:
    """Check one dataset-B row.

    Returns ``(reason, detail, info)``.  ``reason == ""`` means the row is kept;
    every other value is a rejection reason (also the stats key).
    """
    info = _row_info(row)
    sample_id = info["sample_id"]
    if not sample_id:
        return "missing_sample_id", "row has no sample_id (build_dpo_pairs cannot join it)", info

    response = row.get("response")
    if not isinstance(response, str) or not response.strip():
        return "empty_response", "response is empty or not a string", info
    if not format_ok(response):
        return "format", "response fails the format gate", info
    parts = split_think(response)
    if parts is None:
        return "think_block", "response has no single non-empty <think> block", info

    if require_a and a_row is None:
        return "missing_in_a", f"{sample_id} is not present in dataset A", info

    original = row.get("original_response")
    reward_row = row
    if a_row is not None:
        if _prompt_fingerprint(a_row.get("prompt")) != _prompt_fingerprint(row.get("prompt")):
            return "prompt_mismatch", "B prompt is not byte-identical to A prompt", info
        a_response = a_row.get("response")
        if not isinstance(a_response, str) or not a_response.strip():
            return "original_empty", "A response is empty", info
        b_original = row.get("original_response")
        if not isinstance(b_original, str) or not b_original.strip():
            # build_dpo_pairs joins on this field; a row without it cannot pair.
            return "missing_original", "B row has no original_response to pair with A", info
        if b_original != a_response:
            return "original_response_mismatch", "B original_response is not byte-identical to A response", info
        b_gt = ground_truth_of(row)
        a_gt = ground_truth_of(a_row)
        if b_gt and _canonical_json(a_gt) != _canonical_json(b_gt):
            return (
                "ground_truth_mismatch",
                f"A gt {_short(a_gt)} != B gt {_short(b_gt)}",
                info,
            )
        a_source = str(a_row.get("data_source") or "")
        if not info["data_source"]:
            # A B row produced by an older builder may omit the field; A owns it.
            info["data_source"] = a_source
        elif a_source != info["data_source"]:
            return (
                "data_source_mismatch",
                f"A {a_source!r} != B {info['data_source']!r}",
                info,
            )
        original = a_response
        reward_row = a_row  # authoritative ground truth / extra_info
        a_score = _as_float(a_row.get("score"))
        if a_score is not None:
            info["original_stored_score"] = a_score

    if not isinstance(original, str) or not original.strip():
        if not allow_missing_original:
            return "missing_original", "no original_response and no --a row to compare against", info
        info["label_unverifiable"] = True
        logger.debug("[filter] %s kept without label check (--allow_missing_original)", sample_id)
    else:
        if not format_ok(original):
            return "original_format", "original_response fails the format gate", info
        original_parts = split_think(original)
        if original_parts is None:
            return "original_think_block", "original_response has no single non-empty <think> block", info

        info["tail_changed"] = parts[1] != original_parts[1]
        original_label = _label_of(original, info["data_source"], extract_final, extract_logic_answer)
        rewritten_label = _label_of(response, info["data_source"], extract_final, extract_logic_answer)
        info["original_answer"] = _short(original_label)
        info["rewritten_answer"] = _short(rewritten_label)
        if info["tail_changed"] and not allow_tail_change:
            return (
                "tail_changed",
                f"tail {_short(original_parts[1])} -> {_short(parts[1])}",
                info,
            )
        if original_label != rewritten_label:
            return (
                "label_changed",
                f"{_short(original_label)} -> {_short(rewritten_label)}",
                info,
            )

        if check_body_answer:
            reason, detail = _body_answer_check(original_parts[0], parts[0])
            if reason:
                return reason, detail, info

        info["compression_ratio"] = len(response) / max(1, len(original))

    # The stored "score" is A's original reward in the B schema; when --a is
    # given A is authoritative.  Only checked when explicitly requested, because
    # an older artifact may have collected A with a lower threshold.
    if min_original_score is not None and info["original_stored_score"] is not None:
        if info["original_stored_score"] < min_original_score:
            return (
                "original_score",
                f"stored original score {info['original_stored_score']:.4f} < {min_original_score}",
                info,
            )

    info["score"] = float(score_response(reward_fn, reward_row, response, sandbox_url))
    if not math.isfinite(info["score"]):
        return "reward", f"non-finite score {info['score']!r}", info
    if info["score"] < min_score:
        return "reward", f"score {info['score']:.4f} < {min_score}", info

    if check_original_reward and isinstance(original, str) and original.strip():
        info["original_score"] = float(score_response(reward_fn, reward_row, original, sandbox_url))
        if not math.isfinite(info["original_score"]):
            return "original_reward", f"non-finite score {info['original_score']!r}", info
        if info["original_score"] < min_score:
            return "original_reward", f"score {info['original_score']:.4f} < {min_score}", info

    return "", "", info


# ---------------------------------------------------------------------------
# CLI / driver
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--b",
        default="~/data/reasoning_rl/dpo/stage_b/dataset_b.parquet",
        help="dataset-B parquet file / glob / directory",
    )
    parser.add_argument(
        "--a",
        default="",
        help="optional dataset-A parquet; verifies sample_id/prompt/original_response/ground_truth still agree",
    )
    parser.add_argument("--output_dir", default="~/data/reasoning_rl/dpo/stage_b_filtered")
    parser.add_argument("--min_score", type=float, default=1.0, help="B response must re-score at least this")
    parser.add_argument(
        "--min_original_score",
        type=float,
        default=None,
        help="if set, the stored original score must reach this (default: no original-score check)",
    )
    parser.add_argument(
        "--check_original_reward",
        action="store_true",
        help="re-score the original response too (doubles reward cost; code rows run model code)",
    )
    parser.add_argument(
        "--check_body_answer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reject a changed/new \\boxed{} inside the think body (default: on)",
    )
    parser.add_argument(
        "--allow_tail_change",
        action="store_true",
        help="allow the post-</think> tail to change if the parsed answer label is unchanged",
    )
    parser.add_argument(
        "--allow_missing_original",
        action="store_true",
        help="keep rows without original_response (reward-only check; label is unverifiable)",
    )
    parser.add_argument(
        "--sandbox_fusion_url",
        default=os.environ.get("SANDBOX_FUSION_URL", ""),
        help="code reward sandbox (default: $SANDBOX_FUSION_URL; empty = local prime_code fallback)",
    )
    parser.add_argument("--score_workers", type=int, default=8, help="reward-scoring threads")
    parser.add_argument("--limit", type=int, default=0, help="stop after this many B rows (0 = all)")
    parser.add_argument("--log_every", type=int, default=100, help="progress every N rows (0 = off)")
    parser.add_argument("--dry_run", action="store_true", help="run every check but write nothing")
    return parser


def _load_a_index(path: str) -> dict[str, dict]:
    rows = read_parquet_rows(path)
    index: dict[str, dict] = {}
    duplicates = 0
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            continue
        if sample_id in index:
            duplicates += 1
            continue
        index[sample_id] = row
    logger.info(
        "loaded %d dataset-A rows from %s (%d duplicate sample_ids skipped)",
        len(rows),
        path,
        duplicates,
    )
    return index


def _rejected_row(row: dict, reason: str, detail: str, info: dict) -> dict:
    """The original B row plus fixed-type audit columns (no None -> stable schema).

    ``-999.0`` means "not computed": a row rejected before the reward call (e.g.
    a changed tail) is not executed/scored, and its stored score stays in the
    row's own ``rewritten_score`` / ``score`` columns.
    """
    rejected = dict(row)
    rejected["b_filter_reason"] = reason
    rejected["b_filter_detail"] = _short(detail, 500)
    rejected["b_filter_score"] = float(info.get("score") if info.get("score") is not None else -999.0)
    rejected["b_filter_original_score"] = _first_float(
        info.get("original_score"), info.get("original_stored_score")
    )
    rejected["b_filter_original_answer"] = str(info.get("original_answer") or "")
    rejected["b_filter_rewritten_answer"] = str(info.get("rewritten_answer") or "")
    return rejected


def _audit_line(row: dict, reason: str, detail: str, info: dict) -> dict:
    response = row.get("response") or ""
    original = row.get("original_response") or ""
    return {
        "sample_id": info.get("sample_id", ""),
        "task_id": info.get("task_id", ""),
        "data_source": info.get("data_source", ""),
        "reason": reason,
        "detail": _short(detail, 500),
        "score": info.get("score"),
        "stored_score": info.get("stored_score"),
        "original_score": info.get("original_score"),
        "original_stored_score": info.get("original_stored_score"),
        "tail_changed": info.get("tail_changed", False),
        "label_unverifiable": info.get("label_unverifiable", False),
        "original_answer": info.get("original_answer", ""),
        "rewritten_answer": info.get("rewritten_answer", ""),
        "original_chars": len(original) if isinstance(original, str) else 0,
        "rewritten_chars": len(response) if isinstance(response, str) else 0,
    }


def _build_stats(
    rows: list[dict],
    kept: list[dict],
    kept_infos: list[dict],
    rejected: list[dict],
    rejected_infos: list[dict],
    reasons: Counter,
    a_index: dict[str, dict] | None,
    stored_mismatches: list[dict],
    elapsed: float,
) -> dict:
    per_source: dict[str, dict[str, int]] = defaultdict(lambda: {"input": 0, "kept": 0, "dropped": 0})
    for row in rows:
        per_source[str(row.get("data_source") or "")]["input"] += 1
    for row in kept:
        per_source[str(row.get("data_source") or "")]["kept"] += 1
    for row in rejected:
        per_source[str(row.get("data_source") or "")]["dropped"] += 1

    kept_scores = [info["score"] for info in kept_infos if info.get("score") is not None]
    rejected_scores = [info["score"] for info in rejected_infos if info.get("score") is not None]
    kept_ratios = [info["compression_ratio"] for info in kept_infos if info.get("compression_ratio") is not None]

    return {
        "rows": len(rows),
        "kept": len(kept),
        "dropped": len(rejected),
        "kept_ratio": round(len(kept) / max(1, len(rows)), 4),
        "drop_reasons": dict(reasons.most_common()),
        "label_changes": {
            "tail": reasons.get("tail_changed", 0),
            "parsed": reasons.get("label_changed", 0),
            "body_boxed": reasons.get("body_answer_changed", 0) + reasons.get("body_answer_added", 0),
            "tail_allowed": sum(1 for info in kept_infos if info.get("tail_changed")),
        },
        "by_data_source": dict(sorted(per_source.items())),
        "kept_scores": summarise(kept_scores),
        "dropped_scores": summarise(rejected_scores),
        "kept_compression_ratio": summarise(kept_ratios),
        "stored_score_mismatches": {
            "count": len(stored_mismatches),
            "examples": stored_mismatches[:20],
        },
        "a_rows": len(a_index) if a_index is not None else 0,
        "elapsed_seconds": round(elapsed, 2),
    }


def _write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    started = time.time()
    b_path = os.path.expanduser(args.b)
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if os.path.isdir(b_path):
        b_abs, out_abs = os.path.abspath(b_path), os.path.abspath(output_dir)
        if out_abs == b_abs or out_abs.startswith(b_abs + os.sep):
            logger.warning(
                "--output_dir is inside the --b directory; a second run would also read the filtered/rejected "
                "artifacts. Use a sibling output directory."
            )

    rows = read_parquet_rows(b_path)
    rows.sort(key=lambda row: (str(row.get("task_id") or ""), str(row.get("sample_id") or "")))
    if args.limit > 0:
        rows = rows[: args.limit]
        logger.info("--limit: only the first %d rows are checked", len(rows))
    logger.info("loaded %d dataset-B rows from %s", len(rows), b_path)

    a_index = None
    if args.a:
        a_index = _load_a_index(os.path.expanduser(args.a))

    reward_fn = load_reward()
    format_ok, extract_final = load_verifier_helpers()
    sandbox_url = args.sandbox_fusion_url or None

    code_rows = sum(1 for row in rows if str(row.get("data_source") or "").startswith("code"))
    if code_rows and not sandbox_url:
        logger.warning(
            "%d code rows will be verified by the local prime_code fallback (no SANDBOX_FUSION_URL): "
            "model-generated code runs on this host; lower --score_workers if the machine is small",
            code_rows,
        )

    def _process(item: tuple[int, dict]) -> tuple[str, str, dict]:
        index, row = item
        sample_id = str(row.get("sample_id") or "")
        a_row = a_index.get(sample_id) if a_index is not None else None
        try:
            return check_row(
                row,
                reward_fn=reward_fn,
                format_ok=format_ok,
                extract_final=extract_final,
                extract_logic_answer=extract_logic_answer,
                sandbox_url=sandbox_url,
                min_score=args.min_score,
                min_original_score=args.min_original_score,
                check_body_answer=args.check_body_answer,
                allow_tail_change=args.allow_tail_change,
                allow_missing_original=args.allow_missing_original,
                check_original_reward=args.check_original_reward,
                a_row=a_row,
                require_a=a_index is not None,
            )
        except Exception as exc:  # noqa: BLE001 - one bad row must not abort the filter
            logger.exception("[filter] row %d (%s) crashed during checks", index, sample_id or "?")
            return f"error:{type(exc).__name__}", str(exc)[:500], _row_info(row)

    reasons: Counter = Counter()
    kept: list[dict] = []
    kept_infos: list[dict] = []
    rejected: list[dict] = []
    rejected_infos: list[dict] = []
    audited: list[dict] = []
    kept_ids: set[str] = set()
    stored_mismatches: list[dict] = []

    workers = max(1, args.score_workers)
    batch_size = max(1, workers * 8)
    total = len(rows)
    processed = 0
    last_logged = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, total, batch_size):
            batch = rows[start : start + batch_size]
            results = list(pool.map(_process, enumerate(batch, start=start)))
            for row, (reason, detail, info) in zip(batch, results, strict=True):
                sample_id = info.get("sample_id") or str(row.get("sample_id") or "")
                if not reason and sample_id and sample_id in kept_ids:
                    reason, detail = "duplicate_sample_id", sample_id
                if not reason:
                    kept.append(row)
                    kept_infos.append(info)
                    if sample_id:
                        kept_ids.add(sample_id)
                else:
                    reasons[reason] += 1
                    rejected.append(_rejected_row(row, reason, detail, info))
                    rejected_infos.append(info)
                    audited.append(_audit_line(row, reason, detail, info))

                stored = info.get("stored_score")
                rescored = info.get("score")
                if stored is not None and rescored is not None and abs(stored - rescored) > 1e-9:
                    stored_mismatches.append(
                        {
                            "sample_id": sample_id,
                            "data_source": info.get("data_source", ""),
                            "stored": stored,
                            "rescored": rescored,
                        }
                    )

            processed += len(batch)
            if args.log_every and (processed - last_logged >= args.log_every or processed == total):
                last_logged = processed
                logger.info(
                    "[progress] %d/%d kept=%d dropped=%d %s",
                    processed,
                    total,
                    len(kept),
                    len(rejected),
                    dict(reasons),
                )

    stats = _build_stats(
        rows,
        kept,
        kept_infos,
        rejected,
        rejected_infos,
        reasons,
        a_index,
        stored_mismatches,
        time.time() - started,
    )
    logger.info("stats: %s", json.dumps(stats, ensure_ascii=False))

    if args.dry_run:
        logger.info("[dry-run] no files written")
        return 0

    filtered_path = os.path.join(output_dir, "dataset_b_filtered.parquet")
    rejected_path = os.path.join(output_dir, "dataset_b_rejected.parquet")
    report_path = os.path.join(output_dir, "b_filter_rejected.jsonl")
    stats_path = os.path.join(output_dir, "b_filter_stats.json")

    if kept:
        write_parquet_rows(kept, filtered_path)
        logger.info("wrote %d kept rows to %s", len(kept), filtered_path)
    else:
        logger.warning("no rows passed the filter; %s was not written", filtered_path)
    if rejected:
        write_parquet_rows(rejected, rejected_path)
        logger.info("wrote %d rejected rows to %s", len(rejected), rejected_path)
    else:
        logger.info("no rejected rows; %s was not written", rejected_path)
    _write_jsonl(report_path, audited)
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
    logger.info("wrote rejection report to %s and stats to %s", report_path, stats_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
