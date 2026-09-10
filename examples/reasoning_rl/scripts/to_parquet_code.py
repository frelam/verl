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
"""
Preprocess code-domain training data for Qwen3-4B reasoning RL (DESIGN.md section 2.2).

Skeleton  : DeepCoder training set (~24k; already deduped, already time-split vs LiveCodeBench)
Supplement: deepmind/code_contests train split (~13k), codeparrot/apps train split (~10k)

TACO / PrimeIntellect verifiable-coding-problems are intentionally NOT added:
DeepCoder-train already covers TACO-verified + PrimeIntellect (see DESIGN.md section 2.2).

ground_truth format matches verl prime_code / sandbox_fusion consumers:
    json string of {"inputs": [...], "outputs": [...] (, "fn_name": ...)}
    Canonical test shapes (enforced by normalize_in_outs):
      * stdio problems: inputs[i] / outputs[i] are plain strings.
      * call-based (fn_name) problems: inputs[i] is a JSON-per-line string
        (one JSON value per function argument); outputs[i] is the expected
        return value as a JSON string (single-element list unwrapped once,
        TACO convention). prime_code parses these with json.loads and calls
        fn(*args); sandbox_fusion feeds inputs[i] verbatim as stdin to its
        fn_name wrapper and string-compares stdout with outputs[i].
    Raw APPS rows deviate from this contract (call-based inputs[i] is a list
    of args, outputs[i] is list-wrapped; some stdio inputs are lists of
    lines) and score 0 on BOTH verifiers if stored unmodified.

Usage:
    python scripts/to_parquet_code.py \
        --local_save_dir ~/data/reasoning_rl/code
"""

import argparse
import hashlib
import json
import os
import sys

import datasets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dedup import exact_dedup, minhash_near_dedup

MAX_TESTS_PER_PROBLEM = 20  # cap for reward-time timeout control

STDIO_INSTRUCTION = (
    "Write a complete Python program that reads the input from standard input and prints "
    "the answer to standard output. Wrap your code in ```python and ```."
)
FUNCTION_INSTRUCTION = "Implement the required function(s). Wrap your code in ```python and ```."

DEEPCODER_REPO = "agentica-org/DeepCoder-Preview-Dataset"
CODECONTESTS_REPO = "deepmind/code_contests"
APPS_REPO = "codeparrot/apps"


def make_extra_info(split, index, task_id, source, difficulty=""):
    # Keep keys/types identical across ALL domains so per-domain parquets can be
    # concatenated later by mix.py without schema conflicts.
    return {
        "split": split,
        "index": index,
        "task_id": task_id,
        "domain": "code",
        "source": source,
        "difficulty": str(difficulty),
        "prior_solve_rate": -1.0,
        "seed": -1,
    }


def _canon_stdio_field(v) -> str:
    """stdio test field -> plain string (APPS stores some as lists of lines)."""
    if isinstance(v, str):
        return v
    if isinstance(v, list | tuple):
        return "\n".join(str(x).rstrip("\n") for x in v)
    return str(v)


def _canon_call_input(v) -> str:
    """call-based test input -> JSON-per-line string (one JSON value per arg).

    prime_code parses each line with json.loads and calls fn(*args);
    sandbox_fusion pipes the string verbatim as stdin to its fn_name wrapper,
    which does the same. Raw APPS stores the args as a list instead.
    """
    if isinstance(v, str):
        return v
    if isinstance(v, list | tuple):
        return "\n".join(json.dumps(a) for a in v)
    return json.dumps(v)


def _canon_call_output(v) -> str:
    """call-based expected return -> JSON string.

    prime_code json.loads-es it and compares structurally (tolerating a
    list-wrapped value); sandbox_fusion string-compares it against the
    wrapper's stdout (json.dumps of the result for containers/bool/None,
    str() for int/float/str), so a single-element list is unwrapped once
    (TACO convention: expected returns are list-wrapped).
    """
    if isinstance(v, str):
        return v
    if isinstance(v, list | tuple) and len(v) == 1:
        v = v[0]
    return json.dumps(v)


def normalize_in_outs(raw) -> dict | None:
    """Parse raw test-case field into {"inputs","outputs"(,"fn_name")}; None if invalid.

    Test shapes are canonicalised to the verifier contract (see module docstring);
    already-canonical rows (DeepCoder, code_contests) pass through unchanged.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
    if not isinstance(raw, dict):
        return None
    inputs, outputs = raw.get("inputs"), raw.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list) or not inputs:
        return None
    if len(inputs) != len(outputs):
        return None
    fn_name = raw.get("fn_name")
    if fn_name:
        canon_i = [_canon_call_input(v) for v in inputs[:MAX_TESTS_PER_PROBLEM]]
        canon_o = [_canon_call_output(v) for v in outputs[:MAX_TESTS_PER_PROBLEM]]
    else:
        canon_i = [_canon_stdio_field(v) for v in inputs[:MAX_TESTS_PER_PROBLEM]]
        canon_o = [_canon_stdio_field(v) for v in outputs[:MAX_TESTS_PER_PROBLEM]]
    in_outs = {"inputs": canon_i, "outputs": canon_o}
    if fn_name:
        in_outs["fn_name"] = str(fn_name)
    return in_outs


def tests_key(in_outs: dict) -> str:
    """Hash of the first few test cases; catches cross-source duplicates of the same
    problem even when problem statements were reworded (DESIGN.md section 2.2)."""
    payload = json.dumps({"i": in_outs["inputs"][:3], "o": in_outs["outputs"][:3]}, sort_keys=True, default=str)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def make_code_row(problem: str, in_outs: dict, task_id: str, source: str, difficulty="") -> dict:
    instruction = FUNCTION_INSTRUCTION if "fn_name" in in_outs else STDIO_INSTRUCTION
    content = problem.strip() + "\n\n" + instruction
    return {
        "data_source": f"code_{source}",
        "prompt": [{"role": "user", "content": content}],
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": json.dumps(in_outs)},
        "extra_info": make_extra_info("train", 0, task_id=task_id, source=source, difficulty=difficulty),
        "_dedup_text": problem.strip(),
    }


def load_deepcoder(path: str | None) -> list[dict]:
    ds = datasets.load_dataset(path or DEEPCODER_REPO, split="train")
    rows, skipped = [], 0
    for idx, ex in enumerate(ds):
        problem = ex.get("problem") or ex.get("question") or ex.get("description") or ""
        # Tolerant field probing: repo schema may rename the tests field.
        in_outs = None
        for field in ("input_output", "tests", "ground_truth"):
            if ex.get(field):
                in_outs = normalize_in_outs(ex[field])
                if in_outs is not None:
                    break
        if not problem.strip() or in_outs is None:
            skipped += 1
            continue
        rows.append(make_code_row(problem, in_outs, task_id=f"deepcoder-{idx}", source="deepcoder"))
    print(f"[deepcoder] loaded {len(rows)} rows, skipped {skipped} (no tests / no problem)")
    return rows


def load_codecontests(path: str | None, trust_remote_code: bool) -> list[dict]:
    ds = datasets.load_dataset(path or CODECONTESTS_REPO, split="train", trust_remote_code=trust_remote_code)
    rows, skipped = [], 0
    for idx, ex in enumerate(ds):
        problem = ex.get("description") or ""
        inputs, outputs = [], []
        for field in ("public_tests", "private_tests", "generated_tests"):
            remaining = MAX_TESTS_PER_PROBLEM - len(inputs)
            if remaining <= 0:
                break
            tests = ex.get(field) or {}
            inputs.extend(tests.get("input", [])[:remaining])
            outputs.extend(tests.get("output", [])[:remaining])
        in_outs = normalize_in_outs({"inputs": inputs, "outputs": outputs})
        if not problem.strip() or in_outs is None:
            skipped += 1
            continue
        rows.append(
            make_code_row(
                problem,
                in_outs,
                task_id=f"codecontests-{idx}",
                source="codecontests",
                difficulty=ex.get("difficulty", ""),
            )
        )
    print(f"[codecontests] loaded {len(rows)} rows, skipped {skipped} (no tests / no problem)")
    return rows


def load_apps(path: str | None, trust_remote_code: bool) -> list[dict]:
    ds = datasets.load_dataset(path or APPS_REPO, split="train", trust_remote_code=trust_remote_code)
    rows, skipped = [], 0
    for idx, ex in enumerate(ds):
        problem = ex.get("question") or ""
        in_outs = normalize_in_outs(ex.get("input_output"))
        if not problem.strip() or in_outs is None:
            skipped += 1
            continue
        rows.append(
            make_code_row(
                problem,
                in_outs,
                task_id=f"apps-{ex.get('problem_id', idx)}",
                source="apps",
                difficulty=ex.get("difficulty", ""),
            )
        )
    print(f"[apps] loaded {len(rows)} rows, skipped {skipped} (no tests / no problem)")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default="~/data/reasoning_rl/code")
    parser.add_argument("--deepcoder_path", default=None, help="Local path override for DeepCoder train set.")
    parser.add_argument("--codecontests_path", default=None, help="Local path override for code_contests.")
    parser.add_argument("--apps_path", default=None, help="Local path override for APPS.")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--minhash_threshold", type=float, default=0.6)
    parser.add_argument("--no_minhash", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None, help="Debug: cap rows per source.")
    args = parser.parse_args()

    # DeepCoder first: it is the curated skeleton and wins dup collisions.
    rows = (
        load_deepcoder(args.deepcoder_path)
        + load_codecontests(args.codecontests_path, args.trust_remote_code)
        + load_apps(args.apps_path, args.trust_remote_code)
    )
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    # Level 1: test-case hash dedup (cross-source same problem, reworded statement).
    seen_tests: set[str] = set()
    kept = []
    for r in rows:
        k = tests_key(json.loads(r["reward_model"]["ground_truth"]))
        if k not in seen_tests:
            seen_tests.add(k)
            kept.append(r)
    print(f"[dedup] test-hash: dropped {len(rows) - len(kept)}, kept {len(kept)}")
    rows = kept

    # Level 2/3: problem-text exact + MinHash near-dedup.
    texts = [r["_dedup_text"] for r in rows]
    keep = exact_dedup(texts)
    rows = [rows[i] for i in keep]
    print(f"[dedup] text exact: kept {len(rows)} / {len(texts)}")

    if not args.no_minhash:
        texts = [r["_dedup_text"] for r in rows]
        keep = minhash_near_dedup(texts, threshold=args.minhash_threshold)
        rows = [rows[i] for i in keep]
        print(f"[dedup] minhash(thr={args.minhash_threshold}): kept {len(rows)}")

    for r in rows:
        r.pop("_dedup_text")
    # Re-index after dedup so extra_info.index is contiguous.
    for i, r in enumerate(rows):
        r["extra_info"]["index"] = i

    out_ds = datasets.Dataset.from_list(rows)
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    out_path = os.path.join(local_save_dir, "train_code.parquet")
    out_ds.to_parquet(out_path)

    from collections import Counter

    per_source = Counter(r["data_source"] for r in rows)
    print(f"[done] wrote {len(rows)} rows {dict(per_source)} -> {out_path}")
