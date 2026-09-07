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
Preprocess math-domain training data for Qwen3-4B reasoning RL (DESIGN.md section 2.1).

Main pool : SynthLabsAI/Big-Math-RL-Verified (~251k, NuminaMath-derived, rule-verified)
Supplement: BytedTsinghua-SIA/DAPO-Math-17k  (~17k, already in verl schema, integer answers)

NuminaMath-1.5 raw and OpenR1-Math-220k are intentionally NOT used: both are
NuminaMath-lineage and fully overlap with Big-Math's verified subset.

Usage:
    python scripts/to_parquet_math.py \
        --local_save_dir ~/data/reasoning_rl/math
"""

import argparse
import os
import sys

import datasets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dedup import exact_dedup, minhash_near_dedup

MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."

BIGMATH_REPO = "SynthLabsAI/Big-Math-RL-Verified"
DAPO_REPO = "BytedTsinghua-SIA/DAPO-Math-17k"


def make_extra_info(split, index, task_id, source, prior_solve_rate=-1.0):
    # Keep keys/types identical across ALL domains so per-domain parquets can be
    # concatenated later by mix.py without schema conflicts.
    return {
        "split": split,
        "index": index,
        "task_id": task_id,
        "domain": "math",
        "source": source,
        "difficulty": "",
        "prior_solve_rate": float(prior_solve_rate),
        "seed": -1,
    }


def load_bigmath(path: str | None) -> list[dict]:
    ds = datasets.load_dataset(path or BIGMATH_REPO, split="train")
    required = {"problem", "answer"}
    missing = required - set(ds.column_names)
    if missing:
        raise ValueError(
            f"Big-Math schema changed: missing columns {missing}; "
            f"available columns are {ds.column_names}. Update the field mapping in load_bigmath()."
        )
    rows = []
    skipped = 0
    for idx, ex in enumerate(ds):
        problem = (ex.get("problem") or "").strip()
        answer = str(ex.get("answer") or "").strip()
        if not problem or not answer:
            skipped += 1
            continue
        rows.append(
            {
                "data_source": "math_bigmath",
                "prompt": [{"role": "user", "content": problem + " " + MATH_INSTRUCTION}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": make_extra_info(
                    "train",
                    idx,
                    task_id=f"bigmath-{idx}",
                    source=str(ex.get("source") or "big_math"),
                    prior_solve_rate=ex.get("llama8b_solve_rate") if ex.get("llama8b_solve_rate") is not None else -1.0,
                ),
                "_dedup_text": problem,
            }
        )
    print(f"[bigmath] loaded {len(rows)} rows, skipped {skipped} empty rows")
    return rows


def load_dapo(path: str | None) -> list[dict]:
    ds = datasets.load_dataset(path or DAPO_REPO, "default", split="train")
    rows = []
    for idx, ex in enumerate(ds):
        content = ex["prompt"][0]["content"].strip()
        if "\\boxed{}" not in content:
            content = content + " " + MATH_INSTRUCTION
        ground_truth = str(ex["reward_model"]["ground_truth"]).strip()
        rows.append(
            {
                "data_source": "math_dapo",
                "prompt": [{"role": "user", "content": content}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": make_extra_info("train", idx, task_id=f"dapo-{idx}", source="dapo_math_17k"),
                "_dedup_text": content.replace(MATH_INSTRUCTION, "").strip(),
            }
        )
    print(f"[dapo] loaded {len(rows)} rows")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default="~/data/reasoning_rl/math")
    parser.add_argument("--big_math_path", default=None, help="Local path override for Big-Math-RL-Verified.")
    parser.add_argument("--dapo_path", default=None, help="Local path override for DAPO-Math-17k.")
    parser.add_argument("--minhash_threshold", type=float, default=0.6)
    parser.add_argument("--no_minhash", action="store_true", help="Skip MinHash near-dedup (exact dedup only).")
    parser.add_argument("--max_samples", type=int, default=None, help="Debug: cap rows per source.")
    args = parser.parse_args()

    # DAPO first: it is the smaller curated pool and wins exact/near-dup collisions.
    rows = load_dapo(args.dapo_path) + load_bigmath(args.big_math_path)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    texts = [r["_dedup_text"] for r in rows]

    keep = exact_dedup(texts)
    rows = [rows[i] for i in keep]
    print(f"[dedup] exact: kept {len(rows)} / {len(texts)}")

    if not args.no_minhash:
        texts = [r["_dedup_text"] for r in rows]
        keep = minhash_near_dedup(texts, threshold=args.minhash_threshold)
        dropped = len(rows) - len(keep)
        rows = [rows[i] for i in keep]
        print(f"[dedup] minhash(thr={args.minhash_threshold}): dropped {dropped}, kept {len(rows)}")

    for r in rows:
        r.pop("_dedup_text")
    # Re-index after dedup so extra_info.index is contiguous.
    for i, r in enumerate(rows):
        r["extra_info"]["index"] = i

    out_ds = datasets.Dataset.from_list(rows)
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    out_path = os.path.join(local_save_dir, "train_math.parquet")
    out_ds.to_parquet(out_path)

    n_dapo = sum(1 for r in rows if r["data_source"] == "math_dapo")
    print(f"[done] wrote {len(rows)} rows ({n_dapo} dapo + {len(rows) - n_dapo} bigmath) -> {out_path}")
