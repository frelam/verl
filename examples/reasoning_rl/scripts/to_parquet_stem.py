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
Preprocess STEM-domain training data for Qwen3-4B reasoning RL (DESIGN.md section 2.4).

Source: MiniByte-666/Dr.SCI — the repo ships TWO parquet files:
  - Dr_SCI_verifiable.parquet  (~461k, extra_info.match_rule=True, short
    rule-checkable answers) -> THIS is the training pool (DESIGN.md section 2.4)
  - Dr_SCI_open-ended.parquet  (rubric/LLM-judge style proofs) -> NOT used:
    no rule verifier, would inject unverifiable reward noise.

The verifiable parquet's prompt is already chat-format with a
"The final answer is: $\\boxed{$ANSWER}$" instruction, so it is kept as-is and
the reward side routes stem_* to math_verify (DESIGN.md section 6).

Tolerant loader: rows with empty prompt/ground_truth are skipped, and
match_rule is re-checked per row (schema-drift guard).

Usage:
    python scripts/to_parquet_stem.py \
        --local_save_dir ~/data/reasoning_rl/stem
"""

import argparse
import os
import sys

import datasets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dedup import exact_dedup, minhash_near_dedup

DRSCI_REPO = "MiniByte-666/Dr.SCI"
DRSCI_VERIFIABLE_FILE = "Dr_SCI_verifiable.parquet"


def make_extra_info(split, index, task_id, source, difficulty=""):
    # Keep keys/types identical across ALL domains so per-domain parquets can be
    # concatenated later by mix.py without schema conflicts.
    return {
        "split": split,
        "index": index,
        "task_id": task_id,
        "domain": "stem",
        "source": source,
        "difficulty": str(difficulty),
        "prior_solve_rate": -1.0,
        "seed": -1,
    }


def load_drsci(path: str | None, max_rows: int | None = None) -> list[dict]:
    if path:
        ds = datasets.load_dataset("parquet", data_files=path, split="train")
    else:
        ds = datasets.load_dataset(DRSCI_REPO, data_files=DRSCI_VERIFIABLE_FILE, split="train")
    rows, skipped = [], 0
    for idx, ex in enumerate(ds):
        if max_rows is not None and len(rows) >= max_rows:
            break
        extra = ex.get("extra_info") or {}
        # Schema-drift guard: the verifiable file is all match_rule=True, but
        # re-check per row so a repo update never leaks rubric-judged rows in.
        match_rule = extra.get("match_rule", True)
        if isinstance(match_rule, str):
            match_rule = match_rule.strip().lower() == "true"
        if not match_rule:
            skipped += 1
            continue
        reward_model = ex.get("reward_model") or {}
        ground_truth = str(reward_model.get("ground_truth") or extra.get("reference_answer") or "").strip()
        prompt = ex.get("prompt") or []
        content = prompt[0].get("content") if prompt and isinstance(prompt[0], dict) else ""
        if not content.strip() or not ground_truth:
            skipped += 1
            continue
        subject = str(extra.get("subject") or "unknown")
        rows.append(
            {
                "data_source": "stem_drsci",
                "prompt": [{"role": "user", "content": content}],
                "ability": "stem",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": make_extra_info(
                    "train",
                    0,
                    task_id=f"drsci-{idx}",
                    source=f"drsci_{subject}",
                    difficulty=extra.get("difficulty", ""),
                ),
                "_dedup_text": content.strip(),
            }
        )
    print(f"[drsci] loaded {len(rows)} rows, skipped {skipped} (not match_rule / empty)")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default="~/data/reasoning_rl/stem")
    parser.add_argument("--drsci_path", default=None, help="Local parquet override for Dr.SCI verifiable file.")
    parser.add_argument(
        "--max_rows", type=int, default=None, help="Cap pool size (Dr.SCI is 461k; e.g. 150000 for the v1 mix)."
    )
    parser.add_argument("--minhash_threshold", type=float, default=0.6)
    parser.add_argument("--no_minhash", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None, help="Debug: cap rows loaded.")
    args = parser.parse_args()

    rows = load_drsci(args.drsci_path, max_rows=args.max_rows or args.max_samples)

    # Pool-level dedup (Dr.SCI is web-distilled: same question can appear with
    # slightly different phrasing across source corpora).
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
    for i, r in enumerate(rows):
        r["extra_info"]["index"] = i

    out_ds = datasets.Dataset.from_list(rows)
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    out_path = os.path.join(local_save_dir, "train_stem.parquet")
    out_ds.to_parquet(out_path)

    from collections import Counter

    per_source = Counter(r["extra_info"]["source"] for r in rows)
    print(f"[done] wrote {len(rows)} rows {dict(per_source)} -> {out_path}")
