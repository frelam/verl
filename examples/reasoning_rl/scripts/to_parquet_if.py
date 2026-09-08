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
Preprocess instruction-following data for Qwen3-4B reasoning RL (DESIGN.md section 2.5).

Source: nvidia/Nemotron-RL-instruction_following (~46.4k rows)
  - WildChat-1M prompts + Open-Instruct / IFBench verifiable constraints.
  - NeMo-Gym compatible jsonl with fields: prompt, instruction_id_list, kwargs, ...
  - Each row has up to 5 constraints (instruction_id_list) with per-constraint kwargs.
  - ground_truth is NOT a single answer; it is the constraint list itself.
    The reward side (compute_score.py) re-checks each constraint against the
    model response using the IFBench verification functions.

Usage:
    python scripts/to_parquet_if.py \
        --local_save_dir ~/data/reasoning_rl/if
"""

import argparse
import json
import os
import sys

import datasets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dedup import exact_dedup, minhash_near_dedup

IF_REPO = "nvidia/Nemotron-RL-instruction_following"


def make_extra_info(split, index, task_id, source, difficulty=""):
    # Keep keys/types identical across ALL domains so per-domain parquets can be
    # concatenated later by mix.py without schema conflicts.
    return {
        "split": split,
        "index": index,
        "task_id": task_id,
        "domain": "if",
        "source": source,
        "difficulty": str(difficulty),
        "prior_solve_rate": -1.0,
        "seed": -1,
    }


def _normalize_constraint(raw) -> dict | None:
    """Normalize one constraint entry to {id, kwargs}; None if malformed."""
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


def load_nemotron_if(path: str | None, max_rows: int | None = None) -> list[dict]:
    """Load nvidia/Nemotron-RL-instruction_following.

    The repo ships a single train.jsonl. Each row has:
      - prompt: list of chat messages (usually [{"role": "user", ...}])
      - instruction_id_list: list of constraint dicts (or parallel lists)
      - kwargs: list of kwargs dicts (parallel to instruction_id_list)
      - other metadata (task_id, source, etc.) tolerated but not required
    """
    if path:
        ds = datasets.load_dataset("json", data_files=path, split="train")
    else:
        ds = datasets.load_dataset(IF_REPO, split="train")

    rows, skipped = [], 0
    for idx, ex in enumerate(ds):
        if max_rows is not None and len(rows) >= max_rows:
            break

        # ---- prompt ----
        prompt = ex.get("prompt")
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        if not isinstance(prompt, list) or not prompt:
            skipped += 1
            continue
        # Ensure chat-message dicts with content strings.
        clean_prompt = []
        for msg in prompt:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "user")).strip() or "user"
            content = str(msg.get("content", "")).strip()
            if content:
                clean_prompt.append({"role": role, "content": content})
        if not clean_prompt:
            skipped += 1
            continue

        # ---- constraints ----
        raw_ids = ex.get("instruction_id_list")
        raw_kwargs = ex.get("kwargs")
        constraints = []
        if isinstance(raw_ids, list) and raw_ids:
            if isinstance(raw_ids[0], dict):
                # Already a list of constraint dicts.
                constraints = [_normalize_constraint(c) for c in raw_ids]
                constraints = [c for c in constraints if c is not None]
            else:
                # Parallel lists: ids + kwargs.
                ids = [str(i) for i in raw_ids]
                kwargs_list = raw_kwargs if isinstance(raw_kwargs, list) else [None] * len(ids)
                if len(kwargs_list) < len(ids):
                    kwargs_list.extend([None] * (len(ids) - len(kwargs_list)))
                for cid, kw in zip(ids, kwargs_list):
                    c = _normalize_constraint({"id": cid, "kwargs": kw})
                    if c is not None:
                        constraints.append(c)
        if not constraints:
            skipped += 1
            continue

        # ---- task_id / source ----
        task_id = ex.get("task_id") or ex.get("id") or f"if-{idx}"
        source = str(ex.get("source") or "nemotron_if")

        rows.append(
            {
                "data_source": "if_nemotron",
                "prompt": clean_prompt,
                "ability": "if",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps({"constraints": constraints}, ensure_ascii=False),
                },
                "extra_info": make_extra_info(
                    "train",
                    0,
                    task_id=str(task_id),
                    source=source,
                    difficulty="",
                ),
                "_dedup_text": clean_prompt[-1]["content"],
            }
        )
    print(f"[nemotron_if] loaded {len(rows)} rows, skipped {skipped} (empty prompt / no constraints)")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default="~/data/reasoning_rl/if")
    parser.add_argument("--if_path", default=None, help="Local jsonl/parquet override for Nemotron-RL-instruction_following.")
    parser.add_argument("--max_rows", type=int, default=None, help="Cap pool size (debug/budget).")
    parser.add_argument("--minhash_threshold", type=float, default=0.6)
    parser.add_argument("--no_minhash", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None, help="Debug: cap rows loaded.")
    args = parser.parse_args()

    rows = load_nemotron_if(args.if_path, max_rows=args.max_rows or args.max_samples)

    # Pool-level dedup on the final user message (WildChat prompts can repeat
    # with different constraint draws; we keep the first occurrence).
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
    out_path = os.path.join(local_save_dir, "train_if.parquet")
    out_ds.to_parquet(out_path)

    from collections import Counter

    per_source = Counter(r["extra_info"]["source"] for r in rows)
    print(f"[done] wrote {len(rows)} rows {dict(per_source)} -> {out_path}")
