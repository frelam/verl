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
Stratified domain mixing for reasoning RL (DESIGN.md section 5).

v1 ratios: math 45% / code 25% / logic 15% / STEM 15%.

Reads the per-domain parquets produced by to_parquet_{math,code,logic,stem}.py,
samples each domain to its share (without replacement when the pool is large
enough, else with replacement and a warning), shuffles with a fixed seed and
writes the final train parquet. A small stratified validation split is carved
out FIRST (before training sampling) so the run script works out of the box;
for real experiments prefer the fixed eval protocol of DESIGN.md section 7
(AIME / MATH-500 / LCB / GPQA) over this in-distribution val split.

Usage:
    python scripts/mix.py \
        --input_dir ~/data/reasoning_rl \
        --output_dir ~/data/reasoning_rl/final
"""

import argparse
import json
import os
import random
from collections import Counter

import datasets

DOMAIN_RATIOS = {"math": 0.45, "code": 0.25, "logic": 0.15, "stem": 0.15}
DOMAIN_FILES = {
    "math": "math/train_math.parquet",
    "code": "code/train_code.parquet",
    "logic": "logic/train_logic.parquet",
    "stem": "stem/train_stem.parquet",
}


def load_domain(input_dir: str, domain: str, path_override: str | None) -> list[dict]:
    rel = path_override or DOMAIN_FILES[domain]
    path = rel if os.path.isabs(rel) else os.path.join(input_dir, rel)
    ds = datasets.load_dataset("parquet", data_files=path, split="train")
    rows = ds.to_list()
    # Normalise extra_info keys across domains: difficulty_tag.py adds
    # pass_rate only to tagged pools, and a struct schema mismatch would break
    # the final Dataset.from_list merge.
    for r in rows:
        r["extra_info"].setdefault("pass_rate", -1.0)
    print(f"[mix] {domain}: {len(rows)} rows from {path}")
    return rows


def stratified_take(rng: random.Random, rows: list[dict], n: int, domain: str) -> list[dict]:
    if n <= len(rows):
        return rng.sample(rows, n)
    print(f"[mix] WARNING: {domain} pool ({len(rows)}) smaller than target share ({n}); sampling with replacement")
    return [rng.choice(rows) for _ in range(n)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="~/data/reasoning_rl")
    parser.add_argument("--output_dir", default="~/data/reasoning_rl/final")
    for domain in DOMAIN_RATIOS:
        parser.add_argument(f"--{domain}_path", default=None, help="Override parquet path for this domain.")
    parser.add_argument(
        "--total_size",
        type=int,
        default=None,
        help="Total training rows. Default: the largest size fillable without replacement.",
    )
    parser.add_argument(
        "--val_size", type=int, default=256, help="Stratified val rows carved out before train sampling (0 disables)."
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = os.path.expanduser(args.input_dir)
    output_dir = os.path.expanduser(args.output_dir)
    rng = random.Random(args.seed)

    pools = {d: load_domain(input_dir, d, getattr(args, f"{d}_path")) for d in DOMAIN_RATIOS}
    for d, rows in pools.items():
        if not rows:
            raise ValueError(f"domain {d!r} is empty; run its to_parquet script first")

    # Carve the val split first so val rows never leak into train.
    val_rows = []
    if args.val_size > 0:
        for d, rows in pools.items():
            n_val = max(1, int(args.val_size * DOMAIN_RATIOS[d]))
            n_val = min(n_val, len(rows) // 10)  # never take more than 10% of a domain
            taken = rng.sample(rows, n_val)
            taken_ids = {id(r) for r in taken}
            pools[d] = [r for r in rows if id(r) not in taken_ids]
            val_rows.extend(taken)

    if args.total_size is not None:
        total = args.total_size
    else:
        # Largest total fillable without replacement: min over domains of
        # pool_size / ratio.
        total = int(min(len(pools[d]) / DOMAIN_RATIOS[d] for d in DOMAIN_RATIOS))

    train_rows = []
    for d, rows in pools.items():
        target = int(total * DOMAIN_RATIOS[d])
        train_rows.extend(stratified_take(rng, rows, target, d))

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    # Re-index so extra_info.index is contiguous in the final artifacts.
    for i, r in enumerate(train_rows):
        r["extra_info"]["index"] = i
    for i, r in enumerate(val_rows):
        r["extra_info"]["index"] = i
        r["extra_info"]["split"] = "val"

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.parquet")
    datasets.Dataset.from_list(train_rows).to_parquet(train_path)
    val_path = None
    if val_rows:
        val_path = os.path.join(output_dir, "val.parquet")
        datasets.Dataset.from_list(val_rows).to_parquet(val_path)

    stats = {
        "total_train": len(train_rows),
        "total_val": len(val_rows),
        "seed": args.seed,
        "ratios": DOMAIN_RATIOS,
        "train_by_ability": dict(Counter(r["ability"] for r in train_rows)),
        "train_by_source": dict(Counter(r["extra_info"]["source"] for r in train_rows)),
        "val_by_ability": dict(Counter(r["ability"] for r in val_rows)),
    }
    stats_path = os.path.join(output_dir, "mix_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[done] train={len(train_rows)} -> {train_path}")
    if val_path:
        print(f"[done] val={len(val_rows)} -> {val_path}")
    print(f"[done] stats -> {stats_path}: {json.dumps(stats['train_by_ability'])}")
