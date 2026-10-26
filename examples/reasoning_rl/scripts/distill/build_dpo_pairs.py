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
"""Dataset C: join A (verbose original) and B (shortened rewrite) into DPO pairs.

Each pair is one question with two verified-correct responses::

    chosen   = B: the judge's condensed chain of thought
    rejected = A: the original rollout of the RL checkpoint

so optimising the DPO objective teaches "same answer, less redundant reasoning".
The join is strict on purpose -- ``sample_id`` must exist in both artifacts, and
the prompt and the original response must be byte-identical.  A mismatch means
the two stages did not run on the same input, and silently pairing them would
train on a different question's answer.

The validation split is carved by ``task_id`` (never by row) so the same
question cannot appear in both train and val.

Usage::

    python examples/reasoning_rl/scripts/distill/build_dpo_pairs.py \
        --a ~/data/reasoning_rl/dpo/stage_a/dataset_a.parquet \
        --b ~/data/reasoning_rl/dpo/stage_b/dataset_b.parquet \
        --output_dir ~/data/reasoning_rl/dpo \
        --val_ratio 0.02

Outputs ``train.parquet`` / ``val.parquet`` (columns: ``prompt``, ``chosen``,
``rejected`` + audit metadata) and ``dpo_stats.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from collections import Counter, defaultdict

from common import (
    load_verifier_helpers,
    read_parquet_rows,
    summarise,
    split_think,
    write_parquet_rows,
)

logger = logging.getLogger("build_dpo_pairs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a", default="~/data/reasoning_rl/dpo/stage_a/dataset_a.parquet")
    parser.add_argument("--b", default="~/data/reasoning_rl/dpo/stage_b/dataset_b.parquet")
    parser.add_argument("--output_dir", default="~/data/reasoning_rl/dpo")
    parser.add_argument("--val_ratio", type=float, default=0.02, help="share of question groups held out (0 = no val)")
    parser.add_argument("--min_savings", type=float, default=0.1, help="chosen must be at least this much shorter")
    parser.add_argument("--max_pairs_per_task", type=int, default=0, help="cap pairs per question (0 = all)")
    parser.add_argument(
        "--no_final_answer_check",
        action="store_true",
        help="skip the parsed-final-answer equality check (B already verifies it with the reward)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _prompt_fingerprint(prompt) -> str:
    return json.dumps(prompt, ensure_ascii=False, sort_keys=True)


def _build_pairs(args: argparse.Namespace) -> tuple[list[dict], Counter, Counter]:
    a_rows = read_parquet_rows(os.path.expanduser(args.a))
    b_rows = read_parquet_rows(os.path.expanduser(args.b))
    logger.info("dataset A: %d trajectories; dataset B: %d rewrites", len(a_rows), len(b_rows))

    a_by_id: dict[str, dict] = {}
    for row in a_rows:
        sample_id = row.get("sample_id")
        if sample_id and sample_id not in a_by_id:
            a_by_id[sample_id] = row

    dropped = Counter()
    per_task = Counter()
    pairs: list[dict] = []
    seen: set[tuple[str, str]] = set()
    extract_final = None if args.no_final_answer_check else load_verifier_helpers()[1]

    for b_row in b_rows:
        sample_id = b_row.get("sample_id")
        if not sample_id:
            dropped["missing_sample_id"] += 1
            continue
        a_row = a_by_id.get(sample_id)
        if a_row is None:
            dropped["missing_in_a"] += 1
            continue
        if _prompt_fingerprint(a_row.get("prompt")) != _prompt_fingerprint(b_row.get("prompt")):
            dropped["prompt_mismatch"] += 1
            continue
        if a_row.get("response") != b_row.get("original_response"):
            dropped["original_response_mismatch"] += 1
            continue

        chosen = b_row.get("response") or ""
        rejected = a_row.get("response") or ""
        if not chosen.strip() or not rejected.strip():
            dropped["empty_response"] += 1
            continue
        if chosen.strip() == rejected.strip():
            dropped["identical"] += 1
            continue
        if len(chosen) > (1.0 - args.min_savings) * len(rejected):
            dropped["not_short_enough"] += 1
            continue
        if split_think(chosen) is None or split_think(rejected) is None:
            dropped["not_think_compliant"] += 1
            continue
        if extract_final is not None and extract_final(chosen) != extract_final(rejected):
            dropped["final_answer_changed"] += 1
            continue

        task_id = str(b_row.get("task_id") or a_row.get("task_id") or "")
        if args.max_pairs_per_task > 0 and per_task[task_id] >= args.max_pairs_per_task:
            dropped["task_cap"] += 1
            continue
        key = (_prompt_fingerprint(b_row.get("prompt")), chosen, rejected)
        if key in seen:
            dropped["duplicate_pair"] += 1
            continue
        seen.add(key)
        per_task[task_id] += 1

        pairs.append(
            {
                "prompt": b_row.get("prompt"),
                "chosen": chosen,
                "rejected": rejected,
                "task_id": task_id,
                "data_source": str(b_row.get("data_source") or a_row.get("data_source") or ""),
                "ability": str(b_row.get("ability") or a_row.get("ability") or ""),
                "sample_id": sample_id,
                "chosen_score": float(b_row.get("rewritten_score", b_row.get("score", 1.0))),
                "rejected_score": float(a_row.get("score", 1.0)),
                "chosen_tokens": int(b_row.get("rewritten_tokens") or 0),
                "rejected_tokens": int(a_row.get("response_tokens") or 0),
                "chosen_chars": len(chosen),
                "rejected_chars": len(rejected),
                "compression_ratio": round(len(chosen) / max(1, len(rejected)), 4),
            }
        )

    logger.info("matched %d pairs (dropped: %s)", len(pairs), dict(dropped))
    return pairs, dropped, per_task


def _split_by_task(pairs: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    if val_ratio <= 0 or not pairs:
        return pairs, []
    groups: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        groups[pair["task_id"]].append(pair)
    task_ids = sorted(groups)
    random.Random(seed).shuffle(task_ids)
    target = max(1, round(len(pairs) * val_ratio))
    train: list[dict] = []
    val: list[dict] = []
    for task_id in task_ids:
        if len(val) < target:
            val.extend(groups[task_id])
        else:
            train.extend(groups[task_id])
    return train, val


def _stats(train: list[dict], val: list[dict], dropped: Counter) -> dict:
    pairs = train + val
    return {
        "pairs": len(pairs),
        "train_pairs": len(train),
        "val_pairs": len(val),
        "questions": len({pair["task_id"] for pair in pairs}),
        "by_data_source": dict(Counter(pair["data_source"] for pair in pairs)),
        "by_ability": dict(Counter(pair["ability"] for pair in pairs)),
        "compression_ratio": summarise(pair["compression_ratio"] for pair in pairs),
        "token_savings": summarise(
            max(0, pair["rejected_tokens"] - pair["chosen_tokens"]) for pair in pairs
        ),
        "dropped": dict(dropped),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    pairs, dropped, per_task = _build_pairs(args)
    if not pairs:
        raise SystemExit("no valid DPO pairs; check that A and B were built from the same dataset")

    train, val = _split_by_task(pairs, args.val_ratio, args.seed)
    if not train:
        raise SystemExit("train split is empty; lower --val_ratio")
    write_parquet_rows(train, os.path.join(output_dir, "dpo_train.parquet"))
    logger.info("wrote %d train pairs to %s/dpo_train.parquet", len(train), output_dir)
    if val:
        write_parquet_rows(val, os.path.join(output_dir, "dpo_val.parquet"))
        logger.info("wrote %d val pairs to %s/dpo_val.parquet", len(val), output_dir)
    else:
        logger.warning("validation split is empty (val_ratio=%.3f); training will run without eval", args.val_ratio)

    stats = _stats(train, val, dropped)
    stats["pairs_per_question"] = summarise(per_task.values())
    with open(os.path.join(output_dir, "dpo_stats.json"), "w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
    logger.info("stats: %s", json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
