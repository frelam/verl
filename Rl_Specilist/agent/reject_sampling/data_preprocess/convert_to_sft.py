# Copyright 2025 Individual Contributor
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
"""Convert reject sampling trajectories to SFT parquet format.

Reads the JSONL file produced by ``trajectory_collector``, deduplicates by
prompt, keeps top-K trajectories per prompt, and outputs a parquet file
compatible with ``verl.utils.dataset.MultiTurnSFTDataset``.

Output format:
    - ``messages`` column: list of {role, content} dicts
    - ``tools`` column: list of tool schemas (may be empty)
    - ``data_source`` column: string identifier

Usage::

    python -m Rl_Specilist.agent.reject_sampling.data_preprocess.convert_to_sft \\
        --input ~/data/reject_sampling/collected_trajectories.jsonl \\
        --output_dir ~/data/reject_sampling_sft \\
        --top_k 2
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

import pandas as pd

from Rl_Specilist.agent.reject_sampling.reward.trajectory_collector import (
    get_stats,
    read_trajectories,
)


def filter_and_deduplicate(
    trajectories: list[dict],
    top_k: int = 2,
    min_score: float = 0.0,
) -> list[dict]:
    """Group by prompt, keep top-K by score.

    Args:
        trajectories: List of trajectory records from JSONL.
        top_k: Max trajectories to keep per prompt.
        min_score: Minimum score threshold.

    Returns:
        Filtered and deduplicated list of trajectories.
    """
    # Group by prompt_hash
    groups: dict[str, list[dict]] = defaultdict(list)
    for traj in trajectories:
        if traj.get("score", 0.0) >= min_score:
            groups[traj["prompt_hash"]].append(traj)

    # Sort each group by score descending, take top-K
    result = []
    for prompt_hash, group in groups.items():
        group.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        result.extend(group[:top_k])

    return result


def to_sft_dataframe(trajectories: list[dict]) -> pd.DataFrame:
    """Convert trajectories to SFT-compatible DataFrame.

    The output has columns: messages, tools, data_source
    matching the format expected by MultiTurnSFTDataset.
    """
    rows = []
    for traj in trajectories:
        rows.append({
            "messages": traj["messages"],
            "tools": traj.get("tools", []),
            "data_source": traj.get("data_source", "unknown"),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Convert reject sampling trajectories to SFT parquet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default=os.path.expanduser(
            os.environ.get("TRAJECTORY_FILE", "~/data/reject_sampling/collected_trajectories.jsonl")
        ),
        help="Input JSONL file with collected trajectories.",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.expanduser("~/data/reject_sampling_sft"),
        help="Output directory for SFT parquet files.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=2,
        help="Max trajectories to keep per prompt (default: 2).",
    )
    parser.add_argument(
        "--min_score",
        type=float,
        default=0.7,
        help="Minimum score threshold (default: 0.7).",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="Validation set ratio (default: 0.05).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Expand paths
    input_path = os.path.expanduser(args.input)
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Read trajectories
    print(f"Reading trajectories from: {input_path}")
    trajectories = read_trajectories(input_path)

    if not trajectories:
        print("ERROR: No trajectories found. Run reject sampling first.")
        print(f"  Expected file: {input_path}")
        exit(1)

    # Print stats
    stats = get_stats(input_path)
    print(f"\nCollected trajectory stats:")
    print(f"  Total trajectories: {stats['total']}")
    print(f"  Unique prompts:     {stats['unique_prompts']}")
    print(f"  By source:          {stats.get('by_source', {})}")
    print(f"  By judge:           {stats.get('by_judge', {})}")
    print(f"  Score range:        {stats.get('score_min', 0):.2f} - {stats.get('score_max', 0):.2f}")
    print(f"  Score mean:         {stats.get('score_mean', 0):.2f}")

    # Filter and deduplicate
    print(f"\nFiltering: min_score={args.min_score}, top_k={args.top_k}")
    filtered = filter_and_deduplicate(
        trajectories,
        top_k=args.top_k,
        min_score=args.min_score,
    )
    print(f"  After filtering: {len(filtered)} trajectories")

    if not filtered:
        print("ERROR: No trajectories passed the filter. Try lowering --min_score.")
        exit(1)

    # Convert to DataFrame
    df = to_sft_dataframe(filtered)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Print distribution
    from collections import Counter
    source_counts = Counter(df["data_source"])
    print(f"\nSFT data distribution:")
    for source, count in sorted(source_counts.items()):
        print(f"  {source:20s}  {count:>6d}")

    # Split train/val
    if args.val_ratio > 0:
        n_val = max(1, int(len(df) * args.val_ratio))
        val_df = df.iloc[:n_val].reset_index(drop=True)
        train_df = df.iloc[n_val:].reset_index(drop=True)
    else:
        train_df = df
        val_df = df.iloc[:0]

    # Save
    train_path = os.path.join(output_dir, "train.parquet")
    train_df.to_parquet(train_path)
    print(f"\nTrain saved: {train_path} ({len(train_df)} samples)")

    if len(val_df) > 0:
        val_path = os.path.join(output_dir, "val.parquet")
        val_df.to_parquet(val_path)
        print(f"Val saved:   {val_path} ({len(val_df)} samples)")

    # Print a sample
    if len(train_df) > 0:
        sample = train_df.iloc[0]
        print(f"\nSample [0]:")
        print(f"  data_source: {sample['data_source']}")
        print(f"  messages count: {len(sample['messages'])}")
        print(f"  roles: {[m['role'] for m in sample['messages']]}")
        print(f"  tools count: {len(sample['tools'])}")

    print(f"\n✅ SFT data ready at: {output_dir}")
    print(f"\nNext step: run SFT training")
    print(f"  bash Rl_Specilist/agent/reject_sampling/run_sft_from_reject.sh 8 ~/data/sft_ckpt")


if __name__ == "__main__":
    main()
