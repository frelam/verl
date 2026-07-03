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
"""
Prepare agent SFT datasets for verl training.

Supported datasets (all from HuggingFace Hub):
  - Nanbeige/ToolMind                          (tool-use, 369k)
  - nvidia/Open-SWE-Traces                     (SWE agent trajectories, 207k)
  - nvidia/SWE-Zero-openhands-trajectories     (SWE agent trajectories, 318k)
  - m-a-p/TerminalTraj                         (terminal agent trajectories, 20k+)
  - OpenResearcher/OpenResearcher-Dataset      (deep research trajectories, 96k)

Each dataset is converted to a parquet file with a ``messages`` column (list of
{role, content, ...} dicts) and optionally a ``tools`` column, matching the
format expected by ``verl.utils.dataset.MultiTurnSFTDataset``.

Usage:
    # Process all datasets with default sample counts (full datasets)
    python examples/data_preprocess/prepare_agent_sft_data.py \
        --output_dir ~/data/agent_sft

    # Dynamically configure sample count per dataset
    python examples/data_preprocess/prepare_agent_sft_data.py \
        --output_dir ~/data/agent_sft \
        --toolmind_n 10000 \
        --open_swe_traces_n 20000 \
        --swe_zero_n 20000 \
        --terminaltraj_n 5000 \
        --openresearcher_n 10000

    # Use a val split ratio
    python examples/data_preprocess/prepare_agent_sft_data.py \
        --output_dir ~/data/agent_sft \
        --val_ratio 0.02

    # Only process specific datasets
    python examples/data_preprocess/prepare_agent_sft_data.py \
        --output_dir ~/data/agent_sft \
        --datasets toolmind terminaltraj
"""
from __future__ import annotations

import argparse
import json
import os
import random
import traceback
from typing import Any, Optional

import pandas as pd
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Dataset registry — maps a short name to (hf_repo, config, split, description)
# ---------------------------------------------------------------------------
DATASET_REGISTRY = {
    "toolmind": {
        "repo": "Nanbeige/ToolMind",
        "config": None,
        "split": "test",
        "description": "Large-scale reasoning-enhanced tool-use dataset (369k)",
    },
    "open_swe_traces": {
        "repo": "nvidia/Open-SWE-Traces",
        "config": "openhands",
        "split": "train",
        "description": "SWE agent trajectories via OpenHands/SWE-agent (207k)",
    },
    "swe_zero": {
        "repo": "nvidia/SWE-Zero-openhands-trajectories",
        "config": None,
        "split": "train",
        "description": "Execution-free SWE agent trajectories (318k)",
    },
    "terminaltraj": {
        "repo": "m-a-p/TerminalTraj",
        "config": None,
        "split": "train",
        "description": "Terminal agent trajectories from Dockerized envs (20k+)",
    },
    "openresearcher": {
        "repo": "OpenResearcher/OpenResearcher-Dataset",
        "config": "seed_42",
        "split": "train",
        "description": "Long-horizon deep research trajectories (96k)",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sample_dataset(rows: list, n: int, seed: int = 42) -> list:
    """Randomly sample *n* rows from *rows*. n=-1 means use all."""
    if n is None or n < 0 or n >= len(rows):
        return rows
    rng = random.Random(seed)
    indices = rng.sample(range(len(rows)), n)
    return [rows[i] for i in indices]


def clean_content(content: Any) -> str:
    """Ensure content is a non-null string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


def normalize_message(msg: dict) -> Optional[dict]:
    """Normalise a single message dict to {role, content, ...extras}.

    Returns None if the message is unusable.
    """
    role = msg.get("role")
    if role is None:
        return None
    role = str(role).lower()

    # Map 'ai' role (used by some SWE-agent datasets) to 'assistant'
    if role == "ai":
        role = "assistant"

    content = clean_content(msg.get("content"))

    out: dict[str, Any] = {"role": role, "content": content}

    # Preserve tool_calls if present (assistant messages with function calls)
    if "tool_calls" in msg and msg["tool_calls"]:
        out["tool_calls"] = msg["tool_calls"]

    # Preserve reasoning_content if present
    if msg.get("reasoning_content"):
        out["reasoning_content"] = msg["reasoning_content"]

    return out


def normalize_messages(messages: list) -> list[dict]:
    """Normalise a list of messages, dropping invalid ones."""
    result = []
    for msg in messages:
        norm = normalize_message(msg) if isinstance(msg, dict) else None
        if norm is not None:
            result.append(norm)
    return result


def parse_tools_field(tools: Any) -> list[dict]:
    """Parse a tools field that may be a list of JSON strings or list of dicts."""
    if tools is None:
        return []
    result = []
    for t in tools:
        if isinstance(t, dict):
            result.append(t)
        elif isinstance(t, str):
            try:
                result.append(json.loads(t))
            except json.JSONDecodeError:
                continue
    return result


def to_dataframe(messages_list: list[list[dict]], tools_list: Optional[list[list[dict]]] = None) -> pd.DataFrame:
    """Build a DataFrame from lists of messages (and optionally tools)."""
    data: dict[str, Any] = {"messages": messages_list}
    if tools_list is not None:
        data["tools"] = tools_list
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Per-dataset converters
# ---------------------------------------------------------------------------

def convert_toolmind(rows: list[dict]) -> tuple[list[list[dict]], Optional[list[list[dict]]]]:
    """ToolMind: conversations -> messages, tools stays as tools."""
    messages_out, tools_out = [], []
    for row in rows:
        conversations = row.get("conversations")
        if not conversations:
            continue
        messages = normalize_messages(conversations)
        if not messages:
            continue
        messages_out.append(messages)
        tools = parse_tools_field(row.get("tools"))
        tools_out.append(tools if tools else [])
    return messages_out, tools_out


def convert_open_swe_traces(rows: list[dict]) -> tuple[list[list[dict]], Optional[list[list[dict]]]]:
    """Open-SWE-Traces: trajectory -> messages, tools (list[str]) -> parsed."""
    messages_out, tools_out = [], []
    for row in rows:
        trajectory = row.get("trajectory")
        if not trajectory:
            continue
        messages = normalize_messages(trajectory)
        if not messages:
            continue
        messages_out.append(messages)
        tools = parse_tools_field(row.get("tools"))
        tools_out.append(tools if tools else [])
    return messages_out, tools_out


def convert_swe_zero(rows: list[dict]) -> tuple[list[list[dict]], Optional[list[list[dict]]]]:
    """SWE-Zero: trajectory -> messages, no tools."""
    messages_out = []
    for row in rows:
        trajectory = row.get("trajectory")
        if not trajectory:
            continue
        messages = normalize_messages(trajectory)
        if not messages:
            continue
        messages_out.append(messages)
    return messages_out, None


def convert_terminaltraj(rows: list[dict]) -> tuple[list[list[dict]], Optional[list[list[dict]]]]:
    """TerminalTraj: already has 'messages' field with {role, content}."""
    messages_out = []
    for row in rows:
        messages = row.get("messages")
        if not messages:
            continue
        messages = normalize_messages(messages)
        if not messages:
            continue
        messages_out.append(messages)
    return messages_out, None


def _convert_openresearcher_content_block(block: Any) -> Optional[dict]:
    """Convert a single GPT-OSS content block to a normalised message fragment.

    GPT-OSS messages have ``content`` as a list of blocks, each carrying a
    ``channel_config`` and a payload (``text``, tool calls, etc.).  We extract
    the text / tool-call information into a standard structure.
    """
    if not isinstance(block, dict):
        return None

    # Text block
    text = block.get("text")
    if text is not None:
        return {"type": "text", "text": clean_content(text)}

    # Tool call block (GPT-OSS style)
    tool_call = block.get("tool_call") or block.get("function_call")
    if tool_call and isinstance(tool_call, dict):
        fn = tool_call.get("function", tool_call)
        name = fn.get("name", "")
        arguments = fn.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass
        return {"type": "tool_call", "name": name, "arguments": arguments}

    # Tool response block
    tool_result = block.get("tool_result") or block.get("output")
    if tool_result is not None:
        return {"type": "tool_result", "content": clean_content(tool_result)}

    return None


def convert_openresearcher(rows: list[dict]) -> tuple[list[list[dict]], Optional[list[list[dict]]]]:
    """OpenResearcher: GPT-OSS native format -> standard messages.

    The ``messages`` field is a list of dicts with ``role``, ``channel`` and
    ``content`` (a list of content blocks).  We flatten each message's content
    blocks into a single string, and convert tool calls into ``tool_calls``.
    """
    messages_out = []
    for row in rows:
        raw_messages = row.get("messages")
        if not raw_messages:
            continue
        converted = []
        for msg in raw_messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).lower()
            if not role:
                continue
            if role == "ai":
                role = "assistant"

            raw_content = msg.get("content")
            text_parts: list[str] = []
            tool_calls: list[dict] = []
            tool_result_text = ""

            if isinstance(raw_content, list):
                for block in raw_content:
                    frag = _convert_openresearcher_content_block(block)
                    if frag is None:
                        continue
                    if frag["type"] == "text":
                        text_parts.append(frag["text"])
                    elif frag["type"] == "tool_call":
                        tool_calls.append({
                            "function": {
                                "name": frag["name"],
                                "arguments": frag["arguments"],
                            }
                        })
                    elif frag["type"] == "tool_result":
                        tool_result_text = frag["content"]
            elif isinstance(raw_content, str):
                text_parts.append(raw_content)

            content = "\n".join(t for t in text_parts if t).strip()
            if tool_result_text:
                # Tool results come as user messages with <tool_response> wrapper
                role = "tool"
                content = tool_result_text

            norm: dict[str, Any] = {"role": role, "content": content}
            if tool_calls:
                norm["tool_calls"] = tool_calls
            converted.append(norm)

        converted = normalize_messages(converted)
        if converted:
            messages_out.append(converted)
    return messages_out, None


CONVERTERS = {
    "toolmind": convert_toolmind,
    "open_swe_traces": convert_open_swe_traces,
    "swe_zero": convert_swe_zero,
    "terminaltraj": convert_terminaltraj,
    "openresearcher": convert_openresearcher,
}


# ---------------------------------------------------------------------------
# Main processing logic
# ---------------------------------------------------------------------------

def process_dataset(
    name: str,
    n_samples: int,
    output_dir: str,
    seed: int = 42,
    val_ratio: float = 0.0,
) -> dict[str, int]:
    """Download, convert and save a single dataset.

    Returns a dict with ``train`` and ``val`` sample counts.
    """
    info = DATASET_REGISTRY[name]
    converter = CONVERTERS[name]
    print(f"\n{'=' * 70}")
    print(f"Processing [{name}]: {info['description']}")
    print(f"  repo   = {info['repo']}")
    print(f"  config = {info['config']}")
    print(f"  split  = {info['split']}")
    print(f"  n      = {n_samples}")
    print(f"{'=' * 70}")

    # Load from HuggingFace Hub
    load_kwargs: dict[str, Any] = {"path": info["repo"], "split": info["split"]}
    if info["config"]:
        load_kwargs["name"] = info["config"]
    print(f"  Loading dataset from HuggingFace ...")
    ds = load_dataset(**load_kwargs)
    rows = list(ds)
    print(f"  Loaded {len(rows)} rows")

    # Sample
    rows = sample_dataset(rows, n_samples, seed=seed)
    print(f"  Sampled {len(rows)} rows")

    # Convert
    print(f"  Converting to verl format ...")
    messages_list, tools_list = converter(rows)
    print(f"  Converted {len(messages_list)} valid samples")

    if not messages_list:
        print(f"  WARNING: no valid samples for [{name}], skipping.")
        return {"train": 0, "val": 0}

    # Ensure a consistent tools column (empty list per row if dataset has no tools)
    if tools_list is None:
        tools_list = [[] for _ in range(len(messages_list))]

    # Build dataframe
    df = to_dataframe(messages_list, tools_list)

    # Split into train / val
    n_val = int(len(df) * val_ratio) if val_ratio > 0 else 0
    if n_val > 0:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        val_df = df.iloc[:n_val].reset_index(drop=True)
        train_df = df.iloc[n_val:].reset_index(drop=True)
    else:
        train_df = df
        val_df = df.iloc[:0]

    # Save
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, f"{name}_train.parquet")
    train_df.to_parquet(train_path)
    print(f"  Saved train ({len(train_df)} rows) -> {train_path}")

    val_path = None
    if n_val > 0 and len(val_df) > 0:
        val_path = os.path.join(output_dir, f"{name}_val.parquet")
        val_df.to_parquet(val_path)
        print(f"  Saved val   ({len(val_df)} rows) -> {val_path}")

    return {"train": len(train_df), "val": len(val_df)}


def merge_parquets(output_dir: str, names: list[str], val_ratio: float) -> None:
    """Merge per-dataset parquets into combined train/val parquets."""
    train_dfs, val_dfs = [], []
    for name in names:
        train_path = os.path.join(output_dir, f"{name}_train.parquet")
        if os.path.exists(train_path):
            train_dfs.append(pd.read_parquet(train_path))
        if val_ratio > 0:
            val_path = os.path.join(output_dir, f"{name}_val.parquet")
            if os.path.exists(val_path):
                val_dfs.append(pd.read_parquet(val_path))

    if train_dfs:
        combined = pd.concat(train_dfs, ignore_index=True)
        combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)
        combined_path = os.path.join(output_dir, "train.parquet")
        combined.to_parquet(combined_path)
        print(f"\nCombined train: {len(combined)} rows -> {combined_path}")

    if val_dfs:
        combined_val = pd.concat(val_dfs, ignore_index=True)
        combined_val = combined_val.sample(frac=1.0, random_state=42).reset_index(drop=True)
        combined_val_path = os.path.join(output_dir, "val.parquet")
        combined_val.to_parquet(combined_val_path)
        print(f"Combined val:   {len(combined_val)} rows -> {combined_val_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare agent SFT datasets for verl training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--output_dir", type=str, default="~/data/agent_sft",
                        help="Output directory for parquet files.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_REGISTRY.keys()),
                        choices=list(DATASET_REGISTRY.keys()),
                        help="Which datasets to process (default: all).")
    # Per-dataset sample counts (-1 = use full dataset)
    parser.add_argument("--toolmind_n", type=int, default=-1, help="ToolMind samples (-1 = all).")
    parser.add_argument("--open_swe_traces_n", type=int, default=-1, help="Open-SWE-Traces samples (-1 = all).")
    parser.add_argument("--swe_zero_n", type=int, default=-1, help="SWE-Zero samples (-1 = all).")
    parser.add_argument("--terminaltraj_n", type=int, default=-1, help="TerminalTraj samples (-1 = all).")
    parser.add_argument("--openresearcher_n", type=int, default=-1, help="OpenResearcher samples (-1 = all).")
    parser.add_argument("--val_ratio", type=float, default=0.02,
                        help="Fraction of each dataset to reserve for validation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--no_merge", action="store_true",
                        help="Do not merge per-dataset parquets into combined train/val.")

    args = parser.parse_args()
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Map dataset names to their sample counts
    sample_counts = {
        "toolmind": args.toolmind_n,
        "open_swe_traces": args.open_swe_traces_n,
        "swe_zero": args.swe_zero_n,
        "terminaltraj": args.terminaltraj_n,
        "openresearcher": args.openresearcher_n,
    }

    print(f"Output directory: {output_dir}")
    print(f"Datasets to process: {args.datasets}")
    print(f"Validation ratio: {args.val_ratio}")

    stats: dict[str, dict[str, int]] = {}
    for name in args.datasets:
        try:
            stats[name] = process_dataset(
                name=name,
                n_samples=sample_counts[name],
                output_dir=output_dir,
                seed=args.seed,
                val_ratio=args.val_ratio,
            )
        except Exception as e:
            print(f"\nERROR processing [{name}]: {e}")
            traceback.print_exc()
            stats[name] = {"train": 0, "val": 0}

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total_train, total_val = 0, 0
    for name, counts in stats.items():
        print(f"  {name:20s}  train={counts['train']:>8d}  val={counts['val']:>6d}")
        total_train += counts["train"]
        total_val += counts["val"]
    print(f"  {'TOTAL':20s}  train={total_train:>8d}  val={total_val:>6d}")

    # Merge
    if not args.no_merge:
        print("\nMerging parquets ...")
        merge_parquets(output_dir, args.datasets, args.val_ratio)

    print("\nDone! Data is ready at:", output_dir)


if __name__ == "__main__":
    main()
