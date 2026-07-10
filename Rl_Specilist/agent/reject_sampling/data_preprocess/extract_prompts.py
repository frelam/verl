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
"""Extract prompts from 4 agent SFT datasets for reject sampling rollout.

This script reads the raw datasets downloaded by ``download_datasets.sh`` and
produces verl-compatible rollout parquet files. Each sample contains:

* ``prompt`` — list of ``{role, content}`` dicts (system + user message only,
  no assistant responses, so the policy generates fresh trajectories).
* ``tools`` — list of OpenAI function schemas (ToolMind / Open-SWE-Traces have
  them; TerminalTraj / SWE-Zero get an empty list).
* ``data_source`` — string identifier used by the reward function to route
  verification logic.
* ``reward_model`` — ``{style, ground_truth}`` where ``ground_truth`` is
  extracted from the *original* trajectory's last assistant turn.
* ``extra_info`` — dataset-specific metadata (verification type, repo info,
  original trajectory, etc.) consumed by ``judge_reward.compute_score``.

The output format matches ``Rl_Specilist/agent/RL/data_preprocess/prepare_math_multiturn.py``
so it can be fed directly into ``verl.trainer.main_ppo``.

Usage::

    python -m Rl_Specilist.agent.reject_sampling.data_preprocess.extract_prompts \\
        --raw_dir ~/data/reject_sampling/raw \\
        --output_dir ~/data/reject_sampling/prompts \\
        --max_samples 500 \\
        --datasets toolmind terminaltraj open_swe_traces swe_zero
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Optional

import pandas as pd
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Dataset registry — (hf_repo, config, split)
# Mirrors examples/data_preprocess/prepare_agent_sft_data.py:67-98
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
}

# Default system prompt for agent rollout (no tools specified yet — the
# ToolAgentLoop will inject tool schemas via chat_template).
DEFAULT_SYSTEM_PROMPT = (
    "You are an AI assistant with tool access. "
    "Reason step by step inside <think>...</think> before calling tools. "
    "Call the appropriate tool when needed, observe the result, and continue "
    "until the task is complete. Be concise and precise."
)


# ---------------------------------------------------------------------------
# Helpers (adapted from prepare_agent_sft_data.py)
# ---------------------------------------------------------------------------

def _clean_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


ROLE_MAP = {
    "human": "user", "user": "user",
    "gpt": "assistant", "assistant": "assistant", "model": "assistant",
    "ai": "assistant", "chatgpt": "assistant",
    "system": "system",
    "tool": "tool", "function": "tool", "observation": "tool",
}


def _normalize_role(raw_role: str) -> Optional[str]:
    return ROLE_MAP.get((raw_role or "").strip().lower())


def _normalize_messages(messages: list[dict]) -> list[dict]:
    """Normalize a list of raw messages to {role, content} dicts."""
    result = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = _normalize_role(msg.get("role", msg.get("from", "")))
        if role is None:
            continue
        content = _clean_content(msg.get("content", msg.get("value", "")))
        out: dict[str, Any] = {"role": role, "content": content}
        # Preserve tool_calls for assistant messages
        if msg.get("tool_calls"):
            out["tool_calls"] = msg["tool_calls"]
        result.append(out)
    return result


def _parse_tools_field(tools: Any) -> list[dict]:
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


def _extract_prompt_messages(raw_messages: list[dict]) -> list[dict]:
    """Extract the prompt portion (system + first user) from a full trajectory.

    Returns a list like ``[{role: system}, {role: user}]``.
    """
    # Find first user message
    first_user = None
    for msg in raw_messages:
        if msg.get("role") == "user" and msg.get("content"):
            first_user = msg
            break

    if first_user is None:
        return []

    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": first_user["content"]},
    ]


def _extract_ground_truth(raw_messages: list[dict]) -> str:
    """Extract the ground-truth answer from the original trajectory.

    Strategy: take the last assistant message's content (this is the
    "correct" answer from the original dataset).
    """
    for msg in reversed(raw_messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            content = msg["content"]
            # Truncate very long ground truths (they're just for judge reference)
            if len(content) > 2000:
                content = content[:2000] + "...[truncated]"
            return content
    return ""


# ---------------------------------------------------------------------------
# Per-dataset extractors
# ---------------------------------------------------------------------------

def extract_toolmind(rows: list[dict]) -> list[dict]:
    """ToolMind: conversations + tools → prompt + ground_truth."""
    out = []
    for row in rows:
        conversations = row.get("conversations")
        if not conversations:
            continue
        messages = _normalize_messages(conversations)
        if not messages:
            continue
        prompt = _extract_prompt_messages(messages)
        if not prompt:
            continue
        ground_truth = _extract_ground_truth(messages)
        tools = _parse_tools_field(row.get("tools"))
        out.append({
            "prompt": prompt,
            "tools": tools,
            "data_source": "toolmind",
            "agent_name": "custom_hermes",  # route to Hermes runner
            "tools_kwargs": {
                "data_source": "toolmind",
                "agent_name": "custom_hermes",
            },
            "reward_model": {
                "style": "judge",
                "ground_truth": ground_truth,
            },
            "extra_info": {
                "dataset": "toolmind",
                "verification_type": "judge",
                "original_trajectory_length": len(messages),
            },
        })
    return out


def extract_open_swe_traces(rows: list[dict]) -> list[dict]:
    """Open-SWE-Traces: trajectory + tools → prompt + ground_truth."""
    out = []
    for row in rows:
        trajectory = row.get("trajectory")
        if not trajectory:
            continue
        messages = _normalize_messages(trajectory)
        if not messages:
            continue
        prompt = _extract_prompt_messages(messages)
        if not prompt:
            continue
        ground_truth = _extract_ground_truth(messages)
        tools = _parse_tools_field(row.get("tools"))
        out.append({
            "prompt": prompt,
            "tools": tools,
            "data_source": "open_swe_traces",
            "agent_name": "custom_claude",  # route to Claude Code runner
            "tools_kwargs": {
                "data_source": "open_swe_traces",
                "agent_name": "custom_claude",
                "repo": row.get("repo", ""),
                "base_commit": row.get("base_commit", row.get("instance_id", "")),
                "instance_id": row.get("instance_id", ""),
            },
            "reward_model": {
                "style": "test",
                "ground_truth": ground_truth,
            },
            "extra_info": {
                "dataset": "open_swe_traces",
                "verification_type": "test",
                "repo": row.get("repo", ""),
                "base_commit": row.get("base_commit", row.get("instance_id", "")),
                "instance_id": row.get("instance_id", ""),
                "original_trajectory_length": len(messages),
            },
        })
    return out


def extract_swe_zero(rows: list[dict]) -> list[dict]:
    """SWE-Zero: trajectory (no tools) → prompt + ground_truth."""
    out = []
    for row in rows:
        trajectory = row.get("trajectory")
        if not trajectory:
            continue
        messages = _normalize_messages(trajectory)
        if not messages:
            continue
        prompt = _extract_prompt_messages(messages)
        if not prompt:
            continue
        ground_truth = _extract_ground_truth(messages)
        out.append({
            "prompt": prompt,
            "tools": [],
            "data_source": "swe_zero",
            "agent_name": "custom_claude",  # route to Claude Code runner
            "tools_kwargs": {
                "data_source": "swe_zero",
                "agent_name": "custom_claude",
                "repo": row.get("repo", ""),
                "base_commit": row.get("base_commit", row.get("instance_id", "")),
                "instance_id": row.get("instance_id", ""),
            },
            "reward_model": {
                "style": "judge",  # execution-free, fall back to judge
                "ground_truth": ground_truth,
            },
            "extra_info": {
                "dataset": "swe_zero",
                "verification_type": "judge",
                "repo": row.get("repo", ""),
                "base_commit": row.get("base_commit", row.get("instance_id", "")),
                "instance_id": row.get("instance_id", ""),
                "original_trajectory_length": len(messages),
            },
        })
    return out


def extract_terminaltraj(rows: list[dict]) -> list[dict]:
    """TerminalTraj: messages → prompt + ground_truth."""
    out = []
    for row in rows:
        messages = row.get("messages")
        if not messages:
            continue
        messages = _normalize_messages(messages)
        if not messages:
            continue
        prompt = _extract_prompt_messages(messages)
        if not prompt:
            continue
        ground_truth = _extract_ground_truth(messages)
        out.append({
            "prompt": prompt,
            "tools": [],  # TerminalTraj uses a bash tool configured at rollout time
            "data_source": "terminaltraj",
            "agent_name": "custom_hermes",  # route to Hermes runner
            "tools_kwargs": {
                "data_source": "terminaltraj",
                "agent_name": "custom_hermes",
            },
            "reward_model": {
                "style": "judge",
                "ground_truth": ground_truth,
            },
            "extra_info": {
                "dataset": "terminaltraj",
                "verification_type": "judge",
                "original_trajectory_length": len(messages),
            },
        })
    return out


EXTRACTORS = {
    "toolmind": extract_toolmind,
    "open_swe_traces": extract_open_swe_traces,
    "swe_zero": extract_swe_zero,
    "terminaltraj": extract_terminaltraj,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _sample_rows(rows: list[dict], n: int, seed: int = 42) -> list[dict]:
    if n <= 0 or n >= len(rows):
        return rows
    rng = random.Random(seed)
    indices = rng.sample(range(len(rows)), n)
    return [rows[i] for i in indices]


def process_dataset(
    name: str,
    raw_dir: str,
    output_dir: str,
    max_samples: int,
    seed: int = 42,
) -> dict[str, int]:
    """Load, extract, and save a single dataset."""
    info = DATASET_REGISTRY[name]
    extractor = EXTRACTORS[name]

    # Try local path first (from download_datasets.sh), fall back to HF Hub
    local_path = os.path.join(raw_dir, name)
    print(f"\n{'=' * 70}")
    print(f"Processing [{name}]: {info['description']}")
    print(f"  local_path = {local_path}")
    print(f"  hf_repo    = {info['repo']}")
    print(f"  max_samples = {max_samples}")
    print(f"{'=' * 70}")

    load_kwargs: dict[str, Any] = {"split": info["split"]}
    if info["config"]:
        load_kwargs["name"] = info["config"]

    if os.path.exists(local_path) and os.listdir(local_path):
        print(f"  Loading from local path...")
        # Remove split from kwargs for local load; load_dataset handles it
        kwargs = {"path": local_path}
        if info["config"]:
            kwargs["name"] = info["config"]
        ds = load_dataset(**kwargs)
        if isinstance(ds, dict):
            data = ds.get(info["split"], ds[list(ds.keys())[0]])
        else:
            data = ds
    else:
        print(f"  Loading from HuggingFace Hub...")
        kwargs = {"path": info["repo"]}
        if info["config"]:
            kwargs["name"] = info["config"]
        ds = load_dataset(**kwargs)
        if isinstance(ds, dict):
            data = ds.get(info["split"], ds[list(ds.keys())[0]])
        else:
            data = ds

    rows = list(data)
    print(f"  Loaded {len(rows)} rows")

    rows = _sample_rows(rows, max_samples, seed=seed)
    print(f"  Sampled {len(rows)} rows")

    print(f"  Extracting prompts...")
    samples = extractor(rows)
    print(f"  Extracted {len(samples)} valid samples")

    if not samples:
        print(f"  WARNING: no valid samples for [{name}], skipping.")
        return {"train": 0, "val": 0}

    # Save as parquet
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(samples)
    out_path = os.path.join(output_dir, f"{name}.parquet")
    df.to_parquet(out_path)
    print(f"  Saved {len(df)} samples -> {out_path}")

    # Print a sample for verification
    if len(samples) > 0:
        s = samples[0]
        print(f"\n  Sample [0]:")
        print(f"    data_source: {s['data_source']}")
        print(f"    agent_name:  {s.get('agent_name', 'N/A')}")
        print(f"    prompt roles: {[m['role'] for m in s['prompt']]}")
        print(f"    user content (first 200 chars): {s['prompt'][-1]['content'][:200]!r}")
        print(f"    tools count: {len(s['tools'])}")
        print(f"    ground_truth (first 200 chars): {s['reward_model']['ground_truth'][:200]!r}")

    return {"train": len(samples), "val": 0}


def merge_parquets(output_dir: str, names: list[str]) -> None:
    """Merge per-dataset parquets into a combined train.parquet."""
    dfs = []
    for name in names:
        path = os.path.join(output_dir, f"{name}.parquet")
        if os.path.exists(path):
            dfs.append(pd.read_parquet(path))

    if not dfs:
        return

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)
    combined_path = os.path.join(output_dir, "train.parquet")
    combined.to_parquet(combined_path)
    print(f"\nCombined train: {len(combined)} rows -> {combined_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract prompts from agent datasets for reject sampling rollout.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--raw_dir",
        default=os.path.expanduser("~/data/reject_sampling/raw"),
        help="Directory containing downloaded raw datasets.",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.expanduser("~/data/reject_sampling/prompts"),
        help="Output directory for parquet files.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_REGISTRY.keys()),
        choices=list(DATASET_REGISTRY.keys()),
        help="Which datasets to process (default: all).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=500,
        help="Max samples per dataset (-1 = all). Default 500 for quick testing.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_merge", action="store_true", help="Don't merge into combined train.parquet")
    args = parser.parse_args()

    print(f"Raw dir:    {args.raw_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Datasets:   {args.datasets}")
    print(f"Max samples: {args.max_samples}")

    os.makedirs(args.output_dir, exist_ok=True)

    stats = {}
    for name in args.datasets:
        try:
            stats[name] = process_dataset(
                name=name,
                raw_dir=args.raw_dir,
                output_dir=args.output_dir,
                max_samples=args.max_samples,
                seed=args.seed,
            )
        except Exception as e:
            print(f"\nERROR processing [{name}]: {e}")
            import traceback
            traceback.print_exc()
            stats[name] = {"train": 0, "val": 0}

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total = 0
    for name, counts in stats.items():
        n = counts["train"]
        print(f"  {name:20s}  {n:>8d} samples")
        total += n
    print(f"  {'TOTAL':20s}  {total:>8d} samples")

    if not args.no_merge:
        print("\nMerging parquets...")
        merge_parquets(args.output_dir, args.datasets)

    print(f"\nDone! Prompts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
