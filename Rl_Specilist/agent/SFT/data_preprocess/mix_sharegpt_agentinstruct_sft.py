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
Mix ShareGPT and THUDM/AgentInstruct datasets for Qwen3-0.6B SFT training.

Outputs verl-compatible parquet files with a `messages` column:
    [{"role": "system"|"user"|"assistant"|"tool", "content": "..."}, ...]

Usage:
    python examples/data_preprocess/mix_sharegpt_agentinstruct_sft.py \\
        --local_save_dir ~/data/sharegpt_agentinstruct_sft

    # Custom mix ratio (ShareGPT : AgentInstruct = 2 : 1)
    python examples/data_preprocess/mix_sharegpt_agentinstruct_sft.py \\
        --sharegpt_ratio 0.67 \\
        --local_save_dir ~/data/sharegpt_agentinstruct_sft

    # Validate a few samples against Qwen3 chat template
    python examples/data_preprocess/mix_sharegpt_agentinstruct_sft.py \\
        --model_path Qwen/Qwen3-0.6B \\
        --max_samples 1000
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from typing import Any, Optional

from datasets import concatenate_datasets, load_dataset

# Default system prompt for agent/tool SFT (not present in raw AgentInstruct / most ShareGPT).
DEFAULT_SYSTEM_PROMPT = "You are an AI assistant with tool access."

# Qwen3 / Qwen2 chat-template markers that must not appear in raw message content.
QWEN_SPECIAL_TOKEN_PATTERNS = [
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"<\|redacted_im_end\|>",
    r"<\|endoftext\|>",
    r"<\|object_ref_start\|>",
    r"<\|object_ref_end\|>",
    r"<\|box_start\|>",
    r"<\|box_end\|>",
    r"<\|quad_start\|>",
    r"<\|quad_end\|>",
    r"<\|vision_start\|>",
    r"<\|vision_end\|>",
    r"<\|vision_pad\|>",
    r"<\|image_pad\|>",
    r"<\|video_pad\|>",
    r"<\|fim_prefix\|>",
    r"<\|fim_middle\|>",
    r"<\|fim_suffix\|>",
    r"<\|fim_pad\|>",
    r"<\|repo_name\|>",
    r"<\|file_sep\|>",
    r"<tool_call>",
    r"</tool_call>",
    r"<tool_response>",
    r"</tool_response>",
    r"<think>",
    r"</think>",
    # Legacy Qwen2 markers occasionally leaked in scraped data
    r"<\|im_end\|>",
]

# ANSI / terminal control sequences (common in AgentInstruct OS trajectories).
ANSI_ESCAPE_RE = re.compile(
    r"\x1b\[[0-9;]*[A-Za-z]"  # CSI sequences
    r"|\x1b\][^\x07]*(?:\x07|\x1b\\)"  # OSC sequences (e.g. window title)
    r"|\x1b[PX^_][^\x1b]*\x1b\\"  # other ESC sequences
    r"|\x1b."  # fallback single ESC
)

# Other control chars except common whitespace.
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

QWEN_TOKEN_RE = re.compile("|".join(QWEN_SPECIAL_TOKEN_PATTERNS), flags=re.IGNORECASE)

ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "model": "assistant",
    "ai": "assistant",
    "chatgpt": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
    "observation": "tool",
}


def sanitize_content(text: str) -> str:
    """Clean raw text so Qwen3 chat_template can wrap it without token leakage."""
    if not isinstance(text, str):
        text = str(text)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = ANSI_ESCAPE_RE.sub("", text)
    text = CONTROL_CHAR_RE.sub("", text)
    text = QWEN_TOKEN_RE.sub("", text)

    # Collapse excessive blank lines introduced by stripping.
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def normalize_role(raw_role: str) -> Optional[str]:
    role = (raw_role or "").strip().lower()
    return ROLE_MAP.get(role)


def conversations_to_messages(
    conversations: list[dict],
    *,
    respect_agentinstruct_loss: bool = True,
    min_assistant_turns: int = 1,
) -> Optional[list[dict]]:
    """Convert ShareGPT / AgentInstruct `conversations` to OpenAI-style messages."""
    messages: list[dict] = []

    for turn in conversations:
        if not isinstance(turn, dict):
            continue

        raw_role = turn.get("from", turn.get("role", ""))
        role = normalize_role(raw_role)
        if role is None:
            continue

        content = turn.get("value", turn.get("content", ""))
        if content is None:
            content = ""
        content = sanitize_content(content)
        if not content and role != "system":
            continue

        if respect_agentinstruct_loss and role == "assistant" and "loss" in turn:
            loss_flag = turn["loss"]
            if loss_flag is False:
                continue

        messages.append({"role": role, "content": content})

    if not messages:
        return None

    assistant_turns = sum(1 for m in messages if m["role"] == "assistant")
    if assistant_turns < min_assistant_turns:
        return None

    # Require alternating user/tool input and assistant output for stable SFT.
    if messages[0]["role"] not in ("system", "user", "tool"):
        return None

    return messages


def ensure_system_prompt(
    messages: list[dict],
    system_prompt: str,
    *,
    replace_existing: bool = False,
) -> list[dict]:
    """
    Prepend or update system message.

    AgentInstruct and most ShareGPT samples do not ship a dedicated system turn;
    task instructions live in the first user message. We add a global system prompt
    for Qwen3 SFT unless one already exists.
    """
    if not system_prompt:
        return messages

    if messages and messages[0]["role"] == "system":
        if replace_existing:
            messages[0]["content"] = system_prompt
        return messages

    return [{"role": "system", "content": system_prompt}] + messages


def process_example(
    example: dict,
    *,
    data_source: str,
    system_prompt: str,
    replace_system: bool,
    respect_agentinstruct_loss: bool,
    min_assistant_turns: int,
) -> Optional[dict]:
    messages: Optional[list[dict]] = None
    respect_loss = respect_agentinstruct_loss if data_source == "agentinstruct" else False

    if "conversations" in example and example["conversations"]:
        messages = conversations_to_messages(
            example["conversations"],
            respect_agentinstruct_loss=respect_loss,
            min_assistant_turns=min_assistant_turns,
        )
    elif "messages" in example and example["messages"]:
        converted = []
        for msg in example["messages"]:
            role = normalize_role(msg.get("role", ""))
            if role is None:
                continue
            content = sanitize_content(msg.get("content", ""))
            if content or role == "system":
                converted.append({"role": role, "content": content})
        if sum(1 for m in converted if m["role"] == "assistant") >= min_assistant_turns:
            messages = converted

    if not messages:
        return None

    messages = ensure_system_prompt(messages, system_prompt, replace_existing=replace_system)

    return {
        "messages": messages,
        "data_source": data_source,
    }


def load_sharegpt_dataset(
    dataset_name: str,
    dataset_config: Optional[str],
    local_dataset_path: Optional[str],
    max_samples: int,
) -> Any:
    print(f"Loading ShareGPT dataset: {dataset_name}" + (f" ({dataset_config})" if dataset_config else ""))
    if local_dataset_path:
        ds = load_dataset(local_dataset_path, dataset_config, split="train")
    else:
        kwargs = {"path": dataset_name}
        if dataset_config:
            kwargs["name"] = dataset_config
        ds = load_dataset(**kwargs)
        if isinstance(ds, dict):
            if "train" in ds:
                ds = ds["train"]
            else:
                ds = ds[list(ds.keys())[0]]
    if max_samples > 0 and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    print(f"  ShareGPT samples: {len(ds)}")
    return ds


def load_agentinstruct_dataset(local_dataset_path: Optional[str], max_samples: int) -> Any:
    dataset_name = "THUDM/AgentInstruct"
    print(f"Loading AgentInstruct dataset: {dataset_name}")
    if local_dataset_path:
        ds_dict = load_dataset(local_dataset_path)
    else:
        ds_dict = load_dataset(dataset_name)

    splits = []
    for split_name, split_ds in ds_dict.items():
        print(f"  split {split_name}: {len(split_ds)}")
        split_ds = split_ds.add_column("data_source", ["agentinstruct"] * len(split_ds))
        splits.append(split_ds)

    ds = concatenate_datasets(splits)
    if max_samples > 0 and len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))
    print(f"  AgentInstruct total samples: {len(ds)}")
    return ds


def map_dataset(ds: Any, data_source: str, num_proc: int, **process_kwargs) -> Any:
    def _fn(example):
        return process_example(example, data_source=data_source, **process_kwargs)

    mapped = ds.map(
        _fn,
        num_proc=num_proc,
        remove_columns=ds.column_names,
        desc=f"Processing {data_source}",
    )
    mapped = mapped.filter(lambda x: x["messages"] is not None)
    return mapped


def mix_datasets(
    sharegpt_ds: Any,
    agentinstruct_ds: Any,
    sharegpt_ratio: float,
    seed: int,
) -> Any:
    """Mix two datasets by target ratio (ShareGPT weight)."""
    n_sharegpt = len(sharegpt_ds)
    n_agent = len(agentinstruct_ds)

    if sharegpt_ratio <= 0:
        mixed = agentinstruct_ds
    elif sharegpt_ratio >= 1:
        mixed = sharegpt_ds
    else:
        # Target: n_sharegpt / (n_sharegpt + n_agent_sel) ≈ sharegpt_ratio
        if n_sharegpt == 0:
            mixed = agentinstruct_ds
        elif n_agent == 0:
            mixed = sharegpt_ds
        else:
            agent_target = int(n_sharegpt * (1 - sharegpt_ratio) / sharegpt_ratio)
            agent_target = max(1, min(agent_target, n_agent))
            sharegpt_part = sharegpt_ds
            agent_part = agentinstruct_ds.shuffle(seed=seed).select(range(agent_target))
            mixed = concatenate_datasets([sharegpt_part, agent_part])
            print(
                f"  Mix: ShareGPT={len(sharegpt_part)}, AgentInstruct={len(agent_part)} "
                f"(ratio≈{len(sharegpt_part) / len(mixed):.2%} ShareGPT)"
            )

    return mixed.shuffle(seed=seed)


def validate_with_tokenizer(samples: list[dict], model_path: str, max_check: int = 5) -> None:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("  [validate] transformers not installed, skip chat template check.")
        return

    print(f"  [validate] Loading tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    for i, sample in enumerate(samples[:max_check]):
        messages = sample["messages"]
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            print(f"  [validate] sample {i}: {len(ids)} tokens, preview={text[:120]!r}...")
        except Exception as e:
            print(f"  [validate] sample {i} FAILED: {e}")


def main():
    parser = argparse.ArgumentParser(description="Mix ShareGPT + AgentInstruct for Qwen3 SFT")
    parser.add_argument(
        "--local_save_dir",
        default="~/data/sharegpt_agentinstruct_sft",
        help="Directory for train.parquet / test.parquet",
    )
    parser.add_argument("--sharegpt_dataset", default="mhgcut/ShareGPT52K", help="HF ShareGPT dataset id")
    parser.add_argument("--sharegpt_config", default=None, help="HF dataset config name (optional)")
    parser.add_argument("--sharegpt_local_path", default=None, help="Local path for ShareGPT dataset")
    parser.add_argument("--agentinstruct_local_path", default=None, help="Local path for AgentInstruct dataset")
    parser.add_argument(
        "--sharegpt_ratio",
        type=float,
        default=0.5,
        help="Target fraction of ShareGPT in the mixed train set (0~1). Default 0.5.",
    )
    parser.add_argument(
        "--system_prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt prepended when missing from a conversation.",
    )
    parser.add_argument(
        "--replace_system",
        action="store_true",
        help="Replace existing system message with --system_prompt instead of keeping it.",
    )
    parser.add_argument(
        "--respect_agentinstruct_loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop AgentInstruct assistant turns with loss=false (default: true).",
    )
    parser.add_argument("--min_assistant_turns", type=int, default=1, help="Minimum assistant turns per sample")
    parser.add_argument("--val_ratio", type=float, default=0.02, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument(
        "--max_sharegpt_samples",
        type=int,
        default=-1,
        help="Cap ShareGPT samples (-1 = all).",
    )
    parser.add_argument(
        "--max_agentinstruct_samples",
        type=int,
        default=-1,
        help="Cap AgentInstruct samples (-1 = all).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Cap final mixed dataset (-1 = all). For quick tests.",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Optional Qwen3 path to validate chat template on a few samples.",
    )
    parser.add_argument("--skip_sharegpt", action="store_true")
    parser.add_argument("--skip_agentinstruct", action="store_true")
    args = parser.parse_args()

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    process_kwargs = dict(
        system_prompt=args.system_prompt,
        replace_system=args.replace_system,
        respect_agentinstruct_loss=args.respect_agentinstruct_loss,
        min_assistant_turns=args.min_assistant_turns,
    )

    datasets_to_mix = []

    if not args.skip_sharegpt:
        raw_sharegpt = load_sharegpt_dataset(
            args.sharegpt_dataset,
            args.sharegpt_config,
            args.sharegpt_local_path,
            args.max_sharegpt_samples,
        )
        proc_sharegpt = map_dataset(raw_sharegpt, "sharegpt", args.num_proc, **process_kwargs)
        datasets_to_mix.append(proc_sharegpt)

    if not args.skip_agentinstruct:
        raw_agent = load_agentinstruct_dataset(args.agentinstruct_local_path, args.max_agentinstruct_samples)
        proc_agent = map_dataset(raw_agent, "agentinstruct", args.num_proc, **process_kwargs)
        datasets_to_mix.append(proc_agent)

    if not datasets_to_mix:
        raise ValueError("Both datasets skipped; nothing to process.")

    if len(datasets_to_mix) == 2:
        mixed = mix_datasets(datasets_to_mix[0], datasets_to_mix[1], args.sharegpt_ratio, args.seed)
    else:
        mixed = datasets_to_mix[0]

    if args.max_samples > 0 and len(mixed) > args.max_samples:
        mixed = mixed.shuffle(seed=args.seed).select(range(args.max_samples))

    source_counts = Counter(mixed["data_source"])
    print("\nMixed dataset distribution:")
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count} ({100.0 * count / len(mixed):.1f}%)")
    print(f"  total: {len(mixed)}")

    if args.model_path:
        validate_with_tokenizer([mixed[i] for i in range(min(5, len(mixed)))], args.model_path)

    if args.val_ratio > 0:
        split = mixed.train_test_split(test_size=args.val_ratio, seed=args.seed)
        train_data, val_data = split["train"], split["test"]
    else:
        train_data, val_data = mixed, None

    train_path = os.path.join(local_save_dir, "train.parquet")
    train_data.to_parquet(train_path)
    print(f"\nTrain saved: {train_path} ({len(train_data)} samples)")

    if val_data is not None:
        val_path = os.path.join(local_save_dir, "test.parquet")
        val_data.to_parquet(val_path)
        print(f"Val saved:   {val_path} ({len(val_data)} samples)")

    print("\nSFT training example (Qwen3-0.6B):")
    print(f"  data.train_files={train_path}")
    if val_data is not None:
        print(f"  data.val_files={os.path.join(local_save_dir, 'test.parquet')}")
    print("  data.messages_key=messages")
    print("  data.ignore_input_ids_mismatch=True")
    print(f"  model.path=Qwen/Qwen3-0.6B")


if __name__ == "__main__":
    main()
