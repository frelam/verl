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
Preprocess the AgentInstruct dataset to parquet format for verl SFT training.

Dataset: https://huggingface.co/datasets/zai-org/AgentInstruct

Usage:
    python examples/data_preprocess/agent_instruct.py \
        --local_save_dir ~/data/agent_instruct

    # With custom split ratio:
    python examples/data_preprocess/agent_instruct.py \
        --local_save_dir ~/data/agent_instruct \
        --val_ratio 0.1
"""

import argparse
import os

import pandas as pd
from datasets import load_dataset


def convert_to_messages(example):
    """
    Convert a single example to verl-compatible messages format.

    The AgentInstruct dataset uses multi-turn ReAct trajectories.
    Each sample has a 'conversations' field containing a list of turns
    with 'from' (role) and 'value' (content) keys.

    Expected output format:
        {
            "messages": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
                ...
            ]
        }
    """
    messages = []

    if "conversations" in example:
        raw_conversations = example["conversations"]
        for turn in raw_conversations:
            role = turn.get("from", turn.get("role", ""))
            content = turn.get("value", turn.get("content", ""))

            if role in ("human", "user"):
                role = "user"
            elif role in ("gpt", "assistant", "model", "ai"):
                role = "assistant"
            elif role == "system":
                role = "system"
            elif role == "tool" or role == "function":
                role = "tool"
            else:
                role = "user"

            messages.append({"role": role, "content": content})

    elif "messages" in example:
        raw_messages = example["messages"]
        for msg in raw_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            messages.append({"role": role, "content": content})

    elif "prompt" in example and "response" in example:
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["response"]},
        ]

    elif "instruction" in example:
        user_content = example["instruction"]
        if example.get("input"):
            user_content = user_content + "\n\n" + example["input"]
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example.get("output", "")},
        ]

    else:
        print(f"Warning: Unknown data format. Available keys: {list(example.keys())}")
        return None

    return {"messages": messages}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_save_dir",
        default="~/data/agent_instruct",
        help="The save directory for the preprocessed dataset.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="Ratio of data to use for validation (default: 0.05).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Maximum number of samples to use (-1 for all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split.",
    )
    args = parser.parse_args()

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    print(f"Loading dataset from zai-org/AgentInstruct...")
    dataset = load_dataset("zai-org/AgentInstruct")

    print(f"Dataset loaded. Available splits: {list(dataset.keys())}")
    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data)} samples")
        if len(split_data) > 0:
            print(f"  Features: {split_data.features}")
            print(f"  First sample keys: {list(split_data[0].keys())}")
            break

    if "train" in dataset:
        data = dataset["train"]
    else:
        split_name = list(dataset.keys())[0]
        data = dataset[split_name]
        print(f"No 'train' split found, using '{split_name}' split.")

    print(f"\nConverting {len(data)} samples to verl messages format...")
    data = data.map(convert_to_messages, remove_columns=data.column_names)
    data = data.filter(lambda x: x["messages"] is not None)

    if args.max_samples > 0 and args.max_samples < len(data):
        data = data.select(range(args.max_samples))
        print(f"Selected {args.max_samples} samples.")

    data = data.shuffle(seed=args.seed)

    if args.val_ratio > 0:
        split_data = data.train_test_split(test_size=args.val_ratio, seed=args.seed)
        train_data = split_data["train"]
        val_data = split_data["test"]
    else:
        train_data = data
        val_data = None

    train_path = os.path.join(local_save_dir, "train.parquet")
    train_data.to_parquet(train_path)
    print(f"Train dataset saved to {train_path} ({len(train_data)} samples)")

    if val_data is not None:
        val_path = os.path.join(local_save_dir, "test.parquet")
        val_data.to_parquet(val_path)
        print(f"Validation dataset saved to {val_path} ({len(val_data)} samples)")

    print("\nDone! You can now use the following in your SFT training script:")
    print(f"  data.train_files={train_path}")
    if val_data is not None:
        print(f"  data.val_files={val_path}")
    print(f"  data.messages_key=messages")
