# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Preprocess GSM8K + MATH datasets for agentic multi-turn RL.

This script produces a parquet dataset in the format expected by verl's
``ToolAgentLoop``. Each sample carries:

* a system prompt that teaches the agent the ``Know -> Search -> Verify ->
  Revise -> Answer`` workflow;
* a user prompt with the question;
* ``tools_kwargs`` wiring the ground-truth into ``submit_answer`` so the
  tool can give correctness feedback during rollout;
* ``reward_model.ground_truth`` for the final outcome reward.

Run::

    python -m Rl_Specilist.agent.RL.data_preprocess.prepare_math_multiturn \
        --local_save_dir ~/data/agentic_math \
        --datasets gsm8k math

This covers capability 1 (format protocol), 2 (tool routing), 3 (planning),
and 6 (failure recovery) from the training plan.
"""

import argparse
import os
import re

import datasets

SYSTEM_PROMPT = (
    "You are a careful math problem solver. "
    "For every problem, follow this workflow:\n"
    "1. <think> Reason about the problem and decide whether you can solve it "
    "mentally or need a tool.\n"
    "2. If you need a precise computation, call the `calculator` tool with a "
    "well-formed expression.\n"
    "3. When you are confident, call `submit_answer` with your answer and an "
    "honest confidence score (0.0-1.0).\n"
    "4. If the tool tells you the answer is incorrect, reflect on what went "
    "wrong, recompute, and resubmit.\n"
    "Do not guess blindly. If you are unsure, lower your confidence. "
    "Always output your reasoning inside <think>...</think> before any tool call."
)

GSM8K_INSTRUCTION = (
    "Solve the problem step by step. Use the `calculator` tool if you need "
    "precise arithmetic. When ready, call `submit_answer` with your final "
    "numerical answer and a confidence score."
)

MATH_INSTRUCTION = (
    "Solve the problem step by step. Use the `calculator` tool if you need "
    "precise arithmetic. Put your final answer in \\boxed{} and call "
    "`submit_answer` with the boxed answer and a confidence score."
)


def extract_gsm8k_solution(answer_str: str) -> str:
    match = re.search(r"#### (\-?[0-9\.\,]+)", answer_str)
    if match is None:
        return answer_str.strip().split("\n")[-1]
    return match.group(0).split("#### ")[1].replace(",", "")


def extract_math_solution(solution_str: str) -> str:
    match = re.search(r"\\boxed\{(.*)\}", solution_str)
    if match:
        return match.group(1)
    return solution_str.strip().split("\n")[-1]


def make_gsm8k_processor(data_source: str):
    def process_fn(example, idx, split):
        question_raw = example["question"]
        answer_raw = example["answer"]
        solution = extract_gsm8k_solution(answer_raw)
        question = question_raw + "\n\n" + GSM8K_INSTRUCTION
        return {
            "data_source": data_source,
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer_raw,
                "question": question_raw,
                "task_type": "gsm8k",
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "submit_answer": {
                        "create_kwargs": {
                            "ground_truth": solution,
                            "task_type": "gsm8k",
                        }
                    }
                },
            },
        }

    return process_fn


def make_math_processor(data_source: str):
    def process_fn(example, idx, split):
        problem_raw = example["problem"]
        solution_raw = example["solution"]
        solution = extract_math_solution(solution_raw)
        question = problem_raw + "\n\n" + MATH_INSTRUCTION
        return {
            "data_source": data_source,
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": solution_raw,
                "question": problem_raw,
                "task_type": "math",
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "submit_answer": {
                        "create_kwargs": {
                            "ground_truth": solution,
                            "task_type": "math",
                        }
                    }
                },
            },
        }

    return process_fn


DATASET_REGISTRY = {
    "gsm8k": {
        "hf_path": "openai/gsm8k",
        "subset": "main",
        "data_source": "openai/gsm8k",
        "processor": make_gsm8k_processor,
    },
    "math": {
        "hf_path": "HuggingFaceH4/MATH-500",
        "subset": None,
        "data_source": "lighteval/MATH",
        "processor": make_math_processor,
    },
}


def _generate_synthetic_gsm8k(n: int = 100):
    """Generate simple synthetic math problems for offline testing.

    Each problem has a question and a numeric answer, mimicking the GSM8K
    schema so the rest of the pipeline can run without network access.
    """
    import random

    random.seed(42)
    rows = []
    for i in range(n):
        a = random.randint(10, 99)
        b = random.randint(2, 20)
        ops = [
            ("+", lambda x, y: x + y, f"{a} + {b}"),
            ("-", lambda x, y: x - y, f"{a} - {b}"),
            ("*", lambda x, y: x * y, f"{a} * {b}"),
        ]
        op_sym, op_fn, expr = random.choice(ops)
        answer = op_fn(a, b)
        question = f"Janet has {a} apples. She {'receives' if op_sym == '+' else 'gives away' if op_sym == '-' else 'multiplies her apples by'} {b}. How many does she have now?"
        if op_sym == "*":
            question = f"If {a} students each bring {b} pencils, how many pencils are there in total?"
        solution_str = f"Question: {question}\nAnswer: {expr} = {answer}\n#### {answer}"
        rows.append({"question": question, "answer": solution_str})
    return rows


def process_dataset(name: str, local_save_dir: str, max_samples: int = -1, local_dataset_path: str = None):
    cfg = DATASET_REGISTRY[name]
    print(f"\n[{name}] Loading from {cfg['hf_path']} ...")

    if local_dataset_path:
        print(f"  Using local dataset path: {local_dataset_path}")
        if cfg["subset"]:
            ds = datasets.load_dataset(local_dataset_path, cfg["subset"])
        else:
            ds = datasets.load_dataset(local_dataset_path)
    else:
        try:
            if cfg["subset"]:
                ds = datasets.load_dataset(cfg["hf_path"], cfg["subset"])
            else:
                ds = datasets.load_dataset(cfg["hf_path"])
        except Exception as e:
            print(f"  WARNING: Could not load from HuggingFace Hub ({e})")
            print(f"  Generating synthetic {name} data for offline testing...")
            from datasets import Dataset

            rows = _generate_synthetic_gsm8k(200)
            ds = {"train": Dataset.from_list(rows[:180]), "test": Dataset.from_list(rows[180:])}
            print(f"  Generated {len(ds['train'])} train + {len(ds['test'])} test synthetic samples")

    splits = {}
    if "train" in ds:
        splits["train"] = ds["train"]
    else:
        first_key = list(ds.keys())[0]
        splits["train"] = ds[first_key]
        print(f"  No 'train' split, using '{first_key}'")

    if "test" in ds:
        splits["test"] = ds["test"]
    else:
        # If no test split, hold out 5% of train
        from datasets import Dataset

        split_data = splits["train"].train_test_split(test_size=0.05, seed=42)
        splits["train"] = split_data["train"]
        splits["test"] = split_data["test"]
        print(f"  No 'test' split, held out 5% of train ({len(splits['test'])} samples)")

    processor = cfg["processor"](cfg["data_source"])
    for split_name, split_data in splits.items():
        if max_samples > 0 and len(split_data) > max_samples:
            split_data = split_data.select(range(max_samples))
            print(f"  Limited {split_name} to {max_samples} samples")

        def fn(example, idx, _split=split_name, _proc=processor):
            return _proc(example, idx, _split)

        split_data = split_data.map(fn, with_indices=True, remove_columns=split_data.column_names)
        out_path = os.path.join(local_save_dir, name, f"{split_name}.parquet")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        split_data.to_parquet(out_path)
        print(f"  Saved {len(split_data)} samples to {out_path}")

    return cfg["data_source"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare math datasets for agentic multi-turn RL.")
    parser.add_argument(
        "--local_save_dir",
        default="~/data/agentic_math",
        help="Root directory for preprocessed parquet files.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["gsm8k", "math"],
        choices=list(DATASET_REGISTRY.keys()),
        help="Which datasets to prepare.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Maximum samples per split (-1 for all). Useful for quick smoke tests.",
    )
    parser.add_argument(
        "--local_dataset_path",
        default=None,
        help="Local path to a pre-downloaded dataset (avoids HuggingFace Hub access).",
    )
    args = parser.parse_args()

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    data_sources = []
    for name in args.datasets:
        ds = process_dataset(name, local_save_dir, args.max_samples, args.local_dataset_path)
        data_sources.append(ds)

    print("\n" + "=" * 60)
    print("Done! Data sources prepared:", data_sources)
    print(f"Output directory: {local_save_dir}")
    print("Use the generated parquet files in your training config:")
    print(f"  data.train_files={local_save_dir}/<dataset>/train.parquet")
    print(f"  data.val_files={local_save_dir}/<dataset>/test.parquet")
