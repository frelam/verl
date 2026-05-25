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

import argparse
import json
import os
import random
import re
from typing import Optional

import datasets
from datasets import concatenate_datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed


DEFAULT_NON_MAX_EFFORT_PROMPT = (
    "You are a mathematics expert. "
    "Please reason step by step, carefully analyzing the problem and deriving the solution logically. "
    "Put your final answer within \\boxed{}."
)


DEFAULT_MAX_EFFORT_PROMPT = (
    "Reason with maximum effort and absolute rigor. No shortcuts are allowed. "
    "You MUST conduct an extremely comprehensive and thorough analysis, "
    "completely deconstructing the problem to find the root principles. "
    "Rigorously verify your logic, covering all possible paths, boundary cases, and adversarial scenarios. "
    "Explicitly document your complete reasoning process, "
    "recording every intermediate step, every alternative approach considered, "
    "and every hypothesis you reject. "
    "Ensure no premise is overlooked or unverified. "
    "Put your final answer within \\boxed{}."
)


GPQA_QUERY_TEMPLATE = (
    "{Question}\n"
    "A. {A}\nB. {B}\nC. {C}\nD. {D}\n\n"
    "Please reason step by step, and put your final answer (only the choice letter) within \\boxed{{}}."
)


def extract_gsm8k_solution(solution_str: str) -> str:
    solution = re.search(r"#### (\-?[0-9\.\,]+)", solution_str)
    if solution is None:
        return ""
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


def extract_math_solution(solution_str: str) -> Optional[str]:
    try:
        return remove_boxed(last_boxed_only_string(solution_str))
    except Exception:
        return None


def _should_use_max_effort(idx: int, max_effort_ratio: float, seed: int) -> bool:
    if max_effort_ratio <= 0:
        return False
    if max_effort_ratio >= 1:
        return True
    rng = random.Random(str(idx) + "_prompt_type_" + str(seed))
    return rng.random() < max_effort_ratio


def make_prompt(question: str, use_max_effort: bool, max_effort_prompt: str, non_max_effort_prompt: str) -> str:
    if use_max_effort:
        return question + "\n\n" + max_effort_prompt
    else:
        return question + "\n\n" + non_max_effort_prompt


def process_gsm8k(example: dict, idx: int, split: str, max_effort_ratio: float,
                  non_max_effort_prompt: str, max_effort_prompt: str,
                  seed: int = 42) -> dict:
    data_source = "openai/gsm8k"

    question_raw = example.pop("question")
    answer_raw = example.pop("answer")

    use_max_effort = _should_use_max_effort(idx, max_effort_ratio, seed)
    if use_max_effort:
        data_source = data_source + '-max-effort'
    prompt_type = "max_effort" if use_max_effort else "standard"

    question = make_prompt(question_raw, use_max_effort, max_effort_prompt, non_max_effort_prompt)
    solution = extract_gsm8k_solution(answer_raw)

    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": question}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": solution},
        "extra_info": {
            "split": split,
            "index": idx,
            "answer": answer_raw,
            "question": question_raw,
            "prompt_type": prompt_type,
        },
    }


def process_math(example: dict, idx: int, split: str, max_effort_ratio: float,
                 non_max_effort_prompt: str, max_effort_prompt: str,
                 seed: int = 42) -> dict:
    data_source = "DigitalLearningGmbH/MATH-lighteval"

    problem = example.pop("problem")
    solution_raw = example.pop("solution")

    use_max_effort = _should_use_max_effort(idx, max_effort_ratio, seed)
    if use_max_effort:
        data_source = data_source + '-max-effort'
    prompt_type = "max_effort" if use_max_effort else "standard"

    question = make_prompt(problem, use_max_effort, max_effort_prompt, non_max_effort_prompt)
    solution = extract_math_solution(solution_raw)

    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": question}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": solution if solution else ""},
        "extra_info": {
            "split": split,
            "index": idx,
            "answer": solution_raw,
            "question": problem,
            "prompt_type": prompt_type,
        },
    }


def process_gpqa_diamond(example: dict, idx: int, split: str, max_effort_ratio: float,
                         non_max_effort_prompt: str, max_effort_prompt: str,
                         seed: int = 42) -> dict:
    data_source = "Idavidrein/gpqa"

    rng = random.Random(str(idx) + "_gpqa_shuffle_" + str(seed))

    choices = [
        example.pop("Incorrect Answer 1").strip(),
        example.pop("Incorrect Answer 2").strip(),
        example.pop("Incorrect Answer 3").strip(),
    ]
    rng.shuffle(choices)
    gold_index = rng.randint(0, 3)
    choices.insert(gold_index, example.pop("Correct Answer").strip())

    question_raw = example.pop("Question")
    query_prompt = GPQA_QUERY_TEMPLATE.format(
        A=choices[0],
        B=choices[1],
        C=choices[2],
        D=choices[3],
        Question=question_raw,
    )
    gold_choice = "ABCD"[gold_index]

    use_max_effort = _should_use_max_effort(idx, max_effort_ratio, seed)
    if use_max_effort:
        data_source = data_source + '-max-effort'
    prompt_type = "max_effort" if use_max_effort else "standard"

    question = make_prompt(query_prompt, use_max_effort, max_effort_prompt, non_max_effort_prompt)

    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": question}],
        "ability": "reasoning",
        "reward_model": {"style": "rule", "ground_truth": gold_choice},
        "extra_info": {
            "split": split,
            "index": idx,
            "answer": gold_choice,
            "question": question_raw,
            "prompt_type": prompt_type,
        },
    }


def process_bbh(example: dict, idx: int, split: str, max_effort_ratio: float,
                non_max_effort_prompt: str, max_effort_prompt: str,
                config_name: str = "", seed: int = 42) -> dict:
    data_source = "lukaemon/bbh"

    question_raw = example.pop("input")
    answer_raw = example.pop("target")

    use_max_effort = _should_use_max_effort(idx, max_effort_ratio, seed)
    if use_max_effort:
        data_source = data_source + '-max-effort'
    prompt_type = "max_effort" if use_max_effort else "standard"

    question = make_prompt(question_raw, use_max_effort, max_effort_prompt, non_max_effort_prompt)

    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": question}],
        "ability": "reasoning",
        "reward_model": {"style": "rule", "ground_truth": answer_raw},
        "extra_info": {
            "split": split,
            "index": idx,
            "answer": answer_raw,
            "question": question_raw,
            "prompt_type": prompt_type,
            "bbh_config": config_name,
        },
    }


def download_and_process_gsm8k(
    local_dataset_path: Optional[str] = None,
    split: str = "train",
    max_effort_ratio: float = 0.3,
    non_max_effort_prompt: str = DEFAULT_NON_MAX_EFFORT_PROMPT,
    max_effort_prompt: str = DEFAULT_MAX_EFFORT_PROMPT,
    seed: int = 42,
) -> datasets.Dataset:
    print(f"Loading GSM8K dataset ({split} split)...")

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path, "main", split=split)
    else:
        dataset = datasets.load_dataset("openai/gsm8k", "main", split=split)

    print(f"  Processing {len(dataset)} examples...")

    def process_fn(example, idx):
        return process_gsm8k(
            example, idx, split, max_effort_ratio,
            non_max_effort_prompt, max_effort_prompt, seed=seed,
        )

    processed_dataset = dataset.map(
        function=process_fn,
        with_indices=True,
        num_proc=8,
        remove_columns=dataset.column_names,
    )

    print(f"  Done! Processed {len(processed_dataset)} examples")
    return processed_dataset


def download_and_process_math(
    local_dataset_path: Optional[str] = None,
    split: str = "train",
    max_effort_ratio: float = 0.3,
    non_max_effort_prompt: str = DEFAULT_NON_MAX_EFFORT_PROMPT,
    max_effort_prompt: str = DEFAULT_MAX_EFFORT_PROMPT,
    seed: int = 42,
) -> datasets.Dataset:
    print(f"Loading MATH-lighteval dataset ({split} split)...")

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path, split=split)
    else:
        dataset = datasets.load_dataset(
            "DigitalLearningGmbH/MATH-lighteval",
            split=split,
        )

    print(f"  Processing {len(dataset)} examples...")

    def process_fn(example, idx):
        return process_math(
            example, idx, split, max_effort_ratio,
            non_max_effort_prompt, max_effort_prompt, seed=seed,
        )

    processed_dataset = dataset.map(
        function=process_fn,
        with_indices=True,
        num_proc=8,
        remove_columns=dataset.column_names,
    )

    print(f"  Done! Processed {len(processed_dataset)} examples")
    return processed_dataset


def download_and_process_gpqa_diamond(
    local_dataset_path: Optional[str] = None,
    split: str = "train",
    max_effort_ratio: float = 0.3,
    non_max_effort_prompt: str = DEFAULT_NON_MAX_EFFORT_PROMPT,
    max_effort_prompt: str = DEFAULT_MAX_EFFORT_PROMPT,
    seed: int = 42,
) -> datasets.Dataset:
    print(f"Loading GPQA Diamond dataset ({split} split)...")

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path, "gpqa_diamond", split=split)
    else:
        dataset = datasets.load_dataset("Idavidrein/gpqa", "gpqa_diamond", split=split)

    print(f"  Processing {len(dataset)} examples...")

    def process_fn(example, idx):
        return process_gpqa_diamond(
            example, idx, split, max_effort_ratio,
            non_max_effort_prompt, max_effort_prompt, seed=seed,
        )

    processed_dataset = dataset.map(
        function=process_fn,
        with_indices=True,
        num_proc=8,
        remove_columns=dataset.column_names,
    )

    print(f"  Done! Processed {len(processed_dataset)} examples")
    return processed_dataset


BBH_CONFIGS = [
    "boolean_expressions", "causal_judgement", "date_understanding",
    "disambiguation_qa", "dyck_languages", "formal_fallacies",
    "geometric_shapes", "hyperbaton", "logical_deduction_five_objects",
    "logical_deduction_seven_objects", "logical_deduction_three_objects",
    "movie_recommendation", "multistep_arithmetic_two", "navigate",
    "object_counting", "penguins_in_a_table", "reasoning_about_colored_objects",
    "ruin_names", "salient_translation_error_detection", "snarks",
    "sports_understanding", "temporal_sequences",
    "tracking_shuffled_objects_five_objects", "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects", "web_of_lies", "word_sorting",
]


def download_and_process_bbh(
    local_dataset_path: Optional[str] = None,
    split: str = "train",
    max_effort_ratio: float = 0.3,
    non_max_effort_prompt: str = DEFAULT_NON_MAX_EFFORT_PROMPT,
    max_effort_prompt: str = DEFAULT_MAX_EFFORT_PROMPT,
    bbh_configs: Optional[list] = None,
    seed: int = 42,
) -> datasets.Dataset:
    configs = bbh_configs if bbh_configs else BBH_CONFIGS
    print(f"Loading BBH dataset ({split} split, {len(configs)} configs)...")

    all_processed = []
    for config_name in configs:
        print(f"  Loading config: {config_name}...")

        if local_dataset_path is not None:
            dataset = datasets.load_dataset(local_dataset_path, config_name, split=split)
        else:
            dataset = datasets.load_dataset("lukaemon/bbh", config_name, split=split)

        def process_fn(example, idx):
            return process_bbh(
                example, idx, split, max_effort_ratio,
                non_max_effort_prompt, max_effort_prompt,
                config_name=config_name, seed=seed,
            )

        processed = dataset.map(
            function=process_fn,
            with_indices=True,
            num_proc=8,
            remove_columns=dataset.column_names,
        )
        all_processed.append(processed)

    merged = concatenate_datasets(all_processed)
    print(f"  Done! Processed {len(merged)} examples across {len(configs)} configs")
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Generate mixed reasoning dataset with standard and max-effort prompt types for verl training"
    )
    parser.add_argument(
        "--local_save_dir",
        default="~/data/mixed_reasoning_prompt_types",
        help="The save directory for the preprocessed dataset.",
    )
    parser.add_argument(
        "--hdfs_dir",
        default=None,
        help="HDFS directory (optional).",
    )
    parser.add_argument(
        "--gsm8k_path",
        default=None,
        help="Local path to the raw GSM8K dataset (optional).",
    )
    parser.add_argument(
        "--math_path",
        default=None,
        help="Local path to the raw MATH-lighteval dataset (optional).",
    )
    parser.add_argument(
        "--gpqa_path",
        default=None,
        help="Local path to the raw GPQA dataset (optional).",
    )
    parser.add_argument(
        "--bbh_path",
        default=None,
        help="Local path to the raw BBH dataset (optional).",
    )
    parser.add_argument(
        "--max_effort_ratio",
        type=float,
        default=0.3,
        help="Ratio of samples assigned the max-effort prompt (default: 0.3).",
    )
    parser.add_argument(
        "--non_max_effort_prompt_file",
        default=None,
        help="Path to a text file containing the non-max-effort (standard) prompt (optional).",
    )
    parser.add_argument(
        "--max_effort_prompt_file",
        default=None,
        help="Path to a text file containing the max-effort prompt (optional).",
    )
    parser.add_argument(
        "--skip_gsm8k",
        action="store_true",
        help="Skip GSM8K dataset.",
    )
    parser.add_argument(
        "--skip_math",
        action="store_true",
        help="Skip MATH-lighteval dataset.",
    )
    parser.add_argument(
        "--skip_gpqa",
        action="store_true",
        help="Skip GPQA Diamond dataset.",
    )
    parser.add_argument(
        "--skip_bbh",
        action="store_true",
        help="Skip BBH dataset.",
    )
    parser.add_argument(
        "--bbh_configs",
        nargs="*",
        default=None,
        help="Specific BBH configs to include (default: all 27). "
        "Example: --bbh_configs boolean_expressions causal_judgement",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Ratio of data held out as test set for datasets without built-in splits "
        "(GPQA, BBH). 0 means all data goes to train. (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for prompt type assignment and shuffling (default: 42).",
    )

    args = parser.parse_args()

    non_max_effort_prompt = DEFAULT_NON_MAX_EFFORT_PROMPT
    max_effort_prompt = DEFAULT_MAX_EFFORT_PROMPT

    if args.non_max_effort_prompt_file is not None:
        with open(args.non_max_effort_prompt_file, "r") as f:
            non_max_effort_prompt = f.read().strip()
        print(f"Loaded non-max-effort prompt from: {args.non_max_effort_prompt_file}")

    if args.max_effort_prompt_file is not None:
        with open(args.max_effort_prompt_file, "r") as f:
            max_effort_prompt = f.read().strip()
        print(f"Loaded max-effort prompt from: {args.max_effort_prompt_file}")

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    random.seed(args.seed)

    train_datasets = []
    test_datasets = []

    if not args.skip_gsm8k:
        train_datasets.append(
            download_and_process_gsm8k(
                local_dataset_path=args.gsm8k_path,
                split="train",
                max_effort_ratio=args.max_effort_ratio,
                non_max_effort_prompt=non_max_effort_prompt,
                max_effort_prompt=max_effort_prompt,
                seed=args.seed,
            )
        )
        test_datasets.append(
            download_and_process_gsm8k(
                local_dataset_path=args.gsm8k_path,
                split="test",
                max_effort_ratio=args.max_effort_ratio,
                non_max_effort_prompt=non_max_effort_prompt,
                max_effort_prompt=max_effort_prompt,
                seed=args.seed,
            )
        )

    if not args.skip_math:
        train_datasets.append(
            download_and_process_math(
                local_dataset_path=args.math_path,
                split="train",
                max_effort_ratio=args.max_effort_ratio,
                non_max_effort_prompt=non_max_effort_prompt,
                max_effort_prompt=max_effort_prompt,
                seed=args.seed,
            )
        )
        test_datasets.append(
            download_and_process_math(
                local_dataset_path=args.math_path,
                split="test",
                max_effort_ratio=args.max_effort_ratio,
                non_max_effort_prompt=non_max_effort_prompt,
                max_effort_prompt=max_effort_prompt,
                seed=args.seed,
            )
        )

    if not args.skip_gpqa:
        gpqa_dataset = download_and_process_gpqa_diamond(
            local_dataset_path=args.gpqa_path,
            split="train",
            max_effort_ratio=args.max_effort_ratio,
            non_max_effort_prompt=non_max_effort_prompt,
            max_effort_prompt=max_effort_prompt,
            seed=args.seed,
        )
        gpqa_dataset = gpqa_dataset.shuffle(seed=args.seed)
        if args.test_ratio > 0:
            split_dataset = gpqa_dataset.train_test_split(test_size=args.test_ratio, seed=args.seed)
            train_datasets.append(split_dataset["train"])
            test_datasets.append(split_dataset["test"])
            print(f"GPQA train/test split: {len(split_dataset['train'])}/{len(split_dataset['test'])}")
        else:
            train_datasets.append(gpqa_dataset)

    if not args.skip_bbh:
        bbh_dataset = download_and_process_bbh(
            local_dataset_path=args.bbh_path,
            split="test",
            max_effort_ratio=args.max_effort_ratio,
            non_max_effort_prompt=non_max_effort_prompt,
            max_effort_prompt=max_effort_prompt,
            bbh_configs=args.bbh_configs,
            seed=args.seed,
        )
        bbh_dataset = bbh_dataset.shuffle(seed=args.seed)
        if args.test_ratio > 0:
            split_dataset = bbh_dataset.train_test_split(test_size=args.test_ratio, seed=args.seed)
            train_datasets.append(split_dataset["train"])
            test_datasets.append(split_dataset["test"])
            print(f"BBH train/test split: {len(split_dataset['train'])}/{len(split_dataset['test'])}")
        else:
            train_datasets.append(bbh_dataset)

    print("\n" + "=" * 60)
    print("Merging datasets...")
    print("=" * 60)

    if train_datasets:
        merged_train = concatenate_datasets(train_datasets)
        print(f"Total training examples: {len(merged_train)}")

        from collections import Counter

        data_sources = merged_train["data_source"]
        source_counts = Counter(data_sources)
        print("\nTraining set distribution by data_source:")
        for source, count in sorted(source_counts.items()):
            print(f"  {source}: {count}")

        extra_infos = merged_train["extra_info"]
        prompt_types = [info["prompt_type"] for info in extra_infos]
        type_counts = Counter(prompt_types)
        print("\nTraining set distribution by prompt_type:")
        for pt, count in sorted(type_counts.items()):
            print(f"  {pt}: {count} ({count / len(prompt_types) * 100:.1f}%)")

        merged_train = merged_train.shuffle(seed=args.seed)

        train_path = os.path.join(local_save_dir, "train.parquet")
        merged_train.to_parquet(train_path)
        print(f"\nSaved training set to: {train_path}")

        example_path = os.path.join(local_save_dir, "train_example.json")
        with open(example_path, "w") as f:
            json.dump(merged_train[0], f, indent=2, ensure_ascii=False)
        print(f"Saved example to: {example_path}")

    if test_datasets:
        merged_test = concatenate_datasets(test_datasets)
        print(f"\nTotal test examples: {len(merged_test)}")

        data_sources = merged_test["data_source"]
        source_counts = Counter(data_sources)
        print("\nTest set distribution by data_source:")
        for source, count in sorted(source_counts.items()):
            print(f"  {source}: {count}")

        extra_infos = merged_test["extra_info"]
        prompt_types = [info["prompt_type"] for info in extra_infos]
        type_counts = Counter(prompt_types)
        print("\nTest set distribution by prompt_type:")
        for pt, count in sorted(type_counts.items()):
            print(f"  {pt}: {count} ({count / len(prompt_types) * 100:.1f}%)")

        test_path = os.path.join(local_save_dir, "test.parquet")
        merged_test.to_parquet(test_path)
        print(f"\nSaved test set to: {test_path}")

        example_path = os.path.join(local_save_dir, "test_example.json")
        with open(example_path, "w") as f:
            json.dump(merged_test[0], f, indent=2, ensure_ascii=False)
        print(f"Saved example to: {example_path}")

    if args.hdfs_dir is not None:
        print(f"\nCopying to HDFS: {args.hdfs_dir}")
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)

    print("\n" + "=" * 60)
    print("Dataset preparation completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
