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
混合多个数学数据集，生成用于verl框架训练的数据集
包含以下数据集:
- OpenR1-Math-220k (default split)
- GSM8K
- MATH-lighteval
- NuminaMath-CoT (子集)
"""

import argparse
import os
import re
import json
from typing import Optional

import datasets
from datasets import concatenate_datasets
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed


def extract_gsm8k_solution(solution_str: str) -> str:
    """从GSM8K格式中提取答案"""
    solution = re.search(r"#### (\-?[0-9\.\,]+)", solution_str)
    if solution is None:
        return ""
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


def extract_math_solution(solution_str: str) -> Optional[str]:
    """从MATH格式中提取答案"""
    try:
        return remove_boxed(last_boxed_only_string(solution_str))
    except Exception:
        return None


def process_openr1_math(example: dict, idx: int, split: str) -> dict:
    """处理OpenR1-Math-220k数据集"""
    data_source = "open-r1/OpenR1-Math-220k"
    
    problem = example.get("problem", "")
    solution = example.get("solution", "")
    answer = example.get("answer", "")
    
    instruction_following = (
        r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        r"The reasoning process MUST BE enclosed within <thinkthink> tags. "
        r"The final answer MUST BE put in \boxed{}."
    )
    
    prompt = problem + " " + instruction_following
    
    data = {
        "data_source": data_source,
        "prompt": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "split": split,
            "index": idx,
            "answer": answer,
            "question": problem,
            "solution": solution,
        },
    }
    return data


def process_gsm8k(example: dict, idx: int, split: str) -> dict:
    """处理GSM8K数据集"""
    data_source = "openai/gsm8k"
    
    question_raw = example.get("question", "")
    answer_raw = example.get("answer", "")
    
    instruction_following = 'Let\'s think step by step and output the final answer after "####".'
    question = question_raw + " " + instruction_following
    solution = extract_gsm8k_solution(answer_raw)
    
    data = {
        "data_source": data_source,
        "prompt": [
            {
                "role": "user",
                "content": question,
            }
        ],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": solution},
        "extra_info": {
            "split": split,
            "index": idx,
            "answer": answer_raw,
            "question": question_raw,
        },
    }
    return data


def process_math(example: dict, idx: int, split: str) -> dict:
    """处理MATH-lighteval数据集"""
    data_source = "DigitalLearningGmbH/MATH-lighteval"
    
    problem = example.get("problem", "")
    solution_raw = example.get("solution", "")
    
    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
    question = problem + " " + instruction_following
    solution = extract_math_solution(solution_raw)
    
    data = {
        "data_source": data_source,
        "prompt": [
            {
                "role": "user",
                "content": question,
            }
        ],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": solution if solution else ""},
        "extra_info": {
            "split": split,
            "index": idx,
            "answer": solution_raw,
            "question": problem,
        },
    }
    return data


def process_numina_math(example: dict, idx: int, split: str) -> dict:
    """处理NuminaMath数据集"""
    data_source = "AI-MO/NuminaMath-CoT"
    
    problem = example.get("problem", "")
    solution_raw = example.get("solution", "")
    
    instruction_following = (
        r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        r"The reasoning process MUST BE enclosed within <thinkthink> tags. "
        r"The final answer MUST BE put in \boxed{}."
    )
    
    prompt = problem + " " + instruction_following
    
    solution = extract_math_solution(solution_raw)
    
    data = {
        "data_source": data_source,
        "prompt": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": solution if solution else ""},
        "extra_info": {
            "split": split,
            "index": idx,
            "answer": solution_raw,
            "question": problem,
        },
    }
    return data


def download_and_process_openr1_math(
    local_dataset_path: Optional[str] = None,
    max_samples: Optional[int] = None,
    split: str = "train",
) -> datasets.Dataset:
    """下载并处理OpenR1-Math-220k数据集"""
    print(f"Loading OpenR1-Math-220k dataset ({split} split)...")
    
    if local_dataset_path is not None:
        dataset = datasets.load_dataset(
            local_dataset_path,
            "default",
            split=split,
        )
    else:
        dataset = datasets.load_dataset(
            "open-r1/OpenR1-Math-220k",
            "default",
            split=split,
            trust_remote_code=True,
        )
    
    if max_samples is not None and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
        print(f"  Subsampled to {max_samples} examples")
    
    print(f"  Processing {len(dataset)} examples...")
    
    def process_fn(example, idx):
        return process_openr1_math(example, idx, split)
    
    processed_dataset = dataset.map(
        function=process_fn,
        with_indices=True,
        num_proc=8,
        remove_columns=dataset.column_names,
    )
    
    print(f"  Done! Processed {len(processed_dataset)} examples")
    return processed_dataset


def download_and_process_gsm8k(
    local_dataset_path: Optional[str] = None,
    split: str = "train",
) -> datasets.Dataset:
    """下载并处理GSM8K数据集"""
    print(f"Loading GSM8K dataset ({split} split)...")
    
    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path, "main", split=split)
    else:
        dataset = datasets.load_dataset("openai/gsm8k", "main", split=split)
    
    print(f"  Processing {len(dataset)} examples...")
    
    def process_fn(example, idx):
        return process_gsm8k(example, idx, split)
    
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
) -> datasets.Dataset:
    """下载并处理MATH-lighteval数据集"""
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
        return process_math(example, idx, split)
    
    processed_dataset = dataset.map(
        function=process_fn,
        with_indices=True,
        num_proc=8,
        remove_columns=dataset.column_names,
    )
    
    print(f"  Done! Processed {len(processed_dataset)} examples")
    return processed_dataset


def download_and_process_numina_math(
    local_dataset_path: Optional[str] = None,
    max_samples: Optional[int] = None,
    split: str = "train",
) -> datasets.Dataset:
    """下载并处理NuminaMath-CoT数据集"""
    print(f"Loading NuminaMath-CoT dataset ({split} split)...")
    
    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path, split=split)
    else:
        dataset = datasets.load_dataset(
            "AI-MO/NuminaMath-CoT",
            split=split,
        )
    
    if max_samples is not None and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
        print(f"  Subsampled to {max_samples} examples")
    
    print(f"  Processing {len(dataset)} examples...")
    
    def process_fn(example, idx):
        return process_numina_math(example, idx, split)
    
    processed_dataset = dataset.map(
        function=process_fn,
        with_indices=True,
        num_proc=8,
        remove_columns=dataset.column_names,
    )
    
    print(f"  Done! Processed {len(processed_dataset)} examples")
    return processed_dataset


def main():
    parser = argparse.ArgumentParser(description="混合多个数学数据集")
    parser.add_argument(
        "--local_save_dir",
        default="~/data/mixed_math_dataset",
        help="预处理数据集的保存目录",
    )
    parser.add_argument(
        "--hdfs_dir",
        default=None,
        help="HDFS目录（可选）",
    )
    parser.add_argument(
        "--openr1_math_path",
        default=None,
        help="OpenR1-Math-220k本地路径（可选）",
    )
    parser.add_argument(
        "--gsm8k_path",
        default=None,
        help="GSM8K本地路径（可选）",
    )
    parser.add_argument(
        "--math_path",
        default=None,
        help="MATH-lighteval本地路径（可选）",
    )
    parser.add_argument(
        "--numina_math_path",
        default=None,
        help="NuminaMath-CoT本地路径（可选）",
    )
    parser.add_argument(
        "--openr1_max_samples",
        type=int,
        default=None,
        help="OpenR1-Math-220k最大样本数（可选，用于调试）",
    )
    parser.add_argument(
        "--numina_max_samples",
        type=int,
        default=10000,
        help="NuminaMath-CoT最大样本数（默认10000）",
    )
    parser.add_argument(
        "--skip_openr1",
        action="store_true",
        help="跳过OpenR1-Math-220k数据集",
    )
    parser.add_argument(
        "--skip_gsm8k",
        action="store_true",
        help="跳过GSM8K数据集",
    )
    parser.add_argument(
        "--skip_math",
        action="store_true",
        help="跳过MATH-lighteval数据集",
    )
    parser.add_argument(
        "--skip_numina",
        action="store_true",
        help="跳过NuminaMath-CoT数据集",
    )
    
    args = parser.parse_args()
    
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    
    train_datasets = []
    test_datasets = []
    
    if not args.skip_openr1:
        train_datasets.append(
            download_and_process_openr1_math(
                local_dataset_path=args.openr1_math_path,
                max_samples=args.openr1_max_samples,
                split="train",
            )
        )
    
    if not args.skip_gsm8k:
        train_datasets.append(
            download_and_process_gsm8k(
                local_dataset_path=args.gsm8k_path,
                split="train",
            )
        )
        test_datasets.append(
            download_and_process_gsm8k(
                local_dataset_path=args.gsm8k_path,
                split="test",
            )
        )
    
    if not args.skip_math:
        train_datasets.append(
            download_and_process_math(
                local_dataset_path=args.math_path,
                split="train",
            )
        )
        test_datasets.append(
            download_and_process_math(
                local_dataset_path=args.math_path,
                split="test",
            )
        )
    
    if not args.skip_numina:
        train_datasets.append(
            download_and_process_numina_math(
                local_dataset_path=args.numina_math_path,
                max_samples=args.numina_max_samples,
                split="train",
            )
        )
    
    print("\n" + "=" * 50)
    print("Merging datasets...")
    print("=" * 50)
    
    if train_datasets:
        merged_train = concatenate_datasets(train_datasets)
        print(f"Total training examples: {len(merged_train)}")
        
        data_sources = merged_train["data_source"]
        from collections import Counter
        source_counts = Counter(data_sources)
        print("\nTraining set distribution:")
        for source, count in sorted(source_counts.items()):
            print(f"  {source}: {count}")
        
        merged_train = merged_train.shuffle(seed=42)
        
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
        from collections import Counter
        source_counts = Counter(data_sources)
        print("\nTest set distribution:")
        for source, count in sorted(source_counts.items()):
            print(f"  {source}: {count}")
        
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
    
    print("\n" + "=" * 50)
    print("Dataset preparation completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
