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

import re
from collections import Counter

from verl.utils.reward_score import math_reward


def length_bonus(text, max_response_length: int = 4000):
    total_words = len(text.split())
    bonus = max(0, 1 - (total_words / max_response_length))
    return bonus

def check_thinking_format(solution_str):
    pattern = r"<think>.*?</think>"
    matches = re.findall(pattern, solution_str, re.DOTALL)
    return len(matches) == 1


def has_repetition(text: str,
                   min_ngram_len: int = 3,
                   max_ngram_len: int = 5,
                   ngram_repeat_threshold: int = 5):
    for n in range(min_ngram_len, max_ngram_len + 1):
        ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
        ngram_counts = Counter(ngrams)
        if any(count >= ngram_repeat_threshold for count in ngram_counts.values()):
            return True
    return False


def wait_score(text: str,
               max_occurrences: int = 10,
               max_ratio: float = 0.01,
               reward_value: float = 1.0,
               use_word_boundary: bool = True) -> float:
    if use_word_boundary:
        pattern = r'\bwait\b'
    else:
        pattern = 'wait'

    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    count = len(matches)
    if count > max_occurrences:
        return 0.0

    if max_ratio is not None:
        total_words = len(text.split())
        if total_words > 0:
            ratio = count / total_words
            if ratio > max_ratio:
                return 0.0
    return reward_value


def _compute_score(
    solution_str,
    ground_truth,
):
    base_score = math_reward.compute_score(solution_str, ground_truth)

    if base_score == 0.0:
        base_score = -0.1

    if check_thinking_format(solution_str):
        format_bonus = 1.0
    else:
        format_bonus = 0.0

    if base_score <= 0:
        format_bonus = 0.0

    if base_score > 0:
        if has_repetition(solution_str):
            repetition_bonus = 0.0
        else:
            repetition_bonus = 1.0
        wait_bonus = wait_score(solution_str)
        if data_source.endswith("max-effort"):
            len_bonus = 1.0
        else:
            len_bonus = length_bonus(solution_str)
    else:
        repetition_bonus = 0.0
        wait_bonus = 0.0
        len_bonus = 0.0

    # score = float(0.7 * base_score + 0.1 * format_bonus + 0.1 * repetition_bonus + 0.1 * wait_bonus)
    score = float(0.7 * base_score + 0.1 * format_bonus + 0.1 * wait_bonus + 0.1 * len_bonus)


    return {
        "score": score,
        "accuracy_reward": float(base_score),
        "format_reward": float(format_bonus),
        # "repetition_reward": float(repetition_bonus),
        "len_reward": float(len_bonus),
        "wait_reward": float(wait_bonus),
    }


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs
):
    return _compute_score(solution_str, ground_truth)
