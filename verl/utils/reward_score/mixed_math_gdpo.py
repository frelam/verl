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

import random
import re
from collections import Counter

from verl.utils.reward_score import math_reward


def check_thinking_format(solution_str):
    pattern = r"⋪.*?⋫"
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


def wait_penalty(text: str,
                 max_occurrences: int = 3,
                 max_ratio: float = 0.02,
                 penalty_value: float = -0.1,
                 use_word_boundary: bool = True) -> float:
    if use_word_boundary:
        pattern = r'\bwait\b'
    else:
        pattern = 'wait'

    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    count = len(matches)
    if count > max_occurrences:
        return penalty_value

    if max_ratio is not None:
        total_words = len(text.split())
        if total_words > 0:
            ratio = count / total_words
            if ratio > max_ratio:
                return penalty_value
    return 0.0


def customize_accuracy_reward_func(
    completions, answer, step, max_possible_reward, min_possible_reward, do_print, **kwargs
):
    rewards = []
    responses = [completion[0]["content"] for completion in completions]

    if do_print:
        print("\n======= Answer ======= ")
        print(answer[0])
        print("\n======= Responses ======= ")
        for idx, response in enumerate(responses):
            print(f"*** Response {idx + 1} ***\n{response}")

    for response, ans in zip(responses, answer, strict=False):
        base_score = math_reward.compute_score(response, ans)
        if base_score == 0.0:
            reward = min_possible_reward
        else:
            reward = max_possible_reward
        rewards.append(reward)

    if do_print:
        print("\n======= Reward for <accuracy> =======")
        print("Reward function for <accuracy> is called ...")
        print(rewards)

    return rewards


def customize_thinking_format_reward_func(
    completions, answer, step, max_possible_reward, min_possible_reward, do_print, **kwargs
):
    rewards = []
    responses = [completion[0]["content"] for completion in completions]

    for response in responses:
        if check_thinking_format(response):
            reward = max_possible_reward
        else:
            reward = min_possible_reward
        rewards.append(reward)

    if do_print:
        print("\n======= Reward for <thinking format> =======")
        print("Reward function for <thinking format> is called ...")
        print(rewards)

    return rewards


def customize_repetition_reward_func(
    completions, answer, step, max_possible_reward, min_possible_reward, do_print, **kwargs
):
    rewards = []
    responses = [completion[0]["content"] for completion in completions]

    for response in responses:
        if has_repetition(response):
            reward = min_possible_reward
        else:
            reward = max_possible_reward
        rewards.append(reward)

    if do_print:
        print("\n======= Reward for <repetition> =======")
        print("Reward function for <repetition> is called ...")
        print(rewards)

    return rewards


def compute_score(data_source, solution_str, ground_truth, extra_info, step=0):
    exp_name = extra_info.get("experiment_name", "")
    if "llama" in exp_name:
        predict_str = (
            solution_str.split("<|start_header_id|>assistant<|end_header_id|>")[-1].split("<|eot_id|>")[0].strip()
        )
    elif "qwen" in exp_name:
        predict_str = solution_str.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
    else:
        raise NotImplementedError(f"Unknown model name: {exp_name}")

    accuracy_max_possible = 1.0
    accuracy_min_possible = -0.1

    format_max_possible = 1.0
    format_min_possible = 0.0

    repetition_max_possible = 1.0
    repetition_min_possible = 0.0

    completions = [[{"role": "assistant", "content": predict_str}]]
    answer = [ground_truth]

    do_print = random.randint(1, 64) == 1

    accuracy_score = customize_accuracy_reward_func(
        completions, answer, step, accuracy_max_possible, accuracy_min_possible, do_print
    )[0]
    format_score = customize_thinking_format_reward_func(
        completions, answer, step, format_max_possible, format_min_possible, do_print
    )[0]
    repetition_score = customize_repetition_reward_func(
        completions, answer, step, repetition_max_possible, repetition_min_possible, do_print
    )[0]

    if accuracy_score <= 0:
        format_score = 0.0
        repetition_score = 0.0

    score = 0.8 * accuracy_score + 0.1 * format_score + 0.1 * repetition_score

    result = {
        "score": score,
        "accuracy_reward": accuracy_score,
        "format_reward": format_score,
        "repetition_reward": repetition_score,
    }

    return result
