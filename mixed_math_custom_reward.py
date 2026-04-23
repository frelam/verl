import re
from typing import Any, Optional, Union,  List, Tuple
from collections import Counter

from verl.utils.reward_score import gsm8k as gsm8k_score
from verl.utils.reward_score import math_reward as math_reward

def check_thinking_format(solution_str):
    pattern = r"<think>.*?</think>"
    match = re.search(pattern, solution_str, re.DOTALL)
    return match is not None and len(matches) == 1

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
        pattern = f'wait'
        
    matches = re.findall(pattern, text, flag=re.IGNORECASE)
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

def _compute_score(
    solution_str,
    ground_truth,
):
    base_score = math_reward.compute_score(solution_str, ground_truth)

    if base_score == 0.0:
        base_score = -0.1
    
    if check_thinking_format(solution_str):
        bonus = 1.0
    else:
        bonus = 0.0

    if base_score <= 0:
        bonus = 0.0

    if base_score > 0:
        if has_repetition(solution_str):
            repetition_bonus = 0.0
        else:
            repetition_bonus = 1.0
    else:
        repetition_bonus = 0.0
    
    return float(0.8 * base_score + 0.1 * bonus + 0.1 * repetition_bonus)


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs
):
    return _compute_score(solution_str, ground_truth)
