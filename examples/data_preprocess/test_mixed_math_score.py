#!/usr/bin/env python3
"""测试混合数学数据集的compute_score函数"""

import sys
import os

sys.path.insert(0, '/home/charles/workspace/verl')

import re
from typing import Optional


_SOLUTION_CLIP_CHARS = 300


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string."""
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    """Remove the LaTeX boxed command from a string."""
    left = "\\boxed{"
    assert s[: len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"
    return s[len(left) : -1]


def strip_string(string: str) -> str:
    """Normalize a string for comparison."""
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = string.replace("\\\\%", "")
    string = string.replace("\\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    string = string.replace(" ", "")
    return string


def is_equiv(str1: str, str2: str) -> bool:
    """Check if two strings are equivalent."""
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def extract_gsm8k_solution(solution_str: str, method: str = "strict") -> Optional[str]:
    """从GSM8K格式中提取答案"""
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        solutions = re.findall(r"#### (\-?[0-9\.\,]+)", solution_str)
        if len(solutions) == 0:
            return None
        else:
            return solutions[-1].replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall(r"(\-?[0-9\.\,]+)", solution_str)
        if len(answer) == 0:
            return None
        invalid_str = ["", "."]
        for final_answer in reversed(answer):
            if final_answer not in invalid_str:
                return final_answer
    return None


def compute_gsm8k_score(solution_str: str, ground_truth: str) -> float:
    """计算GSM8K分数"""
    answer = extract_gsm8k_solution(solution_str, "strict")
    if answer is None:
        return 0.0
    return 1.0 if answer == ground_truth else 0.0


def compute_math_score(solution_str: str, ground_truth: str) -> float:
    """计算MATH分数"""
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            answer = remove_boxed(string_in_last_boxed)
            if is_equiv(answer, ground_truth):
                return 1.0
    except Exception as e:
        print(e)
    return 0.0


def compute_openr1_score(solution_str: str, ground_truth: str) -> float:
    """计算OpenR1分数"""
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]
    
    boxed_str = last_boxed_only_string(solution_str)
    if boxed_str is not None:
        answer = remove_boxed(boxed_str)
        if is_equiv(answer, ground_truth):
            return 1.0
    return 0.0


def compute_score(data_source: str, solution_str: str, ground_truth: str) -> float:
    """根据数据源计算分数"""
    if data_source == "openai/gsm8k":
        return compute_gsm8k_score(solution_str, ground_truth)
    elif data_source in ["DigitalLearningGmbH/MATH-lighteval", "lighteval/MATH", "HuggingFaceH4/MATH-500"]:
        return compute_math_score(solution_str, ground_truth)
    elif data_source in ["open-r1/OpenR1-Math-220k", "AI-MO/NuminaMath-CoT"]:
        return compute_openr1_score(solution_str, ground_truth)
    else:
        return compute_openr1_score(solution_str, ground_truth)


def test_gsm8k():
    """测试GSM8K评分"""
    print("Testing GSM8K...")
    
    solution = "Let's think step by step. The answer is 42. #### 42"
    ground_truth = "42"
    score = compute_score("openai/gsm8k", solution, ground_truth)
    print(f"  Solution: {solution}")
    print(f"  Ground truth: {ground_truth}")
    print(f"  Score: {score}")
    assert score == 1.0, f"Expected 1.0, got {score}"
    
    solution_wrong = "Let's think step by step. The answer is 41. #### 41"
    score_wrong = compute_score("openai/gsm8k", solution_wrong, ground_truth)
    print(f"  Wrong solution score: {score_wrong}")
    assert score_wrong == 0.0, f"Expected 0.0, got {score_wrong}"
    
    print("  ✓ GSM8K tests passed\n")


def test_math():
    """测试MATH评分"""
    print("Testing MATH-lighteval...")
    
    solution = "The answer is \\boxed{42}"
    ground_truth = "42"
    score = compute_score("DigitalLearningGmbH/MATH-lighteval", solution, ground_truth)
    print(f"  Solution: {solution}")
    print(f"  Ground truth: {ground_truth}")
    print(f"  Score: {score}")
    assert score == 1.0, f"Expected 1.0, got {score}"
    
    print("  ✓ MATH tests passed\n")


def test_openr1():
    """测试OpenR1评分"""
    print("Testing OpenR1-Math-220k...")
    
    solution = "After thinking... the answer is \\boxed{42}"
    ground_truth = "42"
    score = compute_score("open-r1/OpenR1-Math-220k", solution, ground_truth)
    print(f"  Solution: {solution}")
    print(f"  Ground truth: {ground_truth}")
    print(f"  Score: {score}")
    assert score == 1.0, f"Expected 1.0, got {score}"
    
    print("  ✓ OpenR1 tests passed\n")


def test_numina():
    """测试NuminaMath评分"""
    print("Testing NuminaMath-CoT...")
    
    solution = "After calculation... the answer is \\boxed{42}"
    ground_truth = "42"
    score = compute_score("AI-MO/NuminaMath-CoT", solution, ground_truth)
    print(f"  Solution: {solution}")
    print(f"  Ground truth: {ground_truth}")
    print(f"  Score: {score}")
    assert score == 1.0, f"Expected 1.0, got {score}"
    
    print("  ✓ NuminaMath tests passed\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing mixed_math compute_score functions")
    print("=" * 60 + "\n")
    
    test_gsm8k()
    test_math()
    test_openr1()
    test_numina()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
