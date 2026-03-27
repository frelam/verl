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
混合数学数据集的评分函数
支持以下数据集:
- open-r1/OpenR1-Math-220k
- openai/gsm8k
- DigitalLearningGmbH/MATH-lighteval
- AI-MO/NuminaMath-CoT
"""

import re
from typing import Optional

from verl.utils.reward_score import gsm8k, math_reward, math_dapo


_SOLUTION_CLIP_CHARS = 300


def extract_openr1_answer(solution_str: str, method: str = "boxed") -> Optional[str]:
    """从OpenR1格式中提取答案
    
    OpenR1数据集的答案通常在\\boxed{}中
    """
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]
    
    if method == "boxed":
        boxed_str = math_reward.last_boxed_only_string(solution_str)
        if boxed_str is not None:
            return math_reward.remove_boxed(boxed_str)
        return None
    elif method == "flexible":
        numbers = re.findall(r"(\-?[0-9\.\\,]+)", solution_str)
        if numbers:
            return numbers[-1].replace(",", "").replace("\\", "")
        return None
    return None


def compute_openr1_score(
    solution_str: str,
    ground_truth: str,
    method: str = "boxed",
    format_score: float = 0.0,
    score: float = 1.0,
) -> float:
    """计算OpenR1-Math-220k数据集的分数
    
    Args:
        solution_str: 模型输出的解答字符串
        ground_truth: 正确答案
        method: 提取答案的方法 ('boxed' 或 'flexible')
        format_score: 格式正确但答案错误的分数
        score: 答案正确的分数
    
    Returns:
        分数 (0.0, format_score, 或 score)
    """
    answer = extract_openr1_answer(solution_str, method)
    if answer is None:
        return 0.0
    
    if math_reward.is_equiv(answer, ground_truth):
        return score
    else:
        return format_score


def compute_numina_score(
    solution_str: str,
    ground_truth: str,
    method: str = "boxed",
    format_score: float = 0.0,
    score: float = 1.0,
) -> float:
    """计算NuminaMath数据集的分数
    
    Args:
        solution_str: 模型输出的解答字符串
        ground_truth: 正确答案
        method: 提取答案的方法 ('boxed' 或 'flexible')
        format_score: 格式正确但答案错误的分数
        score: 答案正确的分数
    
    Returns:
        分数 (0.0, format_score, 或 score)
    """
    answer = extract_openr1_answer(solution_str, method)
    if answer is None:
        return 0.0
    
    if math_reward.is_equiv(answer, ground_truth):
        return score
    else:
        return format_score


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    method: str = "auto",
    format_score: float = 0.0,
    score: float = 1.0,
    **kwargs,
) -> float:
    """根据数据源计算分数的统一接口
    
    Args:
        data_source: 数据集来源标识
        solution_str: 模型输出的解答字符串
        ground_truth: 正确答案
        method: 提取答案的方法 ('auto', 'strict', 'flexible', 'boxed')
        format_score: 格式正确但答案错误的分数
        score: 答案正确的分数
        **kwargs: 额外参数
    
    Returns:
        分数
    """
    if data_source == "open-r1/OpenR1-Math-220k":
        extract_method = "boxed" if method == "auto" else method
        return compute_openr1_score(
            solution_str, ground_truth, extract_method, format_score, score
        )
    
    elif data_source == "openai/gsm8k":
        extract_method = "strict" if method == "auto" else method
        return gsm8k.compute_score(
            solution_str, ground_truth, extract_method, format_score, score
        )
    
    elif data_source in ["DigitalLearningGmbH/MATH-lighteval", "lighteval/MATH", "HuggingFaceH4/MATH-500"]:
        return math_reward.compute_score(solution_str, ground_truth)
    
    elif data_source == "AI-MO/NuminaMath-CoT":
        extract_method = "boxed" if method == "auto" else method
        return compute_numina_score(
            solution_str, ground_truth, extract_method, format_score, score
        )
    
    elif data_source in ["math_dapo", "math", "math_dapo_reasoning"] or data_source.startswith("aime"):
        result = math_dapo.compute_score(solution_str, ground_truth)
        if isinstance(result, dict):
            return result.get("score", 0.0)
        return result
    
    else:
        return compute_openr1_score(solution_str, ground_truth, "boxed", format_score, score)


def compute_score_with_details(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    method: str = "auto",
    format_score: float = 0.0,
    score: float = 1.0,
    **kwargs,
) -> dict:
    """计算分数并返回详细信息
    
    Args:
        data_source: 数据集来源标识
        solution_str: 模型输出的解答字符串
        ground_truth: 正确答案
        method: 提取答案的方法
        format_score: 格式正确但答案错误的分数
        score: 答案正确的分数
        **kwargs: 额外参数
    
    Returns:
        包含分数和详细信息的字典
    """
    final_score = compute_score(
        data_source, solution_str, ground_truth, method, format_score, score, **kwargs
    )
    
    extracted_answer = None
    if data_source == "open-r1/OpenR1-Math-220k":
        extracted_answer = extract_openr1_answer(solution_str, "boxed")
    elif data_source == "openai/gsm8k":
        extracted_answer = gsm8k.extract_solution(solution_str, method if method != "auto" else "strict")
    elif data_source in ["DigitalLearningGmbH/MATH-lighteval", "lighteval/MATH", "HuggingFaceH4/MATH-500"]:
        boxed_str = math_reward.last_boxed_only_string(solution_str)
        if boxed_str:
            extracted_answer = math_reward.remove_boxed(boxed_str)
    elif data_source == "AI-MO/NuminaMath-CoT":
        extracted_answer = extract_openr1_answer(solution_str, "boxed")
    
    return {
        "score": final_score,
        "data_source": data_source,
        "ground_truth": ground_truth,
        "extracted_answer": extracted_answer,
        "is_correct": final_score >= score,
    }
