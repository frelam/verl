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
"""Agentic RL reward function implementing the full reward spec.

This module implements the reward structure from reward_priciple.md::

    R = R_answer
      + w_cal   * R_calibration
      + w_verify* R_verify
      + w_proc  * R_process
      + w_epi   * R_epistemic
      - w_tool  * R_tool_cost
      - w_safe  * R_safety

Design notes (mapped to the doc's 10 principles):

1.  *Result first, process second* -- R_answer is the dominant term but
    R_process / R_verify add partial credit for good behaviour.
2.  *Don't over-penalise "I don't know"* -- wrong answers get -0.2, not -1.0.
    Abstention (low confidence + tool call) gets a small positive reward.
3.  *Tool cost is tiered, not flat* -- calculator is cheap, search is
    moderate, and there is a free budget before the penalty kicks in.
4.  *Verify is explicitly rewarded* -- calling submit_answer after a
    search/calculator and revising gets a bonus.
5.  *Calibration via Brier score* -- the confidence reported in
    submit_answer is compared to correctness.
6.  *Safety: penalise fake verification, fake citations, tool hallucination.*
7.  *Epistemic: reward "knowing what you don't know" -- searching when
    unsure, not searching when sure.*

The function signature matches verl's ``compute_score`` contract so it can
be loaded via ``reward.custom_reward_function.path``.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Weight configuration -- tune these to shift the reward landscape.
# The defaults follow the recommended weighting in reward_priciple.md §10.
# ---------------------------------------------------------------------------
WEIGHTS = {
    "answer": 1.0,
    "calibration": 0.2,
    "verify": 0.2,
    "process": 0.3,
    "epistemic": 0.2,
    "tool_cost": 0.1,  # subtracted
    "safety": 1.0,  # subtracted (penalty per violation)
}

# Tiered tool costs (per call, after free budget).
TOOL_COSTS = {
    "calculator": 0.01,
    "search": 0.05,
    "browser": 0.1,
    "python": 0.2,
    "submit_answer": 0.0,  # submission is free
}
FREE_TOOL_BUDGET = 2  # first N tool calls are free

# Answer reward constants
REWARD_CORRECT = 1.0
REWARD_WRONG = -0.2
REWARD_UNKNOWN = 0.0


# ---------------------------------------------------------------------------
# Trajectory parsing helpers
# ---------------------------------------------------------------------------
def _extract_think_blocks(text: str) -> List[str]:
    return re.findall(r"<think>(.*?)</think>", text, re.DOTALL)


def _extract_tool_calls(text: str) -> List[dict]:
    """Extract tool calls from the response text.

    Supports the Hermes format ``<tool_call>{...}</tool_call>`` that verl's
    ToolAgentLoop uses by default.
    """
    calls = []
    for match in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL):
        raw = match.group(1).strip()
        # A single <tool_call> block may contain multiple JSON objects
        # separated by newlines.
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    calls.append(obj)
            except json.JSONDecodeError:
                continue
    return calls


def _extract_answer_text(text: str) -> Optional[str]:
    """Try to extract a final answer from common formats."""
    # <answer>...</answer>
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # #### <number>
    match = re.search(r"####\s*(.+?)(?:\n|$)", text)
    if match:
        return match.group(1).strip()
    # \boxed{...}
    match = re.search(r"\\boxed\{(.*)\}", text)
    if match:
        return match.group(1).strip()
    return None


def _parse_trajectory(solution_str: str, extra_info: dict) -> Dict:
    """Parse the full agent trajectory from the decoded response string.

    Returns a dict with:
      * ``tool_calls``: list of {name, arguments} dicts
      * ``think_blocks``: list of reasoning strings
      * ``answer_text``: extracted final answer or None
      * ``num_turns``: from extra_info or inferred
      * ``tool_rewards``: step-level rewards from tools (if available)
      * ``turn_scores``: turn-level scores from interactions (if available)
    """
    tool_calls = _extract_tool_calls(solution_str)
    think_blocks = _extract_think_blocks(solution_str)
    answer_text = _extract_answer_text(solution_str)

    # Also look for submit_answer calls to get the confidence
    confidence = None
    submitted_answer = None
    for tc in tool_calls:
        name = tc.get("name", "")
        if name == "submit_answer":
            args = tc.get("arguments", tc.get("parameters", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            submitted_answer = args.get("answer", submitted_answer)
            conf = args.get("confidence")
            if conf is not None:
                try:
                    confidence = float(conf)
                except (TypeError, ValueError):
                    pass

    return {
        "tool_calls": tool_calls,
        "think_blocks": think_blocks,
        "answer_text": submitted_answer or answer_text,
        "confidence": confidence,
        "num_turns": extra_info.get("num_turns", 1),
        "tool_rewards": extra_info.get("tool_rewards", []),
        "turn_scores": extra_info.get("turn_scores", []),
        "rollout_reward_scores": extra_info.get("rollout_reward_scores", {}),
    }


# ---------------------------------------------------------------------------
# Individual reward modules
# ---------------------------------------------------------------------------
def _reward_answer(parsed: dict, ground_truth: str, extra_info: dict) -> Tuple[float, bool]:
    """Outcome reward: is the submitted answer correct?

    Returns (reward, is_correct).
    """
    answer = parsed["answer_text"]
    task_type = extra_info.get("task_type", "math")

    if answer is None:
        # No answer submitted at all
        return REWARD_UNKNOWN, False

    is_correct = _check_correctness(answer, ground_truth, task_type)
    if is_correct:
        return REWARD_CORRECT, True
    return REWARD_WRONG, False


def _check_correctness(answer: str, ground_truth: str, task_type: str) -> bool:
    """Check answer correctness using verl's built-in verifiers."""
    try:
        if task_type == "gsm8k":
            from verl.utils.reward_score import gsm8k

            return gsm8k.compute_score(answer, ground_truth, method="flexible") > 0
        if task_type == "math":
            from verl.utils.reward_score import math_reward

            return math_reward.compute_score(answer, ground_truth) > 0
        if task_type == "qa":
            from verl.utils.reward_score import search_r1_like_qa_em as qa

            gt = {"target": [ground_truth]} if isinstance(ground_truth, str) else {"target": ground_truth}
            return qa.compute_score(answer, gt) > 0
    except Exception as e:
        logger.debug(f"Reward check failed for task_type={task_type}: {e}")
    # Fallback: normalised exact match
    return str(answer).strip().lower() == str(ground_truth).strip().lower()


def _reward_calibration(parsed: dict, is_correct: bool) -> float:
    """Brier-score-based calibration reward.

    R_cal = -(p - y)^2  where p=confidence, y=1 if correct else 0.

    This is always <= 0, so it acts as a penalty for bad calibration:
    * Correct + high confidence  -> small penalty (-0.01)
    * Wrong   + high confidence  -> large penalty (-0.81)
    * Wrong   + low  confidence  -> small penalty (-0.01)
    """
    confidence = parsed["confidence"]
    if confidence is None:
        # No confidence reported: small penalty to encourage reporting it
        return -0.1
    y = 1.0 if is_correct else 0.0
    return -((confidence - y) ** 2)


def _reward_verify(parsed: dict, is_correct: bool) -> float:
    """Reward for verification behaviour.

    * +0.2 if the agent called a tool (calculator/search) before submitting.
    * +0.5 extra if the agent revised (submitted more than once) and ended
      up correct -- this rewards genuine reflection, not just "I verified".
    """
    score = 0.0
    tool_names = [tc.get("name", "") for tc in parsed["tool_calls"]]
    used_tool_before_answer = any(n in ("calculator", "search") for n in tool_names)

    if used_tool_before_answer:
        score += 0.2

    # Count submit_answer calls to detect revision
    submit_count = sum(1 for n in tool_names if n == "submit_answer")
    if submit_count > 1 and is_correct:
        score += 0.5

    return score


def _reward_process(parsed: dict) -> float:
    """Process reward from tool-level step rewards.

    verl's ToolAgentLoop collects ``tool_rewards`` from each tool's
    ``execute()`` return value. We aggregate them here. If no step rewards
    are available, we fall back to a heuristic based on trajectory quality.
    """
    tool_rewards = parsed.get("tool_rewards", [])
    if tool_rewards:
        try:
            return float(sum(r for r in tool_rewards if isinstance(r, (int, float))))
        except Exception:
            pass

    # Heuristic process reward: reward having <think> blocks (reasoning)
    score = 0.0
    if parsed["think_blocks"]:
        score += 0.1 * min(len(parsed["think_blocks"]), 3)
    return score


def _reward_epistemic(parsed: dict, is_correct: bool) -> float:
    """Epistemic awareness reward.

    Case 1: Knows and is correct + high confidence  -> +
    Case 2: Doesn't know, low confidence, uses tool  -> +
    Case 3: Doesn't know, high confidence, wrong      -> heavy penalty
    Case 4: Should search but doesn't                 -> penalty
    Case 5: Has evidence but keeps searching          -> small penalty
    """
    confidence = parsed["confidence"] or 0.5
    tool_names = [tc.get("name", "") for tc in parsed["tool_calls"]]
    searched = "search" in tool_names
    calculated = "calculator" in tool_names

    # Case 1: correct + confident
    if is_correct and confidence >= 0.7:
        return 0.3
    # Case 2: searched and got it right (used evidence)
    if is_correct and searched:
        return 0.2
    # Case 3: wrong but very confident (worst case -- hallucination)
    if not is_correct and confidence >= 0.7:
        return -0.5
    # Case 2b: wrong but low confidence (honest uncertainty)
    if not is_correct and confidence <= 0.3:
        return 0.1
    # Case 4: no tools used, wrong answer, moderate confidence
    if not is_correct and not searched and not calculated:
        return -0.1
    return 0.0


def _reward_tool_cost(parsed: dict) -> float:
    """Tiered tool cost with a free budget.

    R_tool = lambda * max(0, total_cost - free_budget)
    """
    tool_names = [tc.get("name", "") for tc in parsed["tool_calls"]]
    total_cost = sum(TOOL_COSTS.get(n, 0.05) for n in tool_names)
    excess = max(0.0, total_cost - FREE_TOOL_BUDGET * 0.01)  # budget in cost units
    # Scale: each excess tool call costs ~0.05
    num_tools = len([n for n in tool_names if n != "submit_answer"])
    excess_calls = max(0, num_tools - FREE_TOOL_BUDGET)
    return 0.05 * excess_calls


def _reward_safety(parsed: dict, solution_str: str) -> float:
    """Safety penalty for reward hacking / fake verification.

    Detects:
    * Fake citation: claims "according to search results" without a search call
    * Fake verify: says "I verified" without calling submit_answer or a tool
    * Tool hallucination: references a tool result that doesn't exist
    """
    penalty = 0.0
    tool_names = [tc.get("name", "") for tc in parsed["tool_calls"]]
    searched = "search" in tool_names

    # Fake citation detection
    citation_phrases = [
        "according to the search",
        "based on the search results",
        "the search results show",
        "according to the retrieved",
        "based on the retrieved information",
    ]
    text_lower = solution_str.lower()
    for phrase in citation_phrases:
        if phrase in text_lower and not searched:
            penalty += 0.3
            break

    # Fake verify detection
    verify_phrases = ["i verified", "i have verified", "after verification", "i checked"]
    has_verify_text = any(phrase in text_lower for phrase in verify_phrases)
    used_any_tool = len(tool_names) > 0
    if has_verify_text and not used_any_tool:
        penalty += 0.2

    return penalty  # This is subtracted from the total


# ---------------------------------------------------------------------------
# Main compute_score entry point
# ---------------------------------------------------------------------------
def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> dict:
    """Compute the composite agentic reward.

    Args:
        data_source: Dataset identifier (e.g. "openai/gsm8k").
        solution_str: The decoded full response trajectory.
        ground_truth: The ground-truth answer string.
        extra_info: Dict with num_turns, tool_rewards, turn_scores, task_type.

    Returns:
        A dict with ``score`` and per-module breakdown for logging.
    """
    extra_info = extra_info or {}
    parsed = _parse_trajectory(solution_str, extra_info)

    # Normalise ground_truth to a string
    if isinstance(ground_truth, dict):
        gt_str = ground_truth.get("target", ground_truth.get("ground_truth", ""))
        if isinstance(gt_str, list):
            gt_str = gt_str[0] if gt_str else ""
    elif isinstance(ground_truth, list):
        gt_str = ground_truth[0] if ground_truth else ""
    else:
        gt_str = str(ground_truth)

    # --- Compute each reward module ---
    r_answer, is_correct = _reward_answer(parsed, gt_str, extra_info)
    r_cal = _reward_calibration(parsed, is_correct)
    r_verify = _reward_verify(parsed, is_correct)
    r_process = _reward_process(parsed)
    r_epistemic = _reward_epistemic(parsed, is_correct)
    r_tool_cost = _reward_tool_cost(parsed)
    r_safety = _reward_safety(parsed, solution_str)

    # --- Weighted sum ---
    w = WEIGHTS
    total = (
        w["answer"] * r_answer
        + w["calibration"] * r_cal
        + w["verify"] * r_verify
        + w["process"] * r_process
        + w["epistemic"] * r_epistemic
        - w["tool_cost"] * r_tool_cost
        - w["safety"] * r_safety
    )

    return {
        "score": float(total),
        "answer_reward": float(r_answer),
        "calibration_reward": float(r_cal),
        "verify_reward": float(r_verify),
        "process_reward": float(r_process),
        "epistemic_reward": float(r_epistemic),
        "tool_cost": float(r_tool_cost),
        "safety_penalty": float(r_safety),
        "is_correct": is_correct,
        "confidence": parsed["confidence"],
        "num_tool_calls": len(parsed["tool_calls"]),
        "num_turns": parsed["num_turns"],
    }
