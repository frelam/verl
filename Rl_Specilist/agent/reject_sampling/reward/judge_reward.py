# Copyright 2025 Individual Contributor
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
"""DeepSeek API judge for reject sampling trajectory scoring.

This module implements the ``compute_score`` function with the standard
verl reward function signature, so it can be plugged into ``NaiveRewardManager``
via ``reward.custom_reward_function``.

Scoring logic:
  1. If rule-based signal is available (e.g. ``submit_answer`` correctness
     from tool execution, or SWE-bench test pass rate), use it directly.
  2. Otherwise, call DeepSeek API to judge the trajectory quality.
  3. Trajectories with score >= THRESHOLD are saved to disk for SFT.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Optional

from Rl_Specilist.agent.reject_sampling.reward.trajectory_collector import (
    hash_prompt,
    save_trajectory,
)

# ---------------------------------------------------------------------------
# Config (from environment variables)
# ---------------------------------------------------------------------------

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

# Judge threshold: trajectories with score >= this are saved for SFT
THRESHOLD = float(os.environ.get("REJECT_SAMPLING_THRESHOLD", "0.7"))

# When DPO_MODE=1, compute_score only returns the score and skips trajectory
# saving to disk (Online DPO consumes scores in-memory, no SFT data needed).
DPO_MODE = os.environ.get("DPO_MODE", "0") == "1"

# API call config
MAX_RETRIES = int(os.environ.get("JUDGE_MAX_RETRIES", "3"))
RETRY_DELAY = float(os.environ.get("JUDGE_RETRY_DELAY", "1.0"))
API_TIMEOUT = int(os.environ.get("JUDGE_API_TIMEOUT", "60"))
MAX_TRAJECTORY_TOKENS = int(os.environ.get("JUDGE_MAX_TRAJ_TOKENS", "8000"))

# Cache the OpenAI client (DeepSeek-compatible)
_client = None


def _get_client():
    """Lazy-init the OpenAI client (DeepSeek uses OpenAI-compatible API)."""
    global _client
    if _client is None:
        if not DEEPSEEK_API_KEY:
            raise RuntimeError(
                "DEEPSEEK_API_KEY not set. "
                "Export it: export DEEPSEEK_API_KEY=sk-xxxxx"
            )
        from openai import OpenAI

        _client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            timeout=API_TIMEOUT,
        )
    return _client


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are an expert judge evaluating the quality of an AI agent's trajectory.
You will be given:
1. The task (user's prompt)
2. The agent's full trajectory (including reasoning, tool calls, and observations)
3. The ground-truth reference answer (may be empty for open-ended tasks)

Evaluate the trajectory on the following criteria:
- Correctness: Does the agent arrive at the correct answer or complete the task?
- Tool usage: Are tools called appropriately and with correct parameters?
- Reasoning quality: Is the step-by-step reasoning sound and logical?
- Efficiency: Does the agent avoid unnecessary steps or redundant tool calls?
- Error recovery: If the agent encounters errors, does it recover appropriately?

Output a JSON object with:
{
  "score": <float 0.0-1.0>,
  "is_correct": <true|false>,
  "reasoning": "<brief explanation>"
}

Score guidelines:
- 1.0: Perfect trajectory, correct answer, efficient and clean
- 0.8: Correct answer with minor inefficiencies
- 0.6: Correct answer but messy trajectory or excessive steps
- 0.4: Incorrect answer but reasonable approach
- 0.2: Incorrect answer with poor reasoning
- 0.0: Completely wrong or nonsensical trajectory

Output ONLY the JSON object, no other text."""


def _format_messages_for_judge(messages: list[dict[str, Any]]) -> str:
    """Serialize a full multi-turn message list into a readable trajectory string.

    Each message is rendered as ``[role] content``. Tool observations (role=tool)
    are included so the judge can see the actual environment feedback, which is
    critical for evaluating agent trajectories with tool use.
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        # Truncate very long tool outputs
        if role == "tool" and len(str(content)) > 2000:
            content = str(content)[:2000] + "...[truncated]"
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


def _build_judge_prompt(
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any],
) -> list[dict[str, str]]:
    """Build the messages for the DeepSeek judge API call.

    Prefers the structured multi-turn ``messages`` from ``extra_info`` (exposed by
    ToolAgentLoop) over the decoded ``solution_str``, so the judge sees tool
    observations and the full conversation flow.
    """
    task = ""
    # Try to extract the original task from extra_info
    if extra_info.get("question"):
        task = str(extra_info["question"])
    elif extra_info.get("task"):
        task = str(extra_info["task"])

    # If no task in extra_info, try to extract from solution_str (first user message)
    if not task:
        # solution_str is the decoded response; we don't have the prompt here,
        # so use a generic description
        task = "(See agent trajectory for task context)"

    # Prefer structured multi-turn messages when available (Online DPO path);
    # fall back to decoded solution_str for legacy reject_sampling callers.
    full_messages = extra_info.get("messages")
    if full_messages:
        traj = _format_messages_for_judge(full_messages)
    else:
        traj = solution_str

    # Truncate trajectory if too long
    if len(traj) > MAX_TRAJECTORY_TOKENS * 4:  # rough char-to-token estimate
        traj = traj[: MAX_TRAJECTORY_TOKENS * 4] + "\n...[trajectory truncated]"

    user_content = f"""## Task
{task}

## Agent Trajectory
{traj}

## Ground Truth Reference
{ground_truth if ground_truth else "(no ground truth available — judge based on trajectory quality)"}

## Dataset
{extra_info.get("dataset", "unknown")}

Evaluate the agent's trajectory and output a JSON score."""

    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# DeepSeek API call
# ---------------------------------------------------------------------------

def _call_deepseek_judge(
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any],
) -> tuple[float, bool, str]:
    """Call DeepSeek API to judge a trajectory.

    Returns: (score, is_correct, reasoning)
    """
    messages = _build_judge_prompt(solution_str, ground_truth, extra_info)
    client = _get_client()

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=messages,
                max_tokens=512,
                temperature=0.0,  # Deterministic judging
            )
            content = response.choices[0].message.content.strip()

            # Parse JSON from response
            # Try direct parse first
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code block
                json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(1))
                else:
                    # Try to find any JSON object in the text
                    json_match = re.search(r"\{[^{}]*\"score\"[^{}]*\}", content, re.DOTALL)
                    if json_match:
                        result = json.loads(json_match.group(0))
                    else:
                        raise ValueError(f"Cannot parse JSON from judge response: {content[:200]}")

            score = float(result.get("score", 0.0))
            score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
            is_correct = bool(result.get("is_correct", score >= 0.7))
            reasoning = str(result.get("reasoning", ""))

            return score, is_correct, reasoning

        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            continue

    # All retries failed
    print(f"[judge_reward] DeepSeek API failed after {MAX_RETRIES} retries: {last_error}")
    return 0.0, False, f"API error: {last_error}"


# ---------------------------------------------------------------------------
# Trajectory reconstruction
# ---------------------------------------------------------------------------

def _reconstruct_messages(
    solution_str: str,
    prompt_messages: list[dict] | None,
    extra_info: dict[str, Any],
) -> list[dict[str, str]]:
    """Reconstruct the full messages list for SFT.

    The ``solution_str`` is the decoded LLM response. The full trajectory
    (with tool observations) needs to be reconstructed from extra_info
    or the raw rollout data.

    If ``prompt_messages`` is available (from non_tensor_batch), use it as
    the prefix. Otherwise, construct a minimal prompt.
    """
    messages = []

    if prompt_messages:
        # Use the original prompt messages (system + user)
        messages.extend(prompt_messages)
    else:
        # Fallback: construct from extra_info
        messages.append({
            "role": "system",
            "content": "You are an AI assistant with tool access.",
        })
        task = extra_info.get("question", extra_info.get("task", ""))
        if task:
            messages.append({"role": "user", "content": str(task)})

    # Append the agent's response as assistant message
    messages.append({"role": "assistant", "content": solution_str})

    return messages


# ---------------------------------------------------------------------------
# Main entry point — verl reward function signature
# ---------------------------------------------------------------------------

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> dict:
    """Compute the reject sampling score for a trajectory.

    This function is called by ``NaiveRewardManager`` for each generated
    trajectory. It:
      1. Checks for rule-based correctness signals (from tool execution).
      2. Falls back to DeepSeek API judge for open-ended tasks.
      3. Saves trajectories above threshold to disk for SFT.

    Args:
        data_source: Dataset identifier (e.g. "toolmind", "terminaltraj").
        solution_str: The decoded full response trajectory from the LLM.
        ground_truth: The ground-truth answer string.
        extra_info: Dict with rollout metadata (num_turns, tool_rewards, etc.).

    Returns:
        Dict with ``score`` and metadata for logging.
    """
    extra_info = extra_info or {}

    # Normalise ground_truth to string
    if isinstance(ground_truth, dict):
        gt_str = ground_truth.get("target", ground_truth.get("ground_truth", ""))
        if isinstance(gt_str, list):
            gt_str = gt_str[0] if gt_str else ""
    elif isinstance(ground_truth, list):
        gt_str = ground_truth[0] if ground_truth else ""
    else:
        gt_str = str(ground_truth) if ground_truth else ""

    # --- 1. Check rule-based signals ---
    rollout_reward_scores = extra_info.get("rollout_reward_scores", {})
    rule_correct = rollout_reward_scores.get("is_correct", None)
    tool_rewards = rollout_reward_scores.get("tool_rewards", [])

    # If tool execution gave a definitive correctness signal, use it
    if rule_correct is True:
        score = 1.0
        is_correct = True
        judge_source = "rule"
        reasoning = "Rule-based verification: correct"
    elif rule_correct is False:
        # Rule says incorrect — still call judge for partial credit
        score, is_correct, reasoning = _call_deepseek_judge(solution_str, gt_str, extra_info)
        judge_source = "deepseek_after_rule_fail"
    else:
        # --- 2. No rule-based signal → call DeepSeek judge ---
        score, is_correct, reasoning = _call_deepseek_judge(solution_str, gt_str, extra_info)
        judge_source = "deepseek"

    # --- 3. Save trajectory if above threshold (skipped in DPO mode) ---
    saved = False
    if not DPO_MODE and score >= THRESHOLD:
        try:
            # Reconstruct full messages for SFT
            prompt_messages = extra_info.get("prompt_messages")
            # Prefer the full multi-turn messages from extra_info when available;
            # otherwise fall back to reconstructing from solution_str.
            full_messages = extra_info.get("messages")
            if full_messages:
                messages = full_messages
            else:
                messages = _reconstruct_messages(solution_str, prompt_messages, extra_info)
            tools = extra_info.get("tools", [])
            prompt_hash = hash_prompt(prompt_messages or [{"role": "user", "content": gt_str}])

            save_trajectory(
                messages=messages,
                tools=tools,
                data_source=data_source,
                score=score,
                judge_source=judge_source,
                prompt_hash=prompt_hash,
                extra={
                    "num_turns": extra_info.get("num_turns"),
                    "num_tool_calls": len(tool_rewards),
                    "reasoning": reasoning,
                    "is_correct": is_correct,
                },
            )
            saved = True
        except Exception as e:
            print(f"[judge_reward] Failed to save trajectory: {e}")

    return {
        "score": float(score),
        "is_correct": is_correct,
        "judge": judge_source,
        "judge_reasoning": reasoning,
        "saved_for_sft": saved,
        "threshold": THRESHOLD,
        "num_turns": extra_info.get("num_turns"),
        "num_tool_calls": len(tool_rewards),
    }
