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
"""Configurable LLM Judge for batch-level trajectory scoring.

This module implements a ``compute_score`` function compatible with
``BatchRewardManager`` (batch-level) for agent DPO training. Unlike the
per-sample judge in ``judge_reward.py``, this module sends ALL trajectories
in a batch to the judge model together, enabling **relative scoring** where
the judge compares samples within the batch.

Key features:
- **Batch relative scoring**: Judge sees all trajectories at once, ranks them
  relative to each other
- **Configurable system prompt**: Users specify evaluation criteria (format,
  planning correctness, tool call accuracy, tool efficiency, planning
  efficiency) via YAML config or JSON
- **Multi-provider support**: DeepSeek API, OpenAI API, or any OpenAI-compatible
  endpoint (local vLLM/SGLang)
- **Multi-dimensional scoring**: Optional per-dimension scores for GDPO
  (accuracy_reward, format_reward, efficiency_reward, etc.)

Usage:
    # In verl config:
    reward:
      reward_manager:
        source: register
        name: batch              # MUST use BatchRewardManager
      custom_reward_function:
        path: verl.utils.reward_score.llm_judge_reward
        name: compute_score
      judge_config:
        model: "deepseek-chat"
        base_url: "https://api.deepseek.com"
        api_key: "${oc.env:DEEPSEEK_API_KEY}"
        system_prompt: "default"  # or path to custom prompt file
        scoring_mode: "relative"  # "relative" or "absolute"
        dimensions:
          - accuracy_reward
          - format_reward
          - efficiency_reward
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Optional

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# ---------------------------------------------------------------------------
# Default judge system prompts
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """You are an expert evaluator for AI agent trajectories. You will be given
a set of trajectories from the same task. Your job is to evaluate and score
each trajectory RELATIVE to the others in the batch.

Evaluation Criteria:
1. **Format adherence**: Does the agent follow the expected output format?
   Does it properly use tool calling syntax?
2. **Planning correctness**: Is the agent's reasoning and step-by-step plan
   logically sound? Does it identify the right approach to solve the task?
3. **Tool call accuracy**: Are tool calls made with correct parameters?
   Does the agent select the appropriate tool for each sub-task?
4. **Tool call efficiency**: Does the agent minimize unnecessary or redundant
   tool calls? Is the number of calls proportional to the task complexity?
5. **Planning efficiency**: Does the agent take a direct path to the solution,
   or does it wander with unnecessary steps?

For each trajectory, provide:
- A relative score (0.0 to 1.0) compared to others in this batch
- A brief justification (1-2 sentences)

Output a JSON object with this exact structure:
{
  "scores": [
    {"index": 0, "score": 0.85, "reasoning": "..."},
    {"index": 1, "score": 0.42, "reasoning": "..."},
    ...
  ]
}

IMPORTANT:
- Scores must be RELATIVE within this batch. The best trajectory should have
  the highest score, the worst the lowest.
- Use the FULL 0.0-1.0 range. Don't cluster all scores near 0.7-0.9.
- At least one trajectory should score <= 0.3 and at least one >= 0.8.
- Output ONLY the JSON object, no other text."""

GDPO_SYSTEM_PROMPT = """You are an expert evaluator for AI agent trajectories. You will be given
a set of trajectories from the same task. Your job is to evaluate each trajectory
on MULTIPLE INDEPENDENT DIMENSIONS, scoring relative to others in the batch.

Evaluation Dimensions:
1. **accuracy_reward**: Did the agent accomplish the task correctly? (0.0-1.0)
2. **format_reward**: Did the agent follow the expected output/tool format? (0.0-1.0)
3. **planning_reward**: Was the agent's reasoning and planning logically sound? (0.0-1.0)
4. **tool_accuracy_reward**: Were tool calls correct with proper parameters? (0.0-1.0)
5. **efficiency_reward**: Was the trajectory efficient (minimal redundant steps)? (0.0-1.0)

Output a JSON object with this exact structure:
{
  "scores": [
    {
      "index": 0,
      "accuracy_reward": 0.9,
      "format_reward": 1.0,
      "planning_reward": 0.8,
      "tool_accuracy_reward": 0.7,
      "efficiency_reward": 0.6,
      "reasoning": "..."
    },
    ...
  ]
}

IMPORTANT:
- Each dimension is scored 0.0 to 1.0, RELATIVE within this batch.
- Use the FULL range. Don't cluster scores.
- Output ONLY the JSON object, no other text."""

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _get_judge_config(extra_info: dict | None = None) -> dict[str, Any]:
    """Extract judge configuration from environment variables.

    These can also be overridden via ``extra_info`` or set as env vars.
    Priority: extra_info > env var > default.

    Supported keys:
        judge_model: Model name for the judge API.
        judge_base_url: Base URL for the judge API.
        judge_api_key: API key for the judge.
        judge_system_prompt: "default", "gdpo", or a custom prompt string.
        judge_scoring_mode: "relative" (default) or "absolute".
        judge_dimensions: JSON list of dimension names.
        judge_max_tokens: Max tokens in judge response (default 1024).
        judge_temperature: Judge sampling temperature (default 0.0).
        judge_max_retries: Max API call retries (default 3).
        judge_api_timeout: API timeout in seconds (default 120).
    """
    extra_info = extra_info or {}

    def _get(key: str, default: str = "") -> str:
        ek = f"judge_{key}"
        env_key = f"JUDGE_{key.upper()}"
        if extra_info.get(ek):
            return str(extra_info[ek])
        if extra_info.get(key):
            return str(extra_info[key])
        return os.environ.get(env_key, default)

    config = {
        "model": _get("model", "deepseek-chat"),
        "base_url": _get("base_url", "https://api.deepseek.com"),
        "api_key": _get("api_key", os.environ.get("DEEPSEEK_API_KEY", "")),
        "system_prompt": _get("system_prompt", "default"),
        "scoring_mode": _get("scoring_mode", "relative"),
        "dimensions": _get("dimensions", ""),
        "max_tokens": int(_get("max_tokens", "1024")),
        "temperature": float(_get("temperature", "0.0")),
        "max_retries": int(_get("max_retries", "3")),
        "api_timeout": int(_get("api_timeout", "120")),
    }

    if config.get("dimensions"):
        try:
            config["dimensions"] = json.loads(config["dimensions"])
        except (json.JSONDecodeError, TypeError):
            config["dimensions"] = [
                d.strip() for d in str(config["dimensions"]).split(",")
            ]

    return config


def _resolve_system_prompt(config: dict[str, Any]) -> str:
    """Resolve the system prompt from config.

    - "default": Use DEFAULT_SYSTEM_PROMPT (single score, relative).
    - "gdpo": Use GDPO_SYSTEM_PROMPT (multi-dimensional, relative).
    - A file path: Read from file.
    - Any other string: Use directly as the system prompt.
    """
    prompt_spec = config.get("system_prompt", "default")
    if prompt_spec == "default":
        return DEFAULT_SYSTEM_PROMPT
    elif prompt_spec == "gdpo":
        return GDPO_SYSTEM_PROMPT
    elif os.path.isfile(prompt_spec):
        with open(prompt_spec, "r", encoding="utf-8") as f:
            return f.read()
    else:
        # Treat as inline prompt
        return str(prompt_spec)


# ---------------------------------------------------------------------------
# API client (lazy init, supports provider switching)
# ---------------------------------------------------------------------------

_client: Any = None
_client_config_hash: int = 0


def _get_client(config: dict[str, Any]) -> Any:
    """Lazy-init OpenAI-compatible client. Re-inits if config changes."""
    global _client, _client_config_hash
    config_hash = hash(
        (config["base_url"], config["api_key"], config["api_timeout"])
    )
    if _client is None or config_hash != _client_config_hash:
        if not config["api_key"]:
            raise RuntimeError(
                "Judge API key not set. Set JUDGE_API_KEY env var "
                "or pass judge_api_key in config."
            )
        from openai import OpenAI

        _client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
            timeout=config["api_timeout"],
        )
        _client_config_hash = config_hash
    return _client


# ---------------------------------------------------------------------------
# Trajectory formatting
# ---------------------------------------------------------------------------


def _format_messages_for_judge(messages: list[dict[str, Any]]) -> str:
    """Serialize a full multi-turn message list into a readable trajectory."""
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role == "tool" and len(str(content)) > 2000:
            content = str(content)[:2000] + "...[truncated]"
        # Include tool_call_id for tool messages
        tc_id = msg.get("tool_call_id", "")
        tc_info = f" [call_id={tc_id}]" if tc_id else ""
        parts.append(f"[{role}]{tc_info}\n{content}")
    return "\n\n".join(parts)


def _extract_trajectory_text(data_item: dict, extra_info: dict) -> str:
    """Extract the trajectory text from a data item.

    Prefers the full multi-turn ``messages`` from extra_info (exposed by
    ToolAgentLoop), falling back to decoded response text.
    """
    full_messages = extra_info.get("messages")
    if full_messages:
        return _format_messages_for_judge(full_messages)

    # Fallback: use decoded response
    if isinstance(data_item, dict) and "response_str" in data_item:
        return data_item["response_str"]
    return str(data_item)


# ---------------------------------------------------------------------------
# Judge API call (batch)
# ---------------------------------------------------------------------------


def _call_judge_batch(
    trajectories: list[str],
    tasks: list[str],
    ground_truths: list[str],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Send a batch of trajectories to the judge and parse the response.

    Args:
        trajectories: List of trajectory text strings.
        tasks: List of task descriptions.
        ground_truths: List of ground truth strings.
        config: Judge configuration dict.

    Returns:
        List of per-trajectory score dicts, each containing at least
        ``score`` and optionally per-dimension scores.
    """
    system_prompt = _resolve_system_prompt(config)
    client = _get_client(config)

    # Build the batch content
    batch_parts = []
    for i, (traj, task, gt) in enumerate(
        zip(trajectories, tasks, ground_truths)
    ):
        # Truncate long trajectories
        max_chars = int(os.environ.get("JUDGE_MAX_TRAJ_CHARS", "12000"))
        if len(traj) > max_chars:
            traj = traj[:max_chars] + "\n...[trajectory truncated]"

        batch_parts.append(
            f"### Trajectory {i}\n"
            f"**Task**: {task}\n"
            f"**Reference answer**: {gt if gt else '(none)'}\n"
            f"**Trajectory**:\n{traj}\n"
        )

    user_content = "\n---\n".join(batch_parts)
    user_content += (
        f"\n\nEvaluate all {len(trajectories)} trajectories above "
        f"and output a JSON object with scores."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    last_error = None
    for attempt in range(config["max_retries"]):
        try:
            response = client.chat.completions.create(
                model=config["model"],
                messages=messages,
                max_tokens=config["max_tokens"],
                temperature=config["temperature"],
            )
            content = response.choices[0].message.content.strip()

            # Parse JSON from response
            parsed = _parse_judge_response(content, len(trajectories))
            logger.debug(
                f"Judge response parsed: {len(parsed)} scores for "
                f"{len(trajectories)} trajectories"
            )
            return parsed

        except Exception as e:
            last_error = e
            if attempt < config["max_retries"] - 1:
                backoff = 2**attempt
                time.sleep(min(backoff, 30))
                continue

    logger.error(
        f"Judge API failed after {config['max_retries']} retries: {last_error}"
    )
    # Return zero scores on failure
    return [
        {"score": 0.0, "reasoning": f"API error: {last_error}"}
        for _ in trajectories
    ]


def _parse_judge_response(
    content: str, num_expected: int
) -> list[dict[str, Any]]:
    """Parse the judge's JSON response, with fallbacks for common issues."""
    # Try direct parse
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code block
        json_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL
        )
        if json_match:
            result = json.loads(json_match.group(1))
        else:
            # Try to find any JSON object
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
            else:
                raise ValueError(
                    f"Cannot parse JSON from judge response: {content[:500]}"
                )

    # Extract scores list
    if isinstance(result, dict) and "scores" in result:
        scores = result["scores"]
    elif isinstance(result, list):
        scores = result
    else:
        raise ValueError(
            f"Unexpected judge response format: {type(result)}"
        )

    # Ensure we have the right number of scores
    if len(scores) < num_expected:
        logger.warning(
            f"Judge returned {len(scores)} scores but expected {num_expected}. "
            f"Padding with zeros."
        )
        last_idx = max((s.get("index", i) for i, s in enumerate(scores)), default=0)
        for i in range(len(scores), num_expected):
            scores.append({"index": last_idx + i + 1 - len(scores) + len(scores), "score": 0.0})

    # Normalize: ensure "score" key exists
    for s in scores:
        if "score" not in s:
            # Try to compute from dimension scores
            dim_scores = [
                v for k, v in s.items()
                if k.endswith("_reward") and isinstance(v, (int, float))
            ]
            if dim_scores:
                s["score"] = sum(dim_scores) / len(dim_scores)
            else:
                s["score"] = 0.0

    return scores


# ---------------------------------------------------------------------------
# Individual trajectory judge (per-sample, non-batch)
# ---------------------------------------------------------------------------


def _call_judge_single(
    trajectory: str,
    task: str,
    ground_truth: str,
    config: dict[str, Any],
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Send a single trajectory to the judge and parse the response.

    Used for "absolute" scoring mode where each trajectory is scored
    independently.
    """
    system_prompt = _resolve_system_prompt(config)
    client = _get_client(config)

    max_chars = int(os.environ.get("JUDGE_MAX_TRAJ_CHARS", "12000"))
    if len(trajectory) > max_chars:
        trajectory = trajectory[:max_chars] + "\n...[trajectory truncated]"

    user_content = (
        f"## Task\n{task}\n\n"
        f"## Agent Trajectory\n{trajectory}\n\n"
        f"## Reference\n{ground_truth if ground_truth else '(none)'}\n\n"
        f"Evaluate this trajectory and output a JSON score."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    last_error = None
    for attempt in range(config["max_retries"]):
        try:
            response = client.chat.completions.create(
                model=config["model"],
                messages=messages,
                max_tokens=config["max_tokens"],
                temperature=config["temperature"],
            )
            content = response.choices[0].message.content.strip()
            scores = _parse_judge_response(content, 1)
            return scores[0] if scores else {"score": 0.0}
        except Exception as e:
            last_error = e
            if attempt < config["max_retries"] - 1:
                time.sleep(2**attempt)
                continue

    logger.error(f"Single judge call failed: {last_error}")
    return {"score": 0.0, "reasoning": f"API error: {last_error}"}


# ---------------------------------------------------------------------------
# Main entry point — BatchRewardManager compatible
# ---------------------------------------------------------------------------


def compute_score(
    data_sources: list[str],
    solution_strs: list[str],
    ground_truths: list[Any],
    extra_infos: list[dict[str, Any]] | None = None,
    **kwargs,
) -> list[dict[str, Any]]:
    """Compute batch-level judge scores for a group of trajectories.

    This function is compatible with ``BatchRewardManager``, which passes
    all samples as lists. The judge can see all trajectories together and
    produce relative scores.

    Args:
        data_sources: List of dataset identifiers.
        solution_strs: List of decoded response trajectories.
        ground_truths: List of ground-truth answers.
        extra_infos: List of per-sample metadata dicts. Each may contain:
            - ``messages``: Full multi-turn message list from ToolAgentLoop.
            - ``question`` / ``task``: The original task description.
            - ``num_turns``: Number of conversation turns.
            - ``judge_model``: Override judge model.
            - ``judge_base_url``: Override judge base URL.
            - ``judge_system_prompt``: Override system prompt.
            - ``judge_scoring_mode``: "relative" or "absolute".
            - ``judge_dimensions``: JSON list of dimension names.
        **kwargs: Additional keyword arguments.

    Returns:
        List of dicts, each containing at minimum ``score`` (float), plus
        optional per-dimension scores and metadata.
    """
    extra_infos = extra_infos or [{} for _ in range(len(solution_strs))]

    # Merge config from first extra_info (batch-level config is shared)
    config = _get_judge_config(extra_infos[0] if extra_infos else None)
    scoring_mode = config.get("scoring_mode", "relative")

    # Extract trajectories and tasks
    n = len(solution_strs)
    trajectories = []
    tasks = []
    gts: list[str] = []

    for i in range(n):
        extra = extra_infos[i] if i < len(extra_infos) else {}
        traj = _extract_trajectory_text(
            {"response_str": solution_strs[i]}, extra
        )
        trajectories.append(traj)

        task = (
            extra.get("question")
            or extra.get("task")
            or f"Task from {data_sources[i]}"
        )
        tasks.append(str(task))

        gt = ground_truths[i] if i < len(ground_truths) else ""
        if isinstance(gt, dict):
            gt = str(gt.get("target", gt.get("ground_truth", "")))
        gts.append(str(gt) if gt else "")

    # Call judge based on scoring mode
    if scoring_mode == "relative" and n > 1:
        scores = _call_judge_batch(trajectories, tasks, gts, config)
    else:
        # Absolute mode or single sample: score each independently
        scores = []
        for i in range(n):
            s = _call_judge_single(
                trajectories[i], tasks[i], gts[i], config, extra_infos[i]
            )
            s["index"] = i
            scores.append(s)

    # Build return dicts
    results = []
    dimensions = config.get("dimensions", [])
    for i in range(n):
        score_entry = scores[i] if i < len(scores) else {"score": 0.0}
        result = {
            "score": float(score_entry.get("score", 0.0)),
            "judge_reasoning": str(score_entry.get("reasoning", "")),
            "judge_source": config.get("model", "unknown"),
            "scoring_mode": scoring_mode,
        }

        # Multi-dimensional scores for GDPO
        if dimensions:
            for dim in dimensions:
                result[dim] = float(score_entry.get(dim, result["score"]))
        else:
            # Even without explicit dimension config, try to extract
            # dimension scores if the judge returned them
            for key, value in score_entry.items():
                if key.endswith("_reward") and key not in result:
                    result[key] = float(value)

        # Add metadata
        if i < len(extra_infos):
            extra = extra_infos[i]
            result["num_turns"] = extra.get("num_turns")
            result["tool_rewards"] = extra.get("tool_rewards", [])

        results.append(result)

    return results
