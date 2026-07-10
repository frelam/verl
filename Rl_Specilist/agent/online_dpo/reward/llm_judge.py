"""LLM Judge reward for Gateway-based Hermes agent DPO training.

Provides two interfaces:

1. **``compute_score``** — compatible with verl's ``BatchRewardManager``
   for batch-level LLM Judge scoring.  Sends all trajectories to the Judge
   model together for relative scoring.

2. **``judge_single``** — scores a single trajectory, used by the runner
   inline (``custom_hermes_runner._evaluate_reward`` calls this).

Configuration (via environment variables):
    JUDGE_MODEL       — model name (default: deepseek-chat)
    JUDGE_BASE_URL    — API base URL (default: https://api.deepseek.com)
    DEEPSEEK_API_KEY  — API key (required)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────────────────────


def _judge_config() -> tuple[str, str, str]:
    """Return (model, base_url, api_key) from environment."""
    model = os.environ.get("JUDGE_MODEL", "deepseek-chat")
    base_url = os.environ.get("JUDGE_BASE_URL", "https://api.deepseek.com")
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("JUDGE_API_KEY", "")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY or JUDGE_API_KEY environment variable is required")
    return model, base_url, api_key


# ── Scoring ─────────────────────────────────────────────────────────────────


async def judge_single(
    task: str,
    agent_output: str,
    *,
    rubric: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Score a single agent trajectory.

    Args:
        task: The original task / prompt.
        agent_output: The agent's stdout (final answer + tool logs).
        rubric: Custom scoring rubric.  If None, uses a default.
        model: Judge model name.
        base_url: LLM Judge API base URL.
        api_key: LLM Judge API key.

    Returns:
        dict with keys: ``reward_score`` (float 0-1), ``judge_reason`` (str).
    """
    _model, _base_url, _api_key = _judge_config()
    model = model or _model
    base_url = base_url or _base_url
    api_key = api_key or _api_key

    if rubric is None:
        rubric = (
            "Evaluate whether the agent successfully completed the task.\n"
            "Consider:\n"
            "1. Did the agent produce the correct output?\n"
            "2. Was the approach reasonable and efficient?\n"
            "3. Did the agent use tools appropriately?\n\n"
            "Score: 1.0 = fully correct, 0.5 = partially correct, 0.0 = incorrect."
        )

    judge_prompt = (
        f"## Task\n{task[:2000]}\n\n"
        f"## Agent Output\n{agent_output[:3000]}\n\n"
        f"## Scoring Rubric\n{rubric}\n\n"
        "Respond with a JSON object:\n"
        '{"score": <0.0-1.0 float>, "reason": "<brief explanation>"}'
    )

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{base_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are an expert evaluator. Always respond with valid JSON."},
                    {"role": "user", "content": judge_prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 256,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    # Parse response
    judge_text = data["choices"][0]["message"]["content"]
    try:
        # Try JSON parse
        judge_result = json.loads(judge_text)
        score = float(judge_result.get("score", 0.5))
        reason = judge_result.get("reason", judge_text[:200])
    except (json.JSONDecodeError, KeyError, ValueError):
        score = 0.5
        reason = judge_text[:200]

    return {
        "reward_score": min(max(score, 0.0), 1.0),
        "judge_reason": reason,
    }


async def judge_batch(
    tasks: list[str],
    outputs: list[str],
    *,
    rubric: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Score a batch of agent trajectories with relative comparison.

    All trajectories are sent together, enabling the Judge to compare
    them against each other (relative scoring for DPO).

    Args:
        tasks: List of task strings.
        outputs: List of agent output strings (same length as tasks).
        rubric: Custom scoring rubric.
        model: Judge model name.
        base_url: LLM Judge API base URL.
        api_key: LLM Judge API key.

    Returns:
        List of dicts, each with ``reward_score`` and ``judge_reason``.
    """
    _model, _base_url, _api_key = _judge_config()
    model = model or _model
    base_url = base_url or _base_url
    api_key = api_key or _api_key

    if rubric is None:
        rubric = (
            "Evaluate each agent's performance on its task.\n"
            "Score each agent relative to the others in this batch.\n"
            "Score: 1.0 = excellent (best in batch), 0.7 = good, "
            "0.5 = acceptable, 0.3 = poor, 0.0 = failed.\n\n"
            "IMPORTANT: Use the FULL range 0.0-1.0. Do not cluster scores together."
        )

    # Build batch evaluation prompt
    samples_text = ""
    for i, (task, output) in enumerate(zip(tasks, outputs)):
        samples_text += (
            f"### Sample {i}\n"
            f"**Task:** {task[:800]}\n"
            f"**Output:** {output[:1200]}\n\n"
        )

    judge_prompt = (
        f"Evaluate the following {len(tasks)} agent trajectories.\n\n"
        f"{samples_text}"
        f"## Scoring Rubric\n{rubric}\n\n"
        "Respond with a JSON object:\n"
        '{"scores": [<float for sample 0>, <float for sample 1>, ...], '
        '"reasons": ["<reason for sample 0>", "<reason for sample 1>", ...]}'
    )

    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(
            f"{base_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are an expert evaluator. Always respond with valid JSON."},
                    {"role": "user", "content": judge_prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 1024,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    # Parse response
    judge_text = data["choices"][0]["message"]["content"]
    try:
        judge_result = json.loads(judge_text)
        scores = judge_result.get("scores", [])
        reasons = judge_result.get("reasons", [])
    except (json.JSONDecodeError, KeyError):
        # Fallback: equal scores
        scores = [0.5] * len(tasks)
        reasons = [judge_text[:200]] * len(tasks)

    # Ensure correct length
    while len(scores) < len(tasks):
        scores.append(0.5)
    while len(reasons) < len(tasks):
        reasons.append("")

    return [
        {
            "reward_score": min(max(float(s), 0.0), 1.0),
            "judge_reason": r[:500] if r else "",
        }
        for s, r in zip(scores[:len(tasks)], reasons[:len(tasks)])
    ]


# ── Verl batch reward interface ─────────────────────────────────────────────


def compute_score(
    data_source: Any,
    solution_str: Any,
    ground_truth: Any = None,
    extra_info: Any = None,
    **kwargs: Any,
) -> float:
    """Verl-compatible reward function for BatchRewardManager / RewardLoopWorker.

    The runner (``custom_hermes_runner``) posts ``reward_info`` (including
    ``reward_score``) to the Gateway session's ``reward_info_url``.  The
    framework merges that into ``extra_info``, so this function simply reads
    the already-computed score.

    When no pre-computed score is present (e.g. validation, or non-Gateway
    mode), this falls back to 0.0 — the caller should arrange for scoring
    via ``judge_single`` or ``judge_batch`` upstream.

    Args:
        data_source: Task identifier or source.
        solution_str: The agent's response text.
        ground_truth: Optional ground truth for comparison.
        extra_info: Additional metadata, may contain ``reward_score``.
        **kwargs: Additional verl-injected arguments.

    Returns:
        A reward score (0.0 - 1.0).
    """
    if extra_info and isinstance(extra_info, dict):
        if "reward_score" in extra_info:
            return float(extra_info["reward_score"])
    return 0.0
