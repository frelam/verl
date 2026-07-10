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
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ── Prompt loading ───────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Rl_Specilist/agent/online_dpo/
_PROMPTS_DIR = _PROJECT_ROOT / "prompts"

JUDGE_PROMPT_MAP: dict[str, str | None] = {
    # Exact matches for known data_source values
    "math": "math_judge.txt",
    "coding": "coding_judge.txt",
    "terminal": "terminal_judge.txt",
    # Fuzzy matches — data_source values from extract_prompts.py
    "terminaltraj": "terminal_judge.txt",
    "toolmind": None,        # no domain-specific prompt yet; uses default rubric
    "swe_zero": "coding_judge.txt",
    "open_swe_traces": "coding_judge.txt",
}


def load_judge_prompt(data_source: str) -> str | None:
    """Load a dataset-specific judge prompt from ``prompts/``.

    Args:
        data_source: Dataset identifier (e.g. ``"terminaltraj"``, ``"math"``).

    Returns:
        The prompt text if a matching file is found, or ``None`` if no
        domain-specific prompt is configured (caller should use the default).
    """
    data_source = (data_source or "").strip().lower()
    if not data_source:
        return None

    filename = JUDGE_PROMPT_MAP.get(data_source)
    if filename is None:
        # Try fuzzy match: check if any known key is a substring of data_source
        for key, fname in JUDGE_PROMPT_MAP.items():
            if fname and key in data_source:
                filename = fname
                break

    if filename is None:
        logger.info("No judge prompt configured for data_source=%r; using default rubric.", data_source)
        return None

    prompt_path = _PROMPTS_DIR / filename
    if not prompt_path.is_file():
        logger.warning("Judge prompt file not found: %s; using default rubric.", prompt_path)
        return None

    logger.info("Loaded judge prompt for data_source=%r from %s", data_source, prompt_path)
    return prompt_path.read_text(encoding="utf-8")


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
    system_prompt: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Score a single agent trajectory.

    Args:
        task: The original task / prompt.
        agent_output: The agent's stdout (final answer + tool logs).
        rubric: Custom scoring rubric.  If None, uses a default.
        system_prompt: Custom system prompt for the judge model.
            If provided, replaces the default system message.
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

    system_content = system_prompt or "You are an expert evaluator. Always respond with valid JSON."

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
                    {"role": "system", "content": system_content},
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
    system_prompt: str | None = None,
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
        system_prompt: Custom system prompt for the judge model.
            If provided, replaces the default system message.
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

    system_content = system_prompt or "You are an expert evaluator. Always respond with valid JSON."

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
                    {"role": "system", "content": system_content},
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


async def compute_score_batch(
    data_sources: list[str],
    solution_strs: list[str],
    ground_truths: list[Any],
    extra_infos: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Batch-level judge scoring compatible with ``BatchRewardManager``.

    Groups trajectories by ``data_source``, loads the appropriate
    dataset-specific judge prompt for each group, and calls
    ``judge_batch`` for relative scoring within each group.

    Args:
        data_sources: Dataset identifiers (e.g. ``"terminaltraj"``, ``"math"``).
        solution_strs: Agent output strings.
        ground_truths: Ground-truth answers (unused by judge, kept for compatibility).
        extra_infos: Per-sample metadata dicts.
        **kwargs: Additional arguments (unused).

    Returns:
        List of dicts, each with ``score`` (float) and optional metadata.
    """
    extra_infos = extra_infos or [{} for _ in solution_strs]
    n = len(solution_strs)
    if n == 0:
        return []

    # Build task strings from extra_info
    tasks: list[str] = []
    for i in range(n):
        extra = extra_infos[i] if i < len(extra_infos) else {}
        task = (
            extra.get("question")
            or extra.get("task")
            or f"Task from {data_sources[i]}"
        )
        tasks.append(str(task))

    # Group indices by data_source
    groups: dict[str, list[int]] = {}
    for i, ds in enumerate(data_sources):
        ds_key = (ds or "unknown").strip().lower()
        groups.setdefault(ds_key, []).append(i)

    # Score each group with its dataset-specific prompt
    all_scores: list[dict[str, Any] | None] = [None] * n

    for ds, indices in groups.items():
        prompt = load_judge_prompt(ds)
        group_tasks = [tasks[i] for i in indices]
        group_outputs = [solution_strs[i][:12000] for i in indices]

        try:
            group_scores = await judge_batch(
                tasks=group_tasks,
                outputs=group_outputs,
                rubric=prompt,  # dataset-specific prompt or None (default)
            )
        except Exception:
            logger.warning(
                "Batch judge failed for data_source=%r; assigning default scores.",
                ds,
                exc_info=True,
            )
            group_scores = [
                {"reward_score": 0.5, "judge_reason": "batch judge API error"}
                for _ in group_tasks
            ]

        for local_idx, score_dict in zip(indices, group_scores):
            result = {
                "score": float(score_dict.get("reward_score", 0.0)),
                "judge_reasoning": str(score_dict.get("judge_reason", "")),
                "data_source": ds,
            }
            # Forward any extra dimension scores
            for key, value in score_dict.items():
                if key not in ("reward_score", "judge_reason", "score"):
                    result[key] = value
            all_scores[local_idx] = result

    # Ensure every position has a score
    for i in range(n):
        if all_scores[i] is None:
            all_scores[i] = {"score": 0.0, "data_source": data_sources[i]}

    return all_scores
