"""Tool RL reward for verl — 3 dimensions, rule-based (label mode only).

Ported from slime ``examples/tool_rl/reward/reward.py``. The LLM-judge
(RM v2) mode is intentionally NOT migrated — samples without structured
ground-truth tool calls are filtered out at data preparation time
(see ``examples/tool_rl/prepare_data.py``).

Reward Dimensions
-----------------
==============  ==============================  ======  =============
Dim             Name                            Weight  Source
==============  ==============================  ======  =============
Dim 1           工具调用正确性 (Tool Correctness)  0.60    Label match
Dim 2           回复格式合规 (Format)              0.20    Verifier
Dim 3           工具调用格式 (Tool Call Format)    0.20    Verifier
==============  ==============================  ======  =============

Dim 1 — rule-based, order-independent matching against ground-truth labels:
  - Tool name match  → 0.5  (binary per label call)
  - Param content    → 0.5  (value match per label param)
  Calling a tool not declared in the prompt incurs ``-0.1`` per call on
  Dim 1, floored at 0.0.

Dim 2 — Verifier (format):
  0.6 if all tool_calls after reasoning + 0.4 × count/N for think before
  each call.

Dim 3 — Verifier (tool call format vs label):
  1/N per label call completely matched (name + param names + param types);
  full score when the label has no tool calls.

verl integration
----------------
Loaded via ``reward.custom_reward_function.path`` / ``.name=compute_score``.
Per-sample fields travel in the dataset's ``extra_info`` column:

- ``extra_info["tools"]``              — available tool schemas
- ``extra_info["ground_truth_calls"]`` — structured GT tool calls
  (``[]`` = label says no tools needed)
- ``extra_info["task_id"]``            — for logging

Optional knobs (env vars):
- ``TOOL_RL_REWARD_WEIGHTS`` — JSON dict overriding dimension weights,
  e.g. ``'{"tool_correctness": 0.6, "format": 0.2, "tool_call": 0.2}'``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Make sibling / package imports work regardless of how this file is loaded
# (verl loads custom reward functions by file path via importlib).
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.tool_rl.reward.verifier import (  # noqa: E402
    compute_verifier_scores,
    match_tool_calls_against_label,
    parse_ground_truth_calls,
    parse_qwen_tool_calls,
    undeclared_tool_penalty,
)

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS: dict[str, float] = {
    "tool_correctness": 0.60,
    "format": 0.20,
    "tool_call": 0.20,
}


def _get_weights() -> dict[str, float]:
    """Resolve weights from ``TOOL_RL_REWARD_WEIGHTS`` JSON, else defaults."""
    raw = os.environ.get("TOOL_RL_REWARD_WEIGHTS")
    defaults = dict(DEFAULT_WEIGHTS)
    if not raw:
        return defaults
    try:
        override = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Invalid TOOL_RL_REWARD_WEIGHTS JSON: %r", raw)
        return defaults
    if not isinstance(override, dict):
        return defaults
    for k in defaults:
        if k in override:
            defaults[k] = float(override[k])
    total = sum(defaults.values())
    if total > 0:
        defaults = {k: v / total for k, v in defaults.items()}
    return defaults


def _to_list(value: Any) -> list:
    """Normalise parquet/numpy containers to a plain Python list."""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _to_dict_list(value: Any) -> list[dict[str, Any]]:
    """Normalise a (possibly numpy) list of tool-call/tool-schema dicts."""
    result = []
    for item in _to_list(value):
        if hasattr(item, "tolist"):
            item = item.tolist()
        if isinstance(item, dict):
            result.append(dict(item))
    return result


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """verl custom reward entry point (label mode only).

    Args:
        data_source: Dataset source tag (unused, kept for the verl protocol).
        solution_str: Decoded model response (``<think>`` + ``<tool_call>``).
        ground_truth: Readable ground-truth label string (logging only).
        extra_info: Per-sample fields — must carry ``tools`` and
            ``ground_truth_calls`` (see module docstring).

    Returns:
        Dict with ``score`` plus per-dimension breakdown; extra keys are
        surfaced by the reward manager as ``reward_extra_info`` metrics.
    """
    extra_info = dict(extra_info or {})
    available_tools = _to_dict_list(extra_info.get("tools"))
    ground_truth_calls = extra_info.get("ground_truth_calls", None)
    task_id = extra_info.get("task_id", "unknown")

    # ``None`` (no label) should have been filtered at data prep; degrade
    # gracefully to an empty label so the run does not crash.
    if ground_truth_calls is None:
        logger.warning(
            "[tool_rl] %s: no ground_truth_calls in extra_info — "
            "treating as empty label (RM mode is not migrated)",
            task_id,
        )
        ground_truth_calls = []

    weights = _get_weights()

    parsed_gt = parse_ground_truth_calls(_to_dict_list(ground_truth_calls))
    expects_no_tools = len(parsed_gt) == 0

    # Wrap the single-turn response as a pseudo-trajectory for the verifier.
    trajectory = [{"turn": 0, "text": solution_str, "type": "turn"}]
    output_calls = parse_qwen_tool_calls(solution_str)

    # ── Dim 2 + Dim 3: Verifier (rule-based) ──
    verifier = compute_verifier_scores(
        trajectory,
        available_tools=available_tools,
        label_calls=parsed_gt,
        expects_no_tools=expects_no_tools,
    )
    format_score = verifier["format_compliance"]
    tool_call_score = verifier["tool_call_format"]

    # ── Dim 1: rule-based label matching ──
    name_score, param_score = match_tool_calls_against_label(output_calls, parsed_gt)
    tool_correctness = 0.5 * name_score + 0.5 * param_score

    # Dim 1 undeclared-tool penalty: -0.1 per undeclared call, floored at 0.
    undeclared_penalty = undeclared_tool_penalty(output_calls, available_tools)
    if undeclared_penalty > 0:
        tool_correctness = max(0.0, tool_correctness - undeclared_penalty)

    # ── Weighted sum (negatives allowed so blind guessing scores < 0) ──
    total = (
        weights["tool_correctness"] * tool_correctness
        + weights["format"] * format_score
        + weights["tool_call"] * tool_call_score
    )
    total = min(1.0, total)

    logger.info(
        "[tool_rl] %s: total=%.3f correctness=%.3f(name=%.3f+param=%.3f) "
        "format=%.3f tool_call=%.3f",
        task_id, total, tool_correctness, name_score, param_score,
        format_score, tool_call_score,
    )

    return {
        "score": total,
        "tool_correctness": tool_correctness,
        "name_score": name_score,
        "param_content_score": param_score,
        "format_compliance": format_score,
        "tool_call_format": tool_call_score,
    }
