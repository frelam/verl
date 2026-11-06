"""verl custom reward function for the *single-turn-capable* agentic task types.

verl wiring (launch script sets these)::

    reward.custom_reward_function.path=agentic_rl_data/scripts/verl_reward_bridge.py
    reward.custom_reward_function.name=compute_score
    reward.reward_manager.name=naive

The naive reward manager accepts a dict return value; it takes ``result["score"]``
and copies every key into ``reward_extra_info`` (so the per-dimension breakdown
shows up in training metrics).

Scope
-----
This bridge scores task types that need **no environment state**:

    st_fc, irrelevance, abstain, search_qa, math_ci

``mt_fc`` and ``code_exec`` need the env terminal state (``env_state``) and are
therefore scored **inside the multi-turn agent loop**
(``agentic_rl_data.verl.agent_loop_agentic``), which sets ``reward_score``
directly.  If such a sample reaches this bridge (misconfiguration), it returns
0.0 with ``info["unsupported_single_turn"]=True`` instead of silently scoring
garbage.

Inputs available to verl's custom reward:
    data_source  : row ``data_source`` ("agentic_rl")
    solution_str : decoded model response
    ground_truth : ``reward_model.ground_truth`` (readable label, logging only)
    extra_info   : the parquet ``extra_info`` struct (agentic_task JSON blob)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:  # agentic_rl_data imports + examples.tool_rl parser
    sys.path.insert(0, str(_ROOT))

#: Task types that this bridge can score without an environment state.
SINGLE_TURN_TASK_TYPES = frozenset({"st_fc", "irrelevance", "abstain", "search_qa", "math_ci"})

#: Task types that require the multi-turn agent loop (env_state / code tests).
ENV_REQUIRED_TASK_TYPES = frozenset({"mt_fc", "code_exec"})


def _as_mapping(value: Any) -> dict[str, Any]:
    """Normalise numpy 0-d/array/dict-like values from ``non_tensor_batch``."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):  # pragma: no cover - exotic dtypes
            return {}
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _parse_calls(solution_str: str) -> list[dict[str, Any]]:
    """Parse ``<tool_call>`` blocks with the same parser as the tool_rl reward."""
    try:
        from examples.tool_rl.reward.verifier import parse_qwen_tool_calls
    except ImportError:  # pragma: no cover - the repo root is expected on sys.path
        return []
    try:
        calls = parse_qwen_tool_calls(solution_str or "")
    except Exception:  # noqa: BLE001 - a malformed response must never crash rollout
        return []
    return [call for call in calls if isinstance(call, dict)]


def build_trajectory(solution_str: str, calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Single-turn trajectory as frozen in CONTRACTS C3 (no env_state yet)."""
    return {
        "turns": [
            {
                "step": 1,
                "assistant": {"content": solution_str or "", "tool_calls": calls},
                "tool_results": [],
            }
        ],
        "final_answer": solution_str or None,
        "num_turns": 1,
        "done": True,
        "env_state": None,
    }


def _task_from_extra(extra_info: Mapping[str, Any], data_source: str) -> dict[str, Any]:
    """Rebuild the canonical task from ``extra_info["agentic_task"]``."""
    raw = extra_info.get("agentic_task")
    if isinstance(raw, str) and raw.strip():
        try:
            task = json.loads(raw)
            if isinstance(task, Mapping):
                return dict(task)
        except json.JSONDecodeError:
            pass
    # Degraded task (should not happen with exported parquet): keep the bridge
    # functional so one bad row cannot crash a training step.
    return {
        "id": str(extra_info.get("task_id") or "unknown"),
        "source": str(extra_info.get("source") or "unknown"),
        "task_type": str(extra_info.get("task_type") or "st_fc"),
        "messages": [],
        "tools": [],
        "env": {"type": "none", "config": {}},
        "reference": {
            "final_answer": None,
            "final_answers": None,
            "gt_calls": [],
            "gt_turns": None,
            "tool_returns": None,
            "unit_tests": None,
            "final_state": None,
        },
        "meta": {
            "max_turns": int(extra_info.get("max_turns") or 15),
            "difficulty": None,
            "pass_rate": None,
            "license": None,
        },
        "_degraded_bridge_task": True,
        "_data_source": data_source,
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any = None,
    extra_info: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """verl custom reward entry point (single-turn task types only)."""
    info: dict[str, Any] = {"bridge": "agentic_rl_single_turn"}
    try:
        extra = _as_mapping(extra_info)
        task = _task_from_extra(extra, str(data_source))
        task_type = str(task.get("task_type") or extra.get("task_type") or "st_fc")
        info["task_type"] = task_type
        info["task_id"] = str(task.get("id") or extra.get("task_id") or "")
        if task_type not in SINGLE_TURN_TASK_TYPES:
            info["unsupported_single_turn"] = True
            info["unsupported_reason"] = (
                f"task_type {task_type!r} requires env_state; score it in the "
                "agentic_agent loop (agentic_rl_data.verl.agent_loop_agentic)"
            )
            return {"score": 0.0, **info}
        calls = _parse_calls(str(solution_str or ""))
        info["num_calls"] = len(calls)
        trajectory = build_trajectory(str(solution_str or ""), calls)
        from agentic_rl_data.rewards import dispatch_reward

        result = dispatch_reward(task, trajectory)
        score = float(result.get("score", 0.0))
        info.update({str(key): value for key, value in (result.get("info") or {}).items()})
        return {"score": max(0.0, min(1.0, score)), **info}
    except Exception as exc:  # noqa: BLE001 - never crash the reward worker
        info["bridge_error"] = f"{type(exc).__name__}: {exc}"
        return {"score": 0.0, **info}


__all__ = [
    "ENV_REQUIRED_TASK_TYPES",
    "SINGLE_TURN_TASK_TYPES",
    "build_trajectory",
    "compute_score",
]
