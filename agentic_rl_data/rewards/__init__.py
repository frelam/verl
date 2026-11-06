"""Registry of the per-``task_type`` reward functions (plan §5).

Every reward is a pure function with the frozen signature::

    reward(task: dict, trajectory: dict) -> {"score": float, "info": dict}

Use :func:`get_reward` to look up one family by ``task["task_type"]`` and
:func:`dispatch_reward` for the full dispatch.  Unknown task types raise
``ValueError`` with the known set instead of silently scoring 0 (a mistyped
task type must fail loudly - it is a data/schema bug, not a policy failure).

:func:`get_reward` also accepts the ``(task, trajectory)`` call shape used by
``scripts/replay_check.py``: with one argument it returns the reward function,
with two arguments it dispatches and returns the reward payload.  The registry
is a plain immutable mapping: no mutable module-level state (plan §5 "每个 reward
函数是纯函数"; CONTRACTS C4 score range ``[0, 1]``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from . import base
from .abstain import TASK_TYPE as ABSTAIN_TASK_TYPE
from .abstain import reward as abstain_reward
from .code_exec import TASK_TYPE as CODE_EXEC_TASK_TYPE
from .code_exec import reward as code_exec_reward
from .irrelevance import TASK_TYPE as IRRELEVANCE_TASK_TYPE
from .irrelevance import reward as irrelevance_reward
from .math_ci import TASK_TYPE as MATH_CI_TASK_TYPE
from .math_ci import reward as math_ci_reward
from .mt_fc import TASK_TYPE as MT_FC_TASK_TYPE
from .mt_fc import reward as mt_fc_reward
from .search_qa import TASK_TYPE as SEARCH_QA_TASK_TYPE
from .search_qa import reward as search_qa_reward
from .st_fc import TASK_TYPE as ST_FC_TASK_TYPE
from .st_fc import reward as st_fc_reward

RewardFn = Callable[[dict, dict], dict]

REWARDS: dict[str, RewardFn] = {
    ST_FC_TASK_TYPE: st_fc_reward,
    IRRELEVANCE_TASK_TYPE: irrelevance_reward,
    MT_FC_TASK_TYPE: mt_fc_reward,
    SEARCH_QA_TASK_TYPE: search_qa_reward,
    MATH_CI_TASK_TYPE: math_ci_reward,
    CODE_EXEC_TASK_TYPE: code_exec_reward,
    ABSTAIN_TASK_TYPE: abstain_reward,
}

TASK_TYPES = frozenset(REWARDS)

#: Convenience re-export; the canonical constant lives in ``schema.SCORE_RANGE``.
MAX_SCORE = base.MAX_SCORE


def _lookup(task_type: Any) -> RewardFn:
    """Resolve a reward function from a task type (strict, explicit errors)."""
    if not isinstance(task_type, str):
        raise TypeError(f"task_type must be a string, got {type(task_type).__name__}")
    key = task_type.strip()
    try:
        return REWARDS[key]
    except KeyError:
        raise ValueError(
            f"unknown task_type {task_type!r}; known task types: {sorted(REWARDS)}"
        ) from None


def get_reward(task_or_type: Any, trajectory: Any = None) -> Any:
    """Return the reward function for a task type (or dispatch one task).

    * ``get_reward("st_fc")`` -> the registered reward callable;
    * ``get_reward(task, trajectory)`` -> the reward payload (replay_check
      compatibility: ``scripts/replay_check.py`` probes ``get_reward`` and calls
      it with ``(task, trajectory)``).

    Raises:
        TypeError: when the task type is not a string, or when a task mapping
            is passed without a trajectory.
        ValueError: when the task type is unknown (lists the known types).
    """
    if isinstance(task_or_type, Mapping):
        if trajectory is None:
            raise TypeError(
                "get_reward(task, trajectory) requires both the task and the "
                "trajectory; use get_reward(task_type) for a registry lookup"
            )
        return dispatch_reward(task_or_type, trajectory)
    return _lookup(task_or_type)


def dispatch_reward(task: Any, trajectory: Any) -> dict:
    """Dispatch one task to its registered reward via ``task["task_type"]``."""
    if not isinstance(task, Mapping):
        raise TypeError(f"task must be a mapping, got {type(task).__name__}")
    return _lookup(task.get("task_type"))(task, trajectory)


#: Aliases for consumers that probe one of these names (replay_check, verl
#: custom-reward adapters).
reward_task = dispatch_reward
compute_reward = dispatch_reward

__all__ = [
    "REWARDS",
    "TASK_TYPES",
    "RewardFn",
    "get_reward",
    "dispatch_reward",
    "reward_task",
    "compute_reward",
    "MAX_SCORE",
    "base",
    "st_fc_reward",
    "irrelevance_reward",
    "mt_fc_reward",
    "search_qa_reward",
    "math_ci_reward",
    "code_exec_reward",
    "abstain_reward",
]
