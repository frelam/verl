"""Task-environment protocol and shared observation / termination helpers.

Authoritative spec: ``agentic_rl_data_plan.md`` §4.1 (protocol, no state leak,
``max_turns`` => timeout) and §2 (Task / Trajectory schema).  This module is
stdlib-only so every environment can be imported without torch / verl.

Contract
--------

Environment protocol (implemented by :class:`BaseEnv`, consumed by the rollout
adapter and by ``scripts/replay_check.py``)::

    env = build_env(task)                  # one instance per task
    obs = env.reset(task, max_turns=None)  # rebuilds every internal field
    obs, done, info = env.step(action)     # one assistant turn per call
    env.close()

``action`` is one full assistant turn::

    {"content": str,
     "tool_calls": [{"id": str | None, "name": str, "arguments": dict}, ...]}

A bare ``str`` action is accepted and means "content only, no tool call".
``reset`` accepts an optional ``max_turns`` override for the turn curriculum
(plan §7, 5 -> 10 -> 15); the effective cap is the **minimum** of the task's
``meta.max_turns``, the curriculum override, any ``env.config.max_turns`` and
an environment-specific cap (e.g. Search-R1's 8 search rounds).

Unified observation (``obs``) — aligned with ``schema.Trajectory``'s
``tool_results`` entry shape::

    {"step": int,                  # assistant turns consumed so far
     "tool_results": [{"id": str | None, "name": str,
                       "content": str, "ok": bool}, ...],
     "content": str,              # human-readable rendering for the policy
     "env_state": dict,           # snapshot of the state contract below
     "info": dict}                # always empty today (reserved)

Terminal ``info`` (every env, every termination path)::

    {"timeout": bool,             # True iff max_turns was exhausted
     "final_answer": str | None,
     "env_state": dict}           # only present when done=True

``trajectory["env_state"]`` is built by :func:`make_env_state` and has a
**stable shape for every env** (unused fields are ``None`` / empty):

    type, impl, goal_satisfied, replay_ok, failed_turn, matched_turns,
    gt_turns, checks, state, rendered, num_turns, timeout, extras

Envs fill it through ``_env_state_fields()``.  Reward functions branch on
``type`` + ``impl`` only; see ``reports/env_setup.md`` for the field-level
table.

State isolation: no module-level mutable state exists; every mutable object
lives on the env instance and ``reset()`` reconstructs all of it.  Distinct
task instances can therefore be stepped concurrently from different threads.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Mapping, Protocol, runtime_checkable

try:  # pragma: no cover - normal package import
    from ..schema import DEFAULT_MAX_TURNS
except ImportError:  # pragma: no cover - allows a loose single-file import
    DEFAULT_MAX_TURNS = 15

#: Environment kinds understood by :func:`build_env`.
ENV_KINDS = frozenset({"none", "mock_fc", "search", "python", "code_tests"})

#: Default rendering cap for tool-result content inside one observation.
DEFAULT_MAX_RESULT_CHARS = 4096

#: Keys that always exist in ``trajectory["env_state"]``.
ENV_STATE_KEYS = (
    "type",
    "impl",
    "goal_satisfied",
    "replay_ok",
    "failed_turn",
    "matched_turns",
    "gt_turns",
    "checks",
    "state",
    "rendered",
    "num_turns",
    "timeout",
    "code_tests",
    "extras",
)


class EnvError(RuntimeError):
    """Raised for protocol misuse and unrecoverable environment failures.

    Recoverable failures (a tool call that cannot be executed, a search backend
    timeout, failing unit tests) are returned as ``tool_results[i].ok = False``
    instead — a rollout must never crash the trainer.
    """


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _positive_int(value: Any, default: int | None = None) -> int | None:
    """Coerce a config value to a positive ``int`` (``None`` when invalid)."""
    if value is None or isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    if number <= 0:
        return default
    return number


def make_tool_result(
    name: str,
    content: Any = "",
    *,
    ok: bool = True,
    call_id: str | None = None,
) -> dict[str, Any]:
    """Build one ``tool_results`` entry (schema.Trajectory aligned)."""
    return {
        "id": call_id,
        "name": str(name),
        "content": content if isinstance(content, str) else str(content),
        "ok": bool(ok),
    }


def truncate_text(text: Any, max_chars: int = DEFAULT_MAX_RESULT_CHARS) -> tuple[str, bool]:
    """Truncate ``text`` to ``max_chars`` characters (head kept).

    Returns ``(text, truncated)``.  ``max_chars <= 0`` disables truncation.
    """
    if max_chars <= 0:
        return (text if isinstance(text, str) else str(text)), False
    text = text if isinstance(text, str) else str(text)
    if len(text) <= max_chars:
        return text, False
    if max_chars < 1:
        return "", True
    # Marker must never make the result longer than ``max_chars`` (T4 unit test).
    marker = f"...[{len(text) - max_chars}ch]"
    if len(marker) > max_chars:
        marker = marker[:max_chars]
    budget = max(0, max_chars - len(marker))
    return text[:budget] + marker, True


def normalise_checks(checks: Any) -> list[dict[str, Any]]:
    """Canonicalise goal-check records to ``{"name", "ok", "detail"}``.

    C3 fixes the shape of ``env_state["checks"]``; environments build it from
    arbitrary config, so the base layer normalises (and drops non-mapping
    entries) instead of trusting subclasses.  ``detail`` is truncated for
    diagnosability.
    """
    normalised: list[dict[str, Any]] = []
    for index, item in enumerate(checks or []):
        if not _is_mapping(item):
            continue
        name = item.get("name")
        detail = item.get("detail")
        normalised.append(
            {
                "name": str(name) if name is not None else f"check[{index}]",
                "ok": bool(item.get("ok", False)),
                "detail": truncate_text(
                    "" if detail is None else str(detail), 500
                )[0],
            }
        )
    return normalised


def normalize_tool_call(
    call: Any,
    *,
    index: int = 0,
    step: int = 0,
) -> dict[str, Any]:
    """Normalise one tool call to ``{"id", "name", "arguments"}``.

    Accepts the flat OpenAI-chat shape as well as the nested
    ``{"function": {"name": ..., "arguments": ...}}`` shape; ``arguments`` may
    be a mapping or a JSON string.  Raises :class:`EnvError` on malformed input.
    """
    if not _is_mapping(call):
        raise EnvError(f"tool call #{index} must be a mapping, got {type(call).__name__}")

    name = call.get("name")
    arguments: Any = call.get("arguments")
    function = call.get("function")
    if _is_mapping(function):
        if not isinstance(name, str) or not name.strip():
            name = function.get("name")
        if arguments is None:
            arguments = function.get("arguments")

    if not isinstance(name, str) or not name.strip():
        raise EnvError(f"tool call #{index} must carry a non-empty string name")

    if arguments is None:
        parsed_arguments: dict[str, Any] = {}
    elif _is_mapping(arguments):
        parsed_arguments = dict(arguments)
    elif isinstance(arguments, str):
        text = arguments.strip()
        if not text:
            parsed_arguments = {}
        else:
            try:
                loaded = json.loads(text)
            except json.JSONDecodeError as exc:
                raise EnvError(
                    f"tool call #{index} ({name!r}) has non-JSON string arguments: {exc}"
                ) from exc
            if not _is_mapping(loaded):
                raise EnvError(
                    f"tool call #{index} ({name!r}) arguments must decode to a mapping"
                )
            parsed_arguments = dict(loaded)
    else:
        raise EnvError(
            f"tool call #{index} ({name!r}) arguments must be a mapping or JSON string"
        )

    call_id = call.get("id")
    if not isinstance(call_id, str) or not call_id.strip():
        call_id = f"call-{step}-{index}"
    return {"id": call_id, "name": name.strip(), "arguments": parsed_arguments}


def normalize_action(action: Any, *, step: int = 0) -> dict[str, Any]:
    """Normalise one assistant turn to ``{"content", "tool_calls"}``.

    A bare string is treated as content with no tool calls.  Missing
    ``content`` becomes ``""``.  ``tool_calls`` may be ``None``.
    """
    if isinstance(action, str):
        return {"content": action, "tool_calls": []}
    if not _is_mapping(action):
        raise EnvError(
            f"action must be a mapping or str, got {type(action).__name__}"
        )

    content = action.get("content", "")
    if content is None:
        content = ""
    if not isinstance(content, str):
        content = str(content)

    raw_calls = action.get("tool_calls")
    if raw_calls is None:
        raw_calls = action.get("calls")
    if raw_calls is None:
        raw_calls = []
    if not isinstance(raw_calls, (list, tuple)):
        raise EnvError("action.tool_calls must be a list")

    calls = [
        normalize_tool_call(call, index=index, step=step)
        for index, call in enumerate(raw_calls)
    ]
    return {"content": content, "tool_calls": calls}


def make_env_state(
    env_type: str = "none",
    impl: str = "local",
    *,
    goal_satisfied: bool | None = None,
    replay_ok: bool | None = None,
    failed_turn: int | None = None,
    matched_turns: int | None = None,
    gt_turns: int | None = None,
    checks: list[dict[str, Any]] | None = None,
    state: dict[str, Any] | None = None,
    rendered: str = "",
    num_turns: int = 0,
    timeout: bool = False,
    code_tests: dict[str, Any] | None = None,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the stable ``trajectory["env_state"]`` payload (see module doc)."""
    return {
        "type": str(env_type),
        "impl": str(impl),
        "goal_satisfied": goal_satisfied,
        "replay_ok": replay_ok,
        "failed_turn": failed_turn,
        "matched_turns": matched_turns,
        "gt_turns": gt_turns,
        "checks": normalise_checks(checks),
        "state": state,
        "rendered": rendered,
        "num_turns": int(num_turns),
        "timeout": bool(timeout),
        "code_tests": code_tests,
        "extras": dict(extras or {}),
    }


def validate_env_state(state: Any) -> list[str]:
    """Return contract violations for a ``trajectory["env_state"]`` payload.

    Enforces C3 shapes: stable keys present, terminal verdicts are
    ``bool | None``, ``checks`` is a list of ``{name, ok, detail}`` records,
    ``code_tests`` is ``dict | None``, and ``num_turns`` / ``timeout`` /
    ``extras`` have the right types.  Used by tests and the replay checker.
    """
    errors: list[str] = []
    if not _is_mapping(state):
        return [f"env_state must be a mapping, got {type(state).__name__}"]

    for key in ENV_STATE_KEYS:
        if key not in state:
            errors.append(f"env_state missing stable key {key!r}")
    if not isinstance(state.get("type"), str) or not state.get("type"):
        errors.append("env_state.type must be a non-empty string")
    if not isinstance(state.get("impl"), str) or not state.get("impl"):
        errors.append("env_state.impl must be a non-empty string")
    for key in ("goal_satisfied", "replay_ok"):
        value = state.get(key)
        if value is not None and not isinstance(value, bool):
            errors.append(f"env_state.{key} must be a bool or null")
    for key in ("failed_turn", "matched_turns", "gt_turns"):
        value = state.get(key)
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"env_state.{key} must be a non-negative int or null")
    num_turns = state.get("num_turns")
    if not isinstance(num_turns, int) or isinstance(num_turns, bool) or num_turns < 0:
        errors.append("env_state.num_turns must be a non-negative int")
    if not isinstance(state.get("timeout"), bool):
        errors.append("env_state.timeout must be a bool")
    checks = state.get("checks")
    if not isinstance(checks, list):
        errors.append("env_state.checks must be a list")
    else:
        for index, item in enumerate(checks):
            if not _is_mapping(item):
                errors.append(f"env_state.checks[{index}] must be a mapping")
                continue
            for field in ("name", "ok", "detail"):
                if field not in item:
                    errors.append(f"env_state.checks[{index}] missing {field!r}")
            if not isinstance(item.get("name"), str):
                errors.append(f"env_state.checks[{index}].name must be a string")
            if not isinstance(item.get("ok"), bool):
                errors.append(f"env_state.checks[{index}].ok must be a bool")
            if not isinstance(item.get("detail"), str):
                errors.append(f"env_state.checks[{index}].detail must be a string")
    if state.get("state") is not None and not _is_mapping(state.get("state")):
        errors.append("env_state.state must be a mapping or null")
    if not isinstance(state.get("rendered"), str):
        errors.append("env_state.rendered must be a string")
    if state.get("code_tests") is not None and not _is_mapping(state.get("code_tests")):
        errors.append("env_state.code_tests must be a mapping or null")
    if not _is_mapping(state.get("extras")):
        errors.append("env_state.extras must be a mapping")
    return errors


# ---------------------------------------------------------------------------
# protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class EnvProtocol(Protocol):
    """Structural protocol implemented by every environment (plan §4.1)."""

    env_type: str
    impl: str

    def reset(
        self,
        task: Mapping[str, Any] | None = None,
        *,
        max_turns: int | None = None,
    ) -> dict[str, Any]:
        """Rebuild internal state from ``task`` and return the initial obs."""

    def step(self, action: Any) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        """Consume one assistant turn; return ``(obs, done, info)``."""

    def close(self) -> None:
        """Release resources; must be idempotent."""


# ---------------------------------------------------------------------------
# base implementation
# ---------------------------------------------------------------------------


class BaseEnv:
    """Default environment implementation with shared turn/timeout bookkeeping.

    Subclasses override the small hooks:

    * ``_setup(task, config)`` — build domain state (called by ``reset``).
    * ``_handle_action(action)`` -> ``(tool_results, done, info, final_answer)``
      — default runs ``_handle_tool_calls`` and treats a no-call turn as a
      final answer.
    * ``_handle_tool_calls(calls)`` -> ``(tool_results, info)`` — execute calls.
    * ``_env_state_fields()`` — extra ``env_state`` values.
    * ``_turn_cap(task, config)`` — environment-specific turn cap.
    * ``_teardown()`` — release temp dirs / clients in ``close()``.
    """

    env_type: str = "none"
    impl: str = "local"
    #: Maximum characters per tool-result content (0 disables truncation).
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS

    def __init__(
        self,
        task: Mapping[str, Any] | None = None,
        *,
        max_turns: int | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self._task: Mapping[str, Any] | None = None
        self.config: dict[str, Any] = {}
        self._max_turns: int = DEFAULT_MAX_TURNS
        self._steps: int = 0
        self._turns: list[dict[str, Any]] = []
        self._done: bool = False
        self._timeout: bool = False
        self._closed: bool = False
        self._final_answer: str | None = None
        self._env_state: dict[str, Any] | None = None
        self._last_obs: dict[str, Any] | None = None
        if task is not None:
            self.reset(task, max_turns=max_turns, config=config)

    # -- lifecycle ------------------------------------------------------

    def reset(
        self,
        task: Mapping[str, Any] | None = None,
        *,
        max_turns: int | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Reset all instance state from ``task`` and return the initial obs.

        ``task=None`` replays the previously stored task (CONTRACTS.md C1).
        ``config`` (when given) is merged over ``task["env"]["config"]``; it is
        how tests inject fake backends and how the curriculum raises/lowers
        ``max_turns``.  A closed env may be reset again: ``reset`` reopens it
        after ``close()`` released external resources (T4-06).
        """
        if task is None:
            task = self._task
        if not _is_mapping(task):
            raise EnvError("reset(task) requires a Task mapping")
        self._task = task
        self.config = self._build_config(task, config)
        self._max_turns = self._resolve_max_turns(task, max_turns, self.config)

        self._steps = 0
        self._turns = []
        self._done = False
        self._timeout = False
        self._closed = False
        self._final_answer = None
        self._last_obs = None
        self._env_state = None

        self._setup(task, self.config)
        self._env_state = self._build_env_state()
        obs = self._make_obs(
            step=0,
            tool_results=[],
            content=self._initial_content(),
            env_state=self._env_state,
        )
        self._last_obs = obs
        return obs

    def step(self, action: Any) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        """Consume one assistant turn and return ``(obs, done, info)``."""
        if self._task is None:
            raise EnvError("step() called before reset(task)")
        if self._closed:
            raise EnvError("step() called on a closed env")
        if self._done:
            raise EnvError("episode is finished; call reset(task) before step()")

        action = normalize_action(action, step=self._steps + 1)
        self._steps += 1

        tool_results, turn_done, info, final_answer = self._handle_action(action)
        if final_answer is not None:
            self._final_answer = final_answer

        # plan §4.1: exhausting max_turns ends the episode as a timeout; no
        # extra penalty is applied here (reward decides, see §5 "不额外惩罚").
        if not turn_done and self._steps >= self._max_turns:
            turn_done = True
            self._timeout = True

        turn = {
            "step": self._steps,
            "assistant": {
                "content": action["content"],
                "tool_calls": [dict(call) for call in action["tool_calls"]],
            },
            "tool_results": [dict(result) for result in tool_results],
        }
        self._turns.append(turn)

        if turn_done:
            self._done = True
        self._env_state = self._build_env_state()
        if turn_done:
            info["timeout"] = bool(self._timeout)
            info["final_answer"] = self._final_answer
            info["env_state"] = self._env_state
        else:
            info.setdefault("timeout", False)
            info.setdefault("final_answer", None)
        info["step"] = self._steps

        obs = self._make_obs(
            step=self._steps,
            tool_results=tool_results,
            content=self._turn_content(tool_results, info),
            env_state=self._env_state,
        )
        self._last_obs = obs
        return obs, bool(self._done), info

    def close(self) -> None:
        """Release resources (idempotent), keeping the recorded trajectory.

        ``step()`` on a closed env raises :class:`EnvError`; a subsequent
        ``reset(task)`` reopens the instance (T4-06).
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._teardown()
        finally:
            pass

    def __enter__(self) -> "BaseEnv":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- read-only views -------------------------------------------------

    @property
    def task(self) -> Mapping[str, Any] | None:
        return self._task

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def done(self) -> bool:
        return self._done

    @property
    def timeout(self) -> bool:
        return self._timeout

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def final_answer(self) -> str | None:
        return self._final_answer

    @property
    def max_turns(self) -> int:
        return self._max_turns

    @property
    def env_state(self) -> dict[str, Any] | None:
        return self._env_state

    @property
    def last_obs(self) -> dict[str, Any] | None:
        return self._last_obs

    @property
    def replay_ok(self) -> bool | None:
        """Replay verdict; ``None`` for envs that do not replay."""
        return None

    @property
    def goal_satisfied(self) -> bool | None:
        """Terminal-state verdict; ``None`` for envs without a goal spec."""
        state = self._env_state
        if state is None:
            return None
        return state.get("goal_satisfied")

    def trajectory(self) -> dict[str, Any]:
        """Return the normalised ``schema.Trajectory`` for the current episode.

        This is the exact payload reward functions receive; the training
        framework's rollout adapter should build the same structure.
        """
        turns = copy.deepcopy(self._turns)
        state = self._env_state
        if state is None:
            state = self._build_env_state()
        return {
            "turns": turns,
            "final_answer": self._final_answer,
            "num_turns": len(turns),
            "done": bool(self._done),
            "env_state": copy.deepcopy(state),
        }

    # -- hooks -----------------------------------------------------------

    def _setup(self, task: Mapping[str, Any], config: dict[str, Any]) -> None:
        """Initialise domain state; called after counters are reset."""

    def _handle_action(
        self,
        action: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], bool, dict[str, Any], str | None]:
        """Default turn semantics: calls are executed, no-call = final answer."""
        if not action["tool_calls"]:
            return [], True, {}, action["content"]
        results, info = self._handle_tool_calls(action["tool_calls"])
        return results, False, info, None

    def _handle_tool_calls(
        self,
        calls: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Execute tool calls; base implementation returns "unavailable"."""
        message = f"tool execution is not available for env.type={self.env_type!r}"
        results = [
            make_tool_result(call["name"], message, ok=False, call_id=call["id"])
            for call in calls
        ]
        return results, {"tool_execution": "unavailable"}

    def _env_state_fields(self) -> dict[str, Any]:
        """Extra ``make_env_state`` kwargs for this environment."""
        return {
            "num_turns": len(self._turns),
            "timeout": bool(self._timeout),
        }

    def _turn_cap(
        self,
        task: Mapping[str, Any],
        config: dict[str, Any],
    ) -> int | None:
        """Environment-specific hard turn cap (min'd with everything else)."""
        return None

    def _teardown(self) -> None:
        """Release external resources (temp dirs, sessions)."""

    # -- internals -------------------------------------------------------

    @staticmethod
    def _build_config(
        task: Mapping[str, Any],
        override: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        env = task.get("env") if _is_mapping(task.get("env")) else {}
        raw_config = env.get("config")
        config: dict[str, Any] = copy.deepcopy(dict(raw_config)) if _is_mapping(raw_config) else {}
        if override:
            config.update(dict(override))
        return config

    def _resolve_max_turns(
        self,
        task: Mapping[str, Any],
        override: int | None,
        config: Mapping[str, Any],
    ) -> int:
        candidates: list[int] = []
        meta = task.get("meta") if _is_mapping(task.get("meta")) else {}
        for raw in (meta.get("max_turns"), override, config.get("max_turns")):
            value = _positive_int(raw)
            if value is not None:
                candidates.append(value)
        cap = self._turn_cap(task, config if _is_mapping(config) else {})
        cap = _positive_int(cap)
        if cap is not None:
            candidates.append(cap)
        if not candidates:
            return DEFAULT_MAX_TURNS
        return min(candidates)

    def _build_env_state(self) -> dict[str, Any]:
        fields = dict(self._env_state_fields() or {})
        fields.setdefault("num_turns", len(self._turns))
        fields.setdefault("timeout", bool(self._timeout))
        # A timeout is a failed episode (plan §4.1: "超限视为失败").  No
        # terminal verdict may survive it, otherwise a reward that trusts
        # goal_satisfied / replay_ok would award a full score to a truncated
        # trajectory.  Diagnostics (checks / state) are kept untouched.
        if fields.get("timeout") and self.env_type == "mock_fc":
            fields["goal_satisfied"] = False
            fields["replay_ok"] = False
        return make_env_state(self.env_type, self.impl, **fields)

    def _make_obs(
        self,
        *,
        step: int,
        tool_results: list[dict[str, Any]],
        content: str,
        env_state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "step": step,
            "tool_results": tool_results,
            "content": content,
            "env_state": env_state,
            "info": {},
        }

    def _initial_content(self) -> str:
        """Policy-visible text returned by ``reset`` (usually empty)."""
        return ""

    def _turn_content(
        self,
        tool_results: list[dict[str, Any]],
        info: dict[str, Any],
    ) -> str:
        """Render this turn's tool results for the policy."""
        if not tool_results:
            return ""
        if len(tool_results) == 1:
            return truncate_text(tool_results[0]["content"], self.max_result_chars)[0]
        return "\n".join(
            f"[{result['name']}] "
            f"{truncate_text(result['content'], self.max_result_chars)[0]}"
            for result in tool_results
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<{type(self).__name__} env_type={self.env_type!r} "
            f"impl={self.impl!r} steps={self._steps} done={self._done}>"
        )


class NullEnv(BaseEnv):
    """Environment for ``env.type == "none"`` tasks (single-turn / no tools).

    ``st_fc`` / ``irrelevance`` / ``abstain`` tasks carry no environment: the
    reward compares the emitted calls/text directly.  The env therefore ends
    the episode after one assistant turn and reports tool calls as
    "not executed" (``ok=False``) instead of inventing results.
    """

    env_type = "none"
    impl = "local"

    def _handle_action(
        self,
        action: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], bool, dict[str, Any], str | None]:
        if action["tool_calls"]:
            results = [
                make_tool_result(
                    call["name"],
                    "tool execution is unavailable when env.type == 'none'",
                    ok=False,
                    call_id=call["id"],
                )
                for call in action["tool_calls"]
            ]
            return results, True, {"tool_execution": "unavailable"}, None
        return [], True, {}, action["content"]

    def _env_state_fields(self) -> dict[str, Any]:
        return {
            "num_turns": len(self._turns),
            "timeout": bool(self._timeout),
            "extras": {"tool_execution": "unavailable"},
        }


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------


def build_env(
    task: Mapping[str, Any],
    *,
    max_turns: int | None = None,
    impl: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> BaseEnv:
    """Instantiate the environment declared by ``task["env"]``.

    ``impl`` forces an implementation for dual-implementation envs
    (``mock_fc``: ``"stateful"`` / ``"replay"``); ``config`` is merged over
    ``task["env"]["config"]``.  Imports of heavy modules are lazy so building a
    ``none`` env never pulls in fastapi / faiss.
    """
    if not _is_mapping(task):
        raise EnvError("build_env() requires a Task mapping")
    env = task.get("env") if _is_mapping(task.get("env")) else {}
    env_type = env.get("type", "none")
    if env_type not in ENV_KINDS:
        raise EnvError(f"unknown env.type {env_type!r}; expected one of {sorted(ENV_KINDS)}")

    merged = dict(config or {})

    if env_type == "mock_fc":
        from .mock_fc import MockFCEnv, ReplayMockFCEnv

        impl_name = (impl or merged.get("impl") or _mock_impl_from_task(task)).lower()
        if impl_name == "stateful":
            return MockFCEnv(task, max_turns=max_turns, config=merged or None)
        if impl_name == "replay":
            return ReplayMockFCEnv(task, max_turns=max_turns, config=merged or None)
        raise EnvError(
            f"mock_fc impl must be 'stateful' or 'replay', got {impl_name!r}"
        )

    if env_type == "search":
        from .search import SearchEnv

        return SearchEnv(task, max_turns=max_turns, config=merged or None)

    if env_type == "python":
        from .python_env import PythonEnv

        return PythonEnv(task, max_turns=max_turns, config=merged or None)

    if env_type == "code_tests":
        from .code_tests import CodeTestsEnv

        return CodeTestsEnv(task, max_turns=max_turns, config=merged or None)

    return NullEnv(task, max_turns=max_turns, config=merged or None)


def _mock_impl_from_task(task: Mapping[str, Any]) -> str:
    """Infer the mock_fc implementation from the task config (``auto``)."""
    env = task.get("env") if _is_mapping(task.get("env")) else {}
    config = env.get("config") if _is_mapping(env.get("config")) else {}
    impl = config.get("impl")
    if isinstance(impl, str) and impl.lower() in ("stateful", "replay"):
        return impl.lower()
    if config.get("domain") or config.get("domain_obj") or config.get("goal"):
        return "stateful"
    return "replay"


__all__ = [
    "BaseEnv",
    "DEFAULT_MAX_RESULT_CHARS",
    "ENV_KINDS",
    "ENV_STATE_KEYS",
    "EnvError",
    "EnvProtocol",
    "NullEnv",
    "build_env",
    "make_env_state",
    "make_tool_result",
    "normalize_action",
    "normalize_tool_call",
    "normalise_checks",
    "truncate_text",
    "validate_env_state",
]
