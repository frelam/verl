"""Pure (dependency-free) helpers for the verl agent-loop adapter.

``agentic_rl_data/verl/agent_loop_agentic.py`` is a verl coroutine; this module
holds the parts that do **not** need verl, torch or numpy so they can be
unit-tested in a plain Python environment:

* tool-schema normalisation (the parquet ``tools`` column -> OpenAI tool list),
* ``extra_info["agentic_task"]`` parsing and turn-cap resolution,
* ``FunctionCall`` -> env action / trajectory tool-call conversion,
* ``obs["tool_results"]`` -> chat ``tool`` messages / trajectory results,
* trajectory assembly (CONTRACTS C3/C4 / ``schema.Trajectory``),
* fail-closed reward dispatch (a reward exception never crashes rollout),
* the fixed-schema ``reward_extra_info`` payload.

Why the fixed ``reward_extra_info`` schema
-------------------------------------------
``verl.experimental.agent_loop.agent_loop.AgentLoopWorker._postprocess`` takes
``reward_extra_infos[0]`` (the *first* sample of the worker chunk) as the key
template and then indexes every other sample with it::

    reward_extra_keys = list(reward_extra_infos[0].keys())
    non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

Mixed ``task_type`` batches return different reward ``info`` keys, so a raw
per-sample dict would raise ``KeyError`` on the second sample and abort the
rollout.  We therefore wrap the raw reward info into :data:`REWARD_INFO_KEYS`
(always present) and keep the full payload as a JSON string for debugging.
"""

from __future__ import annotations

import copy
import json
import math
from typing import Any, Callable, Iterable, Mapping

from ..schema import validate_reward_result

#: Every sample's ``extra_fields["reward_extra_info"]`` carries exactly these
#: keys (see module docstring for why this set must be constant).
REWARD_INFO_KEYS: tuple[str, ...] = (
    "agentic/task_type",
    "agentic/reward_score",
    "agentic/reward_error",
    "agentic/reward_timeout",
    "agentic/num_turns",
    "agentic/done",
    "agentic/reward_info",
)

#: Fallback turn cap when the task does not declare ``meta.max_turns``.
DEFAULT_MAX_TURNS = 15

#: Cap for the JSON dump of the raw reward info (avoid unbounded batch payloads).
MAX_REWARD_INFO_JSON_CHARS = 4000

#: Error marker used when the row carries no ``extra_info["agentic_task"]``.
MISSING_TASK_REWARD_ERROR = (
    "missing agentic_task: reward_score left to the configured verl reward manager"
)

#: Binary string types: some ``non_tensor_batch`` cells arrive as ``bytes`` /
#: ``bytearray`` (e.g. when a dataset column is decoded lazily); normalise them
#: to ``str`` before JSON parsing.
_NUMPY_STR_TYPES = (bytes, bytearray)


# ---------------------------------------------------------------------------
# small conversion helpers
# ---------------------------------------------------------------------------


def as_mapping(value: Any) -> dict[str, Any]:
    """Best-effort conversion of a ``non_tensor_batch`` cell to a plain dict."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            value = item()
        except (TypeError, ValueError):  # pragma: no cover - exotic dtypes
            return {}
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _as_tool_mapping(tool: Any) -> dict[str, Any] | None:
    """Convert one tool cell (dict / OpenAI dict / pydantic model) to a dict."""
    if isinstance(tool, Mapping):
        return dict(tool)
    model_dump = getattr(tool, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(exclude_unset=True, exclude_none=True)
        except TypeError:  # pragma: no cover - defensive
            dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)
    return None


def to_dict_list(value: Any) -> list[dict[str, Any]]:
    """Normalise a (possibly NumPy) list-of-dicts cell to ``list[dict]``.

    The parquet ``tools`` column arrives as a NumPy object array of Arrow
    structs; entries may also be pydantic tool schemas in other call sites.
    Non-mapping entries are dropped instead of crashing the rollout.
    """
    if value is None:
        return []
    if isinstance(value, str) or isinstance(value, _NUMPY_STR_TYPES):
        # Some legacy exporters store the whole tool list as a JSON string.
        decoded = value.decode("utf-8", errors="replace") if not isinstance(value, str) else value
        text = decoded.strip()
        if not text:
            return []
        try:
            value = json.loads(text)
        except (json.JSONDecodeError, TypeError, ValueError):
            return []
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if isinstance(value, Mapping):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    result: list[dict[str, Any]] = []
    for item in value:
        item = getattr(item, "tolist", lambda: item)()
        mapping = _as_tool_mapping(item)
        if mapping:
            result.append(mapping)
    return result


def _parse_json_mapping(value: Any) -> Mapping | None:
    """Parse a JSON string/bytes that must decode to an object (else ``None``)."""
    if isinstance(value, _NUMPY_STR_TYPES):
        try:
            value = value.decode("utf-8")  # type: ignore[union-attr]
        except UnicodeDecodeError:
            return None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, Mapping) else None


def normalize_parameters(value: Any) -> dict[str, Any]:
    """Return a JSON-schema ``object`` parameter block.

    The parquet exporter stores ``parameters`` as a JSON string (pyarrow cannot
    unify heterogeneous parameter structs); chat templates and tool parsers
    need a mapping with ``type`` / ``properties`` / ``required``.
    """
    parsed: Mapping | None = None
    if isinstance(value, _NUMPY_STR_TYPES):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        parsed = _parse_json_mapping(value)
        value = parsed if parsed is not None else {}
    if not isinstance(value, Mapping):
        value = {}
    parameters = dict(value)
    parameters.setdefault("type", "object")
    if not isinstance(parameters.get("properties"), Mapping):
        parameters["properties"] = {}
    if not isinstance(parameters.get("required"), list):
        parameters["required"] = []
    return parameters


def normalize_tool(tool: Any) -> dict[str, Any] | None:
    """One tool schema -> OpenAI ``{"type": "function", "function": {...}}``.

    Accepts either the exporter's flat struct ``{name, description,
    parameters}`` (``parameters`` may be a JSON string) or an already
    OpenAI-wrapped dict.  Returns ``None`` for entries without a usable name.
    """
    mapping = _as_tool_mapping(tool)
    if mapping is None:
        return None
    function = mapping.get("function")
    if isinstance(function, Mapping):
        name = function.get("name")
        description = function.get("description", "")
        parameters = function.get("parameters")
        strict = function.get("strict")
    else:
        name = mapping.get("name")
        description = mapping.get("description", "")
        parameters = mapping.get("parameters")
        strict = mapping.get("strict")
    if not isinstance(name, str) or not name.strip():
        return None
    payload: dict[str, Any] = {
        "name": name.strip(),
        "description": "" if description is None else str(description),
        "parameters": normalize_parameters(parameters),
    }
    if strict is not None:
        payload["strict"] = bool(strict)
    return {"type": "function", "function": payload}


def normalize_tools_for_template(value: Any) -> list[dict[str, Any]]:
    """Dataset ``tools`` cell -> OpenAI tool list for chat templates / parsers."""
    return [tool for tool in (normalize_tool(item) for item in to_dict_list(value)) if tool]


def tool_names(value: Any) -> list[str]:
    """Declared tool names (schema order preserved)."""
    return [tool["function"]["name"] for tool in normalize_tools_for_template(value)]


# ---------------------------------------------------------------------------
# task / dataset-row parsing
# ---------------------------------------------------------------------------


def parse_agentic_task(raw: Any) -> dict[str, Any] | None:
    """Rebuild the canonical unified task from ``extra_info["agentic_task"]``.

    The exporter stores the whole task as a JSON string (see
    ``agentic_rl_data.scripts.export_verl_parquet.task_to_row``); a mapping or
    a NumPy 0-d object is accepted too.  Invalid JSON silently degrades to
    ``None`` so the caller can fall back to a single-turn rollout instead of
    crashing the agent loop.
    """
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        return dict(raw)
    item = getattr(raw, "item", None)
    if callable(item) and not isinstance(raw, (str, bytes)):
        try:
            raw = item()
        except (TypeError, ValueError):  # pragma: no cover - exotic dtypes
            return None
        if isinstance(raw, Mapping):
            return dict(raw)
    parsed = _parse_json_mapping(raw)
    return dict(parsed) if parsed is not None else None


def task_task_type(task: Any) -> str | None:
    """``task["task_type"]`` as a non-empty string (``None`` when absent)."""
    if not isinstance(task, Mapping):
        return None
    value = task.get("task_type")
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def task_env_type(task: Any) -> str:
    """``task["env"]["type"]`` as a lowercase string (``"none"`` when absent)."""
    if not isinstance(task, Mapping):
        return "none"
    env = task.get("env")
    if not isinstance(env, Mapping):
        return "none"
    value = env.get("type")
    if not isinstance(value, str) or not value.strip():
        return "none"
    return value.strip().lower()


def task_has_env(task: Any) -> bool:
    """True when the task declares a real (non-``none``) environment."""
    return task_env_type(task) not in ("", "none")


def resolve_max_turns(task: Any, default: int = DEFAULT_MAX_TURNS) -> int:
    """Effective assistant-turn cap from ``task["meta"]["max_turns"]``."""
    if not isinstance(task, Mapping):
        return int(default)
    meta = task.get("meta")
    if not isinstance(meta, Mapping):
        return int(default)
    value = meta.get("max_turns")
    if isinstance(value, bool):
        return int(default)
    try:
        number = int(value)
    except (TypeError, ValueError):
        return int(default)
    return number if number > 0 else int(default)


# ---------------------------------------------------------------------------
# tool call conversion (FunctionCall -> env action / trajectory)
# ---------------------------------------------------------------------------


def parse_tool_call_arguments(raw: Any) -> tuple[dict[str, Any], str | None]:
    """Return ``(arguments dict, error)`` for a ``FunctionCall.arguments`` value.

    ``arguments`` is a JSON *string* from verl's tool parsers (see
    ``FunctionCall`` in ``verl/experimental/agent_loop/tool_parser.py``); the
    env protocol (``agentic_rl_data/envs/base.py``) accepts a mapping or a JSON
    string, and the frozen trajectory schema requires a mapping.
    """
    if raw is None:
        return {}, None
    if isinstance(raw, Mapping):
        return dict(raw), None
    if isinstance(raw, _NUMPY_STR_TYPES):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:  # pragma: no cover - defensive
            return {}, f"invalid arguments encoding: {exc}"
    if not isinstance(raw, str):
        return {}, f"unsupported arguments type {type(raw).__name__}"
    text = raw.strip()
    if not text:
        return {}, None
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        return {}, f"invalid JSON arguments: {exc}"
    if not isinstance(parsed, Mapping):
        return {}, "arguments JSON is not an object"
    return dict(parsed), None


def _call_parts(call: Any) -> tuple[str | None, Any, str | None]:
    """Extract ``(name, arguments, tool_call_id)`` from a tool call object."""
    if isinstance(call, Mapping):
        name = call.get("name")
        arguments = call.get("arguments")
        call_id = call.get("tool_call_id")
        if call_id is None:
            call_id = call.get("id")
        function = call.get("function")
        if isinstance(function, Mapping):
            if not isinstance(name, str) or not name.strip():
                name = function.get("name")
            if arguments is None:
                arguments = function.get("arguments")
    else:
        name = getattr(call, "name", None)
        arguments = getattr(call, "arguments", None)
        call_id = getattr(call, "tool_call_id", None)
        if call_id is None:
            call_id = getattr(call, "id", None)
    return name, arguments, call_id


def build_calls(
    calls: Iterable[Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Convert parsed tool calls to env-action / trajectory forms.

    Returns ``(action_calls, trajectory_calls, errors)``.  Both call lists use
    the frozen shapes::

        action:     {"id": str | None, "name": str, "arguments": dict}
        trajectory: {"id": str | None, "name": str, "arguments": dict}

    Malformed calls (missing name) are skipped and reported instead of
    crashing the rollout; invalid JSON arguments degrade to ``{}`` (the env
    then returns an ``ok=False`` tool result, which the reward sees).
    """
    action_calls: list[dict[str, Any]] = []
    trajectory_calls: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, call in enumerate(calls or []):
        name, arguments, call_id = _call_parts(call)
        if not isinstance(name, str) or not name.strip():
            errors.append(f"tool call #{index}: missing non-empty name")
            continue
        arguments_dict, argument_error = parse_tool_call_arguments(arguments)
        if argument_error:
            errors.append(f"tool call #{index} ({name}): {argument_error}")
        call_payload = {
            "id": call_id if isinstance(call_id, str) and call_id.strip() else None,
            "name": name.strip(),
            "arguments": arguments_dict,
        }
        action_calls.append(dict(call_payload))
        trajectory_calls.append(dict(call_payload))
    return action_calls, trajectory_calls, errors


# ---------------------------------------------------------------------------
# observation -> messages / trajectory
# ---------------------------------------------------------------------------


def _iter_tool_results(tool_results: Any) -> list[Mapping]:
    if tool_results is None:
        return []
    tolist = getattr(tool_results, "tolist", None)
    if callable(tolist):
        tool_results = tolist()
    if isinstance(tool_results, Mapping):
        tool_results = [tool_results]
    if not isinstance(tool_results, (list, tuple)):
        return []
    return [item for item in tool_results if isinstance(item, Mapping)]


def _result_content(result: Mapping) -> str:
    content = result.get("content")
    if content is None:
        return ""
    return content if isinstance(content, str) else str(content)


def tool_results_to_messages(tool_results: Any) -> list[dict[str, Any]]:
    """``obs["tool_results"]`` -> OpenAI chat ``tool`` messages.

    The env observation (CONTRACTS C2) carries ``id`` / ``name`` / ``content``
    / ``ok`` per executed call; the chat template only needs ``role`` and
    ``content`` (plus ``tool_call_id`` when the model emitted an id).
    """
    messages: list[dict[str, Any]] = []
    for result in _iter_tool_results(tool_results):
        message: dict[str, Any] = {"role": "tool", "content": _result_content(result)}
        call_id = result.get("id")
        if isinstance(call_id, str) and call_id.strip():
            message["tool_call_id"] = call_id
        messages.append(message)
    return messages


def tool_results_to_trajectory(tool_results: Any) -> list[dict[str, Any]]:
    """``obs["tool_results"]`` -> ``schema.Trajectory.turns[*].tool_results``."""
    results: list[dict[str, Any]] = []
    for result in _iter_tool_results(tool_results):
        name = result.get("name")
        results.append(
            {
                "id": result.get("id"),
                "name": str(name).strip() if isinstance(name, str) and name.strip() else "unknown",
                "content": _result_content(result),
                "ok": bool(result.get("ok", True)),
            }
        )
    return results


def make_turn(
    step: int,
    content: Any,
    tool_calls: Iterable[Mapping] | None = None,
    tool_results: Iterable[Mapping] | None = None,
) -> dict[str, Any]:
    """Build one ``schema.Trajectory`` turn (``assistant`` + ``tool_results``)."""
    return {
        "step": int(step),
        "assistant": {
            "content": content if isinstance(content, str) else ("" if content is None else str(content)),
            "tool_calls": [dict(call) for call in (tool_calls or []) if isinstance(call, Mapping)],
        },
        "tool_results": [
            dict(result) for result in (tool_results or []) if isinstance(result, Mapping)
        ],
    }


def final_answer_from_turns(turns: Iterable[Mapping] | None) -> str | None:
    """Last assistant content when the final turn emitted no tool call."""
    turns = list(turns or [])
    if not turns:
        return None
    last = turns[-1]
    assistant = last.get("assistant") if isinstance(last, Mapping) else None
    if not isinstance(assistant, Mapping) or assistant.get("tool_calls"):
        return None
    content = assistant.get("content")
    return content if isinstance(content, str) and content else None


def build_trajectory(
    turns: Iterable[Mapping] | None,
    *,
    final_answer: Any = None,
    done: bool = False,
    env_state: Mapping | None = None,
) -> dict[str, Any]:
    """Assemble the frozen ``trajectory`` payload handed to reward functions.

    ``env_state`` is copied verbatim from the env's terminal ``info``
    (CONTRACTS C3: reward reads only the trajectory).  ``None`` means the
    episode never reached the env's terminal state (timeout / token budget /
    env failure); :mod:`agentic_rl_data.rewards.mt_fc` and ``code_exec``
    explicitly handle the missing state instead of silently scoring it.
    """
    turn_list = [dict(turn) for turn in (turns or []) if isinstance(turn, Mapping)]
    state_copy: dict[str, Any] | None = None
    if isinstance(env_state, Mapping):
        try:
            state_copy = copy.deepcopy(dict(env_state))
        except Exception:  # pragma: no cover - defensive (non-copyable extra)
            state_copy = dict(env_state)
    return {
        "turns": turn_list,
        "final_answer": final_answer if isinstance(final_answer, str) else None,
        "num_turns": len(turn_list),
        "done": bool(done),
        "env_state": state_copy,
    }


def single_turn_trajectory(
    content: Any,
    tool_calls: Iterable[Mapping] | None = None,
    *,
    final_answer: Any = None,
) -> dict[str, Any]:
    """One assistant turn, no env state (the ``env.type == "none"`` fallback)."""
    turn = make_turn(1, content, tool_calls, [])
    answer = final_answer if isinstance(final_answer, str) else final_answer_from_turns([turn])
    return build_trajectory([turn], final_answer=answer, done=True, env_state=None)


# ---------------------------------------------------------------------------
# reward dispatch (fail-closed)
# ---------------------------------------------------------------------------


def _default_dispatch(task: Any, trajectory: Any) -> Any:
    """Lazy import keeps this module importable without the reward package."""
    from agentic_rl_data.rewards import dispatch_reward

    return dispatch_reward(task, trajectory)


def clamp_score(value: Any) -> tuple[float, str | None]:
    """Coerce a reward score to ``[0, 1]``; returns ``(score, error)``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0, f"score must be a number, got {type(value).__name__}"
    number = float(value)
    if not math.isfinite(number):
        return 0.0, f"score must be finite, got {value!r}"
    if number < 0.0:
        return 0.0, f"score {number!r} below 0.0 (clamped)"
    if number > 1.0:
        return 1.0, f"score {number!r} above 1.0 (clamped)"
    return number, None


def fail_closed_reward(
    task: Any,
    trajectory: Any,
    dispatch: Callable[[Any, Any], Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Dispatch the task reward and **never raise**.

    Fail-closed semantics (plan §5 / CONTRACTS C4):
    * reward exception / non-mapping result / contract violation -> ``0.0``
      with ``info["reward_error"]`` describing the failure;
    * a valid payload is clamped into the frozen ``[0, 1]`` range.

    Returns ``(score, info)``; ``info`` is always a dict.
    """
    if not isinstance(task, Mapping):
        return 0.0, {"reward_error": "task must be a mapping to compute a reward"}
    dispatcher = dispatch or _default_dispatch
    try:
        result = dispatcher(task, trajectory)
    except Exception as exc:  # noqa: BLE001 - a reward bug must not kill rollout
        return 0.0, {"reward_error": f"reward raised {type(exc).__name__}: {exc}"}
    if not isinstance(result, Mapping):
        return 0.0, {"reward_error": f"reward returned {type(result).__name__}, expected a mapping"}
    raw_info = result.get("info")
    info = dict(raw_info) if isinstance(raw_info, Mapping) else {}
    errors = validate_reward_result(result)
    if errors:
        return 0.0, {**info, "reward_error": "reward contract violation: " + "; ".join(errors)}
    score, score_error = clamp_score(result.get("score"))
    if score_error:
        info.setdefault("reward_error", score_error)
    return score, info


def normalize_reward_info(
    info: Any,
    *,
    task_type: Any = "",
    score: float = 0.0,
    num_turns: int = 0,
    done: bool = False,
    timeout: bool = False,
    reward_error: str = "",
    max_json_chars: int = MAX_REWARD_INFO_JSON_CHARS,
) -> dict[str, Any]:
    """Wrap raw reward ``info`` into the fixed :data:`REWARD_INFO_KEYS` schema.

    ``verl``'s ``_postprocess`` indexes every sample of a worker chunk with the
    **first** sample's key set, so this dict must have identical keys for every
    task type (see module docstring).
    """
    payload = dict(info) if isinstance(info, Mapping) else {}
    try:
        dumped = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        dumped = str(payload)
    if max_json_chars > 0 and len(dumped) > max_json_chars:
        dumped = dumped[:max_json_chars] + "...[truncated]"
    if not reward_error:
        raw_error = payload.get("reward_error")
        reward_error = raw_error if isinstance(raw_error, str) else ""
    return {
        "agentic/task_type": str(task_type or ""),
        "agentic/reward_score": float(clamp_score(score)[0]),
        "agentic/reward_error": str(reward_error),
        "agentic/reward_timeout": bool(timeout),
        "agentic/num_turns": int(num_turns),
        "agentic/done": bool(done),
        "agentic/reward_info": dumped,
    }


__all__ = [
    "DEFAULT_MAX_TURNS",
    "MAX_REWARD_INFO_JSON_CHARS",
    "MISSING_TASK_REWARD_ERROR",
    "REWARD_INFO_KEYS",
    "as_mapping",
    "build_calls",
    "build_trajectory",
    "clamp_score",
    "fail_closed_reward",
    "final_answer_from_turns",
    "make_turn",
    "normalize_parameters",
    "normalize_reward_info",
    "normalize_tool",
    "normalize_tools_for_template",
    "parse_agentic_task",
    "parse_tool_call_arguments",
    "resolve_max_turns",
    "single_turn_trajectory",
    "task_env_type",
    "task_has_env",
    "task_task_type",
    "to_dict_list",
    "tool_names",
    "tool_results_to_messages",
    "tool_results_to_trajectory",
]
