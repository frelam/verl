"""Fixture builders for the reward unit tests (stdlib only, no pytest import).

All builders produce payloads that satisfy the frozen schema v2.1:
``schema.validate_task`` needs the 8 task keys, all 7 ``reference`` keys
(``final_answer`` / ``final_answers`` / ``gt_calls`` / ``gt_turns`` /
``tool_returns`` / ``unit_tests`` / ``final_state``) and the 4 ``meta`` keys
(``max_turns`` / ``difficulty`` / ``pass_rate`` / ``license``);
``schema.validate_trajectory`` needs ``turns[*].tool_results[i]`` to be
``{"name": non-empty str, "content": str}``.

The tests import this module via::

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fixtures.reward_fixtures import task, call, turn, trajectory, ...
"""

from __future__ import annotations

from typing import Any

#: Allowed env type per task type (mirror of schema.ENV_FOR_TASK_TYPE).
TASK_ENV_TYPES: dict[str, str] = {
    "mt_fc": "mock_fc",
    "st_fc": "none",
    "irrelevance": "none",
    "search_qa": "search",
    "math_ci": "python",
    "code_exec": "code_tests",
    "abstain": "none",
}

GET_WEATHER: dict[str, Any] = {
    "name": "get_weather",
    "description": "Get the current weather for a single city.",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["city"],
    },
}

GET_FORECAST: dict[str, Any] = {
    "name": "get_forecast",
    "description": "Get a multi-day forecast for a single city.",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "days": {"type": "integer"},
        },
        "required": ["city", "days"],
    },
}

GET_BULK_WEATHER: dict[str, Any] = {
    "name": "get_bulk_weather",
    "description": "Get the current weather for several cities at once.",
    "parameters": {
        "type": "object",
        "properties": {
            "cities": {"type": "array", "items": {"type": "string"}},
            "unit": {"type": "string"},
        },
        "required": ["cities"],
    },
}

SEARCH: dict[str, Any] = {
    "name": "search",
    "description": "Search a local Wikipedia index and return snippets.",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
        "required": ["query"],
    },
}

PYTHON: dict[str, Any] = {
    "name": "python",
    "description": "Execute Python code in a sandbox and return stdout.",
    "parameters": {
        "type": "object",
        "properties": {"code": {"type": "string"}},
        "required": ["code"],
    },
}

DEFAULT_TOOLS: dict[str, list[dict[str, Any]]] = {
    "mt_fc": [GET_WEATHER, GET_FORECAST],
    "st_fc": [GET_WEATHER, GET_FORECAST, GET_BULK_WEATHER],
    "irrelevance": [GET_WEATHER],
    "search_qa": [SEARCH],
    "math_ci": [PYTHON],
    "code_exec": [],
    "abstain": [GET_WEATHER],
}


def call(name: str, arguments: dict | None = None, *, call_id: str | None = None) -> dict:
    """One assistant tool call."""
    return {"id": call_id, "name": name, "arguments": dict(arguments or {})}


def tool_result(
    name: str,
    content: str = "ok",
    ok: bool = True,
    *,
    result_id: str | None = None,
) -> dict:
    """One env ``tool_results`` entry (schema v2.1: name + content required)."""
    return {"id": result_id, "name": name, "content": content, "ok": ok}


def turn(
    step: int,
    calls: list | None = None,
    content: str = "",
    results: list | None = None,
) -> dict:
    """One trajectory turn."""
    return {
        "step": step,
        "assistant": {"content": content, "tool_calls": list(calls or [])},
        "tool_results": list(results or []),
    }


def turns_from_call_groups(call_groups: list[list[dict]]) -> list[dict]:
    """Turn a list of per-turn call groups into valid turns (results auto-filled)."""
    turns: list[dict] = []
    for index, group in enumerate(call_groups, start=1):
        calls = [dict(call_spec) for call_spec in group]
        results = [tool_result(call_spec.get("name", "")) for call_spec in calls]
        turns.append(turn(index, calls, results=results))
    return turns


def trajectory(
    turns: list | None = None,
    final_answer: str | None = None,
    *,
    env_state_value: dict | None = None,
    done: bool = True,
    num_turns: int | None = None,
) -> dict:
    """One normalised trajectory."""
    turns = list(turns or [])
    return {
        "turns": turns,
        "final_answer": final_answer,
        "num_turns": len(turns) if num_turns is None else num_turns,
        "done": done,
        "env_state": env_state_value,
    }


def env_state(**overrides: Any) -> dict:
    """A complete ``trajectory["env_state"]`` snapshot (CONTRACTS C3)."""
    state: dict[str, Any] = {
        "type": "mock_fc",
        "impl": None,
        "goal_satisfied": None,
        "replay_ok": None,
        "failed_turn": None,
        "matched_turns": None,
        "gt_turns": None,
        "checks": [],
        "state": None,
        "rendered": "",
        "num_turns": 0,
        "timeout": False,
        "code_tests": None,
        "extras": {},
    }
    state.update(overrides)
    return state


def task(
    task_type: str,
    *,
    tools: list | None = None,
    gt_calls: list | None = None,
    final_answer: str | None = None,
    final_answers: list | None = None,
    final_state: dict | None = None,
    unit_tests: str | None = None,
    gt_turns: list | None = None,
    tool_returns: list | None = None,
    env_type: str | None = None,
    env_config: dict | None = None,
    meta: dict | None = None,
    user_content: str = "Please help with the task.",
    task_id: str = "fixture-001",
    source: str = "unit_test",
) -> dict:
    """One schema-valid task (all frozen v2.1 required keys filled)."""
    reference = {
        "final_answer": final_answer,
        "final_answers": final_answers,
        "gt_calls": gt_calls if gt_calls is not None else [],
        "gt_turns": gt_turns,
        "tool_returns": tool_returns,
        "unit_tests": unit_tests,
        "final_state": final_state,
    }
    meta_values: dict[str, Any] = {
        "max_turns": 15,
        "difficulty": None,
        "pass_rate": None,
        "license": None,
    }
    if task_type == "mt_fc":
        # schema v2.2: mt_fc tasks MUST declare where the replay gt returns
        # come from (CONTRACTS C6).
        meta_values["gt_returns_source"] = "reference"
    if meta:
        meta_values.update(meta)
    return {
        "id": task_id,
        "source": source,
        "task_type": task_type,
        "messages": [
            {"role": "system", "content": "You are a helpful tool-using assistant."},
            {"role": "user", "content": user_content},
        ],
        "tools": list(DEFAULT_TOOLS[task_type] if tools is None else tools),
        "env": {
            "type": env_type or TASK_ENV_TYPES[task_type],
            "config": dict(env_config or {}),
        },
        "reference": reference,
        "meta": meta_values,
    }
