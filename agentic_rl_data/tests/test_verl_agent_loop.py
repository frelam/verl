"""Unit tests for the verl agent-loop adapter (``agentic_rl_data.verl``).

**Not executed in the authoring session**: this machine has no working shell
(``pwsh`` dies with ``0xC0000142``), so the suite below was written but never
run.  Run it on the user machine with::

    cd D:\\lhy\\verl\\verl-toolRL
    python -m pytest agentic_rl_data/tests/test_verl_agent_loop.py -q

The first block only imports :mod:`agentic_rl_data.verl.trajectory` (stdlib),
so it runs **without verl / torch**.  The last block is skipped by
``pytest.importorskip("verl")`` on machines without the training stack; it
checks that the loop registers as ``agentic_agent``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agentic_rl_data.verl import trajectory as tr

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _env_task(
    *,
    task_type: str = "mt_fc",
    env_type: str = "mock_fc",
    max_turns: int = 5,
    tools: list | None = None,
    gt_calls: list | None = None,
) -> dict[str, Any]:
    """Minimal unified task mapping (shape of ``extra_info["agentic_task"]``)."""
    return {
        "id": "src-000123",
        "source": "mock",
        "task_type": task_type,
        "messages": [{"role": "user", "content": "book a table"}],
        "tools": tools if tools is not None else [],
        "env": {"type": env_type, "config": {}},
        "reference": {
            "final_answer": None,
            "final_answers": None,
            "gt_calls": gt_calls if gt_calls is not None else [],
            "gt_turns": None,
            "tool_returns": None,
            "unit_tests": None,
            "final_state": None,
        },
        "meta": {
            "max_turns": max_turns,
            "difficulty": None,
            "pass_rate": None,
            "license": None,
        },
    }


def _exporter_tool_cell() -> dict[str, str]:
    """One ``tools`` cell exactly as ``export_verl_parquet`` writes it."""
    return {
        "name": "get_weather",
        "description": "Look up the weather",
        "parameters": json.dumps(
            {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            }
        ),
    }


# ---------------------------------------------------------------------...
# tool schema normalisation (flat parquet struct -> OpenAI tool list)
# ---------------------------------------------------------------------...


def test_normalize_tools_wraps_flat_parquet_struct() -> None:
    tools = tr.normalize_tools_for_template([_exporter_tool_cell()])
    assert len(tools) == 1
    tool = tools[0]
    assert tool["type"] == "function"
    function = tool["function"]
    assert function["name"] == "get_weather"
    assert function["description"] == "Look up the weather"
    assert function["parameters"]["type"] == "object"
    assert function["parameters"]["properties"] == {"city": {"type": "string"}}
    assert function["parameters"]["required"] == ["city"]


def test_normalize_tools_passes_through_openai_shape() -> None:
    source = {
        "type": "function",
        "function": {
            "name": "search",
            "description": "web search",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    }
    tools = tr.normalize_tools_for_template([source])
    assert tools[0]["function"]["name"] == "search"
    assert tools[0]["function"]["parameters"]["required"] == []
    assert tr.tool_names([source]) == ["search"]


def test_normalize_tools_handles_empty_none_and_numpy_like_cells() -> None:
    assert tr.normalize_tools_for_template(None) == []
    assert tr.normalize_tools_for_template([]) == []
    # pyarrow -> numpy object arrays expose ``tolist``
    item = SimpleNamespace(tolist=lambda: [_exporter_tool_cell()])
    assert tr.tool_names(item) == ["get_weather"]
    # malformed entries are dropped, never crash rollout
    assert tr.normalize_tools_for_template([{"description": "no name"}, 42, "junk"]) == []
    # legacy exporters may store the whole list as a JSON string
    assert tr.tool_names(json.dumps([_exporter_tool_cell()])) == ["get_weather"]


def test_normalize_parameters_tolerates_bad_json_string() -> None:
    parameters = tr.normalize_parameters("{not-json")
    assert parameters == {"type": "object", "properties": {}, "required": []}


# ---------------------------------------------------------------------...
# task parsing / turn cap
# ---------------------------------------------------------------------...


def test_parse_agentic_task_accepts_json_string_mapping_and_invalid() -> None:
    task = _env_task()
    assert tr.parse_agentic_task(json.dumps(task)) == task
    assert tr.parse_agentic_task(task) == task
    assert tr.parse_agentic_task(None) is None
    assert tr.parse_agentic_task("") is None
    assert tr.parse_agentic_task("not json") is None
    assert tr.parse_agentic_task("[1, 2, 3]") is None


def test_task_env_type_and_has_env() -> None:
    assert tr.task_has_env(_env_task(env_type="mock_fc")) is True
    assert tr.task_has_env(_env_task(env_type="none")) is False
    assert tr.task_env_type(None) == "none"
    assert tr.task_env_type(_env_task(env_type="SEARCH")) == "search"
    assert tr.task_has_env(None) is False


def test_resolve_max_turns() -> None:
    assert tr.resolve_max_turns(_env_task(max_turns=5)) == 5
    assert tr.resolve_max_turns({"meta": {"max_turns": 0}}) == tr.DEFAULT_MAX_TURNS
    assert tr.resolve_max_turns({"meta": {"max_turns": True}}) == tr.DEFAULT_MAX_TURNS
    assert tr.resolve_max_turns(None) == tr.DEFAULT_MAX_TURNS


# ---------------------------------------------------------------------...
# FunctionCall -> env action / trajectory tool calls
# ---------------------------------------------------------------------...


def test_parse_tool_call_arguments() -> None:
    assert tr.parse_tool_call_arguments('{"city": "Paris"}') == ({"city": "Paris"}, None)
    assert tr.parse_tool_call_arguments({"city": "Paris"}) == ({"city": "Paris"}, None)
    assert tr.parse_tool_call_arguments(None) == ({}, None)
    assert tr.parse_tool_call_arguments("") == ({}, None)
    arguments, error = tr.parse_tool_call_arguments("{broken")
    assert arguments == {}
    assert error is not None
    assert "invalid JSON" in error
    arguments, error = tr.parse_tool_call_arguments("[1, 2]")
    assert arguments == {}
    assert error is not None
    assert "not an object" in error


def test_build_calls_from_function_call_like_inputs() -> None:
    # verl's ToolParser returns pydantic ``FunctionCall`` objects:
    #   FunctionCall(arguments='{"city": "Paris"}', name="get_weather", tool_call_id="call_1")
    calls = [
        SimpleNamespace(name="get_weather", arguments='{"city": "Paris"}', tool_call_id="call_1"),
        {"name": "search", "arguments": '{"q": "x"}'},
    ]
    action_calls, trajectory_calls, errors = tr.build_calls(calls)
    assert errors == []
    assert action_calls == [
        {"id": "call_1", "name": "get_weather", "arguments": {"city": "Paris"}},
        {"id": None, "name": "search", "arguments": {"q": "x"}},
    ]
    assert trajectory_calls == action_calls


def test_build_calls_reports_malformed_calls_without_raising() -> None:
    calls = [
        {"name": "", "arguments": "{}"},
        {"name": "broken", "arguments": "{not-json"},
    ]
    action_calls, trajectory_calls, errors = tr.build_calls(calls)
    assert action_calls == [{"id": None, "name": "broken", "arguments": {}}]
    assert trajectory_calls == action_calls
    assert len(errors) == 2
    assert "missing non-empty name" in errors[0]
    assert "invalid JSON" in errors[1]


# ---------------------------------------------------------------------...
# obs["tool_results"] -> chat messages / trajectory results
# ---------------------------------------------------------------------...


def test_tool_results_to_messages() -> None:
    results = [
        {"id": "call_1", "name": "get_weather", "content": "sunny", "ok": True},
        {"id": None, "name": "search", "content": "", "ok": False},
    ]
    messages = tr.tool_results_to_messages(results)
    assert messages == [
        {"role": "tool", "content": "sunny", "tool_call_id": "call_1"},
        {"role": "tool", "content": ""},
    ]
    assert tr.tool_results_to_messages(None) == []
    assert tr.tool_results_to_messages("junk") == []


def test_tool_results_to_trajectory_shape() -> None:
    results = [
        {"id": "call_1", "name": "get_weather", "content": "sunny", "ok": True},
        {"name": "", "content": 7},
    ]
    converted = tr.tool_results_to_trajectory(results)
    assert converted[0] == {"id": "call_1", "name": "get_weather", "content": "sunny", "ok": True}
    # env contract violations degrade instead of crashing the reward
    assert converted[1] == {"id": None, "name": "unknown", "content": "7", "ok": True}


# ---------------------------------------------------------------------...
# trajectory assembly
# ---------------------------------------------------------------------...


def test_make_turn_and_build_trajectory() -> None:
    call = {"id": "call_1", "name": "get_weather", "arguments": {"city": "Paris"}}
    result = {"id": "call_1", "name": "get_weather", "content": "sunny", "ok": True}
    turn = tr.make_turn(1, "calling", [call], [result])
    env_state = {"type": "mock_fc", "impl": "replay", "timeout": False, "goal_satisfied": False}

    trajectory = tr.build_trajectory([turn], final_answer="done", done=True, env_state=env_state)
    assert trajectory["num_turns"] == 1
    assert trajectory["done"] is True
    assert trajectory["final_answer"] == "done"
    assert trajectory["env_state"] == env_state
    # CONTRACTS C3: env_state is copied verbatim (mutating the source must not leak)
    env_state["timeout"] = True
    assert trajectory["env_state"]["timeout"] is False


def test_build_trajectory_without_env_state() -> None:
    turn = tr.make_turn(1, "no tools needed", [], [])
    trajectory = tr.build_trajectory([turn], final_answer="no tools needed", done=True, env_state=None)
    assert trajectory["env_state"] is None
    assert trajectory["turns"][0]["assistant"]["content"] == "no tools needed"
    assert trajectory["turns"][0]["tool_results"] == []


def test_single_turn_trajectory_env_missing_degrade() -> None:
    """``env.type == "none"`` / env build failure path (task card fallback)."""
    trajectory = tr.single_turn_trajectory("the answer")
    assert trajectory["num_turns"] == 1
    assert trajectory["done"] is True
    assert trajectory["env_state"] is None
    assert trajectory["final_answer"] == "the answer"

    call = {"id": None, "name": "get_weather", "arguments": {"city": "Paris"}}
    trajectory_with_call = tr.single_turn_trajectory("", [call])
    assert trajectory_with_call["turns"][0]["assistant"]["tool_calls"] == [call]
    # a turn that emitted a tool call is not a final answer
    assert trajectory_with_call["final_answer"] is None


# ---------------------------------------------------------------------...
# fail-closed reward dispatch
# ---------------------------------------------------------------------...


def test_fail_closed_reward_returns_score_and_info() -> None:
    task = _env_task(task_type="st_fc", env_type="none")
    trajectory = tr.single_turn_trajectory("answer")

    def dispatch(_task: Any, _trajectory: Any) -> dict[str, Any]:
        return {"score": 0.75, "info": {"dim": 0.25}}

    score, info = tr.fail_closed_reward(task, trajectory, dispatch=dispatch)
    assert score == 0.75
    assert info == {"dim": 0.25}


def test_fail_closed_reward_swallows_exceptions() -> None:
    def dispatch(_task: Any, _trajectory: Any) -> dict[str, Any]:
        raise RuntimeError("reward blew up")

    score, info = tr.fail_closed_reward(_env_task(), {}, dispatch=dispatch)
    assert score == 0.0
    assert "reward raised RuntimeError" in info["reward_error"]


def test_fail_closed_reward_rejects_bad_payloads() -> None:
    def bad_type(_task: Any, _trajectory: Any) -> list:
        return [1, 2, 3]

    def bad_score(_task: Any, _trajectory: Any) -> dict[str, Any]:
        return {"score": "high", "info": {}}

    def out_of_range(_task: Any, _trajectory: Any) -> dict[str, Any]:
        return {"score": 1.25, "info": {}}

    def non_mapping_info(_task: Any, _trajectory: Any) -> dict[str, Any]:
        return {"score": 0.5, "info": None}

    for dispatcher in (bad_type, bad_score, out_of_range, non_mapping_info):
        score, info = tr.fail_closed_reward(_env_task(), {}, dispatch=dispatcher)
        assert score == 0.0
        assert isinstance(info, dict)
        assert "reward_error" in info


def test_fail_closed_reward_requires_task_mapping() -> None:
    score, info = tr.fail_closed_reward(None, {}, dispatch=lambda *_: {"score": 1.0, "info": {}})
    assert score == 0.0
    assert info["reward_error"] == "task must be a mapping to compute a reward"


def test_fail_closed_reward_with_real_dispatch_for_st_fc() -> None:
    """Integration with the real reward registry (stdlib only, no verl)."""
    tool = {
        "name": "get_weather",
        "description": "Look up the weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
    task = _env_task(
        task_type="st_fc",
        env_type="none",
        tools=[tool],
        gt_calls=[[{"name": "get_weather", "arguments": {"city": "Paris"}}]],
    )
    call = {"id": None, "name": "get_weather", "arguments": {"city": "Paris"}}
    result = {"id": None, "name": "get_weather", "content": "sunny", "ok": True}
    turn = tr.make_turn(1, "", [call], [result])
    trajectory = tr.build_trajectory([turn], final_answer=None, done=True, env_state=None)
    score, info = tr.fail_closed_reward(task, trajectory)
    assert score == 1.0
    assert info["task_type"] == "st_fc"


# ---------------------------------------------------------------------...
# reward_extra_info: fixed key set (verl _postprocess indexes sample 0's keys)
# ---------------------------------------------------------------------...


def test_normalize_reward_info_has_fixed_keys_for_every_task_type() -> None:
    infos = [
        ({"dim": 0.8, "task_type": "mt_fc"}, "mt_fc"),
        ({"all_calls_matched": True, "task_type": "st_fc"}, "st_fc"),
        ({"bridge": "single_turn", "bridge_error": "x"}, ""),
    ]
    key_sets = []
    for info, task_type in infos:
        payload = tr.normalize_reward_info(info, task_type=task_type, score=0.5, num_turns=2, done=True)
        assert tuple(payload.keys()) == tr.REWARD_INFO_KEYS
        key_sets.append(frozenset(payload.keys()))
    assert len(set(key_sets)) == 1


def test_normalize_reward_info_defaults_and_serialisation() -> None:
    payload = tr.normalize_reward_info(
        {"detail": {"a": 1}}, task_type="mt_fc", score=1.0, num_turns=3, done=False, timeout=True
    )
    assert payload["agentic/task_type"] == "mt_fc"
    assert payload["agentic/reward_score"] == 1.0
    assert payload["agentic/reward_error"] == ""
    assert payload["agentic/reward_timeout"] is True
    assert payload["agentic/num_turns"] == 3
    assert payload["agentic/done"] is False
    assert json.loads(payload["agentic/reward_info"]) == {"detail": {"a": 1}}


def test_normalize_reward_info_missing_task_degrade() -> None:
    payload = tr.normalize_reward_info(
        {"reward_error": tr.MISSING_TASK_REWARD_ERROR},
        task_type="",
        score=0.0,
        num_turns=1,
        done=True,
    )
    assert payload["agentic/reward_error"] == tr.MISSING_TASK_REWARD_ERROR
    assert payload["agentic/reward_score"] == 0.0


# ---------------------------------------------------------------------...
# config wiring (pure file checks; no verl import needed)
# ---------------------------------------------------------------------...


def test_agent_loop_config_yaml_matches_registration() -> None:
    config_path = _REPO_ROOT / "agentic_rl_data" / "verl" / "agent_loop_config.yaml"
    assert config_path.is_file(), config_path
    text = config_path.read_text(encoding="utf-8")
    assert "name: agentic_agent" in text
    assert "_target_: agentic_rl_data.verl.agent_loop_agentic.AgenticAgentLoop" in text


# ---------------------------------------------------------------------...
# verl-gated: registration + class wiring (skipped without the training stack)
# ---------------------------------------------------------------------...


def test_agentic_agent_is_registered_in_verl_when_available() -> None:
    pytest.importorskip("verl")
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase, _agent_loop_registry

    from agentic_rl_data.verl.agent_loop_agentic import AgenticAgentLoop

    assert issubclass(AgenticAgentLoop, AgentLoopBase)
    assert "agentic_agent" in _agent_loop_registry
    assert _agent_loop_registry["agentic_agent"]["_target_"].endswith(
        "agentic_rl_data.verl.agent_loop_agentic.AgenticAgentLoop"
    )


def test_agentic_agent_loop_exposes_expected_entry_point_when_available() -> None:
    pytest.importorskip("verl")
    from agentic_rl_data.verl.agent_loop_agentic import AgenticAgentLoop

    assert callable(AgenticAgentLoop.run)
    assert AgenticAgentLoop.run.__name__ == "run"
