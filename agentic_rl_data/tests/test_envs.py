"""Unit tests for the environment protocol, mock_fc implementations and the
``max_turns`` / state-isolation contracts (plan §4, CONTRACTS.md C1–C3).

**Not executed in the authoring session** (the machine has no working shell:
``pwsh`` fails with ``0xC0000142``).  Run on the user machine with::

    cd D:\\lhy\\verl\\verl-toolRL
    python -m pytest agentic_rl_data/tests/test_envs.py -q

All tests are dependency-free (stdlib only) except pytest itself.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping

import pytest

from agentic_rl_data import schema
from agentic_rl_data.envs.base import (
    EnvError,
    NullEnv,
    build_env,
    make_env_state,
    normalise_checks,
    normalize_action,
    truncate_text,
    validate_env_state,
)
from agentic_rl_data.envs.code_tests import CodeTestsEnv
from agentic_rl_data.envs.mock_fc import (
    MockFCEnv,
    ReplayMockFCEnv,
    deep_equal,
    evaluate_goal,
    match_call_group,
    resolve_gt_returns,
)
from agentic_rl_data.envs.python_env import PythonEnv
from agentic_rl_data.envs.search import SearchEnv


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def _reference(**overrides: Any) -> dict[str, Any]:
    reference: dict[str, Any] = {
        "final_answer": None,
        "final_answers": None,
        "gt_calls": [],
        "gt_turns": None,
        "tool_returns": None,
        "unit_tests": None,
        "final_state": None,
    }
    reference.update(overrides)
    return reference


def make_task(
    *,
    task_type: str = "mt_fc",
    env_type: str = "mock_fc",
    config: Mapping[str, Any] | None = None,
    reference: Mapping[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    max_turns: int = 15,
    task_id: str = "test-1",
    source: str = "unit-test",
    user: str = "please do the task",
    gt_returns_source: str | None = None,
) -> dict[str, Any]:
    if gt_returns_source is None and task_type == "mt_fc":
        # schema v2.2: required for mt_fc (singular "reference", not plural).
        gt_returns_source = "reference"
    meta: dict[str, Any] = {
        "max_turns": max_turns,
        "difficulty": None,
        "pass_rate": None,
        "license": None,
    }
    if gt_returns_source is not None:
        meta["gt_returns_source"] = gt_returns_source
    task = {
        "id": task_id,
        "source": source,
        "task_type": task_type,
        "messages": [{"role": "user", "content": user}],
        "tools": list(tools or []),
        "env": {"type": env_type, "config": dict(config or {})},
        "reference": _reference(**(reference or {})),
        "meta": meta,
    }
    errors = schema.validate_task(task)
    assert not errors, f"fixture task is schema-invalid: {errors}"
    return task


def tool(name: str, parameters: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "description": f"tool {name}",
        "parameters": dict(parameters or {"type": "object", "properties": {}}),
    }


def call(name: str, **arguments: Any) -> dict[str, Any]:
    return {"name": name, "arguments": dict(arguments)}


def replay_task(
    *,
    task_id: str = "replay-1",
    gt_calls: list[list[dict[str, Any]]] | None = None,
    tool_returns: list[list[dict[str, Any]]] | None = None,
    config: Mapping[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    max_turns: int = 15,
) -> dict[str, Any]:
    return make_task(
        task_id=task_id,
        config={"impl": "replay", **(dict(config or {}))},
        reference={
            "gt_calls": gt_calls or [],
            "tool_returns": tool_returns,
            "final_answer": "The task is complete.",
        },
        tools=tools if tools is not None else [tool("get_weather"), tool("book_flight")],
        max_turns=max_turns,
    )


def stateful_task(
    *,
    task_id: str = "stateful-1",
    domain: str = "filesystem",
    config: Mapping[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    max_turns: int = 15,
) -> dict[str, Any]:
    return make_task(
        task_id=task_id,
        config={"impl": "stateful", "domain": domain, **(dict(config or {}))},
        tools=tools
        if tools is not None
        else [
            tool("mkdir"),
            tool("touch"),
            tool("write_file"),
            tool("read_file"),
            tool("list_dir"),
            tool("delete_file"),
            tool("delete_dir"),
            tool("move_file"),
            tool("copy_file"),
            tool("get_weather"),
            tool("mystery"),
        ],
        max_turns=max_turns,
    )


def assert_trajectory_valid(trajectory: Mapping[str, Any]) -> None:
    errors = schema.validate_trajectory(trajectory)
    assert not errors, f"trajectory is schema-invalid: {errors}"


# ---------------------------------------------------------------------------
# base protocol helpers
# ---------------------------------------------------------------------------


def test_normalize_action_accepts_string_and_nested_function_shape() -> None:
    action = normalize_action("hello", step=1)
    assert action == {"content": "hello", "tool_calls": []}

    nested = normalize_action(
        {
            "content": "",
            "tool_calls": [
                {"function": {"name": "f", "arguments": '{"x": 1}'}, "id": "abc"}
            ],
        },
        step=1,
    )
    assert nested["tool_calls"][0]["name"] == "f"
    assert nested["tool_calls"][0]["arguments"] == {"x": 1}
    assert nested["tool_calls"][0]["id"] == "abc"

    with pytest.raises(EnvError):
        normalize_action({"tool_calls": [{"arguments": {}}]})
    with pytest.raises(EnvError):
        normalize_action(123)


def test_make_env_state_is_stable() -> None:
    state = make_env_state("mock_fc", "replay", replay_ok=True, matched_turns=2)
    for key in ("type", "impl", "goal_satisfied", "replay_ok", "failed_turn",
                "matched_turns", "gt_turns", "checks", "state", "rendered",
                "num_turns", "timeout", "code_tests", "extras"):
        assert key in state, f"missing stable env_state key {key}"
    assert state["goal_satisfied"] is None
    assert state["code_tests"] is None
    assert state["extras"] == {}


def test_truncate_text() -> None:
    short, truncated = truncate_text("abc", 10)
    assert short == "abc" and truncated is False
    long_text, truncated = truncate_text("x" * 100, 20)
    assert truncated is True and len(long_text) <= 20


# ---------------------------------------------------------------------------
# replay implementation
# ---------------------------------------------------------------------------


def test_replay_full_match_exposes_replay_ok() -> None:
    gt_calls = [
        [call("get_weather", city="Beijing")],
        [call("book_flight", city="Shanghai")],
    ]
    returns = [
        [{"name": "get_weather", "content": "sunny", "ok": True}],
        [{"name": "book_flight", "content": "booked", "ok": True}],
    ]
    task = replay_task(gt_calls=gt_calls, tool_returns=returns)
    env = build_env(task)
    assert isinstance(env, ReplayMockFCEnv)

    obs = env.reset(task)
    assert obs["step"] == 0 and obs["tool_results"] == []
    assert obs["env_state"]["impl"] == "replay"

    obs, done, info = env.step({"content": "", "tool_calls": gt_calls[0]})
    assert done is False
    assert obs["tool_results"][0]["name"] == "get_weather"
    assert obs["tool_results"][0]["content"] == "sunny"
    assert obs["tool_results"][0]["ok"] is True
    assert obs["tool_results"][0]["id"]  # generated id, one-to-one with the call

    obs, done, info = env.step({"content": "", "tool_calls": gt_calls[1]})
    assert done is False

    obs, done, info = env.step({"content": "done", "tool_calls": []})
    assert done is True
    assert env.replay_ok is True
    state = env.env_state
    assert state["replay_ok"] is True
    assert state["matched_turns"] == 2
    assert state["gt_turns"] == 2
    assert state["failed_turn"] is None
    assert state["extras"]["gt_returns_source"] == "reference"
    assert info["timeout"] is False
    assert info["final_answer"] == "done"
    trajectory = env.trajectory()
    assert trajectory["done"] is True
    assert_trajectory_valid(trajectory)
    env.close()


def test_replay_mismatch_terminates_with_failed_turn() -> None:
    gt_calls = [[call("get_weather", city="Beijing")]]
    task = replay_task(gt_calls=gt_calls, tool_returns=[[{"name": "get_weather", "content": "sunny"}]])
    env = ReplayMockFCEnv(task)
    env.reset()
    obs, done, info = env.step(
        {"content": "", "tool_calls": [call("get_weather", city="Shanghai")]}
    )
    assert done is True
    assert env.replay_ok is False
    assert env.env_state["failed_turn"] == 0
    assert info["replay"]["ok"] is False
    assert "no ground-truth match" in info["replay"]["reason"]
    assert obs["tool_results"][0]["ok"] is False


def test_replay_parallel_calls_are_order_insensitive() -> None:
    gt_group = [call("get_weather", city="Beijing"), call("book_flight", city="Shanghai")]
    task = replay_task(gt_calls=[gt_group])
    env = ReplayMockFCEnv(task)
    env.reset()
    obs, done, info = env.step(
        {"content": "", "tool_calls": [gt_group[1], gt_group[0]]}
    )
    assert done is False
    assert env.env_state["matched_turns"] == 1
    obs, done, _ = env.step({"content": "done"})
    assert done is True and env.replay_ok is True


def test_replay_extra_call_after_gt_exhausted_fails() -> None:
    task = replay_task(gt_calls=[[call("get_weather", city="Beijing")]])
    env = ReplayMockFCEnv(task)
    env.reset()
    env.step({"content": "", "tool_calls": [call("get_weather", city="Beijing")]})
    _, done, info = env.step({"content": "", "tool_calls": [call("book_flight", city="Paris")]})
    assert done is True
    assert env.replay_ok is False
    assert "exhausted" in info["replay"]["reason"]


def test_replay_early_final_answer_fails() -> None:
    task = replay_task(gt_calls=[[call("get_weather", city="Beijing")]])
    env = ReplayMockFCEnv(task)
    env.reset()
    _, done, info = env.step({"content": "I answer without tools"})
    assert done is True
    assert env.replay_ok is False
    assert "before ground-truth" in info["replay"]["reason"]


def test_replay_empty_gt_calls_passes_on_final_answer() -> None:
    task = replay_task(gt_calls=[])
    env = ReplayMockFCEnv(task)
    env.reset()
    _, done, _ = env.step({"content": "answer"})
    assert done is True
    assert env.replay_ok is True


def test_replay_matching_is_loose_by_default_and_exact_on_demand() -> None:
    gt = [call("get_weather", city=" Beijing ", days=3)]
    loose, _ = match_call_group([call("get_weather", city="beijing", days=3.0)], gt, mode="loose")
    assert loose is True
    exact, _ = match_call_group([call("get_weather", city="beijing", days=3.0)], gt, mode="exact")
    assert exact is False
    # Extra argument keys are not tolerated.
    _, reason = match_call_group([call("get_weather", city="Beijing", days=3, unit="C")], gt)
    assert "no ground-truth match" in reason


def test_replay_is_deterministic() -> None:
    task = replay_task(
        gt_calls=[[call("get_weather", city="Beijing")]],
        tool_returns=[[{"name": "get_weather", "content": "sunny"}]],
    )

    def run() -> dict[str, Any]:
        env = ReplayMockFCEnv(task)
        env.reset()
        env.step({"content": "", "tool_calls": [call("get_weather", city="Beijing")]})
        env.step({"content": "done"})
        return env.trajectory()

    first, second = run(), run()
    assert json.dumps(first, sort_keys=True, ensure_ascii=False) == json.dumps(
        second, sort_keys=True, ensure_ascii=False
    )


def test_replay_placeholder_source_is_recorded_not_silent() -> None:
    task = replay_task(gt_calls=[[call("get_weather", city="Beijing")]])
    env = ReplayMockFCEnv(task)
    env.reset()
    obs, _, _ = env.step({"content": "", "tool_calls": [call("get_weather", city="Beijing")]})
    # T4-02: a missing GT return is an ok=False result, never a fake success.
    assert obs["tool_results"][0]["ok"] is False
    assert "ground-truth return missing" in obs["tool_results"][0]["content"]
    _, _, info = env.step({"content": "done"})
    assert env.replay_ok is True
    extras = env.env_state["extras"]
    assert extras["gt_returns_source"] == "placeholder"
    assert extras["degraded_replay"] is True
    assert extras["gt_returns_missing"] == 1
    assert extras["gt_returns_missing_ratio"] == 1.0
    assert info["degraded_replay"] is True


def test_replay_missing_returns_downgrade_source_and_flag_ok_false() -> None:
    # Two parallel calls; only one return exists -> ratio 0.5 > threshold 0.0.
    gt_calls = [[call("get_weather", city="Beijing"), call("book_flight", city="Paris")]]
    task = replay_task(
        gt_calls=gt_calls,
        tool_returns=[[{"name": "get_weather", "content": "sunny", "ok": True}]],
        config={"max_missing_ratio": 0.0},
    )
    env = ReplayMockFCEnv(task)
    env.reset()
    obs, done, _ = env.step(
        {"content": "", "tool_calls": [call("get_weather", city="Beijing"), call("book_flight", city="Paris")]}
    )
    assert done is False
    by_name = {result["name"]: result for result in obs["tool_results"]}
    assert by_name["get_weather"]["ok"] is True
    assert by_name["book_flight"]["ok"] is False
    assert "ground-truth return missing" in by_name["book_flight"]["content"]
    _, done, _ = env.step({"content": "done"})
    assert done is True
    assert env.replay_ok is True
    extras = env.env_state["extras"]
    assert extras["gt_returns_missing"] == 1
    assert extras["gt_returns_missing_ratio"] == 0.5
    assert extras["gt_returns_declared_source"] == "reference"
    assert extras["gt_returns_source"] == "placeholder"  # downgraded, visible
    assert extras["degraded_replay"] is True


def test_replay_missing_returns_below_threshold_keeps_source() -> None:
    gt_calls = [[call("get_weather", city="Beijing"), call("book_flight", city="Paris")]]
    task = replay_task(
        gt_calls=gt_calls,
        tool_returns=[[{"name": "get_weather", "content": "sunny", "ok": True}]],
        config={"max_missing_ratio": 0.75},
    )
    env = ReplayMockFCEnv(task)
    env.reset()
    env.step(
        {"content": "", "tool_calls": [call("get_weather", city="Beijing"), call("book_flight", city="Paris")]}
    )
    env.step({"content": "done"})
    extras = env.env_state["extras"]
    assert extras["gt_returns_missing"] == 1
    assert extras["gt_returns_source"] == "reference"
    assert extras["degraded_replay"] is False


def test_resolve_gt_returns_priority() -> None:
    task = replay_task(
        gt_calls=[[call("get_weather", city="Beijing")]],
        tool_returns=[[{"name": "get_weather", "content": "from reference"}]],
        config={"gt_returns": [[{"name": "get_weather", "content": "from config"}]]},
    )
    groups, source = resolve_gt_returns(task, task["env"]["config"])
    assert source == "config"
    assert groups[0][0]["content"] == "from config"

    task = replay_task(
        gt_calls=[[call("get_weather", city="Beijing")]],
        tool_returns=[[{"name": "get_weather", "content": "from reference"}]],
    )
    groups, source = resolve_gt_returns(task, task["env"]["config"])
    assert source == "reference"

    task = replay_task(
        gt_calls=[[call("get_weather", city="Beijing")]],
        tool_returns=None,
    )
    task["reference"]["gt_turns"] = [{"tool_results": [{"name": "get_weather", "content": "gt turn"}]}]
    groups, source = resolve_gt_returns(task, task["env"]["config"])
    assert source == "gt_turns"
    assert groups[0][0]["content"] == "gt turn"

    task = replay_task(gt_calls=[[call("get_weather", city="Beijing")]])
    _, source = resolve_gt_returns(task, task["env"]["config"])
    assert source == "placeholder"


# ---------------------------------------------------------------------------
# stateful implementation
# ---------------------------------------------------------------------------


def test_stateful_filesystem_goal_and_snapshot() -> None:
    task = stateful_task(
        config={
            "initial_state": {"files": {}, "dirs": ["/", "/home", "/home/user"]},
            "goal": [
                {"name": "notes created", "check": "file_exists", "path": "/home/user/notes.txt"},
                {
                    "name": "notes content",
                    "check": "file_content",
                    "path": "/home/user/notes.txt",
                    "expected": "hello",
                },
            ],
        }
    )
    env = MockFCEnv(task)
    obs = env.reset()
    assert obs["env_state"]["goal_satisfied"] is False
    obs, done, info = env.step(
        {
            "content": "",
            "tool_calls": [
                call("write_file", path="/home/user/notes.txt", content="hello"),
            ],
        }
    )
    assert done is False
    assert obs["tool_results"][0]["ok"] is True
    assert "/home/user/notes.txt" in obs["tool_results"][0]["content"]

    obs, done, info = env.step({"content": "done", "tool_calls": []})
    assert done is True
    state = env.env_state
    assert state["goal_satisfied"] is True
    assert all(item["ok"] for item in state["checks"])
    assert state["state"]["files"]["/home/user/notes.txt"] == "hello"
    assert state["extras"]["final_answer_matches"] is None
    assert_trajectory_valid(env.trajectory())
    assert len(state["checks"]) == 2


def test_stateful_goal_failure_and_read_errors() -> None:
    task = stateful_task(
        config={
            "goal": [
                {"check": "file_exists", "path": "/missing.txt"},
            ]
        }
    )
    env = MockFCEnv(task)
    env.reset()
    obs, _, _ = env.step(
        {"content": "", "tool_calls": [call("read_file", path="/missing.txt")]}
    )
    assert obs["tool_results"][0]["ok"] is False
    assert "not found" in obs["tool_results"][0]["content"]
    _, done, _ = env.step({"content": "done"})
    assert done is True
    assert env.env_state["goal_satisfied"] is False


def test_stateful_canned_response_substitution() -> None:
    task = stateful_task(
        config={
            "domain": "social",
            "responses": {"get_weather": {"content": "sunny in {city}", "ok": True}},
            "goal": [
                {"check": "value", "path": "next_post_id", "expected": 2},
            ],
        },
        tools=[tool("post"), tool("get_weather"), tool("mystery")],
    )
    env = MockFCEnv(task)
    env.reset()
    obs, _, _ = env.step(
        {"content": "", "tool_calls": [call("get_weather", city="Beijing")]}
    )
    assert obs["tool_results"][0]["content"] == "sunny in Beijing"
    assert obs["tool_results"][0]["ok"] is True

    obs, _, _ = env.step({"content": "", "tool_calls": [call("mystery")]})
    assert obs["tool_results"][0]["ok"] is False
    assert "unknown tool" in obs["tool_results"][0]["content"]

    obs, _, _ = env.step({"content": "", "tool_calls": [call("post", content="hi")]})
    assert obs["tool_results"][0]["ok"] is True
    _, done, _ = env.step({"content": "done"})
    assert done is True
    assert env.env_state["goal_satisfied"] is True


def test_stateful_trading_domain() -> None:
    task = make_task(
        config={
            "impl": "stateful",
            "domain": "trading",
            "initial_state": {"balance": 1000.0, "prices": {"AAPL": 100.0}},
            "goal": [
                {"check": "value", "path": "balance", "expected": 800.0, "mode": "approx"},
                {"check": "value", "path": "holdings.AAPL", "expected": 2},
            ],
        },
        tools=[tool("buy"), tool("get_balance"), tool("get_holdings")],
    )
    env = MockFCEnv(task)
    env.reset()
    obs, _, _ = env.step(
        {"content": "", "tool_calls": [call("buy", symbol="AAPL", quantity=2)]}
    )
    assert obs["tool_results"][0]["ok"] is True
    assert '"balance":800.0' in obs["tool_results"][0]["content"].replace(" ", "")
    obs, _, _ = env.step(
        {"content": "", "tool_calls": [call("buy", symbol="AAPL", quantity=10)]}
    )
    assert obs["tool_results"][0]["ok"] is False
    assert "insufficient funds" in obs["tool_results"][0]["content"]
    _, done, _ = env.step({"content": "done"})
    assert done is True
    assert env.env_state["goal_satisfied"] is True


def test_stateful_no_goal_keeps_goal_satisfied_null() -> None:
    task = stateful_task()
    env = MockFCEnv(task)
    env.reset()
    _, done, _ = env.step({"content": "done"})
    assert done is True
    assert env.env_state["goal_satisfied"] is None
    assert env.env_state["state"]["dirs"][0] == "/"


def test_stateful_rejects_unknown_domain() -> None:
    task = stateful_task()
    task["env"]["config"]["domain"] = "does-not-exist"
    with pytest.raises(EnvError):
        MockFCEnv(task)


def test_stateful_declared_tool_check() -> None:
    task = stateful_task(tools=[tool("list_dir")])
    env = MockFCEnv(task)
    env.reset()
    obs, _, _ = env.step(
        {"content": "", "tool_calls": [call("write_file", path="/x.txt", content="x")]}
    )
    assert obs["tool_results"][0]["ok"] is False
    assert "not declared" in obs["tool_results"][0]["content"]


def test_stateful_final_answer_matching_modes() -> None:
    task = stateful_task(config={"final_answer_mode": "contains", "goal": []})
    task["reference"]["final_answer"] = "the notes file"
    env = MockFCEnv(task)
    env.reset()
    _, done, _ = env.step({"content": "I created the notes file", "tool_calls": []})
    assert done is True
    assert env.env_state["extras"]["final_answer_matches"] is True


def test_evaluate_goal_check_kinds() -> None:
    state = {"files": {"/a.txt": "hello"}, "dirs": ["/", "/home"], "balance": 10.0}
    checks = [
        {"check": "file_exists", "path": "/a.txt"},
        {"check": "file_not_exists", "path": "/missing.txt"},
        {"check": "file_content", "path": "/a.txt", "expected": "hel", "mode": "contains"},
        {"check": "dir_exists", "path": "/home"},
        {"check": "entries", "path": "/", "expected": ["a.txt", "home"]},
        {"check": "value", "path": "balance", "expected": 10.0, "mode": "gte"},
        {"check": "count", "path": "dirs", "expected": 2},
        {"check": "all_of", "checks": [{"check": "file_exists", "path": "/a.txt"}]},
        {"check": "any_of", "any_of": [{"check": "file_exists", "path": "/b.txt"}]},
    ]
    results = evaluate_goal(checks, state)
    assert [item["ok"] for item in results] == [True, True, True, True, True, True, True, True, False]
    assert len(results) == len(checks)

    bad = evaluate_goal([{"check": "nonsense", "path": "x"}], state)
    assert bad[0]["ok"] is False
    assert "unknown goal check kind" in bad[0]["detail"]


def test_deep_equal_semantics() -> None:
    assert deep_equal({"a": " X ", "b": 2.0}, {"a": "x", "b": 2}) is True
    assert deep_equal({"a": 1}, {"a": 1, "b": 2}) is False
    assert deep_equal([1, 2], [1, 2]) is True
    assert deep_equal([1, 2], [2, 1]) is False
    assert deep_equal({"a": 1}, {"a": 1}, loose=False) is True
    assert deep_equal({"a": 1.0}, {"a": 1}, loose=False) is False


# ---------------------------------------------------------------------------
# max_turns / termination
# ---------------------------------------------------------------------------


def test_max_turns_timeout_forces_failed_terminal_verdict() -> None:
    task = stateful_task(
        config={"goal": [{"check": "file_exists", "path": "/never.txt"}]},
        max_turns=2,
    )
    env = MockFCEnv(task)
    env.reset()
    _, done, info = env.step({"content": "", "tool_calls": [call("list_dir", path="/")]})
    assert done is False
    assert info["timeout"] is False
    # A step before timeout still reports the assertion verdict (False here).
    assert env.env_state["goal_satisfied"] is False
    _, done, info = env.step({"content": "", "tool_calls": [call("list_dir", path="/")]})
    assert done is True
    assert info["timeout"] is True
    assert env.timeout is True
    assert env.env_state["timeout"] is True
    # T4-01: no terminal verdict may survive a timeout.
    assert env.env_state["goal_satisfied"] is False
    assert info["env_state"]["goal_satisfied"] is False
    assert env.env_state["replay_ok"] is False
    # Diagnostic checks are preserved.
    assert env.env_state["checks"]
    assert env.trajectory()["done"] is True
    with pytest.raises(EnvError):
        env.step({"content": ""})


def test_timeout_does_not_promote_replay_ok() -> None:
    task = replay_task(gt_calls=[[call("get_weather", city="Beijing")]], max_turns=1)
    env = ReplayMockFCEnv(task)
    env.reset()
    _, done, info = env.step(
        {"content": "", "tool_calls": [call("get_weather", city="Beijing")]}
    )
    assert done is True
    assert info["timeout"] is True
    assert env.env_state["replay_ok"] is False


def test_curriculum_override_lowers_max_turns() -> None:
    task = stateful_task(max_turns=15)
    env = MockFCEnv(task)
    env.reset(task, max_turns=5)
    assert env.max_turns == 5
    env.reset(task, max_turns=10)
    assert env.max_turns == 10
    # env.config cap participates in the minimum as well
    task["env"]["config"]["max_turns"] = 3
    env.reset(task, max_turns=10)
    assert env.max_turns == 3


def test_final_answer_on_last_allowed_turn_is_not_timeout() -> None:
    task = stateful_task(max_turns=1)
    env = MockFCEnv(task)
    env.reset()
    _, done, info = env.step({"content": "done", "tool_calls": []})
    assert done is True
    assert info["timeout"] is False
    assert env.timeout is False


def test_reset_without_task_reuses_stored_task() -> None:
    task = stateful_task()
    env = MockFCEnv(task)
    env.reset()
    env.step({"content": "", "tool_calls": [call("touch", path="/a.txt")]})
    env.reset()  # task=None per CONTRACTS C1
    assert env.steps == 0
    assert env.env_state["state"]["files"] == {}
    assert env.task is task


def test_close_keeps_trajectory_and_reset_reopens() -> None:
    task = stateful_task()
    env = MockFCEnv(task)
    env.reset()
    env.step({"content": "", "tool_calls": [call("touch", path="/a.txt")]})
    env.close()
    assert env.closed is True
    # Trajectory stays readable for the adapter/reward after close.
    trajectory = env.trajectory()
    assert trajectory["num_turns"] == 1
    assert not schema.validate_trajectory(trajectory)
    with pytest.raises(EnvError):
        env.step({"content": ""})
    env.close()  # idempotent
    # T4-06: reset(task) reopens the instance with a clean state.
    env.reset(task)
    assert env.closed is False
    assert env.env_state["state"]["files"] == {}
    _, done, _ = env.step({"content": "done", "tool_calls": []})
    assert done is True


def test_env_state_shape_is_valid_and_checks_normalised() -> None:
    task = stateful_task(
        config={
            "goal": [
                {"name": "a", "check": "file_exists", "path": "/a.txt",
                 "detail": "x" * 2000},
            ]
        }
    )
    env = MockFCEnv(task)
    env.reset()
    env.step({"content": "done", "tool_calls": []})
    state = env.env_state
    assert not validate_env_state(state)
    for check in state["checks"]:
        assert set(check) == {"name", "ok", "detail"}
        assert isinstance(check["name"], str)
        assert isinstance(check["ok"], bool)
        assert isinstance(check["detail"], str)
        assert len(check["detail"]) <= 500

    normalised = normalise_checks([{"name": "n", "ok": True, "detail": None}, "junk"])
    assert normalised == [{"name": "n", "ok": True, "detail": ""}]
    bad = make_env_state("mock_fc", "stateful")
    bad["goal_satisfied"] = "yes"
    bad["checks"] = [{"name": 5}]
    errors = validate_env_state(bad)
    assert any("goal_satisfied" in error for error in errors)
    assert any("checks[0]" in error for error in errors)


def test_null_env_terminates_single_turn() -> None:
    task = make_task(task_type="st_fc", env_type="none", gt_returns_source=None)
    env = NullEnv(task)
    obs, done, info = env.step({"content": "", "tool_calls": [call("f", x=1)]})
    assert done is True
    assert obs["tool_results"][0]["ok"] is False
    assert info["final_answer"] is None
    env2 = NullEnv(make_task(task_type="abstain", env_type="none", gt_returns_source=None))
    _, done, info = env2.step({"content": "I cannot answer"})
    assert done is True
    assert info["final_answer"] == "I cannot answer"


def _filesystem_env(task_id: str, filename: str) -> MockFCEnv:
    task = stateful_task(
        task_id=task_id,
        config={
            "goal": [{"check": "file_exists", "path": filename}],
        },
    )
    return MockFCEnv(task)


def test_two_instances_do_not_share_state() -> None:
    env_a = _filesystem_env("env-a", "/a.txt")
    env_b = _filesystem_env("env-b", "/b.txt")
    env_a.reset()
    env_b.reset()
    env_a.step({"content": "", "tool_calls": [call("touch", path="/a.txt")]})
    env_b.step({"content": "", "tool_calls": [call("write_file", path="/b.txt", content="B")]})
    _, done_a, _ = env_a.step({"content": "done"})
    _, done_b, _ = env_b.step({"content": "done"})
    assert env_a.env_state["goal_satisfied"] is True
    assert env_b.env_state["goal_satisfied"] is True
    assert "/a.txt" in env_a.env_state["state"]["files"]
    assert "/b.txt" not in env_a.env_state["state"]["files"]
    assert "/b.txt" in env_b.env_state["state"]["files"]
    assert "/a.txt" not in env_b.env_state["state"]["files"]


def test_reset_clears_previous_task_state() -> None:
    env = _filesystem_env("first", "/a.txt")
    env.reset()
    env.step({"content": "", "tool_calls": [call("touch", path="/a.txt")]})
    second = stateful_task(task_id="second", config={"goal": []})
    env.reset(second)
    assert env.env_state["state"]["files"] == {}


def test_concurrent_instances_are_thread_isolated() -> None:
    def worker(filename: str) -> dict[str, Any]:
        env = _filesystem_env(filename, filename)
        env.reset()
        for _ in range(5):
            env.step(
                {"content": "", "tool_calls": [call("write_file", path=filename, content=filename)]}
            )
        env.step({"content": "done"})
        return env.env_state["state"]["files"]

    filenames = [f"/f{index}.txt" for index in range(6)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(worker, filenames))
    for filename, files in zip(filenames, results):
        assert list(files.keys()) == [filename]
        assert files[filename] == filename


# ---------------------------------------------------------------------------
# NullEnv / factory
# ---------------------------------------------------------------------------


def test_build_env_dispatch() -> None:
    assert isinstance(build_env(make_task(task_type="st_fc", env_type="none")), NullEnv)
    replay = build_env(replay_task(gt_calls=[[call("get_weather", city="X")]]))
    assert isinstance(replay, ReplayMockFCEnv)

    stateful = build_env(stateful_task())
    assert isinstance(stateful, MockFCEnv)
    assert stateful.impl == "stateful"

    search_task = make_task(
        task_type="search_qa",
        env_type="search",
        config={"backend": "inmemory", "docs": [{"doc_id": "1", "title": "T", "text": "x"}]},
    )
    assert isinstance(build_env(search_task), SearchEnv)

    python_task = make_task(task_type="math_ci", env_type="python", config={})
    assert isinstance(build_env(python_task), PythonEnv)

    code_task = make_task(
        task_type="code_exec",
        env_type="code_tests",
        config={},
        reference={"unit_tests": "x = 1"},
    )
    assert isinstance(build_env(code_task), CodeTestsEnv)


def test_obs_tool_results_align_with_trajectory() -> None:
    task = stateful_task()
    env = MockFCEnv(task)
    env.reset()
    obs, _, _ = env.step(
        {"content": "", "tool_calls": [call("touch", path="/a.txt"), call("list_dir", path="/")]}
    )
    trajectory = env.trajectory()
    assert len(obs["tool_results"]) == 2
    assert obs["tool_results"] == trajectory["turns"][0]["tool_results"]
    assert trajectory["turns"][0]["assistant"]["tool_calls"][0]["name"] == "touch"
    assert obs["tool_results"][0]["id"] == trajectory["turns"][0]["assistant"]["tool_calls"][0]["id"]
    assert_trajectory_valid(trajectory)
