"""Interleaved-user, abstention and parallel gates for stateful mock_fc."""

from __future__ import annotations

import copy
from typing import Any

from agentic_rl_data.envs.base import build_env
from agentic_rl_data.generators.stateful_tasks import FS_TOOLS
from agentic_rl_data.rewards import dispatch_reward


def _task(config: dict[str, Any], *, gt_calls: list | None = None) -> dict[str, Any]:
    return {
        "id": "dialogue-test",
        "source": "unit_test",
        "task_type": "mt_fc",
        "messages": [{"role": "user", "content": "Start."}],
        "tools": copy.deepcopy(FS_TOOLS),
        "env": {
            "type": "mock_fc",
            "config": {
                "impl": "stateful",
                "domain": "filesystem",
                "initial_state": {"files": {}, "dirs": ["/"]},
                "goal": [],
                "max_turns": 12,
                "process_term": "efficiency",
                **config,
            },
        },
        "reference": {
            "final_answer": None,
            "final_answers": None,
            "gt_calls": gt_calls or [],
            "gt_turns": None,
            "tool_returns": None,
            "unit_tests": None,
            "final_state": None,
        },
        "meta": {
            "max_turns": 12,
            "expected_turns": 4,
            "difficulty": 0.0,
            "pass_rate": None,
            "license": "test",
            "gt_returns_source": "config",
        },
    }


def _call(name: str, **arguments: Any) -> dict[str, Any]:
    return {"name": name, "arguments": arguments}


def _step(env, content: str = "", calls: list | None = None):
    return env.step({"content": content, "tool_calls": calls or []})


def test_interleaved_user_preserves_state_and_rewards_outcome() -> None:
    first_goal = [{"check": "file_content", "path": "/a.txt", "expected": "alpha", "name": "a"}]
    final_goal = [
        {"check": "file_content", "path": "/b.txt", "expected": "alpha", "name": "b"},
        {"check": "file_not_exists", "path": "/a.txt", "name": "a_removed"},
    ]
    task = _task(
        {
            "dialogue": {
                "stages": [
                    {
                        "kind": "execute",
                        "goal": first_goal,
                        "required_tools": ["write_file"],
                        "must_change": first_goal,
                        "reply": "none",
                        "next_user": "Now move it to /b.txt.",
                    },
                    {
                        "kind": "execute",
                        "goal": final_goal,
                        "required_tools": ["move_file"],
                        "must_change": final_goal,
                        "reply": "none",
                    },
                ]
            }
        }
    )
    env = build_env(task)
    env.reset(task)
    _step(env, calls=[_call("write_file", path="/a.txt", content="alpha")])
    obs, done, info = _step(env, "First step done.")
    assert done is False
    assert info["next_user_message"] == "Now move it to /b.txt."
    assert env.final_answer is None
    _step(env, calls=[_call("move_file", source="/a.txt", destination="/b.txt")])
    _, done, info = _step(env, "All done.")
    assert done is True
    assert info["env_state"]["goal_satisfied"] is True
    assert info["env_state"]["extras"]["dialogue"]["complete"] is True
    assert dispatch_reward(task, env.trajectory())["score"] == 1.0


def test_premature_reply_fails_instead_of_releasing_user() -> None:
    goal = [{"check": "file_content", "path": "/a.txt", "expected": "alpha", "name": "a"}]
    task = _task(
        {
            "strict_outcome": True,
            "dialogue": {
                "stages": [
                    {
                        "kind": "execute",
                        "goal": goal,
                        "required_tools": ["write_file"],
                        "must_change": goal,
                        "reply": "none",
                        "next_user": "This must never leak.",
                    }
                ]
            },
        }
    )
    env = build_env(task)
    env.reset(task)
    _, done, info = _step(env, "I will do it later.")
    assert done is True
    assert "next_user_message" not in info
    assert info["env_state"]["goal_satisfied"] is False
    assert dispatch_reward(task, env.trajectory())["score"] == 0.0


def test_clarification_requires_slot_and_forbids_calls() -> None:
    goal = [{"check": "file_content", "path": "/plan.txt", "expected": "friday", "name": "plan"}]
    task = _task(
        {
            "strict_outcome": True,
            "dialogue": {
                "stages": [
                    {
                        "kind": "clarify",
                        "goal": [],
                        "reply": "request_info",
                        "slot_keywords": [["date", "day"]],
                        "allow_tools": False,
                        "next_user": "Use Friday.",
                    },
                    {
                        "kind": "execute",
                        "goal": goal,
                        "required_tools": ["write_file"],
                        "must_change": goal,
                        "reply": "none",
                    },
                ]
            },
        }
    )
    env = build_env(task)
    env.reset(task)
    _, done, info = _step(env, "Which date should I use?")
    assert done is False
    assert info["next_user_message"] == "Use Friday."
    _step(env, calls=[_call("write_file", path="/plan.txt", content="friday")])
    _, done, info = _step(env, "Done.")
    assert done is True
    assert info["env_state"]["goal_satisfied"] is True

    env.reset(task)
    _, done, info = _step(env, "Sure, I can help.")
    assert done is True
    assert info["env_state"]["goal_satisfied"] is False

    env.reset(task)
    _step(env, calls=[_call("write_file", path="/plan.txt", content="guessed")])
    _, done, info = _step(env, "Which date should I use?")
    assert done is True
    assert info["env_state"]["goal_satisfied"] is False
    assert info["env_state"]["extras"]["dialogue"]["violations"]


def test_missing_capability_requires_refusal_without_calls() -> None:
    task = _task(
        {
            "strict_outcome": True,
            "dialogue": {
                "stages": [
                    {
                        "kind": "abstain",
                        "goal": [],
                        "reply": "no_valid_tools",
                        "allow_tools": False,
                    }
                ]
            },
        }
    )
    env = build_env(task)
    env.reset(task)
    _, done, info = _step(env, "I have no suitable tools for that.")
    assert done is True
    assert info["env_state"]["goal_satisfied"] is True


def test_parallel_requirement_rejects_serialising_the_same_writes() -> None:
    calls = [
        _call("write_file", path="/a.txt", content="a"),
        _call("write_file", path="/b.txt", content="b"),
    ]
    goal = [
        {"check": "file_content", "path": "/a.txt", "expected": "a", "name": "a"},
        {"check": "file_content", "path": "/b.txt", "expected": "b", "name": "b"},
    ]
    task = _task({"goal": goal, "required_parallel": [calls]})

    parallel = build_env(task)
    parallel.reset(task)
    _step(parallel, calls=calls)
    _, done, info = _step(parallel, "Done.")
    assert done is True
    assert info["env_state"]["goal_satisfied"] is True

    serial = build_env(task)
    serial.reset(task)
    for call in calls:
        _step(serial, calls=[call])
    _, done, info = _step(serial, "Done.")
    assert done is True
    assert info["env_state"]["goal_satisfied"] is False
    parallel_check = next(c for c in info["env_state"]["checks"] if c["name"] == "required_parallel")
    assert parallel_check["ok"] is False


def test_tool_and_parameter_aliases_with_nested_payload() -> None:
    goal = [{"check": "file_content", "path": "/a.txt", "expected": "alpha", "name": "a"}]
    task = _task(
        {
            "goal": goal,
            "tool_aliases": {"file.write": "write_file"},
            "param_aliases": {"file.write": {"target": "path", "text": "content"}},
            "param_wrappers": {"file.write": ["payload"]},
        }
    )
    task["tools"] = [
        {
            "name": "file.write",
            "description": "Write one file.",
            "parameters": {"type": "object", "properties": {"payload": {"type": "object"}}},
        }
    ]
    env = build_env(task)
    env.reset(task)
    _step(env, calls=[_call("file.write", payload={"target": "/a.txt", "text": "alpha"})])
    _, done, info = _step(env, "Done.")
    assert done is True
    assert info["env_state"]["goal_satisfied"] is True
