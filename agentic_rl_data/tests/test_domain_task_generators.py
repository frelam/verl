from __future__ import annotations

import copy
import importlib
from collections import Counter, defaultdict
from typing import Any

import pytest

from agentic_rl_data.envs.base import build_env
from agentic_rl_data.envs.mock_fc import DOMAINS
from agentic_rl_data.rewards import dispatch_reward
from agentic_rl_data.schema import validate_task


_SOURCES = [
    ("agentic_rl_data.generators.vehicle_travel_tasks", "vehicle_travel_mt",
     {"vehicle", "travel", "trading"}),
    ("agentic_rl_data.generators.social_ops_tasks", "social_ops_mt",
     {"microblog", "messaging", "ticketing"}),
]


def _reply(answer: str | None) -> str:
    return f"All done. <answer>{answer}</answer>" if answer is not None else "All done."


def _replay(task: dict[str, Any], groups: list[list[dict[str, Any]]]) -> dict[str, Any]:
    env = build_env(task)
    env.reset(task)
    try:
        for group in groups:
            _, done, _ = env.step({"content": "", "tool_calls": copy.deepcopy(group)})
            if done:
                break
        if not env.done:
            env.step({"content": _reply(task["reference"]["final_answer"]), "tool_calls": []})
        return env.trajectory()
    finally:
        env.close()


@pytest.mark.parametrize(("module_name", "source", "domains"), _SOURCES)
def test_domain_generator_schema_determinism_and_real_reward(
    module_name: str, source: str, domains: set[str]
) -> None:
    module = importlib.import_module(module_name)
    stats: dict[str, Any] = {}
    first = list(module.generate_tasks(6, seed=29, stats=stats))
    second = list(module.generate_tasks(6, seed=29))
    assert first == second
    assert stats["emitted"] == 6
    assert stats["rejected"] == {}
    assert Counter(task["meta"]["domain"] for task in first) == Counter({domain: 2 for domain in domains})
    scenarios: dict[str, set[str]] = defaultdict(set)
    for task in first:
        domain = task["meta"]["domain"]
        scenarios[domain].add(task["meta"]["scenario"])
        assert validate_task(task) == []
        assert task["source"] == source
        assert task["task_type"] == "mt_fc"
        assert sum(message["role"] == "user" for message in task["messages"]) == 1
        assert task["meta"]["user_turn_count"] == 1
        assert task["meta"]["schema_style"] in {
            "canonical", "snake_prefix", "snake_suffix", "camelCase", "dotted",
        }
        assert task["meta"]["gt_returns_source"] == "config"
        assert task["env"]["config"]["process_term"] == "efficiency"
        assert task["env"]["config"]["domain"] == domain
        assert task["env"]["config"]["initial_state"]
        assert task["env"]["config"]["goal"]
        plan = task["reference"]["gt_calls"]
        assert 4 <= len(plan) + 1 <= 10
        assert task["meta"]["expected_turns"] == len(plan) + 1
        assert task["meta"]["parallel_group_count"] == sum(len(group) > 1 for group in plan)
        tool_names = [tool["name"] for tool in task["tools"]]
        assert set(DOMAINS[domain].HANDLERS) <= set(task["meta"]["canonical_tools"])
        assert len(tool_names) == task["meta"]["tool_count_target"]
        assert len(tool_names) <= 20
        aliases = task["env"]["config"]["tool_aliases"]
        if aliases:
            assert set(task["meta"]["canonical_tools"]) == set(aliases.values())
        else:
            assert task["meta"]["canonical_tools"] == tool_names
        for tool in task["tools"]:
            assert tool["description"]
            parameters = tool["parameters"]
            assert parameters["type"] == "object"
            assert set(parameters.get("required", [])) <= set(parameters["properties"])
        assert {call["name"] for group in plan for call in group} <= set(tool_names)
        good = _replay(task, plan)
        assert (good["env_state"] or {})["goal_satisfied"] is True
        assert dispatch_reward(task, good)["score"] == pytest.approx(1.0)
        idle = _replay(task, [])
        assert (idle["env_state"] or {})["goal_satisfied"] is False
        assert dispatch_reward(task, idle)["score"] == 0.0
    assert all(len(scenarios[domain]) == 2 for domain in domains)
