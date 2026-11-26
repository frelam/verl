from __future__ import annotations

import copy
import json
from collections import Counter

from agentic_rl_data.generators import procedural_st_tasks as generator
from agentic_rl_data.loaders import procedural_st as loader
from agentic_rl_data.rewards import st_fc
from agentic_rl_data.schema import validate_task


def _trajectory(calls: list[dict]) -> dict:
    return {
        "turns": [
            {
                "step": 1,
                "assistant": {"content": "" if calls else "Done.", "tool_calls": copy.deepcopy(calls)},
                "tool_results": [],
            }
        ],
        "final_answer": None,
        "num_turns": 1,
        "done": True,
        "env_state": None,
    }


def test_small_generation_schema_and_all_scenarios() -> None:
    tasks = list(generator.generate_tasks(40, seed=17))
    assert len(tasks) == 40
    counts = Counter(task["meta"]["scenario"] for task in tasks)
    assert counts == {
        "multiple": 8,
        "parallel": 8,
        "parallel_multiple": 8,
        "parameter_precision": 16,
    }
    assert {task["meta"]["naming_style"] for task in tasks} == set(generator.NAMING_STYLES)
    assert {task["meta"]["description_style"] for task in tasks} == set(generator.DESCRIPTION_STYLES)

    for task in tasks:
        assert validate_task(task) == []
        assert task["source"] == generator.SOURCE
        assert task["task_type"] == "st_fc"
        assert task["env"] == {"type": "none", "config": {"max_turns": 1}}
        assert task["meta"]["max_turns"] == 1
        assert len(task["reference"]["gt_calls"]) == 1
        calls = task["reference"]["gt_calls"][0]
        assert calls
        assert generator.MIN_TOOLS <= len(task["tools"]) <= generator.MAX_TOOLS
        assert len(task["tools"]) > len({call["name"] for call in calls})
        assert {call["name"] for call in calls} <= {tool["name"] for tool in task["tools"]}
        assert all(isinstance(call["arguments"], dict) for call in calls)
        assert "tool_aliases" not in task["env"]["config"]
        json.dumps(task, ensure_ascii=False, sort_keys=True)
        assert generator.verify_task(task)["score"] == 1.0

        names = [call["name"] for call in calls]
        scenario = task["meta"]["scenario"]
        if scenario == "multiple":
            assert 2 <= len(calls) <= 4
            assert len(set(names)) == len(calls)
        elif scenario == "parallel":
            assert 2 <= len(calls) <= 4
            assert len(set(names)) == 1
            assert len({json.dumps(call["arguments"], sort_keys=True) for call in calls}) == len(calls)
        elif scenario == "parallel_multiple":
            assert 2 <= len(calls) <= 4
            assert 1 < len(set(names)) < len(calls)
        else:
            assert len(calls) == 1


def test_parameter_precision_schema_and_arguments() -> None:
    tasks = [
        task
        for task in generator.generate_tasks(24, seed=8)
        if task["meta"]["scenario"] == "parameter_precision"
    ]
    assert tasks
    assert {task["meta"]["optional_policy"] for task in tasks} == {"include", "omit"}
    for task in tasks:
        call = task["reference"]["gt_calls"][0][0]
        tool = next(tool for tool in task["tools"] if tool["name"] == call["name"])
        properties = tool["parameters"]["properties"]
        payload = properties["payload"]
        records = payload["properties"]["records"]
        assert payload["type"] == "object"
        assert records["type"] == "array"
        assert records["items"]["type"] == "object"
        assert payload["properties"]["mode"]["enum"]
        assert payload["properties"]["unit"]["enum"]
        assert payload["properties"]["amount"]["type"] == "number"
        assert properties["temperature_c"]["type"] == "number"
        assert "operator_note" not in tool["parameters"]["required"]
        assert properties["operator_note"]["default"] == "none"
        assert properties["retry_limit"]["default"] == 0

        arguments = call["arguments"]
        assert isinstance(arguments["payload"]["reference"], str)
        assert 2 <= len(arguments["payload"]["records"]) <= 3
        assert all(isinstance(record, dict) for record in arguments["payload"]["records"])
        if task["meta"]["optional_policy"] == "include":
            assert {"operator_note", "retry_limit"} <= set(arguments)
        else:
            assert {"operator_note", "retry_limit"}.isdisjoint(arguments)


def test_fixed_seed_is_deterministic() -> None:
    first = list(generator.generate_tasks(16, seed=123))
    second = list(generator.generate_tasks(16, seed=123))
    different = list(generator.generate_tasks(16, seed=124))
    assert first == second
    assert different != first
    assert any(
        left["reference"]["gt_calls"] != right["reference"]["gt_calls"]
        for left, right in zip(first, different)
    )


def test_thin_loader_generates_without_registry_changes() -> None:
    tasks = list(loader.load_tasks({"n": 7, "seed": 31}))
    assert loader.SOURCE == "procedural_st"
    assert len(tasks) == 7
    assert all(validate_task(task) == [] for task in tasks)
    assert all(task["task_type"] == "st_fc" for task in tasks)


def test_st_fc_reward_rejects_missing_extra_wrong_value_and_wrong_type() -> None:
    task = next(
        task
        for task in generator.generate_tasks(20, seed=4)
        if task["meta"]["scenario"] == "parameter_precision"
    )
    calls = task["reference"]["gt_calls"][0]
    assert st_fc.reward(task, _trajectory(calls))["score"] == 1.0
    assert st_fc.reward(task, _trajectory(calls[:-1]))["score"] == 0.0

    extra = copy.deepcopy(calls) + [copy.deepcopy(calls[0])]
    assert st_fc.reward(task, _trajectory(extra))["score"] == 0.0

    wrong_value = copy.deepcopy(calls)
    wrong_value[0]["arguments"]["request_tag"] += "-incorrect"
    assert st_fc.reward(task, _trajectory(wrong_value))["score"] == 0.0

    wrong_type = copy.deepcopy(calls)
    wrong_type[0]["arguments"]["request_tag"] = {"not": "a string"}
    assert st_fc.reward(task, _trajectory(wrong_type))["score"] == 0.0
