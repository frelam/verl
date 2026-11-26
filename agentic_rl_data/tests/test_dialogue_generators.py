"""P1/P2 generator contracts and real-env behavioural checks."""

from __future__ import annotations

import copy
from collections import Counter

import pytest

from agentic_rl_data import schema
from agentic_rl_data.envs.base import build_env
from agentic_rl_data.generators import abstain_tasks, interleaved_tasks
from agentic_rl_data.generators.schema_styles import DESCRIPTION_STYLES, NAME_STYLES
from agentic_rl_data.loaders import interleaved_mt, stateful_abstain
from agentic_rl_data.rewards import dispatch_reward

_STAGE_KEYS = {
    "kind",
    "goal",
    "invariants",
    "required_tools",
    "must_change",
    "reply",
    "slot_keywords",
    "next_user",
}


@pytest.fixture(scope="module")
def interleaved():
    return list(interleaved_tasks.generate_tasks(12, seed=7, verify=True))


@pytest.fixture(scope="module")
def abstentions():
    return list(abstain_tasks.generate_tasks(9, seed=5, fault_prob=1 / 3, schema_prob=1.0, verify=True))


def _replay(task):
    env = build_env(task)
    env.reset(task)
    try:
        for index, (group, turn) in enumerate(zip(task["reference"]["gt_calls"], task["reference"]["gt_turns"])):
            _, done, info = env.step({"content": turn["content"], "tool_calls": copy.deepcopy(group)})
            assert info.get("next_user_message") == turn.get("next_user"), (task["id"], index)
            if done:
                assert index == len(task["reference"]["gt_calls"]) - 1
                break
        trajectory = env.trajectory()
        assert trajectory["done"] is True
        return trajectory, dispatch_reward(task, trajectory)
    finally:
        env.close()


def _idle(task):
    env = build_env(task)
    env.reset(task)
    try:
        env.step({"content": "All done.", "tool_calls": []})
        return dispatch_reward(task, env.trajectory())
    finally:
        env.close()


def _without_retry(task):
    retry = task["meta"]["retry_groups"][0]
    env = build_env(task)
    env.reset(task)
    try:
        for index, (group, turn) in enumerate(zip(task["reference"]["gt_calls"], task["reference"]["gt_turns"])):
            if index == retry or env.done:
                continue
            env.step({"content": turn["content"], "tool_calls": copy.deepcopy(group)})
        if not env.done:
            env.step({"content": "All done.", "tool_calls": []})
        return dispatch_reward(task, env.trajectory())
    finally:
        env.close()


def test_interleaved_determinism_quotas_and_schema(interleaved):
    again = list(interleaved_tasks.generate_tasks(12, seed=7, verify=False))
    assert again == interleaved
    assert Counter(task["meta"]["domain"] for task in interleaved) == {"filesystem": 4, "trading": 4, "retail": 4}
    assert {task["meta"]["user_turn_count"] for task in interleaved} == {3, 4, 5, 6}
    assert sum(task["meta"]["n_faults"] for task in interleaved) == 3
    for task in interleaved:
        assert schema.validate_task(task) == []
        assert task["task_type"] == "mt_fc"
        assert task["env"]["type"] == "mock_fc"
        assert task["env"]["config"]["impl"] == "stateful"
        assert task["env"]["config"]["process_term"] == "efficiency"
        assert 0 < task["meta"]["expected_turns"] <= task["meta"]["max_turns"] <= 24
        assert task["meta"]["gt_returns_source"] == "config"
        assert task["meta"]["parallel_group_count"] == sum(
            len(group) > 1 for group in task["reference"]["gt_calls"]
        )


def test_future_users_never_leak_and_stage_actions_align(interleaved, abstentions):
    for task in interleaved + abstentions:
        assert [message["role"] for message in task["messages"]] == ["system", "user"]
        initial_prompt = "\n".join(message["content"] for message in task["messages"])
        stages = task["env"]["config"]["dialogue"]["stages"]
        assert len(stages) == task["meta"]["user_turn_count"]
        groups = task["reference"]["gt_calls"]
        turns = task["reference"]["gt_turns"]
        assert len(groups) == len(turns) == task["meta"]["expected_turns"]
        for index, stage in enumerate(stages):
            assert _STAGE_KEYS <= set(stage)
            next_user = stage["next_user"]
            if index + 1 < len(stages):
                assert isinstance(next_user, str) and next_user.strip()
                assert next_user not in initial_prompt
                assert sum(turn.get("next_user") == next_user for turn in turns) == 1
            else:
                assert next_user is None
        for group, turn in zip(groups, turns):
            if group:
                assert turn["content"] == ""
                assert turn["next_user"] is None
            else:
                assert isinstance(turn["content"], str) and turn["content"]


def test_interleaved_contains_append_correction_and_goal_change(interleaved):
    for task in interleaved:
        kinds = {stage["kind"] for stage in task["env"]["config"]["dialogue"]["stages"]}
        assert {"append", "correction", "change_goal"} <= kinds
        for stage in task["env"]["config"]["dialogue"]["stages"]:
            assert stage["required_tools"]
            assert stage["must_change"]


def test_abstention_classes_and_protocol(abstentions):
    assert Counter(task["meta"]["scenario"] for task in abstentions) == {
        "miss_func": 3,
        "miss_param": 3,
        "inferable": 3,
    }
    assert sum(task["meta"]["n_faults"] for task in abstentions) == 3
    for task in abstentions:
        assert schema.validate_task(task) == []
        assert task["env"]["config"]["strict_outcome"] is True
        kind = task["meta"]["scenario"]
        stages = task["env"]["config"]["dialogue"]["stages"]
        groups = task["reference"]["gt_calls"]
        if kind == "miss_func":
            assert task["meta"]["user_turn_count"] == 1
            assert groups == [[]]
            assert stages[0]["reply"] == "no_valid_tools"
            assert stages[0]["allow_tools"] is False
            assert task["meta"]["n_faults"] == 0
        elif kind == "miss_param":
            assert task["meta"]["user_turn_count"] == 2
            assert groups[0] == []
            assert stages[0]["reply"] == "request_info"
            assert stages[0]["slot_keywords"]
            assert stages[0]["next_user"] == task["reference"]["gt_turns"][0]["next_user"]
            assert stages[0]["allow_tools"] is False
            assert any(groups[1:])
        else:
            assert task["meta"]["user_turn_count"] == 1
            assert groups[0]
            assert stages[0]["reply"] == "none"


def test_schema_overlay_is_synchronised_and_keeps_anchors(interleaved, abstentions):
    tasks = interleaved + abstentions
    assert {task["meta"]["schema_style"] for task in tasks} >= set(NAME_STYLES)
    assert {task["meta"]["description_style"] for task in tasks if task["meta"]["schema_style"] != "canonical"} >= set(
        DESCRIPTION_STYLES
    )
    modes = {
        mode
        for task in tasks
        for mode in task["meta"].get("parameter_styles", {}).values()
    }
    assert {"canonical", "alias", "nested"} <= modes
    for task in tasks:
        declared = {tool["name"] for tool in task["tools"]}
        config = task["env"]["config"]
        assert declared == set(config["tool_aliases"])
        assert set(task["meta"]["canonical_tools"]) == set(config["tool_aliases"].values())
        if task["meta"]["schema_style"] != "canonical":
            assert task["meta"]["schema_anchor_tools"]
        for group in task["reference"]["gt_calls"]:
            assert {call["name"] for call in group} <= declared
        assert {fault["tool"] for fault in config.get("faults", [])} <= declared
        for stage in config["dialogue"]["stages"]:
            assert set(stage["required_tools"]) <= declared
        for tool_name, wrappers in config.get("param_wrappers", {}).items():
            assert tool_name in declared
            schema_properties = next(tool for tool in task["tools"] if tool["name"] == tool_name)["parameters"]["properties"]
            assert set(wrappers) <= set(schema_properties)


def test_real_env_gt_scores_one_and_idle_zero(interleaved, abstentions):
    selected = []
    for domain in interleaved_tasks.DOMAINS:
        selected.append(next(task for task in interleaved if task["meta"]["domain"] == domain))
    for kind in abstain_tasks.KINDS:
        selected.append(next(task for task in abstentions if task["meta"]["scenario"] == kind))
    for task in selected:
        trajectory, reward = _replay(task)
        assert trajectory["env_state"]["goal_satisfied"] is True, task["id"]
        assert reward["score"] == pytest.approx(1.0), (task["id"], reward)
        assert _idle(task)["score"] == 0.0, task["id"]


def test_abstention_negative_behaviours_fail_closed(abstentions):
    missing_function = next(task for task in abstentions if task["meta"]["scenario"] == "miss_func")
    env = build_env(missing_function)
    env.reset(missing_function)
    try:
        env.step({"content": "", "tool_calls": [{"name": missing_function["tools"][0]["name"], "arguments": {}}]})
        if not env.done:
            env.step({"content": missing_function["reference"]["gt_turns"][0]["content"], "tool_calls": []})
        assert dispatch_reward(missing_function, env.trajectory())["score"] == 0.0
    finally:
        env.close()

    missing_param = next(task for task in abstentions if task["meta"]["scenario"] == "miss_param")
    first_call = next(copy.deepcopy(group) for group in missing_param["reference"]["gt_calls"] if group)
    env = build_env(missing_param)
    env.reset(missing_param)
    try:
        env.step({"content": "", "tool_calls": first_call})
        if not env.done:
            env.step({"content": "All done.", "tool_calls": []})
        assert dispatch_reward(missing_param, env.trajectory())["score"] == 0.0
    finally:
        env.close()

    inferable = next(task for task in abstentions if task["meta"]["scenario"] == "inferable")
    env = build_env(inferable)
    env.reset(inferable)
    try:
        env.step({"content": "Could you please clarify the missing details?", "tool_calls": []})
        assert dispatch_reward(inferable, env.trajectory())["score"] == 0.0
    finally:
        env.close()


def test_dropping_fault_retry_never_earns_full_reward(interleaved, abstentions):
    for task in interleaved + abstentions:
        if task["meta"]["n_faults"]:
            assert _without_retry(task)["score"] < 1.0, task["id"]


def test_thin_loaders_generate_without_raw_input():
    first = list(interleaved_mt.load_tasks({"n": 2, "seed": 2, "verify": False}))
    second = list(stateful_abstain.load_tasks({"n": 3, "seed": 2, "verify": False}))
    assert len(first) == 2
    assert len(second) == 3
    assert all(task["source"] == "interleaved_mt" for task in first)
    assert all(task["source"] == "stateful_abstain" for task in second)
