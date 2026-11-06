"""v2 agentic RL: procedural stateful tasks, fault injection, efficiency term,
multi-hop local search, and the reward-hacking probes they must resist.

Every test drives the *real* env + reward (no mocked env_state), so a
regression in either side shows up here.
"""

from __future__ import annotations

import copy
import random
from collections import Counter

import pytest

from agentic_rl_data import schema
from agentic_rl_data.envs.base import build_env
from agentic_rl_data.envs.mock_fc import DEFAULT_FAULT_MESSAGE, answer_tag_matches
from agentic_rl_data.envs.search import SEARCH_HARD_MAX_ROUNDS
from agentic_rl_data.generators import DOMAIN_GENERATORS, generate_tasks, verify_task
from agentic_rl_data.generators.stateful_tasks import to_task
from agentic_rl_data.loaders import load_source
from agentic_rl_data.rewards import dispatch_reward
from agentic_rl_data.rewards.mt_fc import process_mode


def _run(task, groups, reply="All done.", *, max_steps=None):
    env = build_env(task)
    env.reset(task)
    for group in groups:
        if env.done:
            break
        env.step({"content": "", "tool_calls": copy.deepcopy(group)})
    if not env.done:
        env.step({"content": reply, "tool_calls": []})
    trajectory = env.trajectory()
    env.close()
    return trajectory, dispatch_reward(task, trajectory)


def _reply(task):
    answer = task["reference"]["final_answer"]
    return f"Done. <answer>{answer}</answer>" if answer is not None else "Done."


@pytest.fixture(scope="module")
def tasks():
    return list(generate_tasks(240, seed=11))


# ---------------------------------------------------------------------------
# generator: validity, determinism, coverage
# ---------------------------------------------------------------------------


def test_generated_tasks_are_schema_valid_and_deterministic(tasks):
    assert all(schema.validate_task(task) == [] for task in tasks)
    again = list(generate_tasks(240, seed=11))
    assert [t["id"] for t in again] == [t["id"] for t in tasks]
    assert again[17] == tasks[17]


def test_generator_covers_every_template_and_long_horizons():
    sample = list(generate_tasks(600, seed=5))
    templates = {t["meta"]["template"] for t in sample}
    expected = {fn.__name__.split("_", 1)[1] for fns in DOMAIN_GENERATORS.values() for fn, _ in fns}
    assert templates == expected
    turns = Counter(t["meta"]["expected_turns"] for t in sample)
    assert max(turns) >= 12, turns
    assert sum(v for k, v in turns.items() if k >= 10) / len(sample) >= 0.15
    assert all(t["meta"]["max_turns"] <= 15 for t in sample)
    assert 0.15 <= sum(1 for t in sample if t["meta"]["n_faults"]) / len(sample) <= 0.45


def test_gt_plan_scores_full_and_idle_scores_zero(tasks):
    for task in tasks[:60]:
        _, good = _run(task, task["reference"]["gt_calls"], _reply(task))
        assert good["score"] == pytest.approx(1.0), (task["id"], good["info"]["result_detail"])
        _, idle = _run(task, [], _reply(task))
        # no tool call on a tool task: result 0, efficiency 0, format zeroed
        assert idle["score"] == 0.0, task["id"]
        assert idle["info"]["idle_without_calls"] is True


def test_stateful_tasks_use_efficiency_term(tasks):
    assert all(process_mode(task) == "efficiency" for task in tasks)


# ---------------------------------------------------------------------------
# fault injection = error recovery
# ---------------------------------------------------------------------------


def _fault_task(tasks):
    return next(t for t in tasks if t["meta"]["n_faults"])


def test_injected_fault_returns_error_and_leaves_state_untouched(tasks):
    task = _fault_task(tasks)
    tool = task["env"]["config"]["faults"][0]["tool"]
    plan = task["reference"]["gt_calls"]
    first = next(i for i, g in enumerate(plan) if g[0]["name"] == tool)
    env = build_env(task)
    env.reset(task)
    for group in plan[:first]:
        env.step({"content": "", "tool_calls": copy.deepcopy(group)})
    before = copy.deepcopy(env.env_state["state"])
    obs, done, _ = env.step({"content": "", "tool_calls": copy.deepcopy(plan[first])})
    assert done is False
    assert obs["tool_results"][0]["ok"] is False
    assert obs["tool_results"][0]["content"] == DEFAULT_FAULT_MESSAGE
    assert env.env_state["state"] == before
    obs, _, _ = env.step({"content": "", "tool_calls": copy.deepcopy(plan[first])})
    assert obs["tool_results"][0]["ok"] is True
    assert tool in env.env_state["extras"]["recovered_tools"]
    env.close()


def test_skipping_the_retry_fails_the_task(tasks):
    task = _fault_task(tasks)
    tool = task["env"]["config"]["faults"][0]["tool"]
    plan = task["reference"]["gt_calls"]
    first = next(i for i, g in enumerate(plan) if g[0]["name"] == tool)
    no_retry = plan[: first + 1] + plan[first + 2 :]
    _, result = _run(task, no_retry, _reply(task))
    assert result["info"]["result"] == 0.0
    assert result["score"] <= 0.1 + 1e-9


# ---------------------------------------------------------------------------
# reward hacking probes
# ---------------------------------------------------------------------------


def _answer_task(tasks):
    return next(t for t in tasks if t["meta"]["requires_answer"] and not t["meta"]["n_faults"])


def test_missing_or_wrong_answer_fails_answer_tasks(tasks):
    task = _answer_task(tasks)
    plan = task["reference"]["gt_calls"]
    assert _run(task, plan, "Done.")[1]["info"]["result"] == 0.0
    assert _run(task, plan, "<answer>-1</answer>")[1]["info"]["result"] == 0.0
    gold = task["reference"]["final_answer"]
    # answer stuffing outside the tag does not count
    assert _run(task, plan, f"maybe {gold} or 0 or 1")[1]["info"]["result"] == 0.0


def test_answer_tag_matching_is_strict():
    assert answer_tag_matches("x <answer>12.50</answer>", "12.5")
    assert answer_tag_matches("<answer>$1,234.00</answer>", "1234.00")
    assert not answer_tag_matches("<answer>12.51</answer>", "12.5")
    assert not answer_tag_matches("<answer>12.5 or 13</answer>", "12.5")
    assert not answer_tag_matches("12.5", "12.5")
    assert answer_tag_matches("<answer>a</answer> <answer>Paris</answer>", "paris")


def test_touching_decoys_breaks_the_goal():
    rng = random.Random(0)
    gen = DOMAIN_GENERATORS["retail"][0][0](rng, 2)  # retail_cancel
    task = to_task(gen, task_id="probe-cancel", level=2, slack=3)
    verify_task(task)
    orders = gen.initial_state["orders"]
    # "cancel everything pending" must fail: only the named order may change
    greedy = [g for g in task["reference"]["gt_calls"] if g[0]["name"] != "cancel_pending_order"]
    reason = task["reference"]["gt_calls"][-1][0]["arguments"]["reason"]
    greedy += [
        [{"name": "cancel_pending_order", "arguments": {"order_id": o, "reason": reason}}]
        for o, v in orders.items()
        if v["status"] == "pending"
    ]
    n_pending = sum(1 for v in orders.values() if v["status"] == "pending")
    _, result = _run(task, greedy)
    if n_pending > 1:
        assert result["info"]["result"] == 0.0


def test_undeclared_tool_call_is_rejected_by_env(tasks):
    task = tasks[0]
    env = build_env(task)
    env.reset(task)
    obs, _, _ = env.step({"content": "", "tool_calls": [{"name": "rm_rf", "arguments": {}}]})
    assert obs["tool_results"][0]["ok"] is False
    assert env.env_state["extras"]["invalid_calls"] == 1
    env.close()


def test_efficiency_term_decays_and_is_success_only(tasks):
    task = next(t for t in tasks if not t["meta"]["n_faults"] and t["meta"]["max_turns"] >= 10)
    plan = task["reference"]["gt_calls"]
    _, base = _run(task, plan, _reply(task))
    # pad with harmless reads (first read call repeated) up to max_turns - 1
    read = plan[0]
    pad = task["meta"]["max_turns"] - len(plan) - 1
    _, padded = _run(task, [read] * pad + plan, _reply(task))
    assert base["score"] == pytest.approx(1.0)
    if padded["info"]["result"] == 1.0 and pad > 2:
        assert padded["info"]["turn_match"] < 1.0
        assert padded["score"] < base["score"]
    # success-only: a failed attempt never gets the efficiency term
    _, failed = _run(task, plan[:1], _reply(task))
    assert failed["info"]["turn_match"] == 0.0


def test_timeout_never_pays_result(tasks):
    task = tasks[3]
    read = task["reference"]["gt_calls"][0]
    trajectory, result = _run(task, [read] * 20)
    assert trajectory["env_state"]["timeout"] is True
    assert result["info"]["result"] == 0.0


# ---------------------------------------------------------------------------
# multi-hop local search
# ---------------------------------------------------------------------------


def _hotpot_record(i=0):
    return {
        "_id": f"q{i}",
        "question": f"Where was the director of Film {i} born?",
        "answer": f"Townville{i}",
        "context": [
            [f"Film {i}", [f"Film {i} is a drama directed by Person{i}."]],
            [f"Person{i}", [f"Person{i} was born in Townville{i}."]],
            ["Noise", ["Unrelated text."]],
        ],
        "supporting_facts": [[f"Film {i}", 0], [f"Person{i}", 0]],
    }


def test_multihop_loader_builds_self_contained_search_tasks():
    task = next(iter(load_source("multihop_local", {"records": [_hotpot_record()]})))
    assert schema.validate_task(task) == []
    config = task["env"]["config"]
    assert config["backend"] == "inmemory" and len(config["docs"]) == 3
    assert task["meta"]["hops"] == 2
    env = build_env(task)
    env.reset(task)
    assert env.max_turns == 10
    env.step({"content": "", "tool_calls": [{"name": "search", "arguments": {"query": "Film 0 director"}}]})
    env.step({"content": "", "tool_calls": [{"name": "search", "arguments": {"query": "Person0 born"}}]})
    env.step({"content": "<answer>Townville0</answer>", "tool_calls": []})
    assert dispatch_reward(task, env.trajectory())["score"] == pytest.approx(1.0)
    env.close()


def test_multihop_loader_skips_yes_no_and_unanswerable():
    yes_no = dict(_hotpot_record(1), answer="yes")
    unanswerable = {"id": "m1", "question": "q?", "answer": "x", "answerable": False,
                    "paragraphs": [{"title": "t", "paragraph_text": "p", "is_supporting": True}]}
    assert list(load_source("multihop_local", {"records": [yes_no, unanswerable]})) == []


def test_search_rounds_capped_at_hard_max():
    task = next(iter(load_source("multihop_local", {"records": [_hotpot_record()], "max_turns": 30})))
    env = build_env(task)
    env.reset(task)
    assert env.max_turns == SEARCH_HARD_MAX_ROUNDS
    env.close()


# ---------------------------------------------------------------------------
# agent-loop helpers (no verl needed)
# ---------------------------------------------------------------------------


def test_broken_tool_call_detection_and_format_issue():
    pytest.importorskip("verl")
    from agentic_rl_data.verl.agent_loop_agentic import _looks_like_broken_tool_call, format_error_message

    assert _looks_like_broken_tool_call('<tool_call>{"name": "x", "arguments": {oops}</tool_call>')
    assert not _looks_like_broken_tool_call("The answer is 4.")
    assert format_error_message(["bad json"]).startswith("Error: your tool call could not be parsed")


def test_unparsed_tool_call_markup_is_a_format_issue(tasks):
    from agentic_rl_data.rewards import base

    trajectory = {
        "turns": [{"step": 1, "assistant": {"content": "<tool_call>{bad", "tool_calls": []}, "tool_results": []}],
        "final_answer": None, "num_turns": 1, "done": False, "env_state": None,
    }
    score, info = base.format_report(tasks[0], trajectory)
    assert score == 0.0
    assert "unparsed_tool_call" in info["issues"][0]["issues"]
