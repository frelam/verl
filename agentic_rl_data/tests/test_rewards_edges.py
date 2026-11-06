"""Unit tests (edge cases): ``search_qa`` / ``math_ci`` / ``code_exec`` /
``abstain`` plus cross-cutting reward-contract checks.

**NOT EXECUTED** in the authoring session: this machine cannot start
``pwsh``/``python`` (process start dies with ``0xC0000142``), so this file is a
static deliverable.  Run it in a working shell::

    cd D:\\lhy\\verl\\verl-toolRL
    python -m pytest agentic_rl_data/tests/test_rewards_edges.py -q

Required edge cases from the task card (plan §5): empty trajectory / empty
answer, multi-answer F1, answer equivalence, all-tests-pass vs partial vs
missing summary, correct clarification vs fabricated answer, and the
``irrelevance`` vs ``abstain`` boundary.  Every result must pass
``schema.validate_reward_result`` and stay in ``[0, 1]``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `agentic_rl_data`
sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests dir, for `fixtures`

from fixtures.reward_fixtures import (  # noqa: E402
    call,
    env_state,
    task,
    trajectory,
    turn,
)

from agentic_rl_data import schema  # noqa: E402
from agentic_rl_data.rewards import (  # noqa: E402
    REWARDS,
    dispatch_reward,
    get_reward,
)
from agentic_rl_data.rewards import (  # noqa: E402
    abstain,
    code_exec,
    irrelevance,
    math_ci,
    search_qa,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def check_result(result: dict) -> dict:
    """Assert the payload satisfies the frozen reward contract."""
    assert isinstance(result, dict)
    assert schema.validate_reward_result(result) == [], schema.validate_reward_result(result)
    assert isinstance(result["score"], float)
    assert 0.0 <= result["score"] <= 1.0
    assert isinstance(result["info"], dict)
    json.dumps(result["info"], ensure_ascii=False, sort_keys=True)
    return result


def score_of(task_payload: dict, trajectory_payload: dict) -> float:
    return check_result(dispatch_reward(task_payload, trajectory_payload))["score"]


def score_of_search(task_payload: dict, trajectory_payload: dict) -> float:
    return check_result(search_qa.reward(task_payload, trajectory_payload))["score"]


# ---------------------------------------------------------------------------
# search_qa
# ---------------------------------------------------------------------------

class TestSearchQa:
    def _trajectory(self, answer: str | None, *, content: str = "I looked it up.") -> dict:
        return trajectory(
            [turn(1, content=content)],
            final_answer=answer,
        )

    def test_normalised_em_scores_one(self) -> None:
        task_payload = task("search_qa", final_answer="The Eiffel Tower")
        trajectory_payload = self._trajectory("eiffel tower.")
        result = check_result(search_qa.reward(task_payload, trajectory_payload))
        assert math_isclose(result["score"], 1.0)
        assert result["info"]["em"] == 1.0
        assert result["info"]["answer_source"] == "reference.final_answer"

    def test_multiple_answers_use_max_em_f1(self) -> None:
        task_payload = task(
            "search_qa",
            final_answer="Paris",
            final_answers=["Paris", "City of Lights"],
        )
        trajectory_payload = self._trajectory("City of Lights")
        result = check_result(search_qa.reward(task_payload, trajectory_payload))
        assert math_isclose(result["score"], 1.0)
        assert result["info"]["answer_source"] == "reference.final_answers"
        assert result["info"]["best_gold"] == "City of Lights"

    def test_partial_f1_earns_partial_answer_score(self) -> None:
        task_payload = task("search_qa", final_answer="New York")
        trajectory_payload = self._trajectory("New York City")
        result = check_result(search_qa.reward(task_payload, trajectory_payload))
        # f1 = 0.8, partial credit = 0.8 * f1 = 0.64, format = 1.0
        # -> 0.9 * 0.64 + 0.1 * 1.0
        assert math_isclose(result["score"], 0.676)
        assert 0.0 < result["info"]["f1"] < 1.0
        assert result["info"]["em"] == 0.0

    def test_wrong_answer_earns_no_format_credit(self) -> None:
        task_payload = task("search_qa", final_answer="Paris")
        trajectory_payload = self._trajectory("London")
        result = check_result(search_qa.reward(task_payload, trajectory_payload))
        # format credit is gated on a non-zero answer score (no 0.1 floor)
        assert result["score"] == 0.0
        assert result["info"]["answer_score"] == 0.0

    def test_answer_stuffing_earns_no_partial_credit(self) -> None:
        task_payload = task("search_qa", final_answer="Paris")
        stuffed = "Paris or London or Berlin or Rome or Madrid or Vienna or Prague"
        result = check_result(search_qa.reward(task_payload, self._trajectory(stuffed)))
        assert result["info"]["answer_too_long"] is True
        assert result["score"] == 0.0

    def test_exact_match_beats_partial_match(self) -> None:
        task_payload = task("search_qa", final_answer="New York")
        exact = score_of_search(task_payload, self._trajectory("New York"))
        partial = score_of_search(task_payload, self._trajectory("New York City"))
        assert exact == 1.0 and partial < 0.7

    def test_empty_prediction_scores_zero(self) -> None:
        task_payload = task("search_qa", final_answer="Paris")
        trajectory_payload = trajectory([turn(1, content="I could not find it.")], final_answer=None)
        result = check_result(search_qa.reward(task_payload, trajectory_payload))
        assert result["score"] == 0.0
        assert result["info"]["reason"] == "empty_prediction"

    def test_missing_reference_answer_scores_zero(self) -> None:
        task_payload = task("search_qa", final_answer=None, final_answers=None)
        result = check_result(
            search_qa.reward(task_payload, self._trajectory("Paris"))
        )
        assert result["score"] == 0.0
        assert result["info"]["reason"] == "missing_reference_answer"

    def test_wrong_answer_with_format_violation_scores_zero(self) -> None:
        task_payload = task("search_qa", final_answer="Paris")
        gt = [[call("search", {"query": "capital of France"})]]
        trajectory_payload = trajectory(
            [turn(1, list(gt[0]), content="")],  # call without tool_results
            final_answer="London",
        )
        result = check_result(search_qa.reward(task_payload, trajectory_payload))
        assert result["score"] == 0.0
        assert result["info"]["format_score"] == 0.0


# ---------------------------------------------------------------------------
# math_ci
# ---------------------------------------------------------------------------

class TestMathCi:
    def _trajectory(self, answer: str | None) -> dict:
        return trajectory([turn(1, content="Worked it out.")], final_answer=answer)

    def test_boxed_fraction_equivalence(self) -> None:
        task_payload = task("math_ci", final_answer="1/2")
        trajectory_payload = self._trajectory("\\boxed{\\frac{1}{2}}")
        result = check_result(math_ci.reward(task_payload, trajectory_payload))
        assert result["score"] == 1.0
        assert result["info"]["matched"] is True
        assert result["info"]["used_boxed"] is True

    def test_decimal_and_fraction_equivalence(self) -> None:
        task_payload = task("math_ci", final_answer="1/2")
        assert score_of(task_payload, self._trajectory("0.5")) == 1.0

    def test_wrong_answer_scores_zero(self) -> None:
        task_payload = task("math_ci", final_answer="1/2")
        result = check_result(math_ci.reward(task_payload, self._trajectory("2/3")))
        assert result["score"] == 0.0
        assert result["info"]["matched"] is False

    def test_missing_prediction_scores_zero(self) -> None:
        task_payload = task("math_ci", final_answer="1/2")
        result = check_result(math_ci.reward(task_payload, self._trajectory(None)))
        assert result["score"] == 0.0
        assert result["info"]["reason"] == "missing_prediction"

    def test_missing_reference_answer_scores_zero(self) -> None:
        task_payload = task("math_ci", final_answer=None)
        result = check_result(math_ci.reward(task_payload, self._trajectory("5")))
        assert result["score"] == 0.0
        assert result["info"]["reason"] == "missing_reference_answer"

    def test_unknown_tool_calls_do_not_change_answer_score(self) -> None:
        task_payload = task("math_ci", final_answer="1/2")
        gt = [[call("python", {"code": "print(1/2)"})]]
        trajectory_payload = trajectory(
            [turn(1, list(gt[0]))],
            final_answer="0.5",
        )
        # Missing tool_results would break the *format* gate, but math_ci does
        # not use format; the answer equivalence alone decides 1.0.
        result = check_result(math_ci.reward(task_payload, trajectory_payload))
        assert result["score"] == 1.0
        assert result["info"]["n_tool_calls"] == 1

    def test_verl_compatible_entry_points(self) -> None:
        assert math_ci.compute_score("\\boxed{5}", "5") == 1.0
        assert math_ci.compute_score("\\boxed{5}", "6") == 0.0
        assert math_ci.is_equiv("x=5", "5") is True
        assert math_ci.is_equiv("0.50", "0.500") is True
        assert math_ci.is_equiv("5\\text{ cm}", "5") is True
        assert math_ci.is_equiv("2", "3") is False


# ---------------------------------------------------------------------------
# code_exec
# ---------------------------------------------------------------------------

def _code_tests(**overrides) -> dict:
    summary = {
        "total": 3,
        "passed": 3,
        "failed": 0,
        "all_passed": True,
        "exit_code": 0,
        "duration_s": 1.25,
        "truncated": False,
        "failures": [],
    }
    summary.update(overrides)
    return summary


def _code_traj(env_state_value) -> dict:
    return trajectory([turn(1, content="Ran the tests.")], env_state_value=env_state_value)


class TestCodeExec:
    def test_all_tests_passed_scores_one(self) -> None:
        task_payload = task("code_exec", unit_tests="assert add(1, 2) == 3")
        env_state_value = env_state(
            type="code_tests", impl="local", code_tests=_code_tests()
        )
        result = check_result(code_exec.reward(task_payload, _code_traj(env_state_value)))
        assert result["score"] == 1.0
        assert result["info"]["summary_source"] == "env_state.code_tests"

    def test_partial_failure_scores_zero_with_failure_preview(self) -> None:
        task_payload = task("code_exec")
        summary = _code_tests(
            passed=2,
            failed=1,
            all_passed=False,
            failures=[{"name": "test_add", "message": "assert 3 == 4"}],
        )
        env_state_value = env_state(type="code_tests", impl="local", code_tests=summary)
        result = check_result(code_exec.reward(task_payload, _code_traj(env_state_value)))
        assert result["score"] == 0.0
        assert result["info"]["reason"] == "tests_failed"
        assert result["info"]["failures"][0]["name"] == "test_add"

    def test_missing_summary_scores_zero_without_crash(self) -> None:
        task_payload = task("code_exec")
        result = check_result(code_exec.reward(task_payload, trajectory()))
        assert result["score"] == 0.0
        assert result["info"]["missing_test_summary"] is True
        assert result["info"]["missing_env_state"] is True

    def test_null_code_tests_scores_zero(self) -> None:
        task_payload = task("code_exec")
        env_state_value = env_state(type="code_tests", impl="local", code_tests=None)
        result = check_result(code_exec.reward(task_payload, _code_traj(env_state_value)))
        assert result["score"] == 0.0
        assert result["info"]["missing_test_summary"] is True

    def test_inconsistent_summary_fails_closed(self) -> None:
        task_payload = task("code_exec")
        summary = _code_tests(total=2, passed=1, failed=0, all_passed=True)
        env_state_value = env_state(type="code_tests", impl="local", code_tests=summary)
        result = check_result(code_exec.reward(task_payload, _code_traj(env_state_value)))
        assert result["score"] == 0.0
        assert result["info"]["reason"] == "inconsistent_test_summary"

    def test_zero_tests_scores_zero(self) -> None:
        task_payload = task("code_exec")
        summary = _code_tests(total=0, passed=0, failed=0, all_passed=True)
        env_state_value = env_state(type="code_tests", impl="local", code_tests=summary)
        result = check_result(code_exec.reward(task_payload, _code_traj(env_state_value)))
        assert result["score"] == 0.0
        assert result["info"]["reason"] == "no_tests_collected"

    def test_extras_tests_is_not_a_scoring_source(self) -> None:
        """CONTRACTS C5/D6: only ``env_state["code_tests"]`` counts, never ``info``."""
        task_payload = task("code_exec")
        env_state_value = env_state(
            type="code_tests",
            impl="local",
            code_tests=None,
            extras={"tests": _code_tests(), "info_tests": _code_tests()},
        )
        result = check_result(code_exec.reward(task_payload, _code_traj(env_state_value)))
        assert result["score"] == 0.0
        assert result["info"]["missing_test_summary"] is True

    def test_derives_all_passed_when_missing_but_counts_consistent(self) -> None:
        task_payload = task("code_exec")
        summary = _code_tests(all_passed=None, total=4, passed=4, failed=0)
        env_state_value = env_state(type="code_tests", impl="local", code_tests=summary)
        result = check_result(code_exec.reward(task_payload, _code_traj(env_state_value)))
        assert result["score"] == 1.0
        assert result["info"]["all_passed_derived"] is True

    def test_failure_messages_are_truncated_in_info(self) -> None:
        task_payload = task("code_exec")
        long_message = "x" * 300
        summary = _code_tests(
            passed=0,
            failed=1,
            all_passed=False,
            failures=[{"name": "test_big", "message": long_message}],
        )
        env_state_value = env_state(type="code_tests", impl="local", code_tests=summary)
        result = check_result(code_exec.reward(task_payload, _code_traj(env_state_value)))
        assert result["score"] == 0.0
        assert len(result["info"]["failures"][0]["message"]) <= 200


# ---------------------------------------------------------------------------
# abstain
# ---------------------------------------------------------------------------

class TestAbstain:
    def _task(self) -> dict:
        return task("abstain")

    def _trajectory(self, reply: str) -> dict:
        return trajectory([turn(1, content=reply)])

    def test_request_info_scores_one(self) -> None:
        reply = "Could you please tell me which city you mean?"
        result = check_result(abstain.reward(self._task(), self._trajectory(reply)))
        assert result["score"] == 1.0
        assert result["info"]["abstention_class"] == "request_info"

    def test_no_valid_tools_scores_one(self) -> None:
        reply = "Sorry, I don't have access to real-time data and no available tool fits this request."
        result = check_result(abstain.reward(self._task(), self._trajectory(reply)))
        assert result["score"] == 1.0
        assert result["info"]["abstention_class"] == "no_valid_tools"

    def test_fabricated_direct_answer_scores_zero(self) -> None:
        reply = "The answer is 42."
        result = check_result(abstain.reward(self._task(), self._trajectory(reply)))
        assert result["score"] == 0.0
        assert result["info"]["abstention_class"] == "guess"

    def test_hedged_guess_scores_zero(self) -> None:
        reply = "I'm sorry, but I can't be sure - the answer is 42."
        result = check_result(abstain.reward(self._task(), self._trajectory(reply)))
        assert result["score"] == 0.0
        assert result["info"]["abstention_class"] == "guess"

    def test_question_bolted_on_answer_scores_zero(self) -> None:
        """Ask-dominance: an answer with a trailing question is still a guess."""
        reply = "The capital of France is Paris. Which city?"
        result = check_result(abstain.reward(self._task(), self._trajectory(reply)))
        assert result["score"] == 0.0
        assert result["info"]["abstention_class"] == "guess"

    def test_rhetorical_tail_scores_zero(self) -> None:
        reply = "The answer is 42, what?"
        result = check_result(abstain.reward(self._task(), self._trajectory(reply)))
        assert result["score"] == 0.0

    def test_spurious_tool_call_scores_zero(self) -> None:
        reply = "Let me check the weather."
        trajectory_payload = trajectory(
            [
                turn(
                    1,
                    [call("get_weather", {"city": "Paris"})],
                    content=reply,
                    results=[],
                )
            ]
        )
        result = check_result(abstain.reward(self._task(), trajectory_payload))
        assert result["score"] == 0.0
        assert result["info"]["abstention_class"] == "spurious_call"

    def test_empty_reply_scores_zero(self) -> None:
        result = check_result(abstain.reward(self._task(), trajectory([turn(1, content="")])))
        assert result["score"] == 0.0
        assert result["info"]["reply_is_empty"] is True

    def test_chinese_clarification_scores_one(self) -> None:
        reply = "抱歉，信息不足。请问你想查询哪个城市？"
        result = check_result(abstain.reward(self._task(), self._trajectory(reply)))
        assert result["score"] == 1.0
        assert result["info"]["abstention_class"] == "request_info"

    def test_final_answer_fallback_when_no_assistant_content(self) -> None:
        trajectory_payload = trajectory([turn(1, content="")], final_answer="Which city do you mean?")
        result = check_result(abstain.reward(self._task(), trajectory_payload))
        assert result["info"]["abstention_class"] == "request_info"
        assert result["info"]["reply_is_empty"] is False

    def test_boundary_with_irrelevance(self) -> None:
        """irrelevance = decision only; abstain = reply quality too."""
        guess_reply = "The answer is 42."
        clarify_reply = "Could you please tell me which city you mean?"
        guess_trajectory = trajectory([turn(1, content=guess_reply)])
        clarify_trajectory = trajectory([turn(1, content=clarify_reply)])
        assert score_of(task("irrelevance"), guess_trajectory) == 1.0
        assert score_of(task("abstain"), guess_trajectory) == 0.0
        assert score_of(task("irrelevance"), clarify_trajectory) == 1.0
        assert score_of(task("abstain"), clarify_trajectory) == 1.0


# ---------------------------------------------------------------------------
# cross-cutting
# ---------------------------------------------------------------------------

def test_all_reward_constants_match_frozen_score_range() -> None:
    for task_type in sorted(REWARDS):
        function = get_reward(task_type)
        assert callable(function)
    for module in (abstain, code_exec, irrelevance, math_ci, search_qa):
        assert module.MAX_SCORE == schema.SCORE_RANGE[1]


def test_every_reward_returns_json_serialisable_info() -> None:
    minimal_tasks = {
        "st_fc": task("st_fc", gt_calls=[[call("get_weather", {"city": "Paris"})]]),
        "irrelevance": task("irrelevance"),
        "mt_fc": task("mt_fc", gt_calls=[[call("get_weather", {"city": "Paris"})]]),
        "search_qa": task("search_qa", final_answer="Paris"),
        "math_ci": task("math_ci", final_answer="1/2"),
        "code_exec": task("code_exec"),
        "abstain": task("abstain"),
    }
    for task_type, task_payload in minimal_tasks.items():
        result = check_result(get_reward(task_type)(task_payload, trajectory([])))
        assert result["info"]["task_type"] == task_type


def test_scores_stay_within_frozen_range_for_extreme_inputs() -> None:
    # Pathological payloads must never raise or escape [0, 1].
    pathological_trajectory = {
        "turns": [],
        "final_answer": "\u0000" * 10,
        "num_turns": 0,
        "done": True,
        "env_state": {"goal_satisfied": None, "replay_ok": None},
    }
    for task_type in sorted(REWARDS):
        result = check_result(get_reward(task_type)(task(task_type), pathological_trajectory))
        assert 0.0 <= result["score"] <= 1.0


def test_mt_fc_weight_terms_are_used_in_score() -> None:
    from agentic_rl_data.rewards import mt_fc as mt_fc_module

    gt = [[call("get_weather", {"city": "Paris"})]]
    task_payload = task("mt_fc", gt_calls=gt)
    # terminal wrong (goal_satisfied False), per-turn 1/1, format 1/1
    trajectory_payload = trajectory(
        [turn(1, list(gt[0]), results=[{"name": "get_weather", "content": "ok", "ok": True}])],
        env_state_value=env_state(
            impl="stateful", goal_satisfied=False, matched_turns=1, gt_turns=1
        ),
    )
    result = check_result(mt_fc_module.reward(task_payload, trajectory_payload))
    expected = (
        mt_fc_module.RESULT_WEIGHT * 0.0
        + mt_fc_module.TURN_MATCH_WEIGHT * 1.0
        + mt_fc_module.FORMAT_WEIGHT * 1.0
    )
    assert result["score"] == pytest.approx(expected, abs=1e-9)


def math_isclose(value: float, expected: float) -> bool:
    """Local ``math.isclose`` wrapper with the reward arithmetic tolerance."""
    import math

    return math.isclose(value, expected, abs_tol=1e-9)


def test_reward_modules_import_only_stdlib() -> None:
    """Rewards must not import torch / verl (plan §8; D1 phase 1)."""
    import ast
    import inspect
    import sys

    for task_type in sorted(REWARDS):
        function = get_reward(task_type)
        module = sys.modules[function.__module__]
        path = inspect.getsourcefile(module)
        assert path is not None
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                imported.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module.split(".")[0])
        assert "torch" not in imported
        assert "verl" not in imported
