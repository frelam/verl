"""Unit tests: reward registry + ``st_fc`` / ``irrelevance`` / ``mt_fc``.

**NOT EXECUTED** in the authoring session: this machine cannot start
``pwsh``/``python`` (process start dies with ``0xC0000142``), so this file is a
static deliverable.  Run it in a working shell::

    cd D:\\lhy\\verl\\verl-toolRL
    python -m pytest agentic_rl_data/tests/test_rewards.py -q

Coverage (plan §5, §8 A3): positive cases, negative cases and the required
edge cases - empty trajectory, permuted parallel calls, duplicate calls, "label
needs no tool but a call was emitted", right tool / wrong arguments,
"all calls wrong but format legal" (must stay low), weights and score range.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `agentic_rl_data`
sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests dir, for `fixtures`

from fixtures.reward_fixtures import (  # noqa: E402
    call,
    env_state,
    task,
    tool_result,
    trajectory,
    turn,
    turns_from_call_groups,
)

from agentic_rl_data import schema  # noqa: E402
from agentic_rl_data.rewards import (  # noqa: E402
    REWARDS,
    TASK_TYPES,
    dispatch_reward,
    get_reward,
)
from agentic_rl_data.rewards import base, irrelevance, mt_fc as mt_fc_module, st_fc  # noqa: E402


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
    """Dispatch + contract-check + return the score."""
    return check_result(dispatch_reward(task_payload, trajectory_payload))["score"]


def assert_fixtures_valid() -> None:
    """Fixture builders must produce schema-valid payloads for every task type."""
    for task_type in sorted(schema.TASK_TYPES):
        task_payload = task(task_type)
        assert schema.validate_task(task_payload) == [], (
            f"{task_type}: {schema.validate_task(task_payload)}"
        )
        env_state_value = None
        if task_type == "mt_fc":
            env_state_value = env_state(
                impl="replay", replay_ok=True, matched_turns=0, gt_turns=0
            )
        trajectory_payload = trajectory([], env_state_value=env_state_value)
        assert schema.validate_trajectory(trajectory_payload) == [], (
            f"{task_type}: {schema.validate_trajectory(trajectory_payload)}"
        )


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

class TestRewardRegistry:
    def test_registry_covers_all_frozen_task_types(self) -> None:
        assert set(REWARDS) == set(schema.TASK_TYPES)
        assert TASK_TYPES == schema.TASK_TYPES
        assert all(callable(function) for function in REWARDS.values())

    def test_get_reward_returns_registered_function(self) -> None:
        for task_type in sorted(REWARDS):
            assert callable(get_reward(task_type))

    def test_unknown_task_type_fails_loudly(self) -> None:
        with pytest.raises(ValueError, match="unknown task_type"):
            get_reward("not_a_task_type")

    def test_non_string_task_type_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            get_reward(None)  # type: ignore[arg-type]

    def test_dispatch_reward_uses_task_type(self) -> None:
        gt_calls = [[call("get_weather", {"city": "Paris"})]]
        task_payload = task("st_fc", gt_calls=gt_calls)
        trajectory_payload = trajectory(turns_from_call_groups(gt_calls))
        assert check_result(dispatch_reward(task_payload, trajectory_payload))["score"] == 1.0

    def test_get_reward_two_argument_form_for_replay_check(self) -> None:
        """``scripts/replay_check.py`` probes ``get_reward(task, trajectory)``."""
        gt_calls = [[call("get_weather", {"city": "Paris"})]]
        task_payload = task("st_fc", gt_calls=gt_calls)
        trajectory_payload = trajectory(turns_from_call_groups(gt_calls))
        result = get_reward(task_payload, trajectory_payload)
        assert check_result(result)["score"] == 1.0
        with pytest.raises(TypeError):
            get_reward(task_payload)

    def test_mt_fc_weights_sum_to_one_and_result_dominates(self) -> None:
        assert mt_fc_module.RESULT_WEIGHT == pytest.approx(0.8)
        assert mt_fc_module.TURN_MATCH_WEIGHT == pytest.approx(0.1)
        assert mt_fc_module.FORMAT_WEIGHT == pytest.approx(0.1)
        total = (
            mt_fc_module.RESULT_WEIGHT
            + mt_fc_module.TURN_MATCH_WEIGHT
            + mt_fc_module.FORMAT_WEIGHT
        )
        assert total == pytest.approx(1.0, abs=1e-9)
        assert mt_fc_module.RESULT_WEIGHT > mt_fc_module.TURN_MATCH_WEIGHT
        assert mt_fc_module.RESULT_WEIGHT > mt_fc_module.FORMAT_WEIGHT
        # The outcome term must dominate both auxiliary terms combined (plan §10).
        assert mt_fc_module.RESULT_WEIGHT > (
            mt_fc_module.TURN_MATCH_WEIGHT + mt_fc_module.FORMAT_WEIGHT
        )

    def test_max_score_constant_matches_schema(self) -> None:
        assert mt_fc_module.MAX_SCORE == schema.SCORE_RANGE[1]
        assert st_fc.MAX_SCORE == schema.SCORE_RANGE[1]
        assert irrelevance.MAX_SCORE == schema.SCORE_RANGE[1]


# ---------------------------------------------------------------------------
# st_fc
# ---------------------------------------------------------------------------

class TestStFc:
    def _task(self, gt_calls, **kwargs) -> dict:
        return task("st_fc", gt_calls=gt_calls, **kwargs)

    def test_exact_call_scores_one(self) -> None:
        gt = [[call("get_weather", {"city": "Paris", "unit": "celsius"})]]
        task_payload = self._task(gt)
        trajectory_payload = trajectory(turns_from_call_groups(gt), final_answer="21C in Paris.")
        assert score_of(task_payload, trajectory_payload) == 1.0

    def test_parallel_calls_match_order_independently(self) -> None:
        first = call("get_weather", {"city": "Paris"})
        second = call("get_forecast", {"city": "Paris", "days": 3})
        task_payload = self._task([[first, second]])
        # Emitted in the opposite order: must still be a match.
        trajectory_payload = trajectory(turns_from_call_groups([[second, first]]))
        result = check_result(st_fc.reward(task_payload, trajectory_payload))
        assert result["score"] == 1.0
        assert len(result["info"]["matched_calls"]) == 2

    def test_missing_parallel_call_scores_zero(self) -> None:
        first = call("get_weather", {"city": "Paris"})
        second = call("get_forecast", {"city": "Paris", "days": 3})
        task_payload = self._task([[first, second]])
        trajectory_payload = trajectory(turns_from_call_groups([[first]]))
        result = check_result(st_fc.reward(task_payload, trajectory_payload))
        assert result["score"] == 0.0
        assert result["info"]["missed_calls"]

    def test_wrong_tool_name_scores_zero(self) -> None:
        task_payload = self._task([[call("get_weather", {"city": "Paris"})]])
        wrong = [call("get_forecast", {"city": "Paris", "days": 3})]
        trajectory_payload = trajectory(turns_from_call_groups([wrong]))
        assert score_of(task_payload, trajectory_payload) == 0.0

    def test_right_tool_wrong_argument_value_scores_zero(self) -> None:
        task_payload = self._task([[call("get_weather", {"city": "Paris"})]])
        wrong = [call("get_weather", {"city": "Lyon"})]
        trajectory_payload = trajectory(turns_from_call_groups([wrong]))
        assert score_of(task_payload, trajectory_payload) == 0.0

    def test_duplicate_call_scores_zero(self) -> None:
        gt = [[call("get_weather", {"city": "Paris"})]]
        task_payload = self._task(gt)
        duplicate = [call("get_weather", {"city": "Paris"}), call("get_weather", {"city": "Paris"})]
        trajectory_payload = trajectory(turns_from_call_groups([duplicate]))
        result = check_result(st_fc.reward(task_payload, trajectory_payload))
        assert result["score"] == 0.0
        assert result["info"]["no_extra_calls"] is False
        assert result["info"]["extra_calls"]

    def test_label_needs_no_tools_but_call_emitted_scores_zero(self) -> None:
        task_payload = self._task([])
        trajectory_payload = trajectory(
            turns_from_call_groups([[call("get_weather", {"city": "Paris"})]])
        )
        assert score_of(task_payload, trajectory_payload) == 0.0

    def test_label_needs_no_tools_and_no_call_scores_one(self) -> None:
        task_payload = self._task([])
        trajectory_payload = trajectory([turn(1, content="Paris is the capital of France.")])
        assert score_of(task_payload, trajectory_payload) == 1.0

    def test_empty_trajectory_scores_zero(self) -> None:
        task_payload = self._task([[call("get_weather", {"city": "Paris"})]])
        result = check_result(st_fc.reward(task_payload, trajectory()))
        assert result["score"] == 0.0
        assert result["info"]["empty_trajectory"] is True
        assert result["info"]["format_ok"] is False

    def test_structured_argument_stringification_still_matches(self) -> None:
        """AST-ish matching: a JSON-stringified list equals the structured value."""
        gt = [[call("get_bulk_weather", {"cities": ["Paris", "Lyon"]})]]
        emitted = [call("get_bulk_weather", {"cities": '["Paris", "Lyon"]'})]
        task_payload = self._task(gt)
        trajectory_payload = trajectory(turns_from_call_groups([emitted]))
        result = check_result(st_fc.reward(task_payload, trajectory_payload))
        assert result["score"] == 1.0

    def test_numeric_tolerance_in_argument_matching(self) -> None:
        gt = [[call("get_forecast", {"city": "Paris", "days": 3})]]
        emitted = [call("get_forecast", {"city": "paris", "days": 3.0})]
        task_payload = self._task(gt)
        trajectory_payload = trajectory(turns_from_call_groups([emitted]))
        assert score_of(task_payload, trajectory_payload) == 1.0

    def test_single_turn_prediction_does_not_require_tool_results(self) -> None:
        """st_fc predicts calls without executing them (D20 / CONTRACTS C14).

        Emitting the correct call with no tool_results is the normal single-turn
        case (env.type == "none"): the format gate must not punish it.
        """
        gt = [[call("get_weather", {"city": "Paris"})]]
        task_payload = self._task(gt)
        trajectory_payload = trajectory([turn(1, list(gt[0]))])  # matched call, no results
        result = check_result(st_fc.reward(task_payload, trajectory_payload))
        assert result["score"] == 1.0
        assert result["info"]["format_ok"] is True
        assert result["info"]["all_calls_matched"] is True

    def test_unknown_argument_breaks_the_format_gate(self) -> None:
        gt = [[call("get_weather", {"city": "Paris", "lang": "fr"})]]
        task_payload = self._task(gt)
        trajectory_payload = trajectory(turns_from_call_groups(gt))
        result = check_result(st_fc.reward(task_payload, trajectory_payload))
        assert result["score"] == 0.0
        assert result["info"]["all_calls_matched"] is True
        assert result["info"]["format_ok"] is False

    def test_calls_without_declared_tools_break_the_format_gate(self) -> None:
        """review T3-05: a call emitted when no tools are declared is invalid.

        The argument values still match the label, so only the format gate can
        zero this sample.
        """
        gt = [[call("get_weather", {"city": "Paris"})]]
        task_payload = task("st_fc", tools=[], gt_calls=gt)
        trajectory_payload = trajectory(turns_from_call_groups(gt))
        result = check_result(st_fc.reward(task_payload, trajectory_payload))
        assert result["info"]["all_calls_matched"] is True
        assert result["score"] == 0.0
        assert result["info"]["format_ok"] is False


# ---------------------------------------------------------------------------
# irrelevance
# ---------------------------------------------------------------------------

class TestIrrelevance:
    def test_no_tool_call_scores_one(self) -> None:
        task_payload = task("irrelevance")
        trajectory_payload = trajectory([turn(1, content="The capital of France is Paris.")])
        assert score_of(task_payload, trajectory_payload) == 1.0

    def test_any_tool_call_scores_zero(self) -> None:
        task_payload = task("irrelevance")
        gt = [[call("get_weather", {"city": "Paris"})]]
        trajectory_payload = trajectory(turns_from_call_groups(gt))
        result = check_result(irrelevance.reward(task_payload, trajectory_payload))
        assert result["score"] == 0.0
        assert result["info"]["n_output_calls"] == 1

    def test_empty_trajectory_scores_one_per_literal_rule(self) -> None:
        """Plan §5 is literal: no tool call = 1 (flagged for hacking analysis)."""
        task_payload = task("irrelevance")
        result = check_result(irrelevance.reward(task_payload, trajectory()))
        assert result["score"] == 1.0
        assert result["info"]["empty_trajectory"] is True

    def test_results_without_assistant_call_score_one(self) -> None:
        task_payload = task("irrelevance")
        trajectory_payload = trajectory(
            [turn(1, results=[tool_result("get_weather")], content="")]
        )
        assert score_of(task_payload, trajectory_payload) == 1.0

    def test_every_reward_keeps_score_in_range(self) -> None:
        assert_fixtures_valid()


# ---------------------------------------------------------------------------
# mt_fc
# ---------------------------------------------------------------------------

def _mt_groups() -> list[list[dict]]:
    return [
        [call("get_weather", {"city": "Paris"})],
        [call("get_forecast", {"city": "Paris", "days": 3})],
    ]


def _mt_task(gt_calls=None, **kwargs) -> dict:
    return task("mt_fc", gt_calls=_mt_groups() if gt_calls is None else gt_calls, **kwargs)


class TestMtFc:
    def test_goal_satisfied_scores_full_marks(self) -> None:
        gt = _mt_groups()
        task_payload = _mt_task(gt)
        trajectory_payload = trajectory(
            turns_from_call_groups(gt),
            env_state_value=env_state(
                impl="stateful",
                goal_satisfied=True,
                matched_turns=2,
                gt_turns=2,
                checks=[{"name": "booking_exists", "ok": True, "detail": ""}],
            ),
        )
        result = check_result(mt_fc_module.reward(task_payload, trajectory_payload))
        assert math.isclose(result["score"], 1.0, abs_tol=1e-9)
        assert result["info"]["result_basis"] == "goal_satisfied"
        assert result["info"]["result"] == 1.0
        assert result["info"]["checks"]["passed"] == 1

    def test_goal_not_satisfied_caps_at_auxiliary_terms(self) -> None:
        gt = _mt_groups()
        task_payload = _mt_task(gt)
        trajectory_payload = trajectory(
            turns_from_call_groups(gt),
            env_state_value=env_state(
                impl="stateful", goal_satisfied=False, matched_turns=2, gt_turns=2
            ),
        )
        result = check_result(mt_fc_module.reward(task_payload, trajectory_payload))
        # 0.8 x 0 + 0.1 x 1 + 0.1 x 1
        assert math.isclose(result["score"], 0.2, abs_tol=1e-9)
        assert result["info"]["result_basis"] == "goal_satisfied"

    def test_result_term_dominates_replay_matching(self) -> None:
        """Terminal state right + calls wrong must beat terminal wrong + calls right."""
        gt = _mt_groups()
        # (a) terminal state correct, emitted calls do not match the gt calls.
        wrong_emitted = [
            call("get_forecast", {"city": "Lyon", "days": 1}),
            call("get_weather", {"city": "Lyon"}),
        ]
        terminal_ok = trajectory(
            turns_from_call_groups([wrong_emitted]),
            env_state_value=env_state(
                impl="stateful", goal_satisfied=True, matched_turns=0, gt_turns=2
            ),
        )
        # (b) terminal state wrong, calls perfectly matched.
        terminal_bad = trajectory(
            turns_from_call_groups(gt),
            env_state_value=env_state(
                impl="stateful", goal_satisfied=False, matched_turns=2, gt_turns=2
            ),
        )
        score_terminal_ok = score_of(_mt_task(gt), terminal_ok)
        score_terminal_bad = score_of(_mt_task(gt), terminal_bad)
        assert score_terminal_ok > score_terminal_bad
        assert math.isclose(score_terminal_ok, 0.9, abs_tol=1e-9)
        assert math.isclose(score_terminal_bad, 0.2, abs_tol=1e-9)

    def test_replay_full_match_scores_full_marks(self) -> None:
        gt = _mt_groups()
        task_payload = _mt_task(gt)
        trajectory_payload = trajectory(
            turns_from_call_groups(gt),
            env_state_value=env_state(
                impl="replay", replay_ok=True, matched_turns=2, gt_turns=2
            ),
        )
        result = check_result(mt_fc_module.reward(task_payload, trajectory_payload))
        assert math.isclose(result["score"], 1.0, abs_tol=1e-9)
        assert result["info"]["result_basis"] == "replay"

    def test_replay_not_ok_scores_low(self) -> None:
        gt = _mt_groups()
        task_payload = _mt_task(gt)
        trajectory_payload = trajectory(
            turns_from_call_groups(gt),
            env_state_value=env_state(
                impl="replay", replay_ok=False, matched_turns=2, gt_turns=2
            ),
        )
        result = check_result(mt_fc_module.reward(task_payload, trajectory_payload))
        assert math.isclose(result["score"], 0.2, abs_tol=1e-9)

    def test_replay_partial_turn_match_scales_auxiliary_term(self) -> None:
        gt = _mt_groups()
        task_payload = _mt_task(gt)
        trajectory_payload = trajectory(
            turns_from_call_groups([gt[0]]),
            env_state_value=env_state(
                impl="replay", replay_ok=True, matched_turns=1, gt_turns=2
            ),
        )
        result = check_result(mt_fc_module.reward(task_payload, trajectory_payload))
        # 0.8 + 0.1 x 0.5 + 0.1 x 1.0
        assert math.isclose(result["score"], 0.95, abs_tol=1e-9)
        assert result["info"]["turn_match"] == pytest.approx(0.5)

    def test_empty_gt_turn_group_aligns_with_reply_turn(self) -> None:
        """review T3-03: ``gt_calls=[[call], []]`` + pure reply turn must be full marks."""
        first = call("get_weather", {"city": "Paris"})
        task_payload = task("mt_fc", gt_calls=[[first], []])
        turns = [
            turn(1, [first], results=[tool_result("get_weather")]),
            turn(2, content="Done."),
        ]
        trajectory_payload = trajectory(
            turns,
            env_state_value=env_state(impl="stateful", goal_satisfied=True),
        )
        result = check_result(mt_fc_module.reward(task_payload, trajectory_payload))
        assert result["info"]["turn_match"] == pytest.approx(1.0)
        assert math.isclose(result["score"], 1.0, abs_tol=1e-9)

    def test_final_state_fallback(self) -> None:
        task_payload = _mt_task([], final_state={"balance": 100.0})
        trajectory_payload = trajectory(
            [turn(1, content="Done.")],
            env_state_value=env_state(state={"balance": 100}),
        )
        result = check_result(mt_fc_module.reward(task_payload, trajectory_payload))
        assert math.isclose(result["score"], 1.0, abs_tol=1e-9)
        assert result["info"]["result_basis"] == "final_state"

    def test_final_state_mismatch_scores_result_zero(self) -> None:
        task_payload = _mt_task([], final_state={"balance": 100.0})
        trajectory_payload = trajectory(
            [turn(1, content="Done.")],
            env_state_value=env_state(state={"balance": 99}),
        )
        result = check_result(mt_fc_module.reward(task_payload, trajectory_payload))
        assert math.isclose(result["score"], 0.2, abs_tol=1e-9)

    def test_final_answer_fallback(self) -> None:
        task_payload = _mt_task([], final_answer="Paris")
        trajectory_payload = trajectory(
            [turn(1, content="The answer is Paris.")],
            final_answer="paris.",
            env_state_value=env_state(),
        )
        result = check_result(mt_fc_module.reward(task_payload, trajectory_payload))
        assert math.isclose(result["score"], 1.0, abs_tol=1e-9)
        assert result["info"]["result_basis"] == "final_answer"

    def test_final_answer_mismatch_scores_result_zero(self) -> None:
        task_payload = _mt_task([], final_answer="Paris")
        trajectory_payload = trajectory(
            [turn(1, content="The answer is Lyon.")],
            final_answer="Lyon",
            env_state_value=env_state(),
        )
        result = check_result(mt_fc_module.reward(task_payload, trajectory_payload))
        assert math.isclose(result["score"], 0.2, abs_tol=1e-9)

    def test_missing_result_signal_scores_low(self) -> None:
        gt = _mt_groups()
        task_payload = _mt_task(gt)
        trajectory_payload = trajectory(
            turns_from_call_groups(gt),
            env_state_value=env_state(
                impl="replay", matched_turns=2, gt_turns=2
            ),
        )
        result = check_result(mt_fc_module.reward(task_payload, trajectory_payload))
        assert math.isclose(result["score"], 0.2, abs_tol=1e-9)
        assert result["info"]["result_basis"] == "replay"
        assert result["info"]["result_detail"]["missing_result_signal"] is True

    def test_missing_env_state_is_unfinished_not_a_crash(self) -> None:
        gt = _mt_groups()
        task_payload = _mt_task(gt, final_answer="Paris")
        trajectory_payload = trajectory(
            turns_from_call_groups(gt),
            final_answer="Paris",
            env_state_value=None,
        )
        result = check_result(mt_fc_module.reward(task_payload, trajectory_payload))
        assert math.isclose(result["score"], 0.2, abs_tol=1e-9)
        assert result["info"]["missing_env_state"] is True
        # Priority (b/c) must not rescue a rollout that never finished.
        assert result["info"]["result"] == 0.0

    def test_placeholder_replay_does_not_earn_result_term(self) -> None:
        """review T3-02/C6: degraded replay must not pay the 0.8 result term."""
        gt = _mt_groups()
        task_payload = _mt_task(gt)
        common = dict(impl="replay", replay_ok=True, matched_turns=2, gt_turns=2)
        clean = trajectory(
            turns_from_call_groups(gt),
            env_state_value=env_state(**common),
        )
        degraded = trajectory(
            turns_from_call_groups(gt),
            env_state_value=env_state(
                **common, extras={"gt_returns_source": "placeholder"}
            ),
        )
        clean_result = check_result(mt_fc_module.reward(task_payload, clean))
        degraded_result = check_result(mt_fc_module.reward(task_payload, degraded))
        assert degraded_result["info"]["degraded_replay"] is True
        assert clean_result["info"]["degraded_replay"] is False
        assert degraded_result["info"]["result"] == 0.0
        assert degraded_result["info"]["result_suppressed"] == "degraded_replay"
        assert degraded_result["score"] <= 0.2 + 1e-9
        assert math.isclose(clean_result["score"], 1.0, abs_tol=1e-9)

    def test_static_placeholder_marker_also_counts_as_degraded(self) -> None:
        """schema v2.2 ``meta.gt_returns_source == "placeholder"`` is enough."""
        gt = _mt_groups()
        task_payload = _mt_task(gt, meta={"gt_returns_source": "placeholder"})
        trajectory_payload = trajectory(
            turns_from_call_groups(gt),
            env_state_value=env_state(
                impl="replay", replay_ok=True, matched_turns=2, gt_turns=2
            ),
        )
        result = check_result(mt_fc_module.reward(task_payload, trajectory_payload))
        assert result["info"]["degraded_replay"] is True
        assert result["info"]["result"] == 0.0
        assert result["score"] <= 0.2 + 1e-9

    def test_timeout_cannot_earn_result_term(self) -> None:
        """review T3-01/plan §4.1: 超限视为失败, even if the terminal verdict is True."""
        gt = _mt_groups()
        task_payload = _mt_task(gt)
        trajectory_payload = trajectory(
            turns_from_call_groups(gt),
            env_state_value=env_state(
                impl="stateful",
                goal_satisfied=True,
                timeout=True,
                matched_turns=2,
                gt_turns=2,
            ),
        )
        result = check_result(mt_fc_module.reward(task_payload, trajectory_payload))
        # 0.8 x 0 (timeout) + 0.1 x 1 + 0.1 x 1
        assert math.isclose(result["score"], 0.2, abs_tol=1e-9)
        assert result["info"]["timeout"] is True
        assert result["info"]["result_basis"] == "timeout"
        assert result["info"]["result"] == 0.0
        assert result["info"]["result_suppressed"] == "timeout"

    def test_all_wrong_but_format_legal_stays_low(self) -> None:
        gt = _mt_groups()
        task_payload = _mt_task(gt)
        emitted = [call("get_forecast", {"city": "Lyon", "days": 1})]
        trajectory_payload = trajectory(
            turns_from_call_groups([emitted]),
            env_state_value=env_state(
                impl="replay", replay_ok=False, matched_turns=0, gt_turns=2
            ),
        )
        result = check_result(mt_fc_module.reward(task_payload, trajectory_payload))
        # Format-only credit remains: 0.8 x 0 + 0.1 x 0 + 0.1 x 1 = 0.1.
        assert math.isclose(result["score"], 0.1, abs_tol=1e-9)
        assert result["score"] < 0.2

    def test_weights_are_reported_in_info(self) -> None:
        result = check_result(
            mt_fc_module.reward(
                _mt_task(),
                trajectory(
                    turns_from_call_groups(_mt_groups()),
                    env_state_value=env_state(
                        impl="replay", replay_ok=True, matched_turns=2, gt_turns=2
                    ),
                ),
            )
        )
        assert result["info"]["weights"] == {
            "result": 0.8,
            "turn_match": 0.1,
            "format": 0.1,
        }


# ---------------------------------------------------------------------------
# cross-cutting
# ---------------------------------------------------------------------------

def test_rewards_are_pure_and_do_not_mutate_inputs() -> None:
    gt_calls = [[call("get_weather", {"city": "Paris"})]]
    for task_type in sorted(REWARDS):
        if task_type in ("search_qa", "math_ci", "code_exec", "abstain", "irrelevance"):
            task_payload = task(task_type)
        else:
            task_payload = task(task_type, gt_calls=gt_calls)
        trajectory_payload = trajectory(
            turns_from_call_groups(gt_calls),
            env_state_value=env_state(impl="replay", replay_ok=True, matched_turns=1, gt_turns=1),
        )
        task_copy = copy.deepcopy(task_payload)
        trajectory_copy = copy.deepcopy(trajectory_payload)
        first = check_result(dispatch_reward(task_payload, trajectory_payload))
        second = check_result(dispatch_reward(task_payload, trajectory_payload))
        assert first == second
        assert task_payload == task_copy
        assert trajectory_payload == trajectory_copy


def test_empty_trajectory_never_crashes_for_any_reward() -> None:
    for task_type in sorted(REWARDS):
        task_payload = task(task_type)
        result = check_result(dispatch_reward(task_payload, trajectory()))
        assert 0.0 <= result["score"] <= 1.0
        assert isinstance(result["info"].get("task_type"), str)


def test_make_result_records_raw_score_and_clamps() -> None:
    """review T3-04: clamping must be visible, never silently mask a bug."""
    result = base.make_result(1.0000000000000002, {"probe": "float_sum"}, task_type="st_fc")
    assert result["score"] == 1.0
    assert result["info"]["score_clamped"] is True
    assert result["info"]["score_raw"] == 1.0000000000000002

    clean = base.make_result(0.5, {"probe": "clean"}, task_type="st_fc")
    assert clean["info"]["score_clamped"] is False
    assert clean["info"]["score_raw"] == 0.5

    with pytest.raises(schema.SchemaError):
        base.make_result(1.5, {}, strict=True)
    with pytest.raises(schema.SchemaError):
        base.make_result(float("nan"), {}, strict=True)


def test_values_equal_int_precision_is_exact() -> None:
    """review T3-06: int-int comparison must not use the float tolerance."""
    assert base.values_equal(10**9, 10**9 + 1) is False
    assert base.values_equal(3, 3.0) is True
    assert base.values_equal(0, False) is False


def test_fixture_builders_are_schema_valid() -> None:
    assert_fixtures_valid()
