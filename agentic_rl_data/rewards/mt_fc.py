"""``mt_fc`` reward - multi-turn function calling (plan §5, CONTRACTS C4).

    score = 0.8 x result + 0.1 x per-turn match + 0.1 x format

The result term (0.8) is the *outcome* and strictly dominates: it is not an
alias of call matching (plan §10 - replay matching must not become the main
signal).  Frozen priority for the result term (CONTRACTS C4, order immutable):

1. ``trajectory["env_state"]["goal_satisfied"]`` non-null -> env verdict
   (stateful ``mock_fc``, A scheme);
2. else ``task.reference["final_state"]`` and ``env_state["state"]`` both
   non-null -> deep value comparison;
3. else ``reference["final_answer"]`` non-null -> normalised EM against
   ``trajectory["final_answer"]``;
4. else replay mode -> ``env_state["replay_ok"]``.

Per-turn term: trust ``env_state["matched_turns"] / ["gt_turns"]`` when the
env reports them (B scheme); otherwise derive the ratio from emitted call
groups vs ``gt_calls`` (A-scheme stateful envs do not always report replay
statistics).  Format term is the structural gate (``base.format_report``).

Missing ``env_state`` means the rollout never finished: the result term is 0
and ``info["missing_env_state"] = True`` (CONTRACTS C3); an exception is never
raised.  A timed-out rollout cannot earn the result term either: plan §4.1
("超限视为失败") forces ``result_basis = "timeout"`` and the result term to 0
without any extra penalty (CONTRACTS C1).  Placeholder replay
(``env_state.extras["gt_returns_source"] == "placeholder"`` or the static
``task.meta["gt_returns_source"] == "placeholder"``) is a degraded replay:
``info["degraded_replay"] = True`` and the 0.8 result term is not paid
(CONTRACTS C6, review T3-02) -- never rely on T2 filtering it offline.

All scores are clamped into the frozen ``[0, 1]`` range (CONTRACTS C4).
"""

from __future__ import annotations

from collections.abc import Mapping

from . import base

TASK_TYPE = "mt_fc"
RESULT_WEIGHT = 0.8
TURN_MATCH_WEIGHT = 0.1
FORMAT_WEIGHT = 0.1
MAX_SCORE = 1.0
#: turns beyond the reference plan that still earn the full efficiency term
EFFICIENCY_FREE_TURNS = 2

_RESULT_BASES = ("goal_satisfied", "final_state", "final_answer", "replay", "timeout")


def _env_int(value):
    """A plain non-bool int, else ``None``."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _checks_info(env: Mapping) -> dict:
    """Aggregate the stateful env's ``checks`` list (diagnostics only)."""
    checks = base.as_list(env.get("checks"))
    total = 0
    passed = 0
    failed: list[str] = []
    for check in checks:
        if not isinstance(check, Mapping):
            continue
        total += 1
        if check.get("ok") is True:
            passed += 1
        else:
            name = check.get("name")
            failed.append(str(name) if name is not None else "<unnamed>")
    return {"total": total, "passed": passed, "failed": failed[:10]}


def _resolve_result(task: dict, trajectory: dict) -> tuple[bool, str | None, dict]:
    """Resolve the 0.8 result term (CONTRACTS C4 priority order)."""
    env = base.env_state(trajectory)
    ref = base.reference(task)
    detail: dict = {}

    goal = env.get("goal_satisfied")
    if goal is not None:
        detail["goal_satisfied"] = goal
        if not isinstance(goal, bool):
            detail["non_bool_goal_satisfied"] = True
        return bool(goal), "goal_satisfied", detail

    ref_state = ref.get("final_state")
    obs_state = env.get("state")
    if isinstance(ref_state, Mapping) and isinstance(obs_state, Mapping):
        matched = base.values_equal(ref_state, obs_state)
        detail["state_match"] = matched
        return matched, "final_state", detail

    expected = ref.get("final_answer")
    if isinstance(expected, str):
        predicted = base.final_answer(trajectory)
        matched = bool(predicted is not None and base.text_equals(predicted, expected))
        detail["expected_final_answer"] = expected
        detail["predicted_final_answer"] = predicted
        detail["final_answer_match"] = matched
        return matched, "final_answer", detail

    replay = env.get("replay_ok")
    detail["replay_ok"] = replay
    if isinstance(replay, bool):
        return replay, "replay", detail
    detail["missing_result_signal"] = True
    return False, "replay", detail


def _turn_match(task: dict, trajectory: dict) -> dict:
    """0.1 per-turn match term: env-reported ratio, else derived from turns."""
    env = base.env_state(trajectory)
    matched = _env_int(env.get("matched_turns"))
    total = _env_int(env.get("gt_turns"))
    if total is not None and total >= 0:
        if total == 0:
            fraction = 1.0 if not base.nonempty_call_groups(trajectory) else 0.0
        else:
            count = matched if matched is not None else 0
            fraction = max(0.0, min(1.0, count / total))
        return {
            "matched": matched,
            "total": total,
            "fraction": fraction,
            "source": "env_state:matched_turns/gt_turns",
            "full_match": fraction >= 1.0 - 1e-9,
        }
    derived = base.derive_turn_match(trajectory, task)
    return {
        "matched": derived["matched"],
        "total": derived["total"],
        "fraction": derived["fraction"],
        "source": derived["source"],
        "full_match": derived["full_match"],
        "leftover_observed": derived["leftover_observed"],
    }


def process_mode(task: dict) -> str:
    """``"turn_match"`` (default) or ``"efficiency"`` (``env.config.process_term``)."""
    config = base.as_mapping(base.as_mapping(base.as_mapping(task).get("env")).get("config"))
    mode = config.get("process_term")
    return "efficiency" if mode == "efficiency" else "turn_match"


def _efficiency(task: dict, trajectory: dict, result_ok: bool) -> dict:
    """0.1 process term for outcome-verified (stateful) tasks.

    Matching the emitted calls against *one* reference plan punishes valid
    alternative plans and pays for imitation, so procedural stateful tasks
    opt into an efficiency term instead (``env.config.process_term =
    "efficiency"``):

    * it is **paid only on success** - a failed or idle trajectory can never
      collect it (otherwise "stop early" would be a free 0.1);
    * ``num_turns <= expected_turns + 2`` earns the full term (retries and a
      verification read are free), then it decays linearly to 0 at
      ``meta.max_turns``.

    There is deliberately no "recovery bonus": rewarding recovered errors
    would pay the policy for causing errors first.  Recovery is learned
    because injected faults block the goal until the call is retried.
    """
    meta = base.as_mapping(base.as_mapping(task).get("meta"))
    num_turns = len(base.iter_turns(trajectory))
    expected = meta.get("expected_turns")
    max_turns = meta.get("max_turns")
    info = {"num_turns": num_turns, "expected_turns": expected, "max_turns": max_turns,
            "source": "efficiency"}
    if not result_ok:
        return {**info, "fraction": 0.0, "reason": "paid_only_on_success"}
    if not (base._is_int(expected) and base._is_int(max_turns)):
        return {**info, "fraction": 1.0, "reason": "no_turn_budget"}
    free = expected + EFFICIENCY_FREE_TURNS
    if num_turns <= free or max_turns <= free:
        return {**info, "fraction": 1.0}
    fraction = max(0.0, min(1.0, (max_turns - num_turns) / float(max_turns - free)))
    return {**info, "fraction": fraction}


def reward(task: dict, trajectory: dict) -> dict:
    """Pure reward for one ``mt_fc`` task/trajectory pair."""
    format_score, format_info = base.format_report(task, trajectory)
    mode = process_mode(task)
    turn = _turn_match(task, trajectory)
    env = base.env_state(trajectory)
    env_present = base.has_env_state(trajectory)
    timeout = env.get("timeout") is True
    degraded = base.degraded_replay(trajectory, task)

    if not env_present:
        raw_result_ok = False
        result_basis = None
        result_detail = {"missing_result_signal": True}
        missing_env_state = True
    elif timeout:
        # plan §4.1 ("超限视为失败"): a timed-out rollout must not earn the
        # 0.8 result term even when the env terminal verdict is True.
        raw_result_ok = False
        result_basis = "timeout"
        result_detail = {
            "timeout": True,
            "goal_satisfied": env.get("goal_satisfied"),
            "replay_ok": env.get("replay_ok"),
        }
        missing_env_state = False
    else:
        raw_result_ok, result_basis, result_detail = _resolve_result(task, trajectory)
        missing_env_state = False

    result_suppressed = None
    if missing_env_state:
        result_value = 0.0
        result_suppressed = "missing_env_state"
    elif timeout:
        result_value = 0.0
        result_suppressed = "timeout"
    elif degraded and result_basis in ("replay", None):
        # CONTRACTS C6 / review T3-02: a placeholder (degraded) replay must
        # not pay the result term; do not rely on T2 filtering it offline.
        # Only the replay branch consumes gt returns; stateful terminal /
        # final_answer verdicts do not, so they are unaffected.
        result_value = 0.0
        result_suppressed = "degraded_replay"
    else:
        result_value = 1.0 if raw_result_ok else 0.0

    idle_without_calls = False
    if mode == "efficiency":
        turn = {**_efficiency(task, trajectory, result_value >= 1.0), "turn_match_reference": turn}
        # A trajectory that never calls a tool on a task that needs tools must
        # not out-score an honest failed attempt through the format term alone.
        if result_value < 1.0 and not base.all_calls(trajectory) and base.flatten_gt_calls(task):
            format_score = 0.0
            idle_without_calls = True

    score = base.weighted_sum(
        [
            (RESULT_WEIGHT, result_value),
            (TURN_MATCH_WEIGHT, turn["fraction"]),
            (FORMAT_WEIGHT, format_score),
        ]
    )

    info = {
        "weights": {
            "result": RESULT_WEIGHT,
            "turn_match": TURN_MATCH_WEIGHT,
            "format": FORMAT_WEIGHT,
        },
        "result_basis": result_basis,
        "result": result_value,
        "raw_result": 1.0 if raw_result_ok else 0.0,
        "result_suppressed": result_suppressed,
        "turn_match": turn["fraction"],
        "process_term": mode,
        "idle_without_calls": idle_without_calls,
        "format": format_score,
        "components": {
            "result": result_value,
            "turn_match": turn["fraction"],
            "format": format_score,
        },
        "missing_env_state": missing_env_state,
        "env_state_present": env_present,
        "env_impl": env.get("impl"),
        "goal_satisfied": env.get("goal_satisfied"),
        "replay_ok": env.get("replay_ok"),
        "failed_turn": env.get("failed_turn"),
        "timeout": timeout,
        "degraded_replay": degraded,
        "checks": _checks_info(env),
        "result_detail": result_detail,
        "turn_match_detail": turn,
        "format_detail": format_info,
        "result_bases_allowed": list(_RESULT_BASES),
    }
    return base.make_result(score, info, task_type=TASK_TYPE)
