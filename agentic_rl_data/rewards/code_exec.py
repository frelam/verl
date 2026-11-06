"""``code_exec`` reward - unit tests all pass (plan §4.4 / §5).

``score = 1.0`` iff the last ``code_tests`` run reports ``all_passed = True``
with consistent counts (``all_passed == (passed == total)`` and
``total > 0``), else ``0.0`` (plan §5 "单测全部通过 = 1，否则 0").

The **only** test-result source is ``trajectory["env_state"]["code_tests"]``
(CONTRACTS C4/C5, D6).  The reward never reads env logging fields such as
``info["tests"]``, never executes code and never sees the candidate / reference
source (plan §4.4).  Missing summary -> ``0.0`` +
``info["missing_test_summary"] = True`` (no exception).  A summary whose
``all_passed`` contradicts the counts is treated as 0.0 and flagged
``inconsistent_test_summary`` (fail-closed, never trust broken telemetry).

Records ``failures`` (name + truncated message) in ``info`` for debugging
reward hacking - the env must already redact candidate/standard-answer code
(CONTRACTS C5).
"""

from __future__ import annotations

from collections.abc import Mapping

from . import base

TASK_TYPE = "code_exec"
MAX_SCORE = 1.0
_FAILURE_PREVIEW_LIMIT = 10
_FAILURE_MESSAGE_LIMIT = 200


def _count(value):
    """A plain non-bool non-negative int, else ``None``."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _failure_preview(summary: Mapping) -> list[dict]:
    """Capped JSON-friendly preview of the failing tests."""
    failures = base.as_list(summary.get("failures"))
    result: list[dict] = []
    for failure in failures[:_FAILURE_PREVIEW_LIMIT]:
        if not isinstance(failure, Mapping):
            continue
        name = failure.get("name")
        message = failure.get("message")
        result.append(
            {
                "name": name if isinstance(name, str) else "",
                "message": (message if isinstance(message, str) else "")[:_FAILURE_MESSAGE_LIMIT],
            }
        )
    return result


def reward(task: dict, trajectory: dict) -> dict:
    """Pure reward for one ``code_exec`` task/trajectory pair."""
    env_present = base.has_env_state(trajectory)
    env = base.env_state(trajectory)
    summary_value = env.get("code_tests")
    summary = summary_value if isinstance(summary_value, Mapping) else None

    passed = _count(summary.get("passed")) if summary is not None else None
    failed = _count(summary.get("failed")) if summary is not None else None
    total = _count(summary.get("total")) if summary is not None else None
    all_passed = summary.get("all_passed") if summary is not None else None
    failures = _failure_preview(summary) if summary is not None else []

    ok = False
    reason = "missing_test_summary"
    inconsistent = None
    all_passed_derived = False

    if summary is not None:
        if not isinstance(all_passed, bool) and passed is not None and total is not None:
            all_passed = total > 0 and passed == total
            all_passed_derived = True

        if isinstance(all_passed, bool):
            counts_known = passed is not None and total is not None
            if counts_known and all_passed != (passed == total):
                inconsistent = True
            if (
                failed is not None
                and passed is not None
                and total is not None
                and failed != max(0, total - passed)
            ):
                inconsistent = True
            if inconsistent:
                reason = "inconsistent_test_summary"
            elif total is not None and total <= 0:
                reason = "no_tests_collected"
            elif all_passed is True:
                ok = True
                reason = "all_tests_passed"
            else:
                reason = "tests_failed"
        else:
            reason = "missing_all_passed"

    info = {
        "missing_env_state": not env_present,
        "missing_test_summary": summary is None,
        "all_passed": all_passed if isinstance(all_passed, bool) else None,
        "all_passed_derived": all_passed_derived,
        "total": total,
        "passed": passed,
        "failed": failed,
        "inconsistent_test_summary": inconsistent,
        "failures": failures,
        "exit_code": summary.get("exit_code") if summary is not None else None,
        "duration_s": summary.get("duration_s") if summary is not None else None,
        "truncated": summary.get("truncated") if summary is not None else None,
        "summary_source": "env_state.code_tests",
        "reason": reason,
    }
    score = MAX_SCORE if ok else 0.0
    return base.make_result(score, info, task_type=TASK_TYPE)
