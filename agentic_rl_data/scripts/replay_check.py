"""Plan §3 step 5: replay every ``mock_fc`` task's ground-truth trajectory.

For each task the checker instantiates its mock environment, replays
``reference.gt_calls`` turn by turn (injecting the ground-truth tool returns),
then finishes with the reference final answer.  It asserts:

1. the environment reaches a terminal state without timing out;
2. the implementation's terminal verdict holds — ``replay_ok is True`` for the
   replay implementation, or ``goal_satisfied is True`` /
   ``reference.final_state`` deep-equal to ``env_state["state"]`` for the
   stateful implementation (CONTRACTS.md C4);
3. if ``agentic_rl_data.rewards.get_reward`` is importable, the reward returns
   a full score (``>= --min-score``, default 1.0); otherwise the replay verdict
   itself is the fallback and the report marks ``reward_used=False``.

It never silently degrades:

* schema-invalid tasks (including missing ``reference`` v2 keys) are reported
  as failures, not skipped (CONTRACTS.md C7);
* ``env_state.extras["gt_returns_source"]`` is recorded per task and the
  placeholder ratio is summarised (CONTRACTS.md C6 / D7), so the mixture step
  can filter or down-weight degraded replay data.

Usage::

    python -m agentic_rl_data.scripts.replay_check \
        --tasks data/raw_unified/funreason_mt.jsonl data/raw_unified/apigen_mt.jsonl \
        --out reports/replay_check.json

Failure list is printed to stdout; exit code 1 when any task fails.  This
checker is read-only with respect to the data; it writes only the optional
JSON report.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .. import schema
from ..envs.base import BaseEnv, EnvError, build_env, validate_env_state
from ..envs.mock_fc import MISSING_RETURN_PREFIX, deep_equal

#: task types that can carry a mock_fc environment (plan §3 step 5).
DEFAULT_TASK_TYPES = ("mt_fc",)
SCORE_EPSILON = 1e-9


# ---------------------------------------------------------------------------
# reward integration (rewards/ is written by another seat and may be absent)
# ---------------------------------------------------------------------------


def load_reward() -> tuple[Any, str | None]:
    """Return ``(reward_callable | None, note)``.

    Accepts either ``get_reward(task, trajectory)`` or a ``REWARDS`` /
    ``REWARD_FUNCTIONS`` registry mapping ``task_type -> callable``.
    """
    try:
        from .. import rewards as rewards_module  # noqa: F401  (optional)
    except Exception as exc:  # ImportError, or a half-written parallel module
        return None, f"rewards module unavailable: {type(exc).__name__}: {exc}"

    for name in ("get_reward", "reward_task", "compute_reward"):
        function = getattr(rewards_module, name, None)
        if callable(function):
            return function, f"agentic_rl_data.rewards.{name}"

    registry = None
    for name in ("REWARDS", "REWARD_FUNCTIONS", "REGISTRY"):
        candidate = getattr(rewards_module, name, None)
        if isinstance(candidate, Mapping):
            registry = candidate
            break
    if isinstance(registry, Mapping):

        def dispatch(task: Mapping[str, Any], trajectory: Mapping[str, Any]) -> Any:
            function = registry.get(task.get("task_type"))
            if function is None:
                raise KeyError(
                    f"no reward registered for task_type {task.get('task_type')!r}"
                )
            return function(task, trajectory)

        return dispatch, "agentic_rl_data.rewards registry"

    return None, "rewards module has neither get_reward nor a REWARDS registry"


# ---------------------------------------------------------------------------
# one task
# ---------------------------------------------------------------------------


def _failure(
    task: Mapping[str, Any] | None,
    stage: str,
    reason: str,
    **extra: Any,
) -> dict[str, Any]:
    task = task if isinstance(task, Mapping) else {}
    record: dict[str, Any] = {
        "task_id": task.get("id"),
        "source": task.get("source"),
        "task_type": task.get("task_type"),
        "ok": False,
        "stage": stage,
        "reason": reason,
    }
    for key, value in extra.items():
        record[key] = value
    return record


def _reference_final_answer(reference: Mapping[str, Any]) -> str:
    answer = reference.get("final_answer")
    if isinstance(answer, str) and answer.strip():
        return answer
    answers = reference.get("final_answers")
    if isinstance(answers, list):
        for item in answers:
            if isinstance(item, str) and item.strip():
                return item
    return ""


def _mock_impl(task: Mapping[str, Any], forced: str) -> str | None:
    """Resolve which mock_fc implementation to use (``None`` = build_env auto)."""
    forced = (forced or "auto").strip().lower()
    if forced == "replay":
        return "replay"
    if forced == "stateful":
        return "stateful"
    return None


def check_task(
    task: Mapping[str, Any],
    *,
    impl: str = "auto",
    min_score: float = 1.0,
    reward_fn: Any | None = None,
    strict_tool_errors: bool = True,
) -> dict[str, Any]:
    """Replay one task; return a report record (``ok`` field is the verdict)."""
    record: dict[str, Any] = {
        "task_id": None,
        "source": None,
        "task_type": None,
        "impl": None,
        "ok": False,
        "stage": None,
        "reason": None,
        "reward_used": False,
        "reward_backend": None,
        "score": None,
        "replay_ok": None,
        "goal_satisfied": None,
        "gt_returns_source": None,
        "meta_gt_returns_source": None,
        "gt_returns_source_mismatch": False,
        "gt_returns_missing": 0,
        "degraded_replay": False,
        "warnings": [],
    }
    if not isinstance(task, Mapping):
        return _failure(None, "schema", "task is not a mapping")

    record["task_id"] = task.get("id")
    record["source"] = task.get("source")
    record["task_type"] = task.get("task_type")

    schema_errors = schema.validate_task(task)
    if schema_errors:
        return _failure(
            task,
            "schema",
            "schema validation failed",
            details=schema_errors[:10],
        )

    env_type = (task.get("env") or {}).get("type")
    if env_type != "mock_fc":
        return _failure(task, "skip", f"env.type is {env_type!r}, not 'mock_fc'")
    task_type = task.get("task_type")
    if task_type not in DEFAULT_TASK_TYPES and task_type != "st_fc":
        # replay checking only applies to multi-turn / single-turn tool calling.
        return _failure(task, "skip", f"task_type {task_type!r} has no gt trajectory")

    forced_impl = _mock_impl(task, impl)
    config = (task.get("env") or {}).get("config")
    config = config if isinstance(config, Mapping) else {}
    tolerate_tool_errors = bool(config.get("tolerate_tool_errors", not strict_tool_errors))

    reference = task["reference"]
    groups = schema.gt_call_groups(task)

    try:
        env: BaseEnv = build_env(task, impl=forced_impl)
    except EnvError as exc:
        return _failure(task, "config", f"could not build env: {exc}")
    except Exception as exc:  # defensive
        return _failure(
            task, "config", f"env construction raised {type(exc).__name__}: {exc}"
        )

    record["impl"] = getattr(env, "impl", None)
    tool_errors: list[dict[str, Any]] = []
    missing_returns = 0
    is_replay = getattr(env, "impl", None) == "replay"
    try:
        env.reset(task)
        for turn_index, group in enumerate(groups):
            action = {"content": "", "tool_calls": [dict(call) for call in group]}
            obs, done, info = env.step(action)
            for result in obs.get("tool_results") or []:
                if result.get("ok", True):
                    continue
                content = str(result.get("content", ""))
                # Missing GT returns are degraded data, not a tool-execution
                # error: they are counted and reported (CONTRACTS C6 / T4-02).
                if is_replay and content.startswith(MISSING_RETURN_PREFIX):
                    missing_returns += 1
                    continue
                tool_errors.append(
                    {
                        "turn": turn_index,
                        "tool": result.get("name"),
                        "content": content[:200],
                    }
                )
            if done:
                if env.timeout:
                    return _failure(
                        task,
                        "timeout",
                        f"episode timed out after {turn_index + 1} GT turn(s) "
                        f"(max_turns={env.max_turns}, gt_turns={len(groups)})",
                        gt_returns_source=_gt_returns_source(env),
                    )
                return _failure(
                    task,
                    "early_termination",
                    f"episode terminated while replaying GT turn {turn_index}",
                    replay_ok=env.replay_ok,
                    gt_returns_source=_gt_returns_source(env),
                )
        if not env.done:
            final_answer = _reference_final_answer(reference)
            _, done, _ = env.step({"content": final_answer, "tool_calls": []})
            if not done:
                return _failure(
                    task,
                    "no_termination",
                    "episode did not terminate after the reference final answer",
                )
    except EnvError as exc:
        return _failure(task, "env", f"env protocol error: {exc}")
    except Exception as exc:  # defensive: one bad task must not stop the run
        return _failure(
            task,
            "env",
            f"env raised {type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(limit=3),
        )
    finally:
        try:
            env.close()
        except Exception:  # pragma: no cover - defensive
            pass

    if env.timeout:
        return _failure(task, "timeout", "final answer turn timed out")

    if tool_errors and not tolerate_tool_errors:
        return _failure(
            task,
            "tool_error",
            f"{len(tool_errors)} GT tool call(s) returned ok=False during replay",
            tool_errors=tool_errors[:5],
            gt_returns_source=_gt_returns_source(env),
        )
    if tool_errors:
        record["warnings"].append(
            f"{len(tool_errors)} GT tool call(s) returned ok=False (tolerated; "
            "check env.config.tolerate_tool_errors)"
        )
    if missing_returns:
        record["gt_returns_missing"] = missing_returns
        record["warnings"].append(
            f"{missing_returns} GT tool call(s) had no ground-truth return; "
            "the env injected an ok=False placeholder (CONTRACTS C6)"
        )

    env_state = env.env_state or {}
    extras = env_state.get("extras") if isinstance(env_state.get("extras"), Mapping) else {}
    record["goal_satisfied"] = env_state.get("goal_satisfied")
    record["replay_ok"] = env_state.get("replay_ok")
    record["gt_returns_source"] = _gt_returns_source(env)
    record["gt_returns_missing"] = int(extras.get("gt_returns_missing") or 0)
    record["degraded_replay"] = bool(extras.get("degraded_replay"))
    record["meta_gt_returns_source"] = (task.get("meta") or {}).get(
        "gt_returns_source"
    )
    declared_source = record["meta_gt_returns_source"]
    if (
        isinstance(declared_source, str)
        and isinstance(record["gt_returns_source"], str)
        and declared_source != record["gt_returns_source"]
    ):
        # CONTRACTS C6 v2.1: runtime wins, but the mismatch must be visible.
        record["gt_returns_source_mismatch"] = True
        record["warnings"].append(
            f"meta.gt_returns_source={declared_source!r} but runtime source="
            f"{record['gt_returns_source']!r}; runtime wins (CONTRACTS C6)"
        )

    trajectory_errors = schema.validate_trajectory(env.trajectory())
    if trajectory_errors:
        return _failure(
            task,
            "trajectory",
            "normalised trajectory is schema-invalid",
            details=trajectory_errors[:10],
        )
    env_state_errors = validate_env_state(env_state)
    if env_state_errors:
        return _failure(
            task,
            "env_state",
            "env_state breaks CONTRACTS C3",
            details=env_state_errors[:10],
        )
    trajectory = env.trajectory()

    # ---- terminal verdict (CONTRACTS.md C4) ----------------------------
    if env.impl == "replay":
        if env_state.get("replay_ok") is not True:
            replay_info = (env_state.get("extras") or {}).get("last_replay") or {}
            return _failure(
                task,
                "replay",
                "GT trajectory did not match completely: "
                + str(replay_info.get("reason") or "replay_ok is not True"),
                failed_turn=env_state.get("failed_turn"),
                matched_turns=env_state.get("matched_turns"),
                gt_turns=env_state.get("gt_turns"),
                gt_returns_source=record["gt_returns_source"],
            )
    else:
        goal_satisfied = env_state.get("goal_satisfied")
        final_state = reference.get("final_state")
        actual_state = env_state.get("state")
        if goal_satisfied is not None:
            if goal_satisfied is not True:
                return _failure(
                    task,
                    "goal",
                    "stateful env goal_satisfied is False",
                    checks=env_state.get("checks"),
                    gt_returns_source=record["gt_returns_source"],
                )
        elif isinstance(final_state, Mapping) and isinstance(actual_state, Mapping):
            if not deep_equal(final_state, actual_state, loose=True):
                return _failure(
                    task,
                    "final_state",
                    "env_state['state'] does not deep-equal reference.final_state",
                    expected=final_state,
                    actual=actual_state,
                    gt_returns_source=record["gt_returns_source"],
                )
        else:
            record["warnings"].append(
                "no terminal assertion (goal_satisfied/final_state) and no replay_ok; "
                "only tool replay was validated"
            )

    # ---- reward full-score assertion (plan §3 step 5) -----------------
    if reward_fn is not None:
        try:
            result = reward_fn(task, trajectory)
        except Exception as exc:
            return _failure(
                task,
                "reward",
                f"reward raised {type(exc).__name__}: {exc}",
                gt_returns_source=record["gt_returns_source"],
            )
        reward_errors = schema.validate_reward_result(result)
        if reward_errors:
            return _failure(
                task,
                "reward",
                "reward result is schema-invalid",
                details=reward_errors[:10],
            )
        score = float(result["score"])
        record["reward_used"] = True
        record["reward_backend"] = getattr(reward_fn, "__qualname__", "reward")
        record["score"] = score
        if score + SCORE_EPSILON < float(min_score):
            return _failure(
                task,
                "reward_score",
                f"reward score {score:.4f} < required {min_score:.4f}",
                reward_info=result.get("info"),
                gt_returns_source=record["gt_returns_source"],
            )

    record["ok"] = True
    record["stage"] = "passed"
    record["reason"] = "GT trajectory replays and the terminal verdict holds"
    return record


def _gt_returns_source(env: BaseEnv) -> str | None:
    state = env.env_state or {}
    extras = state.get("extras") if isinstance(state.get("extras"), Mapping) else {}
    source = extras.get("gt_returns_source")
    return str(source) if isinstance(source, str) else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _expand_task_paths(patterns: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    missing = [str(path) for path in unique if not path.exists()]
    if missing:
        raise SystemExit(f"[replay_check] input file(s) not found: {missing}")
    return unique


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agentic_rl_data.scripts.replay_check",
        description=(
            "Plan §3 step 5: replay mock_fc ground-truth trajectories and assert "
            "the reward is full."
        ),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        required=True,
        help="JSONL files (globs accepted) with unified tasks",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="write a JSON report to this path (failure list included)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=1.0,
        help="minimum reward score that counts as 'full' (default 1.0)",
    )
    parser.add_argument(
        "--impl",
        choices=("auto", "replay", "stateful"),
        default="auto",
        help="mock_fc implementation to exercise (default: task config / auto)",
    )
    parser.add_argument(
        "--reward",
        choices=("auto", "none"),
        default="auto",
        help="use agentic_rl_data.rewards.get_reward when importable (default auto)",
    )
    parser.add_argument(
        "--task-types",
        default=",".join(DEFAULT_TASK_TYPES),
        help="comma-separated task types to check (default: mt_fc)",
    )
    parser.add_argument("--limit", type=int, default=None, help="max tasks to check")
    parser.add_argument(
        "--allow-tool-errors",
        action="store_true",
        help="tolerate GT tool calls that return ok=False (recorded as warnings)",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=50,
        help="maximum failures kept in the JSON report (all are counted)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="only print the summary line",
    )
    args = parser.parse_args(argv)

    task_types = tuple(
        item.strip() for item in str(args.task_types).split(",") if item.strip()
    ) or DEFAULT_TASK_TYPES

    reward_fn = None
    reward_note: str | None = None
    if args.reward == "auto":
        reward_fn, reward_note = load_reward()
    else:
        reward_note = "disabled by --reward none"

    paths = _expand_task_paths(args.tasks)
    total = 0
    passed = 0
    failed = 0
    skipped = 0
    failures: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    degraded_count = 0
    placeholder_count = 0
    warnings_count = 0
    reward_used_count = 0
    gt_returns_missing_total = 0
    source_mismatch_count = 0

    for path in paths:
        for task in schema.load_jsonl(path):
            if args.limit is not None and total >= int(args.limit):
                break
            task_type = task.get("task_type") if isinstance(task, Mapping) else None
            if task_type not in task_types:
                skipped += 1
                continue
            total += 1
            record = check_task(
                task,
                impl=args.impl,
                min_score=float(args.min_score),
                reward_fn=reward_fn,
                strict_tool_errors=not bool(args.allow_tool_errors),
            )
            if record.get("stage") == "skip":
                skipped += 1
                total -= 1
                continue
            source = record.get("gt_returns_source")
            if isinstance(source, str):
                source_counts[source] = source_counts.get(source, 0) + 1
                if source == "placeholder":
                    placeholder_count += 1
            gt_returns_missing_total += int(record.get("gt_returns_missing") or 0)
            if record.get("gt_returns_source_mismatch"):
                source_mismatch_count += 1
            if record.get("degraded_replay"):
                degraded_count += 1
            if record.get("reward_used"):
                reward_used_count += 1
            for warning in record.get("warnings") or []:
                warnings_count += 1
                if not args.quiet:
                    print(f"[warn] {record.get('task_id')}: {warning}")
            if record.get("ok"):
                passed += 1
                continue
            failed += 1
            failures.append(record)
            if not args.quiet:
                print(
                    f"[FAIL] {record.get('task_id')} "
                    f"({record.get('source')}, {record.get('task_type')}, "
                    f"impl={record.get('impl') or args.impl}): "
                    f"[{record.get('stage')}] {record.get('reason')}"
                )
        if args.limit is not None and total >= int(args.limit):
            break

    source_total = sum(source_counts.values())
    placeholder_ratio = (
        round(placeholder_count / source_total, 4) if source_total else 0.0
    )

    summary = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "reward_used": reward_used_count,
        "reward_backend": reward_note,
        "gt_returns_source_counts": dict(sorted(source_counts.items())),
        "gt_returns_source_mismatch": source_mismatch_count,
        "gt_returns_missing": gt_returns_missing_total,
        "placeholder_count": placeholder_count,
        "placeholder_ratio": placeholder_ratio,
        "degraded_replay_count": degraded_count,
        "warnings": warnings_count,
        "config": {
            "impl": args.impl,
            "min_score": float(args.min_score),
            "task_types": list(task_types),
            "allow_tool_errors": bool(args.allow_tool_errors),
            "inputs": [str(path) for path in paths],
        },
    }

    print(
        f"[replay_check] tasks={total} passed={passed} failed={failed} "
        f"skipped={skipped} reward_used={reward_used_count} "
        f"placeholder_ratio={placeholder_ratio}"
    )
    if source_counts:
        print(f"[replay_check] gt_returns_source: {dict(sorted(source_counts.items()))}")
    if reward_note:
        print(f"[replay_check] reward backend: {reward_note}")
    if failures and args.max_failures > 0:
        kept = failures[: int(args.max_failures)]
        print(f"[replay_check] {len(failures)} failure(s), showing {len(kept)}:")
        for record in kept:
            print(
                f"  - {record.get('task_id')} [{record.get('stage')}] "
                f"{record.get('reason')}"
            )

    if args.out:
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "failures": failures[: max(1, int(args.max_failures))] if failures else [],
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"[replay_check] report written to {args.out}")

    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = ["DEFAULT_TASK_TYPES", "check_task", "load_reward", "main"]
