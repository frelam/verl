"""Contract-level acceptance tests for ``agentic_rl_data``.

Owner: critic (T5).  These tests encode the **frozen contracts** (``schema.py``
v2.2, ``CONTRACTS.md`` v2/v2.1) plus plan §5/§8 requirements.  They are written
ahead of the T1–T4 implementations: a red test here means the contract is not
satisfied, not that the test is wrong.

**NOT EXECUTED in this session** (this machine cannot run ``pwsh``/Python:
``0xC0000142``).  User-side verification command::

    cd D:\\lhy\\verl\\verl-toolRL
    python -m pytest agentic_rl_data/tests/test_acceptance_contract.py -q

Tests use synthetic fixtures only (``cfg["records"]`` / ``tmp_path``); no
network, no HF download, no torch/verl import.

Known-red tests (documented in ``reports/review_r1.md``; expected to turn green
after the owner fixes them — do NOT weaken them):

* ``test_mt_fc_placeholder_does_not_earn_result_term``  -> review T3-02 (P1)
* ``test_mt_fc_timeout_cannot_earn_result_term``       -> review T3-01/T4-01 (P0)
* ``test_search_loader_env_config_is_constructible``   -> review cross-seat
  T1/T4-01 (P0: loader backend name/key mismatch vs SearchEnv)
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from agentic_rl_data import schema


TASK_TYPES = sorted(schema.TASK_TYPES)


# ---------------------------------------------------------------------------
# fixtures: one schema-valid task per task_type (reference 7 keys + meta 4 keys)
# ---------------------------------------------------------------------------

_WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Return the weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}
_SEARCH_TOOL = {
    "name": "search",
    "description": "Search a local Wikipedia index.",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}
_FETCH_TOOL = {
    "name": "fetch",
    "description": "Fetch a page by doc id.",
    "parameters": {
        "type": "object",
        "properties": {"doc_id": {"type": "string"}},
        "required": ["doc_id"],
    },
}
_PYTHON_TOOL = {
    "name": "python",
    "description": "Execute Python code in a sandbox.",
    "parameters": {
        "type": "object",
        "properties": {"code": {"type": "string"}},
        "required": ["code"],
    },
}


def _reference(**overrides) -> dict:
    """A ``reference`` mapping with all 7 mandatory keys present."""
    value = {
        "final_answer": None,
        "final_answers": None,
        "gt_calls": [],
        "gt_turns": None,
        "tool_returns": None,
        "unit_tests": None,
        "final_state": None,
    }
    value.update(overrides)
    return value


def _meta(**overrides) -> dict:
    """A ``meta`` mapping with all keys schema v2.2 requires."""
    value = {"max_turns": 1, "difficulty": None, "pass_rate": None, "license": "test-only"}
    value.update(overrides)
    return value


def _task(task_type: str, *, task_id: str = "fixture-1", **overrides) -> dict:
    """Build one synthetic task; ``overrides`` replace top-level/default fields."""
    env_by_type = {
        "st_fc": {"type": "none", "config": {"max_turns": 1}},
        "irrelevance": {"type": "none", "config": {"max_turns": 1}},
        "mt_fc": {"type": "mock_fc", "config": {"impl": "replay", "max_turns": 3}},
        "search_qa": {"type": "search", "config": {"max_turns": 8}},
        "math_ci": {"type": "python", "config": {"max_turns": 8}},
        "code_exec": {"type": "code_tests", "config": {"max_turns": 15}},
        "abstain": {"type": "none", "config": {"max_turns": 1}},
    }
    tools_by_type = {
        "st_fc": [_WEATHER_TOOL],
        "irrelevance": [_WEATHER_TOOL],
        "mt_fc": [_WEATHER_TOOL],
        "search_qa": [_SEARCH_TOOL, _FETCH_TOOL],
        "math_ci": [_PYTHON_TOOL],
        "code_exec": [],
        "abstain": [_WEATHER_TOOL],
    }
    reference_by_type = {
        "st_fc": _reference(gt_calls=[[{"name": "get_weather", "arguments": {"city": "Paris"}}]]),
        "irrelevance": _reference(gt_calls=[]),
        "mt_fc": _reference(
            gt_calls=[
                [{"name": "get_weather", "arguments": {"city": "Paris"}}],
                [{"name": "get_weather", "arguments": {"city": "Lyon"}}],
            ],
            tool_returns=[
                [{"name": "get_weather", "content": "22C", "ok": True}],
                [{"name": "get_weather", "content": "25C", "ok": True}],
            ],
        ),
        "search_qa": _reference(final_answer="Paris", final_answers=["Paris", "paris, france"]),
        "math_ci": _reference(final_answer="42"),
        "code_exec": _reference(unit_tests="assert add(1, 1) == 2"),
        "abstain": _reference(gt_calls=[]),
    }
    meta_by_type = {
        "st_fc": _meta(),
        "irrelevance": _meta(),
        "mt_fc": _meta(max_turns=3, gt_returns_source="reference"),
        "search_qa": _meta(max_turns=8),
        "math_ci": _meta(max_turns=8),
        "code_exec": _meta(max_turns=15),
        "abstain": _meta(),
    }
    task = {
        "id": task_id,
        "source": "synthetic",
        "task_type": task_type,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Please answer the request."},
        ],
        "tools": tools_by_type[task_type],
        "env": env_by_type[task_type],
        "reference": reference_by_type[task_type],
        "meta": meta_by_type[task_type],
    }
    task.update(overrides)
    return task


def _env_state(**overrides) -> dict:
    """A full CT C3 ``trajectory["env_state"]`` payload."""
    state = {
        "type": "mock_fc",
        "impl": "replay",
        "goal_satisfied": None,
        "replay_ok": None,
        "failed_turn": None,
        "matched_turns": None,
        "gt_turns": None,
        "checks": [],
        "state": None,
        "rendered": "",
        "num_turns": 0,
        "timeout": False,
        "code_tests": None,
        "extras": {},
    }
    state.update(overrides)
    return state


def _turn(call_name=None, arguments=None, *, call_id=None, content="", results=None) -> dict:
    calls = []
    if call_name is not None:
        calls = [{"id": call_id or "call-1", "name": call_name, "arguments": arguments or {}}]
    return {
        "step": 1,
        "assistant": {"content": content, "tool_calls": calls},
        "tool_results": list(results or []),
    }


def _result(name: str, content: str, *, call_id: str | None = "call-1") -> dict:
    return {"id": call_id, "name": name, "content": content, "ok": True}


def _trajectory(task_type: str, *, env_state: dict | None = None) -> dict:
    """A trajectory that should reach full marks for the matching fixture."""
    if task_type == "st_fc":
        return {
            "turns": [
                _turn(
                    "get_weather",
                    {"city": "Paris"},
                    call_id="call-1",
                    content="Calling get_weather.",
                    results=[_result("get_weather", "22C", call_id="call-1")],
                )
            ],
            "final_answer": "It is 22C in Paris.",
            "num_turns": 1,
            "done": True,
            "env_state": None,
        }
    if task_type == "irrelevance":
        return {
            "turns": [_turn(None, None, content="No tool is needed here.")],
            "final_answer": "No tool is needed here.",
            "num_turns": 1,
            "done": True,
            "env_state": None,
        }
    if task_type == "mt_fc":
        state = env_state or _env_state(
            replay_ok=True,
            matched_turns=2,
            gt_turns=2,
            timeout=False,
            num_turns=3,
            extras={"gt_returns_source": "reference", "degraded_replay": False},
        )
        return {
            "turns": [
                _turn(
                    "get_weather",
                    {"city": "Paris"},
                    call_id="call-1",
                    content="Turn 1.",
                    results=[_result("get_weather", "22C", call_id="call-1")],
                ),
                _turn(
                    "get_weather",
                    {"city": "Lyon"},
                    call_id="call-2",
                    content="Turn 2.",
                    results=[_result("get_weather", "25C", call_id="call-2")],
                ),
                _turn(None, None, content="Done."),
            ],
            "final_answer": None,
            "num_turns": 3,
            "done": True,
            "env_state": state,
        }
    if task_type == "search_qa":
        return {
            "turns": [_turn(None, None, content="The answer is Paris.")],
            "final_answer": "Paris",
            "num_turns": 1,
            "done": True,
            "env_state": None,
        }
    if task_type == "math_ci":
        return {
            "turns": [_turn(None, None, content="The answer is 42.")],
            "final_answer": "42",
            "num_turns": 1,
            "done": True,
            "env_state": None,
        }
    if task_type == "code_exec":
        return {
            "turns": [_turn(None, None, content="Implemented add().")],
            "final_answer": None,
            "num_turns": 1,
            "done": True,
            "env_state": _env_state(
                type="code_tests",
                impl="local",
                num_turns=1,
                code_tests={
                    "total": 2,
                    "passed": 2,
                    "failed": 0,
                    "all_passed": True,
                    "exit_code": 0,
                    "duration_s": 0.4,
                    "truncated": False,
                    "failures": [],
                },
            ),
        }
    if task_type == "abstain":
        return {
            "turns": [_turn(None, None, content="Could you tell me which city you mean?")],
            "final_answer": None,
            "num_turns": 1,
            "done": True,
            "env_state": None,
        }
    raise AssertionError(f"no trajectory fixture for {task_type}")


FIXTURES = [_task(task_type, task_id=f"fixture-{task_type}") for task_type in TASK_TYPES]


# ---------------------------------------------------------------------------
# 1. registry + schema contract
# ---------------------------------------------------------------------------


def test_registry_covers_every_task_type() -> None:
    """Every schema task_type has exactly one registered reward (plan §5)."""
    from agentic_rl_data import rewards

    assert isinstance(rewards.REWARDS, dict)
    assert set(rewards.REWARDS) == set(schema.TASK_TYPES), (
        f"registry/schema mismatch: missing={sorted(set(schema.TASK_TYPES) - set(rewards.REWARDS))} "
        f"extra={sorted(set(rewards.REWARDS) - set(schema.TASK_TYPES))}"
    )
    for task_type in TASK_TYPES:
        assert callable(rewards.get_reward(task_type))


@pytest.mark.parametrize("task", FIXTURES, ids=lambda t: t["task_type"])
def test_fixture_tasks_are_schema_valid(task: dict) -> None:
    assert schema.validate_task(task) == [], f"fixture task must be schema-valid: {task['id']}"


@pytest.mark.parametrize("task", FIXTURES, ids=lambda t: t["task_type"])
def test_fixture_trajectories_are_schema_valid(task: dict) -> None:
    trajectory = _trajectory(task["task_type"])
    assert schema.validate_trajectory(trajectory) == [], (
        f"fixture trajectory must be schema-valid: {task['task_type']}"
    )


# ---------------------------------------------------------------------------
# 2. reward payload contract (plan §5: always validate_reward_result-valid)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", FIXTURES, ids=lambda t: t["task_type"])
def test_reward_payload_satisfies_frozen_contract(task: dict) -> None:
    from agentic_rl_data import rewards

    reward = rewards.get_reward(task["task_type"])
    result = reward(task, _trajectory(task["task_type"]))
    assert schema.validate_reward_result(result) == [], (
        f"{task['task_type']} reward violates {{'score','info'}} contract: {result!r}"
    )
    assert isinstance(result["score"], (int, float)) and math.isfinite(float(result["score"]))
    assert 0.0 <= float(result["score"]) <= 1.0, (
        f"{task['task_type']} score {result['score']} outside frozen [0, 1]"
    )
    assert isinstance(result["info"], dict)


@pytest.mark.parametrize("task", FIXTURES, ids=lambda t: t["task_type"])
def test_gt_trajectory_gets_full_marks(task: dict) -> None:
    """plan §8 #3: the ground-truth trajectory must score full marks.

    Tolerance is deliberate: plan §5 weights (0.8 + 0.1 + 0.1) must not be
    asserted with ``== 1.0`` (critic Q-21).
    """
    from agentic_rl_data import rewards

    result = rewards.get_reward(task["task_type"])(task, _trajectory(task["task_type"]))
    assert math.isclose(float(result["score"]), 1.0, abs_tol=1e-6), (
        f"{task['task_type']} GT trajectory scored {result['score']}, expected 1.0; info={result['info']}"
    )


# ---------------------------------------------------------------------------
# 3. negative / boundary contract checks
# ---------------------------------------------------------------------------


def test_validate_task_rejects_missing_reference_key() -> None:
    task = copy.deepcopy(FIXTURES[0])
    task["reference"].pop("tool_returns")
    errors = schema.validate_task(task)
    assert any("tool_returns" in error for error in errors), errors


def test_validate_task_rejects_none_gt_calls() -> None:
    task = copy.deepcopy(FIXTURES[0])
    task["reference"]["gt_calls"] = None
    assert schema.validate_task(task), "gt_calls=None must be rejected (use [])"


def test_validate_task_requires_mt_fc_static_marker() -> None:
    task = copy.deepcopy(_task("mt_fc"))
    task["meta"].pop("gt_returns_source")
    errors = schema.validate_task(task)
    assert any("gt_returns_source" in error for error in errors), errors


def test_validate_task_rejects_plural_static_marker() -> None:
    """schema v2.2 enum is singular ``reference``; ``references`` must be rejected.

    This is the exact failure mode of ``loaders/common.py`` at the time of
    writing (review T1-01).
    """
    task = copy.deepcopy(_task("mt_fc"))
    task["meta"]["gt_returns_source"] = "references"
    errors = schema.validate_task(task)
    assert any("gt_returns_source" in error for error in errors), errors


def test_validate_task_rejects_env_type_mismatch() -> None:
    task = copy.deepcopy(_task("search_qa"))
    task["env"]["type"] = "python"
    assert schema.validate_task(task), "search_qa with env.type=python must be rejected"


def test_validate_reward_result_rejects_out_of_range_scores() -> None:
    base = {"score": 0.5, "info": {"ok": True}}
    assert schema.validate_reward_result(base) == []
    for bad in (-0.1, 1.5, float("nan"), float("inf")):
        result = {"score": bad, "info": {}}
        assert schema.validate_reward_result(result), f"score {bad} must be rejected"
    assert schema.validate_reward_result({"score": 0.5, "info": None}), "info=None must be rejected"


def test_validate_task_rejects_unknown_gt_returns_source() -> None:
    task = copy.deepcopy(_task("mt_fc"))
    task["meta"]["gt_returns_source"] = "config"  # legal value, control
    assert schema.validate_task(task) == []
    task["meta"]["gt_returns_source"] = "tool_returns"  # not in the frozen enum
    assert schema.validate_task(task), "unknown enum value must be rejected"


# ---------------------------------------------------------------------------
# 4. env_state shape contract (CT C3; K6)
# ---------------------------------------------------------------------------


def test_env_state_helper_has_frozen_keys() -> None:
    from agentic_rl_data.envs.base import ENV_STATE_KEYS, make_env_state

    state = make_env_state("mock_fc", "replay")
    assert tuple(sorted(state)) == tuple(sorted(ENV_STATE_KEYS))
    assert set(ENV_STATE_KEYS) >= {
        "type",
        "impl",
        "goal_satisfied",
        "replay_ok",
        "failed_turn",
        "matched_turns",
        "gt_turns",
        "checks",
        "state",
        "rendered",
        "num_turns",
        "timeout",
        "code_tests",
        "extras",
    }


# ---------------------------------------------------------------------------
# 5. requirement-level checks that currently FAIL (review findings; do not weaken)
# ---------------------------------------------------------------------------


def test_mt_fc_timeout_cannot_earn_result_term() -> None:
    """CT C1 / plan §4.1: 超限视为失败 -> timeout cannot earn the 0.8 result term.

    Currently RED: reward does not read ``env_state.timeout`` (review T3-02)
    and the stateful env does not clear ``goal_satisfied`` on timeout
    (review T4-01).  Expected fixed score: <= 0.2 (only auxiliary terms).
    """
    from agentic_rl_data import rewards

    task = _task("mt_fc")
    state = _env_state(
        goal_satisfied=True,
        replay_ok=True,
        matched_turns=2,
        gt_turns=2,
        timeout=True,
        num_turns=3,
    )
    result = rewards.get_reward("mt_fc")(task, _trajectory("mt_fc", env_state=state))
    assert float(result["score"]) <= 0.2 + 1e-9, (
        "timeout must not earn the 0.8 result term (CT C1 / plan §4.1); "
        f"got {result['score']} (info={result['info']})"
    )


def test_mt_fc_placeholder_does_not_earn_result_term() -> None:
    """CT C6: placeholder replay must not earn the 0.8 result term.

    Currently RED: reward only sets ``info.degraded_replay`` but still pays
    0.8 for ``replay_ok`` (review T3-02).  Expected fixed score: <= 0.2.
    """
    from agentic_rl_data import rewards

    task = _task("mt_fc")
    state = _env_state(
        replay_ok=True,
        matched_turns=2,
        gt_turns=2,
        extras={"gt_returns_source": "placeholder"},
    )
    result = rewards.get_reward("mt_fc")(task, _trajectory("mt_fc", env_state=state))
    assert float(result["score"]) <= 0.2 + 1e-9, (
        "degraded replay (placeholder) must not earn the 0.8 result term; "
        f"got {result['score']} (info={result['info']})"
    )


# ---------------------------------------------------------------------------
# 6. loaders: contract + schema-valid output (synthetic records, no network)
# ---------------------------------------------------------------------------


_MT_RECORD = {
    "id": "synthetic-mt-1",
    "tools": [_WEATHER_TOOL],
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Plan a two-city weather check."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}],
        },
        {"role": "tool", "name": "get_weather", "content": "22C"},
        {"role": "assistant", "content": "Paris is 22C."},
    ],
    "final_answer": "Paris is 22C.",
}


def test_loader_modules_expose_contract() -> None:
    import importlib

    loaders_dir = Path("agentic_rl_data/loaders")
    if not loaders_dir.exists():  # pragma: no cover - path safety for other cwds
        loaders_dir = Path(__file__).resolve().parents[1] / "loaders"
    names = sorted(
        path.stem
        for path in loaders_dir.glob("*.py")
        if path.stem not in {"common", "__init__"}
    )
    assert names, f"no loader modules under {loaders_dir}"
    for name in names:
        module = importlib.import_module(f"agentic_rl_data.loaders.{name}")
        assert isinstance(getattr(module, "SOURCE", None), str) and module.SOURCE, (
            f"{name} must expose a non-empty SOURCE"
        )
        assert callable(getattr(module, "load_tasks", None)), f"{name} must expose load_tasks(cfg)"


def test_loader_mt_tasks_schema_valid() -> None:
    """Multi-turn loaders must emit schema-valid tasks from a synthetic record.

    Currently RED for a P0 reason (review T1-01): ``loaders/common.py:999``
    writes the plural ``"references"`` into ``meta.gt_returns_source``, which
    schema v2.2 rejects, and ``load_mt_tasks_from_records`` swallows the error
    with ``except Exception ... continue`` -> zero mt_fc tasks emitted.
    """
    from agentic_rl_data.loaders.common import load_mt_tasks_from_records

    tasks = list(
        load_mt_tasks_from_records(
            [_MT_RECORD],
            source="synthetic",
            cfg={"max_turns": 1, "max_samples": 1},
        )
    )
    assert tasks, "synthetic multi-turn record produced no task (silent drop)"
    for task in tasks:
        assert schema.validate_task(task) == [], f"loader emitted an invalid task: {task['id']} {schema.validate_task(task)}"
        assert task["task_type"] == "mt_fc"
        assert task["meta"].get("gt_returns_source") in schema.GT_RETURNS_SOURCES


def test_single_turn_loader_schema_valid() -> None:
    from agentic_rl_data.loaders import apigen, hammer

    apigen_task = next(
        apigen.load_tasks(
            {
                "records": [
                    {
                        "id": "synthetic-apigen-1",
                        "query": "What is the weather in Paris?",
                        "tools": json.dumps([_WEATHER_TOOL]),
                        "answers": json.dumps([{"name": "get_weather", "arguments": {"city": "Paris"}}]),
                    }
                ],
                "max_samples": 1,
            }
        )
    )
    assert schema.validate_task(apigen_task) == []

    hammer_task = next(
        hammer.load_tasks(
            {
                "records": [
                    {
                        "id": "synthetic-hammer-1",
                        "query": "Tell me a joke.",
                        "tools": json.dumps([_WEATHER_TOOL]),
                        "answers": "[]",
                    }
                ],
                "max_samples": 1,
            }
        )
    )
    assert schema.validate_task(hammer_task) == []
    assert hammer_task["task_type"] == "irrelevance"


# ---------------------------------------------------------------------------
# 7. mixture reproducibility (plan §8 #1) + static placeholder marker (D11)
# ---------------------------------------------------------------------------


def _synthetic_mixture_config(raw_dir: Path) -> dict:
    """A minimal 5-category config used only by this reproducibility test."""
    source_names = ["single_turn", "multi_turn", "search_qa", "math_ci", "code_exec"]
    categories = {
        name: {
            "share": 0.2,
            "heldout": 200,
            "turn_cap": 15,
            "on_unavailable": "fail",
            "sources": [{"name": f"sync_{name}", "weight": 1.0}],
        }
        for name in source_names
    }
    return {
        "version": 1,
        "seed": 12345,
        "total_train": 50,
        "paths": {"raw_dir": str(raw_dir)},
        "output": {"dir": "ignored", "train": "train.jsonl", "heldout": "heldout.jsonl", "stats": "mixture_stats.json"},
        "passrate": {"enabled": False},
        "heldout": {"per_category": 200, "group_aware": False, "oversample": 1},
        "filters": {"max_gt_turns": 15, "invalid_schema": "drop"},
        "curriculum": {"enabled": False},
        "sources": {
            f"sync_{name}": {
                "file": f"{name}.jsonl",
                "task_type": "st_fc",
                "required": True,
                "license": None,
                "note": "synthetic for tests",
            }
            for name in source_names
        },
        "categories": categories,
    }


def _write_synthetic_raw(raw_dir: Path, rows_per_source: int = 240) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    source_names = ["single_turn", "multi_turn", "search_qa", "math_ci", "code_exec"]
    for source_name in source_names:
        tasks = []
        for index in range(rows_per_source):
            task_id = f"{source_name}-{index:04d}"
            task = _task("st_fc", task_id=task_id)
            # Stable-dedup needs a unique normalised user query per row.
            task["messages"][1]["content"] = f"Question {task_id}: what is the weather?"
            tasks.append(task)
        schema.dump_jsonl(tasks, raw_dir / f"{source_name}.jsonl")


def test_mixture_reproducible_twice(tmp_path: Path) -> None:
    """plan §8 #1: two builds of the same inputs must be byte-identical."""
    from agentic_rl_data.scripts import build_mixture as bm

    raw_dir = tmp_path / "raw"
    _write_synthetic_raw(raw_dir)
    cfg = bm.parse_config(_synthetic_mixture_config(raw_dir))

    out_a = tmp_path / "out_a"
    stats_a = bm.build_mixture(cfg, raw_dir=raw_dir, out_dir=out_a, use_loaders=False)
    out_b = tmp_path / "out_b"
    stats_b = bm.build_mixture(cfg, raw_dir=raw_dir, out_dir=out_b, use_loaders=False)

    train_a = (out_a / cfg.train_name).read_bytes()
    train_b = (out_b / cfg.train_name).read_bytes()
    held_a = (out_a / cfg.heldout_name).read_bytes()
    held_b = (out_b / cfg.heldout_name).read_bytes()
    assert train_a == train_b, "train.jsonl is not byte-reproducible across two builds"
    assert held_a == held_b, "heldout.jsonl is not byte-reproducible across two builds"
    assert stats_a["outputs"]["train"]["sha256"] == stats_b["outputs"]["train"]["sha256"]
    assert stats_a["total_train_actual"] == stats_b["total_train_actual"]
    assert len(train_a.splitlines()) == stats_a["total_train_actual"]


def test_mixture_offline_marker_honours_meta() -> None:
    """D11/CT C6: T2 must filter placeholders offline from ``meta`` (not env).

    Currently RED (review T2-02): ``build_mixture.gt_returns_source()`` derives
    the value from the data structures and ignores the frozen ``meta`` marker.
    """
    from agentic_rl_data.scripts import build_mixture as bm

    task = _task("mt_fc")
    task["meta"]["gt_returns_source"] = "placeholder"
    task["reference"]["tool_returns"] = [
        [{"name": "get_weather", "content": "22C", "ok": True}],
        [{"name": "get_weather", "content": "25C", "ok": True}],
    ]
    kind, origin = bm.gt_returns_source(task)
    assert kind == "placeholder", (
        "offline placeholder filtering must honour meta.gt_returns_source "
        f"(D11/CT C6), got {kind!r}"
    )
    assert origin == "static", f"static meta declaration must win, got origin={origin!r}"
    assert kind in schema.GT_RETURNS_SOURCES


def test_search_loader_env_config_is_constructible() -> None:
    """Cross-seat contract: the env config produced by T1 loaders must be
    accepted by T4's env factory (``build_env``).

    Currently RED (review T1/T4 cross-seat): ``loaders/common.py:608-614``
    emits ``backend="local_wikipedia_e5"`` + ``retriever={"url","topk"}`` while
    ``envs/search.py:1144-1209`` accepts only ``inmemory|http|e5`` and keys
    ``url`` / ``top_k`` -> every ``search_qa`` task raises ``EnvError`` on env
    construction.
    """
    from agentic_rl_data.envs.base import build_env
    from agentic_rl_data.loaders.common import (
        SEARCH_SYSTEM,
        SEARCH_TOOLS,
        build_messages,
        build_reference,
        build_task,
        env_search,
    )

    task = build_task(
        source="synthetic",
        task_id="search-env-contract-1",
        task_type="search_qa",
        messages=build_messages(SEARCH_SYSTEM, ["What is the capital of France?"]),
        tools=SEARCH_TOOLS,
        env=env_search(max_turns=8),
        reference=build_reference(final_answer="Paris", final_answers=["Paris"], gt_calls=[]),
        max_turns=8,
    )
    env = build_env(task)  # must not raise EnvError
    assert getattr(env, "env_type", None) == "search"
