"""Unit tests for the data pipeline pure functions (no network, no heavy deps).

Covers: turn counting (CONTRACTS C9), exact dedup, MinHash near dedup,
decontamination n-grams / index / matching, and the build_raw pure layer.

Run from the repository root::

    python -m pytest agentic_rl_data/tests/test_pipeline_units.py -q

These tests were *not executed in the authoring sandbox* (pwsh / python cannot
start, 0xC0000142); run them in a working shell.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from agentic_rl_data.schema import task_dedup_key, validate_task

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    """Import ``agentic_rl_data/scripts/<name>.py`` without a scripts package."""
    path = REPO_ROOT / "agentic_rl_data" / "scripts" / f"{name}.py"
    module_name = f"dsh_pipeline_{name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


filter_rounds = _load_script("filter_rounds")
dedup = _load_script("dedup")
decontam = _load_script("decontam")
build_raw = _load_script("build_raw")

WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


def _task(
    task_id: str,
    user: str,
    *,
    tools=None,
    gt_calls=None,
    expected_turns=None,
    gt_turns=None,
    source="apigen",
    task_type="st_fc",
):
    reference = {
        "final_answer": None,
        "final_answers": None,
        "gt_calls": gt_calls if gt_calls is not None else [[]],
        "gt_turns": gt_turns,
        "tool_returns": None,
        "unit_tests": None,
        "final_state": None,
    }
    meta = {"max_turns": 15, "difficulty": None, "pass_rate": None, "license": None}
    if expected_turns is not None:
        meta["expected_turns"] = expected_turns
    return {
        "id": task_id,
        "source": source,
        "task_type": task_type,
        "messages": [{"role": "user", "content": user}],
        "tools": tools or [],
        "env": {"type": "none", "config": {}},
        "reference": reference,
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# turn filter (CONTRACTS C9)
# ---------------------------------------------------------------------------


def test_effective_turns_uses_max_of_all_sources():
    assert filter_rounds.effective_turns(_task("t1", "hi")) == 1
    task = _task(
        "t2",
        "hi",
        gt_calls=[[{"name": "a"}], [{"name": "b"}]],
        expected_turns=5,
        gt_turns=[{"step": 1}, {"step": 2}, {"step": 3}, {"step": 4}, {"step": 5}, {"step": 6}],
    )
    assert filter_rounds.effective_turns(task) == 6


def test_filter_tasks_drops_over_budget_and_records_reason():
    tasks = [
        _task("short", "q", gt_calls=[[{"name": "a"}]], expected_turns=2),
        _task("long", "q", gt_calls=[[{"name": "a"}]], expected_turns=16, source="toolmind"),
        _task("asearcher-long", "q", expected_turns=20, source="asearcher", task_type="search_qa"),
    ]
    kept, dropped = filter_rounds.filter_tasks(tasks, max_turns=15)
    assert {task["id"] for task in kept} == {"short"}
    assert {item["id"] for item in dropped} == {"long", "asearcher-long"}
    assert all(item["reason"] == "too_many_turns" for item in dropped)
    # per-source override raises the ASearcher budget.
    kept2, _ = filter_rounds.filter_tasks(
        tasks, max_turns=15, source_max_turns={"asearcher": 20}
    )
    assert "asearcher-long" in {task["id"] for task in kept2}


def test_filter_tasks_never_mutates_input():
    task = _task("keep", "q")
    kept, _ = filter_rounds.filter_tasks([task], max_turns=15)
    assert kept[0]["id"] == "keep"
    assert task["reference"]["gt_calls"] == [[]]


# ---------------------------------------------------------------------------
# exact dedup
# ---------------------------------------------------------------------------


def test_exact_dedup_ignores_tool_menu_order():
    tool_a = dict(WEATHER_TOOL, name="a")
    tool_b = dict(WEATHER_TOOL, name="b")
    first = _task("d1", "Weather in Paris?", tools=[tool_a, tool_b])
    second = _task("d2", "weather in paris", tools=[tool_b, tool_a])
    assert task_dedup_key(first) == task_dedup_key(second)
    kept, dropped = dedup.exact_dedup([first, second])
    assert [task["id"] for task in kept] == ["d1"]
    assert dropped[0]["matched_id"] == "d1"


def test_exact_dedup_keeps_different_queries():
    tasks = [_task("e1", "Weather in Paris?"), _task("e2", "Weather in Tokyo?")]
    kept, dropped = dedup.exact_dedup(tasks)
    assert len(kept) == 2 and dropped == []


# ---------------------------------------------------------------------------
# MinHash near dedup
# ---------------------------------------------------------------------------


def test_minhash_estimate_identical_and_disjoint():
    text = "the quick brown fox jumps over the lazy dog near the river bank"
    sig_a = dedup.minhash_signature(dedup.task_shingles(_task("n1", text)), num_perm=64)
    sig_b = dedup.minhash_signature(dedup.task_shingles(_task("n2", text)), num_perm=64)
    assert dedup.jaccard_estimate(sig_a, sig_b) == 1.0

    other = "quantum chromodynamics lattice gauge theory renormalisation group"
    sig_c = dedup.minhash_signature(dedup.task_shingles(_task("n3", other)), num_perm=64)
    assert dedup.jaccard_estimate(sig_a, sig_c) < 0.3


def test_near_dedup_drops_near_duplicate():
    base = "How do I reset my account password safely and quickly today please"
    copy = "How do I reset my account password safely and quickly today"
    tasks = [_task("near-1", base), _task("near-2", copy), _task("near-3", "completely different subject matter entirely")]
    kept, dropped = dedup.near_dedup(tasks, threshold=0.6, num_perm=64, bands=8)
    assert {task["id"] for task in kept} == {"near-1", "near-3"}
    assert dropped and dropped[0]["matched_id"] == "near-1"
    assert dropped[0]["reason"] == "near_duplicate"


def test_near_dedup_against_pool_drops_only_new_tasks():
    existing = _task("existing-1", "How do I reset my account password safely and quickly today")
    incoming = _task("incoming-1", "How do I reset my account password safely and quickly today please")
    kept, dropped = dedup.near_dedup(
        [incoming], against=[existing], threshold=0.6, num_perm=64, bands=8
    )
    assert kept == []
    assert dropped[0]["matched_id"] == "existing-1"


def test_jaccard_exact_identical_sets():
    shingles = {1, 2, 3}
    assert dedup.jaccard_exact(shingles, shingles) == 1.0
    assert dedup.jaccard_exact({1, 2}, {3, 4}) == 0.0


# ---------------------------------------------------------------------------
# decontamination
# ---------------------------------------------------------------------------


def _bfcl_records():
    return [
        {
            "id": "bfcl_simple_1",
            "question": [[{"role": "user", "content": "What is the capital of France?"}]],
            "function": [WEATHER_TOOL],
        }
    ]


def test_text_ngrams_short_and_long():
    short = decontam.text_ngrams("Paris", k=13)
    assert short  # char n-gram fallback for a single token
    long = decontam.text_ngrams("what is the capital of france", k=13)
    assert "what is the capital of france" in long


def test_build_index_and_match_query_contamination():
    index = decontam.build_index(_bfcl_records(), k=13)
    assert index["num_records"] == 1
    task = _task("c1", "What is the capital of France?", tools=[])
    match = decontam.match_task(task, index)
    assert match["contaminated"]
    assert "bfcl_simple_1" in match["matched_ids"]
    clean = _task("c2", "Explain how photosynthesis works in plants.", tools=[])
    assert not decontam.match_task(clean, index)["contaminated"]


def test_match_tool_definition_contamination():
    index = decontam.build_index(_bfcl_records(), k=13)
    task = _task(
        "c3",
        "Check the weather.",
        tools=[dict(WEATHER_TOOL, description="Get the current weather for a city.")],
    )
    match = decontam.match_task(task, index)
    assert match["contaminated"]
    assert match["tool_name_hits"] == ["get_weather"]


def test_decontaminate_drops_and_keeps():
    index = decontam.build_index(_bfcl_records(), k=13)
    tasks = [
        _task("drop-me", "What is the capital of France?", tools=[]),
        _task("keep-me", "How do neural networks learn?", tools=[]),
    ]
    kept, dropped = decontam.decontaminate(tasks, index)
    assert [task["id"] for task in kept] == ["keep-me"]
    assert dropped[0]["id"] == "drop-me"
    assert dropped[0]["reason"] == "bfcl_contamination"
    assert dropped[0]["matched_ids"] == ["bfcl_simple_1"]


def test_empty_decontamination_index_is_fatal():
    with pytest.raises(ValueError):
        decontam.build_index([], k=13)


def test_approximate_jaccard_channel_matches_paraphrase():
    # Same text -> Jaccard 1.0 on 5-shingles; the approximate channel must be
    # present in the match payload (exact n-gram hits are checked separately).
    text = "what is the capital city of france in europe please tell me now"
    index = decontam.build_index([{"id": "bfcl_para", "question": text, "function": []}], k=13)
    task = _task("para", text, tools=[])
    match = decontam.match_task(task, index)
    assert match["approx_jaccard"] >= 0.9
    assert match["contaminated"]
    assert decontam.jaccard_shingles({1, 2, 3}, {1, 2, 3}) == 1.0


def test_stats_are_available_from_build_source():
    records = [
        {"id": "ok", "query": "Weather in Paris?", "tools": json.dumps([WEATHER_TOOL]),
         "answers": json.dumps([{"name": "get_weather", "arguments": {"city": "Paris"}}])},
        {"id": "no-answer", "query": "Weather in Paris?", "tools": json.dumps([WEATHER_TOOL]),
         "answers": "[]"},
        {"id": "missing-query", "tools": json.dumps([WEATHER_TOOL]),
         "answers": json.dumps([{"name": "get_weather", "arguments": {"city": "Paris"}}])},
    ]
    tasks, stats = build_raw.build_source("apigen", {"records": records})
    assert stats["rows"] == 3
    assert stats["loaded"] == len(tasks) == 1
    assert stats["skipped_by_reason"].get("no_parseable_answer") == 1
    assert stats["skipped_by_reason"].get("missing_query") == 1
    assert stats["unattributed_skips"] == 0


def test_hammer_missing_answers_field_is_not_an_irrelevance_negative():
    tasks, stats = build_raw.build_source(
        "hammer",
        {
            "records": [
                {"id": "missing", "query": "Weather in Paris?", "tools": json.dumps([WEATHER_TOOL])},
                {"id": "explicit-empty", "query": "Write a poem.", "tools": json.dumps([WEATHER_TOOL]),
                 "answers": "[]"},
            ]
        },
    )
    assert stats["skipped_by_reason"].get("missing_answers_field") == 1
    assert [task["task_type"] for task in tasks] == ["irrelevance"]


def test_read_any_records_handles_jsonl_with_json_suffix(tmp_path):
    path = tmp_path / "bfcl_dump.json"
    path.write_text(
        "\n".join(json.dumps(record) for record in _bfcl_records()),
        encoding="utf-8",
    )
    records = decontam.read_any_records(path)
    assert len(records) == 1
    assert records[0]["id"] == "bfcl_simple_1"


# ---------------------------------------------------------------------------
# build_raw pure layer
# ---------------------------------------------------------------------------


def test_build_source_returns_valid_tasks_and_stats():
    records = [
        {
            "id": 1,
            "query": "Weather in Paris?",
            "tools": json.dumps([WEATHER_TOOL]),
            "answers": json.dumps([{"name": "get_weather", "arguments": {"city": "Paris"}}]),
        }
    ]
    tasks, stats = build_raw.build_source("apigen", {"records": records})
    assert stats["loaded"] == 1
    assert stats["skipped_invalid"] == 0
    assert stats["task_types"] == {"st_fc": 1}
    assert all(validate_task(task) == [] for task in tasks)
