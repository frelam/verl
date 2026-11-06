"""Tests for verl export + single-turn reward bridge.

NOT EXECUTED in the authoring session (``pwsh`` fails with ``0xC0000142``).
Run on a machine with pyarrow installed::

    python -m pytest agentic_rl_data/tests/test_verl_export_bridge.py -q

Coverage:
* ``task_to_row`` / ``normalize_tool`` / ``build_rows`` (schema-valid task only,
  invalid rows are counted or raise, never silently dropped),
* parquet write -> read round-trip with the explicit Arrow schema
  (``pyarrow`` required; skipped otherwise) and ``tools_for_model`` JSON parsing,
* ``verl_reward_bridge.compute_score`` for single-turn task types, its
  fail-closed behaviour for env-required types/malformed responses, and the
  degraded-task fallback.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures.reward_fixtures import GET_WEATHER, task  # noqa: E402

from agentic_rl_data.scripts import export_verl_parquet as export  # noqa: E402
from agentic_rl_data.scripts import verl_reward_bridge as bridge  # noqa: E402

ST_FC_TASK = task(
    "st_fc",
    task_id="st-1",
    tools=[GET_WEATHER],
    gt_calls=[[{"name": "get_weather", "arguments": {"city": "Paris"}}]],
    final_answer="It is sunny in Paris.",
    meta={"license": "MIT", "pass_rate": 0.5},
)

SOLUTION = (
    "<think>I should call the weather tool for Paris.</think>\n"
    '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
)


# ---------------------------------------------------------------------------
# export helpers
# ---------------------------------------------------------------------------


def test_normalize_tool_stores_parameters_as_json_string():
    tool = export.normalize_tool(GET_WEATHER)
    assert tool["name"] == "get_weather"
    assert isinstance(tool["parameters"], str)
    parsed = json.loads(tool["parameters"])
    assert parsed["required"] == ["city"]
    # round-trip back for model / chat-template use
    back = export.tools_for_model([tool])
    assert back[0]["parameters"]["properties"]["city"]["type"] == "string"


def test_task_to_row_contract():
    row = export.task_to_row(ST_FC_TASK, 7)
    assert row["data_source"] == export.VERL_DATA_SOURCE
    assert row["prompt"] == ST_FC_TASK["messages"]
    assert row["tools"][0]["name"] == "get_weather"
    assert row["reward_model"]["style"] == "rule"
    info = row["extra_info"]
    assert info["index"] == 7
    assert info["task_id"] == "st-1"
    assert info["task_type"] == "st_fc"
    assert info["max_turns"] == 15
    assert info["tool_selection"] == ["get_weather"]
    # the full canonical task must survive the JSON round-trip
    assert json.loads(info["agentic_task"]) == ST_FC_TASK
    assert json.loads(info["ground_truth_calls"]) == [
        [{"name": "get_weather", "arguments": {"city": "Paris"}}]
    ]


def test_build_rows_rejects_invalid_and_can_count(tmp_path):
    bad = dict(ST_FC_TASK)
    bad.pop("reference")  # breaks schema v2.2
    with pytest.raises(export.ExportError):
        export.build_rows([bad])
    rows, invalid = export.build_rows([bad], allow_invalid=True)
    assert rows == []
    assert invalid and "reference" in " ".join(invalid[0]["errors"])


def test_export_split_dry_run(tmp_path):
    mix = tmp_path / "mix"
    mix.mkdir()
    from agentic_rl_data.schema import dump_jsonl

    dump_jsonl([ST_FC_TASK], mix / "train.jsonl")
    stats = export.export_split(mix / "train.jsonl", tmp_path / "train.parquet", dry_run=True)
    assert stats["rows"] == 1
    assert stats["by_task_type"] == {"st_fc": 1}
    assert stats["sha256"] is None
    assert not (tmp_path / "train.parquet").exists()


def test_export_unmappable_bridge_task_keeps_flow(tmp_path):
    """``_task_from_extra`` must degrade, never raise, when agentic_task is absent."""
    out = bridge.compute_score("agentic_rl", SOLUTION, None, extra_info=None)
    assert set(out) >= {"score", "bridge", "task_type"}
    assert 0.0 <= out["score"] <= 1.0


def test_export_split_dry_run_deterministic(tmp_path):
    from agentic_rl_data.schema import dump_jsonl

    mix = tmp_path / "mix"
    mix.mkdir()
    dump_jsonl([ST_FC_TASK, ST_FC_TASK], mix / "train.jsonl")
    first = export.export_split(mix / "train.jsonl", tmp_path / "a.parquet", dry_run=True)
    second = export.export_split(mix / "train.jsonl", tmp_path / "b.parquet", dry_run=True)
    assert json.dumps(first["by_task_type"], sort_keys=True) == json.dumps(
        second["by_task_type"], sort_keys=True
    )
    assert first["rows"] == second["rows"] == 2


# ---------------------------------------------------------------------------
# parquet round-trip (needs pyarrow)
# ---------------------------------------------------------------------------


def test_parquet_roundtrip(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from agentic_rl_data.schema import dump_jsonl

    mix = tmp_path / "mix"
    mix.mkdir()
    dump_jsonl([ST_FC_TASK], mix / "train.jsonl")
    out = tmp_path / "train.parquet"
    stats = export.export_split(mix / "train.jsonl", out)
    assert stats["rows"] == 1 and stats["sha256"]

    table = pq.read_table(out)
    assert table.column_names == [
        "data_source",
        "prompt",
        "tools",
        "reward_model",
        "extra_info",
    ]
    row = table.to_pylist()[0]
    assert row["data_source"] == "agentic_rl"
    assert row["prompt"][0]["role"] == "system"
    assert row["tools"][0]["parameters"].startswith("{")  # JSON string on disk
    assert json.loads(row["extra_info"]["agentic_task"])["id"] == "st-1"
    assert bridge._task_from_extra(dict(row["extra_info"]), "agentic_rl")["id"] == "st-1"


# ---------------------------------------------------------------------------
# reward bridge
# ---------------------------------------------------------------------------


def test_bridge_single_turn_st_fc_does_not_crash():
    extra = {"agentic_task": json.dumps(ST_FC_TASK)}
    out = bridge.compute_score("agentic_rl", SOLUTION, "It is sunny.", extra)
    assert out["task_type"] == "st_fc"
    assert out["num_calls"] == 1
    # D20/C14: single-turn prediction must not fail the tool_results pairing gate
    assert out.get("format_ok") is True
    assert out["score"] >= 0.99
    assert 0.0 <= out["score"] <= 1.0


def test_bridge_irrelevance_rule():
    irrelevance_task = task("irrelevance", task_id="irr-1", gt_calls=[], final_answer="No tool needed.")
    extra = {"agentic_task": json.dumps(irrelevance_task)}
    without_calls = bridge.compute_score("agentic_rl", "<think>answer directly</think>Yes.", None, extra)
    assert without_calls["score"] == 1.0
    with_calls = bridge.compute_score("agentic_rl", SOLUTION, None, extra)
    assert with_calls["score"] == 0.0


def test_bridge_env_required_type_is_fail_closed():
    mt_task = task("mt_fc", task_id="mt-1", gt_calls=[[{"name": "get_weather", "arguments": {"city": "Paris"}}]])
    out = bridge.compute_score("agentic_rl", SOLUTION, None, {"agentic_task": json.dumps(mt_task)})
    assert out["score"] == 0.0
    assert out["unsupported_single_turn"] is True
    assert "agentic_agent" in out["unsupported_reason"]


def test_bridge_malformed_solution_never_raises():
    extra = {"agentic_task": json.dumps(ST_FC_TASK)}
    for solution in ("", "garbage <tool_call>{" , "<tool_call>not json</tool_call>"):
        out = bridge.compute_score("agentic_rl", solution, None, extra)
        assert isinstance(out["score"], float)
        assert 0.0 <= out["score"] <= 1.0


def test_bridge_extra_info_as_array_item():
    """non_tensor_batch may hand over 0-d numpy objects; ``_as_mapping`` normalises."""

    class ArrayLike:
        def __init__(self, value):
            self._value = value

        def item(self):
            return self._value

    extra = ArrayLike({"agentic_task": json.dumps(ST_FC_TASK)})
    out = bridge.compute_score("agentic_rl", SOLUTION, None, extra)
    assert out["task_type"] == "st_fc"
