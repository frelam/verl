"""Small local end-to-end smoke for every new procedural source (no downloads)."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from agentic_rl_data.loaders import load_source
from agentic_rl_data.schema import validate_task
from agentic_rl_data.rewards import dispatch_reward
from agentic_rl_data.scripts import export_verl_parquet as export
from agentic_rl_data.scripts.replay_check import check_task
from agentic_rl_data.scripts.build_mixture import build_mixture as build_mixture_data
from agentic_rl_data.scripts.build_mixture import load_config, parse_config
from agentic_rl_data.scripts.build_raw import build_all


@pytest.mark.parametrize(
    ("source", "count"),
    [
        ("interleaved_mt", 2),
        ("stateful_abstain", 3),
        ("vehicle_travel_mt", 3),
        ("social_ops_mt", 3),
        ("parallel_mt", 2),
        ("procedural_st", 4),
    ],
)
def test_generated_source_loader_replay_and_export(source: str, count: int) -> None:
    tasks = list(load_source(source, {"n": count, "seed": 2026, "fault_prob": 0.0}))
    assert len(tasks) == count
    for task in tasks:
        assert validate_task(task) == []
        if task["task_type"] == "mt_fc":
            report = check_task(task, reward_fn=dispatch_reward, strict_tool_errors=False)
            assert report["ok"], report
            assert report["score"] == pytest.approx(1.0)
        row = export.task_to_row(task, 0)
        restored = json.loads(row["extra_info"]["agentic_task"])
        assert restored == task
        # Future instructions live only in env config, never in the initial prompt.
        dialogue = task["env"]["config"].get("dialogue")
        if dialogue:
            prompt_users = [m["content"] for m in row["prompt"] if m["role"] == "user"]
            assert len(prompt_users) == 1
            for stage in dialogue.get("stages") or []:
                if stage.get("next_user"):
                    assert stage["next_user"] not in prompt_users


def test_round6_config_is_balanced() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "mixture_v2.yaml"
    cfg = load_config(path)
    assert cfg.total_train == 12000
    assert [stage.max_turns for stage in cfg.stages] == [8, 16, 24]
    expected = {
        "multi_turn": 6000,
        "single_turn": 2400,
        "search_qa": 2400,
        "math_ci": 960,
        "irrelevance": 240,
    }
    for category in cfg.categories:
        assert sum(source.weight for source in category.sources) == pytest.approx(1.0)
        assert sum(source.cap or 0 for source in category.sources) == expected[category.name]
    multi = {source.name: source.cap for source in cfg.category("multi_turn").sources}
    assert multi["procedural_mt"] == 1200
    assert multi["toucan"] + multi["apigen_mt"] == 600
    assert cfg.category("multi_turn").turn_cap == 24


def test_mini_raw_mixture_and_parquet_pipeline(tmp_path) -> None:
    sources = [
        "interleaved_mt", "stateful_abstain", "vehicle_travel_mt",
        "social_ops_mt", "parallel_mt", "procedural_st",
    ]
    raw_dir = tmp_path / "raw"
    report = build_all(
        {source: {"n": 30, "seed": 91, "fault_prob": 0.0} for source in sources},
        sources=sources,
        out_dir=raw_dir,
    )
    assert report["total_loaded"] == 180

    base = load_config(Path(__file__).resolve().parents[1] / "configs" / "mixture_v2.yaml").raw
    raw = copy.deepcopy(base)
    raw["total_train"] = 120
    raw["paths"] = {"raw_dir": str(raw_dir)}
    raw["output"].update({"dir": str(tmp_path / "mix"), "emit_curriculum": False})
    raw["passrate"]["enabled"] = False
    raw["filters"]["decontam"]["enabled"] = False
    raw["curriculum"]["enabled"] = False
    raw["sources"] = {
        source: {
            "file": f"{source}.jsonl",
            "task_type": "st_fc" if source == "procedural_st" else "mt_fc",
            "required": True,
            "license": "generated (repo-internal)",
        }
        for source in sources
    }
    raw["categories"] = {
        "multi_turn": {
            "share": 0.75,
            "heldout": 200,
            "turn_cap": 24,
            "on_unavailable": "fail",
            "sources": [
                {"name": source, "weight": 1 / 5, "cap": 18}
                for source in sources if source != "procedural_st"
            ],
        },
        "single_turn": {
            "share": 0.25,
            "heldout": 200,
            "turn_cap": 15,
            "on_unavailable": "fail",
            "sources": [{"name": "procedural_st", "weight": 1.0, "cap": 30}],
        },
    }
    parsed = parse_config(raw)
    cfg = replace(
        parsed,
        heldout_per_category=1,
        categories=tuple(replace(category, heldout=1) for category in parsed.categories),
    )
    stats = build_mixture_data(cfg, raw_dir=raw_dir, out_dir=tmp_path / "mix")
    assert stats["total_train_actual"] == 120
    train_stats = export.export_split(tmp_path / "mix" / "train.jsonl", tmp_path / "train.parquet")
    assert train_stats["rows"] == 120
    heldout_stats = export.export_split(tmp_path / "mix" / "heldout.jsonl", tmp_path / "heldout.parquet")
    assert heldout_stats["rows"] == 2
