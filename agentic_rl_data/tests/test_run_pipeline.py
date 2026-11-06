"""Tests for ``scripts/run_pipeline.py`` (the one-command data build).

NOT EXECUTED in the authoring session (``pwsh`` fails with ``0xC0000142``).
Run on a normal machine::

    python -m pytest agentic_rl_data/tests/test_run_pipeline.py -q

No step is actually executed here: every test targets config validation, the
file helpers, step selection and the ``--dry-run`` plan (which must write
nothing).  Executing real steps needs source data / BFCL data / a rollout
pass-rate file and is covered by the user-side commands in the report.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_rl_data.schema import dump_jsonl
from agentic_rl_data.scripts import run_pipeline


def _config(tmp_path: Path, **overrides) -> dict:
    paths = {
        "raw_dir": str(tmp_path / "data" / "raw_unified"),
        "filtered_dir": str(tmp_path / "data" / "filtered_rounds"),
        "deduped_dir": str(tmp_path / "data" / "deduped"),
        "decontaminated_dir": str(tmp_path / "data" / "decontaminated"),
        "passrated_dir": str(tmp_path / "data" / "passrated"),
        "mixture_dir": str(tmp_path / "data" / "mixture_v1"),
        "reports_dir": str(tmp_path / "reports"),
        "mixture_config": "agentic_rl_data/configs/mixture.yaml",
        "verl_dir": str(tmp_path / "data" / "verl"),
    }
    config = {
        "paths": paths,
        "build_raw": {"sources": [], "cfg_json": None, "max_samples": None},
        "rounds": {"max_turns": 15, "asearcher_max_turns": 15},
        "dedup": {"threshold": 0.85, "num_perm": 64, "bands": 16, "k": 5, "against": []},
        "decontam": {"bfcl_path": None, "extra_paths": []},
        "replay": {"enabled": True, "limit": 10, "task_types": "mt_fc"},
        "passrate": {"enabled": False, "rates": None},
        "mixture": {"total_train": None},
        "report": {"strict": False, "auto_allow_passrate": True},
    }
    config.update(overrides)
    return config


def test_load_config_rejects_missing_path_keys(tmp_path: Path):
    bad = {"paths": {"raw_dir": str(tmp_path)}}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(run_pipeline.PipelineError):
        run_pipeline.load_config(path)


def test_load_config_rejects_invalid_json(tmp_path: Path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(run_pipeline.PipelineError):
        run_pipeline.load_config(path)


def test_jsonl_files_skips_underscore_and_sorts(tmp_path: Path):
    directory = tmp_path / "d"
    directory.mkdir()
    dump_jsonl([{"id": "x"}], directory / "b.jsonl")
    dump_jsonl([{"id": "y"}], directory / "a.jsonl")
    dump_jsonl([{"id": "z"}], directory / "_combined.jsonl")
    (directory / "notes.txt").write_text("noise", encoding="utf-8")
    files = run_pipeline.jsonl_files(directory)
    assert [Path(name).name for name in files] == ["a.jsonl", "b.jsonl"]


def test_merge_and_split_roundtrip(tmp_path: Path):
    a1 = tmp_path / "a_src" / "s1.jsonl"
    a2 = tmp_path / "a_src" / "s2.jsonl"
    dump_jsonl(
        [
            {"id": "1", "source": "src_a", "payload": {"v": 1}},
            {"id": "2", "source": "src_b", "payload": {"v": 2}},
        ],
        a1,
    )
    dump_jsonl([{"id": "3", "source": "src_a", "payload": {"v": 3}}], a2)
    merged = tmp_path / "out" / "_combined.jsonl"
    assert run_pipeline.merge_jsonl([a1, a2], merged) == 3
    counts = run_pipeline.split_by_source(merged, tmp_path / "out")
    assert counts == {"src_a": 2, "src_b": 1}
    # round-trip content is preserved, per source
    assert [row["id"] for row in _read(tmp_path / "out" / "src_a.jsonl")] == ["1", "3"]
    assert [row["id"] for row in _read(tmp_path / "out" / "src_b.jsonl")] == ["2"]


def _read(path: Path) -> list[dict]:
    from agentic_rl_data.schema import load_jsonl

    return list(load_jsonl(path))


def test_select_steps_from_to_and_skip():
    assert run_pipeline.select_steps() == list(run_pipeline.STEP_ORDER)
    assert run_pipeline.select_steps(start="dedup", stop="mix") == [
        "dedup",
        "decontam",
        "replay",
        "passrate",
        "mix",
    ]
    assert run_pipeline.select_steps(skip=("replay", "passrate")) == [
        "load",
        "rounds",
        "dedup",
        "decontam",
        "mix",
        "report",
        "export",
    ]
    with pytest.raises(run_pipeline.PipelineError):
        run_pipeline.select_steps(skip=("nope",))
    with pytest.raises(run_pipeline.PipelineError):
        run_pipeline.select_steps(start="report", stop="load")


def test_dry_run_plans_all_steps_without_writing(tmp_path: Path, capsys):
    config_path = tmp_path / "pipeline.json"
    config_path.write_text(json.dumps(_config(tmp_path)), encoding="utf-8")
    code = run_pipeline.main(["--config", str(config_path), "--dry-run"])
    assert code == 0
    out = capsys.readouterr().out
    for step in run_pipeline.STEP_ORDER:
        assert f"{step:<9}" in out or f"[pipeline] {step}" in out
    assert "[dry-run]" in out
    # optional steps are skipped with a reason, not silently dropped
    assert "decontam.bfcl_path not configured" in out
    assert "passrate.enabled=false" in out
    # nothing was written: no data dirs, no manifest
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "_pipeline_manifest.json").exists()
    assert "export → " in out


def test_dry_run_reports_configured_optional_steps(tmp_path: Path, capsys):
    cfg = _config(tmp_path)
    cfg["decontam"]["bfcl_path"] = str(tmp_path / "bfcl")
    cfg["passrate"]["enabled"] = True
    cfg["passrate"]["rates"] = str(tmp_path / "rates.jsonl")
    config_path = tmp_path / "pipeline.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    code = run_pipeline.main(["--config", str(config_path), "--dry-run"])
    assert code == 0
    out = capsys.readouterr().out
    assert "decontam → " in out
    assert "passrate → " in out


def test_bad_config_returns_2(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"paths": {}}), encoding="utf-8")
    assert run_pipeline.main(["--config", str(path)]) == 2


def test_unknown_step_returns_2(tmp_path: Path):
    config_path = tmp_path / "pipeline.json"
    config_path.write_text(json.dumps(_config(tmp_path)), encoding="utf-8")
    assert run_pipeline.main(["--config", str(config_path), "--skip", "nope"]) == 2
