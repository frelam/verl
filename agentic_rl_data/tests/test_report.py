"""Tests for ``scripts/report.py`` (plan §3 step 8 / §8 A2/A3/A4).

NOT EXECUTED in the authoring session: this machine's ``pwsh`` fails with
``0xC0000142`` so no Python process can start.  Run on a normal machine::

    python -m pytest agentic_rl_data/tests/test_report.py -q

Covers (review R2-01..R2-12):
* distributions recomputed from JSONL (turn / tool / pass-rate / license),
* ``[NOT RUN]`` instead of invented numbers when a step report is missing,
* strict *fail-closed* semantics: missing upstream steps, empty held-out,
  decontamination **residual** hits (removed hits are audit-only), empty BFCL
  index, replay skipped / reward-unused passes, split overlap (id / dedup-key /
  group), schema-invalid rows, stale mixture stats, sha mismatch and the
  optional provenance requirement,
* deterministic rendering (byte-identical re-render, no timestamps),
* standard ``data/`` layout discovery (R2-04).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agentic_rl_data.schema import dump_jsonl
from agentic_rl_data.scripts.report import (
    build_report,
    compute_distributions,
    compute_overlap,
    render_markdown,
)

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def make_task(
    task_type: str = "st_fc",
    *,
    task_id: str = "fixture-1",
    source: str = "unit_test",
    tools: int | None = None,
    gt_turns: int = 1,
    pass_rate: float | None = None,
    license_name: str | None = None,
    group_id: str | None = None,
    gt_returns_source: str | None = None,
) -> dict:
    """One schema-valid task (all frozen v2.2 keys present)."""
    env_type = {
        "mt_fc": "mock_fc",
        "st_fc": "none",
        "irrelevance": "none",
        "search_qa": "search",
        "math_ci": "python",
        "code_exec": "code_tests",
        "abstain": "none",
    }[task_type]
    tool_list = [
        {"name": f"tool_{i}", "description": f"tool {i}", "parameters": {"type": "object", "properties": {}}}
        for i in range(tools if tools is not None else 1)
    ]
    gt_calls = [[{"name": "tool_0", "arguments": {"x": 1}}] for _ in range(gt_turns)]
    meta = {
        "max_turns": 15,
        "difficulty": None,
        "pass_rate": pass_rate,
        "license": license_name,
    }
    if group_id is not None:
        meta["group_id"] = group_id
    if task_type == "mt_fc":
        meta["gt_returns_source"] = gt_returns_source or "placeholder"
    return {
        "id": task_id,
        "source": source,
        "task_type": task_type,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"question {task_id}"},
        ],
        "tools": tool_list,
        "env": {"type": env_type, "config": {}},
        "reference": {
            "final_answer": None,
            "final_answers": None,
            "gt_calls": gt_calls,
            "gt_turns": None,
            "tool_returns": None,
            "unit_tests": None,
            "final_state": None,
        },
        "meta": meta,
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_mixture(
    root: Path,
    train: list[dict],
    heldout: list[dict] | None = None,
    *,
    record_sha: bool = True,
) -> Path:
    mix_dir = root / "mixture_v1"
    mix_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(train, mix_dir / "train.jsonl")
    dump_jsonl(heldout or [], mix_dir / "heldout.jsonl")
    stats: dict = {
        "seed": 20251105,
        "total_train_target": len(train),
        "total_train_actual": len(train),
        "heldout_target": len(heldout or []),
        "heldout_actual": len(heldout or []),
        "config_sha256": "deadbeef",
        "outputs": {},
        "categories": {
            "single_turn": {
                "share_target": 0.2,
                "share_effective": 0.2,
                "train_actual": len(train),
                "heldout_actual": len(heldout or []),
                "placeholder_train": 0,
            }
        },
        "sources": {},
        "filters_applied": {"passrate": False, "decontam": False},
        "warnings": [],
    }
    if record_sha:
        stats["outputs"]["train"] = {
            "path": str(mix_dir / "train.jsonl"),
            "count": len(train),
            "sha256": _sha(mix_dir / "train.jsonl"),
        }
        stats["outputs"]["heldout"] = {
            "path": str(mix_dir / "heldout.jsonl"),
            "count": len(heldout or []),
            "sha256": _sha(mix_dir / "heldout.jsonl"),
        }
    (mix_dir / "mixture_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return mix_dir


def _write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _decontam(path: Path, *, removed: int = 0, residual: int = 0, records: int = 42) -> str:
    """Decontam report: ``removed`` is audit, ``residual`` is the strict gate."""
    payload: dict = {
        "bfcl_records": records,
        "index_records": records,
        "bfcl_path": "/tmp/bfcl",
        "total_in": 10,
        "total_kept": 10 - removed,
        "total_dropped": removed,
        "residual_hits": residual,
        "hit_counts_by_source": {},
        "residual_samples_by_source": {},
        "sources": {"unit_test": {"in": 10, "kept": 10 - removed, "dropped": removed, "residual_hits": residual}},
    }
    return _write_json(path, payload)


def _replay(path: Path, *, failed: int = 0, skipped: int = 0, reward_used: int = 3, passed: int = 3) -> str:
    return _write_json(
        path,
        {
            "summary": {
                "total": passed + failed + skipped,
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "reward_used": reward_used,
                "placeholder_ratio": 0.0,
            },
            "failures": [],
        },
    )


def _all_reports(tmp_path: Path) -> dict[str, str]:
    """Minimal reports for every step (strict passes only when all are present)."""
    return {
        "build_raw": _write_json(tmp_path / "_build_report.json", {"sources": {}, "total_loaded": 0}),
        "filter_rounds": _write_json(
            tmp_path / "_filter_rounds_report.json",
            {"sources": {}, "total_in": 0, "total_kept": 0, "total_dropped": 0},
        ),
        "dedup": _write_json(tmp_path / "_dedup_report.json", {"stats": {"input": 0, "kept": 0}}),
        "decontam": _decontam(tmp_path / "_decontam_report.json"),
        "replay": _replay(tmp_path / "_replay_report.json"),
        "passrate": _write_json(tmp_path / "_passrate_report.json", {"stats": {"kept": 0, "dropped": 0}}),
    }


# ---------------------------------------------------------------------------
# distributions / overlap
# ---------------------------------------------------------------------------


def test_distributions_recomputed_from_tasks():
    tasks = [
        make_task("st_fc", task_id="a", tools=1, gt_turns=1, pass_rate=0.5, license_name="MIT"),
        make_task("mt_fc", task_id="b", tools=7, gt_turns=4, pass_rate=0.25, license_name="[UNVERIFIED] HF"),
        make_task("search_qa", task_id="c", tools=2, pass_rate=None, group_id="sess-1"),
        make_task("code_exec", task_id="d", tools=12, gt_turns=1, pass_rate=0.75),
    ]
    dist = compute_distributions(tasks)
    assert dist["total"] == 4
    assert dist["task_types"] == {"code_exec": 1, "mt_fc": 1, "search_qa": 1, "st_fc": 1}
    assert dist["tool_counts"] == {"0": 0, "1-2": 2, "3-5": 0, "6-10": 1, "over_10": 1}
    # Search-QA tasks carry 1 gt turn here, so three rows land in the "1" bucket.
    assert dist["turns"]["buckets"]["1"] == 3
    # Only the mt_fc row has 4 gt turns; the rest are single-turn rows.
    assert dist["turns"]["buckets"]["2-3"] == 0
    assert dist["turns"]["buckets"]["4-5"] == 1
    assert dist["pass_rate"]["count"] == 3
    assert dist["pass_rate"]["missing"] == 1
    assert dist["unverified_license_rows"] == 1
    assert dist["with_group_id"] == 1
    assert dist["group_id_coverage"] == pytest.approx(0.25)
    assert dist["gt_returns_sources"]["placeholder"] == 1
    assert dist["gt_returns_sources"]["<none>"] == 3


def test_overlap_detects_ids_keys_and_groups():
    train = [
        make_task("st_fc", task_id="same", group_id="g1"),
        make_task("st_fc", task_id="train-only", group_id="g2"),
    ]
    heldout = [
        make_task("st_fc", task_id="same", group_id="g1"),
        make_task("st_fc", task_id="held-only", group_id="g3"),
    ]
    overlap = compute_overlap(train, heldout)
    assert overlap["id_overlap"] == 1
    assert overlap["dedup_key_overlap"] == 1  # same id -> same normalised query/tools
    assert overlap["group_overlap"] == 1
    assert overlap["train_with_group"] == 2
    assert overlap["heldout_with_group"] == 2


# ---------------------------------------------------------------------------
# rendering / missing reports
# ---------------------------------------------------------------------------


def test_missing_upstream_reports_mark_not_run(tmp_path: Path):
    mix_dir = _write_mixture(tmp_path, [make_task("st_fc", task_id="a")], [make_task("st_fc", task_id="b")])
    report = build_report(mixture_dir=mix_dir, raw_dir=tmp_path / "raw_unified", strict=False)
    assert "decontam" in report["not_run"]
    assert "replay" in report["not_run"]
    assert report["checks"]["decontam_residual_hits"] is None
    assert report["checks"]["replay_failed"] is None

    markdown = render_markdown(report)
    assert "**[NOT RUN]**" in markdown
    assert "去污染未验证" in markdown
    assert "命中总数：0" not in markdown  # no invented acceptance numbers
    assert render_markdown(report) == markdown  # deterministic
    assert "generated_at" not in markdown


def test_report_writes_distribution_sections(tmp_path: Path):
    train = [make_task("st_fc", task_id="a", tools=3, gt_turns=2, pass_rate=0.5)]
    held = [make_task("mt_fc", task_id="b", tools=6, gt_turns=1)]
    mix_dir = _write_mixture(tmp_path, train, held)
    report = build_report(mixture_dir=mix_dir, raw_dir=tmp_path / "raw", strict=False)
    markdown = render_markdown(report)
    assert "## 5. 分布" in markdown
    assert '"3-5": 1' in markdown
    assert "group_id 覆盖" in markdown
    assert "| single_turn |" in markdown
    # filters_applied is rendered even when false (T2-04)
    assert "过滤是否真的执行" in markdown


def test_missing_train_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        build_report(mixture_dir=tmp_path / "nope", raw_dir=tmp_path, strict=False)


def test_standard_data_layout_is_discovered(tmp_path: Path):
    """R2-04: reports under data/{filtered_rounds,deduped,decontaminated} are found."""
    data = tmp_path / "data"
    raw_dir = data / "raw_unified"
    mix_dir = _write_mixture(data, [make_task("st_fc", task_id="a")], [make_task("st_fc", task_id="b")])
    _write_json(raw_dir / "_build_report.json", {"sources": {}, "total_loaded": 0})
    _write_json(
        data / "filtered_rounds" / "_filter_rounds_report.json",
        {"sources": {}, "total_in": 0, "total_kept": 0, "total_dropped": 0},
    )
    _write_json(data / "deduped" / "_dedup_report.json", {"stats": {"input": 0, "kept": 0}})
    _decontam(data / "decontaminated" / "_decontam_report.json")
    _replay(data / "decontaminated" / "_replay_report.json")
    report = build_report(mixture_dir=mix_dir, raw_dir=raw_dir, strict=False)
    assert report["steps"]["build_raw"]["status"] == "run"
    assert report["steps"]["filter_rounds"]["status"] == "run"
    assert report["steps"]["dedup"]["status"] == "run"
    assert report["steps"]["decontam"]["status"] == "run"
    assert report["steps"]["replay"]["status"] == "run"


# ---------------------------------------------------------------------------
# strict: missing steps / held-out / stats
# ---------------------------------------------------------------------------


def test_strict_fails_when_steps_missing(tmp_path: Path):
    mix_dir = _write_mixture(tmp_path, [make_task("st_fc", task_id="a")], [make_task("st_fc", task_id="b")])
    report = build_report(mixture_dir=mix_dir, raw_dir=tmp_path / "raw", strict=True)
    failures = report["strict_failures"]
    for step in ("build_raw", "filter_rounds", "dedup", "decontam", "replay", "passrate"):
        assert any(f"{step} report missing" in item for item in failures)
    assert any("heldout" in item for item in failures) is False  # held-out is non-empty here


def test_allow_missing_steps_relaxes_passrate_only(tmp_path: Path):
    mix_dir = _write_mixture(tmp_path, [make_task("st_fc", task_id="a")], [make_task("st_fc", task_id="b")])
    reports = _all_reports(tmp_path)
    # Remove both the explicit entry and the file so discovery cannot find it.
    Path(reports.pop("passrate")).unlink()
    report = build_report(
        mixture_dir=mix_dir,
        raw_dir=tmp_path / "raw",
        explicit_reports=reports,
        strict=True,
        allow_missing_steps=("passrate",),
    )
    assert report["strict_failures"] == []
    with pytest.raises(ValueError):
        build_report(
            mixture_dir=mix_dir,
            raw_dir=tmp_path / "raw",
            explicit_reports=reports,
            strict=True,
            allow_missing_steps=("not_a_step",),
        )


def test_strict_fails_on_empty_heldout(tmp_path: Path):
    mix_dir = _write_mixture(tmp_path, [make_task("st_fc", task_id="a")], [])
    report = build_report(
        mixture_dir=mix_dir,
        raw_dir=tmp_path / "raw",
        explicit_reports=_all_reports(tmp_path),
        strict=True,
    )
    assert any("heldout split is empty" in item for item in report["strict_failures"])


def test_strict_fails_without_mixture_stats(tmp_path: Path):
    mix_dir = _write_mixture(tmp_path, [make_task("st_fc", task_id="a")], [make_task("st_fc", task_id="b")])
    (mix_dir / "mixture_stats.json").unlink()
    report = build_report(
        mixture_dir=mix_dir,
        raw_dir=tmp_path / "raw",
        explicit_reports=_all_reports(tmp_path),
        strict=True,
    )
    assert report["checks"]["mixture_stats_present"] is False
    assert any("mixture_stats.json missing" in item for item in report["strict_failures"])


# ---------------------------------------------------------------------------
# strict: decontamination semantics (R2-01, the P0)
# ---------------------------------------------------------------------------


def test_removed_hits_are_audit_only_and_strict_passes(tmp_path: Path):
    mix_dir = _write_mixture(tmp_path, [make_task("st_fc", task_id="a")], [make_task("st_fc", task_id="b")])
    reports = _all_reports(tmp_path)
    reports["decontam"] = _decontam(tmp_path / "_decontam_report.json", removed=3, residual=0)
    report = build_report(
        mixture_dir=mix_dir, raw_dir=tmp_path / "raw", explicit_reports=reports, strict=True
    )
    assert report["checks"]["decontam_hits_removed"] == 3
    assert report["checks"]["decontam_residual_hits"] == 0
    assert report["strict_failures"] == []


def test_residual_hits_fail_strict(tmp_path: Path):
    mix_dir = _write_mixture(tmp_path, [make_task("st_fc", task_id="a")], [make_task("st_fc", task_id="b")])
    reports = _all_reports(tmp_path)
    reports["decontam"] = _decontam(tmp_path / "_decontam_report.json", removed=1, residual=2)
    report = build_report(
        mixture_dir=mix_dir, raw_dir=tmp_path / "raw", explicit_reports=reports, strict=True
    )
    assert report["checks"]["decontam_residual_hits"] == 2
    assert any("residual_hits=2" in item for item in report["strict_failures"])


def test_unknown_residual_fails_strict(tmp_path: Path):
    """Legacy decontam report without residual_hits must not pass (fail-closed)."""
    mix_dir = _write_mixture(tmp_path, [make_task("st_fc", task_id="a")], [make_task("st_fc", task_id="b")])
    reports = _all_reports(tmp_path)
    legacy = _write_json(
        tmp_path / "_decontam_legacy.json",
        {"bfcl_records": 42, "total_in": 1, "total_kept": 1, "total_dropped": 0, "sources": {}},
    )
    reports["decontam"] = legacy
    report = build_report(
        mixture_dir=mix_dir, raw_dir=tmp_path / "raw", explicit_reports=reports, strict=True
    )
    assert any("residual_hits unknown" in item for item in report["strict_failures"])


def test_empty_bfcl_index_fails_strict(tmp_path: Path):
    mix_dir = _write_mixture(tmp_path, [make_task("st_fc", task_id="a")], [make_task("st_fc", task_id="b")])
    reports = _all_reports(tmp_path)
    reports["decontam"] = _decontam(tmp_path / "_decontam_report.json", records=0)
    report = build_report(
        mixture_dir=mix_dir, raw_dir=tmp_path / "raw", explicit_reports=reports, strict=True
    )
    assert any("index is empty" in item for item in report["strict_failures"])


# ---------------------------------------------------------------------------
# strict: replay evidence, overlap, schema, sha
# ---------------------------------------------------------------------------


def test_replay_skipped_or_unused_reward_fails(tmp_path: Path):
    mix_dir = _write_mixture(tmp_path, [make_task("st_fc", task_id="a")], [make_task("st_fc", task_id="b")])
    reports = _all_reports(tmp_path)
    reports["replay"] = _replay(tmp_path / "_replay_report.json", skipped=2)
    report = build_report(
        mixture_dir=mix_dir, raw_dir=tmp_path / "raw", explicit_reports=reports, strict=True
    )
    assert any("skipped=2" in item for item in report["strict_failures"])

    reports["replay"] = _replay(tmp_path / "_replay_report.json", passed=3, reward_used=1)
    report = build_report(
        mixture_dir=mix_dir, raw_dir=tmp_path / "raw", explicit_reports=reports, strict=True
    )
    assert any("reward_used=1" in item for item in report["strict_failures"])


def test_strict_fails_on_group_overlap(tmp_path: Path):
    shared = make_task("st_fc", task_id="a", group_id="g1")
    other = make_task("st_fc", task_id="b", group_id="g1")
    mix_dir = _write_mixture(tmp_path, [shared], [other])
    report = build_report(
        mixture_dir=mix_dir,
        raw_dir=tmp_path / "raw",
        explicit_reports=_all_reports(tmp_path),
        strict=True,
    )
    assert report["checks"]["group_overlap"] == 1
    assert any("group overlap" in item for item in report["strict_failures"])


def test_strict_fails_on_sha_mismatch(tmp_path: Path):
    mix_dir = _write_mixture(tmp_path, [make_task("st_fc", task_id="a")], [make_task("st_fc", task_id="b")])
    stats_path = mix_dir / "mixture_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["outputs"]["train"]["sha256"] = "0" * 64
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    report = build_report(
        mixture_dir=mix_dir,
        raw_dir=tmp_path / "raw",
        explicit_reports=_all_reports(tmp_path),
        strict=True,
    )
    assert report["checks"]["train_sha256_match"] is False
    assert any("sha256 mismatch" in item for item in report["strict_failures"])


def test_strict_fails_on_schema_invalid_rows(tmp_path: Path):
    bad = make_task("st_fc", task_id="bad")
    bad["reference"].pop("final_state")  # violates schema v2.2 required keys
    mix_dir = _write_mixture(tmp_path, [bad], [make_task("st_fc", task_id="b")])
    report = build_report(
        mixture_dir=mix_dir,
        raw_dir=tmp_path / "raw",
        explicit_reports=_all_reports(tmp_path),
        strict=True,
    )
    assert report["checks"]["schema_invalid_train"] == 1
    assert any("schema-invalid train" in item for item in report["strict_failures"])


def test_provenance_requirement_is_opt_in(tmp_path: Path):
    mix_dir = _write_mixture(tmp_path, [make_task("st_fc", task_id="a")], [make_task("st_fc", task_id="b")])
    reports = _all_reports(tmp_path)
    report = build_report(
        mixture_dir=mix_dir,
        raw_dir=tmp_path / "raw",
        explicit_reports=reports,
        strict=True,
        require_provenance=True,
    )
    assert report["strict_failures"] != []
    assert any("input_sha256" in item for item in report["strict_failures"])

    for step, path in reports.items():
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["input_sha256"] = "a" * 64
        Path(path).write_text(json.dumps(payload), encoding="utf-8")
    report = build_report(
        mixture_dir=mix_dir,
        raw_dir=tmp_path / "raw",
        explicit_reports=reports,
        strict=True,
        require_provenance=True,
    )
    assert report["strict_failures"] == []
