"""Tests for the mixture stage: deterministic build + pass-rate filtering.

Prompt card: T2 acceptance requires (a) two runs with the same seed are byte
identical, (b) category shares within tolerance, (c) train/held-out disjoint,
(d) missing/empty sources fail gracefully. Additional coverage: C6 placeholder
filtering (static declaration vs inference), pass-rate rules, config validation and
the N=8 noise math.

.. warning::
   These tests were **NOT executed on the authoring machine**: this environment
   cannot start a shell (``pwsh`` fails with ``0xC0000142``), so Python/pytest
   cannot run. They are written to be run in any normal environment:

       cd <repo root>
       python -m pytest agentic_rl_data/tests/test_mixture.py -v

   Every fixture is synthetic (no network, no HF download) and uses the frozen
   ``schema.py`` contract, so the run is self-contained.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_script(name: str):
    """Load a script as a module without relying on scripts/__init__.py (T1 owns it).

    The module must be registered in ``sys.modules`` before execution: the
    script uses ``from __future__ import annotations`` + ``@dataclass``, and
    ``dataclasses._is_type`` resolves the annotation namespace through
    ``sys.modules[cls.__module__]`` (a bare exec leaves it ``None`` and raises
    ``AttributeError``).
    """
    path = REPO_ROOT / "agentic_rl_data" / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


bm = _load_script("build_mixture")
fp = _load_script("filter_by_passrate")

from agentic_rl_data import schema  # noqa: E402  (after sys.path bootstrap)


# ===========================================================================
# synthetic fixtures (schema v2: 7 reference keys, 4 meta keys)
# ===========================================================================

ENV_FOR_TASK_TYPE = {
    "st_fc": "none",
    "irrelevance": "none",
    "mt_fc": "mock_fc",
    "search_qa": "search",
    "math_ci": "python",
    "code_exec": "code_tests",
    "abstain": "none",
}

SOURCE_TASK_TYPE = {
    "apigen": "st_fc",
    "sealtools": "st_fc",
    "toolace": "st_fc",
    "apibank": "st_fc",
    "hammer": "irrelevance",
    "funreason_mt": "mt_fc",
    "apigen_mt": "mt_fc",
    "toucan": "mt_fc",
    "search_r1": "search_qa",
    "asearcher": "search_qa",
    "reasoning_math": "math_ci",
    "simplerl_math3_5": "math_ci",
    "code_exec": "code_exec",
}

CATEGORY_MEMBERS = {
    "single_turn": [
        ("apigen", 0.27),
        ("sealtools", 0.27),
        ("toolace", 0.165),
        ("apibank", 0.045),
        ("hammer", 0.25),
    ],
    "multi_turn": [
        ("funreason_mt", 0.40),
        ("apigen_mt", 0.25),
        ("toucan", 0.35),
    ],
    "search_qa": [("search_r1", 0.60), ("asearcher", 0.40)],
    "math_ci": [("reasoning_math", 0.90), ("simplerl_math3_5", 0.10)],
    "code_exec": [("code_exec", 1.0)],
}

CATEGORY_SHARES = {
    "single_turn": 0.20,
    "multi_turn": 0.30,
    "search_qa": 0.25,
    "math_ci": 0.15,
    "code_exec": 0.10,
}


def make_task(
    source: str,
    task_type: str,
    index: int,
    *,
    pass_rate: float | None = 0.5,
    group_id: str | None = None,
    gt_returns_source: str | None = "reference",
    turns: int = 2,
    expected_turns: int | None = None,
) -> dict:
    """Build one schema-valid task (v2 contract)."""
    if task_type in ("st_fc", "mt_fc") and turns > 0:
        gt_calls = [[{"name": f"tool_{index}", "arguments": {"x": index}}] for _ in range(turns)]
        gt_turns = None
    else:
        gt_calls = []
        gt_turns = None
    tool_returns = None
    if task_type == "mt_fc":
        if gt_returns_source in ("reference", None):
            tool_returns = [[{"name": f"tool_{index}", "content": "ok"}] for _ in range(turns)]
        elif gt_returns_source == "gt_turns":
            gt_turns = [
                {"tool_results": [{"name": f"tool_{index}", "content": "ok", "ok": True}]}
                for _ in range(turns)
            ]
        # "placeholder" -> tool_returns stays None and gt_turns stays None
    meta = {
        "max_turns": 15,
        "difficulty": None,
        "pass_rate": pass_rate,
        "license": "test-fixture",
    }
    if group_id is not None:
        meta["group_id"] = group_id
    if expected_turns is not None:
        meta["expected_turns"] = expected_turns
    if task_type == "mt_fc":
        meta["gt_returns_source"] = gt_returns_source or "placeholder"
    return {
        "id": f"{source}-{index:04d}",
        "source": source,
        "task_type": task_type,
        "messages": [
            {"role": "system", "content": "You are a tool-using assistant."},
            {"role": "user", "content": f"synthetic {source} task {index}"},
        ],
        "tools": [
            {
                "name": f"tool_{index}",
                "description": "test tool",
                "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}},
            }
        ],
        "env": {"type": ENV_FOR_TASK_TYPE[task_type], "config": {}},
        "reference": {
            "final_answer": f"answer-{index}",
            "final_answers": None,
            "gt_calls": gt_calls,
            "gt_turns": gt_turns,
            "tool_returns": tool_returns,
            "unit_tests": None,
            "final_state": None,
        },
        "meta": meta,
    }


def _raw_config(
    *,
    group_aware: bool = False,
    total_train: int = 1000,
    heldout: int = 200,
    placeholder_policy: str = "drop",
    placeholder_keep_share: float = 0.0,
    oversample: int = 8,
) -> dict:
    """A compact mixure config as a dict (parse_config is tested separately)."""
    sources = {}
    for source, task_type in SOURCE_TASK_TYPE.items():
        sources[source] = {
            "file": f"{source}.jsonl",
            "task_type": task_type,
            "required": source == "apigen",
            "enabled": True,
            "license": "test-fixture",
            "note": "synthetic test source",
        }
    # Disabled sources from the real config must not block parse validation.
    for disabled in ("tau_bench_train", "toolmind", "a2search", "retool", "when2call"):
        sources[disabled] = {
            "file": f"{disabled}.jsonl",
            "task_type": "mt_fc",
            "required": False,
            "enabled": False,
            "license": None,
            "note": "disabled test source",
        }
    categories = {}
    for cat_name, members in CATEGORY_MEMBERS.items():
        categories[cat_name] = {
            "share": CATEGORY_SHARES[cat_name],
            "heldout": heldout,
            "turn_cap": 8 if cat_name == "search_qa" else 15,
            "on_unavailable": "redistribute" if cat_name == "code_exec" else "fail",
            "sources": [
                {"name": name, "weight": weight, "cap": None} for name, weight in members
            ],
        }
    return {
        "version": 1,
        "seed": 7,
        "total_train": total_train,
        "paths": {"raw_dir": "raw"},
        "output": {
            "dir": "out",
            "train": "train.jsonl",
            "heldout": "heldout.jsonl",
            "stats": "mixture_stats.json",
            "emit_curriculum": False,
        },
        "passrate": {
            "enabled": True,
            "band": {"min": 0.125, "max": 0.875},
            "n_samples": 8,
            "missing": "keep",
            "smoothed": False,
            "require_report": False,
        },
        "heldout": {"per_category": heldout, "group_aware": group_aware, "oversample": oversample},
        "filters": {
            "max_gt_turns": 15,
            "invalid_schema": "drop",
            "dedup": {
                "exact": True,
                "approximate": {
                    "enabled": True,
                    "method": "minhash",
                    "jaccard": 0.85,
                    "num_perm": 64,
                    "priority_sources": ["toolmind", "toolace", "apigen"],
                },
            },
            "decontam": {
                "enabled": True,
                "bfcl_v3": True,
                "bfcl_v4": True,
                "ngram": 5,
                "jaccard": 0.90,
                "hit_policy": "drop",
                "report_required": False,
            },
            "mt_fc": {
                "placeholder_gt_returns": placeholder_policy,
                "placeholder_keep_share": placeholder_keep_share,
                "placeholder_weight": 0.5,
            },
        },
        "curriculum": {
            "enabled": True,
            "filter_gt_turns": 15,
            "stages": [
                {"max_turns": 5, "start_fraction": 0.0, "end_fraction": 0.2},
                {"max_turns": 10, "start_fraction": 0.2, "end_fraction": 0.5},
                {"max_turns": 15, "start_fraction": 0.5, "end_fraction": 1.0},
            ],
        },
        "sources": sources,
        "categories": categories,
    }


def write_sources(
    raw_dir: Path,
    *,
    per_source: int = 360,
    passrate_variation: bool = False,
    placeholder_sources: tuple[str, ...] = (),
    invalid_row_in: str | None = None,
) -> None:
    """Materialise synthetic raw_unified/*.jsonl files."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for source, task_type in SOURCE_TASK_TYPE.items():
        path = raw_dir / f"{source}.jsonl"
        tasks = []
        for index in range(per_source):
            pass_rate = 0.5
            if passrate_variation:
                if index % 10 == 0:
                    pass_rate = 0.0
                elif index % 10 == 1:
                    pass_rate = 1.0
            placeholder = source in placeholder_sources
            tasks.append(
                make_task(
                    source,
                    task_type,
                    index,
                    pass_rate=pass_rate,
                    gt_returns_source="placeholder" if placeholder else "reference",
                    turns=1 if task_type in ("st_fc", "irrelevance") else 3,
                )
            )
        with path.open("w", encoding="utf-8") as handle:
            for task in tasks:
                handle.write(json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n")
            if invalid_row_in == source:
                handle.write('{"id": "broken", "source": "broken"}\n')


def build_default(tmp_path: Path, **kwargs):
    cfg = bm.parse_config(_raw_config())
    return bm.build_mixture(cfg, raw_dir=tmp_path / "raw", out_dir=tmp_path / "out", **kwargs)


def read_jsonl(path: Path) -> list[dict]:
    return list(schema.load_jsonl(path))


# ===========================================================================
# 1. config parsing / YAML fallback
# ===========================================================================

def test_yaml_subset_parser_handles_nested_maps_and_sequences():
    text = """
sources:
  apigen:
    file: apigen.jsonl
    required: true
categories:
  single:
    share: 0.2
    sources:
      - name: apigen
        weight: 1.0
        cap: 10
stages:
  - max_turns: 5
    start: 0.0
flags:
  enabled: true
  ratio: 0.125
  license: null
"""
    parsed = bm._parse_yaml_subset(text)
    assert parsed["sources"]["apigen"]["required"] is True
    assert parsed["categories"]["single"]["sources"][0] == {
        "name": "apigen",
        "weight": 1.0,
        "cap": 10,
    }
    assert parsed["stages"][0]["max_turns"] == 5
    assert parsed["flags"]["enabled"] is True
    assert parsed["flags"]["license"] is None


def test_real_config_parses_validates_and_matches_plan():
    config = bm.load_config(REPO_ROOT / "agentic_rl_data" / "configs" / "mixture.yaml")
    assert config.seed == 20251105
    assert config.total_train == 8000
    assert [cat.name for cat in config.categories] == [
        "single_turn",
        "multi_turn",
        "search_qa",
        "math_ci",
        "code_exec",
    ]
    assert [cat.share for cat in config.categories] == pytest.approx(
        [0.20, 0.30, 0.25, 0.15, 0.10], abs=1e-9
    )
    assert all(cat.heldout >= 200 for cat in config.categories)
    assert config.heldout_per_category == 300
    assert config.band_min == 0.125 and config.band_max == 0.875
    assert config.n_samples == 8
    assert [stage.max_turns for stage in config.stages] == [5, 10, 15]
    assert config.placeholder_gt_returns == "drop"
    assert config.sources["tau_bench_train"].enabled is False
    # plan section 0 existing single-turn rows are present
    for source in ("apigen", "sealtools", "toolace", "hammer", "apibank"):
        assert config.sources[source].enabled and config.sources[source].required


def test_config_rejects_share_sum_not_one():
    raw = _raw_config()
    raw["categories"]["single_turn"]["share"] = 0.21
    with pytest.raises(bm.MixtureConfigError, match="shares must sum to 1.0"):
        bm.parse_config(raw)


def test_config_rejects_small_heldout():
    raw = _raw_config(heldout=100)
    with pytest.raises(bm.MixtureConfigError, match="at least 200"):
        bm.parse_config(raw)


def test_config_rejects_unknown_category_source():
    raw = _raw_config()
    raw["categories"]["single_turn"]["sources"][0]["name"] = "unknown_source"
    with pytest.raises(bm.MixtureConfigError, match="unknown source"):
        bm.parse_config(raw)


def test_config_rejects_placeholder_keep_share_with_drop():
    raw = _raw_config(placeholder_policy="drop", placeholder_keep_share=0.2)
    with pytest.raises(bm.MixtureConfigError, match="conflicts"):
        bm.parse_config(raw)


# ===========================================================================
# 2. build: determinism, shares, held-out disjointness
# ===========================================================================

@pytest.fixture(scope="module")
def fixture_raw(tmp_path_factory):
    raw_dir = tmp_path_factory.mktemp("raw_unified")
    write_sources(raw_dir)
    return raw_dir


@pytest.fixture(scope="module")
def built_once(fixture_raw, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("mixture_v1")
    cfg = bm.parse_config(_raw_config())
    stats = bm.build_mixture(cfg, raw_dir=fixture_raw, out_dir=out_dir)
    return cfg, stats, out_dir


def test_two_runs_byte_identical(fixture_raw, tmp_path):
    cfg = bm.parse_config(_raw_config())
    stats_a = bm.build_mixture(cfg, raw_dir=fixture_raw, out_dir=tmp_path / "a")
    stats_b = bm.build_mixture(cfg, raw_dir=fixture_raw, out_dir=tmp_path / "b")
    assert stats_a["outputs"]["train"]["sha256"] == stats_b["outputs"]["train"]["sha256"]
    assert stats_a["outputs"]["heldout"]["sha256"] == stats_b["outputs"]["heldout"]["sha256"]
    # Run in the same directory: overwriting must be byte-identical as well.
    stats_c = bm.build_mixture(cfg, raw_dir=fixture_raw, out_dir=tmp_path / "a")
    assert stats_c["outputs"]["train"]["sha256"] == stats_a["outputs"]["train"]["sha256"]
    stats_path_a = tmp_path / "a" / "mixture_stats.json"
    first_bytes = stats_path_a.read_bytes()
    bm.build_mixture(cfg, raw_dir=fixture_raw, out_dir=tmp_path / "a")
    assert stats_path_a.read_bytes() == first_bytes


def test_category_shares_within_tolerance(built_once):
    _cfg, stats, _out = built_once
    total = stats["total_train_actual"]
    assert total == 1000
    for cat in stats["categories"].values():
        assert abs(cat["share_effective"] - cat["share_target"]) < 0.01


def test_quota_per_category_exact(built_once):
    _cfg, stats, _out = built_once
    expected = {"single_turn": 200, "multi_turn": 300, "search_qa": 250, "math_ci": 150, "code_exec": 100}
    for name, count in expected.items():
        assert stats["categories"][name]["train_actual"] == count


def test_heldout_size_and_disjointness(built_once):
    _cfg, stats, out_dir = built_once
    train = read_jsonl(out_dir / "train.jsonl")
    heldout = read_jsonl(out_dir / "heldout.jsonl")
    train_ids = {task["id"] for task in train}
    held_ids = {task["id"] for task in heldout}
    assert train_ids.isdisjoint(held_ids)
    assert len(heldout) == 1000  # 5 categories * 200
    for cat in stats["categories"].values():
        assert cat["heldout_actual"] >= 200
    # no duplicate task ids overall
    assert len(train_ids) == len(train) and len(held_ids) == len(heldout)


def test_heldout_respects_group_boundaries(tmp_path):
    raw_dir = tmp_path / "raw"
    write_sources(raw_dir)
    cfg = bm.parse_config(_raw_config(group_aware=True, oversample=10))
    # Give every 5 consecutive tasks in apigen one session group (and declare the
    # size, otherwise partial groups cannot be detected -- [NO-GROUP] limitation).
    path = raw_dir / "apigen.jsonl"
    tasks = read_jsonl(path)
    for i, task in enumerate(tasks):
        task["meta"]["group_id"] = f"apigen-session-{i // 5:03d}"
        task["meta"]["group_size"] = 5
    schema.dump_jsonl(tasks, path)
    stats = bm.build_mixture(cfg, raw_dir=raw_dir, out_dir=tmp_path / "out")
    train = read_jsonl(tmp_path / "out" / "train.jsonl")
    heldout = read_jsonl(tmp_path / "out" / "heldout.jsonl")
    train_ids = {task["id"] for task in train}
    heldout_ids = {task["id"] for task in heldout}
    train_groups = {task["meta"].get("group_id") for task in train if task["source"] == "apigen"}
    held_groups = {task["meta"].get("group_id") for task in heldout if task["source"] == "apigen"}
    assert train_groups.isdisjoint(held_groups)
    for group in train_groups | held_groups:
        members = {
            task["id"]
            for task in train + heldout
            if task["source"] == "apigen" and task["meta"].get("group_id") == group
        }
        # no session may be split across the split; truncated sessions may be
        # smaller than 5 only when they were not selected for either side.
        assert members <= train_ids or members <= heldout_ids
    assert stats["categories"]["single_turn"]["heldout_groups"] > 0


def test_different_seed_changes_selection(fixture_raw, tmp_path):
    cfg = bm.parse_config(_raw_config())
    stats_a = bm.build_mixture(cfg, raw_dir=fixture_raw, out_dir=tmp_path / "seed7", seed=7)
    stats_b = bm.build_mixture(cfg, raw_dir=fixture_raw, out_dir=tmp_path / "seed99", seed=99)
    train_a = {task["id"] for task in read_jsonl(tmp_path / "seed7" / "train.jsonl")}
    train_b = {task["id"] for task in read_jsonl(tmp_path / "seed99" / "train.jsonl")}
    assert stats_a["seed"] == 7 and stats_b["seed"] == 99
    assert train_a != train_b


def test_caps_scale_linearly_with_total_train():
    import dataclasses

    cfg = bm.parse_config(_raw_config())  # baseline total_train=1000
    assert bm._scale_cap(cfg, 550) == 550  # baseline: identity
    cfg_half = dataclasses.replace(cfg, total_train=500)  # same baseline in cfg.raw
    assert bm._scale_cap(cfg_half, 550) == 275
    cfg_double = dataclasses.replace(cfg, total_train=2000)
    assert bm._scale_cap(cfg_double, 550) == 1100


# ===========================================================================
# 3. build: graceful failures + guards
# ===========================================================================

def test_missing_required_source_fails_clearly(tmp_path):
    raw_dir = tmp_path / "raw"
    write_sources(raw_dir)
    (raw_dir / "apigen.jsonl").unlink()
    cfg = bm.parse_config(_raw_config())
    with pytest.raises(bm.MixtureDataError, match="apigen"):
        bm.build_mixture(cfg, raw_dir=raw_dir, out_dir=tmp_path / "out")


def test_empty_required_source_fails_clearly(tmp_path):
    raw_dir = tmp_path / "raw"
    write_sources(raw_dir)
    (raw_dir / "apigen.jsonl").write_text("", encoding="utf-8")
    cfg = bm.parse_config(_raw_config())
    with pytest.raises(bm.MixtureDataError, match="empty"):
        bm.build_mixture(cfg, raw_dir=raw_dir, out_dir=tmp_path / "out")


def test_missing_optional_source_is_tolerated(tmp_path):
    raw_dir = tmp_path / "raw"
    write_sources(raw_dir)
    (raw_dir / "simplerl_math3_5.jsonl").unlink()
    cfg = bm.parse_config(_raw_config())
    stats = bm.build_mixture(cfg, raw_dir=raw_dir, out_dir=tmp_path / "out")
    assert stats["sources"]["math_ci:simplerl_math3_5"]["input"] == "missing_optional"
    assert stats["categories"]["math_ci"]["train_actual"] == 150


def test_schema_invalid_rows_are_dropped_and_counted(tmp_path):
    raw_dir = tmp_path / "raw"
    write_sources(raw_dir, per_source=60, invalid_row_in="apigen")
    cfg = bm.parse_config(_raw_config(total_train=200, heldout=200))
    stats = bm.build_mixture(cfg, raw_dir=raw_dir, out_dir=tmp_path / "out")
    source_stats = stats["sources"]["single_turn:apigen"]
    assert source_stats["invalid_schema"] == 1
    assert source_stats["invalid_examples"]


def test_passrate_guard_drops_zero_and_one(tmp_path):
    raw_dir = tmp_path / "raw"
    write_sources(raw_dir, passrate_variation=True)
    cfg = bm.parse_config(_raw_config(total_train=400))
    stats = bm.build_mixture(cfg, raw_dir=raw_dir, out_dir=tmp_path / "out")
    train = read_jsonl(tmp_path / "out" / "train.jsonl")
    assert all(0.0 < task["meta"]["pass_rate"] < 1.0 for task in train)
    dropped = {
        reason
        for source_stats in stats["sources"].values()
        for reason in source_stats["dropped"]
    }
    assert {"all_correct", "all_wrong"} <= dropped


def test_source_loader_mapping_is_parsed():
    raw = _raw_config()
    raw["sources"]["code_exec"] = {
        "file": "code_exec.jsonl",
        "task_type": "code_exec",
        "required": False,
        "enabled": True,
        "license": None,
        "loader": "code_exec",
        "loader_cfg": {"kind": "apps"},
        "note": "single T1 loader",
    }
    raw["categories"]["code_exec"]["sources"] = [{"name": "code_exec", "weight": 1.0, "cap": 800}]
    cfg = bm.parse_config(raw)
    assert cfg.sources["code_exec"].loader == "code_exec"
    assert cfg.sources["code_exec"].loader_cfg == {"kind": "apps"}


def test_placeholder_static_declaration_drops_rows(tmp_path):
    raw_dir = tmp_path / "raw"
    write_sources(raw_dir, placeholder_sources=("funreason_mt",))
    cfg = bm.parse_config(_raw_config())
    stats = bm.build_mixture(cfg, raw_dir=raw_dir, out_dir=tmp_path / "out")
    funreason = stats["sources"]["multi_turn:funreason_mt"]
    assert funreason["placeholder_static"] == 360
    assert funreason["dropped"]["degraded_replay_placeholder"] == 360
    assert stats["placeholder"]["static_declared"] >= 360
    assert stats["placeholder"]["inferred_fallback"] == 0


def test_gt_returns_source_prefers_static_declaration():
    task = make_task("funreason_mt", "mt_fc", 0, gt_returns_source="placeholder")
    assert bm.gt_returns_source(task) == ("placeholder", "static")
    # legacy row without the static key falls back to inference from reference
    task = make_task("funreason_mt", "mt_fc", 0)
    task["meta"].pop("gt_returns_source")
    assert bm.gt_returns_source(task) == ("reference", "inferred")


def test_placeholder_keep_share_bounds_train_rows(tmp_path):
    raw_dir = tmp_path / "raw"
    write_sources(raw_dir, placeholder_sources=("funreason_mt", "apigen_mt_5k", "toucan_1_5m"))
    cfg = bm.parse_config(
        _raw_config(placeholder_policy="keep", placeholder_keep_share=0.25)
    )
    stats = bm.build_mixture(cfg, raw_dir=raw_dir, out_dir=tmp_path / "out")
    multi = stats["categories"]["multi_turn"]
    assert multi["placeholder_budget"] == 75  # floor(300 * 0.25)
    assert multi["placeholder_train"] <= multi["placeholder_budget"]
    assert stats["placeholder"]["share_of_mt_train"] <= 0.25 + 1e-9


def test_emit_curriculum_writes_stage_files(tmp_path):
    raw_dir = tmp_path / "raw"
    write_sources(raw_dir)
    cfg = bm.parse_config(_raw_config())
    bm.build_mixture(cfg, raw_dir=raw_dir, out_dir=tmp_path / "out", emit_curriculum=True)
    stage5 = read_jsonl(tmp_path / "out" / "curriculum" / "train_max_turns5.jsonl")
    stage10 = read_jsonl(tmp_path / "out" / "curriculum" / "train_max_turns10.jsonl")
    stage15 = read_jsonl(tmp_path / "out" / "curriculum" / "train_max_turns15.jsonl")
    assert all(bm.effective_turns(task) <= 5 for task in stage5)
    assert len(stage5) <= len(stage10) <= len(stage15)
    assert len(stage15) == 1000


def test_dry_run_writes_nothing(fixture_raw, tmp_path):
    cfg = bm.parse_config(_raw_config())
    bm.build_mixture(cfg, raw_dir=fixture_raw, out_dir=tmp_path / "out", dry_run=True)
    assert not (tmp_path / "out" / "train.jsonl").exists()
    assert not (tmp_path / "out" / "mixture_stats.json").exists()


# ===========================================================================
# 4. filter_by_passrate unit + CLI
# ===========================================================================

def _rates(*pairs) -> dict:
    return {f"task-{i}": fp.RateRecord(task_id=f"task-{i}", pass_rate=rate) for i, rate in enumerate(pairs)}


def test_filter_keeps_only_strictly_interior_rates():
    tasks = [
        {"id": f"task-{i}", "task_type": "st_fc", "meta": {"license": "x", "max_turns": 15}}
        for i in range(3)
    ]
    stats = fp.filter_tasks(tasks, _rates(0.0, 1.0, 0.5), band=None)
    assert [task["id"] for task in stats["tasks"]] == ["task-2"]
    assert stats["drop_reasons"]["all_wrong"] == 1
    assert stats["drop_reasons"]["all_correct"] == 1


def test_filter_band_removes_outside_values_and_writes_meta():
    task = {"id": "task-0", "task_type": "st_fc", "meta": {"license": "x", "max_turns": 15}}
    stats = fp.filter_tasks([task], _rates(0.95), band=(0.125, 0.875))
    assert stats["kept"] == 0
    assert stats["drop_reasons"]["outside_passrate_band"] == 1
    stats = fp.filter_tasks([task], _rates(0.3), band=(0.125, 0.875))
    kept = stats["tasks"][0]
    assert kept["meta"]["pass_rate"] == 0.3
    assert kept["meta"]["license"] == "x"  # C10: no meta key removed


def test_filter_missing_policy():
    task = {"id": "task-0", "task_type": "st_fc", "meta": {"license": "x", "max_turns": 15}}
    kept = fp.filter_tasks([task], {}, missing="keep")
    assert kept["kept"] == 1 and kept["kept_without_rate"] == 1
    dropped = fp.filter_tasks([task], {}, missing="drop")
    assert dropped["kept"] == 0 and dropped["drop_reasons"]["missing_rate"] == 1


def test_filter_min_samples():
    record = fp.RateRecord(task_id="task-0", pass_rate=0.5, successes=2, trials=4)
    stats = fp.filter_tasks(
        [{"id": "task-0", "task_type": "st_fc", "meta": {}}], {"task-0": record}, min_samples=8
    )
    assert stats["kept"] == 0 and stats["drop_reasons"]["too_few_samples"] == 1


def test_parse_rates_formats():
    text = (
        '{"task-a": 0.25, "task-b": {"pass_rate": 0.5, "successes": 4, "trials": 8}}\n'
    )
    records, errors = fp.parse_rates_text(text)
    assert errors == []
    assert records["task-a"].pass_rate == 0.25
    assert records["task-b"].successes == 4 and records["task-b"].trials == 8

    jsonl = '{"task_id": "task-a", "pass_rate": 0.25}\n'
    records, _ = fp.parse_rates_text(jsonl)
    assert records["task-a"].pass_rate == 0.25

    tsv = "task-a\t0.25\t8\n"
    records, errors = fp.parse_rates_text(tsv)
    assert errors == [] and records["task-a"].trials == 8


def test_noise_math_matches_plan_assumptions():
    table = fp.noise_rows(8)
    assert table["max_standard_error"] == pytest.approx(0.5 / 8 ** 0.5, rel=1e-9)
    assert table["grid_resolution"] == 0.125
    row8 = next(row for row in table["rows"] if row["n"] == 8)
    assert row8["drop_rate_at_p_0.05"] == pytest.approx(0.95 ** 8 + 0.05 ** 8, rel=1e-9)
    assert row8["drop_rate_at_p_0.05"] > 0.65  # the documented N=8 false-drop cost
    row100 = next(row for row in table["rows"] if row["n"] == 100)
    assert row100["max_se"] == pytest.approx(0.05, rel=1e-9)
    assert fp.wilson_interval(4, 8)[0] >= 0.0
    assert fp.jeffreys_mean(4, 8) == pytest.approx(4.5 / 9.0)


def test_filter_cli_end_to_end(tmp_path):
    tasks_path = tmp_path / "tasks.jsonl"
    rates_path = tmp_path / "rates.jsonl"
    out_path = tmp_path / "kept.jsonl"
    report_path = tmp_path / "report.json"
    tasks = [make_task("apigen", "st_fc", i, pass_rate=0.5) for i in range(6)]
    for i, task in enumerate(tasks):
        task["meta"]["pass_rate"] = None
    schema.dump_jsonl(tasks, tasks_path)
    with rates_path.open("w", encoding="utf-8") as handle:
        for i, rate in enumerate([0.0, 1.0, 0.5, 0.375, 0.625, 0.9]):
            handle.write(json.dumps({"task_id": tasks[i]["id"], "pass_rate": rate}) + "\n")
    code = fp.main(
        [
            "--tasks", str(tasks_path),
            "--rates", str(rates_path),
            "--out", str(out_path),
            "--report", str(report_path),
        ]
    )
    assert code == 0
    kept = read_jsonl(out_path)
    assert {task["id"] for task in kept} == {tasks[i]["id"] for i in (2, 3, 4)}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["stats"]["kept"] == 3
    assert report["noise"]["n_samples"] == 8


# ===========================================================================
# 5. build CLI (one command, plan section 8 acceptance #1)
# ===========================================================================

def test_build_cli_runs_with_real_config(tmp_path):
    raw_dir = tmp_path / "raw"
    write_sources(raw_dir, per_source=40)
    # every enabled source must exist; disabled sources are simply absent
    code = bm.main(
        [
            "--config", str(REPO_ROOT / "agentic_rl_data" / "configs" / "mixture.yaml"),
            "--raw-dir", str(raw_dir),
            "--out-dir", str(tmp_path / "out"),
            "--dry-run",
            "--quiet",
        ]
    )
    assert code == 0
