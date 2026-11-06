"""Step 8: generate ``reports/data_report.md`` from the pipeline artifacts.

Authoritative spec: ``agentic_rl_data_plan.md`` §3 step 8 and §8 acceptance #4::

    reports/data_report.md 含每来源的加载数、各步丢弃数与原因、最终配比、
    轮数 / 工具数分布、通过率分布。

Design rules
------------
* **Numbers must be traceable.** Every number is either recomputed from
  ``data/mixture_v1/{train,heldout}.jsonl`` or read from an upstream JSON
  report; the markdown states the source file next to the table.
* **Never pretend a step ran.** When an upstream report is missing the section
  is rendered as ``[NOT RUN]`` and listed in §8; ``--strict`` turns that into a
  non-zero exit instead of silently shipping a "final" report (critic T2-04).
* **Deterministic output.** No timestamps, dicts sorted, so repeated runs are
  byte-identical (plan §8 A1 spirit; diffable in PRs).
* **Read-only.** This script never rewrites tasks; it only reads and renders.

Usage
-----
.. code-block:: bash

    python -m agentic_rl_data.scripts.report \
        --mixture-dir data/mixture_v1 --raw-dir data/raw_unified \
        --out reports/data_report.md

The exit code is 0 on success; with ``--strict`` it is 1 when any acceptance
check fails (see :data:`STRICT_RULES`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:  # allow direct execution from the repo root
    sys.path.insert(0, str(_ROOT))

from agentic_rl_data.schema import (  # noqa: E402
    load_jsonl,
    task_dedup_key,
    validate_task,
)
from agentic_rl_data.scripts.filter_rounds import effective_turns  # noqa: E402

#: Upstream report file names searched under the candidate directories.
REPORT_SPECS: dict[str, tuple[str, ...]] = {
    "build_raw": ("_build_report.json",),
    "filter_rounds": ("_filter_rounds_report.json",),
    "dedup": ("_dedup_report.json",),
    "decontam": ("_decontam_report.json",),
    "replay": ("_replay_report.json", "_replay_check_report.json"),
    "passrate": ("_passrate_report.json",),
}

#: Glob patterns tried after the explicit file names above.
REPORT_GLOBS: dict[str, tuple[str, ...]] = {
    "passrate": ("*passrate*.report.json",),
    "replay": ("*replay*.json",),
}

#: Checks enforced by ``--strict`` (documented in the rendered report).
STRICT_RULES: tuple[str, ...] = (
    "steps: build_raw / filter_rounds / dedup / decontam / replay / passrate reports present "
    "(relax individually with --allow-missing-steps, e.g. passrate before rollout exists)",
    "decontam: report present, BFCL records > 0, residual_hits == 0 (plan §8 A2); "
    "removed hits are an audit column and may be > 0",
    "replay: failed == 0, skipped == 0, reward_used >= passed (plan §8 A3)",
    "mixture: heldout non-empty; id / dedup-key / session-group overlap all 0",
    "schema: every train/heldout row passes schema.validate_task",
    "reproducibility: mixture_stats.json present with recorded sha256 and matching row count, "
    "and recomputed train/heldout sha256 equals the recorded value",
    "provenance (optional, --require-provenance): every upstream report carries input_sha256",
)

DEFAULT_OUT = "agentic_rl_data/reports/data_report.md"


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _load_json(path: str | Path) -> dict[str, Any] | None:
    """Read a JSON object; ``None`` when absent or unparsable (reported later)."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_dirs(mixture_dir: Path, raw_dir: Path, report_dir: Path | None) -> list[Path]:
    """Directories searched for upstream reports (standard ``data/`` layout only).

    Deliberately does **not** include the current working directory: tests and
    renders must be hermetic (a stale real report must never leak into a
    fixture-based report).  Use ``--report-dir`` for non-standard layouts.
    """
    dirs = [
        mixture_dir,
        mixture_dir.parent,
        raw_dir,
        raw_dir.parent,
        raw_dir.parent / "filtered_rounds",
        raw_dir.parent / "rounds_filtered",
        raw_dir.parent / "deduped",
        raw_dir.parent / "decontaminated",
    ]
    if report_dir is not None:
        dirs.append(report_dir)
    seen: list[Path] = []
    for directory in dirs:
        if directory not in seen:
            seen.append(directory)
    return seen


def discover_reports(
    *,
    mixture_dir: str | Path,
    raw_dir: str | Path,
    report_dir: str | Path | None = None,
    explicit: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    """Locate upstream reports: ``{step: path}`` (only steps that exist)."""
    found: dict[str, Path] = {}
    explicit = dict(explicit or {})
    for step, path_text in explicit.items():
        path = Path(path_text)
        if path.is_file():
            found[step] = path
    dirs = _candidate_dirs(Path(mixture_dir), Path(raw_dir), Path(report_dir) if report_dir else None)
    for step, names in REPORT_SPECS.items():
        if step in found:
            continue
        for name in names:
            for directory in dirs:
                candidate = directory / name
                if candidate.is_file():
                    found[step] = candidate
                    break
            if step in found:
                break
        if step in found:
            continue
        for pattern in REPORT_GLOBS.get(step, ()):
            for directory in dirs:
                matches = sorted(directory.glob(pattern))
                if matches:
                    found[step] = matches[0]
                    break
            if step in found:
                break
    return found


# ---------------------------------------------------------------------------
# distributions (all recomputed from the JSONL, never hard-coded)
# ---------------------------------------------------------------------------


def _turn_buckets(tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    buckets = {"1": 0, "2-3": 0, "4-5": 0, "6-10": 0, "11-15": 0, "over_15": 0}
    values: list[int] = []
    for task in tasks:
        # Display floor: CONTRACTS C9 gives 0 for QA tasks without gt calls, but
        # build_mixture's own turn_distribution reports those as 1.  Keep one
        # floor here so data_report can be cross-checked against mixture_stats
        # (review R2-05); filtering itself still uses the C9 formula.
        turns = max(1, effective_turns(task))
        values.append(turns)
        if turns <= 1:
            buckets["1"] += 1
        elif turns <= 3:
            buckets["2-3"] += 1
        elif turns <= 5:
            buckets["4-5"] += 1
        elif turns <= 10:
            buckets["6-10"] += 1
        elif turns <= 15:
            buckets["11-15"] += 1
        else:
            buckets["over_15"] += 1
    if not values:
        return {"buckets": buckets, "count": 0, "mean": None, "min": None, "max": None}
    return {
        "buckets": buckets,
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def _tool_count_buckets(tasks: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    buckets = {"0": 0, "1-2": 0, "3-5": 0, "6-10": 0, "over_10": 0}
    for task in tasks:
        tools = task.get("tools")
        count = len(tools) if isinstance(tools, list) else 0
        if count == 0:
            buckets["0"] += 1
        elif count <= 2:
            buckets["1-2"] += 1
        elif count <= 5:
            buckets["3-5"] += 1
        elif count <= 10:
            buckets["6-10"] += 1
        else:
            buckets["over_10"] += 1
    return buckets


def _pass_rate_distribution(tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rates: list[float] = []
    missing = 0
    for task in tasks:
        meta = task.get("meta") if isinstance(task.get("meta"), Mapping) else {}
        rate = _as_float(meta.get("pass_rate"))
        if rate is None:
            missing += 1
        else:
            rates.append(rate)
    if not rates:
        return {"count": 0, "missing": missing, "buckets": {}}
    buckets = {"(0,0.25]": 0, "0.25-0.5": 0, "0.5-0.625": 0, "0.625-0.75": 0, "0.75-1)": 0}
    for rate in rates:
        if rate <= 0.25:
            buckets["(0,0.25]"] += 1
        elif rate <= 0.5:
            buckets["0.25-0.5"] += 1
        elif rate <= 0.625:
            buckets["0.5-0.625"] += 1
        elif rate <= 0.75:
            buckets["0.625-0.75"] += 1
        else:
            buckets["0.75-1)"] += 1
    ordered = sorted(rates)
    return {
        "count": len(rates),
        "missing": missing,
        "mean": sum(rates) / len(rates),
        "median": ordered[len(ordered) // 2],
        "buckets": buckets,
    }


def compute_distributions(tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """All plan §3 step 8 distributions for one split."""
    task_types: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    env_types: Counter[str] = Counter()
    gt_sources: Counter[str] = Counter()
    licenses: Counter[str] = Counter()
    max_turns: Counter[str] = Counter()
    with_group = 0
    with_gt_returns = 0
    unverified_license = 0
    for task in tasks:
        task_types[str(task.get("task_type"))] += 1
        sources[str(task.get("source"))] += 1
        env = task.get("env") if isinstance(task.get("env"), Mapping) else {}
        env_types[str(env.get("type"))] += 1
        meta = task.get("meta") if isinstance(task.get("meta"), Mapping) else {}
        source_kind = meta.get("gt_returns_source")
        gt_sources[str(source_kind) if source_kind is not None else "<none>"] += 1
        license_name = meta.get("license")
        if isinstance(license_name, str):
            licenses[license_name] += 1
            if "unverified" in license_name.lower():
                unverified_license += 1
        else:
            licenses["<null>"] += 1
        turns = meta.get("max_turns")
        if isinstance(turns, int) and not isinstance(turns, bool):
            max_turns[str(turns)] += 1
        if meta.get("group_id"):
            with_group += 1
        reference = task.get("reference") if isinstance(task.get("reference"), Mapping) else {}
        if reference.get("tool_returns"):
            with_gt_returns += 1
    return {
        "total": len(tasks),
        "task_types": dict(sorted(task_types.items())),
        "sources": dict(sorted(sources.items())),
        "env_types": dict(sorted(env_types.items())),
        "gt_returns_sources": dict(sorted(gt_sources.items())),
        "licenses": dict(sorted(licenses.items())),
        "unverified_license_rows": unverified_license,
        "max_turns_buckets": dict(sorted(max_turns.items(), key=lambda item: int(item[0]))),
        "with_group_id": with_group,
        "group_id_coverage": with_group / len(tasks) if tasks else 0.0,
        "with_gt_tool_returns": with_gt_returns,
        "turns": _turn_buckets(tasks),
        "tool_counts": _tool_count_buckets(tasks),
        "pass_rate": _pass_rate_distribution(tasks),
    }


def compute_overlap(
    train: Sequence[Mapping[str, Any]], heldout: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Id / dedup-key / session-group overlap between the two splits."""
    train_ids = {str(task.get("id")) for task in train}
    held_ids = {str(task.get("id")) for task in heldout}
    train_keys = {task_dedup_key(task) for task in train}
    held_keys = {task_dedup_key(task) for task in heldout}
    train_groups = {
        str(task.get("meta", {}).get("group_id"))
        for task in train
        if isinstance(task.get("meta"), Mapping) and task["meta"].get("group_id")
    }
    held_groups = {
        str(task.get("meta", {}).get("group_id"))
        for task in heldout
        if isinstance(task.get("meta"), Mapping) and task["meta"].get("group_id")
    }
    return {
        "id_overlap": len(train_ids & held_ids),
        "dedup_key_overlap": len(train_keys & held_keys),
        "group_overlap": len(train_groups & held_groups),
        "train_with_group": len(train_groups),
        "heldout_with_group": len(held_groups),
    }


# ---------------------------------------------------------------------------
# report assembly
# ---------------------------------------------------------------------------


def _step_summary(step: str, data: Mapping[str, Any] | None) -> dict[str, Any]:
    """Compact, tool-agnostic digest of one upstream report."""
    if data is None:
        return {"status": "not_run"}
    entry: dict[str, Any] = {"status": "run"}
    if step == "build_raw":
        entry["total_loaded"] = data.get("total_loaded")
        entry["sources"] = {
            name: {
                "loaded": payload.get("loaded"),
                "skipped_invalid": payload.get("skipped_invalid"),
                "rows": payload.get("rows"),
                "emitted": payload.get("emitted"),
                "skipped_by_reason": payload.get("skipped_by_reason"),
                "unattributed_skips": payload.get("unattributed_skips", payload.get("unattributed")),
            }
            for name, payload in (data.get("sources") or {}).items()
            if isinstance(payload, Mapping)
        }
    elif step in ("filter_rounds", "decontam"):
        entry["total_in"] = data.get("total_in")
        entry["total_kept"] = data.get("total_kept")
        entry["total_dropped"] = data.get("total_dropped")
        entry["per_source"] = {
            name: {
                "in": payload.get("in"),
                "kept": payload.get("kept"),
                "dropped": payload.get("dropped"),
                "residual_hits": payload.get("residual_hits"),
            }
            for name, payload in (data.get("sources") or {}).items()
            if isinstance(payload, Mapping)
        }
        if step == "decontam":
            entry["bfcl_records"] = data.get("bfcl_records")
            entry["index_records"] = data.get("index_records")
            entry["bfcl_path"] = data.get("bfcl_path")
            # audit: detected and already removed
            entry["hits_removed"] = data.get("total_dropped")
            # gate: hits still present in the kept split (must be 0)
            entry["residual_hits"] = data.get("residual_hits")
            entry["hit_counts_by_source"] = data.get("hit_counts_by_source")
            entry["residual_samples_by_source"] = data.get("residual_samples_by_source")
    elif step == "dedup":
        stats = data.get("stats") if isinstance(data.get("stats"), Mapping) else {}
        entry["input"] = stats.get("input")
        entry["exact_dropped"] = stats.get("exact_dropped")
        entry["near_dropped"] = stats.get("near_dropped")
        entry["kept"] = stats.get("kept")
        entry["against_tasks"] = data.get("against_tasks")
    elif step == "passrate":
        stats = data.get("stats") if isinstance(data.get("stats"), Mapping) else {}
        entry["n_samples"] = (data.get("params") or {}).get("n_samples")
        entry["kept"] = stats.get("kept")
        entry["dropped"] = stats.get("dropped")
        entry["kept_without_rate"] = stats.get("kept_without_rate", stats.get("missing_rate"))
    elif step == "replay":
        summary = data.get("summary") if isinstance(data.get("summary"), Mapping) else {}
        for key in (
            "total",
            "passed",
            "failed",
            "skipped",
            "reward_used",
            "placeholder_ratio",
            "gt_returns_source_mismatch",
        ):
            entry[key] = summary.get(key)
        entry["gt_returns_source_counts"] = summary.get("gt_returns_source_counts")
    return entry


def build_report(
    *,
    mixture_dir: str | Path = "agentic_rl_data/data/mixture_v1",
    raw_dir: str | Path = "agentic_rl_data/data/raw_unified",
    report_dir: str | Path | None = None,
    explicit_reports: Mapping[str, str] | None = None,
    strict: bool = False,
    allow_missing_steps: Sequence[str] = (),
    require_provenance: bool = False,
) -> dict[str, Any]:
    """Assemble the full report dict (pure: no writes, deterministic).

    ``strict`` gates plan §8 acceptance.  ``allow_missing_steps`` relaxes one or
    more upstream reports (e.g. ``("passrate",)`` before the rollout backend
    exists); ``require_provenance`` additionally demands an ``input_sha256`` in
    every upstream report (staleness guard, review R2-06 — opt-in because the
    step scripts do not write it yet).
    """
    allowed = {str(step) for step in allow_missing_steps}
    unknown_allowed = allowed - set(REPORT_SPECS)
    if unknown_allowed:
        raise ValueError(f"unknown --allow-missing-steps entries: {sorted(unknown_allowed)}")

    mixture_dir = Path(mixture_dir)
    raw_dir = Path(raw_dir)
    train_path = mixture_dir / "train.jsonl"
    held_path = mixture_dir / "heldout.jsonl"
    stats_path = mixture_dir / "mixture_stats.json"

    if not train_path.is_file():
        raise FileNotFoundError(f"missing {train_path}; run build_mixture first (plan §8 A1)")

    train = list(load_jsonl(train_path))
    heldout = list(load_jsonl(held_path)) if held_path.is_file() else []
    mixture_stats = _load_json(stats_path)

    paths = discover_reports(
        mixture_dir=mixture_dir,
        raw_dir=raw_dir,
        report_dir=report_dir,
        explicit=explicit_reports,
    )
    steps: dict[str, Any] = {}
    for step in REPORT_SPECS:
        path = paths.get(step)
        data = _load_json(path) if path is not None else None
        summary = _step_summary(step, data)
        if path is not None:
            summary["path"] = str(path)
        summary["input_sha256"] = (
            data.get("input_sha256") if isinstance(data, Mapping) else None
        )
        steps[step] = summary

    train_invalid = [str(task.get("id")) for task in train if validate_task(task)]
    held_invalid = [str(task.get("id")) for task in heldout if validate_task(task)]
    overlap = compute_overlap(train, heldout)
    train_sha = _sha256_file(train_path)
    held_sha = _sha256_file(held_path) if held_path.is_file() else None
    recorded = (mixture_stats or {}).get("outputs") or {}
    recorded_train_sha = (
        (recorded.get("train") or {}).get("sha256") if isinstance(recorded, Mapping) else None
    )
    recorded_held_sha = (
        (recorded.get("heldout") or {}).get("sha256") if isinstance(recorded, Mapping) else None
    )

    decontam = steps.get("decontam") or {}
    decontam_records = _as_int(decontam.get("bfcl_records"))
    # Audit column: hits DETECTED AND REMOVED by decontam.py (may be > 0).
    decontam_hits_removed = _as_int(decontam.get("hits_removed"))
    # Gating column: hits still present in the kept split (must be 0, plan §8 A2).
    decontam_residual = _as_int(decontam.get("residual_hits"))
    replay = steps.get("replay") or {}
    replay_failed = _as_int(replay.get("failed"))
    replay_skipped = _as_int(replay.get("skipped"))
    replay_passed = _as_int(replay.get("passed"))
    replay_reward_used = _as_int(replay.get("reward_used"))
    recorded_train_count = (
        (recorded.get("train") or {}).get("count") if isinstance(recorded, Mapping) else None
    )

    checks: dict[str, Any] = {
        "train_rows": len(train),
        "heldout_rows": len(heldout),
        "schema_invalid_train": len(train_invalid),
        "schema_invalid_heldout": len(held_invalid),
        "invalid_train_examples": train_invalid[:20],
        "id_overlap": overlap["id_overlap"],
        "dedup_key_overlap": overlap["dedup_key_overlap"],
        "group_overlap": overlap["group_overlap"],
        "train_sha256": train_sha,
        "train_sha256_recorded": recorded_train_sha,
        "train_sha256_match": (
            None if not recorded_train_sha else recorded_train_sha == train_sha
        ),
        "heldout_sha256": held_sha,
        "heldout_sha256_recorded": recorded_held_sha,
        "heldout_sha256_match": (
            None if not recorded_held_sha or held_sha is None else recorded_held_sha == held_sha
        ),
        "decontam_bfcl_records": decontam_records,
        "decontam_hits_removed": decontam_hits_removed,
        "decontam_residual_hits": decontam_residual,
        "decontam_zero_residual": (
            None if decontam_residual is None else decontam_residual == 0
        ),
        "decontam_provenance_known": bool(decontam.get("input_sha256")),
        "replay_failed": replay_failed,
        "replay_skipped": replay_skipped,
        "replay_reward_used": replay_reward_used,
        "mixture_stats_present": mixture_stats is not None,
        "recorded_train_count": recorded_train_count,
        "train_count_matches_stats": (
            None
            if recorded_train_count is None
            else recorded_train_count == len(train)
        ),
    }

    not_run = [name for name, entry in steps.items() if entry.get("status") != "run"]

    strict_failures: list[str] = []
    if strict:
        for step in REPORT_SPECS:
            if steps[step].get("status") != "run" and step not in allowed:
                strict_failures.append(
                    f"{step} report missing ([NOT RUN]) — acceptance unverified "
                    f"(allow with --allow-missing-steps {step})"
                )
        if steps["decontam"].get("status") == "run":
            if decontam_records in (None, 0):
                strict_failures.append(
                    "decontam index is empty: 0 BFCL records means '0 residual hits' is meaningless"
                )
            if decontam_residual is None:
                strict_failures.append(
                    "decontam residual_hits unknown: re-run decontam.py (v2.4 writes it); "
                    "removed hits are audit-only and cannot prove the final split is clean"
                )
            elif decontam_residual != 0:
                strict_failures.append(
                    f"decontam residual_hits={decontam_residual}; final split must contain 0 hits "
                    f"(plan §8 A2). Removed hits (audit): {decontam_hits_removed}"
                )
        if steps["replay"].get("status") == "run":
            if replay_failed != 0:
                strict_failures.append(f"replay_check failed={replay_failed}; expected 0")
            if replay_skipped not in (None, 0):
                strict_failures.append(
                    f"replay_check skipped={replay_skipped}; skips hide unverified trajectories"
                )
            if replay_passed is not None and (
                replay_reward_used is None or replay_reward_used < replay_passed
            ):
                strict_failures.append(
                    f"replay_check reward_used={replay_reward_used} < passed={replay_passed}; "
                    "passes without the real reward are not acceptance evidence"
                )
        if checks["heldout_rows"] <= 0:
            strict_failures.append(
                "heldout split is empty; build a real held-out before claiming acceptance"
            )
        if mixture_stats is None:
            strict_failures.append("mixture_stats.json missing: reproducibility cannot be verified")
        else:
            if not recorded_train_sha:
                strict_failures.append(
                    "mixture_stats.json records no train sha256; reproducibility cannot be verified"
                )
            if checks["train_count_matches_stats"] is False:
                strict_failures.append(
                    "train row count != mixture_stats.outputs.train.count (stale stats?)"
                )
        if checks["schema_invalid_train"]:
            strict_failures.append(
                f"schema-invalid train rows: {checks['schema_invalid_train']}"
            )
        if checks["schema_invalid_heldout"]:
            strict_failures.append(
                f"schema-invalid heldout rows: {checks['schema_invalid_heldout']}"
            )
        if checks["id_overlap"]:
            strict_failures.append(
                f"train/heldout id overlap: {checks['id_overlap']}"
            )
        if checks["dedup_key_overlap"]:
            strict_failures.append(
                f"train/heldout dedup-key overlap: {checks['dedup_key_overlap']}"
            )
        if checks["group_overlap"]:
            strict_failures.append(
                f"train/heldout group overlap: {checks['group_overlap']} (session-level leakage)"
            )
        if checks["train_sha256_match"] is False:
            strict_failures.append("train sha256 mismatch vs mixture_stats.json (not reproducible)")
        if checks["heldout_sha256_match"] is False:
            strict_failures.append("heldout sha256 mismatch vs mixture_stats.json (not reproducible)")
        if require_provenance:
            for step in REPORT_SPECS:
                if steps[step].get("status") == "run" and not steps[step].get("input_sha256"):
                    strict_failures.append(
                        f"{step} report has no input_sha256 (provenance required)"
                    )

    warnings: list[str] = []
    if steps["passrate"].get("status") != "run":
        warnings.append(
            "pass-rate filter report missing: meta.pass_rate may be null for every row "
            "(plan §3 step 6); mark the corresponding section [NOT RUN]"
        )
    if mixture_stats is None:
        warnings.append(f"mixture stats missing: {stats_path}")
    if checks["decontam_zero_residual"] is None:
        warnings.append(
            "decontam residual hits unknown (report missing or legacy decontam.py); "
            "0-hit acceptance cannot be claimed"
        )
    if checks["decontam_hits_removed"]:
        warnings.append(
            f"decontam removed {checks['decontam_hits_removed']} hit(s) — audit column only; "
            "the gating number is residual_hits"
        )
    if not train:
        warnings.append("train split is empty")

    return {
        "generator": "agentic_rl_data.scripts.report",
        "inputs": {
            "mixture_dir": str(mixture_dir),
            "raw_dir": str(raw_dir),
            "train": str(train_path),
            "heldout": str(held_path) if held_path.is_file() else None,
            "mixture_stats": str(stats_path) if stats_path.is_file() else None,
        },
        "steps": steps,
        "not_run": not_run,
        "mixture": {
            "stats": {
                key: (mixture_stats or {}).get(key)
                for key in (
                    "seed",
                    "total_train_target",
                    "total_train_actual",
                    "heldout_target",
                    "heldout_actual",
                    "config_sha256",
                    "filters_applied",
                    "warnings",
                )
            },
            "categories": (mixture_stats or {}).get("categories") or {},
            "source_pools": (mixture_stats or {}).get("sources") or {},
        },
        "distributions": {
            "train": compute_distributions(train),
            "heldout": compute_distributions(heldout),
        },
        "checks": checks,
        "warnings": warnings,
        "strict_failures": strict_failures,
    }


# ---------------------------------------------------------------------------
# markdown rendering (deterministic)
# ---------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    rows = list(rows)
    if not rows:
        return ["_（无数据）_", ""]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(cell) for cell in row) + " |")
    lines.append("")
    return lines


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render the report dict as deterministic markdown (no timestamps)."""
    checks = report.get("checks") or {}
    dist_train = (report.get("distributions") or {}).get("train") or {}
    dist_held = (report.get("distributions") or {}).get("heldout") or {}
    mixture = report.get("mixture") or {}
    stats = mixture.get("stats") or {}
    steps = report.get("steps") or {}
    lines: list[str] = []
    add = lines.append

    add("# agentic_rl_data — 数据报告（data_report.md）")
    add("")
    add("> 生成器：`agentic_rl_data.scripts.report`（只读、确定性输出，无时间戳）。")
    add("> 规格映射：plan §3 step 8（每来源加载数 / 各步丢弃与原因 / 最终配比 / 轮数·工具数·通过率分布）"
        " + §8 A2/A4（去污染 0 命中、报告齐备）。")
    add("> 严格的**未执行**标注：`[NOT RUN]` 表示对应上游报告不存在，数字未知，不代表通过。")
    add("")

    # ---- 0. summary -----------------------------------------------------
    add("## 0. 摘要")
    add("")
    summary_rows = [
        ("train 行数", checks.get("train_rows")),
        ("heldout 行数", checks.get("heldout_rows")),
        ("train 目标 / 实际", f"{stats.get('total_train_target')} / {stats.get('total_train_actual')}"),
        ("heldout 目标 / 实际", f"{stats.get('heldout_target')} / {stats.get('heldout_actual')}"),
        ("seed", stats.get("seed")),
        ("config sha256", (stats.get("config_sha256") or "")[:16] + "…" if stats.get("config_sha256") else None),
        ("schema 非法（train / heldout）", f"{checks.get('schema_invalid_train')} / {checks.get('schema_invalid_heldout')}"),
        ("train/heldout id 重叠", checks.get("id_overlap")),
        ("train/heldout dedup-key 重叠", checks.get("dedup_key_overlap")),
        ("去污染残留命中（门禁）", checks.get("decontam_residual_hits")),
        ("去污染已剔除（审计）", checks.get("decontam_hits_removed")),
        ("replay 失败 / 跳过", f"{_fmt(checks.get('replay_failed'))} / {_fmt(checks.get('replay_skipped'))}"),
    ]
    lines.extend(_md_table(("指标", "值"), summary_rows))
    add("")

    # ---- 1. upstream steps ---------------------------------------------
    add("## 1. 上游各步骤状态（[RUN] / [NOT RUN]）")
    add("")
    step_rows = []
    for name, entry in steps.items():
        step_rows.append((name, "RUN" if entry.get("status") == "run" else "**NOT RUN**", entry.get("path", "—")))
    lines.extend(_md_table(("步骤", "状态", "报告文件"), step_rows))
    if report.get("not_run"):
        add(f"未执行步骤：`{', '.join(report['not_run'])}`。相关小节按 [NOT RUN] 渲染，不提供推断数字。")
        add("")

    # ---- 2. per-source load --------------------------------------------
    add("## 2. 每来源加载与丢弃（plan §3 step 1）")
    add("")
    build_raw = steps.get("build_raw") or {}
    if build_raw.get("status") == "run":
        rows = []
        for name, payload in sorted((build_raw.get("sources") or {}).items()):
            rows.append(
                (
                    name,
                    payload.get("rows"),
                    payload.get("loaded"),
                    payload.get("emitted"),
                    payload.get("skipped_invalid"),
                    payload.get("skipped_by_reason"),
                    payload.get("unattributed_skips"),
                )
            )
        lines.extend(
            _md_table(
                ("source", "rows", "loaded", "emitted", "skipped_invalid", "skipped_by_reason", "unattributed_skips"),
                rows,
            )
        )
        add(f"来源：`{build_raw.get('path')}`")
    else:
        add("**[NOT RUN]** 未找到 `_build_report.json`（`scripts/build_raw.py` 的输出）。")
    add("")

    # ---- 3. drops per step ---------------------------------------------
    add("## 3. 各步丢弃数与原因")
    add("")
    filter_rounds = steps.get("filter_rounds") or {}
    add("### 3.1 轮数过滤（step 2）")
    add("")
    if filter_rounds.get("status") == "run":
        lines.extend(
            _md_table(
                ("source", "in", "kept", "dropped"),
                [
                    (name, payload.get("in"), payload.get("kept"), payload.get("dropped"))
                    for name, payload in sorted((filter_rounds.get("per_source") or {}).items())
                ],
            )
        )
        add(f"来源：`{filter_rounds.get('path')}`")
    else:
        add("**[NOT RUN]** 未找到 `_filter_rounds_report.json`。")
    add("")

    add("### 3.2 去重（step 3）")
    add("")
    dedup = steps.get("dedup") or {}
    if dedup.get("status") == "run":
        lines.extend(
            _md_table(
                ("指标", "值"),
                [
                    ("input", dedup.get("input")),
                    ("exact_dropped", dedup.get("exact_dropped")),
                    ("near_dropped", dedup.get("near_dropped")),
                    ("kept", dedup.get("kept")),
                    ("against_tasks", dedup.get("against_tasks")),
                ],
            )
        )
        add(f"来源：`{dedup.get('path')}`")
    else:
        add("**[NOT RUN]** 未找到 `_dedup_report.json`。")
    add("")

    add("### 3.3 去污染（step 4）")
    add("")
    decontam = steps.get("decontam") or {}
    if decontam.get("status") == "run":
        lines.extend(
            _md_table(
                ("source", "in", "kept", "dropped（命中，审计）", "residual_hits（门禁）"),
                [
                    (
                        name,
                        payload.get("in"),
                        payload.get("kept"),
                        payload.get("dropped"),
                        payload.get("residual_hits"),
                    )
                    for name, payload in sorted((decontam.get("per_source") or {}).items())
                ],
            )
        )
        add(
            f"BFCL 记录数：{decontam.get('bfcl_records')}；索引记录数：{decontam.get('index_records')}；"
            f"来源：`{decontam.get('bfcl_path')}`。"
        )
        add(
            f"命中**已剔除**（审计，可 > 0）：{decontam.get('hits_removed')}；"
            f"**残留命中（门禁，必须为 0）**：{decontam.get('residual_hits')}（plan §8 A2）。"
        )
        add(f"来源：`{decontam.get('path')}`")
    else:
        add("**[NOT RUN]** 未找到 `_decontam_report.json` —— 去污染未验证，不得宣称 0 命中。")
    add("")

    add("### 3.4 通过率过滤（step 6）")
    add("")
    passrate = steps.get("passrate") or {}
    if passrate.get("status") == "run":
        lines.extend(
            _md_table(
                ("指标", "值"),
                [
                    ("n_samples", passrate.get("n_samples")),
                    ("kept", passrate.get("kept")),
                    ("dropped", passrate.get("dropped")),
                    ("kept_without_rate", passrate.get("kept_without_rate")),
                ],
            )
        )
        add(f"来源：`{passrate.get('path')}`")
    else:
        add("**[NOT RUN]** 未找到通过率过滤报告（需要用户提供 rollout pass-rate 文件，见 open_questions Q-02）。")
    add("")

    add("### 3.5 GT 轨迹重放（step 5）")
    add("")
    replay = steps.get("replay") or {}
    if replay.get("status") == "run":
        lines.extend(
            _md_table(
                ("指标", "值"),
                [
                    ("total", replay.get("total")),
                    ("passed", replay.get("passed")),
                    ("failed", replay.get("failed")),
                    ("skipped", replay.get("skipped")),
                    ("reward_used", replay.get("reward_used")),
                    ("placeholder_ratio", replay.get("placeholder_ratio")),
                ],
            )
        )
        add(f"来源：`{replay.get('path')}`")
    else:
        add("**[NOT RUN]** 未找到 replay 报告（`scripts/replay_check.py --out`）。")
    add("")

    # ---- 4. mixture ------------------------------------------------------
    add("## 4. 最终配比（plan §6 / §7）")
    add("")
    categories = mixture.get("categories") or {}
    if categories:
        rows = []
        for name, payload in sorted(categories.items()):
            if not isinstance(payload, Mapping):
                continue
            rows.append(
                (
                    name,
                    payload.get("share_target"),
                    payload.get("share_effective"),
                    payload.get("train_actual"),
                    payload.get("heldout_actual"),
                    payload.get("placeholder_train"),
                )
            )
        lines.extend(
            _md_table(("类别", "share_target", "share_effective", "train_actual", "heldout_actual", "placeholder_train"), rows)
        )
    else:
        add("**[NOT RUN]** `mixture_stats.json` 缺失或没有 categories。")
    add("")
    filters_applied = (stats.get("filters_applied") or {})
    if filters_applied:
        add("**过滤是否真的执行**（来自 `mixture_stats.json`，未执行即 false）：")
        add("")
        lines.extend(_md_table(("开关", "值"), sorted(filters_applied.items())))
    add("")

    # ---- 5. distributions ------------------------------------------------
    add("## 5. 分布（plan §3 step 8）")
    add("")
    for label, dist in (("train", dist_train), ("heldout", dist_held)):
        add(f"### 5.{1 if label == 'train' else 2} {label}（n={dist.get('total')}）")
        add("")
        add(f"- task_type：`{json.dumps(dist.get('task_types') or {}, ensure_ascii=False, sort_keys=True)}`")
        add(f"- env.type：`{json.dumps(dist.get('env_types') or {}, ensure_ascii=False, sort_keys=True)}`")
        add(f"- source：`{json.dumps(dist.get('sources') or {}, ensure_ascii=False, sort_keys=True)}`")
        turns = dist.get("turns") or {}
        add(f"- 轮数分布：`{json.dumps(turns.get('buckets') or {}, ensure_ascii=False, sort_keys=True)}`"
            f"（mean={_fmt(turns.get('mean'))}, min={_fmt(turns.get('min'))}, max={_fmt(turns.get('max'))}）")
        add(f"- 工具数分布：`{json.dumps(dist.get('tool_counts') or {}, ensure_ascii=False, sort_keys=True)}`")
        pr = dist.get("pass_rate") or {}
        add(f"- 通过率分布：`{json.dumps(pr.get('buckets') or {}, ensure_ascii=False, sort_keys=True)}`"
            f"（count={pr.get('count')}, missing={pr.get('missing')}, mean={_fmt(pr.get('mean'))}, median={_fmt(pr.get('median'))}）")
        add(f"- gt_returns_source：`{json.dumps(dist.get('gt_returns_sources') or {}, ensure_ascii=False, sort_keys=True)}`"
            f"（with_gt_tool_returns={dist.get('with_gt_tool_returns')}）")
        add(f"- license：`{json.dumps(dist.get('licenses') or {}, ensure_ascii=False, sort_keys=True)}`"
            f"（含 [UNVERIFIED] 行数：{dist.get('unverified_license_rows')}）")
        add(f"- group_id 覆盖：{dist.get('with_group_id')} / {dist.get('total')}"
            f"（{_fmt(dist.get('group_id_coverage'))}）")
        add("")
        add("_所有分布均从对应 JSONL 重算，可直接用文件行数复核。_")
        add("")

    # ---- 6. checks -------------------------------------------------------
    add("## 6. 一致性与可复现性检查")
    add("")
    lines.extend(
        _md_table(
            ("检查", "结果", "说明"),
            [
                ("train sha256 与 stats 一致", _fmt(checks.get("train_sha256_match")), (checks.get("train_sha256") or "")[:16] + "…"),
                ("heldout sha256 与 stats 一致", _fmt(checks.get("heldout_sha256_match")), (checks.get("heldout_sha256") or "—")[:16] + "…" if checks.get("heldout_sha256") else "—"),
                ("train/heldout id 重叠", checks.get("id_overlap"), "应为 0"),
                ("train/heldout dedup-key 重叠", checks.get("dedup_key_overlap"), "应为 0"),
                ("train/heldout group 重叠", checks.get("group_overlap"), "有 group_id 时应为 0"),
                ("schema 非法（train）", checks.get("schema_invalid_train"), "应为 0"),
                ("schema 非法（heldout）", checks.get("schema_invalid_heldout"), "应为 0"),
                ("去污染残留命中", checks.get("decontam_residual_hits"), "必须为 0（plan §8 A2）；已剔除数仅审计"),
                ("去污染已剔除（审计）", checks.get("decontam_hits_removed"), "可 > 0，不作门禁"),
                ("replay failed / skipped", f"{_fmt(checks.get('replay_failed'))} / {_fmt(checks.get('replay_skipped'))}", "均应为 0（plan §8 A3）"),
            ],
        )
    )
    add("")

    # ---- 7. not-run / strict --------------------------------------------
    add("## 7. 未执行项与严格模式")
    add("")
    add(f"- [NOT RUN] 步骤：`{', '.join(report.get('not_run') or []) or '无'}`")
    add("- 严格模式规则：")
    for rule in STRICT_RULES:
        add(f"  - {rule}")
    failures = report.get("strict_failures") or []
    if failures:
        add("")
        add(f"**严格模式失败 {len(failures)} 项：**")
        for item in failures:
            add(f"- {item}")
    elif report.get("inputs"):
        add("")
        add("（未启用 `--strict`，或全部检查通过；启用时以上任意失败都会返回非 0。）")
    add("")
    warnings = report.get("warnings") or []
    if warnings:
        add("**警告：**")
        for item in warnings:
            add(f"- {item}")
        add("")

    # ---- 8. reproduce ----------------------------------------------------
    add("## 8. 复现命令")
    add("")
    add("```powershell")
    add("cd D:\\lhy\\verl\\verl-toolRL")
    add("python agentic_rl_data/scripts/build_raw.py --all --cfg-json <cfg> --out-dir agentic_rl_data/data/raw_unified")
    add("python agentic_rl_data/scripts/filter_rounds.py --in-dir agentic_rl_data/data/raw_unified --out-dir agentic_rl_data/data/filtered_rounds")
    add("python agentic_rl_data/scripts/dedup.py --in agentic_rl_data/data/filtered_rounds/*.jsonl --out agentic_rl_data/data/deduped/train.jsonl")
    add("python agentic_rl_data/scripts/decontam.py --in-dir agentic_rl_data/data/deduped --bfcl-path <bfcl-data> --out-dir agentic_rl_data/data/decontaminated")
    add("python agentic_rl_data/scripts/replay_check.py --limit 10 --out agentic_rl_data/data/_replay_report.json")
    add("python -m agentic_rl_data.scripts.build_mixture --config agentic_rl_data/configs/mixture.yaml --out data/mixture_v1")
    add("python -m agentic_rl_data.scripts.report --mixture-dir data/mixture_v1 --raw-dir agentic_rl_data/data/raw_unified --out agentic_rl_data/reports/data_report.md --strict")
    add("# 通过率筛选（step 6）需要 rollout 结果；未产出时加 --allow-missing-steps passrate")
    add("python agentic_rl_data/scripts/report.py --mixture-dir data/mixture_v1 --strict --allow-missing-steps passrate")
    add("```")
    add("")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="report",
        description="Render reports/data_report.md from pipeline artifacts (plan §3 step 8).",
    )
    parser.add_argument("--mixture-dir", default="agentic_rl_data/data/mixture_v1")
    parser.add_argument("--raw-dir", default="agentic_rl_data/data/raw_unified")
    parser.add_argument("--report-dir", default=None, help="extra directory to search for upstream reports")
    parser.add_argument(
        "--report",
        action="append",
        default=[],
        metavar="STEP=PATH",
        help="explicit upstream report, e.g. --report decontam=/tmp/_decontam_report.json",
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"markdown output (default: {DEFAULT_OUT})")
    parser.add_argument("--json-out", default=None, help="also write the machine-readable report JSON")
    parser.add_argument("--strict", action="store_true", help="fail on any acceptance check")
    parser.add_argument(
        "--allow-missing-steps",
        default="",
        help="comma separated steps allowed to be [NOT RUN] under --strict "
        "(e.g. 'passrate' before rollout pass rates exist)",
    )
    parser.add_argument(
        "--require-provenance",
        action="store_true",
        help="also require every upstream report to carry input_sha256 (staleness guard)",
    )
    return parser


def _parse_explicit(items: Sequence[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--report expects STEP=PATH, got {item!r}")
        step, path = item.split("=", 1)
        parsed[step.strip()] = path.strip()
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    allowed = tuple(
        part.strip() for part in str(args.allow_missing_steps).split(",") if part.strip()
    )
    report = build_report(
        mixture_dir=args.mixture_dir,
        raw_dir=args.raw_dir,
        report_dir=args.report_dir,
        explicit_reports=_parse_explicit(args.report),
        strict=args.strict,
        allow_missing_steps=allowed,
        require_provenance=args.require_provenance,
    )
    markdown = render_markdown(report)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    checks = report["checks"]
    print(
        f"[report] train={checks['train_rows']} heldout={checks['heldout_rows']} "
        f"not_run={report['not_run']} "
        f"decontam_residual={checks['decontam_residual_hits']} "
        f"decontam_removed={checks['decontam_hits_removed']} "
        f"replay_failed={checks['replay_failed']} replay_skipped={checks['replay_skipped']}"
    )
    print(f"[report] wrote {out_path}")
    failures = report["strict_failures"]
    if failures:
        for item in failures:
            print(f"[report][STRICT] {item}")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "STRICT_RULES",
    "build_report",
    "compute_distributions",
    "compute_overlap",
    "discover_reports",
    "main",
    "render_markdown",
]
