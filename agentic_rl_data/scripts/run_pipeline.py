"""One-command data build pipeline (plan §3 steps 1-8).

Wraps the individual step scripts so a single command can go from source loaders
to ``data/mixture_v1`` + ``reports/data_report.md``::

    python -m agentic_rl_data.scripts.run_pipeline \
        --config agentic_rl_data/configs/pipeline.json

Step order (plan §3)::

    load → rounds → dedup → decontam → replay → passrate → mix → report

Design rules
------------
* **Fail-closed but configurable.** A step whose prerequisites are configurable
  (BFCL data, rollout pass-rate file) is reported as ``skipped`` with the exact
  reason instead of silently producing a "final" dataset.
* **No hidden shell.** Every step is executed in-process by calling the step
  module's ``main(argv)``; ``--dry-run`` prints the plan and writes nothing, so
  the same command can be inspected before running.
* **Deterministic manifest.** ``<data>/_pipeline_manifest.json`` records step
  status and the exact output dirs, no timestamps.
* **Per-source preservation.** ``dedup`` and ``passrate`` operate on the
  combined stream (cross-source duplicates must be visible), then the result is
  split back into ``{source}.jsonl`` because ``build_mixture`` reads per-source
  files.

Exit codes: ``0`` all steps ok/skipped; ``1`` any step failed; ``2`` bad config
or CLI usage.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:  # allow direct execution from the repo root
    sys.path.insert(0, str(_ROOT))

from agentic_rl_data.schema import dump_jsonl, load_jsonl  # noqa: E402
from agentic_rl_data.scripts import (  # noqa: E402
    build_mixture,
    build_raw,
    decontam,
    dedup,
    export_verl_parquet,
    filter_by_passrate,
    filter_rounds,
    replay_check,
    report,
)

STEP_ORDER: tuple[str, ...] = (
    "load",
    "rounds",
    "dedup",
    "decontam",
    "replay",
    "passrate",
    "mix",
    "report",
    "export",
)

DEFAULT_CONFIG = "agentic_rl_data/configs/pipeline.json"
MANIFEST_NAME = "_pipeline_manifest.json"
COMBINED_NAME = "_combined.jsonl"

#: Path keys every pipeline config must provide.
REQUIRED_PATH_KEYS: tuple[str, ...] = (
    "raw_dir",
    "filtered_dir",
    "deduped_dir",
    "decontaminated_dir",
    "passrated_dir",
    "mixture_dir",
    "reports_dir",
    "mixture_config",
    "verl_dir",
)


class PipelineError(RuntimeError):
    """Configuration / prerequisite problem (exit code 2)."""


class SkipStep(Exception):
    """Step intentionally not executed; recorded as ``skipped`` (exit 0)."""


class StepFailed(Exception):
    """Step ran and failed; recorded as ``failed`` (exit code 1)."""


# ---------------------------------------------------------------------------
# config + small helpers
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> dict[str, Any]:
    """Read the pipeline JSON config; validate required path keys."""
    path = Path(path)
    if not path.is_file():
        raise PipelineError(f"config not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PipelineError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PipelineError(f"config root must be a JSON object: {path}")
    paths = payload.get("paths")
    if not isinstance(paths, dict):
        raise PipelineError("config.paths must be an object")
    missing = [
        key
        for key in REQUIRED_PATH_KEYS
        if not isinstance(paths.get(key), str) or not str(paths[key]).strip()
    ]
    if missing:
        raise PipelineError(f"config.paths missing keys: {missing}")
    return payload


def jsonl_files(directory: str | Path) -> list[str]:
    """Sorted ``*.jsonl`` files in a directory, skipping ``_``-prefixed scratch."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return [
        str(path)
        for path in sorted(directory.glob("*.jsonl"))
        if path.is_file() and not path.name.startswith("_")
    ]


def split_by_source(input_path: str | Path, out_dir: str | Path) -> dict[str, int]:
    """Split a combined JSONL into ``{sanitised-source}.jsonl`` files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[dict[str, Any]]] = {}
    for task in load_jsonl(input_path):
        source = str(task.get("source") or "unknown")
        groups.setdefault(source, []).append(task)
    counts: dict[str, int] = {}
    for source, tasks in sorted(groups.items()):
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", source) or "unknown"
        counts[source] = dump_jsonl(tasks, out_dir / f"{safe}.jsonl")
    return counts


def merge_jsonl(files: Sequence[str | Path], out_path: str | Path) -> int:
    """Merge JSONL files into one (order preserved) for cross-source steps."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return dump_jsonl(
        (task for path in files for task in load_jsonl(path)),
        out_path,
    )


def select_steps(
    steps: Sequence[str] = STEP_ORDER,
    *,
    start: str | None = None,
    stop: str | None = None,
    skip: Sequence[str] = (),
) -> list[str]:
    """Apply ``--from/--to/--skip`` to the fixed step order."""
    for name in (start, stop):
        if name is not None and name not in STEP_ORDER:
            raise PipelineError(f"unknown step {name!r}; known: {list(STEP_ORDER)}")
    skipped = list(skip)
    unknown_skip = [name for name in skipped if name not in STEP_ORDER]
    if unknown_skip:
        raise PipelineError(f"unknown --skip entries: {unknown_skip}")
    begin = STEP_ORDER.index(start) if start else 0
    end = STEP_ORDER.index(stop) + 1 if stop else len(STEP_ORDER)
    if begin > end:
        raise PipelineError("--from comes after --to")
    return [
        name
        for index, name in enumerate(STEP_ORDER)
        if begin <= index < end and name not in skipped
    ]


def _config_mapping(cfg: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = cfg.get(key)
    return dict(value) if isinstance(value, Mapping) else {}


def _positive_int(value: Any, name: str, default: int) -> int:
    if value is None:
        return default
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise PipelineError(f"{name} must be an int, got {value!r}") from exc
    if number <= 0:
        raise PipelineError(f"{name} must be positive, got {number}")
    return number


# ---------------------------------------------------------------------------
# step implementations: (cfg, paths, stage_dir, dry_run) -> (status, detail, new_stage)
# ---------------------------------------------------------------------------


def _require_dir_files(directory: str, *, step: str) -> list[str]:
    files = jsonl_files(directory)
    if not files:
        raise StepFailed(f"{step}: no *.jsonl under {directory} (did an earlier step run?)")
    return files


def step_load(cfg, paths, stage, dry):
    b = _config_mapping(cfg, "build_raw")
    sources = [str(item) for item in b.get("sources") or [] if str(item).strip()]
    argv = ["--out-dir", paths["raw_dir"]]
    if sources:
        for source in sources:
            argv += ["--source", source]
    else:
        argv += ["--all"]
    if b.get("cfg_json"):
        argv += ["--cfg-json", str(b["cfg_json"])]
    if b.get("max_samples") is not None:
        argv += ["--max-samples", str(_positive_int(b["max_samples"], "build_raw.max_samples", 1))]
    if dry:
        return "run", f"load → {paths['raw_dir']} ({'all sources' if not sources else ', '.join(sources)})", paths["raw_dir"]
    code = build_raw.main(argv)
    if code:
        raise StepFailed(f"load: build_raw exited {code}")
    return "run", f"load → {paths['raw_dir']}", paths["raw_dir"]


def step_rounds(cfg, paths, stage, dry):
    r = _config_mapping(cfg, "rounds")
    max_turns = _positive_int(r.get("max_turns"), "rounds.max_turns", 15)
    asearcher = _positive_int(r.get("asearcher_max_turns"), "rounds.asearcher_max_turns", 15)
    if dry:
        return "run", f"rounds → {paths['filtered_dir']} (≤{max_turns} turns)", paths["filtered_dir"]
    _require_dir_files(stage, step="rounds")
    code = filter_rounds.main(
        [
            "--in-dir",
            stage,
            "--out-dir",
            paths["filtered_dir"],
            "--max-turns",
            str(max_turns),
            "--asearcher-max-turns",
            str(asearcher),
        ]
    )
    if code:
        raise StepFailed(f"rounds: filter_rounds exited {code}")
    return "run", f"rounds → {paths['filtered_dir']}", paths["filtered_dir"]


def step_dedup(cfg, paths, stage, dry):
    d = _config_mapping(cfg, "dedup")
    if dry:
        return "run", f"dedup → {paths['deduped_dir']} (exact + MinHash over all sources)", paths["deduped_dir"]
    files = _require_dir_files(stage, step="dedup")
    combined = Path(paths["deduped_dir"]) / COMBINED_NAME
    argv = [
        "--in",
        *files,
        "--out",
        str(combined),
        "--report",
        str(Path(paths["deduped_dir"]) / "_dedup_report.json"),
        "--threshold",
        str(float(d.get("threshold", 0.85))),
        "--num-perm",
        str(_positive_int(d.get("num_perm"), "dedup.num_perm", 64)),
        "--bands",
        str(_positive_int(d.get("bands"), "dedup.bands", 16)),
        "--k",
        str(_positive_int(d.get("k"), "dedup.k", 5)),
    ]
    for against in d.get("against") or []:
        argv += ["--against", str(against)]
    code = dedup.main(argv)
    if code:
        raise StepFailed(f"dedup: exited {code}")
    counts = split_by_source(combined, paths["deduped_dir"])
    return "run", f"dedup → {paths['deduped_dir']} ({sum(counts.values())} rows, {len(counts)} sources)", paths["deduped_dir"]


def step_decontam(cfg, paths, stage, dry):
    d = _config_mapping(cfg, "decontam")
    bfcl = d.get("bfcl_path")
    if not bfcl:
        raise SkipStep("decontam.bfcl_path not configured (plan §8 A2 stays unverified)")
    if dry:
        return "run", f"decontam → {paths['decontaminated_dir']} (bfcl={bfcl})", paths["decontaminated_dir"]
    files = _require_dir_files(stage, step="decontam")
    argv = [
        # Explicit files, not --in-dir: the stage dir also holds `_combined.jsonl`
        # (cross-source scratch), which must not be decontaminated twice.
        "--in",
        *files,
        "--bfcl-path",
        str(bfcl),
        "--out-dir",
        paths["decontaminated_dir"],
        "--k",
        str(_positive_int(d.get("k"), "decontam.k", 13)),
        "--min-hits",
        str(_positive_int(d.get("min_hits"), "decontam.min_hits", 1)),
        "--jaccard-threshold",
        str(float(d.get("jaccard_threshold", 0.90))),
    ]
    for extra in d.get("extra_paths") or []:
        argv += ["--extra-path", str(extra)]
    code = decontam.main(argv)
    if code:
        raise StepFailed(f"decontam: exited {code}")
    return "run", f"decontam → {paths['decontaminated_dir']}", paths["decontaminated_dir"]


def step_replay(cfg, paths, stage, dry):
    r = _config_mapping(cfg, "replay")
    if r.get("enabled", True) is False:
        raise SkipStep("replay.enabled=false")
    limit = _positive_int(r.get("limit"), "replay.limit", 10)
    report_path = Path(paths["raw_dir"]).parent / "_replay_report.json"
    if dry:
        return "run", f"replay → {report_path} (limit={limit})", stage
    files = _require_dir_files(stage, step="replay")
    argv = [
        "--tasks",
        *files,
        "--out",
        str(report_path),
        "--limit",
        str(limit),
        "--task-types",
        str(r.get("task_types", "mt_fc")),
    ]
    if r.get("allow_tool_errors"):
        argv += ["--allow-tool-errors"]
    code = replay_check.main(argv)
    if code:
        if r.get("fail_on_error", False):
            raise StepFailed(f"replay: exited {code} (fail_on_error=true)")
        return "warn", f"replay → {report_path} (some trajectories failed, see report)", stage
    return "run", f"replay → {report_path}", stage


def step_passrate(cfg, paths, stage, dry):
    pr = _config_mapping(cfg, "passrate")
    if not pr.get("enabled"):
        raise SkipStep("passrate.enabled=false (needs a rollout pass-rate file)")
    rates = pr.get("rates")
    if not rates:
        raise SkipStep("passrate.rates not configured (rollout backend is plan §9 TODO)")
    if dry:
        return "run", f"passrate → {paths['passrated_dir']} (rates={rates})", paths["passrated_dir"]
    files = _require_dir_files(stage, step="passrate")
    combined = Path(paths["passrated_dir"]) / COMBINED_NAME
    merged = merge_jsonl(files, combined)
    argv = [
        "--tasks",
        str(combined),
        "--rates",
        str(rates),
        "--out",
        str(combined),
        "--report",
        str(Path(paths["passrated_dir"]) / "_passrate_report.json"),
        "--band",
        str(pr.get("band", "0.125,0.875")),
        "--n-samples",
        str(_positive_int(pr.get("n_samples"), "passrate.n_samples", 8)),
        "--missing",
        str(pr.get("missing", "keep")),
    ]
    if pr.get("smoothed"):
        argv += ["--smoothed"]
    code = filter_by_passrate.main(argv)
    if code:
        raise StepFailed(f"passrate: exited {code}")
    counts = split_by_source(combined, paths["passrated_dir"])
    return "run", f"passrate → {paths['passrated_dir']} ({merged} in, {sum(counts.values())} kept)", paths["passrated_dir"]


def step_mix(cfg, paths, stage, dry):
    m = _config_mapping(cfg, "mixture")
    src_files = jsonl_files(stage)
    if not src_files and not dry:
        raise StepFailed(f"mix: no per-source input files under {stage}")
    argv = [
        "--config",
        paths["mixture_config"],
        "--raw-dir",
        stage,
        "--out-dir",
        paths["mixture_dir"],
    ]
    if m.get("total_train") is not None:
        argv += ["--total-train", str(_positive_int(m["total_train"], "mixture.total_train", 1))]
    if m.get("seed") is not None:
        argv += ["--seed", str(int(m["seed"]))]
    if m.get("emit_curriculum"):
        argv += ["--emit-curriculum"]
    if m.get("use_loaders"):
        argv += ["--use-loaders"]
    if dry:
        return "run", f"mix → {paths['mixture_dir']} (raw={stage})", paths["mixture_dir"]
    code = build_mixture.main(argv)
    if code:
        raise StepFailed(f"mix: build_mixture exited {code}")
    return "run", f"mix → {paths['mixture_dir']}", paths["mixture_dir"]


def step_report(cfg, paths, stage, dry, *, strict_override: bool | None, report_dir: str | None):
    r = _config_mapping(cfg, "report")
    strict = bool(r.get("strict", False)) if strict_override is None else strict_override
    allowed = [str(item) for item in r.get("allow_missing_steps") or []]
    if r.get("auto_allow_passrate", True) and "passrate" not in allowed:
        allowed.append("passrate")
    out_md = Path(report_dir or paths["reports_dir"]) / "data_report.md"
    out_json = Path(report_dir or paths["reports_dir"]) / "data_report.json"
    if dry:
        return "run", f"report → {out_md} (strict={strict}, allow_missing={allowed})", stage
    argv = [
        "--mixture-dir",
        paths["mixture_dir"],
        "--raw-dir",
        paths["raw_dir"],
        "--out",
        str(out_md),
        "--json-out",
        str(out_json),
    ]
    if strict:
        argv += ["--strict"]
    if allowed:
        argv += ["--allow-missing-steps", ",".join(sorted(set(allowed)))]
    if r.get("require_provenance"):
        argv += ["--require-provenance"]
    code = report.main(argv)
    if code:
        raise StepFailed(f"report: strict acceptance failed (exit {code}); see {out_md}")
    return "run", f"report → {out_md}", stage


def step_export(cfg, paths, stage, dry):
    """Plan §9 (verl variant): mixture JSONL -> verl RLHFDataset parquet."""
    e = _config_mapping(cfg, "export")
    if e.get("enabled", True) is False:
        raise SkipStep("export.enabled=false")
    mixture_dir = paths["mixture_dir"]
    out_dir = paths["verl_dir"]
    argv = ["--mixture-dir", mixture_dir, "--out-dir", out_dir]
    if e.get("allow_invalid"):
        argv += ["--allow-invalid"]
    if dry:
        return "run", f"export → {out_dir}/{{train,heldout}}.parquet", stage
    if not (Path(mixture_dir) / "train.jsonl").is_file():
        raise StepFailed(f"export: missing {Path(mixture_dir) / 'train.jsonl'} (run mix first)")
    code = export_verl_parquet.main(argv)
    if code:
        raise StepFailed(f"export: export_verl_parquet exited {code}")
    return "run", f"export → {out_dir}", stage


#: name -> (runner, needs_stage_files)
STEP_RUNNERS = {
    "load": step_load,
    "rounds": step_rounds,
    "dedup": step_dedup,
    "decontam": step_decontam,
    "replay": step_replay,
    "passrate": step_passrate,
    "mix": step_mix,
    "export": step_export,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_pipeline",
        description="Run the plan §3 data pipeline end to end (load → … → report).",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help=f"pipeline JSON (default: {DEFAULT_CONFIG})")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    parser.add_argument("--from", dest="start_step", default=None, help="first step (load/rounds/dedup/decontam/replay/passrate/mix/report)")
    parser.add_argument("--to", dest="stop_step", default=None, help="last step (inclusive)")
    parser.add_argument("--skip", default="", help="comma separated steps to skip")
    parser.add_argument("--keep-going", action="store_true", help="continue after a failed step")
    parser.add_argument("--strict-report", action="store_true", help="force report --strict")
    parser.add_argument("--no-strict-report", action="store_true", help="force report without --strict")
    parser.add_argument("--report-dir", default=None, help="override reports_dir for data_report.md")
    parser.add_argument("--manifest", default=None, help="manifest path (default: <data>/_pipeline_manifest.json)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except PipelineError as exc:
        print(f"[pipeline] CONFIG ERROR: {exc}", file=sys.stderr)
        return 2
    paths = cfg["paths"]
    skip = [item.strip() for item in str(args.skip).split(",") if item.strip()]
    try:
        selected = select_steps(
            start=args.start_step,
            stop=args.stop_step,
            skip=skip,
        )
    except PipelineError as exc:
        print(f"[pipeline] ERROR: {exc}", file=sys.stderr)
        return 2

    strict_override: bool | None = None
    if args.strict_report and args.no_strict_report:
        print("[pipeline] ERROR: --strict-report and --no-strict-report are mutually exclusive", file=sys.stderr)
        return 2
    if args.strict_report:
        strict_override = True
    elif args.no_strict_report:
        strict_override = False

    stage = paths["raw_dir"]
    records: list[dict[str, str]] = []
    dry = bool(args.dry_run)
    if dry:
        print("[pipeline] dry-run: no files will be written")
    for name in selected:
        try:
            if name == "report":
                status, detail, new_stage = step_report(
                    cfg,
                    paths,
                    stage,
                    dry,
                    strict_override=strict_override,
                    report_dir=args.report_dir,
                )
            else:
                status, detail, new_stage = STEP_RUNNERS[name](cfg, paths, stage, dry)
        except SkipStep as exc:
            status, detail, new_stage = "skipped", str(exc), stage
        except StepFailed as exc:
            status, detail, new_stage = "failed", str(exc), stage
        except Exception as exc:  # noqa: BLE001 - report the failure, keep the plan visible
            status, detail, new_stage = "failed", f"{type(exc).__name__}: {exc}", stage
        records.append({"step": name, "status": status, "detail": detail})
        marker = "[dry-run] " if dry else ""
        print(f"{marker}[pipeline] {name:<9} {status:<7} {detail}")
        if status == "failed" and not args.keep_going:
            print("[pipeline] stopping after failure (use --keep-going to continue)", file=sys.stderr)
            break
        if new_stage:
            stage = new_stage

    failed = [row for row in records if row["status"] == "failed"]
    skipped = [row["step"] for row in records if row["status"] == "skipped"]
    if dry:
        print(f"[pipeline] dry-run complete: {len(records)} steps planned, "
              f"skipped={skipped or 'none'}")
        return 1 if failed else 0

    manifest_path = Path(args.manifest) if args.manifest else Path(paths["raw_dir"]).parent / MANIFEST_NAME
    manifest = {
        "generator": "agentic_rl_data.scripts.run_pipeline",
        "config": str(args.config),
        "paths": {key: str(value) for key, value in paths.items()},
        "steps": records,
        "failed": [row["step"] for row in failed],
        "skipped": skipped,
        "stage_after": stage,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[pipeline] manifest: {manifest_path}")
    if failed:
        print(f"[pipeline] FAILED steps: {[row['step'] for row in failed]}", file=sys.stderr)
        return 1
    print("[pipeline] all steps ok")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "STEP_ORDER",
    "PipelineError",
    "SkipStep",
    "StepFailed",
    "jsonl_files",
    "load_config",
    "main",
    "merge_jsonl",
    "select_steps",
    "split_by_source",
]
