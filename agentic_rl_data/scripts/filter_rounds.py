"""Step 2: turn-count filter (plan §3 step 2, CONTRACTS C9).

Frozen rule::

    effective_turns = max(len(reference.gt_calls),
                          meta.expected_turns or 0,
                          len(reference.gt_turns or 0))
    drop when effective_turns > threshold

Thresholds: global default 15; per-source overrides are configurable and
ASearcher is the documented special case (its original RL runs allow >100
turns).  All drop decisions are recorded (id / source / task_type / turns).

Pure layer: :func:`effective_turns` and :func:`filter_tasks`.
CLI layer: reads ``*.jsonl`` under ``--in-dir`` (or explicit files), writes
``{out_dir}/{source}.jsonl`` plus ``_filter_rounds_report.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agentic_rl_data.schema import dump_jsonl, load_jsonl  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_MAX_TURNS = 15
DEFAULT_SOURCE_MAX_TURNS = {"asearcher": 15}


def effective_turns(task: Mapping[str, Any]) -> int:
    """Frozen effective-turn count for one task (CONTRACTS C9)."""
    reference = task.get("reference") if isinstance(task.get("reference"), Mapping) else {}
    meta = task.get("meta") if isinstance(task.get("meta"), Mapping) else {}
    gt_calls = reference.get("gt_calls")
    gt_turns = reference.get("gt_turns")
    expected = meta.get("expected_turns")
    return max(
        len(gt_calls) if isinstance(gt_calls, list) else 0,
        int(expected) if isinstance(expected, int) and not isinstance(expected, bool) else 0,
        len(gt_turns) if isinstance(gt_turns, list) else 0,
    )


def threshold_for(source: str, max_turns: int, source_max_turns: Mapping[str, int] | None = None) -> int:
    overrides = dict(source_max_turns or DEFAULT_SOURCE_MAX_TURNS)
    return int(overrides.get(source, max_turns))


def filter_tasks(
    tasks: Iterable[Mapping[str, Any]],
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
    source_max_turns: Mapping[str, int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(kept, dropped)``; ``dropped`` entries carry the reason."""
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for task in tasks:
        source = str(task.get("source", ""))
        turns = effective_turns(task)
        limit = threshold_for(source, max_turns, source_max_turns)
        if turns > limit:
            dropped.append(
                {
                    "id": task.get("id"),
                    "source": source,
                    "task_type": task.get("task_type"),
                    "effective_turns": turns,
                    "threshold": limit,
                    "reason": "too_many_turns",
                }
            )
            continue
        kept.append(dict(task))
    return kept, dropped


def filter_files(
    in_paths: Iterable[str | Path],
    out_dir: str | Path,
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
    source_max_turns: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Filter JSONL files, write per-source outputs and return a report dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "max_turns": max_turns,
        "source_max_turns": dict(source_max_turns or DEFAULT_SOURCE_MAX_TURNS),
        "sources": {},
        "total_in": 0,
        "total_kept": 0,
        "total_dropped": 0,
    }
    for in_path in in_paths:
        path = Path(in_path)
        tasks = list(load_jsonl(path))
        kept, dropped = filter_tasks(tasks, max_turns=max_turns, source_max_turns=source_max_turns)
        source = str(tasks[0].get("source")) if tasks else path.stem
        out_path = out_dir / f"{source}.jsonl"
        written = dump_jsonl(kept, out_path)
        entry = report["sources"].setdefault(
            source,
            {"source": source, "input_files": [], "in": 0, "kept": 0, "dropped": 0, "drop_details": []},
        )
        entry["input_files"].append(str(path))
        entry["in"] += len(tasks)
        entry["kept"] += written
        entry["dropped"] += len(dropped)
        entry["drop_details"].extend(dropped[:50])
        entry["output"] = str(out_path)
        report["total_in"] += len(tasks)
        report["total_kept"] += written
        report["total_dropped"] += len(dropped)
    report_path = out_dir / "_filter_rounds_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Drop tasks whose effective turn count exceeds the budget.")
    parser.add_argument("--in-dir", default=None, help="directory with raw_unified/*.jsonl")
    parser.add_argument("--in", dest="in_files", action="append", default=[], help="explicit input JSONL")
    parser.add_argument("--out-dir", default="agentic_rl_data/data/filtered_rounds")
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--asearcher-max-turns", type=int, default=DEFAULT_SOURCE_MAX_TURNS["asearcher"])
    args = parser.parse_args(argv)

    in_paths: list[Path] = [Path(path) for path in args.in_files]
    if args.in_dir:
        in_paths.extend(sorted(Path(args.in_dir).glob("*.jsonl")))
    if not in_paths:
        parser.error("pass --in-dir or --in FILE")
    overrides = dict(DEFAULT_SOURCE_MAX_TURNS)
    overrides["asearcher"] = args.asearcher_max_turns
    report = filter_files(in_paths, args.out_dir, max_turns=args.max_turns, source_max_turns=overrides)
    print(json.dumps({k: v for k, v in report.items() if k != "sources"}, ensure_ascii=False, indent=2))
    for source, stats in report["sources"].items():
        print(f"{source}: in={stats['in']} kept={stats['kept']} dropped={stats['dropped']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
