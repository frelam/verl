"""Step 1: load each raw source -> ``data/raw_unified/{source}.jsonl``.

Pure layer: :func:`build_source` returns ``(tasks, stats)`` and
:func:`build_all` returns the aggregate report.  CLI layer writes the JSONL
files with ``schema.dump_jsonl`` (deterministic: sorted keys, UTF-8) plus a
``_build_report.json`` summary.

Usage (run from the repository root; the script bootstraps ``sys.path``)::

    python agentic_rl_data/scripts/build_raw.py --all \
        --cfg-json configs/raw_paths.json --out-dir agentic_rl_data/data/raw_unified

    # one source, local dump
    python agentic_rl_data/scripts/build_raw.py --source funreason_mt \
        --cfg '{"path": "/data/funreason_mt/train.jsonl"}'

``configs/raw_paths.json`` is a ``{source: cfg}`` mapping; ``cfg.json_path`` /
``cfg.parquet_path`` are accepted aliases of ``path`` (see loader docstrings).
No network access happens unless a loader is given only an ``hf_dataset`` id.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Mapping

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agentic_rl_data.loaders import available_sources, load_source  # noqa: E402
from agentic_rl_data.loaders.common import SourceStats  # noqa: E402
from agentic_rl_data.schema import (  # noqa: E402
    dump_jsonl,
    validate_task,
)

logger = logging.getLogger(__name__)


def build_source(
    source: str,
    cfg: Mapping[str, Any] | None = None,
    *,
    max_samples: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load one source; return ``(valid_tasks, stats)``.

    ``stats`` = ``{"source", "rows", "loaded", "skipped_invalid",
    "skipped_total", "unattributed_skips", "skipped_by_reason", "task_types",
    "errors"}``.  Row/skip counters come from ``common.SourceStats`` passed via
    ``cfg["_stats"]`` (T1-02): loaders record every drop reason they can
    attribute; the remainder is reported as ``unattributed_skips`` rather than
    silently disappearing.
    """
    cfg = dict(cfg or {})
    if max_samples is not None:
        cfg["max_samples"] = int(max_samples)
    stats_handle = SourceStats()
    cfg["_stats"] = stats_handle
    tasks: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "source": source,
        "loaded": 0,
        "skipped_invalid": 0,
        "task_types": {},
        "errors": [],
    }
    for task in load_source(source, cfg):
        errors = validate_task(task)
        if errors:
            stats["skipped_invalid"] += 1
            stats_handle.skip("invalid_schema")
            if len(stats["errors"]) < 20:
                stats["errors"].append({"id": task.get("id"), "errors": errors})
            continue
        tasks.append(task)
        stats_handle.emit()
        stats["loaded"] += 1
        task_type = str(task.get("task_type"))
        stats["task_types"][task_type] = stats["task_types"].get(task_type, 0) + 1
    stats.update(stats_handle.as_dict())
    return tasks, stats


def build_all(
    cfg_map: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    sources: list[str] | None = None,
    max_samples: int | None = None,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build every requested source; optionally write JSONL + aggregate report."""
    cfg_map = dict(cfg_map or {})
    requested = list(sources or available_sources())
    report: dict[str, Any] = {"sources": {}, "total_loaded": 0, "out_dir": None}
    for source in requested:
        try:
            tasks, stats = build_source(source, cfg_map.get(source), max_samples=max_samples)
        except Exception as exc:  # noqa: BLE001 - report the failure, keep going
            logger.warning("source %s failed: %s", source, exc)
            report["sources"][source] = {
                "source": source,
                "loaded": 0,
                "skipped_invalid": 0,
                "task_types": {},
                "errors": [{"fatal": str(exc)}],
            }
            continue
        report["sources"][source] = stats
        report["total_loaded"] += stats["loaded"]
        if out_dir is not None:
            path = Path(out_dir) / f"{source}.jsonl"
            dump_jsonl(tasks, path)
            stats["path"] = str(path)
    if out_dir is not None:
        report["out_dir"] = str(out_dir)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        report_path = Path(out_dir) / "_build_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        report["report_path"] = str(report_path)
    return report


def _parse_cfg_json(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    path = Path(raw)
    if path.exists() and path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build unified raw task JSONL per source.")
    parser.add_argument("--source", action="append", default=[], help="source name (repeatable)")
    parser.add_argument("--all", action="store_true", help="build all registered sources")
    parser.add_argument("--cfg", default=None, help="JSON string or path with one source cfg")
    parser.add_argument("--cfg-json", default=None, help="JSON string or path with a {source: cfg} mapping")
    parser.add_argument("--out-dir", default="agentic_rl_data/data/raw_unified")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    sources = args.source or (available_sources() if args.all else [])
    if not sources:
        parser.error("pass --source NAME (repeatable) or --all")

    cfg_map = _parse_cfg_json(args.cfg_json)
    one_cfg = _parse_cfg_json(args.cfg) if args.cfg else None
    if one_cfg is not None:
        for source in sources:
            cfg_map.setdefault(source, dict(one_cfg))
    # raw output always goes through schema.dump_jsonl (deterministic bytes).
    report = build_all(cfg_map, sources=sources, max_samples=args.max_samples, out_dir=args.out_dir)
    print(json.dumps({k: v for k, v in report.items() if k != "sources"}, ensure_ascii=False, indent=2))
    for source, stats in report["sources"].items():
        print(f"{source}: loaded={stats['loaded']} invalid={stats['skipped_invalid']} types={stats['task_types']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
