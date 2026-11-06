#!/usr/bin/env python3
"""Pass-rate filtering for the Agentic RL phase-1 dataset (plan section 3 step 6).

Reads a rollout result file of ``(task_id, pass_rate)`` pairs plus the task JSONL,
keeps only tasks whose observed pass rate is strictly inside ``(0, 1)`` (all-correct
and all-wrong tasks are removed -- both have zero GRPO advantage variance), and
optionally applies a recommended band (default ``[1/N, 1-1/N]``) on top.

Outputs the filtered JSONL (with ``meta.pass_rate`` / sample counts written back,
never other meta keys deleted per CONTRACTS.md C10) and a JSON report that includes
the N=8 estimation-noise analysis required by the task card.

Why the band: with N=8 the observed pass rate grid is ``{0, .125, ..., 1}``, so the
mandatory open-interval rule already excludes all-zero / all-one rows; the config
band ``[0.125, 0.875]`` is exactly equivalent on that grid. The real cost is the
false-drop probability ``P(k=0 | p) + P(k=N | p) = (1-p)^N + p^N``: at N=8 a task
with true pass rate 5% is dropped 66% of the time. That is deliberate (near-zero
variance groups are wasted rollout), but it must be documented because it sets an
effective difficulty floor; see ``reports/mixture_analysis.md`` section 6.

Usage
-----
    python -m agentic_rl_data.scripts.filter_by_passrate \
        --tasks data/raw_unified/funreason_mt.jsonl \
        --rates data/passrate/funreason_mt.jsonl \
        --out data/raw_unified/funreason_mt.filtered.jsonl \
        --report data/passrate/funreason_mt.report.json \
        --band 0.125,0.875 --n-samples 8 --missing keep

    # noise table only, no task IO
    python -m agentic_rl_data.scripts.filter_by_passrate --noise-table

Rate file formats accepted (auto-detected):
  * JSONL: {"task_id": "...", "pass_rate": 0.5}  (or {"id": ..., "successes": 3, "trials": 8})
  * JSON : list of the above, or {"task_id": 0.5} mapping
  * TSV  : ``task_id<TAB or comma>pass_rate[<TAB>n_samples]``

This file is stdlib-only apart from the repository's frozen ``schema`` helpers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

try:  # pragma: no cover - exercised on every run
    from agentic_rl_data import schema as schema
except ImportError as _import_exc:  # pragma: no cover
    raise SystemExit(
        f"cannot import agentic_rl_data.schema (cwd={Path.cwd()}): {_import_exc}"
    ) from _import_exc

#: [SUGGEST] recommended retention band from the plan / task card.
DEFAULT_BAND: tuple[float, float] = (0.125, 0.875)
DEFAULT_N_SAMPLES = 8
KEEP_REASONS = ("all_wrong", "all_correct", "outside_passrate_band", "too_few_samples")
ID_KEYS = ("task_id", "id", "task", "prompt_id")


class PassRateError(ValueError):
    """Invalid pass-rate input or filter parameters."""


# ===========================================================================
# 1. noise / estimation math (used by the report and by ablation planning)
# ===========================================================================

def estimate_se(pass_rate: float, n: int) -> float:
    """Standard error of an observed pass rate with ``n`` samples."""
    if n <= 0:
        raise PassRateError(f"n must be positive, got {n}")
    p = min(max(float(pass_rate), 0.0), 1.0)
    return math.sqrt(p * (1.0 - p) / n)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a pass-rate estimate (better near 0 / 1)."""
    if n <= 0:
        raise PassRateError(f"n must be positive, got {n}")
    if not (0 <= successes <= n):
        raise PassRateError(f"successes must be in [0, {n}], got {successes}")
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half_width = (
        z
        * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
        / denominator
    )
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def jeffreys_mean(successes: int, n: int) -> float:
    """Jeffreys-smoothed pass rate ``(k + 0.5) / (N + 1)``."""
    if n <= 0:
        raise PassRateError(f"n must be positive, got {n}")
    return (successes + 0.5) / (n + 1.0)


def false_exclusion_rate(pass_rate: float, n: int) -> float:
    """P(drop | true pass rate) for the mandatory (0,1) rule.

    A row is dropped when all N rollouts fail or all N succeed:
    ``(1-p)^N + p^N``. For N=8 and p=0.05 this is 0.663 -- the documented
    false-drop cost of using N=8 as a zero/one detector.
    """
    if n <= 0:
        raise PassRateError(f"n must be positive, got {n}")
    p = min(max(float(pass_rate), 0.0), 1.0)
    return (1.0 - p) ** n + p**n


def noise_rows(n_samples: int) -> dict[str, Any]:
    """N=8 estimation-noise payload embedded in the report."""
    rows = []
    for n in sorted({4, 8, 16, 24, 32, 64, 100, n_samples}):
        rows.append(
            {
                "n": n,
                "grid": 1.0 / n,
                "max_se": 0.5 / math.sqrt(n),
                "ci95_halfwidth_worst": 1.96 * 0.5 / math.sqrt(n),
                "drop_rate_at_p_0.05": false_exclusion_rate(0.05, n),
                "drop_rate_at_p_0.10": false_exclusion_rate(0.10, n),
                "drop_rate_at_p_0.20": false_exclusion_rate(0.20, n),
            }
        )
    return {
        "n_samples": n_samples,
        "grid_resolution": 1.0 / n_samples,
        "max_standard_error": 0.5 / math.sqrt(n_samples),
        "worst_case_ci95_halfwidth": 1.96 * 0.5 / math.sqrt(n_samples),
        "rows": rows,
        "recommendation": (
            "N=8 is a zero/one detector, not a difficulty estimator: "
            f"max SE={0.5 / math.sqrt(n_samples):.3f}, grid={1.0 / n_samples:.3f}. "
            "For SE<=0.05 use N>=100; for difficulty banding use N=16/32 on border "
            "tasks or a two-stage re-sample. Keep the mandatory open-interval rule."
        ),
    }


# ===========================================================================
# 2. rate-file parsing (deterministic, tolerant, all errors reported)
# ===========================================================================

@dataclass(frozen=True)
class RateRecord:
    task_id: str
    pass_rate: float
    successes: int | None = None
    trials: int | None = None

    @property
    def n_samples(self) -> int | None:
        return self.trials


def _record_from_mapping(row: Mapping[str, Any], *, source: str, line: int) -> RateRecord:
    tid = next((row[key] for key in ID_KEYS if isinstance(row.get(key), str) and row[key]), None)
    if not tid:
        raise PassRateError(f"{source}:{line}: missing task id (expected one of {ID_KEYS})")
    successes = row.get("successes", row.get("k"))
    trials = row.get("trials", row.get("n_samples", row.get("n")))
    rate = row.get("pass_rate")
    if rate is None and isinstance(successes, (int, float)) and isinstance(trials, int) and trials > 0:
        rate = float(successes) / trials
    if rate is None:
        raise PassRateError(f"{source}:{line}: missing pass_rate (or successes+trials)")
    if not isinstance(rate, (int, float)) or isinstance(rate, bool):
        raise PassRateError(f"{source}:{line}: pass_rate must be a number, got {rate!r}")
    rate = float(rate)
    if not (0.0 <= rate <= 1.0):
        raise PassRateError(f"{source}:{line}: pass_rate must be in [0, 1], got {rate}")
    n_samples: int | None = None
    for candidate in (trials, row.get("n_samples"), row.get("n")):
        if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
            n_samples = candidate
            break
    successes_int: int | None = None
    if isinstance(successes, int) and not isinstance(successes, bool) and successes >= 0:
        successes_int = successes
    return RateRecord(task_id=tid, pass_rate=rate, successes=successes_int, trials=n_samples)


def parse_rates_text(text: str, *, source: str = "<rates>") -> tuple[dict[str, RateRecord], list[str]]:
    """Parse a rate file; returns (records, parse_errors). Never silently drops."""
    errors: list[str] = []
    records: dict[str, RateRecord] = {}
    stripped = text.lstrip()
    payload: Any = None
    payload_error: json.JSONDecodeError | None = None
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            # JSONL (one object per line) also starts with "{": fall through to
            # the line-oriented parser instead of rejecting the whole file.
            payload_error = exc
    if payload is not None:
        if isinstance(payload, dict):
            # Either one record object or {task_id: pass_rate|record}
            if any(key in payload for key in ID_KEYS):
                entries = [payload]
            else:
                entries = []
                for tid, value in payload.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        entries.append({"task_id": tid, "pass_rate": float(value)})
                    elif isinstance(value, Mapping):
                        entries.append({"task_id": tid, **value})
                    else:
                        errors.append(f"{source}: bad value for {tid!r}: {value!r}")
        elif isinstance(payload, list):
            entries = payload
        else:
            raise PassRateError(f"{source}: top-level JSON must be an object or list")
        for i, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                errors.append(f"{source}: entry {i} is not an object")
                continue
            try:
                record = _record_from_mapping(entry, source=source, line=i)
            except PassRateError as exc:
                errors.append(str(exc))
                continue
            if record.task_id in records:
                errors.append(f"{source}: duplicate task id {record.task_id!r} (last one wins)")
            records[record.task_id] = record
        return records, errors
    if payload_error is not None and stripped.startswith("["):
        # A top-level JSON array cannot be JSONL: it is genuinely malformed.
        raise PassRateError(f"{source}: invalid JSON: {payload_error}") from payload_error

    # line-oriented: JSONL objects or two/three column TSV/CSV
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{source}:{lineno}: invalid JSON: {exc}")
                continue
            if not isinstance(entry, Mapping):
                errors.append(f"{source}:{lineno}: JSONL row must be an object")
                continue
            try:
                record = _record_from_mapping(entry, source=source, line=lineno)
            except PassRateError as exc:
                errors.append(str(exc))
                continue
        else:
            separator = "\t" if "\t" in line else ","
            parts = [part.strip() for part in line.split(separator)]
            if len(parts) < 2:
                errors.append(f"{source}:{lineno}: expected task_id{separator}pass_rate")
                continue
            tid, raw_rate = parts[0], parts[1]
            try:
                rate = float(raw_rate)
            except ValueError:
                errors.append(f"{source}:{lineno}: pass_rate is not a number: {raw_rate!r}")
                continue
            n_samples = None
            if len(parts) > 2:
                try:
                    parsed_n = int(float(parts[2]))
                    if parsed_n > 0:
                        n_samples = parsed_n
                except ValueError:
                    errors.append(f"{source}:{lineno}: n_samples is not an int: {parts[2]!r}")
            if not (0.0 <= rate <= 1.0):
                errors.append(f"{source}:{lineno}: pass_rate out of [0,1]: {rate}")
                continue
            record = RateRecord(task_id=tid, pass_rate=rate, trials=n_samples)
        if record.task_id in records:
            errors.append(f"{source}: duplicate task id {record.task_id!r} (last one wins)")
        records[record.task_id] = record
    return records, errors


def load_rates(path: str | Path) -> tuple[dict[str, RateRecord], list[str]]:
    rates_path = Path(path)
    if not rates_path.exists():
        raise PassRateError(f"rates file not found: {rates_path}")
    text = rates_path.read_text(encoding="utf-8")
    return parse_rates_text(text, source=str(rates_path))


def load_tasks(path: str | Path) -> list[dict[str, Any]]:
    tasks_path = Path(path)
    if not tasks_path.exists():
        raise PassRateError(f"tasks file not found: {tasks_path}")
    tasks = list(schema.load_jsonl(tasks_path))
    problems = []
    for i, task in enumerate(tasks):
        issues = schema.validate_task(task)
        if issues:
            problems.append(f"task #{i} id={task.get('id')!r}: {issues[:3]}")
    if problems:
        raise PassRateError(
            "input tasks are not schema-valid (filtering must not silently accept them):\n  - "
            + "\n  - ".join(problems[:10])
        )
    return tasks


# ===========================================================================
# 3. filtering (pure)
# ===========================================================================

def classify_rate(
    record: RateRecord,
    *,
    band: tuple[float, float] | None,
    min_samples: int = 1,
) -> str | None:
    """Return a drop reason, or None when the row may be kept."""
    if record.trials is not None and record.trials < min_samples:
        return "too_few_samples"
    rate = record.pass_rate
    if rate <= 0.0:
        return "all_wrong"
    if rate >= 1.0:
        return "all_correct"
    if band is not None and (rate < band[0] or rate > band[1]):
        return "outside_passrate_band"
    return None


def _write_back_meta(
    task: Mapping[str, Any], record: RateRecord, *, smoothed: bool
) -> dict[str, Any]:
    new_task = dict(task)
    meta = dict(task.get("meta") or {})
    meta["pass_rate"] = record.pass_rate
    meta["pass_rate_samples"] = (
        record.trials if record.trials is not None else None
    )
    meta["pass_rate_source"] = "rollout"
    if smoothed and record.trials:
        successes = record.successes
        if successes is None:
            successes = int(round(record.pass_rate * record.trials))
        meta["pass_rate_jeffreys"] = jeffreys_mean(successes, record.trials)
    new_task["meta"] = meta
    return new_task


def filter_tasks(
    tasks: Sequence[Mapping[str, Any]],
    rates: Mapping[str, RateRecord],
    *,
    band: tuple[float, float] | None = DEFAULT_BAND,
    missing: str = "keep",
    smoothed: bool = False,
    min_samples: int = 1,
) -> dict[str, Any]:
    """Apply the pass-rate rule; returns kept tasks + detailed drop accounting."""
    if missing not in ("keep", "drop"):
        raise PassRateError("missing must be 'keep' or 'drop'")
    if band is not None and not (0.0 < band[0] < band[1] < 1.0):
        raise PassRateError(f"band must satisfy 0 < min < max < 1, got {band}")
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    kept_without_rate = 0
    kept_by_task_type: dict[str, int] = {}
    dropped_by_task_type: dict[str, int] = {}

    for task in tasks:
        tid = str(task.get("id"))
        task_type = str(task.get("task_type"))
        record = rates.get(tid)
        if record is None:
            if missing == "drop":
                dropped.append({"id": tid, "task_type": task_type, "reason": "missing_rate"})
                reasons["missing_rate"] = reasons.get("missing_rate", 0) + 1
                dropped_by_task_type[task_type] = dropped_by_task_type.get(task_type, 0) + 1
                continue
            kept.append(dict(task))
            kept_without_rate += 1
            kept_by_task_type[task_type] = kept_by_task_type.get(task_type, 0) + 1
            continue
        reason = classify_rate(record, band=band, min_samples=min_samples)
        if reason is not None:
            dropped.append({"id": tid, "task_type": task_type, "reason": reason})
            reasons[reason] = reasons.get(reason, 0) + 1
            dropped_by_task_type[task_type] = dropped_by_task_type.get(task_type, 0) + 1
            continue
        kept.append(_write_back_meta(task, record, smoothed=smoothed))
        kept_by_task_type[task_type] = kept_by_task_type.get(task_type, 0) + 1

    return {
        "input_tasks": len(tasks),
        "kept": len(kept),
        "kept_without_rate": kept_without_rate,
        "dropped": len(dropped),
        "drop_reasons": reasons,
        "kept_by_task_type": kept_by_task_type,
        "dropped_by_task_type": dropped_by_task_type,
        "dropped_examples": dropped[:200],
        "tasks": kept,
    }


def build_report(
    params: Mapping[str, Any],
    stats: Mapping[str, Any],
    *,
    tasks_path: str,
    rates_path: str,
    parse_errors: Sequence[str],
) -> dict[str, Any]:
    """Deterministic JSON report (no timestamps; safe to diff between runs)."""
    n_samples = int(params.get("n_samples", DEFAULT_N_SAMPLES))
    return {
        "generator": "agentic_rl_data.scripts.filter_by_passrate",
        "tasks": tasks_path,
        "rates": rates_path,
        "params": dict(params),
        "stats": {
            key: value for key, value in stats.items() if key != "tasks"
        },
        "parse_errors": list(parse_errors),
        "noise": noise_rows(n_samples),
        "notes": [
            "Mandatory rule: pass_rate must be strictly inside (0,1); all-correct and "
            "all-wrong rows always dropped (zero GRPO advantage variance).",
            "Band [1/N, 1-1/N] is equivalent to the open interval on the N-sample grid; "
            "it exists to make the intended difficulty range explicit.",
            "N=8 is a zero/one detector: P(drop | p=0.05) = 0.663. Use N>=16/32 for "
            "difficulty banding; see the noise table.",
            "meta.pass_rate is written back as the raw observed rate; meta keys are "
            "never removed (CONTRACTS.md C10).",
        ],
    }


def _band_from_text(text: str | None) -> tuple[float, float] | None:
    if text is None or text.strip().lower() in ("default", ""):
        return DEFAULT_BAND
    if text.strip().lower() in ("off", "none", "open"):
        return None
    separator = "," if "," in text else " "
    parts = [part for part in text.replace(" ", "").split(separator) if part]
    if len(parts) != 2:
        raise PassRateError(f"band must be 'lo,hi' or 'off', got {text!r}")
    lo, hi = float(parts[0]), float(parts[1])
    if not (0.0 < lo < hi < 1.0):
        raise PassRateError(f"band must satisfy 0 < lo < hi < 1, got {text!r}")
    return lo, hi


# ===========================================================================
# 4. CLI
# ===========================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="filter_by_passrate",
        description=(
            "Keep tasks whose rollout pass rate is strictly inside (0,1) "
            "(plan section 3 step 6), write meta.pass_rate back and emit a report."
        ),
    )
    parser.add_argument("--tasks", help="input task JSONL")
    parser.add_argument("--rates", help="rollout pass-rate file (JSONL/JSON/TSV)")
    parser.add_argument("--out", help="filtered task JSONL output (default: <tasks>.filtered.jsonl)")
    parser.add_argument("--report", help="JSON report output (default: <out>.report.json)")
    parser.add_argument(
        "--band",
        default="0.125,0.875",
        help="retention band 'lo,hi' or 'off' (default: 0.125,0.875)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help="rollout samples per task used to produce the rates (default: 8)",
    )
    parser.add_argument(
        "--missing",
        choices=("keep", "drop"),
        default="keep",
        help="policy for tasks without a pass rate (default: keep; strict pipeline: drop)",
    )
    parser.add_argument(
        "--smoothed",
        action="store_true",
        help="also store meta.pass_rate_jeffreys (raw pass_rate is always kept)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="drop rows with fewer samples than this (0 = no check; 8 for strict)",
    )
    parser.add_argument("--dry-run", action="store_true", help="compute and print only")
    parser.add_argument("--noise-table", action="store_true", help="print the N-noise table and exit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        if args.noise_table:
            table = noise_rows(args.n_samples)
            for row in table["rows"]:
                print(
                    f"N={row['n']:>3} grid={row['grid']:.4f} max_SE={row['max_se']:.4f} "
                    f"drop@p=.05={row['drop_rate_at_p_0.05']:.3f} "
                    f"drop@p=.10={row['drop_rate_at_p_0.10']:.3f} "
                    f"drop@p=.20={row['drop_rate_at_p_0.20']:.3f}"
                )
            print(table["recommendation"])
            return 0
        if not args.tasks or not args.rates:
            raise PassRateError("--tasks and --rates are required (or use --noise-table)")
        band = _band_from_text(args.band)
        tasks = load_tasks(args.tasks)
        rates, parse_errors = load_rates(args.rates)
        stats = filter_tasks(
            tasks,
            rates,
            band=band,
            missing=args.missing,
            smoothed=args.smoothed,
            min_samples=args.min_samples,
        )
        kept = stats["tasks"]
        out_path = Path(args.out) if args.out else Path(args.tasks).with_suffix(".filtered.jsonl")
        report_path = Path(args.report) if args.report else out_path.with_suffix(".report.json")
        if not args.dry_run:
            schema.dump_jsonl(kept, out_path)
            report = build_report(
                {
                    "band": list(band) if band is not None else None,
                    "missing": args.missing,
                    "smoothed": args.smoothed,
                    "min_samples": args.min_samples,
                    "n_samples": args.n_samples,
                    "policy": "observed",
                },
                stats,
                tasks_path=str(args.tasks),
                rates_path=str(args.rates),
                parse_errors=parse_errors,
            )
            report["output"] = {"path": str(out_path), "count": len(kept)}
            report["output"]["sha256"] = hashlib.sha256(out_path.read_bytes()).hexdigest()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
        print(
            f"[filter_by_passrate] tasks={stats['input_tasks']} kept={stats['kept']} "
            f"dropped={stats['dropped']} reasons={stats['drop_reasons']} "
            f"kept_without_rate={stats['kept_without_rate']} band={band}"
        )
        if parse_errors:
            print(f"[filter_by_passrate] WARNING: {len(parse_errors)} rate parse error(s)", file=sys.stderr)
            for error in parse_errors[:10]:
                print(f"  {error}", file=sys.stderr)
        if not args.dry_run:
            print(f"  out={out_path} report={report_path}")
        return 0
    except (PassRateError, OSError, schema.SchemaError) as exc:
        print(f"[filter_by_passrate] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
