"""Export unified task JSONL (from ``build_mixture``) to verl RLHFDataset parquet.

Plan §9 was written for slime; this project trains with **verl**, so this module
is the verl replacement of the data-format item.  It mirrors the on-disk schema
already used by ``examples/tool_rl/prepare_data_v2.py`` (which verl's
``RLHFDataset`` + async agent loops consume):

    data_source : str                       -> "agentic_rl"
    prompt      : list[{role, content}]     -> chat messages (RLHFDataset prompt_key)
    tools       : list[{name, description, parameters(JSON string)}]
    reward_model: {style, ground_truth}     -> rule style
    extra_info  : struct{...}               -> per-sample metadata + the full task

Why the full task in ``extra_info["agentic_task"]``: the multi-turn agent loop
(``agentic_rl_data.verl.agent_loop_agentic``) rebuilds the env / reference from
the row at rollout time; storing the canonical task JSON (not a re-derived
subset) keeps the reward's ground truth identical to training-data validation.

Tool ``parameters`` are stored as JSON **strings** on disk (same reason as the
existing tool_rl parquet: pyarrow cannot unify structs across heterogeneous
parameter shapes).  Consumers must ``json.loads`` them:
``agentic_rl_data.scripts.export_verl_parquet.normalize_tool_list``.

Usage
-----
.. code-block:: bash

    python -m agentic_rl_data.scripts.export_verl_parquet \
        --mixture-dir agentic_rl_data/data/mixture_v1 \
        --out-dir agentic_rl_data/data/verl

The output is deterministic for a fixed input (row order follows the JSONL).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:  # allow direct execution from the repo root
    sys.path.insert(0, str(_ROOT))

from agentic_rl_data.schema import TASK_TYPES, load_jsonl, validate_task  # noqa: E402

VERL_DATA_SOURCE = "agentic_rl"
DEFAULT_MIXTURE_DIR = "agentic_rl_data/data/mixture_v1"
DEFAULT_OUT_DIR = "agentic_rl_data/data/verl"


class ExportError(RuntimeError):
    """Invalid input or missing optional dependency."""


# ---------------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------------


def normalize_tool(tool: Mapping[str, Any]) -> dict[str, str]:
    """One tool schema in the on-disk struct shape (parameters as JSON string)."""
    parameters = tool.get("parameters")
    if not isinstance(parameters, str):
        parameters = json.dumps(parameters or {}, ensure_ascii=False, sort_keys=True)
    return {
        "name": str(tool.get("name") or ""),
        "description": str(tool.get("description") or ""),
        "parameters": parameters,
    }


def normalize_tool_list(tools: Any) -> list[dict[str, str]]:
    """List form of :func:`normalize_tool` (empty list for missing tools)."""
    return [normalize_tool(tool) for tool in tools or [] if isinstance(tool, Mapping)]


def tools_for_model(tools: Any) -> list[dict[str, Any]]:
    """Round-trip helper: on-disk tool cells -> dicts with parsed parameters.

    Agent loops / chat templates need ``parameters`` as a dict; the parquet
    stores a JSON string (see module docstring).
    """
    parsed: list[dict[str, Any]] = []
    for tool in tools or []:
        if not isinstance(tool, Mapping):
            continue
        parameters = tool.get("parameters")
        if isinstance(parameters, str):
            try:
                parameters = json.loads(parameters)
            except json.JSONDecodeError:
                parameters = {}
        parsed.append(
            {
                "name": str(tool.get("name") or ""),
                "description": str(tool.get("description") or ""),
                "parameters": parameters if isinstance(parameters, Mapping) else {},
            }
        )
    return parsed


def _ground_truth_label(task: Mapping[str, Any]) -> str:
    """Readable GT label for ``reward_model`` (logging/back-compat only)."""
    reference = task.get("reference") if isinstance(task.get("reference"), Mapping) else {}
    final_answer = reference.get("final_answer")
    if isinstance(final_answer, str) and final_answer:
        return final_answer
    gt_calls = reference.get("gt_calls")
    if gt_calls:
        return json.dumps(gt_calls, ensure_ascii=False, sort_keys=True)
    return ""


def task_to_row(task: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Convert one unified task into one verl RLHFDataset row.

    The row is deliberately explicit (no pandas inference): ``extra_info`` is a
    struct of nullable, natively-typed fields; JSON blobs stay strings.
    """
    messages = task.get("messages")
    if not isinstance(messages, list):
        raise ExportError(f"task {task.get('id')!r}: messages must be a list")
    prompt = [
        {"role": str(message.get("role")), "content": str(message.get("content"))}
        for message in messages
        if isinstance(message, Mapping)
    ]
    tools = normalize_tool_list(task.get("tools"))
    reference = task.get("reference") if isinstance(task.get("reference"), Mapping) else {}
    meta = task.get("meta") if isinstance(task.get("meta"), Mapping) else {}
    gt_calls = reference.get("gt_calls")
    return {
        "data_source": VERL_DATA_SOURCE,
        "prompt": prompt,
        "tools": tools,
        "reward_model": {"style": "rule", "ground_truth": _ground_truth_label(task)},
        "extra_info": {
            "index": int(index),
            "task_id": str(task.get("id") or f"row-{index}"),
            "source": str(task.get("source") or "unknown"),
            "task_type": str(task.get("task_type") or ""),
            # Canonical task JSON: the agent loop rebuilds env/reference from it.
            "agentic_task": json.dumps(task, ensure_ascii=False, sort_keys=True),
            "ground_truth_calls": (
                json.dumps(gt_calls, ensure_ascii=False, sort_keys=True)
                if gt_calls is not None
                else None
            ),
            "gt_returns_source": (
                str(meta["gt_returns_source"]) if meta.get("gt_returns_source") is not None else None
            ),
            "max_turns": int(meta.get("max_turns") or 15),
            "tool_selection": [tool["name"] for tool in tools],
        },
    }


def build_rows(
    tasks: Iterable[Mapping[str, Any]],
    *,
    start_index: int = 0,
    allow_invalid: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert tasks to rows; returns ``(rows, invalid)``.

    Invalid tasks are **not silently dropped**: with ``allow_invalid`` they are
    returned in the second list (with their schema errors) so the caller can
    report them; otherwise :class:`ExportError` is raised.
    """
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for offset, task in enumerate(tasks):
        errors = validate_task(task)
        if errors:
            if not allow_invalid:
                raise ExportError(
                    f"task {task.get('id')!r} is schema-invalid: {errors}"
                )
            invalid.append({"task_id": str(task.get("id")), "errors": errors})
            continue
        rows.append(task_to_row(task, start_index + offset))
    return rows, invalid


# ---------------------------------------------------------------------------
# parquet schema (mirrors examples/tool_rl/prepare_data_v2.py)
# ---------------------------------------------------------------------------


def parquet_schema() -> Any:
    """Explicit Arrow schema; raises a clear error when pyarrow is missing."""
    try:
        import pyarrow as pa
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise ExportError(
            "pyarrow is required to export verl parquet: pip install pyarrow"
        ) from exc
    tool_struct = pa.struct(
        [
            ("name", pa.string()),
            ("description", pa.string()),
            ("parameters", pa.string()),  # JSON blob (see normalize_tool)
        ]
    )
    return pa.schema(
        [
            ("data_source", pa.string()),
            ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
            ("tools", pa.list_(tool_struct)),
            ("reward_model", pa.struct([("style", pa.string()), ("ground_truth", pa.string())])),
            (
                "extra_info",
                pa.struct(
                    [
                        ("index", pa.int64()),
                        ("task_id", pa.string()),
                        ("source", pa.string()),
                        ("task_type", pa.string()),
                        ("agentic_task", pa.string()),
                        ("ground_truth_calls", pa.string()),
                        ("gt_returns_source", pa.string()),
                        ("max_turns", pa.int64()),
                        ("tool_selection", pa.list_(pa.string())),
                    ]
                ),
            ),
        ]
    )


def write_parquet(rows: Sequence[Mapping[str, Any]], path: str | Path) -> str:
    """Write rows with the explicit schema; returns the file sha256."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise ExportError("pyarrow is required: pip install pyarrow") from exc
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(rows), schema=parquet_schema())
    pq.write_table(table, path)
    return sha256_file(path)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# one split + CLI
# ---------------------------------------------------------------------------


def export_split(
    input_path: str | Path,
    output_path: str | Path,
    *,
    max_samples: int | None = None,
    allow_invalid: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Export one JSONL split; returns a deterministic stats dict."""
    input_path = Path(input_path)
    if not input_path.is_file():
        raise ExportError(f"input JSONL not found: {input_path}")
    tasks: list[Mapping[str, Any]] = []
    for index, task in enumerate(load_jsonl(input_path)):
        if max_samples is not None and index >= max_samples:
            break
        tasks.append(task)
    rows, invalid = build_rows(tasks, allow_invalid=allow_invalid)
    by_type: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for row in rows:
        info = row["extra_info"]
        by_type[info["task_type"]] = by_type.get(info["task_type"], 0) + 1
        by_source[info["source"]] = by_source.get(info["source"], 0) + 1
    stats: dict[str, Any] = {
        "input": str(input_path),
        "output": str(output_path),
        "rows": len(rows),
        "skipped_invalid": len(invalid),
        "invalid_examples": invalid[:10],
        "by_task_type": dict(sorted(by_type.items())),
        "by_source": dict(sorted(by_source.items())),
    }
    if dry_run:
        stats["sha256"] = None
        return stats
    stats["sha256"] = write_parquet(rows, output_path)
    return stats


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="export_verl_parquet",
        description="Export unified task JSONL to verl RLHFDataset parquet (plan §9 -> verl).",
    )
    parser.add_argument("--mixture-dir", default=DEFAULT_MIXTURE_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--splits", default="train:train.jsonl,heldout:heldout.jsonl",
                        help="comma separated name:filename pairs")
    parser.add_argument("--max-samples", type=int, default=None, help="per-split cap (smoke runs)")
    parser.add_argument("--allow-invalid", action="store_true", help="count invalid rows instead of failing")
    parser.add_argument("--dry-run", action="store_true", help="convert and report, write nothing")
    return parser


def _parse_splits(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ExportError(f"--splits expects name:filename, got {item!r}")
        name, filename = item.split(":", 1)
        pairs.append((name.strip(), filename.strip()))
    if not pairs:
        raise ExportError("--splits is empty")
    return pairs


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    mixture_dir = Path(args.mixture_dir)
    out_dir = Path(args.out_dir)
    try:
        splits = _parse_splits(args.splits)
    except ExportError as exc:
        print(f"[export_verl] ERROR: {exc}", file=sys.stderr)
        return 2
    meta: dict[str, Any] = {"generator": "agentic_rl_data.scripts.export_verl_parquet", "splits": {}}
    failed = False
    for name, filename in splits:
        input_path = mixture_dir / filename
        output_path = out_dir / f"{name}.parquet"
        try:
            stats = export_split(
                input_path,
                output_path,
                max_samples=args.max_samples,
                allow_invalid=args.allow_invalid,
                dry_run=args.dry_run,
            )
        except (ExportError, OSError, ValueError) as exc:
            print(f"[export_verl] ERROR ({name}): {exc}", file=sys.stderr)
            failed = True
            continue
        meta["splits"][name] = stats
        print(
            f"[export_verl] {name:<7} rows={stats['rows']} skipped_invalid={stats['skipped_invalid']} "
            f"tasks={stats['by_task_type']} out={stats['output']}"
        )
    if not args.dry_run:
        meta_path = out_dir / "verl_meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[export_verl] meta: {meta_path}")
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "VERL_DATA_SOURCE",
    "ExportError",
    "build_rows",
    "export_split",
    "main",
    "normalize_tool",
    "normalize_tool_list",
    "parquet_schema",
    "task_to_row",
    "tools_for_model",
    "write_parquet",
]
