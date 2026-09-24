#!/usr/bin/env python3
"""Convert tool_rl v2 parquets whose ``extra_info`` is an Arrow map type.

``prepare_data_v2.write_parquet`` used to declare ``extra_info`` as
``map<string, string>``.  HF ``datasets`` (used by verl's RLHFDataset via
``datasets.load_dataset("parquet", ...)``) cannot map an Arrow map type to
a datasets dtype and crashes before reading a single row::

    ValueError: Arrow type map<string, string ('extra_info')> does not have
    a datasets dtype equivalent.

The writer now emits a natively-typed struct (see
``prepare_data_v2._parquet_schema``).  This script converts *existing*
files in place — no re-download, no re-sampling, no re-splitting.  In the
old format every non-string ``extra_info`` value was JSON-encoded (see the
old ``_cell``), so the conversion parses the JSON scalars back to their
native types (int index, bool answerable_direct, list tools); the JSON
blobs (ground_truth_calls, augment_detail) stay strings, exactly as the
reward expects.

Usage
-----
.. code-block:: bash

    # report only
    python examples/tool_rl/fix_map_parquet.py --data-dir /path/to/data --dry-run
    # convert in place, keeping <stem>.parquet.bak
    python examples/tool_rl/fix_map_parquet.py --data-dir /path/to/data
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.tool_rl.prepare_data_v2 import _parquet_schema, _tool_cells  # noqa: E402


def _decode(value: Any) -> Any:
    """Best-effort decode of a JSON-encoded scalar; strings stay strings."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def map_to_struct_extra(pairs: Any) -> dict[str, Any] | None:
    """Turn an ``extra_info`` map row (list of (key, value) pairs) into the
    struct row expected by ``_parquet_schema``.  Returns None if the row is
    not a map (already converted)."""
    if pairs is None or not isinstance(pairs, list):
        return None
    raw: dict[str, Any] = {}
    for k, v in pairs:
        raw[str(k)] = v
    tools = _decode(raw.get("tools"))
    return {
        "index": int(_decode(raw.get("index", 0))),
        "task_id": str(_decode(raw.get("task_id", "")) or ""),
        "source": str(_decode(raw.get("source", "")) or ""),
        # ``_tool_cells`` re-JSON-encodes parameter dicts that the map
        # format's blanket JSON-encoding had flattened back to dicts.
        "tools": _tool_cells(tools if isinstance(tools, list) else []),
        # JSON blobs: keep as string / None
        "ground_truth_calls": raw.get("ground_truth_calls"),
        "augmented": str(_decode(raw.get("augmented", "")) or ""),
        "answerable_direct": bool(_decode(raw.get("answerable_direct", False))),
        "augment_detail": raw.get("augment_detail"),
    }


def fix_file(path: Path, *, dry_run: bool, verify: bool) -> str:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    if not any(f.name == "extra_info" and str(f.type).startswith("map") for f in table.schema):
        return "skip (no map extra_info)"

    rows = table.to_pylist()
    converted = 0
    for r in rows:
        extra = map_to_struct_extra(r.get("extra_info"))
        if extra is not None:
            r["extra_info"] = extra
            converted += 1

    if dry_run:
        return f"would convert {converted}/{len(rows)} rows"

    new_table = table.from_pylist(rows, schema=_parquet_schema())
    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    pq.write_table(new_table, path)

    note = f"converted {converted}/{len(rows)} rows, backup -> {backup.name}"
    if verify:
        import datasets

        datasets.load_dataset("parquet", data_files=str(path))["train"]
        note += "; datasets.load_dataset OK"
    return note


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert map-typed extra_info columns to struct (HF datasets readable).")
    parser.add_argument("--data-dir", required=True, help="Directory holding the parquet files")
    parser.add_argument(
        "--files",
        default="train.parquet,val.parquet",
        help="Comma-separated file names inside --data-dir (missing ones are skipped).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report without writing anything.")
    parser.add_argument("--no-verify", action="store_true", help="Skip the post-write datasets.load_dataset check.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    failures = 0
    for name in (n.strip() for n in args.files.split(",")):
        if not name:
            continue
        path = data_dir / name
        if not path.exists():
            print(f"[fix] {path} not found — skipped")
            continue
        try:
            print(f"[fix] {path.name}: {fix_file(path, dry_run=args.dry_run, verify=not args.no_verify)}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[fix] {path.name}: FAILED — {exc}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
