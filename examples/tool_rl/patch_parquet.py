#!/usr/bin/env python3
"""Repair parquets generated *before* the tool_rl data-prep fixes in 3483165c.

Commit ``3483165c`` fixed two data-preparation defects, both of which are
already baked into previously written ``train.parquet`` / ``val.parquet``:

1. ``_normalize_tools`` returned a *shared* ``parameters`` object for every
   sample declaring the same tool, while ``_augment_param_rename`` mutates it
   in place.  A rename performed for one sample therefore rewrote other
   samples' declared schemas but not their labels.  Such a row carries a
   ground-truth argument key that no longer exists in the schema of the tool
   it belongs to, so a schema-conformant call can never match it (Dim1 caps
   at 0.5, Dim3 is always 0).  These rows are **dropped**: the label keeps the
   pre-rename key, so the original schema cannot be recovered from the parquet
   alone, and they are ~1% of the mix.

2. ``load_sealtools`` now withholds tools that are indistinguishable from a
   label tool (identical description), because Seal-Tools' label records the
   API the query was reverse-engineered from rather than a uniquely correct
   answer.  Affected rows **keep their label tool and lose the twin
   distractor** — the same effect as regenerating, which would have drawn a
   different same-field distractor in its place.

The same twin also breaks a rarer row type: a ``desc_replace`` negative is a
positive whose label tool's description was swapped for an unrelated one and
whose label was then emptied.  With the sibling still declared the query
stays perfectly servable, so the "no tools needed" label punished the correct
call.  The patch recovers the pre-swap description from the file-wide majority
and withholds the twin there too.

Row-local: no raw dataset, download or re-shuffling is involved, so the
existing mix (negative ratio, hard-replay tags, train/val split) is preserved.

Usage
-----
.. code-block:: bash

    # report only
    python examples/tool_rl/patch_parquet.py --data-dir "$HOME/data/tool_rl" --dry-run
    # write <stem>.patched.parquet next to the originals
    python examples/tool_rl/patch_parquet.py --data-dir "$HOME/data/tool_rl"
    # replace in place, keeping <stem>.parquet.bak
    python examples/tool_rl/patch_parquet.py --data-dir "$HOME/data/tool_rl" --in-place
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


def _to_tool_list(value: Any) -> list[dict[str, Any]]:
    """Normalise the ``tools`` column to plain dicts.

    pandas hands nested columns back as numpy object arrays, so ``tolist()``
    has to run before the container check.
    """
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, float) or not isinstance(value, list | tuple):  # NaN on an empty column
        return []
    return [dict(t) for t in value if isinstance(t, dict)]


def _to_call_list(value: Any) -> list[dict[str, Any]]:
    """Normalise ``extra_info["ground_truth_calls"]`` (JSON string) to dicts."""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, float):  # NaN
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list | tuple):
        return []
    out = []
    for item in value:
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except json.JSONDecodeError:
                continue
        if isinstance(item, dict):
            out.append(dict(item))
    return out


def _description(tool: dict[str, Any]) -> str:
    return str(tool.get("description", "")).strip().lower()


def _canonical_descriptions(records: list[dict[str, Any]]) -> dict[str, str]:
    """Most common description declared for each tool across the file.

    ``desc_replace`` swaps a label tool's description for an unrelated one, so
    that row's own description no longer matches the sibling API it used to be
    indistinguishable from.  The file-wide majority recovers the original.
    """
    counts: dict[str, Counter] = {}
    for row in records:
        for tool in _to_tool_list(row.get("tools")):
            name = tool.get("name")
            if name:
                counts.setdefault(name, Counter())[_description(tool)] += 1
    return {name: counter.most_common(1)[0][0] for name, counter in counts.items()}


def patch_row(
    tools: list[dict[str, Any]],
    gt_calls: list[dict[str, Any]],
    canonical: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], bool, int]:
    """Repair one row; return ``(tools, drop, n_twins_removed)``.

    ``drop`` is True when the label cannot be matched against the declared
    schema (defect 1); ``n_twins_removed`` counts indistinguishable distractors
    withheld (defect 2).  ``canonical`` maps a tool name to the description it
    carries elsewhere in the file, which recovers the twin of a
    ``desc_replace`` label tool whose description was swapped away.
    """
    canonical = canonical or {}
    gt_names = {c.get("name", "") for c in gt_calls}

    # Descriptions no declared distractor may duplicate: those of the label
    # tools, plus the pre-swap description of a ``desc_replace`` label tool.
    protected: set[str] = set()
    # The swapped tool itself: it now carries the unrelated description, and
    # must not be mistaken for a twin of its own original.
    replaced: set[str] = set()
    for tool in tools:
        name = tool.get("name", "")
        if name in gt_names:
            protected.add(_description(tool))
        original = canonical.get(name)
        if original and original != _description(tool):
            replaced.add(name)
            protected.add(original)

    kept: list[dict[str, Any]] = []
    twins_removed = 0
    for tool in tools:
        name = tool.get("name", "")
        if name not in gt_names and name not in replaced and _description(tool) in protected:
            twins_removed += 1
            continue
        kept.append(tool)

    declared = {t.get("name"): t for t in kept}
    for call in gt_calls:
        tool = declared.get(call.get("name"))
        if tool is None:
            continue  # label tool not declared at all: a different defect
        # A parquet round trip unifies nested struct fields, so a property the
        # tool does not declare comes back as ``key: None`` instead of being
        # absent — both mean "not declared".
        properties = ((tool.get("parameters") or {}).get("properties")) or {}
        if any(properties.get(key) is None for key in (call.get("arguments") or {})):
            return kept, True, twins_removed
    return kept, False, twins_removed


def patch_frame(df, name: str) -> tuple[Any, dict[str, int]]:
    """Patch one parquet frame in memory; returns ``(df, stats)``."""
    stats = {
        "rows_in": len(df),
        "dropped_unmatchable_label": 0,
        "rows_with_twin_removed": 0,
        "twins_removed": 0,
        "rows_out": 0,
    }
    out = []
    records = df.to_dict("records")
    canonical = _canonical_descriptions(records)
    for row in records:
        extra = row.get("extra_info")
        extra = dict(extra) if isinstance(extra, dict) else {}
        gt_calls = _to_call_list(extra.get("ground_truth_calls"))
        tools, drop, twins = patch_row(_to_tool_list(row.get("tools")), gt_calls, canonical)
        if drop:
            stats["dropped_unmatchable_label"] += 1
            continue
        stats["twins_removed"] += twins
        if twins:
            stats["rows_with_twin_removed"] += 1
        row["tools"] = tools
        extra["tools"] = tools
        extra["index"] = len(out)  # keep the id compact after dropping rows
        row["extra_info"] = extra
        out.append(row)

    import pandas as pd

    stats["rows_out"] = len(out)
    print(
        f"[patch] {name}: {stats['rows_in']} rows -> {stats['rows_out']} "
        f"(dropped {stats['dropped_unmatchable_label']} unmatchable-label, "
        f"withheld {stats['twins_removed']} twin tool(s) from "
        f"{stats['rows_with_twin_removed']} rows)"
    )
    return pd.DataFrame(out), stats


def patch_file(path: Path, *, in_place: bool, dry_run: bool) -> None:
    import pandas as pd

    df = pd.read_parquet(path)
    patched, _ = patch_frame(df, path.name)
    if dry_run or in_place:
        if not dry_run:
            backup = path.with_suffix(path.suffix + ".bak")
            shutil.copy2(path, backup)
            print(f"[patch] backup -> {backup}")
    if dry_run:
        print(f"[patch] dry run: {path} not written")
        return
    target = path if in_place else path.with_name(path.stem + ".patched" + path.suffix)
    patched.to_parquet(target, index=False)
    print(f"[patch] wrote {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch pre-3483165c tool_rl parquets in place.")
    parser.add_argument("--data-dir", required=True, help="Directory holding train.parquet / val.parquet")
    parser.add_argument(
        "--files",
        default="train.parquet,val.parquet",
        help="Comma-separated file names inside --data-dir (missing ones are skipped).",
    )
    parser.add_argument("--in-place", action="store_true", help="Overwrite the original (keeps a .bak copy).")
    parser.add_argument("--dry-run", action="store_true", help="Report the changes without writing anything.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    for name in (n.strip() for n in args.files.split(",")):
        if not name:
            continue
        path = data_dir / name
        if not path.exists():
            print(f"[patch] {path} not found — skipped")
            continue
        patch_file(path, in_place=args.in_place, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
