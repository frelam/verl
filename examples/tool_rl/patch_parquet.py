#!/usr/bin/env python3
"""Repair parquets generated *before* the tool_rl data-prep fix in 3483165c.

``_normalize_tools`` used to return a *shared* ``parameters`` object for every
sample declaring the same tool, while ``_augment_param_rename`` mutates that
object in place.  A parameter renamed for one sample therefore also rewrote
other samples' declared schemas — but their labels kept the old key, so a
schema-conformant call could never match them (Dim1 caps at 0.5, Dim3 is 0).
Measured on Seal-Tools: 331/5000 rows, spread over 168 tools.

The rename itself is intentional: the declared parameter name changed, so the
label must change with it.  This script applies exactly that repair to an
existing parquet, without re-downloading, re-sampling distractors, re-running
the negative mix or re-splitting train/val.

How the new key is found
------------------------
``_augment_param_rename`` moves the whole property spec (type + description)
under the new name, so the **description is the anchor** back to the original:

    tool.jsonl : country       -> "The country ..."        (original)
    schema     : input_value   -> "The country ..."        (renamed)

Only the rows whose key cannot be resolved are dropped (3 of 352 damaged calls
on the Seal-Tools run).  The raw tool table is fetched through
``prepare_data._fetch_raw`` (same cache the loader uses, 2.5 MB); when it is
unavailable the script falls back to a conservative heuristic — repair only
when the call has exactly one unmapped key and the tool exactly one
generically-named property — and drops the rest.

Sibling tools (identical descriptions, e.g. ``getMatchInfo`` /
``getFootballMatchInfo``) are **left alone** on positive rows: the reward
accepts either as the label's tool, see ``reward/verifier.py:_tool_equivalence``.
They are withheld on ``desc_replace`` negatives, where the label was emptied
after swapping the label tool's description — a sibling still answers that
query, so abstention would be punished (the generator fix in
``_augment_desc_replace`` swaps siblings too; this covers existing files).

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
import re
import shutil
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.tool_rl.prepare_data import (  # noqa: E402
    _GENERIC_PARAM_NAMES,
    _IRRELEVANT_DESCRIPTIONS,
    _SEAL_TOOLS_REPO,
    _fetch_raw,
    _format_gt,
)

_REFERENCE_RE = re.compile(r"\nReference:\n(.*)$", re.DOTALL)


# ============================================================================
# Container normalisation — pandas hands nested columns back as arrays
# ============================================================================


def _to_tool_list(value: Any) -> list[dict[str, Any]]:
    """Normalise the ``tools`` column to plain dicts."""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, float) or not isinstance(value, list | tuple):  # NaN
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


# ============================================================================
# Description anchor (original parameter names)
# ============================================================================


def _load_sealtools_anchor() -> dict[str, dict[str, Any]]:
    """``tool -> {"description": ..., "params": {name: description}}``."""
    try:
        text = _fetch_raw(
            _SEAL_TOOLS_REPO,
            "Seal-Tools_Dataset/tool.jsonl",
            branch="master",
            timeout=300,
        )
    except Exception as exc:  # noqa: BLE001 - any fetch failure falls back
        print(f"[patch] tool table unavailable ({exc}) — using the name heuristic")
        return {}
    if not text:
        print("[patch] tool table unavailable — using the name heuristic")
        return {}

    anchor: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        name = raw.get("api_name", "")
        if not name:
            continue
        anchor[name] = {
            "description": str(raw.get("api_description", "")),
            "params": {
                param: str((spec or {}).get("description", ""))
                for param, spec in (raw.get("parameters") or {}).items()
                if isinstance(spec, dict)
            },
        }
    return anchor


def _normalise(text: Any) -> str:
    return str(text or "").strip().lower()


def _resolve_key(
    tool_name: str,
    key: str,
    properties: dict[str, Any],
    anchor: dict[str, dict[str, Any]],
    unmapped: list[str],
) -> str | None:
    """Find the parameter ``key`` was renamed to, or None."""
    original = anchor.get(tool_name, {}).get("params", {}).get(key)
    if original:
        candidates = [
            param
            for param, spec in properties.items()
            if isinstance(spec, dict) and _normalise(spec.get("description")) == _normalise(original)
        ]
        if len(candidates) == 1:
            return candidates[0]

    # Fallback: only when the mapping cannot be ambiguous.
    if len(unmapped) == 1:
        generic = [p for p, spec in properties.items() if isinstance(spec, dict) and p in _GENERIC_PARAM_NAMES]
        if len(generic) == 1:
            return generic[0]
    return None


def repair_row(
    tools: list[dict[str, Any]],
    gt_calls: list[dict[str, Any]],
    anchor: dict[str, dict[str, str]] | None = None,
) -> tuple[list[dict[str, Any]] | None, int]:
    """Rename label keys to the schema's current parameter names.

    Returns ``(repaired_calls, keys_renamed)``; ``repaired_calls`` is None when
    a key cannot be mapped and the row has to be dropped.
    """
    anchor = anchor or {}
    by_name = {t.get("name"): t for t in tools}

    repaired: list[dict[str, Any]] = []
    renamed_total = 0
    for call in gt_calls:
        tool = by_name.get(call.get("name"))
        if tool is None:
            repaired.append(call)  # label tool not declared: a different defect
            continue
        # A parquet round trip unifies nested struct fields, so a property the
        # tool does not declare comes back as ``key: None`` instead of absent.
        properties = ((tool.get("parameters") or {}).get("properties")) or {}
        args = dict(call.get("arguments") or {})
        unmapped = [k for k in args if properties.get(k) is None]
        if not unmapped:
            repaired.append({"name": call.get("name", ""), "arguments": args})
            continue

        for key in unmapped:
            new_key = _resolve_key(call.get("name", ""), key, properties, anchor, unmapped)
            if new_key is None:
                return None, renamed_total
            args[new_key] = args.pop(key)
            renamed_total += 1
        repaired.append({"name": call.get("name", ""), "arguments": args})
    return repaired, renamed_total


def strip_orphan_siblings(
    tools: list[dict[str, Any]],
    anchor: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Withhold tools that still answer a ``desc_replace`` negative's query.

    ``_augment_desc_replace`` swaps the label tool's description for an
    unrelated one and empties the label, so the row must be answered by
    abstaining.  A declared tool that was indistinguishable from the label
    tool (its description *equalled* the original one) still fits the query —
    leaving it in makes the reward score the correct call as spurious.
    """
    anchor = anchor or {}
    irrelevant = {_normalise(d) for d in _IRRELEVANT_DESCRIPTIONS}
    swapped = {t.get("name") for t in tools if _normalise(t.get("description")) in irrelevant}
    originals = {
        _normalise(anchor.get(name, {}).get("description"))
        for name in swapped
        if anchor.get(name, {}).get("description")
    }
    if not originals:
        return tools, 0
    kept = [t for t in tools if _normalise(t.get("description")) not in originals]
    return kept, len(tools) - len(kept)


def _relabel(label: str, calls: list[dict[str, Any]]) -> str:
    """Rebuild the human-readable label string, keeping a Reference trailer."""
    if not calls:
        return label
    rebuilt = _format_gt(calls)
    match = _REFERENCE_RE.search(label or "")
    return rebuilt + ("\nReference:\n" + match.group(1) if match else "")


# ============================================================================
# Frame / file plumbing
# ============================================================================


def patch_frame(df, name: str, anchor: dict[str, dict[str, Any]] | None = None) -> tuple[Any, dict[str, int]]:
    """Repair one parquet frame in memory; returns ``(df, stats)``."""
    import pandas as pd

    stats = {
        "rows_in": len(df),
        "rows_repaired": 0,
        "keys_renamed": 0,
        "rows_dropped": 0,
        "rows_with_sibling_withheld": 0,
        "siblings_withheld": 0,
        "rows_out": 0,
    }
    out = []
    for row in df.to_dict("records"):
        extra = row.get("extra_info")
        extra = dict(extra) if isinstance(extra, dict) else {}
        gt_calls = _to_call_list(extra.get("ground_truth_calls"))
        if gt_calls:
            repaired, renamed = repair_row(_to_tool_list(row.get("tools")), gt_calls, anchor)
            if repaired is None:
                stats["rows_dropped"] += 1
                continue
            if renamed:
                stats["rows_repaired"] += 1
                stats["keys_renamed"] += renamed
                gt_calls = repaired
                extra["ground_truth_calls"] = json.dumps(gt_calls, ensure_ascii=False)
                reward_model = dict(row.get("reward_model") or {})
                reward_model["ground_truth"] = _relabel(reward_model.get("ground_truth", ""), gt_calls)
                row["reward_model"] = reward_model
        elif extra.get("augmented") == "desc_replace":
            tools, withheld = strip_orphan_siblings(_to_tool_list(row.get("tools")), anchor)
            if withheld:
                stats["rows_with_sibling_withheld"] += 1
                stats["siblings_withheld"] += withheld
                row["tools"] = tools
                extra["tools"] = tools

        extra["index"] = len(out)  # keep the id compact after dropping rows
        row["extra_info"] = extra
        out.append(row)

    stats["rows_out"] = len(out)
    print(
        f"[patch] {name}: {stats['rows_in']} rows -> {stats['rows_out']} "
        f"(repaired {stats['rows_repaired']} rows / {stats['keys_renamed']} keys, "
        f"dropped {stats['rows_dropped']} unresolvable, "
        f"withheld {stats['siblings_withheld']} orphan sibling(s) from "
        f"{stats['rows_with_sibling_withheld']} desc_replace rows)"
    )
    return pd.DataFrame(out), stats


def patch_file(
    path: Path,
    *,
    in_place: bool,
    dry_run: bool,
    anchor: dict[str, dict[str, Any]] | None,
) -> None:
    import pandas as pd

    patched, _ = patch_frame(pd.read_parquet(path), path.name, anchor)
    if dry_run:
        print(f"[patch] dry run: {path} not written")
        return
    if in_place:
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
        print(f"[patch] backup -> {backup}")
    target = path if in_place else path.with_name(path.stem + ".patched" + path.suffix)
    patched.to_parquet(target, index=False)
    print(f"[patch] wrote {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair pre-3483165c tool_rl parquets (label parameter keys).")
    parser.add_argument("--data-dir", required=True, help="Directory holding train.parquet / val.parquet")
    parser.add_argument(
        "--files",
        default="train.parquet,val.parquet",
        help="Comma-separated file names inside --data-dir (missing ones are skipped).",
    )
    parser.add_argument("--in-place", action="store_true", help="Overwrite the original (keeps a .bak copy).")
    parser.add_argument("--dry-run", action="store_true", help="Report the changes without writing anything.")
    args = parser.parse_args()

    anchor = _load_sealtools_anchor()
    print(f"[patch] description anchor: {len(anchor)} tools")

    data_dir = Path(args.data_dir).expanduser()
    for name in (n.strip() for n in args.files.split(",")):
        if not name:
            continue
        path = data_dir / name
        if not path.exists():
            print(f"[patch] {path} not found — skipped")
            continue
        patch_file(path, in_place=args.in_place, dry_run=args.dry_run, anchor=anchor)


if __name__ == "__main__":
    main()
