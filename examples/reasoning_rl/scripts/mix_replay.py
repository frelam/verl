#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Replay a previous ``mix.py`` run while swapping in rebuilt domain parquets.

Use case
--------
You rebuilt ``code/`` and/or ``logic/`` (e.g. after the reward-side fixes) and
want a new ``train.parquet``/``val.parquet`` that keeps the *same* domain mix as
the previous run instead of being recomputed from the new pool sizes.

This script reads the previous run's ``mix_stats.json`` and replays:

* ``ratios``        -- same domain proportions (including a mid-training ``if`` share);
* ``seed``          -- same RNG seed;
* ``total_train``   -- same training-row count, so the achievable total is pinned
                       (``mix.py`` would otherwise re-derive it from pool sizes);
* ``val_by_ability``-- same per-domain val counts.

Same sampling algorithm as ``mix.py`` (val carved first, then per-domain
``int(total * ratio)`` rows), so the new artifact is directly comparable to the
old one.  ``train_by_source`` is *not* controllable: within a domain the pool is
sampled uniformly, so source shares follow the pool composition -- the report
prints old vs new so a rebuild that dropped rows is visible.

Configuration
-------------
Edit the ``CONFIG`` block below (or pass the matching CLI flags) and run::

    python3 examples/reasoning_rl/scripts/mix_replay.py

    # or fully explicit
    python3 examples/reasoning_rl/scripts/mix_replay.py \
        --old_dir  ~/data/reasoning_rl/final \
        --input_dir ~/data/reasoning_rl \
        --new_code  ~/data/reasoning_rl/code_v2/train_code.parquet \
        --new_logic ~/data/reasoning_rl/logic_v2/train_logic.parquet \
        --output_dir ~/data/reasoning_rl/final_v2

Nothing is written with ``--dry_run``; use ``--strict`` to abort instead of
silently sampling with replacement when a rebuilt pool cannot fill its share.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mix import DOMAIN_FILES, load_domain, stratified_take  # noqa: E402

# ================================ CONFIG ================================
# Directory of the PREVIOUS mix run (must contain mix_stats.json).
OLD_DIR = "~/data/reasoning_rl/final"
# Raw per-domain directory holding the unchanged domains (math/, stem/, [if/]).
# Rebuilt domains are overridden below.
INPUT_DIR = "~/data/reasoning_rl"
# Rebuilt code / logic output. Accepts either the parquet file or the directory
# the rebuild wrote to (the script looks for train_<domain>.parquet inside it).
# Leave as "" to reuse INPUT_DIR/code, INPUT_DIR/logic.
NEW_CODE = "~/data/reasoning_rl/code_v2"
NEW_LOGIC = "~/data/reasoning_rl/logic_v2"
# Where the new train.parquet / val.parquet / mix_stats.json are written.
OUTPUT_DIR = "~/data/reasoning_rl/final_v2"
# Optional overrides; None = replay the old run's value.
TOTAL_SIZE = None  # int -> force this many training rows
VAL_SIZE = None  # int -> force mix.py's val_size (per-domain count = int(V*ratio))
SEED = None  # int -> force the RNG seed
# =======================================================================

_REPORT_KEYS = ("total_train", "total_val", "ratios", "train_by_ability", "val_by_ability")


def _expand(path: str | None) -> str | None:
    return os.path.expanduser(path) if path else path


def _load_old_stats(old_stats_path: str) -> dict:
    if not os.path.isfile(old_stats_path):
        raise SystemExit(
            f"[mix_replay] previous mix_stats.json not found: {old_stats_path}\n"
            "Point --old_dir/--old_stats at the directory of the previous mix run."
        )
    with open(old_stats_path, encoding="utf-8") as f:
        old = json.load(f)
    if not isinstance(old, dict) or "ratios" not in old:
        raise SystemExit(f"[mix_replay] {old_stats_path} does not look like a mix.py mix_stats.json")
    return old


def _ordered_ratios(raw_ratios: dict) -> dict[str, float]:
    """Same iteration order mix.py uses (base domains, then 'if')."""
    ratios = {d: float(raw_ratios[d]) for d in DOMAIN_FILES if d in raw_ratios}
    missing = set(raw_ratios) - set(ratios)
    if missing:
        raise SystemExit(f"[mix_replay] unknown domain(s) in old ratios: {sorted(missing)}")
    if not ratios:
        raise SystemExit("[mix_replay] old mix_stats.json has an empty 'ratios' map")
    return ratios


def _val_counts(
    ratios: dict[str, float],
    pool_sizes: dict[str, int],
    old: dict,
    val_size_override: int | None,
) -> dict[str, int]:
    """Per-domain val row counts, replaying the previous run.

    Preference order: explicit ``--val_size`` -> the old per-domain
    ``val_by_ability`` counts -> brute-force the ``val_size`` that reproduces
    ``total_val`` -> the mix.py default of 256.
    """

    def counts_for(val_size: int) -> dict[str, int]:
        if val_size <= 0:
            return dict.fromkeys(ratios, 0)
        return {d: min(max(1, int(val_size * r)), pool_sizes[d] // 10) for d, r in ratios.items()}

    if val_size_override is not None:
        return counts_for(val_size_override)

    val_by_ability = old.get("val_by_ability") or {}
    total_val = old.get("total_val")
    by_ability = {d: int(val_by_ability[d]) for d in ratios if d in val_by_ability}
    if total_val is not None and len(by_ability) == len(ratios) and sum(by_ability.values()) == int(total_val):
        return {d: min(by_ability[d], pool_sizes[d] // 10) for d in ratios}

    if total_val is not None and int(total_val) > 0:
        for val_size in range(1, 2001):
            candidate = counts_for(val_size)
            if sum(candidate.values()) == int(total_val):
                return candidate

    return counts_for(256)


def replay(
    old_stats_path: str,
    input_dir: str,
    output_dir: str,
    path_overrides: dict[str, str | None] | None = None,
    total_size: int | None = None,
    val_size: int | None = None,
    seed: int | None = None,
    strict: bool = False,
    dry_run: bool = False,
) -> tuple[dict, dict]:
    """Build a new mix replaying the old run's configuration.

    Returns ``(new_stats, old_stats)``.
    """
    old = _load_old_stats(old_stats_path)
    ratios = _ordered_ratios(old["ratios"])
    seed = int(old.get("seed", 42)) if seed is None else int(seed)
    path_overrides = path_overrides or {}

    pools: dict[str, list[dict]] = {}
    for domain in ratios:
        pools[domain] = load_domain(input_dir, domain, path_overrides.get(domain))
    for domain, rows in pools.items():
        if not rows:
            raise SystemExit(f"[mix_replay] domain {domain!r} is empty; rebuild/path it first")

    pool_sizes = {d: len(rows) for d, rows in pools.items()}
    val_counts = _val_counts(ratios, pool_sizes, old, val_size)

    rng = random.Random(seed)

    # 1) carve val first so val rows never leak into train (same as mix.py).
    val_rows: list[dict] = []
    for domain, rows in pools.items():
        n_val = val_counts.get(domain, 0)
        if n_val <= 0:
            continue
        n_val = min(n_val, len(rows))
        taken = rng.sample(rows, n_val)
        taken_ids = {id(r) for r in taken}
        pools[domain] = [r for r in rows if id(r) not in taken_ids]
        val_rows.extend(taken)

    # 2) total training rows: pin to the old run's size.
    if total_size is None:
        total_size = old.get("total_train")
    if total_size is None:
        total_size = int(min(len(pools[d]) / ratios[d] for d in ratios))
    total_size = int(total_size)

    train_rows: list[dict] = []
    replenished: list[dict] = []
    for domain, rows in pools.items():
        target = int(total_size * ratios[domain])
        if target > len(rows):
            replenished.append({"domain": domain, "pool": len(rows), "target": target})
            if strict:
                raise SystemExit(
                    f"[mix_replay] strict: {domain} pool has {len(rows)} rows but needs {target} "
                    f"(total_size={total_size}); lower --total_size or rebuild a larger pool"
                )
        train_rows.extend(stratified_take(rng, rows, target, domain))

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    for i, row in enumerate(train_rows):
        row["extra_info"]["index"] = i
    for i, row in enumerate(val_rows):
        row["extra_info"]["index"] = i
        row["extra_info"]["split"] = "val"

    new_stats = {
        "total_train": len(train_rows),
        "total_val": len(val_rows),
        "seed": seed,
        "ratios": ratios,
        "train_by_ability": dict(Counter(r["ability"] for r in train_rows)),
        "train_by_source": dict(Counter(r["extra_info"]["source"] for r in train_rows)),
        "val_by_ability": dict(Counter(r["ability"] for r in val_rows)),
    }

    report = _compare(old, new_stats)
    report["replayed_from"] = old_stats_path
    report["input_dir"] = os.path.abspath(input_dir)
    report["output_dir"] = os.path.abspath(output_dir)
    report["val_counts"] = val_counts
    report["pool_sizes"] = pool_sizes
    report["replenished_with_replacement"] = replenished

    if not dry_run:
        import datasets

        os.makedirs(output_dir, exist_ok=True)
        datasets.Dataset.from_list(train_rows).to_parquet(os.path.join(output_dir, "train.parquet"))
        if val_rows:
            datasets.Dataset.from_list(val_rows).to_parquet(os.path.join(output_dir, "val.parquet"))
        with open(os.path.join(output_dir, "mix_stats.json"), "w", encoding="utf-8") as f:
            json.dump(new_stats, f, indent=2, ensure_ascii=False)
        with open(os.path.join(output_dir, "mix_replay_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    _print_report(report, new_stats, dry_run=dry_run)
    return new_stats, old


def _compare(old: dict, new: dict) -> dict:
    comparison = {key: old.get(key) == new.get(key) for key in _REPORT_KEYS}
    comparison["train_by_source"] = old.get("train_by_source") == new.get("train_by_source")
    return {
        "configuration_replayed": all(comparison[k] for k in _REPORT_KEYS),
        "matches_previous": comparison,
        "old_train_by_source": old.get("train_by_source"),
        "new_train_by_source": new.get("train_by_source"),
    }


def _print_report(report: dict, new_stats: dict, dry_run: bool) -> None:
    mode = "DRY RUN (nothing written)" if dry_run else "wrote train.parquet / val.parquet / mix_stats.json"
    print(f"[mix_replay] {mode}")
    print(
        f"[mix_replay] total_train={new_stats['total_train']} "
        f"total_val={new_stats['total_val']} seed={new_stats['seed']}"
    )
    print(f"[mix_replay] ratios={new_stats['ratios']}")
    print("[mix_replay] replay check vs previous run:")
    for key, ok in report["matches_previous"].items():
        print(f"    {key:20s} {'OK' if ok else 'DIFF'}")
    if report["matches_previous"]["train_by_source"]:
        print("[mix_replay] train_by_source: unchanged")
    else:
        print(f"[mix_replay] train_by_source OLD: {report['old_train_by_source']}")
        print(f"[mix_replay] train_by_source NEW: {report['new_train_by_source']}")
    if report["replenished_with_replacement"]:
        print("[mix_replay] WARNING: domain pool smaller than its target share -> sampled WITH replacement:")
        for item in report["replenished_with_replacement"]:
            print(f"    {item['domain']}: pool={item['pool']} target={item['target']} (duplicates possible)")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old_dir", default=None, help="Previous mix run directory containing mix_stats.json.")
    parser.add_argument("--old_stats", default=None, help="Explicit path to the previous mix_stats.json.")
    parser.add_argument("--input_dir", default=None, help="Raw per-domain directory (unchanged domains).")
    parser.add_argument("--output_dir", default=None, help="Destination for the new train/val/stats.")
    parser.add_argument("--new_code", default=None, help="Rebuilt code/train_code.parquet.")
    parser.add_argument("--new_logic", default=None, help="Rebuilt logic/train_logic.parquet.")
    parser.add_argument("--math_path", default=None, help="Override math parquet.")
    parser.add_argument("--stem_path", default=None, help="Override stem parquet.")
    parser.add_argument("--if_path", default=None, help="Override if parquet.")
    parser.add_argument("--total_size", type=int, default=None, help="Force training-row total.")
    parser.add_argument("--val_size", type=int, default=None, help="Force mix.py's val_size.")
    parser.add_argument("--seed", type=int, default=None, help="Force RNG seed.")
    parser.add_argument("--strict", action="store_true", help="Abort instead of sampling with replacement.")
    parser.add_argument("--dry_run", action="store_true", help="Plan only; write nothing.")
    return parser.parse_args(argv)


def _resolve_override(domain: str, override: str | None, input_dir: str) -> str | None:
    """Resolve a per-domain parquet override to an existing file.

    Accepts either the parquet itself or the rebuild output *directory*
    (``<dir>/train_<domain>.parquet``), so pointing ``--new_code`` at
    ``.../code_v2`` works as well as ``.../code_v2/train_code.parquet``.  A
    relative path is resolved against ``--input_dir``, matching ``mix.py``'s own
    override handling.
    """
    if not override:
        return None
    path = _expand(override)
    if not os.path.isabs(path):
        path = os.path.join(input_dir, path)
    path = os.path.abspath(path)
    if os.path.isdir(path):
        candidate = os.path.join(path, f"train_{domain}.parquet")
        if os.path.isfile(candidate):
            return candidate
        found = sorted(glob.glob(os.path.join(path, "*.parquet")))
        hint = f" Found instead: {found}" if found else " The directory is empty."
        raise SystemExit(
            f"[mix_replay] {domain!r} override is a directory ({path}) with no train_{domain}.parquet.{hint}"
        )
    if not os.path.isfile(path):
        raise SystemExit(f"[mix_replay] {domain!r} override not found: {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    old_dir = os.path.abspath(_expand(args.old_dir if args.old_dir is not None else OLD_DIR))
    old_stats_path = (
        os.path.abspath(_expand(args.old_stats)) if args.old_stats else os.path.join(old_dir, "mix_stats.json")
    )
    input_dir = os.path.abspath(_expand(args.input_dir if args.input_dir is not None else INPUT_DIR))
    output_dir = os.path.abspath(_expand(args.output_dir if args.output_dir is not None else OUTPUT_DIR))

    raw_overrides = {
        "code": args.new_code if args.new_code is not None else NEW_CODE,
        "logic": args.new_logic if args.new_logic is not None else NEW_LOGIC,
        "math": args.math_path,
        "stem": args.stem_path,
        "if": args.if_path,
    }
    path_overrides = {domain: _resolve_override(domain, value, input_dir) for domain, value in raw_overrides.items()}

    total_size = args.total_size if args.total_size is not None else TOTAL_SIZE
    val_size = args.val_size if args.val_size is not None else VAL_SIZE
    seed = args.seed if args.seed is not None else SEED

    replay(
        old_stats_path=old_stats_path,
        input_dir=input_dir,
        output_dir=output_dir,
        path_overrides=path_overrides,
        total_size=total_size,
        val_size=val_size,
        seed=seed,
        strict=args.strict,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
