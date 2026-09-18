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
"""Stage-2 mixer: the hallucination domain + the stage-1 mix (design doc section 7).

Follows ``mix.py``'s conventions deliberately -- carve val before sampling train,
scale by ratio, no replacement (warn instead of silently duplicating), shuffle,
re-index ``extra_info.index``, write ``mix_stats.json`` and mark val rows with
``extra_info.split = "val"``.  The parquet I/O helpers are imported from
``mix.py`` rather than re-implemented so the two artifacts cannot drift.

What is different from ``mix.py``
---------------------------------

The hallucination domain is **not** sampled by a single ratio.  Design doc
section 4.9.3 table B fixes a per-cell quota over ``(branch, data_source)``, and
the three v0.10 knobs rescale those cells:

* ``--halluc_total`` (20,000) -- the size of the whole hallucination domain;
* ``--halluc_unsolvable_ratio`` (0.6) -- its unsolvable share;
* ``--halluc_four_tier_ratio`` (0.6) -- the four-tier share *within* the unsolvable side.

``(branch, data_source)`` is a complete key for every cell: the SUM rows that go
four-tier and the SUM rows that go three-tier differ by branch, and the
synthesised SUM rows differ again by branch (``solvable_numeric``).  So the quota
table needs no extra ``error_type`` coupling to the adapters.

One cell deviates from the table as printed.  Table B row 1's 2,400 synthesised
distractor rows are split SUM 1,000 / UMWP 600 / K&K 400 / main pool 400, but a
K&K question answers with a *role sequence*, so its 400 rows cannot carry
``solvable_numeric``; they carry ``solvable_roles`` and share a cell with the
2,000 K&K rows of row 2.  The total is unchanged -- 4,400 + 2,000 stays 6,400,
and the solvable side stays 8,000 -- only the branch bookkeeping moves.  See
``distractor_synth.py``, which emits them that way.

Two invariants are hard, not advisory, and the mixer refuses to write an artifact
that violates them:

1. every template must carry **both** solvable and unsolvable rows.  If an option
   block or a verdict prompt correlated with solvability the policy could read the
   label off the prompt -- the design doc calls this out as a 100% shortcut.
2. a row's branch must agree with its own ``ground_truth.solvable``.

A cell whose pool is smaller than its quota is **filled short and reported**, with
the leftover budget redistributed to cells on the same side that have surplus.
Redistribution stays inside a side, so the 60/40 unsolvable split and the
four-tier/three-tier split survive a shortfall; the total is only allowed to come
out below ``--halluc_total`` when the pools genuinely cannot fill it.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mix import read_parquet_rows, resolve_parquet_files  # noqa: E402

import schema  # noqa: E402

# The artifact is written with schema's writer rather than mix.py's.  Both chunk
# by 10,000, but mix.py infers the Arrow schema from whichever rows land in chunk
# 1, and the hallucination rows are deliberately heterogeneous (only K&K fills
# `canonical_solution`, only the four-tier sources fill `options`).  If chunk 1
# happened to hold none of those, pyarrow would lock the column to list<null> and
# every later chunk carrying a real value would fail to cast.  schema's writer
# infers from a per-data_source probe instead, which covers every source by
# construction.
write_rows_parquet = schema.write_rows_parquet
from schema import BRANCH_SOLVABLE_JUDGE, BRANCH_SOLVABLE_NUMERIC, BRANCH_SOLVABLE_ROLES  # noqa: E402
from schema import BRANCH_UNSOLVABLE_BARE, BRANCH_UNSOLVABLE_DIAG  # noqa: E402

SOLVABLE_BRANCHES = (BRANCH_SOLVABLE_NUMERIC, BRANCH_SOLVABLE_ROLES, BRANCH_SOLVABLE_JUDGE)
UNSOLVABLE_BRANCHES = (BRANCH_UNSOLVABLE_DIAG, BRANCH_UNSOLVABLE_BARE)

#: The directory every adapter writes its ``DEFAULT_OUT`` into, and therefore the
#: default ``--halluc_dir``.  ``test_mix_halluc.py`` asserts the two stay equal for
#: every ``*_adapter.py`` in this directory: a mismatch here is silent -- the mixer
#: finds no rows and dies with "no hallucination rows found", which reads like a
#: missing build rather than a typo.
DEFAULT_ADAPTER_DIR = "~/data/reasoning_rl/halluc/built"

# Design doc section 4.9.3 table B, at the default knobs (20,000 / 0.6 / 0.6).
# Keyed by (branch, data_source) -- see the module docstring for why that key is
# complete.  ``SOURCE_MAIN`` is the stage-1 math pool used for the fourth
# synthesised-distractor slice; it is absent on a box without stage-1 data, which
# the shortfall reporting handles rather than hiding.
#: The K&K slice of the synthesised distractor pool (400 rows) answers with a role
#: sequence, not a number, so it carries ``solvable_roles`` rather than
#: ``solvable_numeric`` -- see ``distractor_synth.py``'s module docstring.  The row
#: total is unchanged (4,400 + 2,000); only the branch bookkeeping moves, and the
#: two K&K cells are therefore one cell of 2,400 here.
DEFAULT_QUOTA: dict[tuple[str, str], int] = {
    # solvable numeric + injected distractor -- 4,400
    (BRANCH_SOLVABLE_NUMERIC, schema.SOURCE_GSMIC): 2000,
    (BRANCH_SOLVABLE_NUMERIC, schema.SOURCE_SUM): 1000,
    (BRANCH_SOLVABLE_NUMERIC, schema.SOURCE_UMWP): 600,
    (BRANCH_SOLVABLE_NUMERIC, schema.SOURCE_MAIN): 400,
    # solvable role words -- K&K 2,000 + its 400 synthesised-distractor rows
    (BRANCH_SOLVABLE_ROLES, schema.SOURCE_KK): 2400,
    # solvable judgment, option block -- 900
    (BRANCH_SOLVABLE_JUDGE, schema.SOURCE_UMWP): 550,
    (BRANCH_SOLVABLE_JUDGE, schema.SOURCE_SUM): 350,
    # solvable judgment, no option block -- 700
    (BRANCH_SOLVABLE_JUDGE, schema.SOURCE_CREPE): 500,
    (BRANCH_SOLVABLE_JUDGE, schema.SOURCE_KUQ): 200,
    # unsolvable four-tier (diagnosis with options) -- 7,200
    (BRANCH_UNSOLVABLE_DIAG, schema.SOURCE_SUM): 5094,
    (BRANCH_UNSOLVABLE_DIAG, schema.SOURCE_UMWP): 1449,
    (BRANCH_UNSOLVABLE_DIAG, schema.SOURCE_FALSEQA): 657,
    # unsolvable three-tier (bare) -- 4,800
    (BRANCH_UNSOLVABLE_BARE, schema.SOURCE_SUM): 2000,
    (BRANCH_UNSOLVABLE_BARE, schema.SOURCE_TREECUT): 1084,
    (BRANCH_UNSOLVABLE_BARE, schema.SOURCE_UMWP): 840,
    (BRANCH_UNSOLVABLE_BARE, schema.SOURCE_CREPE): 400,
    (BRANCH_UNSOLVABLE_BARE, schema.SOURCE_MIP): 276,
    (BRANCH_UNSOLVABLE_BARE, schema.SOURCE_KUQ): 200,
}


def _largest_remainder(shares: dict, total: int) -> dict:
    """Apportion ``total`` proportionally to ``shares``, summing to exactly ``total``."""
    weight = sum(shares.values())
    if weight <= 0:
        raise ValueError("cannot apportion a zero-weight quota group")
    exact = {key: total * value / weight for key, value in shares.items()}
    floors = {key: int(value) for key, value in exact.items()}
    remainder = total - sum(floors.values())
    # Ties broken by the larger fractional part, then by key for determinism.
    order = sorted(exact, key=lambda key: (-(exact[key] - floors[key]), key))
    for key in order[:remainder]:
        floors[key] += 1
    return floors


def scaled_quota(total: int, unsolvable_ratio: float, four_tier_ratio: float) -> dict[tuple[str, str], int]:
    """Rescale :data:`DEFAULT_QUOTA` to the requested totals.

    Groups are apportioned independently -- solvable / unsolvable, then
    four-tier / three-tier inside the unsolvable side -- and each group is made to
    sum exactly, so the three knobs are honoured to the row instead of drifting by
    rounding.
    """
    if not 0.0 < unsolvable_ratio < 1.0:
        raise ValueError(f"--halluc_unsolvable_ratio must be in (0, 1), got {unsolvable_ratio}")
    if not 0.0 < four_tier_ratio < 1.0:
        raise ValueError(f"--halluc_four_tier_ratio must be in (0, 1), got {four_tier_ratio}")

    n_unsolvable = round(total * unsolvable_ratio)
    groups: list[tuple[list[tuple[str, str]], int]] = []
    groups.append(([c for c in DEFAULT_QUOTA if c[0] in SOLVABLE_BRANCHES], total - n_unsolvable))
    groups.append(([c for c in DEFAULT_QUOTA if c[0] == BRANCH_UNSOLVABLE_DIAG], round(n_unsolvable * four_tier_ratio)))
    groups.append(
        ([c for c in DEFAULT_QUOTA if c[0] == BRANCH_UNSOLVABLE_BARE], n_unsolvable - round(n_unsolvable * four_tier_ratio))
    )

    quota: dict[tuple[str, str], int] = {}
    for cells, group_total in groups:
        quota.update(_largest_remainder({c: DEFAULT_QUOTA[c] for c in cells}, group_total))
    return quota


def index_pool(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """Bucket hallucination rows by ``(branch, data_source)``.

    Also rejects rows whose branch and ``ground_truth.solvable`` disagree, because
    a row in the wrong bucket would be counted against the wrong quota *and*
    scored as a different contract branch than the one it was written for.
    """
    pool: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        info = row.get("extra_info", {})
        branch, data_source = info.get("branch", ""), row.get("data_source", "")
        try:
            payload = json.loads(row.get("reward_model", {}).get("ground_truth", "{}"))
        except ValueError:
            payload = {}
        solvable = payload.get("solvable")
        if branch in SOLVABLE_BRANCHES and solvable is not True:
            raise ValueError(f"{info.get('task_id')}: branch {branch!r} but solvable={solvable!r}")
        if branch in UNSOLVABLE_BRANCHES and solvable is not False:
            raise ValueError(f"{info.get('task_id')}: branch {branch!r} but solvable={solvable!r}")
        pool[(branch, data_source)].append(row)
    return pool


def carve_val(
    pool: dict[tuple[str, str], list[dict]],
    quota: dict[tuple[str, str], int],
    n_val: int,
    rng: random.Random,
) -> tuple[list[dict], dict[tuple[str, str], list[dict]]]:
    """Take the val rows out of the pools *before* the train quota is filled.

    Design doc section 7.1 (and ``mix.py``): val is carved first so a val row can
    never also be sampled into train, and so ``--halluc_total`` means the train
    quota to the row.  The carve is apportioned over the quota cells, which keeps
    val's branch mix proportional to train's -- a val split that is all four-tier
    would not exercise the other contract branches.
    """
    if n_val <= 0:
        return [], {cell: list(rows) for cell, rows in pool.items()}

    weights = {cell: quota.get(cell, 0) for cell in pool}
    per_cell = _largest_remainder(weights, min(n_val, sum(len(v) for v in pool.values())))

    val_rows: list[dict] = []
    reduced: dict[tuple[str, str], list[dict]] = {}
    for cell, rows in pool.items():
        take = min(per_cell.get(cell, 0), len(rows))
        if take:
            chosen = rng.sample(rows, take)
            chosen_ids = {id(r) for r in chosen}
            val_rows.extend(chosen)
            reduced[cell] = [r for r in rows if id(r) not in chosen_ids]
        else:
            reduced[cell] = list(rows)
    return val_rows, reduced


def fill_cells(
    pool: dict[tuple[str, str], list[dict]],
    quota: dict[tuple[str, str], int],
    rng: random.Random,
    spill_across_tiers: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Take each cell's quota, redistributing shortfalls within the same group.

    Returns ``(rows, report)`` where ``report`` is one entry per cell with its
    target, the amount filled from its own pool, and how much it received (or
    gave up) in redistribution.

    Redistribution boundaries are the *knob* boundaries, not just the
    solvable/unsolvable one: by default a four-tier deficit stays inside the
    four-tier group (and three-tier inside three-tier, solvable inside solvable),
    so ``--halluc_four_tier_ratio`` still describes the artifact.  Letting a
    four-tier shortfall spill into three-tier cells would silently move that ratio
    -- with UMWP's four-tier pool measured at ~670 against a 1,449 quota, the real
    build would drift from 60/40 to 55/45 without anything in the log saying so.
    ``spill_across_tiers`` opts in to the total-preserving behaviour instead; the
    achieved ratios are recorded either way.
    """
    report: list[dict] = []
    taken: dict[tuple[str, str], list[dict]] = {}
    deficits: dict[str, int] = defaultdict(int)
    for cell, target in sorted(quota.items()):
        available = pool.get(cell, [])
        take = min(target, len(available))
        chosen = rng.sample(available, take) if take < len(available) else list(available)
        taken[cell] = chosen
        deficit = target - take
        if deficit:
            deficits[_group_of(cell[0])] += deficit
        report.append({"cell": list(cell), "target": target, "filled": take, "shortfall": deficit, "redistributed": 0})

    if spill_across_tiers:
        # Any group's deficit may be covered by any cell, so a shortfall never
        # shrinks the total.  Every group is seeded with the total, including the
        # groups that had no deficit of their own -- otherwise they would never be
        # offered as a spill target, which is the whole point of the flag.
        total_deficit = sum(deficits.values())
        deficits = defaultdict(int, dict.fromkeys(("solvable", "four_tier", "three_tier"), total_deficit))

    for entry in report:
        group = _group_of(entry["cell"][0])
        if deficits[group] <= 0:
            continue
        cell = tuple(entry["cell"])
        already = {id(r) for r in taken[cell]}
        spare = len(pool.get(cell, [])) - len(already)
        if spare <= 0:
            continue
        extra = min(spare, deficits[group])
        remaining = [r for r in pool.get(cell, []) if id(r) not in already]
        taken[cell].extend(rng.sample(remaining, extra))
        entry["redistributed"] = extra
        deficits[group] -= extra

    rows: list[dict] = []
    for cell in taken:
        rows.extend(taken[cell])
    return rows, report


def _group_of(branch: str) -> str:
    """The apportionment group a branch belongs to -- also a redistribution boundary.

    Three groups, not two: the four-tier/three-tier split is its own knob, so it
    is its own budget.
    """
    if branch in SOLVABLE_BRANCHES:
        return "solvable"
    return "four_tier" if branch == BRANCH_UNSOLVABLE_DIAG else "three_tier"


def check_template_balance(rows: list[dict]) -> dict:
    """Assert each template carries both classes; return the measured table.

    This is design doc section 5.2's hard constraint.  A template that only ever
    appears on one side of the solvable/unsolvable boundary is a prompt-level label
    leak, and the fix belongs in the adapters (route some rows of the other side
    through it), not in a tolerance here.
    """
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        template = row["extra_info"].get("template", "?")
        solvable = json.loads(row["reward_model"]["ground_truth"]).get("solvable")
        counts[template]["solvable" if solvable else "unsolvable"] += 1

    unbalanced = [t for t, c in counts.items() if not (c["solvable"] and c["unsolvable"])]
    if unbalanced:
        detail = {t: dict(counts[t]) for t in unbalanced}
        raise ValueError(
            f"template(s) {unbalanced} appear on only one side of the solvable/unsolvable "
            f"boundary: {detail}. Design doc section 5.2 requires every template to contain "
            f"both, or the prompt itself leaks the label."
        )
    return {t: dict(c) for t, c in sorted(counts.items())}


def dedup_against(rows: list[dict], reference_texts: set[str]) -> tuple[list[dict], int]:
    """Drop rows whose question text exactly matches a stage-1 pool text.

    Design doc section 7.2.  MiP is built from GSM8K/SVAMP/MATH, which the stage-1
    mix also draws on, so an exact-text pass is the cheap floor; the MinHash pass
    in ``dedup.py`` covers near-duplicates and is run separately by the caller.
    """
    kept, dropped = [], 0
    for row in rows:
        text = row["prompt"][0]["content"]
        key = " ".join(text.split()).casefold()
        if key in reference_texts:
            dropped += 1
            continue
        kept.append(row)
    return kept, dropped


def _question_key(row: dict) -> str:
    return " ".join(row["prompt"][0]["content"].split()).casefold()


def build_parser() -> argparse.ArgumentParser:
    """The CLI.  Split out from :func:`main` so tests can read the defaults."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--halluc_dir",
        default=DEFAULT_ADAPTER_DIR,
        help="adapter output directory -- the one every adapter's DEFAULT_OUT points into",
    )
    parser.add_argument("--stage1_path", default=None, help="stage-1 train.parquet / val.parquet / directory")
    parser.add_argument("--output_dir", default="~/data/reasoning_rl/final_halluc")
    parser.add_argument("--halluc_total", type=int, default=20000)
    parser.add_argument("--halluc_unsolvable_ratio", type=float, default=0.6)
    parser.add_argument("--halluc_four_tier_ratio", type=float, default=0.6)
    parser.add_argument("--old_domain_ratio", type=float, default=0.7, help="stage-1 share of the stage-2 mix")
    parser.add_argument(
        "--spill_across_tiers",
        action="store_true",
        help="Let a four-tier shortfall be covered by three-tier cells. Preserves the total but "
        "moves the four-tier/three-tier ratio away from --halluc_four_tier_ratio.",
    )
    parser.add_argument("--val_size", type=int, default=256, help="hallucination-domain val rows (0 disables)")
    parser.add_argument("--no-dedup", action="store_true", help="skip the section 7.2 exact-text dedup")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    rng = random.Random(args.seed)
    halluc_dir = os.path.expanduser(args.halluc_dir)
    output_dir = os.path.expanduser(args.output_dir)

    files = resolve_parquet_files(halluc_dir)
    halluc_rows = read_parquet_rows(halluc_dir)
    if not halluc_rows:
        raise ValueError(f"no hallucination rows found under {halluc_dir}")
    print(f"[mix_halluc] {len(halluc_rows)} rows from {len(files)} adapter parquet(s)")

    quota = scaled_quota(args.halluc_total, args.halluc_unsolvable_ratio, args.halluc_four_tier_ratio)
    pool = index_pool(halluc_rows)
    missing = [cell for cell in quota if cell not in pool]
    if missing:
        print(f"[mix_halluc] {len(missing)} quota cell(s) have no adapter output: {missing}")

    # Carve val out of the pools first, then fill the train quota (mix.py's order).
    val_rows, pool = carve_val(pool, quota, args.val_size, rng)
    rows, cell_report = fill_cells(pool, quota, rng, spill_across_tiers=args.spill_across_tiers)
    filled_total = len(rows)
    if filled_total < args.halluc_total:
        print(
            f"[mix_halluc] WARNING: hallucination domain filled {filled_total} of "
            f"{args.halluc_total} rows; the pools are short. See mix_stats.json -> cells."
        )

    # --- optional stage-1 mixing (section 7.1) ---------------------------------
    old_rows: list[dict] = []
    if args.stage1_path:
        stage1_path = os.path.expanduser(args.stage1_path)
        if not (os.path.exists(stage1_path) or os.path.isdir(stage1_path)):
            raise FileNotFoundError(f"--stage1_path does not exist: {stage1_path}")
        stage1_rows = read_parquet_rows(stage1_path)
        n_old = round(filled_total * args.old_domain_ratio / (1.0 - args.old_domain_ratio))
        if n_old > len(stage1_rows):
            print(f"[mix_halluc] WARNING: stage-1 pool {len(stage1_rows)} < {n_old} requested; taking all")
            n_old = len(stage1_rows)
        old_rows = rng.sample(stage1_rows, n_old)
        print(f"[mix_halluc] stage-1 rows: {len(old_rows)} (old_domain_ratio={args.old_domain_ratio})")
    else:
        print("[mix_halluc] no --stage1_path: emitting the hallucination domain only")

    if not args.no_dedup:
        reference = {_question_key(r) for r in old_rows} if old_rows else set()
        rows, dropped_old = dedup_against(rows, reference)
        # Also drop exact duplicates inside the hallucination domain itself: the
        # same source question can reach two branches (e.g. SUM answerable used for
        # both the synthesised and the judgment slice).
        seen: set[str] = set()
        unique: list[dict] = []
        for row in rows:
            key = _question_key(row)
            if key in seen:
                continue
            seen.add(key)
            unique.append(row)
        dropped_self = len(rows) - len(unique)
        rows = unique
        print(f"[mix_halluc] dedup: -{dropped_old} overlapping stage-1, -{dropped_self} internal")
    else:
        dropped_old = dropped_self = 0

    all_train = rows + old_rows
    rng.shuffle(all_train)
    rng.shuffle(val_rows)
    for i, row in enumerate(all_train):
        row["extra_info"]["index"] = i
        row["extra_info"].setdefault("pass_rate", -1.0)
    for i, row in enumerate(val_rows):
        row["extra_info"]["index"] = i
        row["extra_info"]["split"] = "val"
        row["extra_info"].setdefault("pass_rate", -1.0)

    templates = check_template_balance(rows)
    n_diag = sum(1 for r in rows if r["extra_info"]["branch"] == BRANCH_UNSOLVABLE_DIAG)
    n_bare = sum(1 for r in rows if r["extra_info"]["branch"] == BRANCH_UNSOLVABLE_BARE)
    n_solvable = sum(1 for r in rows if r["extra_info"]["branch"] in SOLVABLE_BRANCHES)
    achieved_ratios = {
        "solvable_share": n_solvable / len(rows) if rows else 0.0,
        "four_tier_share_of_unsolvable": n_diag / (n_diag + n_bare) if (n_diag + n_bare) else 0.0,
    }
    schema.normalise_extra_info(all_train + val_rows)

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.parquet")
    write_rows_parquet(all_train, train_path)
    val_path = None
    if val_rows:
        val_path = os.path.join(output_dir, "val.parquet")
        write_rows_parquet(val_rows, val_path)

    stats = {
        "config": {
            "halluc_total": args.halluc_total,
            "halluc_unsolvable_ratio": args.halluc_unsolvable_ratio,
            "halluc_four_tier_ratio": args.halluc_four_tier_ratio,
            "old_domain_ratio": args.old_domain_ratio,
            "val_size": args.val_size,
            "seed": args.seed,
        },
        "total_train": len(all_train),
        "total_val": len(val_rows),
        "halluc_rows_in_train": len(rows),
        "halluc_target_vs_filled": {"target": args.halluc_total, "filled": len(rows)},
        "requested_ratios": {
            "unsolvable_share": args.halluc_unsolvable_ratio,
            "four_tier_share_of_unsolvable": args.halluc_four_tier_ratio,
        },
        "achieved_ratios": achieved_ratios,
        "dropped_overlapping_stage1": dropped_old,
        "dropped_duplicate_questions": dropped_self,
        "cells": cell_report,
        "train_by_ability": dict(Counter(r["ability"] for r in all_train)),
        "train_by_source": dict(Counter(r["extra_info"].get("source", "") for r in all_train)),
        "halluc_by_branch": dict(Counter(r["extra_info"]["branch"] for r in rows)),
        "halluc_by_template": dict(Counter(r["extra_info"]["template"] for r in rows)),
        "halluc_by_data_source": dict(Counter(r["data_source"] for r in rows)),
        "template_balance": templates,
    }
    with open(os.path.join(output_dir, "mix_stats.json"), "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"[done] train={len(all_train)} -> {train_path}")
    if val_path:
        print(f"[done] val={len(val_rows)} -> {val_path}")
    print(f"[done] hallucination by branch: {stats['halluc_by_branch']}")
    print(f"[done] template balance: {templates}")
    short = [c for c in cell_report if c["shortfall"] - c["redistributed"] > 0]
    if short:
        print(f"[done] {len(short)} cell(s) under-filled after redistribution:")
        for cell in short:
            print(f"    {cell['cell']}: {cell['filled'] + cell['redistributed']}/{cell['target']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
