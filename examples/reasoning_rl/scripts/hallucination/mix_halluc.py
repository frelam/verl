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
section 4.8 table B fixes a per-cell quota over ``(branch, data_source)``, and
two knobs rescale those cells:

* ``--halluc_total`` (20,000) -- the size of the whole hallucination domain;
* ``--halluc_unsolvable_ratio`` (0.6) -- its unsolvable share.

``(branch, data_source)`` is a complete key for every cell: the synthesised
distractor rows of table B row 1 keep their base pool's data_source while the
pure rows of the same pool use a different branch (``solvable_two_layer`` for
UMWP-answerable, ``solvable_roles`` for K&K), and TreeCut's positives and
negatives differ by branch.  So the quota table needs no extra ``error_type``
coupling to the adapters.

Side weights: SUM's pair rows (table B row 6) carry **both** sides in one row, so
the 60/40 accounting of section 4.8 counts them 0.5/0.5.  The quota therefore has
three kinds of cell -- solvable (weight 1 to the solvable side), unsolvable
(weight 1 to the unsolvable side) and pair (0.5 to each) -- and
:func:`scaled_quota` apportions **side weight**, not row count.  That is what
makes the default knobs reproduce table B exactly: 5,000 solvable rows + 6,000
pair rows = 8,000 solvable-equivalent (40%) and 9,000 unsolvable rows + 6,000 pair
rows = 12,000 unsolvable-equivalent (60%), 20,000 rows in total.

One cell deviates from the table's *labels*, not its arithmetic.  Table B row 1's
400 synthesised distractor rows are built on UMWP-answerable 200 / K&K 100 /
stage-1 main pool 100 (section 4.7).  A K&K question answers with a D19
``name: role`` mapping, so its 100 rows cannot carry the numeric branch: they
join the K&K cell (1,600 + 100 = 1,700) and the numeric synth cells hold the
other 300.  Every side weight is unchanged, so the 40/60 split and the 20,000
total are exactly the table's.

Two invariants are hard, not advisory, and the mixer refuses to write an artifact
that violates them:

1. every template must carry **both** solvable and unsolvable rows.  If an option
   block or a verdict prompt correlated with solvability the policy could read the
   label off the prompt -- the design doc calls this out as a 100% shortcut.
   Template C is exempt by construction (every SUM pair row is ``solvable=true``
   yet contains an unanswerable question too, section 4.8), and ``B_judge`` is
   counted inside the B family because it is B's verdict-only variant.
2. a row's branch must agree with its own ``ground_truth.solvable``.

A cell whose pool is smaller than its quota is **filled short and reported**, with
the leftover budget redistributed to cells in the same group that have surplus.
Redistribution stays inside a group (solvable / four-tier / three-tier; the pair
cell is its own group and never absorbs a deficit), so the 60/40 unsolvable split
and the four-tier/three-tier composition survive a shortfall; the total is only
allowed to come out below ``--halluc_total`` when the pools genuinely cannot fill
it.

Stage 3: ``--stage3_mix`` (D30)
-------------------------------

With ``--stage3_mix`` the mixer builds the stage-3 domain-balanced mix instead
of the stage-2 table: the base table is :data:`STAGE3_QUOTA`, which holds every
stage-2 source rebalanced so that the hallucination domain is **25% solvable /
75% unsolvable** by side weight and its domain weights are logic 20% /
commonsense 20% / math 60% (logic 2,800 = 700/2,100; commonsense 2,800 =
700/2,100; math 8,400 = 2,100/6,300).  With ``--old_domain_ratio 0.6`` the whole
artifact is then 70/30 solvable/unsolvable (0.6 * 1.0 + 0.4 * 0.25 = 0.70).  The
new logic unsolvable side is the K&K ``kkcut:`` synthesis
(``kk_adapter.py --emit-unsolvable``, D30) -- the repair-layer contract of D29
is retired from scoring, so stage 3 no longer reads ``--repair-contract``
output and the old ``--repair_mode`` is gone.  Commonsense grows through
CREPE's ``--quota-overrides`` (400 judgement + 1,280 bare).

Three ordering rules differ from stage 2 under this flag, all to keep the
artifact on the quota table rather than silently drifting off it:

* the exact-text and MinHash passes run **before** the fill and anchor on the
  whole stage-1 train pool, so the quota cells are sized at the post-dedup
  supply and a hallucination row cannot duplicate an unsampled stage-1 row;
* val is carved **out of the filled rows** (proportionally per cell) instead of
  out of the pools, so ``--halluc_total`` counts train + val and the train cells
  keep their requested composition; ``--strict_cells`` makes a short cell a hard
  error instead of a cross-cell backfill.

The stage-1 replay is stratified by ability under this flag (``--code_share``,
default 0.24): the hallucination domain has no code source, so a uniform replay
would dilute code from ~25% of the old domain to ~15% of the whole train set.
Code is drawn to the requested share of the *whole* train set and the remaining
abilities keep the old domain's proportions, which leaves math/code/logic close
to stage 1's own mix.
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
from schema import BRANCH_SOLVABLE_JUDGE, BRANCH_SOLVABLE_NUMERIC, BRANCH_SOLVABLE_PAIR  # noqa: E402
from schema import BRANCH_SOLVABLE_ROLES, BRANCH_SOLVABLE_TWO_LAYER  # noqa: E402
from schema import BRANCH_UNSOLVABLE_BARE, BRANCH_UNSOLVABLE_DIAG  # noqa: E402

#: Branches whose rows are solvable end to end (weight 1 to the solvable side).
SOLVABLE_BRANCHES = (
    BRANCH_SOLVABLE_NUMERIC,
    BRANCH_SOLVABLE_ROLES,
    BRANCH_SOLVABLE_TWO_LAYER,
    BRANCH_SOLVABLE_JUDGE,
)
#: SUM's pair rows: one row carries a solvable and an unanswerable question, so it
#: counts 0.5 to each side (design doc section 4.8's 60/40 accounting).
PAIR_BRANCHES = (BRANCH_SOLVABLE_PAIR,)
UNSOLVABLE_BRANCHES = (BRANCH_UNSOLVABLE_DIAG, BRANCH_UNSOLVABLE_BARE)

#: Side weight of each branch kind (a pair row carries half of each side).
BRANCH_SIDE_WEIGHT = {branch: 1.0 for branch in SOLVABLE_BRANCHES}
BRANCH_SIDE_WEIGHT.update({branch: 1.0 for branch in UNSOLVABLE_BRANCHES})
BRANCH_SIDE_WEIGHT.update({branch: 0.5 for branch in PAIR_BRANCHES})

#: The directory every adapter writes its ``DEFAULT_OUT`` into, and therefore the
#: default ``--halluc_dir``.  ``test_mix_halluc.py`` asserts the two stay equal for
#: every ``*_adapter.py`` in this directory: a mismatch here is silent -- the mixer
#: finds no rows and dies with "no hallucination rows found", which reads like a
#: missing build rather than a typo.
DEFAULT_ADAPTER_DIR = "~/data/reasoning_rl/halluc/built"

# Design doc section 4.8 table B, at the default knobs (20,000 / 0.6).  Keyed by
# (branch, data_source) -- see the module docstring for why that key is complete.
# The rows sum to 20,000: 5,000 pure-solvable + 6,000 pair + 9,000 unsolvable, and
# the side weights (pair = 0.5 each) give 8,000 / 12,000, i.e. the documented
# 40% / 60%.  ``SOURCE_MAIN`` is the stage-1 math pool used as a base for the
# synthesised distractor slice; it is absent on a box without stage-1 data, which
# the shortfall reporting handles rather than hiding.
#: The K&K cell holds 1,600 pure K&K rows (D20/thresholds of section 4.1) plus the
#: 100 synthesised-distractor rows whose base question is a K&K puzzle.  Those 100
#: answer with a D19 name->role mapping, so they cannot carry the numeric branch;
#: every side weight is unchanged, so table B's 40/60 arithmetic still holds.
DEFAULT_QUOTA: dict[tuple[str, str], int] = {
    # --- table B row 1: solvable numeric, optional injected distractor -- 1,072
    (BRANCH_SOLVABLE_NUMERIC, schema.SOURCE_GSMIC): 772,
    (BRANCH_SOLVABLE_NUMERIC, schema.SOURCE_UMWP): 200,
    (BRANCH_SOLVABLE_NUMERIC, schema.SOURCE_MAIN): 100,
    # --- table B row 2: solvable name -> role pairs -- 1,700 (1,600 + 100 synth)
    (BRANCH_SOLVABLE_ROLES, schema.SOURCE_KK): 1700,
    # --- table B row 3: solvable judgement + answer, placeholder options -- 1,478
    (BRANCH_SOLVABLE_TWO_LAYER, schema.SOURCE_UMWP): 550,
    (BRANCH_SOLVABLE_TWO_LAYER, schema.SOURCE_FALSEQA): 928,
    # --- table B row 4: solvable judgement only -- 250
    (BRANCH_SOLVABLE_JUDGE, schema.SOURCE_CREPE): 250,
    # --- table B row 5: solvable numeric with a placeholder option block -- 500
    (BRANCH_SOLVABLE_NUMERIC, schema.SOURCE_TREECUT): 500,
    # --- table B row 6: SUM pair task (0.5 solvable / 0.5 unsolvable) -- 6,000
    (BRANCH_SOLVABLE_PAIR, schema.SOURCE_SUM): 6000,
    # --- table B row 7: unsolvable four-tier (diagnosis with options) -- 5,835
    (BRANCH_UNSOLVABLE_DIAG, schema.SOURCE_FALSEQA): 928,
    (BRANCH_UNSOLVABLE_DIAG, schema.SOURCE_TREECUT): 4907,
    # --- table B row 8: unsolvable three-tier (bare refusal) -- 3,165
    (BRANCH_UNSOLVABLE_BARE, schema.SOURCE_UMWP): 2489,
    (BRANCH_UNSOLVABLE_BARE, schema.SOURCE_CREPE): 400,
    (BRANCH_UNSOLVABLE_BARE, schema.SOURCE_MIP): 276,
}

#: Groups whose budgets are apportioned independently, i.e. the boundaries a
#: shortfall may not cross.  The pair cell is alone in its group, so it can never
#: absorb a solvable or unsolvable deficit (which would silently move the 40/60
#: accounting).
QUOTA_GROUPS = ("solvable", "four_tier", "three_tier", "pair")

#: Stage 3 (D30, ``--stage3_mix``): the domain-balanced table.  Same sources and
#: branches as stage 2, four moves:
#:
#: * **the whole mix is 25/75 solvable/unsolvable by side weight** -- the old
#:   domain is 100% solvable, so with old_domain_ratio 0.6 this is what makes the
#:   *artifact* 70/30 (0.6 old solvable + 0.1 halluc solvable = 0.7);
#: * **every hallucination domain splits 25/75** -- logic 700/2,100, commonsense
#:   700/2,100 (two_layer FalseQA 300 + judge CREPE 400 vs diag FalseQA 820 +
#:   bare CREPE 1,280), math 2,100/6,300 (the SUM pair's 2,200 rows count 1,100
#:   to each side);
#: * **domain weights** are logic 20% (2,800) / commonsense 20% (2,800) / math
#:   60% (8,400) of the 14,000, which keeps hallucination math:logic at 3:1 --
#:   the old domain's own 45:15 ratio -- so adding the hallucination domain does
#:   not tilt math vs logic;
#: * the logic unsolvable side is **synthesised**: K&K has no native unsolvable
#:   rows, so the 2,100 come from ``kk_adapter.py --emit-unsolvable``'s
#:   ``kkcut:`` cells (D30) -- delete a certified-necessary statement, the 2^N
#:   enumeration is the certificate, the contract is the three-tier bare refusal
#:   like MiP's.  One deletion per parent caps the yield at ~490, so the stage-3
#:   build passes ``--kkcut-per-group 0`` (every admissible deletion).
#:
#: Supply notes, all measured on the real stage-2 build: MiP is capped below its
#: strict-certification supply (the 2026-10-16 artifact filled 267 of 270 after
#: the near-duplicate pass, so 270 is unreachable and ``--strict_cells`` would
#: always fail); TreeCut keeps 3,030 of its ~3,230 post-dedup negatives and 320 of
#: its ~405 positives; UMWP ships ~2,600 unsolvable rows, so 1,925 is safe; CREPE
#: needs its adapter's ``--quota-overrides`` to ship past the stage-2 650
#: (400 + 1,280).  GSM-IC/Main are the tightest cells: the near-duplicate pass
#: against the *whole* stage-1 pool removed 405 of GSM-IC's 790 and 74 of Main's
#: 99, leaving ~385 / ~26 -- 340 / 15 keeps a margin.  At the default knobs
#: :func:`scaled_quota` reproduces this table to the row.
STAGE3_QUOTA: dict[tuple[str, str], int] = {
    # --- logic: 700 solvable + 2,100 unsolvable = 2,800 (20%) ---------------
    (BRANCH_SOLVABLE_ROLES, schema.SOURCE_KK): 700,
    (BRANCH_UNSOLVABLE_BARE, schema.SOURCE_KK): 2100,
    # --- math: 2,100 solvable + 6,300 unsolvable = 8,400 (60%) --------------
    (BRANCH_SOLVABLE_NUMERIC, schema.SOURCE_GSMIC): 340,
    (BRANCH_SOLVABLE_NUMERIC, schema.SOURCE_UMWP): 155,
    (BRANCH_SOLVABLE_NUMERIC, schema.SOURCE_MAIN): 15,
    (BRANCH_SOLVABLE_NUMERIC, schema.SOURCE_TREECUT): 320,
    (BRANCH_SOLVABLE_TWO_LAYER, schema.SOURCE_UMWP): 170,
    (BRANCH_SOLVABLE_PAIR, schema.SOURCE_SUM): 2200,
    (BRANCH_UNSOLVABLE_DIAG, schema.SOURCE_TREECUT): 3030,
    (BRANCH_UNSOLVABLE_BARE, schema.SOURCE_UMWP): 1925,
    (BRANCH_UNSOLVABLE_BARE, schema.SOURCE_MIP): 245,
    # --- commonsense: 700 solvable + 2,100 unsolvable = 2,800 (20%) ---------
    (BRANCH_SOLVABLE_TWO_LAYER, schema.SOURCE_FALSEQA): 300,
    (BRANCH_SOLVABLE_JUDGE, schema.SOURCE_CREPE): 400,
    (BRANCH_UNSOLVABLE_DIAG, schema.SOURCE_FALSEQA): 820,
    (BRANCH_UNSOLVABLE_BARE, schema.SOURCE_CREPE): 1280,
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


def side_weight(cell: tuple[str, str], rows: int) -> float:
    """Side weight of ``rows`` rows in ``cell`` (pairs count half to each side)."""
    return rows * BRANCH_SIDE_WEIGHT[cell[0]]


def scaled_quota(
    total: int, unsolvable_ratio: float, base: dict[tuple[str, str], int] | None = None
) -> dict[tuple[str, str], int]:
    """Rescale a quota table to the requested total and side ratio.

    ``base`` is :data:`DEFAULT_QUOTA` (stage 2) by default and
    :data:`STAGE3_QUOTA` for ``--stage3_mix``.  The apportionment is done in
    **side weight**, not row count (see the module docstring): SUM's pair rows
    contribute 0.5 to each side, so the requested ``unsolvable_ratio`` describes
    the artifact the way section 4.8 accounts for it rather than over-counting
    the pairs as solvable rows.  The pair budget keeps the base table's share of
    the total, then the two sides are apportioned over their own cells in
    proportion to the base and made to sum exactly, so the default knobs
    reproduce the base table to the row.
    """
    if base is None:
        base = DEFAULT_QUOTA
    if not 0.0 < unsolvable_ratio < 1.0:
        raise ValueError(f"--halluc_unsolvable_ratio must be in (0, 1), got {unsolvable_ratio}")

    default_total = sum(base.values())
    pair_cells = [cell for cell in base if cell[0] in PAIR_BRANCHES]
    pair_rows = round(sum(base[cell] for cell in pair_cells) * total / default_total)
    half_pair = pair_rows / 2.0
    solvable_weight = (1.0 - unsolvable_ratio) * total - half_pair
    unsolvable_weight = unsolvable_ratio * total - half_pair
    if solvable_weight < 0 or unsolvable_weight < 0:
        raise ValueError(
            f"halluc_total={total} with unsolvable_ratio={unsolvable_ratio} leaves "
            f"{solvable_weight:.0f}/{unsolvable_weight:.0f} side weight outside the pair rows; "
            f"the pair share of the pool is fixed at {pair_rows / total:.2f} of the total"
        )

    quota: dict[tuple[str, str], int] = {}
    for group, target in (
        ("solvable", round(solvable_weight)),
        ("four_tier", round(unsolvable_weight * _four_tier_share(base))),
        ("three_tier", round(unsolvable_weight * (1.0 - _four_tier_share(base)))),
    ):
        cells = [cell for cell in base if _group_of(cell[0]) == group]
        quota.update(_largest_remainder({cell: base[cell] for cell in cells}, target))
    quota.update(_largest_remainder({cell: base[cell] for cell in pair_cells}, pair_rows))
    return quota


def _four_tier_share(base: dict[tuple[str, str], int]) -> float:
    """The base table's four-tier / three-tier split of the unsolvable side."""
    four = sum(q for (branch, _), q in base.items() if branch == BRANCH_UNSOLVABLE_DIAG)
    bare = sum(q for (branch, _), q in base.items() if branch == BRANCH_UNSOLVABLE_BARE)
    return four / (four + bare)


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
        if branch in PAIR_BRANCHES and solvable is not True:
            # A SUM pair row is solvable=true at row level (the pair contains a
            # solvable question); the 0.5/0.5 split lives in the quota weights.
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


def _pair_id(row: dict) -> str:
    """The row's train/val atomicity key (empty when the row stands alone)."""
    return str(row.get("extra_info", {}).get("pair_id", "") or "")


def _split_group_key(row: dict) -> str:
    """The key a row must share its train/val side with (empty = stands alone).

    Two kinds of near-duplicate groups must not straddle the split:

    * D27's FalseQA twins, joined by ``pair_id``;
    * D30's kkcut siblings -- several rows that delete a different single
      sentence from the *same* parent puzzle, so they differ by one sentence and
      would otherwise leak across the boundary.  Their only shared identifier is
      the ``kkcut:{family}:{N}ppl:{index}`` prefix of the 5-segment task_id.
    """
    pair_id = _pair_id(row)
    if pair_id:
        return pair_id
    task_id = str(row.get("extra_info", {}).get("task_id", ""))
    if task_id.startswith("kkcut:"):
        return task_id.rsplit(":", 1)[0]
    return ""


def enforce_pair_atomicity(
    val_rows: list[dict],
    pool: dict[tuple[str, str], list[dict]],
) -> tuple[list[dict], list[dict]]:
    """Keep a paired row's twins on the same side of the train/val boundary.

    Design doc D27: FalseQA's answerable and unanswerable rows are index-aligned
    twins whose prompts differ by one fragment, so letting one land in val while
    the other trains is a near-duplicate leak.  ``carve_val`` samples per pool
    cell, and the twins live in different cells, so the check has to happen across
    cells after the carve: a row whose ``pair_id`` appears fewer than twice in val
    is returned to the train pool.  Rows without a ``pair_id`` are untouched.
    """
    counts = Counter(pid for pid in (_pair_id(r) for r in val_rows) if pid)
    kept: list[dict] = []
    returned: list[dict] = []
    for row in val_rows:
        pid = _pair_id(row)
        if pid and counts[pid] < 2:
            returned.append(row)
        else:
            kept.append(row)
    for row in returned:
        cell = (row["extra_info"]["branch"], row["data_source"])
        pool.setdefault(cell, []).append(row)
    return kept, returned


def carve_val_from_filled(
    rows: list[dict],
    quota: dict[tuple[str, str], int],
    n_val: int,
    rng: random.Random,
) -> tuple[list[dict], list[dict]]:
    """Carve the val split out of the *already filled* train rows (stage 3).

    ``carve_val`` takes val out of the pools before the train quota is filled,
    which is fine when every cell has spare supply.  Stage 3 sizes several cells
    at exactly their supply (MiP, GSM-IC, TreeCut), so carving first would starve
    those cells and the shortfall would be backfilled from another data source --
    the domain weights would drift without anything in the log saying so.

    Carving after the fill keeps the train cell counts equal to the quota table to
    the row; val is then taken proportionally out of each cell, so it keeps the
    train composition.  ``--halluc_total`` therefore counts train + val.

    A D27 pair or a D30 kkcut sibling group that was split by the carve is
    returned to train whole (see :func:`_split_group_key`), so val can come in
    slightly under ``n_val``.
    """
    if n_val <= 0:
        return [], list(rows)

    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(row["extra_info"]["branch"], row["data_source"])].append(row)

    weights = {cell: quota.get(cell, len(bucket)) for cell, bucket in buckets.items()}
    per_cell = _largest_remainder(weights, min(n_val, len(rows)))

    val_rows: list[dict] = []
    train_rows: list[dict] = []
    for cell, bucket in buckets.items():
        take = min(per_cell.get(cell, 0), len(bucket))
        chosen = rng.sample(bucket, take) if take else []
        chosen_ids = {id(r) for r in chosen}
        val_rows.extend(chosen)
        train_rows.extend([r for r in bucket if id(r) not in chosen_ids])

    # D27 pair twins, and D30's kkcut siblings (several cuts of one parent puzzle
    # differ by a single sentence, so they are near-duplicates of each other): a
    # group that did not land in val *whole* goes back to train rather than
    # leaking across the boundary.
    group_sizes = Counter(key for key in (_split_group_key(r) for r in rows) if key)
    val_counts = Counter(key for key in (_split_group_key(r) for r in val_rows) if key)
    kept: list[dict] = []
    for row in val_rows:
        key = _split_group_key(row)
        if key and val_counts[key] != group_sizes[key]:
            train_rows.append(row)
        else:
            kept.append(row)
    return kept, train_rows


def fill_cells(
    pool: dict[tuple[str, str], list[dict]],
    quota: dict[tuple[str, str], int],
    rng: random.Random,
    spill_across_tiers: bool = False,
    redistribute: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Take each cell's quota, redistributing shortfalls within the same group.

    Returns ``(rows, report)`` where ``report`` is one entry per cell with its
    target, the amount filled from its own pool, and how much it received (or
    gave up) in redistribution.

    Redistribution boundaries are the *group* boundaries, not just the
    solvable/unsolvable one: by default a four-tier deficit stays inside the
    four-tier group (and three-tier inside three-tier, solvable inside solvable,
    pair inside pair), so the requested 40/60 split still describes the artifact.
    Letting a four-tier shortfall spill into three-tier cells would silently move
    the four-tier/three-tier composition -- the real build would drift without
    anything in the log saying so.  ``spill_across_tiers`` opts in to the
    total-preserving behaviour instead; the achieved ratios are recorded either
    way.

    ``redistribute=False`` disables the backfill entirely (stage-3 strict mode):
    a cell that cannot meet its target keeps its shortfall, so the caller can
    fail on it instead of letting another data source silently absorb it and
    move the domain weights.
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
        if not redistribute or deficits[group] <= 0:
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

    Four groups, not two: the four-tier/three-tier split is fixed by table B, and
    the pair cell is a group of its own so a solvable or unsolvable shortfall can
    never be backfilled with pair rows (which would move the 40/60 accounting
    without anything in the log saying so).
    """
    if branch in PAIR_BRANCHES:
        return "pair"
    if branch in SOLVABLE_BRANCHES:
        return "solvable"
    return "four_tier" if branch == BRANCH_UNSOLVABLE_DIAG else "three_tier"


def check_template_balance(rows: list[dict]) -> dict:
    """Assert each template carries both classes; return the measured table.

    This is design doc sections 4.8/5.2's hard constraint.  A template that only
    ever appears on one side of the solvable/unsolvable boundary is a prompt-level
    label leak, and the fix belongs in the adapters (route some rows of the other
    side through it), not in a tolerance here.

    Three exemptions, all structural rather than tolerances:

    * template C is SUM's pair task: every row is ``solvable=true`` *and* carries
      an unanswerable question, so it covers both sides by construction
      (design doc section 4.8's hard-constraint check);
    * template ``B_judge`` is template B's verdict-only variant, used by the
      CREPE-normal judgement rows; it is counted inside the B family, which must
      still carry both classes (B has the numeric/K&K solvable rows and the
      three-tier unsolvable rows);
    * template ``A_repair`` was stage 3's repair contract (D29, retired from
      the stage-3 build by D30; the shape is kept so pre-D29 artifacts still
      validate and audit): ``validate_row`` ties it to ``repair_task`` 1:1, so
      it can only ever carry four-tier unsolvable rows -- there is no solvable
      side the template could leak, and the solvability judgement it skips is
      the skill stage 2 already owns.
    """
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        template = row["extra_info"].get("template", "?")
        family = schema.TEMPLATE_B if template == schema.TEMPLATE_B_JUDGE else template
        solvable = json.loads(row["reward_model"]["ground_truth"]).get("solvable")
        counts[family]["solvable" if solvable else "unsolvable"] += 1

    exempt = (schema.TEMPLATE_C, schema.TEMPLATE_A_REPAIR)
    unbalanced = [
        t for t, c in counts.items() if t not in exempt and not (c["solvable"] and c["unsolvable"])
    ]
    if unbalanced:
        detail = {t: dict(counts[t]) for t in unbalanced}
        raise ValueError(
            f"template(s) {unbalanced} appear on only one side of the solvable/unsolvable "
            f"boundary: {detail}. Design doc section 4.8 requires every non-pair template to "
            f"contain both, or the prompt itself leaks the label."
        )
    return {t: dict(c) for t, c in sorted(counts.items())}


def dedup_against(rows: list[dict], reference_texts: set[str]) -> tuple[list[dict], int]:
    """Drop rows whose question text exactly matches a stage-1 pool text.

    Design doc section 7.2.  MiP is built from GSM8K/SVAMP/MATH, which the stage-1
    mix also draws on, so an exact-text pass is the cheap floor; the MinHash pass
    below covers near-duplicates (reworded statements of the same problem).
    """
    kept, dropped = [], 0
    for row in rows:
        key = _question_key(row)
        if key in reference_texts:
            dropped += 1
            continue
        kept.append(row)
    return kept, dropped


def near_dedup_against_stage1(
    rows: list[dict], anchor_texts: list[str], *, threshold: float
) -> tuple[list[dict], int, int]:
    """Drop rows whose question is a MinHash near-duplicate of a stage-1 row.

    Design doc section 7.2 calls this a *hard prerequisite*: MiP (GSM8K/SVAMP/
    MATH), UMWP (GSM8K/SVAMP/MultiArith/ASDiv) and SUM (DeepScaleR) share their
    underlying problems with the Big-Math main pool, so an exact-text pass alone
    leaves reworded duplicates in the stage-2 mix and silently inflates that
    slice's weight.

    The stage-1 rows are the *anchors*: they go into the greedy pass first, so
    they are always kept and a hallucination row that near-duplicates one is the
    row that gets dropped.  Rows are only ever compared against the stage-1 pool,
    never against each other -- D27's FalseQA twins (label=0 / label=1 differ by
    one replaced fragment, Jaccard well above the threshold) must both survive.
    A dropped row that carries a ``pair_id`` therefore takes its twin with it.

    Similarity is measured on the *presented prompt* (template included).  The
    shared instruction block is identical for every row of a template, so it
    lowers the effective bar on the question text; measured on the real pools an
    unrelated pair still lands well under 0.6 while a reworded duplicate of the
    same problem lands well above it.

    Returns ``(kept_rows, dropped, dropped_pair_twins)``.
    """
    from dedup import minhash_near_dedup

    texts = [str(text) for text in anchor_texts] + [_question_key(row) for row in rows]
    kept_indices = set(minhash_near_dedup(texts, threshold=threshold))
    n_anchor = len(anchor_texts)
    dropped_ids = {row["extra_info"]["task_id"] for i, row in enumerate(rows) if n_anchor + i not in kept_indices}
    if not dropped_ids:
        return rows, 0, 0
    dropped_pairs = {
        row["extra_info"].get("pair_id")
        for row in rows
        if row["extra_info"]["task_id"] in dropped_ids and row["extra_info"].get("pair_id")
    }
    kept: list[dict] = []
    twins = 0
    for row in rows:
        info = row["extra_info"]
        if info["task_id"] in dropped_ids:
            continue
        if info.get("pair_id") and info["pair_id"] in dropped_pairs:
            # Its twin left the pool, so this row can no longer be judged against
            # a paired counterpart (section 4.3 / D27).
            twins += 1
            continue
        kept.append(row)
    dropped = len(rows) - len(kept)
    return kept, dropped, twins


def _question_key(row: dict) -> str:
    """Normalised problem text of a row, template instruction block removed."""
    return " ".join(schema.question_of(row).split()).casefold()


def read_stage1_pool(path: str) -> tuple[list[dict], list[dict]]:
    """Stage-1 ``(train_rows, val_rows)``, kept apart (design doc section 7.1).

    A *directory* is split on ``train.parquet`` / ``val.parquet`` instead of being
    read recursively: ``read_parquet_rows`` walks ``**/*.parquet``, so pointing
    ``--stage1_path`` at stage-1's output directory would fold stage-1's own val
    rows into the stage-2 **train** set -- exactly the leakage section 7.2 works
    to avoid.  Every other parquet in the directory is treated as train (a sharded
    dump), and a single file is train-only.
    """
    if not os.path.isdir(path):
        return read_parquet_rows(path), []
    val_path = os.path.join(path, "val.parquet")
    val_rows = read_parquet_rows(val_path) if os.path.exists(val_path) else []
    train_rows: list[dict] = []
    for file in resolve_parquet_files(path):
        if os.path.abspath(file) == os.path.abspath(val_path):
            continue
        train_rows.extend(read_parquet_rows(file))
    if not val_rows:
        print("[mix_halluc] stage-1 directory has no val.parquet; stage-2 val stays hallucination-only")
    return train_rows, val_rows


def sample_old_domain(
    rows: list[dict],
    n_old: int,
    total_train: int,
    code_share: float,
    rng: random.Random,
) -> tuple[list[dict], dict]:
    """Sample the stage-1 replay rows, optionally pinning the code share.

    A uniform stage-1 sample carries the pool's own ability mix (measured: math
    45.0%, code 24.9%, logic 15.2%, stem 14.8%).  The hallucination domain has no
    code source, so a uniform replay dilutes code from ~25% of the old domain to
    ~15% of the whole train set.  ``code_share`` pins ``ability == "code"`` at
    that share of the *whole* train set and splits the rest over the remaining
    abilities in proportion to their pool sizes -- the old domain's shape among
    the non-code abilities is preserved, so math/code/logic end up close to the
    stage-1 mix.

    The returned report records what was actually drawn (the code pool may be
    too small, in which case the missing rows go to the other abilities and the
    caller sees the shortfall).
    """
    if not code_share or code_share <= 0 or n_old <= 0:
        chosen = rng.sample(rows, n_old)
        return chosen, {
            "code_target": 0,
            "code_rows": sum(1 for row in chosen if row.get("ability") == "code"),
            "shortfall": 0,
            "by_ability": dict(Counter(str(row.get("ability", "")) for row in chosen)),
        }

    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("ability", ""))].append(row)
    code_pool = buckets.pop("code", [])
    code_target = round(code_share * total_train)
    n_code = min(code_target, len(code_pool))
    allocations = _largest_remainder(
        {ability: len(bucket) for ability, bucket in buckets.items()},
        max(n_old - n_code, 0),
    ) if buckets else {}
    allocations = {ability: min(take, len(buckets[ability])) for ability, take in allocations.items()}
    # A capped ability leaves budget behind; hand it to the others in proportion
    # to their remaining supply (never below n_old rows unless the pool is short).
    leftover = n_old - n_code - sum(allocations.values())
    while leftover > 0:
        spare = {a: len(b) - allocations[a] for a, b in buckets.items() if len(b) > allocations[a]}
        if not spare:
            break
        extra = _largest_remainder(spare, min(leftover, sum(spare.values())))
        for ability, take in extra.items():
            allocations[ability] += take
        leftover = n_old - n_code - sum(allocations.values())

    chosen = rng.sample(code_pool, n_code) if n_code else []
    for ability, take in allocations.items():
        if take:
            chosen.extend(rng.sample(buckets[ability], take))
    return chosen, {
        "code_target": code_target,
        "code_rows": n_code,
        "shortfall": code_target - n_code,
        "by_ability": dict(Counter(str(row.get("ability", "")) for row in chosen)),
    }


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
    parser.add_argument(
        "--halluc_unsolvable_ratio",
        type=float,
        default=None,
        help="unsolvable share of the hallucination domain by side weight "
        "(default: 0.6, or 0.75 with --stage3_mix)",
    )
    parser.add_argument(
        "--stage3_mix",
        action="store_true",
        help="Stage 3 (D30): build the domain-balanced mix from STAGE3_QUOTA instead "
        "of the stage-2 table B -- the hallucination domain is 25%% solvable / 75%% "
        "unsolvable by side weight, so with --old_domain_ratio 0.6 the whole "
        "artifact is 70/30; its domain weights are logic 20%% / commonsense 20%% / "
        "math 60%%.  --halluc_total counts train + val.  Needs the kkcut cells "
        "(kk_adapter --emit-unsolvable --kkcut-per-group 0) and CREPE's "
        "--quota-overrides in --halluc_dir.",
    )
    parser.add_argument(
        "--old_domain_ratio",
        type=float,
        default=None,
        help="stage-1 share of the mix (default: 0.7, or 0.6 with --stage3_mix)",
    )
    parser.add_argument(
        "--strict_cells",
        action="store_true",
        help="Stage 3 only: fail instead of redistributing when a quota cell is short, so a "
        "missing data source cannot silently move the domain weights.",
    )
    parser.add_argument(
        "--spill_across_tiers",
        action="store_true",
        help="Let a four-tier shortfall be covered by three-tier cells. Preserves the total but "
        "moves the table-B four-tier/three-tier composition.",
    )
    parser.add_argument("--val_size", type=int, default=256, help="hallucination-domain val rows (0 disables)")
    parser.add_argument("--no-dedup", action="store_true", help="skip the section 7.2 exact-text dedup")
    parser.add_argument(
        "--minhash_threshold",
        type=float,
        default=0.6,
        help="Section 7.2 near-duplicate Jaccard threshold against the stage-1 pool (0 disables). "
        "Only runs when --stage1_path is given: rows are compared with stage-1, never with each other, "
        "so D27's FalseQA twins both survive.",
    )
    parser.add_argument(
        "--code_share",
        type=float,
        default=None,
        help="Share of the *whole train set* that must be ability == 'code' in the stage-1 "
        "replay (default: 0.24 with --stage3_mix, 0 = uniform stage-1 sampling). The "
        "hallucination domain has no code source, so without this a uniform replay "
        "dilutes code to ~0.6 * its stage-1 share.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    rng = random.Random(args.seed)
    halluc_dir = os.path.expanduser(args.halluc_dir)
    output_dir = os.path.expanduser(args.output_dir)
    # The two ratio knobs default per mode: stage 2 keeps the stage-2 table's
    # 0.6 / 0.7, stage 3 defaults to the 70/30 artifact (0.75 inside the
    # hallucination domain, 0.6 old-domain share).
    unsolvable_ratio = args.halluc_unsolvable_ratio
    if unsolvable_ratio is None:
        unsolvable_ratio = 0.75 if args.stage3_mix else 0.6
    old_domain_ratio = args.old_domain_ratio
    if old_domain_ratio is None:
        old_domain_ratio = 0.6 if args.stage3_mix else 0.7
    code_share = args.code_share
    if code_share is None:
        code_share = 0.24 if args.stage3_mix else 0.0

    files = resolve_parquet_files(halluc_dir)
    halluc_rows = read_parquet_rows(halluc_dir)
    if not halluc_rows:
        raise ValueError(f"no hallucination rows found under {halluc_dir}")
    print(f"[mix_halluc] {len(halluc_rows)} rows from {len(files)} adapter parquet(s)")

    base = STAGE3_QUOTA if args.stage3_mix else DEFAULT_QUOTA
    quota = scaled_quota(args.halluc_total, unsolvable_ratio, base=base)

    # --- stage-1 pool (section 7.1), read up front ------------------------------
    # Stage 3 anchors its dedup on the *whole* stage-1 train pool rather than on
    # the sampled old rows: a hallucination row must not survive only because the
    # stage-1 row it duplicates happened not to be sampled this time.
    stage1_rows: list[dict] = []
    old_val_rows: list[dict] = []
    if args.stage1_path:
        stage1_path = os.path.expanduser(args.stage1_path)
        if not (os.path.exists(stage1_path) or os.path.isdir(stage1_path)):
            raise FileNotFoundError(f"--stage1_path does not exist: {stage1_path}")
        stage1_rows, old_val_rows = read_stage1_pool(stage1_path)

    dropped_old = dropped_self = 0
    minhash_dropped = minhash_twins = 0

    if args.stage3_mix:
        # Stage 3 dedups *before* the fill.  Its quota table is sized at the
        # post-dedup supply, so dropping rows afterwards (stage 2's order) is what
        # made the earlier artifact land short in exactly the cells the near-
        # duplicate pass hits hardest -- GSM-IC 800 -> 385, main 100 -> 26.
        if not args.no_dedup:
            reference = {_question_key(r) for r in stage1_rows}
            halluc_rows, dropped_old = dedup_against(halluc_rows, reference)
            # Also drop exact duplicates inside the hallucination domain itself:
            # the same source question can reach two branches (e.g. SUM answerable
            # used for both the synthesised and the judgment slice).
            seen: set[str] = set()
            unique: list[dict] = []
            for row in halluc_rows:
                key = _question_key(row)
                if key in seen:
                    continue
                seen.add(key)
                unique.append(row)
            dropped_self = len(halluc_rows) - len(unique)
            halluc_rows = unique
            print(
                f"[mix_halluc] dedup (pre-fill): -{dropped_old} overlapping stage-1, "
                f"-{dropped_self} internal"
            )
        if stage1_rows and args.minhash_threshold > 0:
            anchor_texts = [_question_key(r) for r in stage1_rows]
            halluc_rows, minhash_dropped, minhash_twins = near_dedup_against_stage1(
                halluc_rows, anchor_texts, threshold=args.minhash_threshold
            )
            print(
                f"[mix_halluc] section 7.2 MinHash (Jaccard >= {args.minhash_threshold}) against "
                f"{len(anchor_texts)} stage-1 row(s): -{minhash_dropped} near-duplicates"
                + (f" (incl. {minhash_twins} pair twin(s) kept atomic)" if minhash_twins else "")
            )

    pool = index_pool(halluc_rows)
    missing = [cell for cell in quota if cell not in pool]
    if missing:
        print(f"[mix_halluc] {len(missing)} quota cell(s) have no adapter output: {missing}")

    if args.stage3_mix:
        rows, cell_report = fill_cells(
            pool,
            quota,
            rng,
            spill_across_tiers=args.spill_across_tiers,
            redistribute=not args.strict_cells,
        )
        short = [entry for entry in cell_report if entry["shortfall"] - entry["redistributed"] > 0]
        if short:
            detail = "; ".join(
                f"{entry['cell']}: {entry['filled'] + entry['redistributed']}/{entry['target']}"
                for entry in short
            )
            if args.strict_cells:
                raise ValueError(
                    f"--strict_cells: {len(short)} stage-3 cell(s) cannot meet their quota ({detail}). "
                    "Rebuild the adapter output or lower --halluc_total; the mix will not silently "
                    "shift a deficit onto another data source."
                )
            print(f"[mix_halluc] WARNING: {len(short)} stage-3 cell(s) under-filled: {detail}")
        # Val is carved out of the filled rows so the train cells keep the quota
        # table to the row; --halluc_total therefore counts train + val.
        val_rows, rows = carve_val_from_filled(rows, quota, args.val_size, rng)
    else:
        # Carve val out of the pools first, then fill the train quota (mix.py's order).
        val_rows, pool = carve_val(pool, quota, args.val_size, rng)
        val_rows, returned = enforce_pair_atomicity(val_rows, pool)
        if returned:
            print(f"[mix_halluc] val: returned {len(returned)} row(s) whose pair twin was not in val")
        rows, cell_report = fill_cells(pool, quota, rng, spill_across_tiers=args.spill_across_tiers)

    filled_total = len(rows)
    filled_all = filled_total + len(val_rows) if args.stage3_mix else filled_total
    if filled_all < args.halluc_total:
        print(
            f"[mix_halluc] WARNING: hallucination domain filled {filled_all} of "
            f"{args.halluc_total} rows; the pools are short. See mix_stats.json -> cells."
        )

    # --- optional stage-1 mixing (section 7.1) ---------------------------------
    old_rows: list[dict] = []
    old_report: dict = {"code_target": 0, "code_rows": 0, "shortfall": 0, "by_ability": {}}
    if args.stage1_path:
        n_old = round(filled_total * old_domain_ratio / (1.0 - old_domain_ratio))
        if n_old > len(stage1_rows):
            print(f"[mix_halluc] WARNING: stage-1 pool {len(stage1_rows)} < {n_old} requested; taking all")
            n_old = len(stage1_rows)
        old_rows, old_report = sample_old_domain(
            stage1_rows, n_old, filled_total + n_old, code_share, rng
        )
        print(
            f"[mix_halluc] stage-1 rows: {len(old_rows)} (old_domain_ratio={old_domain_ratio}, "
            f"code {old_report['code_rows']}/{old_report['code_target']})"
        )
        if old_report["shortfall"] > 0:
            print(
                f"[mix_halluc] WARNING: code pool short by {old_report['shortfall']} row(s); "
                f"the whole train set will miss the {code_share:.0%} code target"
            )
        if len(old_rows) < n_old:
            print(
                f"[mix_halluc] WARNING: stage-1 pool could only supply {len(old_rows)} of "
                f"{n_old} replay rows; the old-domain share will come out below "
                f"{old_domain_ratio:.2f}"
            )
        if old_val_rows:
            # Section 7.1 keeps the stage-1 rows verbatim; keeping its val rows in
            # the stage-2 val is what makes an old-domain regression visible,
            # because the carved rows are hallucination-only (section 12 Q2).
            val_rows = list(val_rows) + old_val_rows
            print(f"[mix_halluc] stage-1 val rows preserved: {len(old_val_rows)}")
    else:
        print("[mix_halluc] no --stage1_path: emitting the hallucination domain only")

    if not args.stage3_mix and not args.no_dedup:
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

    # Section 7.2's near-duplicate pass.  It needs the stage-1 pool as the anchor
    # set, so it is a no-op without --stage1_path (the hallucination domain is
    # never near-deduped against itself: D27's twins and UMWP's paired rows are
    # *supposed* to be near-identical).  Stage 3 already ran it pre-fill above.
    if not args.stage3_mix and old_rows and args.minhash_threshold > 0:
        anchor_texts = [_question_key(r) for r in old_rows]
        rows, minhash_dropped, minhash_twins = near_dedup_against_stage1(
            rows, anchor_texts, threshold=args.minhash_threshold
        )
        print(
            f"[mix_halluc] section 7.2 MinHash (Jaccard >= {args.minhash_threshold}) against "
            f"{len(anchor_texts)} stage-1 row(s): -{minhash_dropped} near-duplicates"
            + (f" (incl. {minhash_twins} pair twin(s) kept atomic)" if minhash_twins else "")
        )

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
    # Side weights, not row counts: a SUM pair row carries one solvable and one
    # unanswerable question, so it counts 0.5 to each side (section 4.8).
    n_solvable_rows = sum(1 for r in rows if r["extra_info"]["branch"] in SOLVABLE_BRANCHES)
    n_pair = sum(1 for r in rows if r["extra_info"]["branch"] in PAIR_BRANCHES)
    half_pairs = n_pair * BRANCH_SIDE_WEIGHT[BRANCH_SOLVABLE_PAIR]
    solvable_weight = n_solvable_rows + half_pairs
    unsolvable_weight = (len(rows) - n_solvable_rows - n_pair) + half_pairs
    total_weight = solvable_weight + unsolvable_weight
    achieved_ratios = {
        "solvable_share": solvable_weight / total_weight if total_weight else 0.0,
        "unsolvable_share": unsolvable_weight / total_weight if total_weight else 0.0,
        "four_tier_share_of_unsolvable": n_diag / (n_diag + n_bare) if (n_diag + n_bare) else 0.0,
        # Whole artifact (stage-1 + hallucination).  The stage-1 domain is
        # solvable-only by construction, so ``overall_solvable_share`` is the
        # number the stage-3 70/30 target is read from: 0.6 old + 0.4 * 0.25.
        "overall_solvable_share": (len(old_rows) + solvable_weight) / (len(old_rows) + total_weight)
        if (len(old_rows) + total_weight)
        else 0.0,
        # Code only ever comes from the stage-1 replay, so this is the number the
        # --code_share target is read from.
        "overall_code_share": sum(1 for r in all_train if r.get("ability") == "code") / len(all_train)
        if all_train
        else 0.0,
        "old_domain_share_of_train": len(old_rows) / len(all_train) if all_train else 0.0,
    }
    halluc_rows_in_val = len(val_rows) - len(old_val_rows)
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
            "halluc_unsolvable_ratio": unsolvable_ratio,
            "stage3_mix": args.stage3_mix,
            "old_domain_ratio": old_domain_ratio,
            "val_size": args.val_size,
            "strict_cells": args.strict_cells,
            "code_share": code_share,
            "seed": args.seed,
        },
        "total_train": len(all_train),
        "total_val": len(val_rows),
        "stage1_val_rows_kept_in_val": len(old_val_rows),
        "halluc_rows_in_train": len(rows),
        "halluc_rows_in_val": halluc_rows_in_val,
        "halluc_target_vs_filled": {
            # Stage 2 carves val out of the pools, so --halluc_total is the train
            # quota; stage 3 carves val out of the filled rows, so the total
            # counts train + val.
            "target": args.halluc_total,
            "filled": len(rows) + halluc_rows_in_val if args.stage3_mix else len(rows),
            "filled_train": len(rows),
            "filled_val": halluc_rows_in_val,
        },
        "requested_ratios": {
            # Side weights, so the SUM pairs count 0.5/0.5 the way section 4.8
            # accounts for them; the four-tier composition is fixed by the base
            # table (stage-2 table B, or STAGE3_QUOTA under --stage3_mix).
            "unsolvable_share": unsolvable_ratio,
            "old_domain_share": old_domain_ratio,
            "code_share": code_share,
        },
        "row_kinds": {
            "solvable_rows": n_solvable_rows,
            "pair_rows": n_pair,
            "unsolvable_rows": len(rows) - n_solvable_rows - n_pair,
            "side_weights": {"solvable": solvable_weight, "unsolvable": unsolvable_weight},
        },
        "achieved_ratios": achieved_ratios,
        "dropped_overlapping_stage1": dropped_old,
        "dropped_duplicate_questions": dropped_self,
        "dropped_minhash_near_duplicates": minhash_dropped,
        "dropped_minhash_pair_twins": minhash_twins,
        "minhash_threshold": args.minhash_threshold,
        "cells": cell_report,
        "harvested_old_domain": {
            "requested_rows": len(old_rows),
            "code_target": old_report["code_target"],
            "code_rows": old_report["code_rows"],
            "code_shortfall": old_report["shortfall"],
            "by_ability": old_report["by_ability"],
        },
        "train_by_ability": dict(Counter(r["ability"] for r in all_train)),
        "train_by_source": dict(Counter(r["extra_info"].get("source", "") for r in all_train)),
        "halluc_by_branch": dict(Counter(r["extra_info"]["branch"] for r in rows)),
        "halluc_by_template": dict(Counter(r["extra_info"]["template"] for r in rows)),
        "halluc_by_data_source": dict(Counter(r["data_source"] for r in rows)),
        # Domain weights of the halluc slice (logic / commonsense / math), so the
        # stage-3 20/20/60 target is readable from the artifact's own stats.
        "halluc_by_domain": dict(
            Counter(schema.SOURCES.get(r["data_source"], ("?", ""))[0] for r in rows)
        ),
        "template_balance": templates,
    }
    with open(os.path.join(output_dir, "mix_stats.json"), "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"[done] train={len(all_train)} -> {train_path}")
    if val_path:
        print(f"[done] val={len(val_rows)} -> {val_path}")
    print(f"[done] hallucination by branch: {stats['halluc_by_branch']}")
    print(f"[done] hallucination by domain: {stats['halluc_by_domain']}")
    print(f"[done] template balance: {templates}")
    short = [c for c in cell_report if c["shortfall"] - c["redistributed"] > 0]
    if short:
        print(f"[done] {len(short)} cell(s) under-filled after redistribution:")
        for cell in short:
            print(f"    {cell['cell']}: {cell['filled'] + cell['redistributed']}/{cell['target']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
