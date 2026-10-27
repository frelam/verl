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
"""Tests for the stage-2 mixer's quota arithmetic, balance gate and val carve.

The fixtures are synthetic rows in the shape the adapters produce, so these tests
run without the downloaded raw data.  What they pin down is the *mixer's*
behaviour: that the default knobs reproduce design doc section 4.8 table B to the
row, that the 60/40 accounting counts SUM's pair rows 0.5/0.5, what happens when a
cell's pool is short, that a paired row never straddles the train/val boundary,
and that an unbalanced template aborts the write rather than producing a subtly
leaky artifact.
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mix_halluc
import schema
from schema import (
    BRANCH_SOLVABLE_JUDGE,
    BRANCH_SOLVABLE_NUMERIC,
    BRANCH_SOLVABLE_PAIR,
    BRANCH_SOLVABLE_ROLES,
    BRANCH_SOLVABLE_TWO_LAYER,
    BRANCH_UNSOLVABLE_BARE,
    BRANCH_UNSOLVABLE_DIAG,
    SOURCE_CREPE,
    SOURCE_FALSEQA,
    SOURCE_GSMIC,
    SOURCE_KK,
    SOURCE_MAIN,
    SOURCE_MIP,
    SOURCE_SUM,
    SOURCE_TREECUT,
    SOURCE_UMWP,
    TEMPLATE_A,
    TEMPLATE_B,
    TEMPLATE_B_JUDGE,
    TEMPLATE_C,
)

PLACEHOLDER_OPTIONS = [{"id": c, "text": f"span {c}"} for c in "ABC"]

# (branch, data_source, template, solvable, pool_size) -- the shape of each cell's
# pool.  Every pool is a little larger than its table-B quota (the mix has to
# carve val first *and* fill train), except ``(unsolvable_bare, MIP)``, which is
# deliberately short so the shortfall path is exercised.  There is no
# ``(solvable_numeric, KK)`` cell: a K&K answer is a D19 name->role mapping, so the
# 100 synthesised-distractor K&K rows join the 1,600 K&K role rows -- hence 1,700.
CELL_SHAPES = [
    (BRANCH_SOLVABLE_NUMERIC, SOURCE_GSMIC, TEMPLATE_B, True, 900),
    (BRANCH_SOLVABLE_NUMERIC, SOURCE_UMWP, TEMPLATE_B, True, 300),
    (BRANCH_SOLVABLE_NUMERIC, SOURCE_MAIN, TEMPLATE_B, True, 200),
    (BRANCH_SOLVABLE_ROLES, SOURCE_KK, TEMPLATE_B, True, 2000),
    (BRANCH_SOLVABLE_TWO_LAYER, SOURCE_UMWP, TEMPLATE_A, True, 700),
    (BRANCH_SOLVABLE_TWO_LAYER, SOURCE_FALSEQA, TEMPLATE_A, True, 1100),
    (BRANCH_SOLVABLE_JUDGE, SOURCE_CREPE, TEMPLATE_B_JUDGE, True, 350),
    (BRANCH_SOLVABLE_NUMERIC, SOURCE_TREECUT, TEMPLATE_A, True, 650),
    (BRANCH_SOLVABLE_PAIR, SOURCE_SUM, TEMPLATE_C, True, 7000),
    (BRANCH_UNSOLVABLE_DIAG, SOURCE_FALSEQA, TEMPLATE_A, False, 1100),
    (BRANCH_UNSOLVABLE_DIAG, SOURCE_TREECUT, TEMPLATE_A, False, 5400),
    (BRANCH_UNSOLVABLE_BARE, SOURCE_UMWP, TEMPLATE_B, False, 2700),
    (BRANCH_UNSOLVABLE_BARE, SOURCE_CREPE, TEMPLATE_B, False, 500),
    (BRANCH_UNSOLVABLE_BARE, SOURCE_MIP, TEMPLATE_B, False, 100),  # short pool on purpose
]


def make_cell_rows(branch, data_source, template, solvable, n, seed=0):
    """Synthetic rows in the shape an adapter emits for this cell."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        options = None
        extra = {"task_id": f"{data_source}-{branch}-{i}", "index": i, "seed": seed}
        gt_kwargs = {"solvable": solvable}
        question_b = None
        if solvable:
            gt_kwargs["answer"] = "42"
            if branch == BRANCH_SOLVABLE_ROLES:
                # D19: a name -> surface role word mapping, plus the row's own
                # [truth-teller, liar] words.
                gt_kwargs["answer"] = {"Oliver": "knight", "Ethan": "knave"}
                gt_kwargs["role_words"] = ["knight", "knave"]
            elif branch == BRANCH_SOLVABLE_TWO_LAYER:
                if data_source == SOURCE_FALSEQA:
                    gt_kwargs["solvable_answer"] = True
                    gt_kwargs["answer"] = "a teacher"
                    extra["pair_id"] = f"falseqa:train:{i}"
                else:
                    gt_kwargs["two_layer"] = True
                options = [dict(opt) for opt in PLACEHOLDER_OPTIONS]
            elif branch == BRANCH_SOLVABLE_JUDGE:
                gt_kwargs["judgment_only"] = True
            elif branch == BRANCH_SOLVABLE_PAIR:
                gt_kwargs["pair_task"] = True
                gt_kwargs["answerable_id"] = "A" if i % 2 == 0 else "B"
                question_b = f"Second question {i}?"
            elif template == TEMPLATE_A:
                # TreeCut positive: a placeholder block with no correct item.
                options = [dict(opt) for opt in PLACEHOLDER_OPTIONS]
        else:
            if template == TEMPLATE_A:
                options = [dict(opt) for opt in PLACEHOLDER_OPTIONS]
                gt_kwargs["correct_option_id"] = rng.choice("ABC")
                gt_kwargs["has_diagnosis_label"] = True
                gt_kwargs["perturbation_type"] = "missing_condition"
                if data_source == SOURCE_FALSEQA:
                    extra["pair_id"] = f"falseqa:train:{i}"
        rows.append(
            schema.make_row(
                data_source=data_source,
                question=f"Question {data_source} {branch} {i}?",
                ground_truth=schema.ground_truth_json(schema.ground_truth_payload(**gt_kwargs)),
                template=template,
                branch=branch,
                extra_info=extra,
                options=options,
                question_b=question_b,
            )
        )
    return rows


@pytest.fixture
def build_dir(tmp_path):
    """A directory of adapter outputs shaped like the real ones."""
    path = tmp_path / "build"
    path.mkdir()
    rows = []
    for branch, data_source, template, solvable, n in CELL_SHAPES:
        rows.extend(make_cell_rows(branch, data_source, template, solvable, n))
    # Split across three files to also exercise multi-file reading.
    for i in range(3):
        schema.write_rows_parquet(rows[i :: 3], str(path / f"part{i}.parquet"))
    return str(path)


def side_weights(quota):
    """(solvable, unsolvable) side weight of a quota table (pairs count 0.5)."""
    solvable = sum(v for k, v in quota.items() if k[0] in mix_halluc.SOLVABLE_BRANCHES)
    unsolvable = sum(v for k, v in quota.items() if k[0] in mix_halluc.UNSOLVABLE_BRANCHES)
    pair = sum(v for k, v in quota.items() if k[0] in mix_halluc.PAIR_BRANCHES)
    return solvable + pair / 2, unsolvable + pair / 2


class TestQuotaScaling:
    def test_defaults_reproduce_table_b(self):
        quota = mix_halluc.scaled_quota(20000, 0.6)
        assert quota == mix_halluc.DEFAULT_QUOTA
        assert sum(quota.values()) == 20000
        # 5,000 pure-solvable rows + 6,000 pair rows + 9,000 unsolvable rows.
        assert quota[(BRANCH_SOLVABLE_NUMERIC, SOURCE_GSMIC)] == 772
        assert quota[(BRANCH_SOLVABLE_NUMERIC, SOURCE_TREECUT)] == 500
        assert quota[(BRANCH_SOLVABLE_ROLES, SOURCE_KK)] == 1700
        assert quota[(BRANCH_SOLVABLE_TWO_LAYER, SOURCE_UMWP)] == 550
        assert quota[(BRANCH_SOLVABLE_TWO_LAYER, SOURCE_FALSEQA)] == 928
        assert quota[(BRANCH_SOLVABLE_JUDGE, SOURCE_CREPE)] == 250
        assert quota[(BRANCH_SOLVABLE_PAIR, SOURCE_SUM)] == 6000
        assert quota[(BRANCH_UNSOLVABLE_DIAG, SOURCE_FALSEQA)] == 928
        assert quota[(BRANCH_UNSOLVABLE_DIAG, SOURCE_TREECUT)] == 4907
        assert quota[(BRANCH_UNSOLVABLE_BARE, SOURCE_UMWP)] == 2489
        assert quota[(BRANCH_UNSOLVABLE_BARE, SOURCE_CREPE)] == 400
        assert quota[(BRANCH_UNSOLVABLE_BARE, SOURCE_MIP)] == 276
        assert (BRANCH_SOLVABLE_NUMERIC, SOURCE_KK) not in quota

    def test_defaults_give_the_documented_40_60_split(self):
        # The 60/40 of section 4.8 counts SUM's pair rows 0.5/0.5; a row-count
        # reading would report 11,000/20,000 = 55% solvable instead.
        solvable, unsolvable = side_weights(mix_halluc.scaled_quota(20000, 0.6))
        assert (solvable, unsolvable) == (8000, 12000)
        assert solvable / (solvable + unsolvable) == pytest.approx(0.4)

    def test_four_tier_composition_is_table_bs(self):
        quota = mix_halluc.scaled_quota(20000, 0.6)
        four = sum(v for k, v in quota.items() if k[0] == BRANCH_UNSOLVABLE_DIAG)
        three = sum(v for k, v in quota.items() if k[0] == BRANCH_UNSOLVABLE_BARE)
        assert (four, three) == (5835, 3165)

    @pytest.mark.parametrize(
        "total,unsolvable_ratio", [(20000, 0.6), (1000, 0.5), (333, 0.6), (20000, 0.7), (77, 0.25)]
    )
    def test_groups_always_sum_exactly(self, total, unsolvable_ratio):
        quota = mix_halluc.scaled_quota(total, unsolvable_ratio)
        assert sum(quota.values()) == total
        solvable, unsolvable = side_weights(quota)
        # Rounding is at most one row per group on each side.
        assert unsolvable == pytest.approx(unsolvable_ratio * total, abs=line_groups(quota) + 2)
        assert solvable == pytest.approx((1 - unsolvable_ratio) * total, abs=line_groups(quota) + 2)

    def test_rejects_out_of_range_ratios(self):
        with pytest.raises(ValueError, match="halluc_unsolvable_ratio"):
            mix_halluc.scaled_quota(20000, 1.0)
        with pytest.raises(ValueError, match="halluc_unsolvable_ratio"):
            mix_halluc.scaled_quota(20000, 0.0)

    def test_rejects_a_pair_share_that_swallows_a_side(self):
        # 30% of the total is pair rows, i.e. 15% of side weight per side, so a 0.1
        # unsolvable ratio leaves negative unsolvable weight; the mixer must say so
        # rather than return a negative quota.
        with pytest.raises(ValueError, match="pair share"):
            mix_halluc.scaled_quota(20000, 0.1)

    def test_every_cell_is_represented(self):
        # A quota cell with no default entry silently loses its rows.
        assert set(mix_halluc.scaled_quota(20000, 0.6)) == set(mix_halluc.DEFAULT_QUOTA)

    def test_the_branch_constants_all_have_a_group(self):
        for branch in schema.BRANCHES:
            assert mix_halluc._group_of(branch) in mix_halluc.QUOTA_GROUPS


def line_groups(quota):
    """How many independently rounded groups a quota has (the rounding budget)."""
    return len({mix_halluc._group_of(cell[0]) for cell in quota})


class TestIndexPool:
    def test_buckets_by_branch_and_source(self):
        rows = make_cell_rows(BRANCH_SOLVABLE_NUMERIC, SOURCE_GSMIC, TEMPLATE_B, True, 5)
        pool = mix_halluc.index_pool(rows)
        assert len(pool[(BRANCH_SOLVABLE_NUMERIC, SOURCE_GSMIC)]) == 5

    def test_accepts_pair_and_two_layer_rows(self):
        rows = make_cell_rows(BRANCH_SOLVABLE_PAIR, SOURCE_SUM, TEMPLATE_C, True, 2)
        rows += make_cell_rows(BRANCH_SOLVABLE_TWO_LAYER, SOURCE_FALSEQA, TEMPLATE_A, True, 2)
        pool = mix_halluc.index_pool(rows)
        assert len(pool[(BRANCH_SOLVABLE_PAIR, SOURCE_SUM)]) == 2
        assert len(pool[(BRANCH_SOLVABLE_TWO_LAYER, SOURCE_FALSEQA)]) == 2

    def test_rejects_solvable_flag_mismatch(self):
        rows = make_cell_rows(BRANCH_UNSOLVABLE_DIAG, SOURCE_TREECUT, TEMPLATE_A, False, 1)
        rows[0]["reward_model"]["ground_truth"] = schema.ground_truth_json(
            schema.ground_truth_payload(solvable=True, answer="1")
        )
        with pytest.raises(ValueError, match="solvable"):
            mix_halluc.index_pool(rows)

    def test_rejects_a_two_layer_row_marked_unsolvable(self):
        rows = make_cell_rows(BRANCH_SOLVABLE_TWO_LAYER, SOURCE_FALSEQA, TEMPLATE_A, True, 1)
        rows[0]["reward_model"]["ground_truth"] = schema.ground_truth_json(
            schema.ground_truth_payload(solvable=False, has_diagnosis_label=True, correct_option_id="A")
        )
        with pytest.raises(ValueError, match="solvable"):
            mix_halluc.index_pool(rows)

    def test_rejects_malformed_ground_truth(self):
        rows = make_cell_rows(BRANCH_UNSOLVABLE_BARE, SOURCE_MIP, TEMPLATE_B, False, 1)
        rows[0]["reward_model"]["ground_truth"] = "not json"
        with pytest.raises(ValueError, match="solvable"):
            mix_halluc.index_pool(rows)


class TestFillCells:
    def test_short_pool_is_reported_and_redistributed_within_side(self):
        pool = mix_halluc.index_pool(
            make_cell_rows(BRANCH_UNSOLVABLE_DIAG, SOURCE_UMWP, TEMPLATE_A, False, 3)
            + make_cell_rows(BRANCH_UNSOLVABLE_DIAG, SOURCE_TREECUT, TEMPLATE_A, False, 100)
        )
        quota = {
            (BRANCH_UNSOLVABLE_DIAG, SOURCE_UMWP): 50,
            (BRANCH_UNSOLVABLE_DIAG, SOURCE_TREECUT): 20,
        }
        rows, report = mix_halluc.fill_cells(pool, quota, random.Random(0))
        by_cell = {tuple(e["cell"]): e for e in report}
        short = by_cell[(BRANCH_UNSOLVABLE_DIAG, SOURCE_UMWP)]
        assert short["filled"] == 3 and short["shortfall"] == 47
        # The other four-tier cell had 80 spare, so it absorbs the deficit.
        assert by_cell[(BRANCH_UNSOLVABLE_DIAG, SOURCE_TREECUT)]["redistributed"] == 47
        assert len(rows) == 70

    def test_deficit_does_not_cross_the_tier_boundary(self):
        # A four-tier cell runs short; the three-tier cells are full of surplus.
        # The table-B composition must win over backfilling the total.
        pool = mix_halluc.index_pool(
            make_cell_rows(BRANCH_UNSOLVABLE_DIAG, SOURCE_UMWP, TEMPLATE_A, False, 2)
            + make_cell_rows(BRANCH_UNSOLVABLE_BARE, SOURCE_UMWP, TEMPLATE_B, False, 500)
        )
        quota = {(BRANCH_UNSOLVABLE_DIAG, SOURCE_UMWP): 40, (BRANCH_UNSOLVABLE_BARE, SOURCE_UMWP): 5}
        rows, report = mix_halluc.fill_cells(pool, quota, random.Random(3))
        assert len(rows) == 2 + 5
        by_cell = {tuple(e["cell"]): e for e in report}
        assert by_cell[(BRANCH_UNSOLVABLE_BARE, SOURCE_UMWP)]["redistributed"] == 0

    def test_spill_across_tiers_preserves_the_total_when_asked(self):
        pool = mix_halluc.index_pool(
            make_cell_rows(BRANCH_UNSOLVABLE_DIAG, SOURCE_UMWP, TEMPLATE_A, False, 2)
            + make_cell_rows(BRANCH_UNSOLVABLE_BARE, SOURCE_UMWP, TEMPLATE_B, False, 500)
        )
        quota = {(BRANCH_UNSOLVABLE_DIAG, SOURCE_UMWP): 40, (BRANCH_UNSOLVABLE_BARE, SOURCE_UMWP): 5}
        rows, _ = mix_halluc.fill_cells(pool, quota, random.Random(3), spill_across_tiers=True)
        assert len(rows) == 40 + 5

    def test_four_tier_deficit_prefers_four_tier_surplus(self):
        pool = mix_halluc.index_pool(
            make_cell_rows(BRANCH_UNSOLVABLE_DIAG, SOURCE_UMWP, TEMPLATE_A, False, 2)
            + make_cell_rows(BRANCH_UNSOLVABLE_DIAG, SOURCE_TREECUT, TEMPLATE_A, False, 100)
            + make_cell_rows(BRANCH_UNSOLVABLE_BARE, SOURCE_UMWP, TEMPLATE_B, False, 100)
        )
        quota = {
            (BRANCH_UNSOLVABLE_DIAG, SOURCE_UMWP): 30,
            (BRANCH_UNSOLVABLE_DIAG, SOURCE_TREECUT): 10,
            (BRANCH_UNSOLVABLE_BARE, SOURCE_UMWP): 5,
        }
        rows, report = mix_halluc.fill_cells(pool, quota, random.Random(4))
        by_cell = {tuple(e["cell"]): e for e in report}
        # 28 rows flow to the other four-tier cell, never to the three-tier one.
        assert by_cell[(BRANCH_UNSOLVABLE_DIAG, SOURCE_TREECUT)]["redistributed"] == 28
        assert by_cell[(BRANCH_UNSOLVABLE_BARE, SOURCE_UMWP)]["redistributed"] == 0
        assert len(rows) == 40 + 5

    def test_pair_deficit_is_not_backfilled_by_the_solvable_side(self):
        # The pair cell is its own group: letting solvable surplus fill it would
        # move the 40/60 accounting without saying so.
        pool = mix_halluc.index_pool(
            make_cell_rows(BRANCH_SOLVABLE_PAIR, SOURCE_SUM, TEMPLATE_C, True, 2)
            + make_cell_rows(BRANCH_SOLVABLE_NUMERIC, SOURCE_GSMIC, TEMPLATE_B, True, 500)
        )
        quota = {
            (BRANCH_SOLVABLE_PAIR, SOURCE_SUM): 50,
            (BRANCH_SOLVABLE_NUMERIC, SOURCE_GSMIC): 10,
        }
        rows, report = mix_halluc.fill_cells(pool, quota, random.Random(5))
        by_cell = {tuple(e["cell"]): e for e in report}
        assert by_cell[(BRANCH_SOLVABLE_NUMERIC, SOURCE_GSMIC)]["redistributed"] == 0
        assert len(rows) == 2 + 10

    def test_no_duplicate_rows_after_redistribution(self):
        pool = mix_halluc.index_pool(
            make_cell_rows(BRANCH_UNSOLVABLE_BARE, SOURCE_MIP, TEMPLATE_B, False, 5)
            + make_cell_rows(BRANCH_UNSOLVABLE_BARE, SOURCE_UMWP, TEMPLATE_B, False, 50)
        )
        quota = {(BRANCH_UNSOLVABLE_BARE, SOURCE_MIP): 10, (BRANCH_UNSOLVABLE_BARE, SOURCE_UMWP): 5}
        rows, _ = mix_halluc.fill_cells(pool, quota, random.Random(1))
        ids = [r["extra_info"]["task_id"] for r in rows]
        assert len(ids) == len(set(ids))

    def test_shortfall_stays_on_its_own_side(self):
        # A short SOLVABLE cell must not be backfilled from the unsolvable pool,
        # which would move the 40/60 split.
        pool = mix_halluc.index_pool(
            make_cell_rows(BRANCH_SOLVABLE_ROLES, SOURCE_KK, TEMPLATE_B, True, 2)
            + make_cell_rows(BRANCH_UNSOLVABLE_BARE, SOURCE_UMWP, TEMPLATE_B, False, 50)
        )
        quota = {(BRANCH_SOLVABLE_ROLES, SOURCE_KK): 20, (BRANCH_UNSOLVABLE_BARE, SOURCE_UMWP): 10}
        rows, report = mix_halluc.fill_cells(pool, quota, random.Random(2))
        # 2 solvable + 10 unsolvable, and nothing crossed the boundary to make up
        # the solvable deficit.
        assert len(rows) == 2 + 10
        branches = [r["extra_info"]["branch"] for r in rows]
        assert branches.count(BRANCH_SOLVABLE_ROLES) == 2
        assert branches.count(BRANCH_UNSOLVABLE_BARE) == 10
        assert not any(e["redistributed"] for e in report)


class TestPairAtomicity:
    """D27: a FalseQA pair's twins must not straddle the train/val boundary."""

    def test_a_val_row_without_its_twin_returns_to_train(self):
        rows = make_cell_rows(BRANCH_SOLVABLE_TWO_LAYER, SOURCE_FALSEQA, TEMPLATE_A, True, 4)
        twin = make_cell_rows(BRANCH_UNSOLVABLE_DIAG, SOURCE_FALSEQA, TEMPLATE_A, False, 4)
        pool = mix_halluc.index_pool(rows + twin)
        val = [pool[(BRANCH_SOLVABLE_TWO_LAYER, SOURCE_FALSEQA)][0]]
        kept, returned = mix_halluc.enforce_pair_atomicity(val, pool)
        assert kept == []
        assert len(returned) == 1
        # The returned row is back in its own pool cell.
        assert len(pool[(BRANCH_SOLVABLE_TWO_LAYER, SOURCE_FALSEQA)]) == 5

    def test_both_twins_stay_in_val_together(self):
        rows = make_cell_rows(BRANCH_SOLVABLE_TWO_LAYER, SOURCE_FALSEQA, TEMPLATE_A, True, 4)
        twin = make_cell_rows(BRANCH_UNSOLVABLE_DIAG, SOURCE_FALSEQA, TEMPLATE_A, False, 4)
        pool = mix_halluc.index_pool(rows + twin)
        val = [pool[(BRANCH_SOLVABLE_TWO_LAYER, SOURCE_FALSEQA)][0], pool[(BRANCH_UNSOLVABLE_DIAG, SOURCE_FALSEQA)][0]]
        kept, returned = mix_halluc.enforce_pair_atomicity(val, pool)
        assert len(kept) == 2 and returned == []

    def test_unpaired_rows_are_untouched(self):
        rows = make_cell_rows(BRANCH_UNSOLVABLE_BARE, SOURCE_UMWP, TEMPLATE_B, False, 2)
        pool = mix_halluc.index_pool(rows)
        val = list(pool[(BRANCH_UNSOLVABLE_BARE, SOURCE_UMWP)])
        kept, returned = mix_halluc.enforce_pair_atomicity(val, pool)
        assert len(kept) == 2 and returned == []


class TestTemplateBalance:
    def test_balanced_templates_pass(self):
        rows = make_cell_rows(BRANCH_SOLVABLE_NUMERIC, SOURCE_GSMIC, TEMPLATE_B, True, 2)
        rows += make_cell_rows(BRANCH_UNSOLVABLE_BARE, SOURCE_MIP, TEMPLATE_B, False, 2)
        assert mix_halluc.check_template_balance(rows) == {TEMPLATE_B: {"solvable": 2, "unsolvable": 2}}

    def test_one_sided_template_is_fatal(self):
        rows = make_cell_rows(BRANCH_SOLVABLE_NUMERIC, SOURCE_GSMIC, TEMPLATE_B, True, 2)
        with pytest.raises(ValueError, match="only one side"):
            mix_halluc.check_template_balance(rows)

    def test_judge_variant_is_counted_inside_the_b_family(self):
        # B_judge is B's verdict-only variant; a B-family count that carries both
        # sides is balanced even when each individual string is one-sided.
        rows = make_cell_rows(BRANCH_SOLVABLE_JUDGE, SOURCE_CREPE, TEMPLATE_B_JUDGE, True, 2)
        rows += make_cell_rows(BRANCH_UNSOLVABLE_BARE, SOURCE_CREPE, TEMPLATE_B, False, 2)
        assert mix_halluc.check_template_balance(rows) == {TEMPLATE_B: {"solvable": 2, "unsolvable": 2}}

    def test_template_c_is_exempt(self):
        # Every SUM pair row is solvable=true yet carries an unanswerable question,
        # so C covers both sides by construction (section 4.8).
        rows = make_cell_rows(BRANCH_SOLVABLE_PAIR, SOURCE_SUM, TEMPLATE_C, True, 3)
        assert mix_halluc.check_template_balance(rows) == {TEMPLATE_C: {"solvable": 3}}

    def test_template_a_solvable_and_unsolvable_is_balanced(self):
        rows = make_cell_rows(BRANCH_SOLVABLE_TWO_LAYER, SOURCE_UMWP, TEMPLATE_A, True, 2)
        rows += make_cell_rows(BRANCH_UNSOLVABLE_DIAG, SOURCE_TREECUT, TEMPLATE_A, False, 2)
        assert mix_halluc.check_template_balance(rows) == {TEMPLATE_A: {"solvable": 2, "unsolvable": 2}}


class TestAdapterOutputAgreement:
    """Every adapter must write where the mixer reads by default.

    This is a typo-class guard, and it earns its place: ``--halluc_dir`` shipped
    defaulting to ``.../halluc/build`` while every adapter's ``DEFAULT_OUT`` pointed
    at ``.../halluc/built``.  Nothing failed at build time -- the mixer simply found
    no rows and raised "no hallucination rows found", which reads like a missing
    build rather than a one-character bug.
    """

    @staticmethod
    def _default_out(path):
        """Read a module-level ``DEFAULT_OUT`` out of an adapter's source.

        Parsed rather than imported so the check cannot be defeated by an import
        side effect, and so a half-written adapter reports as a parse error instead
        of a mysterious failure elsewhere.
        """
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(t, ast.Name) and t.id == "DEFAULT_OUT" for t in node.targets):
                continue
            # `DEFAULT_OUT = os.path.expanduser("...")` or a plain string.
            value = node.value
            if isinstance(value, ast.Call):
                assert len(value.args) == 1, f"{path.name}: DEFAULT_OUT call takes one argument"
                value = value.args[0]
            assert isinstance(value, ast.Constant) and isinstance(value.value, str), (
                f"{path.name}: DEFAULT_OUT is not a literal path"
            )
            return value.value
        raise AssertionError(f"{path.name}: no module-level DEFAULT_OUT")

    def test_every_adapter_writes_into_the_default_halluc_dir(self):
        adapter_dir = os.path.dirname(os.path.abspath(mix_halluc.__file__))
        sources = [p for p in sorted(Path(adapter_dir).glob("*_adapter.py")) if not p.name.startswith("test_")]
        sources += sorted(Path(adapter_dir).glob("distractor_synth.py"))
        assert sources, "no adapter sources found next to mix_halluc.py"
        wanted = os.path.expanduser(mix_halluc.DEFAULT_ADAPTER_DIR)
        for source in sources:
            out = self._default_out(source)
            assert os.path.dirname(os.path.expanduser(out)) == wanted, (
                f"{source.name} writes to {out!r}, but mix_halluc.py reads {wanted!r} by default"
            )

    def test_the_cli_default_is_that_same_dir(self):
        """The constant is only useful if the flag actually defaults to it."""
        parser = mix_halluc.build_parser()
        default = parser.get_default("halluc_dir")
        assert os.path.expanduser(default) == os.path.expanduser(mix_halluc.DEFAULT_ADAPTER_DIR)

    def test_every_report_is_named_after_its_own_out(self):
        """A scratch ``--out`` must not overwrite the report the build cites.

        Same typo class as above, measured on a real run: ``distractor_synth``
        built 334 rows to a slice directory and wrote its funnel report to the
        canonical ``built/d17_report.json``, leaving a full-quota report
        describing a slice.
        """
        adapter_dir = os.path.dirname(os.path.abspath(mix_halluc.__file__))
        sources = [p for p in sorted(Path(adapter_dir).glob("*_adapter.py")) if not p.name.startswith("test_")]
        sources += sorted(Path(adapter_dir).glob("distractor_synth.py"))
        reporters = []
        for source in sources:
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == "report_path_for":
                    reporters.append(source.name)
        assert reporters, "no adapter declares report_path_for"
        for name in reporters:
            module = importlib.import_module(name[:-3])
            assert module.report_path_for("/tmp/scratch/x.parquet") == "/tmp/scratch/x_report.json"


class TestEndToEnd:
    def run_mix(self, build_dir, tmp_path, **overrides):
        args = {
            "halluc_dir": build_dir,
            "output_dir": str(tmp_path / "final"),
            "halluc_total": 2000,
            "val_size": 64,
            "seed": 7,
        }
        args.update(overrides)
        flags: list[str] = []
        for key, value in args.items():
            flags += [f"--{key}", str(value)]
        assert mix_halluc.main(flags) == 0
        return args["output_dir"]

    def test_artifacts_and_stats(self, build_dir, tmp_path):
        out = self.run_mix(build_dir, tmp_path)
        train = mix_halluc.read_parquet_rows(os.path.join(out, "train.parquet"))
        val = mix_halluc.read_parquet_rows(os.path.join(out, "val.parquet"))
        assert len(train) == 2000
        assert len(val) <= 64
        stats = json.load(open(os.path.join(out, "mix_stats.json")))
        assert stats["total_train"] == 2000
        assert stats["halluc_target_vs_filled"]["target"] == 2000
        assert sum(stats["halluc_by_branch"].values()) == 2000
        # Indices are re-contiguous and val is labelled.
        assert [r["extra_info"]["index"] for r in train] == list(range(2000))
        assert all(r["extra_info"]["split"] == "val" for r in val)
        assert all(r["extra_info"]["pass_rate"] == -1.0 for r in train)

    def test_proportions_match_the_knobs_in_side_weight(self, build_dir, tmp_path):
        out = self.run_mix(build_dir, tmp_path, halluc_total=4000)
        stats = json.load(open(os.path.join(out, "mix_stats.json")))
        ratios = stats["achieved_ratios"]
        assert ratios["solvable_share"] == pytest.approx(0.4, abs=0.01)
        assert ratios["unsolvable_share"] == pytest.approx(0.6, abs=0.01)
        kinds = stats["row_kinds"]
        # A row-count reading would call the pair rows solvable and report ~55%.
        assert kinds["pair_rows"] == pytest.approx(0.3 * stats["halluc_rows_in_train"], abs=2)

    def test_every_train_row_validates(self, build_dir, tmp_path):
        out = self.run_mix(build_dir, tmp_path, halluc_total=500)
        train = mix_halluc.read_parquet_rows(os.path.join(out, "train.parquet"))
        schema.validate_rows(train)

    def test_no_val_row_shares_a_pair_with_train(self, build_dir, tmp_path):
        out = self.run_mix(build_dir, tmp_path, halluc_total=4000, val_size=200)
        train = mix_halluc.read_parquet_rows(os.path.join(out, "train.parquet"))
        val = mix_halluc.read_parquet_rows(os.path.join(out, "val.parquet"))
        train_pairs = {r["extra_info"]["pair_id"] for r in train if r["extra_info"]["pair_id"]}
        val_pairs = {r["extra_info"]["pair_id"] for r in val if r["extra_info"]["pair_id"]}
        assert not (train_pairs & val_pairs)

    def test_short_cell_shows_up_in_stats(self, build_dir, tmp_path):
        # (unsolvable_bare, MIP) is the deliberately short cell in CELL_SHAPES.
        out = self.run_mix(build_dir, tmp_path, halluc_total=20000)
        stats = json.load(open(os.path.join(out, "mix_stats.json")))
        cell = next(e for e in stats["cells"] if e["cell"] == [BRANCH_UNSOLVABLE_BARE, SOURCE_MIP])
        assert cell["shortfall"] > 0
        assert stats["halluc_target_vs_filled"]["filled"] <= 20000


class TestStage1Pool:
    """Section 7.1: stage-1 rows come in as train.parquet + val.parquet.

    Pointing ``--stage1_path`` at stage-1's output *directory* used to read it
    recursively, which folded stage-1's own val rows into stage-2's train set.
    These pin the split down in both directions: val out of train, and val kept
    in val (otherwise a stage-2 run has no old-domain eval signal at all).
    """

    def stage1_dir(self, tmp_path, n_train=60, n_val=5):
        path = tmp_path / "stage1"
        path.mkdir()
        train = make_cell_rows(BRANCH_SOLVABLE_NUMERIC, SOURCE_GSMIC, TEMPLATE_B, True, n_train, seed=11)
        val = make_cell_rows(BRANCH_SOLVABLE_NUMERIC, SOURCE_GSMIC, TEMPLATE_B, True, n_val, seed=12)
        for row in train:
            row["extra_info"]["task_id"] = f"s1train-{row['extra_info']['task_id']}"
        for row in val:
            row["extra_info"]["task_id"] = f"s1val-{row['extra_info']['task_id']}"
        schema.write_rows_parquet(train, str(path / "train.parquet"))
        schema.write_rows_parquet(val, str(path / "val.parquet"))
        return str(path)

    def test_read_stage1_pool_keeps_val_out_of_train(self, tmp_path):
        train, val = mix_halluc.read_stage1_pool(self.stage1_dir(tmp_path))
        train_ids = {r["extra_info"]["task_id"] for r in train}
        val_ids = {r["extra_info"]["task_id"] for r in val}
        assert train_ids and val_ids
        assert not (train_ids & val_ids)
        assert all(task_id.startswith("s1train-") for task_id in train_ids)
        assert all(task_id.startswith("s1val-") for task_id in val_ids)

    def test_stage1_val_rows_land_in_val_not_train(self, build_dir, tmp_path):
        stage1 = self.stage1_dir(tmp_path)
        out = TestEndToEnd().run_mix(build_dir, tmp_path, halluc_total=200, val_size=20, stage1_path=stage1)
        train = {r["extra_info"]["task_id"] for r in mix_halluc.read_parquet_rows(os.path.join(out, "train.parquet"))}
        val = mix_halluc.read_parquet_rows(os.path.join(out, "val.parquet"))
        val_ids = {r["extra_info"]["task_id"] for r in val}
        assert any(task_id.startswith("s1val-") for task_id in val_ids)
        assert not any(task_id.startswith("s1val-") for task_id in train)
        assert all(r["extra_info"]["split"] == "val" for r in val)
        stats = json.load(open(os.path.join(out, "mix_stats.json")))
        assert stats["stage1_val_rows_kept_in_val"] == 5
        assert stats["total_val"] == len(val)

    def test_stage1_train_rows_are_sampled_into_train(self, build_dir, tmp_path):
        stage1 = self.stage1_dir(tmp_path)
        out = TestEndToEnd().run_mix(build_dir, tmp_path, halluc_total=200, val_size=20, stage1_path=stage1)
        train = mix_halluc.read_parquet_rows(os.path.join(out, "train.parquet"))
        assert any(r["extra_info"]["task_id"].startswith("s1train-") for r in train)


class TestNearDedup:
    """Section 7.2's MinHash pass: reworded stage-1 duplicates must not survive."""

    @staticmethod
    def long_question():
        """A ~395-token question, so a two-word edit stays a near-duplicate."""
        clauses = [
            f"on day {i} a caravan moves {i * 3} crates from the harbour warehouse to stall {i + 2}"
            for i in range(1, 25)
        ]
        return (
            "The chronicle records that "
            + ", and ".join(clauses)
            + ". How many crates reach the stalls in total?"
        )

    @classmethod
    def reworded_question(cls):
        """The same problem with two words swapped: Jaccard ~0.72, above 0.6."""
        return cls.long_question().replace("chronicle", "record").replace("harbour", "dock")

    UNRELATED = (
        "A train leaves the central station travelling at sixty kilometres per hour while a second "
        "train leaves two hours later from the same platform at ninety kilometres per hour along an "
        "identical route; how long after the first departure do the two trains meet on the line?"
    )

    def make_row(self, question, task_id, pair_id=None):
        extra = {"task_id": task_id}
        if pair_id:
            extra["pair_id"] = pair_id
        return schema.make_row(
            data_source=SOURCE_GSMIC,
            question=question,
            ground_truth=schema.ground_truth_json(schema.ground_truth_payload(solvable=True, answer="42")),
            template=TEMPLATE_B,
            branch=BRANCH_SOLVABLE_NUMERIC,
            extra_info=extra,
        )

    def test_reworded_duplicate_is_dropped_and_unrelated_row_survives(self):
        rows = [self.make_row(self.reworded_question(), "dup"), self.make_row(self.UNRELATED, "keep")]
        kept, dropped, twins = mix_halluc.near_dedup_against_stage1(
            rows, [self.long_question()], threshold=0.6
        )
        assert [row["extra_info"]["task_id"] for row in kept] == ["keep"]
        assert (dropped, twins) == (1, 0)

    def test_near_identical_query_rows_both_survive(self):
        """Query rows are compared with the anchors only, never with each other.

        D27's twins, UMWP's answerable/unanswerable partners and D30's kkcut
        siblings are deliberately near-identical; an all-pairs greedy pass would
        collapse each cluster to its first row (measured: the whole kkcut channel
        went to zero).
        """
        rows = [
            self.make_row(self.long_question(), "query-a"),
            self.make_row(self.reworded_question(), "query-b"),
        ]
        kept, dropped, twins = mix_halluc.near_dedup_against_stage1(
            rows, [self.UNRELATED], threshold=0.6
        )
        assert [row["extra_info"]["task_id"] for row in kept] == ["query-a", "query-b"]
        assert (dropped, twins) == (0, 0)

    def test_a_dropped_twin_takes_its_pair_with_it(self):
        rows = [
            self.make_row(self.reworded_question(), "twin-a", pair_id="falseqa:train:7"),
            self.make_row(self.UNRELATED, "twin-b", pair_id="falseqa:train:7"),
        ]
        kept, dropped, twins = mix_halluc.near_dedup_against_stage1(
            rows, [self.long_question()], threshold=0.6
        )
        assert kept == []
        assert (dropped, twins) == (2, 1)


class TestStage3Mix:
    """Stage 3 (D30): ``--stage3_mix`` builds the domain-balanced table.

    The hallucination domain is 25/75 solvable/unsolvable by side weight and its
    domain weights are logic 20% / commonsense 20% / math 60%; with the old
    domain at 60% (and solvable-only) the whole artifact is 70/30.  The new logic
    unsolvable side is the kkcut synthesis cell.  The old ``--repair_mode``
    (D29) is retired from scoring, so there is no repair quota to test anymore.
    """

    # Pools shaped like the stage-3 adapter outputs: every cell comfortably
    # above its STAGE3_QUOTA target so the test exercises the quota arithmetic,
    # not shortfall redistribution.
    STAGE3_CELL_SHAPES = [
        (BRANCH_SOLVABLE_ROLES, SOURCE_KK, TEMPLATE_B, True, 1400),
        (BRANCH_UNSOLVABLE_BARE, SOURCE_KK, TEMPLATE_B, False, 2600),
        (BRANCH_SOLVABLE_NUMERIC, SOURCE_GSMIC, TEMPLATE_B, True, 900),
        (BRANCH_SOLVABLE_NUMERIC, SOURCE_UMWP, TEMPLATE_B, True, 300),
        (BRANCH_SOLVABLE_NUMERIC, SOURCE_MAIN, TEMPLATE_B, True, 200),
        (BRANCH_SOLVABLE_NUMERIC, SOURCE_TREECUT, TEMPLATE_A, True, 650),
        (BRANCH_SOLVABLE_TWO_LAYER, SOURCE_UMWP, TEMPLATE_A, True, 700),
        (BRANCH_SOLVABLE_PAIR, SOURCE_SUM, TEMPLATE_C, True, 7000),
        (BRANCH_UNSOLVABLE_DIAG, SOURCE_TREECUT, TEMPLATE_A, False, 3600),
        (BRANCH_UNSOLVABLE_BARE, SOURCE_UMWP, TEMPLATE_B, False, 2600),
        (BRANCH_UNSOLVABLE_BARE, SOURCE_MIP, TEMPLATE_B, False, 350),
        (BRANCH_SOLVABLE_TWO_LAYER, SOURCE_FALSEQA, TEMPLATE_A, True, 1100),
        (BRANCH_SOLVABLE_JUDGE, SOURCE_CREPE, TEMPLATE_B_JUDGE, True, 800),
        (BRANCH_UNSOLVABLE_DIAG, SOURCE_FALSEQA, TEMPLATE_A, False, 1100),
        (BRANCH_UNSOLVABLE_BARE, SOURCE_CREPE, TEMPLATE_B, False, 1700),
    ]

    def test_stage3_table_sums_to_the_total(self):
        quota = mix_halluc.STAGE3_QUOTA
        assert sum(quota.values()) == 14000
        # 14,000 / 0.75 is the table's own knob pair: it must reproduce it to
        # the row, otherwise the build script's numbers and the table drift.
        assert mix_halluc.scaled_quota(14000, 0.75, base=quota) == quota

    def test_stage3_table_domain_arithmetic(self):
        """logic 20% / commonsense 20% / math 60%, each 25/75 inside."""
        solvable = defaultdict(int)
        unsolvable = defaultdict(int)
        for (branch, data_source), rows in mix_halluc.STAGE3_QUOTA.items():
            domain = schema.SOURCES[data_source][0]
            if branch in mix_halluc.PAIR_BRANCHES:
                solvable[domain] += rows / 2
                unsolvable[domain] += rows / 2
            elif branch in mix_halluc.SOLVABLE_BRANCHES:
                solvable[domain] += rows
            else:
                unsolvable[domain] += rows
        assert solvable == {"logic": 700, "commonsense": 700, "math": 2100}
        assert unsolvable == {"logic": 2100, "commonsense": 2100, "math": 6300}

    def test_stage3_table_has_the_kkcut_cell(self):
        # The synthesised logic unsolvable side (D30): kk_adapter --emit-unsolvable.
        assert mix_halluc.STAGE3_QUOTA[(BRANCH_UNSOLVABLE_BARE, SOURCE_KK)] == 2100
        # Supply-capped cells, each sized below the measured post-dedup supply so
        # --strict_cells can actually pass: MiP 267, TreeCut positives 405,
        # TreeCut negatives 3,230, GSM-IC 385, main 26.
        assert mix_halluc.STAGE3_QUOTA[(BRANCH_UNSOLVABLE_BARE, SOURCE_MIP)] == 245
        assert mix_halluc.STAGE3_QUOTA[(BRANCH_UNSOLVABLE_DIAG, SOURCE_TREECUT)] == 3030
        assert mix_halluc.STAGE3_QUOTA[(BRANCH_SOLVABLE_NUMERIC, SOURCE_TREECUT)] == 320
        assert mix_halluc.STAGE3_QUOTA[(BRANCH_SOLVABLE_NUMERIC, SOURCE_GSMIC)] == 340
        assert mix_halluc.STAGE3_QUOTA[(BRANCH_SOLVABLE_NUMERIC, SOURCE_MAIN)] == 15

    def test_stage3_mix_end_to_end(self, tmp_path):
        build = tmp_path / "built_stage3"
        build.mkdir()
        rows = []
        for branch, data_source, template, solvable, n in self.STAGE3_CELL_SHAPES:
            rows.extend(make_cell_rows(branch, data_source, template, solvable, n))
        schema.write_rows_parquet(rows, str(build / "halluc.parquet"))
        out = tmp_path / "stage3"
        assert mix_halluc.main([
            "--stage3_mix",
            "--halluc_dir", str(build),
            "--output_dir", str(out),
            "--halluc_total", "2000",
            "--val_size", "60",
            "--seed", "11",
        ]) == 0
        train = mix_halluc.read_parquet_rows(os.path.join(str(out), "train.parquet"))
        val = mix_halluc.read_parquet_rows(os.path.join(str(out), "val.parquet"))
        # --halluc_total counts train + val under --stage3_mix.
        assert len(train) + len(val) == 2000
        # Pair atomicity may return a few split twins to train, so val can come
        # in under the requested carve (the D27 rule, exercised for real here).
        assert 0 < len(val) <= 60
        schema.validate_rows(train + val)
        stats = json.load(open(os.path.join(str(out), "mix_stats.json")))
        assert stats["config"]["stage3_mix"] is True
        assert stats["config"]["halluc_unsolvable_ratio"] == pytest.approx(0.75)
        assert stats["config"]["old_domain_ratio"] == pytest.approx(0.6)
        assert stats["achieved_ratios"]["unsolvable_share"] == pytest.approx(0.75, abs=0.02)
        # The rescaled table keeps the domain shares at 20/20/60.
        domain = stats["halluc_by_domain"]
        total = sum(domain.values())
        assert domain["logic"] / total == pytest.approx(0.20, abs=0.01)
        assert domain["commonsense"] / total == pytest.approx(0.20, abs=0.01)
        assert domain["math"] / total == pytest.approx(0.60, abs=0.01)

    def test_stage3_strict_cells_refuses_a_short_cell(self, tmp_path):
        """--strict_cells fails on a short cell instead of backfilling it."""
        build = tmp_path / "built_stage3"
        build.mkdir()
        rows = []
        for branch, data_source, template, solvable, n in self.STAGE3_CELL_SHAPES:
            # Starve the K&K unsolvable cell: at total=2000 it needs ~300 rows.
            supply = 10 if (branch, data_source) == (BRANCH_UNSOLVABLE_BARE, SOURCE_KK) else n
            rows.extend(make_cell_rows(branch, data_source, template, solvable, supply))
        schema.write_rows_parquet(rows, str(build / "halluc.parquet"))
        out = tmp_path / "stage3"
        with pytest.raises(ValueError, match="strict_cells"):
            mix_halluc.main([
                "--stage3_mix",
                "--strict_cells",
                "--halluc_dir", str(build),
                "--output_dir", str(out),
                "--halluc_total", "2000",
                "--val_size", "60",
                "--seed", "7",
            ])

    def test_stage3_mix_runs_pair_atomicity(self, tmp_path):
        """Stage 3 uses the stage-2 FalseQA twins, so the val atomicity rule applies."""
        build = tmp_path / "built_stage3"
        build.mkdir()
        rows = []
        for branch, data_source, template, solvable, n in self.STAGE3_CELL_SHAPES:
            rows.extend(make_cell_rows(branch, data_source, template, solvable, n))
        schema.write_rows_parquet(rows, str(build / "halluc.parquet"))
        out = tmp_path / "stage3"
        assert mix_halluc.main([
            "--stage3_mix",
            "--halluc_dir", str(build),
            "--output_dir", str(out),
            "--halluc_total", "2000",
            "--val_size", "200",
            "--seed", "5",
        ]) == 0
        train = mix_halluc.read_parquet_rows(os.path.join(str(out), "train.parquet"))
        val = mix_halluc.read_parquet_rows(os.path.join(str(out), "val.parquet"))
        # No pair_id may straddle the boundary (D27), on either side.
        val_ids = {r["extra_info"].get("pair_id") for r in val if r["extra_info"].get("pair_id")}
        train_ids = {r["extra_info"].get("pair_id") for r in train if r["extra_info"].get("pair_id")}
        assert not val_ids & train_ids

    def test_stage3_val_keeps_kkcut_siblings_together(self, tmp_path):
        """The several cuts of one parent are near-duplicates: no straddling (D30)."""
        build = tmp_path / "built_stage3"
        build.mkdir()
        rows = []
        for branch, data_source, template, solvable, n in self.STAGE3_CELL_SHAPES:
            rows.extend(make_cell_rows(branch, data_source, template, solvable, n))
        # Reshape the K&K unsolvable cell into kkcut sibling groups: four cuts of
        # one parent differ by a single sentence.
        kk_cell = (BRANCH_UNSOLVABLE_BARE, SOURCE_KK)
        for i, row in enumerate(
            [r for r in rows if (r["extra_info"]["branch"], r["data_source"]) == kk_cell]
        ):
            row["extra_info"]["task_id"] = f"kkcut:perturbed_statement:4ppl:{i // 4}:{i % 4}"
        schema.write_rows_parquet(rows, str(build / "halluc.parquet"))
        out = tmp_path / "stage3"
        assert mix_halluc.main([
            "--stage3_mix",
            "--halluc_dir", str(build),
            "--output_dir", str(out),
            "--halluc_total", "2000",
            "--val_size", "400",
            "--seed", "5",
        ]) == 0
        train = mix_halluc.read_parquet_rows(os.path.join(str(out), "train.parquet"))
        val = mix_halluc.read_parquet_rows(os.path.join(str(out), "val.parquet"))

        def kkcut_parents(rows_):
            return {
                r["extra_info"]["task_id"].rsplit(":", 1)[0]
                for r in rows_
                if r["extra_info"]["task_id"].startswith("kkcut:")
            }

        assert val and kkcut_parents(val)
        assert not kkcut_parents(val) & kkcut_parents(train)
        # Whole sibling groups are drawn into val, so the carve neither falls
        # short of its target nor skips the kkcut channel (the old row-wise carve
        # returned almost every group and left val with a single kkcut row).
        assert len(val) >= 400
        assert sum(1 for r in val if r["extra_info"]["task_id"].startswith("kkcut:")) >= 40

    def test_stage3_pins_the_code_share_in_the_old_domain(self, tmp_path):
        """The hallucination domain has no code, so the replay must oversample it."""
        build = tmp_path / "built_stage3"
        build.mkdir()
        rows = []
        for branch, data_source, template, solvable, n in self.STAGE3_CELL_SHAPES:
            rows.extend(make_cell_rows(branch, data_source, template, solvable, n))
        schema.write_rows_parquet(rows, str(build / "halluc.parquet"))

        stage1 = tmp_path / "stage1"
        stage1.mkdir()
        old_rows = []
        for i, (ability, n) in enumerate((("code", 2000), ("math", 2000), ("logic", 1000), ("stem", 1000))):
            chunk = make_cell_rows(
                BRANCH_SOLVABLE_NUMERIC, SOURCE_GSMIC, TEMPLATE_B, True, n, seed=10 + i
            )
            for row in chunk:
                row["ability"] = ability
                row["extra_info"]["task_id"] = f"s1-{ability}-{row['extra_info']['task_id']}"
            old_rows.extend(chunk)
        schema.write_rows_parquet(old_rows, str(stage1 / "train.parquet"))

        out = tmp_path / "stage3"
        assert mix_halluc.main([
            "--stage3_mix",
            "--halluc_dir", str(build),
            "--stage1_path", str(stage1),
            "--output_dir", str(out),
            "--halluc_total", "2000",
            "--val_size", "60",
            "--seed", "3",
        ]) == 0
        stats = json.load(open(os.path.join(str(out), "mix_stats.json")))
        harvested = stats["harvested_old_domain"]
        assert harvested["code_rows"] == harvested["code_target"]
        assert stats["config"]["code_share"] == pytest.approx(0.24)
        assert stats["achieved_ratios"]["overall_code_share"] == pytest.approx(0.24, abs=0.005)
        assert stats["achieved_ratios"]["old_domain_share_of_train"] == pytest.approx(0.6, abs=0.002)

    def test_old_domain_sampling_reports_a_short_code_pool(self):
        """A code pool below the target is reported, and the row budget still holds."""
        rows = []
        for i in range(10):
            rows.append({"ability": "code", "extra_info": {"task_id": f"c{i}"}})
        for i in range(100):
            rows.append({"ability": "math", "extra_info": {"task_id": f"m{i}"}})
        rng = random.Random(0)
        chosen, report = mix_halluc.sample_old_domain(rows, 60, 200, 0.24, rng)
        # 0.24 * 200 = 48 wanted, only 10 exist.
        assert report["code_target"] == 48
        assert report["code_rows"] == 10
        assert report["shortfall"] == 38
        assert len(chosen) == 60
        assert len({row["extra_info"]["task_id"] for row in chosen}) == 60


def make_if_rows(n, prefix="if", seed=0):
    """Stage-4 IF rows in the shape ``if_adapter.py`` emits."""
    rows = []
    for i in range(n):
        rows.append(
            {
                "data_source": "if_nemotron",
                "prompt": [{"role": "user", "content": f"Instruction-following request {prefix} {i}."}],
                "ability": "if",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps(
                        {
                            "constraints": [{"id": "punctuation:no_comma", "kwargs": {}}],
                            "grading_mode": "binary",
                        }
                    ),
                },
                "extra_info": {
                    "split": "train",
                    "index": i,
                    "task_id": f"{prefix}-{i}",
                    "domain": "if",
                    "source": "wildchat",
                    "difficulty": "",
                    "prior_solve_rate": -1.0,
                    "seed": seed,
                    "pass_rate": -1.0,
                },
            }
        )
    return rows


class TestStage4InstructionFollowing:
    """Stage 4 (D31): the IF domain is an additive ``--if_ratio`` share.

    The stage-3 domain keeps its absolute size, so the IF rows only grow the
    total; ``--exclude_path`` / ``--old_pool_path`` are the freshness levers.
    """

    def _stage3_build(self, tmp_path, scale=1):
        build = tmp_path / "built_stage3"
        build.mkdir()
        rows = []
        for branch, data_source, template, solvable, n in TestStage3Mix.STAGE3_CELL_SHAPES:
            rows.extend(make_cell_rows(branch, data_source, template, solvable, n * scale))
        schema.write_rows_parquet(rows, str(build / "halluc.parquet"))
        return str(build)

    @staticmethod
    def _stage1_pool(tmp_path, name="stage1", counts=(("code", 3000), ("math", 3000), ("logic", 1000), ("stem", 1000))):
        stage1 = tmp_path / name
        stage1.mkdir()
        rows = []
        for ability, n in counts:
            for i in range(n):
                row = schema.make_row(
                    data_source=SOURCE_GSMIC,
                    question=f"{name} {ability} question {i}?",
                    ground_truth=schema.ground_truth_json(
                        schema.ground_truth_payload(solvable=True, answer="42")
                    ),
                    template=TEMPLATE_B,
                    branch=BRANCH_SOLVABLE_NUMERIC,
                    extra_info={"task_id": f"{name}-{ability}-{i}", "index": i, "seed": 0},
                )
                row["ability"] = ability
                rows.append(row)
        schema.write_rows_parquet(rows, str(stage1 / "train.parquet"))
        return str(stage1), rows

    def test_if_train_target_matches_the_share(self):
        # 15% of the final train set means IF = 0.15 / 0.85 of the non-IF part.
        assert mix_halluc.if_train_target(0.15, 33250) == 5868
        assert mix_halluc.if_train_target(0.0, 33250) == 0

    def test_if_ratio_requires_if_path(self, tmp_path):
        build = self._stage3_build(tmp_path)
        with pytest.raises(ValueError, match="--if_path"):
            mix_halluc.main([
                "--stage3_mix",
                "--halluc_dir", build,
                "--output_dir", str(tmp_path / "out"),
                "--halluc_total", "400",
                "--if_ratio", "0.15",
            ])

    def test_if_domain_is_injected_at_the_requested_share(self, tmp_path):
        build = self._stage3_build(tmp_path)
        stage1, stage1_rows = self._stage1_pool(tmp_path)
        if_pool = tmp_path / "if"
        if_pool.mkdir()
        schema.write_rows_parquet(make_if_rows(2000), str(if_pool / "if.parquet"))

        out = tmp_path / "stage4"
        assert mix_halluc.main([
            "--stage3_mix",
            "--halluc_dir", build,
            "--stage1_path", stage1,
            "--if_path", str(if_pool),
            "--if_ratio", "0.15",
            "--output_dir", str(out),
            "--halluc_total", "2000",
            "--val_size", "60",
            "--seed", "13",
        ]) == 0

        train = mix_halluc.read_parquet_rows(os.path.join(str(out), "train.parquet"))
        val = mix_halluc.read_parquet_rows(os.path.join(str(out), "val.parquet"))
        stats = json.load(open(os.path.join(str(out), "mix_stats.json")))

        non_if = sum(1 for row in train if row["ability"] != "if")
        if_train = [row for row in train if row["ability"] == "if"]
        # The IF rows are additive: the non-IF part is exactly the stage-3 mix
        # (halluc train + old replay) and the IF count is ratio / (1 - ratio).
        assert len(if_train) == mix_halluc.if_train_target(0.15, non_if)
        assert stats["achieved_ratios"]["if_share_of_train"] == pytest.approx(0.15, abs=0.003)
        assert stats["config"]["if_ratio"] == pytest.approx(0.15)
        assert stats["if_rows_in_train"] == len(if_train)
        # Val carries IF rows too, carved from the pool before the train draw.
        if_val = [row for row in val if row["ability"] == "if"]
        assert if_val
        assert stats["achieved_ratios"]["if_share_of_val"] > 0
        # No prompt may straddle train/val.
        train_prompts = {mix_halluc._question_key(row) for row in train}
        val_prompts = {mix_halluc._question_key(row) for row in val}
        assert not train_prompts & val_prompts
        # The old domain still holds its share of the non-IF part.
        assert stats["achieved_ratios"]["old_domain_share_of_train"] == pytest.approx(0.6 * 0.85, abs=0.01)
        # Old-domain replay came from the stage-1 pool, not from the IF pool.
        old_ids = {row["extra_info"]["task_id"] for row in train if row["ability"] != "if"}
        assert old_ids & {row["extra_info"]["task_id"] for row in stage1_rows}

    def test_if_prompt_duplicating_the_reasoning_domains_is_dropped(self, tmp_path):
        build = self._stage3_build(tmp_path)
        stage1, _ = self._stage1_pool(tmp_path)
        base_args = [
            "--stage3_mix",
            "--halluc_dir", build,
            "--stage1_path", stage1,
            "--if_ratio", "0.15",
            "--halluc_total", "400",
            "--val_size", "0",
            "--if_val_size", "0",
            "--seed", "17",
        ]

        # First run: learn which hallucination question actually made it into the
        # train artifact (the fill samples a quota out of a much larger pool).
        first_pool = tmp_path / "if_first"
        first_pool.mkdir()
        schema.write_rows_parquet(make_if_rows(300), str(first_pool / "if.parquet"))
        first_out = tmp_path / "stage4_first"
        assert mix_halluc.main([*base_args, "--if_path", str(first_pool), "--output_dir", str(first_out)]) == 0
        train = mix_halluc.read_parquet_rows(os.path.join(str(first_out), "train.parquet"))
        halluc = next(row for row in train if str(row["data_source"]).startswith("halluc_"))
        duplicate = schema.question_of(halluc)

        # Second run: an IF row re-uses that question verbatim and must be dropped
        # so the same request is not trained twice in one artifact.
        if_rows = make_if_rows(300) + [
            {
                "data_source": "if_nemotron",
                "prompt": [{"role": "user", "content": duplicate}],
                "ability": "if",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps({"constraints": [{"id": "punctuation:no_comma"}]}),
                },
                "extra_info": {"task_id": "if-duplicate", "index": 999, "domain": "if", "source": "wildchat"},
            }
        ]
        if_pool = tmp_path / "if"
        if_pool.mkdir()
        schema.write_rows_parquet(if_rows, str(if_pool / "if.parquet"))

        out = tmp_path / "stage4"
        assert mix_halluc.main([*base_args, "--if_path", str(if_pool), "--output_dir", str(out)]) == 0
        train = mix_halluc.read_parquet_rows(os.path.join(str(out), "train.parquet"))
        assert all(row["extra_info"]["task_id"] != "if-duplicate" for row in train)
        stats = json.load(open(os.path.join(str(out), "mix_stats.json")))
        assert stats["if_domain"]["dropped_duplicate_of_reasoning_domains"] == 1

    def test_exclude_path_removes_previously_trained_rows(self, tmp_path):
        build = self._stage3_build(tmp_path)
        stage1, stage1_rows = self._stage1_pool(tmp_path, counts=(("code", 600), ("math", 600), ("logic", 300), ("stem", 300)))
        seen_dir = tmp_path / "stage3_seen"
        seen_dir.mkdir()
        seen = stage1_rows[:900]
        schema.write_rows_parquet(seen, str(seen_dir / "train.parquet"))
        seen_ids = {row["extra_info"]["task_id"] for row in seen}

        out = tmp_path / "stage4"
        assert mix_halluc.main([
            "--stage3_mix",
            "--halluc_dir", build,
            "--stage1_path", stage1,
            "--exclude_path", str(seen_dir),
            "--output_dir", str(out),
            "--halluc_total", "400",
            "--val_size", "20",
            "--seed", "19",
        ]) == 0
        train = mix_halluc.read_parquet_rows(os.path.join(str(out), "train.parquet"))
        train_ids = {row["extra_info"]["task_id"] for row in train}
        # Every old-domain row must come from the 900-row unseen tail.
        assert not train_ids & seen_ids
        stats = json.load(open(os.path.join(str(out), "mix_stats.json")))
        assert stats["old_domain_rows_excluded"] == 900

    def test_read_old_pool_skips_val_parquet(self, tmp_path):
        pool = tmp_path / "math_pool"
        pool.mkdir()
        schema.write_rows_parquet(
            make_cell_rows(BRANCH_SOLVABLE_NUMERIC, SOURCE_GSMIC, TEMPLATE_B, True, 5),
            str(pool / "train_math.parquet"),
        )
        schema.write_rows_parquet(
            make_cell_rows(BRANCH_SOLVABLE_NUMERIC, SOURCE_GSMIC, TEMPLATE_B, True, 3, seed=1),
            str(pool / "val.parquet"),
        )
        rows = mix_halluc.read_old_pool(str(pool))
        assert len(rows) == 5


class TestStage4BbhGap:
    """Stage 4 (D32): the BBH-gap domain is the second additive share."""

    SOURCES = (
        "bbh_dyck",
        "bbh_geometric_shapes",
        "bbh_disambiguation_qa",
        "bbh_causal_judgement",
        "bbh_movie_recommendation",
    )
    TASKS = {
        "bbh_dyck": "dyck_languages",
        "bbh_geometric_shapes": "geometric_shapes",
        "bbh_disambiguation_qa": "disambiguation_qa",
        "bbh_causal_judgement": "causal_judgement",
        "bbh_movie_recommendation": "movie_recommendation",
    }

    def _bbh_rows(self, n_per_source=600, seed=0):
        rows = []
        for source in self.SOURCES:
            for i in range(n_per_source):
                info = dict(schema.EXTRA_INFO_DEFAULTS)
                info.update(
                    {
                        "split": "train",
                        "index": i,
                        "task_id": f"bbh-{source}-{i}",
                        "domain": "bbh",
                        "source": f"BBH-gap {source}",
                        "difficulty": "",
                        "prior_solve_rate": -1.0,
                        "seed": seed,
                        "bbh_task": self.TASKS[source],
                    }
                )
                rows.append(
                    {
                        "data_source": source,
                        "prompt": [{"role": "user", "content": f"{source} question {i}?"}],
                        "ability": "bbh",
                        "reward_model": {
                            "style": "rule",
                            "ground_truth": json.dumps({"task": self.TASKS[source], "answer": "A"}),
                        },
                        "extra_info": info,
                    }
                )
        return rows

    def test_additive_target_matches_both_shares(self):
        # 15% IF + 10% BBH over a non-additive part N.
        assert mix_halluc.additive_target(0.15, 0.10, 33250) == 6650
        assert mix_halluc.additive_target(0.10, 0.15, 33250) == 4433
        assert mix_halluc.additive_target(0.0, 0.10, 33250) == 0

    def test_bbh_ratio_requires_bbh_path(self, tmp_path):
        build = TestStage4InstructionFollowing()._stage3_build(tmp_path)
        with pytest.raises(ValueError, match="--bbh_path"):
            mix_halluc.main([
                "--stage3_mix",
                "--halluc_dir", build,
                "--output_dir", str(tmp_path / "out"),
                "--halluc_total", "400",
                "--bbh_ratio", "0.10",
            ])

    def test_parse_bbh_weights_accepts_short_names(self):
        weights = mix_halluc.parse_bbh_weights(
            "dyck=0.3,geometric=0.2,disambiguation=0.2,causal=0.15,movie=0.15"
        )
        assert sum(weights.values()) == pytest.approx(1.0)
        assert weights["bbh_dyck"] == pytest.approx(0.3)
        with pytest.raises(ValueError, match="unknown --bbh_weights"):
            mix_halluc.parse_bbh_weights("nope=1.0")

    def test_apportion_with_supply_caps_a_short_source(self):
        allocation = mix_halluc._apportion_with_supply(
            {"a": 0.5, "b": 0.5}, 100, {"a": 10, "b": 100}
        )
        assert allocation == {"a": 10, "b": 90}

    def test_bbh_domain_is_injected_at_the_requested_share(self, tmp_path):
        helper = TestStage4InstructionFollowing()
        build = helper._stage3_build(tmp_path)
        stage1, _ = helper._stage1_pool(tmp_path)

        if_pool = tmp_path / "if"
        if_pool.mkdir()
        schema.write_rows_parquet(make_if_rows(3000), str(if_pool / "if.parquet"))
        bbh_pool = tmp_path / "bbh"
        bbh_pool.mkdir()
        schema.write_rows_parquet(self._bbh_rows(), str(bbh_pool / "bbh.parquet"))

        out = tmp_path / "stage4"
        assert mix_halluc.main([
            "--stage3_mix",
            "--halluc_dir", build,
            "--stage1_path", stage1,
            "--if_path", str(if_pool),
            "--if_ratio", "0.15",
            "--bbh_path", str(bbh_pool),
            "--bbh_ratio", "0.10",
            "--output_dir", str(out),
            "--halluc_total", "2000",
            "--val_size", "60",
            "--seed", "23",
        ]) == 0

        train = mix_halluc.read_parquet_rows(os.path.join(str(out), "train.parquet"))
        val = mix_halluc.read_parquet_rows(os.path.join(str(out), "val.parquet"))
        stats = json.load(open(os.path.join(str(out), "mix_stats.json")))

        non_add = [row for row in train if row["ability"] not in ("if", "bbh")]
        if_train = [row for row in train if row["ability"] == "if"]
        bbh_train = [row for row in train if row["ability"] == "bbh"]
        assert len(if_train) == mix_halluc.additive_target(0.15, 0.10, len(non_add))
        assert len(bbh_train) == mix_halluc.additive_target(0.10, 0.15, len(non_add))
        assert stats["achieved_ratios"]["if_share_of_train"] == pytest.approx(0.15, abs=0.005)
        assert stats["achieved_ratios"]["bbh_share_of_train"] == pytest.approx(0.10, abs=0.005)
        assert stats["config"]["bbh_ratio"] == pytest.approx(0.10)
        assert stats["bbh_rows_in_train"] == len(bbh_train)
        # The per-source weights survive a pool that is larger than the quota.
        by_source = Counter(row["data_source"] for row in bbh_train)
        assert set(by_source) == set(self.SOURCES)
        assert by_source["bbh_dyck"] > by_source["bbh_movie_recommendation"]
        # Both additive domains contribute val rows, and no prompt straddles.
        assert any(row["ability"] == "bbh" for row in val)
        assert any(row["ability"] == "if" for row in val)
        train_prompts = {mix_halluc._question_key(row) for row in train}
        val_prompts = {mix_halluc._question_key(row) for row in val}
        assert not train_prompts & val_prompts
        # The stage-3 cells keep their absolute size: the non-additive part is
        # independent of the two additive ratios.
        assert stats["achieved_ratios"]["old_domain_share_of_train"] == pytest.approx(
            0.6 * (1 - 0.15 - 0.10), abs=0.01
        )


class TestStage4AbilityMatch:
    """Stage 4 follow-up: keep the stage-3 ability mix in the reasoning part.

    Hard math counts inside the math budget, so it replaces old-domain math
    replay one-for-one; the freed replay slots go to the abilities the new
    domains under-supply, and the artifact total is unchanged.
    """

    def test_stage3_profile_quotas(self):
        quotas, report = mix_halluc.old_domain_ability_quotas(
            mix_halluc.STAGE3_ABILITY_PROFILE,
            reasoning_rows=33250,
            halluc_counts={"math": 7980, "logic": 2660, "commonsense": 2660},
            fixed_counts={"math": 6650},
            n_old=13300,
        )
        assert quotas == {
            "math": 532,
            "code": 7980,
            "logic": 2394,
            "commonsense": 0,
            "stem": 2394,
        }
        assert sum(quotas.values()) == 13300
        assert report["targets"] == mix_halluc.STAGE3_ABILITY_PROFILE

    def test_overflow_is_absorbed_without_changing_the_total(self):
        # A math target below halluc + hard math clamps to zero; the residue is
        # taken from the largest quota so the replay total stays n_old.
        quotas, _ = mix_halluc.old_domain_ability_quotas(
            {"math": 1000, "code": 1000},
            reasoning_rows=2500,
            halluc_counts={"math": 1200},
            fixed_counts={"math": 600},
            n_old=700,
        )
        assert quotas["math"] == 0
        assert sum(quotas.values()) == 700

    def test_read_ability_reference(self, tmp_path):
        path = tmp_path / "stage3_stats.json"
        path.write_text(
            json.dumps({"train_by_ability": {"math": 10, "code": 5, "logic": 3, "empty": 0}}),
            encoding="utf-8",
        )
        assert mix_halluc.read_ability_reference(str(path)) == {
            "math": 10,
            "code": 5,
            "logic": 3,
        }
        path.write_text(json.dumps({"total_train": 1}), encoding="utf-8")
        with pytest.raises(ValueError, match="train_by_ability"):
            mix_halluc.read_ability_reference(str(path))

    def test_sample_old_domain_by_ability_reports_shortfall(self):
        rows = (
            [{"ability": "code"}] * 5
            + [{"ability": "math"}] * 1
            + [{"ability": "logic"}] * 0
        )
        chosen, report = mix_halluc.sample_old_domain_by_ability(
            rows, {"code": 3, "math": 2, "logic": 1}, random.Random(0)
        )
        assert report["actual_by_ability"] == {"code": 3, "math": 1, "logic": 0}
        assert report["shortfall_by_ability"] == {"code": 0, "math": 1, "logic": 1}
        assert len(chosen) == 4

    def test_ability_match_requires_stage3_mix(self, tmp_path):
        build = TestStage4InstructionFollowing()._stage3_build(tmp_path)
        with pytest.raises(ValueError, match="require --stage3_mix"):
            mix_halluc.main([
                "--halluc_dir", build,
                "--output_dir", str(tmp_path / "out"),
                "--halluc_total", "400",
                "--match_stage3_abilities",
            ])

    def test_ability_matched_domain_keeps_the_total_and_the_mix(self, tmp_path):
        helper = TestStage4InstructionFollowing()
        build = helper._stage3_build(tmp_path)
        stage1, _ = helper._stage1_pool(tmp_path)

        if_pool = tmp_path / "if"
        if_pool.mkdir()
        schema.write_rows_parquet(make_if_rows(3000), str(if_pool / "if.parquet"))
        bbh_pool = tmp_path / "bbh"
        bbh_pool.mkdir()
        schema.write_rows_parquet(TestStage4BbhGap()._bbh_rows(), str(bbh_pool / "bbh.parquet"))
        hard_pool = tmp_path / "hard_math"
        hard_pool.mkdir()
        schema.write_rows_parquet(
            TestStage4HardMath()._hard_math_rows(), str(hard_pool / "hard_math.parquet")
        )

        # Small-scale stage-3 reference: same 45.6/24/15.2/8/7.2 shape as the
        # real artifact, summing to reasoning = halluc_train + old + hard math.
        reference = {
            "math": 2200,
            "code": 1160,
            "logic": 740,
            "commonsense": 388,
            "stem": 362,
        }
        assert sum(reference.values()) == 4850
        reference_path = tmp_path / "stage3_stats.json"
        reference_path.write_text(
            json.dumps({"train_by_ability": reference}), encoding="utf-8"
        )

        out = tmp_path / "stage4"
        assert mix_halluc.main([
            "--stage3_mix",
            "--halluc_dir", build,
            "--stage1_path", stage1,
            "--if_path", str(if_pool),
            "--if_ratio", "0.15",
            "--bbh_path", str(bbh_pool),
            "--bbh_ratio", "0.10",
            "--hard_math_path", str(hard_pool),
            "--hard_math_ratio", "0.15",
            "--old_domain_ratio", "0.5",
            "--ability_reference_path", str(reference_path),
            "--output_dir", str(out),
            "--halluc_total", "2000",
            "--val_size", "60",
            "--seed", "37",
        ]) == 0

        train = mix_halluc.read_parquet_rows(os.path.join(str(out), "train.parquet"))
        stats = json.load(open(os.path.join(str(out), "mix_stats.json")))
        # Total is the same as the unmatched build at this scale.
        assert stats["total_train"] == 6467
        assert stats["ability_match"]["targets"] == reference
        achieved = stats["achieved_ratios"]["ability_share_of_reasoning"]
        for ability, target in reference.items():
            assert achieved.get(ability, 0.0) == pytest.approx(target / 4850, abs=0.01)
        # Old domain total and share are unchanged; only its internal mix moved.
        assert stats["harvested_old_domain"]["requested_rows"] == 1940
        assert stats["achieved_ratios"]["old_domain_share_of_train"] == pytest.approx(0.30, abs=0.01)
        # Math is supplied mostly by hard math + the halluc domain, not replay.
        old_actual = stats["harvested_old_domain"]["actual_by_ability"]
        assert old_actual["code"] == 1160
        assert old_actual["math"] == 66
        assert old_actual["logic"] == 352
        assert old_actual["stem"] == 362


class TestStage4HardMath:
    """Stage 4 (D34): the hard-math domain is the third additive share.

    The stage-4 data plan cuts the old-domain replay to 30% of the final train
    set and fills the freed share with AIME-adjacent problems; the total volume
    stays at the previous stage-4 scale.
    """

    SOURCES = ("math_hard_ace_reason", "math_hard_deepscaler", "math_hard_deepmath")

    def _hard_math_rows(self, n_per_source=600, seed=0):
        rows = []
        for source in self.SOURCES:
            for i in range(n_per_source):
                info = dict(schema.EXTRA_INFO_DEFAULTS)
                info.update(
                    {
                        "split": "train",
                        "index": i,
                        "task_id": f"{source}-{i}",
                        "domain": "math",
                        "source": source,
                        "difficulty": "6",
                        "prior_solve_rate": -1.0,
                        "seed": seed,
                        "pass_rate": 0.5,
                    }
                )
                rows.append(
                    {
                        "data_source": source,
                        "prompt": [
                            {
                                "role": "user",
                                "content": f"{source} question {i}? Please reason step by step, and put your final answer within \\boxed{{}}.",
                            }
                        ],
                        "ability": "math",
                        "reward_model": {"style": "rule", "ground_truth": str(i)},
                        "extra_info": info,
                    }
                )
        return rows

    def test_additive_targets_keep_all_three_shares(self):
        # 15% IF + 10% BBH + 15% hard math over N = 26,600 (halluc 13,300 + old 13,300).
        targets = mix_halluc.additive_targets(
            {"if": 0.15, "bbh": 0.10, "hard_math": 0.15}, 26600
        )
        assert targets == {"if": 6650, "bbh": 4433, "hard_math": 6650}
        assert sum(targets.values()) + 26600 == 44333
        assert mix_halluc.additive_targets({"if": 0.0}, 100) == {"if": 0}

    def test_additive_targets_reject_an_impossible_total(self):
        with pytest.raises(ValueError, match="no room"):
            mix_halluc.additive_targets({"a": 0.6, "b": 0.4}, 100)

    def test_parse_hard_math_weights_accepts_short_names(self):
        weights = mix_halluc.parse_hard_math_weights(
            "ace_reason=0.4,deepscaler=0.35,deepmath=0.25"
        )
        assert sum(weights.values()) == pytest.approx(1.0)
        assert weights["math_hard_deepmath"] == pytest.approx(0.25)
        with pytest.raises(ValueError, match="unknown --hard_math_weights"):
            mix_halluc.parse_hard_math_weights("nope=1.0")

    def test_hard_math_ratio_requires_path(self, tmp_path):
        build = TestStage4InstructionFollowing()._stage3_build(tmp_path)
        with pytest.raises(ValueError, match="--hard_math_path"):
            mix_halluc.main([
                "--stage3_mix",
                "--halluc_dir", build,
                "--output_dir", str(tmp_path / "out"),
                "--halluc_total", "400",
                "--hard_math_ratio", "0.15",
            ])

    def test_hard_math_domain_is_injected_at_the_requested_share(self, tmp_path):
        helper = TestStage4InstructionFollowing()
        build = helper._stage3_build(tmp_path)
        stage1, _ = helper._stage1_pool(tmp_path, counts=(("code", 3000), ("math", 3000), ("logic", 1000), ("stem", 1000)))

        if_pool = tmp_path / "if"
        if_pool.mkdir()
        schema.write_rows_parquet(make_if_rows(3000), str(if_pool / "if.parquet"))
        bbh_pool = tmp_path / "bbh"
        bbh_pool.mkdir()
        schema.write_rows_parquet(
            TestStage4BbhGap()._bbh_rows(), str(bbh_pool / "bbh.parquet")
        )
        hard_pool = tmp_path / "hard_math"
        hard_pool.mkdir()
        schema.write_rows_parquet(self._hard_math_rows(), str(hard_pool / "hard_math.parquet"))

        out = tmp_path / "stage4"
        assert mix_halluc.main([
            "--stage3_mix",
            "--halluc_dir", build,
            "--stage1_path", stage1,
            "--if_path", str(if_pool),
            "--if_ratio", "0.15",
            "--bbh_path", str(bbh_pool),
            "--bbh_ratio", "0.10",
            "--hard_math_path", str(hard_pool),
            "--hard_math_ratio", "0.15",
            "--old_domain_ratio", "0.5",
            "--output_dir", str(out),
            "--halluc_total", "2000",
            "--val_size", "60",
            "--seed", "29",
        ]) == 0

        train = mix_halluc.read_parquet_rows(os.path.join(str(out), "train.parquet"))
        val = mix_halluc.read_parquet_rows(os.path.join(str(out), "val.parquet"))
        stats = json.load(open(os.path.join(str(out), "mix_stats.json")))

        hard_sources = set(self.SOURCES)
        non_add = [
            row
            for row in train
            if row["data_source"] not in hard_sources and row["ability"] not in ("if", "bbh")
        ]
        hard_train = [row for row in train if row["data_source"] in hard_sources]
        targets = mix_halluc.additive_targets(
            {"if": 0.15, "bbh": 0.10, "hard_math": 0.15}, len(non_add)
        )
        assert len(hard_train) == targets["hard_math"]
        assert stats["achieved_ratios"]["hard_math_share_of_train"] == pytest.approx(0.15, abs=0.005)
        assert stats["config"]["hard_math_ratio"] == pytest.approx(0.15)
        assert stats["hard_math_rows_in_train"] == len(hard_train)
        # The per-source weights survive a pool larger than the quota.
        by_source = Counter(row["data_source"] for row in hard_train)
        assert set(by_source) == set(self.SOURCES)
        assert by_source["math_hard_ace_reason"] > by_source["math_hard_deepmath"]
        # The measured pass rate survives the mix (it is the difficulty signal).
        assert all(row["extra_info"]["pass_rate"] == pytest.approx(0.5) for row in hard_train)
        # Every additive domain contributes val rows, and no prompt straddles.
        assert any(row["data_source"] in self.SOURCES for row in val)
        train_prompts = {mix_halluc._question_key(row) for row in train}
        val_prompts = {mix_halluc._question_key(row) for row in val}
        assert not train_prompts & val_prompts
        # With old_domain_ratio 0.5 the old domain is 30% of the final set
        # (0.5 * (1 - 0.15 - 0.10 - 0.15)), i.e. the stage-4 data-plan target.
        assert stats["achieved_ratios"]["old_domain_share_of_train"] == pytest.approx(
            0.5 * (1 - 0.15 - 0.10 - 0.15), abs=0.01
        )

    def test_hard_math_prompt_duplicating_the_reasoning_domains_is_dropped(self, tmp_path):
        helper = TestStage4InstructionFollowing()
        build = helper._stage3_build(tmp_path)
        stage1, _ = helper._stage1_pool(tmp_path, counts=(("code", 1200), ("math", 1200), ("logic", 400), ("stem", 400)))
        base_args = [
            "--stage3_mix",
            "--halluc_dir", build,
            "--stage1_path", stage1,
            "--hard_math_ratio", "0.15",
            "--halluc_total", "400",
            "--val_size", "0",
            "--hard_math_val_size", "0",
            "--seed", "31",
        ]

        first_pool = tmp_path / "hard_first"
        first_pool.mkdir()
        schema.write_rows_parquet(self._hard_math_rows(300), str(first_pool / "hard_math.parquet"))
        first_out = tmp_path / "stage4_first"
        assert mix_halluc.main([*base_args, "--hard_math_path", str(first_pool), "--output_dir", str(first_out)]) == 0
        train = mix_halluc.read_parquet_rows(os.path.join(str(first_out), "train.parquet"))
        halluc = next(row for row in train if str(row["data_source"]).startswith("halluc_"))
        duplicate = schema.question_of(halluc)

        rows = self._hard_math_rows(300) + [
            {
                "data_source": "math_hard_deepmath",
                "prompt": [{"role": "user", "content": duplicate}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": "42"},
                "extra_info": {"task_id": "hard-duplicate", "index": 999, "domain": "math"},
            }
        ]
        hard_pool = tmp_path / "hard_math"
        hard_pool.mkdir()
        schema.write_rows_parquet(rows, str(hard_pool / "hard_math.parquet"))

        out = tmp_path / "stage4"
        assert mix_halluc.main([*base_args, "--hard_math_path", str(hard_pool), "--output_dir", str(out)]) == 0
        train = mix_halluc.read_parquet_rows(os.path.join(str(out), "train.parquet"))
        assert all(row["extra_info"]["task_id"] != "hard-duplicate" for row in train)
        stats = json.load(open(os.path.join(str(out), "mix_stats.json")))
        assert stats["hard_math_domain"]["dropped_duplicate_of_reasoning_domains"] >= 1

    def test_hard_math_pool_with_an_unweighted_source_is_rejected(self, tmp_path):
        helper = TestStage4InstructionFollowing()
        build = helper._stage3_build(tmp_path)
        pool = tmp_path / "hard_math"
        pool.mkdir()
        rows = self._hard_math_rows(10)
        # DAPO is a supported adapter source but has no default mixer weight;
        # silently dropping it would under-fill the domain, so the mixer refuses.
        rows.append(
            {
                "data_source": "math_hard_dapo",
                "prompt": [{"role": "user", "content": "DAPO problem? Please reason step by step."}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": "1"},
                "extra_info": {"task_id": "hard-dapo-0", "index": 999, "domain": "math"},
            }
        )
        schema.write_rows_parquet(rows, str(pool / "hard_math.parquet"))
        with pytest.raises(ValueError, match="weights do not cover"):
            mix_halluc.main([
                "--stage3_mix",
                "--halluc_dir", build,
                "--hard_math_path", str(pool),
                "--hard_math_ratio", "0.15",
                "--output_dir", str(tmp_path / "out"),
                "--halluc_total", "400",
                "--val_size", "0",
                "--hard_math_val_size", "0",
            ])
