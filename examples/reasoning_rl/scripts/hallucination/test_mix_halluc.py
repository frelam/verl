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
