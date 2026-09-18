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
behaviour: how table B's quotas rescale under the three knobs, what happens when a
cell's pool is short, and that an unbalanced template aborts the write rather than
producing a subtly leaky artifact.
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
    BRANCH_SOLVABLE_ROLES,
    BRANCH_UNSOLVABLE_BARE,
    BRANCH_UNSOLVABLE_DIAG,
    SOURCE_CREPE,
    SOURCE_FALSEQA,
    SOURCE_GSMIC,
    SOURCE_KK,
    SOURCE_KUQ,
    SOURCE_MAIN,
    SOURCE_MIP,
    SOURCE_SUM,
    SOURCE_TREECUT,
    SOURCE_UMWP,
    TEMPLATE_A,
    TEMPLATE_B,
    TEMPLATE_B_JUDGE,
)

# (branch, data_source, template, solvable, n_rows) -- the shape of each cell's pool.
# There is no ``(solvable_numeric, KK)`` cell: K&K answers with a role sequence, so
# the 400 synthesised-distractor K&K rows join the 2,000 K&K role rows in
# ``(solvable_roles, KK)`` -- hence 2,400 there.
CELL_SHAPES = [
    (BRANCH_SOLVABLE_NUMERIC, SOURCE_GSMIC, TEMPLATE_B, True, 2400),
    (BRANCH_SOLVABLE_NUMERIC, SOURCE_SUM, TEMPLATE_B, True, 1200),
    (BRANCH_SOLVABLE_NUMERIC, SOURCE_UMWP, TEMPLATE_B, True, 700),
    (BRANCH_SOLVABLE_NUMERIC, SOURCE_MAIN, TEMPLATE_B, True, 500),
    (BRANCH_SOLVABLE_ROLES, SOURCE_KK, TEMPLATE_B, True, 2400),
    (BRANCH_SOLVABLE_JUDGE, SOURCE_UMWP, TEMPLATE_A, True, 700),
    (BRANCH_SOLVABLE_JUDGE, SOURCE_SUM, TEMPLATE_A, True, 450),
    (BRANCH_SOLVABLE_JUDGE, SOURCE_CREPE, TEMPLATE_B_JUDGE, True, 600),
    (BRANCH_SOLVABLE_JUDGE, SOURCE_KUQ, TEMPLATE_B_JUDGE, True, 300),
    (BRANCH_UNSOLVABLE_DIAG, SOURCE_SUM, TEMPLATE_A, False, 5600),
    (BRANCH_UNSOLVABLE_DIAG, SOURCE_UMWP, TEMPLATE_A, False, 700),  # short pool on purpose
    (BRANCH_UNSOLVABLE_DIAG, SOURCE_FALSEQA, TEMPLATE_A, False, 700),
    (BRANCH_UNSOLVABLE_BARE, SOURCE_SUM, TEMPLATE_B, False, 2200),
    (BRANCH_UNSOLVABLE_BARE, SOURCE_TREECUT, TEMPLATE_B, False, 1200),
    (BRANCH_UNSOLVABLE_BARE, SOURCE_UMWP, TEMPLATE_B, False, 1000),
    (BRANCH_UNSOLVABLE_BARE, SOURCE_CREPE, TEMPLATE_B_JUDGE, False, 500),
    (BRANCH_UNSOLVABLE_BARE, SOURCE_MIP, TEMPLATE_B, False, 300),
    (BRANCH_UNSOLVABLE_BARE, SOURCE_KUQ, TEMPLATE_B_JUDGE, False, 300),
]


def make_cell_rows(branch, data_source, template, solvable, n, seed=0):
    """Synthetic rows in the shape an adapter emits for this cell."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        options = None
        extra = {"task_id": f"{data_source}-{branch}-{i}", "index": i, "seed": seed}
        gt_kwargs = {"solvable": solvable}
        if solvable:
            gt_kwargs["answer"] = "42"
            if branch == BRANCH_SOLVABLE_ROLES:
                gt_kwargs["role_words"] = ["knight", "knave"]
            if template in (TEMPLATE_A, TEMPLATE_B_JUDGE):
                gt_kwargs["judgment_only"] = True
            if template == TEMPLATE_A:
                options = [{"id": c, "text": f"span {c} {i}"} for c in "ABC"]
        else:
            if template == TEMPLATE_A:
                options = [{"id": c, "text": f"span {c} {i}"} for c in "ABC"]
                gt_kwargs["correct_option_id"] = rng.choice("ABC")
                gt_kwargs["has_diagnosis_label"] = True
                gt_kwargs["perturbation_type"] = "missing_condition"
        rows.append(
            schema.make_row(
                data_source=data_source,
                question=f"Question {data_source} {branch} {i}?",
                ground_truth=schema.ground_truth_json(schema.ground_truth_payload(**gt_kwargs)),
                template=template,
                branch=branch,
                extra_info=extra,
                options=options,
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


class TestQuotaScaling:
    def test_defaults_reproduce_table_b(self):
        quota = mix_halluc.scaled_quota(20000, 0.6, 0.6)
        assert sum(quota.values()) == 20000
        solvable = sum(v for k, v in quota.items() if k[0] in mix_halluc.SOLVABLE_BRANCHES)
        unsolvable = sum(v for k, v in quota.items() if k[0] in mix_halluc.UNSOLVABLE_BRANCHES)
        assert solvable == 8000
        assert unsolvable == 12000
        four = sum(v for k, v in quota.items() if k[0] == BRANCH_UNSOLVABLE_DIAG)
        three = sum(v for k, v in quota.items() if k[0] == BRANCH_UNSOLVABLE_BARE)
        assert (four, three) == (7200, 4800)
        assert quota[(BRANCH_SOLVABLE_NUMERIC, SOURCE_GSMIC)] == 2000
        assert quota[(BRANCH_UNSOLVABLE_DIAG, SOURCE_SUM)] == 5094
        # The K&K branch deviation: 2,000 role rows + the 400 synthesised-distractor
        # ones, which cannot carry `solvable_numeric` (a K&K gold is a role sequence).
        assert quota[(BRANCH_SOLVABLE_ROLES, SOURCE_KK)] == 2400
        assert (BRANCH_SOLVABLE_NUMERIC, SOURCE_KK) not in quota

    @pytest.mark.parametrize(
        "total,unsolvable_ratio,four_tier_ratio",
        [(20000, 0.6, 0.6), (1000, 0.5, 0.5), (333, 0.6, 0.6), (20000, 0.7, 0.3), (77, 0.25, 0.75)],
    )
    def test_groups_always_sum_exactly(self, total, unsolvable_ratio, four_tier_ratio):
        quota = mix_halluc.scaled_quota(total, unsolvable_ratio, four_tier_ratio)
        assert sum(quota.values()) == total
        unsolvable = sum(v for k, v in quota.items() if k[0] in mix_halluc.UNSOLVABLE_BRANCHES)
        assert unsolvable == round(total * unsolvable_ratio)
        four = sum(v for k, v in quota.items() if k[0] == BRANCH_UNSOLVABLE_DIAG)
        assert four == round(unsolvable * four_tier_ratio)

    def test_rejects_out_of_range_ratios(self):
        with pytest.raises(ValueError, match="halluc_unsolvable_ratio"):
            mix_halluc.scaled_quota(20000, 1.0, 0.6)
        with pytest.raises(ValueError, match="halluc_four_tier_ratio"):
            mix_halluc.scaled_quota(20000, 0.6, 0.0)

    def test_every_cell_is_represented(self):
        # A quota cell with no default entry silently loses its rows.
        assert set(mix_halluc.scaled_quota(20000, 0.6, 0.6)) == set(mix_halluc.DEFAULT_QUOTA)


class TestIndexPool:
    def test_buckets_by_branch_and_source(self):
        rows = make_cell_rows(BRANCH_SOLVABLE_NUMERIC, SOURCE_GSMIC, TEMPLATE_B, True, 5)
        pool = mix_halluc.index_pool(rows)
        assert len(pool[(BRANCH_SOLVABLE_NUMERIC, SOURCE_GSMIC)]) == 5

    def test_rejects_solvable_flag_mismatch(self):
        rows = make_cell_rows(BRANCH_UNSOLVABLE_DIAG, SOURCE_SUM, TEMPLATE_A, False, 1)
        rows[0]["reward_model"]["ground_truth"] = schema.ground_truth_json(
            schema.ground_truth_payload(solvable=True, answer="1")
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
            + make_cell_rows(BRANCH_UNSOLVABLE_DIAG, SOURCE_SUM, TEMPLATE_A, False, 100)
        )
        quota = {
            (BRANCH_UNSOLVABLE_DIAG, SOURCE_UMWP): 50,
            (BRANCH_UNSOLVABLE_DIAG, SOURCE_SUM): 20,
        }
        rows, report = mix_halluc.fill_cells(pool, quota, random.Random(0))
        by_cell = {tuple(e["cell"]): e for e in report}
        short = by_cell[(BRANCH_UNSOLVABLE_DIAG, SOURCE_UMWP)]
        assert short["filled"] == 3 and short["shortfall"] == 47
        # SUM had 80 spare, so it absorbs the deficit rather than the mix shrinking.
        assert by_cell[(BRANCH_UNSOLVABLE_DIAG, SOURCE_SUM)]["redistributed"] == 47
        assert len(rows) == 70

    def test_deficit_does_not_cross_the_tier_boundary(self):
        # A four-tier cell runs short; the three-tier cells are full of surplus.
        # The tier knob must win over backfilling the total.
        pool = mix_halluc.index_pool(
            make_cell_rows(BRANCH_UNSOLVABLE_DIAG, SOURCE_UMWP, TEMPLATE_A, False, 2)
            + make_cell_rows(BRANCH_UNSOLVABLE_BARE, SOURCE_SUM, TEMPLATE_B, False, 500)
        )
        quota = {(BRANCH_UNSOLVABLE_DIAG, SOURCE_UMWP): 40, (BRANCH_UNSOLVABLE_BARE, SOURCE_SUM): 5}
        rows, report = mix_halluc.fill_cells(pool, quota, random.Random(3))
        assert len(rows) == 2 + 5
        by_cell = {tuple(e["cell"]): e for e in report}
        assert by_cell[(BRANCH_UNSOLVABLE_BARE, SOURCE_SUM)]["redistributed"] == 0

    def test_spill_across_tiers_preserves_the_total_when_asked(self):
        pool = mix_halluc.index_pool(
            make_cell_rows(BRANCH_UNSOLVABLE_DIAG, SOURCE_UMWP, TEMPLATE_A, False, 2)
            + make_cell_rows(BRANCH_UNSOLVABLE_BARE, SOURCE_SUM, TEMPLATE_B, False, 500)
        )
        quota = {(BRANCH_UNSOLVABLE_DIAG, SOURCE_UMWP): 40, (BRANCH_UNSOLVABLE_BARE, SOURCE_SUM): 5}
        rows, _ = mix_halluc.fill_cells(pool, quota, random.Random(3), spill_across_tiers=True)
        assert len(rows) == 40 + 5

    def test_four_tier_deficit_prefers_four_tier_surplus(self):
        pool = mix_halluc.index_pool(
            make_cell_rows(BRANCH_UNSOLVABLE_DIAG, SOURCE_UMWP, TEMPLATE_A, False, 2)
            + make_cell_rows(BRANCH_UNSOLVABLE_DIAG, SOURCE_SUM, TEMPLATE_A, False, 100)
            + make_cell_rows(BRANCH_UNSOLVABLE_BARE, SOURCE_TREECUT, TEMPLATE_B, False, 100)
        )
        quota = {
            (BRANCH_UNSOLVABLE_DIAG, SOURCE_UMWP): 30,
            (BRANCH_UNSOLVABLE_DIAG, SOURCE_SUM): 10,
            (BRANCH_UNSOLVABLE_BARE, SOURCE_TREECUT): 5,
        }
        rows, report = mix_halluc.fill_cells(pool, quota, random.Random(4))
        by_cell = {tuple(e["cell"]): e for e in report}
        # 28 rows flow to the other four-tier cell, never to the three-tier one.
        assert by_cell[(BRANCH_UNSOLVABLE_DIAG, SOURCE_SUM)]["redistributed"] == 28
        assert by_cell[(BRANCH_UNSOLVABLE_BARE, SOURCE_TREECUT)]["redistributed"] == 0
        assert len(rows) == 40 + 5

    def test_no_duplicate_rows_after_redistribution(self):
        pool = mix_halluc.index_pool(
            make_cell_rows(BRANCH_UNSOLVABLE_BARE, SOURCE_MIP, TEMPLATE_B, False, 5)
            + make_cell_rows(BRANCH_UNSOLVABLE_BARE, SOURCE_SUM, TEMPLATE_B, False, 50)
        )
        quota = {(BRANCH_UNSOLVABLE_BARE, SOURCE_MIP): 10, (BRANCH_UNSOLVABLE_BARE, SOURCE_SUM): 5}
        rows, _ = mix_halluc.fill_cells(pool, quota, random.Random(1))
        ids = [r["extra_info"]["task_id"] for r in rows]
        assert len(ids) == len(set(ids))

    def test_shortfall_stays_on_its_own_side(self):
        # A short SOLVABLE cell must not be backfilled from the unsolvable pool,
        # which would move the 60/40 split.
        pool = mix_halluc.index_pool(
            make_cell_rows(BRANCH_SOLVABLE_ROLES, SOURCE_KK, TEMPLATE_B, True, 2)
            + make_cell_rows(BRANCH_UNSOLVABLE_BARE, SOURCE_SUM, TEMPLATE_B, False, 50)
        )
        quota = {(BRANCH_SOLVABLE_ROLES, SOURCE_KK): 20, (BRANCH_UNSOLVABLE_BARE, SOURCE_SUM): 10}
        rows, report = mix_halluc.fill_cells(pool, quota, random.Random(2))
        # 2 solvable + 10 unsolvable, and nothing crossed the boundary to make up
        # the solvable deficit.
        assert len(rows) == 2 + 10
        branches = [r["extra_info"]["branch"] for r in rows]
        assert branches.count(BRANCH_SOLVABLE_ROLES) == 2
        assert branches.count(BRANCH_UNSOLVABLE_BARE) == 10
        assert not any(e["redistributed"] for e in report)


class TestTemplateBalance:
    def test_balanced_templates_pass(self):
        rows = make_cell_rows(BRANCH_SOLVABLE_NUMERIC, SOURCE_GSMIC, TEMPLATE_B, True, 2)
        rows += make_cell_rows(BRANCH_UNSOLVABLE_BARE, SOURCE_MIP, TEMPLATE_B, False, 2)
        assert mix_halluc.check_template_balance(rows) == {TEMPLATE_B: {"solvable": 2, "unsolvable": 2}}

    def test_one_sided_template_is_fatal(self):
        rows = make_cell_rows(BRANCH_SOLVABLE_NUMERIC, SOURCE_GSMIC, TEMPLATE_B, True, 2)
        with pytest.raises(ValueError, match="only one side"):
            mix_halluc.check_template_balance(rows)


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
        assert len(val) == 64
        stats = json.load(open(os.path.join(out, "mix_stats.json")))
        assert stats["total_train"] == 2000
        assert stats["halluc_target_vs_filled"]["target"] == 2000
        assert sum(stats["halluc_by_branch"].values()) == 2000
        # Indices are re-contiguous and val is labelled.
        assert [r["extra_info"]["index"] for r in train] == list(range(2000))
        assert all(r["extra_info"]["split"] == "val" for r in val)
        assert all(r["extra_info"]["pass_rate"] == -1.0 for r in train)

    def test_proportions_match_the_knobs(self, build_dir, tmp_path):
        out = self.run_mix(build_dir, tmp_path, halluc_total=4000)
        train = mix_halluc.read_parquet_rows(os.path.join(out, "train.parquet"))
        solvable = sum(json.loads(r["reward_model"]["ground_truth"])["solvable"] for r in train)
        assert solvable == pytest.approx(0.4 * len(train), abs=1)
        four = sum(1 for r in train if r["extra_info"]["branch"] == BRANCH_UNSOLVABLE_DIAG)
        unsolvable = len(train) - solvable
        assert four == pytest.approx(0.6 * unsolvable, abs=1)

    def test_every_train_row_validates(self, build_dir, tmp_path):
        out = self.run_mix(build_dir, tmp_path, halluc_total=500)
        train = mix_halluc.read_parquet_rows(os.path.join(out, "train.parquet"))
        schema.validate_rows(train)

    def test_short_cell_shows_up_in_stats(self, build_dir, tmp_path):
        # UMWP's four-tier pool is the deliberately short one in CELL_SHAPES.
        out = self.run_mix(build_dir, tmp_path, halluc_total=20000)
        stats = json.load(open(os.path.join(out, "mix_stats.json")))
        cell = next(
            e for e in stats["cells"] if e["cell"] == [BRANCH_UNSOLVABLE_DIAG, SOURCE_UMWP]
        )
        assert cell["shortfall"] > 0
        assert stats["halluc_target_vs_filled"]["filled"] <= 20000
