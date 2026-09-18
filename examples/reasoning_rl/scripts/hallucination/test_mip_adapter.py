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
"""Tests for :mod:`mip_adapter` -- every fixture is inline and tiny.

The rows below are copied verbatim from ``scratch/halluc_recon/mip_report.md``
section 3 and from the shapes it documents (the report's sample rows, the two
conventions it records in section 4.1, and the hazards in section 6).  Nothing
here reads ``/home/charles/data/reasoning_rl/halluc/raw``: each test writes its
own ``{gsm8k,svamp,math,formula}.json`` fixture files into ``tmp_path``, so the
suite runs with no raw data present at all and cannot drift with the download.

Coverage: one test per funnel stage, per emitted branch, per drop reason, both
gold certificates (gsm8k tiers T1/T2 and math ``\\boxed{}``), the placeholder
family, the limit/pair rule, determinism, and the D12 no-options invariant.
"""

from __future__ import annotations

import json

import mip_adapter as adapter
import pytest
import schema

# ---------------------------------------------------------------------------
# inline fixtures (verbatim from the recon report, section 3)
# ---------------------------------------------------------------------------

# recon section 3 "gsm8k.json[0]": one deleted clause ("He runs 60 meters each
# sprint."), one value (60), used by the source's own chain (9*60=540), and a
# gold the last annotation recomputes -> admitted on both branches.
GSM8K_PAIR = {
    "question": "James decides to run 3 sprints 3 times a week.  He runs 60 meters each "
    "sprint.  How many total meters does he run a week?",
    "answer": "He sprints 3*3=<<3*3=9>>9 times\nSo he runs 9*60=<<9*60=540>>540 meters\n#### 540",
    "insufficient_question": "James decides to run 3 sprints 3 times a week. How many total "
    "meters does he run a week?",
}

# A placeholder row (the recon's `220` -> `many` shape): the deleted value is
# replaced by a word, so no numeric value is introduced.
GSM8K_PLACEHOLDER = {
    "question": "A chef has 220 knives. He bought another 20 knives. How many knives does "
    "he have now?",
    "answer": "He has 220+20=<<220+20=240>>240 knives\n#### 240",
    "insufficient_question": "A chef has many knives. He bought another 20 knives. How many "
    "knives does he have now?",
}

# The recon's T2 shape: the last step is done in prose after the last
# annotation, e.g. "he has 25-2 = 23 jewels. #### 23".  The deleted premise
# ("5 more") is a value the annotation chain uses, which is what makes it a
# necessity-passing row whose *gold* is only certifiable from the prose.
GSM8K_T2 = {
    "question": "Aaron has 20 jewels and buys 5 more. Siobhan has 2 fewer jewels than "
    "Aaron. How many jewels does Siobhan have?",
    "answer": "Aaron has 20+5 = <<20+5=25>>25 jewels.\nIf Siobhan has 2 fewer jewels than "
    "Aaron, he has 25-2 = 23 jewels.\n#### 23",
    "insufficient_question": "Aaron has 20 jewels and buys some more. Siobhan has 2 fewer "
    "jewels than Aaron. How many jewels does Siobhan have?",
}

# An admitted math row (a single deleted numeric premise, a gold the solution's
# own last \boxed{} reaches).  Note the blank lines: the question itself embeds
# "\n\n", which is why the prompt must be split at the template marker and not
# at the first blank line.
MATH_PAIR = {
    "solution": "The area of a circle is $\\pi r^2$ with $r = 7$, so it is $49\\pi$.\n"
    "\\boxed{49\\pi}",
    "answer": "49\\pi",
    "subject": "Geometry",
    "level": 2,
    "unique_id": "test/geometry/123.json",
    "insufficient_question": "A circle is drawn in the plane.\n\nIt has some radius.\n\nWhat "
    "is its area?",
    "question": "A circle is drawn in the plane.\n\nIt has radius 7.\n\nWhat is its area?",
}

# recon section 3 "math.json[0]": two deleted numeric premises inside one
# changed region (6 and 4 from the second line's equation), so the row is a
# *drop* case -- it is the row that the "bucket by changed region" convention
# rejects, and the one the [asy] tail makes look like two regions if the prompt
# is sliced at the first blank line.
MATH_TWO_VALUES = {
    "solution": "Both directions are given, so $\\cos\\theta = 0$ and $\\theta = 90^\\circ$.",
    "answer": "90^\\circ",
    "subject": "Precalculus",
    "level": 4,
    "unique_id": "test/precalculus/927.json",
    "insufficient_question": "The set of points $(x,y,z)$ that satisfy\n\\[2x = 3y = -z\\]is a "
    "line.\n\nThere is another line in the space.\n\nFind the angle between these lines, in "
    "degrees.",
    "question": "The set of points $(x,y,z)$ that satisfy\n\\[2x = 3y = -z\\]is a line.\n\nThe "
    "set of points $(x,y,z)$ that satisfy\n\\[6x = -y = -4z\\]is another line.\n\nFind the "
    "angle between these lines, in degrees.",
}

# recon section 3 "svamp.json[0]": cross-problem splice, no `question`, and an
# `answer` (91.0) that belongs to a different problem.
SVAMP_ROW = {
    "insufficient_question": "Dan had $ 3 left with him after he bought a candy bar. If he "
    "had $ 4 at the start How many bottle caps did danny have at first?",
    "answer": 91.0,
}

# recon section 3 "formula.json[0]": no answer, no question.
FORMULA_ROW = {
    "insufficient_question": "What is the value of $(\\gamma - \\arctan(\\tan(\\kappa) + 6 - "
    "\\omega) \\cdot 10^{x})$?",
    "complexity": 3,
    "depth": 3,
}


def write_raw(tmp_path, *, gsm8k=(), svamp=(), math=(), formula=()):
    """Write the four source files and return the directory as a string."""
    for name, rows in (
        ("gsm8k", gsm8k),
        ("svamp", svamp),
        ("math", math),
        ("formula", formula),
    ):
        (tmp_path / f"{name}.json").write_text(json.dumps(list(rows)), encoding="utf-8")
    return str(tmp_path)


def gsm8k_row(question, insufficient_question, answer=None):
    row = dict(GSM8K_PAIR)
    row.update({"question": question, "insufficient_question": insufficient_question})
    if answer is not None:
        row["answer"] = answer
    return row


# ---------------------------------------------------------------------------
# the happy path: both branches, funnel, contract
# ---------------------------------------------------------------------------


class TestEmittedRows:
    def test_pair_emits_both_branches(self, tmp_path):
        raw = write_raw(tmp_path, gsm8k=[GSM8K_PAIR])
        rows, funnel = adapter.build_rows(raw)
        assert [row["extra_info"]["branch"] for row in rows] == [
            adapter.UNSOLVABLE_BRANCH,
            adapter.SOLVABLE_BRANCH,
        ]
        assert funnel["raw_rows"] == 1
        assert funnel[adapter.LIMIT_STAGE] == 1

    def test_unsolvable_row_contract(self, tmp_path):
        raw = write_raw(tmp_path, gsm8k=[GSM8K_PAIR])
        rows, _ = adapter.build_rows(raw)
        unsolvable = rows[0]
        gt = json.loads(unsolvable["reward_model"]["ground_truth"])
        info = unsolvable["extra_info"]

        assert gt["solvable"] is False
        assert gt["answer"] is None  # hard rule: unsolvable rows carry no answer
        assert gt["correct_option_id"] is None
        assert gt["has_diagnosis_label"] is False
        assert gt["perturbation_type"] == "missing_condition"
        assert "judgment_only" not in gt  # this is not a judgment row
        assert info["template"] == schema.TEMPLATE_B
        assert info["branch"] == adapter.UNSOLVABLE_BRANCH
        assert info["solvable"] is False
        assert info["error_type"] == "missing_condition"
        assert info["perturbation_family"] == "deletion"
        assert info["options"] == [] and info["role_words"] == []
        assert info["split"] == "train"
        assert info["difficulty"] == adapter.DIFFICULTY_UNLABELLED
        assert info["deleted_condition_text"] == "He runs 60 meters each sprint."
        assert info["paired_original_text"] == GSM8K_PAIR["question"]
        assert info["perturbed_entity_text"] == GSM8K_PAIR["answer"]
        # the prompt asks for the bare marker and renders no option block
        assert unsolvable["prompt"][0]["content"].startswith(GSM8K_PAIR["insufficient_question"])
        assert "\\boxed{UNSOLVABLE}" in unsolvable["prompt"][0]["content"]
        assert "选项" not in unsolvable["prompt"][0]["content"]

    def test_solvable_row_contract(self, tmp_path):
        raw = write_raw(tmp_path, gsm8k=[GSM8K_PAIR])
        rows, _ = adapter.build_rows(raw)
        solvable = rows[1]
        gt = json.loads(solvable["reward_model"]["ground_truth"])
        info = solvable["extra_info"]

        assert gt["solvable"] is True
        assert gt["answer"] == "540"  # the source's own `####` value
        assert gt["correct_option_id"] is None
        assert info["template"] == schema.TEMPLATE_B
        assert info["branch"] == adapter.SOLVABLE_BRANCH
        assert info["error_type"] == ""  # a well-posed problem is not a defect row
        assert info["task_id"].endswith("-solvable")
        assert info["paired_original_text"] == GSM8K_PAIR["question"]
        assert solvable["prompt"][0]["content"].startswith(GSM8K_PAIR["question"])

    def test_every_row_passes_schema_validation(self, tmp_path):
        raw = write_raw(
            tmp_path, gsm8k=[GSM8K_PAIR, GSM8K_PLACEHOLDER, GSM8K_T2], math=[MATH_PAIR]
        )
        rows, _ = adapter.build_rows(raw)
        assert len(rows) == 8
        for row in rows:
            assert schema.validate_row(row) == []
        schema.normalise_extra_info(rows)
        schema.validate_rows(rows)  # raises on any violation

    def test_math_row_uses_unique_id_level_and_solution_gold(self, tmp_path):
        raw = write_raw(tmp_path, math=[MATH_PAIR])
        rows, _ = adapter.build_rows(raw)
        assert len(rows) == 2
        info = rows[0]["extra_info"]
        assert info["task_id"] == "mip-math-test_geometry_123_json-unsolvable"
        assert info["difficulty"] == "level-2"
        assert rows[0]["prompt"][0]["content"].startswith(MATH_PAIR["insufficient_question"])
        assert json.loads(rows[1]["reward_model"]["ground_truth"])["answer"] == "49\\pi"

    def test_math_row_with_two_deleted_values_is_dropped(self, tmp_path):
        """One changed region, two numeric premises -> not the single-value bucket."""
        raw = write_raw(tmp_path, math=[MATH_TWO_VALUES])
        rows, funnel = adapter.build_rows(raw)
        assert rows == []
        assert funnel["question_differs"] == 1
        assert funnel["single_value_single_region"] == 0

    def test_placeholder_row_is_admitted_with_its_own_family(self, tmp_path):
        raw = write_raw(tmp_path, gsm8k=[GSM8K_PLACEHOLDER])
        rows, _ = adapter.build_rows(raw)
        assert len(rows) == 2
        assert rows[0]["extra_info"]["perturbation_family"] == "placeholder"
        assert rows[0]["extra_info"]["deleted_condition_text"] == "220"

    def test_no_row_carries_options_or_a_diagnosis_label(self, tmp_path):
        raw = write_raw(tmp_path, gsm8k=[GSM8K_PAIR], math=[MATH_PAIR])
        rows, _ = adapter.build_rows(raw)
        for row in rows:
            info = row["extra_info"]
            gt = json.loads(row["reward_model"]["ground_truth"])
            assert info["options"] == []
            assert gt["has_diagnosis_label"] is False
            assert info["correct_option_id"] == ""
            assert "选项：" not in row["prompt"][0]["content"]


# ---------------------------------------------------------------------------
# one test per funnel stage / drop reason
# ---------------------------------------------------------------------------


class TestFunnelStages:
    def test_stage_counts_are_ordered_and_monotone(self, tmp_path):
        raw = write_raw(
            tmp_path,
            gsm8k=[GSM8K_PAIR],
            svamp=[SVAMP_ROW],
            formula=[FORMULA_ROW],
        )
        _, funnel = adapter.build_rows(raw)
        assert list(funnel) == list(adapter.FUNNEL_STAGES)
        counts = list(funnel.values())
        assert counts == sorted(counts, reverse=True)
        assert funnel["raw_rows"] == 3  # 1 gsm8k + 1 svamp + 1 formula
        assert funnel["paired_original_present"] == 1  # svamp and formula cannot pair
        assert funnel[adapter.LIMIT_STAGE] == 1

    def test_unpairable_sources_are_dropped_before_the_diff(self, tmp_path):
        raw = write_raw(tmp_path, svamp=[SVAMP_ROW], formula=[FORMULA_ROW])
        rows, funnel = adapter.build_rows(raw)
        assert rows == []
        assert funnel["raw_rows"] == 2
        assert funnel["paired_original_present"] == 0

    def test_empty_question_is_treated_as_unpaired(self, tmp_path):
        row = dict(SVAMP_ROW)
        row["question"] = ""
        raw = write_raw(tmp_path, svamp=[row])
        rows, funnel = adapter.build_rows(raw)
        assert rows == []
        assert funnel["paired_original_present"] == 0

    def test_identical_question_is_dropped(self, tmp_path):
        """recon hazard 3: 8 gsm8k rows have question == insufficient_question."""
        row = gsm8k_row(GSM8K_PAIR["question"], GSM8K_PAIR["question"])
        raw = write_raw(tmp_path, gsm8k=[row])
        rows, funnel = adapter.build_rows(raw)
        assert rows == []
        assert funnel["paired_original_present"] == 1
        assert funnel["question_differs"] == 0

    def test_whitespace_only_difference_is_dropped(self, tmp_path):
        row = gsm8k_row(GSM8K_PAIR["question"], GSM8K_PAIR["question"].replace("  ", " "))
        raw = write_raw(tmp_path, gsm8k=[row])
        rows, funnel = adapter.build_rows(raw)
        assert rows == []
        assert funnel["question_differs"] == 0

    def test_two_deleted_values_are_dropped(self, tmp_path):
        row = gsm8k_row(
            "Tom has 5 apples and 3 oranges. How many fruits does he have?",
            "Tom has oranges. How many fruits does he have?",
            answer="Tom has 5+3=<<5+3=8>>8 fruits\n#### 8",
        )
        raw = write_raw(tmp_path, gsm8k=[row])
        rows, funnel = adapter.build_rows(raw)
        assert rows == []
        assert funnel["question_differs"] == 1
        assert funnel["single_value_single_region"] == 0

    def test_non_numeric_deletion_is_dropped(self, tmp_path):
        row = gsm8k_row(
            "Tom has 5 apples and 3 oranges. He is happy. How many fruits does he have?",
            "Tom has 5 apples and 3 oranges. How many fruits does he have?",
            answer="Tom has 5+3=<<5+3=8>>8 fruits\n#### 8",
        )
        raw = write_raw(tmp_path, gsm8k=[row])
        rows, funnel = adapter.build_rows(raw)
        assert rows == []
        assert funnel["single_value_single_region"] == 0

    def test_two_regions_are_dropped_even_with_one_value(self, tmp_path):
        """The recon's 12-row trap: one numeric value, but two changed regions."""
        row = gsm8k_row(
            "Tom has 5 apples and 3 oranges. He eats one. How many fruits does he have?",
            "Tom has   apples and 3 oranges. He eats one. How many does he have?",
            answer="Tom has 5+3=<<5+3=8>>8 fruits\n#### 8",
        )
        raw = write_raw(tmp_path, gsm8k=[row])
        rows, funnel = adapter.build_rows(raw)
        assert rows == []
        assert funnel["single_value_single_region"] == 0

    def test_still_visible_value_is_dropped(self, tmp_path):
        """The deleted number is still readable elsewhere in the truncated text."""
        row = gsm8k_row(
            "Tom has 60 apples and 60 oranges. How many fruits does he have?",
            "Tom has 60 apples and some oranges. How many fruits does he have?",
            answer="Tom has 60+60=<<60+60=120>>120 fruits\n#### 120",
        )
        raw = write_raw(tmp_path, gsm8k=[row])
        rows, funnel = adapter.build_rows(raw)
        assert rows == []
        assert funnel["single_value_single_region"] == 1
        assert funnel["deleted_value_absent"] == 0

    def test_inserted_numeric_is_dropped(self, tmp_path):
        """A replacement that introduces a number leaves the question answerable."""
        row = gsm8k_row(
            "Tom has 60 apples and 40 oranges. How many fruits does he have?",
            "Tom has 60 apples and 50 oranges. How many fruits does he have?",
            answer="Tom has 60+40=<<60+40=100>>100 fruits\n#### 100",
        )
        raw = write_raw(tmp_path, gsm8k=[row])
        rows, funnel = adapter.build_rows(raw)
        assert rows == []
        assert funnel["deleted_value_absent"] == 1
        assert funnel["no_inserted_numeric"] == 0

    def test_value_absent_from_the_chain_is_dropped(self, tmp_path):
        """The deleted value is not used by the source's own derivation."""
        row = gsm8k_row(
            "Tom has 7 apples and 40 oranges. How many fruits does he have?",
            "Tom has 7 apples and some oranges. How many fruits does he have?",
            answer="Tom has 7+?=<<7+0=7>>7 fruits\n#### 7",
        )
        raw = write_raw(tmp_path, gsm8k=[row])
        rows, funnel = adapter.build_rows(raw)
        assert funnel["no_inserted_numeric"] == 1
        assert funnel["necessary_value_certified"] == 0

    def test_empty_derivation_chain_is_not_credited(self, tmp_path):
        """recon hazard 2: 4 gsm8k rows have no <<expr=val>> annotation at all.

        The fail-closed rule is to drop them, not to assume they are fine.
        """
        row = gsm8k_row(
            "Tom has 7 apples and 40 oranges. How many fruits does he have?",
            "Tom has 7 apples and some oranges. How many fruits does he have?",
            answer="Scottish Unicorns:27(1/3)=9\nSo the total is 7 fruits\n#### 7",
        )
        raw = write_raw(tmp_path, gsm8k=[row])
        rows, funnel = adapter.build_rows(raw)
        assert rows == []
        assert funnel["necessary_value_certified"] == 0

    def test_uncertifiable_gold_keeps_the_unsolvable_row_only(self, tmp_path):
        """The answer key contradicts the source's own arithmetic: no twin."""
        row = gsm8k_row(
            "Tom has 6 apples and 40 oranges. How many fruits does he have?",
            "Tom has some apples and 40 oranges. How many fruits does he have?",
            answer="Tom has 6+0=<<6+0=6>>6 fruits\nSo he has 24/240 = 0.10 baskets\n#### 10",
        )
        raw = write_raw(tmp_path, gsm8k=[row])
        rows, funnel = adapter.build_rows(raw)
        assert [row["extra_info"]["branch"] for row in rows] == [adapter.UNSOLVABLE_BRANCH]
        assert funnel["necessary_value_certified"] == 1

    def test_value_used_by_the_chain_is_admitted(self, tmp_path):
        """The positive control for the necessity stage: 5 is in <<20+5=25>>."""
        raw = write_raw(tmp_path, gsm8k=[GSM8K_T2])
        rows, funnel = adapter.build_rows(raw)
        assert funnel["necessary_value_certified"] == 1
        assert len(rows) == 2
        assert json.loads(rows[1]["reward_model"]["ground_truth"])["answer"] == "23"


# ---------------------------------------------------------------------------
# certificates and conventions (unit tests)
# ---------------------------------------------------------------------------


class TestCertificates:
    def test_gsm8k_annotation_tier(self):
        gold, tier = adapter._certify_gsm8k(GSM8K_PAIR["answer"])
        assert (gold, tier) == ("540", "T1")

    def test_gsm8k_final_step_tier(self):
        gold, tier = adapter._certify_gsm8k(GSM8K_T2["answer"])
        assert (gold, tier) == ("23", "T2")

    def test_gsm8k_conflicting_key_is_uncertified(self):
        answer = "Tom has 6+0=<<6+0=6>>6 fruits\nSo he has 24/240 = 0.10 baskets\n#### 10"
        assert adapter._certify_gsm8k(answer) == (None, None)

    def test_gsm8k_without_a_key_is_uncertified(self):
        assert adapter._certify_gsm8k("no key here") == (None, None)

    def test_math_gold_is_the_last_boxed_value(self):
        gold, tier = adapter._certify_math(MATH_PAIR)
        assert (gold, tier) == ("49\\pi", "solution_boxed")

    def test_necessary_folds_percentages(self):
        """Convention: necessity compares Decimals and folds 10 <-> 0.1."""
        assert adapter._necessary("10", "the rate is 0.1 of the price") is True
        assert adapter._necessary("0.25", "25 out of every 100") is True
        assert adapter._necessary("5", "we counted 500 items") is True
        assert adapter._necessary("7", "we counted 3 and 4 items") is False
        assert adapter._necessary("7", "") is None  # nothing to check against

    def test_math_gold_without_a_matching_box_is_uncertified(self):
        row = dict(MATH_PAIR, solution="The answer follows from the diagram.")
        assert adapter._certify_math(row) == (None, None)

    def test_last_boxed_survives_nested_braces(self):
        text = "$x=\\boxed{\\frac{1}{2}}$ and finally $\\boxed{3}$"
        assert adapter._last_boxed(text) == "3"

    def test_safe_arithmetic_refuses_non_arithmetic(self):
        assert adapter._safe_arithmetic("__import__('os')") is None
        assert adapter._safe_arithmetic("6/0") is None
        assert adapter._safe_arithmetic("3*3") == 9

    def test_certifiers_refuse_empty_input(self):
        assert adapter._certify_gsm8k("") == (None, None)
        assert adapter._certify_gsm8k("no key here") == (None, None)
        assert adapter._certify_gsm8k("a non-numeric key\n#### abc") == (None, None)
        assert adapter._certify_math({}) == (None, None)
        assert adapter._certify_math({"answer": "", "solution": "\\boxed{1}"}) == (None, None)

    def test_unevaluable_annotation_falls_through_to_the_final_step(self):
        """An annotation whose expression is not arithmetic is skipped, not fatal."""
        answer = (
            "The total is unknown=<<unknown=5>>5 so far.\n"
            "Adding the rest, we get 5+2 = 7 items\n#### 7"
        )
        assert adapter._certify_gsm8k(answer) == ("7", "T2")

    def test_decimal_rejects_non_numeric_tokens(self):
        assert adapter._decimal("abc") is None
        assert adapter._decimal("12.5") == __import__("decimal").Decimal("12.5")

    def test_necessary_rejects_a_non_numeric_target(self):
        assert adapter._necessary("abc", "the value is 3") is False

    def test_chain_and_derivation_text_are_empty_for_unpaired_sources(self):
        assert adapter._chain_text("svamp", SVAMP_ROW) == ""
        assert adapter._derivation_text("formula", FORMULA_ROW) == ""

    def test_load_rejects_a_non_list_file(self, tmp_path):
        for name in ("gsm8k", "svamp", "math", "formula"):
            (tmp_path / f"{name}.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError):
            adapter.build_rows(str(tmp_path))

    def test_membership_uses_string_normalised_numerics(self):
        """Convention (recon 4.1 note 3): `4.00` is not `4` for membership."""
        assert adapter._normalise_number("4.00") == "4.00"
        assert adapter._normalise_number("4.0") == "4"
        assert adapter._normalise_number("1,200") == "1200"
        assert adapter._numeric_values("costs $4.00 today") == {"4.00"}


class TestClassify:
    def test_bucket_single(self):
        diff = adapter._classify("run 60 meters now", "run now")
        assert diff["bucket"] == "single"
        assert diff["n_values"] == 1
        assert diff["pure_deletion"] is True

    def test_bucket_placeholder_is_a_replace(self):
        diff = adapter._classify("a chef has 220 knives", "a chef has many knives")
        assert diff["bucket"] == "single"
        assert diff["pure_deletion"] is False

    def test_bucket_multi_value(self):
        diff = adapter._classify("Tom has 5 apples and 3 oranges", "Tom has oranges")
        assert diff["bucket"] == "multi"
        assert diff["n_values"] == 2

    def test_bucket_zero_value(self):
        diff = adapter._classify("Tom runs fast today", "Tom runs today")
        assert diff["bucket"] == "zero"

    def test_bucket_multi_region(self):
        diff = adapter._classify("5 apples red and 3 oranges", "apples red and oranges")
        assert diff["bucket"] == "multi_region"

    def test_identical_text_classifies_as_none(self):
        assert adapter._classify("same text", "same  text") is None


# ---------------------------------------------------------------------------
# limit, determinism, ids
# ---------------------------------------------------------------------------


class TestLimitAndDeterminism:
    def _raw(self, tmp_path, count=4):
        rows = []
        for index in range(count):
            rows.append(
                gsm8k_row(
                    f"P{index} has 10 apples and 30 oranges. How many fruits?",
                    f"P{index} has 10 apples and some oranges. How many fruits?",
                    answer=f"P{index} has 10+30=<<10+30={40 + index}>>{40 + index} fruits\n"
                    f"#### {40 + index}",
                )
            )
        return write_raw(tmp_path, gsm8k=rows)

    def test_limit_keeps_pairs_whole(self, tmp_path):
        raw = self._raw(tmp_path)
        rows, funnel = adapter.build_rows(raw, limit=4)
        assert len(rows) == 4
        assert funnel[adapter.LIMIT_STAGE] == 2
        branches = [row["extra_info"]["branch"] for row in rows]
        assert branches.count(adapter.UNSOLVABLE_BRANCH) == 2
        assert branches.count(adapter.SOLVABLE_BRANCH) == 2

    def test_odd_limit_rounds_down_to_a_whole_pair_count(self, tmp_path):
        raw = self._raw(tmp_path)
        rows, funnel = adapter.build_rows(raw, limit=3)
        assert len(rows) == 2
        assert funnel[adapter.LIMIT_STAGE] == 1

    def test_limit_below_one_pair_is_rejected(self, tmp_path):
        raw = self._raw(tmp_path)
        with pytest.raises(ValueError):
            adapter.build_rows(raw, limit=1)

    def test_same_seed_is_byte_identical(self, tmp_path):
        raw = self._raw(tmp_path)
        first, funnel_a = adapter.build_rows(raw, seed=7)
        second, funnel_b = adapter.build_rows(raw, seed=7)
        assert funnel_a == funnel_b
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)

    def test_a_new_seed_keeps_the_same_row_set(self, tmp_path):
        raw = self._raw(tmp_path)
        rows_a, _ = adapter.build_rows(raw, seed=0)
        rows_b, _ = adapter.build_rows(raw, seed=1)
        ids_a = sorted(row["extra_info"]["task_id"] for row in rows_a)
        ids_b = sorted(row["extra_info"]["task_id"] for row in rows_b)
        assert ids_a == ids_b

    def test_task_ids_are_stable_and_unique(self, tmp_path):
        raw = self._raw(tmp_path)
        rows, _ = adapter.build_rows(raw)
        ids = [row["extra_info"]["task_id"] for row in rows]
        assert len(ids) == len(set(ids))
        assert all(identifier.startswith("mip-gsm8k-") for identifier in ids)
        assert {identifier.rsplit("-", 1)[1] for identifier in ids} == {
            "solvable",
            "unsolvable",
        }

    def test_index_follows_the_written_order(self, tmp_path):
        raw = self._raw(tmp_path)
        rows, _ = adapter.build_rows(raw)
        assert [row["extra_info"]["index"] for row in rows] == list(range(len(rows)))
        assert all(row["extra_info"]["seed"] == 0 for row in rows)

    def test_difficulty_is_a_string_on_every_row(self, tmp_path):
        raw = write_raw(tmp_path, gsm8k=[GSM8K_PAIR], math=[MATH_PAIR])
        rows, _ = adapter.build_rows(raw)
        assert all(isinstance(row["extra_info"]["difficulty"], str) for row in rows)
        assert {row["extra_info"]["difficulty"] for row in rows} == {
            "unlabelled",
            "level-2",
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_writes_a_readable_parquet(self, tmp_path, capsys):
        raw = write_raw(tmp_path, gsm8k=[GSM8K_PAIR, GSM8K_T2], math=[MATH_PAIR])
        out = tmp_path / "mip.parquet"
        code = adapter.main(["--raw-dir", raw, "--out", str(out), "--seed", "3"])
        assert code == 0
        printed = capsys.readouterr().out
        assert "funnel" in printed
        assert "by_branch" in printed
        assert "by_template" in printed
        assert "by_solvable" in printed

        rows = schema.read_parquet_rows(str(out))
        assert len(rows) > 0
        for row in rows:
            assert schema.validate_row(row) == []
        assert all("options" in row["extra_info"] for row in rows)

    def test_main_fails_closed_when_nothing_survives(self, tmp_path):
        raw = write_raw(tmp_path, svamp=[SVAMP_ROW])
        out = tmp_path / "empty.parquet"
        code = adapter.main(["--raw-dir", raw, "--out", str(out)])
        assert code == 1
        assert not out.exists()
