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
"""Unit tests for the hallucination-resistance reward dispatcher.

Run from the repo root::

    pytest examples/reasoning_rl/reward/test_hallucination_compute_score.py -v

The branch x case matrix of HALLUCINATION_RL_DESIGN.md section 9 is spread over
the ``Test*`` classes below.  Every branch gets a correct-output cell, a
wrong-direction cell and an unparseable cell, and the two cross-cutting rules are
pinned down explicitly:

* ``-1`` applies in exactly two places -- fabricating an answer on an unsolvable
  row and refusing on a ``judgment_only`` row; every other solvable branch scores
  a misrefusal 0 (section 6's pseudocode, section 9's matrix, D24/D26);
* the bare ``\\boxed{UNSOLVABLE}`` string is branch-dependent: +1 on a three-tier
  source, 0 on a four-tier one (section 5.1's "最容易写错的一处").
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hallucination_compute_score import (
    STATUS_ANSWER,
    STATUS_NONE,
    STATUS_SOLVABLE_MARKER,
    STATUS_UNSOLVABLE_BARE,
    STATUS_UNSOLVABLE_OPTION,
    compute_score,
    extract_final,
    kk_match,
    math_match,
    norm_match,
    parse_pair,
    score_from_status,
)


def think_wrap(response: str, think: str = "some reasoning") -> str:
    """A Qwen3-4B thinking-template-compliant response for gate-passing tests."""
    return f"<think>{think}</think>\n\n{response}"


def gt_json(**fields) -> str:
    """A ``reward_model.ground_truth`` JSON string, as the parquet stores it."""
    return json.dumps(fields)


def score(data_source: str, response: str, ground_truth) -> float:
    return compute_score(data_source, think_wrap(response), ground_truth)["score"]


def math_verify_installed() -> bool:
    """Whether the optional ``math-verify`` package is importable.

    It is in requirements.txt, so the training host has it; this dev box does
    not, and without it only exact/structural answer equality is decidable
    (``0.5`` vs ``\\frac{1}{2}`` needs the symbolic parser).
    """
    try:
        import math_verify  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------


class TestExtractFinal:
    def test_last_boxed_wins(self):
        status, answer, _ = extract_final(think_wrap("first \\boxed{1} then \\boxed{2}"))
        assert (status, answer) == (STATUS_ANSWER, "2")

    def test_nested_braces(self):
        status, answer, _ = extract_final("$\\boxed{\\frac{1}{2}}$")
        assert (status, answer) == (STATUS_ANSWER, "\\frac{1}{2}")

    def test_no_boxed(self):
        assert extract_final(think_wrap("the answer is 42")) == (STATUS_NONE, None, None)

    def test_unterminated_boxed(self):
        assert extract_final("\\boxed{42") == (STATUS_NONE, None, None)

    def test_empty_boxed(self):
        assert extract_final("\\boxed{  }") == (STATUS_NONE, None, None)

    def test_answer_that_contains_the_word_unsolvable(self):
        # Only the whole boxed text decides; prose inside an answer is a refusal
        # shape, and the option id in it is unreadable.
        status, answer, _ = extract_final("\\boxed{the problem is unsolvable for small n}")
        assert status == STATUS_UNSOLVABLE_BARE

    def test_solvable_marker(self):
        assert extract_final("\\boxed{SOLVABLE}") == (STATUS_SOLVABLE_MARKER, None, None)

    def test_solvable_marker_case_and_punctuation(self):
        assert extract_final("\\boxed{**Solvable**.}") == (STATUS_SOLVABLE_MARKER, None, None)

    def test_unsolvable_bare(self):
        assert extract_final("\\boxed{UNSOLVABLE}") == (STATUS_UNSOLVABLE_BARE, None, None)

    def test_unsolvable_with_option(self):
        assert extract_final("\\boxed{UNSOLVABLE: b}") == (STATUS_UNSOLVABLE_OPTION, None, "B")

    def test_unsolvable_with_dash_option(self):
        assert extract_final("\\boxed{Unsolvable - C}") == (STATUS_UNSOLVABLE_OPTION, None, "C")

    def test_unsolvable_with_prose_tail_is_bare(self):
        # An option id has to be a single letter; prose is not an option, and
        # silently scoring it 0 would punish a correct refusal for phrasing.
        assert extract_final("\\boxed{UNSOLVABLE: the number of sprints is missing}") == (
            STATUS_UNSOLVABLE_BARE,
            None,
            None,
        )

    def test_marker_inside_a_sentence_still_a_refusal(self):
        assert extract_final("\\boxed{Information is insufficient, so UNSOLVABLE.}") == (
            STATUS_UNSOLVABLE_BARE,
            None,
            None,
        )

    def test_option_on_its_own_line(self):
        assert extract_final("\\boxed{UNSOLVABLE:\nB\n}") == (STATUS_UNSOLVABLE_OPTION, None, "B")

    def test_pair_answer_is_an_answer(self):
        assert extract_final("\\boxed{B: 42}") == (STATUS_ANSWER, "B: 42", None)


# ---------------------------------------------------------------------------
# branch 1: solvable, numeric (GSM-IC, synthesised distractors, TreeCut positives)
# ---------------------------------------------------------------------------


class TestSolvableNumeric:
    SOURCE = "halluc_math_gsmic"
    GT = gt_json(solvable=True, answer="72", has_diagnosis_label=False, perturbation_type="distracting_condition")

    def test_correct_answer_scores(self):
        assert score(self.SOURCE, "\\boxed{72}", self.GT) == 1.0

    def test_wrong_answer_scores_zero(self):
        assert score(self.SOURCE, "\\boxed{71}", self.GT) == 0.0

    def test_misrefusal_scores_zero(self):
        # Section 6's pseudocode and section 9's matrix: only a judgment_only row
        # penalises a refusal, so a numeric row does not (D24/D26 restate it as
        # "误拒 → 0（与普通可解行一致）").
        assert score(self.SOURCE, "\\boxed{UNSOLVABLE}", self.GT) == 0.0

    def test_misrefusal_with_option_scores_zero(self):
        assert score(self.SOURCE, "\\boxed{UNSOLVABLE: B}", self.GT) == 0.0

    def test_misrefusal_with_solvable_marker_is_zero(self):
        # \boxed{SOLVABLE} on a plain numeric row is not an answer: no credit.
        assert score(self.SOURCE, "\\boxed{SOLVABLE}", self.GT) == 0.0

    def test_no_boxed_scores_zero(self):
        assert score(self.SOURCE, "72", self.GT) == 0.0

    def test_latex_answer(self):
        gt = gt_json(solvable=True, answer="\\frac{1}{2}", has_diagnosis_label=False)
        assert score(self.SOURCE, "\\boxed{\\frac{1}{2}}", gt) == 1.0

    @pytest.mark.skipif(not math_verify_installed(), reason="needs the optional math-verify parser")
    def test_latex_equivalent_forms(self):
        gt = gt_json(solvable=True, answer="\\frac{1}{2}", has_diagnosis_label=False)
        assert score(self.SOURCE, "\\boxed{0.5}", gt) == 1.0

    def test_placeholder_option_block_does_not_change_the_branch(self):
        """TreeCut positives (D26) carry a placeholder block the reward ignores."""
        gt = gt_json(
            solvable=True,
            answer="72",
            correct_option_id=None,
            has_diagnosis_label=False,
            perturbation_type=None,
        )
        assert score("halluc_math_treecut", "\\boxed{72}", gt) == 1.0
        assert score("halluc_math_treecut", "\\boxed{UNSOLVABLE: A}", gt) == 0.0


# ---------------------------------------------------------------------------
# branch 2: solvable, name -> role pairs (K&K, D19) -- the reward-inversion defence
# ---------------------------------------------------------------------------


class TestSolvableNameRolePairs:
    SOURCE = "halluc_logic_kk"

    @staticmethod
    def gt(answer, role_words):
        return gt_json(solvable=True, answer=answer, role_words=role_words, has_diagnosis_label=False)

    def test_mapping_match(self):
        gt = self.gt({"Oliver": "knight", "Ethan": "knave"}, ["knight", "knave"])
        assert score(self.SOURCE, "\\boxed{Oliver: knight, Ethan: knave}", gt) == 1.0

    def test_name_order_is_irrelevant(self):
        gt = self.gt({"Oliver": "knight", "Ethan": "knave"}, ["knight", "knave"])
        assert score(self.SOURCE, "\\boxed{Ethan: knave, Oliver: knight}", gt) == 1.0

    def test_case_and_separators_are_irrelevant(self):
        gt = self.gt({"Oliver": "knight", "Ethan": "knave"}, ["knight", "knave"])
        assert score(self.SOURCE, "\\boxed{oliver: KNIGHT; ethan: a knave}", gt) == 1.0
        assert score(self.SOURCE, "\\boxed{**Oliver**: knight\nEthan: knave}", gt) == 1.0

    def test_flip_role_inversion_defence(self):
        """flip_role rows write "knaves always tell the truth": the *surface* words
        are swapped, so a hard-coded K = knight would score truth as falsehood."""
        gt = self.gt({"Oliver": "knave", "Ethan": "knave"}, ["knave", "knight"])
        assert score(self.SOURCE, "\\boxed{Oliver: knave, Ethan: knave}", gt) == 1.0
        # The canonical words are the *wrong* answer on this row.
        assert score(self.SOURCE, "\\boxed{Oliver: knight, Ethan: knight}", gt) == 0.0

    def test_random_pair_inversion_defence(self):
        gt = self.gt({"Quinn": "angel", "Ava": "devil", "Jack": "devil"}, ["angel", "devil"])
        assert score(self.SOURCE, "\\boxed{Quinn: angel, Ava: devil, Jack: devil}", gt) == 1.0
        assert score(self.SOURCE, "\\boxed{Quinn: devil, Ava: angel, Jack: angel}", gt) == 0.0

    def test_old_role_sequence_scores_zero(self):
        """The superseded D11 shape carries no name to check the answer against."""
        gt = self.gt({"Oliver": "knight", "Ethan": "knave"}, ["knight", "knave"])
        assert score(self.SOURCE, "\\boxed{knight knave}", gt) == 0.0
        assert score(self.SOURCE, "\\boxed{knave knight}", gt) == 0.0

    def test_wrong_role_scores_zero(self):
        gt = self.gt({"Oliver": "knight", "Ethan": "knave"}, ["knight", "knave"])
        assert score(self.SOURCE, "\\boxed{Oliver: knave, Ethan: knave}", gt) == 0.0

    def test_missing_name_scores_zero(self):
        gt = self.gt({"Oliver": "knight", "Ethan": "knave"}, ["knight", "knave"])
        assert score(self.SOURCE, "\\boxed{Oliver: knight}", gt) == 0.0

    def test_extra_name_scores_zero(self):
        gt = self.gt({"Oliver": "knight", "Ethan": "knave"}, ["knight", "knave"])
        assert score(self.SOURCE, "\\boxed{Oliver: knight, Ethan: knave, Zoe: knave}", gt) == 0.0

    def test_duplicate_name_scores_zero(self):
        gt = self.gt({"Oliver": "knight", "Ethan": "knave"}, ["knight", "knave"])
        assert score(self.SOURCE, "\\boxed{Oliver: knight, Oliver: knave, Ethan: knave}", gt) == 0.0

    def test_out_of_vocabulary_role_scores_zero(self):
        gt = self.gt({"Oliver": "knight", "Ethan": "knave"}, ["knight", "knave"])
        assert score(self.SOURCE, "\\boxed{Oliver: sage, Ethan: knave}", gt) == 0.0

    def test_misrefusal_scores_zero(self):
        # Section 9: \boxed{UNSOLVABLE} on a K&K row is 0, not -1.
        gt = self.gt({"Oliver": "knight", "Ethan": "knave"}, ["knight", "knave"])
        assert score(self.SOURCE, "\\boxed{UNSOLVABLE}", gt) == 0.0

    def test_missing_role_words_fails_closed(self):
        # An adapter that forgot role_words must not silently score with an
        # assumed knight/knave mapping (design doc section 11 risk 5).
        gt = self.gt({"Oliver": "knave"}, None)
        assert score(self.SOURCE, "\\boxed{Oliver: knave}", gt) == 0.0
        assert kk_match("Oliver: knave", None, {"Oliver": "knave"}) is False


# ---------------------------------------------------------------------------
# branch 3: solvable, judgement + answer (two-layer: UMWP D24, FalseQA D27)
# ---------------------------------------------------------------------------


class TestSolvableTwoLayerNumeric:
    """UMWP-answerable: giving an answer *is* the judgement (D24)."""

    SOURCE = "halluc_math_umwp"
    GT = gt_json(solvable=True, answer="42", two_layer=True, has_diagnosis_label=False)

    def test_answer_scores_full(self):
        assert score(self.SOURCE, "\\boxed{42}", self.GT) == 1.0

    def test_judged_but_wrong_scores_half(self):
        assert score(self.SOURCE, "\\boxed{43}", self.GT) == 0.5

    def test_refusal_fails_the_judgement_layer(self):
        for refusal in ("\\boxed{UNSOLVABLE}", "\\boxed{UNSOLVABLE: B}", "\\boxed{SOLVABLE}"):
            assert score(self.SOURCE, refusal, self.GT) == 0.0, refusal

    def test_no_boxed_scores_zero(self):
        assert score(self.SOURCE, "42", self.GT) == 0.0


class TestSolvableTwoLayerFreeText:
    """FalseQA-answerable: same two layers over a free-text answer (D27)."""

    SOURCE = "halluc_commonsense_falseqa"
    GT = gt_json(
        solvable=True, answer="a teacher", solvable_answer=True, correct_option_id=None, has_diagnosis_label=False
    )

    def test_normalised_match_scores_full(self):
        assert score(self.SOURCE, "\\boxed{a teacher}", self.GT) == 1.0

    def test_case_punctuation_and_articles_are_normalised(self):
        for answer in ("\\boxed{Teacher!}", "\\boxed{the teacher.}", "\\boxed{  TEACHER  }"):
            assert score(self.SOURCE, answer, self.GT) == 1.0, answer

    def test_judged_but_wrong_scores_half(self):
        assert score(self.SOURCE, "\\boxed{a student}", self.GT) == 0.5

    def test_solvable_marker_takes_the_judgement_layer_only(self):
        assert score(self.SOURCE, "\\boxed{SOLVABLE}", self.GT) == 0.5

    def test_misrefusal_scores_zero(self):
        assert score(self.SOURCE, "\\boxed{UNSOLVABLE}", self.GT) == 0.0
        assert score(self.SOURCE, "\\boxed{UNSOLVABLE: B}", self.GT) == 0.0

    def test_no_boxed_scores_zero(self):
        assert score(self.SOURCE, "a teacher", self.GT) == 0.0

    def test_norm_match_directly(self):
        assert norm_match("The Teacher!", "a teacher") is True
        assert norm_match("teacher", "teachers") is False
        assert norm_match(None, "teacher") is False


# ---------------------------------------------------------------------------
# branch 4: SUM's paired judge-then-solve task (D23)
# ---------------------------------------------------------------------------


class TestSumPairTask:
    SOURCE = "halluc_math_sumpair"
    GT = gt_json(
        solvable=True,
        pair_task=True,
        answerable_id="B",
        answer=42,
        has_diagnosis_label=False,
        perturbation_type=None,
    )

    def test_judged_and_solved_scores_full(self):
        assert score(self.SOURCE, "\\boxed{B: 42}", self.GT) == 1.0

    def test_lowercase_and_full_width_colon_parse(self):
        assert score(self.SOURCE, "\\boxed{b：42}", self.GT) == 1.0
        assert score(self.SOURCE, "\\boxed{B : 42}", self.GT) == 1.0

    def test_judged_but_wrong_scores_half(self):
        assert score(self.SOURCE, "\\boxed{B: 43}", self.GT) == 0.5

    def test_wrong_judgement_scores_zero_even_with_the_right_answer(self):
        assert score(self.SOURCE, "\\boxed{A: 42}", self.GT) == 0.0

    def test_unparseable_shapes_score_zero(self):
        for response in ("\\boxed{42}", "\\boxed{UNSOLVABLE}", "\\boxed{A}", "42"):
            assert score(self.SOURCE, response, self.GT) == 0.0, response

    def test_ab_swap_invariance(self):
        """Swapping the two questions and answerable_id must not change the score."""
        swapped = gt_json(
            solvable=True,
            pair_task=True,
            answerable_id="A",
            answer=42,
            has_diagnosis_label=False,
            perturbation_type=None,
        )
        assert score(self.SOURCE, "\\boxed{A: 42}", swapped) == 1.0
        assert score(self.SOURCE, "\\boxed{B: 42}", swapped) == 0.0

    def test_parse_pair_directly(self):
        assert parse_pair("A: 42") == ("A", "42")
        assert parse_pair("b：7") == ("B", "7")
        assert parse_pair("42") == (None, None)
        assert parse_pair(None) == (None, None)


# ---------------------------------------------------------------------------
# branch 5: solvable, judgement only (CREPE-normal)
# ---------------------------------------------------------------------------


class TestSolvableJudgmentOnly:
    SOURCE = "halluc_commonsense_crepe"
    GT = gt_json(solvable=True, answer=None, judgment_only=True, has_diagnosis_label=False, correct_option_id=None)

    def test_solvable_marker_scores(self):
        assert score(self.SOURCE, "\\boxed{SOLVABLE}", self.GT) == 1.0

    def test_misrefusal_is_penalised(self):
        assert score(self.SOURCE, "\\boxed{UNSOLVABLE}", self.GT) == -1.0

    def test_misrefusal_with_option_is_penalised(self):
        # The row has no usable option, but the model still refused.
        assert score(self.SOURCE, "\\boxed{UNSOLVABLE: B}", self.GT) == -1.0

    def test_ordinary_answer_scores_zero(self):
        # Doc section 6: "没判断（给了普通答案/无 \boxed{}）" -> 0.
        assert score(self.SOURCE, "\\boxed{Because water is wet}", self.GT) == 0.0

    def test_no_boxed_scores_zero(self):
        assert score(self.SOURCE, "The premise holds.", self.GT) == 0.0

    def test_judgment_only_never_runs_math_match(self):
        """The audit-only ``answer`` field must not become a hidden numeric gold."""
        gt = gt_json(solvable=True, answer="42", judgment_only=True, has_diagnosis_label=False)
        assert score(self.SOURCE, "\\boxed{42}", gt) == 0.0


# ---------------------------------------------------------------------------
# branch 6: unsolvable, three-tier bare (MiP, UMWP-unanswerable, CREPE-FP)
# ---------------------------------------------------------------------------


class TestUnsolvableThreeTier:
    SOURCE = "halluc_math_mip"
    GT = gt_json(solvable=False, answer=None, correct_option_id=None, has_diagnosis_label=False)

    def test_bare_refusal_scores(self):
        assert score(self.SOURCE, "\\boxed{UNSOLVABLE}", self.GT) == 1.0

    def test_spurious_option_id_still_scores(self):
        # Doc sections 5.1/6: the prompt carries no option block, so an option id
        # is meaningless and the verdict earns full credit.
        assert score(self.SOURCE, "\\boxed{UNSOLVABLE: A}", self.GT) == 1.0

    def test_fabrication_is_penalised(self):
        assert score(self.SOURCE, "\\boxed{540}", self.GT) == -1.0

    def test_no_boxed_scores_zero(self):
        assert score(self.SOURCE, "The problem is missing information.", self.GT) == 0.0

    def test_solvable_marker_scores_zero(self):
        # Claiming an unanswerable question is answerable is neither fabricating a
        # number nor refusing; the contract has no credit for it.
        assert score(self.SOURCE, "\\boxed{SOLVABLE}", self.GT) == 0.0

    def test_source_agnostic(self):
        """Every three-tier source shares one behaviour (design doc section 9)."""
        for source in (
            "halluc_math_mip",
            "halluc_math_umwp",
            "halluc_commonsense_crepe",
        ):
            assert score(source, "\\boxed{UNSOLVABLE}", self.GT) == 1.0, source
            assert score(source, "\\boxed{42}", self.GT) == -1.0, source


# ---------------------------------------------------------------------------
# branch 7: unsolvable, four-tier diagnostic (FalseQA-fake, TreeCut negatives D26)
# ---------------------------------------------------------------------------


class TestUnsolvableFourTier:
    SOURCE = "halluc_commonsense_falseqa"
    GT = gt_json(
        solvable=False,
        answer=None,
        correct_option_id="B",
        has_diagnosis_label=True,
        perturbation_type="contradictory_condition",
        options=[
            {"id": "A", "text": "man -> child"},
            {"id": "B", "text": "man -> women"},
            {"id": "C", "text": "man -> teacher"},
        ],
    )

    def test_correct_option_scores(self):
        assert score(self.SOURCE, "\\boxed{UNSOLVABLE: B}", self.GT) == 1.0

    def test_wrong_option_scores_zero(self):
        assert score(self.SOURCE, "\\boxed{UNSOLVABLE: A}", self.GT) == 0.0
        assert score(self.SOURCE, "\\boxed{UNSOLVABLE: C}", self.GT) == 0.0

    def test_out_of_range_option_scores_zero(self):
        assert score(self.SOURCE, "\\boxed{UNSOLVABLE: Z}", self.GT) == 0.0

    def test_bare_refusal_scores_zero(self):
        """The mirror image of the three-tier branch: the verdict is right but the
        pointer the row asks for is missing (doc section 5.1, "最容易写错的一处")."""
        assert score(self.SOURCE, "\\boxed{UNSOLVABLE}", self.GT) == 0.0

    def test_fabrication_is_penalised(self):
        assert score(self.SOURCE, "\\boxed{42}", self.GT) == -1.0

    def test_no_boxed_scores_zero(self):
        assert score(self.SOURCE, "选项 B 让前提为假。", self.GT) == 0.0

    def test_missing_correct_option_id_fails_closed(self):
        gt = gt_json(solvable=False, answer=None, correct_option_id=None, has_diagnosis_label=True)
        assert score(self.SOURCE, "\\boxed{UNSOLVABLE: B}", gt) == 0.0
        assert score(self.SOURCE, "\\boxed{UNSOLVABLE}", gt) == 0.0

    def test_option_id_case_insensitive(self):
        assert score(self.SOURCE, "\\boxed{Unsolvable: b}", self.GT) == 1.0

    def test_option_shuffle_invariance(self):
        """Shuffling the option block and updating correct_option_id in step must
        not change the score: the reward compares ids, never positions."""
        shuffled = gt_json(
            solvable=False,
            answer=None,
            correct_option_id="C",
            has_diagnosis_label=True,
            options=[
                {"id": "A", "text": "man -> women"},
                {"id": "B", "text": "man -> teacher"},
                {"id": "C", "text": "man -> child"},
            ],
        )
        assert score(self.SOURCE, "\\boxed{UNSOLVABLE: C}", shuffled) == 1.0
        assert score(self.SOURCE, "\\boxed{UNSOLVABLE: B}", shuffled) == 0.0

    def test_treecut_negative_uses_the_same_branch(self):
        gt = gt_json(
            solvable=False,
            answer=None,
            correct_option_id="A",
            has_diagnosis_label=True,
            perturbation_type="missing_condition",
        )
        assert score("halluc_math_treecut", "\\boxed{UNSOLVABLE: A}", gt) == 1.0
        assert score("halluc_math_treecut", "\\boxed{UNSOLVABLE}", gt) == 0.0


# ---------------------------------------------------------------------------
# the bare-marker branch correlation (doc section 9)
# ---------------------------------------------------------------------------


class TestBareMarkerBranchCorrelation:
    def test_same_string_two_scores(self):
        three_tier = gt_json(solvable=False, has_diagnosis_label=False)
        four_tier = gt_json(solvable=False, has_diagnosis_label=True, correct_option_id="B")
        assert score("halluc_math_umwp", "\\boxed{UNSOLVABLE}", three_tier) == 1.0
        assert score("halluc_math_treecut", "\\boxed{UNSOLVABLE}", four_tier) == 0.0

    def test_the_message_names_the_source(self):
        four_tier = gt_json(solvable=False, has_diagnosis_label=True, correct_option_id="B")
        assert score("halluc_math_treecut", "\\boxed{UNSOLVABLE}", four_tier) == 0.0, "halluc_math_treecut"


# ---------------------------------------------------------------------------
# format gate and malformed inputs
# ---------------------------------------------------------------------------


class TestFormatGateAndMalformedInputs:
    SOURCE = "halluc_math_mip"
    GT = gt_json(solvable=False, has_diagnosis_label=False)

    def test_missing_think_block_scores_zero(self):
        assert compute_score(self.SOURCE, "\\boxed{UNSOLVABLE}", self.GT)["score"] == 0.0

    def test_empty_think_block_scores_zero(self):
        assert compute_score(self.SOURCE, "<think></think>\n\n\\boxed{UNSOLVABLE}", self.GT)["score"] == 0.0

    def test_unterminated_think_block_scores_zero(self):
        assert compute_score(self.SOURCE, "<think>hmm\n\n\\boxed{UNSOLVABLE}", self.GT)["score"] == 0.0

    def test_truncated_rollout_without_response_scores_zero(self):
        assert compute_score(self.SOURCE, "<think>hmm</think>\n\n", self.GT)["score"] == 0.0

    def test_bad_json_scores_zero(self):
        assert score(self.SOURCE, "\\boxed{UNSOLVABLE}", "{not json") == 0.0

    def test_json_that_is_not_an_object_scores_zero(self):
        assert score(self.SOURCE, "\\boxed{UNSOLVABLE}", "[1, 2, 3]") == 0.0

    def test_missing_solvable_key_scores_zero(self):
        assert score(self.SOURCE, "\\boxed{UNSOLVABLE}", gt_json(answer="42")) == 0.0

    def test_ground_truth_as_a_dict_is_accepted(self):
        bare = {"solvable": False, "has_diagnosis_label": False}
        assert score_from_status(bare, STATUS_UNSOLVABLE_BARE, None, None) == 1.0

    def test_pathological_response_does_not_raise(self):
        weird = "<think>x</think>\n\n\\boxed{{{{{{" + "[" * 500
        assert compute_score(self.SOURCE, weird, self.GT)["score"] == 0.0


# ---------------------------------------------------------------------------
# decision table, branch by branch, without building a solution string
# ---------------------------------------------------------------------------


#: Ground truths shared by the decision table (module level: the parametrize
#: decorator is evaluated while the class body is still being built).
FREE_TEXT_GT = {"solvable": True, "solvable_answer": True, "answer": "a teacher"}
PAIR_GT = {"solvable": True, "pair_task": True, "answerable_id": "B", "answer": 42}
FOUR_TIER_GT = {"solvable": False, "has_diagnosis_label": True, "correct_option_id": "B"}


class TestDecisionTable:
    @pytest.mark.parametrize(
        ("ground_truth", "status", "answer", "option", "expected"),
        [
            # solvable numeric (misrefusal is 0, not -1)
            ({"solvable": True, "answer": "72"}, STATUS_ANSWER, "72", None, 1.0),
            ({"solvable": True, "answer": "72"}, STATUS_ANSWER, "7", None, 0.0),
            ({"solvable": True, "answer": "72"}, STATUS_UNSOLVABLE_BARE, None, None, 0.0),
            ({"solvable": True, "answer": "72"}, STATUS_UNSOLVABLE_OPTION, None, "B", 0.0),
            ({"solvable": True, "answer": "72"}, STATUS_NONE, None, None, 0.0),
            # solvable two-layer numeric (D24)
            ({"solvable": True, "two_layer": True, "answer": "42"}, STATUS_ANSWER, "42", None, 1.0),
            ({"solvable": True, "two_layer": True, "answer": "42"}, STATUS_ANSWER, "43", None, 0.5),
            ({"solvable": True, "two_layer": True, "answer": "42"}, STATUS_UNSOLVABLE_BARE, None, None, 0.0),
            ({"solvable": True, "two_layer": True, "answer": "42"}, STATUS_SOLVABLE_MARKER, None, None, 0.0),
            # solvable two-layer free text (D27)
            (FREE_TEXT_GT, STATUS_ANSWER, "Teacher!", None, 1.0),
            (FREE_TEXT_GT, STATUS_ANSWER, "a student", None, 0.5),
            (FREE_TEXT_GT, STATUS_SOLVABLE_MARKER, None, None, 0.5),
            (FREE_TEXT_GT, STATUS_UNSOLVABLE_BARE, None, None, 0.0),
            # SUM pair task (D23)
            (PAIR_GT, STATUS_ANSWER, "B: 42", None, 1.0),
            (PAIR_GT, STATUS_ANSWER, "B: 43", None, 0.5),
            (PAIR_GT, STATUS_ANSWER, "A: 42", None, 0.0),
            (PAIR_GT, STATUS_ANSWER, "42", None, 0.0),
            (PAIR_GT, STATUS_UNSOLVABLE_BARE, None, None, 0.0),
            # solvable judgment-only
            ({"solvable": True, "judgment_only": True}, STATUS_SOLVABLE_MARKER, None, None, 1.0),
            ({"solvable": True, "judgment_only": True}, STATUS_UNSOLVABLE_BARE, None, None, -1.0),
            ({"solvable": True, "judgment_only": True}, STATUS_UNSOLVABLE_OPTION, None, "B", -1.0),
            ({"solvable": True, "judgment_only": True}, STATUS_ANSWER, "42", None, 0.0),
            ({"solvable": True, "judgment_only": True}, STATUS_NONE, None, None, 0.0),
            # unsolvable three-tier
            ({"solvable": False, "has_diagnosis_label": False}, STATUS_UNSOLVABLE_BARE, None, None, 1.0),
            ({"solvable": False, "has_diagnosis_label": False}, STATUS_UNSOLVABLE_OPTION, None, "A", 1.0),
            ({"solvable": False, "has_diagnosis_label": False}, STATUS_ANSWER, "5", None, -1.0),
            ({"solvable": False, "has_diagnosis_label": False}, STATUS_NONE, None, None, 0.0),
            ({"solvable": False, "has_diagnosis_label": False}, STATUS_SOLVABLE_MARKER, None, None, 0.0),
            # unsolvable four-tier
            (FOUR_TIER_GT, STATUS_UNSOLVABLE_OPTION, None, "B", 1.0),
            (FOUR_TIER_GT, STATUS_UNSOLVABLE_OPTION, None, "A", 0.0),
            (FOUR_TIER_GT, STATUS_UNSOLVABLE_BARE, None, None, 0.0),
            (FOUR_TIER_GT, STATUS_ANSWER, "1", None, -1.0),
            (FOUR_TIER_GT, STATUS_NONE, None, None, 0.0),
        ],
    )
    def test_cell(self, ground_truth, status, answer, option, expected):
        assert score_from_status(ground_truth, status, answer, option) == expected


# ---------------------------------------------------------------------------
# stage-1 delegation
# ---------------------------------------------------------------------------


class TestDelegation:
    def test_stage_one_sources_are_delegated_verbatim(self):
        """A mixed parquet must score its stage-1 rows exactly as before."""
        import compute_score as base

        cases = [
            ("math_dapo", "\\boxed{42}", "42"),
            ("math_dapo", "\\boxed{41}", "42"),
            ("logic_synlogic", "\\boxed{A,C,D,E}", "A,C,D,E"),
            ("logic_synlogic", "\\boxed{A,C,D}", "A,C,D,E"),
            ("stem_drsci", "\\boxed{7}", "7"),
        ]
        for data_source, solution, ground_truth in cases:
            response = think_wrap(solution)
            expected = base.compute_score(data_source, response, ground_truth)
            assert compute_score(data_source, response, ground_truth) == expected, data_source

    def test_format_gate_matches_stage_one(self):
        """A malformed response scores 0 in stage 1 too, so a mixed batch agrees."""
        import compute_score as base

        for response in ("\\boxed{42}", "<think></think>\n\n\\boxed{42}", "<think>x</think>\n\n"):
            assert (
                compute_score("math_dapo", response, "42")["score"]
                == base.compute_score("math_dapo", response, "42")["score"]
            )

    def test_halluc_math_gsmic_uses_the_math_branch(self):
        """GSM-IC rows are plain numeric solvable rows: same matcher, same cells."""
        gt = gt_json(solvable=True, answer="72", has_diagnosis_label=False)
        assert score("halluc_math_gsmic", "\\boxed{72}", gt) == 1.0
        assert score("halluc_math_gsmic", "\\boxed{71}", gt) == 0.0
        assert score("halluc_math_gsmic", "\\boxed{UNSOLVABLE}", gt) == 0.0

    def test_unknown_halluc_source_still_scores(self):
        """Routing is ground-truth driven, so a new source needs no code change."""
        gt = gt_json(solvable=False, has_diagnosis_label=False)
        assert score("halluc_commonsense_brandnew", "\\boxed{UNSOLVABLE}", gt) == 1.0

    def test_kwargs_reach_the_stage_one_dispatcher(self):
        """reward_kwargs (sandbox URL, ...) must survive the wrapper."""
        import hallucination_compute_score as module

        captured = {}

        def fake_base(*args, **kwargs):
            captured["args"] = args
            captured.update(kwargs)
            return {"score": 0.0}

        original = module._base_compute_score
        module._base_compute_score = fake_base
        try:
            compute_score("math_dapo", think_wrap("\\boxed{1}"), "1", {"task_id": "x"}, sandbox_fusion_url="http://x")
        finally:
            module._base_compute_score = original
        assert captured.get("sandbox_fusion_url") == "http://x"
        assert captured["args"][3] == {"task_id": "x"}

    def test_halluc_rows_never_reach_stage_one(self):
        """``halluc_*`` is not a stage-1 prefix; the base dispatcher would raise."""
        import hallucination_compute_score as module

        def exploding_base(*args, **kwargs):
            raise AssertionError("halluc rows must not be delegated")

        original = module._base_compute_score
        module._base_compute_score = exploding_base
        try:
            gt = gt_json(solvable=False, has_diagnosis_label=False)
            assert score("halluc_math_mip", "\\boxed{UNSOLVABLE}", gt) == 1.0
        finally:
            module._base_compute_score = original


# ---------------------------------------------------------------------------
# math_match directly
# ---------------------------------------------------------------------------


class TestMathMatch:
    def test_exact(self):
        assert math_match("540", "540") is True

    def test_trailing_zero(self):
        assert math_match("540.0", "540") is True

    def test_wrong(self):
        assert math_match("541", "540") is False

    def test_none_prediction(self):
        assert math_match(None, "540") is False

    def test_empty_ground_truth(self):
        assert math_match("540", "") is False

    def test_pathological_prediction_does_not_raise(self):
        assert math_match("\\" * 400 + "{", "540") is False
