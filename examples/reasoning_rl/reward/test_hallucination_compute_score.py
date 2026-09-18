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

The six-branch x three-case matrix of HALLUCINATION_RL_DESIGN.md section 9 is
spread over the ``Test*`` classes below; each branch gets a correct-output cell
(``+1``), a wrong-direction cell (``-1``) and an unparseable cell (``0``).

DESIGN CONFLICTS
----------------

Three places in the design document contradict themselves; the module docstring
of ``hallucination_compute_score.py`` records the resolution and the tests below
pin that resolution down so it cannot drift silently:

1. Misrefusal (``\\boxed{UNSOLVABLE}`` on a solvable row) is ``-1`` on **every**
   solvable branch.  Section 6's pseudocode returns 0 on the numeric branch;
   section 9's matrix lists it as the ``-1`` cell there and on the role-word
   branch.  See ``TestSolvableNumeric.test_misrefusal_is_penalised``.
2. ``halluc_math_gsmic`` gets no special case -- numeric solvable rows all route
   through ``math_match``, so GSM-IC is scored like every other numeric pool.
   See ``TestDelegation.test_halluc_math_gsmic_uses_the_math_branch``.
3. ``\\boxed{UNSOLVABLE: B}`` on a three-tier row is ``+1`` (doc sections 5.1 and
   6) rather than the 0 that section 9's matrix implies.
   See ``TestUnsolvableThreeTier.test_spurious_option_id_still_scores``.
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
        # Only the whole boxed text decides; prose inside an answer stays an answer.
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


# ---------------------------------------------------------------------------
# branch 1/2: solvable, numeric (GSM-IC, SUM-answerable, distractor-synthesised)
# ---------------------------------------------------------------------------


class TestSolvableNumeric:
    SOURCE = "halluc_math_gsmic"
    GT = gt_json(solvable=True, answer="72", has_diagnosis_label=False, perturbation_type="distracting_condition")

    def test_correct_answer_scores(self):
        assert score(self.SOURCE, "\\boxed{72}", self.GT) == 1.0

    def test_wrong_answer_scores_zero(self):
        assert score(self.SOURCE, "\\boxed{71}", self.GT) == 0.0

    def test_misrefusal_is_penalised(self):
        # DESIGN CONFLICT 1: doc section 9 lists this as the -1 cell.
        assert score(self.SOURCE, "\\boxed{UNSOLVABLE}", self.GT) == -1.0

    def test_misrefusal_with_option_is_penalised(self):
        assert score(self.SOURCE, "\\boxed{UNSOLVABLE: B}", self.GT) == -1.0

    def test_misrefusal_with_solvable_marker_is_zero(self):
        # \boxed{SOLVABLE} on a plain numeric row is not a verdict the row asked
        # for, and it is not an answer either: neither +1 nor -1.
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


# ---------------------------------------------------------------------------
# branch 3: solvable, role-word sequence (K&K) -- the reward-inversion defence
# ---------------------------------------------------------------------------


class TestSolvableRoleWords:
    SOURCE = "halluc_logic_kk"

    def test_canonical_words(self):
        gt = gt_json(solvable=True, answer="knight knave", role_words=["knight", "knave"], has_diagnosis_label=False)
        assert score(self.SOURCE, "\\boxed{knight knave}", gt) == 1.0
        assert score(self.SOURCE, "\\boxed{knave knight}", gt) == 0.0

    def test_flip_role_inversion_defence(self):
        """flip_role rows write "knaves always tell the truth": the *surface* words
        are swapped, so a hard-coded K = knight would score truth as falsehood."""
        gt = gt_json(solvable=True, answer="knave knave", role_words=["knave", "knight"], has_diagnosis_label=False)
        assert score(self.SOURCE, "\\boxed{knave knave}", gt) == 1.0
        # The canonical words are the *wrong* answer on this row.
        assert score(self.SOURCE, "\\boxed{knight knight}", gt) == 0.0

    def test_random_pair_inversion_defence(self):
        gt = gt_json(solvable=True, answer="angel devil devil", role_words=["angel", "devil"], has_diagnosis_label=False)
        assert score(self.SOURCE, "\\boxed{angel devil devil}", gt) == 1.0
        assert score(self.SOURCE, "\\boxed{devil angel angel}", gt) == 0.0

    def test_plurals_and_articles(self):
        gt = gt_json(solvable=True, answer="knight knave", role_words=["knight", "knave"], has_diagnosis_label=False)
        assert score(self.SOURCE, "\\boxed{a knight, a knave}", gt) == 1.0
        assert score(self.SOURCE, "\\boxed{knights knaves}", gt) == 1.0

    def test_k_n_literals_rejected(self):
        # Design doc section 12 Q17: the canonical K/N shorthand is not the contract.
        gt = gt_json(solvable=True, answer="knight knave", role_words=["knight", "knave"], has_diagnosis_label=False)
        assert score(self.SOURCE, "\\boxed{K N}", gt) == 0.0

    def test_wrong_length_rejected(self):
        gt = gt_json(solvable=True, answer="knight knave", role_words=["knight", "knave"], has_diagnosis_label=False)
        assert score(self.SOURCE, "\\boxed{knight}", gt) == 0.0
        assert score(self.SOURCE, "\\boxed{knight knave knave}", gt) == 0.0

    def test_prose_answer_rejected(self):
        # Names are out-of-vocabulary tokens: the format is part of the contract.
        gt = gt_json(solvable=True, answer="knight knave", role_words=["knight", "knave"], has_diagnosis_label=False)
        assert score(self.SOURCE, "\\boxed{Ethan is a knight and Abigail is a knave}", gt) == 0.0

    def test_misrefusal_is_penalised(self):
        gt = gt_json(solvable=True, answer="knight knave", role_words=["knight", "knave"], has_diagnosis_label=False)
        assert score(self.SOURCE, "\\boxed{UNSOLVABLE}", gt) == -1.0

    def test_missing_role_words_fails_closed(self):
        # An adapter that forgot role_words must not silently score with an
        # assumed knight/knave mapping (design doc section 11 risk 9).
        gt = gt_json(solvable=True, answer="knave knave", has_diagnosis_label=False)
        assert kk_match("knave knave", None, "knave knave") is False


# ---------------------------------------------------------------------------
# branch 4: solvable, judgment-only (FalseQA-real D14, CREPE-normal, KUQ-known)
# ---------------------------------------------------------------------------


class TestSolvableJudgmentOnly:
    SOURCE = "halluc_commonsense_falseqa"
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
# branch 5: unsolvable, three-tier bare (MiP, SUM-deletion, TreeCut, UMWP cat1,
#           CREPE-false-presupposition, KUQ-unknown)
# ---------------------------------------------------------------------------


class TestUnsolvableThreeTier:
    SOURCE = "halluc_math_mip"
    GT = gt_json(solvable=False, answer=None, correct_option_id=None, has_diagnosis_label=False)

    def test_bare_refusal_scores(self):
        assert score(self.SOURCE, "\\boxed{UNSOLVABLE}", self.GT) == 1.0

    def test_spurious_option_id_still_scores(self):
        # DESIGN CONFLICT 3: doc 5.1/6 say +1, doc 9's matrix implies 0.
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
            "halluc_math_sum",
            "halluc_math_treecut",
            "halluc_math_umwp",
            "halluc_commonsense_crepe",
            "halluc_commonsense_kuq",
        ):
            assert score(source, "\\boxed{UNSOLVABLE}", self.GT) == 1.0, source
            assert score(source, "\\boxed{42}", self.GT) == -1.0, source


# ---------------------------------------------------------------------------
# branch 6: unsolvable, four-tier diagnostic (FalseQA-fake, SUM-visible,
#           UMWP-visible)
# ---------------------------------------------------------------------------


class TestUnsolvableFourTier:
    SOURCE = "halluc_commonsense_falseqa"
    GT = gt_json(
        solvable=False,
        answer=None,
        correct_option_id="B",
        has_diagnosis_label=True,
        perturbation_type="contradictory_condition",
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
            options=[{"id": "A", "text": "space"}, {"id": "B", "text": "gases"}, {"id": "C", "text": "Confucius"}],
        )
        assert score(self.SOURCE, "\\boxed{UNSOLVABLE: C}", shuffled) == 1.0
        assert score(self.SOURCE, "\\boxed{UNSOLVABLE: B}", shuffled) == 0.0


# ---------------------------------------------------------------------------
# the bare-marker branch correlation (doc section 9)
# ---------------------------------------------------------------------------


class TestBareMarkerBranchCorrelation:
    def test_same_string_two_scores(self):
        three_tier = gt_json(solvable=False, has_diagnosis_label=False)
        four_tier = gt_json(solvable=False, has_diagnosis_label=True, correct_option_id="B")
        assert score("halluc_math_sum", "\\boxed{UNSOLVABLE}", three_tier) == 1.0
        assert score("halluc_math_sum", "\\boxed{UNSOLVABLE}", four_tier) == 0.0


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
        assert score_from_status({"solvable": False, "has_diagnosis_label": False}, STATUS_UNSOLVABLE_BARE, None, None) == 1.0

    def test_pathological_response_does_not_raise(self):
        weird = "<think>x</think>\n\n\\boxed{{{{{{" + "[" * 500
        assert compute_score(self.SOURCE, weird, self.GT)["score"] == 0.0


# ---------------------------------------------------------------------------
# decision table, branch by branch, without building a solution string
# ---------------------------------------------------------------------------


class TestDecisionTable:
    @pytest.mark.parametrize(
        ("ground_truth", "status", "answer", "option", "expected"),
        [
            # solvable numeric
            ({"solvable": True, "answer": "72"}, STATUS_ANSWER, "72", None, 1.0),
            ({"solvable": True, "answer": "72"}, STATUS_ANSWER, "7", None, 0.0),
            ({"solvable": True, "answer": "72"}, STATUS_UNSOLVABLE_BARE, None, None, -1.0),
            ({"solvable": True, "answer": "72"}, STATUS_NONE, None, None, 0.0),
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
            ({"solvable": False, "has_diagnosis_label": True, "correct_option_id": "B"}, STATUS_UNSOLVABLE_OPTION, None, "B", 1.0),
            ({"solvable": False, "has_diagnosis_label": True, "correct_option_id": "B"}, STATUS_UNSOLVABLE_OPTION, None, "A", 0.0),
            ({"solvable": False, "has_diagnosis_label": True, "correct_option_id": "B"}, STATUS_UNSOLVABLE_BARE, None, None, 0.0),
            ({"solvable": False, "has_diagnosis_label": True, "correct_option_id": "B"}, STATUS_ANSWER, "1", None, -1.0),
            ({"solvable": False, "has_diagnosis_label": True, "correct_option_id": "B"}, STATUS_NONE, None, None, 0.0),
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
        # DESIGN CONFLICT 2: no special case; solvable numeric rows are all scored
        # by math_match, so the misrefusal penalty applies here too.
        gt = gt_json(solvable=True, answer="72", has_diagnosis_label=False)
        assert score("halluc_math_gsmic", "\\boxed{72}", gt) == 1.0
        assert score("halluc_math_gsmic", "\\boxed{UNSOLVABLE}", gt) == -1.0

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
