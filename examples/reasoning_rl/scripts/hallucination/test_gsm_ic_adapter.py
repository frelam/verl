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
"""Tests for ``gsm_ic_adapter.py`` and the audit in ``verify_gsmic.py``.

Every fixture is inline and tiny.  The source rows are copied verbatim out of
``scratch/halluc_recon/gsmic_report.md`` (which quotes the raw files byte for
byte), together with the matching GSM8K cross-check solutions; the variants
built on top of them exist only to drive one filter branch each and are marked
``# synthetic``.  **Nothing here reads
``/home/charles/data/reasoning_rl/halluc/raw``** -- the fixture directory is
written into ``tmp_path``.

The fixture 2step array is laid out so that every funnel stage drops something
and every drop reason has exactly one owner:

====  =============================================  ===========================
idx   row                                              outcome
====  =============================================  ===========================
0     Steve (verbatim)                                 kept
1     Jewel (verbatim)                                 kept
2     Charles, comma-grouped answer (verbatim)         kept
3     Luke, role-only template / number = n/a          kept
4     James, number-only template / role = n/a         kept
5     Jose, inserted number == gold (verbatim)         dropped: read-off
6-10  5 Steve variants (synthetic)                      kept, then trimmed by the cap
11    Steve with the distractor removed (synthetic)     dropped: replay
12    invented base question (synthetic)                dropped: gold certificate
13    Steve with a wrong answer (synthetic)            dropped: gold certificate
14    Hannah (verbatim)                                kept
15    Hannah duplicate new_question (verbatim)         dropped: duplicate prompt
====  =============================================  ===========================
"""

from __future__ import annotations

import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gsm_ic_adapter as ga  # noqa: E402
import schema  # noqa: E402
import verify_gsmic as vg  # noqa: E402

# ---------------------------------------------------------------------------
# verbatim fixture rows (recon report section 3 + the raw files it quotes)
# ---------------------------------------------------------------------------

STEVE = {
    "original_question": 'Steve is 5\'6".  He grows 6 inches.  How tall is he in inches?',
    "answer": "72",
    "new_question": 'Steve is 5\'6". He grows 6 inches. The height of Emma is 8 feet. '
    "How tall is Steve in inches?",
    "n_steps": 2,
    "role": "Emma",
    "number": "8",
    "sentence_template": "The height of {role} is {number} feet.",
    "role_label": "nonoverlapped",
    "number_label": "in_range",
    "sentence_label": "in_topic",
}

JEWEL = {
    "original_question": (
        "A magazine costs $3 each. Jewel bought 10 magazines to be sold at $3.50 each. "
        "How much will be Jewel gain from selling these?"
    ),
    "answer": "5",
    "new_question": (
        "A magazine costs $3 each. Jewel bought 10 magazines to be sold at $3.50 each. "
        "Jewel's neighbor bought 1000 newspapers. How much will Jewel gain from selling her magazines?"
    ),
    "n_steps": 2,
    "role": "Jewel's neighbor",
    "number": "1000",
    "sentence_template": "{role} bought {number} newspapers.",
    "role_label": "overlapped",
    "number_label": "out_range",
    "sentence_label": "in_topic",
}

# 2step row 78: 640 rows in the source carry a comma-grouped answer, which
# ``float()`` rejects -- the adapter normalises it to a comma-free digit string.
CHARLES = {
    "original_question": (
        "Charles is moving from Springfield, which has 482,653 people, to Greenville, "
        "which has 119,666 fewer people. What is the total population of Springfield "
        "and Greenville?"
    ),
    "answer": "845,640",
    "new_question": (
        "Charles is moving from Springfield, which has 482,653 people, to Greenville, "
        "which has 119,666 fewer people. Emma is 174285 years old. What is the total "
        "population of Springfield and Greenville?"
    ),
    "n_steps": 2,
    "role": "Emma",
    "number": "174285",
    "sentence_template": "{role} is {number} years old.",
    "role_label": "nonoverlapped",
    "number_label": "in_range",
    "sentence_label": "out_topic",
}

# 2step row 661: a {role}-only template, so ``number`` is the literal "n/a" and
# ``number_label`` is "n/a" too (60 such rows in the source).
LUKE = {
    "original_question": (
        "85 paper stars are required to fill a glass jar. Luke has already made 33 stars, "
        "but he needs to fill 4 bottles. How many more stars must Luke make?"
    ),
    "answer": "307",
    "new_question": (
        "85 paper stars are required to fill a glass jar. Luke has already made 33 stars, "
        "but he needs to fill 4 bottles. Luke's mother has some time, but does not wish "
        "to help Luke make stars. How many more stars must Luke make?"
    ),
    "n_steps": 2,
    "role": "Luke's mother",
    "number": "n/a",
    "sentence_template": "{role} has some time, but does not wish to help Luke make stars.",
    "role_label": "overlapped",
    "number_label": "n/a",
    "sentence_label": "in_topic",
}

# 2step row 11: a {number}-only template, so ``role`` is "n/a".
JAMES = {
    "original_question": (
        "James collects all the fruits from his 2 trees.  Each tree has 20 plants.  "
        "Each plant has 1 seed and he plants 60% of those.  How many trees did he plant?"
    ),
    "answer": "24",
    "new_question": (
        "James collects all the fruits from his 2 trees. Each tree has 20 plants. "
        "Each plant has 1 seed and he plants 60% of those. Each seed needs 10 liters "
        "of water per day. How many trees did James plant?"
    ),
    "n_steps": 2,
    "role": "n/a",
    "number": "10",
    "sentence_template": "Each seed needs {number} liters of water per day.",
    "role_label": "n/a",
    "number_label": "in_range",
    "sentence_label": "in_topic",
}

# 2step row 32: the inserted number ("9") *is* the gold answer, so "copy the
# number from the sentence that does not belong" would be a second admissible
# answer; 1,258 source rows look like this and the adapter drops them.
JOSE = {
    "original_question": (
        "Jose needs 12 tablespoons of lemon juice to make a dozen of his lemon cupcakes.  "
        "Every lemon provides 4 tablespoons of lemon juice.  If he needs to make 3 dozen "
        "cupcakes, how many lemons will he need?"
    ),
    "answer": "9",
    "new_question": (
        "Jose needs 12 tablespoons of lemon juice to make a dozen of his lemon cupcakes. "
        "Every lemon provides 4 tablespoons of lemon juice. The shoe size of Mary is 9.. "
        "If he needs to make 3 dozen cupcakes, how many lemons will he need?"
    ),
    "n_steps": 2,
    "role": "Mary",
    "number": "9",
    "sentence_template": "The shoe size of {role} is {number}.",
    "role_label": "nonoverlapped",
    "number_label": "in_range",
    "sentence_label": "out_topic",
}

HANNAH = {
    "original_question": (
        "Hannah has three brothers. Her brothers are all 8 years old. How old is Hannah "
        "if she is twice as old as the sum of her brother's ages?"
    ),
    "answer": "48",
    "new_question": (
        "Hannah has three brothers. Her brothers are all 8 years old. Emma is 25 years old. "
        "How old is Hannah if she is twice as old as the sum of her brother's ages?"
    ),
    "n_steps": 2,
    "role": "Emma",
    "number": "25",
    "sentence_template": "{role} is {number} years old.",
    "role_label": "nonoverlapped",
    "number_label": "in_range",
    "sentence_label": "out_topic",
}

# The source's own duplicate: 2step rows 91 and 1243 share this new_question
# verbatim (80 such groups).  Only the sentence_label differs, so the presented
# prompt text is a literal duplicate.
HANNAH_DUPLICATE = dict(HANNAH, sentence_label="in_topic")

MSTEP = {
    "original_question": (
        "Officer Hopps has to give out 200 tickets in May. The first 15 days he averages "
        "8 tickets a day. How many does he have to average each day for the rest of the "
        "month to reach his required goal?"
    ),
    "answer": "5",
    "new_question": (
        "Officer Hopps has to give out 200 tickets in May. The first 15 days he averages "
        "8 tickets a day. Officer Hopps' mother bought 200 bus tickets in Feburary. "
        "How many does he have to average each day for the rest of the month to reach "
        "his required goal?"
    ),
    "n_steps": 4,
    "role": "Officer Hopps' mother",
    "number": "200",
    "sentence_template": "{role} bought {number} bus tickets in Feburary.",
    "role_label": "overlapped",
    "number_label": "in_range",
    "sentence_label": "in_topic",
}

# ---------------------------------------------------------------------------
# synthetic variants -- one per filter branch, never presented as source rows
# ---------------------------------------------------------------------------


def steve_variant(role: str, number: str) -> dict:
    """A Steve variant with a different injected role/number (``# synthetic``)."""
    return dict(
        STEVE,
        new_question=(
            'Steve is 5\'6". He grows 6 inches. The height of '
            f"{role} is {number} feet. How tall is Steve in inches?"
        ),
        role=role,
        number=number,
    )


# Synthetic: the distractor sentence was never spliced in, so replay cannot trace
# the row back to its template.
STEVE_NO_DISTRACTOR = dict(
    STEVE,
    new_question='Steve is 5\'6". He grows 6 inches. How tall is Steve in inches?',
)

# Synthetic: an original question the GSM8K cross-check corpus does not contain.
UNKNOWN_BASE = {
    "original_question": "A question that the GSM8K cross-check corpus does not contain?",
    "answer": "1",
    "new_question": (
        "A question that the GSM8K cross-check corpus does not contain? Emma is 3 years old."
    ),
    "n_steps": 2,
    "role": "Emma",
    "number": "3",
    "sentence_template": "{role} is {number} years old.",
    "role_label": "nonoverlapped",
    "number_label": "in_range",
    "sentence_label": "out_topic",
}

# Synthetic: replay is clean but the stored answer contradicts the recomputed
# chain (the GSM8K terminal value is 72).
STEVE_WRONG_ANSWER = dict(
    STEVE,
    answer="73",
    new_question=(
        'Steve is 5\'6". He grows 6 inches. The height of Oliver is 8 feet. '
        "How tall is Steve in inches?"
    ),
    role="Oliver",
)

TWO_STEP = [
    STEVE,  # 0
    JEWEL,  # 1
    CHARLES,  # 2
    LUKE,  # 3
    JAMES,  # 4
    JOSE,  # 5  dropped: the inserted number is the gold
    steve_variant("Emma", "12"),  # 6
    steve_variant("Oliver", "3"),  # 7
    steve_variant("Sophia", "50"),  # 8
    steve_variant("Mia", "700"),  # 9
    steve_variant("Noah", "11"),  # 10
    STEVE_NO_DISTRACTOR,  # 11 dropped: replay
    UNKNOWN_BASE,  # 12 dropped: gold certificate
    STEVE_WRONG_ANSWER,  # 13 dropped: gold certificate
    HANNAH,  # 14
    HANNAH_DUPLICATE,  # 15 dropped: duplicate presented question
]

# Question text -> the full GSM8K solution used as the independent certificate.
# Copied verbatim from ``gsm8k_train.jsonl`` for the seven base questions above.
GSM8K_CORPUS = {
    STEVE["original_question"]: (
        "He is 5*12+6=<<5*12+6=66>>66 inches tall before the growth spurt.\n"
        "After growing he is now 66+6=<<66+6=72>>72 inches\n"
        "#### 72"
    ),
    JEWEL["original_question"]: (
        "Jewel's gain for each magazine is $3.50 - $3 = $<<3.5-3=0.50>>0.50.\n"
        "Thus, her total gain will be $0.50 x 10 = $<<0.50*10=5>>5.\n"
        "#### 5"
    ),
    CHARLES["original_question"]: (
        "Greenville has 482,653 - 119,666 = <<482653-119666=362987>>362,987 people.\n"
        "So, the total population of Springfield and Greenville is "
        "482,653 + 362,987 = <<482653+362987=845640>>845,640.\n"
        "#### 845,640"
    ),
    LUKE["original_question"]: (
        "Luke must make 85 x 4 = <<85*4=340>>340 stars in total.\n"
        "He needs to make another 340 - 33 = <<340-33=307>>307 stars.\n"
        "#### 307"
    ),
    JAMES["original_question"]: (
        "He got 20*2=<<20*2=40>>40 seeds\n"
        "That means he plants 40*.6=<<40*.6=24>>24 trees\n"
        "#### 24"
    ),
    JOSE["original_question"]: (
        "He needs 12 tablespoons of lemon juice for every dozen of cupcakes and he’s "
        "making 3 dozen cupcakes so he needs 12*3 = <<12*3=36>>36 tablespoons of lemon juice\n"
        "1 lemon provides 4 tablespoons of lemon juice and he needs 36 tablespoons so "
        "he will need 36/4 = <<36/4=9>>9 lemons\n"
        "#### 9"
    ),
    HANNAH["original_question"]: (
        "The total of Hannah's brothers' ages is 3 * 8 = <<3*8=24>>24.\n"
        "Hannah is twice as old as the total of her brothers' ages, so Hannah is "
        "24 * 2 = <<24*2=48>>48 years old.\n"
        "#### 48"
    ),
    MSTEP["original_question"]: (
        "He has given out 120 tickets because 15 x 8 = <<15*8=120>>120\n"
        "He has 16 days left to had out tickets because 31 - 15 = <<31-15=16>>16\n"
        "He has to give out 80 more because 200 - 120 = <<200-120=80>>80\n"
        "He has to give out 5 a day because 80 / 16 = <<80/16=5>>5\n"
        "#### 5"
    ),
}

# Expected funnel for the fixture above, with ``MAX_PER_BASE_QUESTION`` at its
# default (8).  Every stage drops something, and the two gold-certificate drops
# (unknown base + wrong answer) land in the same stage.
EXPECTED_FUNNEL = {
    "raw_rows": 17,
    "template_replay_verified": 16,
    "gold_certified": 14,
    "distractor_number_not_gold": 13,
    "presented_question_unique": 12,
    "per_base_capped": 12,
    "limit_applied": 12,
}

# The measured pool of the current raw bundle: 100 distinct base questions
# (2step 60 + mstep 40) x the 8-per-base cap = 800 rows, a small buffer over the
# 772-row quota design doc section 4.8 table B row 1 assigns to GSM-IC after D27.
DOCUMENTED_CAP = 8
DOCUMENTED_BASES = 100
DOCUMENTED_POOL = 800

# The seven base questions that survive; every one is a distinct problem.
SURVIVING_BASES = (
    STEVE["original_question"],
    JEWEL["original_question"],
    CHARLES["original_question"],
    LUKE["original_question"],
    JAMES["original_question"],
    HANNAH["original_question"],
    MSTEP["original_question"],
)


def _write_corpus(directory) -> None:
    """Both cross-check files: the loaders require the pair to exist."""
    with (directory / "gsm8k_train.jsonl").open("w", encoding="utf-8") as handle:
        for question, solution in GSM8K_CORPUS.items():
            handle.write(json.dumps({"question": question, "answer": solution}) + "\n")
    (directory / "gsm8k_test.jsonl").write_text("", encoding="utf-8")


@pytest.fixture()
def raw_dir(tmp_path):
    """A miniature raw directory: both source arrays plus the cross-check corpus."""
    directory = tmp_path / "raw"
    directory.mkdir()
    (directory / "GSM-IC_2step.json").write_text(json.dumps(TWO_STEP), encoding="utf-8")
    (directory / "GSM-IC_mstep.json").write_text(json.dumps([MSTEP]), encoding="utf-8")
    _write_corpus(directory)
    return str(directory)


@pytest.fixture()
def built(raw_dir):
    """The fixture rows, built once."""
    rows, funnel = ga.build_rows(raw_dir, seed=0)
    return rows, funnel


@pytest.fixture()
def corpus(raw_dir):
    return vg.load_gsm8k(raw_dir)


def _count_by_base(rows) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        base = row["extra_info"]["paired_original_text"]
        counts[base] = counts.get(base, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# helpers / arithmetic certificate
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_normalise_collapses_the_source_double_spaces(self):
        assert ga.normalise('Steve is 5\'6".  He grows 6 inches.') == (
            'Steve is 5\'6". He grows 6 inches.'
        )
        assert ga.normalise("  a \n\t b  ") == "a b"

    def test_evaluate_expression_covers_the_measured_grammar(self):
        assert ga.evaluate_expression("5*12+6") == 66.0
        assert ga.evaluate_expression("80 / 16") == 5.0
        assert ga.evaluate_expression("31 - 15") == 16.0
        assert ga.evaluate_expression("+7") == 7.0
        assert ga.evaluate_expression("-(3 + 4)") == -7.0
        assert ga.evaluate_expression("$3.50 - 3") == pytest.approx(0.5)
        assert ga.evaluate_expression("482,653 - 119,666") == 362987.0

    @pytest.mark.parametrize("expression", ["2**3", "2%3", "x + 1", "7 // 2", "", "2 +"])
    def test_evaluate_expression_rejects_anything_outside_the_whitelist(self, expression):
        with pytest.raises(ga.AnnotationError):
            ga.evaluate_expression(expression)

    def test_evaluate_expression_rejects_division_by_zero(self):
        with pytest.raises(ga.AnnotationError):
            ga.evaluate_expression("1/0")

    def test_evaluate_expression_rejects_a_non_numeric_constant(self):
        with pytest.raises(ga.AnnotationError):
            ga.evaluate_expression("'a'")

    @pytest.mark.parametrize(
        "text,expected",
        [("72", 72.0), ("845,640", 845640.0), ("$3.50", 3.5), ("number=1000", 1000.0)],
    )
    def test_parse_number(self, text, expected):
        assert ga.parse_number(text) == expected

    def test_parse_number_returns_none_without_a_number(self):
        assert ga.parse_number("n/a") is None
        assert ga.parse_number("") is None

    def test_annotation_pairs_and_distractor_text(self):
        assert ga.annotation_pairs("a <<1+1=2>> b <<2+2=4>>") == [("1+1", "2"), ("2+2", "4")]
        assert ga.distractor_text(STEVE) == "The height of Emma is 8 feet."
        assert ga.distractor_text(LUKE) == (
            "Luke's mother has some time, but does not wish to help Luke make stars."
        )
        # A role-only template leaves the number half of the replay untouched.
        assert ga.distractor_text(JAMES) == "Each seed needs 10 liters of water per day."

    def test_format_number_reads_like_prose(self):
        assert ga._format_number(5.0) == "5"
        assert ga._format_number(0.6) == "0.6"

    def test_gold_is_derived_accepts_a_literal_outside_the_annotations(self):
        # ``<<210/350=0.60>>0.60 or 60% off`` closes with ``#### 60``: 60 is a
        # literal in the derivation, not an annotation value (3,480 source rows).
        solution = "She saved $\\<<210/350=0.60>>0.60 or 60% off.\n#### 60"
        assert ga.gold_is_derived(solution, 60.0) is True
        assert ga.gold_is_derived(solution, 61.0) is False

    def test_gold_is_derived_accepts_a_chain_value(self):
        assert ga.gold_is_derived(GSM8K_CORPUS[STEVE["original_question"]], 72.0) is True
        assert ga.gold_is_derived(GSM8K_CORPUS[STEVE["original_question"]], 999.0) is False

    def test_calculator_rounding_accepts_two_decimal_arithmetic(self):
        # 40*.6 is 24.000000000000004 in binary floating point; the corpus stores
        # the rounded 24.  Exact float comparison would reject this row.
        index = {
            ga.normalise(JAMES["original_question"]): GSM8K_CORPUS[JAMES["original_question"]]
        }
        assert ga.certify_gold(JAMES["original_question"], "24", index) is None


class TestGoldCertificate:
    def test_certifies_the_verbatim_source_rows(self):
        index = {ga.normalise(q): a for q, a in GSM8K_CORPUS.items()}
        for record in (STEVE, JEWEL, CHARLES, LUKE, JAMES, HANNAH, MSTEP):
            assert (
                ga.certify_gold(record["original_question"], ga._gold_answer(record), index)
                is None
            )

    def test_comma_grouped_answer_is_normalised_before_comparison(self):
        index = {
            ga.normalise(CHARLES["original_question"]): GSM8K_CORPUS[CHARLES["original_question"]]
        }
        assert ga._gold_answer(CHARLES) == "845640"
        assert ga.certify_gold(CHARLES["original_question"], "845640", index) is None

    def test_fails_when_the_question_is_not_in_the_corpus(self):
        reason = ga.certify_gold("Nowhere question?", "1", {})
        assert reason is not None and "cross-check corpus" in reason

    def test_fails_when_the_answer_disagrees_with_the_chain(self):
        index = {ga.normalise(STEVE["original_question"]): GSM8K_CORPUS[STEVE["original_question"]]}
        reason = ga.certify_gold(STEVE["original_question"], "73", index)
        assert reason is not None and "disagrees" in reason

    def test_fails_without_any_annotation(self):
        index = {ga.normalise("Q"): "just prose\n#### 1"}
        reason = ga.certify_gold("Q", "1", index)
        assert reason is not None and "annotation" in reason

    def test_fails_on_a_wrong_annotation_value(self):
        index = {ga.normalise("Q"): "1+1 = <<1+1=3>>3\n#### 3"}
        reason = ga.certify_gold("Q", "3", index)
        assert reason is not None and "recomputes to" in reason

    def test_fails_on_an_unevaluable_annotation(self):
        index = {ga.normalise("Q"): "x = <<x+1=3>>3\n#### 3"}
        reason = ga.certify_gold("Q", "3", index)
        assert reason is not None and "not evaluable" in reason

    def test_fails_with_multiple_final_markers(self):
        index = {ga.normalise("Q"): "1+1 = <<1+1=2>>2\n#### 2\n#### 3"}
        reason = ga.certify_gold("Q", "2", index)
        assert reason is not None and "####" in reason

    def test_fails_when_the_answer_is_not_numeric(self):
        index = {ga.normalise("Q"): "1+1 = <<1+1=2>>2\n#### 2"}
        reason = ga.certify_gold("Q", "two", index)
        assert reason is not None and "not a plain number" in reason

    def test_fails_when_the_answer_is_never_derived(self):
        index = {ga.normalise("Q"): "1+1 = <<1+1=2>>2\n#### 5"}
        reason = ga.certify_gold("Q", "5", index)
        assert reason is not None and "neither the chain values" in reason

    def test_missing_cross_check_corpus_is_a_hard_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ga.load_gsm8k_index(str(tmp_path))


class TestRowFilters:
    def test_replay_accepts_the_source_rows(self):
        for record in (STEVE, JEWEL, CHARLES, LUKE, JAMES, HANNAH, MSTEP):
            assert ga._passes_replay({"record": record}) is None

    def test_replay_rejects_a_missing_distractor(self):
        reason = ga._passes_replay({"record": STEVE_NO_DISTRACTOR})
        assert reason is not None and "absent" in reason

    def test_replay_rejects_a_template_without_placeholders(self):
        record = dict(STEVE, sentence_template="A sentence with no placeholder.")
        reason = ga._passes_replay({"record": record})
        assert reason is not None and "no placeholder" in reason

    def test_replay_rejects_a_distractor_already_in_the_original(self):
        record = dict(STEVE, original_question=STEVE["new_question"])
        reason = ga._passes_replay({"record": record})
        assert reason is not None and "already occurs" in reason

    def test_replay_rejects_an_empty_replayed_sentence(self):
        # Both placeholders expand to nothing, so there is no sentence to replay.
        record = dict(STEVE, sentence_template="{role} {number}", role="", number="")
        reason = ga._passes_replay({"record": record})
        assert reason is not None

    def test_readoff_drops_the_row_whose_number_is_the_gold(self):
        assert ga._passes_no_readoff({"record": JOSE}) is not None
        assert ga._passes_no_readoff({"record": STEVE}) is None

    def test_readoff_tolerates_the_placeholder_number(self):
        # LUKE carries the literal "n/a": it is not a number, so it cannot be the
        # gold and the row stays.
        assert ga._passes_no_readoff({"record": LUKE}) is None

    def test_select_for_base_caps_and_uses_a_stable_order(self):
        candidates = [{"record": record} for record in [STEVE] * 5]
        chosen = ga._select_for_base(candidates, 3, random.Random(0))
        assert len(chosen) == 3
        assert chosen == ga._select_for_base(candidates, 3, random.Random(0))

    def test_select_for_base_returns_everything_below_the_cap(self):
        candidates = [{"record": STEVE}]
        assert len(ga._select_for_base(candidates, ga.MAX_PER_BASE_QUESTION, random.Random(0))) == 1

    def test_the_per_base_cap_is_the_documented_eight(self):
        """The cap is the knob; the 800-row pool is 100 bases x 8, not an input."""
        assert ga.MAX_PER_BASE_QUESTION == DOCUMENTED_CAP
        assert DOCUMENTED_BASES * ga.MAX_PER_BASE_QUESTION == DOCUMENTED_POOL

    def test_load_source_numbers_rows_with_their_file_and_ordinal(self, raw_dir):
        candidates = ga.load_source(raw_dir)
        assert [c["file"] for c in candidates[:2]] == ["GSM-IC_2step.json"] * 2
        assert [c["ordinal"] for c in candidates[:2]] == [0, 1]
        assert len(candidates) == len(TWO_STEP) + 1
        assert candidates[-1]["file"] == "GSM-IC_mstep.json"
        assert candidates[-1]["ordinal"] == 0


# ---------------------------------------------------------------------------
# build_rows: the funnel, one stage at a time
# ---------------------------------------------------------------------------


class TestBuildRows:
    def test_funnel_reproduces_the_expected_per_stage_counts(self, built):
        _rows, funnel = built
        assert list(funnel) == list(EXPECTED_FUNNEL)
        assert funnel == EXPECTED_FUNNEL

    def test_every_stage_is_a_prefix_of_the_raw_count_and_monotone(self, built):
        _rows, funnel = built
        counts = list(funnel.values())
        assert counts[0] == EXPECTED_FUNNEL["raw_rows"]
        assert all(later <= earlier for earlier, later in zip(counts, counts[1:], strict=False))

    def test_row_counts_match_the_last_funnel_stage(self, built):
        rows, funnel = built
        assert len(rows) == funnel["limit_applied"]

    def test_the_kept_bases_are_the_expected_ones(self, built):
        rows, _funnel = built
        bases = {row["extra_info"]["paired_original_text"] for row in rows}
        for question in SURVIVING_BASES:
            assert ga.normalise(question) in bases
        # ... and the dropped ones are not.
        assert ga.normalise(JOSE["original_question"]) not in bases
        assert ga.normalise(UNKNOWN_BASE["original_question"]) not in bases

    def test_dropped_rows_really_are_absent(self, built):
        rows, _funnel = built
        answers = {json.loads(row["reward_model"]["ground_truth"])["answer"] for row in rows}
        assert "9" not in answers  # JOSE's read-off row
        assert "73" not in answers  # the wrong-answer row
        assert "1" not in answers  # the unknown-base row
        presented = [row["prompt"][0]["content"] for row in rows]
        assert len(presented) == len(set(presented))  # the duplicate prompt is gone

    def test_steve_contributes_every_variant_below_the_cap(self, built):
        rows, _funnel = built
        counts = _count_by_base(rows)
        assert counts[ga.normalise(STEVE["original_question"])] == 6  # original + 5 variants

    def test_per_base_cap_is_applied(self, raw_dir, monkeypatch):
        monkeypatch.setattr(ga, "MAX_PER_BASE_QUESTION", 2)
        rows, funnel = ga.build_rows(raw_dir, seed=0)
        # Steve owns 6 rows in the fixture; the cap keeps 2 of them.
        counts = _count_by_base(rows)
        assert counts[ga.normalise(STEVE["original_question"])] == 2
        assert funnel["per_base_capped"] == 8  # 2 from Steve + 1 for each of 6 other bases
        assert funnel["per_base_capped"] == len(rows)

    def test_cap_one_keeps_exactly_one_row_per_base(self, raw_dir, monkeypatch):
        monkeypatch.setattr(ga, "MAX_PER_BASE_QUESTION", 1)
        rows, funnel = ga.build_rows(raw_dir, seed=0)
        assert funnel["per_base_capped"] == 7  # seven surviving base questions
        assert set(_count_by_base(rows).values()) == {1}

    @pytest.mark.skipif(
        not os.path.isdir(ga.DEFAULT_RAW_DIR),
        reason="raw GSM-IC bundle not downloaded; the measured pool cannot be rebuilt",
    )
    def test_documented_pool_is_100_bases_times_the_cap(self):
        """The full build yields the documented pool: 100 bases x 8 = 800 rows."""
        rows, funnel = ga.build_rows(ga.DEFAULT_RAW_DIR, seed=0)
        assert funnel["per_base_capped"] == DOCUMENTED_POOL
        counts = _count_by_base(rows)
        assert len(counts) == DOCUMENTED_BASES
        assert set(counts.values()) == {DOCUMENTED_CAP}
        assert len(rows) == DOCUMENTED_POOL

    def test_the_interleave_keeps_a_truncated_build_diverse(self, raw_dir):
        rows, _funnel = ga.build_rows(raw_dir, seed=0)
        # The first seven rows are one per base question, not seven Steve rows.
        assert len(_count_by_base(rows[:7])) == 7

    def test_limit_truncates_and_stays_a_prefix_of_a_larger_build(self, raw_dir):
        full, _ = ga.build_rows(raw_dir, seed=0)
        short, funnel = ga.build_rows(raw_dir, limit=3, seed=0)
        assert len(short) == 3
        assert funnel["limit_applied"] == 3
        assert json.dumps(short, sort_keys=True) == json.dumps(full[:3], sort_keys=True)

    def test_limit_zero_is_empty(self, raw_dir):
        rows, funnel = ga.build_rows(raw_dir, limit=0, seed=0)
        assert rows == []
        assert funnel["limit_applied"] == 0

    def test_build_is_deterministic(self, raw_dir):
        first, _ = ga.build_rows(raw_dir, seed=7)
        second, _ = ga.build_rows(raw_dir, seed=7)
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)

    def test_seed_changes_the_sample_but_not_the_shape(self, raw_dir):
        first, funnel_one = ga.build_rows(raw_dir, seed=0)
        second, funnel_two = ga.build_rows(raw_dir, seed=1)
        assert len(first) == len(second)
        assert funnel_one == funnel_two

    def test_empty_source_files_produce_an_empty_build(self, tmp_path):
        directory = tmp_path / "empty"
        directory.mkdir()
        (directory / "GSM-IC_2step.json").write_text("[]", encoding="utf-8")
        (directory / "GSM-IC_mstep.json").write_text("[]", encoding="utf-8")
        _write_corpus(directory)
        rows, funnel = ga.build_rows(str(directory))
        assert rows == []
        assert set(funnel.values()) == {0}

    def test_a_jsonl_source_file_is_rejected(self, tmp_path):
        (tmp_path / "GSM-IC_2step.json").write_text('{"a": 1}', encoding="utf-8")
        with pytest.raises(ValueError):
            ga.load_source(str(tmp_path))


# ---------------------------------------------------------------------------
# the emitted row
# ---------------------------------------------------------------------------


class TestRowShape:
    def test_every_row_passes_schema_validation(self, built):
        rows, _funnel = built
        for row in rows:
            assert schema.validate_row(row) == []

    def test_branch_template_and_data_source(self, built):
        rows, _funnel = built
        for row in rows:
            info = row["extra_info"]
            assert row["data_source"] == schema.SOURCE_GSMIC
            assert row["ability"] == "math"
            assert info["branch"] == schema.BRANCH_SOLVABLE_NUMERIC
            assert info["template"] == schema.TEMPLATE_B
            # ``source`` and ``domain`` are derived by ``schema.make_row`` from
            # the ``data_source`` (mix_halluc's train_by_source reads the former),
            # so the adapter never has to set them itself.
            assert info["source"] == "GSM-IC"
            assert info["domain"] == "math"

    def test_ground_truth_payload(self, built):
        rows, _funnel = built
        for row in rows:
            payload = json.loads(row["reward_model"]["ground_truth"])
            assert payload["solvable"] is True
            assert payload["answer"]
            assert payload["correct_option_id"] is None
            assert payload["has_diagnosis_label"] is False
            assert payload["perturbation_type"] == "distracting_condition"
            assert "judgment_only" not in payload  # not a judgment row
            assert "role_words" not in payload  # not a role-word row

    def test_no_options_block_anywhere(self, built):
        rows, _funnel = built
        for row in rows:
            assert row["extra_info"]["options"] == []
            assert row["extra_info"]["correct_option_id"] == ""
            assert "选项" not in row["prompt"][0]["content"]

    def test_solvable_row_uses_template_b_with_the_unsolvable_escape_hatch(self, built):
        rows, _funnel = built
        content = rows[0]["prompt"][0]["content"]
        assert "\\boxed{UNSOLVABLE}" in content
        assert "\\boxed{你的最终答案}" in content

    def test_task_id_is_stable_and_unique(self, built):
        rows, _funnel = built
        task_ids = [row["extra_info"]["task_id"] for row in rows]
        assert len(task_ids) == len(set(task_ids))
        assert all(task_id.startswith("gsmic:") for task_id in task_ids)
        # The id is the source's own identity, not a counter over the build: the
        # Steve rows keep their 2step ordinals 0, 6, 7, 8, 9 and 10.
        assert "gsmic:GSM-IC_2step:0" in task_ids
        assert "gsmic:GSM-IC_2step:6" in task_ids
        assert "gsmic:GSM-IC_mstep:0" in task_ids

    def test_extra_info_bookkeeping(self, built):
        rows, _funnel = built
        assert [row["extra_info"]["index"] for row in rows] == list(range(len(rows)))
        for row in rows:
            info = row["extra_info"]
            assert info["split"] == "train"
            assert info["seed"] == 0
            assert isinstance(info["difficulty"], str) and info["difficulty"].startswith("n_steps=")
            assert info["paired_original_text"]
            assert info["distractor_text"]
            assert info["perturbation_type"] == "distracting_condition"
            assert set(info["distractor_labels"]) == {
                "role_label",
                "number_label",
                "sentence_label",
            }

    def test_the_prompt_question_is_the_paired_new_question(self, built):
        rows, _funnel = built
        faces = {ga.normalise(row["prompt"][0]["content"].split("\n\n")[0]) for row in rows}
        sources = {ga.normalise(record["new_question"]) for record in TWO_STEP + [MSTEP]}
        assert faces <= sources

    def test_role_only_row_records_only_the_role(self, built):
        rows, _funnel = built
        luke = next(
            row
            for row in rows
            if row["extra_info"]["paired_original_text"] == ga.normalise(LUKE["original_question"])
        )
        assert luke["extra_info"]["perturbed_entity_text"] == "role=Luke's mother"
        assert luke["extra_info"]["distractor_labels"]["number_label"] == "n/a"

    def test_number_only_row_records_only_the_number(self, built):
        rows, _funnel = built
        james = next(
            row
            for row in rows
            if row["extra_info"]["paired_original_text"] == ga.normalise(JAMES["original_question"])
        )
        assert james["extra_info"]["perturbed_entity_text"] == "number=10"
        assert james["extra_info"]["distractor_labels"]["role_label"] == "n/a"

    def test_comma_grouped_gold_is_stored_comma_free(self, built):
        rows, _funnel = built
        charles = next(
            row
            for row in rows
            if row["extra_info"]["paired_original_text"] == ga.normalise(CHARLES["original_question"])
        )
        assert json.loads(charles["reward_model"]["ground_truth"])["answer"] == "845640"

    def test_difficulty_carries_the_source_native_step_count(self, built):
        rows, _funnel = built
        difficulties = {row["extra_info"]["difficulty"] for row in rows}
        assert "n_steps=2" in difficulties
        assert "n_steps=4" in difficulties  # the mstep row

    def test_describe_reports_every_axis(self, built):
        rows, _funnel = built
        text = ga.describe(rows)
        for axis in ("branch", "template", "solvable", "difficulty", "distinct base questions"):
            assert axis in text
        assert f"distinct task_id: {len(rows)}" in text

    def test_funnel_lines_show_each_stage(self, built):
        _rows, funnel = built
        text = ga._funnel_lines(funnel)
        for stage in funnel:
            assert stage in text


# ---------------------------------------------------------------------------
# verify_gsmic: the audit, and the audit's ability to fail
# ---------------------------------------------------------------------------


class TestVerifyHelpers:
    def test_recompute_annotation(self):
        assert vg.recompute_annotation("5*12+6") == 66.0
        with pytest.raises(vg.AnnotationError):
            vg.recompute_annotation("x+1")

    def test_plain_number(self):
        assert vg.plain_number("845,640 people") == 845640.0
        assert vg.plain_number("n/a") is None

    def test_theorem_gold_accepts_a_source_solution(self):
        ok, detail = vg.theorem_gold(GSM8K_CORPUS[STEVE["original_question"]], 72.0)
        assert ok, detail

    def test_theorem_gold_reports_why_it_fails(self):
        ok, detail = vg.theorem_gold(GSM8K_CORPUS[STEVE["original_question"]], 99.0)
        assert not ok and "terminal" in detail
        ok, detail = vg.theorem_gold("no annotations here\n#### 3", 3.0)
        assert not ok and "annotation" in detail

    def test_balanced_accuracy_is_the_mean_of_recalls(self):
        assert vg.balanced_accuracy([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
        assert vg.balanced_accuracy([0, 0, 1, 1], [1, 1, 0, 0]) == 0.0
        assert vg.balanced_accuracy([0, 0, 1, 1], [0, 0, 0, 0]) == pytest.approx(0.5)

    def test_balanced_accuracy_is_nan_for_a_single_class(self):
        import math

        assert math.isnan(vg.balanced_accuracy([1, 1], [1, 1]))

    def test_positive_control_solves_a_separable_task(self):
        assert vg._positive_control() >= vg.NB_POSITIVE_CONTROL_MIN

    def test_nb_is_near_chance_on_labelled_noise(self):
        rng = random.Random(0)
        words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
        texts = [" ".join(rng.sample(words, 4)) for _ in range(200)]
        labels = [rng.randint(0, 1) for _ in range(200)]
        accuracy, vocab = vg.bow_nb_out_of_fold(texts, labels, seed=0)
        assert vocab == len(words)  # every token clears the support floor
        assert 0.30 <= accuracy <= 0.70

    def test_nb_vocabulary_filter_excludes_rare_tokens(self):
        # "common" appears in 12 documents; each "rare" token in exactly one, so a
        # support floor of 5 keeps only the former.
        texts = ["common"] * 12 + [f"rare{index}" for index in range(8)]
        labels = [index % 2 for index in range(len(texts))]
        _accuracy, vocab = vg.bow_nb_out_of_fold(texts, labels, seed=0, min_doc_support=5)
        assert vocab == 1

    def test_nb_falls_back_to_the_prior_with_an_empty_vocabulary(self):
        # Every token appears in one document, so the support floor empties the
        # vocabulary on all five folds.  The estimator then predicts each fold's
        # training majority; with 40/10 imbalance every fold is majority-0, so the
        # balanced accuracy is exactly the 0.50 of a constant classifier.
        texts = [f"uniquetoken{index}" for index in range(50)]
        labels = [0] * 40 + [1] * 10
        accuracy, vocab = vg.bow_nb_out_of_fold(texts, labels, seed=0, min_doc_support=5)
        assert vocab == 0
        assert accuracy == pytest.approx(0.5)

    def test_inserted_number_reads_the_recorded_field(self):
        assert vg._inserted_number({"perturbed_entity_text": "role=Emma; number=8"}) == 8.0
        assert vg._inserted_number({"perturbed_entity_text": "role=Emma"}) is None
        assert vg._inserted_number({"perturbed_entity_text": "number=12"}) == 12.0
        # A digit inside the role name must not be mistaken for the number.
        assert vg._inserted_number({"perturbed_entity_text": "role=Emma 2; number=8"}) == 8.0

    def test_prompt_question_strips_the_instruction(self, built):
        rows, _funnel = built
        question = vg._prompt_question(rows[0])
        assert "\\boxed" not in question
        assert question in vg.normalise(rows[0]["prompt"][0]["content"])
        assert question == ga.normalise(rows[0]["prompt"][0]["content"].split("\n\n")[0])


class TestVerifyAudit:
    def test_audit_passes_on_a_real_build(self, built, corpus):
        rows, _funnel = built
        result = vg.audit(rows, corpus, sample=12, seed=0)
        result.flush()
        assert result.failures == 0

    def test_audit_proves_the_requested_sample_size(self, built, corpus):
        rows, _funnel = built
        result = vg.audit(rows, corpus, sample=12, seed=0)
        assert any("proved 12/12" in line for line in result.lines)

    def test_audit_clamps_a_sample_larger_than_the_build(self, built, corpus):
        rows, _funnel = built
        result = vg.audit(rows, corpus, sample=len(rows) + 5, seed=0)
        assert result.failures == 0
        assert any(f"proved {len(rows)}/{len(rows)}" in line for line in result.lines)

    def test_audit_fails_on_a_corrupted_distractor_text(self, built, corpus):
        rows, _funnel = built
        corrupted = json.loads(json.dumps(rows))
        corrupted[0]["extra_info"]["distractor_text"] = "A sentence that is not in the prompt."
        result = vg.audit(corrupted, corpus, sample=len(corrupted), seed=0)
        assert result.failures > 0

    def test_audit_fails_on_a_stale_gold(self, built, corpus):
        rows, _funnel = built
        corrupted = json.loads(json.dumps(rows))
        payload = json.loads(corrupted[0]["reward_model"]["ground_truth"])
        payload["answer"] = "999999"
        corrupted[0]["reward_model"]["ground_truth"] = json.dumps(payload)
        result = vg.audit(corrupted, corpus, sample=len(corrupted), seed=0)
        assert result.failures > 0

    def test_audit_fails_when_a_schema_key_is_missing(self, built, corpus):
        rows, _funnel = built
        corrupted = json.loads(json.dumps(rows))
        del corrupted[0]["extra_info"]["task_id"]
        result = vg.audit(corrupted, corpus, sample=1, seed=0)
        assert result.failures > 0

    def test_audit_reports_the_single_sided_source(self, built, corpus):
        rows, _funnel = built
        result = vg.audit(rows, corpus, sample=4, seed=0)
        text = "\n".join(result.lines)
        assert "INAPPLICABLE" in text
        assert "single-sided" in text
        assert "vacuous" in text
        assert "reference, not gated" in text

    def test_audit_does_not_gate_the_surrogate_readings(self, built, corpus):
        rows, _funnel = built
        result = vg.audit(rows, corpus, sample=4, seed=0)
        reference_lines = [line for line in result.lines if "reference, not gated" in line]
        # All three difficulty axes are binary in the fixture, so all three carry
        # a surrogate reading; none of them may flip the exit status.
        assert len(reference_lines) == 3
        assert result.failures == 0


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_main_writes_a_valid_readable_parquet(self, raw_dir, tmp_path, monkeypatch, capsys):
        out = tmp_path / "built" / "gsmic.parquet"
        monkeypatch.setattr(
            sys, "argv", ["gsm_ic_adapter.py", "--raw-dir", raw_dir, "--out", str(out)]
        )
        ga.main()
        captured = capsys.readouterr().out
        assert "funnel:" in captured
        assert "branch:" in captured
        assert out.is_file()

        rows = schema.read_parquet_rows(str(out))
        assert len(rows) == EXPECTED_FUNNEL["limit_applied"]
        for row in rows:
            assert schema.validate_row(row) == []

    def test_main_honours_limit_and_seed(self, raw_dir, tmp_path, monkeypatch, capsys):
        out = tmp_path / "gsmic.parquet"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "gsm_ic_adapter.py",
                "--raw-dir",
                raw_dir,
                "--limit",
                "4",
                "--seed",
                "3",
                "--out",
                str(out),
            ],
        )
        ga.main()
        assert "limit_applied" in capsys.readouterr().out
        assert len(schema.read_parquet_rows(str(out))) == 4

    def test_main_refuses_to_write_nothing(self, tmp_path, monkeypatch):
        directory = tmp_path / "empty"
        directory.mkdir()
        (directory / "GSM-IC_2step.json").write_text("[]", encoding="utf-8")
        (directory / "GSM-IC_mstep.json").write_text("[]", encoding="utf-8")
        _write_corpus(directory)
        monkeypatch.setattr(
            sys,
            "argv",
            ["gsm_ic_adapter.py", "--raw-dir", str(directory), "--out", str(tmp_path / "x.parquet")],
        )
        with pytest.raises(SystemExit):
            ga.main()
        assert not (tmp_path / "x.parquet").exists()

    def test_verify_cli_exits_zero_on_the_built_parquet(self, raw_dir, tmp_path, monkeypatch, capsys):
        out = tmp_path / "gsmic.parquet"
        rows, _funnel = ga.build_rows(raw_dir, seed=0)
        schema.normalise_extra_info(rows)
        schema.write_rows_parquet(rows, str(out))
        monkeypatch.setattr(
            sys,
            "argv",
            ["verify_gsmic.py", "--rows", str(out), "--raw-dir", raw_dir, "--sample", "12"],
        )
        vg.main()
        assert "all checks passed" in capsys.readouterr().out

    def test_verify_cli_exits_nonzero_on_a_corrupted_parquet(self, raw_dir, tmp_path, monkeypatch):
        out = tmp_path / "gsmic.parquet"
        rows, _funnel = ga.build_rows(raw_dir, seed=0)
        for row in rows:
            payload = json.loads(row["reward_model"]["ground_truth"])
            payload["answer"] = "0"
            row["reward_model"]["ground_truth"] = json.dumps(payload)
        schema.normalise_extra_info(rows)
        schema.write_rows_parquet(rows, str(out))
        monkeypatch.setattr(
            sys,
            "argv",
            ["verify_gsmic.py", "--rows", str(out), "--raw-dir", raw_dir, "--sample", "5"],
        )
        with pytest.raises(SystemExit):
            vg.main()

    def test_verify_cli_rejects_an_empty_parquet(self, tmp_path, monkeypatch, raw_dir):
        import pyarrow as pa
        import pyarrow.parquet as pq

        out = tmp_path / "empty.parquet"
        pq.write_table(pa.table({"data_source": pa.array([], type=pa.string())}), str(out))
        monkeypatch.setattr(
            sys, "argv", ["verify_gsmic.py", "--rows", str(out), "--raw-dir", raw_dir]
        )
        with pytest.raises(SystemExit):
            vg.main()

    def test_verify_cli_reports_a_missing_parquet(self, tmp_path, monkeypatch, raw_dir):
        monkeypatch.setattr(
            sys,
            "argv",
            ["verify_gsmic.py", "--rows", str(tmp_path / "nope.parquet"), "--raw-dir", raw_dir],
        )
        with pytest.raises(FileNotFoundError):
            vg.main()
