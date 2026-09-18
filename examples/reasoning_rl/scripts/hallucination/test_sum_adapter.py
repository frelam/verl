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
"""Tests for ``sum_adapter.py``.

The fixtures are real ``train.parquet`` rows, copied verbatim -- the LaTeX, the
escaped backslashes and the odd source spacing are exactly what SUM ships, so the
tests exercise the same text shapes as the real build.  Each fixture carries its
source row index in a comment so a failure can be taken back to the frozen file.
Fixtures are written to ``tmp_path`` in the source's own parquet layout and never
to the downloaded data directory, so the suite is self-contained; the two
``TestRealSource`` tests at the end read the real file and skip when it is absent.

Covers every emitted branch, every funnel stage and its drop reason, the
certificates and their rejections, the five-type classifier (including the
priority order), determinism, ``--limit``, the CLI and an integration call into
``verify_sum.py``.
"""

from __future__ import annotations

import copy
import json
import os
import random
import sys

import pytest

import schema
import sum_adapter as adapter

# ---------------------------------------------------------------------------
# fixtures -- verbatim SUM rows, as (answerable, unanswerable, ground_truth)
# ---------------------------------------------------------------------------

# source row 49: the rewrite swaps "CE" for "EF", an entity the original never
# mentioned -- the residual class, and a pair that also yields a judgment row
MEDIANS = (
    'Medians AD and CE of \\(\\triangle ABC\\) intersect in M. The midpoint of AE is N. '
    'Let the area of \\(\\triangle MNE\\) be k times the area of \\(\\triangle ABC\\). Then k equals:',
    'Medians AD and EF of \\(\\triangle ABC\\) intersect in M. The midpoint of AE is N. '
    'Let the area of \\(\\triangle MNE\\) be k times the area of \\(\\triangle ABC\\). Then k equals:',
    '$\\frac{1}{3}$',
)

# source row 55: the question clause is rewritten into "equals what value?"
FORALL = (
    'For all non-zero numbers x and y such that x = 1/y, '
    '\\(\\left(x-\\frac{1}{x}\\right)\\left(y+\\frac{1}{y}\\right)\\) equals',
    'For all non-zero numbers x and y, '
    '\\(\\left(x-\\frac{1}{x}\\right)\\left(y+\\frac{1}{y}\\right)\\) equals what value?',
    '2',
)

# source row 290: a visible defect whose answerable member cannot host an option
# block, so the pair emits a diag row only (no judgment row)
NUMBERS = (
    'How many numbers between 1 and 2005 are integer multiples of 3 or 4 but not 12?',
    'How many numbers between 1 and 2005 satisfy the property?',
    '502',
)

# source row 51: "7" is deleted and does not survive anywhere in the question
SEDANS = (
    'On average, for every 4 sports cars sold at the local dealership, 7 sedans are sold. '
    'The dealership predicts that it will sell 28 sports cars next month. '
    'How many sedans does it expect to sell?',
    'On average, for every 4 sports cars sold at the local dealership, sedans are sold. '
    'The dealership predicts that it will sell 28 sports cars next month. '
    'How many sedans does it expect to sell?',
    '49',
)

# source row 67: a whole premise clause is deleted
TRIANGLE = (
    'The sides of a triangle have lengths 6.5, 10, and s, where s is a whole number. '
    'What is the smallest possible value of s?',
    'The sides of a triangle have lengths 6.5, 10, and s. What is the smallest possible value of s?',
    '4',
)

# source row 35: the deleted span is the question's own object, so the question
# clause was cut -- the `question_missing` bare type
HARMONIC = (
    'The harmonic mean of a set of non-zero numbers is the reciprocal of the average of the '
    'reciprocals of the numbers. What is the harmonic mean of 1, 2, and 4?',
    'The harmonic mean of a set of non-zero numbers is the reciprocal of the average of the '
    'reciprocals of the numbers. What is the harmonic mean?',
    '\\frac{12}{7}',
)

# source row 2570: the inserted span carries the impossibility word "negative"
PRIME_P = (
    'What expression is never a prime number when $p$ is a prime number?',
    'What expression is never a prime number when $p$ is a prime number, '
    'and $p$ is a negative decimal number?',
    '$p^2+26$',
)

# source row 443: the inserted span is the vague quantifier "some"
TENNIS = (
    'The longest professional tennis match ever played lasted a total of 11 hours and 5 minutes. '
    'How many minutes was this?',
    'The longest professional tennis match ever played lasted a total of 11 hours and some minutes. '
    'How many minutes was this?',
    '665',
)

# source row 2352: the inserted span fires *both* the impossible rule ("negative")
# and the vague rule ("some"); the priority order must label it unrealistic
SLOPE_AB = (
    'If y=a+\\frac{b}{x}, where a and b are constants, and if y=1 when x=-1, and y=5 when x=-5, '
    'then a+b equals:',
    'If y = a + \\frac{b}{x}, where a and b are constants, and if y = 1 when x = -1 and '
    'y = 5 when x is some other negative number, then a+b equals:',
    '11',
)

# source row 7: three separate regions -- the model would have to pick one defect
SOFTBALL = (
    'During the softball season, Judy had 35 hits. Among her hits were 1 home run, 1 triple and '
    '5 doubles. The rest of her hits were single. What percent of her hits were single?',
    'During the softball season, Judy got several hits. Among her hits were 1 home run, 1 triple, '
    'and 5 doubles. The rest of her hits were singles. What percent of her hits were singles?',
    '\\frac{5}{7}',
)

# source row 805: the only change is the operator y != 0 -> y > 0, invisible to a
# word tokenizer -- the operator-only class of the 116 no-diff rows
REMAINDER = (
    'The remainder can be defined for all real numbers x and y with y ≠ 0 by '
    'rem(x, y) = x - y ⌊x/y⌋ where ⌊x/y⌋ denotes the greatest integer less than or equal to x/y. '
    'What is the value of rem(3/8, -2/5)?',
    'The remainder can be defined for all real numbers x and y with y > 0 by '
    'rem(x, y) = x - y ⌊x/y⌋ where ⌊x/y⌋ denotes the greatest integer less than or equal to x/y. '
    'What is the value of rem(3/8, -2/5)?',
    '-\\frac{1}{40}',
)

# source row 2225: "then:" -> "then?" is a punctuation-only change
UNEQUAL = (
    'If $a$ and $b$ are two unequal positive numbers, then:',
    'If a and b are two unequal positive numbers, then?',
    '\\frac {a + b}{2} > \\sqrt {ab} > \\frac {2ab}{a + b}',
)

# source row 50: the inserted "c" is a single character, too short to point at
MIN_C = (
    'Find the minimum value of \\(\\sqrt{x^2+y^2}\\) if \\(5x+12y=60\\).',
    'Find the minimum value of \\(\\sqrt{x^2+y^2}\\) if \\(5x+12y = c\\).',
    '5',
)

# source row 175: the inserted "2112-" already occurs inside the original number
FACTOR = (
    'What is the value of \\(\\frac{(2112-2021)^2}{169}\\)?',
    'What is the value of \\(\\frac{(2112- )^2}{169}\\)?',
    '9',
)

# source row 31: the inserted span cannot host three equal-length option spans
GP = (
    'If x, 2x+2, 3x+3, ... are in geometric progression, the fourth term is:',
    'If x, 2x+2, 3x+3, ... are in geometric progression with the common ratio equal to the '
    'number of apples in an orchard, what is the fourth term?',
    '4x+4',
)

# source row 404: a deletion-only pair where every deleted word still stands
# elsewhere in the question -- nothing was lost
SOUP = (
    'A can of soup can feed 3 adults or 5 children. If there are 5 cans of soup and 15 children '
    'are fed, then how many adults would the remaining soup feed?',
    'A can of soup can feed 3 adults. If there are 5 cans of soup and 15 children are fed, then '
    'how many adults would the remaining soup feed?',
    '3',
)

# source row 35212: the rewrite recasts the imperative as a question -- it deletes
# "Simplify" and inserts "How", which is not a defect, not vague and not an
# impossible value, and which introduces no content word the original lacked.  The
# old certificate pointed the model at "How" as the thing that broke the question;
# the defect-class clause rejects it.  The deleted "Simplify" is a content word
# absent from the variant, so the refusal certificate would hold -- which is what
# makes this fixture the test that a rejected pair is *dropped*, not rebuilt as a
# bare row (the pair's shape, not its certificate, decides the branch).
NON_DEFECT = (
    'Simplify $2w+4w+6w+8w+10w+12$.',
    '2w+4w+6w+8w+10w+12. How?',
    '30w+12',
)

# source rows 436 / 2528: the same text pair twice with golds "2" and "3"
MONTH_A = (
    'A month with 31 days has the same number of Mondays and Wednesdays. How many of the seven '
    'days of the week could be the first day of this month?',
    'A month has the same number of Mondays and Wednesdays. How many of the seven days of the '
    'week could be the first day of this month?',
    '2',
)
MONTH_B_CONFLICT = (*MONTH_A[:2], '3')

# source rows 184 / 2609: the same text pair twice with the same gold "15"
BUG_A = (
    'A bug crawls along a number line, starting at -2. It crawls to -6, then turns around and '
    'crawls to 5. How many units does the bug crawl altogether?',
    'A bug crawls along a number line. It crawls to -6, then turns around and crawls to 5. '
    'How many units does the bug crawl altogether?',
    '15',
)
BUG_B_DUPLICATE = BUG_A

#: The clean half of the suite: every branch, every defect type, every certificate.
CLEAN = [
    MEDIANS,
    FORALL,
    NUMBERS,
    SEDANS,
    TRIANGLE,
    HARMONIC,
    PRIME_P,
    TENNIS,
    SLOPE_AB,
]

#: Rows that must be dropped, one fixture per funnel stage that drops them.
DROPS = [
    SOFTBALL,  # multi-region
    REMAINDER,  # no word-level change (operator only)
    UNEQUAL,  # no word-level change (punctuation only)
    MIN_C,  # pointer: gold shorter than MIN_GOLD_CHARS
    FACTOR,  # pointer: gold already occurs in the answerable member
    GP,  # pointer: no k=3 equal-length option block
    NON_DEFECT,  # pointer: the insertion is not a defect any rule names
    SOUP,  # refusal: the deleted content word survives in the question
]

#: Pair-level duplicates.
DUPLICATES = [MONTH_A, MONTH_B_CONFLICT, BUG_A, BUG_B_DUPLICATE]

_FIELDS = ("answerable_question", "unanswerable_question", "ground_truth")

#: every fixture (14 + 4 rows) read at once, in this order
ALL = [*CLEAN, *DROPS, *DUPLICATES]

#: the funnel of those 21 raw rows, stage by stage
EXPECTED_FUNNEL = {
    "raw_rows": 21,
    "after_malformed_drop": 21,
    "after_duplicate_pair_drop": 18,
    "after_no_word_diff_drop": 16,
    "after_multi_region_drop": 15,
    "after_defect_certificate_drop": 10,
    "rows_before_limit": 13,
    "after_limit": 13,
}

#: the pair-level rejections of that funnel, by the clause that rejected them
#: (the nested ``certificate_drops`` mapping; judge clauses can reject a pair's
#: judgment row without costing the pair's diag row)
EXPECTED_CERTIFICATE_DROPS = {
    "pointer_gold_too_short": 1,
    "pointer_gold_in_answerable_member": 1,
    "pointer_question_cannot_host_options": 1,
    "pointer_insertion_is_not_a_defect": 1,
    "refusal_lost_no_content_word": 1,
    "judge_no_deleted_anchor": 1,
    "judge_question_cannot_host_options": 2,
}

#: the task ids the clean fixtures alone produce (module scope: several tests use it)
CLEAN_ROW_IDS = {"sum-uns-0", "sum-uns-1", "sum-uns-2", "sum-uns-3", "sum-uns-4", "sum-uns-5",
                 "sum-uns-6", "sum-uns-7", "sum-uns-8", "sum-ans-0", "sum-ans-1", "sum-ans-8"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _rows(*specs: tuple) -> list[dict]:
    """Fixture tuples -> source rows, in the source file's own column order."""
    return [dict(zip(_FIELDS, spec)) for spec in specs]


def _write_source(raw_dir, *specs, extra=None) -> str:
    """Write the fixture directory in the source's own parquet layout."""
    import datasets

    raw_dir.mkdir(parents=True, exist_ok=True)
    datasets.Dataset.from_list(_rows(*specs)).to_parquet(raw_dir / adapter.DATA_FILE)
    for name, rows in (extra or {}).items():
        datasets.Dataset.from_list(rows).to_parquet(raw_dir / name)
    return str(raw_dir)


def _build(tmp_path, *specs, **kwargs):
    """Build from a fixture file and finalise the rows exactly like ``main`` does."""
    raw_dir = _write_source(tmp_path, *specs)
    rows, funnel = adapter.build_rows(raw_dir, **kwargs)
    schema.normalise_extra_info(rows)
    return rows, funnel


def _by_id(rows: list[dict]) -> dict[str, dict]:
    return {row["extra_info"]["task_id"]: row for row in rows}


def _payload(row: dict) -> dict:
    return json.loads(row["reward_model"]["ground_truth"])


def _question_of(row: dict) -> str:
    return row["prompt"][0]["content"].partition("\n\n")[0]


def _gold(row: dict) -> str:
    info = row["extra_info"]
    return next(
        option["text"] for option in info["options"] if option["id"] == info["correct_option_id"]
    )


def _clean(tmp_path, **kwargs):
    return _build(tmp_path, *CLEAN, **kwargs)


# ---------------------------------------------------------------------------
# the three emitted branches
# ---------------------------------------------------------------------------


def test_emits_all_three_branches(tmp_path):
    rows, _ = _clean(tmp_path)
    branches = {row["extra_info"]["branch"] for row in rows}
    assert branches == {adapter.BRANCH_JUDGE, adapter.BRANCH_DIAG, adapter.BRANCH_BARE}
    templates = {row["extra_info"]["branch"]: row["extra_info"]["template"] for row in rows}
    assert templates[adapter.BRANCH_JUDGE] == schema.TEMPLATE_A
    assert templates[adapter.BRANCH_DIAG] == schema.TEMPLATE_A
    assert templates[adapter.BRANCH_BARE] == schema.TEMPLATE_B


def test_every_row_satisfies_the_schema_contract(tmp_path):
    rows, _ = _clean(tmp_path)
    assert rows
    for row in rows:
        assert schema.validate_row(row) == []
    assert len({tuple(sorted(row["extra_info"])) for row in rows}) == 1


def test_diag_row_points_at_the_inserted_span(tmp_path):
    rows, _ = _clean(tmp_path)
    diag = _by_id(rows)["sum-uns-0"]
    info = diag["extra_info"]
    assert info["branch"] == adapter.BRANCH_DIAG
    assert info["template"] == schema.TEMPLATE_A
    assert info["perturbed_entity_text"] == "EF"
    assert info["deleted_condition_text"] == "CE"
    assert _gold(diag) == "EF"
    # the gold may not already stand in the answerable member (that text cannot be
    # what made the variant unanswerable); the distractors are not required to
    assert "EF" not in info["paired_original_text"]
    assert "EF" in diag["prompt"][0]["content"]
    for option in info["options"]:
        assert option["text"] in diag["prompt"][0]["content"]
    assert len(info["options"]) == adapter.K_OPTIONS
    assert len({len(schema.words(option["text"])) for option in info["options"]}) == 1


def test_diag_ground_truth_payload_is_the_diagnosis_contract(tmp_path):
    rows, _ = _clean(tmp_path)
    diag = _by_id(rows)["sum-uns-0"]
    assert _payload(diag) == {
        "answer": None,
        "correct_option_id": diag["extra_info"]["correct_option_id"],
        "has_diagnosis_label": True,
        "perturbation_type": "unrelated_entity",
        "solvable": False,
    }


def test_bare_rows_refuse_without_options(tmp_path):
    rows, _ = _clean(tmp_path)
    by_id = _by_id(rows)
    sedans, harmonic = by_id["sum-uns-3"], by_id["sum-uns-5"]
    assert sedans["extra_info"]["error_type"] == adapter.MISSING_NECESSARY_CONDITION
    assert harmonic["extra_info"]["error_type"] == adapter.QUESTION_MISSING
    for row in (sedans, harmonic):
        info = row["extra_info"]
        payload = _payload(row)
        assert info["template"] == schema.TEMPLATE_B
        assert info["options"] == []
        assert payload["answer"] is None
        assert payload["correct_option_id"] is None
        assert payload["has_diagnosis_label"] is False
        assert payload["solvable"] is False
        assert "\\boxed{UNSOLVABLE}" in row["prompt"][0]["content"]
    assert sedans["extra_info"]["deleted_condition_text"] == "7"
    assert harmonic["extra_info"]["deleted_condition_text"] == "of 1, 2, and 4"


def test_bare_row_records_the_whole_deleted_span(tmp_path):
    rows, _ = _clean(tmp_path)
    triangle = _by_id(rows)["sum-uns-4"]
    assert triangle["extra_info"]["deleted_condition_text"] == "where s is a whole number"
    assert triangle["extra_info"]["perturbed_entity_text"] == ""
    # the deleted span really is gone from the presented question
    assert "whole number" not in _question_of(triangle)


def test_solvable_judge_row_shape_and_audit_answer(tmp_path):
    rows, _ = _clean(tmp_path)
    judge = _by_id(rows)["sum-ans-0"]
    info = judge["extra_info"]
    payload = _payload(judge)
    assert info["solvable"] is True
    assert info["template"] == schema.TEMPLATE_A
    assert payload["judgment_only"] is True
    assert payload["solvable"] is True
    assert payload["answer"] == MEDIANS[2]  # the source's own string, audit only
    assert payload["correct_option_id"] is None
    assert payload["has_diagnosis_label"] is False
    assert info["correct_option_id"] == ""
    assert len(info["options"]) == adapter.K_OPTIONS


def test_template_a_carries_both_label_sides_of_one_pair(tmp_path):
    rows, _ = _clean(tmp_path)
    by_id = _by_id(rows)
    judge, diag = by_id["sum-ans-0"], by_id["sum-uns-0"]
    assert _question_of(judge) == MEDIANS[0]
    assert _question_of(diag) == MEDIANS[1]
    assert judge["extra_info"]["paired_original_text"] == _question_of(diag)
    assert diag["extra_info"]["paired_original_text"] == _question_of(judge)
    # same question pair, same option count and length rule, opposite golds
    assert len(judge["extra_info"]["options"]) == len(diag["extra_info"]["options"])
    assert _payload(judge)["solvable"] is not _payload(diag)["solvable"]


def test_judgment_row_is_emitted_only_with_its_unsolvable_partner(tmp_path):
    rows, _ = _clean(tmp_path)
    by_id = _by_id(rows)
    judges = [row for row in rows if row["extra_info"]["branch"] == adapter.BRANCH_JUDGE]
    assert len(judges) == 3
    for judge in judges:
        index = judge["extra_info"]["index"]
        partner = by_id[f"sum-uns-{index}"]
        assert _question_of(partner) == judge["extra_info"]["paired_original_text"]
        assert judge["extra_info"]["error_type"] == partner["extra_info"]["error_type"]
    # NUMBERS certifies as a pointer pair but its answerable member has no mineable
    # anchor, so the pair emits the diag row alone
    assert "sum-uns-2" in by_id
    assert "sum-ans-2" not in by_id


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [
        (MEDIANS, "irrelevant_undefined_entity"),
        (PRIME_P, "unrealistic_self_contradiction"),
        (TENNIS, "ambiguous_key_information"),
        (SLOPE_AB, "unrealistic_self_contradiction"),  # priority: vague + impossible
    ],
)
def test_type_classifier_labels_the_real_fixtures(tmp_path, fixture, expected):
    rows, _ = _build(tmp_path, fixture)
    assert rows[0]["extra_info"]["error_type"] == expected


@pytest.mark.parametrize(
    ("defect_type", "perturbation"),
    [
        (adapter.MISSING_NECESSARY_CONDITION, "missing_condition"),
        (adapter.QUESTION_MISSING, "question_missing"),
        (adapter.UNREALISTIC_SELF_CONTRADICTION, "unrealistic_condition"),
        (adapter.AMBIGUOUS_KEY_INFORMATION, "ambiguous_condition"),
        (adapter.IRRELEVANT_UNDEFINED_ENTITY, "unrelated_entity"),
    ],
)
def test_defect_specs_cover_every_type_once(defect_type, perturbation):
    spec_perturbation, branch = adapter.DEFECT_SPECS[defect_type]
    assert spec_perturbation == perturbation
    assert branch in (adapter.BRANCH_BARE, adapter.BRANCH_DIAG)
    assert perturbation in schema.PERTURBATION_TYPES


def test_defect_specs_partition_the_five_types():
    assert set(adapter.DEFECT_SPECS) == set(adapter.DEFECT_TYPES)
    bare = {t for t, (_, branch) in adapter.DEFECT_SPECS.items() if branch == adapter.BRANCH_BARE}
    diag = {t for t, (_, branch) in adapter.DEFECT_SPECS.items() if branch == adapter.BRANCH_DIAG}
    assert bare == {adapter.MISSING_NECESSARY_CONDITION, adapter.QUESTION_MISSING}
    assert diag == set(adapter.DIAG_PRIORITY)


def test_error_type_never_reaches_the_reward(tmp_path):
    """The type is monitoring-only: the reward payload must not carry it."""
    rows, _ = _clean(tmp_path)
    for row in rows:
        assert "error_type" not in _payload(row)
        assert "error_type" in row["extra_info"]


# ---------------------------------------------------------------------------
# the funnel
# ---------------------------------------------------------------------------


def test_funnel_stage_counts(tmp_path):
    _, funnel = _build(tmp_path, *ALL)
    stages = {key: value for key, value in funnel.items() if isinstance(value, int)}
    assert stages == EXPECTED_FUNNEL
    assert dict(funnel["certificate_drops"]) == EXPECTED_CERTIFICATE_DROPS


def test_every_certificate_drop_reason_is_named(tmp_path):
    """Every rejection names the clause that caused it -- no anonymous losses."""
    _, funnel = _build(tmp_path, *ALL)
    reasons = dict(funnel["certificate_drops"])
    assert reasons == EXPECTED_CERTIFICATE_DROPS
    assert all(isinstance(count, int) and count > 0 for count in reasons.values())
    assert all(
        reason.startswith(("pointer_", "judge_", "refusal_")) for reason in reasons
    )


def test_funnel_is_monotone_and_every_stage_is_named(tmp_path):
    _, funnel = _build(tmp_path, *ALL)
    stages = [key for key, value in funnel.items() if isinstance(value, int)]
    assert set(stages) == set(EXPECTED_FUNNEL)
    assert set(funnel) == set(EXPECTED_FUNNEL) | {"certificate_drops"}
    assert stages[0] == "raw_rows" and stages[-1] == "after_limit"
    # the first six stages count *pairs* and only ever fall; the last two count the
    # rows those pairs emitted, so a certified pair can only grow the count
    pair_stages = stages[: stages.index("rows_before_limit")]
    counts = [funnel[stage] for stage in pair_stages]
    assert counts[0] == len(ALL)
    for earlier, later in zip(counts, counts[1:]):
        assert later <= earlier
    assert funnel["rows_before_limit"] >= funnel["after_limit"]


def test_conflicting_duplicate_pair_drops_both_rows(tmp_path):
    """Rows 436/2528: the same text pair with golds "2" and "3" is contradictory."""
    rows, funnel = _build(tmp_path, MONTH_A, MONTH_B_CONFLICT)
    assert rows == []
    assert funnel["after_duplicate_pair_drop"] == 0


def test_same_gold_duplicate_pair_keeps_the_first(tmp_path):
    rows, funnel = _build(tmp_path, BUG_A, BUG_B_DUPLICATE)
    assert funnel["after_duplicate_pair_drop"] == 1
    assert [row["extra_info"]["index"] for row in rows] == [0]


def test_multi_region_pair_is_dropped(tmp_path):
    rows, funnel = _build(tmp_path, SOFTBALL)
    assert rows == []
    assert funnel["after_multi_region_drop"] == 0


@pytest.mark.parametrize("fixture", [REMAINDER, UNEQUAL])
def test_no_word_diff_pair_is_dropped(tmp_path, fixture):
    assert adapter.word_regions(*fixture[:2]) == []
    rows, funnel = _build(tmp_path, fixture)
    assert rows == []
    assert funnel["after_no_word_diff_drop"] == 0


def test_pointer_gold_shorter_than_the_floor_is_dropped(tmp_path):
    rows, funnel = _build(tmp_path, MIN_C)
    assert len(adapter.word_regions(*MIN_C[:2])[0].inserted.strip()) < adapter.MIN_GOLD_CHARS
    assert rows == []
    assert funnel["after_defect_certificate_drop"] == 0


def test_pointer_gold_that_already_occurs_in_the_answerable_member_is_dropped(tmp_path):
    rows, funnel = _build(tmp_path, FACTOR)
    assert rows == []
    assert funnel["after_defect_certificate_drop"] == 0


def test_pointer_gold_without_a_k_option_block_is_dropped(tmp_path):
    rows, funnel = _build(tmp_path, GP)
    assert rows == []
    assert funnel["after_defect_certificate_drop"] == 0


def test_refusal_certificate_needs_a_lost_content_word(tmp_path):
    """Row 404: every deleted word still stands elsewhere in the question."""
    qa, qu = (adapter.normalise_question(text) for text in SOUP[:2])
    regions = adapter.word_regions(qa, qu)
    assert regions and not [r for r in regions if r.b_text.strip()]
    assert adapter._certify_refusal(qa, qu, regions) is False
    rows, funnel = _build(tmp_path, SOUP)
    assert rows == []
    assert funnel["after_defect_certificate_drop"] == 0


def test_a_dropped_pointer_pair_leaves_no_judgment_row(tmp_path):
    """Both halves of a pair die together: no orphan judgment row."""
    rows, _ = _build(tmp_path, GP)
    assert rows == []


def test_malformed_rows_are_dropped_not_crashed(tmp_path):
    import datasets

    broken = [
        {"answerable_question": "a", "unanswerable_question": "", "ground_truth": "1"},
        {"answerable_question": "a", "unanswerable_question": "b", "ground_truth": "  "},
        {"answerable_question": "a", "unanswerable_question": "b"},  # ground_truth null
    ]
    tmp_path.mkdir(parents=True, exist_ok=True)
    datasets.Dataset.from_list(broken).to_parquet(tmp_path / adapter.DATA_FILE)
    rows, funnel = adapter.build_rows(str(tmp_path))
    assert rows == []
    assert funnel["raw_rows"] == 3
    assert funnel["after_malformed_drop"] == 0


def test_missing_source_file_fails_closed(tmp_path):
    """An unreadable source is a configuration error, not an empty pool."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError):
        adapter.build_rows(str(tmp_path))


# ---------------------------------------------------------------------------
# determinism, --limit and the CLI
# ---------------------------------------------------------------------------


def test_same_seed_same_rows(tmp_path):
    raw_dir = _write_source(tmp_path, *ALL)
    first, _ = adapter.build_rows(raw_dir, seed=0)
    second, _ = adapter.build_rows(raw_dir, seed=0)
    assert first == second


def test_seed_is_recorded_and_does_not_change_membership(tmp_path):
    """The seed only picks option spans; which rows exist must not depend on it."""
    raw_dir = _write_source(tmp_path, *CLEAN)
    first, _ = adapter.build_rows(raw_dir, seed=0)
    second, _ = adapter.build_rows(raw_dir, seed=7)

    def membership(rows):
        return [
            (
                row["extra_info"]["task_id"],
                row["extra_info"]["branch"],
                row["extra_info"]["error_type"],
                _question_of(row),  # the rendered prompt carries the seeded options
            )
            for row in rows
        ]

    assert membership(first) == membership(second)
    assert {row["extra_info"]["seed"] for row in first} == {0}
    assert {row["extra_info"]["seed"] for row in second} == {7}


def test_limit_interleaves_the_branches(tmp_path):
    rows, funnel = _clean(tmp_path, limit=6)
    assert len(rows) == 6
    assert funnel["rows_before_limit"] == len(CLEAN_ROW_IDS)
    assert funnel["after_limit"] == 6
    assert [row["extra_info"]["branch"] for row in rows] == [
        adapter.BRANCH_DIAG,
        adapter.BRANCH_JUDGE,
        adapter.BRANCH_BARE,
        adapter.BRANCH_DIAG,
        adapter.BRANCH_JUDGE,
        adapter.BRANCH_BARE,
    ]


def test_limit_above_the_row_count_is_a_no_op(tmp_path):
    everything, _ = _clean(tmp_path)
    limited, _ = _clean(tmp_path, limit=10_000)
    assert limited == everything


def test_limit_zero_emits_nothing(tmp_path):
    rows, funnel = _clean(tmp_path, limit=0)
    assert rows == []
    assert funnel["rows_before_limit"] == len(CLEAN_ROW_IDS)


def test_main_writes_a_parquet_and_prints_the_funnel(tmp_path, monkeypatch, capsys):
    out = tmp_path / "built" / "sum.parquet"
    report = tmp_path / "report.json"
    raw_dir = _write_source(tmp_path / "raw", *CLEAN, *DROPS)
    monkeypatch.setattr(
        sys,
        "argv",
        ["sum_adapter.py", "--raw-dir", raw_dir, "--out", str(out), "--report", str(report)],
    )
    adapter.main()
    printed = capsys.readouterr().out
    assert "raw_rows" in printed
    assert "after_defect_certificate_drop" in printed
    assert "per branch:" in printed
    assert "per template:" in printed
    assert "per error_type:" in printed
    assert out.is_file()
    rows = schema.read_parquet_rows(str(out))
    assert len(rows) == len(CLEAN_ROW_IDS)
    for row in rows:
        assert schema.validate_row(row) == []

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["rows"] == len(CLEAN_ROW_IDS)
    assert payload["limit"] is None
    assert payload["funnel"]["raw_rows"] == len(CLEAN) + len(DROPS)
    assert payload["funnel"]["after_limit"] == len(CLEAN_ROW_IDS)
    assert set(payload["per_branch"]) == {
        adapter.BRANCH_DIAG,
        adapter.BRANCH_BARE,
        adapter.BRANCH_JUDGE,
    }
    assert payload["per_solvable"] == {"False": 9, "True": 3}


def test_task_ids_are_stable_unique_and_source_derived(tmp_path):
    rows, _ = _clean(tmp_path)
    ids = [row["extra_info"]["task_id"] for row in rows]
    assert len(ids) == len(set(ids))
    assert set(ids) == CLEAN_ROW_IDS
    for row in rows:
        info = row["extra_info"]
        assert info["task_id"] == adapter._task_id(
            info["index"], "ans" if info["solvable"] else "uns"
        )


def test_task_id_encodes_the_side_and_the_position():
    assert adapter._task_id(0, "ans") == "sum-ans-0"
    assert adapter._task_id(2528, "uns") == "sum-uns-2528"


def test_test_split_note_reports_the_absent_file(tmp_path):
    raw_dir = _write_source(tmp_path, *CLEAN)
    assert adapter.test_split_note(raw_dir) == "test.parquet: absent"


def test_test_split_note_describes_a_test_split_without_a_pair(tmp_path):
    extra = {
        adapter.TEST_FILE: [
            {"prompt": "a", "ground_truth": "I don't know."},
            {"prompt": "b", "ground_truth": "I don't know."},
        ]
    }
    raw_dir = _write_source(tmp_path, *CLEAN, extra=extra)
    note = adapter.test_split_note(raw_dir)
    assert "2 rows" in note
    assert "distinct ground_truth=1" in note
    assert "no answerable_question" in note


def test_breakdown_of_the_built_rows(tmp_path):
    rows, _ = _clean(tmp_path)
    branches = adapter._breakdown(rows, lambda row: row["extra_info"]["branch"])
    assert branches == {
        adapter.BRANCH_BARE: 3,
        adapter.BRANCH_DIAG: 6,
        adapter.BRANCH_JUDGE: 3,
    }
    error_types = adapter._breakdown(rows, lambda row: row["extra_info"]["error_type"])
    assert set(error_types) == {
        adapter.MISSING_NECESSARY_CONDITION,
        adapter.QUESTION_MISSING,
        adapter.IRRELEVANT_UNDEFINED_ENTITY,
        adapter.UNREALISTIC_SELF_CONTRADICTION,
        adapter.AMBIGUOUS_KEY_INFORMATION,
    }


# ---------------------------------------------------------------------------
# the text plumbing
# ---------------------------------------------------------------------------


def test_normalise_question_collapses_whitespace():
    assert adapter.normalise_question("  a\n\tb  c \r\n d ") == "a b c d"
    assert adapter.normalise_question(None) == ""
    assert adapter.normalise_question("") == ""


def test_word_regions_replacement_is_one_region():
    regions = adapter.word_regions("the value is 5 now", "the value is 7 now")
    assert len(regions) == 1
    assert regions[0].a_text == "5"
    assert regions[0].b_text == "7"
    assert regions[0].tag == "replace"
    assert "the value is 7 now"[regions[0].b_start :] == "7 now"


def test_word_regions_pure_deletion_has_empty_b_text():
    regions = adapter.word_regions("feed 3 adults or 5 children", "feed 3 adults")
    assert len(regions) == 1
    assert regions[0].a_text == "or 5 children"
    assert regions[0].b_text == ""
    assert regions[0].inserted == ""


def test_word_regions_pure_insertion_has_empty_a_text():
    regions = adapter.word_regions("the value is now", "the value is 7 now")
    assert len(regions) == 1
    assert regions[0].a_text == ""
    assert regions[0].inserted == "7"


def test_word_regions_are_whole_words_not_fragments():
    qa = "John cuts the 100m line into 10m sections now"
    qu = "John cuts the line into sections now"
    regions = adapter.word_regions(qa, qu)
    assert [region.a_text for region in regions] == ["100m", "10m"]
    for region in regions:
        assert qa[region.a_start : region.a_start + len(region.a_text)] == region.a_text


def test_word_regions_preserve_the_character_offsets():
    qa = "Medians AD and CE of triangle ABC intersect in M."
    qu = "Medians AD and EF of triangle ABC intersect in M."
    region = adapter.word_regions(qa, qu)[0]
    assert qu[region.b_start : region.b_start + len(region.b_text)] == "EF"
    assert qa[region.a_start : region.a_start + len(region.a_text)] == "CE"


def test_merge_opcodes_joins_adjacent_changed_runs():
    opcodes = [("delete", 0, 1, 0, 0), ("insert", 1, 1, 0, 1), ("equal", 1, 3, 1, 3)]
    assert adapter._merge_opcodes(opcodes) == [("delete", 0, 1, 0, 1)]
    opcodes = [("equal", 0, 1, 0, 1), ("replace", 1, 2, 1, 2)]
    assert adapter._merge_opcodes(opcodes) == [("replace", 1, 2, 1, 2)]
    opcodes = [("delete", 0, 1, 0, 0), ("equal", 1, 2, 0, 1), ("delete", 2, 3, 1, 1)]
    assert adapter._merge_opcodes(opcodes) == [("delete", 0, 1, 0, 0), ("delete", 2, 3, 1, 1)]
    assert adapter._merge_opcodes([("equal", 0, 1, 0, 1)]) == []


def test_residual_tokens_accepts_a_real_diff_and_rejects_a_partial_one():
    qa, qu = "the value is 5 now", "the value is 7 now"
    assert adapter._residual_tokens(qa, qu, adapter.word_regions(qa, qu)) is True

    class FakeRegion:
        a_text = "the"
        b_text = "a"
        a_start = 0
        b_start = 0

    assert adapter._residual_tokens(qa, qu, [FakeRegion()]) is False


def test_question_clause():
    assert adapter._question_clause("What is x?") == ("What is x?", 0)
    assert adapter._question_clause("Given a. What is x?") == (" What is x?", 8)
    assert adapter._question_clause("no question here") == ("", -1)
    assert adapter._question_clause("") == ("", -1)


def test_clause_cut_detects_a_deleted_question_clause():
    qa = "What is the harmonic mean of 1, 2, and 4?"
    qu = "What is the harmonic mean?"
    assert adapter._clause_cut(qa, qu, adapter.word_regions(qa, qu)) is True
    # a deletion elsewhere in the question is not a cut clause
    qa2 = "The mean of 1, 2, 4 is 5. What is the median?"
    qu2 = "The mean of 1, 2 is 5. What is the median?"
    assert adapter._clause_cut(qa2, qu2, adapter.word_regions(qa2, qu2)) is False
    # no question mark at all means there is no clause to cut
    assert adapter._clause_cut("no question", "no", adapter.word_regions("no question", "no")) is False


def test_clause_cut_needs_the_clause_to_be_gone_from_the_prompt():
    """A clause the presented member still carries has not been cut.

    ``sum-uns-25944`` states its question twice (the problem is quoted, then
    restated); the diff deletes the second copy and the prompt still asks the
    question, so the row lost a condition, not its question.  ``sum-uns-1240``
    is the alignment variant: the word-level diff charges the surviving "In" to
    the deleted run while the presented member keeps the whole clause.
    """
    qa = (
        "The text quotes a problem: \"A rod 5 feet long. Cutting 1 foot from the "
        "base, it weighs 4 jin. How much does the rod weigh?\" This means: \"A rod "
        "(uniformly varying in thickness) 5 feet long. Cutting 1 foot from the "
        "base, it weighs 4 jin. How much does the rod weigh?\" The answer is ____."
    )
    qu = "A rod. Cutting 1 foot from the base, it weighs 4 jin. How much does the rod weigh?"
    assert adapter._clause_cut(qa, qu, adapter.word_regions(qa, qu)) is False
    assert (
        adapter.classify_bare_defect(qa, qu, adapter.word_regions(qa, qu))
        == adapter.MISSING_NECESSARY_CONDITION
    )


def test_certify_pointer_accepts_a_clean_pair():
    qa = "The rectangle has width 4 and height 9. What is the area?"
    qu = "The rectangle has width 4 and height 12. What is the area?"
    emitting = [r for r in adapter.word_regions(qa, qu) if r.b_text.strip()]
    pointer, reason = adapter._certify_pointer(qa, qu, emitting, random.Random(0))
    assert pointer is not None and reason == ""
    assert pointer["gold"] == "12"
    assert pointer["texts"][ord(pointer["correct"]) - ord("A")] == "12"
    assert len(pointer["texts"]) == adapter.K_OPTIONS


def test_certify_pointer_rejections():
    qa = "The rectangle has width 4 and height 9. What is the area?"
    qu = "The rectangle has width 4 and height 12. What is the area?"
    emitting = [r for r in adapter.word_regions(qa, qu) if r.b_text.strip()]
    rng = random.Random(0)
    pointer, reason = adapter._certify_pointer(qa, qu, emitting, rng)
    assert pointer is not None and reason == ""
    # wrong arity: the pointer certificate is for exactly one inserted region
    assert adapter._certify_pointer(qa, qu, [], rng)[1] == "pointer_not_one_inserted_region"
    assert adapter._certify_pointer(qa, qu, emitting * 2, rng)[1] == "pointer_not_one_inserted_region"

    # the gold is a single character: unusable as an option
    short_qa, short_qu = "x = 10 and y = 2", "x = 10 and y = c"
    short = [r for r in adapter.word_regions(short_qa, short_qu) if r.b_text.strip()]
    assert short and adapter._certify_pointer(short_qa, short_qu, short, rng) == (
        None,
        "pointer_gold_too_short",
    )

    # the gold already occurs inside the answerable member
    inside_qa, inside_qu = "(2112-2021)^2 is odd", "(2112- )^2 is odd"
    inside = [r for r in adapter.word_regions(inside_qa, inside_qu) if r.b_text.strip()]
    assert inside and adapter._certify_pointer(inside_qa, inside_qu, inside, rng) == (
        None,
        "pointer_gold_in_answerable_member",
    )
    assert inside[0].inserted.strip() in inside_qa


def test_certify_pointer_locates_the_gold_at_the_region_offset():
    """A gold that occurs earlier in the question must not be certified."""
    qa = "The 9 is 5 for the 9. What is the total?"
    qu = "The 9 is 5 for the 12. What is the total?"
    emitting = [r for r in adapter.word_regions(qa, qu) if r.b_text.strip()]
    assert emitting and qu.find(emitting[0].inserted.strip()) == emitting[0].b_start
    # the same text earlier in the question makes that offset guard bite
    qa2 = "The 12 is 5 for the 9. What is the total?"
    qu2 = "The 12 is 5 for the 12. What is the total?"
    emitting2 = [r for r in adapter.word_regions(qa2, qu2) if r.b_text.strip()]
    assert emitting2 and qu2.find(emitting2[0].inserted.strip()) < emitting2[0].b_start
    assert adapter._certify_pointer(qa2, qu2, emitting2, random.Random(0))[1] == (
        "pointer_gold_in_answerable_member"
    )


def test_certify_pointer_rejects_an_insertion_that_is_not_a_defect():
    """The F1 defect: an insertion no visible rule names must not certify.

    ``and the`` is padding the paraphrase added -- it is not a vague quantifier,
    not an impossible value, and it introduces no content word the original
    lacked.  The clause is checked after the substring clause but before the
    option block, so its rejection reason is its own.
    """
    qa = "A box holds 12 red balls. What is the total?"
    qu = "A box holds and the 12 red balls. What is the total?"
    emitting = [r for r in adapter.word_regions(qa, qu) if r.b_text.strip()]
    assert emitting and emitting[0].inserted.strip() == "and the"
    assert adapter.certifies_as_visible_defect("and the", qa) is False
    assert adapter._certify_pointer(qa, qu, emitting, random.Random(0)) == (
        None,
        "pointer_insertion_is_not_a_defect",
    )
    # the two classes that *are* justified by their own rule still certify
    assert adapter.certifies_as_visible_defect("a negative unknown number", qa) is True
    assert adapter.certifies_as_visible_defect("some", qa) is True
    assert adapter.certifies_as_visible_defect("a hexagonal prism", qa) is True
    assert adapter.certifies_as_visible_defect("the box", qa) is False


def test_a_non_defect_insertion_is_dropped_not_rerouted(tmp_path):
    """A pair that cannot certify as visible is dropped -- not rebuilt as bare.

    Table B maps an inserted region to "SUM-visible", so the pair's shape, not
    the certificate, decides the branch: a rejected pair must not appear as a
    bare row either, or the artifact's branches would stop being a function of
    the shape the design documents.
    """
    raw_dir = _write_source(tmp_path, NON_DEFECT)
    rows, funnel = _build(tmp_path, NON_DEFECT)
    assert rows == []
    assert funnel["after_defect_certificate_drop"] == 0
    assert dict(funnel["certificate_drops"]) == {"pointer_insertion_is_not_a_defect": 1}
    assert raw_dir


def test_certify_refusal_branches():
    qa = "A can of soup feeds 3 adults or 5 children. How many adults does it feed?"
    qu = "A can of soup feeds 3 adults. How many adults does it feed?"
    qa, qu = adapter.normalise_question(qa), adapter.normalise_question(qu)
    assert adapter._certify_refusal(qa, qu, adapter.word_regions(qa, qu)) is True
    # a deleted word that survives elsewhere in the question proves nothing
    qa2 = "He ate 2 of the 2 apples. How many are left?"
    qu2 = "He ate 2 apples. How many are left?"
    qa2, qu2 = adapter.normalise_question(qa2), adapter.normalise_question(qu2)
    regions2 = adapter.word_regions(qa2, qu2)
    assert regions2 and [r.a_text for r in regions2] == ["2 of the"]
    assert adapter._certify_refusal(qa2, qu2, regions2) is False
    # nothing deleted at all
    assert adapter._certify_refusal(qa, qa, []) is False


def test_certify_judge_anchors_on_the_largest_deleted_span():
    qa = "The box holds 12 red balls and 6 blue balls. What is the total?"
    qu = "The box holds red balls and blue balls. What is the total?"
    regions = adapter.word_regions(qa, qu)
    assert [region.a_text for region in regions] == ["12", "6"]
    judge, reason = adapter._certify_judge(qa, regions, random.Random(0))
    assert judge is not None and reason == ""
    assert judge["anchor"] == "12"
    assert len(judge["texts"]) == adapter.K_OPTIONS
    assert adapter._certify_judge(qa, [], random.Random(0))[1] == "judge_no_deleted_anchor"


def test_certify_judge_refuses_a_content_free_anchor():
    """The anchor is appended to the option block as its gold.

    ``mine_option_spans`` filters its *distractors* for content words but appends
    the caller's gold unconditionally, so a caller that hands it a bare function
    word gets back an option block that violates the miner's own rule 3.  The
    clause belongs to the caller, and this is it.
    """
    qa = "The box holds 12 red balls and 6 blue balls. What is the total?"
    qu = "The box 12 red balls 6 blue balls. What is the total?"
    regions = adapter.word_regions(qa, qu)
    # two deletions; the longer one ("holds") is a content word and certifies, the
    # function-word run is what must not be handed in as the anchor
    assert [region.a_text for region in regions] == ["holds", "and"]
    assert schema.is_content_word("holds") and not schema.is_content_word("and")
    filler, reason = adapter._certify_judge(qa, regions[1:], random.Random(0))
    assert filler is None and reason == "judge_anchor_has_no_content_word"
    # the same predicate the audit applies: no option of the block may be a filler
    assert adapter.certifies_as_visible_defect("and", qa) is False


def test_introduces_undefined_entity():
    assert adapter.introduces_undefined_entity("a negative decimal number", "when p is prime")
    assert not adapter.introduces_undefined_entity("the", "the box holds the balls")
    assert not adapter.introduces_undefined_entity("5", "the value is 5 now")


def test_classify_visible_defect_priority_and_residual():
    assert (
        adapter.classify_visible_defect("a negative prime number")
        == adapter.UNREALISTIC_SELF_CONTRADICTION
    )
    assert adapter.classify_visible_defect("some") == adapter.AMBIGUOUS_KEY_INFORMATION
    # both rules fire; the impossible rule wins
    assert (
        adapter.classify_visible_defect("is some other negative number")
        == adapter.UNREALISTIC_SELF_CONTRADICTION
    )
    assert (
        adapter.classify_visible_defect("the number of apples in an orchard")
        == adapter.IRRELEVANT_UNDEFINED_ENTITY
    )


def test_classify_bare_defect_uses_the_question_clause():
    qa = "What is the harmonic mean of 1, 2, and 4?"
    qu = "What is the harmonic mean?"
    assert (
        adapter.classify_bare_defect(qa, qu, adapter.word_regions(qa, qu))
        == adapter.QUESTION_MISSING
    )
    qa2 = "For every 4 cars, 7 sedans are sold. How many sedans for 28 cars?"
    qu2 = "For every 4 cars, sedans are sold. How many sedans for 28 cars?"
    assert (
        adapter.classify_bare_defect(qa2, qu2, adapter.word_regions(qa2, qu2))
        == adapter.MISSING_NECESSARY_CONDITION
    )


def test_is_well_formed_rejects_broken_rows():
    good = _rows(SEDANS)[0]
    assert adapter._is_well_formed(good) is True
    assert adapter._is_well_formed("not a dict") is False
    for key in _FIELDS:
        broken = dict(good)
        broken[key] = ""
        assert adapter._is_well_formed(broken) is False
        broken[key] = 7
        assert adapter._is_well_formed(broken) is False


def test_pair_key_uses_both_members():
    left = _rows(MONTH_A)[0]
    right = _rows(MONTH_B_CONFLICT)[0]
    assert adapter._pair_key(left) == adapter._pair_key(right)
    other = dict(left, answerable_question=left["answerable_question"] + " Why?")
    assert adapter._pair_key(other) != adapter._pair_key(left)


def test_load_source_stamps_positional_indices(tmp_path):
    raw_dir = _write_source(tmp_path, *CLEAN)
    source = adapter.load_source(raw_dir)
    assert len(source) == len(CLEAN)
    assert [row["_index"] for row in source] == list(range(len(CLEAN)))
    with pytest.raises(FileNotFoundError):
        adapter.load_source(str(tmp_path / "nope"))


def test_interleave_by_branch_round_robins():
    rows = [
        {"extra_info": {"branch": "a"}},
        {"extra_info": {"branch": "b"}},
        {"extra_info": {"branch": "a"}},
        {"extra_info": {"branch": "a"}},
    ]
    assert [row["extra_info"]["branch"] for row in adapter._interleave_by_branch(rows)] == [
        "a",
        "b",
        "a",
        "a",
    ]
    assert adapter._interleave_by_branch([]) == []


def test_breakdown_counts_and_sorts():
    rows = [{"x": "b"}, {"x": "a"}, {"x": "b"}]
    assert adapter._breakdown(rows, lambda row: row["x"]) == {"a": 1, "b": 2}
    assert adapter._breakdown([], lambda row: row["x"]) == {}


def test_option_block_ids_are_sequential():
    block = adapter._option_block(["x", "yy", "zzz"])
    assert [option["id"] for option in block] == ["A", "B", "C"]
    assert [option["text"] for option in block] == ["x", "yy", "zzz"]


def test_schema_make_row_shape_is_honoured(tmp_path):
    rows, _ = _clean(tmp_path)
    for row in rows:
        assert set(row) == {"data_source", "prompt", "ability", "reward_model", "extra_info"}
        assert row["data_source"] == schema.SOURCE_SUM
        assert row["ability"] == schema.SOURCES[schema.SOURCE_SUM][0] == "math"
        assert len(row["prompt"]) == 1


def test_decision_constants():
    assert adapter.K_OPTIONS == 3  # D15
    assert adapter.MIN_GOLD_CHARS == 2
    assert adapter.DATA_SOURCE == schema.SOURCE_SUM
    assert adapter.DATA_FILE == "train.parquet"


def test_impossible_and_vague_regexes_are_narrow():
    # a subtraction must not look like a negative number
    assert adapter._IMPOSSIBLE_RE.search("x-1 equals y") is None
    assert adapter._IMPOSSIBLE_RE.search("a negative value") is not None
    assert adapter._IMPOSSIBLE_RE.search("(-3)") is not None
    # "about"/"around" are ordinary English, not vagueness markers
    assert adapter._VAGUE_RE.search("a circle about the origin") is None
    assert adapter._VAGUE_RE.search("a certain period") is not None


# ---------------------------------------------------------------------------
# integration with verify_sum.py
# ---------------------------------------------------------------------------


def test_verify_sum_passes_on_the_fixture_artifact(tmp_path):
    import verify_sum

    raw_dir = _write_source(tmp_path, *ALL)
    rows, _ = adapter.build_rows(raw_dir)
    schema.normalise_extra_info(rows)

    audit = verify_sum.Audit()
    verify_sum.check_contract(rows, audit)
    source = verify_sum.load_source(raw_dir)
    verify_sum.check_source_anchor(rows, audit, source, raw_dir)
    sample = verify_sum.sample_by_branch(rows, per_branch=5, seed=0)
    verify_sum.check_l1(sample, audit)
    verify_sum.check_l2(sample, audit)
    verify_sum.check_l3_options(rows, audit)
    audit.flush()
    assert audit.failures == 0


def test_verify_sum_type_predicate_agrees_with_the_adapter(tmp_path):
    import verify_sum

    rows, _ = _clean(tmp_path)
    for row in rows:
        info = row["extra_info"]
        assert verify_sum.type_predicate(row, info["error_type"]) == ""


def test_verify_sum_flags_a_tampered_pair(tmp_path):
    import verify_sum

    raw_dir = _write_source(tmp_path, *CLEAN)
    rows, _ = adapter.build_rows(raw_dir)
    schema.normalise_extra_info(rows)
    victim = next(row for row in rows if row["extra_info"]["branch"] == adapter.BRANCH_DIAG)
    tampered = copy.deepcopy(victim)
    tampered["extra_info"]["paired_original_text"] = "a question from nowhere"
    audit = verify_sum.Audit()
    verify_sum.check_source_anchor([tampered], audit, verify_sum.load_source(raw_dir), raw_dir)
    assert audit.failures == 1


def test_verify_sum_fails_closed_without_the_raw_file(tmp_path):
    import verify_sum

    rows, _ = _clean(tmp_path)
    audit = verify_sum.Audit()
    verify_sum.check_source_anchor(rows, audit, [], str(tmp_path / "absent"))
    assert audit.failures == 1


def test_verify_sum_yield_recomputation_is_deterministic():
    import verify_sum

    source = _rows(MEDIANS, SEDANS, SOFTBALL, REMAINDER)
    assert verify_sum.measure_pools(source) == {
        "pairs": 4,
        "no_word_change": 1,
        "three_tier_pool": 1,
        "four_tier_pool": 1,
        "multi_region": 1,
    }


def test_verify_sum_l1_rejects_a_wrong_recorded_defect(tmp_path):
    import verify_sum

    raw_dir = _write_source(tmp_path, *CLEAN)
    rows, _ = adapter.build_rows(raw_dir)
    schema.normalise_extra_info(rows)
    victim = next(row for row in rows if row["extra_info"]["branch"] == adapter.BRANCH_DIAG)
    tampered = copy.deepcopy(victim)
    tampered["extra_info"]["perturbed_entity_text"] = "something else"
    audit = verify_sum.Audit()
    verify_sum.check_l1({adapter.BRANCH_DIAG: [tampered]}, audit)
    assert audit.failures == 1


def test_verify_sum_defect_class_agrees_with_the_adapter():
    """The audit's copy of the defect-class rule must decide exactly as the adapter's."""
    import verify_sum

    qa, qu = NON_DEFECT[0], NON_DEFECT[1]
    assert verify_sum.certifies_as_defect("How", qa) is False
    assert verify_sum.certifies_as_defect("some", qa) is True
    assert verify_sum.certifies_as_defect("a negative number", qa) is True
    assert verify_sum.certifies_as_defect("a hexagonal prism", qa) is True
    # a content word that the answerable member already carries names nothing new
    boxed = "The box holds 12 red balls. What is the total?"
    assert verify_sum.certifies_as_defect("the box", boxed) is False
    assert verify_sum.certifies_as_defect("and the", qa) is False
    for fixture in CLEAN:
        for region in adapter.word_regions(
            adapter.normalise_question(fixture[0]), adapter.normalise_question(fixture[1])
        ):
            inserted = region.b_text.strip()
            if not inserted:
                continue
            assert verify_sum.certifies_as_defect(
                inserted, adapter.normalise_question(fixture[0])
            ) is adapter.certifies_as_visible_defect(
                inserted, adapter.normalise_question(fixture[0])
            )
    assert qu  # the fixture pair is the rejecting one


def test_verify_sum_l1_rejects_a_non_defect_gold():
    """The pre-fix artifact's gold: a question word shipped as the defect.

    The row below is exactly what the old certificate emitted for source row
    35212 (the pair of :data:`NON_DEFECT`): every earlier L1 clause holds -- the
    recorded defect re-derives, the multiset balances, the gold is the pair's own
    insertion at the region's offset -- and only the defect-class clause can see
    that nothing in the row justifies it.
    """
    import verify_sum

    qa, qu = NON_DEFECT[0], NON_DEFECT[1]
    extra = adapter._base_extra_info({"_index": 35212}, adapter.IRRELEVANT_UNDEFINED_ENTITY, 0)
    extra["deleted_condition_text"] = "Simplify"
    extra["perturbed_entity_text"] = "How"
    plan = {
        "kind": "diag",
        "defect_type": adapter.IRRELEVANT_UNDEFINED_ENTITY,
        "pointer": {"gold": "How", "texts": ["How", "2w", "4w"], "correct": "A"},
        "qa": qa,
        "qu": qu,
        "extra": extra,
        "ground_truth": NON_DEFECT[2],
    }
    row = adapter._diag_row(plan)
    audit = verify_sum.Audit()
    verify_sum.check_l1({adapter.BRANCH_DIAG: [row]}, audit)
    assert audit.failures == 1
    assert "is not a defect any of the three rules names" in "\n".join(audit.lines)


def test_verify_sum_flags_a_filler_option_but_exempts_a_vague_gold(tmp_path):
    """L3d: a content-free *distractor* is a filler; a vague diag gold is the defect."""
    import verify_sum

    rows, _ = _clean(tmp_path)
    audit = verify_sum.Audit()
    verify_sum.check_l3_option_content(rows, audit)
    assert audit.failures == 0, "\n".join(audit.lines)
    # the vague-quantifier gold of TENNIS is content-free and still exempt
    tennis = next(
        row
        for row in rows
        if row["extra_info"]["task_id"] == "sum-uns-7"
        and row["extra_info"]["branch"] == adapter.BRANCH_DIAG
    )
    gold = _gold(tennis)
    assert not any(schema.is_content_word(token) for token in schema.words(gold))
    tampered = copy.deepcopy(tennis)
    tampered["extra_info"]["options"].append({"id": "D", "text": "the of a"})
    audit = verify_sum.Audit()
    verify_sum.check_l3_option_content([tampered], audit)
    assert audit.failures == 1
    assert "'the of a'" in "\n".join(audit.lines)


def test_verify_sum_clause_rule_uses_the_last_clause_and_needs_it_gone():
    """The audit's copy of the rule, on the shapes that made it disagree.

    ``sum-uns-25944`` states its question twice.  ``str.find(clause)`` located the
    first copy, which the deleted run does not touch, while the adapter tested the
    copy the diff deleted -- the disagreement that made the audit red.  The rule
    also has to require the clause to be gone from the presented member: the first
    copy still asks the question, so the pair lost a condition, not its question.
    """
    import verify_sum

    clause, offset = verify_sum.question_clause("Given a weight. How much does it weigh?")
    assert clause == " How much does it weigh?"
    assert verify_sum.clause_extent(clause, offset) == (offset + 1, offset + len(clause))
    # repeated clause: the offset is the last one, not the first
    repeated = "How much does it weigh? The rod is gold. How much does it weigh?"
    _, second = verify_sum.question_clause(repeated)
    assert second == repeated.rfind(".", 0, repeated.rfind("?")) + 1
    assert second > repeated.find("How much")
    # a clause that really goes is a cut
    qa2 = "What is the harmonic mean of 1, 2, and 4?"
    qu2 = "What is the harmonic mean?"
    assert verify_sum.clause_cut(qa2, qu2, verify_sum.diff_regions(qa2, qu2)) is True
    # a clause the presented member still carries is not
    qa3 = "A rod is gold. How much does it weigh? The rod is 5 feet. How much does it weigh?"
    qu3 = "A rod is gold. How much does it weigh?"
    assert verify_sum.clause_cut(qa3, qu3, verify_sum.diff_regions(qa3, qu3)) is False


def test_verify_sum_l1_coverage_is_exhaustive_by_default():
    """A sampled L1 could not see a single tampered row outside the sample."""
    import verify_sum

    rows = [{"extra_info": {"branch": adapter.BRANCH_DIAG, "task_id": f"sum-uns-{i}"}} for i in range(500)]
    sample = verify_sum.sample_by_branch(rows, per_branch=0, seed=0)
    assert len(sample[adapter.BRANCH_DIAG]) == 500
    # a cap keeps the 50-row floor and silently leaves 450 rows unchecked, which
    # is exactly why the default is 0 and any cap is printed next to the result
    capped = verify_sum.sample_by_branch(rows, per_branch=5, seed=0)
    picked = {row["extra_info"]["task_id"] for row in capped[adapter.BRANCH_DIAG]}
    assert len(picked) == 50
    assert picked < {row["extra_info"]["task_id"] for row in rows}


def test_verify_sum_l3_corpus_is_drawn_at_the_quota_proportions(tmp_path):
    import verify_sum

    rows, _ = _clean(tmp_path)
    texts, labels, composition = verify_sum.quota_corpus(rows, total_rows=1_000, seed=0)
    assert len(texts) == len(labels) == sum(composition.values())
    assert set(composition) == {
        adapter.BRANCH_JUDGE,
        adapter.BRANCH_DIAG,
        adapter.BRANCH_BARE,
    }
    # every branch is represented, and the solvable side is the judge pool alone
    assert all(value > 0 for value in composition.values())
    assert sum(labels) == composition[adapter.BRANCH_JUDGE]


def test_verify_sum_l3_gate_runs_and_reports_a_reading(tmp_path):
    import verify_sum

    rows, _ = _clean(tmp_path)
    audit = verify_sum.Audit()
    verify_sum.check_l3_nb(rows, audit, folds=3, seed=0, min_support=1, total_rows=1_000)
    printed = "\n".join(audit.lines)
    assert "L3c corpus" in printed
    assert "L3c raw out-of-fold balanced accuracy" in printed
    assert "L3c shuffled-label control" in printed


def test_verify_sum_l3_gate_fails_closed_on_a_one_sided_artifact(tmp_path):
    import verify_sum

    rows, _ = _clean(tmp_path)
    solvable = [row for row in rows if row["extra_info"]["solvable"]]
    assert solvable
    audit = verify_sum.Audit()
    verify_sum.check_l3_nb(solvable, audit, folds=3, seed=0, min_support=1, total_rows=1_000)
    assert audit.failures == 1
    assert "not identifiable" in "\n".join(audit.lines)


# ---------------------------------------------------------------------------
# the real source (skipped when the frozen download is absent)
# ---------------------------------------------------------------------------

needs_real = pytest.mark.skipif(
    not os.path.isdir(adapter.DEFAULT_RAW_DIR), reason=f"{adapter.DEFAULT_RAW_DIR} is absent"
)


class TestRealSource:
    @needs_real
    def test_real_slice_builds_and_validates(self):
        rows, funnel = adapter.build_rows(adapter.DEFAULT_RAW_DIR, limit=60)
        schema.normalise_extra_info(rows)
        assert len(rows) == 60
        assert funnel["raw_rows"] == 36_480
        for row in rows:
            assert schema.validate_row(row) == []
        assert {row["extra_info"]["branch"] for row in rows} == {
            adapter.BRANCH_DIAG,
            adapter.BRANCH_JUDGE,
            adapter.BRANCH_BARE,
        }
        # every diag gold is a span of its own rendered question and absent from
        # the answerable member -- the pointer certificate's two visible promises
        for row in rows:
            info = row["extra_info"]
            if info["branch"] != adapter.BRANCH_DIAG:
                continue
            assert _gold(row) in _question_of(row)
            assert _gold(row) not in info["paired_original_text"]

    @needs_real
    def test_real_test_split_note(self):
        note = adapter.test_split_note(adapter.DEFAULT_RAW_DIR)
        assert "test.parquet: 284 rows" in note
        assert "distinct ground_truth=1" in note
        assert "no answerable_question" in note
