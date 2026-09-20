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
"""Tests for ``sum_adapter.py`` -- SUM's D23 pair judgement task.

Covers the contract of design doc sections 4.5 / 5.1 / 5.2 (template C) / 4.8 table
B row 6 / 9:

* one **pair row per usable raw row**, with both questions in the prompt;
* the per-row seeded 50:50 A/B draw, its reproducibility, and the 50:50 +-2pt
  distribution on a quota-sized artifact;
* ``answerable_id`` names the question whose text is the raw ``answerable_question``
  (the section 9 contract assertion), and ``answer`` is the raw ``ground_truth``;
* the 6,000-pair quota and the deterministic spread selection (never "the first N"),
  with ``--limit`` as the override;
* the section 9 reward cells driven end to end through the frozen dispatcher:
  A/B swap invariance, ``\\boxed{A: 42}`` == ``\\boxed{a：42}``, a wrong id scoring 0
  even when the answer text is the gold, and bare / ``UNSOLVABLE`` forms scoring 0;
* well-formedness filtering and the duplicate-pair rules;
* integration with ``verify_sum.py`` (contract, A/B balance, quota, source anchor,
  the Q10 spot check and the raw pool measurement).

The fixtures are real ``train.parquet`` rows, copied verbatim -- the LaTeX, the
escaped backslashes and the odd source spacing are exactly what SUM ships, so the
tests exercise the same text shapes as the real build.  Each fixture carries its
source row index in a comment so a failure can be taken back to the frozen file.
They are written to ``tmp_path`` in the source's own parquet layout and never to the
downloaded data directory, so the suite is self-contained; the ``TestRealSource``
tests at the end read the real file and skip when it is absent.
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

#: ``sum_adapter.py`` is the frozen reward file's consumer; the section 9 score cells
#: are pinned against the real dispatcher (not a copy of its table) where it imports.
_REWARD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "reward")
if _REWARD_DIR not in sys.path:
    sys.path.insert(0, _REWARD_DIR)
try:  # pragma: no cover - the frozen file is always present in this repo
    from hallucination_compute_score import compute_score as _compute_score
except ImportError:  # pragma: no cover
    _compute_score = None

requires_reward = pytest.mark.skipif(_compute_score is None, reason="reward dispatcher is not importable")

# ---------------------------------------------------------------------------
# fixtures -- verbatim SUM rows, as (answerable, unanswerable, ground_truth)
# ---------------------------------------------------------------------------

# source row 49: the rewrite swaps "CE" for "EF"
MEDIANS = (
    "Medians AD and CE of \\(\\triangle ABC\\) intersect in M. The midpoint of AE is N. "
    "Let the area of \\(\\triangle MNE\\) be k times the area of \\(\\triangle ABC\\). Then k equals:",
    "Medians AD and EF of \\(\\triangle ABC\\) intersect in M. The midpoint of AE is N. "
    "Let the area of \\(\\triangle MNE\\) be k times the area of \\(\\triangle ABC\\). Then k equals:",
    "$\\frac{1}{3}$",
)

# source row 55: the question clause is rewritten into "equals what value?"
FORALL = (
    "For all non-zero numbers x and y such that x = 1/y, "
    "\\(\\left(x-\\frac{1}{x}\\right)\\left(y+\\frac{1}{y}\\right)\\) equals",
    "For all non-zero numbers x and y, "
    "\\(\\left(x-\\frac{1}{x}\\right)\\left(y+\\frac{1}{y}\\right)\\) equals what value?",
    "2",
)

# source row 290: the rewrite drops the divisibility conditions
NUMBERS = (
    "How many numbers between 1 and 2005 are integer multiples of 3 or 4 but not 12?",
    "How many numbers between 1 and 2005 satisfy the property?",
    "502",
)

# source row 51: "7" is deleted
SEDANS = (
    "On average, for every 4 sports cars sold at the local dealership, 7 sedans are sold. "
    "The dealership predicts that it will sell 28 sports cars next month. "
    "How many sedans does it expect to sell?",
    "On average, for every 4 sports cars sold at the local dealership, sedans are sold. "
    "The dealership predicts that it will sell 28 sports cars next month. "
    "How many sedans does it expect to sell?",
    "49",
)

# source row 67: a whole premise clause is deleted
TRIANGLE = (
    "The sides of a triangle have lengths 6.5, 10, and s, where s is a whole number. "
    "What is the smallest possible value of s?",
    "The sides of a triangle have lengths 6.5, 10, and s. What is the smallest possible value of s?",
    "4",
)

# source row 35: the deleted span is the question's own object
HARMONIC = (
    "The harmonic mean of a set of non-zero numbers is the reciprocal of the average of the "
    "reciprocals of the numbers. What is the harmonic mean of 1, 2, and 4?",
    "The harmonic mean of a set of non-zero numbers is the reciprocal of the average of the "
    "reciprocals of the numbers. What is the harmonic mean?",
    "\\frac{12}{7}",
)

# source row 2570: the inserted span carries the impossibility word "negative"
PRIME_P = (
    "What expression is never a prime number when $p$ is a prime number?",
    "What expression is never a prime number when $p$ is a prime number, and $p$ is a negative decimal number?",
    "$p^2+26$",
)

# source row 443: the inserted span is the vague quantifier "some"
TENNIS = (
    "The longest professional tennis match ever played lasted a total of 11 hours and 5 minutes. "
    "How many minutes was this?",
    "The longest professional tennis match ever played lasted a total of 11 hours and some minutes. "
    "How many minutes was this?",
    "665",
)

# source row 2352: the rewrite makes the slope depend on an unnamed number
SLOPE_AB = (
    "If y=a+\\frac{b}{x}, where a and b are constants, and if y=1 when x=-1, and y=5 when x=-5, then a+b equals:",
    "If y = a + \\frac{b}{x}, where a and b are constants, and if y = 1 when x = -1 and "
    "y = 5 when x is some other negative number, then a+b equals:",
    "11",
)

# source row 7: three separate regions -- under the superseded design the pair was
# dropped because a pointer gold would have had to pick one of them (D23 has no
# pointer, so the pair is usable like any other)
SOFTBALL = (
    "During the softball season, Judy had 35 hits. Among her hits were 1 home run, 1 triple and "
    "5 doubles. The rest of her hits were single. What percent of her hits were single?",
    "During the softball season, Judy got several hits. Among her hits were 1 home run, 1 triple, "
    "and 5 doubles. The rest of her hits were singles. What percent of her hits were singles?",
    "\\frac{5}{7}",
)

# source row 805: the only change is the operator y != 0 -> y > 0
REMAINDER = (
    "The remainder can be defined for all real numbers x and y with y ≠ 0 by "
    "rem(x, y) = x - y ⌊x/y⌋ where ⌊x/y⌋ denotes the greatest integer less than or equal to x/y. "
    "What is the value of rem(3/8, -2/5)?",
    "The remainder can be defined for all real numbers x and y with y > 0 by "
    "rem(x, y) = x - y ⌊x/y⌋ where ⌊x/y⌋ denotes the greatest integer less than or equal to x/y. "
    "What is the value of rem(3/8, -2/5)?",
    "-\\frac{1}{40}",
)

# source row 2225: "then:" -> "then?" is a punctuation-only change
UNEQUAL = (
    "If $a$ and $b$ are two unequal positive numbers, then:",
    "If a and b are two unequal positive numbers, then?",
    "\\frac {a + b}{2} > \\sqrt {ab} > \\frac {2ab}{a + b}",
)

# source row 50: the inserted "c" is a single character
MIN_C = (
    "Find the minimum value of \\(\\sqrt{x^2+y^2}\\) if \\(5x+12y=60\\).",
    "Find the minimum value of \\(\\sqrt{x^2+y^2}\\) if \\(5x+12y = c\\).",
    "5",
)

# source row 175: the inserted "2112-" already occurs inside the original number
FACTOR = (
    "What is the value of \\(\\frac{(2112-2021)^2}{169}\\)?",
    "What is the value of \\(\\frac{(2112- )^2}{169}\\)?",
    "9",
)

# source row 31: the inserted span cannot host three equal-length option spans
GP = (
    "If x, 2x+2, 3x+3, ... are in geometric progression, the fourth term is:",
    "If x, 2x+2, 3x+3, ... are in geometric progression with the common ratio equal to the "
    "number of apples in an orchard, what is the fourth term?",
    "4x+4",
)

# source row 404: a deletion-only pair where every deleted word still stands elsewhere
SOUP = (
    "A can of soup can feed 3 adults or 5 children. If there are 5 cans of soup and 15 children "
    "are fed, then how many adults would the remaining soup feed?",
    "A can of soup can feed 3 adults. If there are 5 cans of soup and 15 children are fed, then "
    "how many adults would the remaining soup feed?",
    "3",
)

# source row 35212: the rewrite recasts the imperative as a question
NON_DEFECT = (
    "Simplify $2w+4w+6w+8w+10w+12$.",
    "2w+4w+6w+8w+10w+12. How?",
    "30w+12",
)

# source row 436: a month with a fixed number of Mondays and Wednesdays
MONTH_A = (
    "A month with 31 days has the same number of Mondays and Wednesdays. How many of the seven "
    "days of the week could be the first day of this month?",
    "A month has the same number of Mondays and Wednesdays. How many of the seven days of the "
    "week could be the first day of this month?",
    "2",
)
# source row 2528: the same text pair with a different gold -> contradictory input
MONTH_B_CONFLICT = (*MONTH_A[:2], "3")

# source row 184: a pair that also occurs later in the file with the same gold
BUG_A = (
    "A bug crawls along a number line, starting at -2. It crawls to -6, then turns around and "
    "crawls to 5. How many units does the bug crawl altogether?",
    "A bug crawls along a number line. It crawls to -6, then turns around and crawls to 5. "
    "How many units does the bug crawl altogether?",
    "15",
)
BUG_B_DUPLICATE = BUG_A

#: The usable pairs: every shape the superseded diff-driven design cared about.
#: Under D23 none of the shapes routes anything -- each one is one pair row.
PAIRS = [
    MEDIANS,
    FORALL,
    NUMBERS,
    SEDANS,
    TRIANGLE,
    HARMONIC,
    PRIME_P,
    TENNIS,
    SLOPE_AB,
    SOFTBALL,
    REMAINDER,
    UNEQUAL,
    MIN_C,
    FACTOR,
    GP,
    SOUP,
    NON_DEFECT,
]

#: Rows whose gold contradicts an earlier row, or repeats it.
DUPLICATES = [MONTH_A, MONTH_B_CONFLICT, BUG_A, BUG_B_DUPLICATE]

#: every fixture read at once, in this order (17 pairs + 4 duplicate rows)
ALL = [*PAIRS, *DUPLICATES]

_FIELDS = ("answerable_question", "unanswerable_question", "ground_truth")

#: the funnel of those 21 raw rows, stage by stage (one pair -> one row)
EXPECTED_FUNNEL = {
    "raw_rows": 21,
    "after_malformed_drop": 21,
    "after_duplicate_pair_drop": 18,
    "after_quota_cap": 18,
}

#: the rejection clauses behind that funnel
EXPECTED_DUPLICATE_DROPS = {"conflicting_gold": 2, "same_gold_duplicate": 1}

#: seed 0's A/B draw for fixture rows 0..5, pinned literally so a change to the
#: seeded rule fails here rather than silently re-randomising the artifact
SEED0_LABELS = ("A", "B", "B", "A", "A", "A")

_PROMPT_A = "问题 A："
_PROMPT_B = "问题 B："
_PROMPT_TAIL = "\n\n请先判断哪个问题可解"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _rows(*specs: tuple) -> list[dict]:
    """Fixture tuples -> source rows, in the source file's own column order."""
    return [dict(zip(_FIELDS, spec, strict=False)) for spec in specs]


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


def _clean(tmp_path, **kwargs):
    return _build(tmp_path, *PAIRS, **kwargs)


def _by_id(rows: list[dict]) -> dict[str, dict]:
    return {row["extra_info"]["task_id"]: row for row in rows}


def _payload(row: dict) -> dict:
    return json.loads(row["reward_model"]["ground_truth"])


def _prompt_questions(row: dict) -> tuple[str, str]:
    """``(question A, question B)`` taken out of the prompt by string surgery.

    Deliberately a different technique from ``verify_sum.py``'s regex, so neither the
    adapter nor the audit can hide a template change behind a shared parser.
    """
    content = row["prompt"][0]["content"]
    head, mark, rest = content.partition(_PROMPT_A)
    assert mark, "the prompt does not label question A"
    assert head.startswith("下面给出两个问题"), head[:40]
    question_a, mark, rest = rest.partition("\n\n" + _PROMPT_B)
    assert mark, "the prompt does not label question B"
    question_b, mark, rest = rest.partition(_PROMPT_TAIL)
    assert mark, "the prompt has no template tail after question B"
    return question_a, question_b


def _named_questions(row: dict) -> tuple[str, str]:
    """``(answerable question, unanswerable question)`` as the row presents them."""
    question_a, question_b = _prompt_questions(row)
    if _payload(row)["answerable_id"] == "A":
        return question_a, question_b
    return question_b, question_a


def _score(response: str, ground_truth: str) -> float:
    """The frozen dispatcher's score for one SUM solution string."""
    assert _compute_score is not None
    solution = f"<think>some reasoning</think>\n\n{response}"
    return _compute_score(schema.SOURCE_SUM, solution, ground_truth)["score"]


def _pair_ground_truth(answer: str, answerable_id: str) -> str:
    """A SUM gold built the way the adapter builds it (section 3's example)."""
    return schema.build_ground_truth(
        solvable=True,
        answer=answer,
        has_diagnosis_label=False,
        perturbation_type=None,
        pair_task=True,
        answerable_id=answerable_id,
    )


def _swapped(row: dict) -> dict:
    """The same pair with A and B exchanged and ``answerable_id`` flipped (section 9)."""
    question_a, question_b = _prompt_questions(row)
    payload = _payload(row)
    info = row["extra_info"]
    return schema.make_row(
        data_source=schema.SOURCE_SUM,
        question=question_b,
        question_b=question_a,
        ground_truth=_pair_ground_truth(payload["answer"], "B" if payload["answerable_id"] == "A" else "A"),
        template=schema.TEMPLATE_C,
        branch=schema.BRANCH_SOLVABLE_PAIR,
        extra_info={"index": info["index"], "task_id": info["task_id"], "solvable": True},
    )


def _synthetic_pairs(count: int) -> list[tuple[str, str, str]]:
    """``count`` distinct, well-formed pairs.

    The pair task reads nothing out of the text but the two questions, so a synthetic
    pair exercises the same code path as a real one -- which is what makes a
    quota-sized fixture cheap.
    """
    return [
        (
            f"Synthetic problem {i}: a crate holds {i} bolts and {i + 1} nuts. How many fasteners are in the crate?",
            f"Synthetic problem {i}: a crate holds {i} bolts. How many fasteners are in the crate?",
            str(2 * i + 1),
        )
        for i in range(count)
    ]


#: more pairs than the 6,000 quota, so the cap is what shapes the artifact
BIG_PAIRS = _synthetic_pairs(6_500)


@pytest.fixture(scope="module")
def quota_source(tmp_path_factory):
    """A source with more usable pairs than the section 4.8 quota."""
    return _write_source(tmp_path_factory.mktemp("quota_raw"), *BIG_PAIRS)


@pytest.fixture(scope="module")
def quota_build(quota_source):
    """The default (quota-sized) build of :func:`quota_source`."""
    rows, funnel = adapter.build_rows(quota_source)
    schema.normalise_extra_info(rows)
    return rows, funnel


# ---------------------------------------------------------------------------
# the pair contract (section 5.1 / 5.2 template C)
# ---------------------------------------------------------------------------


def test_one_pair_row_per_usable_raw_row(tmp_path):
    rows, funnel = _clean(tmp_path)
    assert len(rows) == len(PAIRS)
    assert list(funnel.values())[0] == len(PAIRS)
    assert funnel["after_quota_cap"] == len(PAIRS)
    assert [row["extra_info"]["index"] for row in rows] == list(range(len(PAIRS)))


def test_every_row_satisfies_the_schema_contract(tmp_path):
    rows, _ = _clean(tmp_path)
    assert rows
    for row in rows:
        assert schema.validate_row(row) == []
    schema.validate_rows(rows)  # the fail-closed batch form
    assert len({tuple(sorted(row["extra_info"])) for row in rows}) == 1


def test_row_shape_is_the_d23_pair_contract(tmp_path):
    rows, _ = _clean(tmp_path)
    for row in rows:
        info = row["extra_info"]
        payload = _payload(row)
        assert row["data_source"] == schema.SOURCE_SUM
        assert row["ability"] == schema.SOURCES[schema.SOURCE_SUM][0] == "math"
        assert set(row) == {"data_source", "prompt", "ability", "reward_model", "extra_info"}
        assert info["branch"] == schema.BRANCH_SOLVABLE_PAIR
        assert info["template"] == schema.TEMPLATE_C
        assert info["solvable"] is True
        assert info["options"] == []
        assert payload["solvable"] is True
        assert payload["pair_task"] is True
        assert payload["answerable_id"] in ("A", "B")
        assert payload["has_diagnosis_label"] is False
        assert payload["perturbation_type"] is None
        assert payload["correct_option_id"] is None
        assert isinstance(payload["answer"], str) and payload["answer"]


def test_ground_truth_is_the_section_3_sum_example(tmp_path):
    """The payload is exactly section 3's SUM row, plus the null option id."""
    rows, _ = _clean(tmp_path)
    row = next(row for row in rows if _payload(row)["answer"] == "49")
    assert _payload(row) == {
        "answer": "49",
        "answerable_id": _payload(row)["answerable_id"],
        "correct_option_id": None,
        "has_diagnosis_label": False,
        "pair_task": True,
        "perturbation_type": None,
        "solvable": True,
    }


def test_both_questions_are_in_the_prompt(tmp_path):
    rows, _ = _clean(tmp_path)
    for row in rows:
        content = row["prompt"][0]["content"]
        question_a, question_b = _prompt_questions(row)
        assert question_a and question_b
        assert question_a != question_b
        assert _PROMPT_A + question_a in content
        assert _PROMPT_B + question_b in content
        # the pair-task wording and the output contract of section 5.2
        assert "请先判断哪个问题可解" in content
        assert "\\boxed{<可解问题的编号>: <最终答案>}" in content
        assert "例如 \\boxed{A: 42}" in content


def test_answerable_id_names_the_raw_answerable_question(tmp_path):
    """Section 9's contract assertion, checked against the fixture text itself."""
    rows, _ = _clean(tmp_path)
    assert rows
    for row in rows:
        index = row["extra_info"]["index"]
        answerable, unanswerable = _named_questions(row)
        assert answerable == adapter.normalise_question(PAIRS[index][0])
        assert unanswerable == adapter.normalise_question(PAIRS[index][1])
        assert _payload(row)["answer"] == PAIRS[index][2]


def test_gold_is_the_raw_source_string(tmp_path):
    """``answer`` is a copy of the column, not a re-derived value."""
    specs = [(SEDANS[0], SEDANS[1], "  49  ")]
    raw_dir = _write_source(tmp_path, *specs)
    rows, _ = adapter.build_rows(raw_dir)
    assert _payload(rows[0])["answer"] == "  49  "


def test_paired_original_text_is_the_unanswerable_member(tmp_path):
    rows, _ = _clean(tmp_path)
    for row in rows:
        assert row["extra_info"]["paired_original_text"] == _named_questions(row)[1]


def test_error_type_is_a_statistics_field_only(tmp_path):
    """D24's rule: the taxonomy never decides a contract branch, and never scores."""
    rows, _ = _clean(tmp_path)
    for row in rows:
        assert row["extra_info"]["error_type"] == adapter.MIXED_ERROR_TYPE
        assert "error_type" not in _payload(row)


def test_superseded_drop_fixtures_now_emit_pair_rows(tmp_path):
    """D23 retires the diff-driven branches: no shape is filtered any more.

    Every fixture below was dropped by the superseded design for a different reason
    (multi-region paraphrase, no word-level change, an insertion no rule names, an
    unminable option anchor, a gold that already occurred in the original).  Under the
    pair contract each one is simply a pair.
    """
    formerly_dropped = [SOFTBALL, REMAINDER, UNEQUAL, NON_DEFECT, GP, MIN_C, FACTOR, SOUP]
    rows, funnel = _clean(tmp_path)
    assert funnel["after_duplicate_pair_drop"] == len(PAIRS)
    indexes = {row["extra_info"]["index"] for row in rows}
    assert {PAIRS.index(spec) for spec in formerly_dropped} <= indexes


# ---------------------------------------------------------------------------
# the A/B randomisation (section 4.5)
# ---------------------------------------------------------------------------


def test_answerable_side_is_a_seeded_per_row_draw(tmp_path):
    rows, _ = _clean(tmp_path)
    for row in rows:
        index = row["extra_info"]["index"]
        expected = "A" if random.Random(f"0:pair:{index}").random() < 0.5 else "B"
        assert _payload(row)["answerable_id"] == expected
    # ... and the draw is pinned literally, so changing the rule is a visible change
    assert tuple(_payload(row)["answerable_id"] for row in rows[:6]) == SEED0_LABELS
    assert {_payload(row)["answerable_id"] for row in rows} == {"A", "B"}


def test_the_draw_follows_the_seed_and_the_index(tmp_path):
    raw_dir = _write_source(tmp_path, *PAIRS)
    first, _ = adapter.build_rows(raw_dir, seed=0)
    second, _ = adapter.build_rows(raw_dir, seed=7)

    def labels(rows):
        return [_payload(row)["answerable_id"] for row in rows]

    assert labels(first) != labels(second)
    for row in second:
        index = row["extra_info"]["index"]
        assert _payload(row)["answerable_id"] == ("A" if random.Random(f"7:pair:{index}").random() < 0.5 else "B")


def test_a_smaller_build_keeps_each_rows_draw(tmp_path):
    """The draw depends on the row, not on how many rows the build keeps."""
    raw_dir = _write_source(tmp_path, *PAIRS)
    full, _ = adapter.build_rows(raw_dir, seed=0)
    small, _ = adapter.build_rows(raw_dir, seed=0, limit=5)
    by_index = {row["extra_info"]["index"]: _payload(row)["answerable_id"] for row in full}
    for row in small:
        assert _payload(row)["answerable_id"] == by_index[row["extra_info"]["index"]]


def test_swapping_the_questions_flips_answerable_id_and_keeps_the_contract(tmp_path):
    rows, _ = _clean(tmp_path)
    for row in rows:
        swapped = _swapped(row)
        assert schema.validate_row(swapped) == []
        assert _prompt_questions(swapped) == _prompt_questions(row)[::-1]
        assert _payload(swapped)["answerable_id"] != _payload(row)["answerable_id"]
        assert _payload(swapped)["answer"] == _payload(row)["answer"]


# ---------------------------------------------------------------------------
# the 6,000-pair quota and the selection order (section 4.8 row 6)
# ---------------------------------------------------------------------------


def test_quota_is_six_thousand_pair_rows():
    assert adapter.PAIR_QUOTA == 6_000
    assert adapter.PAIR_QUOTA == 3_000 + 3_000  # 0.5/0.5 in the section 4.8 accounting


def test_default_build_is_capped_at_the_quota(quota_build):
    rows, funnel = quota_build
    assert len(rows) == adapter.PAIR_QUOTA
    assert funnel["raw_rows"] == len(BIG_PAIRS)
    assert funnel["after_duplicate_pair_drop"] == len(BIG_PAIRS)
    assert funnel["after_quota_cap"] == adapter.PAIR_QUOTA
    for row in rows:
        assert schema.validate_row(row) == []


def test_quota_selection_is_a_spread_shuffle_not_the_first_n(quota_build):
    rows, _ = quota_build
    indexes = [row["extra_info"]["index"] for row in rows]
    assert len(indexes) == len(set(indexes)) == adapter.PAIR_QUOTA
    assert set(indexes) != set(range(adapter.PAIR_QUOTA))
    # the artifact reaches deep into the file instead of stopping at row 6,000
    assert max(indexes) > 6_000
    assert sum(1 for index in indexes if index >= 5_000) > 500


def test_select_pairs_is_deterministic_and_prefix_stable():
    pairs = [{"_index": index} for index in range(10_000)]
    chosen = adapter._select_pairs(pairs, 6_000, 0)
    assert len(chosen) == 6_000
    assert [row["_index"] for row in chosen] == sorted(row["_index"] for row in chosen)
    assert adapter._select_pairs(pairs, 6_000, 0) == chosen
    assert adapter._select_pairs(pairs, 6_000, 1) != chosen
    # a smaller cap is a subset of a larger one, because the cap is not part of the seed
    small = {row["_index"] for row in adapter._select_pairs(pairs, 100, 0)}
    assert small <= {row["_index"] for row in chosen}
    # an under-cap source comes back whole, in source order
    assert adapter._select_pairs(pairs[:5], 100, 0) == pairs[:5]
    assert adapter._select_pairs(pairs, 0, 0) == []
    assert adapter._select_pairs([], 10, 0) == []


def test_ab_split_is_within_two_points_on_the_quota_build(quota_build):
    import verify_sum

    rows, _ = quota_build
    share_a = sum(1 for row in rows if _payload(row)["answerable_id"] == "A") / len(rows)
    assert abs(share_a - 0.5) <= verify_sum.AB_TOLERANCE
    audit = verify_sum.Audit()
    verify_sum.check_ab_balance(rows, audit)
    assert audit.failures == 0, "\n".join(audit.lines)


def test_ab_split_tolerance_widens_below_the_quota(quota_build):
    """A small build is checked at three sigma, not at a bound it cannot resolve."""
    import verify_sum

    rows, _ = quota_build
    audit = verify_sum.Audit()
    verify_sum.check_ab_balance(rows[:300], audit)
    assert audit.failures == 0, "\n".join(audit.lines)
    assert any("resolution" in line for line in audit.lines)
    # ... and a fixed draw is still caught at that size
    fixed = copy.deepcopy(rows[:300])
    for row in fixed:
        payload = _payload(row)
        payload["answerable_id"] = "A"
        row["reward_model"]["ground_truth"] = json.dumps(payload)
    audit = verify_sum.Audit()
    verify_sum.check_ab_balance(fixed, audit)
    assert audit.failures == 1


# ---------------------------------------------------------------------------
# the section 9 reward cells, driven through the frozen dispatcher
# ---------------------------------------------------------------------------


@requires_reward
def test_reward_cells_on_a_built_row(tmp_path):
    rows, _ = _clean(tmp_path)
    row = next(row for row in rows if _payload(row)["answer"] == "49")
    ground_truth = row["reward_model"]["ground_truth"]
    named = _payload(row)["answerable_id"]
    other = "B" if named == "A" else "A"
    assert _score(f"\\boxed{{{named}: 49}}", ground_truth) == 1.0
    assert _score(f"\\boxed{{{other}: 49}}", ground_truth) == 0.0  # wrong id, gold answer text
    assert _score(f"\\boxed{{{named}: 50}}", ground_truth) == 0.5  # right id, wrong answer
    assert _score("\\boxed{49}", ground_truth) == 0.0  # bare number: no id
    assert _score("\\boxed{UNSOLVABLE}", ground_truth) == 0.0
    assert _score("the answer is 49", ground_truth) == 0.0  # no box at all


@requires_reward
def test_reward_parse_equivalence_and_wrong_id(tmp_path):
    """Section 9: ``\\boxed{A: 42}`` == ``\\boxed{a：42}``; a wrong id scores 0."""
    ground_truth = _pair_ground_truth("49", "A")
    assert _score("\\boxed{A: 49}", ground_truth) == 1.0
    assert _score("\\boxed{a：49}", ground_truth) == 1.0  # lower case + full-width colon
    assert _score("\\boxed{A:49}", ground_truth) == 1.0  # no space after the colon
    assert _score("\\boxed{B: 49}", ground_truth) == 0.0
    assert _score("\\boxed{b: 49}", ground_truth) == 0.0
    assert _score("\\boxed{A: 50}", ground_truth) == 0.5
    assert _score("\\boxed{A: 49} and \\boxed{B: 49}", ground_truth) == 0.0  # the last box rules


@requires_reward
def test_reward_ab_swap_invariance(tmp_path):
    """Section 9: exchanging A/B and ``answerable_id`` leaves every cell unchanged."""
    rows, _ = _clean(tmp_path)
    for row in rows:
        swapped = _swapped(row)
        ground_truth = row["reward_model"]["ground_truth"]
        swapped_ground_truth = swapped["reward_model"]["ground_truth"]
        named = _payload(row)["answerable_id"]
        other = "B" if named == "A" else "A"
        gold = _payload(row)["answer"]
        for answer, wrong in ((gold, "0"),):
            assert _score(f"\\boxed{{{named}: {answer}}}", ground_truth) == 1.0
            assert _score(f"\\boxed{{{other}: {answer}}}", swapped_ground_truth) == 1.0
            assert _score(f"\\boxed{{{other}: {answer}}}", ground_truth) == 0.0
            assert _score(f"\\boxed{{{named}: {answer}}}", swapped_ground_truth) == 0.0
            assert _score(f"\\boxed{{{named}: {wrong}}}", ground_truth) == 0.5
            assert _score(f"\\boxed{{{other}: {wrong}}}", swapped_ground_truth) == 0.5


# ---------------------------------------------------------------------------
# the funnel: filtering, dedup, determinism and --limit
# ---------------------------------------------------------------------------


def test_funnel_stage_counts(tmp_path):
    _, funnel = _build(tmp_path, *ALL)
    stages = {key: value for key, value in funnel.items() if isinstance(value, int)}
    assert stages == EXPECTED_FUNNEL
    assert dict(funnel["duplicate_drops"]) == EXPECTED_DUPLICATE_DROPS


def test_funnel_is_monotone_and_every_stage_is_named(tmp_path):
    _, funnel = _build(tmp_path, *ALL)
    stages = [key for key, value in funnel.items() if isinstance(value, int)]
    assert set(stages) == set(EXPECTED_FUNNEL)
    assert set(funnel) == set(EXPECTED_FUNNEL) | {"duplicate_drops"}
    assert stages[0] == "raw_rows" and stages[-1] == "after_quota_cap"
    counts = [funnel[stage] for stage in stages]
    assert counts[0] == len(ALL)
    for earlier, later in zip(counts, counts[1:], strict=False):
        assert later <= earlier


def test_conflicting_duplicate_pair_drops_both_rows(tmp_path):
    """Rows 436/2528: the same text pair with golds "2" and "3" is contradictory."""
    rows, funnel = _build(tmp_path, MONTH_A, MONTH_B_CONFLICT)
    assert rows == []
    assert funnel["after_duplicate_pair_drop"] == 0
    assert dict(funnel["duplicate_drops"]) == {
        "conflicting_gold": 2,
        "same_gold_duplicate": 0,
    }


def test_same_gold_duplicate_pair_keeps_the_first(tmp_path):
    rows, funnel = _build(tmp_path, BUG_A, BUG_B_DUPLICATE)
    assert funnel["after_duplicate_pair_drop"] == 1
    assert [row["extra_info"]["index"] for row in rows] == [0]


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


def test_same_seed_same_rows(tmp_path):
    raw_dir = _write_source(tmp_path, *ALL)
    first, _ = adapter.build_rows(raw_dir, seed=0)
    second, _ = adapter.build_rows(raw_dir, seed=0)
    assert first == second


def test_seed_is_recorded_on_every_row(tmp_path):
    rows, _ = _clean(tmp_path, seed=5)
    assert {row["extra_info"]["seed"] for row in rows} == {5}


def test_limit_overrides_the_quota_and_keeps_the_selection_prefix(tmp_path):
    raw_dir = _write_source(tmp_path, *BIG_PAIRS)
    rows, funnel = adapter.build_rows(raw_dir, limit=25)
    assert len(rows) == 25
    assert funnel["after_quota_cap"] == 25
    full, _ = adapter.build_rows(raw_dir)
    chosen = {row["extra_info"]["index"] for row in rows}
    assert chosen <= {row["extra_info"]["index"] for row in full}


def test_limit_above_the_row_count_is_a_no_op(tmp_path):
    everything, _ = _clean(tmp_path)
    limited, _ = _clean(tmp_path, limit=10_000)
    assert limited == everything


def test_limit_zero_emits_nothing(tmp_path):
    rows, funnel = _clean(tmp_path, limit=0)
    assert rows == []
    assert funnel["after_quota_cap"] == 0


def test_task_ids_are_unique_stable_and_source_derived(tmp_path):
    rows, _ = _clean(tmp_path)
    ids = [row["extra_info"]["task_id"] for row in rows]
    assert len(ids) == len(set(ids))
    assert set(ids) == {f"sum-pair-{index}" for index in range(len(PAIRS))}
    for row in rows:
        assert row["extra_info"]["task_id"] == adapter._task_id(row["extra_info"]["index"])


def test_task_id_encodes_the_source_position():
    assert adapter._task_id(0) == "sum-pair-0"
    assert adapter._task_id(2528) == "sum-pair-2528"


def test_main_writes_a_parquet_and_prints_the_funnel(tmp_path, monkeypatch, capsys):
    out = tmp_path / "built" / "sum.parquet"
    report = tmp_path / "report.json"
    raw_dir = _write_source(tmp_path / "raw", *ALL)
    monkeypatch.setattr(
        sys,
        "argv",
        ["sum_adapter.py", "--raw-dir", raw_dir, "--out", str(out), "--report", str(report)],
    )
    adapter.main()
    printed = capsys.readouterr().out
    assert "raw_rows" in printed
    assert "after_quota_cap" in printed
    assert "per branch:" in printed
    assert "per template:" in printed
    assert "per answerable_id" in printed
    assert out.is_file()
    rows = schema.read_parquet_rows(str(out))
    assert len(rows) == 18
    for row in rows:
        assert schema.validate_row(row) == []

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["rows"] == 18
    assert payload["quota"] == adapter.PAIR_QUOTA
    assert payload["limit"] is None
    assert payload["funnel"]["raw_rows"] == len(ALL)
    assert payload["funnel"]["after_quota_cap"] == 18
    assert set(payload["per_branch"]) == {schema.BRANCH_SOLVABLE_PAIR}
    assert set(payload["per_template"]) == {schema.TEMPLATE_C}
    assert sum(payload["per_answerable_id"].values()) == 18
    assert set(payload["per_answerable_id"]) == {"A", "B"}


def test_breakdown_counts_and_sorts():
    rows = [{"x": "b"}, {"x": "a"}, {"x": "b"}]
    assert adapter._breakdown(rows, lambda row: row["x"]) == {"a": 1, "b": 2}
    assert adapter._breakdown([], lambda row: row["x"]) == {}


def test_test_split_note_reports_the_absent_file(tmp_path):
    raw_dir = _write_source(tmp_path, *PAIRS)
    assert adapter.test_split_note(raw_dir) == "test.parquet: absent"


def test_test_split_note_describes_a_test_split_without_a_pair(tmp_path):
    extra = {
        adapter.TEST_FILE: [
            {"prompt": "a", "ground_truth": "I don't know."},
            {"prompt": "b", "ground_truth": "I don't know."},
        ]
    }
    raw_dir = _write_source(tmp_path, *PAIRS, extra=extra)
    note = adapter.test_split_note(raw_dir)
    assert "2 rows" in note
    assert "distinct ground_truth=1" in note
    assert "no answerable_question" in note


# ---------------------------------------------------------------------------
# the text plumbing
# ---------------------------------------------------------------------------


def test_normalise_question_collapses_whitespace():
    assert adapter.normalise_question("  a\n\tb  c \r\n d ") == "a b c d"
    assert adapter.normalise_question(None) == ""
    assert adapter.normalise_question("") == ""


def test_looks_numeric():
    assert adapter.looks_numeric("49")
    assert adapter.looks_numeric(" -3.5 ")
    assert adapter.looks_numeric("1,200")
    assert not adapter.looks_numeric("\\frac{1}{3}")
    assert not adapter.looks_numeric("4x+4")
    assert not adapter.looks_numeric("")


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
    raw_dir = _write_source(tmp_path, *PAIRS)
    source = adapter.load_source(raw_dir)
    assert len(source) == len(PAIRS)
    assert [row["_index"] for row in source] == list(range(len(PAIRS)))
    with pytest.raises(FileNotFoundError):
        adapter.load_source(str(tmp_path / "nope"))


def test_decision_constants():
    assert adapter.DATA_SOURCE == schema.SOURCE_SUM == "halluc_math_sumpair"
    assert adapter.BRANCH_PAIR == schema.BRANCH_SOLVABLE_PAIR == "solvable_pair"
    assert adapter.DATA_FILE == "train.parquet"
    assert adapter.MIXED_ERROR_TYPE == "mixed"


# ---------------------------------------------------------------------------
# integration with verify_sum.py
# ---------------------------------------------------------------------------


def test_verify_sum_passes_on_the_fixture_artifact(tmp_path):
    """The layers a fixture artifact *can* satisfy: contract, balance, anchor.

    The quota supply check needs a source the size of the real file (see
    :func:`test_verify_sum_quota_check` and the real-source class below), and the Q10
    proxy needs rows that are a sample rather than a curated set of hard shapes (see
    :func:`test_verify_sum_spot_check_proxy_passes_on_a_typical_build`).
    """
    import verify_sum

    raw_dir = _write_source(tmp_path, *PAIRS)
    rows, _ = adapter.build_rows(raw_dir)
    schema.normalise_extra_info(rows)

    audit = verify_sum.Audit()
    verify_sum.check_contract(rows, audit)
    verify_sum.check_ab_balance(rows, audit)
    source = verify_sum.load_source(raw_dir)
    verify_sum.check_source_anchor(rows, audit, source, raw_dir)
    audit.flush()
    assert audit.failures == 0, "\n".join(audit.lines)


def test_verify_sum_rejects_a_tampered_answerable_side(tmp_path):
    import verify_sum

    raw_dir = _write_source(tmp_path, *PAIRS)
    rows, _ = adapter.build_rows(raw_dir)
    schema.normalise_extra_info(rows)
    source = verify_sum.load_source(raw_dir)
    victim = copy.deepcopy(rows[0])
    payload = _payload(victim)
    payload["answerable_id"] = "B" if payload["answerable_id"] == "A" else "A"
    victim["reward_model"]["ground_truth"] = json.dumps(payload)
    audit = verify_sum.Audit()
    verify_sum.check_source_anchor([victim], audit, source, raw_dir)
    assert audit.failures == 1
    assert "answerable_question" in "\n".join(audit.lines)


def test_verify_sum_rejects_a_tampered_prompt_question(tmp_path):
    import verify_sum

    raw_dir = _write_source(tmp_path, *PAIRS)
    rows, _ = adapter.build_rows(raw_dir)
    schema.normalise_extra_info(rows)
    source = verify_sum.load_source(raw_dir)
    victim = copy.deepcopy(rows[0])
    victim["prompt"][0]["content"] = victim["prompt"][0]["content"].replace(
        _PROMPT_A + _prompt_questions(victim)[0], _PROMPT_A + "a question from nowhere"
    )
    audit = verify_sum.Audit()
    verify_sum.check_source_anchor([victim], audit, source, raw_dir)
    assert audit.failures == 1


def test_verify_sum_rejects_a_tampered_gold(tmp_path):
    import verify_sum

    raw_dir = _write_source(tmp_path, *PAIRS)
    rows, _ = adapter.build_rows(raw_dir)
    schema.normalise_extra_info(rows)
    source = verify_sum.load_source(raw_dir)
    victim = copy.deepcopy(rows[0])
    payload = _payload(victim)
    payload["answer"] = "a different answer"
    victim["reward_model"]["ground_truth"] = json.dumps(payload)
    audit = verify_sum.Audit()
    verify_sum.check_source_anchor([victim], audit, source, raw_dir)
    assert audit.failures == 1
    assert "ground_truth" in "\n".join(audit.lines)


def test_verify_sum_fails_closed_without_the_raw_file(tmp_path):
    import verify_sum

    rows, _ = _clean(tmp_path)
    audit = verify_sum.Audit()
    verify_sum.check_source_anchor(rows, audit, [], str(tmp_path / "absent"))
    assert audit.failures == 1


def test_verify_sum_contract_rejects_a_non_pair_payload(tmp_path):
    import verify_sum

    rows, _ = _clean(tmp_path)
    victim = copy.deepcopy(rows[0])
    payload = _payload(victim)
    payload.pop("pair_task")
    victim["reward_model"]["ground_truth"] = json.dumps(payload)
    audit = verify_sum.Audit()
    verify_sum.check_contract([victim], audit)
    assert audit.failures >= 1
    assert "pair_task" in "\n".join(audit.lines)


def test_verify_sum_contract_rejects_a_broken_json_payload(tmp_path):
    import verify_sum

    rows, _ = _clean(tmp_path)
    victim = copy.deepcopy(rows[0])
    victim["reward_model"]["ground_truth"] = "{not json"
    audit = verify_sum.Audit()
    verify_sum.check_contract([victim], audit)
    assert audit.failures >= 1
    assert "not valid JSON" in "\n".join(audit.lines)


def test_verify_sum_prompt_parser_rejects_a_foreign_prompt(tmp_path):
    import verify_sum

    rows, _ = _clean(tmp_path)
    victim = copy.deepcopy(rows[0])
    victim["prompt"][0]["content"] = "just one question?"
    with pytest.raises(ValueError):
        verify_sum.prompt_questions(victim)
    audit = verify_sum.Audit()
    verify_sum.check_contract([victim], audit)
    assert audit.failures >= 1


def test_verify_sum_quota_check(tmp_path):
    import verify_sum

    rows, _ = _clean(tmp_path)
    big_source = _rows(*BIG_PAIRS)  # supply above the quota
    audit = verify_sum.Audit()
    verify_sum.check_quota(rows, audit, big_source)
    assert audit.failures == 0, "\n".join(audit.lines)

    too_many = rows * 400  # 6,800 pair rows
    audit = verify_sum.Audit()
    verify_sum.check_quota(too_many, audit, big_source)
    assert audit.failures == 1
    assert "quota" in "\n".join(audit.lines)

    audit = verify_sum.Audit()
    verify_sum.check_quota(rows, audit, [])
    assert audit.failures == 0  # the supply half is skipped, not silently passed

    audit = verify_sum.Audit()
    verify_sum.check_quota(rows, audit, _rows(*PAIRS))  # supply far below the quota
    assert audit.failures == 1
    assert "supply" in "\n".join(audit.lines)


def test_verify_sum_usable_supply_recomputes_the_dedup_rules():
    import verify_sum

    source = _rows(*PAIRS) + _rows(MONTH_A, MONTH_B_CONFLICT, BUG_A, BUG_B_DUPLICATE)
    source += _rows(*_synthetic_pairs(3))
    usable, conflicting, duplicate = verify_sum.usable_supply(source)
    assert conflicting == 2
    assert duplicate == 1
    assert usable == len(source) - 3


def test_verify_sum_spot_check_proxy_passes_on_a_typical_build(tmp_path):
    """The Q10 proxy on a sample-like fixture.

    :data:`PAIRS` is deliberately the set of hard shapes (three of its 17 pairs carry
    no change a word-level rule can name: two operator/punctuation-only rewrites and a
    deletion whose every word survives elsewhere), so it over-represents the proxy's
    floor; a build of ordinary paraphrases is what the check is calibrated against.
    """
    import verify_sum

    rows, _ = _build(tmp_path, *_synthetic_pairs(40))
    audit = verify_sum.Audit()
    verify_sum.check_spot_check(rows, audit, n=100, seed=0, label_file=None)
    assert audit.failures == 0, "\n".join(audit.lines)
    assert "rule-based proxy" in "\n".join(audit.lines)

    rows, _ = _clean(tmp_path / "curated")
    audit = verify_sum.Audit()
    verify_sum.check_spot_check(rows, audit, n=100, seed=0, label_file=None)
    assert audit.failures == 1  # the curated shapes are hard for the proxy on purpose


def test_verify_sum_spot_check_label_file(tmp_path):
    """``--label-file`` is the doc's human check; a partial review cannot clear it."""
    import verify_sum

    rows, _ = _build(tmp_path, *_synthetic_pairs(40))
    ids = [row["extra_info"]["task_id"] for row in rows]

    complete = tmp_path / "complete.json"
    complete.write_text(json.dumps({task_id: task_id != ids[0] for task_id in ids}), encoding="utf-8")
    audit = verify_sum.Audit()
    verify_sum.check_spot_check(rows, audit, n=100, seed=0, label_file=str(complete))
    assert audit.failures == 0, "\n".join(audit.lines)  # 1/40 = 2.5% is under the line
    assert "human verdicts" in "\n".join(audit.lines)

    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps({ids[0]: True}), encoding="utf-8")
    audit = verify_sum.Audit()
    verify_sum.check_spot_check(rows, audit, n=100, seed=0, label_file=str(partial))
    assert audit.failures == 1
    assert "no verdict" in "\n".join(audit.lines)

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({task_id: False for task_id in ids[:10]}), encoding="utf-8")
    audit = verify_sum.Audit()
    verify_sum.check_spot_check(rows, audit, n=100, seed=0, label_file=str(bad))
    assert audit.failures == 1
    assert "25.0% fail" in "\n".join(audit.lines)


def test_verify_sum_label_is_plausible_proxy():
    import verify_sum

    assert verify_sum.label_is_plausible(SEDANS[0], SEDANS[1]) == ""
    assert verify_sum.label_is_plausible(TENNIS[0], TENNIS[1]) == ""
    assert verify_sum.label_is_plausible(MEDIANS[0], MEDIANS[0]) == "the two members are identical"
    assert "no word-level difference" in verify_sum.label_is_plausible(REMAINDER[0], REMAINDER[1])
    assert verify_sum.label_is_plausible("", "b") == "an empty member"


def test_verify_sum_pair_corpus_labels_follow_the_gold(tmp_path):
    import verify_sum

    rows, _ = _clean(tmp_path)
    texts, labels = verify_sum.pair_corpus(rows, total_rows=0, seed=0)
    assert len(texts) == len(labels) == 2 * len(rows)
    for position, row in enumerate(rows):
        question_a, question_b = _prompt_questions(row)
        assert texts[2 * position] == question_a
        assert texts[2 * position + 1] == question_b
        named = _payload(row)["answerable_id"]
        assert labels[2 * position] == (1 if named == "A" else 0)
        assert labels[2 * position + 1] == (1 if named == "B" else 0)
    assert sum(labels) == len(rows)


def test_verify_sum_l3_pair_reports_both_sides_and_a_control(tmp_path):
    import verify_sum

    rows, _ = _clean(tmp_path)
    audit = verify_sum.Audit()
    verify_sum.check_l3_pair(rows, audit, folds=3, seed=0, min_support=1, total_rows=0)
    printed = "\n".join(audit.lines)
    assert "L3a corpus" in printed
    assert "L3b data reading" in printed
    assert "L3b shuffled-label control" in printed
    assert audit.failures == 0, printed


def test_verify_sum_l3_pair_handles_an_empty_corpus():
    """Every pair contributes one member of each label, so only an empty artifact
    can leave the estimator unidentifiable -- and it is reported, not crashed."""
    import verify_sum

    audit = verify_sum.Audit()
    verify_sum.check_l3_pair([], audit, folds=3, seed=0, min_support=1, total_rows=0)
    assert audit.failures == 0
    assert "empty artifact" in "\n".join(audit.lines)


def test_verify_sum_pool_measurement_is_deterministic():
    import verify_sum

    source = _rows(MEDIANS, SEDANS, SOFTBALL, REMAINDER)
    assert verify_sum.measure_pools(source) == {
        "pairs": 4,
        "no_word_change": 1,
        "three_tier_pool": 1,
        "four_tier_pool": 1,
        "multi_region": 1,
    }


def test_verify_sum_sample_rows_is_exhaustive_by_default():
    import verify_sum

    rows = [{"extra_info": {"task_id": f"sum-pair-{index}", "index": index}} for index in range(500)]
    assert len(verify_sum.sample_rows(rows, per_row=0, seed=0)) == 500
    capped = verify_sum.sample_rows(rows, per_row=5, seed=0)
    assert 50 <= len(capped) < 500  # the family floor, and not exhaustive


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
        assert funnel["after_duplicate_pair_drop"] == 36_342
        for row in rows:
            assert schema.validate_row(row) == []
        assert {row["extra_info"]["branch"] for row in rows} == {schema.BRANCH_SOLVABLE_PAIR}
        assert {row["extra_info"]["template"] for row in rows} == {schema.TEMPLATE_C}
        for row in rows:
            answerable, unanswerable = _named_questions(row)
            assert answerable and unanswerable and answerable != unanswerable

    @needs_real
    def test_real_quota_build_is_balanced_and_anchored(self):
        """The section 9 assertions on the artifact the doc is about."""
        import verify_sum

        rows, funnel = adapter.build_rows(adapter.DEFAULT_RAW_DIR)
        schema.normalise_extra_info(rows)
        assert len(rows) == adapter.PAIR_QUOTA
        assert funnel["after_quota_cap"] == adapter.PAIR_QUOTA
        audit = verify_sum.Audit()
        verify_sum.check_contract(rows, audit)
        verify_sum.check_ab_balance(rows, audit)
        source = verify_sum.load_source(adapter.DEFAULT_RAW_DIR)
        verify_sum.check_quota(rows, audit, source)
        verify_sum.check_source_anchor(rows, audit, source, adapter.DEFAULT_RAW_DIR)
        assert audit.failures == 0, "\n".join(audit.lines)
        assert "the doc's +-2pt" in "\n".join(audit.lines)

    @needs_real
    def test_real_ab_draw_is_the_seeded_rule(self):
        rows, _ = adapter.build_rows(adapter.DEFAULT_RAW_DIR, limit=200)
        for row in rows:
            index = row["extra_info"]["index"]
            assert _payload(row)["answerable_id"] == ("A" if random.Random(f"0:pair:{index}").random() < 0.5 else "B")

    @needs_real
    def test_real_test_split_note(self):
        note = adapter.test_split_note(adapter.DEFAULT_RAW_DIR)
        assert "test.parquet: 284 rows" in note
        assert "distinct ground_truth=1" in note
        assert "no answerable_question" in note
