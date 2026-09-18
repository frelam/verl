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
"""Tests for ``distractor_synth.py`` -- the D17 rule-based distractor engine.

Everything up to the end-to-end block is inline and tiny -- written into
``tmp_path`` in the real sources' own shapes, with the base problems constructed
directly rather than loaded.  The end-to-end block is the exception, and reads
``/home/charles/data/reasoning_rl/halluc/raw`` for the same reason the templates
do: the ``in_topic`` axis is a property of the *pairing* between two inventories
(a base has to share a topic word with a template), so a hand-built pool either
has to be tuned until it does -- which tests the fixture, not the code -- or has
no ``in_topic`` supply at all, and an earlier revision of these tests failed for
exactly that.  The templates themselves always come from the real inventory, in
both halves of the file.

The tests are grouped the way the module is:

* the safety rules -- :func:`ds.unsafe_reason`, :func:`ds.hardcoded_names` -- and
  the rule *order*, because the order is load-bearing: the placeholder count is
  checked before the question form, so a role-only question is reported as
  ``single_placeholder`` (nothing can be labelled on the number axis) rather than
  as a question.
* the template inventory and its funnel.
* actor mining, which is where this module shipped its two real defects: an
  imperative mined as a name (``"Create ate 4 pounds of chocolate."``) and an
  unfillable placeholder interpolated as a number (``"... fed n/a monkeys."``).
  Both have a test that fails without the rule that prevents them.
* the four loaders, including the dual-form handling that would otherwise drop a
  whole source silently.
* rendering one distractor: the three axes are *recomputed from the text that was
  built*, so every labelled request either lands in the requested cell or returns
  ``None`` -- never a differently-labelled row.
* planning and allocation, including the axis that the data refuses to balance at
  the design's 50/50 (see :func:`ds.allocate_role_budget`).
* row emission and the per-row self-check.
* ``synthesise`` end to end -- on a hand-built pool where the module's own
  reporting is what is under test, and on a slice of the real pools where the
  three axes have to come out balanced at the design's targets.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import distractor_synth as ds  # noqa: E402
import schema  # noqa: E402

# ---------------------------------------------------------------------------
# fixtures: templates
# ---------------------------------------------------------------------------

# The funnel fixture, laid out so that every stage drops something and every
# rejection reason has exactly one owner.  "The turtle of {role} weighs
# {number} pounds." appears three times with two different fills, so the
# "most common quadruple wins" rule and the summed ``count`` are both exercised.
TWO_STEP = [
    {
        "sentence_template": "The height of {role} is {number} feet.",
        "role": "Emma",
        "number": "8",
        "role_label": "nonoverlapped",
        "number_label": "in_range",
        "sentence_label": "in_topic",
    },
    {
        "sentence_template": "{role} bought {number} newspapers.",
        "role": "Jack",
        "number": "9",
        "role_label": "nonoverlapped",
        "number_label": "out_range",
        "sentence_label": "in_topic",
    },
    {
        "sentence_template": "The turtle of {role} weighs {number} pounds.",
        "role": "Janele's neighbor",
        "number": "100",
        "role_label": "nonoverlapped",
        "number_label": "out_range",
        "sentence_label": "out_topic",
    },
    {
        "sentence_template": "The turtle of {role} weighs {number} pounds.",
        "role": "Zoe",
        "number": "7",
        "role_label": "nonoverlapped",
        "number_label": "in_range",
        "sentence_label": "out_topic",
    },
    {
        "sentence_template": "The turtle of {role} weighs {number} pounds.",
        "role": "Janele's neighbor",
        "number": "100",
        "role_label": "nonoverlapped",
        "number_label": "out_range",
        "sentence_label": "out_topic",
    },
    # empty: skipped before it is counted at all
    {"sentence_template": "", "role": "Emma", "number": "8"},
    # one owner per rejection reason
    {
        "sentence_template": "The sun rises in the east.",
        "role": "Emma",
        "number": "8",
    },
    {
        "sentence_template": "How many apples does {role} have?",
        "role": "Emma",
        "number": "8",
    },
    {
        "sentence_template": "How many apples does {role} have out of {number}?",
        "role": "Emma",
        "number": "8",
    },
    {
        "sentence_template": "{role} is {number} inches taller than Steve.",
        "role": "Emma",
        "number": "8",
    },
    {
        "sentence_template": "{role} and Steve have {number} apples in total.",
        "role": "Emma",
        "number": "8",
    },
    {
        "sentence_template": "{role} is a knight who guards {number} doors.",
        "role": "Emma",
        "number": "8",
    },
    {
        "sentence_template": "{role} is {number} inches away from Oliver.",
        "role": "Emma",
        "number": "8",
    },
]

MSTEP = [
    {
        "sentence_template": "{role} is {number} years old.",
        "role": "Tom",
        "number": "11",
        "role_label": "nonoverlapped",
        "number_label": "in_range",
        "sentence_label": "out_topic",
    }
]

FUNNEL_EXPECTED = {
    "distinct_total": 11,
    "safe": 4,
    "rejected_by_reason": {
        "additive": 1,
        "hardcoded_name": 1,
        "no_placeholder": 1,
        "question_form": 1,
        "relational": 1,
        "single_placeholder": 1,
        "truth_functional": 1,
    },
}
# The four safe templates cover 1 + 1 + 3 (the turtle, seen three times) + 1 rows.
SAFE_ROWS = 6


def _template(
    text: str,
    *,
    role: str = "Emma",
    number: str = "8",
    role_label: str = "nonoverlapped",
    number_label: str = "in_range",
    sentence_label: str = "in_topic",
    count: int = 1,
) -> ds.Template:
    return ds.Template(
        sentence_template=text,
        role=role,
        number=number,
        role_label=role_label,
        number_label=number_label,
        sentence_label=sentence_label,
        count=count,
    )


HEIGHT = _template("The height of {role} is {number} feet.")
NEWSPAPERS = _template("{role} bought {number} newspapers.")
SALARY = _template("The salary of {role} is ${number} per month.")
# Shares ``magazines`` with JEWEL, so it is the one that can carry ``in_topic``
# there -- and, for the same reason, the one the overlap guard refuses an
# overlapped fill on (`impossible_cells`).
MAGAZINES = _template("{role} sold {number} magazines.")


@pytest.fixture()
def raw_dir(tmp_path):
    """A miniature raw directory holding only the two GSM-IC arrays."""
    directory = tmp_path / "raw"
    gsmic = directory / "gsmic"
    gsmic.mkdir(parents=True)
    (gsmic / "GSM-IC_2step.json").write_text(json.dumps(TWO_STEP), encoding="utf-8")
    (gsmic / "GSM-IC_mstep.json").write_text(json.dumps(MSTEP), encoding="utf-8")
    return directory


# ---------------------------------------------------------------------------
# fixtures: base problems
# ---------------------------------------------------------------------------

JEWEL = ds.BaseQuestion(
    question="Jewel bought 10 magazines to be sold at $3.50 each. How much will she gain?",
    answer="5",
    pool=ds.POOL_SUM,
    uid="sum:train:0",
    meta={"names": ["Jewel"]},
)

BRYAN = ds.BaseQuestion(
    question="Bryan took a look at his books and magazines and counted 12 in all.",
    answer="12",
    pool=ds.POOL_UMWP,
    uid="umwp:1",
    meta={"names": ["Bryan"]},
)

# No actor and no numbers: the base that can carry neither an overlapped fill nor
# an in_range one.
EQUATION = ds.BaseQuestion(
    question="Determine the value of the constant k such that the equation has one root.",
    answer="5",
    pool=ds.POOL_SUM,
    uid="sum:train:1",
    meta={},
)

KK = ds.BaseQuestion(
    question=(
        "There are 4 inhabitants on the island, each of whom is either a knight or "
        "a knave. Ava says that Ben is a knave."
    ),
    answer="knight knave knight knave",
    pool=ds.POOL_KK,
    uid="kk:clean__train__4ppl:0",
    role_words=("knight", "knave"),
    meta={"names": ["Ava", "Ben", "Cara", "Dan"]},
)


def _copy(base: ds.BaseQuestion) -> ds.BaseQuestion:
    """A base whose ``meta`` cannot be written through to the original.

    Needed because :func:`ds.mine_actors` writes ``meta["names"]`` in place: a
    test that mines over a module-level base silently empties the role supply of
    every test that runs after it.
    """
    return dataclasses.replace(base, meta=dict(base.meta))


def _candidate(cell: tuple[str, str, str], order: float, base=None) -> ds.Candidate:
    role_label, number_label, sentence_label = cell
    base = base or ds.BaseQuestion(
        question="q", answer="a", pool=ds.POOL_SUM, uid=f"sum:train:{int(order)}"
    )
    return ds.Candidate(
        base=base,
        distractor=ds.Distractor(
            role="Emma",
            number=str(int(order)),
            sentence="sentence",
            role_label=role_label,
            number_label=number_label,
            sentence_label=sentence_label,
            template="t",
        ),
        question="q",
        order=order,
    )


def _cell_supply(counts: dict[tuple[str, str, str], int]) -> list[ds.Candidate]:
    """Candidates spread evenly over the cells ``counts`` asks for."""
    out: list[ds.Candidate] = []
    order = 0.0
    for cell, count in sorted(counts.items()):
        for _ in range(count):
            out.append(_candidate(cell, order))
            order += 1
    return out


def _feasible_cells() -> list[tuple[str, str, str]]:
    """The cells a real build can fill -- ``impossible_cells`` removed."""
    return sorted(ds._all_cells(ds.AXIS_TARGETS) - ds.impossible_cells())


def _full_supply(
    per_cell: int, overrides: dict[tuple[str, str, str], int] | None = None
) -> list[ds.Candidate]:
    """``per_cell`` candidates in every *reachable* cell, then the overrides.

    A real candidate list never holds an impossible cell -- :func:`ds.choose_distractor`
    returns ``None`` for ``overlapped``/``in_topic`` -- so stocking one is asking the
    allocator to fill the quota with rows the build cannot produce.  It duly does,
    in the third pass, and the sentence axis lands at 0.50 against a 0.45 target:
    a fixture artefact that reads exactly like a balancer defect.
    """
    counts = {cell: per_cell for cell in _feasible_cells()}
    counts.update(overrides or {})
    return _cell_supply(counts)


# ---------------------------------------------------------------------------
# the safety rules
# ---------------------------------------------------------------------------


def test_unsafe_reason_accepts_a_plain_parameterised_sentence():
    assert ds.unsafe_reason("The height of {role} is {number} feet.") is None
    assert ds.unsafe_reason("{role} bought {number} newspapers.") is None


@pytest.mark.parametrize(
    "template, reason",
    [
        ("", "empty"),
        ("   ", "empty"),
        ("The sun rises in the east.", "no_placeholder"),
        # Only one of the two axes can be carried, so nothing can be labelled on
        # the other -- reported as such even though the sentence is also a question.
        ("How many apples does {role} have?", "single_placeholder"),
        ("Find {number} apples.", "single_placeholder"),
        ("How many apples does {role} have out of {number}?", "question_form"),
        ("{role} is {number} inches taller than Steve.", "relational"),
        ("{role} and Steve have {number} apples in total.", "additive"),
        ("{role} is a knight who guards {number} doors.", "truth_functional"),
        ("{role} is {number} inches away from Oliver.", "hardcoded_name"),
    ],
)
def test_unsafe_reason_reports_one_reason_per_rule(template, reason):
    assert ds.unsafe_reason(template) == reason


def test_hardcoded_names_skips_the_sentence_initial_capital():
    """``The height of ...`` opens on a capital that is not a name."""
    assert ds.hardcoded_names("The height of {role} is {number} feet.") == []


@pytest.mark.parametrize("word", ["January", "Monday", "The", "With"])
def test_hardcoded_names_skips_the_safe_proper_allowlist(word):
    assert word in ds.SAFE_PROPER
    assert ds.hardcoded_names(f"{{role}} is {{number}} inches from {word}.") == []


def test_hardcoded_names_finds_a_mid_sentence_proper_noun():
    assert ds.hardcoded_names("{role} is {number} inches taller than Steve.") == ["Steve"]
    assert ds.hardcoded_names("{role} and Oli have {number} apples.") == ["Oli"]


def test_hardcoded_names_ignores_all_caps_tokens():
    assert ds.hardcoded_names("{role} scored {number} points in the NBA.") == []


# ---------------------------------------------------------------------------
# templates: words, rendering, and the funnel
# ---------------------------------------------------------------------------


def test_template_words_strips_both_placeholders():
    assert ds.template_words("The height of {role} is {number} feet.") == {
        "height",
        "feet",
    }


def test_template_words_topic_only_drops_the_generic_vocabulary():
    text = "The height of {role} is {number} feet."
    assert ds.template_words(text, topic_only=True) == ds.Template(
        sentence_template=text,
        role="",
        number="",
        role_label="",
        number_label="",
        sentence_label="",
    ).topic_words
    assert "bought" not in ds.template_words(
        "{role} bought {number} newspapers.", topic_only=True
    )
    assert "bought" in ds.template_words("{role} bought {number} newspapers.")


def test_template_render_substitutes_and_strips():
    assert (
        _template("  {role} is {number} years old.  ").render("Tom", "11")
        == "Tom is 11 years old."
    )


def test_template_render_replaces_every_occurrence():
    assert (
        _template("{role} and {role} ate {number} pies.").render("Tom", "3")
        == "Tom and Tom ate 3 pies."
    )


def test_load_templates_keeps_the_most_common_fill(raw_dir):
    templates = {t.sentence_template: t for t in ds.load_templates(raw_dir)}
    turtle = templates["The turtle of {role} weighs {number} pounds."]
    # The (Janele's neighbor, 100, out_topic) quadruple is the one seen twice.
    assert turtle.role == "Janele's neighbor"
    assert turtle.number == "100"
    assert turtle.count == 3


def test_load_templates_sorts_by_count_then_text(raw_dir):
    templates = ds.load_templates(raw_dir)
    assert [t.count for t in templates] == sorted(
        (t.count for t in templates), reverse=True
    )
    assert len(templates) == FUNNEL_EXPECTED["distinct_total"]


def test_load_templates_skips_blank_rows(raw_dir):
    assert "" not in {t.sentence_template for t in ds.load_templates(raw_dir)}


def test_load_templates_raises_when_a_file_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_raw.py"):
        ds.load_templates(tmp_path)


def test_load_safe_templates_counts_every_reason_once(raw_dir):
    safe, report = ds.load_safe_templates(raw_dir)
    assert report["distinct_total"] == FUNNEL_EXPECTED["distinct_total"]
    assert report["safe"] == FUNNEL_EXPECTED["safe"]
    assert report["rejected_by_reason"] == FUNNEL_EXPECTED["rejected_by_reason"]


def test_load_safe_templates_reports_the_rows_the_safe_set_covers(raw_dir):
    _safe, report = ds.load_safe_templates(raw_dir)
    assert report["rows_covered_by_safe"] == SAFE_ROWS
    assert report["rows_total"] == SAFE_ROWS + 7
    assert len(report["rejected_examples"]) == 7


def test_load_safe_templates_keeps_only_safe_ones(raw_dir):
    safe, _report = ds.load_safe_templates(raw_dir)
    assert all(ds.unsafe_reason(t.sentence_template) is None for t in safe)


# ---------------------------------------------------------------------------
# base names
# ---------------------------------------------------------------------------


def test_base_names_finds_capitalised_content_words():
    assert ds.base_names("Jewel bought 10 magazines.") == ["Jewel"]


def test_base_names_accepts_a_sentence_initial_name():
    """UMWP problems open on their actor, so position 0 cannot simply be skipped."""
    assert ds.base_names("Bryan took a look at his books.") == ["Bryan"]


@pytest.mark.parametrize(
    "opener", ["Determine", "Proposed", "Calculate", "Additionally", "Compute"]
)
def test_base_names_rejects_a_sentence_initial_imperative(opener):
    assert opener.casefold() in ds.NON_NAME_OPENERS
    assert ds.base_names(f"{opener} the value of k in the equation.") == []


def test_base_names_splits_sentences_on_punctuation_not_on_words():
    """`schema.words` drops punctuation, so "b = a/b. Determine" must still split."""
    assert ds.base_names("Let b = a/b. Determine the value.") == []


def test_base_names_skips_safe_proper_and_all_caps():
    assert ds.base_names("In January the NBA had 4 games.") == []


def test_base_names_dedupes_in_document_order():
    assert ds.base_names("Ava met Ben. Cara met Ava.") == ["Ava", "Ben", "Cara"]


def test_base_names_is_the_noisy_heuristic_not_the_filtered_one():
    """"Later" is a sentence-initial capital this function cannot rule out.

    That is by design: :func:`ds.base_names` is the raw pass and
    :func:`ds.actor_candidates` is the filtered one -- the corpus writes "later"
    in lower case, which is what removes it there.
    """
    assert ds.base_names("Ava met Ben. Later Ava met Cara.") == [
        "Ava",
        "Ben",
        "Later",
        "Cara",
    ]
    assert ds.actor_candidates("Ava met Ben. Later Ava met Cara.", {"later"}) == [
        "Ava",
        "Ben",
        "Cara",
    ]


def test_base_names_keeps_a_possessive_as_one_token():
    assert ds.base_names("Katya's mother is a professor.") == ["Katya's"]


# ---------------------------------------------------------------------------
# actor mining
# ---------------------------------------------------------------------------


def _umwp(question: str, uid: str, names=None) -> ds.BaseQuestion:
    return ds.BaseQuestion(
        question=question,
        answer="1",
        pool=ds.POOL_UMWP,
        uid=uid,
        meta={"names": names} if names else {},
    )


def test_actor_candidates_rejects_a_titlecase_verb_attested_in_lower_case():
    """The one signal that separates ``Point`` from ``Alice``: the corpus writes
    "point" and never writes "alice"."""
    text = "Create ate 4 pounds of chocolate."
    assert ds.actor_candidates(text, {"create"}) == []
    # With no vocabulary at all the same token is mined, which is the defect the
    # lowercase vocabulary exists to prevent.
    assert ds.actor_candidates(text, set()) == ["Create"]


@pytest.mark.parametrize("word", ["Chinese", "Texas", "Mr", "Professor", "Christmas"])
def test_actor_candidates_rejects_non_actor_proper_nouns(word):
    assert word in ds.NON_ACTOR_PROPER
    assert ds.actor_candidates(f"{word} arrived. What is {word}?", set()) == []


def test_actor_candidates_rejects_a_latex_macro():
    assert ds.actor_candidates("EndArrow is a macro.", set()) == []


def test_actor_candidates_strips_the_possessive_clitic():
    """The defect: a possessive substituted as an actor renders "Bill's baked ...".

    Measured on the real SUM pool before the strip: 87 of 1,299 actor-bearing bases
    mined a possessive, 25 of which reached the shipped rows.
    """
    assert ds.actor_candidates("Bill's brother has 5 apples.", set()) == ["Bill"]
    assert ds.actor_candidates("James bought 4 apples.", set()) == ["James"]


def test_actor_candidates_rejects_an_opener_at_any_position():
    """The defect: a sentence-initial imperative mined as an actor mid-sentence."""
    assert ds.actor_candidates("Then Proposed ate 4 pounds of chocolate.", set()) == []
    assert ds.actor_candidates("Proposed fed 4 monkeys.", set()) == []


def test_mine_actors_requires_pool_support():
    pools = {
        ds.POOL_SUM: [],
        ds.POOL_UMWP: [
            _umwp("Ava has 2 apples.", "umwp:1"),
            _umwp("Ava has 3 apples.", "umwp:2"),
            _umwp("Ava has 5 apples.", "umwp:3"),
            _umwp("Ben has 7 apples.", "umwp:4"),
        ],
        ds.POOL_KK: [],
    }
    report = ds.mine_actors(pools, min_support=3)
    # Ava is attested three times, Ben once: only Ava clears the floor.
    assert [base.meta["names"] for base in pools[ds.POOL_UMWP]] == [
        ["Ava"],
        ["Ava"],
        ["Ava"],
        [],
    ]
    assert report["pools"][ds.POOL_UMWP]["bases_with_actor"] == 3
    assert report["pools"][ds.POOL_UMWP]["coverage"] == 0.75
    assert report["pools"][ds.POOL_UMWP]["top_actors"] == ["Ava"]


def test_mine_actors_seeds_the_vocabulary_from_the_kk_inhabitants():
    """K&K's puzzle lists its people, so they need no support in UMWP."""
    pools = {ds.POOL_SUM: [], ds.POOL_UMWP: [], ds.POOL_KK: [KK, KK]}
    report = ds.mine_actors(pools, min_support=3)
    assert report["pools"][ds.POOL_KK]["bases_with_actor"] == 2
    assert set(report["pools"][ds.POOL_KK]["top_actors"]) == {"Ava", "Ben", "Cara", "Dan"}


def test_mine_actors_leaves_a_base_without_names_empty():
    """Fail-closed: a base with no mined actor simply has no overlapped cell."""
    equation, jewel = _copy(EQUATION), _copy(JEWEL)
    pools = {ds.POOL_SUM: [equation, jewel], ds.POOL_UMWP: [], ds.POOL_KK: []}
    ds.mine_actors(pools)
    assert ds.base_names_of(equation) == []
    assert ds.base_names_of(jewel) == []


def test_mine_actors_reports_the_supply_it_measured():
    pools = {ds.POOL_SUM: [_copy(JEWEL)], ds.POOL_UMWP: [], ds.POOL_KK: []}
    report = ds.mine_actors(pools)
    assert report["min_pool_support"] == ds.ACTOR_MIN_POOL_SUPPORT
    assert report["pools"][ds.POOL_SUM]["coverage"] == 0.0
    assert report["lowercase_vocabulary"] > 0


def test_lowercase_vocabulary_is_lower_case_content_words_only():
    pool = [_umwp("Ava bought 4 Apples in the Store.", "umwp:1")]
    vocabulary = ds.lowercase_vocabulary({ds.POOL_UMWP: pool})
    assert "bought" in vocabulary
    assert "ava" not in vocabulary
    assert "apples" not in vocabulary


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ([460.0], [460.0]),
        ("[460.0]", [460.0]),
        (460.0, [460.0]),
        ("460.0", [460.0]),
        (None, []),
        ("not a list", []),
    ],
)
def test_answer_values_accepts_both_the_list_and_the_repr(raw, expected):
    assert ds._answer_values(raw) == expected


def test_render_number_drops_a_float_integral_point():
    assert ds._render_number(460.0) == "460"
    assert ds._render_number(460.5) == "460.5"


def test_kk_answer_uses_the_rows_own_word_pair():
    assert (
        ds.kk_answer(["A", "B"], [True, False], {"knight": "truth-teller", "knave": "liar"})
        == "truth-teller liar"
    )
    assert ds.kk_answer(["A"], [True], {}) == "knight"


def _write_sum(path, rows) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "answerable_question": [row[0] for row in rows],
                "ground_truth": [row[1] for row in rows],
            }
        ),
        path,
    )


def test_load_sum_answerable_keys_by_row_ordinal(tmp_path):
    _write_sum(
        tmp_path / "sum" / "train.parquet",
        [("A problem.", "5"), ("", "3"), ("Another.", "7")],
    )
    bases = ds.load_sum_answerable(tmp_path)
    # The blank row is dropped but the ordinal is not renumbered: the key has to
    # stay addressable in the raw file.
    assert [(b.uid, b.answer) for b in bases] == [
        ("sum:train:0", "5"),
        ("sum:train:2", "7"),
    ]


def test_load_sum_answerable_raises_on_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        ds.load_sum_answerable(tmp_path)


def _write_umwp(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_load_umwp_answerable_keeps_only_the_answerable_half(tmp_path):
    _write_umwp(
        tmp_path / "umwp" / "StandardDataset.jsonl",
        [
            {"id": 1, "question": "John buys 3 reels.", "answer": [30.0], "answerable": True},
            {"id": 2, "question": "John cuts it.", "answer": None, "answerable": False},
            {"id": 3, "question": "Mary runs 4 miles.", "answer": [4.0], "answerable": True},
        ],
    )
    bases = ds.load_umwp_answerable(tmp_path)
    assert [(b.uid, b.question, b.answer) for b in bases] == [
        ("umwp:1", "John buys 3 reels.", "30"),
        ("umwp:3", "Mary runs 4 miles.", "4"),
    ]


def test_load_umwp_answerable_reads_a_repr_answer(tmp_path):
    """Older dumps hand back the list as a string; dropping them is the failure."""
    _write_umwp(
        tmp_path / "umwp" / "StandardDataset.jsonl",
        [{"id": 1, "question": "Q?", "answer": "[460.0]", "answerable": "True"}],
    )
    assert ds.load_umwp_answerable(tmp_path)[0].answer == "460"


def test_load_umwp_answerable_keys_by_physical_line(tmp_path):
    """The key is the line, so a skipped record does not renumber the rest."""
    _write_umwp(
        tmp_path / "umwp" / "StandardDataset.jsonl",
        [
            {"id": 1, "question": "Q1?", "answer": [1.0], "answerable": True},
            {"id": 2, "question": "Q2?", "answer": None, "answerable": False},
            {"id": 3, "question": "Q3?", "answer": [3.0], "answerable": True},
        ],
    )
    assert [b.uid for b in ds.load_umwp_answerable(tmp_path)] == ["umwp:1", "umwp:3"]


def _write_kk(directory, stem: str, rows) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "quiz": [row[0] for row in rows],
                "names": [row[1] for row in rows],
                "knight_knave": [row[2] for row in rows],
                "solution": [row[3] for row in rows],
                "index": [row[4] for row in rows],
            }
        ),
        directory / f"clean__train__{stem}.parquet",
    )


def test_load_kk_clean_keys_by_file_and_index(tmp_path):
    """``index`` restarts at 0 in each per-size file, so the file is part of the key."""
    directory = tmp_path / "kk"
    words = {"knight": "knight", "knave": "knave"}
    _write_kk(directory, "4ppl", [("Four person puzzle.", ["A", "B", "C", "D"], words, [True] * 4, 0)])
    _write_kk(directory, "5ppl", [("Five person puzzle.", ["A", "B", "C", "D", "E"], words, [False] * 5, 0)])
    bases = ds.load_kk_clean(tmp_path)
    assert [b.uid for b in bases] == [
        "kk:clean__train__4ppl:0",
        "kk:clean__train__5ppl:0",
    ]
    assert len({b.uid for b in bases}) == 2
    assert bases[0].answer == "knight knight knight knight"
    assert bases[1].answer == "knave knave knave knave knave"


def test_load_kk_clean_applies_the_inhabitant_floor(tmp_path):
    _write_kk(
        tmp_path / "kk",
        "3ppl",
        [("Three is too few.", ["A", "B", "C"], {"knight": "knight", "knave": "knave"}, [True] * 3, 0)],
    )
    assert ds.load_kk_clean(tmp_path) == []
    assert len(ds.load_kk_clean(tmp_path, min_inhabitants=3)) == 1


def test_load_kk_clean_drops_a_solution_of_the_wrong_length(tmp_path):
    _write_kk(
        tmp_path / "kk",
        "4ppl",
        [("Mismatched.", ["A", "B", "C", "D"], {"knight": "knight", "knave": "knave"}, [True], 0)],
    )
    assert ds.load_kk_clean(tmp_path) == []


def test_load_kk_clean_reads_reprs_as_well_as_typed_columns(tmp_path):
    """A JSONL dump hands back reprs where the parquet gives real lists."""
    _write_kk(
        tmp_path / "kk",
        "4ppl",
        [
            (
                "Repr form.",
                "['A', 'B', 'C', 'D']",
                "{'knight': 'knight', 'knave': 'knave'}",
                "[True, False, True, False]",
                0,
            )
        ],
    )
    assert ds.load_kk_clean(tmp_path)[0].answer == "knight knave knight knave"


def _write_main(path, rows) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "prompt": [[{"role": "user", "content": row[0]}] for row in rows],
                "reward_model": [{"ground_truth": row[1]} for row in rows],
                "extra_info": [{"split": row[2]} for row in rows],
            }
        ),
        path,
    )


def test_load_main_pool_reads_the_prompt_and_the_gold_payload(tmp_path):
    path = tmp_path / "stage1" / "train.parquet"
    _write_main(
        path,
        [
            ("How many panes?", json.dumps({"answer": "42", "solvable": True}), "train"),
            ("No payload.", "not json", "train"),
        ],
    )
    bases = ds.load_main_pool(path)
    assert [(b.uid, b.answer, b.pool) for b in bases] == [("main:0", "42", ds.POOL_MAIN)]
    assert bases[0].split == "train"


def test_load_main_pool_honours_the_limit(tmp_path):
    path = tmp_path / "stage1" / "t.parquet"
    _write_main(
        path,
        [(f"Q{i}?", json.dumps({"answer": str(i)}), "train") for i in range(5)],
    )
    assert len(ds.load_main_pool(path, limit=2)) == 2


def test_load_pools_notes_an_absent_stage1_slice(tmp_path):
    pools, notes = ds.load_pools(tmp_path, None)
    assert pools[ds.POOL_MAIN] == []
    assert notes[ds.POOL_MAIN] == "no --stage1-path given"


def test_load_pools_notes_a_missing_raw_file(tmp_path):
    pools, notes = ds.load_pools(tmp_path, None)
    assert pools[ds.POOL_SUM] == []
    assert "raw file missing" in notes[ds.POOL_SUM]


# ---------------------------------------------------------------------------
# rendering one candidate
# ---------------------------------------------------------------------------


def test_inject_puts_the_sentence_just_before_the_question():
    injected = ds.inject("Jewel bought 10 magazines. How much will she gain?", "Tom is 8.")
    assert injected == "Jewel bought 10 magazines. Tom is 8. How much will she gain?"


def test_inject_appends_when_the_base_does_not_end_in_a_question():
    assert ds.inject("Compute the value. It is constant.", "Tom is 8.") == (
        "Compute the value. It is constant. Tom is 8."
    )


def test_inject_collapses_the_newlines_inside_a_multi_line_base():
    """A bare newline would stop the base being a substring of the prompt."""
    injected = ds.inject("Find n such that\nP(0) = P(3). What is n?", "Tom is 8.")
    assert "\n" not in injected
    assert "Find n such that P(0) = P(3)." in injected


def test_inject_returns_the_sentence_for_an_empty_base():
    assert ds.inject("", "Tom is 8.") == "Tom is 8."


def test_inject_normalises_the_sentence_whitespace():
    assert ds.inject("Q?", "Tom  is\n8.") == "Tom is 8. Q?"


def test_choose_distractor_lands_in_the_requested_cell():
    """One legal cell per sentence label: ``overlapped`` implies ``out_topic``.

    The two requests differ in the template as well as the role label, because
    the sentence label is a property of the *pair*: ``MAGAZINES`` shares
    ``magazines`` with the base and so is the in_topic one, while ``HEIGHT`` --
    which shares nothing with the base -- is what an overlapped fill can use.
    """
    got = ds.choose_distractor(
        MAGAZINES,
        JEWEL,
        random.Random(0),
        names=["Emma"],
        numbers=["8"],
        want_role="nonoverlapped",
        want_number="in_range",
        want_sentence="in_topic",
    )
    assert got is not None
    assert got.cell == ("nonoverlapped", "in_range", "in_topic")

    got = ds.choose_distractor(
        HEIGHT,
        JEWEL,
        random.Random(0),
        names=["Emma"],
        numbers=["8"],
        want_role="overlapped",
        want_number="in_range",
        want_sentence="out_topic",
    )
    assert got is not None
    assert got.cell == ("overlapped", "in_range", "out_topic")
    assert got.role == "Jewel"


def test_choose_distractor_returns_none_rather_than_a_different_cell():
    """An explicit request that cannot be met must not silently land elsewhere."""
    got = ds.choose_distractor(
        HEIGHT,
        EQUATION,
        random.Random(0),
        names=["Emma"],
        numbers=["8"],
        want_role="overlapped",
        want_number="in_range",
        want_sentence="in_topic",
    )
    assert got is None


def test_choose_distractor_overlapped_pool_is_the_bases_own_names():
    got = ds.choose_distractor(
        HEIGHT, JEWEL, random.Random(0), names=["Emma"], numbers=["8"], want_role="overlapped"
    )
    assert got is not None
    assert got.role == "Jewel"
    assert got.role_label == "overlapped"


def test_choose_distractor_nonoverlapped_pool_excludes_a_name_the_base_carries():
    """A name the base already mentions cannot be a *non*overlapped fill."""
    roles = set()
    for seed in range(30):
        got = ds.choose_distractor(
            HEIGHT,
            JEWEL,
            random.Random(seed),
            names=["Emma", "Jewel"],
            numbers=["8"],
            want_role="nonoverlapped",
        )
        assert got is not None
        roles.add(got.role)
    assert roles == {"Emma"}


def test_choose_distractor_refuses_an_overlap_that_restates_a_property():
    """The guard: an overlapped fill may not reuse the base's content words."""
    got = ds.choose_distractor(
        NEWSPAPERS,
        ds.BaseQuestion(
            question="Jewel bought 10 newspapers. How many did she buy?",
            answer="10",
            pool=ds.POOL_SUM,
            meta={"names": ["Jewel"]},
        ),
        random.Random(0),
        names=["Emma"],
        numbers=["8"],
        want_role="overlapped",
    )
    assert got is None


def test_choose_distractor_refuses_a_sentence_already_in_the_base():
    """A fill that renders a sentence the base already contains is not a red herring.

    The last-line check compares the *rendered* sentence with the base, and it is
    reachable only for a template that carries no content word of its own: an
    overlapped role is taken from the base's own names, so whenever the render is
    a substring of the base every literal word of the template is a word of the
    base too, and the overlap guard above has already refused it.
    ``{role} is {number}.`` is the shape that gets past that guard --
    ``is`` is a stopword, so ``template_words`` is empty -- and it is what makes
    this check a second line of defence rather than dead code.  No template in the
    shipped inventory has that shape (394 distinct templates, 0 with an empty
    content-word set), so on today's data the guard is the one that fires.
    """
    base = ds.BaseQuestion(
        question="Tom is 11. How old is Tom?",
        answer="11",
        pool=ds.POOL_SUM,
        meta={"names": ["Tom"]},
    )
    template = ds.Template(
        sentence_template="{role} is {number}.",
        role="Tom",
        number="11",
        role_label="overlapped",
        number_label="in_range",
        sentence_label="out_topic",
        count=1,
    )
    assert ds.template_words(template.sentence_template) == set()
    assert (
        ds.choose_distractor(
            template,
            base,
            random.Random(0),
            names=["Emma"],
            numbers=["11"],
            want_role="overlapped",
            want_sentence="out_topic",
        )
        is None
    )


def test_choose_distractor_in_range_falls_back_to_a_whole_number_inside_the_span():
    """GSM-IC's number pool rarely lands inside a MATH problem's range."""
    base = ds.BaseQuestion(
        question="Jewel has 3.5 apples and buys 4.25 more.",
        answer="7.75",
        pool=ds.POOL_SUM,
        meta={"names": ["Jewel"]},
    )
    for seed in range(20):
        got = ds.choose_distractor(
            HEIGHT,
            base,
            random.Random(seed),
            names=["Emma"],
            numbers=["1000"],
            want_number="in_range",
            want_sentence="out_topic",
        )
        assert got is not None
        assert ds._in_range(got.number, ds._float_numbers(base.question))
        # A whole number, never the float the span happens to end on.
        assert got.number.isdigit()


def test_choose_distractor_out_range_needs_a_number_outside_the_span():
    got = ds.choose_distractor(
        HEIGHT,
        JEWEL,
        random.Random(0),
        names=["Emma"],
        numbers=["1000"],
        want_number="out_range",
        want_sentence="out_topic",
    )
    assert got is not None
    assert not ds._in_range(got.number, ds._float_numbers(JEWEL.question))


def test_choose_distractor_number_pool_is_never_the_literal_na():
    """The defect: an unfillable placeholder recorded as ``"n/a"`` reached a row."""
    got = ds.choose_distractor(
        HEIGHT,
        JEWEL,
        random.Random(0),
        names=["Emma"],
        numbers=["n/a"],
        want_number="in_range",
        want_sentence="out_topic",
    )
    assert got is None or got.number.isdigit()


def test_choose_distractor_marks_a_row_it_built_itself_consistently():
    got = ds.choose_distractor(HEIGHT, JEWEL, random.Random(0), names=["Emma"], numbers=["8"])
    assert got is not None
    assert got.sentence == HEIGHT.render(got.role, got.number)
    assert got.template == HEIGHT.sentence_template


def test_choose_distractor_returns_none_without_any_usable_role_pool():
    assert (
        ds.choose_distractor(HEIGHT, EQUATION, random.Random(0), names=[], numbers=["8"])
        is None
    )


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------


def test_impossible_cells_is_overlapped_and_in_topic():
    assert ds.impossible_cells() == {
        ("overlapped", "in_range", "in_topic"),
        ("overlapped", "out_range", "in_topic"),
    }


@pytest.mark.parametrize("quota", [8, 100, 2400])
def test_plan_cells_hits_the_marginals_exactly(quota):
    cells = ds.plan_cells(quota, impossible=ds.impossible_cells())
    assert sum(cells.values()) == quota
    counts = {
        axis: {label: 0 for label in ds.AXIS_TARGETS[axis]} for axis in ds.AXES
    }
    for cell, count in cells.items():
        for axis, label in zip(ds.AXES, cell):
            counts[axis][label] += count
    for axis in ds.AXES:
        for label, fraction in ds.AXIS_TARGETS[axis].items():
            assert abs(counts[axis][label] - fraction * quota) <= 1


def test_plan_cells_never_plans_a_forbidden_cell():
    cells = ds.plan_cells(400, impossible=ds.impossible_cells())
    assert not (set(cells) & ds.impossible_cells())


def test_plan_cells_reports_a_shortfall_instead_of_skewing():
    """45% in_topic cannot carry a 90% overlapped role: the plan must come up short."""
    targets = {
        "role_label": {"overlapped": 0.90, "nonoverlapped": 0.10},
        "number_label": {"in_range": 0.50, "out_range": 0.50},
        "sentence_label": {"in_topic": 0.45, "out_topic": 0.55},
    }
    cells = ds.plan_cells(200, targets, ds.impossible_cells())
    # overlapped can only live in out_topic, whose budget is 110.
    assert sum(cells.values()) <= 200
    overlapped = sum(c for cell, c in cells.items() if cell[0] == "overlapped")
    assert overlapped <= round(0.55 * 200)


@pytest.mark.parametrize(
    "weights, total, expected",
    [
        ({"a": 1.0, "b": 1.0}, 5, {"a": 3, "b": 2}),
        ({"a": 1.0, "b": 1.0}, 4, {"a": 2, "b": 2}),
        ({"a": 0.0, "b": 1.0}, 4, {"a": 0, "b": 4}),
        ({"a": 1.0, "b": 1.0}, 0, {"a": 0, "b": 0}),
    ],
)
def test_largest_remainder_sums_to_exactly_the_total(weights, total, expected):
    got = ds._largest_remainder(weights, total)
    assert got == expected
    assert sum(got.values()) == total


def test_largest_remainder_handles_a_zero_mass():
    assert ds._largest_remainder({"a": 0.0, "b": 0.0}, 4) == {"a": 0, "b": 0}


# ---------------------------------------------------------------------------
# allocation
# ---------------------------------------------------------------------------


def test_allocate_fills_the_quota_and_holds_every_axis():
    selected, report = allocate_fixture(_full_supply(100), 320)
    assert len(selected) == 320
    assert report.shortfall == 0
    assert all(report.balanced.values()), report.achieved


def allocate_fixture(candidates, quota, **kwargs):
    cells = ds.plan_cells(quota, ds.AXIS_TARGETS, ds.impossible_cells())
    return ds.allocate(candidates, quota, cell_targets=cells, pool="test", **kwargs)


def test_allocate_skips_a_cell_the_plan_cannot_reach_and_still_fills_up():
    """A cell with no supply at all must not stall the row count."""
    missing = ("nonoverlapped", "in_range", "in_topic")
    selected, report = allocate_fixture(_full_supply(20, {missing: 0}), 48)
    assert len(selected) == 48
    assert report.selected == 48
    assert "/".join(missing) in report.infeasible_cells


def test_allocate_reports_the_cells_it_had_no_supply_for():
    dry = ("overlapped", "in_range", "out_topic")
    _selected, report = ds.allocate(
        _full_supply(20, {dry: 0}),
        50,
        cell_targets=ds.plan_cells(50, ds.AXIS_TARGETS, ds.impossible_cells()),
        pool="p",
    )
    assert "/".join(dry) in report.infeasible_cells
    assert "/".join(dry) not in report.cells


def test_allocate_keeps_the_axes_it_can_when_one_cell_runs_dry():
    """A dry cell costs the axis it belonged to, and the report says which.

    ``plan_cells`` matches the three marginals *exactly*, so the cells whose plan
    a dry cell's share could have spilled into are already at their budget and
    the second pass -- which refuses to take a row whose labels are not all still
    inside budget -- has nothing left to spend.  The third pass fills the row
    count from whatever is left, in cell order, and the axis that pays is the one
    the dry cell sat on.  It errs *below* its target, which is the direction the
    design can live with: the audit reports the achieved value, and a caption that
    came out at 0.23 in_topic is visible where one that quietly rebalanced itself
    to 0.45 by inventing rows would not be.

    This fixture is what a thin cell looks like from the allocator's side.  On the
    real pools at these quotas (SUM 2,000 rows, all 2,600 UMWP, 200 K&K) none of
    the six cells is dry and the third pass selects nothing at all.
    """
    quota = 200
    thin = ("nonoverlapped", "in_range", "in_topic")
    selected, report = allocate_fixture(_full_supply(60, {thin: 1}), quota)
    assert len(selected) == quota
    assert report.shortfall == 0
    # The two axes the dry cell's rows could not have carried are still on target.
    assert report.balanced["role_label"]
    assert report.balanced["number_label"]
    # The sentence axis is the one that pays, and it pays downwards.
    assert not report.balanced["sentence_label"]
    assert report.achieved["sentence_label"]["in_topic"] < ds.AXIS_TARGETS["sentence_label"]["in_topic"]


def test_allocate_third_pass_ignores_the_budgets_and_the_report_says_so():
    """A single cell's supply cannot fill the quota inside the axis budgets."""
    quota = 100
    cells = ds.plan_cells(quota, ds.AXIS_TARGETS, ds.impossible_cells())
    supply = {cell: 0 for cell in _feasible_cells()}
    supply[("overlapped", "out_range", "out_topic")] = quota
    selected, report = ds.allocate(_cell_supply(supply), quota, cell_targets=cells, pool="p")
    assert len(selected) == quota
    assert report.selected == quota
    assert not report.balanced["role_label"]
    assert not report.balanced["sentence_label"]


def test_allocate_reports_a_shortfall_when_there_is_not_enough_supply():
    selected, report = allocate_fixture(_cell_supply({("overlapped", "in_range", "out_topic"): 3}), 20)
    assert len(selected) == 3
    assert (report.selected, report.shortfall) == (3, 17)


def test_allocate_is_order_stable():
    candidates = _full_supply(10)
    first, _ = allocate_fixture(candidates, 40)
    second, _ = allocate_fixture(candidates, 40)
    assert [c.order for c in first] == [c.order for c in second]


def test_allocate_role_budget_is_proportional_to_what_each_pool_can_build():
    supply = {"sum": 10_000, "umwp": 1_000, "kk": 5_000}
    buildable = {"sum": 1000, "umwp": 600, "kk": 400}
    budget = ds.allocate_role_budget(supply, buildable, 0.50)
    assert budget == {"sum": 500, "umwp": 300, "kk": 200}
    assert sum(budget.values()) == 1000


def test_allocate_role_budget_gives_a_starved_pool_what_it_has():
    """An earlier rule gave one pool nothing whenever a bigger one had room.

    ``kk`` can only name an actor on 40 of its 400 buildable rows, so the global
    budget shrinks to the total room (920, not 1,000) and the split is by room:
    ``kk`` contributes its 40 -- a share of 0.10 against the design's 0.50, which
    is exactly the shortfall the report has to state rather than hide -- and the
    two pools that *can* reach the target still reach it (550/1,000 = 0.55, the
    out_topic cap).  Zeroing the small pool instead would have left 720 rows of
    room unused and dragged the global share to 0.10.
    """
    supply = {"sum": 10_000, "umwp": 10_000, "kk": 40}
    buildable = {"sum": 1000, "umwp": 600, "kk": 400}
    budget = ds.allocate_role_budget(supply, buildable, 0.50)
    assert budget == {"sum": 550, "umwp": 330, "kk": 40}
    assert all(value > 0 for value in budget.values())
    assert sum(budget.values()) < round(0.50 * sum(buildable.values()))


def test_allocate_role_budget_hands_out_its_own_total_not_the_room_total():
    """With the budget under the total room, every pool lands on the same share."""
    supply = {"sum": 10_000, "umwp": 10_000}
    buildable = {"sum": 1000, "umwp": 600}
    budget = ds.allocate_role_budget(supply, buildable, 0.20)
    assert budget == {"sum": 200, "umwp": 120}
    for pool, rows in budget.items():
        assert rows / buildable[pool] == pytest.approx(0.20)


def test_allocate_role_budget_is_capped_by_the_out_topic_room():
    """Overlapped implies out_topic, so a pool cannot exceed its out_topic budget."""
    supply = {"sum": 10_000}
    buildable = {"sum": 1000}
    budget = ds.allocate_role_budget(supply, buildable, 0.50)
    assert budget == {"sum": 500}
    # Raise the request past the 55% out_topic cap and the cap binds.
    assert ds.allocate_role_budget(supply, buildable, 0.90) == {"sum": 550}


def test_allocate_role_budget_is_capped_by_actor_supply():
    supply = {"sum": 30}
    buildable = {"sum": 1000}
    assert ds.allocate_role_budget(supply, buildable, 0.50) == {"sum": 30}


def test_allocate_role_budget_uses_buildable_not_the_requested_quota():
    """The regression: keying the fraction to the *requested* total pinned every
    pool to the out_topic cap (0.55) whenever one pool had no source, because the
    budget then asked for 50% of rows that would never be built."""
    supply = {"sum": 10_000, "umwp": 10_000, "kk": 5_000, "main": 0}
    buildable = {"sum": 1000, "umwp": 600, "kk": 400, "main": 0}
    budget = ds.allocate_role_budget(supply, buildable, 0.50)
    assert sum(budget.values()) == 1000  # 50% of the 2,000 buildable, not of 2,400
    assert budget["main"] == 0
    for pool, share in (("sum", 1000), ("umwp", 600), ("kk", 400)):
        assert budget[pool] == share // 2


# ---------------------------------------------------------------------------
# row emission
# ---------------------------------------------------------------------------


def _emit_one() -> tuple[ds.BaseQuestion, dict]:
    """One in_topic row: the cell that has to be asked for explicitly."""
    distractor = ds.choose_distractor(
        MAGAZINES,
        JEWEL,
        random.Random(0),
        names=["Emma"],
        numbers=["8"],
        want_role="nonoverlapped",
        want_sentence="in_topic",
    )
    assert distractor is not None
    assert distractor.cell == ("nonoverlapped", "in_range", "in_topic")
    candidate = ds.Candidate(
        base=JEWEL,
        distractor=distractor,
        question=ds.inject(JEWEL.question, distractor.sentence),
        order=0.0,
    )
    return JEWEL, ds.make_rows([candidate])[0]


def test_make_rows_carries_the_base_and_the_gold_unchanged():
    base, row = _emit_one()
    payload = json.loads(row["reward_model"]["ground_truth"])
    assert payload["answer"] == base.answer
    assert payload["solvable"] is True
    assert payload["perturbation_type"] == ds.DISTRACTOR_PERTURBATION
    assert row["extra_info"]["paired_original_text"] == base.question


def test_make_rows_prefixes_the_task_id_with_the_upstream_key():
    base, row = _emit_one()
    assert row["extra_info"]["task_id"] == f"d17:{base.uid}"


def test_make_rows_mirrors_perturbation_type_into_extra_info():
    """An empty column silently hides these rows from the D18 breakdown."""
    _base, row = _emit_one()
    assert row["extra_info"]["perturbation_type"] == ds.DISTRACTOR_PERTURBATION


def test_make_rows_records_the_three_labels():
    _base, row = _emit_one()
    labels = row["extra_info"]["distractor_labels"]
    assert set(labels) == set(ds.AXES)
    for axis in ds.AXES:
        assert labels[axis] in ds.AXIS_TARGETS[axis]


def test_make_rows_routes_each_pool_to_its_own_data_source_and_branch():
    for base, data_source, branch in (
        (JEWEL, schema.SOURCE_SUM, schema.BRANCH_SOLVABLE_NUMERIC),
        (BRYAN, schema.SOURCE_UMWP, schema.BRANCH_SOLVABLE_NUMERIC),
        (KK, schema.SOURCE_KK, schema.BRANCH_SOLVABLE_ROLES),
    ):
        distractor = ds.choose_distractor(
            HEIGHT, base, random.Random(1), names=["Zoe"], numbers=["8"]
        )
        assert distractor is not None
        row = ds.make_rows(
            [
                ds.Candidate(
                    base=base,
                    distractor=distractor,
                    question=ds.inject(base.question, distractor.sentence),
                    order=0.0,
                )
            ]
        )[0]
        assert row["data_source"] == data_source
        assert row["extra_info"]["branch"] == branch


def test_make_rows_satisfies_the_schema_contract():
    for base in (JEWEL, KK):
        distractor = ds.choose_distractor(
            HEIGHT, base, random.Random(2), names=["Zoe"], numbers=["8"]
        )
        assert distractor is not None
        row = ds.make_rows(
            [
                ds.Candidate(
                    base=base,
                    distractor=distractor,
                    question=ds.inject(base.question, distractor.sentence),
                    order=0.0,
                )
            ]
        )[0]
        assert schema.validate_row(row) == []


def test_verify_row_accepts_the_row_it_emitted():
    base, row = _emit_one()
    assert ds.verify_row(base, row) == []


def test_verify_row_catches_a_gold_that_moved():
    """A distractor that changed the answer is the one thing D17 promises it does not."""
    base, row = _emit_one()
    payload = json.loads(row["reward_model"]["ground_truth"])
    payload["answer"] = str(payload["answer"]) + "9"
    row["reward_model"]["ground_truth"] = json.dumps(payload)
    assert "gold answer changed by synthesis" in ds.verify_row(base, row)


def test_verify_row_catches_a_row_marked_unsolvable():
    base, row = _emit_one()
    payload = json.loads(row["reward_model"]["ground_truth"])
    payload["solvable"] = False
    row["reward_model"]["ground_truth"] = json.dumps(payload)
    assert "synthesised row is not marked solvable" in ds.verify_row(base, row)


def test_verify_row_catches_an_unexpected_perturbation_type():
    base, row = _emit_one()
    payload = json.loads(row["reward_model"]["ground_truth"])
    payload["perturbation_type"] = "missing_condition"
    row["reward_model"]["ground_truth"] = json.dumps(payload)
    problems = ds.verify_row(base, row)
    assert any("perturbation_type" in problem for problem in problems)


def test_verify_row_catches_a_base_that_is_no_longer_recoverable():
    """The base is deliberately *not* a span of the prompt: inject splits it.

    The check is removability, not containment -- an earlier revision asserted the
    base was a contiguous span and failed 1,395 of 2,000 rows for a reason that
    was never a defect.  The edit below has to leave the distractor sentence alone
    (it replaces a word only the base carries), or the earlier check fails first
    and the removability branch is never reached -- which is how this test passed
    for the wrong reason once already.
    """
    base, row = _emit_one()
    assert "bought" in base.question
    assert "bought" not in row["extra_info"]["distractor_text"]
    row["prompt"][0]["content"] = row["prompt"][0]["content"].replace("bought", "obtained")
    problems = ds.verify_row(base, row)
    assert "base question text is not recoverable once the distractor is removed" in problems
    assert problems == ["base question text is not recoverable once the distractor is removed"]


def test_verify_row_catches_a_prompt_missing_the_distractor():
    base, row = _emit_one()
    row["extra_info"]["distractor_text"] = "A sentence that is not in the prompt."
    assert "distractor sentence is not in the rendered prompt" in ds.verify_row(base, row)


def test_verify_row_catches_a_missing_distractor():
    base, row = _emit_one()
    row["extra_info"]["distractor_text"] = ""
    assert "row has no distractor_text" in ds.verify_row(base, row)


def test_verify_row_catches_a_label_outside_the_axis_vocabulary():
    base, row = _emit_one()
    row["extra_info"]["distractor_labels"]["role_label"] = "sideways"
    assert "role_label is 'sideways'" in ds.verify_row(base, row)


def test_verify_row_accepts_a_prompt_with_extra_text_appended():
    """K&K role rows append the answer-format instruction after the puzzle."""
    base, row = _emit_one()
    row["prompt"][0]["content"] += schema._KK_ROLE_INSTRUCTION
    assert ds.verify_row(base, row) == []


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------


def _pools() -> dict[str, list[ds.BaseQuestion]]:
    """Fresh copies of the module-level bases, minus the main pool's source.

    The copy is not decoration: :func:`ds.mine_actors` writes ``meta["names"]``
    in place, so sharing one ``JEWEL`` across the whole file means the first test
    that runs with ``mine=True`` silently empties the role supply of every test
    after it -- which is precisely how two of these tests came to fail.
    """
    return {
        ds.POOL_SUM: [_copy(JEWEL), _copy(EQUATION)],
        ds.POOL_UMWP: [_copy(BRYAN)],
        ds.POOL_KK: [_copy(KK)],
        ds.POOL_MAIN: [],
    }


# Quotas for the real-pool fixture: enough rows for all three axes to be
# meaningful, few enough that the whole file still runs in seconds.
SMALL_QUOTA = {ds.POOL_SUM: 40, ds.POOL_UMWP: 20, ds.POOL_KK: 10, ds.POOL_MAIN: 0}


@functools.lru_cache(maxsize=None)
def _loaded_pools() -> tuple[dict[str, list[ds.BaseQuestion]], dict[str, str]]:
    """The real pools, read once for the whole file (0.6s) and then copied."""
    return ds.load_pools(ds.DEFAULT_RAW_DIR)


@functools.lru_cache(maxsize=None)
def _safe_templates() -> list[ds.Template]:
    return ds.load_safe_templates(ds.DEFAULT_RAW_DIR)[0]


@pytest.fixture()
def real_pools() -> dict[str, list[ds.BaseQuestion]]:
    """A slice of the real pools, with ``meta`` unshared and nothing mined yet.

    The slice keeps the file fast.  It does not change the axes: all 2,600 UMWP
    rows and 200 of the 5,000 K&K puzzles are here, and those are the two pools
    that feed the actor vocabulary, so the mining behaves as it does in a build.
    """
    pools, _notes = _loaded_pools()
    return {
        ds.POOL_SUM: [_copy(base) for base in pools[ds.POOL_SUM][:2000]],
        ds.POOL_UMWP: [_copy(base) for base in pools[ds.POOL_UMWP]],
        ds.POOL_KK: [_copy(base) for base in pools[ds.POOL_KK][:200]],
        ds.POOL_MAIN: [],
    }


def test_synthesise_balances_all_three_axes_on_real_pools(real_pools):
    """The design's three axes, held at once, on a 70-row slice of real data.

    This is the assertion the whole stage exists for, and it is only meaningful
    with real supply behind every cell: on a hand-built pool the ``in_topic``
    cells are empty and the axis comes out at 0.23 -- see
    :func:`test_allocate_keeps_the_axes_it_can_when_one_cell_runs_dry` for what
    that looks like from the allocator's side.
    """
    rows, report = ds.synthesise(real_pools, templates=_safe_templates(), quota=SMALL_QUOTA, seed=7)
    assert report["requested_total"] == 70
    assert report["built_total"] == 70
    assert len(rows) == 70
    assert all(report["balanced_overall"].values()), report["achieved_overall"]
    for axis in ds.AXES:
        for label, fraction in ds.AXIS_TARGETS[axis].items():
            assert abs(report["achieved_overall"][axis][label] - fraction) <= ds.AXIS_TOLERANCE


def test_synthesise_emits_rows_the_schema_and_the_self_check_accept(real_pools):
    rows, _report = ds.synthesise(
        real_pools, templates=_safe_templates(), quota=SMALL_QUOTA, seed=3
    )
    assert len(rows) == 70
    by_uid = {base.uid: base for pool in real_pools.values() for base in pool}
    for row in rows:
        base = by_uid[row["extra_info"]["task_id"][len("d17:") :]]
        assert schema.validate_row(row) == []
        assert ds.verify_row(base, row) == []


def test_synthesise_uses_each_base_at_most_once(real_pools):
    rows, _report = ds.synthesise(
        real_pools, templates=_safe_templates(), quota=SMALL_QUOTA, seed=5
    )
    ids = [row["extra_info"]["task_id"] for row in rows]
    assert len(ids) == len(set(ids))


def test_synthesise_is_deterministic_for_a_seed():
    templates, _report = ds.load_safe_templates()
    first, _report = ds.synthesise(_pools(), templates=templates, seed=11, mine=False)
    second, _report = ds.synthesise(_pools(), templates=templates, seed=11, mine=False)
    assert [row["extra_info"]["distractor_text"] for row in first] == [
        row["extra_info"]["distractor_text"] for row in second
    ]


def test_synthesise_reports_an_empty_pool_rather_than_skipping_it():
    templates, _report = ds.load_safe_templates()
    _rows, report = ds.synthesise(_pools(), templates=templates, seed=1, mine=False)
    assert report["pools"][ds.POOL_MAIN]["selected"] == 0
    assert report["pools"][ds.POOL_MAIN]["shortfall"] == ds.DEFAULT_POOL_QUOTA[ds.POOL_MAIN]


def test_synthesise_reports_the_role_supply_it_could_not_meet():
    """The design's per-pool 50/50 cannot be built: most MATH problems name nobody."""
    templates, _report = ds.load_safe_templates()
    pools = {ds.POOL_SUM: [EQUATION], ds.POOL_UMWP: [], ds.POOL_KK: [], ds.POOL_MAIN: []}
    _rows, report = ds.synthesise(
        pools, templates=templates, quota={ds.POOL_SUM: 4}, seed=1, mine=False
    )
    assert report["effective_targets"][ds.POOL_SUM]["overlapped_supply"] == 0
    assert report["effective_targets"][ds.POOL_SUM]["overlapped_budget"] == 0
    assert report["role_supply_note"].startswith("overlapped role supply 0 of 1 buildable")


def test_synthesise_reports_the_actor_mining_it_ran(real_pools):
    """Mining is what fills ``meta["names"]``, and the report says how much of it."""
    rows, report = ds.synthesise(
        real_pools, templates=_safe_templates(), quota=SMALL_QUOTA, seed=1
    )
    assert rows
    mining = report["actor_mining"]["pools"]
    # K&K's inhabitants are given, so that pool has to come out complete; the two
    # mined pools do not, and the shortfall is the role axis' design refusal.
    assert mining[ds.POOL_KK]["bases_with_actor"] == len(real_pools[ds.POOL_KK])
    assert 0.0 < mining[ds.POOL_UMWP]["coverage"] < 1.0
    assert 0.0 < mining[ds.POOL_SUM]["coverage"] < mining[ds.POOL_UMWP]["coverage"]
    for pool in (ds.POOL_SUM, ds.POOL_UMWP, ds.POOL_KK):
        assert (
            report["effective_targets"][pool]["overlapped_supply"]
            == mining[pool]["bases_with_actor"]
        )
    # Supply counts bases, the budget counts rows: with ample supply each pool's
    # share is the design's half of what it can build.
    for pool in (ds.POOL_SUM, ds.POOL_UMWP, ds.POOL_KK):
        effective = report["effective_targets"][pool]
        assert effective["overlapped_budget"] == pytest.approx(effective["buildable"] * 0.50)
    assert "overlapped role supply" in report["role_supply_note"]


def test_synthesise_can_skip_mining_and_then_has_no_overlapped_supply(real_pools):
    """``mine=False`` on unmined pools is the fail-closed path, not a silent one."""
    _rows, report = ds.synthesise(
        real_pools, templates=_safe_templates(), quota=SMALL_QUOTA, seed=1, mine=False
    )
    assert report["actor_mining"] is None
    # SUM and UMWP carry no names until mining fills them in, so they lose their
    # overlapped cells rather than falling back to a capitalised verb.
    assert report["effective_targets"][ds.POOL_SUM]["overlapped_supply"] == 0
    assert report["effective_targets"][ds.POOL_UMWP]["overlapped_supply"] == 0
    # K&K is the exception in both directions: its names come from the puzzle.
    assert report["effective_targets"][ds.POOL_KK]["overlapped_supply"] > 0
    assert report["pools"][ds.POOL_SUM]["requested"] == SMALL_QUOTA[ds.POOL_SUM]


def test_synthesise_records_the_design_and_the_effective_targets_separately(real_pools):
    """The design's 50/50 is a target, not a promise: the report carries both."""
    _rows, report = ds.synthesise(
        real_pools, templates=_safe_templates(), quota=SMALL_QUOTA, seed=1, mine=False
    )
    assert report["design_targets"]["role_label"] == ds.AXIS_TARGETS["role_label"]
    # Unmined SUM can name nobody, so its *effective* role target is the one the
    # data allows -- and it is not the design's.
    effective = report["effective_targets"][ds.POOL_SUM]["targets"]["role_label"]
    assert effective == {"overlapped": 0.0, "nonoverlapped": 1.0}
    assert effective != report["design_targets"]["role_label"]
    # K&K can meet it, so its effective target is the design's.
    kk_role = report["effective_targets"][ds.POOL_KK]["targets"]["role_label"]
    assert kk_role == pytest.approx(ds.AXIS_TARGETS["role_label"])
    assert sum(kk_role.values()) == pytest.approx(1.0)
    for pool in ds.POOLS:
        for axis in ds.AXES:
            assert sum(report["effective_targets"][pool]["targets"][axis].values()) == (
                pytest.approx(1.0)
            )


def test_synthesise_reports_the_templates_it_drew_from():
    templates, _report = ds.load_safe_templates()
    _rows, report = ds.synthesise(
        _pools(), templates=templates, quota={ds.POOL_SUM: 4}, seed=1, mine=False
    )
    assert report["requested_total"] == 4
    assert report["seed"] == 1
    assert report["tolerance"] == ds.AXIS_TOLERANCE


def test_synthesise_drops_non_numeric_fills_and_records_them():
    bad = _template("The {role} of the machine is {number} units.", number="n/a")
    safe, _report = ds.load_safe_templates()
    rows, report = ds.synthesise(
        {ds.POOL_SUM: [JEWEL], ds.POOL_UMWP: [], ds.POOL_KK: [], ds.POOL_MAIN: []},
        templates=[bad, *safe],
        quota={ds.POOL_SUM: 2},
        seed=1,
        mine=False,
    )
    assert report["non_numeric_fills_dropped"] == ["n/a"]
    assert all("n/a" not in row["prompt"][0]["content"] for row in rows)
