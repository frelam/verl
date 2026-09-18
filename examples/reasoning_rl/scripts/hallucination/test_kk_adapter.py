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
"""Tests for ``kk_adapter.py`` (and the audit script's checks on its output).

The frozen fixtures below are real K&K rows.  Their text, statements, solution and
``knight_knave`` pair are copied verbatim; ``knight_knave`` is trimmed to the four
keys the adapter reads (the raw rows also carry the puzzle narration) and the
``cot_*`` columns of the raw files are not carried over, since nothing under test
consumes them.  Six of them are the
seven wording variants of the *same* abstract problem -- group ``(4, 100)`` -- which
is what makes the group-dedup and role-word tests exercise the real thing: the
``flip_role`` and ``random_pair`` members carry byte-identical ``statements`` and
``solution`` to the clean member while their prompts spell the two roles
differently, so a canonical-word gold inverts on them.

Small fixtures are written to ``tmp_path`` and never to the downloaded data
directory, so most of the suite is self-contained.  Three tests touch the real raw
directory once (module-scoped) and assert the measured funnel and audit result, so
a change to the corpus or to the selection policy fails loudly instead of silently
shrinking the pool.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import schema
import verify_kk
import kk_adapter as adapter

RAW_DIR = Path("/home/charles/data/reasoning_rl/halluc/raw/kk")

# ---------------------------------------------------------------------------
# fixtures -- real K&K rows, copied from the downloaded parquet files (see the
# module docstring for the two deliberate omissions)
# ---------------------------------------------------------------------------

CLEAN_4PPL = {
    'quiz': 'A very special island is inhabited only by knights and knaves. Knights always tell the truth, and knaves always lie. You meet 4 inhabitants: Ethan, Abigail, David, and Noah. Ethan asserted: "David is a knave or Abigail is a knave". Abigail was heard saying, "If Abigail is a knight then David is a knight". "Noah is a knight and Ethan is a knave" - David. Noah remarked, "Abigail is a knave if and only if Ethan is a knight". So who is a knight and who is a knave?',
    'names': ['Ethan', 'Abigail', 'David', 'Noah'],
    'knight_knave': {'knight': 'knight', 'knave': 'knave', 'a_knight': 'a knight', 'a_knave': 'a knave'},
    'solution': [False, True, True, True],
    'solution_text': 'Ethan is a knave, Abigail is a knight, David is a knight, and Noah is a knight.',
    'solution_text_format': '(1) Ethan is a knave\n(2) Abigail is a knight\n(3) David is a knight\n(4) Noah is a knight',
    'statements': "(('or', ('lying', 2), ('lying', 1)), ('->', ('telling-truth', 1), ('telling-truth', 2)), ('and', ('telling-truth', 3), ('lying', 0)), ('<=>', ('lying', 1), ('telling-truth', 0)))",
    'index': 100,
}

FLIP_ROLE = {
    'quiz': 'A very special island is inhabited only by knaves and knights. Knaves always tell the truth, and knights always lie. You meet 4 inhabitants: Ethan, Abigail, David, and Noah. Ethan asserted: "David is a knight or Abigail is a knight". Abigail was heard saying, "If Abigail is a knave then David is a knave". "Noah is a knave and Ethan is a knight" - David. Noah remarked, "Abigail is a knight if and only if Ethan is a knave". So who is a knave and who is a knight?',
    'names': ['Ethan', 'Abigail', 'David', 'Noah'],
    'knight_knave': {'knight': 'knave', 'knave': 'knight', 'a_knight': 'a knave', 'a_knave': 'a knight'},
    'solution': [False, True, True, True],
    'solution_text': 'Ethan is a knight, Abigail is a knave, David is a knave, and Noah is a knave.',
    'solution_text_format': '(1) Ethan is a knight\n(2) Abigail is a knave\n(3) David is a knave\n(4) Noah is a knave',
    'statements': "(('or', ('lying', 2), ('lying', 1)), ('->', ('telling-truth', 1), ('telling-truth', 2)), ('and', ('telling-truth', 3), ('lying', 0)), ('<=>', ('lying', 1), ('telling-truth', 0)))",
    'index': 100,
}

RANDOM_PAIR = {
    'quiz': 'A very special island is inhabited only by altruists and egoists. Altruists always tell the truth, and egoists always lie. You meet 4 inhabitants: Ethan, Abigail, David, and Noah. Ethan was heard saying, "David is an egoist or Abigail is an egoist". "If Abigail is an altruist then David is an altruist" - Abigail. David remarked, "Noah is an altruist and Ethan is an egoist". According to Noah, "Abigail is an egoist if and only if Ethan is an altruist". So who is an altruist and who is an egoist?',
    'names': ['Ethan', 'Abigail', 'David', 'Noah'],
    'knight_knave': {'knight': 'altruist', 'knave': 'egoist', 'a_knight': 'an altruist', 'a_knave': 'an egoist'},
    'solution': [False, True, True, True],
    'solution_text': 'Ethan is an egoist, Abigail is an altruist, David is an altruist, and Noah is an altruist.',
    'solution_text_format': '(1) Ethan is an egoist\n(2) Abigail is an altruist\n(3) David is an altruist\n(4) Noah is an altruist',
    'statements': "(('or', ('lying', 2), ('lying', 1)), ('->', ('telling-truth', 1), ('telling-truth', 2)), ('and', ('telling-truth', 3), ('lying', 0)), ('<=>', ('lying', 1), ('telling-truth', 0)))",
    'index': 100,
}

PERTURBED_STATEMENT = {
    'quiz': 'A very special island is inhabited only by knights and knaves. Knights always tell the truth, and knaves always lie. You meet 4 inhabitants: Ethan, Abigail, David, and Noah. Ethan asserted: "David is a knave or Abigail is a knave". Abigail was heard saying, "David is a knight and David is a knave". "Noah is a knight and Ethan is a knave" - David. Noah remarked, "Abigail is a knave if and only if Ethan is a knight". So who is a knight and who is a knave?',
    'names': ['Ethan', 'Abigail', 'David', 'Noah'],
    'knight_knave': {'knight': 'knight', 'knave': 'knave', 'a_knight': 'a knight', 'a_knave': 'a knave'},
    'solution': [True, False, False, True],
    'solution_text': 'Ethan is a knight, Abigail is a knave, David is a knave, and Noah is a knight.',
    'solution_text_format': '(1) Ethan is a knight\n(2) Abigail is a knave\n(3) David is a knave\n(4) Noah is a knight',
    'statements': "(('or', ('lying', 2), ('lying', 1)), ('and', ('telling-truth', 2), ('lying', 2)), ('and', ('telling-truth', 3), ('lying', 0)), ('<=>', ('lying', 1), ('telling-truth', 0)))",
    'index': 100,
}

REORDER_STATEMENT = {
    'quiz': 'A very special island is inhabited only by knights and knaves. Knights always tell the truth, and knaves always lie. You meet 4 inhabitants: Ethan, Abigail, David, and Noah. Abigail was heard saying, "If Abigail is a knight then David is a knight". Ethan asserted: "David is a knave or Abigail is a knave". "Noah is a knight and Ethan is a knave" - David. Noah remarked, "Abigail is a knave if and only if Ethan is a knight". So who is a knight and who is a knave?',
    'names': ['Ethan', 'Abigail', 'David', 'Noah'],
    'knight_knave': {'knight': 'knight', 'knave': 'knave', 'a_knight': 'a knight', 'a_knave': 'a knave'},
    'solution': [False, True, True, True],
    'solution_text': 'Ethan is a knave, Abigail is a knight, David is a knight, and Noah is a knight.',
    'solution_text_format': '(1) Ethan is a knave\n(2) Abigail is a knight\n(3) David is a knight\n(4) Noah is a knight',
    'statements': "(('or', ('lying', 2), ('lying', 1)), ('->', ('telling-truth', 1), ('telling-truth', 2)), ('and', ('telling-truth', 3), ('lying', 0)), ('<=>', ('lying', 1), ('telling-truth', 0)))",
    'index': 100,
}

UNCOMMON_NAME = {
    'quiz': 'A very special island is inhabited only by knights and knaves. Knights always tell the truth, and knaves always lie. You meet 4 inhabitants: Vesper, Elodie, Thorsten, and Isolde. Vesper asserted: "Thorsten is a knave or Elodie is a knave". Elodie was heard saying, "If Elodie is a knight then Thorsten is a knight". "Isolde is a knight and Vesper is a knave" - Thorsten. Isolde remarked, "Elodie is a knave if and only if Vesper is a knight". So who is a knight and who is a knave?',
    'names': ['Vesper', 'Elodie', 'Thorsten', 'Isolde'],
    'knight_knave': {'knight': 'knight', 'knave': 'knave', 'a_knight': 'a knight', 'a_knave': 'a knave'},
    'solution': [False, True, True, True],
    'solution_text': 'Vesper is a knave, Elodie is a knight, Thorsten is a knight, and Isolde is a knight.',
    'solution_text_format': '(1) Vesper is a knave\n(2) Elodie is a knight\n(3) Thorsten is a knight\n(4) Isolde is a knight',
    'statements': "(('or', ('lying', 2), ('lying', 1)), ('->', ('telling-truth', 1), ('telling-truth', 2)), ('and', ('telling-truth', 3), ('lying', 0)), ('<=>', ('lying', 1), ('telling-truth', 0)))",
    'index': 100,
}

CLEAN_3PPL = {
    'quiz': 'A very special island is inhabited only by knights and knaves. Knights always tell the truth, and knaves always lie. You meet 3 inhabitants: Michael, Zoey, and Ethan. Michael was heard saying, "Ethan is a knight if and only if Michael is a knight". "Zoey is a knight or Ethan is a knight," Zoey mentioned. Ethan asserted: "Michael is a knave if and only if Zoey is a knave". So who is a knight and who is a knave?',
    'names': ['Michael', 'Zoey', 'Ethan'],
    'knight_knave': {'knight': 'knight', 'knave': 'knave', 'a_knight': 'a knight', 'a_knave': 'a knave'},
    'solution': [True, True, True],
    'solution_text': 'Michael is a knight, Zoey is a knight, and Ethan is a knight.',
    'solution_text_format': '(1) Michael is a knight\n(2) Zoey is a knight\n(3) Ethan is a knight',
    'statements': "(('<=>', ('telling-truth', 2), ('telling-truth', 0)), ('or', ('telling-truth', 1), ('telling-truth', 2)), ('<=>', ('lying', 0), ('lying', 1)))",
    'index': 100,
}

CLEAN_5PPL = {
    'quiz': 'A very special island is inhabited only by knights and knaves. Knights always tell the truth, and knaves always lie. You meet 5 inhabitants: Ethan, Abigail, David, Noah, and Zara. Ethan asserted: "David is a knave or Abigail is a knave". Abigail was heard saying, "If Abigail is a knight then David is a knight". "Noah is a knight and Ethan is a knave" - David. Noah remarked, "Abigail is a knave if and only if Ethan is a knight". "Ethan is a knight" - Zara. So who is a knight and who is a knave?',
    'names': ['Ethan', 'Abigail', 'David', 'Noah', 'Zara'],
    'knight_knave': {'knight': 'knight', 'knave': 'knave', 'a_knight': 'a knight', 'a_knave': 'a knave'},
    'solution': [False, True, True, True, False],
    'solution_text': 'Ethan is a knave, Abigail is a knight, David is a knight, Noah is a knight, and Zara is a knave.',
    'solution_text_format': '(1) Ethan is a knave\n(2) Abigail is a knight\n(3) David is a knight\n(4) Noah is a knight\n(5) Zara is a knave',
    'statements': "(('or', ('lying', 2), ('lying', 1)), ('->', ('telling-truth', 1), ('telling-truth', 2)), ('and', ('telling-truth', 3), ('lying', 0)), ('<=>', ('lying', 1), ('telling-truth', 0)), ('telling-truth', 0))",
    'index': 100,
}

CLEAN_TEST_4PPL = {
    'quiz': 'A very special island is inhabited only by knights and knaves. Knights always tell the truth, and knaves always lie. You meet 4 inhabitants: Aiden, Jacob, Chloe, and Penelope. Aiden remarked, "Jacob is a knight if and only if Penelope is a knave". According to Jacob, "Chloe is a knight if and only if Jacob is a knight". "Jacob is not a knight" - Chloe. Penelope said that Jacob is a knave and Jacob is a knight. So who is a knight and who is a knave?',
    'names': ['Aiden', 'Jacob', 'Chloe', 'Penelope'],
    'knight_knave': {'knight': 'knight', 'knave': 'knave', 'a_knight': 'a knight', 'a_knave': 'a knave'},
    'solution': [False, False, True, False],
    'solution_text': 'Aiden is a knave, Jacob is a knave, Chloe is a knight, and Penelope is a knave.',
    'solution_text_format': '(1) Aiden is a knave\n(2) Jacob is a knave\n(3) Chloe is a knight\n(4) Penelope is a knave',
    'statements': "(('<=>', ('telling-truth', 1), ('lying', 3)), ('<=>', ('telling-truth', 2), ('telling-truth', 1)), ('not', ('telling-truth', 1)), ('and', ('lying', 1), ('telling-truth', 1)))",
    'index': 7,
}

#: The seven wording variants of group ``(4, 100)`` plus the two rows the policy
#: drops (the 3-inhabitant tier and the source test split).
GROUP_FILES: dict[str, list[dict]] = {
    "clean__train__4ppl.parquet": [CLEAN_4PPL],
    "clean__train__3ppl.parquet": [CLEAN_3PPL],
    "clean__test__4ppl.parquet": [CLEAN_TEST_4PPL],
    "perturbed__train__flip_role.parquet": [FLIP_ROLE],
    "perturbed__train__random_pair.parquet": [RANDOM_PAIR],
    "perturbed__train__perturbed_statement.parquet": [PERTURBED_STATEMENT],
    "perturbed__train__reorder_statement.parquet": [REORDER_STATEMENT],
    "perturbed__train__uncommon_name.parquet": [UNCOMMON_NAME],
}

FUNNEL_STAGES = [
    "raw_rows",
    "after_filename_drop",
    "after_malformed_drop",
    "after_enumerator_drop",
    "after_n_ge_4_drop",
    "after_source_test_drop",
    "after_group_dedup",
    "after_limit",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_raw(raw_dir: Path, files: dict[str, list[dict]]) -> str:
    """Write ``{filename: [row, ...]}`` as the source's own parquet layout."""
    import datasets

    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in files.items():
        datasets.Dataset.from_list(rows).to_parquet(raw_dir / name)
    return str(raw_dir)


def _build(tmp_path: Path, files: dict[str, list[dict]] | None = None, **kwargs):
    """Build from a fixture directory and finalise the rows like ``main`` does."""
    raw_dir = _write_raw(tmp_path, GROUP_FILES if files is None else files)
    rows, report = adapter.build_rows(raw_dir, **kwargs)
    schema.normalise_extra_info(rows)
    return rows, report


def _clone(fixture: dict, index: int) -> dict:
    """A copy of ``fixture`` with a different source ``index`` (a new group)."""
    return {**fixture, "index": index}


def _by_task_id(rows: list[dict]) -> dict[str, dict]:
    return {row["extra_info"]["task_id"]: row for row in rows}


# ---------------------------------------------------------------------------
# the enumerator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", [CLEAN_4PPL, FLIP_ROLE, RANDOM_PAIR, PERTURBED_STATEMENT, CLEAN_TEST_4PPL])
def test_enumerator_recovers_the_rows_own_solution(fixture):
    statements = adapter.parse_statements(fixture["statements"], len(fixture["names"]))
    solutions = adapter.enumerate_solutions(statements, len(fixture["names"]))
    assert solutions == [tuple(fixture["solution"])]


def test_enumerate_solutions_treats_a_second_solution_as_not_unique():
    # Two people each asserting that the other tells the truth: two solutions.
    statements = (("telling-truth", 1), ("telling-truth", 0))
    assert len(adapter.enumerate_solutions(statements, 2)) == 2
    assert adapter.unique_solution(statements, 2) is None


def test_enumerate_solutions_can_be_empty():
    # "I am a liar": true(truth[i]) and false(truth[i]) both contradict the field.
    assert adapter.enumerate_solutions((("lying", 0),), 1) == []
    assert adapter.unique_solution((("lying", 0),), 1) is None


@pytest.mark.parametrize(
    "node,expected",
    [
        (("telling-truth", 0), True),
        (("lying", 0), False),
        (("not", ("lying", 0)), True),
        (("and", ("telling-truth", 0), ("lying", 0)), False),
        (("or", ("lying", 0), ("telling-truth", 0)), True),
    ],
)
def test_evaluate_matches_the_operator_spelling(node, expected):
    assert adapter.evaluate(node, (True,)) is expected


def test_evaluate_covers_every_operator():
    truth = (True, False)
    assert adapter.evaluate(("and", ("telling-truth", 0), ("lying", 1)), truth) is True
    assert adapter.evaluate(("and", ("lying", 0), ("telling-truth", 0)), truth) is False
    assert adapter.evaluate(("or", ("lying", 0), ("lying", 1)), truth) is True
    assert adapter.evaluate(("->", ("telling-truth", 0), ("telling-truth", 1)), truth) is False
    assert adapter.evaluate(("->", ("lying", 0), ("telling-truth", 1)), truth) is True
    assert adapter.evaluate(("<=>", ("telling-truth", 0), ("lying", 1)), truth) is True
    assert adapter.evaluate(("<=>", ("telling-truth", 1), ("lying", 1)), truth) is False
    assert adapter.evaluate(("not", ("telling-truth", 0)), truth) is False


def test_evaluate_fails_closed_on_an_unknown_operator():
    with pytest.raises(adapter.StatementError):
        adapter.evaluate(("xor", ("telling-truth", 0), ("lying", 0)), (True,))


@pytest.mark.parametrize(
    "node",
    [
        ("telling-truth", 5),  # index outside the roster
        ("xor", ("lying", 0), ("lying", 1)),  # not an operator
        ("and", ("lying", 0), ("lying", 1), ("lying", 0)),  # wrong arity
        ("not",),  # missing operand
        ("lying",),  # missing index
        "lying",  # not a tuple
    ],
)
def test_check_node_rejects_every_shape_error(node):
    with pytest.raises(adapter.StatementError):
        adapter._check_node(node, 2)


@pytest.mark.parametrize(
    "text,count",
    [
        ("not-a-literal", 4),
        ("{'or': 1}", 4),
        ("()", 4),
        ("(('or', ('lying', 2), ('lying', 1)),)", 4),  # one formula, four people
        ("(('xor', ('lying', 0), ('lying', 1)),)", 2),
        ("(('lying', 9),)", 2),  # index outside the roster
        ("(('lying', True),)", 2),
        ("(('and', ('lying', 0)),)", 2),  # wrong arity
        ("(('not', ('lying', 0), ('lying', 1)),)", 2),
        ("(('not',))", 2),
    ],
)
def test_parse_statements_fails_closed(text, count):
    with pytest.raises(adapter.StatementError):
        adapter.parse_statements(text, count)


def test_parse_statements_rejects_an_absurdly_deep_tree():
    node = ("telling-truth", 0)
    for _ in range(adapter.MAX_STATEMENT_DEPTH + 2):
        node = ("not", node)
    with pytest.raises(adapter.StatementError):
        adapter.parse_statements(repr((node,)), 1)


# ---------------------------------------------------------------------------
# the gold is the surface sequence
# ---------------------------------------------------------------------------


def test_gold_uses_the_rows_own_role_words(tmp_path):
    """``flip_role`` calls the truth-teller "knave"; the gold must follow it.

    The row's own ``solution_text`` says Ethan (a liar) *is a knight*, so the
    surface answer is the mirror of the canonical ``knave knight knight knight``.
    """
    files = {
        "clean__train__4ppl.parquet": [CLEAN_4PPL, _clone(CLEAN_4PPL, 101)],
        "perturbed__train__flip_role.parquet": [_clone(FLIP_ROLE, 100), _clone(FLIP_ROLE, 101)],
    }
    rows, _ = _build(tmp_path, files)
    row = _by_task_id(rows)[adapter.task_id_of("flip_role", 4, 101)]
    assert json.loads(row["reward_model"]["ground_truth"])["answer"] == "knight knave knave knave"
    assert row["extra_info"]["role_words"] == ["knave", "knight"]
    assert row["extra_info"]["canonical_solution"] == [False, True, True, True]
    assert adapter.canonical_gold(["Ethan"] * 4, [False, True, True, True]) == "knave knight knight knight"


def test_gold_uses_a_random_pair_verbatim(tmp_path):
    # Two groups: the first rank takes clean, the second takes the perturbed side.
    files = {
        "clean__train__4ppl.parquet": [CLEAN_4PPL, _clone(CLEAN_4PPL, 101)],
        "perturbed__train__random_pair.parquet": [_clone(RANDOM_PAIR, 100), _clone(RANDOM_PAIR, 101)],
    }
    rows, report = _build(tmp_path, files)
    row = _by_task_id(rows)[adapter.task_id_of("random_pair", 4, 101)]
    assert json.loads(row["reward_model"]["ground_truth"])["answer"] == "egoist altruist altruist altruist"
    assert row["extra_info"]["role_words"] == ["altruist", "egoist"]
    assert report["selected"]["non_canonical_role_words"] == 1


def test_surface_gold_maps_the_truth_word_to_a_true_flag():
    assert adapter.surface_gold([True], ["knight", "knave"]) == "knight"
    assert adapter.surface_gold([False], ["knight", "knave"]) == "knave"
    # A row that spells the truth-teller "knave" must respell the same booleans.
    assert adapter.surface_gold([False, True], ["knave", "knight"]) == "knight knave"
    assert adapter.surface_gold([True, False], ["altruist", "egoist"]) == "altruist egoist"


def test_canonical_gold_is_the_audit_only_code():
    assert adapter.canonical_gold(["Ethan"], [False]) == "knave"
    assert adapter.canonical_gold(["Ethan", "Abigail"], [False, True]) == "knave knight"


def test_ground_truth_carries_role_words_and_no_perturbation_type(tmp_path):
    rows, _ = _build(tmp_path)
    for row in rows:
        payload = json.loads(row["reward_model"]["ground_truth"])
        assert payload["solvable"] is True
        assert payload["correct_option_id"] is None
        assert payload["has_diagnosis_label"] is False
        assert payload["perturbation_type"] is None
        assert payload["role_words"] == row["extra_info"]["role_words"]
        assert len(payload["role_words"]) == 2


def test_prompt_asks_for_the_rows_own_words(tmp_path):
    files = {
        "clean__train__4ppl.parquet": [CLEAN_4PPL, _clone(CLEAN_4PPL, 101)],
        "perturbed__train__random_pair.parquet": [_clone(RANDOM_PAIR, 100), _clone(RANDOM_PAIR, 101)],
    }
    rows, _ = _build(tmp_path, files)
    row = _by_task_id(rows)[adapter.task_id_of("random_pair", 4, 101)]
    prompt = row["prompt"][0]["content"]
    assert "altruist" in prompt and "egoist" in prompt
    assert prompt.startswith(adapter.normalise_text(RANDOM_PAIR["quiz"]))
    # The role instruction is the only place the canonical pair may still appear,
    # and only as an example -- the question itself is the row's own.
    assert prompt.count("altruist") == RANDOM_PAIR["quiz"].count("altruist")


# ---------------------------------------------------------------------------
# the funnel
# ---------------------------------------------------------------------------


def test_funnel_stage_counts(tmp_path):
    rows, report = _build(tmp_path)
    assert list(report["funnel"]) == FUNNEL_STAGES
    assert report["funnel"] == {
        "raw_rows": 8,
        "after_filename_drop": 8,
        "after_malformed_drop": 8,
        "after_enumerator_drop": 8,
        "after_n_ge_4_drop": 7,  # the 3-inhabitant tier
        "after_source_test_drop": 6,  # the source test split
        "after_group_dedup": 1,  # six wording variants of one abstract problem
        "after_limit": 1,
    }
    assert len(rows) == report["funnel"]["after_limit"]


def test_funnel_is_monotone(tmp_path):
    files = dict(GROUP_FILES)
    files["clean__train__4ppl.parquet"] = [CLEAN_4PPL, _clone(CLEAN_4PPL, 101), _clone(CLEAN_4PPL, 102)]
    _, report = _build(tmp_path, files)
    counts = list(report["funnel"].values())
    assert counts == sorted(counts, reverse=True)


def test_report_describes_the_selected_rows(tmp_path):
    rows, report = _build(tmp_path)
    assert report["groups"]["total"] == 1
    assert report["groups"]["size_histogram"] == {"6": 1}
    assert report["groups"]["skipped"] == {}
    assert report["selected"]["clean"] == 1
    assert report["selected"]["perturbed"] == 0
    assert report["selected"]["per_tier"] == {"4ppl": 1}
    assert report["enumerator"]["checked"] == 8
    assert report["enumerator"]["zero_solution"] == 0
    assert report["enumerator"]["multi_solution"] == 0
    assert report["enumerator"]["solution_mismatch"] == 0
    assert report["enumerator"]["solution_text_oracle"] == {"checked": 1, "matches": 1}


# ---------------------------------------------------------------------------
# drops
# ---------------------------------------------------------------------------


def test_a_file_whose_name_is_not_the_contract_is_dropped(tmp_path):
    files = dict(GROUP_FILES)
    files["perturbed__train__surprise_family.parquet"] = [CLEAN_4PPL]
    _, report = _build(tmp_path, files)
    assert report["funnel"]["raw_rows"] == 9
    assert report["funnel"]["after_filename_drop"] == 8


@pytest.mark.parametrize(
    "override",
    [
        {"knight_knave": {"knight": "knight"}},  # no liar word
        {"knight_knave": {"knight": "knight", "knave": "Knight"}},  # the same word twice
        {"knight_knave": "knight and knave"},
        {"names": ["Ethan", "Ethan", "David", "Noah"]},  # a duplicated inhabitant
        {"names": ["Ethan", "", "David", "Noah"]},
        {"names": []},
        {"solution": [False, True, True]},  # shorter than the roster
        {"quiz": "   "},
        {"quiz": None},
        {"statements": ""},
        {"index": None},
        {"index": True},
        {"index": "100"},
    ],
)
def test_malformed_rows_are_dropped_not_crashed(tmp_path, override):
    row = {**CLEAN_4PPL, **override}
    rows, report = _build(tmp_path, {"clean__train__4ppl.parquet": [row]})
    assert rows == []
    assert report["funnel"]["after_malformed_drop"] == 0


@pytest.mark.parametrize(
    "override",
    [
        {"knight_knave": {}},  # an empty struct cannot even be written to parquet
        {"knight_knave": {"knight": None, "knave": "knave"}},
        {"solution": [False, 1, True, True]},  # not a bool
        {"solution": "False True True True"},
        {"solution": None},
        {"names": "not a list"},
        {"index": 100.5},
        {"quiz": 3},
    ],
)
def test_shape_failures_that_cannot_be_serialised(override):
    """The same fail-closed rule for shapes parquet itself refuses to encode."""
    row = {**CLEAN_4PPL, **override, "_family": "clean", "_split": "train", "_file": "x"}
    assert adapter.is_well_formed(row) is False
    assert adapter.certify_row(row) is None


def test_unparseable_statements_are_dropped(tmp_path):
    row = {**CLEAN_4PPL, "statements": "not-a-literal"}
    rows, report = _build(tmp_path, {"clean__train__4ppl.parquet": [row]})
    assert rows == []
    assert report["funnel"]["after_malformed_drop"] == 1
    assert report["enumerator"]["unparseable_statements"] == 1
    assert report["funnel"]["after_enumerator_drop"] == 0


#: Two people who agree, plus a tautology and a contradiction: (t0, t1) free.
TWO_SOLUTIONS = (
    "(('telling-truth', 1), ('telling-truth', 0), "
    "('<=>', ('telling-truth', 0), ('telling-truth', 1)), "
    "('not', ('<=>', ('telling-truth', 2), ('telling-truth', 2))))"
)
#: The first inhabitant says "I am a liar", which no assignment satisfies.
NO_SOLUTION = (
    "(('lying', 0), ('telling-truth', 1), ('telling-truth', 2), ('telling-truth', 3))"
)


def test_a_puzzle_with_two_solutions_is_dropped(tmp_path):
    row = {**CLEAN_4PPL, "statements": TWO_SOLUTIONS}
    rows, report = _build(tmp_path, {"clean__train__4ppl.parquet": [row]})
    assert rows == []
    assert report["enumerator"]["multi_solution"] == 1
    assert report["enumerator"]["checked"] == 0
    assert report["funnel"]["after_enumerator_drop"] == 0


def test_a_puzzle_with_no_solution_is_dropped(tmp_path):
    row = {**CLEAN_4PPL, "statements": NO_SOLUTION}
    rows, report = _build(tmp_path, {"clean__train__4ppl.parquet": [row]})
    assert rows == []
    assert report["enumerator"]["zero_solution"] == 1
    assert report["funnel"]["after_enumerator_drop"] == 0


def test_a_solution_field_that_contradicts_the_enumeration_is_dropped(tmp_path):
    row = {**CLEAN_4PPL, "solution": [True, True, True, True]}
    rows, report = _build(tmp_path, {"clean__train__4ppl.parquet": [row]})
    assert rows == []
    assert report["enumerator"]["solution_mismatch"] == 1


def test_a_row_too_large_to_enumerate_is_dropped(tmp_path, monkeypatch):
    monkeypatch.setattr(adapter, "MAX_ENUMERATED", 3)
    rows, report = _build(tmp_path)
    # Only the 3-inhabitant row survives the bound; the N < 4 stage then drops it.
    assert report["enumerator"]["too_many_inhabitants"] == 7
    assert report["funnel"]["after_enumerator_drop"] == 1
    assert rows == []


# ---------------------------------------------------------------------------
# groups
# ---------------------------------------------------------------------------


def test_only_one_variant_of_a_group_enters_the_pool(tmp_path):
    rows, report = _build(tmp_path)
    assert len(rows) == 1
    assert report["groups"]["total"] == 1
    assert report["funnel"]["after_group_dedup"] == 1


def test_group_members_never_land_on_two_sides(tmp_path):
    # Two groups, each with a clean and a flip_role member: the pool must hold one
    # row per group, and every row must claim the same split.
    files = {
        "clean__train__4ppl.parquet": [CLEAN_4PPL, _clone(CLEAN_4PPL, 101)],
        "perturbed__train__flip_role.parquet": [FLIP_ROLE, _clone(FLIP_ROLE, 101)],
    }
    rows, report = _build(tmp_path, files)
    assert len(rows) == 2
    assert {row["extra_info"]["split"] for row in rows} == {"train"}
    keys = [adapter.group_key_for_task_id(row["extra_info"]["task_id"]) for row in rows]
    assert len(set(keys)) == 2


def test_selection_alternates_clean_and_perturbed(tmp_path):
    files = {
        "clean__train__4ppl.parquet": [_clone(CLEAN_4PPL, 100 + i) for i in range(10)],
        "perturbed__train__flip_role.parquet": [_clone(FLIP_ROLE, 100 + i) for i in range(10)],
    }
    rows, report = _build(tmp_path, files)
    assert len(rows) == 10
    assert report["selected"]["clean"] == 5
    assert report["selected"]["perturbed"] == 5


def test_pick_perturbed_shifts_off_a_missing_family():
    present = {"clean": {}, "perturbed_leaf": {}, "flip_role": {}}
    # offset 0 names perturbed_statement, absent here -> the next present family.
    assert adapter._pick_perturbed(present, 0) is present["perturbed_leaf"]
    assert adapter._pick_perturbed(present, 4) is present["flip_role"]  # uncommon_name absent
    assert adapter._pick_perturbed(present, 5) is present["flip_role"]
    assert adapter._pick_perturbed({"clean": {}}, 0) is None


def test_families_present_keeps_one_member_per_family():
    members = [
        {"_family": "clean", "_file": "b.parquet", "index": 1},
        {"_family": "clean", "_file": "a.parquet", "index": 1},
    ]
    present = adapter._families_present(members)
    assert len(present) == 1
    assert present["clean"]["_file"] == "a.parquet"  # first in (file, index) order


def test_a_perturbed_row_carries_the_clean_sibling_as_paired_original(tmp_path):
    files = {
        "clean__train__4ppl.parquet": [CLEAN_4PPL, _clone(CLEAN_4PPL, 101)],
        "perturbed__train__flip_role.parquet": [_clone(FLIP_ROLE, 100), _clone(FLIP_ROLE, 101)],
    }
    rows, _ = _build(tmp_path, files)
    perturbed = [row for row in rows if row["extra_info"]["perturbation_family"] == "flip_role"]
    assert perturbed, "the second group must supply the perturbed side"
    for row in perturbed:
        assert row["extra_info"]["paired_original_text"] == adapter.normalise_text(CLEAN_4PPL["quiz"])
    for row in rows:
        if row["extra_info"]["perturbation_family"] == "clean":
            assert row["extra_info"]["paired_original_text"] == ""


# ---------------------------------------------------------------------------
# task ids
# ---------------------------------------------------------------------------


def test_task_ids_encode_the_group_and_round_trip(tmp_path):
    rows, _ = _build(tmp_path)
    for row in rows:
        task_id = row["extra_info"]["task_id"]
        assert task_id.startswith("kk:")
        assert adapter.parse_task_id(task_id) == (
            row["extra_info"]["perturbation_family"],
            len(json.loads(row["reward_model"]["ground_truth"])["answer"].split()),
            row["extra_info"]["index"],
        )


@pytest.mark.parametrize(
    "task_id",
    ["", "kk", "kk:clean:4ppl", "kk:clean:4ppl:100:extra", "umwp-ans-1", "kk:surprise:4ppl:1", "kk:clean:xppl:1",
     "kk:clean:4ppl:notanint"],
)
def test_parse_task_id_rejects_anything_else(task_id):
    assert adapter.parse_task_id(task_id) is None


def test_no_private_keys_leak_into_the_artifact(tmp_path):
    rows, _ = _build(tmp_path)
    for row in rows:
        assert not [key for key in row if key.startswith("_")]
        assert not [key for key in row["extra_info"] if key.startswith("_")]


# ---------------------------------------------------------------------------
# the schema contract
# ---------------------------------------------------------------------------


def test_every_row_satisfies_the_schema_contract(tmp_path):
    rows, _ = _build(tmp_path)
    schema.validate_rows(rows)
    assert rows


def test_rows_are_solvable_roles_on_template_b(tmp_path):
    rows, _ = _build(tmp_path)
    for row in rows:
        info = row["extra_info"]
        assert row["data_source"] == schema.SOURCE_KK
        assert info["branch"] == schema.BRANCH_SOLVABLE_ROLES
        assert info["template"] == schema.TEMPLATE_B
        assert info["solvable"] is True
        assert info["options"] == []
        assert info["correct_option_id"] == ""
        assert info["has_diagnosis_label"] is False
        assert info["judgment_only"] is False
        assert info["difficulty"] == "4ppl"
        assert info["domain"] == "logic"
        assert info["source"] == "K-and-K"


def test_difficulty_is_the_string_tier(tmp_path):
    rows, _ = _build(tmp_path)
    row = rows[0]
    assert row["extra_info"]["difficulty"] == "4ppl"
    assert isinstance(row["extra_info"]["difficulty"], str)


# ---------------------------------------------------------------------------
# determinism and the CLI
# ---------------------------------------------------------------------------


def test_same_inputs_produce_identical_rows(tmp_path):
    first, _ = _build(tmp_path)
    second, _ = _build(tmp_path)
    assert first == second


def test_seed_is_recorded_but_does_not_choose_rows(tmp_path):
    first, _ = _build(tmp_path, seed=0)
    second, _ = _build(tmp_path, seed=7)
    assert [row["extra_info"]["task_id"] for row in first] == [row["extra_info"]["task_id"] for row in second]
    assert first[0]["extra_info"]["seed"] == 0
    assert second[0]["extra_info"]["seed"] == 7


def test_interleave_by_tier_round_robins_and_keeps_every_member():
    chosen = [{"names": ["a"] * 4, "index": i} for i in range(3)]
    chosen += [{"names": ["a"] * 5, "index": i} for i in range(2)]
    out = adapter._interleave_by_tier(chosen)
    assert [len(member["names"]) for member in out] == [4, 5, 4, 5, 4]
    assert sorted(map(id, out)) == sorted(map(id, chosen))


def test_limit_keeps_every_tier_represented(tmp_path):
    """A small slice spans the tiers; a tier-ordered prefix is one tier only.

    The pool is built in ``(len(names), index)`` order, so ``chosen[:limit]``
    returned 400 rows all at ``4ppl`` on the real corpus -- the shortest answers
    in the dataset, and the reason ``--limit 400`` reported a single tier.
    """
    files = {
        "clean__train__4ppl.parquet": [_clone(CLEAN_4PPL, 100 + i) for i in range(4)],
        "clean__train__5ppl.parquet": [_clone(CLEAN_5PPL, 100 + i) for i in range(4)],
    }
    rows, report = _build(tmp_path, files, limit=4)
    assert {row["extra_info"]["difficulty"] for row in rows} == {"4ppl", "5ppl"}
    assert report["selected"]["per_tier"] == {"4ppl": 2, "5ppl": 2}


def test_limit_caps_the_pool_without_unbalancing_it(tmp_path):
    files = {
        "clean__train__4ppl.parquet": [_clone(CLEAN_4PPL, 100 + i) for i in range(6)],
        "perturbed__train__flip_role.parquet": [_clone(FLIP_ROLE, 100 + i) for i in range(6)],
    }
    rows, report = _build(tmp_path, files, limit=4)
    assert len(rows) == 4
    assert report["funnel"]["after_limit"] == 4
    assert sum(1 for row in rows if row["extra_info"]["perturbation_family"] == "clean") == 2


def test_limit_above_the_pool_is_a_no_op(tmp_path):
    rows, _ = _build(tmp_path, limit=99)
    assert len(rows) == 1


def test_limit_zero_emits_nothing(tmp_path):
    rows, report = _build(tmp_path, limit=0)
    assert rows == []
    assert report["funnel"]["after_limit"] == 0


def test_main_writes_a_parquet_and_a_report(tmp_path, monkeypatch, capsys):
    raw_dir = _write_raw(tmp_path / "raw", GROUP_FILES)
    out = tmp_path / "kk.parquet"
    report_path = tmp_path / "kk_report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["kk_adapter.py", "--raw-dir", raw_dir, "--out", str(out), "--report", str(report_path)],
    )
    adapter.main()

    printed = capsys.readouterr().out
    assert "funnel (rows remaining after each stage" in printed
    assert "per perturbation_family" in printed
    written = schema.read_parquet_rows(str(out))
    assert len(written) == 1
    schema.validate_rows(written)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["funnel"]["raw_rows"] == 8
    assert report["selected"]["clean"] == 1


def test_load_source_fails_closed_on_a_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        adapter.load_source(str(tmp_path / "nope"))


def test_load_source_fails_closed_on_a_directory_without_parquet(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        adapter.load_source(str(tmp_path / "empty"))


# ---------------------------------------------------------------------------
# the audit script's own checks, on a fixture build
# ---------------------------------------------------------------------------


def test_verify_checks_pass_on_a_fixture_build(tmp_path):
    rows, _ = _build(tmp_path)
    index = verify_kk.load_raw_index(_write_raw(tmp_path / "raw", GROUP_FILES))
    reporter = verify_kk.Reporter()
    verify_kk.check_contract(rows, reporter)
    verify_kk.check_prompt_contract(rows, reporter)
    verify_kk.check_source_anchor(rows, index, str(tmp_path / "raw"), reporter)
    verify_kk.check_sentence_oracles(rows, index, reporter)
    sampled = verify_kk.sample_rows(rows, size=50, seed=0)
    verify_kk.check_l1(sampled, index, reporter)
    verify_kk.check_l2(rows, sampled, index, reporter)
    verify_kk.check_l3(rows, index, reporter)
    assert reporter.failed == []


def test_verify_audit_catches_a_coded_gold(tmp_path):
    """A canonical `knave knight` code in place of the surface words must fail."""
    rows, _ = _build(tmp_path)
    row = _by_task_id(rows)[adapter.task_id_of("clean", 4, 100)]
    payload = json.loads(row["reward_model"]["ground_truth"])
    payload["answer"] = "KNAK"
    row["reward_model"]["ground_truth"] = json.dumps(payload)
    reporter = verify_kk.Reporter()
    verify_kk.check_contract(rows, reporter)
    assert reporter.failed


def test_verify_audit_catches_a_canonical_word_gold(tmp_path):
    """`knave knight knight knight` on a flip_role row is the risk-9 inversion."""
    files = {
        "clean__train__4ppl.parquet": [CLEAN_4PPL, _clone(CLEAN_4PPL, 101)],
        "perturbed__train__flip_role.parquet": [_clone(FLIP_ROLE, 100), _clone(FLIP_ROLE, 101)],
    }
    rows, _ = _build(tmp_path, files)
    perturbed = [row for row in rows if row["extra_info"]["perturbation_family"] == "flip_role"]
    assert perturbed
    row = perturbed[0]
    payload = json.loads(row["reward_model"]["ground_truth"])
    payload["answer"] = "knave knight knight knight"  # canonical words, wrong row
    row["reward_model"]["ground_truth"] = json.dumps(payload)
    index = verify_kk.load_raw_index(_write_raw(tmp_path / "raw", files))
    reporter = verify_kk.Reporter()
    verify_kk.check_l3(rows, index, reporter)
    assert reporter.failed


def test_verify_contract_rejects_a_row_with_an_options_block(tmp_path):
    rows, _ = _build(tmp_path)
    rows[0]["extra_info"]["options"] = [{"id": "A", "text": "nonsense"}]
    reporter = verify_kk.Reporter()
    verify_kk.check_contract(rows, reporter)
    assert reporter.failed


def test_verify_question_of_rejects_a_prompt_without_a_template(tmp_path):
    rows, _ = _build(tmp_path)
    rows[0]["prompt"] = [{"role": "user", "content": "no blank line here"}]
    with pytest.raises(ValueError):
        verify_kk.question_of(rows[0])


PROMPT_TAIL_MUTATIONS = {
    "no_role_instruction": lambda prompt: prompt.replace(schema._KK_ROLE_INSTRUCTION, ""),
    "judge_template": lambda prompt: schema.render_prompt(prompt.split("\n\n")[0], schema.TEMPLATE_B_JUDGE),
    "broken_boxed": lambda prompt: prompt.replace("\\boxed", "BOXED"),
    "extra_instruction": lambda prompt: prompt + " Answer with knight and knave only.",
}


@pytest.mark.parametrize("name", sorted(PROMPT_TAIL_MUTATIONS))
def test_verify_prompt_contract_catches_every_broken_tail(tmp_path, name):
    """The tail is what the model reads; ``template == 'B'`` alone proves nothing."""
    rows, _ = _build(tmp_path)
    mutate = PROMPT_TAIL_MUTATIONS[name]
    for row in rows:
        row["prompt"][0]["content"] = mutate(row["prompt"][0]["content"])
    reporter = verify_kk.Reporter()
    verify_kk.check_prompt_contract(rows, reporter)
    assert reporter.failed


def test_names_in_question_reads_the_roster_in_order(tmp_path):
    rows, _ = _build(tmp_path)
    row = _by_task_id(rows)[adapter.task_id_of("clean", 4, 100)]
    assert verify_kk.names_in_question(verify_kk.question_of(row)) == CLEAN_4PPL["names"]


def test_verify_sentence_oracles_catch_a_flipped_canonical_solution(tmp_path):
    files = {
        "clean__train__4ppl.parquet": [CLEAN_4PPL, _clone(CLEAN_4PPL, 101)],
        "perturbed__train__flip_role.parquet": [_clone(FLIP_ROLE, 100), _clone(FLIP_ROLE, 101)],
    }
    rows, _ = _build(tmp_path, files)
    index = verify_kk.load_raw_index(_write_raw(tmp_path / "raw", files))
    perturbed = [row for row in rows if row["extra_info"]["perturbation_family"] == "flip_role"]
    assert perturbed
    row = perturbed[0]
    row["extra_info"]["canonical_solution"] = [
        not flag for flag in row["extra_info"]["canonical_solution"]
    ]
    reporter = verify_kk.Reporter()
    verify_kk.check_sentence_oracles(rows, index, reporter)
    assert reporter.failed


def test_verify_sentence_oracles_catch_a_reordered_roster(tmp_path):
    rows, _ = _build(tmp_path)
    index = verify_kk.load_raw_index(_write_raw(tmp_path / "raw", GROUP_FILES))
    row = _by_task_id(rows)[adapter.task_id_of("clean", 4, 100)]
    head, sep, tail = row["prompt"][0]["content"].partition("\n\n")
    roster = "Ethan, Abigail, David, and Noah"
    reordered = "Noah, David, Abigail, and Ethan"
    assert roster in head
    row["prompt"][0]["content"] = head.replace(roster, reordered) + sep + tail
    reporter = verify_kk.Reporter()
    verify_kk.check_sentence_oracles(rows, index, reporter)
    assert reporter.failed


# ---------------------------------------------------------------------------
# the real corpus (built once for the module)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_pool(tmp_path_factory):
    """The full real build, run once: 48,076 raw rows -> the shipped pool."""
    if not RAW_DIR.is_dir():
        pytest.skip(f"{RAW_DIR} is not present")
    out = tmp_path_factory.mktemp("kk_real")
    rows, report = adapter.build_rows(str(RAW_DIR))
    schema.normalise_extra_info(rows)
    schema.validate_rows(rows)
    path = out / "kk.parquet"
    schema.write_rows_parquet(rows, str(path))
    return rows, report, str(path)


def test_real_funnel_matches_the_measured_corpus(real_pool):
    _, report, _ = real_pool
    assert report["funnel"] == {
        "raw_rows": 48076,
        "after_filename_drop": 48076,
        "after_malformed_drop": 48076,
        "after_enumerator_drop": 48076,
        "after_n_ge_4_drop": 38423,
        "after_source_test_drop": 34931,
        "after_group_dedup": 5000,
        "after_limit": 5000,
    }
    assert report["enumerator"]["checked"] == 48076
    assert report["enumerator"]["zero_solution"] == 0
    assert report["enumerator"]["multi_solution"] == 0
    assert report["enumerator"]["solution_mismatch"] == 0
    assert report["enumerator"]["solution_text_oracle"]["matches"] == 5000
    assert report["enumerator"]["solution_text_format_oracle"]["matches"] == 5000


def test_real_pool_is_balanced(real_pool):
    rows, report, _ = real_pool
    assert len(rows) == 5000
    assert report["selected"]["clean"] == 2500
    assert report["selected"]["perturbed"] == 2500
    assert report["groups"]["total"] == 5000
    assert report["groups"]["size_histogram"] == {"5": 3, "6": 63, "7": 4934}
    assert report["selected"]["per_tier"] == {f"{tier}ppl": 1000 for tier in range(4, 9)}
    assert len(report["selected"]["per_family"]) == len(adapter.FAMILIES)
    # flip_role and random_pair are exactly the families that respell the roles.
    assert report["selected"]["non_canonical_role_words"] == 833


def test_real_limit_slice_spans_every_tier():
    """The slice build is a corpus, not a tier: guard the measured defect."""
    if not RAW_DIR.is_dir():
        pytest.skip(f"{RAW_DIR} is not present")
    rows, report = adapter.build_rows(str(RAW_DIR), limit=400)
    assert report["selected"]["per_tier"] == {"4ppl": 80, "5ppl": 80, "6ppl": 80, "7ppl": 80, "8ppl": 80}
    assert len(rows) == 400


def test_real_rows_are_unique_per_group_and_unique_ids(real_pool):
    rows, _, _ = real_pool
    task_ids = [row["extra_info"]["task_id"] for row in rows]
    assert len(set(task_ids)) == len(task_ids)
    groups = {adapter.group_key_for_task_id(task_id) for task_id in task_ids}
    assert len(groups) == len(rows)


def test_real_verify_passes(real_pool, capsys):
    _, _, path = real_pool
    with pytest.raises(SystemExit) as excinfo:
        verify_kk.main(["--rows", path, "--sample", "50"])
    assert excinfo.value.code == 0
    printed = capsys.readouterr().out
    assert "RESULT: PASS" in printed


def test_real_verify_fails_without_the_raw_files(real_pool, capsys):
    _, _, path = real_pool
    with pytest.raises(SystemExit) as excinfo:
        verify_kk.main(["--rows", path, "--raw-dir", "/nonexistent", "--sample", "50"])
    assert excinfo.value.code == 1
    assert "RESULT: FAIL" in capsys.readouterr().out
