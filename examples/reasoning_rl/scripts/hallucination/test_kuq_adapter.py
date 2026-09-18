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
"""Unit tests for the KUQ adapter and its audit.

Every fixture is inlined below and copied verbatim (whitespace included) out of
``knowns_unknowns.jsonl`` -- **no test reads the raw download**, so the suite runs
anywhere.  ``FIXTURE_LINES`` is the small end-to-end corpus: it carries one
example of each filter branch (certified both halves, both-verdict duplicate,
repeat question, out-of-scope category, and every certificate failure) so the
funnel can be asserted stage by stage.
"""

from __future__ import annotations

import json

import kuq_adapter
import pytest
import schema
import verify_kuq

# ---------------------------------------------------------------------------
# fixtures (verbatim raw records)
# ---------------------------------------------------------------------------

# Verbatim from the raw file, in file order.  Line numbers below are the index
# into this list, which is what ``extra_info.index`` records.
FIXTURE_LINES = [
    # 0 -- known / hotpotqa, certified, kept
    '{"question": "The 337th Flight Test Squadron (337 FLTS) was most recently part of the 46th Test Wing and based at McClellan Air Force Base, a former United States Air Force base located in the North Highlands area of Sacramento County, in which US state?", "answer": ["California"], "unknown": false, "source": "hotpotqa"}',  # noqa: E501 -- verbatim raw record, kept on one line
    # 1 -- known / triviaqa, certified, kept
    '{"question": "What date is Father\'s Day?", "answer": ["3rd sunday in june"], "unknown": false, "source": "triviaqa"}',  # noqa: E501 -- verbatim raw record, kept on one line
    # 2 -- known / squad, certified, kept
    '{"question": "What type of degree is an M.Div.?", "answer": ["Master of Divinity"], "unknown": false, "source": "squad"}',  # noqa: E501 -- verbatim raw record, kept on one line
    # 3 -- unknown / counterfactual, in scope, kept
    '{"question": "Can a person be double-jointed in their mind?", "answer": ["A person can have two minds at the same time.", "Question does not make sense", "A person can have two minds at the same time."], "unknown": true, "source": "turk", "category": "counterfactual"}',  # noqa: E501 -- verbatim raw record, kept on one line
    # 4 -- unknown / false assumption, in scope, kept
    '{"question": "Assuming that the advancements in space medicine have led to the discovery of a cure for all types of cancer, how would this affect the allocation of healthcare resources on Earth and the global economy, and what ethical questions would arise in the distribution and accessibility of this breakthrough treatment?", "answer": ["Answering this requires accepting an alternate reality where cancer is cured.", "It assumes that space medicine will cure cancer.", "the assumption  is  false"], "unknown": true, "source": "turk", "category": "false assumption"}',  # noqa: E501 -- verbatim raw record, kept on one line
    # 5 -- unknown / ambiguous: certified but out of D18 scope -> category_in_scope drop
    '{"question": "Can a fish breathe in space?", "answer": ["A fish can breathe in space, but it requires oxygen.", "The question itself is wrong.", "However, it is a counterfactual question because it is asking about a hypothetical scenario where a fish is placed in space, which is outside of its natural habitat and environment."], "unknown": true, "source": "turk", "category": "ambiguous"}',  # noqa: E501 -- verbatim raw record, kept on one line
    # 6 -- repeat of line 0's question: certified, dropped as a repeated question
    '{"question": "The 337th Flight Test Squadron (337 FLTS) was most recently part of the 46th Test Wing and based at McClellan Air Force Base, a former United States Air Force base located in the North Highlands area of Sacramento County, in which US state?", "answer": ["California"], "unknown": false, "source": "triviaqa"}',  # noqa: E501 -- verbatim raw record, kept on one line
    # 7/8 -- the real both-verdict question ("What is the population of the city?"):
    # both rows are certified and both are dropped by unambiguous_verdicts
    '{"question": "What is the population of the city?", "answer": ["Indianapolis"], "unknown": false, "source": "squad"}',  # noqa: E501 -- verbatim raw record, kept on one line
    '{"question": "What is the population of the city?", "answer": ["The question does not specify which city is meant, so it cannot be answered.", "The question is ambiguous."], "unknown": true, "source": "turk", "category": "ambiguous"}',  # noqa: E501 -- verbatim raw record, kept on one line
    # 9 -- malformed JSON -> parse drop
    '{"question": "truncated record", "answer": ["x"],',
    # 10 -- valid JSON that is not an object -> parse drop
    '["not", "a", "record"]',
    # 11 -- known row whose every element reads as an explanation -> certificate drop
    '{"question": "Who wrote the letter that was sent to the committee last year?", "answer": ["The letter was written by the chair of the committee, whose name is not given in the passage, so the writer cannot be identified from the text provided."], "unknown": false, "source": "squad"}',  # noqa: E501 -- verbatim raw record, kept on one line
    # 12 -- unknown row whose every element reads as an answer -> certificate drop
    '{"question": "What is the boiling point of mercury?", "answer": ["356.7", "674"], "unknown": true, "source": "turk", "category": "counterfactual"}',  # noqa: E501 -- verbatim raw record, kept on one line
    # 13 -- unknown verdict from a QA corpus source -> certificate drop
    '{"question": "How many moons does the planet have?", "answer": ["The question cannot be answered without knowing which planet is meant, and no planet is named anywhere in the passage."], "unknown": true, "source": "squad", "category": "false assumption"}',  # noqa: E501 -- verbatim raw record, kept on one line
    # 14 -- known row carrying a category -> certificate drop (presence disagrees)
    '{"question": "What colour is the sky on a clear day?", "answer": ["Blue"], "unknown": false, "source": "squad", "category": "ambiguous"}',  # noqa: E501 -- verbatim raw record, kept on one line
    # 15 -- category outside the documented six -> certificate drop
    '{"question": "Is the number of stars in the galaxy finite?", "answer": ["No census of the galaxy has ever been completed, so no count exists and none can be produced from the information available."], "unknown": true, "source": "turk", "category": "nonsense category"}',  # noqa: E501 -- verbatim raw record, kept on one line
    # 16 -- unknown row with no category at all -> certificate drop
    '{"question": "Will the new model be released before the end of the year?", "answer": ["There is no published release date for the model, so nobody outside the lab can answer this question at present."], "unknown": true, "source": "turk"}',  # noqa: E501 -- verbatim raw record, kept on one line
    # 17 -- missing source -> certificate drop
    '{"question": "What is the capital of the country?", "answer": ["Unknown"], "unknown": false}',
    # 18 -- unknown flag is not a bool -> certificate drop
    '{"question": "Is this question answerable at all?", "answer": ["Yes"], "unknown": "yes", "source": "turk"}',
    # 19 -- missing answer list -> certificate drop
    '{"question": "What is the name of the river mentioned above?", "unknown": false, "source": "squad"}',
    # 20 -- answer element is not a string -> certificate drop
    '{"question": "What is the registration number of the aircraft?", "answer": [42, "unknown"], "unknown": false, "source": "squad"}',  # noqa: E501 -- verbatim raw record, kept on one line
    # 21 -- degenerate question (2 word tokens) -> certificate drop
    '{"question": "Why not", "answer": ["Because"], "unknown": true, "source": "turk", "category": "counterfactual"}',
    # 22 -- a blank line is skipped and is not counted as a raw row
    "",
]

# Non-blank lines only: the funnel's first stage counts rows, not newlines.
FIXTURE_RAW_ROWS = sum(1 for line in FIXTURE_LINES if line.strip())

FIXTURE_JSONL = "\n".join(FIXTURE_LINES) + "\n"

# A second, all-questions-same-length fixture: the audit's length-cue check is
# then exactly at chance no matter which threshold a training fold picks.
SAME_LENGTH_LINES = [
    '{"question": "Apple beta gamma delta epsilon zeta", "answer": ["Fruit"], "unknown": false, "source": "squad"}',
    '{"question": "Brick beta gamma delta epsilon zeta", "answer": ["Brick"], "unknown": false, "source": "squad"}',
    '{"question": "Modal beta gamma delta epsilon zeta", "answer": ["This question cannot be answered because the premise it rests on is not well defined and no single answer exists."], "unknown": true, "source": "turk", "category": "false assumption"}',  # noqa: E501 -- verbatim raw record, kept on one line
    '{"question": "Alien beta gamma delta epsilon zeta", "answer": ["The question presupposes something that has never been observed and no accepted answer to it exists anywhere in the literature."], "unknown": true, "source": "turk", "category": "counterfactual"}',  # noqa: E501 -- verbatim raw record, kept on one line
]


def write_fixture(tmp_path, lines=FIXTURE_LINES, name="knowns_unknowns.jsonl"):
    """Materialise an inline fixture as a raw-dir layout."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(exist_ok=True)
    (raw_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(raw_dir)


def build_fixture_rows(tmp_path, lines=SAME_LENGTH_LINES, **kwargs):
    rows, funnel = kuq_adapter.build_rows(write_fixture(tmp_path, lines), **kwargs)
    return rows, funnel


# ---------------------------------------------------------------------------
# task_id / difficulty / form predicates
# ---------------------------------------------------------------------------


def test_task_id_is_stable_and_derived_from_the_question():
    question = "What date is Father's Day?"
    assert kuq_adapter.task_id_for(question) == kuq_adapter.task_id_for(question)
    assert kuq_adapter.task_id_for(question) != kuq_adapter.task_id_for("What date is Mother's Day?")
    # whitespace normalisation: the dedup key must not depend on spacing
    assert kuq_adapter.task_id_for("What  date is\nFather's Day?") == kuq_adapter.task_id_for(question)
    assert kuq_adapter.task_id_for(question).startswith("kuq-")


def test_difficulty_buckets():
    assert kuq_adapter.difficulty_for("one two three") == "short"
    assert kuq_adapter.difficulty_for(" ".join(["w"] * kuq_adapter.DIFFICULTY_SHORT_MAX_WORDS)) == "short"
    assert (
        kuq_adapter.difficulty_for(" ".join(["w"] * (kuq_adapter.DIFFICULTY_SHORT_MAX_WORDS + 1)))
        == "medium"
    )
    assert (
        kuq_adapter.difficulty_for(" ".join(["w"] * kuq_adapter.DIFFICULTY_MEDIUM_MAX_WORDS))
        == "medium"
    )
    assert kuq_adapter.difficulty_for(" ".join(["w"] * (kuq_adapter.DIFFICULTY_MEDIUM_MAX_WORDS + 1))) == "long"


@pytest.mark.parametrize(
    "answers, solvable, expected",
    [
        (["California"], True, None),
        (["3rd sunday in june"], True, None),
        # no determinate span on a solvable row
        (
            ["The letter was written by the chair of the committee, whose name is not given in the passage."],
            True,
            "no determinate answer span (every element reads as an explanation)",
        ),
        ([], True, "empty answer list"),
        # no uncertainty explanation on an unsolvable row
        (["356.7", "674"], False, "no uncertainty explanation (every element reads as an answer)"),
        (
            ["There is no consensus answer to this question and the available evidence does not settle it."],
            False,
            None,
        ),
        (["   "], False, "no uncertainty explanation (every element reads as an answer)"),
    ],
)
def test_payload_form_problem(answers, solvable, expected):
    assert kuq_adapter.payload_form_problem(answers, solvable=solvable) == expected


# ---------------------------------------------------------------------------
# certificate: one case per drop reason
# ---------------------------------------------------------------------------

CERTIFICATE_CASES = {
    "empty question": {"question": "  ", "answer": ["x"], "unknown": False, "source": "squad"},
    "question shorter than 3 word tokens": {
        "question": "Why not",
        "answer": ["Because"],
        "unknown": True,
        "source": "turk",
        "category": "counterfactual",
    },
    "unknown flag is not a bool": {
        "question": "Is this answerable?",
        "answer": ["No"],
        "unknown": 0,
        "source": "squad",
    },
    "missing answer list": {"question": "Where is the river?", "unknown": False, "source": "squad"},
    "answer list holds a non-string or empty element": {
        "question": "Where is the river?",
        "answer": [42],
        "unknown": False,
        "source": "squad",
    },
    "missing source": {"question": "Where is the river?", "answer": ["There"], "unknown": False},
    "unknown row from 'squad', expected 'turk'": {
        "question": "How many moons does the planet have?",
        "answer": ["No planet is named, so the question cannot be answered from the passage."],
        "unknown": True,
        "source": "squad",
        "category": "false assumption",
    },
    "category presence disagrees with the unknown flag": {
        "question": "What colour is the sky on a clear day?",
        "answer": ["Blue"],
        "unknown": False,
        "source": "squad",
        "category": "ambiguous",
    },
    "category 'nonsense category' is outside the documented six": {
        "question": "Is the star count finite?",
        "answer": ["No census of the galaxy has ever been completed, so no count exists at all."],
        "unknown": True,
        "source": "turk",
        "category": "nonsense category",
    },
    "no uncertainty explanation (every element reads as an answer)": {
        "question": "What is the boiling point of mercury?",
        "answer": ["356.7"],
        "unknown": True,
        "source": "turk",
        "category": "counterfactual",
    },
    "no determinate answer span (every element reads as an explanation)": {
        "question": "Who wrote the letter that was sent to the committee last year?",
        "answer": ["The passage does not name the author of the letter anywhere in its text."],
        "unknown": False,
        "source": "squad",
    },
}


@pytest.mark.parametrize("reason", sorted(CERTIFICATE_CASES))
def test_certificate_drop_reasons(reason):
    assert kuq_adapter.certify_record(CERTIFICATE_CASES[reason]) == reason


def test_certificate_accepts_both_halves():
    known = {
        "question": "What date is Father's Day?",
        "answer": ["3rd sunday in june"],
        "unknown": False,
        "source": "triviaqa",
    }
    unknown = {
        "question": "Can a person be double-jointed in their mind?",
        "answer": ["A person can have two minds at the same time.", "Question does not make sense"],
        "unknown": True,
        "source": "turk",
        "category": "counterfactual",
    }
    assert kuq_adapter.certify_record(known) is None
    assert kuq_adapter.certify_record(unknown) is None
    # an unseen source vocabulary is a dropped row, never a guessed label
    drifted = dict(known, source="new_corpus")
    assert kuq_adapter.certify_record(drifted) is not None


# ---------------------------------------------------------------------------
# category scope
# ---------------------------------------------------------------------------


def test_category_in_scope():
    assert kuq_adapter.category_in_scope({"unknown": False, "question": "q"}) is True
    for category in kuq_adapter.UNSOLVABLE_CATEGORIES_IN_SCOPE:
        assert kuq_adapter.category_in_scope({"unknown": True, "category": category}) is True
    for category in set(kuq_adapter.KUQ_CATEGORIES) - kuq_adapter.UNSOLVABLE_CATEGORIES_IN_SCOPE:
        assert kuq_adapter.category_in_scope({"unknown": True, "category": category}) is False


# ---------------------------------------------------------------------------
# row construction
# ---------------------------------------------------------------------------


def test_make_row_shape(tmp_path):
    known = json.loads(FIXTURE_LINES[1])
    row = kuq_adapter.make_kuq_row(known, line_no=1, seed=7)
    assert schema.validate_row(row) == []
    assert row["extra_info"]["task_id"] == kuq_adapter.task_id_for(known["question"])
    assert row["extra_info"]["template"] == schema.TEMPLATE_B_JUDGE
    assert row["extra_info"]["branch"] == schema.BRANCH_SOLVABLE_JUDGE
    assert row["extra_info"]["judgment_only"] is True
    assert row["extra_info"]["index"] == 1
    assert row["extra_info"]["seed"] == 7
    assert row["extra_info"]["split"] == "train"
    assert row["extra_info"]["has_diagnosis_label"] is False
    payload = json.loads(row["reward_model"]["ground_truth"])
    assert payload["solvable"] is True
    assert payload["judgment_only"] is True
    # the raw QA answers stay audit-only metadata
    assert payload["answer"] is None
    assert row["extra_info"]["kuq_answer_text"] == "3rd sunday in june"

    unknown = json.loads(FIXTURE_LINES[3])
    row = kuq_adapter.make_kuq_row(unknown, line_no=3, seed=0)
    assert schema.validate_row(row) == []
    assert row["extra_info"]["branch"] == schema.BRANCH_UNSOLVABLE_BARE
    assert row["extra_info"]["judgment_only"] is False
    assert row["extra_info"]["error_type"] == kuq_adapter.UNSOLVABLE_ERROR_TYPE
    assert row["extra_info"]["perturbation_type"] == "contradictory_condition"
    payload = json.loads(row["reward_model"]["ground_truth"])
    assert payload["solvable"] is False
    assert payload["answer"] is None
    assert payload["correct_option_id"] is None
    assert payload["has_diagnosis_label"] is False
    assert row["extra_info"]["kuq_category"] == "counterfactual"


# ---------------------------------------------------------------------------
# build_rows / funnel
# ---------------------------------------------------------------------------


def test_funnel_stage_by_stage(tmp_path):
    rows, funnel = kuq_adapter.build_rows(write_fixture(tmp_path))
    stages = funnel["stages"]
    diagnostics = funnel["diagnostics"]

    # 22 rows, minus the malformed JSON line and the JSON array; the blank line
    # is not a row at all
    assert stages["raw_lines"] == FIXTURE_RAW_ROWS == len(FIXTURE_LINES) - 1
    assert stages["parsed_rows"] == FIXTURE_RAW_ROWS - 2
    assert diagnostics["missing_or_malformed_lines"] == 2

    # certificate drops: empty question (none in the fixture), short question,
    # flags, malformed records, provenance, scope presence, form
    reasons = diagnostics["certificate_drop_reasons"]
    assert sum(reasons.values()) == stages["parsed_rows"] - stages["certified_records"]
    assert reasons["question shorter than 3 word tokens"] == 1
    assert reasons["unknown flag is not a bool"] == 1
    assert reasons["missing answer list"] == 1
    assert reasons["answer list holds a non-string or empty element"] == 1
    assert reasons["missing source"] == 1
    assert reasons["unknown row from 'squad', expected 'turk'"] == 1
    # both directions of the presence rule: a known row that carries a category
    # and an unknown row that does not
    assert reasons["category presence disagrees with the unknown flag"] == 2
    assert reasons["category 'nonsense category' is outside the documented six"] == 1
    assert reasons["no determinate answer span (every element reads as an explanation)"] == 1
    assert reasons["no uncertainty explanation (every element reads as an answer)"] == 1

    # the real both-verdict question drops both of its rows, then the repeated
    # question drops the later copy only
    assert diagnostics["questions_under_both_verdicts"] == 2
    assert diagnostics["repeated_questions"] == 1
    assert stages["unambiguous_verdicts"] == stages["certified_records"] - 2
    assert stages["deduplicated_questions"] == stages["unambiguous_verdicts"] - 1

    # one certified unknown is out of D18 scope ("ambiguous")
    assert stages["category_in_scope"] == stages["deduplicated_questions"] - 1
    assert stages["unique_task_ids"] == stages["category_in_scope"]
    assert stages["rows_after_limit"] == stages["unique_task_ids"]

    # monotone funnel, each stage below the previous
    counts = list(stages.values())
    assert counts == sorted(counts, reverse=True)
    assert len(rows) == stages["rows_after_limit"]


def test_both_verdict_question_is_dropped_entirely(tmp_path):
    rows, _ = kuq_adapter.build_rows(write_fixture(tmp_path))
    assert all("population of the city" not in row["extra_info"]["task_id"] for row in rows)
    questions = [row["prompt"][0]["content"] for row in rows]
    assert not any("population of the city" in question for question in questions)


def test_rows_have_both_branches_and_validate(tmp_path):
    rows, _ = kuq_adapter.build_rows(write_fixture(tmp_path))
    branches = {row["extra_info"]["branch"] for row in rows}
    assert branches == {schema.BRANCH_SOLVABLE_JUDGE, schema.BRANCH_UNSOLVABLE_BARE}
    for row in rows:
        assert schema.validate_row(row) == []
        payload = json.loads(row["reward_model"]["ground_truth"])
        if payload["solvable"]:
            assert payload["judgment_only"] is True
            assert payload["answer"] is None
        else:
            assert payload["answer"] is None
            assert row["extra_info"]["error_type"] == kuq_adapter.UNSOLVABLE_ERROR_TYPE
        # both halves use one template, the D14 isomorphism requirement
        assert row["extra_info"]["template"] == schema.TEMPLATE_B_JUDGE


def test_missing_raw_file_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        kuq_adapter.build_rows(str(tmp_path / "does-not-exist"))


def test_build_is_deterministic(tmp_path):
    raw_dir = write_fixture(tmp_path)
    first, funnel_a = kuq_adapter.build_rows(raw_dir)
    second, funnel_b = kuq_adapter.build_rows(raw_dir)
    assert [row["extra_info"]["task_id"] for row in first] == [
        row["extra_info"]["task_id"] for row in second
    ]
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert funnel_a["stages"] == funnel_b["stages"]


# ---------------------------------------------------------------------------
# apply_limit
# ---------------------------------------------------------------------------


def fake_rows(n_solvable, n_unsolvable):
    rows = [
        {"extra_info": {"solvable": True, "index": i}} for i in range(n_solvable)
    ] + [
        {"extra_info": {"solvable": False, "index": 100 + i}} for i in range(n_unsolvable)
    ]
    return rows


def test_apply_limit_is_a_noop_when_not_needed():
    import random

    rows = fake_rows(3, 2)
    assert kuq_adapter.apply_limit(rows, None, random.Random(0)) == rows
    assert kuq_adapter.apply_limit(rows, 5, random.Random(0)) == rows
    assert kuq_adapter.apply_limit(rows, 99, random.Random(0)) == rows
    assert kuq_adapter.apply_limit(rows, 0, random.Random(0)) == []
    assert kuq_adapter.apply_limit(rows, -1, random.Random(0)) == []


def test_apply_limit_keeps_both_branches_and_is_sorted():
    import random

    rows = fake_rows(30, 20)
    limited = kuq_adapter.apply_limit(rows, 10, random.Random(0))
    assert len(limited) == 10
    assert sum(row["extra_info"]["solvable"] for row in limited) == 6
    assert sum(not row["extra_info"]["solvable"] for row in limited) == 4
    indexes = [row["extra_info"]["index"] for row in limited]
    assert indexes == sorted(indexes)


def test_apply_limit_never_starves_the_rare_branch():
    import random

    # A 1-in-31 branch rounds to a quota of zero, which would ship a
    # single-branch artifact; the split guarantees it at least one row.
    for n_solvable, n_unsolvable in ((30, 1), (1, 30)):
        rows = fake_rows(n_solvable, n_unsolvable)
        limited = kuq_adapter.apply_limit(rows, 10, random.Random(0))
        assert len(limited) == 10
        assert sum(row["extra_info"]["solvable"] for row in limited) == (
            1 if n_solvable == 1 else len(limited) - 1
        )
        assert sum(not row["extra_info"]["solvable"] for row in limited) == (
            1 if n_unsolvable == 1 else len(limited) - 1
        )


@pytest.mark.parametrize(
    "n_solvable, n_unsolvable",
    [(30, 20), (6, 4), (3, 2), (5, 9), (30, 1), (1, 30), (1, 1), (5, 0), (0, 5), (0, 0)],
)
def test_branch_quotas_never_exceed_the_limit_or_the_branch(n_solvable, n_unsolvable):
    for limit in range(0, n_solvable + n_unsolvable + 3):
        n_s, n_u = kuq_adapter.branch_quotas(n_solvable, n_unsolvable, limit)
        assert n_s + n_u == min(limit, n_solvable + n_unsolvable)
        assert 0 <= n_s <= n_solvable
        assert 0 <= n_u <= n_unsolvable


def test_branch_quotas_hits_the_limit_when_rounding_undershoots():
    # a proportional split rounds to 6 of a limit of 7; the remainder is handed
    # to the branch that still has rows (5+9 pool, 7 rows wanted)
    assert kuq_adapter.branch_quotas(5, 9, 7) == (3, 4)
    assert sum(kuq_adapter.branch_quotas(5, 9, 7)) == 7


def test_apply_limit_depends_on_the_seed_only_through_sampling():
    import random

    rows = fake_rows(6, 4)
    same = kuq_adapter.apply_limit(rows, 5, random.Random(3))
    again = kuq_adapter.apply_limit(rows, 5, random.Random(3))
    assert same == again
    outcomes = {
        tuple(row["extra_info"]["index"] for row in kuq_adapter.apply_limit(rows, 5, random.Random(seed)))
        for seed in range(12)
    }
    assert len(outcomes) > 1


def test_limit_across_builds(tmp_path):
    raw_dir = write_fixture(tmp_path, SAME_LENGTH_LINES)
    rows_a, funnel_a = kuq_adapter.build_rows(raw_dir, limit=2, seed=0)
    rows_b, _ = kuq_adapter.build_rows(raw_dir, limit=2, seed=0)
    rows_c, _ = kuq_adapter.build_rows(raw_dir, limit=2, seed=1)
    assert len(rows_a) == len(rows_b) == len(rows_c) == 2
    assert [r["extra_info"]["task_id"] for r in rows_a] == [r["extra_info"]["task_id"] for r in rows_b]
    assert funnel_a["stages"]["rows_after_limit"] == 2
    assert all(row["extra_info"]["seed"] == 0 for row in rows_a)
    # every limited row is a real member of the unlimited pool
    full, _ = kuq_adapter.build_rows(raw_dir)
    pool = {row["extra_info"]["task_id"] for row in full}
    assert {row["extra_info"]["task_id"] for row in rows_a} <= pool


# ---------------------------------------------------------------------------
# report + parquet round trip
# ---------------------------------------------------------------------------


def test_format_report_mentions_every_stage_and_breakdown(tmp_path):
    rows, funnel = kuq_adapter.build_rows(write_fixture(tmp_path))
    report = kuq_adapter.format_report(rows, funnel)
    for stage in funnel["stages"]:
        assert stage in report
    for heading in (
        "by branch:",
        "by template:",
        "by solvable:",
        "by kuq category (audit only):",
        "by difficulty:",
    ):
        assert heading in report
    assert "certificate drop:" in report
    assert "contradictory duplicates (both verdicts): 2 rows dropped" in report
    assert "repeated questions (same verdict): 1 rows dropped" in report


def test_parquet_round_trip(tmp_path):
    rows, _ = kuq_adapter.build_rows(write_fixture(tmp_path))
    schema.normalise_extra_info(rows)
    schema.validate_rows(rows)
    out = tmp_path / "kuq.parquet"
    schema.write_rows_parquet(rows, str(out))
    assert schema.read_parquet_rows(str(out)) == rows


def test_main_writes_a_valid_parquet(tmp_path, monkeypatch, capsys):
    raw_dir = write_fixture(tmp_path, SAME_LENGTH_LINES)
    out = tmp_path / "out" / "kuq.parquet"
    monkeypatch.setattr(
        "sys.argv",
        [
            "kuq_adapter.py",
            "--raw-dir",
            raw_dir,
            "--limit",
            "3",
            "--out",
            str(out),
            "--seed",
            "5",
        ],
    )
    kuq_adapter.main()
    printed = capsys.readouterr().out
    assert "funnel:" in printed
    assert "by branch:" in printed
    assert "by template:" in printed
    assert "by solvable:" in printed
    assert "by difficulty:" in printed
    assert f"wrote 3 rows -> {out}" in printed

    rows = schema.read_parquet_rows(str(out))
    assert len(rows) == 3
    assert {row["extra_info"]["branch"] for row in rows} == {
        schema.BRANCH_SOLVABLE_JUDGE,
        schema.BRANCH_UNSOLVABLE_BARE,
    }
    for row in rows:
        assert schema.validate_row(row) == []
        assert row["extra_info"]["seed"] == 5


def test_task_id_collision_is_dropped_defensively(tmp_path, monkeypatch):
    # the guard is unreachable with a real sha1 key; it exists so a future key
    # change cannot silently emit two rows for one hard-replay id
    monkeypatch.setattr(kuq_adapter, "task_id_for", lambda question: "kuq-collision")
    rows, funnel = kuq_adapter.build_rows(write_fixture(tmp_path))
    assert len(rows) == 1
    assert funnel["stages"]["unique_task_ids"] == 1
    assert funnel["stages"]["deduplicated_questions"] == 6


# ---------------------------------------------------------------------------
# verify_kuq: axis logic
# ---------------------------------------------------------------------------


def test_l1_axes_polarity(tmp_path):
    rows, _ = build_fixture_rows(tmp_path)
    axes = {row["extra_info"]["solvable"]: verify_kuq.l1_axis_verdicts(row) for row in rows}
    assert axes[True] == {"provenance": True, "annotation": True, "payload": True}
    assert axes[False] == {"provenance": False, "annotation": False, "payload": False}


def test_l1_axes_abstain_on_unknown_vocabulary(tmp_path):
    rows, _ = build_fixture_rows(tmp_path)
    drifted = json.loads(json.dumps(rows[0]))
    drifted["extra_info"]["kuq_source"] = "brand_new_corpus"
    drifted["extra_info"]["kuq_category"] = "outside the taxonomy"
    drifted["extra_info"]["kuq_answer_count"] = 99
    axes = verify_kuq.l1_axis_verdicts(drifted)
    assert axes == {"provenance": None, "annotation": None, "payload": None}


def test_l1_axes_payload_is_undecided_when_both_forms_fit(tmp_path):
    # a turk row whose answer list holds both an answer-shaped and an
    # explanation-shaped element: the axis abstains instead of picking a side
    raw = json.loads(FIXTURE_LINES[3])
    row = kuq_adapter.make_kuq_row(raw, line_no=0, seed=0)
    axes = verify_kuq.l1_axis_verdicts(row)
    assert axes["provenance"] is False
    assert axes["annotation"] is False
    assert axes["payload"] is None


# ---------------------------------------------------------------------------
# verify_kuq: checks
# ---------------------------------------------------------------------------


def test_check_schema_passes_on_built_rows(tmp_path):
    rows, _ = build_fixture_rows(tmp_path)
    assert verify_kuq.check_schema(rows)["ok"] is True


def test_check_schema_reports_violations(tmp_path):
    rows, _ = build_fixture_rows(tmp_path)
    broken = json.loads(json.dumps(rows))
    broken[0]["extra_info"]["options"] = [{"id": "A", "text": "x"}]
    broken[0]["reward_model"]["ground_truth"] = json.dumps(
        {"solvable": False, "answer": "leaked"}
    )
    outcome = verify_kuq.check_schema(broken)
    assert outcome["ok"] is False
    assert "violations" in outcome["detail"]


def test_check_l1_certificate(tmp_path):
    rows, _ = build_fixture_rows(tmp_path)
    outcome = verify_kuq.check_l1_certificate(rows, sample_size=50)
    assert outcome["ok"] is True
    assert "axis disagreements=none" in outcome["detail"]
    assert "unsolvable rows carrying an answer=0" in outcome["detail"]


def test_check_l1_certificate_catches_a_wrong_label(tmp_path):
    rows, _ = build_fixture_rows(tmp_path)
    broken = json.loads(json.dumps(rows))
    # flip a solvable row's verdict while leaving its provenance and categories
    # untouched: the decisive axes must object
    flipped = next(row for row in broken if row["extra_info"]["solvable"])
    flipped["reward_model"]["ground_truth"] = json.dumps(
        {"solvable": False, "answer": None, "has_diagnosis_label": False}
    )
    assert verify_kuq.check_l1_certificate(broken, sample_size=50)["ok"] is False


def test_check_l2_uniqueness(tmp_path):
    rows, _ = build_fixture_rows(tmp_path)
    assert verify_kuq.check_l2_uniqueness(rows)["ok"] is True
    duplicated = json.loads(json.dumps(rows))
    duplicated[1]["extra_info"]["task_id"] = duplicated[0]["extra_info"]["task_id"]
    assert verify_kuq.check_l2_uniqueness(duplicated)["ok"] is False


def test_check_l2_catches_a_shared_question(tmp_path):
    rows, _ = build_fixture_rows(tmp_path)
    shared = json.loads(json.dumps(rows))
    shared[0]["prompt"][0]["content"] = shared[-1]["prompt"][0]["content"]
    assert verify_kuq.check_l2_uniqueness(shared)["ok"] is False


def test_check_l3_option_cues(tmp_path):
    rows, _ = build_fixture_rows(tmp_path)
    assert verify_kuq.check_l3_option_cues(rows)["ok"] is True
    broken = json.loads(json.dumps(rows))
    broken[0]["prompt"][0]["content"] = "选项：\nA. x\n" + broken[0]["prompt"][0]["content"]
    assert verify_kuq.check_l3_option_cues(broken)["ok"] is False


def test_check_l3_template_isomorphism(tmp_path):
    rows, _ = build_fixture_rows(tmp_path)
    outcome = verify_kuq.check_l3_template_isomorphism(rows)
    assert outcome["ok"] is True
    assert "distinct prompt suffixes=1" in outcome["detail"]


def test_check_l3_template_isomorphism_catches_a_mixed_template(tmp_path):
    rows, _ = build_fixture_rows(tmp_path)
    broken = json.loads(json.dumps(rows))
    broken[0]["extra_info"]["template"] = schema.TEMPLATE_B
    broken[0]["prompt"][0]["content"] = schema.render_prompt(
        verify_kuq._prompt_question(broken[0]), schema.TEMPLATE_B
    )
    assert verify_kuq.check_l3_template_isomorphism(broken)["ok"] is False


# ---------------------------------------------------------------------------
# verify_kuq: failure detection on purpose-built broken rows
# ---------------------------------------------------------------------------


def test_l1_certificate_reports_every_kind_of_broken_row(tmp_path):
    # rows built from the full fixture: the unknown half carries answer-shaped
    # and explanation-shaped elements at once, so the payload axis abstains
    rows, _ = kuq_adapter.build_rows(write_fixture(tmp_path))
    outcome = verify_kuq.check_l1_certificate(rows, sample_size=50)
    assert outcome["ok"] is True
    assert "undecided=" in outcome["detail"]
    assert not outcome["detail"].endswith("undecided=0")

    # the same rows in the uniqueness check: an abstaining payload is not a
    # second verdict, so the check still passes and says how many abstained
    outcome = verify_kuq.check_l2_uniqueness(rows)
    assert outcome["ok"] is True
    assert "payload axis undecided on" in outcome["detail"]

    broken = json.loads(json.dumps(rows))
    solvable = next(row for row in broken if row["extra_info"]["solvable"])
    unsolvable = next(row for row in broken if not row["extra_info"]["solvable"])
    # an unsolvable row carrying an answer
    payload = json.loads(unsolvable["reward_model"]["ground_truth"])
    unsolvable["reward_model"]["ground_truth"] = json.dumps(dict(payload, answer="leaked"))
    # a solvable judgment row without judgment_only
    payload = json.loads(solvable["reward_model"]["ground_truth"])
    solvable["reward_model"]["ground_truth"] = json.dumps(dict(payload, judgment_only=False))
    # a prompt that no longer round-trips
    other = next(row for row in broken if row is not solvable and row is not unsolvable)
    other["prompt"][0]["content"] = "tampered"
    outcome = verify_kuq.check_l1_certificate(broken, sample_size=50)
    assert outcome["ok"] is False
    assert "unsolvable rows carrying an answer=1" in outcome["detail"]
    assert "solvable rows without judgment_only=1" in outcome["detail"]
    assert "prompts not equal to render_prompt" in outcome["detail"]


def test_check_l1_certificate_on_no_rows():
    outcome = verify_kuq.check_l1_certificate([], sample_size=50)
    assert outcome["ok"] is False
    assert outcome["detail"] == "no rows"


def test_check_l2_catches_diagnosis_payloads_and_axis_dissent(tmp_path):
    rows, _ = build_fixture_rows(tmp_path)
    broken = json.loads(json.dumps(rows))
    payload = json.loads(broken[0]["reward_model"]["ground_truth"])
    broken[0]["reward_model"]["ground_truth"] = json.dumps(
        dict(payload, has_diagnosis_label=True, correct_option_id="B")
    )
    outcome = verify_kuq.check_l2_uniqueness(broken)
    assert outcome["ok"] is False
    assert "rows carrying a diagnosis payload=1" in outcome["detail"]

    drifted = json.loads(json.dumps(rows))
    drifted[0]["extra_info"]["kuq_source"] = "brand_new_corpus"
    drifted[0]["extra_info"]["options"] = [{"id": "A", "text": "pear"}]
    outcome = verify_kuq.check_l2_uniqueness(drifted)
    assert outcome["ok"] is False
    assert "rows with a dissenting axis=1" in outcome["detail"]
    assert "rows carrying options=1" in outcome["detail"]


def test_check_l3_option_cues_reports_orphan_option_text(tmp_path):
    rows, _ = build_fixture_rows(tmp_path)
    broken = json.loads(json.dumps(rows))
    broken[0]["extra_info"]["options"] = [{"id": "A", "text": "text absent from the question"}]
    outcome = verify_kuq.check_l3_option_cues(broken)
    assert outcome["ok"] is False
    assert "rows with an options list=1" in outcome["detail"]
    assert "rows with option text absent from the question=1" in outcome["detail"]


# ---------------------------------------------------------------------------
# verify_kuq: the Naive Bayes gate
# ---------------------------------------------------------------------------


def test_balanced_accuracy_definition():
    assert verify_kuq.balanced_accuracy([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert verify_kuq.balanced_accuracy([0, 0, 1, 1], [1, 1, 0, 0]) == 0.0
    assert verify_kuq.balanced_accuracy([1, 1, 1, 1], [1, 1, 0, 0]) == 0.5


def test_nb_separates_a_trivially_separable_corpus():
    texts = [f"apple banana cherry {i}" for i in range(12)] + [
        f"rocket galaxy nebula {i}" for i in range(12)
    ]
    labels = [0] * 12 + [1] * 12
    predictions, fold_of = verify_kuq.nb_out_of_fold(texts, labels)
    assert verify_kuq.balanced_accuracy(predictions, labels) == 1.0
    assert set(fold_of.tolist()) == {0, 1, 2, 3, 4}
    assert len(fold_of) == len(texts)


def test_nb_does_not_trip_the_gate_when_the_labels_are_independent_of_the_text():
    # every document carries the same four tokens (the per-document index token
    # is unique, so the >= 5 document support filter removes it): there is no
    # class-linked vocabulary left to learn from
    texts = [f"alpha beta gamma delta {i}" for i in range(20)]
    labels = [i % 2 for i in range(20)]
    predictions, _ = verify_kuq.nb_out_of_fold(texts, labels)
    accuracy = verify_kuq.balanced_accuracy(predictions, labels)
    assert accuracy <= 0.60
    # and a reading *below* chance here is the fold-wise-prior pooling artefact,
    # not evidence about the data: with no surviving token the model can only
    # follow each fold's class prior, and those priors differ per fold
    assert len(set(predictions)) <= 2


def test_nb_is_deterministic():
    texts = [f"word{i % 5} token{i % 3} filler" for i in range(30)]
    labels = [i % 2 for i in range(30)]
    first, folds_a = verify_kuq.nb_out_of_fold(texts, labels, seed=11)
    second, folds_b = verify_kuq.nb_out_of_fold(texts, labels, seed=11)
    assert first == second
    assert folds_a.tolist() == folds_b.tolist()
    _, folds_c = verify_kuq.nb_out_of_fold(texts, labels, seed=12)
    assert folds_a.tolist() != folds_c.tolist()


def test_nb_vocabulary_support_filter_removes_the_signal():
    texts = [f"alpha unique{i} " for i in range(20)] + [f"beta unique{i} " for i in range(20)]
    labels = [0] * 20 + [1] * 20
    unrestricted, _ = verify_kuq.nb_out_of_fold(texts, labels)
    assert verify_kuq.balanced_accuracy(unrestricted, labels) == 1.0
    # no token appears in 25 training documents, so nothing survives the support
    # filter and the model is left with the class prior alone
    restricted, _ = verify_kuq.nb_out_of_fold(texts, labels, min_support=25)
    assert abs(verify_kuq.balanced_accuracy(restricted, labels) - 0.5) <= 0.15


def separable_rows(tmp_path, n_per_class=8):
    lines = []
    for i in range(n_per_class):
        lines.append(
            json.dumps(
                {
                    "question": f"apple banana cherry {i}",
                    "answer": ["Fruit"],
                    "unknown": False,
                    "source": "squad",
                }
            )
        )
    for i in range(n_per_class):
        lines.append(
            json.dumps(
                {
                    "question": f"rocket galaxy nebula {i}",
                    "answer": [
                        "This question cannot be answered because the premise it rests on "
                        "is not well defined and no single answer exists."
                    ],
                    "unknown": True,
                    "source": "turk",
                    "category": "counterfactual",
                }
            )
        )
    rows, _ = kuq_adapter.build_rows(write_fixture(tmp_path, lines))
    return rows


def test_check_l3_nb_enforces_its_gate(tmp_path):
    rows = separable_rows(tmp_path)
    outcome = verify_kuq.check_l3_nb(rows, gate=0.60)
    assert outcome["ok"] is False
    assert "out-of-fold balanced accuracy=" in outcome["detail"]
    # the gate is a real threshold, not a label on the report
    assert verify_kuq.check_l3_nb(rows, gate=1.0)["ok"] is True


def test_check_l3_nb_skips_a_single_class_artifact(tmp_path):
    rows, _ = build_fixture_rows(tmp_path)
    single = [row for row in rows if row["extra_info"]["solvable"]]
    outcome = verify_kuq.check_l3_nb(single)
    assert outcome["ok"] is None
    assert "single-class" in outcome["detail"]


# ---------------------------------------------------------------------------
# verify_kuq: raw-source counts and the driver
# ---------------------------------------------------------------------------


def test_check_raw_source_counts_skips_without_a_raw_dir():
    assert verify_kuq.check_raw_source_counts(None)["ok"] is None
    assert verify_kuq.check_raw_source_counts("/nonexistent/dir")["ok"] is None


def test_check_raw_source_counts_rejects_a_small_file(tmp_path):
    # the fixture is small *and* carries a malformed line: the check must report
    # both as failures rather than raising on the unparseable line
    outcome = verify_kuq.check_raw_source_counts(write_fixture(tmp_path))
    assert outcome["ok"] is False
    assert "expect 6884" in outcome["detail"]
    assert "unparseable lines=2" in outcome["detail"]


def test_audit_returns_every_check(tmp_path):
    rows, _ = build_fixture_rows(tmp_path)
    outcomes = verify_kuq.audit(rows, raw_dir=None)
    names = [outcome["name"] for outcome in outcomes]
    assert names == [
        "schema.validate_row",
        "L1 label certificate",
        "L2 verdict uniqueness",
        "L3a option-presence cue",
        "L3b template isomorphism / length cue",
        "L3c bag-of-words NB",
        "raw-source counts",
    ]
    assert all(outcome["detail"] for outcome in outcomes)


def test_audit_fails_on_an_empty_artifact():
    outcomes = verify_kuq.audit([])
    assert outcomes[0]["ok"] is False


def test_verify_main_exits_nonzero_on_a_failing_artifact(tmp_path, monkeypatch, capsys):
    rows = separable_rows(tmp_path)
    schema.normalise_extra_info(rows)
    schema.validate_rows(rows)
    out = tmp_path / "separable.parquet"
    schema.write_rows_parquet(rows, str(out))

    monkeypatch.setattr(
        "sys.argv",
        [
            "verify_kuq.py",
            "--rows",
            str(out),
            "--raw-dir",
            str(tmp_path / "no-raw-file-here"),
        ],
    )
    assert verify_kuq.main() == 1
    printed = capsys.readouterr().out
    assert "[PASS] schema.validate_row" in printed
    assert "[PASS] L2 verdict uniqueness" in printed
    assert "[FAIL] L3c bag-of-words NB" in printed
    assert "[SKIP] raw-source counts" in printed
    assert "out-of-fold balanced accuracy=" in printed


def test_render_question_round_trip(tmp_path):
    rows, _ = build_fixture_rows(tmp_path)
    raw_questions = {json.loads(line)["question"] for line in SAME_LENGTH_LINES}
    assert {verify_kuq._prompt_question(row) for row in rows} == raw_questions
    for row in rows:
        assert row["prompt"][0]["content"] == schema.render_prompt(
            verify_kuq._prompt_question(row), row["extra_info"]["template"]
        )
