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
"""Tests for ``crepe_adapter.py`` (and the L3 harness in ``verify_crepe.py``).

Every fixture is inline and tiny, and the CREPE / KUQ records are copied verbatim
out of the raw bundle (``crepe_{train,validation,test}.parquet`` and
``knowns_unknowns.jsonl``) -- ids, questions, labels, presuppositions, answers and
categories are the source's own bytes.  Rows built on top of them exist only to
drive one filter branch each and are marked ``# synthetic``.  **Nothing here reads
``/home/charles/data/reasoning_rl/halluc/raw``**: the bundles are written into
``tmp_path`` with pyarrow.

The main bundle is laid out so every funnel stage drops at least as many rows as
it owns and every drop reason has exactly one owner:

====  ==========================================  ==================================
idx   record                                        outcome
====  ==========================================  ==================================
0     CREPE train 2018-09504 ['normal']             kept, branch solvable_judge
1     CREPE train 2018-03713 ['false presupp.']      kept, branch unsolvable_bare
      (its presupposition shares the question's opening span but is not
       verbatim -- a 10-row fixture cannot reproduce the real 14/1295 rate)
2     CREPE train 2018-00818 ['false presupp.']      kept (paraphrase presupposition)
3     CREPE train 2018-24401 ['fp','normal']         dropped: label_uniqueness (dual)
4     CREPE validation 2019_a-01077 ['normal']       kept
5     CREPE test 2019_b-11155 ['false presupp.']     kept
6     KUQ line 0 known, hotpotqa                     kept, branch solvable_judge
7     KUQ line 1 known, triviaqa (4 answers)         kept
8     KUQ line 2 unknown, false assumption           kept, branch unsolvable_bare
9     KUQ line 3 unknown, counterfactual             kept
10    KUQ line 4 unknown, controversial              dropped: category filter
11    KUQ line 5 known, trailing space in question   kept (whitespace stripped)
====  ==========================================  ==================================
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import crepe_adapter as ca  # noqa: E402
import schema  # noqa: E402
import verify_crepe as vc  # noqa: E402

# ---------------------------------------------------------------------------
# verbatim fixture records (raw bundle bytes)
# ---------------------------------------------------------------------------

CREPE_NORMAL = {
    "id": "2018-09504",
    "question": 'When a lawyer "objects" to another lawyers statement, how is it handled.',
    "labels": ["normal"],
    "presuppositions": [],
}
CREPE_FP_OVERLAP = {
    "id": "2018-03713",
    "question": "Ohm's law applies on alternate current till 220V but not further, why ?",
    "labels": [ca.CREPE_FALSE_PRESUPPOSITION],
    "presuppositions": ["Ohm's law applies on alternate current till 220V."],
}
CREPE_FP_PARAPHRASE = {
    "id": "2018-00818",
    "question": "Does eating spicy food destroy beneficial bacteria in your colon?",
    "labels": [ca.CREPE_FALSE_PRESUPPOSITION],
    "presuppositions": [
        "Eating spicy food could potentially destroy beneficial bacteria in one's colon."
    ],
}
CREPE_DUAL = {
    "id": "2018-24401",
    "question": "Is it just humans that have such a hard time giving birth or is it just because we can Express it?",
    "labels": [ca.CREPE_FALSE_PRESUPPOSITION, ca.CREPE_NORMAL],
    "presuppositions": [
        "Humans are perceived to have a hard time giving birth because they are verbal about it."
    ],
}
CREPE_VALIDATION = {
    "id": "2019_a-01077",
    "question": "Why does a cold cause your voice to get deeper?",
    "labels": ["normal"],
    "presuppositions": [],
}
CREPE_TEST = {
    "id": "2019_b-11155",
    "question": "Why we can recover files after deleting them from HDD/SSD drives?",
    "labels": [ca.CREPE_FALSE_PRESUPPOSITION],
    "presuppositions": ["Files are removed from HDD/SSD drives when deleted."],
}

KUQ_KNOWN = {
    "question": "What wrestling promotion was formed by the current NEVER Openweight Champion?",
    "answer": ["Pro Wrestling Fujiwara Group"],
    "unknown": False,
    "source": "hotpotqa",
}
KUQ_KNOWN_MULTI = {
    "question": "In which decade of the 20th century was the FBI set up?",
    "answer": ["1900 s", "nineteen hundreds", "1900s", "1900s disambiguation"],
    "unknown": False,
    "source": "triviaqa",
}
KUQ_UNKNOWN_FA = {
    "question": "Can AI make the best rum brand?",
    "answer": ["AI is incapable of creating Rum let alone rum brands."],
    "unknown": True,
    "source": "turk",
    "category": "false assumption",
}
KUQ_UNKNOWN_CF = {
    "question": "Can a person be double-jointed in their mind?",
    "answer": ["A person can have two minds at the same time."],
    "unknown": True,
    "source": "turk",
    "category": "counterfactual",
}
KUQ_UNKNOWN_OTHER = {
    "question": "Are Moms better than Dads?",
    "answer": ["This is a subjective question."],
    "unknown": True,
    "source": "turk",
    "category": "controversial",
}
KUQ_WHITESPACE = {
    "question": "VfL Wolfsburg's attacking midfielder is what nationality? ",
    "answer": ["Bosnian"],
    "unknown": False,
    "source": "hotpotqa",
}

# The six KUQ lines, in file order: the line index *is* the KUQ source identity.
KUQ_FIXTURE = [
    KUQ_KNOWN,
    KUQ_KNOWN_MULTI,
    KUQ_UNKNOWN_FA,
    KUQ_UNKNOWN_CF,
    KUQ_UNKNOWN_OTHER,
    KUQ_WHITESPACE,
]

CREPE_DEFAULT = {
    "train": [CREPE_NORMAL, CREPE_FP_OVERLAP, CREPE_FP_PARAPHRASE, CREPE_DUAL],
    "validation": [CREPE_VALIDATION],
    "test": [CREPE_TEST],
}

# 12 raw records -> 10 rows: one dual-label drop, one out-of-scope-category drop.
EXPECTED_FUNNEL = {
    "raw_rows": 12,
    "after_question_text": 12,
    "after_label_uniqueness": 11,
    "after_certificate": 11,
    "after_kuq_category_filter": 10,
    "after_cross_label_conflict": 10,
    "after_dedup": 10,
    "after_quota": 10,
    "after_limit": 10,
}

# Interleaved one row per quota group per round (GROUP_ORDER), each group first
# round-robin over its native split -- so the order itself is under test.
EXPECTED_TASK_IDS = [
    "crepe:train:2018-09504",
    "crepe:test:2019_b-11155",
    "kuq:00000",
    "kuq:00002",
    "crepe:validation:2019_a-01077",
    "crepe:train:2018-00818",
    "kuq:00001",
    "kuq:00003",
    "crepe:train:2018-03713",
    "kuq:00005",
]

CREPE_PARQUET_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("question", pa.string()),
        ("labels", pa.list_(pa.string())),
        ("presuppositions", pa.list_(pa.string())),
    ]
)


# ---------------------------------------------------------------------------
# bundle writer
# ---------------------------------------------------------------------------


def write_crepe_split(raw_dir: Path, split: str, records: list[dict], columns=None) -> Path:
    """Write one ``crepe_<split>.parquet``; ``columns`` narrows the schema."""
    columns = columns or list(CREPE_PARQUET_SCHEMA.names)
    table = pa.table(
        {
            "id": pa.array([record.get("id") for record in records], pa.string()),
            "question": pa.array([record.get("question") for record in records], pa.string()),
            "labels": pa.array(
                [list(record.get("labels") or []) for record in records], pa.list_(pa.string())
            ),
            "presuppositions": pa.array(
                [list(record.get("presuppositions") or []) for record in records],
                pa.list_(pa.string()),
            ),
        },
        schema=CREPE_PARQUET_SCHEMA,
    )
    path = raw_dir / f"crepe_{split}.parquet"
    pq.write_table(table.select(columns), path)
    return path


def write_bundle(
    raw_dir: Path,
    crepe: dict[str, list[dict]] | None = None,
    kuq: list[dict] | None = None,
) -> Path:
    """Write a miniature raw bundle.

    ``crepe`` maps split -> records (an omitted split becomes an empty parquet so
    ``load_crepe`` never trips over a missing file).  ``kuq=None`` writes no
    ``knowns_unknowns.jsonl`` at all; ``kuq=[]`` writes an empty one.
    """
    crepe = CREPE_DEFAULT if crepe is None else crepe
    raw_dir.mkdir(parents=True, exist_ok=True)
    for split in ca.CREPE_SPLITS:
        write_crepe_split(raw_dir, split, crepe.get(split) or [])
    if kuq is not None:
        (raw_dir / ca.KUQ_FILE).write_text(
            "".join(json.dumps(record) + "\n" for record in kuq), encoding="utf-8"
        )
    return raw_dir


@pytest.fixture()
def bundle(tmp_path: Path) -> Path:
    return write_bundle(tmp_path, kuq=KUQ_FIXTURE)


def build(raw_dir: Path, **kwargs):
    return ca.build_rows(str(raw_dir), **kwargs)


# ---------------------------------------------------------------------------
# funnel and row assembly on the main bundle
# ---------------------------------------------------------------------------


class TestMainBundle:
    def test_funnel_stages_and_counts(self, bundle: Path):
        rows, funnel = build(bundle)
        assert list(funnel.items()) == list(EXPECTED_FUNNEL.items())
        assert len(rows) == 10

    def test_funnel_starts_at_raw_and_never_grows(self, bundle: Path):
        _, funnel = build(bundle)
        assert next(iter(funnel)) == "raw_rows"
        assert funnel["raw_rows"] == 12  # 6 CREPE + 6 KUQ lines
        values = list(funnel.values())
        assert all(later <= earlier for earlier, later in zip(values, values[1:], strict=False))

    def test_funnel_stage_names_are_the_documented_order(self, bundle: Path):
        _, funnel = build(bundle)
        assert list(funnel) == [
            "raw_rows",
            "after_question_text",
            "after_label_uniqueness",
            "after_certificate",
            "after_kuq_category_filter",
            "after_cross_label_conflict",
            "after_dedup",
            "after_quota",
            "after_limit",
        ]

    def test_row_order_is_group_interleaved_and_split_round_robin(self, bundle: Path):
        rows, _ = build(bundle)
        assert [row["extra_info"]["task_id"] for row in rows] == EXPECTED_TASK_IDS

    def test_task_ids_are_stable_and_derived_from_the_source_identity(self, bundle: Path):
        rows, _ = build(bundle)
        by_id = {row["extra_info"]["task_id"]: row for row in rows}
        assert "crepe:validation:2019_a-01077" in by_id
        assert "kuq:00002" in by_id  # KUQ line index, zero-padded
        assert len(by_id) == len(rows)
        # No uuid / random component: a rebuild reproduces every key exactly.
        again, _ = build(bundle)
        assert [row["extra_info"]["task_id"] for row in again] == list(by_id)

    def test_build_is_deterministic_and_byte_identical(self, bundle: Path, tmp_path: Path):
        first, funnel_a = build(bundle, limit=7)
        second, funnel_b = build(bundle, limit=7)
        assert funnel_a == funnel_b
        assert first == second
        schema.normalise_extra_info(first)
        schema.normalise_extra_info(second)
        path_a, path_b = tmp_path / "a.parquet", tmp_path / "b.parquet"
        schema.write_rows_parquet(first, str(path_a))
        schema.write_rows_parquet(second, str(path_b))
        assert path_a.read_bytes() == path_b.read_bytes()

    def test_breakdown_is_two_fifths_solvable(self, bundle: Path):
        rows, _ = build(bundle)
        assert Counter(row["extra_info"]["branch"] for row in rows) == {
            schema.BRANCH_SOLVABLE_JUDGE: 5,
            schema.BRANCH_UNSOLVABLE_BARE: 5,
        }
        assert Counter(row["extra_info"]["template"] for row in rows) == {ca.TEMPLATE: 10}
        assert Counter(row["data_source"] for row in rows) == {
            schema.SOURCE_CREPE: 5,
            schema.SOURCE_KUQ: 5,
        }


# ---------------------------------------------------------------------------
# per-branch payloads
# ---------------------------------------------------------------------------


class TestBranchPayloads:
    def test_solvable_rows_are_judgment_only(self, bundle: Path):
        rows, _ = build(bundle)
        judges = [r for r in rows if r["extra_info"]["solvable"]]
        assert len(judges) == 5
        for row in judges:
            payload = json.loads(row["reward_model"]["ground_truth"])
            assert payload["solvable"] is True
            assert payload["judgment_only"] is True
            assert payload["answer"] is None
            assert payload["correct_option_id"] is None
            assert payload["has_diagnosis_label"] is False
            assert row["extra_info"]["branch"] == schema.BRANCH_SOLVABLE_JUDGE
            assert row["extra_info"]["judgment_only"] is True
            assert row["extra_info"]["perturbation_type"] == ""
            assert row["extra_info"]["error_type"] == ""
            # the gold marker the reward scores +1 against (D14)
            assert "\\boxed{SOLVABLE}" in row["prompt"][0]["content"]

    def test_unsolvable_rows_carry_no_answer_and_no_diagnosis_label(self, bundle: Path):
        rows, _ = build(bundle)
        bares = [r for r in rows if not r["extra_info"]["solvable"]]
        assert len(bares) == 5
        for row in bares:
            payload = json.loads(row["reward_model"]["ground_truth"])
            assert payload["solvable"] is False
            assert payload["answer"] is None
            assert payload.get("judgment_only", False) is False
            assert payload["has_diagnosis_label"] is False
            assert row["extra_info"]["branch"] == schema.BRANCH_UNSOLVABLE_BARE
            assert row["extra_info"]["has_diagnosis_label"] is False
            assert row["extra_info"]["perturbation_type"] == ca.UNSOLVABLE_PERTURBATION
            assert row["extra_info"]["error_type"] == ca.ERROR_TYPE

    def test_perturbation_type_is_absent_to_the_reward_on_solvable_rows(self, bundle: Path):
        rows, _ = build(bundle)
        for row in rows:
            payload = json.loads(row["reward_model"]["ground_truth"])
            if payload["solvable"]:
                assert "perturbation_type" not in payload or not payload["perturbation_type"]
            else:
                assert payload["perturbation_type"] == ca.UNSOLVABLE_PERTURBATION

    def test_prompt_is_the_question_plus_the_verdict_wording(self, bundle: Path):
        rows, _ = build(bundle)
        by_id = {row["extra_info"]["task_id"]: row for row in rows}
        question = by_id["crepe:train:2018-09504"]["prompt"][0]["content"]
        assert question.startswith(CREPE_NORMAL["question"])
        assert "\\boxed{SOLVABLE}" in question
        assert "UNSOLVABLE" in question
        trailing = by_id["kuq:00005"]["prompt"][0]["content"]
        assert trailing.startswith(KUQ_WHITESPACE["question"].strip())
        assert not trailing.startswith(" ")
        assert ca.TEMPLATE == schema.TEMPLATE_B_JUDGE

    def test_every_row_passes_validate_row_and_round_trips(self, bundle: Path, tmp_path: Path):
        rows, _ = build(bundle)
        schema.normalise_extra_info(rows)
        schema.validate_rows(rows)
        for row in rows:
            assert schema.validate_row(row) == []
        out = tmp_path / "rows.parquet"
        schema.write_rows_parquet(rows, str(out))
        reloaded = schema.read_parquet_rows(str(out))
        assert len(reloaded) == len(rows)
        for row in reloaded:
            assert schema.validate_row(row) == []

    def test_extra_info_records_split_index_seed_and_difficulty(self, bundle: Path):
        rows, _ = build(bundle, seed=7)
        for index, row in enumerate(rows):
            info = row["extra_info"]
            assert info["index"] == index
            assert info["seed"] == 7
            assert isinstance(info["difficulty"], str) and info["difficulty"]
            assert info["split"] in ca.CREPE_SPLITS  # KUQ's native split is "train"
        by_id = {row["extra_info"]["task_id"]: row["extra_info"] for row in rows}
        assert by_id["crepe:train:2018-09504"]["difficulty"] == "normal"
        assert by_id["crepe:train:2018-03713"]["difficulty"] == "false_presupposition"
        assert by_id["kuq:00000"]["difficulty"] == "known"
        assert by_id["kuq:00002"]["difficulty"] == "false_assumption"
        assert by_id["kuq:00003"]["difficulty"] == "counterfactual"
        assert by_id["crepe:test:2019_b-11155"]["split"] == "test"
        assert by_id["kuq:00002"]["split"] == "train"

    def test_no_row_carries_an_options_block_or_role_words(self, bundle: Path):
        rows, _ = build(bundle)
        assert all(not row["extra_info"]["options"] for row in rows)
        assert all(not row["extra_info"]["role_words"] for row in rows)


# ---------------------------------------------------------------------------
# one test per drop branch
# ---------------------------------------------------------------------------


class TestLabelUniquenessDrops:
    def test_crepe_dual_labelled_row_is_dropped(self, bundle: Path):
        rows, funnel = build(bundle)
        assert funnel["after_label_uniqueness"] == funnel["raw_rows"] - 1
        assert not any(CREPE_DUAL["id"] in row["extra_info"]["task_id"] for row in rows)

    @pytest.mark.parametrize(
        "record",
        [
            {"id": "x-1", "question": "q?", "labels": ["not a label"], "presuppositions": []},
            {"id": "x-2", "question": "q?", "labels": [], "presuppositions": []},
            {"id": "", "question": "q?", "labels": ["normal"], "presuppositions": []},
            {"id": "x-3", "question": "q?", "labels": None, "presuppositions": []},
        ],
    )
    def test_crepe_rows_without_exactly_one_known_label_are_dropped(
        self, tmp_path: Path, record: dict
    ):
        bundle = write_bundle(tmp_path, crepe={"train": [record]}, kuq=[])
        rows, funnel = build(bundle)
        assert rows == []
        assert funnel["raw_rows"] == 1
        assert funnel["after_label_uniqueness"] == 0

    def test_crepe_row_without_an_id_is_dropped_even_with_a_good_label(self, tmp_path: Path):
        record = dict(CREPE_NORMAL, id="")
        bundle = write_bundle(tmp_path, crepe={"train": [record]}, kuq=[])
        rows, _ = build(bundle)
        assert rows == []


class TestCertificateDrops:
    def test_solvable_crepe_row_with_a_presupposition_is_dropped(self, tmp_path: Path):
        # synthetic: the source never records normal+presuppositions (0 rows)
        record = dict(CREPE_NORMAL, id="s-1", presuppositions=["Something is assumed."])
        bundle = write_bundle(tmp_path, crepe={"train": [record]}, kuq=[])
        rows, funnel = build(bundle)
        assert rows == []
        assert funnel["after_label_uniqueness"] == 1
        assert funnel["after_certificate"] == 0

    def test_unsolvable_crepe_row_without_a_presupposition_is_dropped(self, tmp_path: Path):
        # synthetic: every real FP row carries at least one presupposition
        record = dict(CREPE_FP_PARAPHRASE, id="s-2", presuppositions=[])
        bundle = write_bundle(tmp_path, crepe={"train": [record]}, kuq=[])
        rows, funnel = build(bundle)
        assert rows == []
        assert funnel["after_certificate"] == 0

    def test_kuq_unknown_row_from_a_non_crowd_source_is_dropped(self, tmp_path: Path):
        # synthetic: every real unknown row has source == "turk"
        record = dict(KUQ_UNKNOWN_FA, source="hotpotqa")
        bundle = write_bundle(tmp_path, crepe={}, kuq=[record])
        rows, funnel = build(bundle)
        assert rows == []
        assert funnel["after_label_uniqueness"] == 1
        assert funnel["after_certificate"] == 0

    def test_kuq_unknown_row_without_a_category_is_dropped(self, tmp_path: Path):
        # synthetic: unknown=True with the category key absent
        record = {k: v for k, v in KUQ_UNKNOWN_FA.items() if k != "category"}
        bundle = write_bundle(tmp_path, crepe={}, kuq=[record])
        rows, funnel = build(bundle)
        assert rows == []
        assert funnel["after_certificate"] == 0

    def test_kuq_known_row_carrying_a_category_is_dropped(self, tmp_path: Path):
        # synthetic: category is an absent key on every real known row
        record = dict(KUQ_KNOWN, category="false assumption")
        bundle = write_bundle(tmp_path, crepe={}, kuq=[record])
        rows, funnel = build(bundle)
        assert rows == []
        assert funnel["after_certificate"] == 0

    @pytest.mark.parametrize(
        "answer",
        [[], [""], ["  "], [None], "not a list", "abc", 7],
    )
    def test_kuq_known_row_without_a_usable_answer_is_dropped(self, tmp_path: Path, answer):
        # synthetic: every real known row has a non-empty string answer list
        record = dict(KUQ_KNOWN, answer=answer)
        bundle = write_bundle(tmp_path, crepe={}, kuq=[record])
        rows, funnel = build(bundle)
        assert rows == []
        assert funnel["after_certificate"] == 0


class TestQuestionTextDrops:
    @pytest.mark.parametrize("question", ["", "   ", "\n\t", None])
    def test_crepe_rows_without_question_text_are_dropped(self, tmp_path: Path, question):
        record = dict(CREPE_NORMAL, id="e-1", question=question)
        bundle = write_bundle(tmp_path, crepe={"train": [record]}, kuq=[])
        rows, funnel = build(bundle)
        assert rows == []
        assert funnel["raw_rows"] == 1
        assert funnel["after_question_text"] == 0

    def test_kuq_rows_without_question_text_are_dropped(self, tmp_path: Path):
        record = dict(KUQ_KNOWN, question="   ")
        bundle = write_bundle(tmp_path, crepe={}, kuq=[KUQ_KNOWN, record])
        rows, funnel = build(bundle)
        assert len(rows) == 1
        assert funnel["after_question_text"] == 1


class TestKuqLabelAndCategoryDrops:
    @pytest.mark.parametrize("unknown", ["true", "false", 1, 0, None])
    def test_kuq_rows_with_a_non_bool_unknown_flag_are_dropped(
        self, tmp_path: Path, unknown
    ):
        record = dict(KUQ_KNOWN, unknown=unknown)
        bundle = write_bundle(tmp_path, crepe={}, kuq=[record])
        rows, funnel = build(bundle)
        assert rows == []
        assert funnel["after_label_uniqueness"] == 0

    @pytest.mark.parametrize(
        "category", ["controversial", "ambiguous", "future unknown", "unsolved problem"]
    )
    def test_out_of_scope_unknown_categories_are_dropped(
        self, tmp_path: Path, category: str
    ):
        record = dict(KUQ_UNKNOWN_OTHER, category=category)
        bundle = write_bundle(tmp_path, crepe={}, kuq=[record])
        rows, funnel = build(bundle)
        assert rows == []
        assert funnel["after_label_uniqueness"] == 1
        assert funnel["after_certificate"] == 1
        assert funnel["after_kuq_category_filter"] == 0

    def test_the_two_in_scope_categories_are_admitted(self, tmp_path: Path):
        bundle = write_bundle(tmp_path, crepe={}, kuq=[KUQ_UNKNOWN_FA, KUQ_UNKNOWN_CF])
        rows, funnel = build(bundle)
        assert len(rows) == 2
        assert funnel["after_kuq_category_filter"] == 2
        assert {json.loads(r["reward_model"]["ground_truth"])["solvable"] for r in rows} == {False}

    def test_crepe_rows_are_untouched_by_the_category_filter(self, tmp_path: Path):
        # CREPE candidates carry category=None, which the filter admits.
        bundle = write_bundle(
            tmp_path, crepe={"train": [CREPE_NORMAL, CREPE_FP_PARAPHRASE]}, kuq=[]
        )
        rows, funnel = build(bundle)
        assert len(rows) == 2
        assert funnel["after_kuq_category_filter"] == 2


class TestCrossLabelConflicts:
    def test_a_question_under_both_labels_loses_every_row(self, tmp_path: Path):
        question = "What is the population of the city?"
        known = dict(KUQ_KNOWN, question=question)
        unknown = dict(KUQ_UNKNOWN_FA, question=question)
        bundle = write_bundle(tmp_path, crepe={}, kuq=[known, unknown])
        rows, funnel = build(bundle)
        assert rows == []
        assert funnel["after_certificate"] == 2
        assert funnel["after_cross_label_conflict"] == 0

    def test_a_cross_source_conflict_also_removes_both_rows(self, tmp_path: Path):
        question = "Is this premise true?"
        crepe = dict(CREPE_NORMAL, id="c-1", question=question)
        kuq = dict(KUQ_UNKNOWN_FA, question=question)
        bundle = write_bundle(tmp_path, crepe={"train": [crepe]}, kuq=[kuq])
        rows, funnel = build(bundle)
        assert rows == []
        assert funnel["after_cross_label_conflict"] == 0

    def test_a_question_under_one_label_survives(self, tmp_path: Path):
        bundle = write_bundle(tmp_path, crepe={}, kuq=[KUQ_KNOWN, dict(KUQ_KNOWN, answer=["x"])])
        rows, funnel = build(bundle)
        assert len(rows) == 1  # same question, same label -> dedup, not conflict
        assert funnel["after_cross_label_conflict"] == 2


class TestDedup:
    def test_a_repeated_question_keeps_the_first_by_order_key(self, tmp_path: Path):
        first = dict(KUQ_KNOWN, answer=["first"])
        second = dict(KUQ_KNOWN, answer=["second"])
        bundle = write_bundle(tmp_path, crepe={}, kuq=[first, second])
        rows, funnel = build(bundle)
        assert funnel["after_cross_label_conflict"] == 2
        assert funnel["after_dedup"] == 1
        assert len(rows) == 1
        assert rows[0]["extra_info"]["task_id"] == "kuq:00000"

    def test_dedup_ignores_case_and_whitespace(self, tmp_path: Path):
        first = dict(KUQ_KNOWN, answer=["a"])
        second = dict(KUQ_KNOWN, question="  " + first["question"].upper() + " ", answer=["b"])
        bundle = write_bundle(tmp_path, crepe={}, kuq=[first, second])
        rows, funnel = build(bundle)
        assert funnel["after_dedup"] == 1
        assert len(rows) == 1
        # The surviving row keeps the question verbatim (no normalisation applied).
        assert rows[0]["prompt"][0]["content"].startswith(first["question"])

    def test_a_surrounding_space_alone_does_not_change_the_prompt(self, tmp_path: Path):
        bundle = write_bundle(tmp_path, crepe={}, kuq=[KUQ_WHITESPACE])
        rows, _ = build(bundle)
        assert len(rows) == 1
        assert rows[0]["prompt"][0]["content"].startswith(KUQ_WHITESPACE["question"].strip())

    def test_distinct_questions_are_all_kept(self, tmp_path: Path):
        bundle = write_bundle(tmp_path, crepe={}, kuq=[KUQ_KNOWN, KUQ_KNOWN_MULTI])
        rows, funnel = build(bundle)
        assert funnel["after_dedup"] == 2
        assert len(rows) == 2


class TestNormaliseQuestion:
    @pytest.mark.parametrize(
        "left,right",
        [
            ("Hello  World", "hello world"),
            ("Hello World ", "HELLO   world"),
            ("\nHello\tWorld", "hello world"),
        ],
    )
    def test_collapses_case_and_whitespace(self, left: str, right: str):
        assert ca.normalise_question(left) == ca.normalise_question(right)

    def test_keeps_word_order_and_punctuation(self):
        assert ca.normalise_question("A b?") != ca.normalise_question("b? A")


# ---------------------------------------------------------------------------
# limit
# ---------------------------------------------------------------------------


class TestLimit:
    def test_limit_truncates_the_interleaved_order(self, bundle: Path):
        rows, funnel = build(bundle, limit=3)
        assert [row["extra_info"]["task_id"] for row in rows] == EXPECTED_TASK_IDS[:3]
        assert funnel["after_limit"] == 3
        # Both labels and both sources survive even a tiny limit.
        assert {row["extra_info"]["solvable"] for row in rows} == {True, False}

    def test_limit_zero_emits_nothing_but_keeps_the_funnel(self, bundle: Path):
        rows, funnel = build(bundle, limit=0)
        assert rows == []
        assert funnel["after_quota"] == 10
        assert funnel["after_limit"] == 0

    def test_a_limit_above_the_pool_is_harmless(self, bundle: Path):
        rows, funnel = build(bundle, limit=999)
        assert len(rows) == 10
        assert funnel["after_limit"] == 10

    def test_a_negative_limit_is_rejected(self, bundle: Path):
        with pytest.raises(ValueError, match="non-negative"):
            build(bundle, limit=-1)


# ---------------------------------------------------------------------------
# quotas, interleaving, sources
# ---------------------------------------------------------------------------


class TestQuotaMechanics:
    def test_sampling_is_seeded_by_the_group_index(self, bundle: Path, monkeypatch):
        monkeypatch.setitem(ca.QUOTAS, (schema.SOURCE_CREPE, schema.BRANCH_SOLVABLE_JUDGE), 1)
        rows, _ = build(bundle, seed=3)
        picked = [
            row["extra_info"]["task_id"]
            for row in rows
            if row["extra_info"]["branch"] == schema.BRANCH_SOLVABLE_JUDGE
            and row["data_source"] == schema.SOURCE_CREPE
        ]
        assert len(picked) == 1
        # Reconstruct exactly what build_rows should have done: the group members
        # are ordered by (source, split, key), then sampled with
        # _group_rng(seed, group_index), then split-round-robined.
        members = [
            ca._crepe_candidate("validation", CREPE_VALIDATION),
            ca._crepe_candidate("train", CREPE_NORMAL),
        ]
        members.sort(key=ca.Candidate.order_key)
        expected = ca._group_rng(3, 0).sample(members, 1)
        assert picked == [expected[0].task_id]
        assert ca._round_robin_by_split(members)[0].task_id == "crepe:train:2018-09504"

    def test_sampling_is_reproducible_under_a_tight_quota(self, bundle: Path, monkeypatch):
        monkeypatch.setitem(ca.QUOTAS, (schema.SOURCE_CREPE, schema.BRANCH_UNSOLVABLE_BARE), 1)
        first, _ = build(bundle, seed=11)
        second, _ = build(bundle, seed=11)
        assert [r["extra_info"]["task_id"] for r in first] == [
            r["extra_info"]["task_id"] for r in second
        ]

    def test_a_source_without_members_is_skipped(self, tmp_path: Path):
        bundle = write_bundle(tmp_path, crepe={}, kuq=[KUQ_KNOWN])
        rows, funnel = build(bundle)
        assert len(rows) == 1
        assert rows[0]["extra_info"]["branch"] == schema.BRANCH_SOLVABLE_JUDGE

    def test_a_group_missing_from_the_quota_table_raises(self, bundle: Path, monkeypatch):
        monkeypatch.setattr(ca, "QUOTAS", {})
        with pytest.raises(ValueError, match="no quota declared"):
            build(bundle)

    def test_interleave_round_robins_over_groups(self):
        groups = [["a1", "a2", "a3"], ["b1", "b2"]]
        assert ca._interleave(groups) == ["a1", "b1", "a2", "b2", "a3"]
        assert ca._interleave([]) == []

    def test_round_robin_by_split_keeps_every_split_at_any_prefix(self):
        members = [
            ca._crepe_candidate("train", dict(CREPE_NORMAL, id=f"t{i}")) for i in range(4)
        ] + [
            ca._crepe_candidate("validation", dict(CREPE_NORMAL, id=f"v{i}")) for i in range(2)
        ] + [ca._crepe_candidate("test", dict(CREPE_NORMAL, id="s0"))]
        ordered = ca._round_robin_by_split(members)
        assert [c.split for c in ordered[:3]] == ["test", "train", "validation"]
        # The property the fix exists for: a prefix of length k already covers k
        # distinct native splits, so no --limit can truncate a whole split away.
        for length in (1, 2, 3):
            assert len({c.split for c in ordered[:length]}) == length
        # Deterministic: the same members always give the same order.
        assert ca._round_robin_by_split(members) == ordered

    def test_group_rng_is_reproducible_and_seed_dependent(self):
        assert ca._group_rng(5, 2).random() == ca._group_rng(5, 2).random()
        assert ca._group_rng(5, 2).random() != ca._group_rng(6, 2).random()
        assert ca._group_rng(5, 2).random() != ca._group_rng(5, 3).random()


class TestSourceSelection:
    def test_crepe_only(self, bundle: Path):
        rows, funnel = build(bundle, sources=(schema.SOURCE_CREPE,))
        assert {row["data_source"] for row in rows} == {schema.SOURCE_CREPE}
        assert funnel["raw_rows"] == 6

    def test_kuq_only_needs_no_crepe_files(self, tmp_path: Path):
        bundle = write_bundle(tmp_path, crepe={}, kuq=[KUQ_KNOWN, KUQ_UNKNOWN_CF])
        rows, funnel = build(bundle, sources=(schema.SOURCE_KUQ,))
        assert {row["data_source"] for row in rows} == {schema.SOURCE_KUQ}
        assert funnel["raw_rows"] == 2

    def test_an_unknown_source_is_rejected(self, bundle: Path):
        with pytest.raises(ValueError, match="unknown source"):
            build(bundle, sources=("nope",))


# ---------------------------------------------------------------------------
# raw-loading failures (fail closed, never invent)
# ---------------------------------------------------------------------------


class TestRawLoading:
    def test_a_missing_crepe_split_raises(self, bundle: Path):
        (bundle / "crepe_validation.parquet").unlink()
        with pytest.raises(FileNotFoundError, match="validation"):
            build(bundle)

    def test_a_missing_kuq_file_raises(self, tmp_path: Path):
        bundle = write_bundle(tmp_path, crepe=CREPE_DEFAULT, kuq=None)
        with pytest.raises(FileNotFoundError, match="knowns_unknowns"):
            build(bundle)

    def test_an_empty_kuq_file_yields_only_crepe_rows(self, tmp_path: Path):
        bundle = write_bundle(tmp_path, crepe=CREPE_DEFAULT, kuq=[])
        rows, funnel = build(bundle)
        assert funnel["raw_rows"] == 6
        assert {row["data_source"] for row in rows} == {schema.SOURCE_CREPE}

    def test_a_missing_required_column_raises(self, tmp_path: Path):
        bundle = write_bundle(tmp_path, crepe={}, kuq=[KUQ_KNOWN])
        write_crepe_split(
            bundle,
            "train",
            [CREPE_NORMAL],
            columns=["id", "question", "labels"],
        )
        with pytest.raises(ValueError, match="presuppositions"):
            build(bundle)

    def test_a_bad_kuq_json_line_raises(self, tmp_path: Path):
        bundle = write_bundle(tmp_path, crepe={}, kuq=[KUQ_KNOWN])
        with open(bundle / ca.KUQ_FILE, "a", encoding="utf-8") as handle:
            handle.write("{not json}\n")
        with pytest.raises(json.JSONDecodeError):
            build(bundle)

    def test_blank_jsonl_lines_are_skipped(self, tmp_path: Path):
        bundle = write_bundle(tmp_path, crepe={}, kuq=[KUQ_KNOWN])
        with open(bundle / ca.KUQ_FILE, "a", encoding="utf-8") as handle:
            handle.write("\n")
        rows, funnel = build(bundle)
        assert funnel["raw_rows"] == 1
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------


class TestCandidate:
    def test_crepe_candidate_fields(self):
        candidate = ca._crepe_candidate("test", CREPE_FP_PARAPHRASE)
        assert candidate.source == schema.SOURCE_CREPE
        assert candidate.split == "test"
        assert candidate.task_id == "crepe:test:2018-00818"
        assert candidate.solvable is False
        assert candidate.branch == schema.BRANCH_UNSOLVABLE_BARE
        assert candidate.perturbation_type == ca.UNSOLVABLE_PERTURBATION
        assert candidate.error_type == ca.ERROR_TYPE
        assert candidate.difficulty == "false_presupposition"
        assert candidate.category is None
        assert candidate.label_ok and candidate.certificate_ok

    def test_crepe_solvable_candidate_has_no_perturbation(self):
        candidate = ca._crepe_candidate("train", CREPE_NORMAL)
        assert candidate.solvable is True
        assert candidate.branch == schema.BRANCH_SOLVABLE_JUDGE
        assert candidate.perturbation_type is None
        assert candidate.error_type == ""
        assert candidate.difficulty == "normal"

    def test_kuq_candidate_index_becomes_the_task_id(self):
        candidate = ca._kuq_candidate(41, KUQ_UNKNOWN_CF)
        assert candidate.task_id == "kuq:00041"
        assert candidate.split == "train"
        assert candidate.solvable is False
        assert candidate.category == "counterfactual"
        assert candidate.difficulty == "counterfactual"

    def test_kuq_known_candidate(self):
        candidate = ca._kuq_candidate(0, KUQ_KNOWN)
        assert candidate.solvable is True
        assert candidate.branch == schema.BRANCH_SOLVABLE_JUDGE
        assert candidate.difficulty == "known"
        assert candidate.category is None

    def test_order_key_sorts_by_source_split_then_key(self):
        keys = sorted(
            [
                ca._crepe_candidate("test", dict(CREPE_NORMAL, id="b")),
                ca._kuq_candidate(3, KUQ_KNOWN),
                ca._crepe_candidate("train", dict(CREPE_NORMAL, id="a")),
            ],
            key=ca.Candidate.order_key,
        )
        assert [c.task_id for c in keys] == ["crepe:test:b", "crepe:train:a", "kuq:00003"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_writes_a_validated_parquet_and_prints_the_funnel(
        self, bundle: Path, tmp_path: Path, monkeypatch, capsys
    ):
        out = tmp_path / "out.parquet"
        monkeypatch.setattr(
            sys,
            "argv",
            ["crepe_adapter.py", "--raw-dir", str(bundle), "--out", str(out), "--seed", "1"],
        )
        ca.main()
        printed = capsys.readouterr().out
        assert "funnel (rows remaining after each stage)" in printed
        for stage in EXPECTED_FUNNEL:
            assert stage in printed
        assert "rows by branch: solvable_judge=5, unsolvable_bare=5" in printed
        assert "rows by template: B_judge=10" in printed
        assert "rows with an options block: 0; with a diagnosis label: 0" in printed
        assert out.exists()
        rows = schema.read_parquet_rows(str(out))
        assert len(rows) == 10
        for row in rows:
            assert schema.validate_row(row) == []

    def test_main_respects_limit_and_sources(
        self, bundle: Path, tmp_path: Path, monkeypatch, capsys
    ):
        out = tmp_path / "limited.parquet"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "crepe_adapter.py",
                "--raw-dir",
                str(bundle),
                "--out",
                str(out),
                "--limit",
                "4",
                "--sources",
                "crepe",
            ],
        )
        ca.main()
        printed = capsys.readouterr().out
        assert "limit: 4" in printed
        assert "sources       : halluc_commonsense_crepe" in printed
        rows = schema.read_parquet_rows(str(out))
        assert len(rows) == 4
        assert {row["data_source"] for row in rows} == {schema.SOURCE_CREPE}

    def test_main_with_kuq_only(self, bundle: Path, tmp_path: Path, monkeypatch, capsys):
        out = tmp_path / "kuq.parquet"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "crepe_adapter.py",
                "--raw-dir",
                str(bundle),
                "--out",
                str(out),
                "--sources",
                "kuq",
            ],
        )
        ca.main()
        printed = capsys.readouterr().out
        assert "sources       : halluc_commonsense_kuq" in printed
        rows = schema.read_parquet_rows(str(out))
        assert {row["data_source"] for row in rows} == {schema.SOURCE_KUQ}
        assert len(rows) == 5

    def test_print_breakdown_handles_an_empty_list(self, capsys):
        ca._print_breakdown([])
        assert "no rows emitted" in capsys.readouterr().out

    def test_main_refuses_to_write_an_empty_parquet(self, tmp_path: Path, monkeypatch):
        bundle = write_bundle(
            tmp_path,
            crepe={"train": [CREPE_DUAL]},
            kuq=[{k: v for k, v in KUQ_UNKNOWN_OTHER.items()}],
        )
        out = tmp_path / "empty.parquet"
        monkeypatch.setattr(
            sys, "argv", ["crepe_adapter.py", "--raw-dir", str(bundle), "--out", str(out)]
        )
        with pytest.raises(SystemExit, match="no rows survived"):
            ca.main()
        assert not out.exists()

    def test_main_rejects_a_missing_bundle(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["crepe_adapter.py", "--raw-dir", str(tmp_path / "nope"), "--out", str(tmp_path / "o.parquet")],
        )
        with pytest.raises(FileNotFoundError):
            ca.main()


# ---------------------------------------------------------------------------
# the audit's own harness (verify_crepe.py)
# ---------------------------------------------------------------------------


class TestVerifyHarness:
    def test_nb_pipeline_finds_no_signal_under_random_labels(self):
        """A green 0.50-ish reading here is what makes a low L3 number meaningful."""
        rng = random.Random(0)
        alphabet = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
        texts = [
            " ".join(rng.choice(alphabet) for _ in range(12)) for _ in range(120)
        ]
        labels = np.array([index % 2 for index in range(len(texts))])
        shuffled = np.array(labels)
        random.Random(11).shuffle(shuffled)
        balanced, folds, _ = vc.nb_out_of_fold(texts, shuffled, min_support=1, seed=0)
        assert len(folds) == 5
        assert 0.40 <= balanced <= 0.60

    def test_nb_pipeline_recovers_a_planted_signal(self):
        """Class vocabularies are disjoint, so a working estimator must separate them."""
        rng = random.Random(0)
        pos = ["zulu", "yankee", "xray", "whisky"]
        neg = ["mike", "november", "oscar", "papa"]
        noise = ["omega", "sigma"]
        texts, labels = [], []
        for _ in range(200):
            texts.append(f"{rng.choice(pos)} {rng.choice(noise)}")
            labels.append(1)
            texts.append(f"{rng.choice(neg)} {rng.choice(noise)}")
            labels.append(0)
        balanced, _, _ = vc.nb_out_of_fold(texts, np.array(labels), min_support=1, seed=0)
        assert balanced > 0.90

    @pytest.mark.parametrize(
        "task_id,expected",
        [
            ("crepe:train:2018-09504", ("crepe", "train:2018-09504")),
            ("kuq:00041", ("kuq", "00041")),
        ],
    )
    def test_decode_task_id(self, task_id: str, expected):
        assert vc.decode_task_id(task_id) == expected

    @pytest.mark.parametrize("task_id", ["nope:1", "crepe:train", "kuq:1:2"])
    def test_decode_task_id_rejects_other_shapes(self, task_id: str):
        with pytest.raises(ValueError, match="unrecognised task_id"):
            vc.decode_task_id(task_id)

    def test_extract_question_removes_the_verdict_tail(self, bundle: Path):
        rows, _ = build(bundle, limit=1)
        content = rows[0]["prompt"][0]["content"]
        recovered = vc.extract_question(content)
        assert recovered == CREPE_NORMAL["question"]
        assert schema.render_prompt(
            recovered, ca.TEMPLATE, options=[], role_words=[]
        ) == content

    def test_extract_question_leaves_a_tail_free_prompt_alone(self):
        assert vc.extract_question("A plain question?") == "A plain question?"

    def test_stratified_sample_covers_every_group(self, bundle: Path):
        rows, _ = build(bundle)
        sample = vc.stratified_sample(rows, 6, random.Random(0))
        assert len(sample) == 6
        assert {row["data_source"] for row in sample} == {
            schema.SOURCE_CREPE,
            schema.SOURCE_KUQ,
        }
        assert {row["extra_info"]["solvable"] for row in sample} == {True, False}

    def test_stratified_sample_stops_at_the_pool_size(self, bundle: Path):
        rows, _ = build(bundle)
        assert len(vc.stratified_sample(rows, 999, random.Random(0))) == len(rows)

    def test_audit_records_failures_and_exit_code(self, capsys):
        audit = vc.Audit()
        audit.record("X1", True, "fine")
        audit.record("X2", False, "broken")
        audit.record("X3", True, "also fine", ["detail line"])
        printed = capsys.readouterr().out
        assert "PASS  X1   fine" in printed
        assert "FAIL  X2   broken" in printed
        assert "detail line" in printed
        assert audit.checks == 3
        assert audit.failures == 1
        assert audit.exit_code == 1

    def test_audit_exit_code_is_zero_when_nothing_fails(self):
        audit = vc.Audit()
        audit.record("X1", True, "fine")
        assert audit.exit_code == 0

    def test_full_audit_runs_every_check_on_a_built_artifact(self, bundle: Path, capsys):
        """End-to-end run of ``run_checks`` against a real adapter build.

        ``L2b`` is the one check that must fail here, and for a reason that has
        nothing to do with the adapter: it asserts the real bundle's per-split
        label-string hit counts (927 / 544 / 751), which a 10-row fixture cannot
        reproduce.  ``L3c`` is left unasserted because with 5 rows per class any
        out-of-fold reading it produces is meaningless.  Pinning the rest is the
        point: it shows the harness certifies a genuine adapter artifact without
        crashing and without being weakened.
        """
        rows, _ = build(bundle)
        audit = vc.run_checks(rows, str(bundle), sample_size=8, seed=0)
        printed = capsys.readouterr().out
        assert audit.checks == 7
        assert audit.exit_code == 1
        assert "FAIL  L2b" in printed
        for code in ("L0", "L1 ", "L1b", "L2 ", "L3a"):
            assert f"PASS  {code}" in printed
        assert "label certificate re-derived from raw for 8 rows" in printed
        assert "0 sampled false presuppositions are verbatim spans" in printed
