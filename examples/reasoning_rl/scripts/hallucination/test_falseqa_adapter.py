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
"""Tests for ``falseqa_adapter.py``.

The fixtures are small hand-built ``question,answer,label`` CSV files written to
``tmp_path``, so the suite is self-contained: it reads nothing outside this
repository and touches no downloaded data.  Each fixture is a *pair* -- the two
members the source stores at index ``k`` of its ``label=1`` and ``label=0``
blocks -- and the writer keeps the source's blocked layout (every fake member
first, then every real one), because a builder that paired adjacent file rows
would pass none of these tests.

Covers both emitted branches, every funnel stage, every drop reason, the
determinism and ``--limit`` contract, the CLI, and the certificate helpers
directly (the diff regions, the gold certificate, the L2 uniqueness gate, both
option miners, the tie-free re-roll, and the source-answer parse).  One test
re-derives the diff with ``verify_falseqa``'s own copy and one checks that the
audit's out-of-fold scorer reproduces ``verify_umwp.bow_nb_oof``, because both
are contracts the artifact's correctness rests on.  Three more pin the audit's own
coverage: the L1/L2 set must be every row unless a fast-mode cap is passed, the
D14 scaffold check must go red when one branch's wording diverges, and the contract
check must go red when a diagnosis row's D18/D13 metadata drifts.

The single drop reason not covered by a *fixture* is
``gold_not_verbatim_in_question``: a real region's fake-side text is a slice of
the presented question, so the guard cannot fire on source data.  It is covered
as a unit test on a hand-built region instead, and the funnel test asserts it is
present-and-zero rather than absent.
"""

from __future__ import annotations

import csv
import json
import pathlib
import random
import sys

import falseqa_adapter as adapter
import pytest
import schema
import verify_falseqa
import verify_umwp

# ---------------------------------------------------------------------------
# fixtures -- ``(question, answer, label)`` triples in the source's own columns
# ---------------------------------------------------------------------------

#: One rewritten token: the paired real question answers with ``eat``.
COLD_FAKE = ("What should a child drink when they have a cold?", "Because a cold is a virus.", "1")
COLD_REAL = ("What should a child eat when they have a cold?", "Because a cold is a virus.", "0")

#: A second clean pair, so branch order and ``--limit`` are observable.
BIRDS_FAKE = ("Why do fish fly south in the winter?", "Fish do not fly.", "1")
BIRDS_REAL = ("Why do birds fly south in the winter?", "Birds do not fly.", "0")

#: Two visible changed regions -- not a certificate.
MULTI_FAKE = ("a dog sat on the rug", "", "1")
MULTI_REAL = ("the cat sat on the mat", "", "0")

#: The two members are the same question, so the pair says nothing.
IDENT_FAKE = ("What colour is the sky?", "", "1")
IDENT_REAL = ("What colour is the sky?", "", "0")

#: The inserted fragment ("to the") carries no content word.
NOCONTENT_FAKE = ("birds fly to the south in the winter", "", "1")
NOCONTENT_REAL = ("birds fly south in the winter", "", "0")

#: The gold ("south") is not locatable by its first token -- it occurs twice.
NONUNIQUE_FAKE = ("the south south is warm", "", "1")
NONUNIQUE_REAL = ("the south is warm", "", "0")

#: The only admissible distractor ("gamma delta") does not occur in the paired
#: real question, whose punctuation is "gamma. delta" (the recon's uneven spacing).
PUNCT_FAKE = ("alpha beta big blue gamma delta", "", "1")
PUNCT_REAL = ("alpha beta xx yy gamma. delta", "", "0")

#: Too short to mine a three-option block on either side.
SHORT_FAKE = ("the dog", "", "1")
SHORT_REAL = ("the cat", "", "0")

#: Pairs, as ``_write_source`` takes them: ``(fake member, real member)``.  The
#: fake member is ``label=1``, the real one ``label=0``.
COLD = (COLD_FAKE, COLD_REAL)
BIRDS = (BIRDS_FAKE, BIRDS_REAL)
MULTI = (MULTI_FAKE, MULTI_REAL)
IDENT = (IDENT_FAKE, IDENT_REAL)
NOCONTENT = (NOCONTENT_FAKE, NOCONTENT_REAL)
NONUNIQUE = (NONUNIQUE_FAKE, NONUNIQUE_REAL)
PUNCT = (PUNCT_FAKE, PUNCT_REAL)
SHORT = (SHORT_FAKE, SHORT_REAL)

#: Every pair, for the drop-reason reachability sweep.
ALL_PAIRS = (COLD, BIRDS, MULTI, IDENT, NOCONTENT, NONUNIQUE, PUNCT, SHORT)

_FIELDS = ("question", "answer", "label")

MARKER_SOLVABLE = "\\boxed{SOLVABLE}"
MARKER_UNSOLVABLE = "\\boxed{UNSOLVABLE"


def _write_rows(raw_dir, rows, split: str = adapter.DEFAULT_SPLIT) -> str:
    """Write ``{split}.csv`` with ``rows`` verbatim, in the given order.

    Nothing is re-ordered or validated: the malformed-layout fixtures need to
    control the file exactly.
    """
    raw_dir = pathlib.Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / adapter.DATA_FILE_TEMPLATE.format(split=split)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_FIELDS)
        for row in rows:
            writer.writerow(row)
    return str(raw_dir)


def _write_source(raw_dir, *pairs, split: str = adapter.DEFAULT_SPLIT) -> str:
    """Write ``{split}.csv`` from ``(fake, real)`` pairs, label-blocked.

    The source stores every ``label=1`` row first, so pair ``k``'s fake member is
    at index ``k`` of the ``label=1`` block and its real member at index ``k`` of
    the ``label=0`` block.  Writing the members interleaved would make the fixture
    unrepresentative and let an adjacent-row pairing bug through.
    """
    rows = [pair[0] for pair in pairs] + [pair[1] for pair in pairs]
    return _write_rows(raw_dir, rows, split=split)


def _build(tmp_path, *pairs, split: str = adapter.DEFAULT_SPLIT, **kwargs):
    return adapter.build_rows(_write_source(tmp_path, *pairs, split=split), **kwargs)


def _by_id(rows: list[dict]) -> dict[str, dict]:
    return {row["extra_info"]["task_id"]: row for row in rows}


# ---------------------------------------------------------------------------
# the two emitted branches
# ---------------------------------------------------------------------------


def test_emits_both_branches_under_template_a(tmp_path):
    rows, _ = _build(tmp_path, COLD)
    assert len(rows) == 2
    assert {row["extra_info"]["branch"] for row in rows} == {adapter.BRANCH_DIAG, adapter.BRANCH_JUDGE}
    assert {row["extra_info"]["template"] for row in rows} == {schema.TEMPLATE_A}


def test_every_row_satisfies_the_schema_contract(tmp_path):
    rows, _ = _build(tmp_path, COLD, BIRDS)
    for row in rows:
        assert schema.validate_row(row) == []
    # the canonical key set is uniform and complete across rows
    key_sets = {tuple(sorted(row["extra_info"])) for row in rows}
    assert key_sets == {tuple(sorted(schema.EXTRA_INFO_KEYS))}
    # the same holds after the normalisation main() applies before writing
    schema.normalise_extra_info(rows)
    schema.validate_rows(rows)


def test_diag_row_points_at_the_rewritten_span(tmp_path):
    rows, _ = _build(tmp_path, COLD)
    diag = _by_id(rows)["falseqa-fake-train-0"]
    info = diag["extra_info"]
    gold = next(option for option in info["options"] if option["id"] == info["correct_option_id"])
    assert gold["text"] == "drink"
    assert info["paired_original_text"] == adapter.normalise_question(COLD_REAL[0])
    # the gold is the fragment the paired real question did *not* say
    assert gold["text"] not in info["paired_original_text"]
    assert gold["text"] in diag["prompt"][0]["content"]
    assert info["deleted_condition_text"] == "eat"
    assert info["perturbed_entity_text"] == "drink"
    assert len(info["options"]) == adapter.K_OPTIONS
    # equal token length is the D15 anti-shortcut constraint, and every distractor
    # must be text that survived the rewrite (the L2 uniqueness proof)
    assert len({len(schema.words(option["text"])) for option in info["options"]}) == 1
    for option in info["options"]:
        if option["id"] != info["correct_option_id"]:
            assert option["text"] in info["paired_original_text"]
    assert json.loads(diag["reward_model"]["ground_truth"]) == {
        "answer": None,
        "correct_option_id": info["correct_option_id"],
        "has_diagnosis_label": True,
        "perturbation_type": "contradictory_condition",
        "solvable": False,
    }
    assert info["solvable"] is False
    assert info["error_type"] == "false_premise_pointable"
    assert MARKER_UNSOLVABLE in diag["prompt"][0]["content"]


def test_judge_row_shape_and_audit_answer(tmp_path):
    rows, _ = _build(tmp_path, COLD)
    judge = _by_id(rows)["falseqa-real-train-0"]
    info = judge["extra_info"]
    payload = json.loads(judge["reward_model"]["ground_truth"])
    assert info["solvable"] is True
    assert payload["judgment_only"] is True
    assert payload["solvable"] is True
    assert payload["answer"] == COLD_REAL[1]  # the source's own text, audit only
    assert payload["correct_option_id"] is None
    assert payload["has_diagnosis_label"] is False
    assert payload["perturbation_type"] is None
    assert info["correct_option_id"] == ""
    assert info["error_type"] == ""  # no defect slot: this row carries no gold defect
    assert info["paired_original_text"] == adapter.normalise_question(COLD_FAKE[0])
    assert len(info["options"]) == adapter.K_OPTIONS
    assert info["index"] == 0
    assert info["split"] == "train"
    assert info["seed"] == 0
    assert info["difficulty"] == ""
    assert judge["prompt"][0]["content"].endswith(schema.render_options_block(info["options"]))


def test_both_branches_share_one_prompt_shape(tmp_path):
    """D14: the two labels must not be tellable apart by prompt shape alone."""
    rows, _ = _build(tmp_path, COLD)
    by_id = _by_id(rows)
    diag, judge = by_id["falseqa-fake-train-0"], by_id["falseqa-real-train-0"]
    assert diag["extra_info"]["template"] == judge["extra_info"]["template"] == schema.TEMPLATE_A
    assert len(diag["extra_info"]["options"]) == len(judge["extra_info"]["options"]) == adapter.K_OPTIONS

    def instruction(row: dict) -> tuple:
        """The template wording between the question and its options, plus the block size."""
        body = row["prompt"][0]["content"].partition("\n\n")[2]
        head, separator, options = body.partition("选项：")
        return head, separator, len(options.strip().splitlines())

    # the same wording and the same block size; only the option texts differ, and
    # those are mined from each row's own question
    assert len({instruction(row) for row in (diag, judge)}) == 1
    # and both prompts offer both verdicts, so the marker is not the discriminator
    for row in (diag, judge):
        prompt = row["prompt"][0]["content"]
        assert MARKER_SOLVABLE in prompt and MARKER_UNSOLVABLE in prompt


def test_option_blocks_are_mined_from_the_rows_own_question(tmp_path):
    rows, _ = _build(tmp_path, COLD, BIRDS)
    for row in rows:
        question = verify_falseqa.question_of(row)
        options = row["extra_info"]["options"]
        assert len(options) == adapter.K_OPTIONS
        assert len({option["text"] for option in options}) == adapter.K_OPTIONS
        assert len({len(schema.words(option["text"])) for option in options}) == 1
        for option in options:
            assert option["text"] in question, option


def test_diag_gold_is_absent_from_the_paired_question_and_every_distractor_present(tmp_path):
    """The L2 uniqueness proof, re-derived from the artifact rather than assumed."""
    rows, _ = _build(tmp_path, COLD, BIRDS)
    diag = [row for row in rows if row["extra_info"]["branch"] == adapter.BRANCH_DIAG]
    assert len(diag) == 2
    for row in diag:
        info = row["extra_info"]
        gold = verify_falseqa._option_text(row, info["correct_option_id"])
        assert gold is not None and gold not in info["paired_original_text"]
        for option in info["options"]:
            if option["text"] != gold:
                assert option["text"] in info["paired_original_text"]


def test_task_ids_are_side_split_and_index_derived(tmp_path):
    assert adapter._task_id(adapter.SIDE_FAKE, "valid", 3) == "falseqa-fake-valid-3"
    assert adapter._task_id(adapter.SIDE_REAL, "test", 0) == "falseqa-real-test-0"
    rows, _ = _build(tmp_path, COLD, BIRDS)
    task_ids = [row["extra_info"]["task_id"] for row in rows]
    assert len(set(task_ids)) == len(task_ids)
    assert all(task_id.startswith("falseqa-") for task_id in task_ids)
    # one pair -> one diag row and one judge row, both keyed on the pair index
    assert task_ids == [
        "falseqa-fake-train-0",
        "falseqa-real-train-0",
        "falseqa-fake-train-1",
        "falseqa-real-train-1",
    ]


# ---------------------------------------------------------------------------
# funnel
# ---------------------------------------------------------------------------


def test_funnel_stage_counts_for_a_clean_pair(tmp_path):
    rows, report = _build(tmp_path, COLD)
    assert list(report["funnel"]) == [
        "raw_rows",
        "after_malformed_drop",
        "pairs_available",
        "pairs_after_limit",
        "pairs_with_region_certificate",
        "rows_built",
    ]
    assert report["funnel"] == {
        "raw_rows": 2,
        "after_malformed_drop": 2,
        "pairs_available": 1,
        "pairs_after_limit": 1,
        "pairs_with_region_certificate": 1,
        "rows_built": 2,
    }
    assert len(rows) == report["funnel"]["rows_built"]
    # every drop reason is reported, even the ones that cannot fire here
    assert tuple(report["drops"]) == adapter.DROP_REASONS
    assert all(count == 0 for count in report["drops"].values())


def test_report_carries_the_breakdowns(tmp_path):
    _, report = _build(tmp_path, COLD, BIRDS)
    assert set(report) == {
        "raw_dir",
        "split",
        "limit",
        "seed",
        "funnel",
        "drops",
        "by_branch",
        "by_template",
        "by_solvable",
        "by_error_type",
    }
    assert report["by_branch"] == {adapter.BRANCH_DIAG: 2, adapter.BRANCH_JUDGE: 2}
    assert report["by_template"] == {schema.TEMPLATE_A: 4}
    assert report["by_solvable"] == {False: 2, True: 2}
    assert report["by_error_type"] == {"": 2, adapter.ERROR_TYPE: 2}
    assert report["split"] == "train" and report["limit"] is None and report["seed"] == 0


def test_a_pair_can_yield_the_judge_row_when_the_diag_row_is_dropped(tmp_path):
    _, report = _build(tmp_path, NOCONTENT)
    assert report["drops"]["gold_has_no_content_word"] == 1
    assert report["funnel"]["rows_built"] == 1
    assert report["by_branch"] == {adapter.BRANCH_JUDGE: 1}


# ---------------------------------------------------------------------------
# drop reasons -- one fixture each
# ---------------------------------------------------------------------------


def test_malformed_rows_are_dropped_not_crashed(tmp_path):
    raw_dir = _write_rows(
        tmp_path,
        [
            COLD_FAKE,
            ("", "an answer with no question", adapter.LABEL_FALSE),
            ("a question nobody labelled", "", "7"),
            COLD_REAL,
        ],
    )
    loaded = adapter.load_source(raw_dir)
    assert len(loaded) == 4
    assert [adapter._is_well_formed(row) for row in loaded] == [True, False, False, True]

    rows, report = adapter.build_rows(raw_dir)
    assert report["funnel"]["raw_rows"] == 4
    assert report["funnel"]["after_malformed_drop"] == 2
    assert report["drops"]["malformed_row"] == 2
    assert report["drops"]["unpaired_label_block"] == 0
    assert len(rows) == 2


def test_a_ragged_label_block_fails_closed(tmp_path):
    """The index alignment is the pairing certificate; a ragged block voids it."""
    raw_dir = _write_rows(tmp_path, [COLD_FAKE, BIRDS_FAKE, COLD_REAL])
    rows, report = adapter.build_rows(raw_dir)
    assert rows == []
    assert report["funnel"]["pairs_available"] == 0
    assert report["drops"]["unpaired_label_block"] == 1
    assert report["drops"]["multi_region_defect"] == 0  # nothing was ever paired


def test_identical_question_pair_is_dropped(tmp_path):
    rows, report = _build(tmp_path, IDENT)
    assert rows == []
    assert report["drops"]["identical_question_pair"] == 1
    assert report["funnel"]["pairs_with_region_certificate"] == 0


def test_multi_region_defect_is_dropped(tmp_path):
    rows, report = _build(tmp_path, MULTI)
    assert rows == []
    assert report["drops"]["multi_region_defect"] == 1
    # two regions means the pointer gold is ambiguous, not merely unparsed
    assert len(adapter.word_regions(MULTI_REAL[0], MULTI_FAKE[0])) == 2


def test_gold_without_a_content_word_drops_only_the_diag_row(tmp_path):
    """"to the" is a defect the model cannot be asked to point at."""
    rows, report = _build(tmp_path, NOCONTENT)
    assert report["drops"]["gold_has_no_content_word"] == 1
    assert [row["extra_info"]["branch"] for row in rows] == [adapter.BRANCH_JUDGE]
    assert report["drops"]["diag_pool_below_k"] == 0


def test_non_unique_first_token_drops_the_pair(tmp_path):
    """The gold "south" occurs twice, so the span is not locatable by its first word."""
    rows, report = _build(tmp_path, NONUNIQUE)
    assert rows == []
    assert report["drops"]["gold_first_token_not_unique"] == 1
    assert report["drops"]["judge_pool_below_k"] == 1


def test_pool_below_k_drops_both_rows(tmp_path):
    rows, report = _build(tmp_path, SHORT)
    assert rows == []
    assert report["drops"]["diag_pool_below_k"] == 1
    assert report["drops"]["judge_pool_below_k"] == 1
    assert report["funnel"]["pairs_with_region_certificate"] == 1  # the pair certified


def test_a_distractor_absent_from_the_original_drops_the_diag_row(tmp_path):
    """A second absent option would be an unearnable second answer."""
    rows, report = _build(tmp_path, PUNCT)
    assert report["drops"]["diag_not_unique_absent_option"] == 1
    assert [row["extra_info"]["branch"] for row in rows] == [adapter.BRANCH_JUDGE]


def test_every_drop_reason_is_reachable_or_documented(tmp_path):
    """Each drop reason fires on at least one fixture, except the documented guard."""
    names = ("cold", "birds", "multi", "ident", "nocontent", "nonunique", "punct", "short")
    fired: set[str] = set()
    for name, pair in zip(names, ALL_PAIRS, strict=True):
        _, report = _build(tmp_path / name, pair)
        fired.update(reason for reason, count in report["drops"].items() if count)

    # the two file-layout faults cannot come from a well-formed pair fixture
    layouts = {
        "malformed": [
            COLD_FAKE,
            ("", "an answer with no question", adapter.LABEL_FALSE),
            ("a question", "", "7"),
            COLD_REAL,
        ],
        "ragged": [COLD_FAKE, BIRDS_FAKE, COLD_REAL],
    }
    for name, rows in layouts.items():
        _, report = adapter.build_rows(_write_rows(tmp_path / name, rows))
        fired.update(reason for reason, count in report["drops"].items() if count)

    assert set(adapter.DROP_REASONS) - fired == {"gold_not_verbatim_in_question"}


# ---------------------------------------------------------------------------
# determinism and --limit
# ---------------------------------------------------------------------------


def test_same_seed_same_rows(tmp_path):
    first, _ = _build(tmp_path / "a", COLD, BIRDS, seed=0)
    second, _ = _build(tmp_path / "b", COLD, BIRDS, seed=0)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_seed_is_recorded_and_the_admitted_set_is_seed_independent(tmp_path):
    seeded, _ = _build(tmp_path / "a", COLD, BIRDS, seed=7)
    unseeded, _ = _build(tmp_path / "b", COLD, BIRDS, seed=0)
    assert {row["extra_info"]["seed"] for row in seeded} == {7}
    assert {row["extra_info"]["task_id"] for row in seeded} == {
        row["extra_info"]["task_id"] for row in unseeded
    }
    # the seed moves the option block (which span is A/B/C), not the row set
    assert json.dumps(seeded, sort_keys=True) != json.dumps(unseeded, sort_keys=True)


def test_limit_caps_pairs_not_rows(tmp_path):
    rows, report = _build(tmp_path, COLD, BIRDS, limit=1)
    assert report["funnel"]["pairs_available"] == 2
    assert report["funnel"]["pairs_after_limit"] == 1
    assert report["funnel"]["pairs_with_region_certificate"] == 1
    assert report["funnel"]["rows_built"] == 2
    assert [row["extra_info"]["index"] for row in rows] == [0, 0]
    assert {row["extra_info"]["task_id"] for row in rows} == {
        "falseqa-fake-train-0",
        "falseqa-real-train-0",
    }
    # a limit must truncate, never rewrite: every kept row is byte-identical
    full, _ = _build(tmp_path / "full", COLD, BIRDS)
    assert json.dumps(rows, sort_keys=True) == json.dumps(full[:2], sort_keys=True)


def test_limit_above_the_pair_count_is_a_no_op(tmp_path):
    rows, report = _build(tmp_path, COLD, limit=99)
    assert len(rows) == 2
    assert report["funnel"]["pairs_after_limit"] == 1


def test_limit_zero_emits_nothing(tmp_path):
    rows, report = _build(tmp_path, COLD, limit=0)
    assert rows == []
    assert report["funnel"]["pairs_after_limit"] == 0
    assert report["funnel"]["rows_built"] == 0


def test_build_rows_rejects_an_unknown_split(tmp_path):
    with pytest.raises(ValueError, match="unknown split"):
        adapter.build_rows(str(tmp_path), split="dev")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_main_writes_a_parquet_and_prints_the_funnel(tmp_path, monkeypatch, capsys):
    raw_dir = tmp_path / "raw"
    _write_source(raw_dir, COLD, BIRDS)
    out = tmp_path / "rows.parquet"
    report_path = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "falseqa_adapter.py",
            "--raw-dir",
            str(raw_dir),
            "--out",
            str(out),
            "--report",
            str(report_path),
            "--limit",
            "1",
        ],
    )
    assert adapter.main() == 0
    printed = capsys.readouterr().out
    assert "after_malformed_drop" in printed
    assert "pairs_available" in printed
    assert "drop reasons" in printed
    assert "per branch:" in printed
    assert "per template:" in printed
    assert "per solvable:" in printed
    assert "per error_type:" in printed

    written = schema.read_parquet_rows(str(out))
    assert len(written) == 2
    assert schema.validate_rows(written) is None
    assert {row["extra_info"]["branch"] for row in written} == {
        adapter.BRANCH_DIAG,
        adapter.BRANCH_JUDGE,
    }
    assert {row["extra_info"]["split"] for row in written} == {"train"}

    report = json.loads(pathlib.Path(report_path).read_text(encoding="utf-8"))
    assert report["funnel"]["pairs_after_limit"] == 1
    assert report["limit"] == 1
    assert tuple(report["drops"]) == adapter.DROP_REASONS
    assert report["raw_dir"] == str(raw_dir)


def test_main_builds_the_named_split(tmp_path, monkeypatch, capsys):
    raw_dir = tmp_path / "raw"
    _write_source(raw_dir, COLD, split="valid")
    out = tmp_path / "valid.parquet"
    monkeypatch.setattr(
        sys,
        "argv",
        ["falseqa_adapter.py", "--raw-dir", str(raw_dir), "--out", str(out), "--split", "valid"],
    )
    assert adapter.main() == 0
    assert "split=valid" in capsys.readouterr().out
    written = schema.read_parquet_rows(str(out))
    assert {row["extra_info"]["split"] for row in written} == {"valid"}
    assert {row["extra_info"]["task_id"] for row in written} == {
        "falseqa-fake-valid-0",
        "falseqa-real-valid-0",
    }


def test_main_fails_closed_on_an_empty_artifact(tmp_path, monkeypatch, capsys):
    raw_dir = tmp_path / "raw"
    _write_source(raw_dir, SHORT)
    out = tmp_path / "rows.parquet"
    monkeypatch.setattr(sys, "argv", ["falseqa_adapter.py", "--raw-dir", str(raw_dir), "--out", str(out)])
    assert adapter.main() == 1
    assert not out.exists()
    assert "nothing written" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# the audit's independent re-derivation
# ---------------------------------------------------------------------------


def test_the_audit_re_derives_the_same_region_as_the_adapter(tmp_path):
    """``verify_falseqa`` re-runs the diff on the artifact instead of importing it."""
    rows, _ = _build(tmp_path, COLD, BIRDS)
    for row in rows:
        real_q, fake_q = verify_falseqa._orient(row)
        audit = verify_falseqa.rederive_regions(real_q, fake_q)
        mine = adapter.word_regions(real_q, fake_q)
        assert len(audit) == len(mine) == 1
        assert audit[0]["fake_text"] == mine[0].b_text
        assert audit[0]["real_text"] == mine[0].a_text
        assert audit[0]["n_fake"] == mine[0].n_b
        assert audit[0]["n_real"] == mine[0].n_a


def test_the_audit_question_head_is_the_adapter_normalised_question(tmp_path):
    rows, _ = _build(tmp_path, COLD)
    diag, judge = _by_id(rows)["falseqa-fake-train-0"], _by_id(rows)["falseqa-real-train-0"]
    assert verify_falseqa.question_of(diag) == adapter.normalise_question(COLD_FAKE[0])
    assert verify_falseqa.question_of(judge) == adapter.normalise_question(COLD_REAL[0])


def test_the_audit_scorer_reproduces_the_house_estimator():
    """The H6 cross-check: the audit's own arithmetic must match ``bow_nb_oof``."""
    texts = [
        "cats drink milk every morning at home",
        "dogs bark loudly at the postman",
        "birds fly over the wide blue sea",
        "fish swim quickly in the cold river",
        "cats sleep all afternoon on the sofa",
        "dogs chase sticks across the green park",
        "birds build nests in the tall tree",
        "fish eat small insects near the rocks",
    ]
    labels = [1, 0, 1, 0, 1, 0, 1, 0]
    margin, truth = verify_falseqa._nb_oof_scores(texts, labels, folds=4, seed=0, min_support=1)
    house = verify_umwp.bow_nb_oof(texts, labels, folds=4, seed=0, min_support=1)
    assert verify_falseqa._argmax_accuracy(margin, truth) == pytest.approx(house)


def test_the_audit_scorer_groups_folds_by_pair():
    """Grouping must place both members of a pair in the same fold."""
    texts = ["cats drink milk", "cats drink milk and cream"] * 4
    labels = [0, 1] * 4
    groups = [(0, 0), (0, 0), (1, 0), (1, 0), (2, 0), (2, 0), (3, 0), (3, 0)]
    margin, truth = verify_falseqa._nb_oof_scores(
        texts, labels, folds=2, seed=0, min_support=1, groups=groups
    )
    assert margin.shape == truth.shape == (8,)


def test_the_audit_audits_every_row_unless_explicitly_capped():
    """L1/L2 re-derive a per-row certificate, so a sample is not a certificate.

    An audit that reads 200 of 700 diagnosis rows passes an artifact whose gold was
    flipped on one of the other 500, so ``per_branch <= 0`` -- the CLI default --
    must cover the whole artifact.  A positive cap stays available as a fast smoke
    mode and honours the ``L1_MIN_SAMPLE`` floor.
    """
    rows = [{"extra_info": {"branch": "b", "task_id": f"t{index}"}} for index in range(60)]
    every = verify_falseqa.sample_by_branch(rows, per_branch=0, seed=0)
    # the returned rows are task-id sorted, so compare to the full set
    assert sorted(row["extra_info"]["task_id"] for row in every["b"]) == sorted(
        row["extra_info"]["task_id"] for row in rows
    )
    assert len(verify_falseqa.sample_by_branch(rows, per_branch=-1, seed=0)["b"]) == 60
    capped = verify_falseqa.sample_by_branch(rows, per_branch=10, seed=0)
    assert len(capped["b"]) == verify_falseqa.L1_MIN_SAMPLE


def test_the_audit_scaffold_check_is_a_gate_on_a_branch_divergence(tmp_path):
    """D14: if one branch's *wording* differs, the model reads the verdict off it."""
    rows, _ = _build(tmp_path, COLD)
    clean = verify_umwp.Reporter()
    verify_falseqa.check_template_isomorphism(rows, clean)
    assert clean.failed == []

    diag = next(row for row in rows if row["extra_info"]["branch"] == adapter.BRANCH_DIAG)
    prompt = diag["prompt"][0]["content"]
    # the diagnosis prompt stops offering the SOLVABLE verdict ...
    diag["prompt"][0]["content"] = prompt.replace(MARKER_SOLVABLE, "\\boxed{YES}", 1)
    reporter = verify_umwp.Reporter()
    verify_falseqa.check_template_isomorphism(rows, reporter)
    assert len(reporter.failed) == 1 and "template isomorphism" in reporter.failed[0]

    # ... or renames the option block both labels are supposed to share
    diag["prompt"][0]["content"] = prompt.replace("选项：", "Options:", 1)
    reporter = verify_umwp.Reporter()
    verify_falseqa.check_template_isomorphism(rows, reporter)
    assert len(reporter.failed) == 1 and "template isomorphism" in reporter.failed[0]


@pytest.mark.parametrize("hostile", ["../../../../etc/passwd", "/etc/passwd", "train.csv"])
def test_the_audit_gates_an_artifact_chosen_split(tmp_path, hostile):
    """``split`` is read out of the artifact and used to build a file path.

    The audit's whole job is to distrust the artifact, so a value naming a CSV the
    source does not have is a boundary violation: unchecked, ``os.path.join`` would
    read that file as CSV and audit it as if it were the source.
    """
    rows, _ = _build(tmp_path, COLD)
    rows[0]["extra_info"]["split"] = hostile
    reporter = verify_umwp.Reporter()
    assert verify_falseqa.check_split_vocabulary(rows, reporter) is False
    assert len(reporter.failed) == 1 and "extra_info.split" in reporter.failed[0]


def test_load_source_split_refuses_a_split_the_source_does_not_have(tmp_path):
    with pytest.raises(ValueError, match="split must be one of"):
        verify_falseqa.load_source_split(str(tmp_path), "../../etc/passwd")


def test_the_audit_contract_check_catches_defect_metadata_drift(tmp_path):
    """The D18/D13 keys a diagnosis row carries are part of the contract."""
    rows, _ = _build(tmp_path, COLD)
    clean = verify_umwp.Reporter()
    verify_falseqa.check_contract(rows, clean)
    assert clean.failed == []

    diag = next(row for row in rows if row["extra_info"]["branch"] == adapter.BRANCH_DIAG)
    diag["extra_info"]["error_type"] = "false_premise_unpointable"
    diag["extra_info"]["perturbation_type"] = "unrelated_entity"
    reporter = verify_umwp.Reporter()
    verify_falseqa.check_contract(rows, reporter)
    assert reporter.failed == ["branch invariants (template A / verdicts / options / pairing)"]


# ---------------------------------------------------------------------------
# unit tests for the helpers
# ---------------------------------------------------------------------------


def test_normalise_question_collapses_whitespace():
    assert adapter.normalise_question("  a  b\n c \n\n d  ") == "a b c d"
    assert adapter.normalise_question(None) == ""
    assert adapter.normalise_question("already clean") == "already clean"


def test_pair_blocks_undoes_the_blocked_layout(tmp_path):
    raw_dir = _write_source(tmp_path, COLD, BIRDS)
    loaded = adapter.load_source(raw_dir)
    assert [row["label"] for row in loaded] == ["1", "1", "0", "0"]
    fake, real = adapter._pair_blocks(loaded)
    assert [row["question"] for row in fake] == [COLD_FAKE[0], BIRDS_FAKE[0]]
    assert [row["question"] for row in real] == [COLD_REAL[0], BIRDS_REAL[0]]


def test_word_regions_reports_only_the_visible_defect():
    insert = adapter.word_regions("alpha beta", "alpha gamma beta")
    assert [(r.tag, r.a_text, r.b_text, r.n_a, r.n_b) for r in insert] == [("insert", "", "gamma", 0, 1)]

    replace = adapter.word_regions("alpha beta gamma", "alpha delta gamma")
    assert [(r.tag, r.a_text, r.b_text) for r in replace] == [("replace", "beta", "delta")]

    # a two-sided change is one region, not a delete plus an insert
    merged = adapter.word_regions("alpha beta gamma delta", "alpha epsilon delta")
    assert len(merged) == 1
    assert merged[0].a_text == "beta gamma" and merged[0].b_text == "epsilon"
    # ... but regions separated by an unchanged run stay separate
    assert len(adapter.word_regions("alpha beta gamma delta", "alpha x gamma y")) == 2

    assert adapter.word_regions("same text", "same text") == []


def test_a_pure_deletion_is_invisible_but_still_recorded():
    """The presented question lost a word; there is nothing to point at."""
    real, fake = "the cat sat on the warm mat", "the cat sat on the mat"
    assert adapter.word_regions(real, fake) == []
    assert adapter.defect_texts(real, fake) == ("warm", "")

    # a deletion and a visible replace are two opcodes but one recorded defect
    real, fake = "the cat sat on the warm mat today", "the cat sat on the big mat"
    regions = adapter.word_regions(real, fake)
    assert [(r.tag, r.a_text, r.b_text) for r in regions] == [("replace", "warm", "big")]
    assert adapter.defect_texts(real, fake) == ("warm today", "big")


def test_merge_joins_adjacent_runs_and_drops_invisible_ones():
    """A hand-built opcode list must not turn one defect into two regions."""
    raw = [
        ("equal", 0, 2, 0, 2),
        ("delete", 2, 4, 2, 2),
        ("insert", 4, 4, 2, 5),
        ("equal", 4, 5, 5, 6),
    ]
    # the delete and the insert are contiguous on both sides, so they are one run
    assert adapter._merge(raw, require_visible=False) == [("delete", 2, 4, 2, 5)]
    # with the fake-side-empty delete dropped first, only the insert is left
    assert adapter._merge(raw, require_visible=True) == [("insert", 4, 4, 2, 5)]
    # runs that are not contiguous on both sides stay separate
    split = [("delete", 0, 1, 0, 0), ("insert", 2, 2, 0, 1)]
    assert adapter._merge(split, require_visible=False) == split
    assert adapter._merge([("equal", 0, 1, 0, 1)], require_visible=False) == []


def test_certify_gold_accepts_a_single_visible_region():
    real, fake = "the cat sat on the mat", "the cat sat on the big mat"
    assert adapter.certify_gold(real, fake, adapter.word_regions(real, fake)) == "big"


def test_certify_gold_rejections():
    # no region at all, or more than one
    assert adapter.certify_gold("a b", "a b", []) is None
    two = adapter.word_regions(MULTI_REAL[0], MULTI_FAKE[0])
    assert len(two) == 2
    assert adapter.certify_gold(MULTI_REAL[0], MULTI_FAKE[0], two) is None

    # an empty fake-side region
    empty = adapter._Region(tag="insert", a_text="", b_text="", a_start=0, b_start=0, n_a=0, n_b=0)
    assert adapter.certify_gold("a b", "a c", [empty]) is None

    # a region whose text is not in the question (defensive: a real region's text
    # is a slice of the question by construction, so this cannot fire on data)
    ghost = adapter._Region(tag="insert", a_text="", b_text="zebra", a_start=2, b_start=2, n_a=0, n_b=1)
    assert adapter.certify_gold("a b", "a c", [ghost]) is None

    # a stop-word-only gold makes an option block worse than no block
    real, fake = "birds fly south in the winter", "birds fly to the south in the winter"
    assert adapter.word_regions(real, fake)[0].b_text == "to the"
    assert adapter.certify_gold(real, fake, adapter.word_regions(real, fake)) is None

    # the gold's first token must locate the span, and here "south" occurs twice
    real, fake = "the south is warm", "the south south is warm"
    assert adapter.certify_gold(real, fake, adapter.word_regions(real, fake)) is None


def test_unique_absent_option():
    real = "alpha beta xx yy gamma. delta"
    assert adapter.unique_absent_option(real, ["alpha beta", "gamma delta", "big blue"], "big blue") is False
    assert (
        adapter.unique_absent_option(
            "alpha beta big blue gamma delta", ["alpha beta", "gamma delta", "big blue"], "big blue"
        )
        is False
    )
    # a distractor absent from the original is a second admissible answer
    assert adapter.unique_absent_option(real, ["alpha beta", "beta xx", "big blue"], "big blue") is True


def test_mine_anchor_option_spans_mines_same_length_spans_of_the_question():
    question = "alpha beta xx yy gamma. delta"
    texts = adapter.mine_anchor_option_spans(question, 2, k=3, rng=random.Random(0))
    assert texts is not None
    assert len(texts) == adapter.K_OPTIONS
    assert len(set(texts)) == adapter.K_OPTIONS
    assert len({len(schema.words(text)) for text in texts}) == 1
    assert all(text in question for text in texts)


def test_mine_anchor_option_spans_fails_closed():
    assert adapter.mine_anchor_option_spans("the cat", 1, k=3, rng=random.Random(0)) is None
    assert adapter.mine_anchor_option_spans("alpha beta", 0, k=3, rng=random.Random(0)) is None


def test_mine_anchor_option_spans_enforces_the_character_band():
    """The band, not the pool, is what makes a far-off candidate unusable."""
    assert adapter.mine_anchor_option_spans("cat dog fox", 1, k=3, rng=random.Random(0)) is not None
    # "internationalization" is 20 characters against a 3-character anchor, so the
    # two 3-character candidates that remain are one short of the block
    assert adapter.mine_anchor_option_spans("cat internationalization dog", 1, k=3, rng=random.Random(0)) is None


def test_mine_distinct_block_prefers_the_first_tie_free_block():
    # "tie-free" means the character lengths are all different, not all the same
    blocks = {
        "row:0": ["a", "bb", "ccc"],
        "row:1": ["aa", "bb", "cc"],
    }
    assert adapter.mine_distinct_block(lambda seed: blocks[seed], "row", 3) == ["a", "bb", "ccc"]


def test_mine_distinct_block_keeps_working_when_no_block_is_tie_free():
    """A row whose question offers no tie-free block keeps the tie, not a hole."""
    assert adapter.mine_distinct_block(lambda seed: ["ab", "cd", "ef"], "row", 3) == ["ab", "cd", "ef"]


def test_mine_distinct_block_reports_an_insufficient_pool_as_none():
    assert adapter.mine_distinct_block(lambda seed: None, "row", 3) is None


def test_mine_distinct_block_is_bounded_and_deterministic():
    seen: list[str] = []

    def mine(seed: str) -> list[str]:
        seen.append(seed)
        return ["ab", "cd", "ef"]

    assert adapter.mine_distinct_block(mine, "row", 3) == ["ab", "cd", "ef"]
    assert seen == [f"row:{attempt}" for attempt in range(adapter.TIE_RETRIES)]
    seen.clear()
    adapter.mine_distinct_block(mine, "row", 3)
    assert seen == [f"row:{attempt}" for attempt in range(adapter.TIE_RETRIES)]


def test_mine_distinct_block_returns_early_on_a_tie_free_block():
    seen: list[str] = []

    def mine(seed: str) -> list[str]:
        seen.append(seed)
        return ["ab", "cd", "ef"] if len(seen) < 3 else ["a", "bb", "ccc"]

    assert adapter.mine_distinct_block(mine, "row", 3) == ["a", "bb", "ccc"]
    assert seen == ["row:0", "row:1", "row:2"]


def test_diag_option_blocks_are_tie_free_when_the_question_offers_one(tmp_path):
    """The tie-free re-roll removes the first-index tie-break's free wins.

    Where the question offers no tie-free block, the row keeps its tie instead of
    being dropped -- that fallback is the documented bound on the re-roll, and the
    L4 audit is what measures whether the residual tie is still inside budget.
    """
    rows, _ = _build(tmp_path, COLD, BIRDS)
    lengths = {
        row["extra_info"]["task_id"]: {
            len(option["text"]) for option in row["extra_info"]["options"]
        }
        for row in rows
        if row["extra_info"]["branch"] == adapter.BRANCH_DIAG
    }
    assert set(lengths) == {"falseqa-fake-train-0", "falseqa-fake-train-1"}
    # BIRDS' question offers three distinct character lengths at the gold's token
    # count, so the block is tie-free ...
    assert len(lengths["falseqa-fake-train-1"]) == adapter.K_OPTIONS
    # ... COLD's offers two, so the tie is kept rather than the row dropped
    assert len(lengths["falseqa-fake-train-0"]) < adapter.K_OPTIONS


def test_audit_answer_parses_the_test_splits_list_repr():
    sentence = "Because cats are larger than mice."
    assert adapter.audit_answer({"answer": sentence}) == sentence
    assert adapter.audit_answer({"answer": "  padded  "}) == "padded"
    assert adapter.audit_answer({"answer": "['first take', 'second take', 'third take']"}) == "first take"
    assert adapter.audit_answer({"answer": '["a", "b"]'}) == "a"
    # an unparseable bracket is returned as-is, and an empty list is left alone
    assert adapter.audit_answer({"answer": "[not a literal"}) == "[not a literal"
    assert adapter.audit_answer({"answer": "[]"}) == "[]"
    assert adapter.audit_answer({"answer": ""}) == ""
    assert adapter.audit_answer({"answer": None}) == ""
    assert adapter.audit_answer({}) == ""


def test_is_well_formed_rejects_broken_rows():
    good = {"question": "why is the sky blue?", "answer": "rayleigh scattering", "label": "1"}
    assert adapter._is_well_formed(good) is True
    assert adapter._is_well_formed("not a dict") is False
    assert adapter._is_well_formed({**good, "question": "   "}) is False
    assert adapter._is_well_formed({**good, "question": None}) is False
    assert adapter._is_well_formed({**good, "label": "2"}) is False
    assert adapter._is_well_formed({key: value for key, value in good.items() if key != "label"}) is False


def test_breakdown_counts_and_sorts():
    rows = [
        {"extra_info": {"branch": "b"}},
        {"extra_info": {"branch": "a"}},
        {"extra_info": {"branch": "a"}},
    ]
    assert adapter._breakdown(rows, lambda row: row["extra_info"]["branch"]) == {"a": 2, "b": 1}


def test_source_constants_match_the_shared_schema():
    assert adapter.DATA_SOURCE == schema.SOURCE_FALSEQA
    assert schema.SOURCES[adapter.DATA_SOURCE] == ("commonsense", "FalseQA")
    assert adapter.BRANCH_DIAG == schema.BRANCH_UNSOLVABLE_DIAG
    assert adapter.BRANCH_JUDGE == schema.BRANCH_SOLVABLE_JUDGE
    assert adapter.ERROR_TYPE in {"false_premise_pointable", "false_premise_unpointable"}
    assert adapter.PERTURBATION_TYPE in schema.PERTURBATION_TYPES
    assert adapter.SPLITS == ("train", "valid", "test")
    assert adapter.DATA_FILE_TEMPLATE.format(split="valid") == "valid.csv"
