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
"""Tests for ``falseqa_adapter.py`` (D21 replacement pairs + D27 two-layer side).

The fixtures are small hand-built ``question,answer,label`` CSV files written to
``tmp_path``, so the suite is self-contained: it reads nothing outside this
repository and touches no downloaded data.  Each fixture is a *pair* -- the two
members the source stores at index ``k`` of its ``label=1`` and ``label=0``
blocks -- and the writer keeps the source's blocked layout (every fake member
first, then every real one), because a builder that paired adjacent file rows
would pass none of these tests.

Every build also writes :data:`VOCAB`, a **carrier pair** that is deliberately
multi-region: it earns no certificate, so it contributes no rows, and its only job
is to put out-of-passage vocabulary into the item bank.  Without it a two-question
fixture has no word outside the passage at all (the paired question is a rewrite
of the presented one, so it adds almost nothing), and every diagnosis row would
legitimately drop as ``pair_distractors_below_k``.

Covers both emitted branches, the replacement-pair structure (D21) and the
two-layer answerable contract (D27), every funnel stage and drop reason, the
pair-identity contract (D27 ``pair_id``), determinism and ``--limit``, the CLI,
the reward cells of the design doc section 9 matrix for both branches (end to end
through the frozen dispatcher), and the audit's own re-derivations.  A handful of
tests pin the audit's gates by mutating a built row and asserting the
corresponding check goes red -- an audit that cannot fail is not an audit.
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

#: One rewritten token: the paired real question answers with ``eat``, so the gold
#: replacement pair is ``drink -> eat``.
COLD_FAKE = ("What should a child drink when they have a cold?", "Because a cold is a virus.", "1")
COLD_REAL = ("What should a child eat when they have a cold?", "Because a child needs food.", "0")

#: A second clean pair, so branch order and ``--limit`` are observable.
BIRDS_FAKE = ("Why do fish fly south in the winter?", "Fish do not fly.", "1")
BIRDS_REAL = ("Why do birds fly south in the winter?", "Birds do not fly.", "0")

#: The vocabulary carrier: two visible changed regions, so the pair earns no
#: certificate and yields no rows -- it exists only to stock the item bank.
VOCAB_FAKE = ("Sharks hunt fish near coral reefs at night", "", "1")
VOCAB_REAL = ("Wolves hunt deer near alpine forests at dawn", "", "0")

#: Two visible changed regions -- not a certificate (the carrier is the one every
#: build carries, so this fixture is used for the multi-region drop reason).
MULTI_FAKE = ("a dog sat on the rug", "", "1")
MULTI_REAL = ("the cat sat on the mat", "", "0")

#: The two members are the same question, so the pair says nothing.
IDENT_FAKE = ("What colour is the sky?", "", "1")
IDENT_REAL = ("What colour is the sky?", "", "0")

#: The inserted fragment ("to the") carries no content word.
NOCONTENT_FAKE = ("birds fly to the south in the winter", "", "1")
NOCONTENT_REAL = ("birds fly south in the winter", "Because birds fly.", "0")

#: The fake fragment ("south") is not locatable by its first token -- it occurs twice.
NONUNIQUE_FAKE = ("the south south is warm", "", "1")
NONUNIQUE_REAL = ("the south is warm", "Because the south is warm.", "0")

#: A pure insertion on the fake side ("warm"), so the gold pair's right item -- the
#: fragment that would repair the premise -- does not exist.
RIGHT_EMPTY_FAKE = ("the cat sat on the warm mat", "", "1")
RIGHT_EMPTY_REAL = ("the cat sat on the mat", "Because the cat sat.", "0")

#: The repairing fragment ("dog") already occurs in the presented question (inside
#: "dogs"), so the gold right item is not out-of-passage.
RIGHT_IN_FAKE = ("how many dogs does the small cat need", "", "1")
RIGHT_IN_REAL = ("how many dogs does the small dog need", "The small dog needs them.", "0")

#: The gold right item ("walking") is the fixture's only ``-ing`` item, so no
#: two qualifying distractors of its type exist: the diagnosis row must be dropped,
#: never padded with an off-type option.
NO_POOL_FAKE = ("Why is the dog running?", "", "1")
NO_POOL_REAL = ("Why is the dog walking?", "Because dogs walk.", "0")

#: The source's own answerable answer is empty, so the two-layer branch has no gold.
EMPTY_ANSWER_FAKE = ("Where do frogs sleep at night?", "", "1")
EMPTY_ANSWER_REAL = ("Where do stones sleep at night?", "", "0")

#: The real question carries no content word, so no in-passage left item can be
#: mined for the answerable side's placeholder pair.
NO_PLACEHOLDER_FAKE = ("the of cat", "", "1")
NO_PLACEHOLDER_REAL = ("the of an", "a sentence", "0")

#: Pairs, as ``_write_source`` takes them: ``(fake member, real member)``.  The
#: fake member is ``label=1``, the real one ``label=0``.
COLD = (COLD_FAKE, COLD_REAL)
BIRDS = (BIRDS_FAKE, BIRDS_REAL)
VOCAB = (VOCAB_FAKE, VOCAB_REAL)
MULTI = (MULTI_FAKE, MULTI_REAL)
IDENT = (IDENT_FAKE, IDENT_REAL)
NOCONTENT = (NOCONTENT_FAKE, NOCONTENT_REAL)
NONUNIQUE = (NONUNIQUE_FAKE, NONUNIQUE_REAL)
RIGHT_EMPTY = (RIGHT_EMPTY_FAKE, RIGHT_EMPTY_REAL)
RIGHT_IN = (RIGHT_IN_FAKE, RIGHT_IN_REAL)
NO_POOL = (NO_POOL_FAKE, NO_POOL_REAL)
EMPTY_ANSWER = (EMPTY_ANSWER_FAKE, EMPTY_ANSWER_REAL)
NO_PLACEHOLDER = (NO_PLACEHOLDER_FAKE, NO_PLACEHOLDER_REAL)

#: Every drop-reason fixture, for the reachability sweep.
ALL_PAIRS = (
    COLD,
    BIRDS,
    MULTI,
    IDENT,
    NOCONTENT,
    NONUNIQUE,
    RIGHT_EMPTY,
    RIGHT_IN,
    NO_POOL,
    EMPTY_ANSWER,
    NO_PLACEHOLDER,
)

_FIELDS = ("question", "answer", "label")

PAIR_ARROW = " -> "
MARKER_ANSWER = "\\boxed{<答案>}"
MARKER_UNSOLVABLE = "\\boxed{UNSOLVABLE"
MARKER_UNSOLVABLE_ID = "\\boxed{UNSOLVABLE: <选项ID>}"

#: The frozen reward dispatcher, imported end to end so the section 9 reward cells
#: are pinned against the real scorer rather than a restatement of it.
_REWARD_DIR = pathlib.Path(__file__).resolve().parents[2] / "reward"
if str(_REWARD_DIR) not in sys.path:
    sys.path.insert(0, str(_REWARD_DIR))
from hallucination_compute_score import compute_score as _reward_compute_score  # noqa: E402


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
    """Build a fixture source; the vocabulary carrier is always appended last."""
    return adapter.build_rows(_write_source(tmp_path, *(pairs + (VOCAB,)), split=split), **kwargs)


def _by_id(rows: list[dict]) -> dict[str, dict]:
    return {row["extra_info"]["task_id"]: row for row in rows}


def _parts(row: dict) -> list[tuple[str, str]]:
    """``[(left, right)]`` for the row's option block, via the audit's parser."""
    return [verify_falseqa.split_option(option["text"]) for option in row["extra_info"]["options"]]


def _gold_text(row: dict) -> str:
    return verify_falseqa._option_text(row, row["extra_info"]["correct_option_id"])


def _reward(row: dict, response: str) -> float:
    """Score one response against a built row through the frozen dispatcher."""
    solution = f"<think>the premise is checked here</think>\n\n{response}"
    return _reward_compute_score(row["data_source"], solution, row["reward_model"]["ground_truth"])["score"]


# ---------------------------------------------------------------------------
# the two emitted branches
# ---------------------------------------------------------------------------


def test_emits_both_branches_under_template_a(tmp_path):
    rows, _ = _build(tmp_path, COLD)
    assert len(rows) == 2
    assert {row["extra_info"]["branch"] for row in rows} == {
        adapter.BRANCH_DIAG,
        adapter.BRANCH_ANSWERABLE,
    }
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


def test_diag_row_carries_the_region_as_a_replacement_pair(tmp_path):
    rows, _ = _build(tmp_path, COLD)
    diag = _by_id(rows)["falseqa-fake-train-0"]
    info = diag["extra_info"]
    gold = _gold_text(diag)
    assert gold == f"drink{PAIR_ARROW}eat"
    # the left item is the fragment the paired real question did *not* say, the
    # right item is exactly what it says there
    assert info["paired_original_text"] == adapter.normalise_question(COLD_REAL[0])
    assert "drink -> " in gold and gold not in info["paired_original_text"]
    assert "eat" in info["paired_original_text"]
    assert gold in diag["prompt"][0]["content"]
    assert info["deleted_condition_text"] == "eat"
    assert info["perturbed_entity_text"] == "drink"
    assert len(info["options"]) == adapter.K_OPTIONS
    assert json.loads(diag["reward_model"]["ground_truth"]) == {
        "answer": None,
        "correct_option_id": info["correct_option_id"],
        "has_diagnosis_label": True,
        "perturbation_type": "contradictory_condition",
        "solvable": False,
    }
    assert info["solvable"] is False
    assert info["error_type"] == "false_premise_pointable"
    assert info["perturbation_type"] == "contradictory_condition"
    assert info["pair_id"] == "train:0"
    assert MARKER_UNSOLVABLE_ID in diag["prompt"][0]["content"]


def test_answerable_row_is_a_placeholder_two_layer_row(tmp_path):
    rows, _ = _build(tmp_path, COLD)
    answerable = _by_id(rows)["falseqa-real-train-0"]
    info = answerable["extra_info"]
    payload = json.loads(answerable["reward_model"]["ground_truth"])
    assert info["solvable"] is True
    assert payload["solvable"] is True
    assert payload["solvable_answer"] is True
    assert "two_layer" not in payload and "judgment_only" not in payload and "pair_task" not in payload
    # the source's own answer, verbatim, is the gold of the answer layer
    assert payload["answer"] == COLD_REAL[1]
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
    assert info["pair_id"] == "train:0"
    assert info["difficulty"] == ""
    assert answerable["prompt"][0]["content"].endswith(schema.render_options_block(info["options"]))


def test_both_branches_share_one_prompt_shape(tmp_path):
    """D18: the two sides must not be tellable apart by prompt shape alone."""
    rows, _ = _build(tmp_path, COLD)
    by_id = _by_id(rows)
    diag, answerable = by_id["falseqa-fake-train-0"], by_id["falseqa-real-train-0"]
    assert diag["extra_info"]["template"] == answerable["extra_info"]["template"] == schema.TEMPLATE_A
    assert len(diag["extra_info"]["options"]) == len(answerable["extra_info"]["options"]) == adapter.K_OPTIONS

    def instruction(row: dict) -> tuple:
        """The template wording between the question and its options, plus the block size."""
        body = row["prompt"][0]["content"].partition("\n\n")[2]
        head, separator, options = body.partition("选项：")
        return head, separator, len(options.strip().splitlines())

    # the same wording and the same block size; only the option texts differ
    assert len({instruction(row) for row in (diag, answerable)}) == 1
    # and both prompts offer both verdicts, so the marker is not the discriminator
    for row in (diag, answerable):
        prompt = row["prompt"][0]["content"]
        assert MARKER_ANSWER in prompt and MARKER_UNSOLVABLE_ID in prompt


def test_every_block_is_a_placeholder_shaped_replacement_pair(tmp_path):
    """Both sides: one identical in-passage left item, three out-of-passage rights."""
    rows, _ = _build(tmp_path, COLD, BIRDS)
    for row in rows:
        question = verify_falseqa.question_of(row)
        options = row["extra_info"]["options"]
        parts = _parts(row)
        assert all(part is not None for part in parts), options
        lefts = {left for left, _ in parts}
        assert len(lefts) == 1
        left = lefts.pop()
        assert left in question
        assert any(schema.is_content_word(token) for token in schema.words(left))
        rights = [right for _, right in parts]
        assert len({right.casefold() for right in rights}) == adapter.K_OPTIONS
        assert len({len(schema.words(right)) for right in rights}) == 1
        assert len({verify_falseqa.item_signature(right) for right in rights}) == 1
        for right in rights:
            assert right.casefold() not in question.casefold(), (row["extra_info"]["task_id"], right)


def test_diag_distractors_are_out_of_passage_same_type_and_same_length(tmp_path):
    """The D21 rule, re-derived: not in the passage, same type, same word count."""
    rows, report = _build(tmp_path, COLD, BIRDS)
    diag = [row for row in rows if row["extra_info"]["branch"] == adapter.BRANCH_DIAG]
    assert diag and report["drops"]["pair_distractors_below_k"] == 0
    for row in diag:
        question = verify_falseqa.question_of(row)
        gold = _gold_text(row)
        _, gold_right = verify_falseqa.split_option(gold)
        for option in row["extra_info"]["options"]:
            _left, right = verify_falseqa.split_option(option["text"])
            assert right.casefold() not in question.casefold()
            assert len(schema.words(right)) == len(schema.words(gold_right))
            assert verify_falseqa.item_signature(right) == verify_falseqa.item_signature(gold_right)


def test_diag_gold_pair_is_the_recorded_replacement_between_the_twins(tmp_path):
    """L2: the left item did not survive the rewrite, the right item is the repair."""
    rows, _ = _build(tmp_path, COLD, BIRDS)
    diag = [row for row in rows if row["extra_info"]["branch"] == adapter.BRANCH_DIAG]
    assert len(diag) == 2
    for row in diag:
        info = row["extra_info"]
        left, right = verify_falseqa.split_option(_gold_text(row))
        assert not verify_falseqa.contains_token_window(info["paired_original_text"], left)
        assert verify_falseqa.contains_token_window(info["paired_original_text"], right)
        # the block is never padded: every option is one of the pool's items
        assert len({option["text"] for option in info["options"]}) == adapter.K_OPTIONS


def test_the_two_sides_of_a_pair_share_one_pair_id(tmp_path):
    """D27: the twins must be identifiable so the mixer can keep them together."""
    rows, _ = _build(tmp_path, COLD, BIRDS)
    by_id = _by_id(rows)
    assert by_id["falseqa-fake-train-0"]["extra_info"]["pair_id"] == "train:0"
    assert by_id["falseqa-real-train-0"]["extra_info"]["pair_id"] == "train:0"
    assert by_id["falseqa-fake-train-1"]["extra_info"]["pair_id"] == "train:1"
    assert by_id["falseqa-real-train-1"]["extra_info"]["pair_id"] == "train:1"
    assert adapter.pair_id("valid", 3) == "valid:3"


def test_task_ids_are_side_split_and_index_derived(tmp_path):
    assert adapter._task_id(adapter.SIDE_FAKE, "valid", 3) == "falseqa-fake-valid-3"
    assert adapter._task_id(adapter.SIDE_REAL, "test", 0) == "falseqa-real-test-0"
    rows, _ = _build(tmp_path, COLD, BIRDS)
    task_ids = [row["extra_info"]["task_id"] for row in rows]
    assert len(set(task_ids)) == len(task_ids)
    assert all(task_id.startswith("falseqa-") for task_id in task_ids)
    # one pair -> one diag row and one answerable row, both keyed on the pair index
    assert task_ids == [
        "falseqa-fake-train-0",
        "falseqa-real-train-0",
        "falseqa-fake-train-1",
        "falseqa-real-train-1",
    ]
    # ... and the task_id is what the audit recovers the pair from
    assert verify_falseqa.parse_task_id("falseqa-real-train-1") == ("real", "train", 1)
    assert verify_falseqa.parse_task_id("falseqa-fake-train-1")[1:] == verify_falseqa.parse_task_id(
        "falseqa-real-train-1"
    )[1:]


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
    # 2 pairs (COLD + the carrier), the carrier is multi-region so it certifies nothing
    assert report["funnel"] == {
        "raw_rows": 4,
        "after_malformed_drop": 4,
        "pairs_available": 2,
        "pairs_after_limit": 2,
        "pairs_with_region_certificate": 1,
        "rows_built": 2,
    }
    assert len(rows) == report["funnel"]["rows_built"]
    # every drop reason is reported, even the ones that cannot fire here
    assert tuple(report["drops"]) == adapter.DROP_REASONS
    assert report["drops"]["multi_region_defect"] == 1  # the carrier, by design
    assert {k: v for k, v in report["drops"].items() if v} == {"multi_region_defect": 1}


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
    assert report["by_branch"] == {adapter.BRANCH_DIAG: 2, adapter.BRANCH_ANSWERABLE: 2}
    assert report["by_template"] == {schema.TEMPLATE_A: 4}
    assert report["by_solvable"] == {False: 2, True: 2}
    assert report["by_error_type"] == {"": 2, adapter.ERROR_TYPE: 2}
    assert report["split"] == "train" and report["limit"] is None and report["seed"] == 0


def test_a_pair_can_yield_the_answerable_row_when_the_diag_row_is_dropped(tmp_path):
    rows, report = _build(tmp_path, NO_POOL)
    assert report["drops"]["pair_distractors_below_k"] == 1
    assert report["funnel"]["rows_built"] == 1
    assert report["by_branch"] == {adapter.BRANCH_ANSWERABLE: 1}


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
    assert rows == []  # no carrier here, so the single pair has no out-of-passage item


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
    # two regions means the replacement gold is ambiguous, not merely unparsed
    assert len(adapter.word_regions(MULTI_REAL[0], MULTI_FAKE[0])) == 2
    assert report["drops"]["multi_region_defect"] == 2  # MULTI + the carrier


def test_gold_without_a_content_word_drops_only_the_diag_row(tmp_path):
    """"to the" is a defect the model cannot be asked to point at."""
    rows, report = _build(tmp_path, NOCONTENT)
    assert report["drops"]["gold_has_no_content_word"] == 1
    assert [row["extra_info"]["branch"] for row in rows] == [adapter.BRANCH_ANSWERABLE]
    assert report["drops"]["pair_distractors_below_k"] == 0


def test_non_unique_first_token_drops_the_diag_row(tmp_path):
    """The fake fragment "south" occurs twice, so it is not locatable by its first word."""
    rows, report = _build(tmp_path, NONUNIQUE)
    assert report["drops"]["gold_first_token_not_unique"] == 1
    assert [row["extra_info"]["branch"] for row in rows] == [adapter.BRANCH_ANSWERABLE]


def test_a_pure_insertion_has_no_replacement_gold(tmp_path):
    """The fake question gained "warm"; there is no right item to offer."""
    rows, report = _build(tmp_path, RIGHT_EMPTY)
    assert report["drops"]["gold_pair_right_empty"] == 1
    assert [row["extra_info"]["branch"] for row in rows] == [adapter.BRANCH_ANSWERABLE]


def test_a_repairing_item_that_is_in_the_passage_drops_the_diag_row(tmp_path):
    """All three right items must be out-of-passage (section 4.3)."""
    rows, report = _build(tmp_path, RIGHT_IN)
    assert report["drops"]["gold_pair_right_in_passage"] == 1
    assert [row["extra_info"]["branch"] for row in rows] == [adapter.BRANCH_ANSWERABLE]


def test_a_short_distractor_pool_drops_the_row_instead_of_padding_it(tmp_path):
    """Fewer than k-1 same-type out-of-passage items means no block, not a bad one."""
    rows, report = _build(tmp_path, NO_POOL)
    assert report["drops"]["pair_distractors_below_k"] == 1
    assert [row["extra_info"]["branch"] for row in rows] == [adapter.BRANCH_ANSWERABLE]
    # and the pool really is empty: no off-type item was substituted
    assert not any(row["extra_info"]["branch"] == adapter.BRANCH_DIAG for row in rows)


def test_an_empty_source_answer_drops_the_answerable_row(tmp_path):
    rows, report = _build(tmp_path, EMPTY_ANSWER)
    assert report["drops"]["answer_empty"] == 1
    assert [row["extra_info"]["branch"] for row in rows] == [adapter.BRANCH_DIAG]


def test_a_passage_without_a_content_span_drops_the_placeholder_row(tmp_path):
    """No in-passage content phrase means no placeholder block -- drop, never pad."""
    rows, report = _build(tmp_path, NO_PLACEHOLDER)
    assert report["drops"]["placeholder_pool_below_k"] == 1
    assert [row["extra_info"]["branch"] for row in rows] == [adapter.BRANCH_DIAG]


def test_every_drop_reason_is_reachable_or_documented(tmp_path):
    """Each drop reason fires on at least one fixture, except the documented guard."""
    fired: set[str] = set()
    for index, pair in enumerate(ALL_PAIRS):
        _, report = _build(tmp_path / f"fixture-{index}", pair)
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

    # ``gold_not_verbatim_in_question`` is the one guard source data cannot reach:
    # a region's fake-side text is a slice of the presented question by
    # construction, so it is covered as a unit test on a hand-built region below.
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
    # the seed moves the option block (which right item is A/B/C), not the row set
    assert json.dumps(seeded, sort_keys=True) != json.dumps(unseeded, sort_keys=True)


def test_limit_caps_pairs_not_rows(tmp_path):
    rows, report = _build(tmp_path, COLD, BIRDS, limit=1)
    assert report["funnel"]["pairs_available"] == 3  # COLD, BIRDS, carrier
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
    assert report["funnel"]["pairs_after_limit"] == 2


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
    _write_source(raw_dir, COLD, BIRDS, VOCAB)
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
        adapter.BRANCH_ANSWERABLE,
    }
    assert {row["extra_info"]["split"] for row in written} == {"train"}

    report = json.loads(pathlib.Path(report_path).read_text(encoding="utf-8"))
    assert report["funnel"]["pairs_after_limit"] == 1
    assert report["limit"] == 1
    assert tuple(report["drops"]) == adapter.DROP_REASONS
    assert report["raw_dir"] == str(raw_dir)


def test_main_builds_the_named_split(tmp_path, monkeypatch, capsys):
    raw_dir = tmp_path / "raw"
    _write_source(raw_dir, COLD, VOCAB, split="valid")
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
    assert {row["extra_info"]["pair_id"] for row in written} == {"valid:0"}


def test_main_fails_closed_on_an_empty_artifact(tmp_path, monkeypatch, capsys):
    raw_dir = tmp_path / "raw"
    _write_source(raw_dir, MULTI, IDENT)
    out = tmp_path / "rows.parquet"
    monkeypatch.setattr(sys, "argv", ["falseqa_adapter.py", "--raw-dir", str(raw_dir), "--out", str(out)])
    assert adapter.main() == 1
    assert not out.exists()
    assert "nothing written" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# the reward cells of the section 9 matrix, end to end
# ---------------------------------------------------------------------------


def test_four_tier_reward_cells_for_a_diag_row(tmp_path):
    """Four-tier (D21/§6): the pointer earns +1, a bare refusal 0, an answer -1."""
    rows, _ = _build(tmp_path, COLD)
    diag = _by_id(rows)["falseqa-fake-train-0"]
    gold_id = diag["extra_info"]["correct_option_id"]
    wrong_id = next(letter for letter in "ABC" if letter != gold_id)
    assert _reward(diag, f"\\boxed{{UNSOLVABLE: {gold_id}}}") == 1.0
    assert _reward(diag, "\\boxed{UNSOLVABLE}") == 0.0
    assert _reward(diag, f"\\boxed{{UNSOLVABLE: {wrong_id}}}") == 0.0
    assert _reward(diag, "\\boxed{42}") == -1.0
    assert _reward(diag, "no box at all") == 0.0


def test_two_layer_reward_cells_for_an_answerable_row(tmp_path):
    """Two-layer (D27/§6): judgement +0.5, normalised answer match +0.5."""
    rows, _ = _build(tmp_path, COLD)
    answerable = _by_id(rows)["falseqa-real-train-0"]
    answer = json.loads(answerable["reward_model"]["ground_truth"])["answer"]
    assert _reward(answerable, f"\\boxed{{{answer}}}") == 1.0
    # norm_match: case, punctuation and articles are not part of the answer
    assert _reward(answerable, f"\\boxed{{{answer.upper()}}}") == 1.0
    assert _reward(answerable, "\\boxed{SOLVABLE}") == 0.5
    assert _reward(answerable, "\\boxed{something else}") == 0.5
    assert _reward(answerable, "\\boxed{UNSOLVABLE}") == 0.0
    assert _reward(answerable, "\\boxed{UNSOLVABLE: A}") == 0.0
    assert _reward(answerable, "no box at all") == 0.0


def test_shuffling_the_option_block_does_not_change_the_score(tmp_path):
    """§9 option-shuffle invariance: the reward compares ids, and the ids follow."""
    rows, _ = _build(tmp_path, COLD, BIRDS)
    diag = _by_id(rows)["falseqa-fake-train-0"]
    gold_id = diag["extra_info"]["correct_option_id"]
    assert _reward(diag, f"\\boxed{{UNSOLVABLE: {gold_id}}}") == 1.0

    shuffled = json.loads(json.dumps(diag))
    info = shuffled["extra_info"]
    texts = [option["text"] for option in info["options"]]
    random.Random(0).shuffle(texts)
    info["options"] = [{"id": "ABC"[index], "text": text} for index, text in enumerate(texts)]
    gold_text = _gold_text(diag)
    info["correct_option_id"] = next(
        option["id"] for option in info["options"] if option["text"] == gold_text
    )
    payload = json.loads(shuffled["reward_model"]["ground_truth"])
    payload["correct_option_id"] = info["correct_option_id"]
    shuffled["reward_model"]["ground_truth"] = json.dumps(payload, ensure_ascii=False, sort_keys=True)

    assert schema.validate_row(shuffled) == []
    assert _reward(shuffled, f"\\boxed{{UNSOLVABLE: {info['correct_option_id']}}}") == 1.0
    # the *old* id is now a distractor, so it must not score
    assert _reward(shuffled, f"\\boxed{{UNSOLVABLE: {gold_id}}}") == (1.0 if gold_id == info["correct_option_id"] else 0.0)


def test_the_gold_option_id_is_the_shuffled_position_of_the_gold_pair(tmp_path):
    rows, _ = _build(tmp_path, COLD, BIRDS)
    for row in rows:
        info = row["extra_info"]
        ids = [option["id"] for option in info["options"]]
        assert ids == ["A", "B", "C"]
        if row["extra_info"]["branch"] != adapter.BRANCH_DIAG:
            assert info["correct_option_id"] == ""
            continue
        payload = json.loads(row["reward_model"]["ground_truth"])
        assert payload["correct_option_id"] == info["correct_option_id"]
        assert _gold_text(row) in [option["text"] for option in info["options"]]


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
    diag, answerable = _by_id(rows)["falseqa-fake-train-0"], _by_id(rows)["falseqa-real-train-0"]
    assert verify_falseqa.question_of(diag) == adapter.normalise_question(COLD_FAKE[0])
    assert verify_falseqa.question_of(answerable) == adapter.normalise_question(COLD_REAL[0])


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

    An audit that reads 200 of 800 diagnosis rows passes an artifact whose gold was
    flipped on one of the other 600, so ``per_branch <= 0`` -- the CLI default --
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


def test_the_audit_l1_l2_l3_gates_pass_on_a_clean_artifact(tmp_path):
    """Every certificate check must be green on the fixture build it audits."""
    rows, _ = _build(tmp_path, COLD, BIRDS)
    reporter = verify_umwp.Reporter()
    sample = verify_falseqa.sample_by_branch(rows, per_branch=0, seed=0)
    verify_falseqa.check_l1(sample, reporter)
    verify_falseqa.check_l2(sample, reporter)
    verify_falseqa.check_l3_options(rows, reporter)
    verify_falseqa.check_answerable_side(rows, reporter)
    verify_falseqa.check_contract(rows, reporter)
    verify_falseqa.check_template_isomorphism(rows, reporter)
    verify_falseqa.check_pair_atomicity(rows, reporter)
    assert reporter.failed == []


def test_the_audit_l1_gate_rejects_a_flipped_gold(tmp_path):
    rows, _ = _build(tmp_path, COLD)
    diag = next(row for row in rows if row["extra_info"]["branch"] == adapter.BRANCH_DIAG)
    gold_id = diag["extra_info"]["correct_option_id"]
    other = next(option for option in diag["extra_info"]["options"] if option["id"] != gold_id)
    # swap the gold and a distractor's *texts*, leaving the id pointing at the wrong pair
    gold_text = _gold_text(diag)
    for option in diag["extra_info"]["options"]:
        if option["id"] == gold_id:
            option["text"] = other["text"]
        elif option["id"] == other["id"]:
            option["text"] = gold_text
    reporter = verify_umwp.Reporter()
    verify_falseqa.check_l1({adapter.BRANCH_DIAG: [diag]}, reporter)
    assert any("pointer gold" in name for name in reporter.failed)


def test_the_audit_l3_gate_rejects_an_in_passage_right_item(tmp_path):
    rows, _ = _build(tmp_path, COLD)
    diag = next(row for row in rows if row["extra_info"]["branch"] == adapter.BRANCH_DIAG)
    question = verify_falseqa.question_of(diag)
    left = _parts(diag)[0][0]
    diag["extra_info"]["options"][1]["text"] = f"{left}{PAIR_ARROW}child"  # "child" is in the question
    assert "child" in question
    reporter = verify_umwp.Reporter()
    verify_falseqa.check_l3_options(rows, reporter)
    assert any(name.startswith("L3c") for name in reporter.failed)


def test_the_audit_pair_atomicity_gate_rejects_a_straddling_pair(tmp_path):
    rows, _ = _build(tmp_path, COLD)
    verify_falseqa.check_pair_atomicity(rows, verify_umwp.Reporter())  # clean build
    diag = next(row for row in rows if row["extra_info"]["branch"] == adapter.BRANCH_DIAG)
    diag["extra_info"]["split"] = "val"  # its twin stays on the train side

    reporter = verify_umwp.Reporter()
    verify_falseqa.check_pair_atomicity(rows, reporter)
    assert any("atomicity" in name for name in reporter.failed)

    diag["extra_info"]["split"] = "train"
    diag["extra_info"]["pair_id"] = "train:999"
    reporter = verify_umwp.Reporter()
    verify_falseqa.check_pair_atomicity(rows, reporter)
    assert any("pair identity" in name for name in reporter.failed)


def test_the_audit_answerable_gate_rejects_a_broken_contract(tmp_path):
    rows, _ = _build(tmp_path, COLD)
    answerable = next(row for row in rows if row["extra_info"]["branch"] == adapter.BRANCH_ANSWERABLE)
    payload = json.loads(answerable["reward_model"]["ground_truth"])
    payload["solvable_answer"] = False
    payload["correct_option_id"] = "A"
    answerable["reward_model"]["ground_truth"] = json.dumps(payload)
    reporter = verify_umwp.Reporter()
    verify_falseqa.check_answerable_side(rows, reporter)
    assert len(reporter.failed) == 1


def test_the_audit_scaffold_check_is_a_gate_on_a_side_divergence(tmp_path):
    """D18: if one side's *wording* differs, the model reads the verdict off it."""
    rows, _ = _build(tmp_path, COLD)
    clean = verify_umwp.Reporter()
    verify_falseqa.check_template_isomorphism(rows, clean)
    assert clean.failed == []

    answerable = next(row for row in rows if row["extra_info"]["branch"] == adapter.BRANCH_ANSWERABLE)
    prompt = answerable["prompt"][0]["content"]
    # the answerable prompt stops offering the UNSOLVABLE verdict ...
    answerable["prompt"][0]["content"] = prompt.replace(MARKER_UNSOLVABLE_ID, "\\boxed{NO}", 1)
    reporter = verify_umwp.Reporter()
    verify_falseqa.check_template_isomorphism(rows, reporter)
    assert len(reporter.failed) == 1 and "template isomorphism" in reporter.failed[0]

    # ... or renames the option block both sides are supposed to share
    answerable["prompt"][0]["content"] = prompt.replace("选项：", "Options:", 1)
    reporter = verify_umwp.Reporter()
    verify_falseqa.check_template_isomorphism(rows, reporter)
    assert len(reporter.failed) == 1 and "template isomorphism" in reporter.failed[0]


@pytest.mark.parametrize("hostile", ["../../../../etc/passwd", "/etc/passwd", "train.csv", "val"])
def test_the_audit_gates_an_artifact_chosen_split(tmp_path, hostile):
    """The source split is read out of the artifact and used to build a file path.

    The audit's whole job is to distrust the artifact, so a value naming a CSV the
    source does not have is a boundary violation: unchecked, ``os.path.join`` would
    read that file as CSV and audit it as if it were the source.
    """
    rows, _ = _build(tmp_path, COLD)
    rows[0]["extra_info"]["task_id"] = f"falseqa-fake-{hostile}-0"
    reporter = verify_umwp.Reporter()
    assert verify_falseqa.check_split_vocabulary(rows, reporter) is False
    assert len(reporter.failed) == 1 and "task_id" in reporter.failed[0]


def test_load_source_split_refuses_a_split_the_source_does_not_have(tmp_path):
    with pytest.raises(ValueError, match="split must be one of"):
        verify_falseqa.load_source_split(str(tmp_path), "../../etc/passwd")


def test_the_audit_contract_check_catches_defect_metadata_drift(tmp_path):
    """The D18/D21 keys a diagnosis row carries are part of the contract."""
    rows, _ = _build(tmp_path, COLD)
    clean = verify_umwp.Reporter()
    verify_falseqa.check_contract(rows, clean)
    assert clean.failed == []

    diag = next(row for row in rows if row["extra_info"]["branch"] == adapter.BRANCH_DIAG)
    diag["extra_info"]["error_type"] = "false_premise_unpointable"
    diag["extra_info"]["perturbation_type"] = "unrelated_entity"
    reporter = verify_umwp.Reporter()
    verify_falseqa.check_contract(rows, reporter)
    assert reporter.failed == ["branch invariants (template A / verdicts / placeholder / pairing)"]


def test_the_audit_writes_the_section_9_spot_check_sample(tmp_path):
    rows, _ = _build(tmp_path, COLD, BIRDS)
    out = tmp_path / "spot.jsonl"
    verify_falseqa.write_spot_check(rows, str(out), seed=0)
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert lines  # the fixture has two diagnosis rows, both sampled
    for entry in lines:
        assert set(entry) == {
            "task_id",
            "split",
            "index",
            "question",
            "gold_pair",
            "distractor_pairs",
        }
        assert entry["gold_pair"] and len(entry["distractor_pairs"]) == adapter.K_OPTIONS - 1


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


def test_certify_gold_pair_accepts_a_single_visible_region():
    real, fake = "the cat sat on the mat", "the cat sat on the big mat"
    assert adapter.certify_gold_pair(real, fake, adapter.word_regions(real, fake)) is None  # pure insertion

    real, fake = "the cat sat on the soft mat", "the cat sat on the big mat"
    assert adapter.certify_gold_pair(real, fake, adapter.word_regions(real, fake)) == ("big", "soft")


def test_gold_pair_problem_reports_every_gate():
    """Each certificate is reported by name, so the funnel can say why it dropped."""
    real, fake = "the cat sat on the soft mat", "the cat sat on the big mat"
    assert adapter.gold_pair_problem(real, fake, adapter.word_regions(real, fake)) is None

    # no region at all, or more than one
    assert adapter.gold_pair_problem("a b", "a b", []) == "multi_region_defect"
    two = adapter.word_regions(MULTI_REAL[0], MULTI_FAKE[0])
    assert len(two) == 2
    assert adapter.gold_pair_problem(MULTI_REAL[0], MULTI_FAKE[0], two) == "multi_region_defect"

    # a region whose text is not in the question (defensive: a real region's text
    # is a slice of the question by construction, so this cannot fire on data)
    ghost = adapter._Region(tag="insert", a_text="", b_text="zebra", a_start=2, b_start=2, n_a=0, n_b=1)
    assert adapter.gold_pair_problem("a b", "a c", [ghost]) == "gold_not_verbatim_in_question"

    # a stop-word-only left item makes an option block worse than no block
    real, fake = "birds fly south in the winter", "birds fly to the south in the winter"
    assert adapter.gold_pair_problem(real, fake, adapter.word_regions(real, fake)) == "gold_has_no_content_word"

    # the left item's first token must locate the span, and here "south" occurs twice
    real, fake = "the south is warm", "the south south is warm"
    assert adapter.gold_pair_problem(real, fake, adapter.word_regions(real, fake)) == "gold_first_token_not_unique"

    # a pure insertion has no repairing fragment to offer
    assert (
        adapter.gold_pair_problem(RIGHT_EMPTY_REAL[0], RIGHT_EMPTY_FAKE[0], adapter.word_regions(RIGHT_EMPTY_REAL[0], RIGHT_EMPTY_FAKE[0]))
        == "gold_pair_right_empty"
    )

    # ... and the repairing fragment must itself be out-of-passage
    assert (
        adapter.gold_pair_problem(RIGHT_IN_REAL[0], RIGHT_IN_FAKE[0], adapter.word_regions(RIGHT_IN_REAL[0], RIGHT_IN_FAKE[0]))
        == "gold_pair_right_in_passage"
    )


def test_item_signature_separates_the_types_the_doc_cares_about():
    # the doc's "same type" reading of one of its own examples: man -> women
    assert adapter.item_signature("women") == adapter.item_signature("men")
    assert adapter.item_signature("women") == adapter.item_signature("children")
    # a plural is a different type from a plain noun
    assert adapter.item_signature("women") != adapter.item_signature("adults")
    # capitalisation, word count and the morphological class are all part of it
    assert adapter.item_signature("Academy of") != adapter.item_signature("rainy days")
    assert adapter.item_signature("fish") != adapter.item_signature("coral reefs")
    assert adapter.item_signature("walking")[3] == "ing"
    assert adapter.item_signature("rainy days") != adapter.item_signature("Academy of")


def test_build_item_bank_indexes_windows_by_signature():
    bank = adapter.build_item_bank(["Sharks hunt fish near coral reefs at night"])
    assert bank[adapter.item_signature("fish")]["fish"] == "fish"
    assert "reefs" in bank[adapter.item_signature("reefs")]
    assert bank[adapter.item_signature("coral reefs")]["coral reefs"] == "coral reefs"


def test_out_of_passage_right_pool_filters_and_tiers():
    bank = adapter.build_item_bank(["Sharks hunt fish near coral reefs at night", "Wolves hunt deer at dawn"])
    signature = adapter.item_signature("fish")  # 1 token, lower, plain
    pool = adapter.out_of_passage_right_pool(
        bank, signature=signature, char_len=4, passage="hunt wolves at dawn", exclude="", minimum=2
    )
    assert pool is not None
    folded = "hunt wolves at dawn"
    for item in pool:
        assert item.casefold() not in folded
        assert len(schema.words(item)) == 1
        assert adapter.item_signature(item) == signature
    # the closest character-length tier wins when it is big enough ...
    tiered = adapter.out_of_passage_right_pool(
        bank, signature=signature, char_len=4, passage="", exclude="", minimum=2
    )
    assert {len(item) for item in tiered} == {4}
    # ... and the pool is None (never padded) when the type cannot field k-1 items
    assert (
        adapter.out_of_passage_right_pool(
            bank, signature=adapter.item_signature("walking"), char_len=7, passage="", exclude="", minimum=2
        )
        is None
    )


def test_out_of_passage_right_pool_excludes_the_gold_right_item():
    bank = adapter.build_item_bank(["Sharks hunt fish near coral reefs at night"])
    signature = adapter.item_signature("fish")
    pool = adapter.out_of_passage_right_pool(
        bank, signature=signature, char_len=4, passage="", exclude="fish", minimum=1
    )
    assert pool is not None and "fish" not in pool


def test_mine_content_spans_requires_a_content_word():
    assert "child" in adapter.mine_content_spans("What should a child eat?", 1)
    assert adapter.mine_content_spans("the of a", 1) == []
    assert adapter.mine_content_spans("the of a", 0) == []
    spans = adapter.mine_content_spans("What should a child eat?", 2)
    assert "a child" in spans and "should a" not in spans


def test_mine_placeholder_options_mirrors_a_reference_and_fails_closed():
    bank = adapter.build_item_bank(["Sharks hunt fish near coral reefs at night"])
    reference = (1, 1, adapter.item_signature("fish"), 4)
    mined = adapter.mine_placeholder_options("where do cats sleep", bank, [reference], "row", 3)
    assert mined is not None
    options, left, rights = mined
    assert len(options) == 3 and [option["id"] for option in options] == ["A", "B", "C"]
    assert left in "where do cats sleep"
    assert len(rights) == 3 and len(set(rights)) == 3
    for right in rights:
        assert right not in "where do cats sleep"
        assert adapter.item_signature(right) == adapter.item_signature("fish")
    # no reference shape can be satisfied -> None (the caller drops the row)
    assert adapter.mine_placeholder_options("the of a", bank, [reference], "row", 3) is None


def test_split_option_accepts_only_well_formed_pairs():
    assert verify_falseqa.split_option("men -> women") == ("men", "women")
    assert verify_falseqa.split_option("men -> women -> children") is None
    assert verify_falseqa.split_option("men") is None
    assert verify_falseqa.split_option(" -> women") is None
    assert verify_falseqa.split_option("men -> ") is None


def test_contains_token_window_is_not_a_substring_test():
    assert verify_falseqa.contains_token_window("what should women do", "women")
    assert not verify_falseqa.contains_token_window("what should women do", "men")
    assert not verify_falseqa.contains_token_window("what should women do", "woman")
    assert verify_falseqa.contains_token_window("what should women do", "should women")


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
    assert adapter.BRANCH_ANSWERABLE == schema.BRANCH_SOLVABLE_TWO_LAYER
    assert adapter.ERROR_TYPE in {"false_premise_pointable", "false_premise_unpointable"}
    assert adapter.PERTURBATION_TYPE in schema.PERTURBATION_TYPES
    assert adapter.SPLITS == ("train", "valid", "test")
    assert adapter.DATA_FILE_TEMPLATE.format(split="valid") == "valid.csv"
