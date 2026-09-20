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
"""Tests for ``umwp_adapter.py`` (design doc D24, sections 4.6 / 4.8 / 5.1 / 5.2 / 9).

The fixtures are real ``StandardDataset.jsonl`` rows, copied verbatim -- the
questions below (including their doubled spaces, doubled periods and missing
spaces before ``How``) are exactly what UMWP ships, so the tests exercise the same
text shapes as the real build.  Small fixtures are written to ``tmp_path`` and
never to the downloaded data directory, so the suite is self-contained: it reads
nothing outside this repository.

What the contract is, after D24:

* the **answerable** side ships as ``solvable_two_layer`` -- template A, a k=3
  *placeholder* option block with no correct item, ``answer`` = the source's own
  number, ``correct_option_id=null``;
* the **unanswerable** side ships as ``unsolvable_bare`` -- template B, **no
  option block at all**, ``\\boxed{UNSOLVABLE}`` gold, with the defect class kept
  for statistics only (it no longer decides the branch);
* the pair certificate is the pairing / residual-content check, so the
  token-invisible (character-level) defects ship too.

Covers every funnel stage, both emitted branches, the drop branches (stray answer,
duplicate question, ambiguous partner, unknown category, unanswerable question
without a category, answerable question without a placeholder block, malformed
line), the certificate helpers, and the section 9 assertions: the option-block
contract per side, the no-shortcut gate (length + BoW NB <= random + 5pt) and the
reward cells pinned against the frozen reward module.
"""

from __future__ import annotations

import json
import random
import sys

import pytest
import schema
import umwp_adapter as adapter
import verify_umwp

# ---------------------------------------------------------------------------
# fixtures -- verbatim UMWP rows
# ---------------------------------------------------------------------------

# every tuple is (id, question, answerable, answer, category, relevant_ids, source)
# exactly as it appears in data/StandardDataset.jsonl

# real umwp row id 2 (answerable side of a category-2 pair)
BIRDS_ANSWERABLE = (
    2,
    "3 birds were sitting on the fence. 2 more birds and 6 more storks came to join them..How many more storks than birds are sitting on the fence?",
    True,
    [1.0],
    None,
    None,
    "SVAMP",
)

# real umwp row id 2502 (category 2: "6 more" -> "some")
BIRDS_UNANSWERABLE = (
    2502,
    "3 birds were sitting on the fence. 2 more birds and some storks came to join them. How many more storks than birds are sitting on the fence?",
    False,
    None,
    2,
    [2],
    "SVAMP",
)

# real umwp row id 1111
FISH_ANSWERABLE = (
    1111,
    "John buys 3 reels of 100m fishing line.  He cuts it into 10m sections.  How many sections does he get?",
    True,
    [30.0],
    None,
    None,
    "GSM8K",
)

# real umwp row id 3611 (category 1: "10m" dropped, "with equal length" inserted)
FISH_UNANSWERABLE = (
    3611,
    "John buys 3 reels of 100m fishing line.  He cuts it into sections with equal length.  How many sections does he get?",
    False,
    None,
    1,
    [1111],
    "GSM8K",
)

# real umwp row id 1347
JUGGLE_ANSWERABLE = (
    1347,
    "Jeanette is practicing her juggling. Each week she can juggle 2 more objects than the week before. If she starts out juggling 3 objects and practices for 5 weeks, how many objects can she juggle?",
    True,
    [13.0],
    None,
    None,
    "GSM8K",
)

# real umwp row id 3847 (category 5: the question clause is truncated)
JUGGLE_UNANSWERABLE = (
    3847,
    "Jeanette is practicing her juggling. Each week she can juggle 2 more objects than the week before. If she starts out juggling 3 objects and practices for 5 weeks, how many?",
    False,
    None,
    5,
    [1347],
    "GSM8K",
)

# real umwp row id 113; its partner 2613 is the stray-int row
STRAY_ANSWERABLE = (
    113,
    "The cave is 1218 feet deep and they are already at 849 feet. If they are travelling at speed of 17.How much farther until they reach the end of the cave?",
    True,
    [369.0],
    None,
    None,
    "SVAMP",
)

# real umwp row id 2613: unanswerable with a stray int answer 2
STRAY_UNANSWERABLE = (
    2613,
    "The cave is 1218 feet deep, and they are already at 849 feet. If they are traveling at a certain fast speed, how much farther until they reach the end of the cave?",
    False,
    2,
    1,
    [113],
    "SVAMP",
)

# real umwp row id 512; 744 repeats this question verbatim
DUP_ANSWERABLE_A = (
    512,
    " A toy store had 6 giant stuffed bears in stock when they got another shipment with 18 bears in it. The put the bears onto shelves with 6 on each shelf. How many shelves did they use? ",
    True,
    [4.0],
    None,
    None,
    "MultiArith",
)

# real umwp row id 744 -- same normalised question as 512, same label
DUP_ANSWERABLE_B = (
    744,
    " A toy store had 6 giant stuffed bears in stock when they got another shipment with 18 bears in it. The put the bears onto shelves with 6 on each shelf. How many shelves did they use? ",
    True,
    [4.0],
    None,
    None,
    "MultiArith",
)

# real umwp row: the unanswerable partner of 512
PARTNER_OF_512 = (
    3012,
    "A toy store had 6 giant stuffed bears in stock when they got another shipment with 18 bears in it. They put the bears onto shelves with 0 on each shelf. How many shelves did they use? ",
    False,
    None,
    3,
    [512],
    "MultiArith",
)

# real umwp row: the unanswerable partner of 744 (dropped with its partner)
PARTNER_OF_744 = (
    3244,
    "A toy store had 6 giant stuffed bears in stock when they got another shipment with 18 bears in it. The put the tiger onto shelves with 6 on each shelf. How many shelves did they use? ",
    False,
    None,
    3,
    [744],
    "MultiArith",
)

# real umwp row id 857; 3357 asks the identical question with the opposite label
CONFLICT_ANSWERABLE = (
    857,
    "Tate finishes high school in 1 year less than normal.  It takes him 3 times that long to get his bachelor's degree and Ph.D.  How many years did he spend in high school and college?",
    True,
    [12.0],
    None,
    None,
    "GSM8K",
)

# real umwp row id 3357
CONFLICT_UNANSWERABLE = (
    3357,
    "Tate finishes high school in 1 year less than normal.  It takes him 3 times that long to get his bachelor's degree and Ph.D.  How many years did he spend in high school and college?",
    False,
    None,
    1,
    [857],
    "GSM8K",
)

# real umwp row: answerable partner of 2503
INSERT_ANSWERABLE = (
    3,
    "Randy has 58 blocks. He uses 27 blocks to build a tower and 53 blocks to build a house..How many blocks did he use to build the tower and the house altogether?",
    True,
    [80.0],
    None,
    None,
    "SVAMP",
)

# real umwp row id 2503 (category 2, pure insertion of "less than")
INSERT_UNANSWERABLE = (
    2503,
    "Randy has 58 blocks. He uses 27 blocks to build a tower and less than 53 blocks to build a house. How many blocks did he use to build the tower and the house altogether?",
    False,
    None,
    2,
    [3],
    "SVAMP",
)

# real umwp row: answerable partner of 2524
MULTI_ANSWERABLE = (
    24,
    "Brenda's mother made cookies for 14. If each of them had 30 cookies.How many cookies did she prepare?",
    True,
    [420.0],
    None,
    None,
    "SVAMP",
)

# real umwp row id 2524 (category 2, two change regions)
MULTI_UNANSWERABLE = (
    2524,
    "Brenda's mother made cookies for some children. If each of them had 30 cookies. How many cookies did she make?",
    False,
    None,
    2,
    [24],
    "SVAMP",
)

# real umwp row: answerable partner of 2510
SIGNFLIP_ANSWERABLE = (
    10,
    "Bobby ate 23 pieces of candy. If he initially had 30 pieces of candy.How many pieces of candy does he still have left?",
    True,
    [7.0],
    None,
    None,
    "SVAMP",
)

# real umwp row id 2510 (category 3, a sign flip invisible to a word diff)
SIGNFLIP_UNANSWERABLE = (
    2510,
    "Bobby ate -23 pieces of candy. If he initially had 30 pieces of candy.How many pieces of candy does he still have left?",
    False,
    None,
    3,
    [10],
    "SVAMP",
)

# real umwp row: answerable partner of 2532
NONUMBER_ANSWERABLE = (
    32,
    "Jack received 5 emails in the morning, 8 emails in the afternoon and 72 emails in the evening..How many emails did Jack receive in the morning and afternoon?",
    True,
    [13.0],
    None,
    None,
    "SVAMP",
)

# real umwp row id 2532 (category 1, nothing numeric is actually lost)
NONUMBER_UNANSWERABLE = (
    2532,
    "Jack received 5 emails in the morning, 8 emails in the afternoon and 72 emails in the evening. How many emails will Jack receive in the tomorrow morning and afternoon?",
    False,
    None,
    1,
    [32],
    "SVAMP",
)

# real umwp rows 88/2588 (category 3, a number swap).  Under the old pointer
# contract the answerable side was dropped ("4" is too short to mine an anchored
# option block); the anchorless placeholder miner takes it.
MARBLE_ANSWERABLE = (
    88,
    "Josh had 9 marbles in his collection. He lost some marbles. If he has 4 marbles now.How many marbles did he lose?",
    True,
    [5.0],
    None,
    None,
    "SVAMP",
)

MARBLE_UNANSWERABLE = (
    2588,
    "Josh had 9 marbles in his collection. He lost some marbles. "
    "If he has 11 marbles now.How many marbles did he lose?",
    False,
    None,
    3,
    [88],
    "SVAMP",
)

# A synthetic pair whose answerable question cannot field k=3 distinct
# equal-length content spans, so the *answerable* side is dropped while the
# unanswerable side still ships (the "drop the row only if it cannot mine" rule).
SHORT_ANSWERABLE = (9001, "What is 2?", True, [2.0], None, None, "SVAMP")
SHORT_UNANSWERABLE = (9002, "What is some?", False, None, 2, [9001], "SVAMP")

THREE_WAY = [
    BIRDS_ANSWERABLE,
    BIRDS_UNANSWERABLE,
    FISH_ANSWERABLE,
    FISH_UNANSWERABLE,
    JUGGLE_ANSWERABLE,
    JUGGLE_UNANSWERABLE,
]

_FIELDS = ("id", "question", "answerable", "answer", "category", "relevant_ids", "source")


def _rows(*specs: tuple) -> list[dict]:
    """Fixture tuples -> source rows, in the source file's own field order."""
    return [dict(zip(_FIELDS, spec, strict=False)) for spec in specs]


def _write_source(raw_dir, *specs) -> str:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / adapter.DATA_FILE
    with open(path, "w", encoding="utf-8") as handle:
        for row in _rows(*specs):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(raw_dir)


def _build(tmp_path, *specs, **kwargs):
    return adapter.build_rows(_write_source(tmp_path, *specs), **kwargs)


def _by_id(rows: list[dict]) -> dict[str, dict]:
    return {row["extra_info"]["task_id"]: row for row in rows}


def _sides(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """``(answerable rows, unanswerable rows)`` of the built pool."""
    return (
        [row for row in rows if row["extra_info"]["solvable"]],
        [row for row in rows if not row["extra_info"]["solvable"]],
    )


# ---------------------------------------------------------------------------
# the two emitted branches
# ---------------------------------------------------------------------------


def test_emits_both_branches(tmp_path):
    rows, _ = _build(tmp_path, *THREE_WAY)
    branches = {row["extra_info"]["branch"] for row in rows}
    assert branches == {adapter.BRANCH_TWO_LAYER, adapter.BRANCH_BARE}
    assert len(rows) == 6
    templates = {row["extra_info"]["branch"]: row["extra_info"]["template"] for row in rows}
    assert templates[adapter.BRANCH_TWO_LAYER] == schema.TEMPLATE_A
    assert templates[adapter.BRANCH_BARE] == schema.TEMPLATE_B


def test_every_row_satisfies_the_schema_contract(tmp_path):
    rows, _ = _build(tmp_path, *THREE_WAY)
    for row in rows:
        assert schema.validate_row(row) == []
    # the canonical key set is uniform across rows
    key_sets = {tuple(sorted(row["extra_info"])) for row in rows}
    assert len(key_sets) == 1


def test_two_layer_row_shape(tmp_path):
    rows, _ = _build(tmp_path, *THREE_WAY)
    row = _by_id(rows)["umwp-ans-2"]
    info = row["extra_info"]
    ground_truth = json.loads(row["reward_model"]["ground_truth"])
    assert info["solvable"] is True
    assert info["branch"] == adapter.BRANCH_TWO_LAYER
    assert info["template"] == schema.TEMPLATE_A
    assert ground_truth == {
        "answer": "1.0",  # the source's own number: this is the real gold now (D24)
        "correct_option_id": None,
        "has_diagnosis_label": False,
        "perturbation_type": None,
        "solvable": True,
        "two_layer": True,
    }
    assert info["correct_option_id"] == ""
    # the answerable row has no defect of its own, so it carries no class into the
    # section 4.8 defect-type statistics (the partner's is not copied onto it)
    assert info["perturbation_type"] == ""
    assert info["error_type"] == ""
    assert info["index"] == 2
    assert info["split"] == "train"
    assert info["seed"] == 0
    assert info["difficulty"] == "SVAMP"  # the base problem's origin, not a UMWP field
    # the pair's defect stays in the audit fields
    assert info["deleted_condition_text"] == "6 more"
    assert info["perturbed_entity_text"] == "some"
    assert info["paired_original_text"] == adapter.normalise_question(BIRDS_UNANSWERABLE[1])


def test_bare_rows_refuse_without_options(tmp_path):
    rows, _ = _build(tmp_path, *THREE_WAY)
    by_id = _by_id(rows)
    fish, juggle, birds = by_id["umwp-uns-3611"], by_id["umwp-uns-3847"], by_id["umwp-uns-2502"]
    assert fish["extra_info"]["error_type"] == "key_information_missing"
    assert juggle["extra_info"]["error_type"] == "question_missing"
    assert birds["extra_info"]["error_type"] == "ambiguous_key_information"
    for row in (fish, juggle, birds):
        info = row["extra_info"]
        ground_truth = json.loads(row["reward_model"]["ground_truth"])
        assert info["branch"] == adapter.BRANCH_BARE
        assert info["template"] == schema.TEMPLATE_B
        assert info["options"] == []
        assert ground_truth["answer"] is None
        assert ground_truth["has_diagnosis_label"] is False
        assert ground_truth["correct_option_id"] is None
        assert "two_layer" not in ground_truth
    assert birds["extra_info"]["perturbation_type"] == "ambiguous_condition"
    # the fish row lost a number AND gained text (the design doc's claim that cat1
    # is invisible in the question is false for this row); the audit records both
    assert fish["extra_info"]["deleted_condition_text"] == "10m"
    assert fish["extra_info"]["perturbed_entity_text"] == "with equal length"
    assert juggle["extra_info"]["deleted_condition_text"] == "objects can she juggle"
    assert juggle["extra_info"]["perturbed_entity_text"] == ""


def test_template_a_is_solvable_only_and_template_b_unsolvable_only(tmp_path):
    """D24 moved UMWP's unsolvable arm to template B; its template A is solvable-only.

    The design doc's A-must-hold-both-sides constraint (D18) is a *mix-level*
    constraint: the A-unsolvable rows come from FalseQA/TreeCut.  What this adapter
    must guarantee is that its own A rows are the two-layer solvable ones with a
    placeholder block, and its B rows the bare refusals with none.
    """
    rows, _ = _build(tmp_path, *THREE_WAY)
    assert {(row["extra_info"]["template"], row["extra_info"]["solvable"]) for row in rows} == {
        (schema.TEMPLATE_A, True),
        (schema.TEMPLATE_B, False),
    }


# ---------------------------------------------------------------------------
# section 9: the two sides' option-block contract
# ---------------------------------------------------------------------------


def test_section9_answerable_side_carries_a_placeholder_block(tmp_path):
    """Every answerable row: template A, k=3 same-question spans, no correct item."""
    rows, _ = _build(tmp_path, *THREE_WAY)
    answerable, _ = _sides(rows)
    assert answerable
    for row in answerable:
        info = row["extra_info"]
        ground_truth = json.loads(row["reward_model"]["ground_truth"])
        question = row["prompt"][0]["content"]
        texts = [option["text"] for option in info["options"]]
        assert ground_truth["correct_option_id"] is None
        assert info["correct_option_id"] == ""
        assert [option["id"] for option in info["options"]] == ["A", "B", "C"]
        assert len(set(texts)) == 3
        assert all(text in question for text in texts)  # same-question spans (section 4.4)
        assert len({len(schema.words(text)) for text in texts}) == 1  # equal length (D15)
        assert all(any(schema.is_content_word(token) for token in schema.words(text)) for text in texts)
        # the placeholder block is not an answer key: the gold is not among the options
        assert str(ground_truth["answer"]).strip() not in {text.strip() for text in texts}
        assert "选项：" in question


def test_section9_unanswerable_side_carries_no_options_block(tmp_path):
    """Section 5.1: a three-tier row's prompt must not offer options at all."""
    rows, _ = _build(tmp_path, *THREE_WAY)
    _, unanswerable = _sides(rows)
    assert unanswerable
    for row in unanswerable:
        assert row["extra_info"]["options"] == []
        assert json.loads(row["reward_model"]["ground_truth"])["correct_option_id"] is None
        assert "选项：" not in row["prompt"][0]["content"]
        assert "\\boxed{UNSOLVABLE}" in row["prompt"][0]["content"]


def test_section9_sides_pass_the_no_shortcut_gate(tmp_path):
    """Section 9's gate: length and BoW NB <= random + 5pt on the shipped sides."""
    rows, _ = _build(tmp_path, *THREE_WAY)
    texts = [verify_umwp.question_of(row) for row in rows]
    labels = [1 if row["extra_info"]["solvable"] else 0 for row in rows]
    assert sum(labels) == len(labels) - sum(labels)  # both sides present
    assert verify_umwp.length_oof(texts, labels, folds=2, seed=0) <= verify_umwp.L3_MAX_BALANCED_ACCURACY
    assert verify_umwp.bow_nb_oof(texts, labels, folds=2, seed=0, min_support=1) <= verify_umwp.L3_MAX_BALANCED_ACCURACY


def test_no_shortcut_gate_has_teeth():
    """The gate is a real threshold: a leaking corpus must read above it."""
    leaking_lengths = ["tiny question"] * 10 + ["a much longer question with many more words in it"] * 10
    labels = [1] * 10 + [0] * 10
    assert verify_umwp.length_oof(leaking_lengths, labels, folds=2, seed=0) > verify_umwp.L3_MAX_BALANCED_ACCURACY
    leaking_tokens = ["alpha " + "filler " * 10] * 10 + ["omega " + "filler " * 10] * 10
    assert verify_umwp.bow_nb_oof(leaking_tokens, labels, folds=2, seed=0, min_support=1) > (
        verify_umwp.L3_MAX_BALANCED_ACCURACY
    )
    flat = ["the same question text"] * 20
    assert verify_umwp.length_oof(flat, labels, folds=2, seed=0) == 0.5


def test_verifier_audit_accepts_the_built_fixture_rows(tmp_path):
    """The whole §9 audit, run on a real fixture build (source anchor included)."""
    raw_dir = _write_source(tmp_path, *THREE_WAY)
    rows, _ = adapter.build_rows(raw_dir)
    reporter = verify_umwp.Reporter()
    verify_umwp.check_contract(rows, reporter)
    verify_umwp.check_source_anchor(rows, reporter, verify_umwp.load_source_index(raw_dir), raw_dir)
    verify_umwp.check_sides(rows, reporter)
    verify_umwp.check_l3_options(rows, reporter)
    sample = verify_umwp.sample_by_branch(rows, per_branch=50, seed=0)
    verify_umwp.check_l1(sample, reporter)
    verify_umwp.check_l2(sample, reporter)
    verify_umwp.check_l3_shortcuts(rows, reporter, folds=2, seed=0, min_support=1, sweep=False)
    assert reporter.failed == []
    assert len(reporter.results) >= 12


# ---------------------------------------------------------------------------
# section 9: reward cells
# ---------------------------------------------------------------------------


def test_section9_reward_cells_two_layer_and_three_tier(tmp_path):
    """The two shipped sides' cells, scored end to end by the frozen reward file."""
    module = verify_umwp.load_reward_module()

    def score(row: dict, ground_truth: str, final: str) -> float:
        return float(
            module.compute_score(row["data_source"], verify_umwp._rollout(final), ground_truth, row["extra_info"])[
                "score"
            ]
        )

    rows, _ = _build(tmp_path, *THREE_WAY)
    two_layer = _by_id(rows)["umwp-ans-2"]
    ground_truth = two_layer["reward_model"]["ground_truth"]
    assert score(two_layer, ground_truth, r"\boxed{1.0}") == 1.0  # judged solvable + solved
    assert score(two_layer, ground_truth, r"\boxed{1}") == 1.0  # 1.0 == 1 as a number
    assert score(two_layer, ground_truth, r"\boxed{2}") == 0.5  # judged, not solved
    assert score(two_layer, ground_truth, r"\boxed{SOLVABLE}") == 0.0  # verdict is not an answer
    assert score(two_layer, ground_truth, r"\boxed{UNSOLVABLE}") == 0.0  # misrefusal
    assert score(two_layer, ground_truth, "the answer is not boxed") == 0.0

    bare = _by_id(rows)["umwp-uns-2502"]
    bare_gt = bare["reward_model"]["ground_truth"]
    assert score(bare, bare_gt, r"\boxed{UNSOLVABLE}") == 1.0
    assert score(bare, bare_gt, r"\boxed{UNSOLVABLE: B}") == 1.0  # option id is meaningless here
    assert score(bare, bare_gt, r"\boxed{42}") == -1.0  # fabricated answer
    assert score(bare, bare_gt, "the question cannot be answered") == 0.0

    # the branch-dependent cell of section 5.1: same marker, four-tier ground truth
    four_tier = json.dumps({**json.loads(bare_gt), "has_diagnosis_label": True, "correct_option_id": "B"})
    assert score(bare, four_tier, r"\boxed{UNSOLVABLE}") == 0.0
    assert score(bare, four_tier, r"\boxed{UNSOLVABLE: B}") == 1.0


# ---------------------------------------------------------------------------
# funnel
# ---------------------------------------------------------------------------


def test_funnel_stage_counts_for_a_clean_fixture(tmp_path):
    rows, funnel = _build(tmp_path, *THREE_WAY)
    assert list(funnel) == [
        "raw_rows",
        "after_malformed_drop",
        "after_stray_answer_drop",
        "after_duplicate_question_drop",
        "after_pair_resolution_drop",
        "after_defect_class_drop",
        "after_defect_certificate_drop",
        "after_option_mining_drop",
        "after_limit",
    ]
    assert funnel == {
        "raw_rows": 6,
        "after_malformed_drop": 6,
        "after_stray_answer_drop": 6,
        "after_duplicate_question_drop": 6,
        "after_pair_resolution_drop": 6,
        "after_defect_class_drop": 6,
        "after_defect_certificate_drop": 6,
        "after_option_mining_drop": 6,
        "after_limit": 6,
    }
    assert len(rows) == funnel["after_limit"]


def test_funnel_is_monotone_and_counts_only_drops(tmp_path):
    _, funnel = _build(
        tmp_path,
        STRAY_ANSWERABLE,
        STRAY_UNANSWERABLE,
        CONFLICT_ANSWERABLE,
        CONFLICT_UNANSWERABLE,
        MULTI_ANSWERABLE,
        MULTI_UNANSWERABLE,
        SIGNFLIP_ANSWERABLE,
        SIGNFLIP_UNANSWERABLE,
    )
    counts = list(funnel.values())
    assert counts == sorted(counts, reverse=True)
    # 8 rows in; the stray-answer row and both members of the contradictory
    # duplicate pair are lost, and id 744's partner is orphaned -- the multi-region
    # and sign-flip pairs now ship on both sides (the certificate only demands the
    # pairing / residual-content identity)
    assert funnel == {
        "raw_rows": 8,
        "after_malformed_drop": 8,
        "after_stray_answer_drop": 7,
        "after_duplicate_question_drop": 5,
        "after_pair_resolution_drop": 4,
        "after_defect_class_drop": 4,
        "after_defect_certificate_drop": 4,
        "after_option_mining_drop": 4,
        "after_limit": 4,
    }


def test_stray_answer_row_and_its_partner_dropped(tmp_path):
    rows, funnel = _build(tmp_path, STRAY_ANSWERABLE, STRAY_UNANSWERABLE)
    assert rows == []
    assert funnel["after_stray_answer_drop"] == 1  # the stray-int row itself
    assert funnel["after_pair_resolution_drop"] == 0  # its partner has no partner left


def test_conflicting_duplicate_question_drops_both_rows(tmp_path):
    rows, funnel = _build(tmp_path, CONFLICT_ANSWERABLE, CONFLICT_UNANSWERABLE)
    assert rows == []
    assert funnel["after_duplicate_question_drop"] == 0


def test_same_label_duplicate_question_keeps_one(tmp_path):
    rows, funnel = _build(tmp_path, DUP_ANSWERABLE_A, DUP_ANSWERABLE_B, PARTNER_OF_512, PARTNER_OF_744)
    assert funnel["after_duplicate_question_drop"] == 3  # id 744 dropped, 512 kept
    assert funnel["after_pair_resolution_drop"] == 2  # 744's partner loses its partner
    assert [row["extra_info"]["task_id"] for row in rows] == ["umwp-ans-512", "umwp-uns-3012"]
    assert rows[0]["extra_info"]["index"] == 512  # the lowest id survives


def test_multi_region_defect_ships_both_sides(tmp_path):
    """Two changed regions no longer disqualify anything: the bare side points at
    nothing and the placeholder block has no anchor to mine around."""
    rows, funnel = _build(tmp_path, MULTI_ANSWERABLE, MULTI_UNANSWERABLE)
    assert funnel["after_defect_certificate_drop"] == 2
    assert funnel["after_option_mining_drop"] == 2
    by_id = _by_id(rows)
    assert by_id["umwp-ans-24"]["extra_info"]["solvable"] is True
    assert by_id["umwp-uns-2524"]["extra_info"]["solvable"] is False
    assert by_id["umwp-uns-2524"]["extra_info"]["deleted_condition_text"] == "14 prepare"
    assert by_id["umwp-uns-2524"]["extra_info"]["perturbed_entity_text"] == "some children make"


def test_token_invisible_sign_flip_ships_with_a_character_level_defect(tmp_path):
    """ "23" -> "-23" has no word region; the bare contract does not need one."""
    rows, funnel = _build(tmp_path, SIGNFLIP_ANSWERABLE, SIGNFLIP_UNANSWERABLE)
    assert funnel["after_defect_certificate_drop"] == 2
    assert funnel["after_option_mining_drop"] == 2
    by_id = _by_id(rows)
    assert set(by_id) == {"umwp-ans-10", "umwp-uns-2510"}
    for task_id in by_id:
        info = by_id[task_id]["extra_info"]
        assert info["deleted_condition_text"] == ""
        assert info["perturbed_entity_text"] == "-"  # the sign, recorded at character level
    assert by_id["umwp-uns-2510"]["extra_info"]["perturbation_type"] == "unrealistic_condition"
    assert by_id["umwp-ans-10"]["extra_info"]["perturbation_type"] == ""


def test_missing_information_label_without_information_loss_still_ships(tmp_path):
    """ "did" -> "will tomorrow" keeps every number; the class no longer drops the row.

    D24 makes the class statistics-only, so the source's category-1 label is kept
    on the bare row even where the artifact cannot prove the text shape (the
    verifier reports those rows as a statistics caveat).
    """
    rows, funnel = _build(tmp_path, NONUMBER_ANSWERABLE, NONUMBER_UNANSWERABLE)
    assert funnel["after_defect_certificate_drop"] == 2
    by_id = _by_id(rows)
    uns = by_id["umwp-uns-2532"]
    assert uns["extra_info"]["error_type"] == "key_information_missing"
    assert uns["extra_info"]["options"] == []
    assert by_id["umwp-ans-32"]["extra_info"]["solvable"] is True


def test_pure_insertion_ships_both_sides_with_an_empty_deletion(tmp_path):
    """The inserted span is not in the answerable member; ``deleted`` is empty."""
    rows, funnel = _build(tmp_path, INSERT_ANSWERABLE, INSERT_UNANSWERABLE)
    assert funnel["after_option_mining_drop"] == 2
    uns = _by_id(rows)["umwp-uns-2503"]
    assert uns["extra_info"]["deleted_condition_text"] == ""
    assert uns["extra_info"]["perturbed_entity_text"] == "less than"
    assert _by_id(rows)["umwp-ans-3"]["extra_info"]["solvable"] is True


def test_answerable_row_without_a_placeholder_block_is_dropped_alone(tmp_path):
    """The answerable side needs k distinct equal-length spans; the bare side does not."""
    rows, funnel = _build(tmp_path, SHORT_ANSWERABLE, SHORT_UNANSWERABLE)
    assert funnel["after_defect_certificate_drop"] == 2
    assert funnel["after_option_mining_drop"] == 1
    assert [row["extra_info"]["task_id"] for row in rows] == ["umwp-uns-9002"]


def test_a_link_to_an_unanswerable_row_is_not_a_pair(tmp_path):
    """``relevant_ids`` must name the answerable member; anything else fails closed."""
    orphan = list(FISH_UNANSWERABLE)
    orphan[5] = [2502]  # 2502 is itself unanswerable
    rows, funnel = _build(tmp_path, BIRDS_ANSWERABLE, BIRDS_UNANSWERABLE, tuple(orphan))
    assert funnel["after_pair_resolution_drop"] == 2
    assert {row["extra_info"]["task_id"] for row in rows} == {"umwp-ans-2", "umwp-uns-2502"}


def test_unknown_category_fails_closed(tmp_path):
    """``None`` means "unlabelled" and is admitted; an unknown code means the file
    changed shape."""
    corrupted = list(BIRDS_UNANSWERABLE)
    corrupted[4] = 9
    rows, funnel = _build(tmp_path, BIRDS_ANSWERABLE, tuple(corrupted))
    assert rows == []
    assert funnel["after_pair_resolution_drop"] == 2
    assert funnel["after_defect_class_drop"] == 0


def test_unlabelled_pair_is_admitted_and_classified_when_provable(tmp_path):
    """Section 4.8's category-less rows: admitted, and the class is derived from
    the text when a refusal rule proves it (here: a number is gone)."""
    unlabelled = list(BIRDS_UNANSWERABLE)
    unlabelled[4] = None
    rows, funnel = _build(tmp_path, BIRDS_ANSWERABLE, tuple(unlabelled))
    assert funnel["after_defect_class_drop"] == 2
    by_id = _by_id(rows)
    uns = by_id["umwp-uns-2502"]
    assert uns["extra_info"]["error_type"] == "key_information_missing"
    assert uns["extra_info"]["perturbation_type"] == "missing_condition"
    assert json.loads(uns["reward_model"]["ground_truth"])["perturbation_type"] == "missing_condition"
    assert by_id["umwp-ans-2"]["extra_info"]["error_type"] == ""
    assert schema.validate_row(uns) == []


def test_unlabelled_pair_without_a_provable_shape_stays_outside_the_table(tmp_path):
    """No source label and no textual proof -> the row ships with no class at all."""
    unlabelled = list(SIGNFLIP_UNANSWERABLE)
    unlabelled[4] = None
    rows, _ = _build(tmp_path, SIGNFLIP_ANSWERABLE, tuple(unlabelled))
    uns = _by_id(rows)["umwp-uns-2510"]
    assert uns["extra_info"]["error_type"] == ""
    assert uns["extra_info"]["perturbation_type"] == ""
    assert json.loads(uns["reward_model"]["ground_truth"])["perturbation_type"] is None
    assert schema.validate_row(uns) == []


def test_ambiguous_partner_drops_the_answerable_row_only(tmp_path):
    """Two unanswerable rows link to one answerable row: its defect is undefined.

    Only the *answerable* side is ambiguous -- "which of the two defects applies to
    me?" -- so it fails closed, while each unanswerable row still knows the one
    answerable question it was derived from.
    """
    second = list(FISH_UNANSWERABLE)
    second[5] = [2]
    rows, funnel = _build(tmp_path, BIRDS_ANSWERABLE, BIRDS_UNANSWERABLE, tuple(second))
    task_ids = {row["extra_info"]["task_id"] for row in rows}
    assert "umwp-ans-2" not in task_ids
    assert {"umwp-uns-2502", "umwp-uns-3611"} <= task_ids
    assert funnel["after_pair_resolution_drop"] == 2


def test_malformed_lines_are_dropped_not_crashed(tmp_path):
    path = tmp_path / adapter.DATA_FILE
    good = json.dumps(_rows(BIRDS_ANSWERABLE)[0])
    path.write_text(
        "\n".join(
            [
                good,
                "this is not json",
                "[]",
                json.dumps({"id": "not-an-int", "question": "x", "answerable": True}),
                json.dumps({"id": 99, "question": "x", "answerable": "yes"}),
                json.dumps({"id": 98, "question": "   ", "answerable": False}),
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    source = adapter.load_source(str(tmp_path))
    assert len(source) == 6  # the unparseable line is kept for counting
    well_formed = [row for row in source if adapter._is_well_formed(row)]
    assert well_formed == [_rows(BIRDS_ANSWERABLE)[0]]

    rows, funnel = adapter.build_rows(str(tmp_path))
    assert rows == []
    assert funnel["raw_rows"] == 6
    assert funnel["after_malformed_drop"] == 1


# ---------------------------------------------------------------------------
# determinism and --limit
# ---------------------------------------------------------------------------


def test_same_seed_same_rows(tmp_path):
    first, _ = _build(tmp_path / "a", *THREE_WAY, seed=0)
    second, _ = _build(tmp_path / "b", *THREE_WAY, seed=0)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_seed_is_recorded_and_membership_is_seed_independent(tmp_path):
    seeded, _ = _build(tmp_path / "a", *THREE_WAY, seed=7)
    unseeded, _ = _build(tmp_path / "b", *THREE_WAY, seed=0)
    assert {row["extra_info"]["seed"] for row in seeded} == {7}
    assert {row["extra_info"]["task_id"] for row in seeded} == {row["extra_info"]["task_id"] for row in unseeded}
    assert json.dumps(seeded, sort_keys=True) != json.dumps(unseeded, sort_keys=True)


def test_limit_interleaves_the_branches(tmp_path):
    rows, funnel = _build(tmp_path, *THREE_WAY, limit=3)
    assert funnel["after_option_mining_drop"] == 6
    assert funnel["after_limit"] == 3
    counter: dict[str, int] = {}
    for row in rows:
        counter[row["extra_info"]["branch"]] = counter.get(row["extra_info"]["branch"], 0) + 1
    assert counter == {adapter.BRANCH_TWO_LAYER: 2, adapter.BRANCH_BARE: 1}
    # a limit must truncate, never rewrite: every kept row is byte-identical
    full, _ = _build(tmp_path / "full", *THREE_WAY)
    assert json.dumps(rows, sort_keys=True) == json.dumps(full[:3], sort_keys=True)


def test_default_emits_the_full_eligible_pool(tmp_path):
    """``limit=None`` (the CLI default) is the full pool; the mix selects the quotas."""
    rows, funnel = _build(tmp_path, *THREE_WAY)
    assert funnel["after_limit"] == funnel["after_option_mining_drop"] == len(rows) == 6
    assert len(_sides(rows)[0]) == 3 and len(_sides(rows)[1]) == 3


def test_limit_above_the_row_count_is_a_no_op(tmp_path):
    rows, funnel = _build(tmp_path, *THREE_WAY, limit=99)
    assert len(rows) == 6
    assert funnel["after_limit"] == 6


def test_limit_zero_emits_nothing(tmp_path):
    rows, funnel = _build(tmp_path, *THREE_WAY, limit=0)
    assert rows == []
    assert funnel["after_limit"] == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_main_writes_a_parquet_and_prints_the_funnel(tmp_path, monkeypatch, capsys):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_source(raw_dir, *THREE_WAY)
    out = tmp_path / "rows.parquet"
    monkeypatch.setattr(sys, "argv", ["umwp_adapter.py", "--raw-dir", str(raw_dir), "--out", str(out), "--limit", "3"])
    adapter.main()
    printed = capsys.readouterr().out
    assert "after_option_mining_drop" in printed
    assert "per branch:" in printed
    assert "per template:" in printed
    assert "per solvable:" in printed
    assert "per perturbation_type:" in printed
    written = schema.read_parquet_rows(str(out))
    assert len(written) == 3
    assert schema.validate_rows(written) is None
    assert {row["extra_info"]["branch"] for row in written} == {adapter.BRANCH_TWO_LAYER, adapter.BRANCH_BARE}


def test_cli_default_writes_the_full_pool(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_source(raw_dir, *THREE_WAY)
    out = tmp_path / "rows.parquet"
    monkeypatch.setattr(sys, "argv", ["umwp_adapter.py", "--raw-dir", str(raw_dir), "--out", str(out)])
    adapter.main()
    written = schema.read_parquet_rows(str(out))
    assert len(written) == 6
    assert schema.validate_rows(written) is None


def test_task_ids_are_stable_and_unique(tmp_path):
    rows, _ = _build(tmp_path, *THREE_WAY)
    task_ids = [row["extra_info"]["task_id"] for row in rows]
    # branches are interleaved (two-layer, bare) so a small --limit keeps both
    assert task_ids == [
        "umwp-ans-2",
        "umwp-uns-2502",
        "umwp-ans-1111",
        "umwp-uns-3611",
        "umwp-ans-1347",
        "umwp-uns-3847",
    ]
    assert len(set(task_ids)) == len(task_ids)
    assert adapter._task_id({"id": 5, "answerable": True}) == "umwp-ans-5"
    assert adapter._task_id({"id": 5, "answerable": False}) == "umwp-uns-5"


# ---------------------------------------------------------------------------
# unit tests for the helpers
# ---------------------------------------------------------------------------


def test_normalise_question_collapses_whitespace():
    assert adapter.normalise_question("  a  b\n c \n\n d  ") == "a b c d"
    assert adapter.normalise_question(None) == ""
    assert adapter.normalise_question("already clean") == "already clean"


def test_word_regions_shapes():
    insert = adapter.word_regions("alpha beta", "gamma of alpha beta")
    assert [(r.tag, r.a_text, r.b_text) for r in insert] == [("insert", "", "gamma of")]

    delete = adapter.word_regions("alpha beta gamma", "alpha gamma")
    assert [(r.tag, r.a_text, r.b_text) for r in delete] == [("delete", "beta", "")]

    replace = adapter.word_regions("alpha beta gamma", "alpha delta gamma")
    assert [(r.tag, r.a_text, r.b_text) for r in replace] == [("replace", "beta", "delta")]

    # a two-sided change is one region, not a delete plus an insert
    merged = adapter.word_regions("alpha beta gamma delta", "alpha epsilon delta")
    assert len(merged) == 1
    assert merged[0].a_text == "beta gamma" and merged[0].b_text == "epsilon"
    # ... but regions separated by an unchanged run stay separate
    assert len(adapter.word_regions("alpha beta gamma delta", "alpha x gamma y")) == 2

    assert adapter.word_regions("same text", "same text") == []


def test_merge_opcodes_joins_adjacent_changed_runs():
    """difflib reports a two-sided change as one ``replace``, so the join is a
    guard rather than a hot path; a split opcode list must still give one region."""
    raw = [
        ("equal", 0, 2, 0, 2),
        ("delete", 2, 4, 2, 2),
        ("insert", 4, 4, 2, 5),
        ("equal", 4, 5, 5, 6),
    ]
    assert adapter._merge_opcodes(raw) == [("delete", 2, 4, 2, 5)]
    # runs that are not contiguous stay separate
    split = [("delete", 0, 1, 0, 0), ("insert", 2, 2, 0, 1)]
    assert adapter._merge_opcodes(split) == split
    assert adapter._merge_opcodes([("equal", 0, 1, 0, 1)]) == []


def test_word_regions_are_whole_words_not_fragments():
    """A repeated word must not drag the alignment into a mid-word span."""
    qa = "Debby bought 264 water bottles. If she drank 15 bottles a day for 11 days."
    qu = "Debby bought 264 water bottles. If she drank 15 bottles a day for a few days."
    regions = adapter.word_regions(qa, qu)
    assert [(r.a_text, r.b_text) for r in regions] == [("11", "a few")]
    assert all(region.a_text == region.a_text.strip() for region in regions)


def test_residual_tokens_rejects_a_region_that_is_not_the_whole_difference():
    a, b = "one two three", "one two three"
    assert adapter._residual_tokens(a, b, []) is True
    regions = adapter.word_regions("one two three", "one three")
    assert adapter._residual_tokens("one two three", "one three", regions) is True


def test_residual_tokens_accepts_a_token_invisible_pair():
    """No regions means "the token sequences are equal" -- the sign-flip case."""
    qa = "Bobby ate 23 pieces of candy."
    qu = "Bobby ate -23 pieces of candy."
    assert adapter.word_regions(qa, qu) == []
    assert adapter._residual_tokens(qa, qu, []) is True
    assert adapter._residual_tokens("one two", "one three", []) is False


def test_character_defect_records_a_token_invisible_edit():
    qa = "Bobby ate 23 pieces of candy."
    qu = "Bobby ate -23 pieces of candy."
    assert adapter.character_defect(qa, qu) == ("", "-")
    assert adapter.defect_texts(qa, qu, []) == ("", "-")
    # the word-region record wins whenever there is one
    regions = adapter.word_regions("alpha beta gamma", "alpha delta gamma")
    assert adapter.defect_texts("alpha beta gamma", "alpha delta gamma", regions) == ("beta", "delta")


def test_question_clause():
    assert adapter._question_clause("A has 3. How many?") == " How many?"
    assert adapter._question_clause("How many more?") == "How many more?"
    assert adapter._question_clause("no question here.") == ""


def test_certify_refusal_branches():
    qa = "A cat eats 5 fish and sleeps. How many fish does it eat?"
    qu = "A cat eats and sleeps. How many fish does it eat?"
    regions = adapter.word_regions(qa, qu)
    # category 1: the number 5 is gone from the question
    assert adapter._certify_refusal(qa, qu, regions, 1) is True
    # category 5: the removal is outside the question clause
    assert adapter._certify_refusal(qa, qu, regions, 5) is False
    # category 5: a removal inside the clause counts
    inside_qa = "Bob has 4 apples. How many apples does Bob have?"
    inside_qu = "Bob has 4 apples. How many does Bob have?"
    assert adapter._certify_refusal(inside_qa, inside_qu, adapter.word_regions(inside_qa, inside_qu), 5) is True
    # category 1 with nothing numeric lost at all
    plain_qa = "Bob has some apples. How many apples does Bob have?"
    plain_qu = "Bob has apples. How many apples does Bob have?"
    assert adapter._certify_refusal(plain_qa, plain_qu, adapter.word_regions(plain_qa, plain_qu), 1) is False
    # category 5 with no interrogative sentence to compare against
    no_mark_qa = "Bob has 4 apples. how many apples does Bob have"
    no_mark_qu = "Bob has 4 apples. how many does Bob have"
    assert adapter._certify_refusal(no_mark_qa, no_mark_qu, adapter.word_regions(no_mark_qa, no_mark_qu), 5) is False
    # an untouched question has no certificate at all
    assert adapter._certify_refusal(qa, qa, [], 1) is False
    assert adapter._certify_refusal(qa, qa, [], 5) is False


def test_mine_placeholder_spans_returns_k_equal_length_spans():
    question = "Bryan has 9 books and 46 magazines in each of his 10 bookshelves. How many magazines does he have?"
    spans = adapter.mine_placeholder_spans(question, 3, rng=random.Random(0))
    assert spans is not None
    assert len(spans) == 3 and len(set(spans)) == 3
    assert all(span in question for span in spans)
    assert len({len(schema.words(span)) for span in spans}) == 1
    assert all(any(schema.is_content_word(token) for token in schema.words(span)) for span in spans)
    # deterministic for a fixed seed
    assert adapter.mine_placeholder_spans(question, 3, rng=random.Random(0)) == spans


def test_mine_placeholder_spans_excludes_the_gold_answer():
    question = "A basket holds 12 apples and 7 oranges. How many apples are in the basket?"
    spans = adapter.mine_placeholder_spans(question, 3, rng=random.Random(0), exclude=("12",))
    assert spans is not None
    assert "12" not in spans


def test_mine_placeholder_spans_needs_k_distinct_spans():
    assert adapter.mine_placeholder_spans("What is 2?", 3, rng=random.Random(0)) is None
    assert adapter.mine_placeholder_spans("", 3, rng=random.Random(0)) is None


def test_answer_text_reads_only_a_real_number():
    assert adapter._answer_text({"answer": [1.0]}) == "1.0"
    assert adapter._answer_text({"answer": [30]}) == "30"
    assert adapter._answer_text({"answer": None}) == ""
    # the two stray unanswerable rows carry a bare int; that is not a gold
    assert adapter._answer_text({"answer": 2}) == ""
    assert adapter._answer_text({"answer": []}) == ""
    assert adapter._answer_text({"answer": [True]}) == ""
    assert adapter._answer_text({"answer": [float("inf")]}) == ""
    assert adapter._answer_text({}) == ""


def test_pair_category_prefers_the_label_and_rejects_unknown_codes():
    answerable = _rows(BIRDS_ANSWERABLE)[0]
    unanswerable = _rows(BIRDS_UNANSWERABLE)[0]
    assert adapter._pair_category(answerable, unanswerable) == 2
    assert adapter._pair_category(unanswerable, answerable) == 2
    assert adapter._pair_category(answerable, {**unanswerable, "category": None}) is None
    assert adapter._pair_category(answerable, {**unanswerable, "category": 1}) == 1
    for broken in (9, "2", 2.0, True):
        with pytest.raises(ValueError):
            adapter._pair_category(answerable, {**unanswerable, "category": broken})


def test_is_well_formed_rejects_broken_rows():
    good = _rows(BIRDS_ANSWERABLE)[0]
    assert adapter._is_well_formed(good) is True
    assert adapter._is_well_formed("not a dict") is False
    assert adapter._is_well_formed({**good, "id": "1"}) is False
    assert adapter._is_well_formed({**good, "question": ""}) is False
    assert adapter._is_well_formed({**good, "answerable": 1}) is False
    assert adapter._is_well_formed({key: value for key, value in good.items() if key != "answerable"}) is False


def test_partner_map_flags_ambiguous_links():
    answerable = _rows(BIRDS_ANSWERABLE)[0]
    first = _rows(BIRDS_UNANSWERABLE)[0]
    second = {**first, "id": 2599}
    by_id, partners = adapter._partner_map([answerable, first, second])
    assert set(by_id) == {2, 2502, 2599}
    assert partners == {}  # two claimants -> no partner for either

    _, single = adapter._partner_map([answerable, first])
    assert single[2]["id"] == 2502

    # an unanswerable row with no link at all, or a link that is not a 1-element
    # list, contributes no partner and does not crash
    for broken in (None, 2, [2, 3], []):
        _, partners = adapter._partner_map([answerable, {**first, "relevant_ids": broken}])
        assert partners == {}
    # a link that names another unanswerable row is not a pair
    other = {**first, "id": 2599, "relevant_ids": [4]}
    _, partners = adapter._partner_map([answerable, {**first, "relevant_ids": [2599]}, other])
    assert partners == {}


def test_interleave_by_branch_round_robins():
    rows = [
        {"extra_info": {"branch": "a"}},
        {"extra_info": {"branch": "a"}},
        {"extra_info": {"branch": "b"}},
    ]
    assert [row["extra_info"]["branch"] for row in adapter._interleave_by_branch(rows)] == ["a", "b", "a"]


def test_breakdown_counts_and_sorts():
    rows = [{"extra_info": {"branch": "b"}}, {"extra_info": {"branch": "a"}}, {"extra_info": {"branch": "a"}}]
    assert adapter._breakdown(rows, lambda row: row["extra_info"]["branch"]) == {"a": 2, "b": 1}


@pytest.mark.parametrize("category,expected", sorted(adapter.CATEGORIES.items()))
def test_category_table_is_complete(category, expected):
    error_type, perturbation = expected
    assert error_type and perturbation
    assert perturbation in schema.PERTURBATION_TYPES or perturbation == "question_missing"
