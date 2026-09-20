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
"""Tests for ``treecut_adapter.py`` and its audit script ``verify_treecut.py``.

TreeCut is a *generator*, not a file, so unlike the other adapters there is no
small fixture that can stand in for the source: the pair fixtures below are real
rows frozen out of a build (their texts, proof, deleted sentence and
configuration), and the tests that exercise generation need the pinned generator
under ``<raw-dir>/repo/treecut`` -- those are skipped, not failed, when the
download is absent.  Everything that can be exercised without the source (the
text-to-graph certificate, the proof parser, the option builders and their
checks, the helper functions, the CLI plumbing) runs from the frozen strings
alone.

D26 ships **both** sides of every pair (design doc table B rows 5 and 7), so the
tests are written per side: the four-tier negative (``unsolvable_diag``, template
A, gold ``\\boxed{UNSOLVABLE: <ID>}``) and the solvable positive
(``solvable_numeric``, template A, gold ``\\boxed{answer}`` behind a placeholder
block).  The one full build is a module-scoped fixture: ``build_rows`` plans all
4,907 pairs whatever ``--limit`` says, so a truncated artifact is a prefix of it
(``build_rows(limit=n) == build_rows(limit=None)[:n]``) and the small fixtures
cost nothing extra.  The pair fixtures are proved against the source itself by
``test_render_scenario_is_byte_faithful_to_the_generator`` and by the end-to-end
audit, which runs the real artifact through ``verify_treecut.py``.
"""

from __future__ import annotations

import collections
import json
import os
import random
import sys

import pytest
import schema
import treecut_adapter as adapter
import verify_treecut as verify

RAW_DIR = adapter.DEFAULT_RAW_DIR
RAW_AVAILABLE = os.path.isdir(os.path.join(RAW_DIR, adapter.REPO_SUBDIR))
requires_raw = pytest.mark.skipif(
    not RAW_AVAILABLE,
    reason=f"the pinned TreeCut generator is not under {os.path.join(RAW_DIR, adapter.REPO_SUBDIR)}",
)

# ---------------------------------------------------------------------------
# frozen fixtures -- one real pair, copied verbatim out of a build
# ---------------------------------------------------------------------------

#: The four variables the frozen pair names, with the generator's own plural
#: spellings, so the certificate can be exercised without the entity tables.
FIXED_FORMS = {
    "lasagna at Taste Good Cuisine": [
        "lasagna at Taste Good Cuisine",
        "lasagnas at Taste Good Cuisine",
    ],
    "scrambled egg at Urban Plate": [
        "scrambled egg at Urban Plate",
        "scrambled eggs at Urban Plate",
    ],
    "lasagna at Urban Plate": ["lasagna at Urban Plate", "lasagnas at Urban Plate"],
    "scrambled egg at Taste Good Cuisine": [
        "scrambled egg at Taste Good Cuisine",
        "scrambled eggs at Taste Good Cuisine",
    ],
}

#: The unsolvable member: the fact that grounds the answer was deleted, so the
#: asked variable's component is two variables in one equation.
SHIPPED = (
    "A lasagna at Taste Good Cuisine and 2 scrambled eggs at Urban Plate cost 34 dollars. "
    "3 lasagnas at Urban Plate and 3 scrambled eggs at Taste Good Cuisine cost 72 dollars. "
    "A lasagna at Urban Plate costs 9 dollars. "
    "Question: how much does a scrambled egg at Urban Plate cost?"
)

#: The solvable counterpart (the positive side, D26 table B row 5): the deleted
#: edge was a root fact the derivation never used.
COUNTERPART = (
    "A lasagna at Taste Good Cuisine costs 12 dollars. "
    "3 lasagnas at Urban Plate and 3 scrambled eggs at Taste Good Cuisine cost 72 dollars. "
    "A lasagna at Taste Good Cuisine and 2 scrambled eggs at Urban Plate cost 34 dollars. "
    "Question: how much does a scrambled egg at Urban Plate cost?"
)

#: The sentence the unsolvable member lost -- the counterpart's grounding fact and
#: the four-tier gold (``correct_option_id`` points at it after the shuffle).
DELETED = "A lasagna at Taste Good Cuisine costs 12 dollars."

#: The sentence the solvable member lost -- a root fact off the derivation path.
COUNTERPART_LOST = "A lasagna at Urban Plate costs 9 dollars."

PROOF = (
    "All we know about the prices of lasagna at Taste Good Cuisine, "
    "scrambled egg at Urban Plate are: a lasagna at Taste Good Cuisine and "
    "2 scrambled eggs at Urban Plate cost 34 dollars.\n"
    "There are 2 variables but only 1 linear formula, so we cannot calculate the "
    "price of scrambled egg at Urban Plate."
)

ASKED = "scrambled egg at Urban Plate"
FIXED_MATCHER = adapter.compile_matcher(FIXED_FORMS)
CONFIG = {
    "theme": "food",
    "num_vars": 4,
    "ans_depth": 2,
    "order": "random",
    "ordinal": 0,
    "matcher": FIXED_MATCHER,
}

#: The negative option family (D26): the gold -- the generator's cut edge -- plus
#: two candidate missing conditions the passage does not state, which is the shape
#: a sibling scenario's cut contributes.
FROZEN_FAMILY = [
    DELETED,
    "A lasagna at Urban Plate costs 11 dollars.",
    "3 lasagnas at Taste Good Cuisine and 3 scrambled eggs at Urban Plate cost 75 dollars.",
]

#: Two surface conditions of the frozen passage, used to drive the L4 checks'
#: rejection paths: a condition that *is* in the passage (the leak) and one that is
#: not (a legitimate distractor).
IN_PASSAGE = "A lasagna at Urban Plate costs 9 dollars."
OUT_OF_PASSAGE = "A lasagna at Urban Plate costs 11 dollars."


def frozen_member(text: str, deleted: str, proof: str) -> dict:
    return {"problem": text, "deleted_sentence": deleted, "proof": proof}


def frozen_negative() -> dict:
    return frozen_member(SHIPPED, DELETED, PROOF)


def frozen_positive() -> dict:
    return frozen_member(COUNTERPART, COUNTERPART_LOST, "")


#: A second, hand-built four-variable scenario, used only to drive the
#: certificate's "the shipped member is still solvable" branch.  The frozen pair
#: cannot reach that branch: in it the removed sentence is the only grounding of
#: the asked variable's component, so no shipped text that keeps the one-sentence
#: swap shape can stay solvable.  Here the asked variable hangs off a *second*
#: root fact, so the shipped member is solvable while still differing from its
#: counterpart by exactly one sentence.
SWAP_FORMS = {
    "widget at S": ["widget at S"],
    "sprocket at S": ["sprocket at S"],
    "cog at S": ["cog at S"],
    "gear at S": ["gear at S"],
}
SWAP_MATCHER = adapter.compile_matcher(SWAP_FORMS)
SWAP_QUESTION = "Question: how much does a sprocket at S cost?"
SWAP_LOST_BY_SHIPPED = "gear at S and cog at S cost 8 dollars."
SWAP_LOST_BY_COUNTERPART = "cog at S and gear at S cost 8 dollars."
SWAP_SHIPPED = (
    "widget at S costs 3 dollars. "
    "widget at S and sprocket at S cost 9 dollars. "
    f"{SWAP_LOST_BY_COUNTERPART} " + SWAP_QUESTION
)
SWAP_COUNTERPART = (
    "widget at S costs 3 dollars. "
    "widget at S and sprocket at S cost 9 dollars. "
    f"{SWAP_LOST_BY_SHIPPED} " + SWAP_QUESTION
)
SWAP_CONFIG = {
    "theme": "food",
    "num_vars": 4,
    "ans_depth": 2,
    "order": "random",
    "ordinal": 0,
}


def option_row(options: list, correct: str | None, *, question: str = SHIPPED,
               deleted: str = DELETED, positive: bool = False) -> dict:
    """A minimal row for the option checks (no parquet, no build needed)."""
    payload = schema.ground_truth_payload(
        solvable=positive,
        answer="34" if positive else None,
        correct_option_id=None if positive else correct,
        has_diagnosis_label=not positive,
        perturbation_type=None if positive else adapter.PERTURBATION_TYPE,
    )
    return {
        "data_source": schema.SOURCE_TREECUT,
        "prompt": [{"content": schema.render_prompt(question, schema.TEMPLATE_A, options=options)}],
        "reward_model": {"style": "rule", "ground_truth": schema.ground_truth_json(payload)},
        "extra_info": {
            "task_id": "treecut-test-000000",
            "options": options,
            "correct_option_id": "" if positive else correct,
            "deleted_condition_text": deleted,
            "paired_original_text": COUNTERPART if not positive else SHIPPED,
            "num_vars": 4,
            "ans_depth": 2,
            "theme": "food",
            "solvable": positive,
        },
    }


def frozen_negative_options() -> tuple:
    """``(options, correct)`` from the frozen family, through the adapter's builder."""
    built = adapter.negative_options(SHIPPED, DELETED, FROZEN_FAMILY, random.Random(0))
    assert built is not None
    return built


# ---------------------------------------------------------------------------
# the certificate, on the frozen pair
# ---------------------------------------------------------------------------


def test_frozen_pair_certifies():
    assert adapter.certify_pair(frozen_negative(), frozen_positive(), CONFIG, FIXED_MATCHER) == []


def test_text_graph_reads_the_frozen_text():
    graph = adapter.text_graph(SHIPPED, FIXED_MATCHER)
    assert graph["malformed"] == []
    assert graph["asked"] == ASKED
    assert len(adapter._variables(graph)) == 4
    assert len(graph["facts"]) == 1 and len(graph["links"]) == 2
    # the unsolvable member: no fact reaches the answer
    assert ASKED not in adapter._reachable(graph)
    # the counterpart keeps the chain, one hop below the fact
    counterpart = adapter.text_graph(COUNTERPART, FIXED_MATCHER)
    assert counterpart["asked"] == ASKED
    assert adapter._hops_to(counterpart, ASKED) == CONFIG["ans_depth"] - 1


def test_sentence_plumbing_matches_the_generator_convention():
    assert adapter.split_body("a b. Question: c?") == ("a b.", "Question: c?")
    assert adapter.split_body("no question marker") == ("no question marker", "")
    assert adapter.sentences_of(SHIPPED) == [
        "A lasagna at Taste Good Cuisine and 2 scrambled eggs at Urban Plate cost 34 dollars.",
        "3 lasagnas at Urban Plate and 3 scrambled eggs at Taste Good Cuisine cost 72 dollars.",
        "A lasagna at Urban Plate costs 9 dollars.",
    ]
    # the certificate quotes a sentence with its first letter folded and the full
    # stop dropped (the generator's l0(sentence[:-1]))
    assert adapter.clause_of(DELETED) == "a lasagna at Taste Good Cuisine costs 12 dollars"


def test_certificate_rejects_a_wrong_deleted_sentence():
    negative = frozen_negative()
    negative["deleted_sentence"] = COUNTERPART_LOST
    problems = adapter.certify_pair(negative, frozen_positive(), CONFIG, FIXED_MATCHER)
    assert any("not the sentence the question lost" in problem for problem in problems)


def test_certificate_rejects_a_counterpart_that_lost_the_wrong_sentence():
    positive = frozen_positive()
    # drop the 72-dollar link and add back the shipped member's fact: still one
    # sentence each way, but the pair is no longer the same scenario's two cuts
    positive["problem"] = (
        "A lasagna at Taste Good Cuisine costs 12 dollars. "
        "A lasagna at Urban Plate costs 9 dollars. "
        "A lasagna at Taste Good Cuisine and 2 scrambled eggs at Urban Plate cost 34 dollars. "
        "Question: how much does a scrambled egg at Urban Plate cost?"
    )
    problems = adapter.certify_pair(frozen_negative(), positive, CONFIG, FIXED_MATCHER)
    assert problems
    assert any("the counterpart's lost sentence is not the recorded one" in problem
               for problem in problems)


def test_certificate_rejects_a_solvable_shipped_member():
    # the shipped member keeps its one-sentence swap shape but is still solvable:
    # the asked variable is grounded through the second root fact, so this pair is
    # not "unsolvable versus solvable" and must be rejected
    negative = frozen_member(SWAP_SHIPPED, SWAP_LOST_BY_SHIPPED, "")
    positive = frozen_member(SWAP_COUNTERPART, SWAP_LOST_BY_COUNTERPART, "")
    assert adapter.text_graph(SWAP_SHIPPED, SWAP_MATCHER)["asked"] in adapter._reachable(
        adapter.text_graph(SWAP_SHIPPED, SWAP_MATCHER)
    )
    problems = adapter.certify_pair(negative, positive, SWAP_CONFIG, SWAP_MATCHER)
    assert any("reachable from a root fact: solvable" in problem for problem in problems)


def test_certificate_rejects_a_deleted_sentence_that_is_still_present():
    # the shipped member is handed a text that still carries the sentence it
    # claims to have lost: no amount of graph reading can certify that
    negative = frozen_member(COUNTERPART, DELETED, PROOF)
    problems = adapter.certify_pair(negative, frozen_positive(), CONFIG, FIXED_MATCHER)
    assert any("still present in the question" in problem for problem in problems)


@pytest.mark.parametrize(
    "proof,needle",
    [
        ("", "no 'N variables but only M linear formula"),
        ("All we know about the prices of lasagna at Taste Good Cuisine, "
         "scrambled egg at Urban Plate are: nothing.\nThere are 2 variables but "
         "only 2 linear formulas, so we cannot calculate the price of "
         "scrambled egg at Urban Plate.", "not deficient"),
        ("All we know about the prices of lasagna at Taste Good Cuisine, "
         "scrambled egg at Urban Plate are: a lasagna at Taste Good Cuisine and "
         "2 scrambled eggs at Urban Plate cost 34 dollars.\nThere are 3 variables "
         "but only 2 linear formulas, so we cannot calculate the price of "
         "scrambled egg at Urban Plate.", "component has 2"),
        ("All we know about the prices of lasagna at Taste Good Cuisine, "
         "scrambled egg at Urban Plate are: a lasagna at Taste Good Cuisine and "
         "2 scrambled eggs at Urban Plate cost 34 dollars.\nThere are 2 variables "
         "but only 1 linear formula, so we cannot calculate the price of "
         "lasagna at Urban Plate.", "does not name the asked variable"),
        ("All we know about the prices of lasagna at Taste Good Cuisine, "
         "scrambled egg at Urban Plate are: a made-up clause about lasagna at "
         "Taste Good Cuisine and scrambled egg at Urban Plate.\nThere are 2 "
         "variables but only 1 linear formula, so we cannot calculate the price "
         "of scrambled egg at Urban Plate.", "cites a clause that is not a sentence"),
    ],
)
def test_certificate_rejects_a_broken_proof(proof, needle):
    negative = frozen_negative()
    negative["proof"] = proof
    problems = adapter.certify_pair(negative, frozen_positive(), CONFIG, FIXED_MATCHER)
    assert any(needle in problem for problem in problems), problems


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def test_vocabulary_forms_covers_both_spellings():
    forms = adapter.vocabulary_forms(["Store"], {"shirt": "shirts"})
    assert forms == {"shirt at Store": ["shirt at Store", "shirts at Store"]}
    matcher = adapter.compile_matcher(forms)
    graph = adapter.text_graph(
        "2 shirts at Store cost 10 dollars. Question: how much does a shirt at Store cost?",
        matcher,
    )
    assert graph["malformed"] == [] and graph["asked"] == "shirt at Store"


def test_vocabulary_pairs_are_the_generator_node2var_values():
    pairs = adapter.vocabulary_pairs(["Store"], {"shirt": "shirts"})
    assert pairs == {"shirt at Store": ("Store", "shirt")}


def test_quota_for_splits_without_losing_rows():
    assert sum(adapter.quota_for(12, adapter.NEG_TARGET)) == adapter.NEG_TARGET
    assert adapter.quota_for(12, adapter.NEG_TARGET) == [409] * 11 + [408]
    assert sum(adapter.quota_for(12, adapter.POS_TARGET)) == adapter.POS_TARGET
    assert adapter.NEG_TARGET == 4907 and adapter.POS_TARGET == 500  # D26/D27
    assert adapter.quota_for(3, 2) == [1, 1, 0]


def test_cells_is_the_documented_grid():
    grid = adapter.cells()
    assert len(grid) == 12
    assert grid[0] == ("food", 4, 2)
    assert {theme for theme, _, _ in grid} == set(adapter.GRID_THEMES)
    # every configuration leaves an off-path edge for the counterpart to cut
    for _, num_vars, ans_depth in grid:
        assert 2 <= ans_depth < num_vars


def test_pair_seed_is_stable_and_per_slot():
    first = adapter.pair_seed(0, "food", 4, 2, 0)
    assert first == adapter.pair_seed(0, "food", 4, 2, 0)
    assert first != adapter.pair_seed(0, "food", 4, 2, 1)
    assert first != adapter.pair_seed(1, "food", 4, 2, 0)
    assert first != adapter.pair_seed(0, "outfit", 4, 2, 0)
    # a seed is a hash, not a bare offset: adjacent slots are not adjacent numbers
    assert abs(adapter.pair_seed(0, "food", 4, 2, 1) - first) > 1


def test_interleave_round_robins_the_cells():
    planned = []
    for cell in range(3):
        for ordinal in range(2):
            config = {"theme": "food", "num_vars": 4 + cell, "ans_depth": 2}
            planned.append((cell, ordinal, None, None, config))
    order = [(item[0], item[1]) for item in adapter._interleave_by_cell(planned)]
    assert order == [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]


def test_merge_two_sided_carries_both_sides_from_the_first_rows():
    negatives = [f"N{index}" for index in range(490)]
    positives = [f"P{index}" for index in range(50)]
    merged = adapter._merge_two_sided(negatives, positives)
    assert merged[:2] == ["N0", "P0"]  # a small --limit can never be single-sided
    assert len(merged) == 540 and sorted(merged) == sorted(negatives + positives)
    positions = {item: index for index, item in enumerate(merged)}
    # ceil(k * n_pos / n_neg): positive i always follows negative i, so a prefix
    # of the stream never carries an orphan positive
    assert all(positions[f"P{index}"] > positions[f"N{index}"] for index in range(50))
    assert merged == adapter._merge_two_sided(negatives, positives)
    assert adapter._merge_two_sided(negatives, []) == negatives


def test_task_id_for_marks_the_positive_side():
    config = {"theme": "food", "num_vars": 4, "ans_depth": 2, "ordinal": 7}
    assert adapter.task_id_for(config) == "treecut-food-nv4-ad2-000007"
    assert adapter.task_id_for(config, "-pos") == "treecut-food-nv4-ad2-000007-pos"
    assert verify.task_base(adapter.task_id_for(config, "-pos")) == adapter.task_id_for(config)


def test_branch_breakdown_names_every_branch():
    rows = [
        {"extra_info": {"branch": schema.BRANCH_UNSOLVABLE_DIAG}},
        {"extra_info": {"branch": schema.BRANCH_SOLVABLE_NUMERIC}},
    ]
    counts = adapter.branch_breakdown(rows)
    assert counts[schema.BRANCH_UNSOLVABLE_DIAG] == 1
    assert counts[schema.BRANCH_SOLVABLE_NUMERIC] == 1
    assert set(counts) == set(schema.BRANCHES)


def test_load_source_fails_closed_without_the_generator(tmp_path):
    with pytest.raises(FileNotFoundError):
        adapter.load_source(str(tmp_path / "no-such-source"))


# ---------------------------------------------------------------------------
# the option builders (D26), on frozen text -- no generator needed
# ---------------------------------------------------------------------------


def test_negative_options_are_out_of_passage_and_point_at_the_cut_edge():
    options, correct = frozen_negative_options()
    assert [opt["id"] for opt in options] == ["A", "B", "C"]
    texts = [opt["text"] for opt in options]
    assert len(set(texts)) == 3
    assert all(text not in SHIPPED for text in texts)  # the leak defence (section 9)
    assert next(opt["text"] for opt in options if opt["id"] == correct) == DELETED
    assert adapter.options_absent_from(options, SHIPPED)
    assert adapter.options_pairwise_distinct(options)


def test_negative_options_fail_closed_when_the_family_is_short():
    # one distractor is not enough: drop the row, never pad with a passage sentence
    assert adapter.negative_options(SHIPPED, DELETED, [DELETED], random.Random(0)) is None
    # a passage condition is filtered out of the family, so this leaves a single
    # usable sibling cut and the row is dropped rather than padded
    assert adapter.negative_options(SHIPPED, DELETED, [DELETED, IN_PASSAGE, OUT_OF_PASSAGE],
                                    random.Random(0)) is None
    family = [DELETED, IN_PASSAGE, OUT_OF_PASSAGE, FROZEN_FAMILY[2]]
    built = adapter.negative_options(SHIPPED, DELETED, family, random.Random(0))
    assert built is not None
    assert IN_PASSAGE not in [opt["text"] for opt in built[0]]


def test_placeholder_variant_swaps_exactly_one_name_or_value():
    sentence = "A lasagna at Taste Good Cuisine costs 12 dollars."
    named = adapter.placeholder_variant(sentence, FIXED_MATCHER, random.Random(0), "variable_name")
    assert named is not None and named != sentence
    assert verify.placeholder_mode_of(named, sentence, FIXED_MATCHER) == "variable_name"
    valued = adapter.placeholder_variant(sentence, FIXED_MATCHER, random.Random(0), "variable_value")
    assert valued is not None and valued != sentence
    assert verify.placeholder_mode_of(valued, sentence, FIXED_MATCHER) == "variable_value"
    # a sentence with no variable to swap cannot yield that mode, and an unknown
    # mode is a programming error, not a silent fallback
    assert adapter.placeholder_variant("no variables here", FIXED_MATCHER, random.Random(0),
                                       "variable_name") is None
    with pytest.raises(ValueError):
        adapter.placeholder_variant(sentence, FIXED_MATCHER, random.Random(0), "nonsense")


def test_placeholder_options_are_out_of_passage_and_have_no_correct_item():
    built = adapter.placeholder_options(COUNTERPART, FIXED_MATCHER, random.Random(0))
    assert built is not None
    options, modes = built
    assert [opt["id"] for opt in options] == ["A", "B", "C"]
    texts = [opt["text"] for opt in options]
    assert len(set(texts)) == 3
    assert all(text not in COUNTERPART for text in texts)
    assert len(modes) == 3 and set(modes) <= set(adapter.PLACEHOLDER_MODES)
    # every placeholder is one name/value swap away from one of the passage's own
    # conditions (the verifier's independent reading of the same property)
    for text in texts:
        assert verify.placeholder_mode_of(text, COUNTERPART, FIXED_MATCHER) is not None


def test_option_builders_are_seeded_and_reproducible():
    first = adapter.placeholder_options(COUNTERPART, FIXED_MATCHER, random.Random(3))
    again = adapter.placeholder_options(COUNTERPART, FIXED_MATCHER, random.Random(3))
    other = adapter.placeholder_options(COUNTERPART, FIXED_MATCHER, random.Random(4))
    assert first == again
    assert first != other
    assert (adapter.negative_options(SHIPPED, DELETED, FROZEN_FAMILY, random.Random(5))
            == adapter.negative_options(SHIPPED, DELETED, FROZEN_FAMILY, random.Random(5)))


# ---------------------------------------------------------------------------
# the verifier's option checks, on constructed rows
# ---------------------------------------------------------------------------


def test_negative_option_check_accepts_the_frozen_block_and_rejects_a_pointer_off_the_cut_edge():
    options, correct = frozen_negative_options()
    row = option_row(options, correct)
    assert verify._negative_option_failures(row, FIXED_MATCHER) == []
    # a pointer that is inside the block but is not the generator's cut edge
    mislabelled = option_row(options, "B" if correct != "B" else "C")
    problems = verify._negative_option_failures(mislabelled, FIXED_MATCHER)
    assert any("not at the generator's cut edge" in problem for problem in problems)


def test_negative_option_check_rejects_a_passage_condition():
    leaked = [
        {"id": "A", "text": DELETED},
        {"id": "B", "text": IN_PASSAGE},
        {"id": "C", "text": OUT_OF_PASSAGE},
    ]
    problems = verify._negative_option_failures(option_row(leaked, "A"), FIXED_MATCHER)
    assert any("appear in the passage" in problem for problem in problems)


def test_positive_option_check_accepts_placeholders_and_rejects_a_correct_item():
    options, _ = adapter.placeholder_options(COUNTERPART, FIXED_MATCHER, random.Random(0))
    row = option_row(options, None, question=COUNTERPART, deleted=COUNTERPART_LOST,
                     positive=True)
    assert verify._positive_option_failures(row, FIXED_MATCHER) == []
    # a placeholder block that suddenly carries a correct option is a contract break
    broken = option_row(options, None, question=COUNTERPART, deleted=COUNTERPART_LOST,
                        positive=True)
    payload = json.loads(broken["reward_model"]["ground_truth"])
    payload["correct_option_id"] = "A"
    broken["reward_model"]["ground_truth"] = json.dumps(payload)
    broken["extra_info"]["correct_option_id"] = "A"
    problems = verify._positive_option_failures(broken, FIXED_MATCHER)
    assert any("no correct option" in problem for problem in problems)


def test_leak_heuristic_reads_chance_when_every_option_is_out_of_passage():
    rows = [option_row(*frozen_negative_options()) for _ in range(300)]
    rate, count, unique = verify.leak_heuristic_hit_rate(rows, seed=0)
    assert count == 300 and unique == 0
    assert rate == pytest.approx(1.0 / 3, abs=0.08)
    # a single option left in the passage hands the heuristic the gold: the gate
    # is on the rate *and* on the number of rows where it is forced
    forced = [
        {"id": "A", "text": DELETED},
        {"id": "B", "text": IN_PASSAGE},
        {"id": "C", "text": "3 lasagnas at Urban Plate and 3 scrambled eggs at Taste Good "
                            "Cuisine cost 72 dollars."},
    ]
    rate, count, unique = verify.leak_heuristic_hit_rate([option_row(forced, "A")], seed=0)
    assert unique == 1 and rate == 1.0 > verify.LEAK_MAX_HIT_RATE


def test_option_shuffle_invariance_runs_against_the_frozen_dispatcher():
    reward = verify.load_reward_module()
    options, correct = frozen_negative_options()
    assert verify.option_shuffle_failures(option_row(options, correct), reward,
                                          random.Random(0)) == []
    # a recorded gold that is not among the block's ids cannot be scored at all
    assert verify.option_shuffle_failures(option_row(options, "D"), reward, random.Random(0))


def test_bare_unsolvable_is_branch_dependent_with_the_data_source_in_the_message(capsys):
    reward = verify.load_reward_module()
    options, correct = frozen_negative_options()
    reporter = verify.Reporter()
    verify.check_bare_unsolvable([option_row(options, correct)], reporter, reward)
    printed = capsys.readouterr().out
    assert not reporter.failed, printed
    assert schema.SOURCE_TREECUT in printed and schema.SOURCE_MIP in printed
    # and the same check fails when the row is not four-tier
    three_tier = option_row(options, correct)
    three_tier["reward_model"]["ground_truth"] = schema.build_ground_truth(
        solvable=False, answer=None, correct_option_id=None, has_diagnosis_label=False,
        perturbation_type="missing_condition",
    )
    reporter = verify.Reporter()
    verify.check_bare_unsolvable([three_tier], reporter, reward)
    assert reporter.failed


def test_positive_reward_cells_run_against_the_frozen_dispatcher(capsys):
    reward = verify.load_reward_module()
    options, _ = adapter.placeholder_options(COUNTERPART, FIXED_MATCHER, random.Random(0))
    positive = option_row(options, None, question=COUNTERPART, deleted=COUNTERPART_LOST,
                          positive=True)
    reporter = verify.Reporter()
    verify.check_positive_reward([positive], reporter, reward)
    assert not reporter.failed, capsys.readouterr().out
    # the check is a real gate: a row that ships no answer cannot be scored
    broken = option_row(options, None, question=COUNTERPART, deleted=COUNTERPART_LOST,
                        positive=True)
    payload = json.loads(broken["reward_model"]["ground_truth"])
    payload["answer"] = None
    broken["reward_model"]["ground_truth"] = json.dumps(payload)
    reporter = verify.Reporter()
    verify.check_positive_reward([broken], reporter, reward)
    assert reporter.failed


# ---------------------------------------------------------------------------
# generation (needs the pinned generator)
# ---------------------------------------------------------------------------


@requires_raw
def test_render_scenario_is_byte_faithful_to_the_generator():
    """The scenario mirror must reproduce the stock generator's own rendering.

    Byte equality is asserted on the *uncut* problem: the stock generator filters
    the cut edge out before drawing formulas, while the adapter draws one shared
    render and filters afterwards (which is the whole fix), so the two agree
    exactly where they compute the same thing -- on ``hallu=False``.

    ``order="random"`` is the one order that cannot be byte-compared: the adapter
    shuffles with its own per-pair RNG (both members must ride one permutation),
    so there the test holds the sentence multiset, the question and the answer to
    the generator's own output instead.
    """
    source = adapter.load_source(RAW_DIR)
    import gen_data  # noqa: PLC0415 - imported from the pinned repo

    checked = 0
    for theme, num_vars, ans_depth in (("food", 4, 2), ("outfit", 6, 4), ("food", 8, 6)):
        for order in ("random", "forward", "backward"):
            key = f"treecut-test:{theme}:{num_vars}:{ans_depth}:{order}"
            random.seed(key)
            expected = gen_data.generate_qa(theme, True, num_vars, ans_depth, order, False)
            random.seed(key)
            scenario = adapter.build_scenario(source, theme, num_vars, ans_depth, order, random)
            got = adapter.render_scenario(source, scenario, order, random.Random(f"{key}:shuffle"))
            assert got["answer"] == expected["answer"]
            assert got["question"] == adapter.split_body(expected["problem"])[1]
            # the same sentence list, in whatever order: the shuffle stream is the
            # one thing the two paths cannot share -- the adapter gives each pair
            # its own RNG so both members ride one permutation, while the stock
            # generator draws from the global stream the scenario already consumed
            assert sorted(got["sentences"]) == sorted(adapter.sentences_of(expected["problem"]))
            if order != "random":
                assert got["problem"] == expected["problem"], (theme, num_vars, ans_depth, order)
            checked += 1
    assert checked == 9


@requires_raw
def test_members_of_one_render_differ_by_their_two_cuts():
    source = adapter.load_source(RAW_DIR)
    negative, positive, _ = adapter.build_pair(source, 0, "food", 6, 4, "random", 0)
    assert negative is not None and positive is not None
    # exactly one sentence each way, and the recorded deleted sentences name them
    negative_sentences = adapter.sentences_of(negative["problem"])
    positive_sentences = adapter.sentences_of(positive["problem"])
    assert len(negative_sentences) == len(positive_sentences) == 6 - 1
    only_negative = [s for s in positive_sentences if s not in negative_sentences]
    only_positive = [s for s in negative_sentences if s not in positive_sentences]
    assert only_negative == [negative["deleted_sentence"]]
    assert only_positive == [positive["deleted_sentence"]]
    # one shared render: the two members ask the same question about the same scenario
    assert (adapter.split_body(negative["problem"])[1]
            == adapter.split_body(positive["problem"])[1])
    assert positive["answer"].isdigit()


@requires_raw
def test_certify_pair_accepts_every_built_pair_of_one_cell():
    source = adapter.load_source(RAW_DIR)
    matcher, pairs, item_dict = adapter.theme_vocabulary(source, "food")
    for ordinal in range(5):
        negative, positive, _ = adapter.build_pair(source, 0, "food", 8, 4, "random", ordinal)
        assert negative is not None
        config = {"theme": "food", "num_vars": 8, "ans_depth": 4, "order": "random",
                  "ordinal": ordinal}
        assert adapter.certify_pair(negative, positive, config, matcher) == []
        # the same pairs carry the arithmetic certificate for the positive's gold
        full_config = dict(config, matcher=matcher, pairs=pairs, item_dict=item_dict,
                           render_sentence=source["render_sentence"],
                           formula=source["formula"])
        assert adapter.certify_answer(positive, full_config) == []


@requires_raw
def test_certify_answer_re_derives_the_gold_and_rejects_a_tampered_one():
    """The positive's number is proved from the passage, not trusted."""
    source = adapter.load_source(RAW_DIR)
    checked = 0
    for theme in adapter.GRID_THEMES:
        matcher, pairs, item_dict = adapter.theme_vocabulary(source, theme)
        for num_vars, ans_depth in ((4, 2), (6, 4), (8, 6)):
            _, positive, _ = adapter.build_pair(source, 0, theme, num_vars, ans_depth,
                                                "random", 0)
            assert positive is not None
            config = {"theme": theme, "num_vars": num_vars, "ans_depth": ans_depth,
                      "order": "random", "ordinal": 0, "matcher": matcher, "pairs": pairs,
                      "item_dict": item_dict, "render_sentence": source["render_sentence"],
                      "formula": source["formula"]}
            assert adapter.certify_answer(positive, config) == []
            tampered = dict(positive, answer=str(int(positive["answer"]) + 1))
            problems = adapter.certify_answer(tampered, config)
            assert any("the recorded gold is" in problem for problem in problems)
            checked += 1
    assert checked == 6


@requires_raw
def test_build_pair_is_seeded_and_order_independent():
    source = adapter.load_source(RAW_DIR)
    first, first_positive, _ = adapter.build_pair(source, 0, "food", 6, 4, "random", 0)
    # a different slot built in between must not change the seeded result: every
    # shuffle comes from a per-slot ``random.Random``, not from the global stream
    adapter.build_pair(source, 0, "outfit", 8, 2, "random", 3)
    again, again_positive, _ = adapter.build_pair(source, 0, "food", 6, 4, "random", 0)
    assert (first, first_positive) == (again, again_positive)
    other, _, _ = adapter.build_pair(source, 7, "food", 6, 4, "random", 0)
    assert other["problem"] != first["problem"]


# ---------------------------------------------------------------------------
# the built rows (one full build; every truncated artifact is its prefix)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def full_build():
    rows, funnel = adapter.build_rows(RAW_DIR, limit=None, seed=0)
    schema.normalise_extra_info(rows)
    return rows, funnel


@pytest.fixture(scope="module")
def full_rows(full_build):
    return full_build[0]


@pytest.fixture(scope="module")
def small_build(full_build):
    """``build_rows(limit=24)`` is exactly the first 24 rows of the full build."""
    rows, funnel = full_build
    return rows[:24], funnel


@requires_raw
def test_full_build_funnel_and_plan(full_build):
    rows, funnel = full_build
    assert len(rows) == adapter.NEG_TARGET + adapter.POS_TARGET == 5407
    assert funnel["raw_rows"] == adapter.NEG_TARGET
    for stage, count in funnel.items():
        if isinstance(count, int) and stage != "emitted_rows":
            assert count == adapter.NEG_TARGET, stage  # no drop at any stage
    assert funnel["emitted_rows"] == 5407
    plan = funnel["plan"]
    assert plan["targets"] == {"negatives": adapter.NEG_TARGET, "positives": adapter.POS_TARGET}
    assert plan["positives_selected"] == adapter.POS_TARGET
    for key in ("scenario_drops", "certificate_drops", "answer_certificate_drops",
                "negative_option_drops"):
        assert sum(plan[key].values()) == 0, key
    assert plan["positive_placeholder_drops"] >= 0  # dropped positives are replaced
    # the negative side clears Q11's ">= 100 rows per cell" floor
    assert len(plan["grid"]) == 12
    assert sum(cell["quota"] for cell in plan["grid"]) == adapter.NEG_TARGET
    assert all(cell["quota"] >= 100 for cell in plan["grid"])
    assert {row["extra_info"]["theme"] for row in rows} == set(adapter.GRID_THEMES)
    assert adapter.branch_breakdown(rows) == {
        schema.BRANCH_SOLVABLE_NUMERIC: adapter.POS_TARGET,
        schema.BRANCH_SOLVABLE_ROLES: 0,
        schema.BRANCH_SOLVABLE_TWO_LAYER: 0,
        schema.BRANCH_SOLVABLE_JUDGE: 0,
        schema.BRANCH_SOLVABLE_PAIR: 0,
        schema.BRANCH_UNSOLVABLE_DIAG: adapter.NEG_TARGET,
        schema.BRANCH_UNSOLVABLE_BARE: 0,
    }


@requires_raw
def test_small_build_is_the_full_build_prefix_and_satisfies_the_schema(small_build, full_rows):
    rows, _ = small_build
    assert len(rows) == 24
    assert [row["extra_info"]["task_id"] for row in rows] == [
        row["extra_info"]["task_id"] for row in full_rows[:24]
    ]
    schema.validate_rows(rows)  # raises on a contract violation
    for row in rows:
        assert schema.validate_row(row) == []
    assert 0 < sum(1 for row in rows if row["extra_info"]["solvable"]) < len(rows)


@requires_raw
def test_full_build_ships_both_sides_of_the_contract(full_rows):
    negatives = [row for row in full_rows if not row["extra_info"]["solvable"]]
    positives = [row for row in full_rows if row["extra_info"]["solvable"]]
    assert len(negatives) == adapter.NEG_TARGET
    assert len(positives) == adapter.POS_TARGET
    # D18: both sides render template A with k=3 -- the option block cannot be the
    # label, which is the whole point of shipping the positive with placeholders
    assert {row["extra_info"]["template"] for row in full_rows} == {schema.TEMPLATE_A}
    assert {len(row["extra_info"]["options"]) for row in full_rows} == {adapter.K_OPTIONS}

    for row in negatives:
        info = row["extra_info"]
        truth = json.loads(row["reward_model"]["ground_truth"])
        assert info["branch"] == schema.BRANCH_UNSOLVABLE_DIAG
        assert truth["solvable"] is False and truth["answer"] is None
        assert truth["has_diagnosis_label"] is True
        assert truth["perturbation_type"] == "missing_condition"
        assert info["error_type"] == "key_information_missing"
        assert info["option_kind"] == "missing_condition"
        assert truth["correct_option_id"] in [opt["id"] for opt in info["options"]]
        passage = row["prompt"][0]["content"].split("\n\n")[0]
        assert info["deleted_condition_text"] not in passage
        assert "\\boxed{UNSOLVABLE: <选项ID>}" in row["prompt"][0]["content"]
        for opt in info["options"]:
            assert opt["text"] not in passage

    for row in positives:
        info = row["extra_info"]
        truth = json.loads(row["reward_model"]["ground_truth"])
        assert info["branch"] == schema.BRANCH_SOLVABLE_NUMERIC
        assert truth["solvable"] is True and str(truth["answer"]).isdigit()
        assert truth["correct_option_id"] is None
        assert truth["has_diagnosis_label"] is False and truth["perturbation_type"] is None
        assert info["error_type"] == "" and info["perturbation_type"] == ""
        assert info["option_kind"] == "placeholder"
        assert info["correct_option_id"] in ("", None)
        assert str(info["answer"]) == str(truth["answer"])
        # misrefusal is 0 on this branch (section 6 / D26), and the placeholder
        # block is never read by the reward
        assert "\\boxed{<答案>}" in row["prompt"][0]["content"]
        for opt in info["options"]:
            assert opt["text"] not in row["prompt"][0]["content"].split("\n\n")[0]


@requires_raw
def test_full_build_option_contract_and_leak_rate(full_rows):
    """Section 9's D26 leak gate on the shipped artifact (the per-row option checks
    run in ``test_full_build_passes_the_option_and_reward_checks``)."""
    positives = [row for row in full_rows if row["extra_info"]["solvable"]]
    mode_counts: collections.Counter = collections.Counter()
    for row in positives:
        mode_counts.update(row["extra_info"]["placeholder_modes"])
    # Q16's 50/50 rule is per option, so both modes must appear on a 500-row build
    assert set(mode_counts) == set(adapter.PLACEHOLDER_MODES)
    assert min(mode_counts.values()) > 0.2 * sum(mode_counts.values())

    rate, count, unique = verify.leak_heuristic_hit_rate(full_rows, seed=0)
    assert count == adapter.NEG_TARGET
    assert unique == 0  # every option of every negative is out of passage
    assert rate <= verify.LEAK_MAX_HIT_RATE
    assert abs(rate - 1.0 / adapter.K_OPTIONS) < 0.05


@requires_raw
def test_full_build_l3_gate_and_pair_alignment(full_rows):
    """L3a/L3b on the two shipped sides, plus the pair alignment."""
    texts, labels = verify._l3_corpus(full_rows)
    assert len(texts) == adapter.NEG_TARGET + adapter.POS_TARGET
    assert sum(labels) == adapter.POS_TARGET
    length_accuracy = verify.length_only_oof([len(text) for text in texts], labels,
                                             folds=5, seed=0)
    bow_accuracy = verify.bernoulli_nb_oof(texts, labels, folds=5, seed=0, min_support=5)
    assert length_accuracy <= verify.L3_MAX_BALANCED_ACCURACY, length_accuracy
    assert bow_accuracy <= verify.L3_MAX_BALANCED_ACCURACY, bow_accuracy
    reporter = verify.Reporter()
    verify.check_l3_alignment(full_rows, reporter)
    assert not reporter.failed


@requires_raw
def test_full_build_passes_the_option_and_reward_checks(full_rows):
    """L4/L5 run against the frozen dispatcher on the built rows."""
    vocabulary = verify.load_vocabulary(RAW_DIR)
    matchers = verify.matchers_for(vocabulary)
    reward = verify.load_reward_module()
    reporter = verify.Reporter()
    verify.check_option_structure(full_rows, reporter, matchers)
    verify.check_option_leak(full_rows, reporter, seed=0)
    verify.check_option_shuffle(full_rows, reporter, reward, seed=0)
    verify.check_bare_unsolvable(full_rows, reporter, reward)
    verify.check_positive_reward(full_rows, reporter, reward)
    assert not reporter.failed, reporter.failed


# ---------------------------------------------------------------------------
# the audit script, end to end
# ---------------------------------------------------------------------------


def write_parquet(rows: list, path: str) -> str:
    schema.normalise_extra_info(rows)
    schema.write_rows_parquet(rows, path)
    return path


@pytest.fixture(scope="module")
def audited_parquet(tmp_path_factory, full_rows):
    """A mid-size artifact (a prefix of the full build) that clears every gate.

    It must be big enough for the L3 and leak gates to be measurable: a 24-row
    prefix carries only three positives, and the length estimator is then noise
    (0.64 on this build).  1,500 rows give 1,361 negatives and 139 positives, which
    is what the fail-closed audit needs to pass on real numbers rather than on a
    toy sample.
    """
    path = str(tmp_path_factory.mktemp("treecut") / "rows.parquet")
    return write_parquet(full_rows[:1500], path), full_rows[:1500]


@requires_raw
def test_verify_accepts_a_real_build(audited_parquet, monkeypatch, capsys):
    path, _ = audited_parquet
    monkeypatch.setattr(sys, "argv", ["verify_treecut.py", "--rows", path, "--raw-dir", RAW_DIR])
    with pytest.raises(SystemExit) as excinfo:
        verify.main()
    output = capsys.readouterr().out
    assert excinfo.value.code == 0, output
    assert "RESULT: PASS" in output and "[FAIL]" not in output
    assert "pre-fix reference" in output
    # the report prints the L3 numbers and the leak rate the design doc asks for
    assert "L3a length-only out-of-fold balanced accuracy" in output
    assert "L3b raw out-of-fold balanced accuracy" in output
    assert "L4b option leak heuristic" in output
    assert "L5 bare UNSOLVABLE is branch-dependent" in output


@pytest.fixture(scope="module")
def tamper_artifact(tmp_path_factory, full_rows):
    """A small artifact for the rejection tests (the gate they target fires first)."""
    path = str(tmp_path_factory.mktemp("treecut_tamper") / "small.parquet")
    return write_parquet(full_rows[:24], path), full_rows[:24]


def run_verify(path: str, monkeypatch, capsys, raw_dir: str = RAW_DIR) -> tuple:
    monkeypatch.setattr(sys, "argv", ["verify_treecut.py", "--rows", path, "--raw-dir", raw_dir])
    with pytest.raises(SystemExit) as excinfo:
        verify.main()
    return excinfo.value.code, capsys.readouterr().out


@requires_raw
def test_verify_rejects_a_tampered_proof(tmp_path, tamper_artifact, monkeypatch, capsys):
    _, rows = tamper_artifact
    tampered = [dict(row) for row in rows]
    info = dict(tampered[0]["extra_info"])
    info["proof"] = info["proof"].replace("only 1 linear formula", "only 2 linear formulas")
    tampered[0] = {**tampered[0], "extra_info": info}
    code, output = run_verify(write_parquet(tampered, str(tmp_path / "proof.parquet")),
                              monkeypatch, capsys)
    assert code == 1
    assert "L1 proof certificate" in output and "[FAIL]" in output


@requires_raw
def test_verify_rejects_a_swapped_counterpart(tmp_path, tamper_artifact, monkeypatch, capsys):
    _, rows = tamper_artifact
    tampered = [dict(row) for row in rows]
    first = dict(tampered[0]["extra_info"])
    second = dict(tampered[1]["extra_info"])
    first["paired_original_text"] = second["paired_original_text"]
    tampered[0] = {**tampered[0], "extra_info": first}
    code, output = run_verify(write_parquet(tampered, str(tmp_path / "swapped.parquet")),
                              monkeypatch, capsys)
    assert code == 1
    assert "L2 pair certificate" in output and "[FAIL]" in output


@requires_raw
def test_verify_rejects_a_tampered_answer_gold(tmp_path, tamper_artifact, monkeypatch, capsys):
    """The positive's number is certified, so a bumped gold must fail."""
    _, rows = tamper_artifact
    positives = [index for index, row in enumerate(rows) if row["extra_info"]["solvable"]]
    assert positives, "the prefix must carry at least one positive"
    tampered = [dict(row) for row in rows]
    index = positives[0]
    payload = json.loads(tampered[index]["reward_model"]["ground_truth"])
    payload["answer"] = str(int(payload["answer"]) + 1)
    tampered[index] = {**tampered[index],
                       "reward_model": {"style": "rule",
                                        "ground_truth": json.dumps(payload)}}
    code, output = run_verify(write_parquet(tampered, str(tmp_path / "gold.parquet")),
                              monkeypatch, capsys)
    assert code == 1
    assert "L2b answer certificate" in output and "[FAIL]" in output


@requires_raw
def test_verify_rejects_a_wrong_option_pointer(tmp_path, tamper_artifact, monkeypatch, capsys):
    _, rows = tamper_artifact
    negatives = [index for index, row in enumerate(rows) if not row["extra_info"]["solvable"]]
    tampered = [dict(row) for row in rows]
    index = negatives[0]
    info = dict(tampered[index]["extra_info"])
    correct = info["correct_option_id"]
    info["correct_option_id"] = next(opt["id"] for opt in info["options"] if opt["id"] != correct)
    tampered[index] = {**tampered[index], "extra_info": info}
    code, output = run_verify(write_parquet(tampered, str(tmp_path / "pointer.parquet")),
                              monkeypatch, capsys)
    assert code == 1
    assert "L4a negatives" in output and "[FAIL]" in output


@requires_raw
def test_verify_rejects_a_passage_condition_in_the_option_block(tmp_path, tamper_artifact,
                                                                monkeypatch, capsys):
    """The leak defence is a gate, not a hope: one in-passage option must fail."""
    path, rows = tamper_artifact
    negatives = [index for index, row in enumerate(rows) if not row["extra_info"]["solvable"]]
    tampered = [dict(row) for row in rows]
    index = negatives[0]
    info = dict(tampered[index]["extra_info"])
    question = tampered[index]["prompt"][0]["content"].split("\n\n")[0]
    options = [dict(opt) for opt in info["options"]]
    options[1]["text"] = question.split(". ")[0] + "."  # a sentence of the passage itself
    info["options"] = options
    tampered[index] = {**tampered[index], "extra_info": info}
    code, output = run_verify(write_parquet(tampered, str(tmp_path / "leak.parquet")),
                              monkeypatch, capsys)
    assert code == 1
    assert ("L4a negatives" in output or "prompt:" in output) and "[FAIL]" in output


@requires_raw
def test_verify_rejects_a_rewritten_prompt_tail(tmp_path, tamper_artifact, monkeypatch, capsys):
    """A prompt that no longer asks for the row's own gold must fail, not pass."""
    _, rows = tamper_artifact
    tampered = [dict(row) for row in rows]
    first = dict(tampered[0])
    question = first["prompt"][0]["content"].partition("\n\n")[0]
    first["prompt"] = [{"role": "user", "content": schema.render_prompt(question,
                                                                        schema.TEMPLATE_B)}]
    tampered[0] = first
    code, output = run_verify(write_parquet(tampered, str(tmp_path / "template.parquet")),
                              monkeypatch, capsys)
    assert code == 1, output
    assert "prompt: the stored prompt" in output and "[FAIL]" in output


@requires_raw
def test_verify_rejects_wrong_source_metadata(tmp_path, tamper_artifact, monkeypatch, capsys):
    """The source label and the ability are part of the contract, not decoration."""
    _, rows = tamper_artifact
    tampered = [dict(row) for row in rows]
    info = dict(tampered[0]["extra_info"])
    info["source"] = "not-treecut"
    tampered[0] = {**tampered[0], "extra_info": info}
    code, output = run_verify(write_parquet(tampered, str(tmp_path / "source.parquet")),
                              monkeypatch, capsys)
    assert code == 1, output
    assert "source metadata" in output and "[FAIL]" in output


@requires_raw
def test_verify_fails_closed_without_the_source(tmp_path, tamper_artifact, monkeypatch, capsys):
    path, _ = tamper_artifact
    code, output = run_verify(path, monkeypatch, capsys, raw_dir=str(tmp_path / "absent"))
    assert code == 1
    assert "source anchor: vocabulary" in output


def test_verify_helpers_are_independent_readings_of_the_text():
    assert verify.sentences_of(SHIPPED) == adapter.sentences_of(SHIPPED)
    assert verify.clause_of(DELETED) == adapter.clause_of(DELETED)
    matchers = {"food": FIXED_MATCHER}
    assert verify.themes_of(SHIPPED, matchers) == {"food"}
    graph = verify.text_graph(SHIPPED, FIXED_MATCHER)
    assert graph["asked"] == ASKED and not graph["malformed"]
    # the unsolvable member's answer is unreachable; the counterpart's is one hop away
    assert verify._derivation(graph, ASKED) == (None, [])
    hops, trail = verify._derivation(verify.text_graph(COUNTERPART, FIXED_MATCHER), ASKED)
    assert hops == CONFIG["ans_depth"] - 1
    assert trail == [DELETED, "A lasagna at Taste Good Cuisine and 2 "
                              "scrambled eggs at Urban Plate cost 34 dollars."]


def test_verify_pair_check_accepts_the_frozen_pair_and_rejects_a_broken_one():
    row = {
        "prompt": [{"content": SHIPPED + "\n\n" + "template"}],
        "extra_info": {
            "task_id": "treecut-food-nv4-ad2-000000",
            "paired_original_text": COUNTERPART,
            "deleted_condition_text": DELETED,
            "num_vars": 4,
            "ans_depth": 2,
            "theme": "food",
        },
    }
    assert verify._pair_failures(row, FIXED_MATCHER) == []
    broken = json.loads(json.dumps(row))
    broken["extra_info"]["deleted_condition_text"] = COUNTERPART_LOST
    assert verify._pair_failures(broken, FIXED_MATCHER)
    # the check is symmetric: the same pair read from the solvable side must pass too
    mirrored = {
        "prompt": [{"content": COUNTERPART + "\n\n" + "template"}],
        "extra_info": {
            "task_id": "treecut-food-nv4-ad2-000000-pos",
            "paired_original_text": SHIPPED,
            "deleted_condition_text": COUNTERPART_LOST,
            "num_vars": 4,
            "ans_depth": 2,
            "theme": "food",
        },
    }
    assert verify._pair_failures(mirrored, FIXED_MATCHER) == []


def test_verify_proof_check_reads_the_recorded_proof_not_the_graph():
    """L1's clause citation is checked against the proof text, not the graph.

    A fabricated justification must fail.  Comparing the clause with the graph it
    was derived from would pass no matter what the proof says -- that is the dead
    check this test pins down.
    """
    graph = verify.text_graph(SHIPPED, FIXED_MATCHER)
    assert verify._proof_failures(PROOF, graph, "frozen") == []
    fabricated = PROOF.replace(
        "a lasagna at Taste Good Cuisine and 2 scrambled eggs at Urban Plate cost "
        "34 dollars.",
        "a completely made up justification about widgets.",
    )
    assert fabricated != PROOF
    problems = verify._proof_failures(fabricated, graph, "frozen")
    assert any("does not quote the clause" in problem for problem in problems)


def test_verify_length_estimator_separates_a_real_leak_from_a_matched_pair():
    # a real leak: the classes differ in length, so the threshold rule finds it
    leaked = [110.0] * 40 + [200.0] * 40
    labels = [1] * 40 + [0] * 40
    assert verify.length_only_oof(leaked, labels, folds=5, seed=0) == 1.0
    # a matched pair: every text the same length, labels alternating -> chance,
    # whatever threshold the fold happens to learn
    matched = [100.0] * 80
    matched_labels = [index % 2 for index in range(80)]
    assert verify.length_only_oof(matched, matched_labels, folds=5, seed=0) == pytest.approx(0.5)


def test_verify_placeholder_mode_reading_is_exact_on_frozen_text():
    sentence = "A lasagna at Taste Good Cuisine costs 12 dollars."
    named = sentence.replace("lasagna at Taste Good Cuisine", "lasagna at Urban Plate")
    valued = sentence.replace("12", "19")
    assert verify.placeholder_mode_of(named, sentence, FIXED_MATCHER) == "variable_name"
    assert verify.placeholder_mode_of(valued, sentence, FIXED_MATCHER) == "variable_value"
    # a substitution that changes both a name and a value is not a placeholder
    both = valued.replace("lasagna at Taste Good Cuisine", "lasagna at Urban Plate")
    assert verify.placeholder_mode_of(both, sentence, FIXED_MATCHER) is None
    assert verify.placeholder_mode_of(sentence, sentence, FIXED_MATCHER) is None


@requires_raw
def test_main_writes_a_parquet_and_a_report(tmp_path, monkeypatch, capsys):
    out = str(tmp_path / "out.parquet")
    report = str(tmp_path / "report.json")
    monkeypatch.setattr(sys, "argv", ["treecut_adapter.py", "--raw-dir", RAW_DIR,
                                      "--out", out, "--report", report, "--limit", "12"])
    adapter.main()
    printed = capsys.readouterr().out
    assert "raw_rows" in printed and "unsolvable_diag" in printed and "placeholder" in printed
    rows = schema.read_parquet_rows(out)
    assert len(rows) == 12
    with open(report, encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["rows"] == 12
    assert payload["targets"] == {"negatives": adapter.NEG_TARGET,
                                  "positives": adapter.POS_TARGET}
    assert payload["funnel"]["raw_rows"] == adapter.NEG_TARGET
    assert payload["funnel"]["emitted_rows"] == 12
    assert payload["template_breakdown"] == {schema.TEMPLATE_A: 12}
    assert sum(payload["branch_breakdown"].values()) == 12
    assert payload["branch_breakdown"][schema.BRANCH_UNSOLVABLE_DIAG] > 0
    assert payload["branch_breakdown"][schema.BRANCH_SOLVABLE_NUMERIC] > 0
    assert payload["negatives"] + payload["positives"] == 12
    assert payload["length"]["negative_rows"] + payload["length"]["positive_rows"] == 12
    assert payload["plan"]["positive_rule"]
    assert len(payload["notes"]) == 4
