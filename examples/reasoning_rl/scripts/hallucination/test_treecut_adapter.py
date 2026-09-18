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
download is absent.  Everything that can be exercised without the source
(the text-to-graph certificate, the proof parser, the helper functions, the CLI
plumbing) runs from the frozen strings alone.

The two things the frozen fixtures cannot prove are proved against the source
itself: ``test_render_scenario_is_byte_faithful_to_the_generator`` seeds the
global ``random`` module and demands the generator's own ``problem``/``answer``
back, and ``test_verify_accepts_a_real_build`` /
``test_verify_rejects_a_tampered_*`` run the audit end to end on a real small
build, so the certificate is held to the artifact and not to this module's
helpers.
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

#: The shipped member: the fact that grounds the answer was deleted, so the asked
#: variable's component is two variables in one equation.
SHIPPED = (
    "A lasagna at Taste Good Cuisine and 2 scrambled eggs at Urban Plate cost 34 dollars. "
    "3 lasagnas at Urban Plate and 3 scrambled eggs at Taste Good Cuisine cost 72 dollars. "
    "A lasagna at Urban Plate costs 9 dollars. "
    "Question: how much does a scrambled egg at Urban Plate cost?"
)

#: The counterpart: the deleted edge was a root fact the derivation never used.
COUNTERPART = (
    "A lasagna at Taste Good Cuisine costs 12 dollars. "
    "3 lasagnas at Urban Plate and 3 scrambled eggs at Taste Good Cuisine cost 72 dollars. "
    "A lasagna at Taste Good Cuisine and 2 scrambled eggs at Urban Plate cost 34 dollars. "
    "Question: how much does a scrambled egg at Urban Plate cost?"
)

#: The sentence the shipped member lost -- the counterpart's grounding fact.
DELETED = "A lasagna at Taste Good Cuisine costs 12 dollars."

#: The sentence the counterpart lost -- a root fact off the derivation path.
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
    # the shipped member is the unsolvable one: no fact reaches the answer
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


def test_quota_for_splits_without_losing_rows():
    assert adapter.quota_for(12, 1084) == [91] * 4 + [90] * 8
    assert sum(adapter.quota_for(12, 1084)) == adapter.ROW_TARGET
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


def test_branch_breakdown_names_every_branch():
    rows = [{"extra_info": {"branch": schema.BRANCH_UNSOLVABLE_BARE}}]
    counts = adapter.branch_breakdown(rows)
    assert counts[schema.BRANCH_UNSOLVABLE_BARE] == 1
    assert set(counts) == {
        schema.BRANCH_SOLVABLE_NUMERIC,
        schema.BRANCH_SOLVABLE_ROLES,
        schema.BRANCH_SOLVABLE_JUDGE,
        schema.BRANCH_UNSOLVABLE_DIAG,
        schema.BRANCH_UNSOLVABLE_BARE,
    }


def test_load_source_fails_closed_without_the_generator(tmp_path):
    with pytest.raises(FileNotFoundError):
        adapter.load_source(str(tmp_path / "no-such-source"))


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


@requires_raw
def test_certify_pair_accepts_every_built_pair_of_one_cell():
    source = adapter.load_source(RAW_DIR)
    matcher = adapter.compile_matcher(
        adapter.vocabulary_forms(source["vocabulary"]["food"]["entities"],
                                source["vocabulary"]["food"]["item_dict"])
    )
    for ordinal in range(5):
        negative, positive, _ = adapter.build_pair(source, 0, "food", 8, 4, "random", ordinal)
        assert negative is not None
        config = {"theme": "food", "num_vars": 8, "ans_depth": 4, "order": "random",
                  "ordinal": ordinal}
        assert adapter.certify_pair(negative, positive, config, matcher) == []


# ---------------------------------------------------------------------------
# the built rows
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_build():
    return adapter.build_rows(RAW_DIR, limit=24, seed=0)


@pytest.fixture(scope="module")
def full_build():
    return adapter.build_rows(RAW_DIR, limit=None, seed=0)


@requires_raw
def test_small_build_satisfies_the_schema_contract(small_build):
    rows, funnel = small_build
    assert len(rows) == 24
    assert funnel["after_limit"] == 24
    schema.normalise_extra_info(rows)
    schema.validate_rows(rows)  # raises on a contract violation
    for row in rows:
        assert schema.validate_row(row) == []
        info = row["extra_info"]
        truth = json.loads(row["reward_model"]["ground_truth"])
        assert info["branch"] == schema.BRANCH_UNSOLVABLE_BARE
        assert info["template"] == schema.TEMPLATE_B
        assert info["solvable"] is False and truth["solvable"] is False
        assert truth["answer"] is None and truth["correct_option_id"] is None
        assert truth["perturbation_type"] == adapter.PERTURBATION_TYPE
        assert info["error_type"] == adapter.ERROR_TYPE
        assert row["data_source"] == schema.SOURCE_TREECUT
        assert info["task_id"].startswith("treecut-")
        assert info["paired_original_text"] and info["deleted_condition_text"]
        assert info["deleted_condition_text"] not in row["prompt"][0]["content"]
        # the gold lives in the template's refusal line, not in a number
        assert "\\boxed{UNSOLVABLE}" in row["prompt"][0]["content"]


@requires_raw
def test_limit_interleaves_every_cell(small_build):
    rows, _ = small_build
    cells = [(row["extra_info"]["theme"], row["extra_info"]["num_vars"],
              row["extra_info"]["ans_depth"]) for row in rows]
    assert len(set(cells)) == 12  # 24 rows over 12 cells
    assert collections.Counter(cells) == {cell: 2 for cell in set(cells)}
    assert cells[:12] == list(dict.fromkeys(cells))


@requires_raw
def test_full_build_funnel_and_plan(full_build):
    rows, funnel = full_build
    assert len(rows) == adapter.ROW_TARGET == 1084
    assert funnel["raw_rows"] == 1084
    assert funnel["after_scenario_reject_drop"] == 1084
    assert funnel["after_certificate_drop"] == 1084
    assert funnel["after_limit"] == 1084
    plan = funnel["plan"]
    assert sum(plan["scenario_drops"].values()) == 0
    assert sum(plan["certificate_drops"].values()) == 0
    assert plan["retried_slots"] <= plan["attempts"]
    assert len(plan["grid"]) == 12
    assert sum(cell["quota"] for cell in plan["grid"]) == adapter.ROW_TARGET
    assert {row["extra_info"]["theme"] for row in rows} == set(adapter.GRID_THEMES)
    assert adapter.branch_breakdown(rows) == {
        schema.BRANCH_SOLVABLE_NUMERIC: 0,
        schema.BRANCH_SOLVABLE_ROLES: 0,
        schema.BRANCH_SOLVABLE_JUDGE: 0,
        schema.BRANCH_UNSOLVABLE_DIAG: 0,
        schema.BRANCH_UNSOLVABLE_BARE: 1084,
    }
    # the doc's "at least 100 per cell" cannot be met at 1,084 rows over 12 cells
    assert all(88 <= cell["quota"] <= 92 for cell in plan["grid"])


@requires_raw
def test_build_is_deterministic_and_seed_sensitive():
    def projection(rows):
        return [
            (row["extra_info"]["task_id"], row["prompt"][0]["content"],
             row["extra_info"]["proof"], row["extra_info"]["paired_original_text"])
            for row in rows
        ]

    first, _ = adapter.build_rows(RAW_DIR, limit=12, seed=0)
    again, _ = adapter.build_rows(RAW_DIR, limit=12, seed=0)
    other, _ = adapter.build_rows(RAW_DIR, limit=12, seed=7)
    assert projection(first) == projection(again)
    assert projection(first) != projection(other)
    assert [row["extra_info"]["task_id"] for row in first] == [
        row["extra_info"]["task_id"] for row in again
    ]


@requires_raw
def test_only_the_bare_branch_and_refusal_template_are_emitted(full_build):
    rows, _ = full_build
    assert {row["extra_info"]["template"] for row in rows} == {schema.TEMPLATE_B}
    assert {row["extra_info"]["order"] for row in rows} <= set(verify.ORDER_VALUES)
    assert len({row["extra_info"]["task_id"] for row in rows}) == len(rows)


# ---------------------------------------------------------------------------
# the audit script, end to end
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def audited_parquet(tmp_path_factory):
    directory = tmp_path_factory.mktemp("treecut")
    path = str(directory / "rows.parquet")
    rows, _ = adapter.build_rows(RAW_DIR, limit=24, seed=0)
    schema.normalise_extra_info(rows)
    schema.write_rows_parquet(rows, path)
    return path, rows


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


@requires_raw
def test_verify_rejects_a_tampered_proof(tmp_path, audited_parquet, monkeypatch, capsys):
    path, rows = audited_parquet
    tampered = [dict(row) for row in rows]
    info = dict(tampered[0]["extra_info"])
    info["proof"] = info["proof"].replace("only 1 linear formula", "only 2 linear formulas")
    tampered[0] = {**tampered[0], "extra_info": info}
    out = str(tmp_path / "tampered_proof.parquet")
    schema.write_rows_parquet(tampered, out)
    monkeypatch.setattr(sys, "argv", ["verify_treecut.py", "--rows", out, "--raw-dir", RAW_DIR])
    with pytest.raises(SystemExit) as excinfo:
        verify.main()
    output = capsys.readouterr().out
    assert excinfo.value.code == 1
    assert "L1 proof certificate" in output and "[FAIL]" in output


@requires_raw
def test_verify_rejects_a_swapped_counterpart(tmp_path, audited_parquet, monkeypatch, capsys):
    path, rows = audited_parquet
    tampered = [dict(row) for row in rows]
    first = dict(tampered[0]["extra_info"])
    second = dict(tampered[1]["extra_info"])
    first["paired_original_text"] = second["paired_original_text"]
    tampered[0] = {**tampered[0], "extra_info": first}
    out = str(tmp_path / "swapped.parquet")
    schema.write_rows_parquet(tampered, out)
    monkeypatch.setattr(sys, "argv", ["verify_treecut.py", "--rows", out, "--raw-dir", RAW_DIR])
    with pytest.raises(SystemExit) as excinfo:
        verify.main()
    output = capsys.readouterr().out
    assert excinfo.value.code == 1
    assert "L2 pair certificate" in output and "[FAIL]" in output


@requires_raw
def test_verify_rejects_a_rewritten_prompt_tail(tmp_path, audited_parquet,
                                                monkeypatch, capsys):
    """A prompt that no longer asks for the row's own gold must fail, not pass."""
    path, rows = audited_parquet
    tampered = [dict(row) for row in rows]
    first = dict(tampered[0])
    question = first["prompt"][0]["content"].partition("\n\n")[0]
    first["prompt"] = [{
        "role": "user",
        "content": schema.render_prompt(question, schema.TEMPLATE_A,
                                        options=[{"id": "A", "text": "7"}]),
    }]
    tampered[0] = first
    out = str(tmp_path / "template_a.parquet")
    schema.write_rows_parquet(tampered, out)
    monkeypatch.setattr(sys, "argv", ["verify_treecut.py", "--rows", out, "--raw-dir", RAW_DIR])
    with pytest.raises(SystemExit) as excinfo:
        verify.main()
    output = capsys.readouterr().out
    assert excinfo.value.code == 1, output
    assert "prompt: the stored prompt" in output and "[FAIL]" in output


@requires_raw
def test_verify_rejects_wrong_source_metadata(tmp_path, audited_parquet,
                                              monkeypatch, capsys):
    """The source label and the ability are part of the contract, not decoration."""
    path, rows = audited_parquet
    tampered = [dict(row) for row in rows]
    first = dict(tampered[0])
    info = dict(first["extra_info"])
    info["source"] = "not-treecut"
    first["extra_info"] = info
    tampered[0] = first
    out = str(tmp_path / "source_label.parquet")
    schema.write_rows_parquet(tampered, out)
    monkeypatch.setattr(sys, "argv", ["verify_treecut.py", "--rows", out, "--raw-dir", RAW_DIR])
    with pytest.raises(SystemExit) as excinfo:
        verify.main()
    output = capsys.readouterr().out
    assert excinfo.value.code == 1, output
    assert "source metadata" in output and "[FAIL]" in output


@requires_raw
def test_verify_fails_closed_without_the_source(tmp_path, audited_parquet, monkeypatch, capsys):
    path, _ = audited_parquet
    monkeypatch.setattr(sys, "argv", ["verify_treecut.py", "--rows", path,
                                      "--raw-dir", str(tmp_path / "absent")])
    with pytest.raises(SystemExit) as excinfo:
        verify.main()
    output = capsys.readouterr().out
    assert excinfo.value.code == 1
    assert "source anchor: vocabulary" in output


def test_verify_helpers_are_independent_readings_of_the_text():
    assert verify.sentences_of(SHIPPED) == adapter.sentences_of(SHIPPED)
    assert verify.clause_of(DELETED) == adapter.clause_of(DELETED)
    matchers = {"food": FIXED_MATCHER}
    assert verify.themes_of(SHIPPED, matchers) == {"food"}
    graph = verify.text_graph(SHIPPED, FIXED_MATCHER)
    assert graph["asked"] == ASKED and not graph["malformed"]
    # the shipped member's answer is unreachable; the counterpart's is one hop away
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


@requires_raw
def test_main_writes_a_parquet_and_a_report(tmp_path, monkeypatch, capsys):
    out = str(tmp_path / "out.parquet")
    report = str(tmp_path / "report.json")
    monkeypatch.setattr(sys, "argv", ["treecut_adapter.py", "--raw-dir", RAW_DIR,
                                      "--out", out, "--report", report, "--limit", "12"])
    adapter.main()
    printed = capsys.readouterr().out
    assert "raw_rows" in printed and "unsolvable_bare" in printed
    rows = schema.read_parquet_rows(out)
    assert len(rows) == 12
    with open(report, encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["rows"] == 12
    assert payload["funnel"]["raw_rows"] == adapter.ROW_TARGET
    assert payload["funnel"]["after_limit"] == 12
    assert payload["branch_breakdown"][schema.BRANCH_UNSOLVABLE_BARE] == 12
    assert payload["template_breakdown"] == {schema.TEMPLATE_B: 12}
    assert payload["length"]["shipped_body_chars_mean"] > 0
    assert len(payload["unsupported"]) == 2
