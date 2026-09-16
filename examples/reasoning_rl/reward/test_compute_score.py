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
"""Unit tests for the reasoning_rl four-domain reward dispatcher.

Run from the repo root:  pytest examples/reasoning_rl/reward/test_compute_score.py -v
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_score import (
    _align_callable_name,
    compute_score,
    extract_logic_answer,
    format_ok,
    logic_answer_match,
)


def think_wrap(response: str, think: str = "some reasoning") -> str:
    """A Qwen3-4B thinking-template-compliant response for gate-passing tests."""
    return f"<think>{think}</think>\n\n{response}"


# ---------------------------------------------------------------------------
# answer extraction
# ---------------------------------------------------------------------------


class TestExtractLogicAnswer:
    def test_answer_tag_last_occurrence(self):
        sol = "thinking <answer>wrong</answer> more <answer> 42 </answer>"
        assert extract_logic_answer(sol) == "42"

    def test_boxed_fallback(self):
        assert extract_logic_answer("reasoning ... $\\boxed{\\frac{1}{2}}$ done") == "\\frac{1}{2}"

    def test_boxed_balanced_braces(self):
        assert extract_logic_answer("so $\\boxed{{2, 3}}$") == "{2, 3}"

    def test_final_answer_line(self):
        assert extract_logic_answer("blah\nFinal Answer: B\n") == "B"

    def test_the_answer_is_line(self):
        # SynLogic web_of_lies / cryptarithm convention.
        assert extract_logic_answer("reasoning...\nThe answer is **no, no, yes**.\n") == "**no, no, yes**"

    def test_answer_tag_beats_boxed(self):
        sol = "$\\boxed{99}$ then <answer>42</answer>"
        assert extract_logic_answer(sol) == "42"

    def test_fenced_block_fallback(self):
        # SynLogic skyscraper_puzzle / zebra_puzzle require a fenced block.
        sol = "analysis...\n最终解答：\n```python\n[[3, 2], [1, 4]]\n```\n"
        assert extract_logic_answer(sol) == "[[3, 2], [1, 4]]"

    def test_fences_stripped_inside_answer_tag(self):
        sol = '<answer>```json\n{"a": 1}\n```</answer>'
        assert extract_logic_answer(sol) == '{"a": 1}'

    def test_no_answer(self):
        assert extract_logic_answer("no final answer here") is None


# ---------------------------------------------------------------------------
# normalised matching
# ---------------------------------------------------------------------------


class TestLogicAnswerMatch:
    def test_text_casefold_and_whitespace(self):
        assert logic_answer_match("  The   CAT ", "the cat")

    def test_numeric_equivalence(self):
        assert logic_answer_match("42.0", "42")
        assert logic_answer_match("0.5", "0.5")

    def test_numeric_mismatch(self):
        assert not logic_answer_match("41", "42")

    def test_grid_structural(self):
        # SynLogic arc_agi style: ground truth serialised via json.dumps.
        gt = json.dumps([[1, 9], [0, 1]])
        assert logic_answer_match("[[1, 9], [0, 1]]", gt)
        assert logic_answer_match("[[1,9],[0,1]]", gt)
        assert not logic_answer_match("[[1, 9], [0, 2]]", gt)

    def test_grid_numeric_tolerance(self):
        gt = json.dumps([[1.0, 2.0]])
        assert logic_answer_match("[[1, 2]]", gt)

    def test_none_prediction(self):
        assert not logic_answer_match(None, "42")

    def test_empty_ground_truth(self):
        assert not logic_answer_match("42", "")

    def test_enigmata_whitespace_grid(self):
        # Rows separated by double spaces / newlines collapse identically.
        gt = "7 0 7  0 7 0\n7 0 0"
        assert logic_answer_match("7 0 7 0 7 0 7 0 0", gt)

    def test_comma_spacing_invariant(self):
        # SynLogic boolean_expressions gt is "A,C,D,E"; models naturally
        # write "A, C, D, E".
        assert logic_answer_match("A, C, D, E", "A,C,D,E")
        assert not logic_answer_match("A, C, D, F", "A,C,D,E")

    def test_inner_quotes_invariant(self):
        # SynLogic cipher gt is "[[WORD]]" (unparseable); the prompt example
        # tells the model to output "[['WORD']]".
        assert logic_answer_match("[['HILDRQHZPQ']]", "[[HILDRQHZPQ]]")
        assert not logic_answer_match("[['HILDRQHZPX']]", "[[HILDRQHZPQ]]")

    def test_bold_emphasis_stripped(self):
        # SynLogic web_of_lies asks for 'The answer is **yes, no**'.
        assert logic_answer_match("**no, no, no, yes**", "no, no, no, yes")

    def test_token_spacing_still_significant(self):
        # Loose form must not collapse spaces between tokens.
        assert not logic_answer_match("123", "1 2 3")

    def test_unordered_coord_tasks_canonicalised(self):
        # norinori dominos: domino order and cell order within a domino are
        # both irrelevant; minesweeper/star_placement behave the same way.
        gt = json.dumps([[[3, 1], [3, 2]], [[5, 1], [5, 2]]])
        assert logic_answer_match("[[(5, 1), (5, 2)], [(3, 2), (3, 1)]]", gt, task="norinori")
        assert not logic_answer_match("[[(5, 1), (5, 2)], [(3, 2), (3, 3)]]", gt, task="norinori")
        gt_ms = json.dumps([[3, 7], [4, 6]])
        assert logic_answer_match("[(4, 6), (3, 7)]", gt_ms, task="minesweeper")

    def test_grid_order_still_significant_without_task(self):
        # The same shuffled-rows input must NOT match for grid tasks (campsite)
        # or when no task is supplied.
        gt = json.dumps([[3, 4], [1, 2]])
        assert not logic_answer_match("[[1, 2], [3, 4]]", gt, task="campsite")
        assert not logic_answer_match("[[1, 2], [3, 4]]", gt)


# ---------------------------------------------------------------------------
# dispatcher contract
# ---------------------------------------------------------------------------


class TestComputeScore:
    def test_logic_correct(self):
        gt = json.dumps({"answer": "42", "task": "sudoku"})
        res = compute_score("logic_synlogic", think_wrap("<answer>42</answer>"), gt)
        assert res == {"score": 1.0}

    def test_logic_wrong(self):
        gt = json.dumps({"answer": "42", "task": "sudoku"})
        res = compute_score("logic_enigmata", think_wrap("<answer>43</answer>"), gt)
        assert res == {"score": 0.0}

    def test_logic_missing_answer(self):
        gt = json.dumps({"answer": "42", "task": "sudoku"})
        res = compute_score("logic_reasoning_gym", think_wrap("i give up"), gt)
        assert res == {"score": 0.0}

    def test_logic_bare_ground_truth_tolerated(self):
        # Non-JSON ground truth (schema drift) falls back to raw string compare.
        res = compute_score("logic_arc", think_wrap("<answer>yes</answer>"), "yes")
        assert res == {"score": 1.0}

    def test_unknown_source_raises(self):
        with pytest.raises(NotImplementedError):
            compute_score("openai/gsm8k", think_wrap("x"), "y")

    def test_stem_routes_to_math(self):
        # Only runs when math-verify is installed; otherwise the fallback
        # math_dapo path is exercised — either way the contract must hold.
        res = compute_score("stem_drsci", think_wrap("The final answer is: $\\boxed{105}$"), "105")
        assert isinstance(res, dict) and set(res) == {"score"}
        assert res["score"] in (0.0, 1.0)

    def test_math_route(self):
        res = compute_score("math_bigmath", think_wrap("The answer is $\\boxed{7}$."), "7")
        assert isinstance(res["score"], float)


# ---------------------------------------------------------------------------
# format gate (Qwen3-4B thinking template)
# ---------------------------------------------------------------------------


class TestFormatOk:
    def test_compliant_passes(self):
        assert format_ok(think_wrap("the final answer"))

    def test_missing_think_block(self):
        assert not format_ok("the final answer is 42")

    def test_leading_whitespace_before_think(self):
        # The template starts assistant content immediately with <think>.
        assert not format_ok("\n<think>reasoning</think>\n\nanswer")

    def test_unterminated_think(self):
        assert not format_ok("<think>reasoning forever, truncated")

    def test_empty_think_is_no_thinking_shortcut(self):
        # "<think>\n\n</think>" is what the template inserts when
        # enable_thinking=False — not a thinking response.
        assert not format_ok("<think>\n\n</think>\n\nanswer")
        assert not format_ok("<think>  </think>\n\nanswer")

    def test_empty_response(self):
        assert not format_ok("<think>reasoning</think>")
        assert not format_ok("<think>reasoning</think>\n\n  ")

    def test_stray_think_tags_in_response(self):
        assert not format_ok("<think>reasoning</think>\n\nanswer </think> extra")
        assert not format_ok("<think>reasoning</think>\n\n<think>again</think> answer")

    def test_think_body_may_contain_anything_but_close_tag(self):
        assert format_ok("<think>2+2... wait, let me re-check.\nStill 4.</think>\n\n\\boxed{4}")


class TestFormatGate:
    def test_gate_zeroes_correct_answer_with_bad_format(self):
        # A correct boxed answer still scores 0 when the think block is missing.
        res = compute_score("math_bigmath", "The answer is $\\boxed{7}$.", "7")
        assert res == {"score": 0.0}

    def test_gate_zeroes_all_domains(self):
        for data_source, gt in (
            ("math_bigmath", "42"),
            ("stem_drsci", "42"),
            ("logic_synlogic", json.dumps({"answer": "42"})),
            ("code_deepcoder", json.dumps({"inputs": ["1\n"], "outputs": ["1\n"]})),
        ):
            res = compute_score(data_source, "no think block at all", gt)
            assert res == {"score": 0.0}, data_source

    def test_gate_passes_compliant_response_to_verifier(self):
        gt = json.dumps({"answer": "42", "task": "sudoku"})
        res = compute_score("logic_synlogic", think_wrap("<answer>42</answer>"), gt)
        assert res == {"score": 1.0}


# ---------------------------------------------------------------------------
# Enigmata task-aware verification (audit regressions)
# ---------------------------------------------------------------------------


class TestEnigmataTaskAware:
    def test_game24_expression_with_meta(self):
        # Raw gt is a prose-wrapped, non-unique expression; the model's different
        # but mathematically valid expression must score 1.
        gt = json.dumps(
            {"answer": "The answer is: (12-10)*(12*1) = 24", "task": "game24", "meta": {"question": [10, 12, 12, 1]}}
        )
        assert compute_score("logic_enigmata", think_wrap("<answer>(12-10)*(12*1)</answer>"), gt) == {"score": 1.0}
        assert compute_score("logic_enigmata", think_wrap("<answer>(12-10)*(12+1)</answer>"), gt) == {"score": 0.0}

    def test_game24_bare_target_is_not_rewarded(self):
        gt = json.dumps(
            {"answer": "The answer is: (12-10)*(12*1) = 24", "task": "game24", "meta": {"question": [10, 12, 12, 1]}}
        )
        assert compute_score("logic_enigmata", think_wrap("<answer>24</answer>"), gt) == {"score": 0.0}

    def test_countdown_legacy_row_without_meta(self):
        # Rows built before the meta change still get expression checking (target
        # is recoverable from the "= 59" suffix).
        gt = json.dumps({"answer": "The answer is: 6*11-(15+13)/4 = 59", "task": "countdown"})
        assert compute_score("logic_enigmata", think_wrap("<answer>6*11-(15+13)/4</answer>"), gt) == {"score": 1.0}
        assert compute_score("logic_enigmata", think_wrap("<answer>59</answer>"), gt) == {"score": 0.0}

    def test_maze_prose_prefixed_ground_truth(self):
        gt = json.dumps({"answer": "The answer is: (1,1)->(1,2)->(2,2)", "task": "maze"})
        assert compute_score("logic_enigmata", think_wrap("<answer>(1,1)->(1,2)->(2,2)</answer>"), gt) == {"score": 1.0}
        assert compute_score("logic_enigmata", think_wrap("<answer>(9,9)->(9,8)</answer>"), gt) == {"score": 0.0}

    def test_stack_permutation_with_meta(self):
        meta = {"input_sequence": [1, 2, 4, 3], "output_sequence": [1, 4, 3, 2]}
        gt = json.dumps(
            {"answer": "The output sequence is a valid stack permutation.", "task": "stack_permutation", "meta": meta}
        )
        ops = '["Push(1)", "Pop()", "Push(2)", "Push(4)", "Pop()", "Push(3)", "Pop()", "Pop()"]'
        assert compute_score("logic_enigmata", think_wrap(f"<answer>{ops}</answer>"), gt) == {"score": 1.0}
        bad = '["Push(1)", "Push(2)", "Push(4)", "Push(3)", "Pop()", "Pop()", "Pop()", "Pop()"]'
        assert compute_score("logic_enigmata", think_wrap(f"<answer>{bad}</answer>"), gt) == {"score": 0.0}

    def test_grid_answer_is_format_agnostic(self):
        # Enigmata stores the grid as space separated text; models emit nested lists.
        gt = json.dumps({"answer": "7 0\n7 0", "task": "arc_agi"})
        assert compute_score("logic_enigmata", think_wrap("<answer>[[7, 0], [7, 0]]</answer>"), gt) == {"score": 1.0}
        assert compute_score("logic_enigmata", think_wrap("<answer>[[7, 1], [7, 0]]</answer>"), gt) == {"score": 0.0}

    def test_generic_enigmata_prose_prefix_stripped(self):
        gt = json.dumps({"answer": "The answer is: yes", "task": "not_a_special_task"})
        assert compute_score("logic_enigmata", think_wrap("<answer>yes</answer>"), gt) == {"score": 1.0}

    def test_other_logic_sources_unaffected(self):
        gt = json.dumps({"answer": "42", "task": "sudoku"})
        assert compute_score("logic_synlogic", think_wrap("<answer>42</answer>"), gt) == {"score": 1.0}
        assert compute_score("logic_reasoning_gym", think_wrap("<answer>42</answer>"), gt) == {"score": 1.0}


# ---------------------------------------------------------------------------
# code callable-name alignment (fn_name contract gap)
# ---------------------------------------------------------------------------


class TestCodeCallableNameAlignment:
    GT = json.dumps({"inputs": ['"ab"'], "outputs": ["AB"], "fn_name": "make_acronym"})

    def test_exact_name_left_alone(self):
        sol = think_wrap("```python\ndef make_acronym(s):\n    return s.upper()\n```")
        assert _align_callable_name(sol, self.GT) == self.GT

    def test_camel_case_variant_rewritten(self):
        sol = think_wrap("```python\ndef makeAcronym(s):\n    return s.upper()\n```")
        aligned = json.loads(_align_callable_name(sol, self.GT))
        assert aligned["fn_name"] == "makeAcronym"
        assert aligned["inputs"] == ['"ab"']

    def test_single_unrelated_top_level_function_rewritten(self):
        sol = think_wrap("```python\ndef solution(s):\n    return s.upper()\n```")
        assert json.loads(_align_callable_name(sol, self.GT))["fn_name"] == "solution"

    def test_ambiguous_multiple_functions_left_alone(self):
        sol = think_wrap("```python\ndef helper(s):\n    return s\ndef other(s):\n    return s\n```")
        assert _align_callable_name(sol, self.GT) == self.GT

    def test_solution_class_layout_left_alone(self):
        sol = think_wrap("```python\nclass Solution:\n    def make_acronym(self, s):\n        return s\n```")
        assert _align_callable_name(sol, self.GT) == self.GT

    def test_stdio_payload_without_fn_name_untouched(self):
        gt = json.dumps({"inputs": ["1\n"], "outputs": ["1\n"]})
        sol = think_wrap("```python\ndef anything():\n    pass\n```")
        assert _align_callable_name(sol, gt) == gt
