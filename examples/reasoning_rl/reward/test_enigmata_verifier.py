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
"""Regression tests for the task-aware Enigmata verifier.

These lock in the fix for the false negatives documented in the audit: the raw
Enigmata ``answer`` field carries prose scaffolding ("The answer is: ...") and,
for game24/countdown, a non-unique arithmetic expression, so string comparison
scored correct responses 0.

Run from the repo root:
    pytest examples/reasoning_rl/reward/test_enigmata_verifier.py -v
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enigmata_verifier import (
    HANDLED_TASKS,
    strip_prose_prefix,
    verify_enigmata,
)


class TestStripProsePrefix:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("The answer is: 42", "42"),
            ("The answer is 42", "42"),
            ("The final answer is: 42", "42"),
            ("Answer: 42", "42"),
            ("answer is 42", "42"),
            ("42", "42"),
            ("The output sequence is a valid stack permutation.", "The output sequence is a valid stack permutation."),
        ],
    )
    def test_prefixes(self, raw, expected):
        assert strip_prose_prefix(raw) == expected


class TestGame24:
    META = {"question": [10, 12, 12, 1], "target": None}
    GT = "The answer is: (12-10)*(12*1) = 24"

    def test_correct_alternative_expression(self):
        # A mathematically valid solution that is NOT the stored string.
        assert verify_enigmata("(12-10)*(12*1)", self.GT, "game24", {"question": [10, 12, 12, 1]}) is True

    def test_bare_target_rejected_when_numbers_known(self):
        # Reward-hacking guard: "24" must not pass when the inputs are known.
        assert verify_enigmata("24", self.GT, "game24", {"question": [10, 12, 12, 1]}) is False

    def test_wrong_expression_rejected(self):
        assert verify_enigmata("(12-10)*(12+1)", self.GT, "game24", {"question": [10, 12, 12, 1]}) is False

    def test_wrong_numbers_rejected(self):
        assert verify_enigmata("(12-10)*(11+1)", self.GT, "game24", {"question": [10, 12, 12, 1]}) is False

    def test_legacy_row_without_meta_still_requires_an_expression(self):
        # No meta -> number usage cannot be checked, but a bare constant is still
        # not an acceptable expression.
        assert verify_enigmata("(12-10)*(12*1)", self.GT, "game24") is True
        assert verify_enigmata("24", self.GT, "game24") is False

    def test_cannot_form(self):
        gt = "cannot form 24"
        assert verify_enigmata("cannot form 24", gt, "game24", self.META) is True
        assert verify_enigmata("6*4", gt, "game24", self.META) is False


class TestCountdown:
    META = {"question": [6, 11, 15, 4, 13], "target": 59}
    GT = "The answer is: 6*11-(15+13)/4 = 59"

    def test_correct(self):
        assert verify_enigmata("6*11-(15+13)/4", self.GT, "countdown", self.META) is True

    def test_bare_target_rejected(self):
        assert verify_enigmata("59", self.GT, "countdown", self.META) is False

    def test_target_recovered_from_ground_truth_without_meta(self):
        assert verify_enigmata("6*11-(15+13)/4", self.GT, "countdown") is True
        assert verify_enigmata("6*11-(15+13)/5", self.GT, "countdown") is False


class TestMaze:
    GT = "The answer is: (1,1)->(1,2)->(2,2)->(3,2)->(3,3)->(4,3)->(5,3)->(5,4)->(5,5)"
    MAZE = [["S", ".", ".", ".", "."], ["B", ".", ".", ".", "."], [".", ".", ".", ".", "."]]
    META = {"question": MAZE, "height": 3, "width": 5}

    def test_legacy_exact_path(self):
        assert verify_enigmata("(1,1)->(1,2)->(2,2)->(3,2)->(3,3)->(4,3)->(5,3)->(5,4)->(5,5)", self.GT, "maze") is True

    def test_legacy_wrong_path(self):
        assert verify_enigmata("(1,1)->(1,2)->(1,3)", self.GT, "maze") is False

    def test_meta_valid_grid_path(self):
        # 3x5 grid, no obstacle on this route.
        path = "[(1, 1), (1, 2), (2, 2), (3, 2), (3, 3), (3, 4), (3, 5)]"
        assert verify_enigmata(path, self.GT, "maze", self.META) is True

    def test_meta_path_through_obstacle_rejected(self):
        path = "[(1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (3, 4), (3, 5)]"
        assert verify_enigmata(path, self.GT, "maze", self.META) is False

    def test_meta_non_adjacent_rejected(self):
        assert verify_enigmata("[(1, 1), (3, 3), (3, 4), (3, 5)]", self.GT, "maze", self.META) is False

    def test_no_path_marker(self):
        gt = "not exist the path from start to end."
        assert verify_enigmata("not exist the path from start to end.", gt, "maze", self.META) is True
        assert verify_enigmata("(1,1)->(1,2)", gt, "maze", self.META) is False


class TestStackPermutation:
    META = {"input_sequence": [1, 2, 4, 3], "output_sequence": [1, 4, 3, 2]}
    GT = "The output sequence is a valid stack permutation."

    OPS = ["Push(1)", "Pop()", "Push(2)", "Push(4)", "Pop()", "Push(3)", "Pop()", "Pop()"]

    def test_valid_sequence_with_meta(self):
        assert verify_enigmata(str(self.OPS), self.GT, "stack_permutation", self.META) is True

    def test_invalid_sequence_with_meta(self):
        bad = ["Push(1)", "Push(2)", "Push(4)", "Push(3)", "Pop()", "Pop()", "Pop()", "Pop()"]
        assert verify_enigmata(str(bad), self.GT, "stack_permutation", self.META) is False

    def test_invalid_case_requires_marker(self):
        gt = "The output sequence is not a valid stack permutation."
        assert verify_enigmata("The output sequence is not a valid stack permutation.", gt, "stack_permutation") is True
        assert verify_enigmata(str(self.OPS), gt, "stack_permutation") is False

    def test_legacy_without_meta_is_strict(self):
        # No sequences in ground_truth -> cannot validate safely -> no credit.
        assert verify_enigmata(str(self.OPS), self.GT, "stack_permutation") is False


class TestDispatch:
    def test_unhandled_task_returns_none(self):
        assert verify_enigmata("[[1, 2], [3, 4]]", "[[1, 2], [3, 4]]", "sudoku") is None

    def test_handled_tasks_set(self):
        assert HANDLED_TASKS == {
            "game24",
            "countdown",
            "maze",
            "stack_permutation",
            "eight_puzzle",
            "fifteen_puzzle",
            "nine_puzzle",
            "sixteen_puzzle",
            "twiddle",
            "hamiltonian_path",
            "hamiltonian_cycle",
            "car_painting",
            "campsite",
            "star_battle",
            "full_crosswords",
            "tic_tac_toe",
            "zebra_logic",
        }


class TestSlidingPuzzle:
    """8/15 puzzle: the stored answer is the *initial* board, so only replaying the
    move sequence can grade the response (real audited row)."""

    BOARD = [[6, 3, 1], [2, 0, 7], [5, 4, 8]]
    META = {"question": BOARD}
    SOLUTION = "URDLULDRURDLULDRDLURDR"  # BFS, blank-moves reading
    TILE_READING = SOLUTION.translate(str.maketrans("UDLR", "DURL"))

    def test_solution_accepted(self):
        assert verify_enigmata(self.SOLUTION, "[[6, 3, 1], [2, 0, 7], [5, 4, 8]]", "eight_puzzle", self.META) is True

    def test_prompt_tile_reading_accepted(self):
        # The prompt says the *tile* moves; the official verifier replays the
        # inverse reading, so both sequences are the same solution.
        assert verify_enigmata(self.TILE_READING, "x", "eight_puzzle", self.META) is True

    def test_wrong_sequence_rejected(self):
        assert verify_enigmata("LRUD", "x", "eight_puzzle", self.META) is False

    def test_stored_initial_board_is_not_an_answer(self):
        assert verify_enigmata("[[6, 3, 1], [2, 0, 7], [5, 4, 8]]", "x", "eight_puzzle", self.META) is False

    def test_unsolvable_instance(self):
        meta = {"question": [[1, 2, 3], [4, 5, 6], [8, 7, 0]]}
        assert verify_enigmata("No feasible move path exists.", "x", "fifteen_puzzle", meta) is True
        assert verify_enigmata("LR", "x", "fifteen_puzzle", meta) is False

    def test_no_solution_claim_on_solvable_instance(self):
        assert verify_enigmata("No feasible move path exists.", "x", "eight_puzzle", self.META) is False

    def test_without_meta_is_strict(self):
        assert verify_enigmata(self.SOLUTION, "x", "eight_puzzle") is False


class TestShiftPuzzle:
    """Nine/sixteen puzzle: same initial-state trap, moves are circular shifts."""

    def test_left_shift_accepted(self):
        meta = {"question": [[2, 3, 1], [4, 5, 6], [7, 8, 9]]}
        assert verify_enigmata('["R12"]', "x", "nine_puzzle", meta) is True

    def test_right_shift_reading_accepted(self):
        # The prompt never states the rotation direction, so a sequence that
        # solves the puzzle under the other reading is the same answer.
        meta = {"question": [[2, 3, 1], [4, 5, 6], [7, 8, 9]]}
        assert verify_enigmata('["R11"]', "x", "nine_puzzle", meta) is True

    def test_wrong_move_rejected(self):
        meta = {"question": [[2, 3, 1], [4, 5, 6], [7, 8, 9]]}
        assert verify_enigmata('["C11"]', "x", "nine_puzzle", meta) is False


class TestTwiddle:
    META = {"question": [[4, 1, 3], [7, 5, 6], [8, 2, 9]]}
    ANSWER = "[[1, 0], [0, 0]]"

    def test_generator_answer_accepted(self):
        assert verify_enigmata(self.ANSWER, self.ANSWER, "twiddle", self.META) is True

    def test_alternative_valid_sequence_accepted(self):
        # Four turns of the same 2x2 block are the identity, so this still solves it.
        assert verify_enigmata("[[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]", "x", "twiddle", self.META) is True

    def test_arrow_notation_accepted(self):
        assert verify_enigmata("(1,0)->(0,0)", "x", "twiddle", self.META) is True

    def test_wrong_sequence_rejected(self):
        assert verify_enigmata("[[0, 0]]", "x", "twiddle", self.META) is False


class TestHamiltonian:
    PATH_META = {"question": "4\n0 1\n1 2\n2 3"}
    CYCLE_META = {"question": "4\n0 1\n1 2\n2 3\n3 0"}

    def test_path_accepted(self):
        assert verify_enigmata("[0, 1, 2, 3]", "x", "hamiltonian_path", self.PATH_META) is True

    def test_reversed_path_accepted(self):
        assert verify_enigmata("[3, 2, 1, 0]", "x", "hamiltonian_path", self.PATH_META) is True

    def test_broken_path_rejected(self):
        assert verify_enigmata("[0, 2, 1, 3]", "x", "hamiltonian_path", self.PATH_META) is False

    def test_non_permutation_rejected(self):
        assert verify_enigmata("[0, 1, 2, 2]", "x", "hamiltonian_path", self.PATH_META) is False

    def test_cycle_with_and_without_repeat(self):
        assert verify_enigmata("[0, 1, 2, 3, 0]", "x", "hamiltonian_cycle", self.CYCLE_META) is True
        assert verify_enigmata("[0, 3, 2, 1]", "x", "hamiltonian_cycle", self.CYCLE_META) is True

    def test_cycle_without_closing_edge_rejected(self):
        assert verify_enigmata("[0, 1, 3, 2]", "x", "hamiltonian_cycle", self.CYCLE_META) is False

    def test_no_instance(self):
        meta = {"question": "3\n0 1"}
        assert verify_enigmata("NO", "NO", "hamiltonian_path", meta) is True
        assert verify_enigmata("[0, 1, 2]", "NO", "hamiltonian_path", meta) is False
        assert verify_enigmata("NO", "[0, 1, 2]", "hamiltonian_path", self.PATH_META) is False


class TestCarPainting:
    """Audited row: the stored order is one of several optimal ones."""

    META = {
        "car_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "colors": ["A", "A", "B", "A", "B", "A", "A", "A", "B", "A"],
        "K": 4,
        "min_switches": 2,
    }
    ANSWER = "[1, 2, 5, 3, 9, 7, 8, 4, 6, 10]"

    def test_stored_answer_accepted(self):
        assert verify_enigmata(self.ANSWER, self.ANSWER, "car_painting", self.META) is True

    def test_alternative_optimal_order_accepted(self):
        assert verify_enigmata("[2, 1, 5, 3, 9, 7, 8, 4, 6, 10]", "x", "car_painting", self.META) is True

    def test_shift_budget_enforced(self):
        assert verify_enigmata("[10, 2, 3, 4, 5, 6, 7, 8, 9, 1]", "x", "car_painting", self.META) is False

    def test_missing_car_rejected(self):
        assert verify_enigmata("[1, 2, 5, 3, 9, 7, 8, 4, 6, 6]", "x", "car_painting", self.META) is False

    def test_without_meta_is_strict(self):
        assert verify_enigmata(self.ANSWER, self.ANSWER, "car_painting") is False


class TestBoardTasks:
    CAMPSITE_GT = (
        "total number of tents: 4\n"
        "tents in each row: 2 0 2 0\n"
        "tents in each column: 2 0 2 0\n"
        "<begin_board>\n* X * X\n. . . .\n* X * .\n. . X .\n<end_board>"
    )
    BOARD = "<begin_board>\n* X * X\n. . . .\n* X * .\n. . X .\n<end_board>"

    def test_campsite_board_only_response_accepted(self):
        # The prompt asks for the board inside <begin_board>; the ground truth
        # additionally stores the constraint header.
        assert verify_enigmata(self.BOARD, self.CAMPSITE_GT, "campsite") is True

    def test_campsite_header_repeat_accepted(self):
        body = self.BOARD.replace("<begin_board>\n", "")
        assert verify_enigmata(self.CAMPSITE_GT.split("<begin_board>")[0] + body, self.CAMPSITE_GT, "campsite") is True

    def test_campsite_wrong_board_rejected(self):
        wrong = self.BOARD.replace("* X * X", "* X * .")
        assert verify_enigmata(wrong, self.CAMPSITE_GT, "campsite") is False

    def test_star_battle_board_accepted(self):
        gold = ". . . * .\n* . X . X\n. . * . X\n. . . . *\n. * X . X"
        wrapped = "<begin_board>\n" + gold + "\n<end_board>"
        wrong = "<begin_board>\n" + gold.replace("*", ".") + "\n<end_board>"
        assert verify_enigmata(wrapped, gold, "star_battle") is True
        assert verify_enigmata(wrong, gold, "star_battle") is False


class TestCrosswords:
    GT = json.dumps({"across": ["FSLIC", "EXJET", "ASSAL"], "down": ["FEELA", "LAJOS", "CETYL"]})

    def test_json_shape_accepted(self):
        assert verify_enigmata(self.GT, self.GT, "full_crosswords") is True

    def test_prompt_mandated_lines_accepted(self):
        # The prompt's own Answer Format is "across: ..., down: ...", never JSON.
        body = "across: FSLIC, EXJET, ASSAL\ndown: FEELA, LAJOS, CETYL"
        assert verify_enigmata(body, self.GT, "full_crosswords") is True

    def test_wrong_word_rejected(self):
        body = "across: FSLIC, EXJET, ASSAL\ndown: FEELA, LAJOS, CETYL\n"  # ok
        assert verify_enigmata(body, self.GT, "full_crosswords") is True
        bad = "across: FSLIC, EXJET, ASSAL\ndown: FEELA, LAJOS, TEYLC"
        assert verify_enigmata(bad, self.GT, "full_crosswords") is False


class TestZebraLogic:
    GT = (
        "| Food          | papaya    | peas    |\n"
        "| Hobby         | skydiving | cooking |\n"
        "| Music-Genre   | reggae    | punk    |\n"
        "| Transport     | bus       | tram    |"
    )

    def test_verbatim_accepted(self):
        assert verify_enigmata(self.GT, self.GT, "zebra_logic") is True

    def test_markdown_separator_row_accepted(self):
        rows = self.GT.splitlines()
        body = "\n".join([rows[0], "|---|---|---|", *rows[1:]])
        assert verify_enigmata(body, self.GT, "zebra_logic") is True

    def test_wrong_value_rejected(self):
        assert verify_enigmata(self.GT.replace("reggae", "jazz"), self.GT, "zebra_logic") is False


class TestTicTacToe:
    def test_forced_centre_accepted(self):
        meta = {"current_board": [["X", "", ""], ["", "", ""], ["", "", ""]], "active_player": "O"}
        good = json.dumps([["X", "", ""], ["", "O", ""], ["", "", ""]])
        assert verify_enigmata(good, "x", "tic_tac_toe", meta) is True
        # The prompt's own quoted-token shape must parse too.
        assert verify_enigmata('"X" "" ""\n"" "O" ""\n"" "" ""', "x", "tic_tac_toe", meta) is True

    def test_suboptimal_move_rejected(self):
        meta = {"current_board": [["X", "", ""], ["", "", ""], ["", "", ""]], "active_player": "O"}
        bad = json.dumps([["X", "", ""], ["", "", ""], ["O", "", ""]])
        assert verify_enigmata(bad, "x", "tic_tac_toe", meta) is False

    def test_two_marks_rejected(self):
        meta = {"current_board": [["X", "", ""], ["", "", ""], ["", "", ""]], "active_player": "O"}
        bad = json.dumps([["X", "", ""], ["O", "", ""], ["O", "", ""]])
        assert verify_enigmata(bad, "x", "tic_tac_toe", meta) is False

    def test_audited_row(self):
        # Real row: X in a corner, O to move, centre is the only optimal move.
        meta = {
            "current_board": [["", "", "X"], ["", "", ""], ["", "", ""]],
            "active_player": "O",
            "answer": [["", "", "X"], ["", "O", ""], ["", "", ""]],
        }
        assert verify_enigmata('[["", "", "X"], ["", "O", ""], ["", "", ""]]', "x", "tic_tac_toe", meta) is True

    def test_without_meta_is_strict(self):
        assert verify_enigmata('"X" "" ""\n"" "O" ""\n"" "" ""', "x", "tic_tac_toe") is False


class TestSafeArithmetic:
    """The expression evaluator must never execute arbitrary code."""

    @pytest.mark.parametrize(
        "expr,expected",
        [
            ("2+2*3", 8.0),
            ("(12-10)*(12*1)", 24.0),
            ("6*11-(15+13)/4", 59.0),
            ("2**10", 1024.0),
            ("-3+5", 2.0),
        ],
    )
    def test_valid(self, expr, expected):
        from enigmata_verifier import _safe_arith_eval

        assert _safe_arith_eval(expr) == expected

    @pytest.mark.parametrize(
        "expr",
        ["__import__('os').system('ls')", "open('/etc/passwd')", "9**9**9", "1/0", "a+b", "[1,2]", "os"],
    )
    def test_rejected(self, expr):
        from enigmata_verifier import _safe_arith_eval

        assert _safe_arith_eval(expr) is None
