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
        assert HANDLED_TASKS == {"game24", "countdown", "maze", "stack_permutation"}


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
