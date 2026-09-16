# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Local tests for the sandbox_fusion execution contract.

The fn_name wrapper is what actually runs a call-based solution, but it normally
only runs inside the sandbox service; ``build_fn_name_wrapper`` renders it
locally so its callable-resolution rules can be tested on CPU.  ``outputs_match``
is the stdout comparison applied to every test case.
"""

import subprocess
import sys
import tempfile

from verl.utils.reward_score.sandbox_fusion.utils import (
    build_fn_name_wrapper,
    normalize_stdout,
    outputs_match,
)


def run_wrapper(generation: str, fn_name: str, stdin: str) -> str:
    wrapper = build_fn_name_wrapper(generation, fn_name)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(wrapper)
        path = fh.name
    proc = subprocess.run([sys.executable, path], input=stdin, capture_output=True, text=True, timeout=30)
    return proc.stdout


def test_class_solution_can_shadow_a_wrapper_import():
    """`re.search` is imported by the wrapper's prelude; the solution's method must win."""
    generation = """
class Solution:
    def search(self, nums, target):
        return target in nums
"""
    assert run_wrapper(generation, "search", "[2, 5, 6, 0, 0, 1, 2]\n0") == "true\n"


def test_class_solution_with_a_unique_name():
    generation = """
class Solution:
    def strongPasswordChecker(self, s):
        return len(s)
"""
    assert run_wrapper(generation, "strongPasswordChecker", '"abc"') == "3\n"


def test_top_level_function_is_used():
    generation = """
def add(a, b):
    return a + b
"""
    assert run_wrapper(generation, "add", "1\n2") == "3\n"


def test_top_level_function_shadows_wrapper_import():
    generation = """
def count(items, value):
    return items.count(value)
"""
    assert run_wrapper(generation, "count", "[1, 2, 1]\n1") == "2\n"


def test_container_and_string_results_are_serialized_like_the_sandbox():
    assert run_wrapper("def f(x):\n    return [x, x]\n", "f", "3") == "[3, 3]\n"
    assert run_wrapper("def f(x):\n    return x\n", "f", '"hi"') == "hi\n"
    assert run_wrapper("def f(x):\n    return None\n", "f", "1") == "null\n"


def test_missing_callable_prints_nothing():
    assert run_wrapper("x = 1\n", "not_defined", "1") == ""


class TestOutputsMatch:
    """Stored expected outputs often carry trailing spaces / CRLF endings."""

    def test_trailing_newlines_are_ignored(self):
        assert outputs_match("3\n", "3")
        assert outputs_match("3\n\n", "3\n")

    def test_line_trailing_spaces_are_ignored(self):
        assert outputs_match("1\n3\n-1\n", "1\n3 \n-1 \n")

    def test_crlf_endings_are_ignored(self):
        assert outputs_match("    ****\n    *  *\n", "    ****\r\n    *  *\r\n")

    def test_inner_spacing_still_matters(self):
        assert not outputs_match("    ****\n", "     ****\n")
        assert not outputs_match("1 2 3\n", "1 2  3\n")

    def test_different_content_still_fails(self):
        assert not outputs_match("3\n", "4\n")
        assert not outputs_match("1\n2\n3\n", "1\n3\n2\n")

    def test_normalize_stdout_keeps_inner_content(self):
        assert normalize_stdout("a  b \n\n") == "a  b"
