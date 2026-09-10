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
"""Unit tests for code-domain test-case canonicalisation (to_parquet_code.py).

Raw APPS rows deviate from the prime_code / sandbox_fusion contract:
  * call-based (fn_name): inputs[i] is a list of args, outputs[i] list-wrapped
  * stdio: some inputs[i]/outputs[i] are lists of lines
Stored unmodified these score 0 on both verifiers; normalize_in_outs must
canonicalise them while leaving already-canonical rows (DeepCoder,
code_contests) byte-identical.

Run from the repo root:  pytest examples/reasoning_rl/scripts/test_to_parquet_code.py -v
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from to_parquet_code import normalize_in_outs


class TestCallBasedCanonicalisation:
    def test_apps_raw_args_list(self):
        # Raw APPS call-based: inputs[i] is a list of args, outputs[i] list-wrapped.
        raw = {"inputs": [[[1, 2, 3]], [[4, 5]]], "outputs": [[6], [9]], "fn_name": "sum_list"}
        out = normalize_in_outs(raw)
        assert out["fn_name"] == "sum_list"
        # prime_code does [json.loads(line) for line in inputs[i].split("\n")] -> fn(*args)
        assert out["inputs"] == ["[1, 2, 3]", "[4, 5]"]
        assert [json.loads(x) for x in out["inputs"][0].split("\n")] == [[1, 2, 3]]
        # single-element list unwrapped once, then JSON-serialised
        assert out["outputs"] == ["6", "9"]
        assert json.loads(out["outputs"][0]) == 6

    def test_apps_raw_multi_arg(self):
        raw = {"inputs": [[1, [2, 3]]], "outputs": [[4]], "fn_name": "f"}
        out = normalize_in_outs(raw)
        assert out["inputs"] == ["1\n[2, 3]"]
        assert [json.loads(x) for x in out["inputs"][0].split("\n")] == [1, [2, 3]]

    def test_list_returning_fn_stays_wrapped(self):
        # fn returns [6]; APPS wraps expected returns -> [[6]]; unwrap once -> [6]
        raw = {"inputs": [[1]], "outputs": [[[6]]], "fn_name": "f"}
        out = normalize_in_outs(raw)
        assert out["outputs"] == ["[6]"]

    def test_canonical_strings_unchanged(self):
        # DeepCoder-style rows already satisfy the contract -> pass through.
        raw = {"inputs": ["[1, 2, 3]", "[4]"], "outputs": ["6", "4"], "fn_name": "sum_list"}
        out = normalize_in_outs(raw)
        assert out["inputs"] == raw["inputs"]
        assert out["outputs"] == raw["outputs"]


class TestStdioCanonicalisation:
    def test_apps_raw_list_of_lines(self):
        raw = {"inputs": [["4", "1 2 3 4"]], "outputs": [["10"]]}
        out = normalize_in_outs(raw)
        assert out["inputs"] == ["4\n1 2 3 4"]
        assert out["outputs"] == ["10"]
        assert "fn_name" not in out

    def test_list_lines_with_trailing_newlines(self):
        raw = {"inputs": [["4\n", "1 2 3 4\n"]], "outputs": ["10\n"]}
        out = normalize_in_outs(raw)
        assert out["inputs"] == ["4\n1 2 3 4"]

    def test_canonical_strings_unchanged(self):
        # code_contests / DeepCoder stdio rows: already plain strings.
        raw = {"inputs": ["4\n1 2 3 4\n", "1\n7\n"], "outputs": ["10\n", "7\n"]}
        out = normalize_in_outs(raw)
        assert out["inputs"] == raw["inputs"]
        assert out["outputs"] == raw["outputs"]


class TestInvalidRows:
    def test_none_and_empty(self):
        assert normalize_in_outs(None) is None
        assert normalize_in_outs("") is None
        assert normalize_in_outs("not json") is None

    def test_mismatched_lengths(self):
        assert normalize_in_outs({"inputs": ["a"], "outputs": []}) is None

    def test_string_field_json(self):
        out = normalize_in_outs(json.dumps({"inputs": ["1\n"], "outputs": ["1\n"]}))
        assert out == {"inputs": ["1\n"], "outputs": ["1\n"]}
