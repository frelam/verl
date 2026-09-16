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
"""Unit tests for the parquet/reward self-check script.

Run from the repo root:  pytest examples/reasoning_rl/scripts/test_check_reward.py -v
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_reward import (
    audit_dump,
    audit_rows,
    echo_response,
    ground_truth_answer,
    if_constraints,
    print_report,
)


class TestGroundTruthAnswer:
    def test_json_payload(self):
        assert ground_truth_answer(json.dumps({"answer": "42", "task": "maze"})) == "42"

    def test_raw_json_string(self):
        assert ground_truth_answer(json.dumps({"answer": "42"})) == "42"

    def test_bare_ground_truth(self):
        assert ground_truth_answer("yes") == "yes"

    def test_structured_answer(self):
        assert ground_truth_answer(json.dumps({"answer": [1, 2]})) == "[1, 2]"

    def test_empty(self):
        assert ground_truth_answer(json.dumps({"answer": ""})) is None
        assert ground_truth_answer(json.dumps({"task": "maze"})) is None
        assert ground_truth_answer("") is None
        assert ground_truth_answer(None) is None


class TestEchoResponse:
    def test_logic_uses_answer_tags(self):
        assert (
            echo_response("logic_reasoning_gym", json.dumps({"answer": "6"}))
            == "<think>echo</think>\n\n<answer>6</answer>"
        )

    def test_math_uses_boxed(self):
        assert echo_response("math_bigmath", "7") == "<think>echo</think>\n\nThe final answer is: \\boxed{7}"

    def test_logic_bare_style(self):
        assert echo_response("logic_reasoning_gym", json.dumps({"answer": "6"}), "bare") == "<think>echo</think>\n\n6"

    def test_math_ignores_bare_style(self):
        # math/stem have a single prompt-mandated \boxed{} shape.
        assert echo_response("stem_drsci", "7", "bare") == "<think>echo</think>\n\nThe final answer is: \\boxed{7}"

    def test_code_and_if_are_not_echoable(self):
        assert echo_response("code_apps", json.dumps({"inputs": [], "outputs": []})) is None
        assert echo_response("if_nemotron", json.dumps({"constraints": []})) is None
        assert echo_response("code_apps", json.dumps({"inputs": [], "outputs": []}), "bare") is None

    def test_empty_ground_truth(self):
        assert echo_response("logic_synlogic", json.dumps({"answer": ""})) is None


def _row(data_source, ground_truth, source="src", extra=None):
    return {
        "data_source": data_source,
        "reward_model": {"ground_truth": ground_truth},
        "extra_info": {"source": source, **(extra or {})},
    }


class TestAuditRows:
    def _scorer(self, verdicts):
        def compute_score(data_source, solution_str, ground_truth, extra_info=None):
            return {"score": verdicts.get((data_source, ground_truth), 0.0)}

        return compute_score

    def test_counts_rates_and_failures(self):
        gt_ok = json.dumps({"answer": "1", "task": "countdown"})
        gt_bad = json.dumps({"answer": "2", "task": "countdown"})
        rows = [
            _row("logic_reasoning_gym", gt_ok),
            _row("logic_reasoning_gym", gt_bad),
            _row("logic_reasoning_gym", json.dumps({"answer": ""})),  # empty ground truth
            _row("code_apps", json.dumps({"inputs": [], "outputs": []})),  # not echoable
        ]
        verdicts = {("logic_reasoning_gym", gt_ok): 1.0}
        report = audit_rows(rows, self._scorer(verdicts), samples=1)

        assert report["counts"]["logic_reasoning_gym"] == 3
        assert report["counts"]["code_apps"] == 1
        entry = report["per_source"]["logic_reasoning_gym"]
        # logic rows are echoed twice (tagged + bare); the failing gt fails both.
        assert (entry["echoed"], entry["passed"], entry["empty_gt"]) == (2, 1, 1)
        assert (entry["bare_echoed"], entry["bare_passed"]) == (2, 1)
        assert entry["tasks"]["countdown"] == 2
        assert len(entry["failures"]) == 1  # capped by samples
        assert entry["failures"][0]["task"] == "countdown"
        assert entry["failures"][0]["style"] == "tagged"

        code_entry = report["per_source"]["code_apps"]
        assert code_entry["echoed"] == 0  # skipped, never scored
        assert code_entry["skipped"] == 1
        assert code_entry["empty_gt"] == 0  # code payloads carry no "answer" key

    def test_bare_style_failure_is_reported(self):
        # A source whose bare answers are not scoreable must be visible even when
        # the tagged echo passes (the shape the model actually produces).
        gt = json.dumps({"answer": "6", "task": "maze"})

        def compute_score(data_source, solution_str, ground_truth, extra_info=None):
            return {"score": 0.0 if solution_str.endswith("6") and "<answer>" not in solution_str else 1.0}

        entry = audit_rows([_row("logic_reasoning_gym", gt)], compute_score)["per_source"]["logic_reasoning_gym"]
        assert (entry["passed"], entry["echoed"]) == (1, 1)
        assert (entry["bare_passed"], entry["bare_echoed"]) == (0, 1)

    def test_math_rows_are_not_bare_echoed(self):
        gt = "7"

        def compute_score(data_source, solution_str, ground_truth, extra_info=None):
            return {"score": 1.0}

        entry = audit_rows([_row("math_bigmath", gt)], compute_score)["per_source"]["math_bigmath"]
        assert (entry["passed"], entry["bare_echoed"]) == (1, 0)

    def test_extra_info_is_forwarded(self):
        seen = []

        def compute_score(data_source, solution_str, ground_truth, extra_info=None):
            seen.append(extra_info)
            return {"score": 1.0}

        rows = [_row("logic_reasoning_gym", json.dumps({"answer": "6", "task": "maze"}), extra={"seed": 11})]
        audit_rows(rows, compute_score)
        assert seen == [{"source": "src", "seed": 11}] * 2  # tagged + bare

    def test_report_smoke(self, capsys):
        rows = [_row("logic_synlogic", "yes")]
        print_report(audit_rows(rows, self._scorer({("logic_synlogic", "yes"): 1.0})))
        out = capsys.readouterr().out
        assert "logic_synlogic: 1 rows" in out
        assert "echo 1/1 (100%) OK" in out
        assert "bare echo 1/1 (100%) OK" in out


class TestIfConstraintCoverage:
    def test_if_constraints_parsing(self):
        payload = json.dumps({"constraints": [{"id": "keywords:palindrome", "kwargs": {}}]})
        assert if_constraints(payload) == [{"id": "keywords:palindrome", "kwargs": {}}]
        assert if_constraints('{"constraints": "oops"}') == []
        assert if_constraints("not json") == []
        assert if_constraints(None) == []

    def test_supported_rows_are_counted_as_covered(self):
        gt = json.dumps({"constraints": [{"id": "keywords:palindrome", "kwargs": {}}]})

        def compute_score(**kwargs):
            raise AssertionError("if rows must never be echoed")

        entry = audit_rows([_row("if_nemotron", gt)], compute_score)["per_source"]["if_nemotron"]
        assert entry["if_rows"] == 1 and entry["if_covered"] == 1
        assert not entry["unsupported_ids"]

    def test_unimplemented_ids_are_reported(self):
        gt = json.dumps({"constraints": [{"id": "made_up:thing", "kwargs": {}}]})
        entry = audit_rows([_row("if_nemotron", gt)], lambda **kwargs: {"score": 1.0})["per_source"]["if_nemotron"]
        assert entry["if_rows"] == 1 and entry["if_covered"] == 0
        assert entry["unsupported_ids"]["made_up:thing"] == 1

    def test_report_prints_coverage(self, capsys):
        gt = json.dumps({"constraints": [{"id": "keywords:palindrome", "kwargs": {}}]})
        report = audit_rows([_row("if_nemotron", gt)], lambda **kwargs: {"score": 1.0})
        print_report(report)
        assert "constraint coverage 1/1 (100%) OK" in capsys.readouterr().out


class TestAuditDump:
    def test_summarises_generations(self, tmp_path, capsys):
        dump = tmp_path / "5.jsonl"
        dump.write_text(
            "\n".join(
                json.dumps(item)
                for item in [
                    {
                        "gts": json.dumps({"answer": "1", "task": "countdown"}),
                        "score": 1.0,
                        "output": "<answer>1</answer>",
                    },
                    {"gts": json.dumps({"answer": "2", "task": "countdown"}), "score": 0.0, "output": "bare 2"},
                    {"gts": json.dumps({"answer": "x", "task": "word_ladder"}), "score": 0.0, "output": "nope"},
                ]
            ),
            encoding="utf-8",
        )
        audit_dump(str(tmp_path), task_filter="countdown")
        out = capsys.readouterr().out
        assert "countdown: n=2 mean_reward=0.500" in out
        assert "word_ladder" not in out
        assert "bare 2" in out
