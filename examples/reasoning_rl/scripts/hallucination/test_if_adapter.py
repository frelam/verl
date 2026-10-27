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
"""Tests for the stage-4 instruction-following adapter (D31).

Fixtures are written in the two source spellings the adapter accepts, so the
suite runs without downloading ``nvidia/Nemotron-RL-instruction_following``.
What is pinned down here is the *adapter's* contract: constraint normalisation,
the unsupported-id drop (a row whose id the verifier cannot evaluate would score
0 forever), exact/exclusion dedup, the seeded quota draw, and the emitted row
schema the mixer and reward consume.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import if_adapter
import schema

KEYWORDS = {"id": "keywords:existence", "kwargs": {"keywords": ["Paris"]}}
NO_COMMA = {"id": "punctuation:no_comma", "kwargs": {}}


def raw_row(question: str, ids=None, kwargs=None, task_id=None, prompt=None):
    return {
        "prompt": prompt if prompt is not None else [{"role": "user", "content": question}],
        "instruction_id_list": ids if ids is not None else ["keywords:existence", "punctuation:no_comma"],
        "kwargs": kwargs if kwargs is not None else [{"keywords": ["Paris"]}, {}],
        "task_id": task_id or question,
        "source": "wildchat",
    }


class TestConstraintParsing:
    def test_parallel_lists(self):
        constraints = if_adapter.extract_constraints(raw_row("q"))
        assert constraints == [KEYWORDS, NO_COMMA]

    def test_dict_list(self):
        row = {"instruction_id_list": [dict(KEYWORDS), dict(NO_COMMA)]}
        assert if_adapter.extract_constraints(row) == [KEYWORDS, NO_COMMA]

    def test_kwargs_as_json_string(self):
        row = {"instruction_id_list": ["keywords:existence"], "kwargs": ['{"keywords": ["Paris"]}']}
        assert if_adapter.extract_constraints(row) == [KEYWORDS]

    def test_missing_kwargs_becomes_empty_dict(self):
        row = {"instruction_id_list": ["punctuation:no_comma"]}
        assert if_adapter.extract_constraints(row) == [{"id": "punctuation:no_comma", "kwargs": {}}]


class TestCandidateFiltering:
    def test_drops_empty_prompt_and_missing_constraints(self):
        rows = [
            {"prompt": [], "instruction_id_list": ["punctuation:no_comma"]},
            {"prompt": [{"role": "user", "content": "q"}], "instruction_id_list": []},
            raw_row("keep"),
        ]
        candidates, report = if_adapter.collect_candidates(rows)
        assert [c["prompt"][0]["content"] for c in candidates] == ["keep"]
        assert report["funnel"]["empty_prompt"] == 1
        assert report["funnel"]["no_constraints"] == 1
        assert report["funnel"]["admissible"] == 1

    def test_drops_unsupported_instruction_ids(self):
        rows = [
            raw_row("bad", ids=["made_up:thing"], kwargs=[{}]),
            raw_row("good"),
        ]
        candidates, report = if_adapter.collect_candidates(rows)
        assert [c["prompt"][0]["content"] for c in candidates] == ["good"]
        assert report["funnel"]["unsupported_instruction_id"] == 1
        assert report["unsupported_instruction_ids"] == {"made_up:thing": 1}

    def test_exact_dedup_keeps_first_prompt(self):
        rows = [raw_row("same", task_id="a"), raw_row("same", task_id="b"), raw_row("other", task_id="c")]
        candidates, report = if_adapter.collect_candidates(rows)
        assert [c["task_id"] for c in candidates] == ["a", "c"]
        assert report["funnel"]["duplicate_prompt"] == 1

    def test_excluded_prompts_are_dropped(self):
        rows = [raw_row("seen"), raw_row("fresh")]
        excluded = {if_adapter.prompt_key(rows[0]["prompt"])}
        candidates, report = if_adapter.collect_candidates(rows, excluded=excluded)
        assert [c["prompt"][0]["content"] for c in candidates] == ["fresh"]
        assert report["funnel"]["already_trained"] == 1

    def test_prompt_key_joins_multi_turn_prompts(self):
        prompt = [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Write a haiku."},
        ]
        assert if_adapter.prompt_key(prompt) == "be terse. write a haiku."


class TestRowSchema:
    def test_rows_carry_the_mixed_parquet_contract(self):
        candidates, _ = if_adapter.collect_candidates([raw_row("Write about Paris.")])
        rows = if_adapter.build_rows(candidates, seed=7, grading_mode="binary")
        schema.normalise_extra_info(rows)
        row = rows[0]
        assert row["data_source"] == if_adapter.DATA_SOURCE
        assert row["ability"] == "if"
        assert row["extra_info"]["domain"] == "if"
        assert row["extra_info"]["source"] == "wildchat"
        # Every stage-1 / stage-2 key must be present so the mixed parquet keeps
        # one Arrow struct column.
        for key in schema.EXTRA_INFO_KEYS:
            assert key in row["extra_info"]
        payload = json.loads(row["reward_model"]["ground_truth"])
        assert payload["constraints"] == [KEYWORDS, NO_COMMA]
        assert payload["grading_mode"] == "binary"

    def test_quota_is_seeded_and_respected(self):
        raw = [raw_row(f"question {i}", task_id=f"t{i}") for i in range(50)]
        candidates, _ = if_adapter.collect_candidates(raw)
        first = if_adapter.build_rows(candidates, quota=10, seed=3)
        second = if_adapter.build_rows(candidates, quota=10, seed=3)
        assert len(first) == 10
        assert [r["extra_info"]["task_id"] for r in first] == [r["extra_info"]["task_id"] for r in second]
        assert {r["extra_info"]["task_id"] for r in first} != {
            r["extra_info"]["task_id"] for r in if_adapter.build_rows(candidates, quota=10, seed=4)
        }

    def test_grading_mode_is_written(self):
        candidates, _ = if_adapter.collect_candidates([raw_row("q")])
        row = if_adapter.build_rows(candidates, grading_mode="fraction")[0]
        assert json.loads(row["reward_model"]["ground_truth"])["grading_mode"] == "fraction"


class TestMain:
    def test_end_to_end_with_a_local_jsonl(self, tmp_path):
        raw_path = tmp_path / "train.jsonl"
        rows = [raw_row(f"question {i}", task_id=f"t{i}") for i in range(6)]
        rows.append(raw_row("bad", ids=["made_up:thing"], kwargs=[{}], task_id="bad"))
        with raw_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

        out = tmp_path / "built" / "if.parquet"
        assert if_adapter.main(["--if-path", str(raw_path), "--out", str(out), "--quota", "4", "--seed", "1"]) == 0

        written = schema.read_parquet_rows(str(out))
        assert len(written) == 4
        assert all(row["data_source"] == if_adapter.DATA_SOURCE for row in written)
        report = json.load(open(if_adapter.report_path_for(str(out)), encoding="utf-8"))
        assert report["summary"]["rows"] == 4
        assert report["funnel"]["unsupported_instruction_id"] == 1
        assert report["config"]["grading_mode"] == "binary"

    def test_exclude_path_drops_seen_prompts(self, tmp_path):
        raw_path = tmp_path / "train.jsonl"
        with raw_path.open("w", encoding="utf-8") as handle:
            for i in range(4):
                handle.write(json.dumps(raw_row(f"question {i}", task_id=f"t{i}")) + "\n")

        seen = tmp_path / "stage3"
        seen.mkdir()
        schema.write_rows_parquet(
            [{"prompt": [{"role": "user", "content": "question 0"}], "extra_info": {"task_id": "old"}}],
            str(seen / "train.parquet"),
        )
        out = tmp_path / "built" / "if.parquet"
        assert (
            if_adapter.main(
                ["--if-path", str(raw_path), "--out", str(out), "--exclude-path", str(seen)]
            )
            == 0
        )
        written = schema.read_parquet_rows(str(out))
        assert len(written) == 3
        assert {row["prompt"][0]["content"] for row in written} == {"question 1", "question 2", "question 3"}

    def test_empty_pool_fails_without_writing(self, tmp_path):
        raw_path = tmp_path / "train.jsonl"
        raw_path.write_text("", encoding="utf-8")
        out = tmp_path / "built" / "if.parquet"
        assert if_adapter.main(["--if-path", str(raw_path), "--out", str(out)]) == 1
        assert not out.exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
