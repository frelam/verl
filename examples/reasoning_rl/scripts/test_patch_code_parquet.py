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
"""Tests for patch_code_parquet.py (in-place code-data fixes).

Run from the repo root:
    pytest examples/reasoning_rl/scripts/test_patch_code_parquet.py -v
"""

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from patch_code_parquet import (  # noqa: E402
    OLD_FUNCTION_INSTRUCTION,
    _unquote_string_return,
    patch_dataset,
    patch_row,
)
from to_parquet_code import FUNCTION_INSTRUCTION, STDIO_INSTRUCTION  # noqa: E402


def _extra_info():
    return {
        "split": "train",
        "index": 0,
        "task_id": "t",
        "domain": "code",
        "source": "deepcoder",
        "difficulty": "",
        "prior_solve_rate": -1.0,
        "seed": -1,
    }


def _old_row(fn_name=None, outputs=None, inputs=None, problem="Solve it."):
    if fn_name:
        in_outs = {"inputs": inputs or ['"a"'], "outputs": outputs or ["A"], "fn_name": fn_name}
        instruction = OLD_FUNCTION_INSTRUCTION
    else:
        in_outs = {"inputs": inputs or ["1\n"], "outputs": outputs or ["1\n"]}
        instruction = STDIO_INSTRUCTION
    return {
        "data_source": "code_deepcoder",
        "prompt": [{"role": "user", "content": problem + "\n\n" + instruction}],
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": json.dumps(in_outs)},
        "extra_info": _extra_info(),
    }


class TestUnquoteStringReturn:
    @pytest.mark.parametrize("value", ["6", "[3, 4]", "true", "null", "hello", "1.5", "[[1, 2]]"])
    def test_non_string_forms_untouched(self, value):
        assert _unquote_string_return(value) == value

    @pytest.mark.parametrize("value,expected", [('"hi"', "hi"), ('"a b"', "a b"), ('""', "")])
    def test_json_string_form_unquoted(self, value, expected):
        assert _unquote_string_return(value) == expected


class TestPatchRow:
    def test_call_based_row_gets_fn_name_and_unquoted_strings(self):
        row = _old_row(fn_name="make_acronym", outputs=['"MAS"', "6", "[3, 4]", "true"])
        changed, unquoted = patch_row(row)
        assert changed is True
        assert unquoted == 1
        content = row["prompt"][0]["content"]
        assert "make_acronym" in content
        assert OLD_FUNCTION_INSTRUCTION not in content
        assert content.startswith("Solve it.")  # problem text preserved
        gt = json.loads(row["reward_model"]["ground_truth"])
        assert gt["outputs"] == ["MAS", "6", "[3, 4]", "true"]
        assert gt["fn_name"] == "make_acronym"

    def test_stdio_row_untouched(self):
        row = _old_row(outputs=["1\n"])
        before = json.dumps(row, sort_keys=True)
        changed, unquoted = patch_row(row)
        assert (changed, unquoted) == (False, 0)
        assert json.dumps(row, sort_keys=True) == before

    def test_idempotent(self):
        row = _old_row(fn_name="f", outputs=['"hi"'])
        patch_row(row)
        snapshot = json.dumps(row, sort_keys=True)
        changed, unquoted = patch_row(row)
        assert (changed, unquoted) == (False, 0)
        assert json.dumps(row, sort_keys=True) == snapshot

    def test_appends_when_old_instruction_absent(self):
        row = _old_row(fn_name="f")
        row["prompt"][0]["content"] = "Just solve it."
        changed, _ = patch_row(row)
        assert changed is True
        assert row["prompt"][0]["content"].endswith(FUNCTION_INSTRUCTION.replace("{fn_name}", "f"))

    def test_non_call_based_payload_with_missing_fn_name(self):
        row = _old_row()
        row["reward_model"]["ground_truth"] = "not json"
        assert patch_row(row) == (False, 0)


def test_patch_dataset_round_trip(tmp_path):
    import datasets

    rows = [
        _old_row(fn_name="make_acronym", outputs=['"MAS"', "6"]),
        _old_row(fn_name="f", outputs=["[1, 2]"]),
        _old_row(outputs=["1\n"]),
    ]
    src = tmp_path / "train_code.parquet"
    dst = tmp_path / "code_v2" / "train_code.parquet"
    datasets.Dataset.from_list(rows).to_parquet(str(src))

    stats = patch_dataset(str(src), str(dst))
    assert stats["total_rows"] == 3
    assert stats["call_based_rows"] == 2
    assert stats["prompts_updated"] == 2
    assert stats["outputs_unquoted"] == 1

    out = datasets.load_dataset("parquet", data_files=str(dst), split="train").to_list()
    assert len(out) == 3
    assert "make_acronym" in out[0]["prompt"][0]["content"]
    assert json.loads(out[0]["reward_model"]["ground_truth"])["outputs"] == ["MAS", "6"]
    # stdio row byte-identical
    assert out[2]["prompt"] == rows[2]["prompt"]


def test_patch_dataset_dry_run_writes_nothing(tmp_path):
    import datasets

    src = tmp_path / "train_code.parquet"
    dst = tmp_path / "out" / "train_code.parquet"
    datasets.Dataset.from_list([_old_row(fn_name="f", outputs=['"x"'])]).to_parquet(str(src))

    patch_dataset(str(src), str(dst), dry_run=True)
    assert not dst.exists()


class TestShardedInput:
    """The original code parquet may be sharded (00000.parquet, 00001.parquet, ...)."""

    @staticmethod
    def _write_shards(code_dir):
        import datasets

        code_dir.mkdir(parents=True, exist_ok=True)
        shards = [
            [_old_row(fn_name="make_acronym", outputs=['"MAS"'])],
            [_old_row(fn_name="f", outputs=['"x"']), _old_row(outputs=["1\n"])],
        ]
        for i, rows in enumerate(shards):
            datasets.Dataset.from_list(rows).to_parquet(str(code_dir / f"{i:05d}.parquet"))

    def test_directory_input_writes_one_output_per_shard(self, tmp_path):
        import datasets

        code_dir = tmp_path / "code"
        self._write_shards(code_dir)
        out_dir = tmp_path / "code_v2"

        stats = patch_dataset(str(code_dir), str(out_dir))
        assert stats["input_shards"] == 2
        assert stats["total_rows"] == 3
        assert stats["call_based_rows"] == 2
        assert stats["outputs_unquoted"] == 2
        assert [os.path.basename(p) for p in stats["output_files"]] == ["00000.parquet", "00001.parquet"]

        # Sharding is preserved so no single Arrow string column has to hold everything.
        assert not (out_dir / "train_code.parquet").exists()
        first = datasets.load_dataset("parquet", data_files=str(out_dir / "00000.parquet"), split="train").to_list()
        assert len(first) == 1
        assert "make_acronym" in first[0]["prompt"][0]["content"]
        assert json.loads(first[0]["reward_model"]["ground_truth"])["outputs"] == ["MAS"]

        merged = datasets.load_dataset("parquet", data_files=str(out_dir / "*.parquet"), split="train").to_list()
        assert len(merged) == 3

    def test_single_input_and_directory_output_uses_train_code_name(self, tmp_path):
        code_dir = tmp_path / "code"
        self._write_shards(code_dir)
        out_dir = tmp_path / "code_v2"
        stats = patch_dataset(str(code_dir / "00000.parquet"), str(out_dir))
        assert stats["output_files"] == [str(out_dir / "train_code.parquet")]
        assert (out_dir / "train_code.parquet").is_file()

    def test_multiple_file_inputs_respect_order(self, tmp_path):
        code_dir = tmp_path / "code"
        self._write_shards(code_dir)
        files = [str(code_dir / "00001.parquet"), str(code_dir / "00000.parquet")]

        stats = patch_dataset(files, str(tmp_path / "out"))
        assert stats["input_shards"] == 2
        assert stats["total_rows"] == 3
        # explicit file order is respected (00001 first)
        assert stats["input_files"][0].endswith("00001.parquet")
        assert [os.path.basename(p) for p in stats["output_files"]] == ["00001.parquet", "00000.parquet"]

    def test_multiple_shards_with_file_output_is_fatal(self, tmp_path):
        code_dir = tmp_path / "code"
        self._write_shards(code_dir)
        with pytest.raises(SystemExit, match="offset overflow"):
            patch_dataset(str(code_dir), str(tmp_path / "merged.parquet"))

    def test_input_directory_without_parquet_is_fatal(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(SystemExit, match="no \\*.parquet found"):
            patch_dataset(str(empty), str(tmp_path / "out"))

    def test_missing_input_is_fatal(self, tmp_path):
        with pytest.raises(SystemExit, match="input not found"):
            patch_dataset(str(tmp_path / "nope.parquet"), str(tmp_path / "out"))

    def test_single_input_with_parquet_suffix_is_a_file(self, tmp_path):
        import datasets

        src = tmp_path / "00000.parquet"
        datasets.Dataset.from_list([_old_row(fn_name="f", outputs=['"x"'])]).to_parquet(str(src))
        target = tmp_path / "custom_name.parquet"
        patch_dataset(str(src), str(target))
        assert target.is_file()

    def test_prompt_guard_survives_missing_prompt(self):
        # A malformed row must not crash the whole shard.
        row = _old_row(fn_name="f", outputs=['"x"'])
        del row["prompt"]
        assert patch_row(row)[0] is False
