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
"""Tests for mix_replay.py (replaying a previous mix with rebuilt domains).

Run from the repo root:
    pytest examples/reasoning_rl/scripts/test_mix_replay.py -v
"""

import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mix import read_parquet_rows, resolve_parquet_files  # noqa: E402
from mix_replay import _resolve_override, main, replay  # noqa: E402

MIX_PY = os.path.join(HERE, "mix.py")


def _write_domain(path: str, domain: str, rows: list[tuple[str, int]]) -> None:
    """Write a domain parquet from ``[(source, n_rows), ...]``."""
    import datasets

    records = []
    for source, n in rows:
        for i in range(n):
            records.append(
                {
                    "data_source": f"{domain}_{source}",
                    "prompt": [{"role": "user", "content": "q"}],
                    "ability": domain,
                    "reward_model": {"style": "rule", "ground_truth": "x"},
                    "extra_info": {
                        "split": "train",
                        "index": i,
                        "task_id": f"{source}-{i}",
                        "domain": domain,
                        "source": source,
                        "difficulty": "",
                        "prior_solve_rate": -1.0,
                        "seed": -1,
                    },
                }
            )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    datasets.Dataset.from_list(records).to_parquet(path)


@pytest.fixture
def raw_layout(tmp_path):
    """math/stem unchanged; code and logic have a v2 rebuild with new sources."""
    raw = tmp_path / "raw"
    _write_domain(str(raw / "math" / "train_math.parquet"), "math", [("bigmath", 1000)])
    _write_domain(str(raw / "stem" / "train_stem.parquet"), "stem", [("drsci", 1000)])
    _write_domain(str(raw / "code" / "train_code.parquet"), "code", [("deepcoder", 1000)])
    _write_domain(str(raw / "logic" / "train_logic.parquet"), "logic", [("enigmata", 1000)])
    # rebuilt domains: more rows and an extra source each
    _write_domain(str(raw / "code_v2" / "train_code.parquet"), "code", [("deepcoder", 500), ("apps", 900)])
    _write_domain(str(raw / "logic_v2" / "train_logic.parquet"), "logic", [("enigmata", 700), ("synlogic", 800)])
    return raw


def _run_old_mix(raw, out_dir):
    subprocess.run(
        [sys.executable, MIX_PY, "--input_dir", str(raw), "--output_dir", str(out_dir)],
        check=True,
    )
    with open(os.path.join(out_dir, "mix_stats.json"), encoding="utf-8") as f:
        return json.load(f)


def test_replay_keeps_ratios_total_and_ability(raw_layout, tmp_path):
    old_dir = tmp_path / "final"
    old_stats = _run_old_mix(raw_layout, old_dir)

    new_dir = tmp_path / "final_v2"
    new_stats, _ = replay(
        old_stats_path=str(old_dir / "mix_stats.json"),
        input_dir=str(raw_layout),
        output_dir=str(new_dir),
        path_overrides={
            "code": str(raw_layout / "code_v2" / "train_code.parquet"),
            "logic": str(raw_layout / "logic_v2" / "train_logic.parquet"),
        },
    )

    # Configuration replayed exactly.
    assert new_stats["total_train"] == old_stats["total_train"]
    assert new_stats["total_val"] == old_stats["total_val"]
    assert new_stats["ratios"] == old_stats["ratios"]
    assert new_stats["train_by_ability"] == old_stats["train_by_ability"]
    assert new_stats["val_by_ability"] == old_stats["val_by_ability"]

    # The rebuilt pools were actually used.
    assert "apps" in new_stats["train_by_source"]
    assert "synlogic" in new_stats["train_by_source"]
    assert "apps" not in old_stats["train_by_source"]

    # Artifacts written.
    assert (new_dir / "train.parquet").is_file()
    assert (new_dir / "val.parquet").is_file()
    assert (new_dir / "mix_stats.json").is_file()
    report = json.loads((new_dir / "mix_replay_report.json").read_text())
    assert report["configuration_replayed"] is True
    assert report["replenished_with_replacement"] == []


def test_dry_run_writes_nothing(raw_layout, tmp_path):
    old_dir = tmp_path / "final"
    _run_old_mix(raw_layout, old_dir)
    out_dir = tmp_path / "dry"
    replay(
        old_stats_path=str(old_dir / "mix_stats.json"),
        input_dir=str(raw_layout),
        output_dir=str(out_dir),
        dry_run=True,
    )
    assert not out_dir.exists()


def test_strict_aborts_when_pool_too_small(raw_layout, tmp_path):
    old_dir = tmp_path / "final"
    _run_old_mix(raw_layout, old_dir)
    with pytest.raises(SystemExit, match="strict"):
        replay(
            old_stats_path=str(old_dir / "mix_stats.json"),
            input_dir=str(raw_layout),
            output_dir=str(tmp_path / "strict"),
            total_size=100_000,
            strict=True,
        )


def test_explicit_total_size_overrides_old(raw_layout, tmp_path):
    old_dir = tmp_path / "final"
    _run_old_mix(raw_layout, old_dir)
    new_stats, _ = replay(
        old_stats_path=str(old_dir / "mix_stats.json"),
        input_dir=str(raw_layout),
        output_dir=str(tmp_path / "small"),
        total_size=200,
    )
    assert new_stats["total_train"] == sum(int(200 * r) for r in new_stats["ratios"].values())


def test_missing_old_stats_is_fatal(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        replay(
            old_stats_path=str(tmp_path / "nope" / "mix_stats.json"),
            input_dir=str(tmp_path),
            output_dir=str(tmp_path / "out"),
        )


class TestResolveOverride:
    """--new_code/--new_logic accept the rebuild directory as well as the file."""

    def test_none_and_empty(self, tmp_path):
        assert _resolve_override("code", None, str(tmp_path)) is None
        assert _resolve_override("code", "", str(tmp_path)) is None

    def test_file_path(self, tmp_path):
        target = tmp_path / "train_code.parquet"
        target.write_bytes(b"")
        assert _resolve_override("code", str(target), str(tmp_path)) == str(target)

    def test_directory_resolves_to_train_file(self, tmp_path):
        target = tmp_path / "code_v2" / "train_code.parquet"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"")
        assert _resolve_override("code", str(target.parent), str(tmp_path)) == str(target)

    def test_directory_with_shards_returns_glob(self, tmp_path):
        shard_dir = tmp_path / "code_v2"
        shard_dir.mkdir(parents=True)
        for i in range(2):
            (shard_dir / f"{i:05d}.parquet").write_bytes(b"")
        assert _resolve_override("code", str(shard_dir), str(tmp_path)) == str(shard_dir / "*.parquet")

    def test_directory_with_single_shard_returns_file(self, tmp_path):
        shard_dir = tmp_path / "code_v2"
        shard_dir.mkdir(parents=True)
        shard = shard_dir / "00000.parquet"
        shard.write_bytes(b"")
        assert _resolve_override("code", str(shard_dir), str(tmp_path)) == str(shard)

    def test_empty_directory_is_fatal(self, tmp_path):
        empty = tmp_path / "code_v2"
        empty.mkdir()
        with pytest.raises(SystemExit, match="empty"):
            _resolve_override("code", str(empty), str(tmp_path))

    def test_relative_override_resolves_against_input_dir(self, tmp_path):
        target = tmp_path / "code_v2" / "train_code.parquet"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"")
        assert _resolve_override("code", "code_v2", str(tmp_path)) == str(target)

    def test_missing_path_reports_absolute(self, tmp_path):
        with pytest.raises(SystemExit, match="not found"):
            _resolve_override("code", "nope/train_code.parquet", str(tmp_path))


def test_replay_reads_sharded_override(raw_layout, tmp_path):
    """A sharded code_v2 (00000.parquet, ...) is loaded via the glob override."""
    import datasets

    old_dir = tmp_path / "final"
    old_stats = _run_old_mix(raw_layout, old_dir)

    rows = datasets.load_dataset(
        "parquet", data_files=str(raw_layout / "code_v2" / "train_code.parquet"), split="train"
    ).to_list()
    shard_dir = tmp_path / "code_shards"
    shard_dir.mkdir()
    half = len(rows) // 2
    datasets.Dataset.from_list(rows[:half]).to_parquet(str(shard_dir / "00000.parquet"))
    datasets.Dataset.from_list(rows[half:]).to_parquet(str(shard_dir / "00001.parquet"))

    new_stats, _ = replay(
        old_stats_path=str(old_dir / "mix_stats.json"),
        input_dir=str(raw_layout),
        output_dir=str(tmp_path / "final_v2"),
        path_overrides={"code": str(shard_dir)},
    )
    assert new_stats["total_train"] == old_stats["total_train"]
    assert new_stats["train_by_ability"] == old_stats["train_by_ability"]
    assert "apps" in new_stats["train_by_source"]


def test_main_accepts_directory_overrides(raw_layout, tmp_path):
    old_dir = tmp_path / "final"
    _run_old_mix(raw_layout, old_dir)
    out_dir = tmp_path / "final_v2"
    rc = main(
        [
            "--old_dir",
            str(old_dir),
            "--input_dir",
            str(raw_layout),
            "--new_code",
            str(raw_layout / "code_v2"),
            "--new_logic",
            str(raw_layout / "logic_v2"),
            "--output_dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    assert (out_dir / "train.parquet").is_file()
    assert (out_dir / "val.parquet").is_file()


class TestReadParquetRows:
    """mix.load_domain reads parquet in batches instead of datasets.to_list().

    datasets builds one Arrow string column per domain, which overflows 32-bit
    offsets past ~2 GB of text (the Dr.SCI stem pool).  pyarrow RecordBatches
    keep the offsets local, and must return identical rows.
    """

    def test_matches_datasets_output(self, raw_layout):
        import datasets

        path = str(raw_layout / "code_v2" / "train_code.parquet")
        got = read_parquet_rows(path)
        expected = datasets.load_dataset("parquet", data_files=path, split="train").to_list()
        assert len(got) == len(expected)
        assert [r["data_source"] for r in got] == [r["data_source"] for r in expected]
        assert [r["extra_info"]["source"] for r in got] == [r["extra_info"]["source"] for r in expected]
        assert got[0]["prompt"] == expected[0]["prompt"]
        assert got[0]["reward_model"] == expected[0]["reward_model"]

    def test_directory_and_glob(self, raw_layout):
        code_dir = raw_layout / "code_v2"
        assert len(resolve_parquet_files(str(code_dir))) == 1
        assert len(resolve_parquet_files(str(code_dir / "*.parquet"))) == 1
        assert len(read_parquet_rows(str(code_dir))) == 1400

    def test_missing_path_is_fatal(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no parquet files matched"):
            resolve_parquet_files(str(tmp_path / "nope.parquet"))
