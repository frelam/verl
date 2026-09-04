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
"""CPU tests for the tool_rl tiered hard-sample replay (pool + HardReplaySampler + dataset mixing).

The sampler-side tests run against a real in-process TransferQueue (same pattern as
``test_replay_buffer_on_cpu.py``); trajectories additionally carry the dataset row fields
(``raw_prompt``/``extra_info``/...) that the replay export pass reads back.
"""

import random
import uuid

import datasets
import pytest
import torch
import transfer_queue as tq
from omegaconf import OmegaConf

from examples.tool_rl.hard_replay import (
    HARD_REPLAY_TAG,
    HardReplayPool,
    HardReplaySampler,
    _row_key,
    entry_to_row_dict,
    get_hard_pool,
    reset_hard_pool,
)
from examples.tool_rl.tool_rl_dataset import ToolRLHintDataset

POLL_INTERVAL = 0.05

# Metric name used as the DAPO filter metric, mirroring ``algorithm.filter_groups.metric=score``.
METRIC = "acc"

ROW = {
    "raw_prompt": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 6 * 7?"},
    ],
    "tools": [],
    "data_source": "tool_rl_test",
    "reward_model": {"style": "rule", "ground_truth": "42"},
    "extra_info": {"index": 0, "tools_kwargs": {}, "interaction_kwargs": {}},
}


@pytest.fixture(autouse=True)
def _fresh_pool():
    reset_hard_pool()
    yield
    reset_hard_pool()


@pytest.fixture(scope="module")
def tq_init():
    tq.init()
    yield
    tq.close()


@pytest.fixture
def partition_id():
    return f"test-{uuid.uuid4().hex}"


def _uid() -> str:
    # uid must not contain "_" because ReplayBuffer derives it via key.split("_")[0].
    return uuid.uuid4().hex


def _produce_group(partition_id: str, task_id: str, rewards: list[float], extra_info: dict | None = None) -> str:
    """Write one finished group whose trajectories carry row fields plus the reward metric."""
    uid = _uid()
    for session_id, reward in enumerate(rewards):
        fields = dict(ROW)
        ei = dict(ROW["extra_info"])
        ei["task_id"] = task_id
        if extra_info:
            ei.update(extra_info)
        fields["extra_info"] = ei
        fields["input_ids"] = torch.tensor([1, 2, 3])
        fields["extra_fields"] = {"reward_extra_info": {METRIC: float(reward)}}
        tq.kv_put(
            key=f"{uid}_{session_id}_0",
            partition_id=partition_id,
            fields=fields,
            tag={"is_prompt": False, "seq_len": 3, "global_steps": 0},
        )
    tq.kv_put(
        key=uid,
        partition_id=partition_id,
        tag={"is_prompt": True, "status": "finished", "global_steps": 0},
    )
    return uid


def _clear_partition(partition_id: str) -> None:
    keys = list(tq.kv_list(partition_id=partition_id).get(partition_id, {}).keys())
    if keys:
        tq.kv_clear(keys=keys, partition_id=partition_id)


def _make_sampler(**sampler_kwargs) -> HardReplaySampler:
    kwargs = {"filter_metric": METRIC, "train_batch_size": 1}
    kwargs.update(sampler_kwargs)
    return HardReplaySampler(
        trainer_mode="sync",
        trainer_config={},
        max_off_policy_threshold=8,
        max_off_policy_strategy="drop",
        sampler_kwargs=kwargs,
        poll_interval=POLL_INTERVAL,
        # DAPO filtering requires a refill_fn; tests pre-produce every group, so this is a no-op.
        refill_fn=lambda _n: None,
    )


def _classify(rb: HardReplaySampler, partition_id: str):
    """Drive the exact hook sample() uses, without entering the blocking sample loop."""
    rb._sync_metadata_from_transfer_queue()
    return rb._dapo_filtered_keys(partition_id)


# --------------------------------------------------------------------------- #
# HardReplayPool lifecycle (no TransferQueue).
# --------------------------------------------------------------------------- #


def test_pool_tier_for_boundaries():
    pool = HardReplayPool(medium_threshold=0.5, zero_threshold=0.01)
    assert pool.tier_for(0.0) == "hard"
    assert pool.tier_for(0.01) == "hard"  # at the zero threshold -> still hard
    assert pool.tier_for(0.05) == "medium"
    assert pool.tier_for(0.49) == "medium"
    assert pool.tier_for(0.5) is None  # at the medium threshold -> graduated
    assert pool.tier_for(1.0) is None


def test_pool_add_dedups_by_key_and_retiers():
    pool = HardReplayPool()
    assert pool.add("a", {"extra_info": {"task_id": "a"}}, pass_rate=0.0) is True
    assert pool.entries["a"].tier == "hard"
    assert pool.add("a", {"extra_info": {"task_id": "a"}}, pass_rate=0.25) is False
    assert len(pool) == 1 and pool.num_available == 1
    assert pool.entries["a"].tier == "medium" and pool.entries["a"].pass_rate == 0.25


def test_pool_maybe_take_respects_ratio_state_and_due_step():
    pool = HardReplayPool(hard_interval=20)
    pool.add("a", {}, pass_rate=0.0)  # added at step 0
    rng = random.Random(0)
    pool.current_step = 5
    assert pool.maybe_take(rng, 1.0) is None  # not due yet (5 < 20)
    pool.current_step = 20
    assert pool.maybe_take(rng, 0.0) is None  # ratio 0 never replays
    entry = pool.maybe_take(rng, 1.0)
    assert entry is not None and entry.key == "a" and entry.state == "inflight"
    assert entry.last_replay_step == 20
    assert pool.maybe_take(rng, 1.0) is None  # nothing available anymore
    assert pool.num_available == 0


def test_pool_medium_tier_is_due_earlier_than_hard():
    pool = HardReplayPool(medium_interval=10, hard_interval=20)
    pool.add("m", {}, pass_rate=0.25)
    pool.add("h", {}, pass_rate=0.0)
    rng = random.Random(0)
    pool.current_step = 10
    entry = pool.maybe_take(rng, 1.0)
    assert entry is not None and entry.key == "m"  # only the medium entry is due
    pool.current_step = 20
    entry = pool.maybe_take(rng, 1.0)
    assert entry is not None and entry.key == "h"


def test_pool_maybe_take_caps_dispatches_per_step():
    pool = HardReplayPool(max_per_step=2)
    for i in range(5):
        pool.add(f"k{i}", {}, pass_rate=0.0)
    rng = random.Random(0)
    pool.current_step = 100
    assert pool.maybe_take(rng, 1.0) is not None
    assert pool.maybe_take(rng, 1.0) is not None
    assert pool.maybe_take(rng, 1.0) is None  # cap reached for this step
    pool.current_step = 101  # next step resets the counter
    assert pool.maybe_take(rng, 1.0) is not None


def test_pool_resolve_graduates_on_improvement():
    pool = HardReplayPool()
    pool.add("a", {}, pass_rate=0.0)
    pool.current_step = 100
    pool.maybe_take(random.Random(0), 1.0)
    pool.resolve_replay("a", pass_rate=0.75)  # >= medium_threshold -> out of the pool
    assert len(pool) == 0
    pool.resolve_replay("a", pass_rate=0.0)  # unknown key is a no-op


def test_pool_resolve_rearms_and_retiers_when_still_low():
    pool = HardReplayPool()
    pool.add("a", {}, pass_rate=0.0)
    pool.current_step = 100
    pool.maybe_take(random.Random(0), 1.0)
    pool.resolve_replay("a", pass_rate=0.25)  # hard -> medium
    entry = pool.entries["a"]
    assert entry.state == "available" and entry.tier == "medium"
    assert entry.replay_count == 1 and entry.pass_rate == 0.25


def test_pool_gives_up_after_max_replays():
    pool = HardReplayPool(max_replays=2)
    pool.add("a", {}, pass_rate=0.0)
    pool.current_step = 100
    pool.maybe_take(random.Random(0), 1.0)
    pool.resolve_replay("a", pass_rate=0.0)
    assert pool.entries["a"].replay_count == 1
    pool.current_step = 200
    assert pool.maybe_take(random.Random(0), 1.0) is not None
    pool.resolve_replay("a", pass_rate=0.0)
    assert "a" not in pool.entries  # replay_count hit max_replays -> dropped


def test_pool_end_of_pass_recycles_leaked_inflight_entries():
    pool = HardReplayPool(recycle_after_passes=2)
    pool.add("a", {}, pass_rate=0.0)
    pool.current_step = 100
    pool.maybe_take(random.Random(0), 1.0)
    pool.end_of_pass()
    assert pool.entries["a"].state == "inflight"
    pool.end_of_pass()
    pool.end_of_pass()
    assert pool.entries["a"].state == "available"  # replayed group never finished -> re-armed


def test_get_hard_pool_reconfigures_existing_singleton():
    pool = get_hard_pool(medium_interval=7)
    assert pool.medium_interval == 7
    assert get_hard_pool() is pool and pool.medium_interval == 7  # no kwargs -> untouched
    get_hard_pool(medium_interval=10)
    assert pool.medium_interval == 10


def test_row_key_prefers_task_id_and_falls_back_to_prompt_hash():
    assert _row_key({"extra_info": {"task_id": "t-1"}, "raw_prompt": []}) == "t-1"
    hash_key = _row_key({"extra_info": {}, "raw_prompt": [{"role": "user", "content": "q"}]})
    assert hash_key.startswith("hash:")
    assert hash_key == _row_key({"raw_prompt": [{"role": "user", "content": "q"}]})


def test_entry_to_row_dict_tags_replay_and_keeps_raw_prompt():
    pool = HardReplayPool()
    pool.add("t-1", dict(ROW), pass_rate=0.0)
    pool.current_step = 100
    entry = pool.maybe_take(random.Random(0), 1.0)
    row = entry_to_row_dict(entry)
    assert row["extra_info"][HARD_REPLAY_TAG] == "t-1"
    assert row["raw_prompt"] == ROW["raw_prompt"]  # hint-augmented prompt preserved as-is
    assert row["dummy_tensor"].shape == (1,)
    assert row["index"] == ROW["extra_info"]["index"]
    assert row["tools_kwargs"] == {}


# --------------------------------------------------------------------------- #
# HardReplaySampler against a real TransferQueue.
# --------------------------------------------------------------------------- #


def test_sampler_requires_train_batch_size_for_filtering():
    with pytest.raises(ValueError, match="train_batch_size"):
        HardReplaySampler(
            trainer_mode="sync",
            trainer_config={},
            max_off_policy_threshold=8,
            max_off_policy_strategy="drop",
            sampler_kwargs={"filter_metric": METRIC},
        )


def test_sampler_maps_max_replay_fraction_to_per_step_cap():
    rb = _make_sampler()  # train_batch_size=1, default fraction 0.2 -> cap at least 1
    assert rb.pool.max_per_step == 1
    rb = _make_sampler(max_replay_fraction=0.0)
    assert rb.pool.max_per_step == 0  # unlimited


def test_sampler_rejects_async_mode():
    with pytest.raises(ValueError, match="sync"):
        HardReplaySampler(
            trainer_mode="colocate_async",
            trainer_config={},
            max_off_policy_threshold=8,
            max_off_policy_strategy="drop",
            sampler_kwargs={},
        )


def test_all_zero_group_is_filtered_and_exported_as_hard(tq_init, partition_id):
    zero_uid = _produce_group(partition_id, task_id="hard-0", rewards=[0.0, 0.0])
    mixed_uid = _produce_group(partition_id, task_id="easy-0", rewards=[0.0, 1.0])
    rb = _make_sampler()
    try:
        batch, metrics = rb.sample(global_steps=3, partition_id=partition_id, batch_size=1)
        sampled_uids = {key.split("_")[0] for key in batch.keys}
        assert sampled_uids == {mixed_uid}
        assert zero_uid not in sampled_uids
        assert metrics["validation/filter_groups/evicted_samples"] == 1

        pool = get_hard_pool()
        assert pool.current_step == 3  # sample() tracks the training step
        assert set(pool.entries) == {"hard-0"}
        entry = pool.entries["hard-0"]
        assert entry.state == "available" and entry.tier == "hard" and entry.pass_rate == 0.0
        assert entry.row["raw_prompt"] == ROW["raw_prompt"]
        assert entry.row["data_source"] == ROW["data_source"]
    finally:
        _clear_partition(partition_id)


def test_all_one_group_is_filtered_but_not_exported(tq_init, partition_id):
    _produce_group(partition_id, task_id="solved-0", rewards=[1.0, 1.0])
    mixed_uid = _produce_group(partition_id, task_id="easy-1", rewards=[0.0, 1.0])
    rb = _make_sampler()
    try:
        batch, _ = rb.sample(global_steps=0, partition_id=partition_id, batch_size=1)
        assert {key.split("_")[0] for key in batch.keys} == {mixed_uid}
        assert len(get_hard_pool()) == 0
    finally:
        _clear_partition(partition_id)


def test_medium_group_is_trained_and_exported_as_medium(tq_init, partition_id):
    # 1/4 passed -> pass rate 0.25: mixed group (gradient signal) AND medium-tier replay candidate.
    medium_uid = _produce_group(partition_id, task_id="mid-0", rewards=[0.0, 0.0, 0.0, 1.0])
    rb = _make_sampler()
    try:
        batch, _ = rb.sample(global_steps=0, partition_id=partition_id, batch_size=1)
        assert {key.split("_")[0] for key in batch.keys} == {medium_uid}  # trained normally
        pool = get_hard_pool()
        assert set(pool.entries) == {"mid-0"}
        entry = pool.entries["mid-0"]
        assert entry.tier == "medium" and entry.pass_rate == 0.25
    finally:
        _clear_partition(partition_id)


def test_high_pass_mixed_group_is_neither_filtered_nor_exported(tq_init, partition_id):
    # 3/4 passed -> pass rate 0.75 >= medium_threshold: trained, never pooled.
    easy_uid = _produce_group(partition_id, task_id="easy-2", rewards=[1.0, 1.0, 1.0, 0.0])
    rb = _make_sampler()
    try:
        batch, _ = rb.sample(global_steps=0, partition_id=partition_id, batch_size=1)
        assert {key.split("_")[0] for key in batch.keys} == {easy_uid}
        assert len(get_hard_pool()) == 0
    finally:
        _clear_partition(partition_id)


def test_same_hard_group_is_exported_once(tq_init, partition_id):
    _produce_group(partition_id, task_id="hard-1", rewards=[0.0, 0.0])
    rb = _make_sampler()
    try:
        dapo_uids, _ = _classify(rb, partition_id)
        assert len(dapo_uids) == 1
        assert len(rb.pool) == 1
        # A second classification pass over the same finished group must not duplicate.
        _classify(rb, partition_id)
        assert len(rb.pool) == 1
    finally:
        _clear_partition(partition_id)


def test_replayed_group_graduates_when_pass_rate_recovers(tq_init, partition_id):
    _produce_group(partition_id, task_id="hard-2", rewards=[0.0, 0.0])
    rb = _make_sampler()
    try:
        _classify(rb, partition_id)
        rb.pool.current_step = 100
        entry = rb.pool.maybe_take(random.Random(0), 1.0)
        assert entry is not None and entry.state == "inflight"

        # The replayed prompt comes back as a fresh group tagged with HARD_REPLAY_TAG.
        replay_uid = _produce_group(
            partition_id,
            task_id="hard-2",
            rewards=[1.0, 1.0, 0.0, 1.0],  # pass rate 0.75 >= medium_threshold
            extra_info={HARD_REPLAY_TAG: "hard-2"},
        )
        dapo_uids, _ = _classify(rb, partition_id)
        assert replay_uid not in dapo_uids  # mixed group carries gradient signal, kept for training
        assert len(rb.pool) == 0  # graduated out of the pool
    finally:
        _clear_partition(partition_id)


def test_replayed_group_all_zero_again_rearms_pool_entry(tq_init, partition_id):
    _produce_group(partition_id, task_id="hard-3", rewards=[0.0, 0.0])
    rb = _make_sampler(max_replays=3)
    try:
        _classify(rb, partition_id)
        rb.pool.current_step = 100
        assert rb.pool.maybe_take(random.Random(0), 1.0) is not None

        replay_uid = _produce_group(
            partition_id, task_id="hard-3", rewards=[0.0, 0.0], extra_info={HARD_REPLAY_TAG: "hard-3"}
        )
        dapo_uids, _ = _classify(rb, partition_id)
        assert replay_uid in dapo_uids  # still filtered: no gradient signal
        assert len(rb.pool) == 1  # but not duplicated by the export pass
        entry = rb.pool.entries["hard-3"]
        assert entry.state == "available" and entry.tier == "hard" and entry.replay_count == 1
    finally:
        _clear_partition(partition_id)


def test_replayed_group_improving_to_medium_retiers_pool_entry(tq_init, partition_id):
    _produce_group(partition_id, task_id="hard-4", rewards=[0.0, 0.0])
    rb = _make_sampler()
    try:
        _classify(rb, partition_id)
        rb.pool.current_step = 100
        assert rb.pool.maybe_take(random.Random(0), 1.0) is not None

        _produce_group(
            partition_id,
            task_id="hard-4",
            rewards=[0.0, 0.0, 0.0, 1.0],  # pass rate 0.25 -> medium tier
            extra_info={HARD_REPLAY_TAG: "hard-4"},
        )
        _classify(rb, partition_id)
        entry = rb.pool.entries["hard-4"]
        assert entry.state == "available" and entry.tier == "medium" and entry.replay_count == 1
    finally:
        _clear_partition(partition_id)


# --------------------------------------------------------------------------- #
# Dataset-side mixing (replay path never touches the tokenizer).
# --------------------------------------------------------------------------- #


def _write_parquet(tmp_path, name: str) -> str:
    # Empty structs (e.g. tools_kwargs={}) cannot be written to parquet; add a dummy child.
    extra_info = dict(ROW["extra_info"])
    extra_info["tools_kwargs"] = {"__dummy__": 0}
    extra_info["interaction_kwargs"] = {"__dummy__": 0}
    rows = [
        {
            "data_source": ROW["data_source"],
            "prompt": ROW["raw_prompt"],
            "ability": "tool_rl",
            "reward_model": ROW["reward_model"],
            "extra_info": extra_info,
        }
    ]
    path = str(tmp_path / name)
    datasets.Dataset.from_list(rows).to_parquet(path)
    return path


def _make_dataset(path: str) -> ToolRLHintDataset:
    config = OmegaConf.create({"prompt_key": "prompt", "filter_overlong_prompts": False})
    return ToolRLHintDataset(data_files=path, tokenizer=object(), processor=None, config=config)


def test_dataset_replays_due_pool_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("TOOL_RL_HARD_REPLAY", "1")
    monkeypatch.setenv("TOOL_RL_REPLAY_RATIO", "1.0")
    monkeypatch.setenv("TOOL_RL_HINT_INJECTION", "off")
    pool = get_hard_pool()
    pool.add("t-9", dict(ROW), pass_rate=0.0)

    dataset = _make_dataset(_write_parquet(tmp_path, "train.parquet"))
    assert dataset._replay_enabled is True

    pool.current_step = 5  # not due yet (hard tier, interval 20)
    row = dataset[0]
    assert HARD_REPLAY_TAG not in row["extra_info"]

    pool.current_step = 100
    row = dataset[0]
    assert row["extra_info"][HARD_REPLAY_TAG] == "t-9"
    assert row["raw_prompt"] == ROW["raw_prompt"]
    assert pool.entries["t-9"].state == "inflight"


def test_dataset_replay_disabled_by_default_and_for_val_files(tmp_path, monkeypatch):
    monkeypatch.delenv("TOOL_RL_HARD_REPLAY", raising=False)
    monkeypatch.setenv("TOOL_RL_HINT_INJECTION", "off")
    train_dataset = _make_dataset(_write_parquet(tmp_path, "train.parquet"))
    assert train_dataset._replay_enabled is False

    monkeypatch.setenv("TOOL_RL_HARD_REPLAY", "1")
    val_dataset = _make_dataset(_write_parquet(tmp_path, "val.parquet"))
    assert val_dataset._replay_enabled is False
