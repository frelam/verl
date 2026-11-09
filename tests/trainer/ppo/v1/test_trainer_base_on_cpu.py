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

from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer, ReplayBufferAsync
from verl.trainer.ppo.v1.trainer_base import DispatchLedger, PPOTrainer


class _StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


class _CustomSampler:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _trainer_with_filter_groups(filter_groups: dict, trainer_mode: str = "sync") -> _StubTrainer:
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.trainer_mode = trainer_mode
    trainer.async_trajectory_config = {}
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"filter_groups": filter_groups},
            "data": {"train_batch_size": 64, "gen_batch_size": 8},
            "reward": {"reward_model": {"enable": False, "enable_resource_pool": False}},
            "trainer": {
                "v1": {
                    trainer_mode: {},
                    "sampler": {
                        "custom_sampler": None,
                        "max_off_policy_threshold": 1,
                        "max_off_policy_strategy": "drop",
                        "sampler_kwargs": {},
                    },
                }
            },
        }
    )
    return trainer


def test_builtin_sampler_class_follows_trainer_mode():
    sync_sampler = _trainer_with_filter_groups({"enable": False}, trainer_mode="sync")._build_replay_buffer()
    async_samplers = [
        _trainer_with_filter_groups({"enable": True, "metric": "acc"}, trainer_mode=mode)._build_replay_buffer()
        for mode in ("colocate_async", "separate_async")
    ]

    assert type(sync_sampler) is ReplayBuffer
    assert all(type(sampler) is ReplayBufferAsync for sampler in async_samplers)
    assert all(sampler.filter_groups_metric == "acc" for sampler in async_samplers)
    assert all(sampler.train_batch_size is None for sampler in async_samplers)
    assert all(sampler.gen_batch_size is None for sampler in async_samplers)


def test_custom_sampler_skips_builtin_filter_groups_validation():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc"})
    trainer.config.trainer.v1.sampler.custom_sampler = {"path": "custom.py", "name": "CustomSampler"}

    with (
        patch("verl.trainer.ppo.v1.trainer_base.load_extern_type", return_value=_CustomSampler),
        patch.object(trainer, "_resolve_filter_groups_metric") as resolve_filter_groups_metric,
    ):
        sampler = trainer._build_replay_buffer()

    resolve_filter_groups_metric.assert_not_called()
    assert isinstance(sampler, _CustomSampler)
    assert "filter_groups_metric" not in sampler.kwargs
    assert "train_batch_size" not in sampler.kwargs
    assert "gen_batch_size" not in sampler.kwargs
    assert "max_inflight_gen_batches" not in sampler.kwargs
    assert "sync_refill_failed_groups" not in sampler.kwargs


def test_builtin_filter_groups_uses_default_inflight_limit():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc"})

    sampler = trainer._build_replay_buffer()

    assert sampler.filter_groups_metric == "acc"
    assert sampler.train_batch_size == 64
    assert sampler.gen_batch_size == 1
    assert sampler.max_inflight_gen_batches == 1


def test_builtin_filter_groups_forwards_configured_inflight_limit():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc", "max_inflight_gen_batches": 3})

    sampler = trainer._build_replay_buffer()

    assert sampler.max_inflight_gen_batches == 3


def test_builtin_sync_failure_refill_forces_single_prompt_generation():
    trainer = _trainer_with_filter_groups({"enable": False})
    trainer.config.trainer.v1.sampler.sync_refill_failed_groups = True

    sampler = trainer._build_replay_buffer()

    assert sampler.sync_refill_failed_groups is True
    assert sampler.gen_batch_size == 1


def test_sync_failure_refill_overrides_dataloader_generation_batch_size():
    trainer = _trainer_with_filter_groups({"enable": False})
    trainer.config.trainer.v1.sampler.sync_refill_failed_groups = True
    trainer.config.data.update(
        {
            "train_files": [],
            "val_files": [],
            "train_max_samples": -1,
            "val_max_samples": -1,
            "dataloader_num_workers": 0,
            "val_batch_size": 1,
            "validation_shuffle": False,
        }
    )
    trainer.config.trainer.total_epochs = 1
    trainer.config.trainer.total_training_steps = None
    trainer.parameter_sync_step = 1
    trainer.tokenizer = None
    trainer.processor = None

    with (
        patch("verl.trainer.ppo.v1.trainer_base.create_rl_dataset", side_effect=[[{}, {}], [{}]]),
        patch("verl.trainer.ppo.v1.trainer_base.create_rl_sampler", return_value=None),
        patch("verl.trainer.ppo.v1.trainer_base.StatefulDataLoader") as dataloader,
        patch("verl.trainer.ppo.v1.trainer_base.logger.warning") as warning,
    ):
        trainer._init_dataloader()

    assert trainer.config.data.gen_batch_size == 1
    assert dataloader.call_args_list[0].kwargs["batch_size"] == 1
    warning.assert_any_call("data.gen_batch_size=8 is overridden to 1.")


def test_builtin_filter_groups_warns_when_total_generation_limit_is_configured():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc", "max_num_gen_batches": 10})

    with patch("verl.trainer.ppo.v1.trainer_base.logger.warning") as warning:
        trainer._build_replay_buffer()

    warning.assert_called_once_with(
        "algorithm.filter_groups.max_num_gen_batches=%s is ignored by the built-in V1 ReplayBuffer; "
        "use max_inflight_gen_batches to bound concurrent Sync DAPO generation.",
        10,
    )


class _DispatchReplayBuffer:
    def __init__(
        self,
        *,
        outstanding: int,
        potential: int,
        max_outstanding_groups: int,
        high_watermark_ratio: float,
        low_watermark_ratio: float,
        capacity_policy: str,
        drop_once: int = 0,
    ):
        self.outstanding = {f"uid-{index}" for index in range(outstanding)}
        self.potential = potential
        self.max_outstanding_groups = max_outstanding_groups
        self.high_watermark_ratio = high_watermark_ratio
        self.low_watermark_ratio = low_watermark_ratio
        self.capacity_policy = capacity_policy
        self.drop_once = drop_once

    def enforce_capacity(self, global_steps: int, partition_id: str) -> dict:
        del global_steps, partition_id
        dropped = min(self.drop_once, len(self.outstanding))
        self.drop_once = 0
        if not dropped:
            return {}
        self.outstanding = set(sorted(self.outstanding)[dropped:])
        return {"rollout/capacity_dropped_groups": float(dropped)}

    def outstanding_keys(self, partition_id: str) -> set[str]:
        del partition_id
        return set(self.outstanding)

    def get_outstanding_count(self, partition_id: str, sync_metadata: bool = True) -> int:
        del partition_id, sync_metadata
        return len(self.outstanding)

    def remaining_capacity(self, partition_id: str) -> int:
        del partition_id
        return max(0, self.max_outstanding_groups - len(self.outstanding))

    def get_potential_sampleable_count(self, global_steps: int, partition_id: str) -> int:
        del global_steps, partition_id
        return self.potential

    def get_dispatch_metrics(self, global_steps: int, partition_id: str) -> dict:
        del global_steps, partition_id
        return {"rollout/outstanding_groups": float(len(self.outstanding))}


def _trainer_with_dispatch(
    *,
    outstanding: int = 0,
    potential: int = 0,
    max_outstanding_groups: int = 4,
    train_batch_size: int = 2,
    gen_batch_size: int = 2,
    high_watermark_ratio: float = 0.75,
    low_watermark_ratio: float = 0.5,
    capacity_policy: str = "hybrid",
    drop_once: int = 0,
) -> _StubTrainer:
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.config = OmegaConf.create(
        {"data": {"train_batch_size": train_batch_size, "gen_batch_size": gen_batch_size}}
    )
    trainer.global_steps = 3
    trainer._dispatch_ledger = DispatchLedger()
    trainer._dispatch_paused = False
    trainer._dispatch_step_metrics = {}
    trainer.replay_buffer = _DispatchReplayBuffer(
        outstanding=outstanding,
        potential=potential,
        max_outstanding_groups=max_outstanding_groups,
        high_watermark_ratio=high_watermark_ratio,
        low_watermark_ratio=low_watermark_ratio,
        capacity_policy=capacity_policy,
        drop_once=drop_once,
    )
    return trainer


def _record_prompt_dispatch(trainer: _StubTrainer) -> list[tuple]:
    calls = []

    def next_train_batch(num_prompts: int, **kwargs):
        calls.append(("fetch", num_prompts, kwargs))
        return list(range(num_prompts))

    def submit_batch(batch: list, *, optional: bool = False) -> int:
        calls.append(("submit", len(batch), optional))
        return len(batch)

    trainer._next_train_batch = next_train_batch
    trainer._submit_batch_to_rollout = submit_batch
    return calls


def _record_fetched_batch(trainer: _StubTrainer, batch, *, optional: bool) -> None:
    trainer._dispatch_ledger.record_dispatch(
        list(batch["uid"]),
        trainer._last_batch_epochs,
        trainer._last_batch_borrowed,
        optional=optional,
    )


def test_bounded_optional_prefetch_pauses_at_high_and_resumes_at_low_watermark():
    trainer = _trainer_with_dispatch(outstanding=3, max_outstanding_groups=4)
    calls = _record_prompt_dispatch(trainer)

    metrics = trainer._add_batch_to_generate()
    assert calls == []
    assert trainer._dispatch_paused is True
    assert metrics["rollout/dispatch_skipped_steps"] == 1

    # Staying between the watermarks must not release the paused state.
    assert trainer._add_batch_to_generate()["rollout/dispatch_skipped_steps"] == 1
    assert calls == []

    trainer.replay_buffer.outstanding = {"uid-0", "uid-1"}
    trainer._add_batch_to_generate()
    assert trainer._dispatch_paused is False
    assert calls == [
        ("fetch", 2, {"allow_epoch_wrap": False, "required": False}),
        ("submit", 2, True),
    ]

    calls.clear()
    trainer.replay_buffer.outstanding = {f"uid-{index}" for index in range(4)}
    trainer._add_batch_to_generate()
    assert trainer._dispatch_paused is True
    assert calls == []


@pytest.mark.parametrize("capacity", [2, 6])
def test_first_optional_prefetch_can_fill_hard_cap_with_full_batches(capacity: int):
    trainer = _trainer_with_dispatch(max_outstanding_groups=capacity)
    calls = _record_prompt_dispatch(trainer)

    trainer._add_batch_to_generate()

    assert calls == [
        ("fetch", capacity, {"allow_epoch_wrap": False, "required": False}),
        ("submit", capacity, True),
    ]


def test_prepare_step_required_refill_bypasses_paused_optional_watermark():
    trainer = _trainer_with_dispatch(
        outstanding=6,
        potential=0,
        max_outstanding_groups=8,
    )
    trainer._dispatch_paused = True
    calls = _record_prompt_dispatch(trainer)

    trainer.prepare_step()

    assert calls == [
        ("fetch", 2, {"allow_epoch_wrap": True, "required": True}),
        ("submit", 2, False),
    ]
    assert trainer._dispatch_paused is True


def test_capacity_drop_does_not_trigger_immediate_one_for_one_refill():
    trainer = _trainer_with_dispatch(
        outstanding=5,
        potential=4,
        max_outstanding_groups=4,
        drop_once=1,
    )
    calls = _record_prompt_dispatch(trainer)

    metrics = trainer.prepare_step()

    assert len(trainer.replay_buffer.outstanding) == 4
    assert metrics["rollout/capacity_dropped_groups"] == 1
    assert calls == []


def test_required_epoch_wrap_is_borrowed_from_optional_prefetch():
    trainer = _trainer_with_dispatch()
    trainer.train_dataloader = [
        {"raw_prompt": ["prompt-0", "prompt-1"], "index": torch.tensor([0, 1])},
        {"raw_prompt": ["prompt-2", "prompt-3"], "index": torch.tensor([2, 3])},
    ]
    trainer.train_dataloader_it = None

    for _ in range(2):
        batch = trainer._next_train_batch(2, allow_epoch_wrap=False, required=False)
        _record_fetched_batch(trainer, batch, optional=True)

    # Optional prefetch stops at the epoch boundary instead of wrapping into epoch 1.
    assert trainer._next_train_batch(2, allow_epoch_wrap=False, required=False) is None
    assert trainer._dispatch_ledger.current_epoch == 1
    assert trainer._dispatch_ledger.epoch_exhausted is True

    required = trainer._next_train_batch(2, required=True)
    assert trainer._last_batch_epochs == [1, 1]
    assert trainer._last_batch_borrowed == [True, True]
    _record_fetched_batch(trainer, required, optional=False)

    # Two prompts borrowed by required refill cancel one optional gen-batch fetch.
    optional = trainer._next_train_batch(4, allow_epoch_wrap=False, required=False)
    assert len(optional) == 2
    assert trainer._last_batch_epochs == [1, 1]
    assert trainer._last_batch_borrowed == [False, False]
    _record_fetched_batch(trainer, optional, optional=True)

    epoch = trainer._dispatch_ledger.epochs[1]
    assert epoch["dispatched"] == 4
    assert epoch["prefetched"] == 2
    assert epoch["borrowed"] == 2
    assert epoch["optional_skipped"] == 2


def test_dispatch_ledger_checkpoint_round_trip_restores_borrow_debt():
    ledger = DispatchLedger()
    ledger.current_epoch = 3
    ledger.epoch_exhausted = True
    ledger.record_dispatch(["trained", "borrowed"], [2, 3], [False, True], optional=False)
    ledger.record_dispatch(["prefetched"], [3], [False], optional=True)
    ledger.record_trained({"trained"})

    restored = DispatchLedger()
    restored.load_state_dict(ledger.state_dict())

    assert restored.state_dict() == ledger.state_dict()
    assert restored.consume_borrowed(3) == 2
    assert restored.epochs[3]["optional_skipped"] == 1
    assert ledger.epochs[3]["optional_skipped"] == 0
    restored.reconcile({"borrowed", "prefetched"})
    assert restored.uid_epochs == {"borrowed": 3, "prefetched": 3}
