# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""CPU tests for the Laminar-style trajectory-level async stack (arxiv 2510.12633).

Covers:
1. ReplayBufferAsync staleness measured from per-trajectory weight-version tags
   (``min_global_steps``) instead of the prompt's dispatch step.
2. LaminarRolloutManager's drain -> migrate -> pull -> rejoin state machine.
3. GlobalRequestLoadBalancer draining preference and sticky-cache migration support.
4. PPOTrainerLaminarAsync config validation.
"""

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field

import pytest
import torch
import transfer_queue as tq
from omegaconf import OmegaConf

from verl.trainer.ppo.v1.replay_buffer import ReplayBufferAsync
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.trainer.ppo.v1.trainer_laminar_async import PPOTrainerLaminarAsync
from verl.workers.rollout.llm_server import GlobalRequestLoadBalancer
from verl.workers.rollout.rollout_manager import LaminarRolloutManager

POLL_INTERVAL = 0.05


@pytest.fixture(scope="module")
def tq_init():
    tq.init()
    yield
    tq.close()


@pytest.fixture
def partition_id():
    """A unique partition per test to isolate TransferQueue state across tests."""
    return f"test-{uuid.uuid4().hex}"


def _uid() -> str:
    # uid must not contain "_" because ReplayBuffer derives it via key.split("_")[0].
    return uuid.uuid4().hex


def _make_async_rb(max_off_policy_threshold: int = 4, max_off_policy_strategy: str = "drop") -> ReplayBufferAsync:
    return ReplayBufferAsync(
        trainer_mode="laminar_async",
        trainer_config={},
        max_off_policy_threshold=max_off_policy_threshold,
        max_off_policy_strategy=max_off_policy_strategy,
        sampler_kwargs={},
        poll_interval=POLL_INTERVAL,
    )


@dataclass
class VersionedPromptSpec:
    """A prompt group whose trajectories carry weight-version stamps."""

    uid: str
    status: str = "finished"
    dispatch_step: int = 0
    # per-session generating weight version (min_global_steps == max_global_steps per session)
    versions: list[int] = field(default_factory=lambda: [0])


def _produce_versioned(partition_id: str, specs: list[VersionedPromptSpec]) -> None:
    """Write trajectory groups with version tags, then publish the prompt status."""
    for spec in specs:
        for session_id, version in enumerate(spec.versions):
            tq.kv_put(
                key=f"{spec.uid}_{session_id}_0",
                partition_id=partition_id,
                fields={"input_ids": torch.tensor([1, 2, 3])},
                tag={
                    "is_prompt": False,
                    "seq_len": 3,
                    "global_steps": spec.dispatch_step,
                    "min_global_steps": version,
                    "max_global_steps": version,
                },
            )
        tq.kv_put(
            key=spec.uid,
            partition_id=partition_id,
            tag={"is_prompt": True, "status": spec.status, "global_steps": spec.dispatch_step},
        )


def _produce_unversioned(partition_id: str, uid: str, status: str, dispatch_step: int) -> None:
    """Write a group whose trajectories carry no version stamps (legacy behavior)."""
    tq.kv_put(
        key=f"{uid}_0_0",
        partition_id=partition_id,
        fields={"input_ids": torch.tensor([1, 2, 3])},
        tag={"is_prompt": False, "seq_len": 3, "global_steps": dispatch_step},
    )
    tq.kv_put(
        key=uid,
        partition_id=partition_id,
        tag={"is_prompt": True, "status": status, "global_steps": dispatch_step},
    )


class _Sample(threading.Thread):
    """Run the blocking sample loop in a thread so tests can observe the outcome."""

    def __init__(self, rb, partition_id, batch_size, global_steps):
        super().__init__(daemon=True)
        self.rb, self.partition_id, self.batch_size, self.global_steps = rb, partition_id, batch_size, global_steps
        self.batch, self.metrics, self.error = None, None, None

    def run(self):
        try:
            self.batch, self.metrics = self.rb.sample(
                global_steps=self.global_steps, partition_id=self.partition_id, batch_size=self.batch_size
            )
        except Exception as e:  # surfaced via result_or_raise
            self.error = e

    def result_or_raise(self, timeout: float = 10.0):
        self.join(timeout)
        assert not self.is_alive(), "sample did not finish in time"
        if self.error is not None:
            raise self.error
        return self.batch, self.metrics


# --------------------------------------------------------------------------- #
# 1. Version-based staleness in ReplayBufferAsync
# --------------------------------------------------------------------------- #


def test_staleness_uses_generation_version_over_dispatch_step(tq_init, partition_id):
    """A group generated by a fresh replica stays sampleable even if it was dispatched
    long ago (the Laminar case: dispatch step says nothing about the serving version)."""
    rb = _make_async_rb(max_off_policy_threshold=4)
    # dispatched at step 0, but actually generated by a replica serving v9
    old_dispatch = VersionedPromptSpec(uid=_uid(), dispatch_step=0, versions=[9])
    _produce_versioned(partition_id, [old_dispatch])

    consumer = _Sample(rb, partition_id, batch_size=1, global_steps=10)
    consumer.start()
    batch, _ = consumer.result_or_raise()
    assert {key.split("_")[0] for key in batch.keys} == {old_dispatch.uid}


def test_stale_generation_version_is_dropped(tq_init, partition_id):
    """A group generated by a stale replica is evicted even if recently dispatched."""
    rb = _make_async_rb(max_off_policy_threshold=4)
    stale = VersionedPromptSpec(uid=_uid(), dispatch_step=10, versions=[2])  # span 10-2+1=9 > 4
    fresh = VersionedPromptSpec(uid=_uid(), dispatch_step=10, versions=[9])  # span 2
    _produce_versioned(partition_id, [stale, fresh])

    consumer = _Sample(rb, partition_id, batch_size=1, global_steps=10)
    consumer.start()
    batch, metrics = consumer.result_or_raise()
    assert {key.split("_")[0] for key in batch.keys} == {fresh.uid}
    # partition_id is a random test id (not "train") -> "validation" prefix.
    assert metrics["validation/off_policy/evicted_samples"] == 1
    # staleness metrics are measured from the generation version (span 9), not dispatch (span 1)
    assert metrics["validation/off_policy/evicted_samples_staleness/max"] == 9.0


def test_group_version_is_the_oldest_trajectory_version(tq_init, partition_id):
    """A group whose trajectories span versions is judged by its oldest one: a migrated
    trajectory makes the whole group as stale as its oldest segment."""
    rb = _make_async_rb(max_off_policy_threshold=4)
    mixed = VersionedPromptSpec(uid=_uid(), dispatch_step=9, versions=[3, 9])  # min=3 -> span 8 > 4
    fresh = VersionedPromptSpec(uid=_uid(), dispatch_step=9, versions=[9, 9])
    _produce_versioned(partition_id, [mixed, fresh])

    consumer = _Sample(rb, partition_id, batch_size=1, global_steps=10)
    consumer.start()
    batch, _ = consumer.result_or_raise()
    assert {key.split("_")[0] for key in batch.keys} == {fresh.uid}


def test_unversioned_groups_fall_back_to_dispatch_step(tq_init, partition_id):
    """Trajectories without version stamps keep the legacy dispatch-step staleness."""
    rb = _make_async_rb(max_off_policy_threshold=4)
    stale = _uid()
    _produce_unversioned(partition_id, stale, status="finished", dispatch_step=1)  # span 10 > 4
    fresh = _uid()
    _produce_unversioned(partition_id, fresh, status="finished", dispatch_step=9)  # span 2

    consumer = _Sample(rb, partition_id, batch_size=1, global_steps=10)
    consumer.start()
    batch, metrics = consumer.result_or_raise()
    assert {key.split("_")[0] for key in batch.keys} == {fresh}
    assert metrics["validation/off_policy/evicted_samples"] == 1


# --------------------------------------------------------------------------- #
# 2. LaminarRolloutManager state machine
# --------------------------------------------------------------------------- #


class _RemoteMethod:
    """Mimics ``actor.method.remote(...)`` on a plain Python object."""

    def __init__(self, fn):
        self._fn = fn

    async def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class FakeLoadBalancer:
    def __init__(self, inflight: dict[str, int]):
        self.draining: set[str] = set()
        self.inflight = dict(inflight)
        self.sticky_cleared: list[str] = []
        self.set_draining = _RemoteMethod(self._set_draining)
        self.get_all_inflight_counts = _RemoteMethod(lambda: dict(self.inflight))
        self.clear_sticky_for_server = _RemoteMethod(self._clear_sticky)

    def _set_draining(self, server_ids, draining=True):
        for sid in server_ids:
            (self.draining.add if draining else self.draining.discard)(sid)

    def _clear_sticky(self, server_id):
        self.sticky_cleared.append(server_id)
        return 1


class FakeReplica:
    def __init__(self, address: str):
        self.server_address = address
        self.pulled_versions: list[int] = []
        self.abort_calls = 0
        self.resume_calls = 0
        self.pull_error: Exception | None = None

    async def pull_weights(self, version=None):
        if self.pull_error is not None:
            raise self.pull_error
        self.pulled_versions.append(version)

    async def abort_all_requests(self):
        self.abort_calls += 1
        return {"aborted_count": 2, "request_ids": ["r0", "r1"]}

    async def resume_generation(self):
        self.resume_calls += 1


def _make_manager(inflight: dict[str, int], drain_timeout: float = 600.0):
    replicas = [FakeReplica(sid) for sid in inflight]
    lb = FakeLoadBalancer(inflight)
    manager = LaminarRolloutManager(
        replicas=replicas,
        load_balancer_handle=lb,
        poll_interval_seconds=0.01,
        drain_timeout_seconds=drain_timeout,
    )
    return manager, replicas, lb


def _run(coro):
    return asyncio.run(coro)


def test_idle_replica_pulls_and_rejoins():
    manager, replicas, lb = _make_manager({"s0": 0, "s1": 0})
    _run(manager.on_publish(3))

    assert lb.draining == {"s0", "s1"}
    _run(manager._control_once())
    # pull tasks are scheduled; let them finish
    _run(asyncio.sleep(0))
    _run(asyncio.sleep(0))

    assert [r.pulled_versions for r in replicas] == [[3], [3]]
    assert manager.get_versions() == {"s0": 3, "s1": 3}
    assert lb.draining == set()
    # no abort happened, so no resume needed
    assert all(r.resume_calls == 0 for r in replicas)


def test_busy_replica_waits_for_natural_drain():
    manager, replicas, lb = _make_manager({"s0": 0, "s1": 1})
    _run(manager.on_publish(2))
    _run(manager._control_once())
    _run(asyncio.sleep(0))
    _run(asyncio.sleep(0))

    # s0 idle -> pulled; s1 busy -> still on v0, still draining
    assert replicas[0].pulled_versions == [2]
    assert replicas[1].pulled_versions == []
    assert lb.draining == {"s1"}

    # s1 finishes its work -> next control round pulls it
    lb.inflight["s1"] = 0
    _run(manager._control_once())
    _run(asyncio.sleep(0))
    _run(asyncio.sleep(0))
    assert replicas[1].pulled_versions == [2]
    assert lb.draining == set()


def test_long_tail_drain_triggers_migration_and_resume():
    manager, replicas, lb = _make_manager({"s0": 2}, drain_timeout=30.0)
    _run(manager.on_publish(5))
    # backdate the drain start so the timeout has already elapsed
    manager._draining_since["s0"] = time.monotonic() - 31.0

    _run(manager._control_once())
    _run(asyncio.sleep(0))
    _run(asyncio.sleep(0))

    # migration: sticky cache cleared, in-flight aborted (duplicate-prefill resume elsewhere)
    assert lb.sticky_cleared == ["s0"]
    assert replicas[0].abort_calls == 1
    assert replicas[0].pulled_versions == []

    # aborted requests have drained -> pull, then engine resumed before rejoin
    lb.inflight["s0"] = 0
    _run(manager._control_once())
    _run(asyncio.sleep(0))
    _run(asyncio.sleep(0))
    assert replicas[0].pulled_versions == [5]
    assert replicas[0].resume_calls == 1
    assert lb.draining == set()
    assert manager.get_metrics()["migration_count"] == 1
    assert manager.get_metrics()["migrated_request_count"] == 2


def test_migration_disabled_with_zero_timeout():
    manager, replicas, lb = _make_manager({"s0": 3}, drain_timeout=0.0)
    _run(manager.on_publish(5))
    manager._draining_since["s0"] = time.monotonic() - 10_000

    _run(manager._control_once())
    _run(asyncio.sleep(0))
    assert replicas[0].abort_calls == 0
    assert lb.sticky_cleared == []


def test_pull_failure_keeps_replica_draining_and_retries():
    manager, replicas, lb = _make_manager({"s0": 0})
    replicas[0].pull_error = RuntimeError("store unavailable")
    _run(manager.on_publish(4))
    _run(manager._control_once())
    _run(asyncio.sleep(0))
    _run(asyncio.sleep(0))

    assert manager.get_versions()["s0"] == 0
    assert lb.draining == {"s0"}
    assert manager.get_metrics()["pull_failure_count"] == 1

    replicas[0].pull_error = None
    _run(manager._control_once())
    _run(asyncio.sleep(0))
    _run(asyncio.sleep(0))
    assert manager.get_versions()["s0"] == 4
    assert lb.draining == set()


def test_on_publish_ignores_older_versions():
    manager, _replicas, _lb = _make_manager({"s0": 0})
    _run(manager.on_publish(3))
    _run(manager.on_publish(2))  # stale notification, must be ignored
    assert manager._latest_version == 3
    assert manager.get_metrics()["publish_count"] == 1


def test_wait_all_replicas_times_out():
    manager, _replicas, _lb = _make_manager({"s0": 0})
    with pytest.raises(TimeoutError):
        _run(manager.wait_all_replicas(1, timeout_seconds=0.05))


# --------------------------------------------------------------------------- #
# 3. GlobalRequestLoadBalancer draining + migration support
# --------------------------------------------------------------------------- #


def _make_lb():
    # actor handles are opaque to the balancer; plain strings stand in
    servers = {"s0": "handle0", "s1": "handle1"}
    return GlobalRequestLoadBalancer(servers=servers)


def test_new_requests_prefer_non_draining_servers():
    lb = _make_lb()
    lb.set_draining(["s0"], True)
    for i in range(4):
        sid, _ = lb.acquire_server(f"req-{i}")
        assert sid == "s1"


def test_sticky_sessions_stay_on_draining_server():
    lb = _make_lb()
    sid, _ = lb.acquire_server("sticky-req")
    lb.set_draining([sid], True)
    for _ in range(3):
        again, _ = lb.acquire_server("sticky-req")
        assert again == sid


def test_clear_sticky_for_server_reroutes_to_non_draining():
    lb = _make_lb()
    sid, _ = lb.acquire_server("sticky-req")
    assert sid == "s0"  # first request lands on the first (only least-loaded) server
    lb.set_draining(["s0"], True)

    # sticky hit keeps routing to s0 until the entry is cleared for migration
    assert lb.acquire_server("sticky-req")[0] == "s0"
    cleared = lb.clear_sticky_for_server("s0")
    assert cleared == 1
    assert lb.acquire_server("sticky-req")[0] == "s1"


def test_acquire_falls_back_to_draining_when_all_draining():
    lb = _make_lb()
    lb.set_draining(["s0", "s1"], True)
    sid, _ = lb.acquire_server("req-x")
    assert sid in {"s0", "s1"}


# --------------------------------------------------------------------------- #
# 4. PPOTrainerLaminarAsync config validation
# --------------------------------------------------------------------------- #


def _laminar_config(**overrides):
    config = {
        "trainer": {
            "v1": {
                "trainer_mode": "laminar_async",
                "laminar_async": {
                    "parameter_sync_step": 1,
                    "max_inflight_prompts": None,
                },
            }
        },
        "data": {"train_batch_size": 64},
        "actor_rollout_ref": {
            "rollout": {
                "checkpoint_engine": {"backend": "mooncake_store"},
                "nnodes": 1,
                "n_gpus_per_node": 8,
                "calculate_log_probs": True,
            }
        },
        "algorithm": {},
        "reward": {"reward_model": {"enable": False, "enable_resource_pool": False}},
    }
    for dotted_key, value in overrides.items():
        *path, leaf = dotted_key.split(".")
        node = config
        for key in path:
            node = node.setdefault(key, {})
        node[leaf] = value
    return OmegaConf.create(config)


def _assert_init_raises(config, match: str, monkeypatch):
    # The validation asserts run before the heavy base-class init; patch the base init
    # out anyway so a validation bug surfaces as a test failure, not a ray/tq error.
    monkeypatch.setattr(PPOTrainer, "__init__", lambda self, config: None)
    with pytest.raises(AssertionError, match=match):
        PPOTrainerLaminarAsync(config)


def test_trainer_requires_store_backend(monkeypatch):
    config = _laminar_config(**{"actor_rollout_ref.rollout.checkpoint_engine.backend": "nccl"})
    _assert_init_raises(config, "mooncake_store", monkeypatch)


def test_trainer_requires_parameter_sync_step_one(monkeypatch):
    config = _laminar_config(**{"trainer.v1.laminar_async.parameter_sync_step": 4})
    _assert_init_raises(config, "parameter_sync_step", monkeypatch)


def test_trainer_requires_inflight_cap_above_train_batch(monkeypatch):
    config = _laminar_config(**{"trainer.v1.laminar_async.max_inflight_prompts": 8})
    _assert_init_raises(config, "max_inflight_prompts", monkeypatch)


def test_trainer_bypass_mode_requires_rollout_log_probs(monkeypatch):
    config = _laminar_config(
        **{
            "algorithm.rollout_correction": {"bypass_mode": True},
            "actor_rollout_ref.rollout.calculate_log_probs": False,
        }
    )
    _assert_init_raises(config, "calculate_log_probs", monkeypatch)


def test_trainer_rejects_colocated_reward_model(monkeypatch):
    config = _laminar_config(**{"reward.reward_model": {"enable": True, "enable_resource_pool": False}})
    _assert_init_raises(config, "enable_resource_pool", monkeypatch)


def test_trainer_valid_config_passes(monkeypatch):
    monkeypatch.setattr(PPOTrainer, "__init__", lambda self, config: None)
    PPOTrainerLaminarAsync(_laminar_config())
