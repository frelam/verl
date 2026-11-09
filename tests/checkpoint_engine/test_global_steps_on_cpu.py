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

import asyncio
from types import SimpleNamespace

import pytest

from verl.checkpoint_engine import base as checkpoint_engine_base
from verl.checkpoint_engine.base import CheckpointEngineManager, CheckpointEngineWorker
from verl.workers.engine_workers import ActorRolloutRefWorker


class _FakeTrainerEngine:
    def __init__(self):
        self.weights = [("w", object())]

    def get_per_tensor_param(self):
        return iter(self.weights), None


class _FakeCheckpointEngine:
    def __init__(self):
        self.sent_global_steps = None
        self.received_global_steps = None
        self.sent_weights = None

    async def send_weights(self, weights, global_steps=None):
        self.sent_global_steps = global_steps
        self.sent_weights = list(weights)

    def receive_weights(self, global_steps=None):
        self.received_global_steps = global_steps

        async def _weights():
            yield "w", object()

        return _weights()


class _FakeServerAdapter:
    def __init__(self):
        self.global_steps = None
        self.weights = None

    async def update_weights(self, weights, global_steps=None, **kwargs):
        self.global_steps = global_steps
        self.weights = [item async for item in weights]


def test_actor_worker_passes_global_steps_to_checkpoint_engine_send():
    checkpoint_engine = _FakeCheckpointEngine()
    worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
    worker.config = SimpleNamespace(
        rollout=SimpleNamespace(
            checkpoint_engine=SimpleNamespace(backend="modelexpress"),
        ),
    )
    worker.actor = SimpleNamespace(engine=_FakeTrainerEngine())
    worker.checkpoint_engine = checkpoint_engine

    asyncio.run(
        ActorRolloutRefWorker.update_weights.__wrapped__(
            worker,
            global_steps=17,
            mode="auto",
        )
    )

    assert checkpoint_engine.sent_global_steps == 17
    assert checkpoint_engine.sent_weights == worker.actor.engine.weights


def test_checkpoint_worker_passes_global_steps_to_receive_and_rollout_update():
    checkpoint_engine = _FakeCheckpointEngine()
    server_adapter = _FakeServerAdapter()
    worker = CheckpointEngineWorker.__new__(CheckpointEngineWorker)
    worker.checkpoint_engine = checkpoint_engine
    worker.server_adapter = server_adapter

    asyncio.run(
        CheckpointEngineWorker.update_weights.__wrapped__(
            worker,
            global_steps=17,
        )
    )

    assert checkpoint_engine.received_global_steps == 17
    assert server_adapter.global_steps == 17
    assert [name for name, _tensor in server_adapter.weights] == ["w"]


class _FakeActorWorkerGroup:
    def __init__(self, world_size=2):
        self.world_size = world_size
        self.update_calls = []
        self.execute_calls = []
        self.publish_results = {}
        self.snapshots = {}

    def update_weights(self, global_steps=None, mode="auto"):
        self.update_calls.append({"global_steps": global_steps, "mode": mode})
        if mode == "snapshot":
            return self.publish_results.get(
                global_steps,
                [{"latest_weight_version": global_steps} for _ in range(self.world_size)],
            )
        if mode == "snapshot_send":
            return [{"sent_global_steps": global_steps} for _ in range(self.world_size)]
        raise AssertionError(f"unexpected actor update mode: {mode}")

    def execute_checkpoint_engine(self, *args, **kwargs):
        self.execute_calls.append((args, kwargs))
        method = kwargs.get("method", args[0] if args else None)
        if method and method[0] == "get_snapshot_weights":
            version = kwargs["global_steps"][0]
            return self.snapshots[version]
        assert method == ["finalize"] * self.world_size
        return [{"finalized": True} for _ in range(self.world_size)]


class _FakeRolloutWorkerGroup:
    instances = []

    def __init__(self, worker_handles, ray_cls_with_init):
        self.workers = list(worker_handles)
        self.world_size = len(self.workers)
        self.ray_cls_with_init = ray_cls_with_init
        self.update_calls = []
        self.snapshot_calls = []
        self.execute_calls = []
        self.__class__.instances.append(self)

    def update_weights(self, global_steps=None):
        self.update_calls.append(global_steps)
        return [{"received_global_steps": global_steps} for _ in range(self.world_size)]

    def update_weights_from_snapshot(self, weights, global_steps):
        self.snapshot_calls.append((weights, global_steps))
        return [{"installed_global_steps": global_steps} for _ in range(self.world_size)]

    def execute_checkpoint_engine(self, *args, **kwargs):
        self.execute_calls.append((args, kwargs))
        assert args[0] == ["finalize"] * self.world_size
        return [{"finalized": True} for _ in range(self.world_size)]


def _checkpoint_manager(monkeypatch, backend, actor_wg, replicas=()):
    monkeypatch.setattr(
        checkpoint_engine_base.CheckpointEngineRegistry,
        "get",
        staticmethod(lambda _backend: object),
    )
    monkeypatch.setattr(checkpoint_engine_base.ray, "get", lambda references: references)
    config = SimpleNamespace(backend=backend, custom_backend_module=None)
    return CheckpointEngineManager(config=config, actor_wg=actor_wg, replicas=list(replicas))


def test_manager_publishes_each_version_once_and_failed_version_is_invisible(monkeypatch):
    actor_wg = _FakeActorWorkerGroup()
    actor_wg.publish_results[8] = [{"latest_weight_version": 7}, {}]
    manager = _checkpoint_manager(monkeypatch, "kimi_ckpt_engine", actor_wg)

    assert manager.publish_weights(7) == {"latest_weight_version": 7, "already_published": False}
    assert manager.publish_weights(7) == {"latest_weight_version": 7, "already_published": True}
    assert [call["global_steps"] for call in actor_wg.update_calls] == [7]

    with pytest.raises(RuntimeError, match="publication barrier"):
        manager.publish_weights(8)
    assert 8 not in manager._published_versions
    with pytest.raises(KeyError, match="version 8 has not been published"):
        manager.pull_replica(SimpleNamespace(workers=[]), 8)

    actor_wg.publish_results[8] = [{"latest_weight_version": 8}, {}]
    assert manager.publish_weights(8) == {"latest_weight_version": 8, "already_published": False}
    assert [call["global_steps"] for call in actor_wg.update_calls] == [7, 8, 8]


def test_manager_kimi_pull_uses_direct_cpu_snapshot_without_collective(monkeypatch):
    actor_wg = _FakeActorWorkerGroup()
    weight = object()
    actor_wg.snapshots[3] = [{}, {"weight": weight}]
    manager = _checkpoint_manager(monkeypatch, "kimi_ckpt_engine", actor_wg)
    _FakeRolloutWorkerGroup.instances = []
    monkeypatch.setattr(checkpoint_engine_base, "RayWorkerGroup", _FakeRolloutWorkerGroup)
    monkeypatch.setattr(
        manager,
        "build_process_group",
        lambda _rollout: pytest.fail("KIMI single-replica CPU pull must not build a collective"),
    )
    replica = SimpleNamespace(workers=["target-worker"])

    manager.publish_weights(3)
    result = manager.pull_replica(replica, 3)

    assert result["version"] == 3
    assert len(_FakeRolloutWorkerGroup.instances) == 1
    rollout = _FakeRolloutWorkerGroup.instances[0]
    assert rollout.workers == ["target-worker"]
    assert rollout.snapshot_calls == [({"weight": weight}, 3)]
    assert rollout.update_calls == []
    assert rollout.execute_calls == []
    assert [call["mode"] for call in actor_wg.update_calls] == ["snapshot"]
    assert len(actor_wg.execute_calls) == 1
    args, kwargs = actor_wg.execute_calls[0]
    assert args == ()
    assert kwargs == {"method": ["get_snapshot_weights"] * 2, "global_steps": [3, 3]}


def test_manager_nixl_pull_builds_transfer_with_only_target_replica(monkeypatch):
    actor_wg = _FakeActorWorkerGroup()
    target = SimpleNamespace(workers=["target-0", "target-1"])
    other = SimpleNamespace(workers=["other-0"])
    manager = _checkpoint_manager(monkeypatch, "nixl", actor_wg, replicas=[target, other])
    _FakeRolloutWorkerGroup.instances = []
    monkeypatch.setattr(checkpoint_engine_base, "RayWorkerGroup", _FakeRolloutWorkerGroup)
    build_calls = []
    monkeypatch.setattr(manager, "build_process_group", lambda rollout: build_calls.append(rollout))

    manager.publish_weights(5)
    result = manager.pull_replica(target, 5)

    assert result["version"] == 5
    assert len(build_calls) == 1
    rollout = build_calls[0]
    assert rollout.workers == ["target-0", "target-1"]
    assert "other-0" not in rollout.workers
    assert rollout.update_calls == [5]
    assert rollout.snapshot_calls == []
    assert [call["mode"] for call in actor_wg.update_calls] == ["snapshot", "snapshot_send"]
    assert actor_wg.execute_calls[0][0] == (["finalize"] * actor_wg.world_size,)
    assert rollout.execute_calls[0][0] == (["finalize"] * rollout.world_size,)
