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

"""Standalone per-replica drain/pull weight-update coordinator."""

import asyncio
import time
from types import SimpleNamespace
from typing import Any

import ray

from verl.checkpoint_engine import CheckpointEngineManager


@ray.remote
class WeightUpdateCoordinatorActor:
    """Serialize snapshot installation per replica without global aborts."""

    def __init__(
        self,
        checkpoint_engine_config,
        actor_wg,
        replica_workers: dict[str, list[ray.actor.ActorHandle]],
        load_balancer: ray.actor.ActorHandle,
        retry_interval_s: float,
    ) -> None:
        self.replicas = {server_id: SimpleNamespace(workers=workers) for server_id, workers in replica_workers.items()}
        self.manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            actor_wg=actor_wg,
            replicas=list(self.replicas.values()),
        )
        self.load_balancer = load_balancer
        self.retry_interval_s = retry_interval_s
        self._initialized = False
        self._stop = False
        self._wake = asyncio.Event()
        self._task = None
        self._metrics = {
            "drain_wait_seconds": 0.0,
            "drain_wait_count": 0.0,
            "update_duration_seconds": 0.0,
            "update_duration_count": 0.0,
            "update_failures": 0.0,
        }

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="verl-weight-update-coordinator")

    async def stop(self) -> None:
        self._stop = True
        self._wake.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def publish(self, version: int) -> dict[str, Any]:
        """Commit the snapshot before making it visible to replica updates."""
        result = await self.manager.publish_weights(version)
        if not self._initialized:
            await self.load_balancer.initialize_weight_registry.remote(version)
            self._initialized = True
        else:
            await self.load_balancer.set_latest_weight_version.remote(version)
        self._wake.set()
        return result

    async def _run(self) -> None:
        while not self._stop:
            try:
                pending = await self.load_balancer.servers_pending_update.remote()
                for server_id in pending:
                    try:
                        version = await self.load_balancer.mark_server_updating.remote(server_id)
                    except Exception:
                        continue
                    started = time.monotonic()
                    success = True
                    try:
                        await self.manager.pull_replica(self.replicas[server_id], version)
                    except Exception:
                        success = False
                        self._metrics["update_failures"] += 1
                    duration = max(0.0, time.monotonic() - started)
                    versions = await self.load_balancer.get_weight_versions.remote()
                    drain_started = versions.get("drain_started_at", {}).get(server_id)
                    if drain_started is not None:
                        self._metrics["drain_wait_seconds"] += max(0.0, started - drain_started)
                        self._metrics["drain_wait_count"] += 1
                    await self.load_balancer.complete_server_update.remote(server_id, version, success)
                    if success:
                        self._metrics["update_duration_seconds"] += duration
                        self._metrics["update_duration_count"] += 1
            except asyncio.CancelledError:
                raise
            except Exception:
                self._metrics["update_failures"] += 1
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self.retry_interval_s)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()

    def get_metrics(self) -> dict[str, float]:
        return dict(self._metrics)
