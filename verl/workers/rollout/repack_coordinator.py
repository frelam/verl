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

"""Persistent Ray coordinator for Laminar repack dry-runs and migration."""

import asyncio
import logging
from dataclasses import replace
from typing import Any

import ray

from verl.workers.rollout.repack import RepackCoordinatorCore, RepackPlan, select_idle_replicas
from verl.workers.rollout.router import MigrationConflictError, ServerBusyError

logger = logging.getLogger(__name__)


class RepackPlanExecutor:
    """Execute same-version plans through atomic LB reservation and directed abort."""

    def __init__(
        self,
        servers: dict[str, ray.actor.ActorHandle],
        load_balancer: ray.actor.ActorHandle,
        *,
        timeout_s: float,
        fallback_to_lb: bool,
    ) -> None:
        self.servers = servers
        self.load_balancer = load_balancer
        self.timeout_s = timeout_s
        self.fallback_to_lb = fallback_to_lb
        self._cleanup_tasks: set[asyncio.Task] = set()

    async def __call__(self, plan: RepackPlan) -> bool:
        success = True
        for move in plan.moves:
            version = await self.load_balancer.begin_repack_drain.remote(move.source)
            try:
                records = await self.load_balancer.list_inflight.remote(move.source)
                for record in records:
                    request_id = record["request_id"]
                    engine_request_id = record.get("engine_request_id")
                    if not engine_request_id:
                        raise MigrationConflictError(f"Request {request_id!r} has no engine identity")
                    prepared = False
                    try:
                        await self.load_balancer.prepare_migration.remote(
                            request_id, move.destination, expected_attempt=record["attempt"]
                        )
                        prepared = True
                    except (ServerBusyError, MigrationConflictError):
                        if not self.fallback_to_lb:
                            success = False
                            continue
                    try:
                        abort = await asyncio.wait_for(
                            self.servers[move.source].abort_request.remote(
                                engine_request_id,
                                reset_prefix_cache=False,
                                reason="migration",
                            ),
                            timeout=self.timeout_s,
                        )
                    except Exception:
                        if prepared:
                            await self.load_balancer.cancel_migration.remote(request_id)
                        raise
                    if not abort.get("aborted", False):
                        if prepared:
                            # The source completed before abort: leave its original routing intact.
                            await self.load_balancer.cancel_migration.remote(request_id, request_completed=True)
                        continue
                    if prepared:
                        self._schedule_reservation_cleanup(request_id)
                    else:
                        await self.load_balancer.record_migration_result.remote(
                            success=True,
                            reprefill_tokens=0,
                            duration_seconds=0.0,
                            fallback_to_lb=True,
                        )
            finally:
                await self.load_balancer.finish_repack_drain.remote(move.source, version)
        return success

    def _schedule_reservation_cleanup(self, request_id: str) -> None:
        async def cleanup() -> None:
            try:
                await asyncio.sleep(self.timeout_s)
                await self.load_balancer.cancel_migration.remote(request_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Migration reservation cleanup failed for %s: %s", request_id, exc)

        task = asyncio.create_task(cleanup(), name=f"repack-reservation-{request_id}")
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)

    async def close(self) -> None:
        tasks = list(self._cleanup_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._cleanup_tasks.clear()


@ray.remote
class RepackCoordinatorActor:
    """Own a persistent event loop for start/stop, single-flight ticks and cleanup."""

    def __init__(
        self,
        servers: dict[str, ray.actor.ActorHandle],
        load_balancer: ray.actor.ActorHandle,
        repack_config: dict[str, Any],
        request_budget: int,
        stats_interval_s: float,
    ) -> None:
        self.servers = servers
        self.load_balancer = load_balancer
        self._weight_syncing = False
        migration_config = repack_config.get("migration", {})
        self._executor = RepackPlanExecutor(
            servers,
            load_balancer,
            timeout_s=migration_config.get("timeout_s", 30),
            fallback_to_lb=migration_config.get("fallback_to_lb", False),
        )
        self._core = RepackCoordinatorCore(
            self._collect_stats,
            interval_s=repack_config.get("interval_s", 5),
            kv_cache_threshold=repack_config.get("kv_cache_threshold", 0.99),
            roofline_batch_size=repack_config.get("roofline_batch_size") or request_budget,
            metrics_max_age_s=max(2 * stats_interval_s, 2 * repack_config.get("interval_s", 5)),
            enabled=repack_config.get("enable", False),
            migration_enabled=migration_config.get("enable", False),
            execute_plan=self._executor,
            weight_syncing=lambda: self._weight_syncing,
            collection_timeout_s=stats_interval_s,
        )

    async def _collect_stats(self):
        versions = await self.load_balancer.get_weight_versions.remote()
        server_ids = list(self.servers)
        results = await asyncio.gather(
            *[self.servers[server_id].get_metrics_snapshot.remote() for server_id in server_ids],
            return_exceptions=True,
        )
        snapshots = {}
        for server_id, result in zip(server_ids, results, strict=True):
            if isinstance(result, Exception) or result is None:
                snapshots[server_id] = None
                continue
            snapshots[server_id] = replace(
                result,
                weight_version=versions["local"].get(server_id),
                state=versions["state"].get(server_id, "unknown"),
            )
        return snapshots

    async def start(self) -> None:
        await self._core.start()

    async def stop(self) -> None:
        await self._core.stop()
        await self._executor.close()

    async def tick(self):
        return await self._core.tick()

    def set_weight_syncing(self, active: bool) -> None:
        self._weight_syncing = active

    async def get_metrics(self) -> dict[str, Any]:
        idle = select_idle_replicas(
            self._core.history,
            self._core.kv_cache_threshold,
            self._core.roofline_batch_size,
            max_age_s=self._core.metrics_max_age_s,
        )
        return {
            "coordinator": self._core.get_metrics().as_dict(),
            "load_balancer": await self.load_balancer.get_metrics.remote(),
            "idle_replicas": sorted(idle),
            "snapshots": {
                server_id: self._core.history.latest(server_id) for server_id in self._core.history.server_ids
            },
        }
