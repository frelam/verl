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
"""Laminar rollout manager: per-replica weight-version service with drain-based pulls.

https://arxiv.org/abs/2510.12633

After the trainer *publishes* a new weight version to the store (see
``verl/checkpoint_engine/store_checkpoint_engine.py``), this manager drives every stale
rollout replica through the paper's update semantics:

1. **drain**: the replica is marked *draining* in the global load balancer, so it keeps
   serving its sticky in-flight trajectories but receives no new prompts;
2. **migrate** (dynamic repack, simplified): if a draining replica's in-flight work
   outlives ``drain_timeout_seconds``, its remaining requests are aborted. The abort
   returns the tokens generated so far to the caller, whose retry loop re-submits
   ``prompt + partial tokens`` to a non-draining replica -- i.e. Laminar's trajectory
   migration WITHOUT KV-cache transfer, at the cost of a duplicate prefill. Target
   selection is the load balancer's least-loaded rule rather than the paper's
   best-fit bin packing (see ``acquire_server``), which is the deliberate simple
   variant for this first implementation;
3. **pull**: once the replica's in-flight count reaches zero, it pulls the latest weights
   from the store (never aborting ongoing generation);
4. **rejoin**: the replica's recorded version is bumped and it starts accepting new
   prompts again.

Because weight pulls only happen at idle points, one replica serves exactly one weight
version at any time, so every trajectory is stamped with a single, well-defined
``policy_version`` (the server's ``global_steps`` tag). Trajectories migrated mid-flight
span two versions; their ``min_global_steps``/``max_global_steps`` tags record the span.

The class is defined as a plain (non-remote) class so unit tests can drive it directly;
the trainer instantiates it through the ``LaminarRolloutManagerActor`` Ray wrapper.
"""

import asyncio
import logging
import os
import time

import ray

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class LaminarRolloutManager:
    """Drain→pull→rejoin orchestrator for rollout replicas in ``laminar_async`` mode.

    Args:
        replicas: List of ``RolloutReplica`` objects (hold server + worker handles).
        load_balancer_handle: Ray actor handle of the ``GlobalRequestLoadBalancer``
            shared with the AgentLoop clients.
        poll_interval_seconds: Control-loop interval for drain detection.
        max_concurrent_pulls: Upper bound of replicas pulling weights concurrently
            (bounds store read fan-out).
        drain_timeout_seconds: How long a draining replica may keep in-flight work
            before its requests are aborted and migrated (duplicate-prefill resume on
            another replica). ``0`` disables migration: replicas always drain
            naturally, however long that takes.
    """

    def __init__(
        self,
        replicas: list,
        load_balancer_handle: "ray.actor.ActorHandle",
        poll_interval_seconds: float = 1.0,
        max_concurrent_pulls: int = 4,
        drain_timeout_seconds: float = 600.0,
    ) -> None:
        self._lb = load_balancer_handle
        self._poll_interval = poll_interval_seconds
        self._drain_timeout = drain_timeout_seconds
        self._latest_version = 0
        # server_address -> {"replica": replica, "version": int}
        self._servers: dict[str, dict] = {
            replica.server_address: {"replica": replica, "version": 0} for replica in replicas
        }
        self._pulling: set[str] = set()
        self._pull_sem = asyncio.Semaphore(max_concurrent_pulls)
        # server_address -> time.monotonic() when it was first marked draining for the
        # current version; entry removed on successful rejoin.
        self._draining_since: dict[str, float] = {}
        # Servers with an in-flight migration task right now (re-abort guard).
        self._migrating: set[str] = set()
        # Servers whose engine was paused by a migration abort; they need
        # ``resume_generation`` after the weight pull, before rejoining.
        self._aborted: set[str] = set()
        self._stopped = asyncio.Event()
        self._metrics: dict = {
            "publish_count": 0,
            "pull_count": 0,
            "pull_failure_count": 0,
            "total_pull_duration_s": 0.0,
            "max_pull_duration_s": 0.0,
            "migration_count": 0,
            "migration_failure_count": 0,
            "migrated_request_count": 0,
        }

    # ------------------------------------------------------------------ loop

    async def start(self) -> None:
        """Run the drain-detection control loop until :meth:`stop` is called."""
        logger.info(
            f"[LaminarRolloutManager] started with {len(self._servers)} replicas, poll_interval={self._poll_interval}s"
        )
        while not self._stopped.is_set():
            try:
                await self._control_once()
            except Exception as e:
                logger.exception(f"[LaminarRolloutManager] control loop error: {e}")
            await asyncio.sleep(self._poll_interval)

    async def stop(self) -> None:
        self._stopped.set()

    # ------------------------------------------------------------------ API

    async def on_publish(self, version: int) -> None:
        """Trainer hook after weights of ``version`` are committed to the store.

        Marks every replica older than ``version`` as draining; the control loop pulls
        weights for them as they individually drain.
        """
        if version <= self._latest_version:
            return
        self._latest_version = version
        self._metrics["publish_count"] += 1
        stale = [sid for sid, s in self._servers.items() if s["version"] < version]
        if stale:
            now = time.monotonic()
            for sid in stale:
                # Keep the original drain-start time across versions: a replica that is
                # still draining vK when vK+1 publishes has one continuous drain budget.
                self._draining_since.setdefault(sid, now)
            await self._lb.set_draining.remote(stale, True)

    async def wait_all_replicas(self, version: int, timeout_seconds: float = 1800.0) -> None:
        """Block until every replica reached at least ``version`` (startup barrier)."""
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if all(s["version"] >= version for s in self._servers.values()):
                return
            await asyncio.sleep(1.0)
        versions = {sid: s["version"] for sid, s in self._servers.items()}
        raise TimeoutError(f"replicas failed to reach v{version} within {timeout_seconds}s: {versions}")

    def get_versions(self) -> dict[str, int]:
        """Current weight version per server address."""
        return {sid: s["version"] for sid, s in self._servers.items()}

    def get_metrics(self) -> dict:
        versions = list(self.get_versions().values())
        metrics = dict(self._metrics)
        metrics.update(
            {
                "latest_version": self._latest_version,
                "num_replicas": len(versions),
                "num_stale_replicas": sum(1 for v in versions if v < self._latest_version),
                "num_pulling_replicas": len(self._pulling),
                "replica_version_min": min(versions) if versions else 0,
                "replica_version_max": max(versions) if versions else 0,
            }
        )
        return metrics

    # ------------------------------------------------------------------ internals

    async def _control_once(self) -> None:
        if not self._servers:
            return
        inflight = await self._lb.get_all_inflight_counts.remote()
        now = time.monotonic()
        for sid, state in self._servers.items():
            if state["version"] >= self._latest_version or sid in self._pulling:
                continue
            if inflight.get(sid, 0) > 0:
                # Long-tail drain: migrate the remaining in-flight trajectories off this
                # replica so it can reach an idle point and pull (dynamic repack).
                if (
                    self._drain_timeout > 0
                    and sid in self._draining_since
                    and now - self._draining_since[sid] > self._drain_timeout
                    and sid not in self._migrating
                ):
                    self._migrating.add(sid)
                    asyncio.create_task(self._migrate_inflight(sid))
                continue
            self._pulling.add(sid)
            asyncio.create_task(self._pull_and_rejoin(sid))

    async def _migrate_inflight(self, server_id: str) -> None:
        """Migrate in-flight trajectories off a long-tail draining replica.

        Simplified dynamic repack without KV-cache transfer: the sticky cache entries
        pointing at this server are dropped first (so aborted requests do not route
        back), then all in-flight requests are aborted. Each caller receives its tokens
        generated so far with ``stop_reason="aborted"`` and resumes on a non-draining
        replica with ``prompt + partial tokens`` as the new prompt (duplicate prefill).
        The abort pauses the engine, which also closes the race of new requests
        sneaking in before the weight pull; ``_pull_and_rejoin`` reopens it.
        """
        try:
            await self._lb.clear_sticky_for_server.remote(server_id)
            result = await self._servers[server_id]["replica"].abort_all_requests()
            aborted = result.get("aborted_count", 0) if isinstance(result, dict) else 0
            self._aborted.add(server_id)
            self._metrics["migration_count"] += 1
            self._metrics["migrated_request_count"] += aborted
            logger.info(
                f"[LaminarRolloutManager] {server_id} drain exceeded {self._drain_timeout}s, "
                f"migrated {aborted} in-flight requests"
            )
        except Exception as e:
            self._metrics["migration_failure_count"] += 1
            logger.exception(f"[LaminarRolloutManager] {server_id} migration failed: {e}")
        finally:
            self._migrating.discard(server_id)

    async def _pull_and_rejoin(self, server_id: str) -> None:
        async with self._pull_sem:
            start = time.time()
            try:
                await self._servers[server_id]["replica"].pull_weights(version=self._latest_version)
                if server_id in self._aborted:
                    # The migration abort paused the engine; reopen it before rejoining.
                    await self._servers[server_id]["replica"].resume_generation()
                    self._aborted.discard(server_id)
                self._servers[server_id]["version"] = self._latest_version
                self._draining_since.pop(server_id, None)
                await self._lb.set_draining.remote([server_id], False)
                duration = time.time() - start
                self._metrics["pull_count"] += 1
                self._metrics["total_pull_duration_s"] += duration
                self._metrics["max_pull_duration_s"] = max(self._metrics["max_pull_duration_s"], duration)
                logger.info(f"[LaminarRolloutManager] {server_id} pulled v{self._latest_version} in {duration:.2f}s")
            except Exception as e:
                # Keep the server draining and retry on the next control round.
                self._metrics["pull_failure_count"] += 1
                logger.exception(f"[LaminarRolloutManager] {server_id} pull failed: {e}")
            finally:
                self._pulling.discard(server_id)


# Ray actor wrapper used by the trainer; tests drive the plain class above directly.
LaminarRolloutManagerActor = ray.remote(num_cpus=2, max_concurrency=8)(LaminarRolloutManager)
