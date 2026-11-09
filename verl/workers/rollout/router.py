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

import logging
import os
import random
import time
from typing import Any, Protocol
from uuid import uuid4

import ray
from cachetools import LRUCache
from omegaconf import OmegaConf

from verl.utils.import_utils import load_class_from_fqn, resolve_config_path

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DEFAULT_ROUTING_CACHE_SIZE = 10000


class ServerBusyError(RuntimeError):
    """Raised when every replica eligible for a request is at admission capacity."""


class TrajectoryLeaseError(RuntimeError):
    """Raised for invalid, conflicting, or expired trajectory lease operations."""


class MigrationConflictError(RuntimeError):
    """Raised when an in-flight request changed before a migration CAS could run."""


class RequestLoadBalancer(Protocol):
    """Protocol for rollout inference load balancers.

    All strategies must satisfy this interface via structural subtyping.
    """

    def acquire_server(self, request_id: str, **extra) -> tuple[str, Any]:
        """Acquire a server for the given request.

        Args:
            request_id: Request identifier for sticky session routing.
            **extra: Keyword fields declared via :meth:`require_acquire_fields`
                (e.g. ``prompt_ids`` for content-aware routing). Only the
                declared fields are serialized into this RPC.

        Returns:
            A ``(server_id, actor_handle)`` tuple.

        Raises:
            RuntimeError: If no servers are available in the pool.
        """
        ...

    def require_acquire_fields(self) -> list[str]:
        """``generate()`` kwargs this router consumes at acquire time; only
        these are serialized into the ``acquire_server`` RPC (e.g.
        ``["prompt_ids"]``; ``[]`` if routing on ``request_id`` alone)."""
        ...

    def require_release_fields(self) -> list[str]:
        """Identity fields this router consumes at release time (currently
        only ``"request_id"``); ``[]`` if counting by ``server_id`` alone."""
        ...

    def release_server(self, server_id: str, request_id: str | None = None) -> None:
        """Release a server after a request completes.

        Args:
            server_id: Identifier of the server to release.
            request_id: Request identifier. Content-aware balancers that need
                the prompt length at release time look it up by this id from
                their own acquire-time bookkeeping, so the full token list is
                not re-serialized over RPC.
        """
        ...

    def add_servers(self, servers: dict[str, Any]) -> None:
        """Bulk-add servers to the load balancer pool.

        Args:
            servers: Mapping from ``server_id`` to ``actor_handle``.
        """
        ...

    def remove_servers(self, server_ids: list[str]) -> None:
        """Bulk-remove servers from the load balancer pool.

        Args:
            server_ids: List of server identifiers to remove.
        """
        ...

    def get_all_servers(self) -> list[str]:
        """List all active server IDs.

        Returns:
            List of server identifier strings.
        """
        ...

    def get_status(self) -> dict:
        """Return current load balancer state for debugging.

        Returns:
            A dictionary with ``servers``, ``total_inflight``,
            and ``active_servers`` keys.
        """
        ...

    def clear_sticky_cache(self) -> dict:
        """Clear sticky-session state to force request redistribution.

        Called by trainers on training/rollout phase switches and by
        fully-async rebalancing, so all strategies (plugins included)
        must implement it.

        Returns:
            A diagnostics dict, conventionally with ``cleared_entries``
            and ``server_loads`` keys.
        """
        ...

    def get_total_inflight(self) -> int:
        """Return the total in-flight requests across registered servers.

        Polled by fully-async drain/rebalance loops to wait for the
        pool to quiesce.
        """
        ...


class GlobalRequestLoadBalancer:
    """Global sticky-session + in-flight load balancer shared by all AgentLoopWorkers.

    When a sticky session points to a removed server, the cache entry is
    automatically invalidated and a new server is selected.

    This is a plain Python class (not a Ray actor). It is wrapped with
    ``ray.remote(...)`` at instantiation time so callers can subclass it and
    override :meth:`acquire_server` before registering the subclass as an actor.

    Key features:
    - **Atomic acquire**: ``acquire_server()`` returns ``(server_id, handle)``
    - **Sticky Session**: Uses LRUCache to map request_id → server_id, ensuring
      multi-turn conversations route to the same server.
    - **Least-loaded Selection**: When no sticky session exists, selects the
      server with the fewest in-flight requests.
    - **Deterministic Routing**: When ``full_determinism=True``, routes every
      request by ``hash(request_id) % len(servers)`` over the full pool so the
      same request always routes to the same replica across runs.
    - **Dynamic Server Management**: Supports add/remove servers at runtime
      for hybrid scaling.
    """

    def __init__(
        self,
        servers: dict[str, ray.actor.ActorHandle],
        max_cache_size: int = DEFAULT_ROUTING_CACHE_SIZE,
        full_determinism: bool = False,
        max_inflight_requests_per_replica: int | None = None,
        laminar_routing_enabled: bool = False,
    ):
        # Allow empty initial servers: in dynamic-resource-scheduling mode all
        # replicas are hybrid and will be registered later via add_servers().
        if max_inflight_requests_per_replica is not None:
            if not isinstance(max_inflight_requests_per_replica, int) or max_inflight_requests_per_replica <= 0:
                raise ValueError("max_inflight_requests_per_replica must be null or a positive integer")
            if full_determinism:
                raise ValueError("V1 full_determinism does not support per-replica admission control")

        self._servers: dict[str, ray.actor.ActorHandle] = dict(servers)
        self._inflight_requests: dict[str, int] = {sid: 0 for sid in servers}
        self._request_id_to_server: LRUCache = LRUCache(maxsize=max_cache_size)
        self._full_determinism = full_determinism
        self._max_inflight = max_inflight_requests_per_replica
        self._laminar_routing_enabled = laminar_routing_enabled
        self._server_versions: dict[str, int] = {sid: 0 for sid in servers}
        self._server_states: dict[str, str] = {sid: "serving" for sid in servers}
        self._latest_weight_version = 0
        self._drain_started_at: dict[str, float] = {}
        self._trajectory_leases: dict[str, dict[str, Any]] = {}
        self._lease_tokens: dict[str, str] = {}
        self._request_registry: dict[str, dict[str, Any]] = {}
        self._migration_reservations: dict[str, dict[str, Any]] = {}
        self._migration_resumes: dict[str, dict[str, Any]] = {}
        self._metrics: dict[str, float] = {
            "admission_wait_seconds": 0.0,
            "admission_wait_count": 0.0,
            "migration_requests_total": 0.0,
            "migration_success": 0.0,
            "migration_failed": 0.0,
            "migration_reprefill_tokens": 0.0,
            "migration_duration_seconds": 0.0,
            "migration_fallback_to_lb": 0.0,
        }

    def _has_capacity(self, server_id: str) -> bool:
        return self._max_inflight is None or self._inflight_requests[server_id] < self._max_inflight

    def _select_least_loaded(self, candidates: list[str]) -> str:
        min_count = min(self._inflight_requests[sid] for sid in candidates)
        least_loaded = [sid for sid in candidates if self._inflight_requests[sid] == min_count]
        return random.choice(least_loaded)

    def acquire_server(
        self,
        request_id: str,
        target_server_id: str | None = None,
        trajectory_id: str | None = None,
        lease_token: str | None = None,
        engine_request_id: str | None = None,
        attempt: int | None = None,
    ) -> tuple[str, ray.actor.ActorHandle]:
        """Atomically acquire admission capacity and register request identity.

        An unleased sticky request whose replica is full is reselected within the
        same weight-version serving set. A leased trajectory never changes its
        replica through ordinary load balancing.
        """
        if not self._inflight_requests:
            raise RuntimeError("No available servers in load balancer")

        lease = None
        if lease_token is not None:
            lease = self._trajectory_leases.get(self._lease_tokens.get(lease_token, ""))
            if lease is None or lease["token"] != lease_token:
                raise TrajectoryLeaseError("Unknown trajectory lease token")
            if trajectory_id is not None and lease["trajectory_id"] != trajectory_id:
                raise TrajectoryLeaseError("Trajectory id does not match its lease")
            target_server_id = lease["server_id"]

        reservation = self._migration_reservations.get(request_id)
        if reservation is not None:
            target_server_id = reservation["destination"]

        if target_server_id is not None:
            if target_server_id not in self._inflight_requests:
                raise RuntimeError(f"Target server {target_server_id!r} is not registered")
            server_id = target_server_id
            if lease is None and self._server_states[server_id] != "serving" and reservation is None:
                raise ServerBusyError(f"Target server {server_id!r} is {self._server_states[server_id]}")
            if reservation is None and not self._has_capacity(server_id):
                raise ServerBusyError(f"Target server {server_id!r} is full")
        else:
            sticky_id = self._request_id_to_server.get(request_id)
            if sticky_id in self._inflight_requests and self._server_states[sticky_id] == "serving":
                if self._has_capacity(sticky_id):
                    server_id = sticky_id
                else:
                    candidates = [
                        sid
                        for sid in self._inflight_requests
                        if self._server_states[sid] == "serving"
                        and self._server_versions[sid] == self._server_versions[sticky_id]
                        and self._has_capacity(sid)
                    ]
                    if not candidates:
                        raise ServerBusyError("All replicas in the sticky request's version set are full")
                    server_id = self._select_least_loaded(candidates)
            else:
                if sticky_id is not None:
                    self._request_id_to_server.pop(request_id, None)
                sticky_version = self._server_versions.get(sticky_id)
                candidates = [
                    sid
                    for sid in self._inflight_requests
                    if self._server_states[sid] == "serving"
                    and self._has_capacity(sid)
                    and (sticky_version is None or self._server_versions[sid] == sticky_version)
                ]
                if not candidates:
                    if any(state == "serving" for state in self._server_states.values()):
                        raise ServerBusyError("All serving replicas are full")
                    raise ServerBusyError("No serving replicas are currently available")
                if self._full_determinism:
                    server_id = list(self._servers)[hash(request_id) % len(self._servers)]
                else:
                    server_id = self._select_least_loaded(candidates)

        previous = self._request_registry.get(request_id, {})
        current_attempt = previous.get("attempt", 0) if attempt is None else attempt
        if reservation is not None:
            current_attempt = reservation["attempt"]
        else:
            self._inflight_requests[server_id] += 1
        self._request_id_to_server[request_id] = server_id
        self._request_registry[request_id] = {
            "request_id": request_id,
            "engine_request_id": engine_request_id,
            "server_id": server_id,
            "attempt": current_attempt,
            "weight_version": self._server_versions[server_id],
            "trajectory_id": lease["trajectory_id"] if lease else trajectory_id,
            "lease_token": lease_token,
            "acquired_at": time.monotonic(),
        }
        if reservation is not None:
            self._migration_reservations.pop(request_id, None)
            self._migration_resumes[request_id] = {
                "attempt": reservation["attempt"],
                "started_at": reservation["started_at"],
                "source": reservation["source"],
                "destination": reservation["destination"],
            }
        return server_id, self._servers[server_id]

    def release_server(self, server_id: str, request_id: str | None = None) -> None:
        """Release one admission slot; lease lifetime is independent of turns."""
        if server_id not in self._inflight_requests:
            return
        if self._inflight_requests[server_id] > 0:
            self._inflight_requests[server_id] -= 1
        if request_id is not None:
            record = self._request_registry.get(request_id)
            if record is not None and record["server_id"] == server_id:
                self._request_registry.pop(request_id, None)

    def require_acquire_fields(self) -> list[str]:
        """Only Laminar routing metadata is serialized; sampling payloads stay local."""
        if not self._laminar_routing_enabled and self._max_inflight is None:
            return []
        return ["target_server_id", "trajectory_id", "lease_token", "engine_request_id", "attempt"]

    def require_release_fields(self) -> list[str]:
        if not self._laminar_routing_enabled and self._max_inflight is None:
            return []
        return ["request_id"]

    def add_servers(self, servers: dict[str, ray.actor.ActorHandle]) -> None:
        """Atomically add multiple servers to the load balancer pool.

        This is more efficient than calling :meth:`add_server` in a loop
        because it performs a single bulk update on the internal state.

        Args:
            servers: Dict mapping server_id → actor_handle for all servers
                to register.
        """
        for sid, handle in servers.items():
            self._inflight_requests[sid] = 0
            self._servers[sid] = handle
            self._server_versions.setdefault(sid, self._latest_weight_version)
            self._server_states[sid] = "serving"
        logger.info(f"[GlobalLoadBalancer] added {len(servers)} servers")

    def remove_servers(self, server_ids: list[str]) -> None:
        """Atomically remove multiple servers from the load balancer pool.

        More efficient than calling :meth:`remove_server` in a loop.

        Args:
            server_ids: List of server identifiers to remove.
        """
        for sid in server_ids:
            if any(lease["server_id"] == sid for lease in self._trajectory_leases.values()):
                raise TrajectoryLeaseError(f"Cannot remove server {sid!r} with active trajectory leases")
            self._inflight_requests.pop(sid, None)
            self._servers.pop(sid, None)
            self._server_versions.pop(sid, None)
            self._server_states.pop(sid, None)
        logger.info(f"[GlobalLoadBalancer] removed {len(server_ids)} servers")

    def get_inflight_count(self, server_id: str) -> int:
        """Get number of in-flight requests for a server."""
        return self._inflight_requests.get(server_id, 0)

    def get_all_servers(self) -> list[str]:
        """Get list of all active server IDs."""
        return list(self._inflight_requests.keys())

    def clear_sticky_cache(self) -> dict:
        """Clear the sticky-session cache to force request redistribution.

        After clearing, all subsequent ``acquire_server()`` calls will select
        the least-loaded server (based on ``_inflight_requests``), which
        naturally balances load across all active replicas — including newly
        added ones with zero in-flight requests.

        Returns:
            A dict with ``cleared_entries`` (number of cache entries dropped)
            and ``server_loads`` (current per-server inflight counts for
            diagnostics).
        """
        cleared = len(self._request_id_to_server)
        self._request_id_to_server.clear()
        logger.info(
            f"[GlobalLoadBalancer] Sticky cache cleared: {cleared} entries dropped. "
            f"Server loads: {dict(self._inflight_requests)}"
        )
        return {
            "cleared_entries": cleared,
            "server_loads": dict(self._inflight_requests),
        }

    def get_status(self) -> dict:
        """Return current load balancer state for debugging."""
        return {
            "servers": dict(self._inflight_requests),
            "total_inflight": sum(self._inflight_requests.values()),
            "active_servers": len(self._inflight_requests),
            "registered_handles": list(self._servers.keys()),
            "local_versions": dict(self._server_versions),
            "server_states": dict(self._server_states),
            "latest_weight_version": self._latest_weight_version,
        }

    def get_total_inflight(self) -> int:
        """Return the sum of in-flight requests across all currently registered servers."""
        return sum(self._inflight_requests.values())

    def initialize_weight_registry(self, version: int) -> None:
        """Start per-replica pull with no snapshot installed on any replica."""
        self._latest_weight_version = version
        now = time.monotonic()
        for server_id in self._servers:
            self._server_versions[server_id] = -1
            self._server_states[server_id] = "draining"
            self._drain_started_at[server_id] = now

    def set_latest_weight_version(self, version: int) -> None:
        """Publish a committed version and stop new leases on stale replicas."""
        if not isinstance(version, int) or version < self._latest_weight_version:
            raise ValueError("Weight versions must be monotonically non-decreasing integers")
        self._latest_weight_version = version
        for server_id, local_version in self._server_versions.items():
            if local_version < version and self._server_states[server_id] == "serving":
                self._server_states[server_id] = "draining"
                self._drain_started_at[server_id] = time.monotonic()

    def set_local_weight_version(self, server_id: str, version: int) -> None:
        """Register the actual version reported by an adapter after installation."""
        if server_id not in self._server_versions:
            raise KeyError(server_id)
        if version < self._server_versions[server_id] or version > self._latest_weight_version:
            raise ValueError(f"Invalid local weight version {version} for {server_id}")
        self._server_versions[server_id] = version

    def acquire_trajectory_lease(self, trajectory_id: str) -> dict[str, Any]:
        """Bind one ``(uid, session_id)`` trajectory to a replica/version for all turns."""
        if trajectory_id in self._trajectory_leases:
            raise TrajectoryLeaseError(f"Trajectory {trajectory_id!r} already has a lease")
        candidates = [sid for sid, state in self._server_states.items() if state == "serving"]
        if not candidates:
            raise ServerBusyError("No serving replica can accept a trajectory lease")
        latest_available = max(self._server_versions[sid] for sid in candidates)
        candidates = [sid for sid in candidates if self._server_versions[sid] == latest_available]
        lease_counts = {
            sid: sum(lease["server_id"] == sid for lease in self._trajectory_leases.values()) for sid in candidates
        }
        min_load = min(self._inflight_requests[sid] + lease_counts[sid] for sid in candidates)
        candidates = [sid for sid in candidates if self._inflight_requests[sid] + lease_counts[sid] == min_load]
        server_id = random.choice(candidates)
        token = uuid4().hex
        lease = {
            "trajectory_id": trajectory_id,
            "server_id": server_id,
            "weight_version": self._server_versions[server_id],
            "token": token,
        }
        self._trajectory_leases[trajectory_id] = lease
        self._lease_tokens[token] = trajectory_id
        return dict(lease)

    def release_trajectory_lease(self, trajectory_id: str, token: str) -> None:
        lease = self._trajectory_leases.get(trajectory_id)
        if lease is None or lease["token"] != token:
            raise TrajectoryLeaseError(f"Invalid lease release for {trajectory_id!r}")
        self._trajectory_leases.pop(trajectory_id, None)
        self._lease_tokens.pop(token, None)

    def begin_repack_drain(self, server_id: str) -> int:
        if self._server_states.get(server_id) != "serving":
            raise MigrationConflictError(f"Server {server_id!r} is not serving")
        self._server_states[server_id] = "draining"
        return self._server_versions[server_id]

    def finish_repack_drain(self, server_id: str, expected_version: int) -> None:
        if self._server_versions.get(server_id) != expected_version:
            raise MigrationConflictError("Weight version changed during repack")
        if self._server_states.get(server_id) == "draining" and expected_version == self._latest_weight_version:
            self._server_states[server_id] = "serving"

    def servers_pending_update(self) -> list[str]:
        """Return drained replicas for which both leases and generate calls reached zero."""
        return [
            sid
            for sid, state in self._server_states.items()
            if state == "draining"
            and self._server_versions[sid] < self._latest_weight_version
            and self._inflight_requests[sid] == 0
            and not any(lease["server_id"] == sid for lease in self._trajectory_leases.values())
        ]

    def mark_server_updating(self, server_id: str) -> int:
        if server_id not in self.servers_pending_update():
            raise MigrationConflictError(f"Server {server_id!r} is not fully drained")
        self._server_states[server_id] = "updating"
        return self._latest_weight_version

    def complete_server_update(self, server_id: str, version: int, success: bool) -> None:
        if self._server_states.get(server_id) != "updating":
            raise MigrationConflictError(f"Server {server_id!r} is not updating")
        if success:
            self.set_local_weight_version(server_id, version)
            self._server_states[server_id] = (
                "serving" if self._server_versions[server_id] == self._latest_weight_version else "draining"
            )
        else:
            # Keep the old local version visible and retry from draining.
            self._server_states[server_id] = "draining"
        self._drain_started_at[server_id] = time.monotonic()
        if success and self._server_states[server_id] == "serving":
            self._drain_started_at.pop(server_id, None)

    def get_weight_versions(self) -> dict[str, Any]:
        return {
            "latest": self._latest_weight_version,
            "local": dict(self._server_versions),
            "state": dict(self._server_states),
            "drain_started_at": dict(self._drain_started_at),
            "active_leases": len(self._trajectory_leases),
        }

    def list_inflight(self, server_id: str) -> list[dict[str, Any]]:
        return [dict(record) for record in self._request_registry.values() if record["server_id"] == server_id]

    def prepare_migration(
        self, request_id: str, destination_server_id: str, expected_attempt: int | None = None
    ) -> dict[str, Any]:
        """Reserve destination capacity and atomically CAS sticky/attempt/lease."""
        record = self._request_registry.get(request_id)
        if record is None:
            raise MigrationConflictError(f"Request {request_id!r} already completed")
        if request_id in self._migration_reservations:
            raise MigrationConflictError(f"Request {request_id!r} already has a migration reservation")
        if expected_attempt is not None and record["attempt"] != expected_attempt:
            raise MigrationConflictError("Request attempt changed before migration")
        source = record["server_id"]
        if destination_server_id == source or destination_server_id not in self._servers:
            raise MigrationConflictError("Migration destination must be another registered replica")
        if self._server_states[destination_server_id] != "serving":
            raise MigrationConflictError("Migration destination is not serving")
        if self._server_versions[destination_server_id] != record["weight_version"]:
            raise MigrationConflictError("Cross-version trajectory migration is forbidden")
        if not self._has_capacity(destination_server_id):
            raise ServerBusyError("Migration destination is full")

        lease = None
        if record.get("trajectory_id") is not None:
            lease = self._trajectory_leases.get(record["trajectory_id"])
            if lease is None or lease["server_id"] != source:
                raise MigrationConflictError("Trajectory lease no longer matches the source")
            lease["server_id"] = destination_server_id

        attempt = record["attempt"] + 1
        self._inflight_requests[destination_server_id] += 1
        self._migration_reservations[request_id] = {
            "source": source,
            "destination": destination_server_id,
            "attempt": attempt,
            "previous_record": dict(record),
            "started_at": time.monotonic(),
        }
        record.update(
            server_id=destination_server_id,
            attempt=attempt,
            weight_version=self._server_versions[destination_server_id],
        )
        self._request_id_to_server[request_id] = destination_server_id
        self._metrics["migration_requests_total"] += 1
        return dict(self._migration_reservations[request_id])

    def cancel_migration(self, request_id: str, request_completed: bool = False) -> None:
        reservation = self._migration_reservations.pop(request_id, None)
        if reservation is None:
            return
        record = reservation["previous_record"]
        if request_completed:
            self._request_registry.pop(request_id, None)
        else:
            self._request_registry[request_id] = record
        self._request_id_to_server[request_id] = record["server_id"]
        if record.get("trajectory_id") in self._trajectory_leases:
            self._trajectory_leases[record["trajectory_id"]]["server_id"] = record["server_id"]
        if self._inflight_requests.get(reservation["destination"], 0) > 0:
            self._inflight_requests[reservation["destination"]] -= 1
        self._metrics["migration_failed"] += 1

    def record_admission_wait(self, seconds: float) -> None:
        self._metrics["admission_wait_seconds"] += max(0.0, seconds)
        self._metrics["admission_wait_count"] += 1

    def consume_migration_resume(self, request_id: str) -> dict[str, Any] | None:
        return self._migration_resumes.pop(request_id, None)

    def record_migration_result(
        self,
        *,
        success: bool,
        reprefill_tokens: int = 0,
        duration_seconds: float = 0.0,
        fallback_to_lb: bool = False,
    ) -> None:
        self._metrics["migration_success" if success else "migration_failed"] += 1
        self._metrics["migration_reprefill_tokens"] += reprefill_tokens
        self._metrics["migration_duration_seconds"] += max(0.0, duration_seconds)
        self._metrics["migration_fallback_to_lb"] += int(fallback_to_lb)

    def get_metrics(self) -> dict[str, Any]:
        versions = list(self._server_versions.values())
        lag = [max(0, self._latest_weight_version - version) for version in versions]
        return {
            "counters": dict(self._metrics),
            "gauges": {
                "inflight_requests": dict(self._inflight_requests),
                "draining_servers": sum(state == "draining" for state in self._server_states.values()),
                "updating_servers": sum(state == "updating" for state in self._server_states.values()),
                "version_lag": dict(zip(self._server_versions, lag, strict=True)),
                "active_trajectory_leases": len(self._trajectory_leases),
            },
        }


def _create_global_sticky_inflight(
    servers: dict[str, Any],
    full_determinism: bool = False,
    load_balancer_cls: type | None = None,
    max_inflight_requests_per_replica: int | None = None,
    laminar_routing_enabled: bool = False,
):
    """Factory for the default sticky-session + least-inflight strategy.

    Args:
        servers: ``{server_address: actor_handle}`` mapping.
        full_determinism: Rollout-level ``rollout.full_determinism`` flag;
            enables full-hash deterministic routing.
        load_balancer_cls: Class to use as the routing actor. A subclass overrides
            :meth:`acquire_server` and takes full control of routing, so
            ``full_determinism`` is not forwarded to it.
    """
    kwargs = dict(servers=servers, max_cache_size=DEFAULT_ROUTING_CACHE_SIZE)
    # The default GlobalRequestLoadBalancer honors the full_determinism flag
    # in acquire_server. A custom subclass overrides acquire_server and takes
    # full control of routing, so the flag is not forwarded to it.
    if load_balancer_cls is GlobalRequestLoadBalancer:
        kwargs.update(
            full_determinism=full_determinism,
            max_inflight_requests_per_replica=max_inflight_requests_per_replica,
            laminar_routing_enabled=laminar_routing_enabled,
        )
    elif max_inflight_requests_per_replica is not None or laminar_routing_enabled:
        raise ValueError("The injected custom router must declare and implement the enabled Laminar capabilities")
    return ray.remote(load_balancer_cls).remote(**kwargs)


def _load_router_yaml(router_config_path: str) -> dict:
    """Load a router YAML config into a plain dict.

    Hydra ``defaults`` blocks are rejected: inline the referenced configs or
    compose them inside the router plugin's constructor.
    """
    full_path = resolve_config_path(router_config_path)
    if not os.path.isfile(full_path):
        raise FileNotFoundError(f"Router config file not found: {full_path}")

    cfg = OmegaConf.load(full_path)
    if "defaults" in cfg:
        raise ValueError(
            f"Router config {full_path} uses a Hydra 'defaults' block, which is not "
            "supported. Inline the referenced configs or compose them inside the "
            "router plugin's constructor."
        )
    return OmegaConf.to_container(cfg, resolve=True)


def _resolve_router_class(router_class: str) -> type:
    """Validate and import a load-balancer class from an FQN string.

    Raises:
        TypeError: If the resolved object is not callable.
    """
    cls = load_class_from_fqn(router_class, "load balancer class")
    if not callable(cls):
        raise TypeError(
            f"'{router_class}' is not callable (type: {type(cls).__name__}). "
            f"Expected a class with a .remote() constructor."
        )
    return cls


def _create_plugin_extension(
    servers: dict[str, Any],
    router_config_path: str,
    required_capabilities: list[str] | None = None,
    max_inflight_requests_per_replica: int | None = None,
):
    """Factory for a user-defined load balancer loaded from an external YAML.

    The file must contain ``router_class`` (FQN); the whole loaded dict is
    passed to the constructor.

    Args:
        servers: ``{server_address: actor_handle}`` mapping.
        router_config_path: Path to the external YAML file.

    Raises:
        ValueError: If ``router_config_path`` is missing or the YAML lacks
            ``router_class``.
        ImportError: If a module or package cannot be imported.
        AttributeError: If the class does not exist in the module.
    """
    yaml_config = _load_router_yaml(router_config_path)
    missing_capabilities = set(required_capabilities or []) - set(yaml_config.get("capabilities", []))
    if missing_capabilities:
        raise ValueError(
            f"Custom router does not declare required Laminar capabilities: {sorted(missing_capabilities)}"
        )
    if max_inflight_requests_per_replica is not None:
        yaml_config["max_inflight_requests_per_replica"] = max_inflight_requests_per_replica
    router_class = yaml_config.get("router_class", None)
    if not router_class:
        raise ValueError(
            "External router YAML must contain 'router_class'. "
            "Example: router_class: uni_agent.llm_router.KvcAwareRouter"
        )
    cls = _resolve_router_class(router_class)
    logger.info(
        "Creating plugin load balancer from YAML: class=%s, servers=%d, config=%s",
        router_class,
        len(servers),
        yaml_config,
    )
    ray_cls = cls if isinstance(cls, ray.actor.ActorClass) else ray.remote(cls)
    return ray_cls.remote(servers, yaml_config)


def get_router_handle(
    servers: dict[str, Any],
    router_config_path: str | None = None,
    full_determinism: bool = False,
    load_balancer_cls: type | None = None,
    max_inflight_requests_per_replica: int | None = None,
    required_capabilities: list[str] | None = None,
) -> Any:
    """Create a load balancer instance from router configuration.

    Args:
        servers: ``{server_address: actor_handle}`` mapping.
        router_config_path: Optional external router YAML path. When set, a
            user-defined plugin is loaded; otherwise the default sticky-session
            + least-inflight strategy is used.
        full_determinism: Rollout-level ``rollout.full_determinism`` flag,
            forwarded to strategies that support deterministic routing.
        load_balancer_cls: Optional subclass of the default strategy's load
            balancer. Takes precedence over the config-selected strategy; not
            applicable to the YAML plugin.
    """
    required_capabilities = list(required_capabilities or [])
    if max_inflight_requests_per_replica is not None:
        required_capabilities.append("admission_control")
    if router_config_path and load_balancer_cls is None:
        return _create_plugin_extension(
            servers=servers,
            router_config_path=router_config_path,
            required_capabilities=required_capabilities,
            max_inflight_requests_per_replica=max_inflight_requests_per_replica,
        )

    # Programmatic injection (e.g. verl-omni's Deterministic* subclasses)
    # overrides the config-selected strategy. If both are set, the YAML is
    # ignored — warn so the mismatch doesn't fail silently.
    if router_config_path and load_balancer_cls is not None:
        logger.warning(
            "Both router_config_path=%r and load_balancer_cls=%s are set; "
            "the YAML plugin is ignored in favor of the injected subclass.",
            router_config_path,
            getattr(load_balancer_cls, "__name__", load_balancer_cls),
        )
    return _create_global_sticky_inflight(
        servers=servers,
        full_determinism=full_determinism,
        load_balancer_cls=load_balancer_cls or GlobalRequestLoadBalancer,
        max_inflight_requests_per_replica=max_inflight_requests_per_replica,
        laminar_routing_enabled=bool(required_capabilities),
    )
