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
"""Pure-Python metrics, planning, and coordination for Laminar repack.

The planner is deliberately independent of Ray and rollout engine implementations.  Server
managers only need to supply structured snapshots and, when migration is enabled, an execution
callback.  The coordinator never creates an event loop; its background task belongs to the loop
in which :meth:`RepackCoordinatorCore.start` is awaited.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import threading
import time
import urllib.request
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ReplicaMetrics:
    """An immutable per-replica metrics snapshot.

    The optional metric values represent unavailable engine capabilities.  They are retained in
    history so consumers can distinguish missing data from an idle replica.  ``num_tokens`` and
    ``token_budget`` describe real KV-token occupancy/capacity when available; planning falls back
    to the KV-utilization estimate otherwise.  ``request_budget`` is the replica's real request
    capacity and further constrains the global roofline batch size.
    """

    server_id: str
    kv_cache_usage: float | None
    num_running_reqs: int | None
    num_waiting_reqs: int | None
    weight_version: int | str | None
    role: str | None
    timestamp: float
    state: str = "serving"
    num_running_seqs: int | None = None
    num_tokens: int | None = None
    token_budget: int | None = None
    request_budget: int | None = None

    def __post_init__(self) -> None:
        if not self.server_id:
            raise ValueError("server_id must be non-empty")
        if self.kv_cache_usage is not None and (
            not math.isfinite(self.kv_cache_usage) or not 0.0 <= self.kv_cache_usage <= 1.0
        ):
            raise ValueError("kv_cache_usage must be in [0, 1] or None")
        for name in (
            "num_running_reqs",
            "num_waiting_reqs",
            "num_running_seqs",
            "num_tokens",
            "token_budget",
            "request_budget",
        ):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")
        if self.role is not None and not self.role:
            raise ValueError("role must be non-empty or None")
        if not self.state:
            raise ValueError("state must be non-empty")
        if not math.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")

    def is_complete(self) -> bool:
        """Return whether all metrics required for idle/repack decisions are present."""

        return (
            self.kv_cache_usage is not None
            and self.num_running_reqs is not None
            and self.num_waiting_reqs is not None
            and self.weight_version is not None
            and self.role is not None
        )

    def is_fresh(self, now: float, max_age_s: float) -> bool:
        """Return whether this snapshot is in the recent past, including clock-skew safety."""

        _validate_max_age(max_age_s)
        age = now - self.timestamp
        return 0.0 <= age <= max_age_s


class ReplicaMetricsHistory:
    """A bounded, per-replica history with completeness and freshness queries."""

    def __init__(self, maxlen: int = 2, max_age_s: float = 5.0):
        if maxlen < 2:
            raise ValueError("maxlen must be at least 2 for current/previous idleness checks")
        _validate_max_age(max_age_s)
        self.maxlen = maxlen
        self.max_age_s = max_age_s
        self._history: dict[str, deque[ReplicaMetrics]] = {}

    @property
    def server_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._history))

    def add(self, metrics: ReplicaMetrics) -> bool:
        """Add a snapshot, replacing an equal timestamp and rejecting out-of-order samples."""

        if not isinstance(metrics, ReplicaMetrics):
            raise TypeError("metrics must be a ReplicaMetrics instance")
        entries = self._history.setdefault(metrics.server_id, deque(maxlen=self.maxlen))
        if entries and metrics.timestamp < entries[-1].timestamp:
            return False
        if entries and metrics.timestamp == entries[-1].timestamp:
            entries[-1] = metrics
        else:
            entries.append(metrics)
        return True

    def update(self, snapshots: Mapping[str, ReplicaMetrics | None]) -> set[str]:
        """Apply a collection result and return server IDs whose samples were out of order.

        ``None`` clears a server's history.  This prevents a failed collection from reusing an old
        snapshot that still happens to be within its freshness window.
        """

        rejected: set[str] = set()
        for server_id, metrics in snapshots.items():
            if metrics is None:
                self.clear(server_id)
                continue
            if metrics.server_id != server_id:
                raise ValueError(f"snapshot key {server_id!r} does not match server_id {metrics.server_id!r}")
            if not self.add(metrics):
                rejected.add(server_id)
        return rejected

    def clear(self, server_id: str) -> None:
        self._history.pop(server_id, None)

    def latest(self, server_id: str) -> ReplicaMetrics | None:
        entries = self._history.get(server_id)
        return entries[-1] if entries else None

    def current_and_previous(self, server_id: str) -> tuple[ReplicaMetrics, ReplicaMetrics] | None:
        entries = self._history.get(server_id)
        if entries is None or len(entries) < 2:
            return None
        return entries[-1], entries[-2]

    def latest_snapshot(self, now: float, max_age_s: float | None = None) -> dict[str, ReplicaMetrics]:
        """Return complete, fresh current snapshots without mutating history."""

        max_age_s = self.max_age_s if max_age_s is None else max_age_s
        _validate_max_age(max_age_s)
        result: dict[str, ReplicaMetrics] = {}
        for server_id in self.server_ids:
            current = self.latest(server_id)
            if _is_complete_and_fresh(current, now, max_age_s):
                result[server_id] = current
        return result


def parse_prometheus_snapshot(
    text: str,
    *,
    server_id: str,
    weight_version: int | None,
    role: str,
    request_budget: int | None,
    timestamp: float | None = None,
    metric_names: Mapping[str, tuple[str, ...]] | None = None,
) -> ReplicaMetrics | None:
    """Parse required KV/request gauges; incomplete engine output returns ``None``."""
    names = metric_names or {
        "kv_cache_usage": ("vllm:gpu_cache_usage_perc", "sglang:token_usage", "sglang:kv_cache_usage"),
        "num_running_reqs": ("vllm:num_requests_running", "sglang:num_running_reqs"),
        "num_waiting_reqs": (
            "vllm:num_requests_waiting",
            "sglang:num_queue_reqs",
            "sglang:num_waiting_reqs",
        ),
    }
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        sample, _, raw_value = line.rpartition(" ")
        metric_name = sample.split("{", 1)[0]
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if math.isfinite(value):
            values[metric_name] = value

    parsed: dict[str, float | int | None] = {}
    for field_name, candidates in names.items():
        value = next((values[name] for name in candidates if name in values), None)
        parsed[field_name] = value
    if any(parsed[name] is None for name in names):
        return None
    kv_usage = float(parsed["kv_cache_usage"])
    if 1.0 < kv_usage <= 100.0:
        kv_usage /= 100.0
    try:
        return ReplicaMetrics(
            server_id=server_id,
            kv_cache_usage=kv_usage,
            num_running_reqs=int(parsed["num_running_reqs"]),
            num_waiting_reqs=int(parsed["num_waiting_reqs"]),
            weight_version=weight_version,
            role=role,
            timestamp=time.time() if timestamp is None else timestamp,
            request_budget=request_budget,
        )
    except (TypeError, ValueError):
        return None


async def fetch_prometheus_snapshot(
    server_address: str,
    *,
    weight_version: int | None,
    role: str,
    request_budget: int | None,
    timeout_s: float = 2.0,
) -> ReplicaMetrics | None:
    """Fetch one replica's metrics without blocking the Ray actor event loop."""

    def fetch() -> str:
        host, separator, port = server_address.rpartition(":")
        if not separator:
            raise ValueError(f"Invalid server address: {server_address}")
        url = f"http://{host}:{port}/metrics"
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return response.read().decode("utf-8")

    try:
        text = await asyncio.wait_for(asyncio.to_thread(fetch), timeout=timeout_s + 0.1)
    except (OSError, ValueError, TimeoutError, asyncio.TimeoutError):
        return None
    return parse_prometheus_snapshot(
        text,
        server_id=server_address,
        weight_version=weight_version,
        role=role,
        request_budget=request_budget,
    )


def select_idle_replicas(
    history: ReplicaMetricsHistory,
    kv_cache_threshold: float,
    roofline_batch_size: int | None,
    *,
    now: float | None = None,
    max_age_s: float | None = None,
) -> set[str]:
    """Implement Algorithm 1's source selection using current/previous snapshots.

    Missing, incomplete, stale, non-serving, or version/role-transitioning samples never count as
    idle.  Waiting work also disqualifies a replica even when its KV utilization is decreasing.
    """

    if not isinstance(history, ReplicaMetricsHistory):
        raise TypeError("history must be a ReplicaMetricsHistory")
    _validate_threshold(kv_cache_threshold)
    _validate_optional_budget(roofline_batch_size, "roofline_batch_size")
    now = time.time() if now is None else now
    max_age_s = history.max_age_s if max_age_s is None else max_age_s
    _validate_max_age(max_age_s)

    idle: set[str] = set()
    for server_id in history.server_ids:
        pair = history.current_and_previous(server_id)
        if pair is None:
            continue
        current, previous = pair
        if not _is_complete_and_fresh(current, now, max_age_s) or not _is_complete_and_fresh(previous, now, max_age_s):
            continue
        if current.state != "serving" or previous.state != "serving":
            continue
        if current.weight_version != previous.weight_version or current.role != previous.role:
            continue
        request_budget = roofline_batch_size if roofline_batch_size is not None else current.request_budget
        if request_budget is None:
            continue
        if (
            current.kv_cache_usage < min(kv_cache_threshold, previous.kv_cache_usage)
            and current.num_running_reqs < request_budget
            and current.num_waiting_reqs == 0
        ):
            idle.add(server_id)
    return idle


@dataclass(frozen=True)
class RepackMove:
    """One estimated source-to-destination migration in a repack plan."""

    source: str
    destination: str
    weight_version: int | str
    role: str
    num_requests: int
    kv_cache_usage: float
    num_tokens: int | None = None

    @property
    def source_id(self) -> str:
        return self.source

    @property
    def destination_id(self) -> str:
        return self.destination


@dataclass(frozen=True)
class RepackPlan:
    """A serializable Best-Fit plan; no engine operation is performed by the planner."""

    moves: tuple[RepackMove, ...] = ()
    released_replicas: frozenset[str] = frozenset()
    estimated_destination_kv_util: dict[str, float] = field(default_factory=dict)
    estimated_destination_requests: dict[str, int] = field(default_factory=dict)
    estimated_destination_tokens: dict[str, int | None] = field(default_factory=dict)
    dry_run: bool = True

    @property
    def released(self) -> frozenset[str]:
        return self.released_replicas

    @property
    def estimated_moves(self) -> int:
        return len(self.moves)

    @property
    def estimated_requests(self) -> int:
        return sum(move.num_requests for move in self.moves)

    @property
    def estimated_released_replicas(self) -> int:
        return len(self.released_replicas)


def plan_repack(
    stats_by_server: Mapping[str, ReplicaMetrics | None],
    idle_set: set[str] | frozenset[str],
    c_max: float,
    b: int | None,
    *,
    now: float | None = None,
    max_age_s: float = 5.0,
    dry_run: bool = True,
) -> RepackPlan:
    """Plan same-version, same-role migrations with cumulative Best-Fit packing.

    Sources are considered from lowest to highest KV load.  A destination is selected by the
    fullest *projected* KV load that still fits, with server ID providing deterministic tie breaks.
    Released sources cannot subsequently be destinations.  A source that has already received a
    planned migration is skipped, preventing move chains whose estimates would become invalid.
    """

    _validate_threshold(c_max)
    _validate_optional_budget(b, "b")
    _validate_max_age(max_age_s)
    now = time.time() if now is None else now

    eligible: dict[str, ReplicaMetrics] = {}
    for mapping_id, metrics in stats_by_server.items():
        if metrics is None:
            continue
        if not isinstance(metrics, ReplicaMetrics):
            raise TypeError("stats_by_server values must be ReplicaMetrics instances or None")
        if metrics.server_id != mapping_id:
            continue
        if metrics.state == "serving" and _is_complete_and_fresh(metrics, now, max_age_s):
            eligible[mapping_id] = metrics

    sources = sorted(
        (eligible[server_id] for server_id in idle_set if server_id in eligible),
        key=lambda metrics: (metrics.kv_cache_usage, metrics.num_running_reqs, metrics.server_id),
    )
    assigned_kv: dict[str, float] = {}
    assigned_requests: dict[str, int] = {}
    assigned_tokens: dict[str, int] = {}
    unknown_token_estimates: set[str] = set()
    destinations_with_incoming: set[str] = set()
    released: set[str] = set()
    moves: list[RepackMove] = []

    for source in sources:
        if source.server_id in destinations_with_incoming:
            continue
        best: ReplicaMetrics | None = None
        best_projected_kv = -1.0
        for destination in sorted(eligible.values(), key=lambda metrics: metrics.server_id):
            if destination.server_id == source.server_id or destination.server_id in released:
                continue
            if destination.weight_version != source.weight_version or destination.role != source.role:
                continue
            if not _can_fit(destination, source, c_max, b, assigned_kv, assigned_requests, assigned_tokens):
                continue
            projected_kv = (
                destination.kv_cache_usage + assigned_kv.get(destination.server_id, 0.0) + source.kv_cache_usage
            )
            if best is None or projected_kv > best_projected_kv:
                best = destination
                best_projected_kv = projected_kv

        if best is None:
            continue
        destination_id = best.server_id
        moves.append(
            RepackMove(
                source=source.server_id,
                destination=destination_id,
                weight_version=source.weight_version,
                role=source.role,
                num_requests=source.num_running_reqs,
                kv_cache_usage=source.kv_cache_usage,
                num_tokens=source.num_tokens,
            )
        )
        released.add(source.server_id)
        destinations_with_incoming.add(destination_id)
        assigned_kv[destination_id] = assigned_kv.get(destination_id, 0.0) + source.kv_cache_usage
        assigned_requests[destination_id] = assigned_requests.get(destination_id, 0) + source.num_running_reqs
        if source.num_tokens is None:
            unknown_token_estimates.add(destination_id)
        else:
            assigned_tokens[destination_id] = assigned_tokens.get(destination_id, 0) + source.num_tokens

    estimated_kv: dict[str, float] = {}
    estimated_requests: dict[str, int] = {}
    estimated_tokens: dict[str, int | None] = {}
    for destination_id in destinations_with_incoming:
        destination = eligible[destination_id]
        estimated_kv[destination_id] = destination.kv_cache_usage + assigned_kv[destination_id]
        estimated_requests[destination_id] = destination.num_running_reqs + assigned_requests[destination_id]
        if destination.num_tokens is None or destination_id in unknown_token_estimates:
            estimated_tokens[destination_id] = None
        else:
            estimated_tokens[destination_id] = destination.num_tokens + assigned_tokens.get(destination_id, 0)

    return RepackPlan(
        moves=tuple(moves),
        released_replicas=frozenset(released),
        estimated_destination_kv_util=estimated_kv,
        estimated_destination_requests=estimated_requests,
        estimated_destination_tokens=estimated_tokens,
        dry_run=dry_run,
    )


def _can_fit(
    destination: ReplicaMetrics,
    source: ReplicaMetrics,
    c_max: float,
    b: int | None,
    assigned_kv: Mapping[str, float],
    assigned_requests: Mapping[str, int],
    assigned_tokens: Mapping[str, int],
) -> bool:
    destination_id = destination.server_id
    projected_kv = destination.kv_cache_usage + assigned_kv.get(destination_id, 0.0) + source.kv_cache_usage
    if projected_kv > c_max:
        return False

    limits = [limit for limit in (b, destination.request_budget) if limit is not None]
    if not limits:
        return False
    projected_requests = (
        destination.num_running_reqs + assigned_requests.get(destination_id, 0) + source.num_running_reqs
    )
    if projected_requests > min(limits):
        return False

    if destination.token_budget is not None and destination.num_tokens is not None and source.num_tokens is not None:
        projected_tokens = destination.num_tokens + assigned_tokens.get(destination_id, 0) + source.num_tokens
        if projected_tokens > destination.token_budget:
            return False
    return True


@dataclass(frozen=True)
class DurationMetrics:
    """Duration aggregate suitable for counter-like duration reporting."""

    count: int = 0
    total_s: float = 0.0
    last_s: float = 0.0

    @property
    def mean_s(self) -> float:
        return self.total_s / self.count if self.count else 0.0


@dataclass(frozen=True)
class RepackCoordinatorMetrics:
    """Immutable structured coordinator counters, gauges, and duration aggregates."""

    trigger_count: int = 0
    plan_count: int = 0
    plan_empty_count: int = 0
    skipped_overlap: int = 0
    skipped_backoff: int = 0
    skipped_no_idle: int = 0
    skipped_state: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0
    backoff_active: bool = False
    planned_sources: int = 0
    planned_destinations: int = 0
    planned_moves: int = 0
    estimated_released_replicas: int = 0
    migration_attempts: int = 0
    migration_successes: int = 0
    migration_failures: int = 0
    migration_moves: int = 0
    collect_duration: DurationMetrics = DurationMetrics()
    plan_duration: DurationMetrics = DurationMetrics()
    migration_duration: DurationMetrics = DurationMetrics()
    tick_duration: DurationMetrics = DurationMetrics()
    last_run_timestamp: float | None = None
    last_run_age_s: float | None = None
    last_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RepackTickResult:
    """The outcome of one coordinator tick."""

    status: str
    reason: str
    plan: RepackPlan | None = None
    idle_replicas: frozenset[str] = frozenset()
    error: str | None = None


MetricsCollector = Callable[[], Mapping[str, ReplicaMetrics | None] | Awaitable[Mapping[str, ReplicaMetrics | None]]]
PlanExecutor = Callable[[RepackPlan], Any | Awaitable[Any]]
StateProbe = Callable[[], bool | Awaitable[bool]]


class RepackCoordinatorCore:
    """Persistent, single-flight collect -> idle -> plan -> optional execute coordinator."""

    def __init__(
        self,
        collect_stats: MetricsCollector,
        *,
        interval_s: float = 5.0,
        kv_cache_threshold: float = 0.99,
        roofline_batch_size: int | None = None,
        history_size: int = 2,
        metrics_max_age_s: float = 5.0,
        enabled: bool = False,
        migration_enabled: bool = False,
        execute_plan: PlanExecutor | None = None,
        can_run: StateProbe | None = None,
        weight_syncing: StateProbe | None = None,
        collection_timeout_s: float | None = None,
        backoff_base_s: float | None = None,
        backoff_multiplier: float = 2.0,
        max_backoff_s: float = 300.0,
        wall_clock: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ):
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        _validate_threshold(kv_cache_threshold)
        _validate_optional_budget(roofline_batch_size, "roofline_batch_size")
        if collection_timeout_s is not None and collection_timeout_s <= 0:
            raise ValueError("collection_timeout_s must be positive or None")
        backoff_base_s = interval_s if backoff_base_s is None else backoff_base_s
        if backoff_base_s <= 0 or max_backoff_s <= 0:
            raise ValueError("backoff durations must be positive")
        if backoff_multiplier < 1.0:
            raise ValueError("backoff_multiplier must be at least 1")

        self.collect_stats = collect_stats
        self.interval_s = interval_s
        self.kv_cache_threshold = kv_cache_threshold
        self.roofline_batch_size = roofline_batch_size
        self.metrics_max_age_s = metrics_max_age_s
        self.enabled = enabled
        self.migration_enabled = migration_enabled
        self.execute_plan = execute_plan
        self.can_run = can_run
        self.weight_syncing = weight_syncing
        self.collection_timeout_s = collection_timeout_s
        self.backoff_base_s = backoff_base_s
        self.backoff_multiplier = backoff_multiplier
        self.max_backoff_s = max_backoff_s
        self.wall_clock = wall_clock
        self.monotonic_clock = monotonic_clock
        self.history = ReplicaMetricsHistory(maxlen=history_size, max_age_s=metrics_max_age_s)

        self._known_server_ids: set[str] = set()
        self._flight_lock = threading.Lock()
        self._task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None
        self._backoff_until = 0.0
        self._last_plan: RepackPlan | None = None
        self._last_run_timestamp: float | None = None
        self._last_error: str | None = None
        self._consecutive_failures = 0
        self._counters: dict[str, int] = {
            name: 0
            for name in (
                "trigger_count",
                "plan_count",
                "plan_empty_count",
                "skipped_overlap",
                "skipped_backoff",
                "skipped_no_idle",
                "skipped_state",
                "failure_count",
                "planned_sources",
                "planned_destinations",
                "planned_moves",
                "estimated_released_replicas",
                "migration_attempts",
                "migration_successes",
                "migration_failures",
                "migration_moves",
            )
        }
        self._durations: dict[str, list[float | int]] = {
            name: [0, 0.0, 0.0] for name in ("collect", "plan", "migration", "tick")
        }

    @property
    def task(self) -> asyncio.Task[None] | None:
        return self._task

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def last_plan(self) -> RepackPlan | None:
        return self._last_plan

    def _duration_metrics(self, name: str) -> DurationMetrics:
        count, total, last = self._durations[name]
        return DurationMetrics(count=int(count), total_s=float(total), last_s=float(last))

    def get_metrics(self) -> RepackCoordinatorMetrics:
        last_age = (
            max(0.0, self.wall_clock() - self._last_run_timestamp) if self._last_run_timestamp is not None else None
        )
        return RepackCoordinatorMetrics(
            **self._counters,
            consecutive_failures=self._consecutive_failures,
            backoff_active=self.monotonic_clock() < self._backoff_until,
            collect_duration=self._duration_metrics("collect"),
            plan_duration=self._duration_metrics("plan"),
            migration_duration=self._duration_metrics("migration"),
            tick_duration=self._duration_metrics("tick"),
            last_run_timestamp=self._last_run_timestamp,
            last_run_age_s=last_age,
            last_error=self._last_error,
        )

    async def tick(self, *, now: float | None = None) -> RepackTickResult:
        """Run one non-blocking-single-flight repack tick.

        Disabled, overlapping, backing-off, weight-syncing, and state-conflicting ticks are no-ops.
        Operational exceptions are captured in the result and activate exponential backoff rather
        than escaping into the persistent background task.
        """

        if not self.enabled:
            return RepackTickResult(status="skipped", reason="disabled")
        self._counters["trigger_count"] += 1
        if not self._flight_lock.acquire(blocking=False):
            self._counters["skipped_overlap"] += 1
            return RepackTickResult(status="skipped", reason="overlap")
        if self.monotonic_clock() < self._backoff_until:
            self._flight_lock.release()
            self._counters["skipped_backoff"] += 1
            return RepackTickResult(status="skipped", reason="backoff")

        started = self.monotonic_clock()
        self._last_run_timestamp = self.wall_clock()
        stage = "state"
        try:
            if self.weight_syncing is not None and await _resolve_probe(self.weight_syncing):
                self._counters["skipped_state"] += 1
                return RepackTickResult(status="skipped", reason="weight_sync")
            if self.can_run is not None and not await _resolve_probe(self.can_run):
                self._counters["skipped_state"] += 1
                return RepackTickResult(status="skipped", reason="state_conflict")

            stage = "collect"
            collect_started = self.monotonic_clock()
            try:
                collected = self.collect_stats()
                if inspect.isawaitable(collected):
                    if self.collection_timeout_s is None:
                        snapshots = await collected
                    else:
                        snapshots = await asyncio.wait_for(collected, timeout=self.collection_timeout_s)
                else:
                    snapshots = collected
                if not isinstance(snapshots, Mapping):
                    raise TypeError("collect_stats must return a mapping")
            finally:
                self._record_duration("collect", self.monotonic_clock() - collect_started)

            rejected = self.history.update(snapshots)
            for server_id in self._known_server_ids - set(snapshots):
                self.history.clear(server_id)
            for server_id in rejected:
                self.history.clear(server_id)
            self._known_server_ids = set(snapshots)
            current_time = self.wall_clock() if now is None else now

            stage = "idle"
            idle = select_idle_replicas(
                self.history,
                self.kv_cache_threshold,
                self.roofline_batch_size,
                now=current_time,
                max_age_s=self.metrics_max_age_s,
            )
            if not idle:
                self._counters["skipped_no_idle"] += 1
                return RepackTickResult(status="skipped", reason="no_idle")

            stage = "plan"
            plan_started = self.monotonic_clock()
            try:
                plan = plan_repack(
                    self.history.latest_snapshot(current_time, self.metrics_max_age_s),
                    idle,
                    self.kv_cache_threshold,
                    self.roofline_batch_size,
                    now=current_time,
                    max_age_s=self.metrics_max_age_s,
                    dry_run=not self.migration_enabled or self.execute_plan is None,
                )
            finally:
                self._record_duration("plan", self.monotonic_clock() - plan_started)
            self._last_plan = plan
            self._counters["plan_count"] += 1
            self._counters["planned_sources"] += len({move.source for move in plan.moves})
            self._counters["planned_destinations"] += len({move.destination for move in plan.moves})
            self._counters["planned_moves"] += plan.estimated_moves
            self._counters["estimated_released_replicas"] += plan.estimated_released_replicas
            if not plan.moves:
                self._counters["plan_empty_count"] += 1
                self._record_success()
                return RepackTickResult(status="skipped", reason="plan_empty", plan=plan, idle_replicas=frozenset(idle))

            if not self.migration_enabled or self.execute_plan is None:
                self._record_success()
                return RepackTickResult(status="planned", reason="dry_run", plan=plan, idle_replicas=frozenset(idle))

            stage = "migration"
            self._counters["migration_attempts"] += 1
            migration_started = self.monotonic_clock()
            try:
                execution_result = self.execute_plan(plan)
                if inspect.isawaitable(execution_result):
                    execution_result = await execution_result
                if execution_result is False:
                    raise RuntimeError("execution callback reported failure")
            finally:
                self._record_duration("migration", self.monotonic_clock() - migration_started)
            self._counters["migration_successes"] += 1
            self._counters["migration_moves"] += plan.estimated_moves
            self._record_success()
            return RepackTickResult(status="migrated", reason="executed", plan=plan, idle_replicas=frozenset(idle))
        except Exception as exc:
            if stage == "migration":
                self._counters["migration_failures"] += 1
            self._record_failure(exc)
            return RepackTickResult(status="error", reason=f"{stage}_failed", error=self._last_error)
        finally:
            self._record_duration("tick", self.monotonic_clock() - started)
            self._flight_lock.release()

    async def start(self) -> None:
        """Start a persistent task on the caller's running event loop; never create a loop."""

        if not self.enabled or self.running:
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="verl-repack-coordinator")

    async def stop(self) -> None:
        """Cancel and await the persistent task, including an in-flight collection/execution."""

        task = self._task
        if task is None:
            return
        if self._stop_event is not None:
            self._stop_event.set()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            if self._task is task:
                self._task = None
                self._stop_event = None

    async def _run(self) -> None:
        while self._stop_event is not None and not self._stop_event.is_set():
            await self.tick()
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.interval_s)
            except asyncio.TimeoutError:
                pass

    def _record_success(self) -> None:
        self._consecutive_failures = 0
        self._backoff_until = 0.0
        self._last_error = None

    def _record_failure(self, exc: Exception) -> None:
        self._counters["failure_count"] += 1
        self._consecutive_failures += 1
        delay = min(
            self.max_backoff_s,
            self.backoff_base_s * self.backoff_multiplier ** (self._consecutive_failures - 1),
        )
        self._backoff_until = self.monotonic_clock() + delay
        self._last_error = f"{type(exc).__name__}: {exc}"

    def _record_duration(self, name: str, duration: float) -> None:
        self._durations[name][0] += 1
        self._durations[name][1] += max(0.0, duration)
        self._durations[name][2] = max(0.0, duration)


async def _resolve_probe(probe: StateProbe) -> bool:
    result = probe()
    if inspect.isawaitable(result):
        result = await result
    return bool(result)


def _is_complete_and_fresh(metrics: ReplicaMetrics | None, now: float, max_age_s: float) -> bool:
    return metrics is not None and metrics.is_complete() and metrics.is_fresh(now, max_age_s)


def _validate_threshold(value: float) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("KV cache threshold must be in [0, 1]")


def _validate_max_age(value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError("max_age_s must be positive and finite")


def _validate_optional_budget(value: int | None, name: str) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
        raise ValueError(f"{name} must be a non-negative integer or None")


__all__ = [
    "DurationMetrics",
    "RepackCoordinatorCore",
    "RepackCoordinatorMetrics",
    "RepackMove",
    "RepackPlan",
    "RepackTickResult",
    "ReplicaMetrics",
    "ReplicaMetricsHistory",
    "fetch_prometheus_snapshot",
    "parse_prometheus_snapshot",
    "plan_repack",
    "select_idle_replicas",
]
