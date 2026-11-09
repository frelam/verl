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
"""GPU-free tests for the independent Laminar repack core."""

import asyncio
from dataclasses import FrozenInstanceError

import pytest

from verl.workers.rollout import weight_sync_coordinator
from verl.workers.rollout.repack import (
    RepackCoordinatorCore,
    RepackMove,
    RepackPlan,
    ReplicaMetrics,
    ReplicaMetricsHistory,
    parse_prometheus_snapshot,
    plan_repack,
    select_idle_replicas,
)
from verl.workers.rollout.repack_coordinator import RepackCoordinatorActor, RepackPlanExecutor
from verl.workers.rollout.router import GlobalRequestLoadBalancer


def _metrics(server_id, **overrides):
    values = {
        "server_id": server_id,
        "kv_cache_usage": 0.2,
        "num_running_reqs": 1,
        "num_waiting_reqs": 0,
        "weight_version": 7,
        "role": "decode",
        "timestamp": 100.0,
    }
    values.update(overrides)
    return ReplicaMetrics(**values)


def _add_pair(history, server_id, previous_kv, current_kv, **current_overrides):
    previous_overrides = {
        key: current_overrides[key]
        for key in ("weight_version", "role", "num_running_reqs", "num_waiting_reqs", "state")
        if key in current_overrides
    }
    history.add(
        _metrics(
            server_id,
            kv_cache_usage=previous_kv,
            timestamp=99.0,
            **previous_overrides,
        )
    )
    history.add(_metrics(server_id, kv_cache_usage=current_kv, **current_overrides))


def test_replica_metrics_is_immutable_and_structured():
    metrics = _metrics("s0", num_tokens=200, token_budget=1000, request_budget=8)

    assert metrics.state == "serving"
    assert metrics.is_complete()
    assert metrics.num_tokens == 200
    assert metrics.token_budget == 1000
    assert metrics.request_budget == 8
    with pytest.raises(FrozenInstanceError):
        metrics.kv_cache_usage = 0.5

    assert not _metrics("missing", kv_cache_usage=None).is_complete()
    with pytest.raises(ValueError, match="kv_cache_usage"):
        _metrics("bad", kv_cache_usage=1.1)
    with pytest.raises(ValueError, match="num_running_reqs"):
        _metrics("bad", num_running_reqs=-1)


def test_history_is_bounded_and_rejects_out_of_order_samples():
    history = ReplicaMetricsHistory(maxlen=2, max_age_s=5)

    assert history.add(_metrics("s0", timestamp=96.0, kv_cache_usage=0.5))
    assert history.add(_metrics("s0", timestamp=98.0, kv_cache_usage=0.4))
    assert not history.add(_metrics("s0", timestamp=97.0, kv_cache_usage=0.3))
    assert history.current_and_previous("s0")[0].timestamp == 98.0
    assert history.add(_metrics("s0", timestamp=100.0, kv_cache_usage=0.2))
    assert history.current_and_previous("s0")[1].timestamp == 98.0

    replacement = _metrics("s0", timestamp=100.0, kv_cache_usage=0.1)
    assert history.add(replacement)
    assert history.latest("s0") == replacement
    assert history.current_and_previous("s0")[1].timestamp == 98.0


def test_history_none_clears_old_data_and_snapshot_filters_freshness():
    history = ReplicaMetricsHistory(max_age_s=5)
    history.update(
        {
            "fresh": _metrics("fresh", timestamp=100.0),
            "stale": _metrics("stale", timestamp=94.9),
            "future": _metrics("future", timestamp=100.1),
            "incomplete": _metrics("incomplete", timestamp=100.0, num_waiting_reqs=None),
        }
    )

    assert set(history.latest_snapshot(now=100.0)) == {"fresh"}
    history.update({"fresh": None})
    assert history.latest("fresh") is None
    assert history.latest_snapshot(now=100.0) == {}


def test_algorithm_1_idle_selection_requires_decreasing_kv_and_available_requests():
    history = ReplicaMetricsHistory(max_age_s=10)
    _add_pair(history, "idle", 0.8, 0.7, num_running_reqs=3)
    _add_pair(history, "flat", 0.8, 0.8)
    _add_pair(history, "above-threshold", 0.95, 0.92)
    _add_pair(history, "at-request-budget", 0.8, 0.7, num_running_reqs=8)
    _add_pair(history, "waiting", 0.8, 0.7, num_waiting_reqs=1)

    assert select_idle_replicas(history, 0.9, 8, now=100.0) == {"idle"}


def test_idle_requires_two_complete_fresh_compatible_serving_snapshots():
    history = ReplicaMetricsHistory(max_age_s=5)
    history.add(_metrics("one-point", timestamp=100.0))
    _add_pair(history, "stale-previous", 0.8, 0.2)
    history.clear("stale-previous")
    history.add(_metrics("stale-previous", timestamp=94.0, kv_cache_usage=0.8))
    history.add(_metrics("stale-previous", timestamp=100.0, kv_cache_usage=0.2))

    _add_pair(history, "draining", 0.8, 0.2, state="draining")
    _add_pair(history, "missing", 0.8, 0.2, num_waiting_reqs=None)

    history.add(_metrics("version-change", timestamp=99.0, kv_cache_usage=0.8, weight_version=1))
    history.add(_metrics("version-change", timestamp=100.0, kv_cache_usage=0.2, weight_version=2))
    history.add(_metrics("role-change", timestamp=99.0, kv_cache_usage=0.8, role="prefill"))
    history.add(_metrics("role-change", timestamp=100.0, kv_cache_usage=0.2, role="decode"))
    _add_pair(history, "uses-real-request-budget", 0.8, 0.2, num_running_reqs=3)

    assert select_idle_replicas(history, 0.9, None, now=100.0) == set()
    assert select_idle_replicas(history, 0.9, 4, now=100.0) == {"uses-real-request-budget"}

    history.add(
        _metrics(
            "uses-real-request-budget",
            timestamp=100.0,
            kv_cache_usage=0.2,
            num_running_reqs=3,
            request_budget=3,
        )
    )
    assert select_idle_replicas(history, 0.9, None, now=100.0) == set()


def test_best_fit_accumulates_destination_load_and_estimates_metrics():
    stats = {
        "s1": _metrics("s1", kv_cache_usage=0.15, num_running_reqs=1),
        "s2": _metrics("s2", kv_cache_usage=0.20, num_running_reqs=1),
        "s3": _metrics("s3", kv_cache_usage=0.25, num_running_reqs=1),
        "d1": _metrics("d1", kv_cache_usage=0.45, num_running_reqs=2),
        "d2": _metrics("d2", kv_cache_usage=0.40, num_running_reqs=2),
        "d3": _metrics("d3", kv_cache_usage=0.30, num_running_reqs=0),
    }
    idle = {"s1", "s2", "s3"}

    plan = plan_repack(stats, idle, c_max=0.8, b=6, now=100.0)

    assert [(move.source, move.destination) for move in plan.moves] == [
        ("s1", "d1"),
        ("s2", "d1"),
        ("s3", "d2"),
    ]
    assert plan.released_replicas == {"s1", "s2", "s3"}
    assert plan.estimated_moves == 3
    assert plan.estimated_requests == 3
    assert plan.estimated_destination_kv_util == {"d1": pytest.approx(0.8), "d2": pytest.approx(0.65)}
    assert plan.estimated_destination_requests == {"d1": 4, "d2": 3}
    assert idle == {"s1", "s2", "s3"}
    assert plan.dry_run


def test_best_fit_obeys_real_request_and_token_budgets_when_available():
    request_limited = {
        "source": _metrics("source", kv_cache_usage=0.2, num_running_reqs=3),
        "destination": _metrics("destination", kv_cache_usage=0.3, num_running_reqs=2, request_budget=4),
    }
    assert not plan_repack(request_limited, {"source"}, 0.9, 6, now=100.0).moves

    token_limited = {
        "source": _metrics("source", kv_cache_usage=0.2, num_tokens=200),
        "destination": _metrics("destination", kv_cache_usage=0.3, num_tokens=600, token_budget=750),
    }
    assert not plan_repack(token_limited, {"source"}, 0.9, 6, now=100.0).moves

    token_limited["destination"] = _metrics("destination", kv_cache_usage=0.3, num_tokens=500, token_budget=750)
    plan = plan_repack(token_limited, {"source"}, 0.9, 6, now=100.0)
    assert plan.estimated_destination_tokens == {"destination": 700}


def test_plan_excludes_cross_version_pd_state_stale_and_incomplete_destinations():
    stats = {
        "source": _metrics("source", kv_cache_usage=0.2),
        "valid": _metrics("valid", kv_cache_usage=0.3),
        "other-version": _metrics("other-version", kv_cache_usage=0.7, weight_version=8),
        "other-role": _metrics("other-role", kv_cache_usage=0.7, role="prefill"),
        "draining": _metrics("draining", kv_cache_usage=0.7, state="draining"),
        "stale": _metrics("stale", kv_cache_usage=0.7, timestamp=94.0),
        "incomplete": _metrics("incomplete", kv_cache_usage=0.7, num_running_reqs=None),
        "missing": None,
    }

    plan = plan_repack(stats, {"source", "unknown"}, 0.9, 8, now=100.0, max_age_s=5)

    assert [(move.source, move.destination) for move in plan.moves] == [("source", "valid")]
    assert all(move.weight_version == 7 and move.role == "decode" for move in plan.moves)


def test_released_or_incoming_sources_cannot_create_move_chains():
    stats = {
        "s1": _metrics("s1", kv_cache_usage=0.1),
        "s2": _metrics("s2", kv_cache_usage=0.2),
        "s3": _metrics("s3", kv_cache_usage=0.3),
    }

    plan = plan_repack(stats, {"s1", "s2", "s3"}, 0.9, 8, now=100.0)

    assert [(move.source, move.destination) for move in plan.moves] == [("s1", "s3"), ("s2", "s3")]
    assert plan.estimated_destination_requests == {"s3": 3}


@pytest.mark.asyncio
async def test_coordinator_collect_idle_plan_dry_run_has_no_execution_side_effects():
    wall = [100.0]
    calls = 0
    executed = []

    async def collect():
        nonlocal calls
        calls += 1
        source_kv = 0.5 if calls == 1 else 0.2
        return {
            "source": _metrics("source", kv_cache_usage=source_kv, timestamp=wall[0]),
            "destination": _metrics("destination", kv_cache_usage=0.4, timestamp=wall[0]),
        }

    async def execute(plan):
        executed.append(plan)

    coordinator = RepackCoordinatorCore(
        collect,
        enabled=True,
        migration_enabled=False,
        execute_plan=execute,
        roofline_batch_size=8,
        wall_clock=lambda: wall[0],
    )

    first = await coordinator.tick()
    wall[0] += 1.0
    second = await coordinator.tick()

    assert first.reason == "no_idle"
    assert second.status == "planned"
    assert second.reason == "dry_run"
    assert second.plan.dry_run
    assert [(move.source, move.destination) for move in second.plan.moves] == [("source", "destination")]
    assert executed == []
    metrics = coordinator.get_metrics()
    assert metrics.trigger_count == 2
    assert metrics.plan_count == 1
    assert metrics.planned_moves == 1
    assert metrics.collect_duration.count == 2
    assert metrics.plan_duration.count == 1
    assert metrics.migration_duration.count == 0
    assert metrics.tick_duration.count == 2


@pytest.mark.asyncio
async def test_coordinator_missing_metrics_and_state_conflicts_are_noops():
    wall = [100.0]
    collections = 0

    def collect():
        nonlocal collections
        collections += 1
        if collections == 1:
            return {"source": _metrics("source", kv_cache_usage=0.5, timestamp=wall[0])}
        return {"source": None}

    coordinator = RepackCoordinatorCore(collect, enabled=True, roofline_batch_size=8, wall_clock=lambda: wall[0])
    await coordinator.tick()
    result = await coordinator.tick()
    assert result.reason == "no_idle"
    assert coordinator.history.latest("source") is None

    blocked = RepackCoordinatorCore(
        lambda: pytest.fail("state-conflicted tick must not collect"),
        enabled=True,
        can_run=lambda: False,
    )
    assert (await blocked.tick()).reason == "state_conflict"
    assert blocked.get_metrics().skipped_state == 1

    syncing = RepackCoordinatorCore(
        lambda: pytest.fail("weight-syncing tick must not collect"),
        enabled=True,
        weight_syncing=lambda: True,
    )
    assert (await syncing.tick()).reason == "weight_sync"


@pytest.mark.asyncio
async def test_coordinator_single_flight_skips_overlap_instead_of_queueing():
    collecting = asyncio.Event()
    release = asyncio.Event()

    async def collect():
        collecting.set()
        await release.wait()
        return {}

    coordinator = RepackCoordinatorCore(collect, enabled=True, roofline_batch_size=8)
    first = asyncio.create_task(coordinator.tick())
    await collecting.wait()

    overlap = await coordinator.tick()
    assert overlap.reason == "overlap"
    assert not first.done()

    release.set()
    await first
    metrics = coordinator.get_metrics()
    assert metrics.trigger_count == 2
    assert metrics.skipped_overlap == 1
    assert metrics.collect_duration.count == 1


@pytest.mark.asyncio
async def test_failures_exponential_backoff_and_successful_recovery():
    monotonic = [0.0]
    wall = [100.0]
    calls = 0

    def collect():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary collector failure")
        source_kv = 0.5 if calls == 2 else 0.2
        return {
            "source": _metrics("source", kv_cache_usage=source_kv, timestamp=wall[0]),
            "destination": _metrics("destination", kv_cache_usage=0.4, timestamp=wall[0]),
        }

    coordinator = RepackCoordinatorCore(
        collect,
        enabled=True,
        roofline_batch_size=8,
        backoff_base_s=10,
        backoff_multiplier=2,
        wall_clock=lambda: wall[0],
        monotonic_clock=lambda: monotonic[0],
    )

    failed = await coordinator.tick()
    assert failed.status == "error"
    assert failed.reason == "collect_failed"
    assert "temporary collector failure" in failed.error
    assert coordinator.get_metrics().backoff_active

    assert (await coordinator.tick()).reason == "backoff"
    assert calls == 1
    monotonic[0] = 9.9
    assert (await coordinator.tick()).reason == "backoff"
    assert calls == 1

    monotonic[0] = 10.0
    assert (await coordinator.tick()).reason == "no_idle"
    assert calls == 2
    monotonic[0] = 20.0
    wall[0] = 101.0
    recovered = await coordinator.tick()
    assert recovered.status == "planned"
    metrics = coordinator.get_metrics()
    assert metrics.consecutive_failures == 0
    assert not metrics.backoff_active
    assert metrics.skipped_backoff == 2
    assert metrics.failure_count == 1


@pytest.mark.asyncio
async def test_migration_callback_success_failure_and_backoff():
    wall = [100.0]
    monotonic = [0.0]
    collections = 0
    executed = []

    def collect():
        nonlocal collections
        collections += 1
        source_kv = (0.5, 0.2, 0.1)[min(collections - 1, 2)]
        return {
            "source": _metrics("source", kv_cache_usage=source_kv, timestamp=wall[0]),
            "destination": _metrics("destination", kv_cache_usage=0.4, timestamp=wall[0]),
        }

    async def fail_execution(plan):
        assert not plan.dry_run
        raise RuntimeError("migration rejected")

    async def succeed_execution(plan):
        executed.append(plan)

    coordinator = RepackCoordinatorCore(
        collect,
        enabled=True,
        migration_enabled=True,
        execute_plan=fail_execution,
        roofline_batch_size=8,
        backoff_base_s=5,
        wall_clock=lambda: wall[0],
        monotonic_clock=lambda: monotonic[0],
    )
    await coordinator.tick()
    wall[0] += 1.0
    result = await coordinator.tick()

    assert result.reason == "migration_failed"
    metrics = coordinator.get_metrics()
    assert metrics.migration_attempts == 1
    assert metrics.migration_failures == 1
    assert metrics.migration_successes == 0
    assert metrics.migration_duration.count == 1
    assert (await coordinator.tick()).reason == "backoff"

    coordinator.execute_plan = succeed_execution
    monotonic[0] = 5.0
    wall[0] += 1.0
    recovered = await coordinator.tick()
    assert recovered.status == "migrated"
    assert executed == [recovered.plan]
    metrics = coordinator.get_metrics()
    assert metrics.migration_attempts == 2
    assert metrics.migration_successes == 1
    assert metrics.migration_moves == 1
    assert metrics.consecutive_failures == 0


@pytest.mark.asyncio
async def test_migration_enabled_without_callback_stays_dry_run():
    wall = [100.0]
    calls = 0

    def collect():
        nonlocal calls
        calls += 1
        return {
            "source": _metrics("source", kv_cache_usage=0.5 if calls == 1 else 0.2, timestamp=wall[0]),
            "destination": _metrics("destination", kv_cache_usage=0.4, timestamp=wall[0]),
        }

    coordinator = RepackCoordinatorCore(
        collect, enabled=True, migration_enabled=True, roofline_batch_size=8, wall_clock=lambda: wall[0]
    )
    await coordinator.tick()
    wall[0] += 1.0
    result = await coordinator.tick()

    assert result.reason == "dry_run"
    assert result.plan.dry_run
    assert coordinator.get_metrics().migration_attempts == 0


@pytest.mark.asyncio
async def test_persistent_start_stop_cancels_in_flight_tick_and_is_idempotent():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def collect():
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    coordinator = RepackCoordinatorCore(collect, enabled=True, interval_s=60)
    assert coordinator.task is None

    await coordinator.start()
    await coordinator.start()
    await started.wait()
    task = coordinator.task
    assert coordinator.running
    assert task.get_loop() is asyncio.get_running_loop()

    await asyncio.wait_for(coordinator.stop(), timeout=1)
    assert cancelled.is_set()
    assert task.cancelled()
    assert not coordinator.running
    assert coordinator.task is None
    await coordinator.stop()

    disabled = RepackCoordinatorCore(collect, enabled=False)
    await disabled.start()
    assert disabled.task is None
    assert (await disabled.tick()).reason == "disabled"


def test_parse_vllm_prometheus_snapshot_with_labels():
    snapshot = parse_prometheus_snapshot(
        """# HELP vllm:gpu_cache_usage_perc KV usage
vllm:gpu_cache_usage_perc{model_name=\"policy\"} 0.42
vllm:num_requests_running{model_name=\"policy\"} 3
vllm:num_requests_waiting{model_name=\"policy\"} 1
""",
        server_id="s0",
        weight_version=4,
        role="decode",
        request_budget=16,
        timestamp=100.0,
    )
    assert snapshot.kv_cache_usage == 0.42
    assert snapshot.num_running_reqs == 3
    assert snapshot.num_waiting_reqs == 1
    assert snapshot.weight_version == 4


def test_parse_sglang_percent_snapshot_and_missing_metrics():
    text = "sglang:token_usage 25\nsglang:num_running_reqs 2\nsglang:num_queue_reqs 0\n"
    snapshot = parse_prometheus_snapshot(
        text,
        server_id="s1",
        weight_version=8,
        role="default",
        request_budget=8,
        timestamp=100.0,
    )
    assert snapshot.kv_cache_usage == 0.25
    assert (
        parse_prometheus_snapshot(
            "sglang:num_running_reqs 2\n",
            server_id="s1",
            weight_version=8,
            role="default",
            request_budget=8,
        )
        is None
    )


class _ImmediateAwaitable:
    def __init__(self, value=None):
        self.value = value

    def __await__(self):
        async def result():
            return self.value

        return result().__await__()


class _LocalActorHandle:
    def __init__(self, actor):
        self.actor = actor

    def __getattr__(self, name):
        def remote(*args, **kwargs):
            return _ImmediateAwaitable(getattr(self.actor, name)(*args, **kwargs))

        return type("RemoteMethod", (), {"remote": staticmethod(remote)})()


class _AbortServer:
    def __init__(self, result=None, error=None, on_abort=None):
        self.result = {"aborted": True} if result is None else result
        self.error = error
        self.on_abort = on_abort
        self.calls = []

    def abort_request(self, engine_request_id, **kwargs):
        self.calls.append((engine_request_id, kwargs))
        if self.on_abort is not None:
            self.on_abort()
        if self.error is not None:
            raise self.error
        return self.result


class _MetricsAbortServer(_AbortServer):
    def __init__(self, metrics_callback, **kwargs):
        super().__init__(**kwargs)
        self.metrics_callback = metrics_callback

    def get_metrics_snapshot(self):
        return self.metrics_callback()


def _migration_plan():
    return RepackPlan(
        moves=(
            RepackMove(
                source="source",
                destination="destination",
                weight_version=0,
                role="decode",
                num_requests=1,
                kv_cache_usage=0.2,
            ),
        ),
        dry_run=False,
    )


class _AttemptRaceLoadBalancer(GlobalRequestLoadBalancer):
    def list_inflight(self, server_id):
        records = super().list_inflight(server_id)
        # Win the CAS after the executor snapshots the old attempt.
        self._request_registry["req"]["attempt"] += 1
        return records


def _migration_router(max_inflight=None, router_cls=GlobalRequestLoadBalancer):
    router = router_cls(
        servers={"source": None, "destination": None},
        max_inflight_requests_per_replica=max_inflight,
        laminar_routing_enabled=True,
    )
    selected, _ = router.acquire_server("req", target_server_id="source", engine_request_id="engine-req")
    assert selected == "source"
    return router


@pytest.mark.asyncio
async def test_executor_reserves_same_version_destination_before_directed_abort():
    router = _migration_router()
    reserved = []

    def check_reservation():
        records = router.list_inflight("destination")
        assert len(records) == 1
        assert records[0]["attempt"] == 1
        assert records[0]["weight_version"] == 0
        assert router.get_inflight_count("destination") == 1
        reserved.append(True)

    server = _AbortServer(on_abort=check_reservation)
    executor = RepackPlanExecutor(
        {"source": _LocalActorHandle(server)},
        _LocalActorHandle(router),
        timeout_s=60,
        fallback_to_lb=False,
    )
    try:
        assert await executor(_migration_plan())
        assert reserved == [True]
        assert server.calls == [("engine-req", {"reset_prefix_cache": False, "reason": "migration"})]
        assert router.get_weight_versions()["state"]["source"] == "serving"
    finally:
        await executor.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("conflict", ["attempt", "version"])
async def test_executor_cas_conflict_skips_abort_and_restores_source_drain(conflict):
    router_cls = _AttemptRaceLoadBalancer if conflict == "attempt" else GlobalRequestLoadBalancer
    router = _migration_router(router_cls=router_cls)
    if conflict == "version":
        # Simulate a destination advancing before the source starts its drain.
        router._server_versions["destination"] = 1
    server = _AbortServer()
    executor = RepackPlanExecutor(
        {"source": _LocalActorHandle(server)},
        _LocalActorHandle(router),
        timeout_s=60,
        fallback_to_lb=False,
    )

    assert not await executor(_migration_plan())
    assert server.calls == []
    assert router.list_inflight("destination") == []
    assert router.get_weight_versions()["state"]["source"] == "serving"


@pytest.mark.asyncio
async def test_executor_full_destination_does_not_abort_without_fallback():
    router = _migration_router(max_inflight=1)
    router.acquire_server("other", target_server_id="destination", engine_request_id="engine-other")
    server = _AbortServer()
    executor = RepackPlanExecutor(
        {"source": _LocalActorHandle(server)},
        _LocalActorHandle(router),
        timeout_s=60,
        fallback_to_lb=False,
    )

    assert not await executor(_migration_plan())
    assert server.calls == []
    assert [record["request_id"] for record in router.list_inflight("source")] == ["req"]
    assert [record["request_id"] for record in router.list_inflight("destination")] == ["other"]
    assert router.get_weight_versions()["state"]["source"] == "serving"


@pytest.mark.asyncio
async def test_executor_abort_complete_race_cancels_reservation_without_losing_request():
    router = _migration_router()
    server = _AbortServer(
        result={"aborted": False},
        on_abort=lambda: router.release_server("source", request_id="req"),
    )
    executor = RepackPlanExecutor(
        {"source": _LocalActorHandle(server)},
        _LocalActorHandle(router),
        timeout_s=60,
        fallback_to_lb=False,
    )

    assert await executor(_migration_plan())
    assert router.list_inflight("source") == []
    assert router.list_inflight("destination") == []
    assert router.get_total_inflight() == 0
    assert router.get_weight_versions()["state"]["source"] == "serving"
    await executor.close()


@pytest.mark.asyncio
async def test_executor_abort_failure_cleans_reservation_and_restores_source_drain():
    router = _migration_router()
    server = _AbortServer(error=RuntimeError("abort RPC failed"))
    executor = RepackPlanExecutor(
        {"source": _LocalActorHandle(server)},
        _LocalActorHandle(router),
        timeout_s=60,
        fallback_to_lb=False,
    )

    with pytest.raises(RuntimeError, match="abort RPC failed"):
        await executor(_migration_plan())
    records = router.list_inflight("source")
    assert len(records) == 1
    assert records[0]["request_id"] == "req"
    assert records[0]["attempt"] == 0
    assert router.list_inflight("destination") == []
    assert router.get_inflight_count("destination") == 0
    assert router.get_weight_versions()["state"]["source"] == "serving"
    await executor.close()


@pytest.mark.asyncio
async def test_repack_actor_backoff_recovery_and_stop_cancel_cleanup_tasks():
    router = _migration_router()
    wall = [100.0]
    source_usage = iter((0.8, 0.2))
    source = _MetricsAbortServer(lambda: _metrics("source", kv_cache_usage=next(source_usage), timestamp=wall[0]))
    destination = _MetricsAbortServer(
        lambda: _metrics(
            "destination",
            kv_cache_usage=0.1,
            num_running_reqs=0,
            timestamp=wall[0],
        )
    )
    servers = {
        "source": _LocalActorHandle(source),
        "destination": _LocalActorHandle(destination),
    }
    get_versions = router.get_weight_versions
    fail_collection = True

    def flaky_get_versions():
        nonlocal fail_collection
        if fail_collection:
            fail_collection = False
            raise RuntimeError("stats temporarily unavailable")
        return get_versions()

    router.get_weight_versions = flaky_get_versions
    actor_cls = RepackCoordinatorActor.__ray_actor_class__
    coordinator = actor_cls(
        servers=servers,
        load_balancer=_LocalActorHandle(router),
        repack_config={
            "enable": True,
            "interval_s": 60,
            "kv_cache_threshold": 0.99,
            "migration": {"enable": True, "timeout_s": 60, "fallback_to_lb": False},
        },
        request_budget=8,
        stats_interval_s=1,
    )
    monotonic = [0.0]
    coordinator._core.monotonic_clock = lambda: monotonic[0]
    coordinator._core.wall_clock = lambda: wall[0]

    try:
        failed = await coordinator.tick()
        assert failed.status == "error"
        assert failed.reason == "collect_failed"
        assert "stats temporarily unavailable" in failed.error
        assert coordinator._core.get_metrics().backoff_active
        assert (await coordinator.tick()).reason == "backoff"

        monotonic[0] = 60.0
        assert (await coordinator.tick()).reason == "no_idle"
        wall[0] += 1.0
        migrated = await coordinator.tick()
        assert migrated.status == "migrated", migrated
        assert len(source.calls) == 1
        assert coordinator._core.get_metrics().consecutive_failures == 0
        cleanup_task = next(iter(coordinator._executor._cleanup_tasks))

        await coordinator.start()
        background_task = coordinator._core.task
        assert background_task is not None
        await coordinator.stop()

        assert background_task.cancelled()
        assert cleanup_task.cancelled()
        assert coordinator._core.task is None
        assert not coordinator._executor._cleanup_tasks
    finally:
        await coordinator.stop()
        router.cancel_migration("req")


class _FakeWeightManager:
    def __init__(self, publish_started=None, publish_release=None, pull_callback=None):
        self.publish_started = publish_started
        self.publish_release = publish_release
        self.pull_callback = pull_callback
        self.publish_calls = []
        self.pull_calls = []

    async def publish_weights(self, version):
        self.publish_calls.append(version)
        if self.publish_started is not None:
            self.publish_started.set()
        if self.publish_release is not None:
            await self.publish_release.wait()
        return {"latest_weight_version": version}

    async def pull_replica(self, replica, version):
        self.pull_calls.append((replica, version))
        if self.pull_callback is not None:
            await self.pull_callback(len(self.pull_calls))
        return {"version": version}


class _PublishLoadBalancer:
    def __init__(self):
        self.latest = 1
        self.calls = []

    def initialize_weight_registry(self, version):
        self.calls.append(("initialize", version))
        self.latest = version

    def set_latest_weight_version(self, version):
        self.calls.append(("latest", version))
        self.latest = version


class _DrainLoadBalancer(GlobalRequestLoadBalancer):
    def __init__(self, events):
        super().__init__(servers={"s0": None, "s1": None}, laminar_routing_enabled=True)
        self.events = events
        self.updated = {"s0": asyncio.Event(), "s1": asyncio.Event()}

    def mark_server_updating(self, server_id):
        self.events.append(f"mark:{server_id}")
        return super().mark_server_updating(server_id)

    def complete_server_update(self, server_id, version, success):
        super().complete_server_update(server_id, version, success)
        self.events.append(f"complete:{server_id}:{version}:{success}")
        if success:
            self.updated[server_id].set()


class _UpdateEventLoadBalancer(GlobalRequestLoadBalancer):
    def __init__(self):
        super().__init__(servers={"s0": None}, laminar_routing_enabled=True)
        self.failed_update = asyncio.Event()
        self.successful_update = asyncio.Event()

    def complete_server_update(self, server_id, version, success):
        super().complete_server_update(server_id, version, success)
        if version == 2:
            (self.successful_update if success else self.failed_update).set()


def _weight_coordinator(monkeypatch, manager, load_balancer, replicas=None, retry_interval_s=60):
    monkeypatch.setattr(
        weight_sync_coordinator,
        "CheckpointEngineManager",
        lambda **kwargs: manager,
    )
    actor_cls = weight_sync_coordinator.WeightUpdateCoordinatorActor.__ray_metadata__.modified_class
    return actor_cls(
        checkpoint_engine_config=object(),
        actor_wg=object(),
        replica_workers=replicas or {"s0": ["worker-s0"]},
        load_balancer=_LocalActorHandle(load_balancer),
        retry_interval_s=retry_interval_s,
    )


@pytest.mark.asyncio
async def test_weight_publish_barrier_precedes_latest_visibility(monkeypatch):
    publish_started = asyncio.Event()
    publish_release = asyncio.Event()
    manager = _FakeWeightManager(publish_started, publish_release)
    load_balancer = _PublishLoadBalancer()
    coordinator = _weight_coordinator(monkeypatch, manager, load_balancer)
    coordinator._initialized = True

    publish = asyncio.create_task(coordinator.publish(2))
    await asyncio.wait_for(publish_started.wait(), timeout=1)
    assert manager.publish_calls == [2]
    assert load_balancer.latest == 1
    assert load_balancer.calls == []

    publish_release.set()
    assert await asyncio.wait_for(publish, timeout=1) == {"latest_weight_version": 2}
    assert load_balancer.latest == 2
    assert load_balancer.calls == [("latest", 2)]


@pytest.mark.asyncio
async def test_weight_update_waits_for_drain_then_pulls_only_target_replica(monkeypatch):
    events = []
    load_balancer = _DrainLoadBalancer(events)
    selected, _ = load_balancer.acquire_server("busy", target_server_id="s0", engine_request_id="engine-busy")
    assert selected == "s0"
    load_balancer.initialize_weight_registry(4)

    async def record_pull(_call_count):
        events.append("pull")

    manager = _FakeWeightManager(pull_callback=record_pull)
    coordinator = _weight_coordinator(
        monkeypatch,
        manager,
        load_balancer,
        replicas={"s0": ["worker-s0"], "s1": ["worker-s1"]},
        retry_interval_s=0.01,
    )
    await coordinator.start()
    try:
        await asyncio.wait_for(load_balancer.updated["s1"].wait(), timeout=1)
        versions = load_balancer.get_weight_versions()
        assert versions["local"]["s1"] == 4
        assert versions["state"]["s1"] == "serving"
        assert versions["local"]["s0"] == -1
        assert versions["state"]["s0"] == "draining"
        assert len(manager.pull_calls) == 1

        load_balancer.release_server("s0", request_id="busy")
        await asyncio.wait_for(load_balancer.updated["s0"].wait(), timeout=1)
    finally:
        await coordinator.stop()

    pulls = [(replica.workers, version) for replica, version in manager.pull_calls]
    assert pulls == [(["worker-s1"], 4), (["worker-s0"], 4)]
    assert events.index("mark:s1") < events.index("pull") < events.index("complete:s1:4:True")
    assert load_balancer.get_weight_versions()["state"] == {"s0": "serving", "s1": "serving"}
    metrics = coordinator.get_metrics()
    assert metrics["update_duration_count"] == 2
    assert metrics["update_duration_seconds"] >= 0
    assert metrics["drain_wait_count"] == 2
    assert metrics["drain_wait_seconds"] >= 0
    assert metrics["update_failures"] == 0


@pytest.mark.asyncio
async def test_weight_update_failure_keeps_old_version_and_retries(monkeypatch):
    load_balancer = _UpdateEventLoadBalancer()
    load_balancer.initialize_weight_registry(1)
    version = load_balancer.mark_server_updating("s0")
    load_balancer.complete_server_update("s0", version, success=True)
    load_balancer.set_latest_weight_version(2)

    async def fail_first_pull(call_count):
        if call_count == 1:
            raise RuntimeError("temporary pull failure")

    manager = _FakeWeightManager(pull_callback=fail_first_pull)
    coordinator = _weight_coordinator(monkeypatch, manager, load_balancer)
    await coordinator.start()
    try:
        await asyncio.wait_for(load_balancer.failed_update.wait(), timeout=1)
        versions = load_balancer.get_weight_versions()
        assert versions["local"]["s0"] == 1
        assert versions["latest"] == 2
        assert versions["state"]["s0"] == "draining"
        assert len(manager.pull_calls) == 1

        coordinator._wake.set()
        await asyncio.wait_for(load_balancer.successful_update.wait(), timeout=1)
        versions = load_balancer.get_weight_versions()
        assert versions["local"]["s0"] == 2
        assert versions["state"]["s0"] == "serving"
        assert [version for _replica, version in manager.pull_calls] == [2, 2]
        metrics = coordinator.get_metrics()
        assert metrics["update_failures"] == 1
        assert metrics["update_duration_count"] == 1
        assert metrics["update_duration_seconds"] >= 0
        assert metrics["drain_wait_count"] == 2
        assert metrics["drain_wait_seconds"] >= 0
    finally:
        await coordinator.stop()
