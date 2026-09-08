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
"""Store-based publish/pull checkpoint engine for Laminar-style trajectory-level asynchrony.

https://arxiv.org/abs/2510.12633

Weight-service semantics (no global weight-sync point):

- **Publish** (trainer side): each actor rank bucket-serializes the parameters it yields
  under version-immutable keys ``{prefix}/v{version}/rank{r}/bucket_{i}`` plus a per-rank
  manifest. After *all* actor ranks finish (a ``ray.get`` barrier in the trainer driver),
  the driver writes the version manifest and bumps the ``{prefix}/latest`` pointer --
  that pointer bump is the atomic commit point of a generation.
- **Pull** (rollout side): any replica may call ``receive_weights`` at any time after the
  commit. It pins a version (explicit ``global_steps`` or the current ``latest``), reads
  the manifest, fetches every rank's buckets, and yields ``(name, tensor)`` pairs to its
  server adapter. Pulls of different replicas are fully independent.
- **GC**: because a pull always targets the latest version at its start and a new
  generation is published at most once per trainer step, keeping the newest
  ``keep_versions`` (default 2) generations is exactly safe: a pull of ``vK`` can
  overlap with the publish of ``vK+1``, and generations ``<= vK-1`` are unreachable.

Two store clients are provided:

- ``MooncakeWeightStore``: production client backed by ``mooncake.store.MooncakeDistributedStore``
  (host-DRAM KV store with RDMA transport; deploy one store instance per rollout node so
  pulls are served from local/nearby DRAM, mirroring Laminar's per-machine relay workers).
- ``InMemoryWeightStore``: process-global dict for CPU tests and single-process debug only
  (trainer driver and Ray workers live in different processes on a real cluster).
"""

import asyncio
import io
import json
import logging
import os
import threading
import time
from typing import Any, AsyncGenerator, Generator, Protocol

import torch

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

DEFAULT_KEY_PREFIX = "verl/weights"
DEFAULT_KEEP_VERSIONS = 2


# ---------------------------------------------------------------------------
# Weight store clients
# ---------------------------------------------------------------------------


class WeightStoreClient(Protocol):
    """Minimal byte-KV interface required by the weight service."""

    def put(self, key: str, data: bytes) -> None: ...

    def get(self, key: str) -> bytes | None: ...

    def remove(self, key: str) -> None: ...


class InMemoryWeightStore:
    """Process-global in-memory store. For CPU tests / single-process debug only."""

    _global_lock = threading.Lock()
    _global_data: dict[str, bytes] = {}

    def __init__(self, **kwargs) -> None:
        pass

    def put(self, key: str, data: bytes) -> None:
        with self._global_lock:
            self._global_data[key] = bytes(data)

    def get(self, key: str) -> bytes | None:
        with self._global_lock:
            return self._global_data.get(key)

    def remove(self, key: str) -> None:
        with self._global_lock:
            self._global_data.pop(key, None)


class MooncakeWeightStore:
    """Production client backed by mooncake.store.MooncakeDistributedStore.

    kwargs mirror the ``transfer_queue.backend.MooncakeStore`` config section:
    ``metadata_server``, ``master_server_address``, ``local_hostname``, ``protocol``
    (tcp/rdma), ``global_segment_size``, ``local_buffer_size``, ``device_name``.
    """

    def __init__(
        self,
        metadata_server: str = "localhost:50123",
        master_server_address: str = "localhost:50124",
        local_hostname: str = "localhost",
        protocol: str = "tcp",
        global_segment_size: int = 1 << 32,
        local_buffer_size: int = 1 << 30,
        device_name: str = "",
        **kwargs,
    ) -> None:
        try:
            from mooncake.store import MooncakeDistributedStore
        except ImportError as e:
            raise ImportError(
                "mooncake is required for MooncakeWeightStore; install it or use the 'memory' store backend for tests."
            ) from e
        self._store = MooncakeDistributedStore()
        ret = self._store.setup(
            local_hostname,
            metadata_server,
            global_segment_size,
            local_buffer_size,
            protocol,
            device_name,
            master_server_address,
        )
        if ret != 0:
            raise RuntimeError(f"MooncakeDistributedStore.setup failed with code {ret}")

    def put(self, key: str, data: bytes) -> None:
        ret = self._store.put(key, data)
        if ret != 0:
            raise RuntimeError(f"MooncakeDistributedStore.put({key}) failed with code {ret}")

    def get(self, key: str) -> bytes | None:
        if self._store.is_exist(key) <= 0:
            return None
        data = self._store.get(key)
        return bytes(data) if data is not None else None

    def remove(self, key: str) -> None:
        self._store.remove(key)


_STORE_BACKENDS = {
    "memory": InMemoryWeightStore,
    "mooncake": MooncakeWeightStore,
}

# Per-process client cache: a Mooncake store connection is expensive to set up
# (segment registration), so all engine instances in one process share one client.
_clients_lock = threading.Lock()
_clients: dict[tuple, WeightStoreClient] = {}


def get_or_create_client(store_backend: str, store_kwargs: dict) -> WeightStoreClient:
    """Get (or lazily create) the per-process shared store client for this backend/config."""
    cache_key = (store_backend, tuple(sorted((k, repr(v)) for k, v in store_kwargs.items())))
    with _clients_lock:
        if cache_key not in _clients:
            if store_backend not in _STORE_BACKENDS:
                raise ValueError(
                    f"Unknown weight store backend {store_backend!r}; available: {sorted(_STORE_BACKENDS)}"
                )
            _clients[cache_key] = _STORE_BACKENDS[store_backend](**store_kwargs)
        return _clients[cache_key]


def reset_clients_for_test() -> None:
    """Drop all cached clients (test isolation)."""
    with _clients_lock:
        _clients.clear()
    with InMemoryWeightStore._global_lock:
        InMemoryWeightStore._global_data.clear()


# ---------------------------------------------------------------------------
# Key layout + version protocol (shared by engine instances and the trainer driver)
# ---------------------------------------------------------------------------


def _bucket_key(key_prefix: str, version: int, rank: int, index: int) -> str:
    return f"{key_prefix}/v{version}/rank{rank}/bucket_{index:06d}"


def _rank_manifest_key(key_prefix: str, version: int, rank: int) -> str:
    return f"{key_prefix}/v{version}/rank{rank}/manifest"


def _version_manifest_key(key_prefix: str, version: int) -> str:
    return f"{key_prefix}/v{version}/manifest"


def _latest_key(key_prefix: str) -> str:
    return f"{key_prefix}/latest"


def read_latest_version(client: WeightStoreClient, key_prefix: str) -> int | None:
    """Read the latest committed weight version, or None if nothing was published yet."""
    data = client.get(_latest_key(key_prefix))
    return int(data.decode()) if data is not None else None


def read_version_manifest(client: WeightStoreClient, key_prefix: str, version: int) -> dict | None:
    data = client.get(_version_manifest_key(key_prefix, version))
    return json.loads(data.decode()) if data is not None else None


def commit_version(
    client: WeightStoreClient,
    key_prefix: str,
    version: int,
    num_ranks: int,
    keep_versions: int = DEFAULT_KEEP_VERSIONS,
) -> dict:
    """Commit a published generation and garbage-collect unreachable ones.

    Must be called exactly once per generation, after *all* actor ranks finished writing
    their buckets (the trainer driver enforces this via the ``ray.get`` barrier on
    ``actor_wg.update_weights``). The ``latest`` pointer bump is the atomic commit point.

    Returns:
        GC metrics: number of deleted generations and keys.
    """
    client.put(
        _version_manifest_key(key_prefix, version),
        json.dumps({"version": version, "num_ranks": num_ranks, "commit_time": time.time()}).encode(),
    )
    # Atomic commit: readers that started before this point pinned an older version whose
    # keys are immutable, so the pointer bump can never tear a read.
    client.put(_latest_key(key_prefix), str(version).encode())
    return gc_old_versions(client, key_prefix, latest=version, keep_versions=keep_versions)


def gc_old_versions(client: WeightStoreClient, key_prefix: str, latest: int, keep_versions: int) -> dict:
    """Delete generations older than the newest ``keep_versions`` ones.

    Key enumeration is manifest-driven (no prefix-scan API in Mooncake): the version
    manifest gives ``num_ranks`` and each rank manifest gives ``num_buckets``.
    """
    deleted_generations, deleted_keys = 0, 0
    cutoff = latest - keep_versions + 1
    for version in range(0, cutoff):
        manifest = read_version_manifest(client, key_prefix, version)
        if manifest is None:
            continue
        for rank in range(manifest["num_ranks"]):
            rank_data = client.get(_rank_manifest_key(key_prefix, version, rank))
            if rank_data is not None:
                num_buckets = json.loads(rank_data.decode())["num_buckets"]
                for i in range(num_buckets):
                    client.remove(_bucket_key(key_prefix, version, rank, i))
                    deleted_keys += 1
            client.remove(_rank_manifest_key(key_prefix, version, rank))
            deleted_keys += 1
        client.remove(_version_manifest_key(key_prefix, version))
        deleted_keys += 1
        deleted_generations += 1
    if deleted_generations:
        logger.info(f"[StoreCE] GC deleted {deleted_generations} generations ({deleted_keys} keys), latest=v{latest}")
    return {"store/gc_deleted_generations": deleted_generations, "store/gc_deleted_keys": deleted_keys}


# ---------------------------------------------------------------------------
# CheckpointEngine implementation
# ---------------------------------------------------------------------------


@CheckpointEngineRegistry.register("mooncake_store")
class StoreCheckpointEngine(CheckpointEngine):
    """Publish/pull checkpoint engine over a byte-KV weight store (Laminar parameter service).

    There is no communication topology between actor and rollout workers: actor ranks only
    *write* disjoint parameter buckets to the store (``rank`` namespaces their keys), and
    every rollout checkpoint-engine worker independently *reads* the full model for its
    server adapter. ``build_process_group`` is therefore a one-time rank assignment on the
    actor worker group, driven by ``CheckpointEngineManager.publish_weights``.

    engine_kwargs (under ``actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.mooncake_store``):
        keep_versions (int): generations to retain, default 2 (exactly safe, see module docstring).
        key_prefix (str): key namespace, default ``verl/weights``.
        store_backend (str): ``mooncake`` (production) or ``memory`` (tests), default ``mooncake``.
        ...: remaining keys are forwarded to the store client constructor
             (see :class:`MooncakeWeightStore`).
    """

    def __init__(
        self,
        bucket_size: int,
        keep_versions: int = DEFAULT_KEEP_VERSIONS,
        key_prefix: str = DEFAULT_KEY_PREFIX,
        store_backend: str = "mooncake",
        **store_kwargs,
    ) -> None:
        self.bucket_size = bucket_size
        self.keep_versions = keep_versions
        self.key_prefix = key_prefix
        self.store_backend = store_backend
        self.store_kwargs = store_kwargs
        self._rank = 0

    def _client(self) -> WeightStoreClient:
        return get_or_create_client(self.store_backend, self.store_kwargs)

    # ---------------- topology (trivial: rank assignment only) ----------------

    def prepare(self) -> dict[str, Any]:
        return {}

    @classmethod
    def build_topology(
        cls, actor_wg_world_size: int, rollout_world_size: int, metadata: list[dict]
    ) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        # Actor ranks write disjoint key namespaces; rollout workers need no rank.
        return {"rank": list(range(actor_wg_world_size))}, {}

    def init_process_group(self, rank: int = 0, **kwargs) -> None:
        self._rank = rank
        # Eagerly connect so a misconfigured store fails at setup, not at first publish.
        self._client()

    def finalize(self) -> None:
        pass

    # ---------------- publish path (actor side) ----------------

    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ) -> dict:
        """Bucket-serialize this rank's parameters into the store under version-immutable keys."""
        assert global_steps is not None, "StoreCheckpointEngine.send_weights requires global_steps as version"
        client = self._client()
        num_buckets, total_bytes = 0, 0
        bucket: list[tuple[str, torch.Tensor]] = []
        bucket_bytes = 0

        async def flush(index: int) -> None:
            nonlocal total_bytes
            payload = await asyncio.to_thread(_serialize_bucket, bucket)
            await asyncio.to_thread(client.put, _bucket_key(self.key_prefix, global_steps, self._rank, index), payload)
            total_bytes += len(payload)

        for name, tensor in weights:
            tensor = tensor.detach().to(device="cpu", copy=True)
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            tensor_bytes = tensor.numel() * tensor.element_size()
            if bucket and bucket_bytes + tensor_bytes > self.bucket_size:
                await flush(num_buckets)
                num_buckets += 1
                bucket, bucket_bytes = [], 0
            bucket.append((name, tensor))
            bucket_bytes += tensor_bytes
        if bucket:
            await flush(num_buckets)
            num_buckets += 1

        await asyncio.to_thread(
            client.put,
            _rank_manifest_key(self.key_prefix, global_steps, self._rank),
            json.dumps({"num_buckets": num_buckets}).encode(),
        )
        # NOTE: the version manifest + `latest` pointer are committed by the trainer driver
        # (CheckpointEngineManager.publish_weights) after the all-ranks barrier.
        return {
            f"store/rank{self._rank}_bytes_written": total_bytes,
            f"store/rank{self._rank}_num_buckets": num_buckets,
        }

    # ---------------- pull path (rollout side) ----------------

    async def receive_weights(
        self,
        global_steps: int | None = None,
    ) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        """Yield the full model of the pinned (or latest committed) version from the store."""
        client = self._client()
        version = global_steps if global_steps is not None else read_latest_version(client, self.key_prefix)
        assert version is not None, "StoreCheckpointEngine.receive_weights: no committed weight version in store"
        manifest = read_version_manifest(client, self.key_prefix, version)
        assert manifest is not None, f"weight version v{version} not found (already GC'd or never published)"

        for rank in range(manifest["num_ranks"]):
            rank_data = await asyncio.to_thread(client.get, _rank_manifest_key(self.key_prefix, version, rank))
            assert rank_data is not None, f"rank manifest missing for v{version}/rank{rank}"
            num_buckets = json.loads(rank_data.decode())["num_buckets"]
            for index in range(num_buckets):
                payload = await asyncio.to_thread(client.get, _bucket_key(self.key_prefix, version, rank, index))
                assert payload is not None, f"weight bucket missing for v{version}/rank{rank}/bucket_{index}"
                bucket = await asyncio.to_thread(_deserialize_bucket, payload)
                for name, tensor in bucket:
                    yield name, tensor


def _serialize_bucket(bucket: list[tuple[str, torch.Tensor]]) -> bytes:
    buffer = io.BytesIO()
    torch.save(bucket, buffer)
    return buffer.getvalue()


def _deserialize_bucket(payload: bytes) -> list[tuple[str, torch.Tensor]]:
    return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
