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

"""Versioned, immutable weight snapshots for checkpoint-engine integrations.

The store deliberately has no dependency on a particular transport.  A NIXL or
KIMI adapter can use :meth:`WeightSnapshotStore.acquire` to pin a version for the
whole transfer and inspect its tensor metadata before moving any bytes.
"""

import hashlib
import threading
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType

import torch


@dataclass(frozen=True)
class TensorMetadata:
    """Transport-friendly metadata for one tensor in a committed snapshot.

    ``checksum`` is the hexadecimal SHA-256 of the tensor's logical values in
    contiguous row-major byte representation.  Shape and dtype are kept
    separately, so receivers should validate all three before installing data.
    """

    name: str
    shape: torch.Size
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    numel: int
    nbytes: int
    checksum: str


@dataclass
class _VersionEntry:
    version: int
    weights: Mapping[str, torch.Tensor]
    metadata: Mapping[str, TensorMetadata]
    checksums: Mapping[str, str]
    references: int = 0


def _validate_positive_integer(value: int, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{description} must be a positive integer, got {value!r}")
    return int(value)


def _validate_version(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"version must be a non-negative integer, got {value!r}")
    return int(value)


def _checksum_tensor(tensor: torch.Tensor) -> str:
    """Hash all logical tensor bytes without materializing a Python bytes copy."""

    cpu_tensor = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    if cpu_tensor.numel() == 0:
        return digest.hexdigest()

    byte_tensor = cpu_tensor.view(torch.uint8)
    digest.update(memoryview(byte_tensor.numpy()).cast("B"))
    return digest.hexdigest()


class WeightSnapshot:
    """A leased, read-only view of one committed version.

    Iteration yields ``(name, tensor)`` pairs in publication order.  Each tensor
    is cloned on iteration: PyTorch has no general read-only tensor flag, and a
    clone prevents an inference adapter's in-place update from corrupting the
    cached snapshot.  This costs one temporary tensor at a time, while the
    original cached tensors remain stable and privately owned by the store.

    Call :meth:`release` when the transfer finishes, or use the handle as a
    context manager.  Releasing a handle while one of its iterators is active
    delays the store's reference release until that iterator finishes.
    """

    def __init__(self, store: "WeightSnapshotStore", entry: _VersionEntry) -> None:
        self._store = store
        self._entry = entry
        self._lock = threading.Lock()
        self._active_iterators = 0
        self._released = False
        self._store_reference_held = True

    @property
    def version(self) -> int:
        """The positive integer identifying this snapshot."""

        return self._entry.version

    @property
    def metadata(self) -> Mapping[str, TensorMetadata]:
        """An immutable name-to-metadata mapping, safe to inspect after release."""

        return self._entry.metadata

    @property
    def checksums(self) -> Mapping[str, str]:
        """An immutable name-to-SHA-256 mapping, safe to inspect after release."""

        return self._entry.checksums

    @property
    def names(self) -> tuple[str, ...]:
        """Tensor names in publication order."""

        return tuple(self._entry.weights)

    @property
    def released(self) -> bool:
        """Whether no new iterator can be started from this handle."""

        with self._lock:
            return self._released

    def checksum(self, name: str) -> str:
        """Return one tensor's SHA-256, raising ``KeyError`` for an unknown name."""

        return self._entry.checksums[name]

    def _take_store_reference_locked(self) -> bool:
        if self._released and self._active_iterators == 0 and self._store_reference_held:
            self._store_reference_held = False
            return True
        return False

    def release(self) -> None:
        """Release this lease; repeated calls are harmless.

        Cached storage becomes eligible for retention eviction only after this
        call and after all iterators already created from the handle finish.
        Metadata and checksums remain available on the handle itself.
        """

        with self._lock:
            if self._released:
                return
            self._released = True
            release_store_reference = self._take_store_reference_locked()

        if release_store_reference:
            self._store._release(self._entry)

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor]]:
        with self._lock:
            if self._released:
                raise RuntimeError(f"snapshot version {self.version} has been released")
            self._active_iterators += 1

        try:
            for name, tensor in self._entry.weights.items():
                yield name, tensor.clone(memory_format=torch.preserve_format)
        finally:
            with self._lock:
                self._active_iterators -= 1
                release_store_reference = self._take_store_reference_locked()

            if release_store_reference:
                self._store._release(self._entry)

    def items(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Return an iterator over defensive tensor copies."""

        return iter(self)

    def __len__(self) -> int:
        return len(self._entry.weights)

    def __enter__(self) -> "WeightSnapshot":
        with self._lock:
            if self._released:
                raise RuntimeError(f"snapshot version {self.version} has been released")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            # Explicit release/context management is deterministic; finalization
            # is only a last-resort safeguard during interpreter shutdown.
            pass


class WeightSnapshotStore:
    """Thread-safe staging, publication, leasing, and retention of weight versions.

    Versions are positive integers and are never overwritten.  ``publish`` is
    idempotent for a version that already committed, and concurrent publishers
    of the same new version are serialized so only one consumes its ``weights``
    iterable.  A failed export leaves no visible state and the version can be
    retried.

    Every source tensor is detached and cloned into private staging storage,
    then checksummed in full.  Only after the entire iterable succeeds are the
    immutable mapping, metadata, version index, and ``latest_version`` committed
    under one lock.  Consequently readers never observe a partial version or
    later in-place changes to training tensors.

    Args:
        cache_versions: Number of numerically newest committed versions to keep.
            Must be at least two.  Older versions with active leases are pinned;
            they are evicted on release once outside the retention window.

    Notes:
        ``latest_version`` is the greatest committed version, not merely the last
        publish call.  A small set of historical version integers is retained so
        even an evicted version is never exported a second time.
    """

    def __init__(self, cache_versions: int = 2) -> None:
        self.cache_versions = _validate_positive_integer(cache_versions, "cache_versions")
        if self.cache_versions < 2:
            raise ValueError(f"cache_versions must be at least 2, got {cache_versions}")

        self._condition = threading.Condition(threading.RLock())
        self._entries: dict[int, _VersionEntry] = {}
        self._publishing: set[int] = set()
        self._completed_versions: set[int] = set()
        self._latest_version: int | None = None

    @property
    def latest_version(self) -> int | None:
        """The greatest fully committed version, or ``None`` before the first commit."""

        with self._condition:
            return self._latest_version

    @property
    def available_versions(self) -> tuple[int, ...]:
        """Currently cached versions in ascending order, including pinned old versions."""

        with self._condition:
            return tuple(sorted(self._entries))

    @staticmethod
    def _stage_weights(
        weights: Mapping[str, torch.Tensor] | Iterable[tuple[str, torch.Tensor]],
    ) -> Mapping[str, torch.Tensor]:
        source = weights.items() if isinstance(weights, Mapping) else weights
        staged: dict[str, torch.Tensor] = {}

        for item in source:
            try:
                name, tensor = item
            except (TypeError, ValueError) as exc:
                raise ValueError("weights must contain (name, tensor) pairs") from exc

            if not isinstance(name, str) or not name:
                raise ValueError(f"weight names must be non-empty strings, got {name!r}")
            if name in staged:
                raise ValueError(f"duplicate weight name {name!r}")
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"weight {name!r} must be a torch.Tensor, got {type(tensor).__name__}")
            if tensor.layout != torch.strided:
                raise ValueError(f"weight {name!r} must use torch.strided layout, got {tensor.layout}")

            # Resolve lazy conjugate/negative bits before materializing bytes.
            # The private clone also prevents subsequent trainer mutation from
            # changing a published version.
            staged_tensor = tensor.detach().resolve_conj().resolve_neg().clone(memory_format=torch.preserve_format)
            staged[name] = staged_tensor

        return MappingProxyType(staged)

    @staticmethod
    def _build_metadata(weights: Mapping[str, torch.Tensor]) -> Mapping[str, TensorMetadata]:
        metadata = {
            name: TensorMetadata(
                name=name,
                shape=tensor.shape,
                stride=tuple(tensor.stride()),
                dtype=tensor.dtype,
                device=tensor.device,
                numel=tensor.numel(),
                nbytes=tensor.numel() * tensor.element_size(),
                checksum=_checksum_tensor(tensor),
            )
            for name, tensor in weights.items()
        }
        return MappingProxyType(metadata)

    def publish(
        self,
        version: int,
        weights: Mapping[str, torch.Tensor] | Iterable[tuple[str, torch.Tensor]],
    ) -> None:
        """Stage and atomically publish all weights for ``version``.

        ``weights`` can be a name-to-tensor mapping or any iterable of
        ``(name, tensor)`` pairs, including a single-use generator.  If this
        version already committed, this method returns without touching
        ``weights``.  Duplicate names, export errors, and checksum errors abort
        the whole staging attempt and leave the version retryable.
        """

        version = _validate_version(version)
        with self._condition:
            while version in self._publishing:
                self._condition.wait()

            if version in self._completed_versions:
                return
            self._publishing.add(version)

        try:
            # Nothing in this local staging area is visible to readers.  Compute
            # every checksum before taking the commit lock.
            staged_weights = self._stage_weights(weights)
            metadata = self._build_metadata(staged_weights)
            entry = _VersionEntry(
                version=version,
                weights=staged_weights,
                metadata=metadata,
                checksums=MappingProxyType({name: meta.checksum for name, meta in metadata.items()}),
            )

            with self._condition:
                self._entries[version] = entry
                self._completed_versions.add(version)
                if self._latest_version is None or version > self._latest_version:
                    self._latest_version = version
                self._evict_locked()
        finally:
            with self._condition:
                self._publishing.remove(version)
                self._condition.notify_all()

    def _entry_locked(self, version: int | None) -> _VersionEntry:
        if version is None:
            if self._latest_version is None:
                raise KeyError("no weight snapshot has been committed")
            version = self._latest_version
        else:
            version = _validate_version(version)

        try:
            return self._entries[version]
        except KeyError as exc:
            raise KeyError(f"weight snapshot version {version} is not available") from exc

    def acquire(self, version: int | None = None) -> WeightSnapshot:
        """Pin and return a snapshot handle; ``None`` selects the latest version.

        Raises:
            KeyError: If no version has committed or the requested version was
                never committed, is still staging, or has already been evicted.
            ValueError: If an explicitly supplied version is not a positive int.
        """

        with self._condition:
            entry = self._entry_locked(version)
            entry.references += 1
            return WeightSnapshot(self, entry)

    def get(self, version: int | None = None) -> WeightSnapshot:
        """Alias for :meth:`acquire`, for checkpoint-engine read paths."""

        return self.acquire(version)

    def get_metadata(self, version: int | None = None) -> Mapping[str, TensorMetadata]:
        """Return immutable tensor metadata without acquiring a read lease."""

        with self._condition:
            return self._entry_locked(version).metadata

    def get_checksum(self, version: int | None, name: str) -> str:
        """Return one committed tensor checksum without acquiring a read lease."""

        with self._condition:
            return self._entry_locked(version).checksums[name]

    def reference_count(self, version: int) -> int:
        """Return the number of active leases for an available version."""

        with self._condition:
            return self._entry_locked(version).references

    def _release(self, entry: _VersionEntry) -> None:
        with self._condition:
            entry.references -= 1
            self._evict_locked()

    def _evict_locked(self) -> None:
        # Keep the newest N plus any older versions still leased.  On the final
        # release this same calculation removes newly unpinned old versions.
        retained = set(sorted(self._entries)[-self.cache_versions :])
        for version in sorted(self._entries):
            entry = self._entries[version]
            if version not in retained and entry.references == 0:
                del self._entries[version]


__all__ = ["TensorMetadata", "WeightSnapshot", "WeightSnapshotStore"]
