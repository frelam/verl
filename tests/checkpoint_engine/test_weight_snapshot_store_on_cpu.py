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

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from verl.checkpoint_engine.weight_snapshot_store import WeightSnapshotStore


def _read(snapshot):
    return [(name, tensor.clone()) for name, tensor in snapshot]


def test_publish_is_atomic_and_defensively_isolated():
    store = WeightSnapshotStore(cache_versions=2)
    original = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    source = {"weight": original, "bias": torch.tensor([4, 5])}

    store.publish(7, source)
    original.fill_(-100)
    source["bias"] = torch.tensor([99, 99])

    assert store.latest_version == 7
    assert store.available_versions == (7,)
    with store.get() as snapshot:
        assert snapshot.version == 7
        assert snapshot.names == ("weight", "bias")
        assert len(snapshot) == 2

        first_read = dict(_read(snapshot))
        torch.testing.assert_close(first_read["weight"], torch.arange(6, dtype=torch.float32).reshape(2, 3))
        torch.testing.assert_close(first_read["bias"], torch.tensor([4, 5]))

        # A consumer can safely use in-place installation logic on its copy.
        first_read["weight"].fill_(123)
        second_read = dict(snapshot.items())
        torch.testing.assert_close(second_read["weight"], torch.arange(6, dtype=torch.float32).reshape(2, 3))


def test_duplicate_publish_does_not_consume_weights_or_overwrite():
    store = WeightSnapshotStore()
    store.publish(1, iter([("weight", torch.tensor([1, 2, 3]))]))

    def forbidden_export():
        raise AssertionError("a committed version must not be exported again")
        yield

    store.publish(1, forbidden_export())
    store.publish(1, {"weight": torch.tensor([9, 9, 9])})

    with store.acquire(1) as snapshot:
        torch.testing.assert_close(dict(_read(snapshot))["weight"], torch.tensor([1, 2, 3]))


def test_failed_publish_is_invisible_and_version_can_be_retried():
    store = WeightSnapshotStore()
    store.publish(1, {"stable": torch.tensor([1])})

    def broken_export():
        yield "partial", torch.tensor([2])
        raise RuntimeError("export failed")

    with pytest.raises(RuntimeError, match="export failed"):
        store.publish(2, broken_export())

    assert store.latest_version == 1
    assert store.available_versions == (1,)
    with pytest.raises(KeyError, match="version 2"):
        store.get(2)

    store.publish(2, {"complete": torch.tensor([3])})
    assert store.latest_version == 2
    with store.get(2) as snapshot:
        assert snapshot.names == ("complete",)
        torch.testing.assert_close(dict(_read(snapshot))["complete"], torch.tensor([3]))


def test_checksum_and_immutable_metadata_describe_logical_tensor_bytes():
    store = WeightSnapshotStore()
    tensor = torch.arange(6, dtype=torch.int16).reshape(2, 3).transpose(0, 1)
    store.publish(3, {"weight": tensor, "empty": torch.empty(0, dtype=torch.float32)})

    expected = hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()
    empty_expected = hashlib.sha256(b"").hexdigest()
    metadata = store.get_metadata(3)

    assert store.get_checksum(3, "weight") == expected
    assert store.get_checksum(None, "weight") == expected
    assert metadata["weight"].checksum == expected
    assert metadata["weight"].shape == torch.Size([3, 2])
    assert metadata["weight"].dtype == torch.int16
    assert metadata["weight"].device == torch.device("cpu")
    assert metadata["weight"].numel == 6
    assert metadata["weight"].nbytes == 12
    assert metadata["empty"].checksum == empty_expected

    with pytest.raises(TypeError):
        metadata["new"] = metadata["weight"]

    with store.get(3) as snapshot:
        assert snapshot.checksum("weight") == expected
        assert snapshot.checksums["empty"] == empty_expected
        with pytest.raises(TypeError):
            snapshot.checksums["new"] = expected


def test_context_manager_release_is_idempotent_and_blocks_new_reads():
    store = WeightSnapshotStore()
    store.publish(1, {"weight": torch.tensor([1])})

    with store.acquire() as snapshot:
        assert not snapshot.released
        assert store.reference_count(1) == 1
        assert len(_read(snapshot)) == 1

    assert snapshot.released
    assert store.reference_count(1) == 0
    snapshot.release()
    assert store.reference_count(1) == 0
    with pytest.raises(RuntimeError, match="released"):
        next(iter(snapshot))


def test_release_with_active_iterator_keeps_version_pinned_until_exhausted():
    store = WeightSnapshotStore(cache_versions=2)
    store.publish(1, {"first": torch.tensor([1]), "second": torch.tensor([2])})
    snapshot = store.get(1)
    iterator = iter(snapshot)

    assert next(iterator)[0] == "first"
    snapshot.release()
    store.publish(2, {"weight": torch.tensor([2])})
    store.publish(3, {"weight": torch.tensor([3])})
    store.publish(4, {"weight": torch.tensor([4])})

    assert store.reference_count(1) == 1
    assert [name for name, _ in iterator] == ["second"]
    assert 1 not in store.available_versions
    with pytest.raises(RuntimeError, match="released"):
        next(iter(snapshot))


def test_retention_keeps_recent_versions_and_pinned_old_versions():
    store = WeightSnapshotStore(cache_versions=2)
    for version in (1,):
        store.publish(version, {"weight": torch.tensor([version])})
    pinned = store.get(1)

    store.publish(2, {"weight": torch.tensor([2])})
    store.publish(3, {"weight": torch.tensor([3])})
    assert store.available_versions == (1, 2, 3)
    assert store.latest_version == 3

    store.publish(4, {"weight": torch.tensor([4])})
    assert store.available_versions == (1, 3, 4)
    with store.get(1) as snapshot:
        torch.testing.assert_close(dict(_read(snapshot))["weight"], torch.tensor([1]))

    pinned.release()
    assert store.available_versions == (3, 4)
    with pytest.raises(KeyError, match="version 1"):
        store.get(1)


def test_unreferenced_old_versions_are_evicted():
    store = WeightSnapshotStore(cache_versions=3)
    for version in (1, 2, 3, 4, 5):
        store.publish(version, {"weight": torch.tensor([version])})

    assert store.latest_version == 5
    assert store.available_versions == (3, 4, 5)
    for version in (1, 2):
        with pytest.raises(KeyError):
            store.get(version)


def test_concurrent_repeated_reads_are_consistent_and_release_their_references():
    store = WeightSnapshotStore(cache_versions=2)
    expected = {
        "a": torch.arange(32, dtype=torch.float32).reshape(4, 8),
        "b": torch.tensor([1, 3, 5, 7], dtype=torch.int64),
    }
    store.publish(9, expected)
    workers = 8
    barrier = threading.Barrier(workers)

    def read_version(_):
        with store.get(9) as snapshot:
            barrier.wait(timeout=10)
            return dict(_read(snapshot))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(read_version, range(workers)))

    assert len(results) == workers
    for result in results:
        assert result.keys() == expected.keys()
        for name, tensor in expected.items():
            torch.testing.assert_close(result[name], tensor)
    assert store.reference_count(9) == 0


def test_concurrent_same_version_publish_exports_once_and_is_not_visible_mid_staging():
    store = WeightSnapshotStore()
    export_started = threading.Event()
    finish_export = threading.Event()
    export_calls = 0
    export_lock = threading.Lock()

    def exporting_weights():
        nonlocal export_calls
        with export_lock:
            export_calls += 1
        export_started.set()
        assert finish_export.wait(timeout=10)
        yield "weight", torch.tensor([1, 2, 3])

    def forbidden_export():
        raise AssertionError("the duplicate publisher consumed its input")
        yield

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(store.publish, 5, exporting_weights())
        assert export_started.wait(timeout=10)
        second = executor.submit(store.publish, 5, forbidden_export())

        assert store.latest_version is None
        assert store.available_versions == ()
        with pytest.raises(KeyError):
            store.get(5)

        finish_export.set()
        first.result(timeout=10)
        second.result(timeout=10)

    assert export_calls == 1
    assert store.latest_version == 5
    with store.get() as snapshot:
        torch.testing.assert_close(dict(_read(snapshot))["weight"], torch.tensor([1, 2, 3]))


def test_initial_global_step_zero_is_a_valid_snapshot_version():
    store = WeightSnapshotStore()
    store.publish(0, {"weight": torch.tensor([1])})
    with store.get(0) as snapshot:
        assert snapshot.version == 0
        assert dict(_read(snapshot))["weight"].item() == 1


@pytest.mark.parametrize("version", [-1, True, 1.5, "1", None])
def test_invalid_versions_are_rejected(version):
    store = WeightSnapshotStore()
    with pytest.raises(ValueError, match="non-negative integer"):
        store.publish(version, {"weight": torch.tensor([1])})

    if version is not None:
        with pytest.raises(ValueError, match="non-negative integer"):
            store.get(version)


@pytest.mark.parametrize("cache_versions", [0, 1, -2, True, 1.5, "2"])
def test_cache_versions_must_be_an_integer_of_at_least_two(cache_versions):
    with pytest.raises(ValueError):
        WeightSnapshotStore(cache_versions=cache_versions)


def test_missing_versions_and_names_raise_key_errors():
    store = WeightSnapshotStore()
    with pytest.raises(KeyError, match="no weight snapshot"):
        store.get()

    store.publish(1, {"weight": torch.tensor([1])})
    with pytest.raises(KeyError, match="version 99"):
        store.get_metadata(99)
    with pytest.raises(KeyError):
        store.get_checksum(1, "missing")
    with store.get(1) as snapshot:
        with pytest.raises(KeyError):
            snapshot.checksum("missing")


def test_invalid_weight_iterable_does_not_publish_partial_state():
    store = WeightSnapshotStore()
    with pytest.raises(ValueError, match="duplicate weight name"):
        store.publish(
            1,
            [("weight", torch.tensor([1])), ("weight", torch.tensor([2]))],
        )

    assert store.latest_version is None
    assert store.available_versions == ()
    store.publish(1, {"weight": torch.tensor([3])})
    assert store.latest_version == 1
