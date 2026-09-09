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
"""
Near-duplicate detection utilities for training-pool dedup (DESIGN.md section 2).

Two levels:
1. exact_dedup: normalized-text exact match (cheap, always run first).
2. minhash_near_dedup: MinHash + LSH banding, catches variants with small edits
   (numeric perturbation, reworded sentences) that exact match misses.

Self-contained, no third-party deps beyond numpy.
"""

import hashlib
import math
import re

import numpy as np

_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]")

# Prime larger than 2**32, used as the MinHash permutation space so that
# (h1 + i * h2) never overflows uint64 (i < num_perm <= 256).
_M64 = np.uint64(4294967311)


def normalize_text(text: str) -> str:
    """Lowercase, strip non-alphanumeric chars, collapse whitespace."""
    text = text.lower()
    text = _NON_ALNUM_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def exact_key(text: str) -> str:
    return hashlib.md5(normalize_text(text).encode("utf-8")).hexdigest()


def exact_dedup(texts: list[str]) -> list[int]:
    """Return indices of kept rows (first occurrence wins)."""
    seen: set[str] = set()
    keep: list[int] = []
    for i, t in enumerate(texts):
        k = exact_key(t)
        if k not in seen:
            seen.add(k)
            keep.append(i)
    return keep


def _shingle_hashes(text: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Word n-gram shingles -> two uint32 hash arrays (h1, h2) via md5.

    Documents shorter than n words fall back to a single whole-text shingle,
    so identical short texts still collide.
    """
    words = normalize_text(text).split()
    if not words:
        return np.zeros(0, dtype=np.uint64), np.zeros(0, dtype=np.uint64)
    if len(words) < n:
        shingles = {" ".join(words)}
    else:
        shingles = {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}
    h1 = np.empty(len(shingles), dtype=np.uint64)
    h2 = np.empty(len(shingles), dtype=np.uint64)
    for j, s in enumerate(shingles):
        d = hashlib.md5(s.encode("utf-8")).digest()
        h1[j] = int.from_bytes(d[:4], "little")
        h2[j] = int.from_bytes(d[4:8], "little")
    return h1, h2


def _minhash_signature(text: str, num_perm: int, shingle_n: int) -> np.ndarray | None:
    """(num_perm,) uint64 MinHash signature using double hashing h1 + i*h2 (mod M)."""
    h1, h2 = _shingle_hashes(text, shingle_n)
    if h1.shape[0] == 0:
        return None
    i = np.arange(num_perm, dtype=np.uint64)[:, None]  # (P, 1)
    vals = (h1[None, :] + i * h2[None, :]) % _M64  # (P, S)
    return vals.min(axis=1)


def minhash_near_dedup(
    texts: list[str],
    threshold: float = 0.6,
    num_perm: int = 128,
    bands: int = 32,
    shingle_n: int = 5,
) -> list[int]:
    """Greedy MinHash-LSH near-dedup. Returns indices of kept rows.

    A row is dropped if it shares an LSH band bucket with an earlier kept row AND
    their signature similarity (fraction of equal minima, an unbiased estimate of
    Jaccard) is >= threshold. Empty texts are always kept (they carry no signal).

    Performance notes (all decisions are identical to a naive per-candidate loop):

    1. Kept signatures live in one (n_kept, num_perm) array and band-collision
       candidates are scored with vectorized broadcast comparisons instead of a
       per-candidate Python loop. mean(sig == s) >= threshold is evaluated as
       match count >= ceil(threshold * num_perm).
    2. Bucket membership is stored in over-allocated numpy arrays (doubling
       growth, amortized O(1) insert) so queries concatenate array views instead
       of re-converting Python lists — otherwise hub buckets (templates shared
       by many rows) cost O(bucket_size) of list->array boxing per query.
    3. Candidates are filtered in a uint8-truncated copy of the signature space
       first: uint64 equality implies uint8 equality, so no true duplicate is
       rejected. Mod-256 collisions can only create false positives, removed by
       an exact uint64 confirm of the rare survivors. This cuts comparison
       memory traffic ~4x versus a uint32-domain filter.

    Cluster-heavy pools (many near-dup templates) degrade a naive implementation
    to O(N^2) tiny numpy calls; the above keeps the same O(N^2) candidate volume
    but with a much smaller constant.
    """
    assert num_perm % bands == 0
    rows_per_band = num_perm // bands
    # Smallest c with float(c / num_perm) >= threshold. The ceil is exact when
    # num_perm is a power of two (multiply is exact); the loops absorb float error
    # in threshold * num_perm for other num_perm values.
    min_count = math.ceil(threshold * num_perm)
    while min_count > 0 and (min_count - 1) / num_perm >= threshold:
        min_count -= 1
    while min_count / num_perm < threshold:
        min_count += 1
    # Each bucket maps a band key -> [capacity array, used count].
    buckets: list[dict[bytes, list]] = [dict() for _ in range(bands)]
    # Buffers of kept rows' signatures; bucket entries index into these buffers.
    # kept_buf8 holds signatures truncated to uint8 (values are < 2**33); it is
    # a filter-only copy — final decisions always use kept_buf (uint64).
    kept_buf = np.empty((len(texts), num_perm), dtype=np.uint64)
    kept_buf8 = np.empty((len(texts), num_perm), dtype=np.uint8)
    n_kept = 0
    keep: list[int] = []

    # Chunk size for candidate comparisons; bounds transient memory.
    _CMP_CHUNK = 1 << 15

    for idx, t in enumerate(texts):
        sig = _minhash_signature(t, num_perm, shingle_n)
        if sig is None:
            keep.append(idx)
            continue

        q8 = sig.astype(np.uint8)
        cand_arrays: list[np.ndarray] = []
        band_keys: list[bytes] = []
        for b in range(bands):
            key = sig[b * rows_per_band : (b + 1) * rows_per_band].tobytes()
            band_keys.append(key)
            cell = buckets[b].get(key)
            if cell is not None:
                cand_arrays.append(cell[0][: cell[1]])

        is_dup = False
        if cand_arrays:
            cand = np.unique(np.concatenate(cand_arrays)) if len(cand_arrays) > 1 else cand_arrays[0]
            for s in range(0, cand.shape[0], _CMP_CHUNK):
                c = cand[s : s + _CMP_CHUNK]
                # uint8 filter, then exact uint64 confirm of the survivors.
                cnt = (kept_buf8[c] == q8[None, :]).sum(axis=1)
                surv = c[cnt >= min_count]
                if surv.size and int((kept_buf[surv] == sig[None, :]).sum(axis=1).max()) >= min_count:
                    is_dup = True
                    break

        if not is_dup:
            kept_buf[n_kept] = sig
            kept_buf8[n_kept] = q8
            n_kept += 1
            row = n_kept - 1
            for b, key in enumerate(band_keys):
                bucket = buckets[b]
                cell = bucket.get(key)
                if cell is None:
                    arr = np.empty(8, dtype=np.int64)
                    arr[0] = row
                    bucket[key] = [arr, 1]
                else:
                    arr, used = cell
                    if used == arr.size:
                        arr = np.empty(used * 2, dtype=np.int64)
                        arr[:used] = cell[0]
                        cell[0] = arr
                    arr[used] = row
                    cell[1] = used + 1
            keep.append(idx)

    return keep
