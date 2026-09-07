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
    """
    assert num_perm % bands == 0
    rows_per_band = num_perm // bands
    buckets: list[dict[bytes, list[int]]] = [dict() for _ in range(bands)]
    sigs: list[np.ndarray | None] = []
    keep: list[int] = []

    for idx, t in enumerate(texts):
        sig = _minhash_signature(t, num_perm, shingle_n)
        sigs.append(sig)
        if sig is None:
            keep.append(idx)
            continue

        candidates: set[int] = set()
        band_keys: list[bytes] = []
        for b in range(bands):
            key = sig[b * rows_per_band : (b + 1) * rows_per_band].tobytes()
            band_keys.append(key)
            candidates.update(buckets[b].get(key, ()))

        is_dup = False
        for j in candidates:
            if float(np.mean(sig == sigs[j])) >= threshold:
                is_dup = True
                break

        if not is_dup:
            for b, key in enumerate(band_keys):
                buckets[b].setdefault(key, []).append(idx)
            keep.append(idx)

    return keep
