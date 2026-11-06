"""Step 3: exact + MinHash near-duplicate removal (plan §3 step 3).

Exact key: ``schema.task_dedup_key`` (normalised user queries + tools
signature; tool signature is name-sorted, so a re-ordered tool menu no longer
escapes dedup).

Near-duplicate: MinHash over hashed word/char shingles (``schema.text_shingles``)
with LSH banding on the minhash signature.  Pure-Python on purpose: no numpy
/ datasketch dependency, so it runs anywhere.  Exact ``task_dedup_key`` still
runs first.

Cross-source priority is caller-defined: pass the legacy sources first (or via
``--against``) so the overlapping ToolMind / APIGen-MT rows are the ones
dropped, matching the plan's "ToolMind 与 xlam/ToolACE 有重叠，必须去重".

Pure layer: :func:`exact_dedup`, :func:`near_dedup`, :func:`run_dedup`.
CLI layer: ``--in`` files (kept + written), ``--against`` files (drop-only
reference pool), ``--out``, report with per-drop details.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agentic_rl_data.schema import (  # noqa: E402
    dump_jsonl,
    load_jsonl,
    normalize_query,
    task_dedup_key,
    text_shingles,
)

logger = logging.getLogger(__name__)

DEFAULT_NUM_PERM = 128
DEFAULT_BANDS = 16
DEFAULT_THRESHOLD = 0.8


# ---------------------------------------------------------------------------
# exact dedup
# ---------------------------------------------------------------------------


def exact_dedup(tasks: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep the first occurrence of each ``task_dedup_key``."""
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    first_seen: dict[str, str] = {}
    for task in tasks:
        key = task_dedup_key(task)
        if key in first_seen:
            dropped.append(
                {
                    "id": task.get("id"),
                    "source": task.get("source"),
                    "matched_id": first_seen[key],
                    "reason": "exact_duplicate",
                }
            )
            continue
        first_seen[key] = str(task.get("id"))
        kept.append(dict(task))
    return kept, dropped


# ---------------------------------------------------------------------------
# MinHash near dedup (pure python)
# ---------------------------------------------------------------------------


def task_text(task: Mapping[str, Any]) -> str:
    """Concatenated user text used for near-duplicate signatures."""
    messages = task.get("messages")
    if not isinstance(messages, list):
        return ""
    return "\n".join(
        normalize_query(message.get("content"))
        for message in messages
        if isinstance(message, Mapping) and message.get("role") == "user"
    )


def task_shingles(task: Mapping[str, Any], k: int = 5) -> set[int]:
    return text_shingles(task_text(task), k=k)


def minhash_signature(shingles: Iterable[int], num_perm: int = DEFAULT_NUM_PERM) -> tuple[int, ...]:
    """MinHash signature: ``num_perm`` independent blake2b hash families."""
    shingle_list = sorted(set(int(shingle) for shingle in shingles))
    if not shingle_list or num_perm <= 0:
        return tuple([0] * max(num_perm, 0))
    signature: list[int] = []
    for permutation in range(num_perm):
        seed = permutation.to_bytes(4, "big")
        best = None
        for shingle in shingle_list:
            digest = hashlib.blake2b(seed + shingle.to_bytes(8, "big"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            if best is None or value < best:
                best = value
        signature.append(best or 0)
    return tuple(signature)


def jaccard_estimate(sig_a: tuple[int, ...], sig_b: tuple[int, ...]) -> float:
    if not sig_a or not sig_b or len(sig_a) != len(sig_b):
        return 0.0
    equal = sum(1 for left, right in zip(sig_a, sig_b) if left == right)
    return equal / len(sig_a)


def jaccard_exact(shingles_a: set[int], shingles_b: set[int]) -> float:
    if not shingles_a or not shingles_b:
        return 0.0
    intersection = len(shingles_a & shingles_b)
    union = len(shingles_a | shingles_b)
    return intersection / union if union else 0.0


def _band_keys(signature: tuple[int, ...], bands: int) -> list[tuple[int, ...]]:
    bands = max(1, min(int(bands), len(signature))) if signature else 0
    if not bands:
        return []
    rows = max(1, len(signature) // bands)
    keys: list[tuple[int, ...]] = []
    for band in range(bands):
        start = band * rows
        end = len(signature) if band == bands - 1 else start + rows
        if start >= end:
            break
        keys.append(signature[start:end])
    return keys


def near_dedup(
    tasks: Iterable[Mapping[str, Any]],
    *,
    against: Iterable[Mapping[str, Any]] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    num_perm: int = DEFAULT_NUM_PERM,
    bands: int = DEFAULT_BANDS,
    k: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop near-duplicates with MinHash + LSH banding.

    ``against`` tasks form a reference pool that is *not* returned; incoming
    tasks matching the pool are dropped.  Within ``tasks`` the first occurrence
    wins (later duplicates are dropped and record ``matched_id`` + similarity).
    """
    threshold = float(threshold)
    pool: dict[tuple[int, ...], list[tuple[str, str, tuple[int, ...]]]] = {}
    # (band_key) -> [(id, source, signature)]

    def _add_to_pool(task: Mapping[str, Any], signature: tuple[int, ...]) -> None:
        for band_key in _band_keys(signature, bands):
            pool.setdefault(band_key, []).append((str(task.get("id")), str(task.get("source")), signature))

    for task in against or []:
        signature = minhash_signature(task_shingles(task, k=k), num_perm=num_perm)
        _add_to_pool(task, signature)

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for task in tasks:
        shingles = task_shingles(task, k=k)
        signature = minhash_signature(shingles, num_perm=num_perm)
        best: tuple[float, str | None, str | None] = (0.0, None, None)
        candidates: dict[str, tuple[str, tuple[int, ...]]] = {}
        for band_key in _band_keys(signature, bands):
            for candidate_id, candidate_source, candidate_signature in pool.get(band_key, []):
                candidates[candidate_id] = (candidate_source, candidate_signature)
        for candidate_id, (candidate_source, candidate_signature) in candidates.items():
            estimate = jaccard_estimate(signature, candidate_signature)
            if estimate > best[0]:
                best = (estimate, candidate_id, candidate_source)
        if best[0] >= threshold:
            dropped.append(
                {
                    "id": task.get("id"),
                    "source": task.get("source"),
                    "matched_id": best[1],
                    "matched_source": best[2],
                    "similarity": round(best[0], 4),
                    "reason": "near_duplicate",
                }
            )
            continue
        kept.append(dict(task))
        _add_to_pool(task, signature)
    return kept, dropped


def run_dedup(
    tasks: Iterable[Mapping[str, Any]],
    *,
    against: Iterable[Mapping[str, Any]] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    num_perm: int = DEFAULT_NUM_PERM,
    bands: int = DEFAULT_BANDS,
    k: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Exact dedup first, then MinHash; returns ``(kept, dropped, stats)``."""
    tasks = list(tasks)
    kept_exact, dropped_exact = exact_dedup(tasks)
    kept_near, dropped_near = near_dedup(
        kept_exact,
        against=list(against or []),
        threshold=threshold,
        num_perm=num_perm,
        bands=bands,
        k=k,
    )
    stats = {
        "input": len(tasks),
        "exact_dropped": len(dropped_exact),
        "near_dropped": len(dropped_near),
        "kept": len(kept_near),
        "threshold": threshold,
        "num_perm": num_perm,
        "bands": bands,
        "shingle_k": k,
    }
    return kept_near, dropped_exact + dropped_near, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _read_tasks(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for path in paths:
        for task in load_jsonl(path):
            tasks.append(task)
    return tasks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exact + MinHash near-dedup for unified task JSONL.")
    parser.add_argument("--in", dest="in_files", action="append", default=[], required=True)
    parser.add_argument("--against", action="append", default=[], help="drop-only reference pool (e.g. legacy 5 sources)")
    parser.add_argument("--out", default="agentic_rl_data/data/deduped/train.jsonl")
    parser.add_argument("--report", default="agentic_rl_data/data/deduped/_dedup_report.json")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--num-perm", type=int, default=DEFAULT_NUM_PERM)
    parser.add_argument("--bands", type=int, default=DEFAULT_BANDS)
    parser.add_argument("--k", type=int, default=5, help="shingle size")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    tasks = _read_tasks(args.in_files)
    against = _read_tasks(args.against)
    kept, dropped, stats = run_dedup(
        tasks,
        against=against,
        threshold=args.threshold,
        num_perm=args.num_perm,
        bands=args.bands,
        k=args.k,
    )
    dump_jsonl(kept, args.out)
    report = {"stats": stats, "against_tasks": len(against), "dropped_details": dropped[:500]}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
