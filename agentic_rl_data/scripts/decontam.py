"""Step 4: decontamination against BFCL v3/v4 test sets (plan §3 step 4, §10).

Frozen policy: **a single matching 13-gram is enough to drop the task** and the
hit is recorded (id, source, task_type, matched BFCL ids, hit type).  Matching
uses two channels (plan says "n-gram + 近似匹配"):

1. exact 13-word-gram overlap on user text and on tool descriptions,
2. exact tool-name overlap,
3. **approximate Jaccard >= 0.90** on 5-shingles between the task's user text
   and any indexed BFCL prompt (catches paraphrases that share no 13-gram).

An **empty contamination index is a fatal error** (fatal means "decontam
reported 0 hits because it never compared anything" — the acceptance criterion
"BFCL 命中数为 0" would be a false pass).

Test-set source (verified 2026-11-05): ``bfcl-eval`` on PyPI, latest stable
``2026.3.23`` (docs' ``2025.10.27.1`` was stale), Apache-2.0, data ships inside
the wheel.  Pass either the installed package data directory::

    python -c "import bfcl_eval, pathlib; print(pathlib.Path(bfcl_eval.__path__[0]) / 'data')"

or a gorilla-repo BFCL directory, or a JSONL export.  ``--extra-path`` accepts
additional evaluation sets (e.g. AIME / math eval pool) indexed as text-only.

Report format: per-source in/kept/dropped counts, **full hit counts** and up to
50 hit samples per source (never only "50 total").
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agentic_rl_data.loaders.common import normalize_tools  # noqa: E402
from agentic_rl_data.schema import dump_jsonl, load_jsonl, normalize_query, text_shingles  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_K = 13
DEFAULT_MIN_HITS = 1
JACCARD_THRESHOLD = 0.90
SHINGLE_K = 5
_WORD_RE = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------------------
# n-gram helpers
# ---------------------------------------------------------------------------


def text_ngrams(text: Any, k: int = DEFAULT_K) -> set[str]:
    """Word n-grams over normalised text; char n-grams for very short text."""
    norm = normalize_query(text)
    if not norm:
        return set()
    tokens = _WORD_RE.findall(norm)
    if len(tokens) >= 3:
        kk = min(max(1, int(k)), len(tokens))
        return {" ".join(tokens[index : index + kk]) for index in range(len(tokens) - kk + 1)}
    compact = "".join(tokens) or norm
    kk = min(5, len(compact))
    if kk <= 0:
        return set()
    return {compact[index : index + kk] for index in range(len(compact) - kk + 1)}


def _extract_texts(value: Any) -> list[str]:
    """Recursively collect text from BFCL ``question`` shapes."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        for key in ("content", "text", "value", "question"):
            if key in value:
                return _extract_texts(value[key])
        return []
    if isinstance(value, (list, tuple)):
        texts: list[str] = []
        for item in value:
            texts.extend(_extract_texts(item))
        return texts
    return []


def jaccard_shingles(shingles_a: set[int], shingles_b: set[int]) -> float:
    if not shingles_a or not shingles_b:
        return 0.0
    union = shingles_a | shingles_b
    return len(shingles_a & shingles_b) / len(union) if union else 0.0


# ---------------------------------------------------------------------------
# index building
# ---------------------------------------------------------------------------


def collect_contam_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Normalise one BFCL / eval record to ``{"id", "texts", "tools"}``."""
    record_id = record.get("id") or record.get("question_id") or record.get("name")
    texts = _extract_texts(record.get("question", record.get("prompt")))
    tools = record.get("function") or record.get("functions") or record.get("tools") or []
    return {
        "id": str(record_id) if record_id is not None else "unknown",
        "texts": texts,
        "tools": normalize_tools(tools),
    }


def build_index(records: Iterable[Mapping[str, Any]], *, k: int = DEFAULT_K) -> dict[str, Any]:
    """Build the n-gram / tool-definition / 5-shingle index.

    Raises ``ValueError`` when zero records were indexed: an empty index would
    silently report "0 contamination hits" (false pass), which C6/§8.2 forbids.
    """
    query_ngrams: dict[str, set[str]] = {}
    tool_names: dict[str, set[str]] = {}
    tool_ngrams: dict[str, set[str]] = {}
    doc_shingles: dict[str, set[int]] = {}
    ids: set[str] = set()
    count = 0
    for raw in records:
        record = collect_contam_record(raw)
        record_id = record["id"]
        ids.add(record_id)
        count += 1
        shingles: set[int] = set()
        for text in record["texts"]:
            for ngram in text_ngrams(text, k=k):
                query_ngrams.setdefault(ngram, set()).add(record_id)
            shingles |= text_shingles(text, k=SHINGLE_K)
        if shingles:
            doc_shingles[record_id] = shingles
        for tool in record["tools"]:
            name = normalize_query(tool.get("name"))
            if name:
                tool_names.setdefault(name, set()).add(record_id)
            description = tool.get("description") or ""
            for ngram in text_ngrams(description, k=k):
                tool_ngrams.setdefault(ngram, set()).add(record_id)
    if count == 0:
        raise ValueError(
            "contamination index is empty: refusing to report 0 hits without comparing anything "
            "(check --bfcl-path; bfcl-eval data lives under `bfcl_eval/data`)"
        )
    return {
        "k": k,
        "num_records": count,
        "ids": ids,
        "query_ngrams": query_ngrams,
        "tool_names": tool_names,
        "tool_ngrams": tool_ngrams,
        "doc_shingles": doc_shingles,
        "jaccard_threshold": JACCARD_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------


def match_task(
    task: Mapping[str, Any],
    index: Mapping[str, Any],
    *,
    min_hits: int = DEFAULT_MIN_HITS,
    jaccard_threshold: float = JACCARD_THRESHOLD,
) -> dict[str, Any]:
    """Return contamination hit details for one unified task."""
    k = int(index.get("k") or DEFAULT_K)
    matched_ids: set[str] = set()
    query_hits: list[str] = []
    tool_name_hits: list[str] = []
    tool_desc_hits: list[str] = []
    query_texts: list[str] = []
    messages = task.get("messages")
    for message in messages if isinstance(messages, list) else []:
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        text = message.get("content")
        if isinstance(text, str) and text:
            query_texts.append(text)
        for ngram in text_ngrams(text, k=k):
            hits = index["query_ngrams"].get(ngram)
            if hits:
                query_hits.append(ngram)
                matched_ids.update(hits)
    for tool in normalize_tools(task.get("tools")):
        name = normalize_query(tool.get("name"))
        if name and name in index["tool_names"]:
            tool_name_hits.append(str(tool.get("name")))
            matched_ids.update(index["tool_names"][name])
        for ngram in text_ngrams(tool.get("description") or "", k=k):
            hits = index["tool_ngrams"].get(ngram)
            if hits:
                tool_desc_hits.append(ngram)
                matched_ids.update(hits)

    # Approximate channel: exact Jaccard on 5-shingles against every indexed
    # prompt (BFCL is a few thousand short prompts; this is offline prep code).
    query_shingles: set[int] = set()
    for text in query_texts:
        query_shingles |= text_shingles(text, k=SHINGLE_K)
    approx_best = 0.0
    approx_id: str | None = None
    if query_shingles and index.get("doc_shingles"):
        candidates = matched_ids or set(index["ids"])
        for doc_id in candidates:
            shingles = index["doc_shingles"].get(doc_id)
            if not shingles:
                continue
            similarity = jaccard_shingles(query_shingles, shingles)
            if similarity > approx_best:
                approx_best, approx_id = similarity, doc_id
        if approx_best < jaccard_threshold:
            approx_id = None
        else:
            matched_ids.add(str(approx_id))

    contaminated = (
        len(query_hits) >= min_hits
        or bool(tool_name_hits)
        or len(tool_desc_hits) >= min_hits
        or approx_id is not None
    )
    return {
        "contaminated": contaminated,
        "matched_ids": sorted(matched_ids),
        "query_ngram_hits": len(query_hits),
        "query_ngram_examples": sorted(query_hits)[:10],
        "tool_name_hits": tool_name_hits[:10],
        "tool_desc_ngram_hits": len(tool_desc_hits),
        "tool_desc_examples": sorted(tool_desc_hits)[:10],
        "approx_jaccard": round(approx_best, 4),
        "approx_matched_id": approx_id,
    }


def decontaminate(
    tasks: Iterable[Mapping[str, Any]],
    index: Mapping[str, Any],
    *,
    min_hits: int = DEFAULT_MIN_HITS,
    jaccard_threshold: float = JACCARD_THRESHOLD,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop tasks whose query / tool definition overlaps the eval index."""
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for task in tasks:
        match = match_task(task, index, min_hits=min_hits, jaccard_threshold=jaccard_threshold)
        if match["contaminated"]:
            dropped.append(
                {
                    "id": task.get("id"),
                    "source": task.get("source"),
                    "task_type": task.get("task_type"),
                    "reason": "bfcl_contamination",
                    **match,
                }
            )
            continue
        kept.append(dict(task))
    return kept, dropped


def count_residual(
    tasks: Iterable[Mapping[str, Any]],
    index: Mapping[str, Any],
    *,
    min_hits: int = DEFAULT_MIN_HITS,
    jaccard_threshold: float = JACCARD_THRESHOLD,
) -> dict[str, Any]:
    """Re-match the *kept* tasks and count residual contamination hits.

    ``total_dropped`` in the report is the number of hits **detected and
    already removed** (audit column).  The acceptance rule (plan §8 A2) is that
    the **final** split contains 0 hits, so this second pass is what strict
    checking must look at.  A non-zero result means the matcher dropped some
    tasks but the output still overlaps the eval index (bug or stale index).

    Returns ``{"residual_hits": int, "residual_samples": [...]}``; samples are
    capped at 50 for the report.
    """
    residual = 0
    samples: list[dict[str, Any]] = []
    for task in tasks:
        match = match_task(task, index, min_hits=min_hits, jaccard_threshold=jaccard_threshold)
        if match["contaminated"]:
            residual += 1
            if len(samples) < 50:
                samples.append(
                    {
                        "id": task.get("id"),
                        "source": task.get("source"),
                        "task_type": task.get("task_type"),
                        **match,
                    }
                )
    return {"residual_hits": residual, "residual_samples": samples}


# ---------------------------------------------------------------------------
# IO + CLI
# ---------------------------------------------------------------------------


def read_any_records(path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Read JSONL / JSON / a directory of BFCL data (tolerant of ``.json`` JSONL)."""
    path = Path(path)
    records: list[dict[str, Any]] = []
    if path.is_dir():
        for file_path in sorted(path.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in {".json", ".jsonl", ".ndjson"}:
                records.extend(read_any_records(file_path, limit=None))
                if limit is not None and len(records) >= limit:
                    break
        return records[:limit] if limit is not None else records
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return []
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, list):
        records = [item for item in payload if isinstance(item, Mapping)]
    elif isinstance(payload, Mapping):
        records = [payload]
    else:
        for line in stripped.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, Mapping):
                records.append(item)
    return records[:limit] if limit is not None else records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Decontaminate unified tasks against BFCL v3/v4 tests.")
    parser.add_argument("--in-dir", default=None)
    parser.add_argument("--in", dest="in_files", action="append", default=[])
    parser.add_argument("--bfcl-path", required=True, help="bfcl-eval data dir / gorilla BFCL dir / JSONL dump")
    parser.add_argument("--extra-path", action="append", default=[], help="extra eval sets (e.g. AIME prompts)")
    parser.add_argument("--out-dir", default="agentic_rl_data/data/decontaminated")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--min-hits", type=int, default=DEFAULT_MIN_HITS)
    parser.add_argument("--jaccard-threshold", type=float, default=JACCARD_THRESHOLD)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    index = build_index(read_any_records(args.bfcl_path), k=args.k)
    for extra_path in args.extra_path:
        extra_index = build_index(read_any_records(extra_path), k=args.k)
        for key in ("query_ngrams", "tool_names", "tool_ngrams"):
            for ngram, ids in extra_index[key].items():
                index[key].setdefault(ngram, set()).update(ids)
        index["doc_shingles"].update(extra_index["doc_shingles"])
        index["ids"].update(extra_index["ids"])
        index["num_records"] += extra_index["num_records"]
    logger.info("contamination index: %d records", index["num_records"])

    in_paths = [Path(path) for path in args.in_files]
    if args.in_dir:
        in_paths.extend(sorted(Path(args.in_dir).glob("*.jsonl")))
    if not in_paths:
        parser.error("pass --in-dir or --in FILE")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "bfcl_records": index["num_records"],
        "k": args.k,
        "min_hits": args.min_hits,
        "jaccard_threshold": args.jaccard_threshold,
        "sources": {},
        "total_in": 0,
        "total_kept": 0,
        "total_dropped": 0,
        "hit_counts_by_source": {},
        "hit_samples_by_source": {},
        # Detected-and-removed hits are an audit trail; ONLY residual_hits is
        # gated by plan §8 A2 (final split must contain 0 hits).
        "residual_hits": 0,
        "residual_samples_by_source": {},
        "bfcl_path": str(args.bfcl_path),
        "index_records": index["num_records"],
    }
    for in_path in in_paths:
        tasks = list(load_jsonl(in_path))
        kept, dropped = decontaminate(tasks, index, min_hits=args.min_hits, jaccard_threshold=args.jaccard_threshold)
        residual = count_residual(kept, index, min_hits=args.min_hits, jaccard_threshold=args.jaccard_threshold)
        source = str(tasks[0].get("source")) if tasks else in_path.stem
        out_path = out_dir / f"{source}.jsonl"
        written = dump_jsonl(kept, out_path)
        entry = report["sources"].setdefault(
            source,
            {"source": source, "in": 0, "kept": 0, "dropped": 0, "residual_hits": 0},
        )
        entry["in"] += len(tasks)
        entry["kept"] += written
        entry["dropped"] += len(dropped)
        entry["residual_hits"] += residual["residual_hits"]
        report["residual_hits"] += residual["residual_hits"]
        if residual["residual_samples"]:
            bucket = report["residual_samples_by_source"].setdefault(source, [])
            remaining = 50 - len(bucket)
            if remaining > 0:
                bucket.extend(residual["residual_samples"][:remaining])
        report["hit_counts_by_source"][source] = len(dropped)
        samples = report["hit_samples_by_source"].setdefault(source, [])
        remaining = 50 - len(samples)
        if remaining > 0:
            samples.extend(dropped[:remaining])
        report["total_in"] += len(tasks)
        report["total_kept"] += written
        report["total_dropped"] += len(dropped)
    report_path = out_dir / "_decontam_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("bfcl_records", "total_in", "total_kept", "total_dropped", "residual_hits")}, indent=2))
    for source, stats in report["sources"].items():
        print(f"{source}: in={stats['in']} kept={stats['kept']} dropped={stats['dropped']} (samples stored: {len(report['hit_samples_by_source'][source])})")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
