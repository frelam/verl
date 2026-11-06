"""Hammer / xLAM irrelevance loader (single-turn, positive + negative).

Raw source: ``MadeAgents/xlam-irrelevance-7.5k``.  Rows with a non-empty
``answers`` label are positives (``st_fc``); rows with an empty label are
irrelevance negatives (``irrelevance``, plan §1.1 "何时不调用工具").
Parsing mirrors ``examples/tool_rl/prepare_data.py::load_hammer``.

Caveat (verified in the legacy pipeline): the ``answers`` field is a JSON
string; rows where it fails to parse are skipped rather than silently treated
as negatives.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

from agentic_rl_data.loaders.common import (
    GENERIC_TOOL_SYSTEM,
    NO_TOOL_SYSTEM,
    build_messages,
    build_reference,
    build_task,
    iter_source_records,
    normalize_calls,
    parse_json_maybe,
    record_skip,
    safe_task_id,
)

logger = logging.getLogger(__name__)

SOURCE = "hammer"
LICENSE = "cc-by-4.0[UNVERIFIED]"


def _calls_from_answers(value: Any) -> list[dict[str, Any]] | None:
    """Return label calls, or ``None`` when the field is absent / unparseable."""
    parsed = parse_json_maybe(value)
    if parsed is None or parsed == "":
        return []
    if isinstance(parsed, str) and parsed.strip() == "":
        return []
    if isinstance(parsed, (list, tuple)):
        calls = normalize_calls(parsed)
        return calls
    # Unparseable non-empty label: caller drops the row.
    return None


def load_tasks(cfg: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Yield ``irrelevance`` negatives and ``st_fc`` positives.

    cfg keys: ``path`` / ``hf_dataset`` (raw input, required), ``max_samples``,
    ``license``.
    """
    cfg = dict(cfg or {})
    max_samples = int(cfg.get("max_samples") or 0)
    license_name = cfg.get("license", LICENSE)
    emitted = 0
    for index, record in enumerate(iter_source_records(cfg, SOURCE)):
        if max_samples and emitted >= max_samples:
            return
        query = record.get("query") or record.get("instruction")
        if not isinstance(query, str) or not query.strip():
            record_skip(cfg, "missing_query")
            continue
        if "answers" not in record and not any(
            key in record for key in ("ground_truth", "label", "answer")
        ):
            # Missing label field is NOT the same as an explicit empty label:
            # the former is unknowable, the latter is a verified negative (T1-06).
            record_skip(cfg, "missing_answers_field")
            continue
        labels = _calls_from_answers(record.get("answers", "[]"))
        if labels is None:
            logger.debug("%s: record %d has an unparseable answers field", SOURCE, index)
            record_skip(cfg, "unparseable_answers")
            continue
        tools = parse_json_maybe(record.get("tools"))
        if labels:
            task_type, system = "st_fc", GENERIC_TOOL_SYSTEM
            gt_calls = [labels]
        else:
            task_type, system = "irrelevance", NO_TOOL_SYSTEM
            gt_calls = []
        task = build_task(
            source=SOURCE,
            task_id=safe_task_id(SOURCE, record.get("id"), index),
            task_type=task_type,
            messages=build_messages(system, [query]),
            tools=tools,
            env={"type": "none", "config": {"max_turns": 1}},
            reference=build_reference(gt_calls=gt_calls),
            max_turns=1,
            expected_turns=1,
            license=license_name,
            meta_extra={"raw_index": index, "is_irrelevant": not bool(labels)},
        )
        emitted += 1
        yield task


__all__ = ["SOURCE", "LICENSE", "load_tasks"]
