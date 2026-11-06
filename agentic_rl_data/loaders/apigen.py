"""APIGen / xLAM single-turn function-calling loader.

Raw source: ``Salesforce/xlam-function-calling-60k`` (one JSON file with
``query`` / ``tools`` / ``answers`` string fields).  Parsing mirrors
``examples/tool_rl/prepare_data.py::load_apigen``.

The raw download path is parameterised: pass ``cfg["path"]`` to a local JSON
dump, or ``cfg["hf_dataset"]`` to fetch it via ``datasets``.  No network call
happens unless explicitly configured.

License note: the HF dataset card could not be reached from this sandbox (see
``reports/data_sources.md``); ``meta.license`` is therefore marked
``[UNVERIFIED]`` until the user confirms it.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator, Mapping

from agentic_rl_data.loaders.common import (
    GENERIC_TOOL_SYSTEM,
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

SOURCE = "apigen"
LICENSE = "cc-by-4.0[UNVERIFIED]"


def _parse_field(value: Any) -> Any:
    parsed = parse_json_maybe(value)
    return parsed if parsed is not None else value


def load_tasks(cfg: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Yield schema-valid ``st_fc`` tasks (``np.ndarray`` safe, no network by default).

    cfg keys: ``path`` / ``hf_dataset`` (raw input, required), ``max_samples``
    (default 0 = all), ``license`` (override).
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
        tools = _parse_field(record.get("tools"))
        answers = _parse_field(record.get("answers"))
        calls = normalize_calls(answers)
        if not calls:
            # No parseable label: emitting [] would mislabel the row as a
            # "no tools needed" negative, so drop it (reason is logged).
            logger.debug("%s: record %d has no parseable answer calls", SOURCE, index)
            record_skip(cfg, "no_parseable_answer")
            continue
        task = build_task(
            source=SOURCE,
            task_id=safe_task_id(SOURCE, record.get("id"), index),
            task_type="st_fc",
            messages=build_messages(
                GENERIC_TOOL_SYSTEM,
                [query],
            ),
            tools=tools,
            env={"type": "none", "config": {"max_turns": 1}},
            reference=build_reference(gt_calls=[calls]),
            max_turns=1,
            expected_turns=1,
            license=license_name,
            meta_extra={"raw_index": index},
        )
        emitted += 1
        yield task


def load_tasks_stream(cfg: Mapping[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Alias kept for symmetry with the other loaders."""
    return load_tasks(dict(cfg or {}))


__all__ = ["SOURCE", "LICENSE", "load_tasks", "load_tasks_stream"]
