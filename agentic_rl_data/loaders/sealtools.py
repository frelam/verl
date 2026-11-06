"""Seal-Tools loader (single-turn query + calling, tools from ``tool.jsonl``).

Raw source: ``fairyshine/Seal-Tools`` (Apache-2.0), files
``Seal-Tools_Dataset/tool.jsonl`` and ``Seal-Tools_Dataset/train.jsonl``.
Parsing mirrors ``examples/tool_rl/prepare_data.py::load_sealtools``:
nested calls (``API_call_N`` references) are dropped because the single-turn
pipeline has no execution; same-field tools are used as hard distractors and
a few global distractors fill the menu.

cfg keys: ``tools_path`` (required), ``train_path`` (required unless
``tools_path`` points to a directory), ``max_samples``, ``n_distractors``
(default 5), ``seed`` (default 42), ``license``.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Iterator

from agentic_rl_data.loaders.common import (
    GENERIC_TOOL_SYSTEM,
    LoaderError,
    build_messages,
    build_reference,
    build_task,
    first_non_empty,
    iter_path_records,
    read_records,
    record_skip,
    safe_task_id,
)

logger = logging.getLogger(__name__)

SOURCE = "sealtools"
LICENSE = "apache-2.0"

_TYPE_MAP = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
}


def _sealtools_schema(raw: dict[str, Any]) -> dict[str, Any] | None:
    name = raw.get("api_name", "")
    if not isinstance(name, str) or not name.strip():
        return None
    properties: dict[str, Any] = {}
    for param_name, param_info in (raw.get("parameters") or {}).items():
        if not isinstance(param_info, dict):
            continue
        properties[str(param_name)] = {
            "type": _TYPE_MAP.get(str(param_info.get("type", "str")).lower(), "string"),
            "description": str(param_info.get("description", "")),
        }
    return {
        "name": name.strip(),
        "description": str(raw.get("api_description", "")),
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": sorted(raw.get("required") or properties),
        },
    }


def _pick_distractors(pool: list[dict[str, Any]], exclude: set[str], n: int, rng: random.Random) -> list[dict[str, Any]]:
    candidates = [tool for tool in pool if tool.get("name") not in exclude]
    rng.shuffle(candidates)
    return candidates[:n]


def load_tasks(cfg: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Yield single-turn ``st_fc`` tasks from Seal-Tools train split."""
    cfg = dict(cfg or {})
    max_samples = int(cfg.get("max_samples") or 0)
    license_name = cfg.get("license", LICENSE)
    n_distractors = int(cfg.get("n_distractors", 5))
    rng = random.Random(int(cfg.get("seed", 42)))

    tools_path = first_non_empty(cfg.get("tools_path"), cfg.get("tool_path"))
    train_path = first_non_empty(cfg.get("train_path"), cfg.get("path"))
    if tools_path is None or train_path is None:
        raise LoaderError(
            f"{SOURCE}: configure cfg['tools_path'] and cfg['train_path'] "
            "(local copies of Seal-Tools_Dataset/tool.jsonl and train.jsonl)"
        )

    schemas: dict[str, dict[str, Any]] = {}
    fields: dict[str, list[str]] = {}
    for raw in iter_path_records(Path(str(tools_path))):
        schema = _sealtools_schema(raw)
        if schema is None:
            continue
        schemas[schema["name"]] = schema
        fields.setdefault(str(raw.get("field", "")), []).append(schema["name"])
    if not schemas:
        raise LoaderError(f"{SOURCE}: no tool schemas parsed from {tools_path}")
    logger.info("%s: %d tools in %d fields", SOURCE, len(schemas), len(fields))

    pool = list(schemas.values())
    emitted = 0
    n_nested = 0
    for index, sample in enumerate(read_records({"path": str(train_path)})):
        if max_samples and emitted >= max_samples:
            return
        query = sample.get("query")
        calling = sample.get("calling") or []
        if not isinstance(query, str) or not query.strip() or not calling:
            record_skip(cfg, "missing_query_or_calling")
            continue
        gt: list[dict[str, Any]] = []
        nested = False
        for call in calling:
            if not isinstance(call, dict):
                nested = True
                break
            api = str(call.get("api", ""))
            params = call.get("parameters") or {}
            if api not in schemas or not isinstance(params, dict):
                nested = True
                break
            if any(isinstance(value, str) and "API_call" in value for value in params.values()):
                nested = True
                break
            gt.append({"name": api, "arguments": dict(params)})
        if nested or not gt:
            n_nested += 1
            record_skip(cfg, "nested_or_unresolvable")
            continue

        gt_names = {call["name"] for call in gt}
        same_field: dict[str, dict[str, Any]] = {}
        for name in gt_names:
            field = next((key for key, names in fields.items() if name in names), None)
            if field:
                for sibling in fields[field]:
                    if sibling in schemas:
                        same_field.setdefault(sibling, schemas[sibling])
        distractors = _pick_distractors(list(same_field.values()), gt_names, n_distractors, rng)
        if len(distractors) < n_distractors:
            distractors += _pick_distractors(
                pool,
                gt_names | {tool["name"] for tool in distractors},
                n_distractors - len(distractors),
                rng,
            )
        tools = [schemas[name] for name in sorted(gt_names)] + distractors
        rng.shuffle(tools)

        task = build_task(
            source=SOURCE,
            task_id=safe_task_id(SOURCE, sample.get("id"), index),
            task_type="st_fc",
            messages=build_messages(GENERIC_TOOL_SYSTEM, [query]),
            tools=tools,
            env={"type": "none", "config": {"max_turns": 1}},
            reference=build_reference(gt_calls=[gt]),
            max_turns=1,
            expected_turns=1,
            license=license_name,
            meta_extra={"raw_index": index},
        )
        emitted += 1
        yield task

    logger.info("%s: %d samples (%d nested/unresolvable dropped)", SOURCE, emitted, n_nested)


__all__ = ["SOURCE", "LICENSE", "load_tasks"]
