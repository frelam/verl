"""API-Bank loader (level-1 dialogues -> single-turn samples).

Raw source: ``AlibabaResearch/DAMO-ConvAI`` (MIT), toolkit under ``api-bank/``.
Only ``api-bank/lv1-lv2-samples/level-1-given-desc/*.jsonl`` plus the API
classes under ``api-bank/apis/*.py`` are used, mirroring
``examples/tool_rl/prepare_data.py::load_apibank``.  Each API turn becomes one
``st_fc`` sample; preceding User/AI turns are flattened to schema-legal
``system`` / ``user`` messages and previous API output is injected as a
``<tool_response>`` user message.  A few distractor schemas keep tool selection
non-trivial.

cfg keys: ``api_dir`` (directory holding the ``*.py`` API definitions, or
``api_files`` list), ``dialogues_path`` (JSONL file or directory), optional
``n_distractors`` (default 5), ``seed`` (default 42), ``max_samples``,
``license``.
"""

from __future__ import annotations

import ast
import json
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

SOURCE = "apibank"
LICENSE = "mit"

_N_DISTRACTORS = 5
_TYPE_MAP = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
}


def _parse_apibank_api_py(source: str) -> dict[str, Any] | None:
    """Extract a tool schema from one API-Bank API class definition."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        description = ""
        input_params: dict[str, Any] = {}
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            target = stmt.targets[0] if stmt.targets else None
            if not isinstance(target, ast.Name):
                continue
            if target.id == "description":
                try:
                    description = ast.literal_eval(stmt.value)
                except (ValueError, SyntaxError):
                    pass
            elif target.id == "input_parameters":
                try:
                    input_params = ast.literal_eval(stmt.value)
                except (ValueError, SyntaxError):
                    pass
        if not description and not input_params:
            continue
        properties: dict[str, Any] = {}
        for param_name, param_info in (input_params or {}).items():
            if not isinstance(param_info, dict):
                continue
            properties[str(param_name)] = {
                "type": _TYPE_MAP.get(str(param_info.get("type", "str")).lower(), "string"),
                "description": str(param_info.get("description", "")),
            }
        return {
            "name": node.name,
            "description": str(description),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": sorted(properties),
            },
        }
    return None


def _load_api_schemas(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    api_dir = first_non_empty(cfg.get("api_dir"), cfg.get("apis_dir"))
    api_files = cfg.get("api_files")
    files: list[Path] = []
    if api_files:
        files = [Path(str(path)) for path in api_files]
    elif api_dir:
        directory = Path(str(api_dir))
        if not directory.is_dir():
            raise LoaderError(f"{SOURCE}: api_dir is not a directory: {directory}")
        files = sorted(
            path
            for path in directory.glob("*.py")
            if path.name not in {"__init__.py", "api.py", "tool_search.py"}
        )
    if not files:
        raise LoaderError(
            f"{SOURCE}: configure cfg['api_dir'] (local copy of api-bank/apis) "
            "or cfg['api_files']"
        )
    schemas: dict[str, dict[str, Any]] = {}
    for path in files:
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("%s: cannot read %s: %s", SOURCE, path, exc)
            continue
        schema = _parse_apibank_api_py(source)
        if schema:
            schemas[schema["name"]] = schema
    return schemas


def _pick_distractors(
    pool: list[dict[str, Any]], exclude: set[str], n: int, rng: random.Random
) -> list[dict[str, Any]]:
    candidates = [tool for tool in pool if tool.get("name") not in exclude]
    rng.shuffle(candidates)
    return candidates[:n]


def load_tasks(cfg: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Yield single-turn ``st_fc`` samples from API-Bank level-1 dialogues.

    One sample is emitted per API turn, with the preceding dialogue as context
    (User turns kept; API results injected as ``<tool_response>`` user
    messages; AI text dropped because the schema forbids the assistant role).
    """
    cfg = dict(cfg or {})
    max_samples = int(cfg.get("max_samples") or 0)
    license_name = cfg.get("license", LICENSE)
    dialogues_path = first_non_empty(cfg.get("dialogues_path"), cfg.get("path"))
    if dialogues_path is None:
        raise LoaderError(
            f"{SOURCE}: configure cfg['dialogues_path'] (local copy of "
            "api-bank/lv1-lv2-samples/level-1-given-desc)"
        )
    schemas = _load_api_schemas(cfg)
    if not schemas:
        raise LoaderError(f"{SOURCE}: no API schemas parsed")
    pool = list(schemas.values())
    rng = random.Random(int(cfg.get("seed", 42)))
    n_distractors = int(cfg.get("n_distractors", _N_DISTRACTORS))
    emitted = 0
    skipped_turns = 0

    dialogue_path = Path(str(dialogues_path))
    if dialogue_path.is_dir():
        files = sorted(dialogue_path.rglob("*.jsonl"))
    else:
        files = [dialogue_path]

    for path in files:
        history: list[str] = []
        n_calls = 0
        for line in _iter_jsonl_lines(path):
            if max_samples and emitted >= max_samples:
                return
            if not isinstance(line, dict):
                continue
            role = str(line.get("role", ""))
            text = str(line.get("text", "") or "").strip()
            if role == "User":
                history.append(text)
                continue
            if role == "AI":
                # Assistant text is not prompt-legal; keep it out of the prompt.
                continue
            if role != "API":
                continue
            api_name = str(line.get("api_name", ""))
            params = line.get("param_dict") or {}
            output = line.get("result")
            if api_name in schemas and isinstance(params, dict) and history:
                n_calls += 1
                gt = [{"name": api_name, "arguments": dict(params)}]
                tools = [schemas[api_name]] + _pick_distractors(pool, {api_name}, n_distractors, rng)
                rng.shuffle(tools)
                context = history[-8:]
                task = build_task(
                    source=SOURCE,
                    task_id=f"{SOURCE}-{path.stem}-c{n_calls}",
                    task_type="st_fc",
                    messages=build_messages(GENERIC_TOOL_SYSTEM, context),
                    tools=tools,
                    env={"type": "none", "config": {"max_turns": 1}},
                    reference=build_reference(gt_calls=[gt]),
                    max_turns=1,
                    expected_turns=1,
                    license=license_name,
                    meta_extra={
                        "raw_index": n_calls,
                        "dialogue": path.name,
                        "distractors": len(tools) - 1,
                    },
                )
                emitted += 1
                yield task
            else:
                skipped_turns += 1
                record_skip(cfg, "unresolved_api_turn")
            if isinstance(output, dict):
                output = output.get("output")
            try:
                output_text = json.dumps(output, ensure_ascii=False)
            except TypeError:
                output_text = str(output)
            history.append(f"<tool_response>\n{output_text[:500]}\n</tool_response>")

    logger.info("%s: %d samples (%d API turns skipped)", SOURCE, emitted, skipped_turns)


def _iter_jsonl_lines(path: Path) -> Iterator[Any]:
    if not path.exists():
        raise LoaderError(f"{SOURCE}: dialogues file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("%s:%d: invalid JSON line skipped", path, lineno)


__all__ = ["SOURCE", "LICENSE", "load_tasks"]
