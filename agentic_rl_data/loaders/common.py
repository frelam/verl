"""Shared, stdlib-only helpers for the ``agentic_rl_data.loaders`` package.

Everything here is deliberately framework-agnostic and import-light:

* raw-record readers (JSON / JSONL / parquet / optional HF ``datasets``),
* tool-schema / tool-call parsing (OpenAI JSON, Qwen XML, ``<TOOLCALL>``
  blocks and ToolACE's ``[func(k=v)]`` lists),
* a single schema-valid ``Task`` builder (schema v2.1, see ``schema.py`` and
  ``CONTRACTS.md`` C6/C9),
* conversation -> ``(messages, gt_calls, tool_returns, gt_turns)`` extraction
  used by every multi-turn loader,
* env config builders (``none`` / ``mock_fc`` / ``search`` / ``python`` /
  ``code_tests``).

Parsing logic is ported / adapted from ``examples/tool_rl/prepare_data.py``
and ``examples/tool_rl/prepare_data_v2.py`` (read-only reference; this module
does not import or modify them).  The important adaptation is the schema:
``messages`` may only use roles ``system`` / ``user`` (schema.py
``MESSAGE_ROLES``), so previous assistant / tool turns are folded into user
strings as ``<tool_response>`` blocks, exactly like the legacy pipeline does.

No heavy dependency is imported at module import time: ``datasets`` /
``pyarrow`` are imported lazily inside the reader helpers.
"""

from __future__ import annotations

import ast
import copy
import gzip
import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from agentic_rl_data.schema import (
    DEFAULT_MAX_TURNS,
    assert_valid_task,
    normalize_query,
)

logger = logging.getLogger(__name__)

# Generic system prompts. Tool schemas are never embedded in the prompt:
# they travel in ``task["tools"]`` and are rendered by the chat template.
GENERIC_TOOL_SYSTEM = (
    "You are a helpful assistant with access to tools. Use them when needed to answer user queries accurately."
)
NO_TOOL_SYSTEM = (
    "You are a helpful assistant. Determine if tools are needed for the user's request. "
    "If no suitable tool is available, answer directly or ask for the missing information."
)
SEARCH_SYSTEM = (
    "You are a helpful assistant that can search the web. Use the search tool to find relevant "
    "information and fetch a page when its snippet is not enough. Answer the question with a final answer."
)

# Canonical tool schemas used by the search / python / code-exec task types.
SEARCH_TOOL = {
    "name": "search",
    "description": "Search a local Wikipedia index and return the top matching snippets.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "top_k": {"type": "integer", "description": "Number of snippets to return.", "default": 3},
        },
        "required": ["query"],
    },
}
FETCH_TOOL = {
    "name": "fetch",
    "description": "Fetch the full text of a Wikipedia page by its document id.",
    "parameters": {
        "type": "object",
        "properties": {"doc_id": {"type": "string", "description": "Document id returned by search."}},
        "required": ["doc_id"],
    },
}
PYTHON_TOOL = {
    "name": "python",
    "description": "Execute Python code in a sandbox and return stdout / stderr. State does not persist between calls.",
    "parameters": {
        "type": "object",
        "properties": {"code": {"type": "string", "description": "Python code to run."}},
        "required": ["code"],
    },
}
SEARCH_TOOLS = [SEARCH_TOOL, FETCH_TOOL]

_MAX_RETURN_CHARS = 2000


class LoaderError(RuntimeError):
    """Raised when a loader cannot read its raw input at all."""


class SourceStats:
    """Per-source loader counters (CONTRACTS needs rows / emitted / skip reasons).

    Passed through ``cfg["_stats"]``; loaders call :func:`record_skip` on drop
    paths so ``build_raw.py`` can aggregate a data report instead of silently
    losing rows.
    """

    def __init__(self) -> None:
        self.rows = 0
        self.emitted = 0
        self.skipped_by_reason: dict[str, int] = {}

    def row(self) -> None:
        self.rows += 1

    def emit(self) -> None:
        self.emitted += 1

    def skip(self, reason: str) -> None:
        self.skipped_by_reason[reason] = self.skipped_by_reason.get(reason, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        skipped_total = sum(self.skipped_by_reason.values())
        unattributed = max(0, self.rows - self.emitted - skipped_total)
        return {
            "rows": self.rows,
            "emitted": self.emitted,
            "skipped_total": skipped_total,
            "unattributed_skips": unattributed,
            "skipped_by_reason": dict(sorted(self.skipped_by_reason.items())),
        }


def get_stats(cfg: Mapping[str, Any] | None) -> SourceStats | None:
    stats = (cfg or {}).get("_stats")
    return stats if isinstance(stats, SourceStats) else None


def record_skip(cfg: Mapping[str, Any] | None, reason: str) -> None:
    stats = get_stats(cfg)
    if stats is not None:
        stats.skip(reason)


# ---------------------------------------------------------------------------
# small value helpers
# ---------------------------------------------------------------------------


def as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def to_plain(value: Any) -> Any:
    """Recursively convert mappings / sequences to plain JSON-friendly values."""
    if isinstance(value, Mapping):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    if hasattr(value, "tolist"):  # numpy arrays / pandas values from parquet
        return value.tolist()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive
            return str(value)
    return value


def parse_json_maybe(value: Any) -> Any:
    """Parse a JSON/Python-literal string when it looks structured, else pass through."""
    if isinstance(value, (Mapping, list, tuple)):
        return to_plain(value)
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return to_plain(json.loads(text))
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        return to_plain(ast.literal_eval(text))
    except (ValueError, SyntaxError, TypeError):
        return value


def first_non_empty(*values: Any) -> Any:
    """Return the first value that is not ``None`` / empty string / empty container."""
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, dict)) and not value:
            continue
        return value
    return None


def truncate(text: Any, limit: int = _MAX_RETURN_CHARS) -> str:
    text = "" if text is None else str(text)
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def unwrap_text(value: Any) -> str:
    """Best-effort conversion of a message content (str / list-of-parts) to text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("content", "text", "value"):
            if key in value:
                return unwrap_text(value[key])
        return ""
    if isinstance(value, (list, tuple)):
        parts: list[str] = []
        for item in value:
            if isinstance(item, Mapping) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(unwrap_text(item))
        return "".join(parts)
    return str(value)


# ---------------------------------------------------------------------------
# raw input readers
# ---------------------------------------------------------------------------


def read_records(cfg: Mapping[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Yield raw source records from ``cfg``, counting ``cfg["_stats"].rows``.

    Supported cfg keys (in priority order):

    * ``records``  -- an in-memory list / dict of records (tests / fixtures),
    * ``path``     -- a file or directory (JSON / JSONL / parquet),
    * ``hf_dataset`` -- a HuggingFace dataset id (requires ``datasets``).

    Extra cfg keys: ``hf_config``, ``split`` (default ``train``), ``streaming``.
    """
    cfg = dict(cfg or {})
    stats = get_stats(cfg)
    for record in _iter_raw_records(cfg):
        if stats is not None:
            stats.row()
        yield record


def _iter_raw_records(cfg: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    if cfg.get("records") is not None:
        records = cfg["records"]
        if isinstance(records, Mapping):
            records = [records]
        for record in records:
            yield to_plain(record)
        return

    path = first_non_empty(cfg.get("path"), cfg.get("raw_path"), cfg.get("file"), cfg.get("dir"))
    if path is not None:
        yield from iter_path_records(Path(str(path)), cfg)
        return

    dataset = first_non_empty(cfg.get("hf_dataset"), cfg.get("dataset"))
    if dataset:
        yield from iter_hf_records(cfg)
        return

    raise LoaderError(
        "no raw input configured: pass cfg['records'], cfg['path'] or cfg['hf_dataset']"
    )


def iter_path_records(path: Path, cfg: Mapping[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Iterate JSON / JSONL (optionally ``.gz``) / parquet records under a path."""
    cfg = cfg or {}
    if path.is_dir():
        suffixes = {".json", ".jsonl", ".ndjson", ".parquet", ".gz"}
        files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix in suffixes)
        for file_path in files:
            yield from iter_path_records(file_path, cfg)
        return
    if not path.exists():
        raise LoaderError(f"raw input not found: {path}")

    name = path.name.lower()
    if name.endswith((".jsonl", ".ndjson")) or name.endswith(".jsonl.gz"):
        opener = gzip.open if name.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[arg-type]
            for lineno, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("%s:%d: invalid JSON line skipped (%s)", path, lineno, exc)
                    continue
                if isinstance(record, Mapping):
                    yield to_plain(record)
        return

    if name.endswith(".json"):
        with path.open("r", encoding="utf-8") as handle:
            try:
                payload = json.load(handle)
            except json.JSONDecodeError as exc:
                raise LoaderError(f"{path}: invalid JSON: {exc}") from exc
        record_list = _json_payload_records(payload)
        for record in record_list:
            yield to_plain(record)
        return

    if name.endswith(".parquet"):
        yield from _iter_parquet(path)
        return

    # Last resort: try JSONL, then whole-file JSON.
    try:
        yield from iter_path_records_text(path)
    except LoaderError:
        raise LoaderError(f"unsupported raw input format: {path}")


def _json_payload_records(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        # A dict of split -> list, or a single record.
        if "data" in payload and isinstance(payload["data"], list):
            return payload["data"]
        nested = [value for value in payload.values() if isinstance(value, list)]
        if len(nested) == 1 and len(nested[0]) > 0 and isinstance(nested[0][0], Mapping):
            return list(nested[0])
        return [payload]
    return []


def iter_path_records_text(path: Path) -> Iterator[dict[str, Any]]:
    """Fallback JSONL reader for extension-less files."""
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                raise LoaderError(f"{path}: not JSONL (line {lineno})")
            if isinstance(record, Mapping):
                yield to_plain(record)


def _iter_parquet(path: Path) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency
        raise LoaderError(
            f"{path}: reading parquet requires pyarrow (pip install pyarrow)"
        ) from exc
    table = pq.read_table(path)
    for record in table.to_pylist():
        if isinstance(record, Mapping):
            yield to_plain(record)


def iter_hf_records(cfg: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    """Stream records from a HuggingFace dataset (optional ``datasets`` dependency)."""
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency
        raise LoaderError(
            "reading a HF dataset requires `datasets` (pip install datasets); "
            "alternatively pass cfg['path'] to a local JSONL/parquet dump"
        ) from exc
    dataset = load_dataset(
        str(cfg.get("hf_dataset") or cfg.get("dataset")),
        cfg.get("hf_config"),
        split=str(cfg.get("split") or "train"),
        streaming=bool(cfg.get("streaming", False)),
    )
    for record in dataset:
        yield to_plain(record)


# ---------------------------------------------------------------------------
# tool schema parsing
# ---------------------------------------------------------------------------


def normalize_tools(tools: Any) -> list[dict[str, Any]]:
    """Normalise any source tool list to OpenAI-style ``{name, description, parameters}``.

    Accepts OpenAI function objects, bare ``{name, parameters}`` dicts, plain
    parameter-name maps (wrapped into a JSON schema) and JSON strings.  Names
    are de-duplicated (first occurrence wins) because ``validate_task`` rejects
    duplicate tool names.
    """
    tools = parse_json_maybe(tools)
    if isinstance(tools, Mapping):
        tools = [tools]
    if not isinstance(tools, list):
        return []
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for tool in tools:
        schema = normalize_tool(tool)
        if schema is None or schema["name"] in seen:
            continue
        seen.add(schema["name"])
        result.append(schema)
    return result


def normalize_tool(tool: Any) -> dict[str, Any] | None:
    if not isinstance(tool, Mapping):
        return None
    fn = tool.get("function") if isinstance(tool.get("function"), Mapping) else tool
    name = fn.get("name") or tool.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    description = fn.get("description", tool.get("description", ""))
    parameters = fn.get("parameters", fn.get("input_parameters", tool.get("parameters")))
    parameters = parse_json_maybe(parameters)
    if not isinstance(parameters, Mapping):
        parameters = {}
    parameters = dict(parameters)
    if "properties" not in parameters:
        # A plain {param_name: {...}} map -> wrap into a JSON schema.
        properties = {str(key): value for key, value in parameters.items() if isinstance(value, Mapping)}
        parameters = {"type": "object", "properties": properties}
    return {
        "name": name.strip(),
        "description": "" if description is None else str(description),
        "parameters": to_plain(parameters),
    }


# ---------------------------------------------------------------------------
# tool-call parsing
# ---------------------------------------------------------------------------

_QWEN_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
_QWEN_FUNC_RE = re.compile(r"<function=([^>\n]+?)>", re.DOTALL)
_QWEN_PARAM_RE = re.compile(r"<parameter=(\w+)>(.*?)</parameter>", re.DOTALL)
_TOOLCALL_TAG_RE = re.compile(r"<TOOLCALL>(.*?)</TOOLCALL>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def normalize_call(raw: Any) -> dict[str, Any] | None:
    """Normalise one call object / JSON string to ``{"name", "arguments"}``."""
    if isinstance(raw, str):
        raw = parse_json_maybe(raw)
    if not isinstance(raw, Mapping):
        return None
    fn = raw.get("function") if isinstance(raw.get("function"), Mapping) else raw
    name = fn.get("name") or raw.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    arguments = fn.get("arguments", raw.get("arguments", raw.get("parameters")))
    arguments = parse_json_maybe(arguments)
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, Mapping):
        arguments = {}
    return {"name": name.strip(), "arguments": to_plain(dict(arguments))}


def normalize_calls(raw_calls: Any) -> list[dict[str, Any]]:
    raw_calls = parse_json_maybe(raw_calls)
    if isinstance(raw_calls, Mapping):
        raw_calls = [raw_calls]
    if not isinstance(raw_calls, list):
        return []
    calls = [normalize_call(item) for item in raw_calls]
    return [call for call in calls if call is not None]


def parse_tool_calls(text: Any) -> list[dict[str, Any]]:
    """Parse assistant text in any of the supported call formats.

    Order: Qwen XML -> ``<TOOLCALL>`` JSON block -> bare JSON object -> ToolACE
    ``[func(k=v)]`` list.  Returns ``[]`` when the text contains no call.
    """
    text = unwrap_text(text)
    if not text.strip():
        return []
    for parser in (
        _parse_qwen_calls,
        _parse_toolcall_tag_calls,
        _parse_json_calls,
        _parse_toolace_call_list,
    ):
        calls = parser(text)
        if calls:
            return calls
    return []


def _parse_value(raw: str) -> Any:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError, TypeError):
        return raw.strip("'\"")


def _parse_qwen_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for block_match in _QWEN_CALL_RE.finditer(text):
        block = block_match.group(1)
        func_match = _QWEN_FUNC_RE.search(block)
        if not func_match:
            continue
        name = func_match.group(1).strip()
        arguments: dict[str, Any] = {}
        for param_match in _QWEN_PARAM_RE.finditer(block):
            arguments[param_match.group(1)] = _parse_value(param_match.group(2))
        if name:
            calls.append({"name": name, "arguments": arguments})
    return calls


def _parse_toolcall_tag_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for block_match in _TOOLCALL_TAG_RE.finditer(text):
        payload = parse_json_maybe(block_match.group(1).strip())
        calls.extend(normalize_calls(payload))
    return calls


def _iter_json_objects(text: str) -> Iterator[str]:
    """Yield balanced ``{...}`` substrings (string-literal aware)."""
    depth = 0
    start = -1
    in_string = False
    quote = ""
    index = 0
    while index < len(text):
        char = text[index]
        if in_string:
            if char == "\\":
                index += 2
                continue
            if char == quote:
                in_string = False
        elif char in "\"'":
            in_string = True
            quote = char
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                yield text[start : index + 1]
                start = -1
        index += 1


def _parse_json_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for block in _iter_json_objects(text):
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            continue
        calls.extend(normalize_calls(payload))
    return calls


def _split_top_level(text: str, sep: str = ",") -> list[str]:
    parts: list[str] = []
    depth = 0
    in_string = False
    quote = ""
    start = 0
    index = 0
    while index < len(text):
        char = text[index]
        if in_string:
            if char == "\\":
                index += 2
                continue
            if char == quote:
                in_string = False
        elif char in "\"'":
            in_string = True
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == sep and depth == 0:
            parts.append(text[start:index])
            start = index + 1
        index += 1
    parts.append(text[start:])
    return [part.strip() for part in parts if part.strip()]


def _parse_toolace_call_list(text: str) -> list[dict[str, Any]]:
    """Parse ToolACE's ``[func(k=v, ...)]`` assistant call list."""
    text = text.strip()
    if not text.startswith("[") or "(" not in text:
        return []
    start = text.find("[")
    end = text.rfind("]")
    inner = text[start + 1 : end if end > start else len(text)]
    calls: list[dict[str, Any]] = []
    for segment in _split_top_level(inner):
        match = re.match(r"(?s)([^(){}]+?)\s*\((.*)\)\s*$", segment)
        if not match:
            continue
        name = match.group(1).strip().strip("'\"")
        if not name:
            continue
        arguments: dict[str, Any] = {}
        for arg_segment in _split_top_level(match.group(2)):
            if "=" not in arg_segment:
                continue
            key, value = arg_segment.split("=", 1)
            arguments[key.strip()] = _parse_value(value)
        calls.append({"name": name, "arguments": arguments})
    return calls


def extract_answer(text: Any) -> str | None:
    """Last ``<answer>...</answer>`` content, else ``None``."""
    text = unwrap_text(text)
    matches = _ANSWER_RE.findall(text)
    if matches:
        answer = matches[-1].strip()
        return answer or None
    return None


# ---------------------------------------------------------------------------
# env config builders
# ---------------------------------------------------------------------------


def env_none(max_turns: int = 1) -> dict[str, Any]:
    return {"type": "none", "config": {"max_turns": int(max_turns)}}


def env_search(
    max_turns: int = 8,
    retriever: Mapping[str, Any] | None = None,
    *,
    backend: str = "e5",
    url: str | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    """Canonical search env config (CONTRACTS C11 / D12).

    ``{"backend": "e5"|"inmemory"|"http", "url": str|null, "top_k": int,
    "max_rounds": int, "retriever": {...}|null}``.  Legacy aliases
    (``backend="local_wikipedia_e5"``, ``retriever={"url","topk"}``) are
    normalised here so SearchEnv never receives an unknown key.
    """
    if backend not in {"e5", "inmemory", "http"}:
        backend = "e5"
    extra_retriever = dict(retriever or {})
    url = url or extra_retriever.pop("url", None) or None
    top_k = int(extra_retriever.pop("top_k", extra_retriever.pop("topk", top_k)))
    config: dict[str, Any] = {
        "backend": backend,
        "url": url,
        "top_k": top_k,
        "max_rounds": int(max_turns),
        "retriever": extra_retriever or None,
    }
    return {"type": "search", "config": config}


def env_python(max_turns: int = 8) -> dict[str, Any]:
    return {
        "type": "python",
        "config": {"max_turns": int(max_turns), "sandbox": "subprocess", "network": False, "timeout_s": 30},
    }


def env_code_tests(max_turns: int = 15, *, redact: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "code_tests",
        "config": {
            "max_turns": int(max_turns),
            "sandbox": "subprocess",
            "network": False,
            "timeout_s": 30,
            "redact": list(redact or []),
        },
    }


def env_mock_fc(
    max_turns: int = DEFAULT_MAX_TURNS,
    *,
    impl: str = "replay",
    final_state: Mapping[str, Any] | None = None,
    gt_returns: list[list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """``mock_fc`` env config (CONTRACTS C1/C6).

    The canonical replay source is ``reference["tool_returns"]``; ``config``
    only carries the hand-override hook ``gt_returns`` (omitted by default) and
    the stateful-env terminal state.
    """
    config: dict[str, Any] = {"impl": impl, "max_turns": int(max_turns)}
    if impl == "stateful":
        config["domain"] = "synthetic"
        config["final_state"] = dict(final_state or {})
    if gt_returns:
        config["gt_returns"] = to_plain(gt_returns)
    return {"type": "mock_fc", "config": config}


# ---------------------------------------------------------------------------
# Task builder (schema v2.1)
# ---------------------------------------------------------------------------


def build_reference(**overrides: Any) -> dict[str, Any]:
    """Return a ``reference`` mapping with all 7 required keys present."""
    reference: dict[str, Any] = {
        "final_answer": None,
        "final_answers": None,
        "gt_calls": [],
        "gt_turns": None,
        "tool_returns": None,
        "unit_tests": None,
        "final_state": None,
    }
    reference.update(overrides)
    if reference.get("gt_calls") is None:
        reference["gt_calls"] = []
    return reference


def build_task(
    *,
    source: str,
    task_id: str,
    task_type: str,
    messages: Iterable[Mapping[str, Any]],
    tools: Any,
    env: Mapping[str, Any],
    reference: Mapping[str, Any] | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    expected_turns: int | None = None,
    difficulty: Any = None,
    pass_rate: Any = None,
    license: str | None = None,
    meta_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble + validate one task; raises ``AssertionError`` style error on bug."""
    reference_map = build_reference(**dict(reference or {}))
    meta: dict[str, Any] = {
        "max_turns": int(max_turns),
        "difficulty": difficulty,
        "pass_rate": pass_rate,
        "license": license,
    }
    if expected_turns is not None:
        meta["expected_turns"] = int(expected_turns)
    if meta_extra:
        meta.update({str(key): to_plain(value) for key, value in meta_extra.items()})
    task = {
        "id": str(task_id),
        "source": str(source),
        "task_type": str(task_type),
        "messages": [{"role": str(m.get("role")), "content": str(m.get("content"))} for m in messages],
        "tools": normalize_tools(tools),
        "env": {"type": str(env.get("type")), "config": to_plain(dict(env.get("config") or {}))},
        "reference": reference_map,
        "meta": meta,
    }
    errors = _validate_task_errors(task)
    if task["task_type"] in {"st_fc", "mt_fc"} and not task["tools"]:
        # A function-calling task without a declared tool menu is unreplayable /
        # unanswerable (T1-07): fail loudly instead of emitting a dead sample.
        errors.append("function-calling task requires a non-empty tools list")
    if errors:
        raise ValueError(f"invalid task from source={source!r} id={task_id!r}: " + "; ".join(errors))
    return task


def _validate_task_errors(task: Mapping[str, Any]) -> list[str]:
    from agentic_rl_data.schema import validate_task

    return validate_task(task)


def build_messages(system: str | None, user_texts: Iterable[str]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    for text in user_texts:
        text = unwrap_text(text).strip()
        if text:
            messages.append({"role": "user", "content": text})
    return messages


# ---------------------------------------------------------------------------
# conversation -> trajectory extraction (multi-turn loaders)
# ---------------------------------------------------------------------------

_ROLE_MAP = {
    "system": "system",
    "user": "user",
    "human": "user",
    "prompt": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "ai": "assistant",
    "model": "assistant",
    "tool": "tool",
    "function": "tool",
    "observation": "tool",
    "tool_response": "tool",
    "ipython": "tool",
}


def _entry_role(entry: Mapping[str, Any]) -> str:
    for key in ("role", "from", "speaker"):
        if key in entry:
            value = str(entry[key]).strip().lower()
            return _ROLE_MAP.get(value, "other")
    return "other"


def _entry_text(entry: Mapping[str, Any]) -> str:
    for key in ("content", "value", "text"):
        if key in entry:
            return unwrap_text(entry[key])
    return ""


def _entry_calls(entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in ("tool_calls", "tool_calls_json", "calls", "function_call"):
        if key in entry and entry[key]:
            raw = entry[key]
            if key == "function_call" and isinstance(raw, Mapping):
                raw = [raw]
            calls = normalize_calls(raw)
            if calls:
                return calls
    # Fall back to parsing the assistant text (Qwen XML / JSON / ToolACE).
    return parse_tool_calls(_entry_text(entry))


def conversation_entries(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Locate the conversation list in a record across common source layouts."""
    for key in ("messages", "conversations", "conversation", "trajectory", "dialogues", "turns"):
        value = record.get(key)
        if isinstance(value, list) and value and isinstance(value[0], Mapping):
            return [item for item in value if isinstance(item, Mapping)]
    return []


def extract_trajectory(record: Mapping[str, Any]) -> dict[str, Any]:
    """Turn a raw SFT conversation record into a normalised trajectory view.

    Returns::

        {"system": str|None,
         "user_texts": [str],          # messages to render (system/user only)
         "turns": [{"user", "assistant", "calls", "returns"}],
         "gt_calls": [[call, ...]],    # per assistant turn that has calls
         "tool_returns": [[{name, content, ok}], ...] or None,
         "gt_turns": [...] or None,
         "expected_turns": int,
         "final_answer": str|None}

    ``tool_returns`` is ``None`` (placeholder / degraded) unless *every*
    ground-truth call group has at least one recorded return.
    """
    entries = conversation_entries(record)
    if not entries:
        raise LoaderError("record has no messages/conversations list")

    system: str | None = None
    turns: list[dict[str, Any]] = []
    pending_users: list[str] = []
    last_assistant_text = ""

    def _start_turn(assistant_text: str, calls: list[dict[str, Any]]) -> None:
        nonlocal pending_users
        user_text = "\n".join(pending_users).strip()
        if not user_text and not calls and not assistant_text.strip():
            return
        if not user_text and not turns:
            user_text = ""
        turns.append(
            {
                "user": user_text,
                "assistant": assistant_text,
                "calls": calls,
                "returns": [],
            }
        )
        pending_users = []

    for entry in entries:
        role = _entry_role(entry)
        text = _entry_text(entry)
        if role == "system":
            system = text.strip() or system
        elif role == "user":
            pending_users.append(text.strip())
        elif role == "assistant":
            calls = _entry_calls(entry)
            _start_turn(text, calls)
            last_assistant_text = text
        elif role == "tool":
            if turns:
                tool_name = ""
                for key in ("name", "tool_name", "function"):
                    if entry.get(key):
                        tool_name = str(entry[key])
                        break
                if not tool_name and turns[-1]["calls"]:
                    index = len(turns[-1]["returns"])
                    call_list = turns[-1]["calls"]
                    if index < len(call_list):
                        tool_name = str(call_list[index].get("name", ""))
                content = _entry_text(entry)
                turns[-1]["returns"].append(
                    {
                        "name": tool_name or "unknown_tool",
                        "content": truncate(content),
                        "ok": bool(entry.get("ok", True)),
                    }
                )
        # "other" roles are ignored.

    if pending_users and turns:
        # Trailing user message with no assistant answer yet: render it too.
        turns.append({"user": "\n".join(pending_users).strip(), "assistant": "", "calls": [], "returns": []})
        pending_users = []

    if not turns:
        raise LoaderError("conversation has no user/assistant turns")

    user_texts: list[str] = []
    for turn in turns:
        if turn["user"]:
            # The prompt only uses schema-legal roles (system / user); previous
            # tool output is folded in as a ``<tool_response>`` user message.
            user_texts.append(turn["user"])
        if turn["returns"]:
            response_text = "\n".join(
                f"<tool_response>\n{ret['content']}\n</tool_response>" for ret in turn["returns"]
            )
            if user_texts:
                user_texts.append(response_text)
    # Guarantee at least one user message.
    if not user_texts:
        user_texts = ["Proceed with the task."]

    gt_calls = [turn["calls"] for turn in turns if turn["calls"]]
    tool_return_groups: list[list[dict[str, Any]]] = []
    complete_returns = True
    for turn in turns:
        if not turn["calls"]:
            continue
        returns = turn["returns"]
        if not returns:
            complete_returns = False
        tool_return_groups.append(
            [
                {"name": ret["name"], "content": ret["content"], "ok": bool(ret.get("ok", True))}
                for ret in returns
            ]
        )

    gt_turns = []
    for turn in turns:
        if not turn["calls"] and not turn["returns"]:
            continue
        gt_turns.append(
            {
                "user": truncate(turn["user"], 500),
                "assistant": truncate(turn["assistant"], 500),
                "gt_calls": turn["calls"],
                "tool_returns": turn["returns"],
            }
        )

    final_answer = extract_answer(last_assistant_text)
    if final_answer is None:
        final_answer = first_non_empty(
            record.get("final_answer"),
            record.get("answer"),
            record.get("gold_answer"),
        )
        if isinstance(final_answer, (list, tuple)):
            final_answer = str(final_answer[0]) if final_answer else None
        elif final_answer is not None:
            final_answer = str(final_answer)

    expected_turns = len(turns)
    for key in ("expected_turns", "num_turns", "turn_count"):
        value = record.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            expected_turns = value
            break

    return {
        "system": system,
        "user_texts": user_texts,
        "turns": turns,
        "gt_calls": gt_calls,
        "tool_returns": tool_return_groups if complete_returns and tool_return_groups else None,
        "gt_turns": gt_turns or None,
        "expected_turns": expected_turns,
        "final_answer": final_answer,
    }


def load_mt_tasks_from_records(
    records: Iterable[Mapping[str, Any]],
    *,
    source: str,
    cfg: Mapping[str, Any] | None = None,
    default_license: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Convert raw SFT conversations into ``mt_fc`` tasks (replay env).

    ``cfg`` keys: ``max_turns`` (default 15, hard-capped at 15), ``max_samples``
    (default 0 = all), ``id_field`` / ``id_prefix``, ``license``, ``impl``
    (``replay`` default).  ``tool_returns`` is set when the record contains real
    tool observations, otherwise it is ``None`` and ``meta["gt_returns_source"]``
    is ``"placeholder"`` so downstream mixing can filter / down-weight it.
    """
    cfg = dict(cfg or {})
    max_samples = int(cfg.get("max_samples") or 0)
    prefix = str(cfg.get("id_prefix") or source)
    license_name = cfg.get("license", default_license)
    emitted = 0
    for index, record in enumerate(records):
        if max_samples and emitted >= max_samples:
            return
        try:
            view = extract_trajectory(record)
            turn_estimate = max(
                len(view["gt_calls"]),
                int(view.get("expected_turns") or 0),
                len(view.get("gt_turns") or []),
            )
            # meta.max_turns is the RL turn budget, hard-capped at 15 (T1-05).
            # Long SFT traces are not silently truncated: expected_turns carries
            # the estimate so filter_rounds.py (C9) drops them with a count.
            max_turns_value = min(int(cfg.get("max_turns") or DEFAULT_MAX_TURNS), DEFAULT_MAX_TURNS)
            env = env_mock_fc(
                max_turns=max_turns_value,
                impl=str(cfg.get("impl") or "replay"),
                gt_returns=cfg.get("gt_returns") or None,
            )
            reference = build_reference(
                final_answer=view.get("final_answer"),
                gt_calls=view["gt_calls"],
                gt_turns=_compact_gt_turns(view.get("gt_turns")),
                tool_returns=view.get("tool_returns"),
                final_state=cfg.get("final_state"),
            )
            # Frozen enum (CONTRACTS C6/D11): config | reference | gt_turns | placeholder.
            # The canonical source is reference["tool_returns"]; gt_turns is the
            # legacy fallback (partial returns) and placeholder means "no real
            # observation at all" (T2 filters / down-weights those).
            if cfg.get("gt_returns"):
                gt_source = "config"
            elif view.get("tool_returns"):
                gt_source = "reference"
            elif _gt_turns_have_returns(view.get("gt_turns")):
                gt_source = "gt_turns"
            else:
                gt_source = "placeholder"
            meta_extra = {
                "gt_returns_source": gt_source,
                "raw_index": index,
            }
            group_id = first_non_empty(
                record.get(str(cfg.get("id_field") or "id")), record.get("conversation_id")
            )
            if group_id is not None:
                meta_extra["group_id"] = str(group_id)
            if cfg.get("difficulty") is not None:
                meta_extra["difficulty"] = cfg.get("difficulty")
            task = build_task(
                source=source,
                task_id=str(record.get(str(cfg.get("id_field") or "id") or f"{prefix}-{index}")),
                task_type="mt_fc",
                messages=build_messages(
                    view.get("system") or GENERIC_TOOL_SYSTEM, view["user_texts"]
                ),
                tools=first_non_empty(record.get("tools"), record.get("functions"), []),
                env=env,
                reference=reference,
                max_turns=max_turns_value,
                expected_turns=int(view.get("expected_turns") or 0) or None,
                license=license_name,
                meta_extra=meta_extra,
            )
        except Exception as exc:  # noqa: BLE001 - one bad row must not kill the source
            logger.warning("%s: record %d skipped: %s", source, index, exc)
            record_skip(cfg, "invalid_or_unparseable_record")
            continue
        emitted += 1
        yield task


def _compact_gt_turns(gt_turns: Any) -> list[dict[str, Any]] | None:
    """Normalise ``gt_turns`` to the CONTRACTS C6 shape (schema v2.2).

    Shape: ``{"step": int, "assistant": {"content": str, "tool_calls": [...]},
    "tool_results": [{"name", "content", "ok"}]}``.  Every element is a
    mapping (schema v2.2 validates that).
    """
    if not isinstance(gt_turns, list):
        return None
    compact: list[dict[str, Any]] = []
    for index, turn in enumerate(gt_turns, 1):
        if not isinstance(turn, Mapping):
            continue
        compact.append(
            {
                "step": index,
                "assistant": {
                    "content": truncate(turn.get("assistant"), 500),
                    "tool_calls": to_plain(turn.get("gt_calls") or []),
                },
                "tool_results": [
                    {
                        "name": str(ret.get("name") or "unknown_tool"),
                        "content": str(ret.get("content") or ""),
                        "ok": bool(ret.get("ok", True)),
                    }
                    for ret in (turn.get("tool_returns") or [])
                    if isinstance(ret, Mapping)
                ],
            }
        )
    return compact or None


def _gt_turns_have_returns(gt_turns: Any) -> bool:
    if not isinstance(gt_turns, list):
        return False
    for turn in gt_turns:
        if isinstance(turn, Mapping) and any(
            isinstance(ret, Mapping) for ret in (turn.get("tool_returns") or [])
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# generic record-field helpers used by several loaders
# ---------------------------------------------------------------------------


def first_user_text(record: Mapping[str, Any]) -> str:
    """Best-effort user question extraction across source layouts."""
    question = first_non_empty(
        record.get("question"),
        record.get("query"),
        record.get("prompt"),
        record.get("problem"),
        record.get("instruction"),
    )
    if isinstance(question, list):
        for entry in question:
            if isinstance(entry, Mapping) and _entry_role(entry) == "user":
                return _entry_text(entry).strip()
        return ""
    if isinstance(question, str) and question.strip():
        return question.strip()
    entries = conversation_entries(record)
    for entry in entries:
        if _entry_role(entry) == "user":
            return _entry_text(entry).strip()
    return ""


def gold_answers(record: Mapping[str, Any]) -> list[str]:
    """Extract one or more gold answers (``final_answers``) from a record."""
    candidates: list[Any] = []
    reward_model = as_mapping(record.get("reward_model"))
    if reward_model.get("ground_truth") is not None:
        candidates.append(reward_model["ground_truth"])
    for key in ("golden_answers", "answers", "answer", "final_answer", "label", "solution", "gt"):
        if record.get(key) is not None:
            candidates.append(record[key])
    answers: list[str] = []
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            nested = first_non_empty(
                candidate.get("target"), candidate.get("answer"), candidate.get("aliases")
            )
            if nested is not None:
                candidate = nested
            else:
                continue
        if isinstance(candidate, (list, tuple)):
            for item in candidate:
                if isinstance(item, Mapping):
                    value = first_non_empty(item.get("answer"), item.get("text"))
                    if value is not None:
                        answers.append(str(value).strip())
                elif item is not None and str(item).strip():
                    answers.append(str(item).strip())
        elif candidate is not None and str(candidate).strip():
            answers.append(str(candidate).strip())
    # Stable de-duplication, keep order.
    seen: set[str] = set()
    unique: list[str] = []
    for answer in answers:
        if answer in seen:
            continue
        seen.add(answer)
        unique.append(answer)
    return unique


def safe_task_id(prefix: str, value: Any, index: int) -> str:
    raw = first_non_empty(value)
    if raw is None:
        return f"{prefix}-{index}"
    text = re.sub(r"\s+", "-", str(raw).strip())
    text = re.sub(r"[^0-9A-Za-z._:\-]", "_", text)
    return f"{prefix}-{text}" if not text.startswith(prefix) else text


def iter_source_records(cfg: Mapping[str, Any] | None, source: str) -> Iterator[dict[str, Any]]:
    try:
        yield from read_records(cfg)
    except LoaderError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise LoaderError(f"{source}: failed to read raw input: {exc}") from exc


def dedup_keys_ready(tasks: Iterable[Mapping[str, Any]]) -> list[str]:
    """Small helper for scripts / tests: exact dedup keys in input order."""
    from agentic_rl_data.schema import task_dedup_key

    return [task_dedup_key(task) for task in tasks]


__all__ = [
    "GENERIC_TOOL_SYSTEM",
    "NO_TOOL_SYSTEM",
    "SEARCH_SYSTEM",
    "SEARCH_TOOLS",
    "PYTHON_TOOL",
    "LoaderError",
    "read_records",
    "iter_path_records",
    "iter_hf_records",
    "normalize_tools",
    "normalize_calls",
    "parse_tool_calls",
    "extract_answer",
    "env_none",
    "env_search",
    "env_python",
    "env_code_tests",
    "env_mock_fc",
    "build_reference",
    "build_task",
    "build_messages",
    "extract_trajectory",
    "load_mt_tasks_from_records",
    "first_user_text",
    "gold_answers",
    "safe_task_id",
    "as_mapping",
    "as_list",
    "to_plain",
    "parse_json_maybe",
    "first_non_empty",
    "truncate",
    "unwrap_text",
]
