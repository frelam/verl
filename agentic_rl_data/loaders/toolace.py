"""ToolACE loader (multi-turn conversations split into single-turn samples).

Raw source: ``Team-ACE/ToolACE``.  Each record has a ``system`` prompt that
embeds tool schemas as JSON plus a ``conversations`` list of
``{"from": "user"|"assistant"|"tool", "value": ...}`` turns.  Parsing mirrors
``examples/tool_rl/prepare_data.py::load_toolace``: every user turn becomes one
``st_fc`` sample whose label is the next assistant turn's tool calls.

The schema only allows ``system`` / ``user`` roles, so earlier turns are
flattened into user-visible context: prior user text is kept as user messages
and prior tool outputs are wrapped in ``<tool_response>`` blocks (same wire
format as the legacy verl pipeline).  Prior assistant reasoning is dropped
deliberately: it must not leak the SFT trace into an RL prompt.

ToolACE writes assistant calls as ``[func(k=v, ...)]``.  A guard refuses to
continue when >5% of call-shaped assistant turns fail to parse, because
emitting them as ``gt_calls=[]`` would mislabel positives as "no tools needed"
(this mirrors the legacy guard rail).

License note: HF card unreachable from this sandbox -> ``[UNVERIFIED]``.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any, Iterator, Mapping

from agentic_rl_data.loaders.common import (
    GENERIC_TOOL_SYSTEM,
    LoaderError,
    build_messages,
    build_reference,
    build_task,
    iter_source_records,
    normalize_calls,
    normalize_tools,
    parse_tool_calls,
    record_skip,
    safe_task_id,
)

logger = logging.getLogger(__name__)

SOURCE = "toolace"
LICENSE = "apache-2.0[UNVERIFIED]"

_CALL_SHAPED_RE = re.compile(r"^\s*\[.*\(\s*", re.DOTALL)


def _iter_balanced_json(text: str) -> Iterator[str]:
    """Yield every balanced ``[...]`` / ``{...}`` block, string-aware.

    A greedy regex is useless here: ToolACE system prompts embed the tool list
    as JSON and then continue with prose that may itself contain brackets, so
    ``re.search(r"\\[.*\\]", DOTALL)`` spans past the JSON and never parses.
    """
    index = 0
    length = len(text)
    while index < length:
        if text[index] not in "[{":
            index += 1
            continue
        depth = 0
        in_string = False
        escaped = False
        cursor = index
        closed = False
        while cursor < length:
            char = text[cursor]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char in "[{":
                depth += 1
            elif char in "]}":
                depth -= 1
                if depth == 0:
                    closed = True
                    break
            cursor += 1
        if closed:
            yield text[index : cursor + 1]
            index = cursor + 1
        else:
            index += 1


def _extract_tools_from_text(text: str) -> list[dict[str, Any]]:
    """Extract tool definitions embedded in ToolACE's system prompt text.

    Two layouts ship in the public dump: ``{name, description, parameters}`` and
    ``{tool_name, definition, arguments: [...]}``; ``normalize_tools`` converts
    both (see ``common.normalize_tool``).
    """
    for blob in _iter_balanced_json(text):
        candidate: Any = None
        try:
            candidate = json.loads(blob)
        except (json.JSONDecodeError, TypeError):
            # Some rows embed the tool list as a Python literal (single quotes).
            try:
                candidate = ast.literal_eval(blob)
            except (ValueError, SyntaxError, TypeError):
                candidate = None
        if isinstance(candidate, Mapping):
            candidate = [candidate]
        if not isinstance(candidate, list) or not candidate:
            continue
        tools = normalize_tools(candidate)
        if tools:
            return tools
    return []


def _find_next_assistant(conversations: list[Any], index: int) -> str:
    for offset in range(index + 1, len(conversations)):
        turn = conversations[offset]
        if isinstance(turn, dict) and str(turn.get("from")) == "assistant":
            return str(turn.get("value", ""))
    return ""


def _looks_like_call(text: str) -> bool:
    return bool(_CALL_SHAPED_RE.match(text or ""))


def load_tasks(cfg: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Yield single-turn ``st_fc`` samples derived from ToolACE dialogues."""
    cfg = dict(cfg or {})
    max_samples = int(cfg.get("max_samples") or 0)
    license_name = cfg.get("license", LICENSE)
    emitted = 0
    n_call_shaped = 0
    n_unparsed = 0

    for index, record in enumerate(iter_source_records(cfg, SOURCE)):
        if max_samples and emitted >= max_samples:
            break
        system = str(record.get("system", "") or "")
        conversations = record.get("conversations") or record.get("messages") or []
        if not isinstance(conversations, list) or not conversations:
            record_skip(cfg, "empty_conversations")
            continue
        tools = normalize_tools(record.get("tools")) or _extract_tools_from_text(system)
        if not tools:
            # Records whose system prompt renders the tool list as an HTML
            # table (or with unknown keys) cannot be turned into a
            # function-calling task; validate_task would reject them.  Skip
            # explicitly rather than emitting a task with an empty tool list.
            record_skip(cfg, "no_tools")
            continue
        history: list[str] = []

        for turn_index, turn in enumerate(conversations):
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("from") or turn.get("role") or "")
            value = str(turn.get("value") if "value" in turn else turn.get("content", ""))
            if role == "user":
                assistant_text = _find_next_assistant(conversations, turn_index)
                if not assistant_text:
                    record_skip(cfg, "missing_assistant_response")
                    history.append(value)
                    continue
                calls = parse_tool_calls(assistant_text)
                if _looks_like_call(assistant_text):
                    n_call_shaped += 1
                    if not calls:
                        n_unparsed += 1
                task = build_task(
                    source=SOURCE,
                    task_id=safe_task_id(SOURCE, f"{record.get('id', index)}-t{turn_index}", index),
                    task_type="st_fc",
                    messages=build_messages(GENERIC_TOOL_SYSTEM, [*history, value]),
                    tools=tools,
                    env={"type": "none", "config": {"max_turns": 1}},
                    reference=build_reference(gt_calls=[calls] if calls else []),
                    max_turns=1,
                    expected_turns=1,
                    license=license_name,
                    meta_extra={
                        "raw_index": index,
                        "conversation_turn": turn_index,
                        "group_id": str(record.get("id") or index),
                    },
                )
                emitted += 1
                yield task
                history.append(value)
            elif role == "tool":
                history.append(f"<tool_response>\n{value[:500]}\n</tool_response>")
            # assistant reasoning is intentionally not added to the prompt.

    if n_call_shaped and n_unparsed:
        logger.warning(
            "%s: %d/%d call-shaped assistant turns were not parsed", SOURCE, n_unparsed, n_call_shaped
        )
    if n_call_shaped and n_unparsed / n_call_shaped > 0.05:
        raise LoaderError(
            f"{SOURCE}: {n_unparsed}/{n_call_shaped} call-shaped assistant turns could not be "
            "parsed into ground-truth tool calls; fix the parser or check whether the raw "
            "format changed (refusing to mislabel positives as negatives)"
        )


__all__ = ["SOURCE", "LICENSE", "load_tasks"]
