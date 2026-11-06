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

import json
import logging
import re
from typing import Any, Iterator

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


def _extract_tools_from_text(text: str) -> list[dict[str, Any]]:
    """Extract tool definitions embedded in ToolACE's system prompt text."""
    tools: list[dict[str, Any]] = []
    array_match = re.search(r"\[.*\]", text, re.DOTALL)
    if array_match:
        try:
            candidates = json.loads(array_match.group(0))
        except (json.JSONDecodeError, TypeError):
            candidates = None
        if isinstance(candidates, list):
            for candidate in candidates:
                if isinstance(candidate, dict) and candidate.get("name"):
                    tools.append(candidate)
    if not tools:
        depth = 0
        start = -1
        for index, char in enumerate(text):
            if char == "{":
                if depth == 0:
                    start = index
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        obj = json.loads(text[start : index + 1])
                    except json.JSONDecodeError:
                        obj = None
                    if isinstance(obj, dict) and obj.get("name"):
                        tools.append(obj)
                    start = -1
    return normalize_tools(tools)


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
