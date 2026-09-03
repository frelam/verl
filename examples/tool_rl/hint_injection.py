#!/usr/bin/env python3
"""Randomized system-prompt hint injection for the tool_rl experiment.

Motivation
----------
Some tool-use queries are hard to roll out successfully, so we inject a
short behavioural *hint* into the system prompt.  A single fixed hint would
make the policy over-rely on that exact wording, so instead we keep a pool
of paraphrased variants — all carrying the **same three semantics**:

1. Stay humble: never assume or invent information you were not given.
2. Tools may depend on each other: mind the calling order.
3. Reason before making a tool call.

One "variant" is the empty hint (no guidance at all), drawn with
probability ``TOOL_RL_HINT_EMPTY_PROB`` (default 0.25) so the policy also
learns to behave well without any hint.

Injection happens at *training time* in ``ToolRLHintDataset.__getitem__``
(see ``tool_rl_dataset.py``), which guarantees:

- every epoch re-samples a fresh variant per sample (no fixed pairing to
  overfit), and
- all ``rollout.n`` rollouts of one GRPO group share the same variant
  (verl repeats the batch *after* ``__getitem__``).

The chosen variant index is recorded in ``extra_info["hint_id"]``
(``-1`` = empty hint) so per-variant rewards can be analysed offline.

Configuration (environment variables)
-------------------------------------
``TOOL_RL_HINT_MODE``       ``off`` (default) | ``random`` | ``fixed``
                            ``random``: fresh draw on every ``__getitem__``.
                            ``fixed``: deterministic per sample (content
                            hash) — same sample always gets the same
                            variant.  Validation datasets always behave as
                            ``fixed`` regardless of this setting.
``TOOL_RL_HINT_EMPTY_PROB`` probability of the empty hint (default 0.25).
``TOOL_RL_HINT_SEED``       base seed for sampling / hashing (default 42).
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any

# ============================================================================
# Hint variant pool
# ============================================================================
# Every variant conveys the same three points (humility / tool-dependency
# ordering / reason-before-call) with different wording, formatting and
# point order, so the policy cannot latch onto one canonical phrasing.

HINT_VARIANTS: list[str] = [
    # 0 — numbered list, direct
    (
        "Important:\n"
        "1. Stay humble — never assume or invent information you don't have.\n"
        "2. Tools may depend on each other; plan the calling order carefully.\n"
        "3. Reason carefully before every tool call."
    ),
    # 1 — bullet list
    (
        "Keep the following in mind:\n"
        "- Do not fabricate or assume facts; acknowledge what you don't know.\n"
        "- Some tools depend on the results of others, so call them in the right sequence.\n"
        "- Think step by step before invoking any tool."
    ),
    # 2 — plain paragraph
    (
        "Always stay humble and avoid guessing or assuming details. Remember that "
        "tool calls can depend on one another, so decide their order deliberately, "
        "and think through your reasoning before each call."
    ),
    # 3 — XML-wrapped, keyword style
    (
        "<guidelines>\n"
        "- Humility: never invent information; say when you don't know.\n"
        "- Dependencies: tools may rely on each other's outputs — respect the calling order.\n"
        "- Deliberation: reason first, then call the tool.\n"
        "</guidelines>"
    ),
    # 4 — reordered points (dependency → reasoning → humility)
    (
        "Note:\n"
        "1) Tools often depend on each other — call them in a sensible order.\n"
        "2) Think before you act: work out your reasoning prior to each tool call.\n"
        "3) Stay humble; do not assume facts that were not given."
    ),
    # 5 — reflective questions
    (
        "Before answering, ask yourself: Am I assuming anything I shouldn't? Am I "
        "calling tools in the right order given their dependencies? Have I reasoned "
        "through this step before calling the next tool?"
    ),
    # 6 — single-line compact
    (
        "Stay humble and never guess; mind the dependencies between tools and their "
        "calling order; always reason before making a tool call."
    ),
    # 7 — advisory tone
    (
        "You should avoid making assumptions about missing information. You should "
        "consider how the tools depend on each other and sequence your calls "
        "accordingly. You should also think carefully before every tool invocation."
    ),
    # 8 — markdown emphasis
    (
        "**Reminders**\n"
        "1. **No assumptions** — if information is missing, don't make it up.\n"
        "2. **Order matters** — some tools need the output of earlier calls.\n"
        "3. **Reason first** — think through your plan before each tool call."
    ),
    # 9 — terse rules
    (
        "Rules: (1) never hallucinate facts; (2) respect dependencies between tools "
        "when ordering calls; (3) reason before each tool call."
    ),
    # 10 — conversational
    (
        "A few things to remember: Don't pretend to know what you don't. Check "
        "whether one tool's output is needed as another's input, and order your "
        "calls accordingly. Take a moment to reason before each call."
    ),
    # 11 — guidance bullets
    (
        "Guidance:\n"
        "- Be honest about uncertainty instead of guessing.\n"
        "- Plan tool usage: later calls may depend on earlier results.\n"
        "- Think through the problem before invoking tools."
    ),
]

EMPTY_HINT_ID = -1


# ============================================================================
# Sampling
# ============================================================================

def sample_hint_id(rng: random.Random, empty_prob: float) -> int:
    """Draw a variant index; ``EMPTY_HINT_ID`` with probability ``empty_prob``."""
    if rng.random() < empty_prob:
        return EMPTY_HINT_ID
    return rng.randrange(len(HINT_VARIANTS))


def fixed_hint_id(sample_key: str, seed: int, empty_prob: float) -> int:
    """Deterministic variant for a sample (stable across epochs and eval runs)."""
    digest = hashlib.md5(f"{seed}|{sample_key}".encode()).hexdigest()
    bucket = int(digest[:8], 16)
    if (bucket % 10000) / 10000.0 < empty_prob:
        return EMPTY_HINT_ID
    return int(digest[8:16], 16) % len(HINT_VARIANTS)


# ============================================================================
# Injection
# ============================================================================

def inject_hint(messages: list[dict[str, Any]], hint: str) -> list[dict[str, Any]]:
    """Return a copy of ``messages`` with ``hint`` appended to the system prompt.

    A new system message is created if the conversation has none.  Messages
    whose content is not a plain string (e.g. multimodal content lists) are
    left untouched; in that case the hint is prepended as a separate system
    message so injection never silently drops.
    """
    out: list[dict[str, Any]] = []
    injected = False
    for msg in messages:
        msg = dict(msg)
        if not injected and msg.get("role") == "system":
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = f"{content}\n\n{hint}" if content else hint
                injected = True
        out.append(msg)
    if not injected:
        out.insert(0, {"role": "system", "content": hint})
    return out


def sample_key_from_row(row_dict: dict[str, Any]) -> str:
    """Stable identity of a sample for ``fixed`` mode.

    Prefers ``extra_info.task_id``; falls back to the first user message.
    """
    extra = row_dict.get("extra_info") or {}
    task_id = extra.get("task_id")
    if task_id:
        return str(task_id)
    for msg in row_dict.get("raw_prompt") or []:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                return content[:512]
    return str(row_dict.get("index", ""))


def apply_hint_to_row(
    row_dict: dict[str, Any],
    *,
    mode: str,
    rng: random.Random,
    seed: int,
    empty_prob: float,
) -> dict[str, Any]:
    """Inject a sampled hint into ``row_dict["raw_prompt"]`` in place.

    Records the chosen variant in ``row_dict["extra_info"]["hint_id"]``.
    ``mode`` is ``random`` (fresh draw) or ``fixed`` (content-hash
    deterministic); ``off`` is a no-op.
    """
    if mode == "off":
        return row_dict
    messages = row_dict.get("raw_prompt")
    if not isinstance(messages, list) or not messages:
        return row_dict

    if mode == "fixed":
        hint_id = fixed_hint_id(sample_key_from_row(row_dict), seed, empty_prob)
    else:
        hint_id = sample_hint_id(rng, empty_prob)

    if hint_id != EMPTY_HINT_ID:
        row_dict["raw_prompt"] = inject_hint(messages, HINT_VARIANTS[hint_id])

    extra = row_dict.get("extra_info")
    if isinstance(extra, dict):
        extra["hint_id"] = hint_id
    return row_dict


# ============================================================================
# Env config
# ============================================================================

def hint_config_from_env() -> dict[str, Any]:
    """Read hint injection settings from ``TOOL_RL_HINT_*`` env vars."""
    mode = os.environ.get("TOOL_RL_HINT_MODE", "off").strip().lower()
    if mode not in ("off", "random", "fixed"):
        raise ValueError(f"TOOL_RL_HINT_MODE must be off|random|fixed, got {mode!r}")
    return {
        "mode": mode,
        "empty_prob": float(os.environ.get("TOOL_RL_HINT_EMPTY_PROB", "0.25")),
        "seed": int(os.environ.get("TOOL_RL_HINT_SEED", "42")),
    }
