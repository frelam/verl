"""Rule-based abstention classifier for no-tool samples (label mode, no RM).

Motivation
----------
For samples whose label says "no tools needed" (``ground_truth_calls == []``),
a response without tool calls is still one of three behaviours:

1. **request_info**   — asks the user for missing information (desired).
2. **no_valid_tools** — declares none of the available tools fits (desired).
3. **guess**          — fabricates a direct answer (undesired).

Dim 1 must tell them apart without a reward model, so this module classifies
the user-visible reply text (``<think>`` / ``<tool_call>`` blocks stripped)
with keyword and structure rules.

Classification (first match wins)
---------------------------------
1. ``NO_VALID_TOOLS`` — capability/tool negation lexicon matches. Tool-
   explicit phrases ("no suitable tool ...", "no tools are needed ...",
   "none of the tools ...", "don't have access to ...") match bare;
   *generic* negations ("I can't ...", "unable to ...", "not able to
   help ...") additionally require a tool/capability/data-access context
   word within the same sentence.
   Without that guard the cheapest reward hack in keyword mode — a hedged
   guess like "I can't be sure, but the answer is 42" — would score full
   marks.  The trade-off is recall: context-free refusals ("I can't help
   with that") now fall to ``GUESS`` (0.4 instead of 1.0), pushing the
   policy towards explicit, tool-grounded abstention phrasing.
2. ``REQUEST_INFO``   — the reply contains a *genuine* question (a ``?``
   that is not a follow-up courtesy like "let me know if ...") together
   with a clarification / wh- lexicon hit.
3. ``GUESS``          — everything else, including empty replies.

Only the {request_info ∪ no_valid_tools} vs guess boundary affects the
reward (both desired classes score identically), so the lexicons favour
paraphrase coverage over precision between the two desired classes.

Reference-conditioned guess penalty
-----------------------------------
Not every no-tool-label sample desires abstention: ToolACE chitchat /
general-knowledge turns carry an empty label because the *desired*
behaviour is a direct answer, and the dataset's own reference response
(embedded in the label string by ``prepare_data.py``) demonstrates it.
Penalising a correct direct answer as ``GUESS`` would reward-hack the
policy towards blanket refusal.  ``reference_prefers_answer`` therefore
checks the reference response with the same classifier: when it is
itself a direct answer, the caller skips the guess penalty.  Designed-
abstention negatives (hammer, ``desc_replace``, ``no_tools``) carry no
reference and keep the penalty.

Configuration (env vars)
------------------------
``TOOL_RL_ABSTAIN_MODE``   ``keyword`` (default) | ``off``
                           On by default so the shaping survives env
                           propagation losses (reward workers may not
                           inherit the launcher shell's exports); set
                           ``off`` explicitly for the legacy behaviour.
"""

from __future__ import annotations

import os
import re
from enum import IntEnum


class AbstentionClass(IntEnum):
    """Behaviour class of a no-tool-label sample's response."""

    REQUEST_INFO = 0
    NO_VALID_TOOLS = 1
    GUESS = 2
    SPURIOUS_CALL = 3  # emitted tool call(s) although the label needs none


ABSTENTION_NOT_APPLICABLE = -1  # sample/response not in the abstention branch


# ============================================================================
# Text stripping — keep only the user-visible reply
# ============================================================================

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL | re.IGNORECASE)


def strip_non_reply(text: str) -> str:
    """Remove ``<think>`` and ``<tool_call>`` blocks from a response."""
    text = _THINK_BLOCK_RE.sub(" ", text)
    text = _UNCLOSED_THINK_RE.sub(" ", text)
    text = _TOOL_CALL_BLOCK_RE.sub(" ", text)
    return text.strip()


# ============================================================================
# Lexicons (extend here — paraphrase variants beat precision)
# ============================================================================

# Class B — capability / tool negation.
#
# Tool/capability/data-access context that must accompany a *generic*
# negation for it to count as a capability abstention (same-sentence
# 80-char window).  Tool-explicit phrases below match bare.
_CAP_CTX = (
    r"(?:tools?|access|ability|capable|capability|capacity|means|permission"
    r"|authori[sz]ed|apis?|databases?|brows(?:e|ing)|internet|web"
    r"|real[-\s]?time|retriev(?:e|al)|fetch|search)"
)

_NEGATION_RES = [
    re.compile(p, re.IGNORECASE)
    for p in (
        # Generic first-person negation — capability context required.
        r"\bi\s*(?:do\s+not|don't|cannot|can't|can\s+not|am\s+unable"
        r"|'m\s+unable|am\s+not\s+able|'m\s+not\s+able)\b"
        r"[^.?!]{0,80}?\b" + _CAP_CTX + r"\b",
        r"\bunable\s+to\b[^.?!]{0,80}?\b" + _CAP_CTX + r"\b",
        # Tool-explicit phrases — match bare.
        r"\bno\s+(?:suitable|available|appropriate|matching|relevant|adequate)\s+tools?\b",
        r"\bno\s+tools?\s+(?:are|is|were|was)?\s*(?:needed|required|necessary|useful)\b",
        r"\bno\s+tools?\s+(?:can|could|would|will)\b",
        r"\bthere\s+(?:is|are)\s+no\s+(?:(?:suitable|available|appropriate)\s+)?tools?\b",
        r"\bnone\s+of\s+the\s+(?:available\s+|provided\s+|declared\s+)?tools?\b",
        r"\b(?:do\s+not|don't|does\s+not|doesn't)\s+have\s+"
        r"(?:access\s+to|the\s+(?:ability|capability|capacity|means)\s+to"
        r"|a\s+(?:suitable|relevant)\s+tool)\b",
        r"\bbeyond\s+my\s+(?:capabilities|ability|scope)\b",
        r"\bnot\s+(?:able|possible)\s+to\s+(?:help|assist|answer|complete|fulfil|fulfill|perform)\b"
        r"[^.?!]{0,80}?\b" + _CAP_CTX + r"\b",
        r"\bno\s+(?:way|means)\s+to\b[^.?!]{0,80}?\b" + _CAP_CTX + r"\b",
    )
]

# Class A — clarification phrases / question words (a genuine "?" required).
_CLARIFY_RES = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(?:could|can|would|will)\s+you\b",
        r"\bplease\s+(?:provide|clarify|specify|share|tell|let\s+me\s+know)\b",
        r"\b(?:i|we)\s+need\s+(?:to\s+know|more)\b",
        r"\bmore\s+(?:information|details?|context|info)\b",
        r"\b(?:do|did)\s+you\s+mean\b",
        r"\b(?:what|which|when|where|who|whom|whose|why|how)\b",
    )
]

# Trailing courtesy questions that must NOT count as clarification requests.
_COURTESY_RE = re.compile(
    r"(?:let\s+me\s+know\s+if|anything\s+else|what\s+else|something\s+else"
    r"|hope\s+this\s+helps?|feel\s+free\s+to"
    r"|if\s+you\s+(?:have|need)\s+(?:any\s+)?(?:more\s+|further\s+)?questions?"
    r"|does\s+that\s+(?:help|make\s+sense|answer)"
    r"|is\s+there\s+anything)",
    re.IGNORECASE,
)

_COURTESY_WINDOW = 80  # chars before a "?" inspected for courtesy phrasing


def _has_genuine_question(text: str) -> bool:
    """True when a ``?`` exists that is not a follow-up courtesy."""
    for m in re.finditer(r"\?", text):
        window = text[max(0, m.start() - _COURTESY_WINDOW):m.start()]
        if not _COURTESY_RE.search(window):
            return True
    return False


# ============================================================================
# Classification
# ============================================================================

def classify_abstention(response: str) -> AbstentionClass:
    """Classify a no-tool-call response on a no-tool-label sample."""
    text = strip_non_reply(response)
    if not text:
        return AbstentionClass.GUESS
    if any(r.search(text) for r in _NEGATION_RES):
        return AbstentionClass.NO_VALID_TOOLS
    if _has_genuine_question(text) and any(r.search(text) for r in _CLARIFY_RES):
        return AbstentionClass.REQUEST_INFO
    return AbstentionClass.GUESS


# ============================================================================
# Reference-conditioned guess penalty
# ============================================================================

# Marker written by ``prepare_data.py``: the label string is
# ``_format_gt(calls)`` optionally followed by
# ``"\nReference:\n{assistant_response[:1000]}"`` — present only when the
# dataset's own reference response exists (e.g. ToolACE turns).  Designed-
# abstention negatives (hammer, desc_replace, no_tools) have an empty label.
_REFERENCE_RE = re.compile(r"\n?Reference:\n(.*)$", re.DOTALL)


def reference_prefers_answer(label: object) -> bool:
    """True when the label's reference response is itself a direct answer.

    A no-tool sample whose demonstrated behaviour is a direct answer
    (chitchat / general knowledge) must not guess-penalise the model for
    answering directly.  No reference → False (abstention samples keep the
    penalty): an empty reference classifies as ``GUESS``, so the emptiness
    check must come first.
    """
    if not isinstance(label, str):
        return False
    m = _REFERENCE_RE.search(label)
    if not m or not m.group(1).strip():
        return False
    return classify_abstention(m.group(1)) is AbstentionClass.GUESS


# ============================================================================
# Env config
# ============================================================================

def abstain_mode_from_env() -> str:
    """Read ``TOOL_RL_ABSTAIN_MODE``: ``keyword`` (default) | ``off``."""
    mode = os.environ.get("TOOL_RL_ABSTAIN_MODE", "keyword").strip().lower()
    if mode not in ("off", "keyword"):
        raise ValueError(f"TOOL_RL_ABSTAIN_MODE must be off|keyword, got {mode!r}")
    return mode
