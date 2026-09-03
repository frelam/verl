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
1. ``NO_VALID_TOOLS`` — capability/tool negation lexicon matches anywhere
   ("I don't have access to ...", "no suitable tool ...", "unable to ...").
2. ``REQUEST_INFO``   — the reply contains a *genuine* question (a ``?``
   that is not a follow-up courtesy like "let me know if ...") together
   with a clarification / wh- lexicon hit.
3. ``GUESS``          — everything else, including empty replies.

Only the {request_info ∪ no_valid_tools} vs guess boundary affects the
reward (both desired classes score identically), so the lexicons favour
paraphrase coverage over precision between the two desired classes.

Configuration (env vars)
------------------------
``TOOL_RL_ABSTAIN_MODE``   ``off`` (default) | ``keyword``
                           ``off`` keeps the legacy reward behaviour.
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
_NEGATION_RES = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bi\s*(?:do\s+not|don't|cannot|can't|can\s+not|am\s+unable"
        r"|'m\s+unable|am\s+not\s+able|'m\s+not\s+able)\b",
        r"\bunable\s+to\b",
        r"\bno\s+(?:suitable|available|appropriate|matching|relevant|adequate)\s+tools?\b",
        r"\bnone\s+of\s+the\s+(?:available\s+|provided\s+|declared\s+)?tools?\b",
        r"\b(?:do\s+not|don't|does\s+not|doesn't)\s+have\s+"
        r"(?:access\s+to|the\s+(?:ability|capability|capacity|means)\s+to"
        r"|a\s+(?:suitable|relevant)\s+tool)\b",
        r"\bbeyond\s+my\s+(?:capabilities|ability|scope)\b",
        r"\bnot\s+(?:able|possible)\s+to\s+(?:help|assist|answer|complete|fulfil|fulfill|perform)\b",
        r"\bno\s+(?:way|means)\s+to\b",
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
# Env config
# ============================================================================

def abstain_mode_from_env() -> str:
    """Read ``TOOL_RL_ABSTAIN_MODE``: ``off`` (default) | ``keyword``."""
    mode = os.environ.get("TOOL_RL_ABSTAIN_MODE", "off").strip().lower()
    if mode not in ("off", "keyword"):
        raise ValueError(f"TOOL_RL_ABSTAIN_MODE must be off|keyword, got {mode!r}")
    return mode
