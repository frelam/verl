"""``abstain`` reward - correct clarification / refusal (plan §1.5, §5).

``score = 1.0`` iff the policy emitted **no tool call** and the user-visible
reply correctly

* asks for the missing information (``request_info``), or
* declares that no available tool / capability fits (``no_valid_tools``);

else ``0.0``.  A fabricated direct answer (``guess``) and a spurious tool call
(``spurious_call``) score 0.0.

Classification is a rule-based, stdlib-only re-implementation of
``examples/tool_rl/reward/abstention.py`` (same three behaviour classes; no
env-var switches, no torch).  Two structural guards keep the cheapest reward
hacks out (plan §5/§10):

1. **clause locality** - the question mark and the asking signal must be in
   the same clause, so a wh-word in one sentence cannot vouch for a "?" in
   another;
2. **ask dominance** - the asking clauses must make up at least half of the
   visible reply (and more than the declarative remainder), so "The capital of
   France is Paris. Which city?" stays a guess.

Output is the last non-empty assistant ``content`` (``<think>`` /
``<tool_call>`` blocks stripped), falling back to ``trajectory["final_answer"]``
(CONTRACTS C4: turns + final_answer).  A minimal Chinese lexicon is included as
a best-effort extension for Chinese clarifications.

Boundary vs ``irrelevance`` (hammer)
------------------------------------
``irrelevance`` scores only *whether any tool call happened*: a direct answer
or a refusal both get 1.0.  ``abstain`` (When2Call) additionally scores the
reply: a fabricated direct answer gets 0.0.  A correct clarification scores
1.0 under both reward families; the task_type decides which intent applies, not
the classifier (a ``gt_calls=[]`` label alone cannot tell the two apart).
"""

from __future__ import annotations

import re

from . import base

TASK_TYPE = "abstain"
MAX_SCORE = 1.0

REQUEST_INFO = "request_info"
NO_VALID_TOOLS = "no_valid_tools"
GUESS = "guess"
SPURIOUS_CALL = "spurious_call"
DESIRED_CLASSES = frozenset({REQUEST_INFO, NO_VALID_TOOLS})

_REPLY_EXCERPT_LIMIT = 200

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL | re.IGNORECASE)


def strip_non_reply(text: str) -> str:
    """Remove ``<think>`` / ``<tool_call>`` blocks, keeping the visible reply."""
    if not isinstance(text, str):
        return ""
    text = _THINK_BLOCK_RE.sub(" ", text)
    text = _UNCLOSED_THINK_RE.sub(" ", text)
    text = _TOOL_CALL_BLOCK_RE.sub(" ", text)
    return text.strip()


# --- capability / tool negation (class NO_VALID_TOOLS) ---------------------

_CAP_CTX = (
    r"(?:tools?|access|ability|capable|capability|means|permission"
    r"|authori[sz]ed|apis?|databases?|brows(?:e|ing)|internet|web"
    r"|real[-\s]?time|retriev(?:e|al)|fetch|search)"
)

_NEGATION_RES = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bi\s*(?:do\s+not|don't|cannot|can't|can\s+not|am\s+unable|'m\s+unable"
        r"|am\s+not\s+able|'m\s+not\s+able)\b[^.?!]{0,80}?\b" + _CAP_CTX + r"\b",
        r"\bunable\s+to\b[^.?!]{0,80}?\b" + _CAP_CTX + r"\b",
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

# --- clarification requests (class REQUEST_INFO) --------------------------

_CLARIFY_RES = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bplease\s+(?:provide|clarify|specify|share|tell|show|give|confirm"
        r"|check|explain|let\s+me\s+know)\b",
        r"\b(?:tell|show|give|provide|share|specify|clarify|confirm)\s+me\b",
        r"\b(?:i|we)\s+need\s+(?:to\s+know|more)\b",
        r"\bmore\s+(?:information|details?|context|info)\b",
        r"\b(?:do|did)\s+you\s+mean\b",
    )
]

_MODAL_YOU_RE = re.compile(r"\b(?:could|can|would|will)\s+you\b", re.IGNORECASE)
_REQUEST_VERB_RE = re.compile(
    r"\b(?:tell|provide|give|share|clarify|specify|confirm|check|explain|show"
    r"|indicate|state|let\s+me\s+know|help\s+me|send|list)\b",
    re.IGNORECASE,
)
_WH_RE = re.compile(r"\b(?:what|which|when|where|who|whom|whose|why|how)\b", re.IGNORECASE)
_NOMINAL_WH = frozenset({"what", "which", "whose"})
_HOW_QUANT_RE = re.compile(
    r"^(?:many|much|long|often|far|soon|big|deep|large)\b", re.IGNORECASE
)
_WH_CONTRACTION_RE = re.compile(
    r"\b(what|which|who|where|when|why|how)(?:'s|s)\b", re.IGNORECASE
)
_AUX_RE = re.compile(
    r"^(?:is|are|was|were|am|do|does|did|can|could|would|will|should|shall"
    r"|may|might|have|has|had)\b",
    re.IGNORECASE,
)
_AUX_TOK_RE = re.compile(
    r"^(?:is|are|was|were|am|do|does|did|can|could|would|will|should|shall"
    r"|may|might|have|has|had)$",
    re.IGNORECASE,
)
_SUBJ_PRON_RE = re.compile(r"^(?:i|you|we|they|he|she|it)$", re.IGNORECASE)

# Courtesy / rhetorical tails that end in "?" but ask for nothing.
_COURTESY_RE = re.compile(
    r"(?:let\s+me\s+know\s+if|anything\s+else|what\s+else|something\s+else"
    r"|hope\s+this\s+helps?|feel\s+free\s+to"
    r"|if\s+you\s+(?:have|need)\s+(?:any\s+)?(?:more\s+|further\s+)?questions?"
    r"|does\s+that\s+(?:help|make\s+sense|answer)"
    r"|is\s+there\s+anything"
    r"|do\s+you\s+agree|how\s+about\s+that|how\s+does\s+that\s+sound"
    r"|is\s+that\s+(?:correct|right|ok|okay|clear|all)"
    r"|who\s+would\s+have\s+\w+|what\s+do\s+you\s+think|who\s+knows"
    r"|how\s+(?:can|may)\s+i\s+(?:help|assist)|what\s+can\s+i\s+(?:do|help)"
    r"|isn'?t\s+(?:it|that)|don'?t\s+you\s+think"
    r"|what'?s\s+next|what\s+would\s+you\s+like\s+to\s+know"
    r"|(?:could|would|can|will)\s+you\s+(?:imagine|believe|beat\s+that|top\s+that)"
    r"|(?:would|will|can)\s+you\s+look\s+at\s+that"
    r"|(?:correct|right|ok|okay)\s*\?$"
    r"|who(?:'s|\s+(?:is|are|was))\s+asking|who\s+asked|who\s+cares"
    r"|why\s+bother|so\s+what|what\s+of\s+it"
    r"|what'?s\s+the\s+(?:point|use|difference)|why\s+not"
    r"|what\s+does\s+it\s+matter|how\s+should\s+i\s+know|who\s+am\s+i\s+to\s+say"
    r"|which\s+one\s+of\s+us)",
    re.IGNORECASE,
)

_RHETORICAL_RE = re.compile(
    r"\b(?:what|which|who|how|where|when|why)\w*\s*(?:'s|is|are|was|were)?\s*"
    r"(?:the\s+)?(?:point|use|difference|matter|problem|issue|deal|catch)\b"
    r"|\b(?:is|was|are|were)\s+(?:that|this|it)\s+(?:the\s+)?"
    r"(?:point|use|difference|matter|problem|issue|deal|catch)\b"
    r"|\bwhich\s+(?:one|of\s+(?:these|them))\s+(?:is|are|was|were)?\s*"
    r"(?:correct|right|wrong|better|best|the\s+point)\b"
    r"|\b(?:which|what|that|who)\s+you\s+(?:already\s+|may\s+have\s+)?"
    r"(?:know|said|asked|meant|think|want)\b"
    r"|\bi\s+(?:should\s+know|am\s+i\s+to\s+say)\b",
    re.IGNORECASE,
)

_TERMINATOR_RE = re.compile(r"\?|？|\.(?=\s|$)|!(?=\s|$)|\n(?=\s*[A-Z(\[*#<`-]|\s*$)")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
_LEAD_NOISE_RE = re.compile(
    r"^(?:[\s\-*)\].:>#_`'\"]+|\d+[.)]\s*|(?:and|but|or|so|also|then|however"
    r"|thus|therefore|ok|okay|sure|alright|well|hmm|sorry|no\s+problem"
    r"|of\s+course|got\s+it|understood|absolutely|certainly|yes|no|hi|hello"
    r"|right)\b[\s,.:;!-]*)+",
    re.IGNORECASE,
)

# --- Chinese best-effort lexicon (When2Call is English; kept small) --------

_CJK_QUESTION_RE = re.compile(r"[？?]")
_CJK_NEG_RE = re.compile(
    r"(?:无法访问|无法获取|没有(?:可用|合适|相关)的?工具|无法(?:执行|查询|检索)"
    r"|超出.{0,6}(?:能力|范围)|我不具备)"
)
_CJK_CLARIFY_RE = re.compile(
    r"(?:请问|请提供|请说明|能否(?:提供|说明|确认)|是否(?:可以|需要)"
    r"|需要(?:你|您)?提供|更多(?:信息|细节|上下文)|信息不足|不够(?:明确|具体)"
    r"|你想(?:查询|问|了解))"
)


def _all_clauses(text: str):
    """Yield clauses split by sentence terminators, terminators retained."""
    start = 0
    for match in _TERMINATOR_RE.finditer(text):
        yield text[start : match.start()] + match.group(0)
        start = match.end()
    if start < len(text):
        yield text[start:]


def _asks(segment: str) -> bool:
    """True when one clause is an interrogative asking the user for something."""
    segment = _LEAD_NOISE_RE.sub("", segment.strip())
    if not segment:
        return False
    if any(pattern.search(segment) for pattern in _CLARIFY_RES):
        return True
    if _MODAL_YOU_RE.search(segment) and _REQUEST_VERB_RE.search(segment):
        return True
    if _RHETORICAL_RE.search(segment):
        return False
    stripped = _WH_CONTRACTION_RE.sub(r"\1 is", segment)
    wh_match = _WH_RE.match(stripped)
    if wh_match:
        wh = wh_match.group(0).lower()
        rest = stripped[wh_match.end() :].lstrip()
        if not rest:
            return False
        if wh in _NOMINAL_WH:
            if _AUX_RE.match(rest):
                return True
            words = rest.split()
            if words and _SUBJ_PRON_RE.match(words[0]):
                return False
            return True
        if _AUX_RE.match(rest):
            return True
        return wh == "how" and bool(_HOW_QUANT_RE.match(rest))
    return bool(_AUX_RE.match(stripped))


def _is_clarification_clause(clause: str) -> bool:
    """True when one interrogative clause itself asks for missing information."""
    body = clause.strip()
    if len(_WORD_RE.findall(body)) < 2:
        return False
    if _asks(body):
        return True
    tail = body.rsplit(",", 1)[-1]
    return tail != body and _asks(tail)


def _request_info(text: str) -> bool:
    """True when a genuine question dominates the visible reply."""
    total_words = 0
    for clause in _all_clauses(text):
        if _COURTESY_RE.search(clause):
            continue
        total_words += len(_WORD_RE.findall(clause))
    if total_words == 0:
        return False

    asking_words = 0
    found = False
    for clause in _all_clauses(text):
        stripped = clause.rstrip()
        if not (stripped.endswith("?") or stripped.endswith("？")):
            continue
        body = stripped[:-1]
        if _COURTESY_RE.search(body):
            continue
        if _is_clarification_clause(body):
            found = True
            asking_words += len(_WORD_RE.findall(body))
    if not found:
        return False
    remainder = total_words - asking_words
    return asking_words * 2 >= total_words and asking_words > remainder


def classify_reply(reply: str) -> str:
    """Classify a visible reply as ``request_info`` / ``no_valid_tools`` / ``guess``."""
    text = strip_non_reply(reply)
    if not text:
        return GUESS
    if _CJK_NEG_RE.search(text):
        return NO_VALID_TOOLS
    if any(pattern.search(text) for pattern in _NEGATION_RES):
        return NO_VALID_TOOLS
    if _CJK_CLARIFY_RE.search(text) and _CJK_QUESTION_RE.search(text):
        return REQUEST_INFO
    if _request_info(text):
        return REQUEST_INFO
    if _CJK_CLARIFY_RE.search(text):
        return REQUEST_INFO
    return GUESS


def reward(task: dict, trajectory: dict) -> dict:
    """Pure reward for one ``abstain`` task/trajectory pair."""
    calls = base.all_calls(trajectory)
    reply = base.visible_reply(trajectory)
    visible = strip_non_reply(reply)

    if calls:
        classification = SPURIOUS_CALL
    elif not visible:
        classification = GUESS
    else:
        classification = classify_reply(reply)

    desired = classification in DESIRED_CLASSES
    score = MAX_SCORE if desired else 0.0
    info = {
        "abstention_class": classification,
        "desired": desired,
        "desired_classes": sorted(DESIRED_CLASSES),
        "n_output_calls": len(calls),
        "called_calls": [base.call_repr(call) for call in calls[:10]],
        "reply_excerpt": visible[:_REPLY_EXCERPT_LIMIT],
        "reply_is_empty": not bool(visible),
        "gt_calls_empty": not base.flatten_gt_calls(task),
        "expected_behavior": (
            "ask for the missing information (request_info) or declare that no "
            "available tool fits (no_valid_tools); never fabricate an answer or call a tool"
        ),
        "boundary_vs_irrelevance": (
            "irrelevance scores only whether any tool call happened and accepts any "
            "direct answer; abstain scores the reply quality and a fabricated direct "
            "answer is 0.0"
        ),
        "rule": "no call + correct clarification/refusal = 1, else 0 (plan §5)",
    }
    return base.make_result(score, info, task_type=TASK_TYPE)
