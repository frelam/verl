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
2. ``REQUEST_INFO``   — the reply *is* a question: its interrogative
   clauses account for most of it, and those clauses themselves ask for the
   missing information (see "Clause locality and ask-dominance").
3. ``GUESS``          — everything else, including empty replies.

Only the {request_info ∪ no_valid_tools} vs guess boundary affects the
reward (both desired classes score identically), so the lexicons favour
paraphrase coverage over precision between the two desired classes.

Clause locality and ask-dominance
---------------------------------
``REQUEST_INFO`` used to be a reply-global keyword test: any non-courtesy
``?`` plus a clarification/wh lexicon hit *anywhere* in the reply.  That is
trivially hackable on the surfaces the policy controls:

- the ``?`` and the wh-word can come from *different* sentences, so
  "When I consider the data, the answer is 42?" (a fabricated answer with a
  trailing "?") classified as ``REQUEST_INFO``;
- the hint pool is injected into the system prompt, so a policy that echoes
  its instructions supplies the wh-word for free — 11 of the 12 variants
  contain a wh-word, and appending a bare "?" reached full Dim 1;
- a rhetorical tail ("The answer is 42, what?", "Paris. Can you imagine?")
  matched the same lexicon; and
- a fabricated answer followed by a genuine-looking question ("The capital
  of France is Paris. Which city?") is still mostly an answer.

``_request_info`` therefore applies three structural guards:

1. **Clause locality** — the ``?`` and the asking signal must be in the
   same clause.  A clause ends at a ``?``, at a ``.``/``!`` followed by
   whitespace (so "Acme Inc.?" and "Which city...?" stay whole), and
   deliberately *not* at a newline, so a hard-wrapped question survives.
2. **The question must be the reply** — the asking clauses must account for
   at least half the reply's words.  A real abstention is mostly question;
   an answer with a question bolted on is not.  Courtesy clauses are
   excluded from the count on both sides, so a sign-off does not dilute a
   real question.
3. **The clause must actually ask** — the asking signal has to be an
   interrogative use, not just a wh-word.  "Which city should I use, Paris
   or Lyon?" asks; "When I consider the data", "which you already know" and
   "what you said" are declarative/relative fragments.

Rhetorical questions remain an open set that no lexicon can enumerate, so
``_COURTESY_RE`` lists the idioms frequent enough to matter and an unlisted
one can still be echoed.  The residual hole needs a fabricated answer *and*
a rhetorical question *and* for the question to dominate the reply — which
is why ``REQUEST_INFO`` stays a heuristic rather than a guarantee.

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
reference and keep the penalty, and a reference that itself demonstrates
a tool call is never treated as a direct answer.

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

# Class A — clarification phrases / question words.
#
# Unlike ``_NEGATION_RES`` these are matched against the *interrogative
# clause* rather than the whole reply (see ``_request_info``).  The first
# group ("please provide ...", "tell me ...") is specific enough to identify
# a clarification request on its own.  A bare "could you ..."-shaped clause
# is weaker — "Can you imagine?" is phatic — so it only counts together
# with a request verb (``_MODAL_YOU_RE`` + ``_REQUEST_VERB_RE`` below).  A
# wh-word alone is weaker still — "What?" alone is not a request, and "The
# answer is 42, what?" is a rhetorical tail on a fabricated answer — so it
# only asks when it *leads* the clause in an interrogative use
# (``_wh_leads_question`` below).
_CLARIFY_RES = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bplease\s+(?:provide|clarify|specify|share|tell|show|give|confirm"
        r"|check|explain|let\s+me\s+know)\b",
        r"\b(?:tell|show|give|provide|share|specify|clarify|confirm)\s+me\b",
        r"\b(?:i|we)\s+need\s+(?:to\s+know|more)\b",
        r"\bmore\s+(?:information|details?|context|info)\b",
        r"\b(?:do|did)\s+you\s+mean\b",
    )
]

# "Could you ..."-shaped requests need a request verb: bare modal+you is
# phatic as often as not ("Can you imagine?", "Would you look at that?").
_MODAL_YOU_RE = re.compile(r"\b(?:could|can|would|will)\s+you\b", re.IGNORECASE)
_REQUEST_VERB_RE = re.compile(
    r"\b(?:tell|provide|give|share|clarify|specify|confirm|check|explain|show"
    r"|indicate|state|let\s+me\s+know|help\s+me|send|list)\b",
    re.IGNORECASE,
)

_WH_RE = re.compile(r"\b(?:what|which|when|where|who|whom|whose|why|how)\b", re.IGNORECASE)

# Trailing courtesy questions that must NOT count as clarification requests.
_COURTESY_RE = re.compile(
    r"(?:let\s+me\s+know\s+if|anything\s+else|what\s+else|something\s+else"
    r"|hope\s+this\s+helps?|feel\s+free\s+to"
    r"|if\s+you\s+(?:have|need)\s+(?:any\s+)?(?:more\s+|further\s+)?questions?"
    r"|does\s+that\s+(?:help|make\s+sense|answer)"
    r"|is\s+there\s+anything"
    r"|do\s+you\s+agree|how\s+about\s+that|how\s+does\s+that\s+sound"
    r"|is\s+that\s+(?:correct|right|ok|okay|clear|all)"
    # Rhetorical / phatic tails: these end in "?" but ask for nothing.
    r"|who\s+would\s+have\s+\w+|what\s+do\s+you\s+think|who\s+knows"
    r"|how\s+(?:can|may)\s+i\s+(?:help|assist)|what\s+can\s+i\s+(?:do|help)"
    r"|isn'?t\s+(?:it|that)|don'?t\s+you\s+think"
    r"|what'?s\s+next|what\s+would\s+you\s+like\s+to\s+know"
    r"|(?:could|would|can|will)\s+you\s+(?:imagine|believe|beat\s+that|top\s+that)"
    r"|(?:would|will|can)\s+you\s+look\s+at\s+that"
    r"|(?:correct|right|ok|okay)\s*\?$"
    # Rhetorical questions in general are an open set that no lexicon can
    # enumerate; these are the idioms frequent enough to matter.  See the
    # "Known limitation" note in the module docstring.
    r"|who(?:'s|\s+(?:is|are|was))\s+asking|who\s+asked|who\s+cares|why\s+bother|so\s+what"
    r"|what\s+of\s+it|what'?s\s+the\s+(?:point|use|difference)|why\s+not"
    r"|what\s+does\s+it\s+matter|how\s+should\s+i\s+know|who\s+am\s+i\s+to\s+say"
    r"|which\s+one\s+of\s+us)",
    re.IGNORECASE,
)

# Sentence terminator.  "?" always ends a clause.  A "." only does so
# before whitespace or end-of-text, so "Do you mean Acme Inc.?" and
# "Which city...?" stay whole.  A newline only does so when a new
# sentence starts, so a hard-wrapped question ("Which city do you\n
# mean?") is not chopped in half.
_TERMINATOR_RE = re.compile(r"\?|\.(?=\s|$)|!(?=\s|$)|\n(?=\s*[A-Z(\[*#<`-]|\s*$)")

# Interrogative clause body, as matched by ``_TERMINATOR_RE``.
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")

# Leading list/markup noise, addressing and discourse fillers, stripped
# before deciding what kind of clause follows.
_LEAD_NOISE_RE = re.compile(
    r"^(?:[\s\-*)\].:>#_`'\"]+|\d+[.)]\s*|(?:and|but|or|so|also|then|however"
    r"|thus|therefore|ok|okay|sure|alright|well|hmm|sorry|no\s+problem"
    r"|of\s+course|got\s+it|understood|absolutely|certainly|yes|no|hi|hello"
    r"|right)\b[\s,.:;!-]*)+",
    re.IGNORECASE,
)

_MIN_CLAUSE_WORDS = 2

# Auxiliary verbs: a clause that opens "<wh> <aux> ..." is inverted — a
# question by construction ("When is the deadline?"); a bare "<aux> ..."
# clause ending in "?" likewise ("Is that the right city?").
_AUX_RE = re.compile(
    r"^(?:is|are|was|were|am|do|does|did|can|could|would|will|should|shall"
    r"|may|might|have|has|had)\b",
    re.IGNORECASE,
)

# Whole-token auxiliaries, so a scan for "the first auxiliary after the
# wh-word" is not fooled by "can" inside "candidate".
_AUX_TOK_RE = re.compile(
    r"^(?:is|are|was|were|am|do|does|did|can|could|would|will|should|shall"
    r"|may|might|have|has|had)$",
    re.IGNORECASE,
)

# Subject pronouns that can head a *relative* clause ("which city you
# mean") — the shape a policy reaches for when it bolts a hint-echoed
# wh-word onto a fabricated answer.
_SUBJ_PRON_RE = re.compile(r"^(?:i|you|we|they|he|she|it)$", re.IGNORECASE)

# How far past the wh-word the noun phrase may reach before a subject
# pronoun counts as an object rather than a relative head ("What date
# works best for you" — "you" is far out).
_HEAD_WINDOW_CHARS = 12

# "What's the city?" — expand the contraction before the auxiliary scan,
# since ``_AUX_RE`` matches whole tokens and would read "What's" as a noun.
_WH_CONTRACTION_RE = re.compile(r"\b(what|which|who|where|when|why|how)(?:'s|s)\b", re.IGNORECASE)

# Nominal wh-words introduce a noun phrase; the rest are adverbial.
_NOMINAL_WH = {"what", "which", "whose"}

# "How many / how much / how long ..." — "how" only asks without inversion
# when a quantifier follows.
_HOW_QUANT_RE = re.compile(r"^(?:many|much|long|often|far|soon|big|deep|large)\b", re.IGNORECASE)

# "Should I ...?" / "May we ...?" asks the *user* for direction, unlike a
# be-verb inversion ("Is the capital of France Paris?") which asserts a
# proposition in question form.
_MODAL_I_RE = re.compile(
    r"^(?:should|shall|may|might|can|could|would|will|must)\s+(?:i|we)\b",
    re.IGNORECASE,
)

# Vague model-action plans ("Should I go ahead?", "Shall we proceed?"):
# intransitive, no information requested from the user — deliberation.
_VAGUE_PLAN_RE = re.compile(
    r"^(?:go\s+(?:ahead|on|forward)|proceed|carry\s+on|move\s+(?:on|forward)"
    r"|continue|start|begin)\b[\s,.?!]*$",
    re.IGNORECASE,
)

# First-person inversion that talks about the model's own method ("Am I
# assuming anything?", "Have I reasoned this through?", "Should I check
# the data?") is self-directed deliberation, not a question to the user —
# exactly the register of the hint pool, so echoing the hint must not read
# as asking the user.
_META_VERB_RE = re.compile(
    r"\b(?:assum|invent|fabricat|guess|reason|call|invok|hold\s+back|think"
    r"|thought|hallucinat|pretend|plan|order|sequenc|decide|double[- ]?check"
    r"|check|second[- ]?guess)",
    re.IGNORECASE,
)

# The same, minus "assume"/"guess": a modal permission question ("May I
# assume Paris?") proposes a reading of the user's words and asks them to
# confirm — a clarification even though the verb is metacognitive.  Only
# the bare first-person-inversion branch below needs those two.
_META_VERB_MODAL_RE = re.compile(
    r"\b(?:invent|fabricat|reason|call|invok|hold\s+back|think|thought"
    r"|hallucinat|pretend|plan|order|sequenc|decide|double[- ]?check|check"
    r"|second[- ]?guess)",
    re.IGNORECASE,
)

# Abstract nouns that cannot head a genuine clarification request: a bare
# "which <one of these>?" asks the user to grade the reply, not to supply a
# missing value.  "Which is correct?" and "What you said?" are rhetorical
# wherever they appear.
_RHETORICAL_HEAD_RE = re.compile(
    r"^(?:one\s+of\s+us"
    r"|is\s+(?:the\s+)?(?:correct|right|wrong|better|best|point|use"
    r"|difference|matter|problem|issue|deal|catch)"
    r"|(?:the\s+)?(?:point|use|difference|matter|problem|issue|deal|catch)"
    r"|(?:which|what|that|who)\s+you\s+(?:already\s+|may\s+have\s+)?"
    r"(?:know|said|asked|meant|think|want)"
    r"|\bi\s+(?:should\s+know|am\s+i\s+to\s+say))",
    re.IGNORECASE,
)

# The same shapes through other syntax ("What's the problem?", "Is that
# the point?", "Which one is correct?"), matched against the whole clause
# because they need not open with a wh-word.
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


def _interrogative_clauses(text: str):
    """Yield each clause body that is terminated by a ``?``."""
    start = 0
    for m in _TERMINATOR_RE.finditer(text):
        if m.group(0) == "?":
            yield text[start:m.start()]
        start = m.end()
    if start < len(text):
        yield text[start:]


def _all_clauses(text: str):
    """Yield each clause body (question or not) as split by ``_TERMINATOR_RE``.

    The final segment keeps its ``?`` so callers can tell interrogative
    clauses from declarative ones (``_request_info`` strips it before
    classifying).
    """
    start = 0
    for m in _TERMINATOR_RE.finditer(text):
        yield text[start:m.start()] + m.group(0)
        start = m.end()
    if start < len(text):
        yield text[start:]


def _nominal_wh_asks(rest: str) -> bool:
    """True when a determiner wh-word (``what``/``which``/``whose``) asks.

    ``rest`` is what follows the wh-word.  A determiner use introduces a
    noun phrase and either stops there ("Which city?") or continues into an
    inverted clause ("Which city do you mean?").  A bare subject pronoun
    with no auxiliary in between ("which city you mean", "which you already
    know") is a relative clause, not a question — that is the shape a
    policy reaches for when it bolts a hint-echoed wh-word onto an answer.
    """
    if _RHETORICAL_HEAD_RE.match(rest):
        # "Which is correct?", "What you said?" — a request for the user's
        # verdict, not for a missing value.
        return False
    # Only the head of the clause decides: a determiner wh-word opens a
    # short noun phrase, followed either by an auxiliary (question) or
    # immediately by a subject pronoun (relative clause).  A pronoun further
    # along ("What date works best for you, ...") is an object.
    for tok in _WORD_RE.finditer(rest):
        word = tok.group(0)
        if _AUX_TOK_RE.match(word):
            return True
        if _SUBJ_PRON_RE.match(word):
            return False
        if tok.end() > _HEAD_WINDOW_CHARS:
            break
    return True


def _wh_leads_question(seg: str, *, tail: bool) -> bool:
    """True when ``seg`` opens with a wh-word used interrogatively."""
    seg = _WH_CONTRACTION_RE.sub(r"\1 is", seg)
    m = _WH_RE.match(seg)
    if not m:
        return False
    wh = m.group(0).lower()
    rest = seg[m.end():].lstrip()
    if not rest:
        return False
    if wh in _NOMINAL_WH:
        return _nominal_wh_asks(rest)
    if tail:
        # After a comma only a determiner wh-word can start the asking part
        # ("..., which city should I use?"); an adverbial one introduces an
        # adverbial clause ("..., when the data arrives").
        return False
    # "when is X", "who should I ask", "how many results" → inversion.
    if _AUX_RE.match(rest):
        return True
    return wh == "how" and bool(_HOW_QUANT_RE.match(rest))


def _asks(seg: str, *, tail: bool = False) -> bool:
    """True when ``seg`` is an interrogative that asks the user something."""
    seg = _LEAD_NOISE_RE.sub("", seg.strip())
    if not seg:
        return False
    # Unambiguous request phrases win over the rhetorical filter: "Could you
    # please tell me which city you mean?" contains a relative-looking
    # fragment but is a request, whereas the fragment alone is not.
    if any(r.search(seg) for r in _CLARIFY_RES):
        return True
    if _MODAL_YOU_RE.search(seg) and _REQUEST_VERB_RE.search(seg):
        return True
    if _RHETORICAL_RE.search(seg):
        return False
    if _wh_leads_question(seg, tail=tail):
        return True
    # "Should I use the London office?" asks the user for direction.  A
    # first-person inversion only reads as deliberation when it uses
    # metacognitive vocabulary ("Am I assuming anything I shouldn't?" — the
    # hint pool's register) or plans a vague model action with nothing to
    # confirm ("Should I go ahead?"); otherwise it still asks.
    m_modal_i = _MODAL_I_RE.match(seg)
    if (
        not tail
        and m_modal_i
        and not _META_VERB_MODAL_RE.search(seg)
        and not _VAGUE_PLAN_RE.search(seg[m_modal_i.end():].strip())
    ):
        return True
    # Subject-auxiliary inversion ("Is that the right city?") is a question
    # by construction once the clause ends in "?"; the guards that keep it
    # from leaking are clause locality, ask-dominance and ``_COURTESY_RE``.
    # A first-person modal inversion that IS deliberation ("Should I go
    # ahead?") falls through here and is suppressed by the meta/vague
    # vocabulary check.
    deliberation = bool(_META_VERB_RE.search(seg) or _VAGUE_PLAN_RE.search(seg))
    if not deliberation and m_modal_i:
        deliberation = bool(
            _VAGUE_PLAN_RE.search(seg[m_modal_i.end():].strip())
        )
    return bool(_AUX_RE.match(seg) and not deliberation)


def _is_clarification_clause(clause: str) -> bool:
    """True when an interrogative clause body itself asks for information."""
    body = clause.strip()
    if len(_WORD_RE.findall(body)) < _MIN_CLAUSE_WORDS:
        return False
    return _asks(body) or _asks(body.rsplit(",", 1)[-1], tail=True)


def _request_info(text: str) -> bool:
    """True when ``text`` poses a genuine question that asks for information.

    Two structural guards on top of the lexicon, because a fabricated
    answer with a question bolted on must not pass as an abstention:

    1. **Clause locality** — the ``?`` and the asking signal must live in
       the same clause, so a wh-word in one sentence cannot vouch for a
       ``?`` in another.
    2. **The question must be the reply** — the asking clauses must make up
       at least half of the visible reply's words.  A genuine abstention is
       mostly question; "The capital of France is Paris. Which city?" is
       mostly an answer.  Courtesy clauses are excluded from the count on
       both sides, so a sign-off does not dilute a real question.
    """
    total_words = 0
    for clause in _interrogative_clauses(text) if False else _all_clauses(text):
        if _COURTESY_RE.search(clause):
            continue
        total_words += len(_WORD_RE.findall(clause))
    if total_words == 0:
        return False

    asking_words = 0
    found = False
    for clause in _all_clauses(text):
        if not clause.endswith("?"):
            continue
        clause = clause[:-1]
        if _COURTESY_RE.search(clause):
            continue
        if _is_clarification_clause(clause):
            found = True
            asking_words += len(_WORD_RE.findall(clause))
    if not found:
        return False
    # The asking part must dominate: at least half the reply's words AND
    # strictly more than the declarative remainder ("The answer is 42.
    # Which city?" has a 3-word answer and a 2-word question → still a
    # guess).
    return asking_words * 2 >= total_words and asking_words > total_words - asking_words


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
    if _request_info(text):
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


def _reference_shows_tool_call(reference: str) -> bool:
    """True when the reference response itself emits tool call(s).

    A demonstrated tool call is the opposite of a demonstrated direct
    answer.  Without this check a reference like
    ``[Get Competition Standings(timezone=-8.0, ...)]`` classified as
    ``GUESS`` and wrongly exempted the guess penalty — which is how a
    correct tool call scored *below* a fabricated answer on labels that a
    parser had silently emptied.
    """
    # Imported lazily: verifier has no dependency on this module, so the
    # local import keeps the direction of the dependency obvious.
    from examples.tool_rl.reward.verifier import parse_qwen_tool_calls

    return bool(parse_qwen_tool_calls(reference))


def reference_prefers_answer(label: object) -> bool:
    """True when the label's reference response is itself a direct answer.

    A no-tool sample whose demonstrated behaviour is a direct answer
    (chitchat / general knowledge) must not guess-penalise the model for
    answering directly.  No reference → False (abstention samples keep the
    penalty): an empty reference classifies as ``GUESS``, so the emptiness
    check must come first.  A reference that demonstrates a tool call also
    returns False — it does not prefer a direct answer.
    """
    if not isinstance(label, str):
        return False
    m = _REFERENCE_RE.search(label)
    if not m or not m.group(1).strip():
        return False
    reference = m.group(1)
    if _reference_shows_tool_call(reference):
        return False
    return classify_abstention(reference) is AbstentionClass.GUESS


# ============================================================================
# Env config
# ============================================================================

def abstain_mode_from_env() -> str:
    """Read ``TOOL_RL_ABSTAIN_MODE``: ``keyword`` (default) | ``off``."""
    mode = os.environ.get("TOOL_RL_ABSTAIN_MODE", "keyword").strip().lower()
    if mode not in ("off", "keyword"):
        raise ValueError(f"TOOL_RL_ABSTAIN_MODE must be off|keyword, got {mode!r}")
    return mode
