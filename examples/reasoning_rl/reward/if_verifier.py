# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Verifiable-constraint checker for the instruction-following domain.

Implements the Open-Instruct / IFEval-style constraint set used by
``nvidia/Nemotron-RL-instruction_following`` (WildChat prompts + verifiable
constraints, NeMo-Gym convention: reward 1.0 iff EVERY constraint in
``instruction_id_list`` passes).

Each constraint is ``{"id": "<category>:<type>", "kwargs": {...}}``.  Ids
follow the IFEval/Open-Instruct taxonomy, e.g.::

    keywords:existence            {keywords: ["a", "b"]}
    keywords:frequency            {keyword: "the", frequency: 3, relation: "at least"}
    length_constraints:number_words      {num_words: 200, relation: "less than"}
    detectable_format:number_bullet_lists {num_bullets: 3}
    startend:end_checker          {end_phrase: "Is there anything else I can help with?"}
    change_case:english_lowercase {}
    punctuation:no_comma          {}

The checker is self-contained (no third-party deps) and tolerant of malformed
constraints: unknown ids / bad kwargs return False for that constraint rather
than raising, so one bad sample never kills the reward pass.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

_RELATIONS = {
    "less than": lambda a, b: a < b,
    "at least": lambda a, b: a >= b,
    "at most": lambda a, b: a <= b,
    "exactly": lambda a, b: a == b,
    "equal to": lambda a, b: a == b,
    "more than": lambda a, b: a > b,
}


def _check_relation(count: float, relation: str | None, target: float) -> bool:
    if target is None:
        return False
    rel = (relation or "").strip().lower()
    fn = _RELATIONS.get(rel)
    if fn is None:
        # Default to ">=", the most common IFEval semantics.
        fn = _RELATIONS["at least"]
    return fn(count, target)


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text)


def _num_words(text: str) -> int:
    return len(_words(text))


def _sentences(text: str) -> list[str]:
    # Lightweight sentence splitter: split on ., !, ? followed by whitespace/EOS.
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _paragraphs(text: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def _kw(kwargs: dict, *names, default=None):
    for n in names:
        if n in kwargs and kwargs[n] is not None:
            return kwargs[n]
    return default


# ---------------------------------------------------------------------------
# per-category checkers
# ---------------------------------------------------------------------------


def _check_keywords(cid: str, text: str, kw: dict) -> bool:
    lower = text.lower()
    if cid == "existence":
        kws = _kw(kw, "keywords", "keyword", default=[])
        if isinstance(kws, str):
            kws = [kws]
        return all(str(k).lower() in lower for k in kws)
    if cid == "frequency":
        k = str(_kw(kw, "keyword", "word", default="")).lower()
        if not k:
            return False
        n = lower.count(k)
        return _check_relation(n, _kw(kw, "relation"), _kw(kw, "frequency", "num_occurrences"))
    if cid == "forbidden_words":
        kws = _kw(kw, "forbidden_words", "keywords", "words", default=[])
        if isinstance(kws, str):
            kws = [kws]
        return all(str(k).lower() not in lower for k in kws)
    if cid == "letter_frequency":
        letter = str(_kw(kw, "letter", default="")).lower()
        if len(letter) != 1:
            return False
        n = lower.count(letter)
        return _check_relation(n, _kw(kw, "relation"), _kw(kw, "frequency", "num_occurrences"))
    return False


def _check_language(cid: str, text: str, kw: dict) -> bool:
    if cid != "response_language":
        return False
    lang = str(_kw(kw, "language", "lang", default="")).lower()
    if not lang:
        return False
    # Heuristic, dependency-free language check.
    if lang in ("en", "english"):
        # English iff the overwhelming majority of alpha chars are ASCII.
        alpha = [c for c in text if c.isalpha()]
        if not alpha:
            return False
        ascii_ratio = sum(1 for c in alpha if ord(c) < 128) / len(alpha)
        return ascii_ratio > 0.9
    if lang in ("zh", "chinese", "mandarin"):
        return any("一" <= c <= "鿿" for c in text)
    # Fallback: non-ASCII-dominated text counts as "a non-English language".
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return False
    return sum(1 for c in alpha if ord(c) >= 128) / len(alpha) > 0.5


def _check_length(cid: str, text: str, kw: dict) -> bool:
    if cid == "number_words":
        return _check_relation(_num_words(text), _kw(kw, "relation"), _kw(kw, "num_words", "num"))
    if cid == "number_sentences":
        return _check_relation(len(_sentences(text)), _kw(kw, "relation"), _kw(kw, "num_sentences", "num"))
    if cid == "number_paragraphs":
        return _check_relation(len(_paragraphs(text)), _kw(kw, "relation"), _kw(kw, "num_paragraphs", "num"))
    if cid == "number_letters":
        n = sum(1 for c in text if c.isalpha())
        return _check_relation(n, _kw(kw, "relation"), _kw(kw, "num_letters", "num"))
    if cid == "nth_paragraph_first_word":
        paras = _paragraphs(text)
        idx = _kw(kw, "num_paragraphs", "n", "index")
        first = str(_kw(kw, "first_word", default="")).lower()
        try:
            i = int(idx) - 1
        except (TypeError, ValueError):
            return False
        if not (0 <= i < len(paras)) or not first:
            return False
        ws = _words(paras[i])
        return bool(ws) and ws[0].lower() == first
    return False


def _check_detectable_content(cid: str, text: str, kw: dict) -> bool:
    lower = text.lower()
    if cid == "number_placeholders":
        targets = _kw(kw, "placeholders", "placeholder", default=[])
        if isinstance(targets, str):
            targets = [targets]
        return all(str(t).lower() in lower for t in targets)
    if cid == "postscript":
        marker = str(_kw(kw, "postscript_marker", "marker", default="P.S."))
        return marker.lower() in lower
    if cid == "number_bullet_lists" or cid == "number_bullets":
        bullets = re.findall(r"^\s*(?:[-*+]|\d+[.)])\s+\S", text, re.MULTILINE)
        return _check_relation(len(bullets), _kw(kw, "relation", default="exactly"), _kw(kw, "num_bullets", "num"))
    if cid == "constrained_response":
        options = _kw(kw, "options", "choices", default=["yes", "no", "maybe"])
        if isinstance(options, str):
            options = [options]
        return lower.strip() in {str(o).lower() for o in options}
    if cid == "number_highlighted_sections":
        # Markdown highlights like *text* or **text**.
        n = len(re.findall(r"\*[^*\n]+\*", text))
        return _check_relation(n, _kw(kw, "relation", default="at least"), _kw(kw, "num_highlights", "num"))
    if cid == "multiple_sections":
        splitter = str(_kw(kw, "section_splitter", "splitter", default="Section"))
        n = len(re.findall(rf"^\s*{re.escape(splitter)}\s+\d+", text, re.MULTILINE | re.IGNORECASE))
        return _check_relation(n, _kw(kw, "relation", default="at least"), _kw(kw, "num_sections", "num"))
    if cid == "number_repeat_prompt":
        # "repeat the request": count occurrences of the prompt_to_repeat.
        target = str(_kw(kw, "prompt_to_repeat", "prompt", default="")).lower()
        if not target:
            return False
        return target in lower
    return False


def _check_detectable_format(cid: str, text: str, kw: dict) -> bool:
    if cid == "number_bullet_lists":
        return _check_detectable_content("number_bullet_lists", text, kw)
    if cid == "constrained_response":
        return _check_detectable_content("constrained_response", text, kw)
    if cid == "number_highlighted_sections":
        return _check_detectable_content("number_highlighted_sections", text, kw)
    if cid == "multiple_sections":
        return _check_detectable_content("multiple_sections", text, kw)
    if cid == "json_format":
        stripped = text.strip()
        if not (stripped.startswith("{") and stripped.endswith("}")) and not (
            stripped.startswith("[") and stripped.endswith("]")
        ):
            return False
        import json

        try:
            json.loads(stripped)
            return True
        except (json.JSONDecodeError, ValueError):
            return False
    if cid == "title":
        return bool(re.search(r"<<[^>\n]+>>", text))
    return False


def _check_combination(cid: str, text: str, kw: dict) -> bool:
    if cid == "two_responses":
        # IFEval: two responses separated by 6 asterisks "******".
        parts = [p for p in text.split("******") if p.strip()]
        return len(parts) == 2
    if cid == "repeat_prompt":
        return _check_detectable_content("number_repeat_prompt", text, kw)
    return False


def _check_startend(cid: str, text: str, kw: dict) -> bool:
    stripped = text.strip()
    if cid == "end_checker":
        phrase = str(_kw(kw, "end_phrase", "phrase", default="")).strip()
        return bool(phrase) and stripped.endswith(phrase)
    if cid == "quotation":
        return (
            (stripped.startswith('"') and stripped.endswith('"'))
            or (stripped.startswith("'") and stripped.endswith("'"))
            or (stripped.startswith("“") and stripped.endswith("”"))
        )
    return False


def _check_change_case(cid: str, text: str, kw: dict) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    if cid == "capital_word_frequency":
        caps = [w for w in _words(text) if w.isupper() and len(w) > 1]
        return _check_relation(len(caps), _kw(kw, "relation"), _kw(kw, "frequency", "num_occurrences"))
    if cid == "english_capital":
        return all(c.isupper() for c in letters)
    if cid == "english_lowercase":
        return all(c.islower() for c in letters)
    return False


def _check_punctuation(cid: str, text: str, kw: dict) -> bool:
    if cid == "no_comma":
        return "," not in text
    return False


_CATEGORY = {
    "keywords": _check_keywords,
    "language": _check_language,
    "length_constraints": _check_length,
    "detectable_content": _check_detectable_content,
    "detectable_format": _check_detectable_format,
    "combination": _check_combination,
    "startend": _check_startend,
    "change_case": _check_change_case,
    "punctuation": _check_punctuation,
}


def check_constraint(constraint: dict, response: str) -> bool:
    """Return True iff the response satisfies one constraint.

    Tolerant of malformed input: unknown ids / missing kwargs -> False.
    """
    if not isinstance(constraint, dict):
        return False
    cid_full = constraint.get("id") or constraint.get("instruction_id") or ""
    if ":" in cid_full:
        category, cid = cid_full.split(":", 1)
    else:
        category, cid = cid_full, ""
    fn = _CATEGORY.get(category)
    if fn is None:
        return False
    kwargs = constraint.get("kwargs") or {}
    if not isinstance(kwargs, dict):
        kwargs = {}
    try:
        return bool(fn(cid, response, kwargs))
    except Exception:
        return False


def verify_instructions(response: str, constraints: list[dict]) -> tuple[float, list[bool]]:
    """NeMo-Gym convention: reward 1.0 iff EVERY constraint passes.

    Returns (score, per_constraint_follow_list).
    """
    if not constraints:
        return 0.0, []
    follow = [check_constraint(c, response) for c in constraints]
    return (1.0 if all(follow) else 0.0), follow
