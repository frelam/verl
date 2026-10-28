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
"""Verifier for the stage-4 BBH-gap domain (HALLUCINATION_RL_DESIGN.md D32).

The stage-3 checkpoint is weakest on five BIG-Bench-Hard subtasks (see
``BBH_Benchmark_评测分析报告.md``): ``dyck_languages`` (40%), ``disambiguation_qa``
(64%), ``causal_judgement`` (64.7%), ``movie_recommendation`` (66.4%) and
``geometric_shapes`` (74%).  Stage 4 adds fresh training rows for those skills,
built by ``scripts/hallucination/bbh_adapter.py``.  This module scores them.

The rows keep the benchmark's own answer shape instead of imposing the
abstention domain's ``\\boxed{UNSOLVABLE}`` contract:

=====================  ==============================================  ===============
task                   expected answer                                  gold
=====================  ==============================================  ===============
``dyck_languages``     the missing closing brackets, e.g. ``] )``      bracket list
``geometric_shapes``   one option letter, e.g. ``(B)`` or ``B``         letter
``disambiguation_qa``  ``(A)`` / ``(B)`` / ``(C)``                      letter
``causal_judgement``   ``Yes`` / ``No``                                 yes/no
``movie_recommendation`` one option letter, e.g. ``(C)``                letter
=====================  ==============================================  ===============

Both a ``\\boxed{...}`` answer and a bare final-line answer are accepted -- the
BBH prompts themselves do not ask for ``\\boxed{}``, so a model that answers in
the benchmark's native shape must not be punished.  For Dyck the bracket tokens
are compared (whitespace is presentation); for multiple-choice rows the option
letter is compared case-insensitively -- the explicitly stated verdict wins, a
letter merely mentioned mid-sentence is never read, and a fragment that only
ends with ``(B)`` is read as that choice; for causal judgement the stated
verdict wins over an earlier polarity token, else the last yes/no in the
visible response decides.  Every other shape scores 0.

The verifier never raises: a malformed ground truth or an unparseable response
is a 0, which is what the dispatcher's fail-closed contract expects.
"""

from __future__ import annotations

import json
import re

#: The five tasks this domain owns.
TASK_DYCK = "dyck_languages"
TASK_GEOMETRIC = "geometric_shapes"
TASK_DISAMBIGUATION = "disambiguation_qa"
TASK_CAUSAL = "causal_judgement"
TASK_MOVIE = "movie_recommendation"

TASKS = (TASK_DYCK, TASK_GEOMETRIC, TASK_DISAMBIGUATION, TASK_CAUSAL, TASK_MOVIE)
MCQ_TASKS = frozenset({TASK_GEOMETRIC, TASK_DISAMBIGUATION, TASK_MOVIE})

#: ``data_source`` -> task, for the rows whose payload omits ``task``.
SOURCE_TO_TASK = {
    "bbh_dyck": TASK_DYCK,
    "bbh_geometric_shapes": TASK_GEOMETRIC,
    "bbh_disambiguation_qa": TASK_DISAMBIGUATION,
    "bbh_causal_judgement": TASK_CAUSAL,
    "bbh_movie_recommendation": TASK_MOVIE,
}

_BOXED_RE = re.compile(r"\\boxed\s*\{")
_BRACKET_RE = re.compile(r"[()\[\]{}<>]")
_BRACKET_LINE_RE = re.compile(r"[()\[\]{}<>][\s()\[\]{}<>]*")
_TRAILING_BRACKETS_RE = re.compile(r"[()\[\]{}<>][\s()\[\]{}<>]*$")

#: A line that is just ``B`` / ``(B)`` / ``[B]`` / ``B.`` (after markdown noise
#: is stripped).  Used both for a boxed payload and for the last line.
_LETTER_LINE_RE = re.compile(r"^[\s(\[]*([A-Za-z])[\s)\]]*[.:]?$")
#: The fragment ENDS with an option letter ("... therefore (B).").  A letter
#: mentioned in the middle of prose is not read at all: "Option (A) is not
#: correct" must not credit A.
_TRAILING_PAREN_LETTER_RE = re.compile(r"\(([A-Za-z])\)\s*[.!。！]?\s*$")
#: An explicitly stated answer letter ("the answer is (B)", "answer: C").
#: Deliberately narrower than "option|choice": a first-match scan that accepts
#: ``option (A)`` reads "Option (A) is wrong; the answer is (B)" as A -- i.e.
#: it credits a letter the model just rejected.
_ANSWER_LETTER_RE = re.compile(
    r"\banswer\s*(?:is|:|=)\s*\**\s*\(?([A-Za-z])\)?(?![A-Za-z])", re.IGNORECASE
)
#: "option B is correct" / "choice (C) seems best": an affirmation that does
#: not use the word "answer".  Requires the affirmation, so a rejected option
#: ("option (A) is wrong") never matches.
_CORRECT_OPTION_RE = re.compile(
    r"\b(?:option|choice)\s*\(?([A-Za-z])\)?\s*(?:is|seems|appears)?\s*(?:the\s+)?(?:correct|right|best)\b",
    re.IGNORECASE,
)
_YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
#: An explicitly stated yes/no verdict ("... but the answer is no").
_ANSWER_YES_NO_RE = re.compile(r"\b(?:answer|conclusion|verdict)\s*(?:is|:|=)\s*\**\s*(yes|no)\b", re.IGNORECASE)

#: Presentation noise around a boxed / final-line payload.
_MARKUP_STRIP_RE = re.compile(r"^[*`_\"'\s]+|[*`_\"'.\s]+$")


def _last_boxed(text: str) -> str | None:
    """Content of the last balanced ``\\boxed{...}`` (None when absent)."""
    starts = [match.end() for match in _BOXED_RE.finditer(text)]
    if not starts:
        return None
    start = starts[-1]
    depth = 1
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start:index]
    return None


def _visible(solution_str: str) -> str:
    """Everything after ``</think>``; the whole string when no think block exists."""
    head, sep, body = solution_str.partition("</think>")
    return (body if sep else head).strip()


def _last_nonempty_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return ""


def _bracket_tokens(text: str) -> list[str]:
    return _BRACKET_RE.findall(text)


def _strip_markup(text: str) -> str:
    return _MARKUP_STRIP_RE.sub("", text.strip())


def _extract_dyck(text: str) -> list[str] | None:
    """The predicted closing-bracket sequence, as a token list.

    Priority: the last boxed payload, then the last line made only of brackets,
    then a trailing bracket run.  The last fallback is bounded to the end of the
    response so brackets mentioned inside the reasoning cannot win.
    """
    boxed = _last_boxed(text)
    if boxed is not None:
        tokens = _bracket_tokens(boxed)
        if tokens:
            return tokens
    for line in reversed(text.splitlines()):
        candidate = _strip_markup(line)
        if candidate and _BRACKET_LINE_RE.fullmatch(candidate):
            tokens = _bracket_tokens(candidate)
            if tokens:
                return tokens
    match = _TRAILING_BRACKETS_RE.search(text.rstrip().rstrip(".!。！"))
    if match:
        tokens = _bracket_tokens(match.group(0))
        if tokens:
            return tokens
    return None


def _choice_from_text(text: str) -> str | None:
    """One answer-bearing fragment -> option letter, or None when ambiguous.

    Priority matters: the explicit verdict ("the answer is (B)") is read before
    any parenthesised letter, a mid-sentence mention is never read at all (a
    rejected option would otherwise be credited), and only a fragment that
    ENDS with ``(B)`` counts as a bare choice.
    """
    candidate = _strip_markup(text)
    match = _LETTER_LINE_RE.match(candidate)
    if match:
        return match.group(1).upper()
    match = _ANSWER_LETTER_RE.search(candidate)
    if match:
        return match.group(1).upper()
    match = _CORRECT_OPTION_RE.search(candidate)
    if match:
        return match.group(1).upper()
    match = _TRAILING_PAREN_LETTER_RE.search(candidate)
    if match:
        return match.group(1).upper()
    return None


def _extract_choice(text: str) -> str | None:
    """The predicted option letter, upper-cased, or None.

    Accepts ``\\boxed{(B)}``, ``\\boxed{B}``, a bare final-line ``(B)`` / ``B``,
    and an explicit "the answer is B" line.  The generic single-capital-letter
    scan is deliberately **not** used: an isolated "I" in ordinary prose would
    otherwise be read as option I.
    """
    boxed = _last_boxed(text)
    if boxed is not None:
        choice = _choice_from_text(boxed)
        if choice is not None:
            return choice

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines[-4:]):
        choice = _choice_from_text(line)
        if choice is not None:
            return choice
    return None


def _yes_no_from_text(text: str) -> str | None:
    """One fragment -> "yes"/"no", or None.

    The explicit verdict wins ("it might seem yes at first, but the answer is
    no" is a NO); otherwise the LAST polarity token decides, because the final
    clause is where the model states its conclusion.
    """
    match = _ANSWER_YES_NO_RE.search(text)
    if match:
        return match.group(1).lower()
    matches = _YES_NO_RE.findall(text)
    return matches[-1].lower() if matches else None


def _extract_yes_no(text: str) -> str | None:
    boxed = _last_boxed(text)
    if boxed is not None:
        verdict = _yes_no_from_text(boxed)
        if verdict is not None:
            return verdict
    line = _last_nonempty_line(text)
    if line:
        verdict = _yes_no_from_text(line)
        if verdict is not None:
            return verdict
    return _yes_no_from_text(text[-300:])


def _gold_choice(answer) -> str | None:
    candidate = _strip_markup(str(answer))
    match = _LETTER_LINE_RE.match(candidate)
    return match.group(1).upper() if match else None


def _gold_yes_no(answer) -> str | None:
    match = _YES_NO_RE.search(str(answer))
    return match.group(1).lower() if match else None


def verify_bbh(solution_str: str, ground_truth: str | dict, data_source: str | None = None) -> float:
    """Score one BBH-gap row as 1.0 (correct) or 0.0 (wrong / unparseable).

    ``ground_truth`` is the JSON string written by ``bbh_adapter.py``::

        {"task": "dyck_languages", "answer": "] )"}

    The task is read from the payload and falls back to the ``data_source``
    mapping, so an older artifact without ``task`` still scores.
    """
    try:
        payload = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    except (TypeError, ValueError):
        return 0.0
    if not isinstance(payload, dict):
        return 0.0
    task = str(payload.get("task") or SOURCE_TO_TASK.get(str(data_source or ""), ""))
    answer = payload.get("answer")
    if answer is None:
        return 0.0

    text = _visible(solution_str)
    if task == TASK_DYCK:
        prediction = _extract_dyck(text)
        gold = _bracket_tokens(str(answer))
        return 1.0 if prediction and gold and prediction == gold else 0.0
    if task in MCQ_TASKS:
        prediction = _extract_choice(text)
        gold = _gold_choice(answer)
        return 1.0 if prediction and gold and prediction == gold else 0.0
    if task == TASK_CAUSAL:
        prediction = _extract_yes_no(text)
        gold = _gold_yes_no(answer)
        return 1.0 if prediction and gold and prediction == gold else 0.0
    return 0.0


__all__ = [
    "MCQ_TASKS",
    "SOURCE_TO_TASK",
    "TASKS",
    "TASK_CAUSAL",
    "TASK_DISAMBIGUATION",
    "TASK_DYCK",
    "TASK_GEOMETRIC",
    "TASK_MOVIE",
    "verify_bbh",
]
