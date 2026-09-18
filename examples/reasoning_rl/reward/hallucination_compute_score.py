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
"""Hallucination-resistance reward dispatcher (HALLUCINATION_RL_DESIGN.md section 6).

Mounted exactly like the four-domain dispatcher, but pointed at this file::

    reward.custom_reward_function.path=examples/reasoning_rl/reward/hallucination_compute_score.py
    reward.custom_reward_function.name=compute_score

Stage 2 of the reasoning-RL run swaps in this file *and* a new data mix; the old
four domains (`math_*` / `code_*` / `logic_*` / `stem_*` / `if_*`) are delegated
verbatim to ``compute_score.py`` so that mixing both in one parquet changes
nothing about their scoring.

Answer contract (design doc section 5.1)
----------------------------------------

===========================================  =====================================
situation                                    model writes, last line
===========================================  =====================================
solvable, numeric/expression                  ``\\boxed{<answer>}``
solvable, role-word sequence (K&K)            ``\\boxed{<role word> <role word> …}``
solvable, "is this answerable?" only (D14)    ``\\boxed{SOLVABLE}``
unsolvable, with diagnosis label              ``\\boxed{UNSOLVABLE: <option id>}``
unsolvable, without diagnosis label (D12)     ``\\boxed{UNSOLVABLE}``
===========================================  =====================================

Only the **last** balanced ``\\boxed{}`` decides the score; a generation without
one is `status="none"` and scores 0.  Nothing about the reasoning text itself is
graded (design decision D4 dropped the consistency aux term), so a model may say
"contradiction" mid-thought without penalty.

The main reward is ``{+1, 0, -1}`` (design doc section 6).  ``-1`` is symmetric
by construction: fabricating an answer on an unsolvable row *and* refusing on a
solvable row are both punished.

Routing
-------

``data_source`` selects only the *delegate* vs *hallucination* branch; inside the
hallucination branch the ground-truth JSON fields drive the decision (design doc
section 6: "判定依据是 ground_truth 里是否存在 role_words 键，而不是
data_source").  That keeps a new source source-agnostic: any row whose
``ground_truth`` carries ``solvable``/``role_words``/``judgment_only`` is scored
correctly without touching this file.

Deviations from the design doc (each one is a place the doc contradicts itself;
see the ``DESIGN CONFLICTS`` notes in test_hallucination_compute_score.py):

1. **Misrefusal is ``-1`` on every solvable branch**, not just on the
   ``judgment_only`` one.  Doc section 6's pseudocode returns 0 for a numeric
   solvable row answered with ``\\boxed{UNSOLVABLE}``, while section 9's
   six-branch test matrix lists exactly that as the ``-1`` ("错向") cell, and
   says so again for the role-word branch.  We follow the matrix: "don't abstain
   on a solvable problem" is the whole point of the domain, and splitting the
   meaning of ``-1`` by branch would make the two templates differ in a way a
   model can see.
2. **No special case for ``halluc_math_gsmic``.**  Doc section 6 short-circuits
   GSM-IC to ``_math_score(solution_str, gt.answer)`` before parsing the ground
   truth.  That path cannot express the misrefusal penalty and would score a
   1,000-row-per-epoch pool differently from every other numeric pool.  Numeric
   solvable rows route through ``math_match``, which is ``_math_score`` on the
   extracted answer — same matcher, no new comparison logic (section 6's actual
   requirement).
3. **``\\boxed{UNSOLVABLE: B}`` on a three-tier row scores +1.**  Doc sections
   5.1 and 6 both say so explicitly ("识别出不可解即满分"); section 9's matrix
   lists it as the 0 cell.  We follow 5.1/6 and treat the spurious option id as
   format jitter, not as a wrong answer.  Section 10 already tracks the rate of
   option-shaped output on option-less sources as a monitoring signal, which is
   where that anomaly belongs.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys

logger = logging.getLogger(__name__)

try:
    from examples.reasoning_rl.reward.compute_score import _extract_boxed
    from examples.reasoning_rl.reward.compute_score import _math_score
    from examples.reasoning_rl.reward.compute_score import compute_score as _base_compute_score
    from examples.reasoning_rl.reward.compute_score import format_ok
    from examples.reasoning_rl.reward.compute_score import logic_answer_match
except ImportError:
    # Fallback when run as a plain module without the repo on sys.path.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from compute_score import _extract_boxed, _math_score, format_ok, logic_answer_match
    from compute_score import compute_score as _base_compute_score

# The prefix that marks a row as belonging to the hallucination domain.  It is
# deliberately *not* one of the four stage-1 prefixes (math/code/logic/stem/if),
# so a row can never be scored by both dispatchers.
HALLUC_PREFIX = "halluc_"

# ``extract_final`` status values (design doc section 6).
STATUS_ANSWER = "answer"
STATUS_SOLVABLE_MARKER = "solvable_marker"
STATUS_UNSOLVABLE_OPTION = "unsolvable_option"
STATUS_UNSOLVABLE_BARE = "unsolvable_bare"
STATUS_NONE = "none"

_ABSTAIN_STATUSES = frozenset({STATUS_UNSOLVABLE_BARE, STATUS_UNSOLVABLE_OPTION})

# ``\boxed{SOLVABLE}`` / ``\boxed{UNSOLVABLE}`` / ``\boxed{UNSOLVABLE: B}``.
# The optional trailing punctuation absorbs "UNSOLVABLE." and "**UNSOLVABLE**"
# once the markdown emphasis is stripped (prompts ask for a bare marker, so this
# only guards against harmless presentation noise).
_MARKER_STRIP_RE = re.compile(r"^[*`\s]+|[*`\s.!。！]+$")
_SOLVABLE_RE = re.compile(r"^SOLVABLE$", re.IGNORECASE)
_UNSOLVABLE_RE = re.compile(r"^UNSOLVABLE\s*(?:[:\-—]\s*(.*))?$", re.IGNORECASE | re.DOTALL)
# An option id is a single letter (design decision D15 fixes k=3 -> A/B/C).  A
# longer "id" such as ``UNSOLVABLE: missing the number of sprints`` is prose, not
# an option, and is treated as a bare refusal instead of silently scoring 0.
_OPTION_ID_RE = re.compile(r"^([A-Za-z])$")
# A response may decorate the marker with a short preamble inside the box
# ("The premise is false: UNSOLVABLE"). Keep the strict forms above for the
# common case and fall back to this only for the *bare* marker, never for an
# option id -- accepting a trailing option id would make the four-tier branch
# scorable by a substring match.
_UNSOLVABLE_TAIL_RE = re.compile(r"\bUNSOLVABLE\b", re.IGNORECASE)

# Role-word tokenisation (K&K only).  Anything that is not alphanumeric
# separates tokens, so "knave, knight" and "knave/knight" both parse.
_ROLE_SEP_RE = re.compile(r"[^a-z0-9]+")
_ROLE_ARTICLES = frozenset({"a", "an", "the"})


# ---------------------------------------------------------------------------
# final-answer extraction
# ---------------------------------------------------------------------------


def extract_final(solution_str: str) -> tuple[str, str | None, str | None]:
    """Classify the last ``\\boxed{}`` of a response.

    Returns ``(status, answer, option)`` where ``status`` is one of the five
    constants above.  ``answer`` is the boxed text for ``STATUS_ANSWER`` and
    ``None`` otherwise; ``option`` is the upper-cased single-letter id for
    ``STATUS_UNSOLVABLE_OPTION`` and ``None`` otherwise.

    The classification order matters: ``UNSOLVABLE: B`` must not be mistaken for
    an ordinary answer whose text happens to start with "UNSOLVABLE", and
    ``SOLVABLE`` must not be mistaken for an answer that is the literal word
    "solvable" -- in both cases the dedicated marker wins, because the prompts
    reserve those two words for the verdict.
    """
    boxed = _extract_boxed(solution_str)
    if boxed is None:
        return STATUS_NONE, None, None
    text = _MARKER_STRIP_RE.sub("", boxed.strip())
    if not text:
        # ``\boxed{}`` / ``\boxed{ }``: a marker-shaped hole is not an answer.
        return STATUS_NONE, None, None
    if _SOLVABLE_RE.match(text):
        return STATUS_SOLVABLE_MARKER, None, None
    match = _UNSOLVABLE_RE.match(text)
    if match is not None:
        tail = (match.group(1) or "").strip()
        option = tail.upper() if _OPTION_ID_RE.match(tail) else None
        if option is not None:
            return STATUS_UNSOLVABLE_OPTION, None, option
        return STATUS_UNSOLVABLE_BARE, None, None
    if _UNSOLVABLE_TAIL_RE.search(text):
        # "The information given is insufficient -- UNSOLVABLE".  Still a refusal
        # (any option id in it is unreadable), so it earns the bare-marker branch.
        return STATUS_UNSOLVABLE_BARE, None, None
    return STATUS_ANSWER, text, None


# ---------------------------------------------------------------------------
# K&K role-word matching
# ---------------------------------------------------------------------------


def _role_tokens(text: str) -> list[str]:
    """Split a role-word answer into content tokens, dropping articles."""
    tokens = [t for t in _ROLE_SEP_RE.split(text.casefold()) if t]
    return [t for t in tokens if t not in _ROLE_ARTICLES]


def _role_map(role_words) -> dict[str, bool]:
    """Map the row's own surface words to canonical booleans.

    ``role_words`` is ``(truth_teller_word, liar_word)`` as written in *this*
    row's prompt -- for the ``flip_role`` family it is ``("knave", "knight")``,
    for ``random_pair`` ``("angel", "devil")``.  Reading it from the ground truth
    instead of hard-coding knight/knave is what keeps the reward from inverting
    on the 28.7% of K&K rows that do not use the canonical words
    (HALLUCINATION_RL_DESIGN.md section 5.1 / risk 9).
    """
    words = {}
    for word, is_truth_teller in zip(role_words, (True, False), strict=False):
        for key in _role_word_keys(word):
            words[key] = is_truth_teller
    return words


def _role_word_keys(word: str) -> set[str]:
    """Case-folded singular forms a surface word may appear as.

    "knights" / "a knight" / "knight's" all have to reach the same key: the
    prompts of the six perturbation families spell the two role words
    inconsistently (``knight`` vs ``knights``), and a plural must not be scored
    as an out-of-vocabulary token.
    """
    base = _ROLE_SEP_RE.sub("", word.casefold())
    if not base:
        return set()
    keys = {base}
    if base.endswith("s"):
        keys.add(base[:-1])
    if base.endswith("es"):
        keys.add(base[:-2])
    return keys


def _canonical_role_sequence(text: str, roles: dict[str, bool]) -> list[bool] | None:
    """Surface role-word answer -> canonical truth-teller booleans, or None.

    A single out-of-vocabulary token, or an empty answer, returns None: the
    contract asks for the role words **in the order the inhabitants are listed**,
    and a prose answer ("Ethan is a knave, Abigail is a knight") carries the
    inhabitants' names as extra tokens.  Being lenient there would silently
    accept the wrong format and teach the model to ignore the instruction.
    """
    tokens = _role_tokens(text)
    if not tokens:
        return None
    out = []
    for token in tokens:
        value = roles.get(token)
        if value is None and token.endswith("s"):
            value = roles.get(token[:-1])
        if value is None:
            return None
        out.append(value)
    return out


def kk_match(prediction: str | None, role_words, ground_truth_answer: str) -> bool:
    """Score a K&K role-word sequence against the row's own role vocabulary.

    Both sides are mapped through ``role_words`` before comparing, so the
    comparison happens in canonical boolean space and cannot invert.
    """
    if prediction is None or not ground_truth_answer:
        return False
    if not role_words or len(role_words) < 2:
        # No vocabulary means the adapter violated the K&K contract; refuse to
        # guess a mapping rather than risk scoring the opposite of the truth.
        logger.warning("[halluc] K&K row without a usable role_words list; scoring 0")
        return False
    roles = _role_map(role_words)
    predicted = _canonical_role_sequence(prediction, roles)
    expected = _canonical_role_sequence(ground_truth_answer, roles)
    if predicted is None or expected is None:
        return False
    return predicted == expected


# ---------------------------------------------------------------------------
# numeric / expression matching
# ---------------------------------------------------------------------------


def math_match(prediction: str | None, ground_truth: str) -> bool:
    """Compare an already-extracted answer with the gold, using stage-1 matchers.

    ``_math_score`` parses the prediction out of the string it is handed, so the
    extracted answer is re-wrapped in ``\\boxed{}``: that is the shape every
    prompt in the mix asks for and the shape whose brace balancing
    ``_extract_boxed`` has already validated.

    The three layers are the ones DESIGN.md section 6 already promises for
    ``math_*`` -- "math_verify（pip install math-verify）；math_verify 抛错（进程池
    损坏等）**或缺失时** fallback math_dapo（先 \\boxed{} 严格匹配，再 Minerva
    Answer:）".  They are spelled out here instead of delegated wholesale to
    ``_math_score`` because that helper cannot reach its own fallback in the
    "package missing" case: ``verl.utils.reward_score.math_verify.compute_score``
    catches the ``ImportError`` internally, prints, and returns ``0.0`` -- so it
    never raises the ``ImportError`` ``_math_score`` is waiting for, and every
    math row silently scores 0.  Each layer can only *add* credit, so this is
    strictly at least as permissive as the stage-1 path (which is what
    HALLUCINATION_RL_DESIGN.md section 6 asks for: "复用现有数学匹配路径 ... 不新写
    比较逻辑").
    """
    if prediction is None or ground_truth is None:
        return False
    ground_truth = str(ground_truth).strip()
    if not ground_truth:
        return False
    boxed = f"\\boxed{{{prediction}}}"
    try:
        if _math_score(boxed, ground_truth) > 0.5:
            return True
    except Exception:  # never let one bad sample kill the reward worker
        logger.exception("[halluc] math_match: math_verify path crashed")
    try:
        from verl.utils.reward_score import math_dapo

        for strict_box in (True, False):
            res = math_dapo.compute_score(boxed, ground_truth, strict_box_verify=strict_box)
            acc = res.get("acc") if isinstance(res, dict) else res
            if acc:
                return True
    except Exception:
        logger.exception("[halluc] math_match: math_dapo path crashed")
    # Last resort: the stage-1 structural/text comparison, which covers plain
    # integers and expressions that need no symbolic parser (the common case for
    # GSM-IC / SUM / UMWP answers).  Like the layers above it only adds credit.
    try:
        return logic_answer_match(prediction, ground_truth)
    except Exception:
        logger.exception("[halluc] math_match: text path crashed")
        return False


# ---------------------------------------------------------------------------
# ground-truth contract
# ---------------------------------------------------------------------------


def parse_ground_truth(ground_truth) -> dict | None:
    """Parse ``reward_model.ground_truth`` into the hallucination payload.

    The parquet column stores a JSON **string** so new fields never require a
    parquet schema change (design doc section 3).  A dict is accepted too, which
    is what the unit tests and the offline reporting tools pass in.
    """
    if isinstance(ground_truth, dict):
        return ground_truth
    if not isinstance(ground_truth, str):
        return None
    try:
        payload = json.loads(ground_truth)
    except (ValueError, TypeError):
        # ValueError covers JSONDecodeError and the plain ValueError CPython
        # raises for an integer literal past sys.get_int_max_str_digits().
        return None
    return payload if isinstance(payload, dict) else None


def score_from_status(ground_truth: dict, status: str, answer: str | None, option: str | None) -> float:
    """The decision table of design doc sections 6 / 9 (see the module docstring).

    Split out from ``compute_score`` so the whole six-branch matrix can be
    exercised without building a full ``solution_str``.
    """
    if "solvable" not in ground_truth:
        logger.warning("[halluc] ground_truth has no 'solvable' key; scoring 0")
        return 0.0
    solvable = bool(ground_truth["solvable"])

    if solvable:
        # Abstaining on a row that has an answer is a misrefusal, whichever
        # template the row came from (deviation 1 in the module docstring).
        if status in _ABSTAIN_STATUSES:
            return -1.0
        if ground_truth.get("judgment_only"):
            # D14: the row asks "is this answerable?" and the gold is SOLVABLE.
            # A numeric or role-word answer is not a verdict -> 0, and an answer
            # that exists (e.g. FalseQA label=0's free text) is never compared.
            return 1.0 if status == STATUS_SOLVABLE_MARKER else 0.0
        if status != STATUS_ANSWER:
            return 0.0
        role_words = ground_truth.get("role_words")
        if role_words:
            return 1.0 if kk_match(answer, role_words, ground_truth.get("answer") or "") else 0.0
        return 1.0 if math_match(answer, ground_truth.get("answer")) else 0.0

    # Unsolvable: the model must refuse.  Fabricating an answer costs more than
    # refusing imprecisely, because a fabricated answer is exactly the failure
    # mode this domain exists to remove.
    if status == STATUS_ANSWER:
        return -1.0
    if status not in _ABSTAIN_STATUSES:
        return 0.0
    if not ground_truth.get("has_diagnosis_label"):
        # Three-tier source (MiP, SUM-deletion, TreeCut, UMWP cat1, CREPE,
        # KUQ): naming the right verdict is the whole task.  An option id here
        # is meaningless -- the prompt carries no option block -- so it does not
        # demote the score (deviation 3 in the module docstring).
        return 1.0
    # Four-tier source: the verdict alone is worth 0, only the pointer earns +1.
    correct = ground_truth.get("correct_option_id")
    if correct is None:
        logger.warning("[halluc] four-tier row without correct_option_id; no refusal can score")
        return 0.0
    return 1.0 if (status == STATUS_UNSOLVABLE_OPTION and option == str(correct).upper()) else 0.0


def score_halluc_row(data_source: str, solution_str: str, ground_truth, extra_info=None) -> float:
    """Score one hallucination-domain row (format gate + extraction + table)."""
    if not format_ok(solution_str):
        return 0.0
    payload = parse_ground_truth(ground_truth)
    if payload is None:
        logger.warning("[halluc] unparseable ground_truth for %s; scoring 0", data_source)
        return 0.0
    try:
        status, answer, option = extract_final(solution_str)
        return float(score_from_status(payload, status, answer, option))
    except Exception:
        logger.exception("[halluc] scoring crashed for %s; scoring 0", data_source)
        return 0.0


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
):
    """Route ``halluc_*`` rows to the abstention scorer, everything else to stage 1.

    Returns ``{"score": float}`` so the naive reward manager lifts ``score`` into
    ``reward_extra_info`` and DAPO ``filter_groups.metric=score`` keeps working.
    ``**kwargs`` (sandbox URL, semaphores, memory limit, ...) are forwarded to the
    stage-1 dispatcher untouched, so a single mixed parquet serves both.
    """
    if not isinstance(data_source, str) or not data_source.startswith(HALLUC_PREFIX):
        return _base_compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)
    try:
        return {"score": score_halluc_row(data_source, solution_str, ground_truth, extra_info)}
    except Exception:
        logger.exception("[halluc] dispatcher crashed for %s; scoring 0", data_source)
        return {"score": 0.0}
