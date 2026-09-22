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
"""Four-domain reward dispatcher for reasoning RL (DESIGN.md section 6).

Mounted without touching verl source::

    reward.custom_reward_function.path=examples/reasoning_rl/reward/compute_score.py
    reward.custom_reward_function.name=compute_score
    # sandbox URL rides through reward_kwargs (merged into every call):
    +reward.custom_reward_function.reward_kwargs.sandbox_fusion_url=http://<host>/run_code

Routing (data_source prefix -> verifier):

===================  =====================================================
``math_*``           math_verify (``pip install math-verify``); falls back
                     to verl's math_dapo when the package is missing; last
                     resorts for ground-truth format noise: a k-digit rounded
                     decimal gt (DAPO keeps e.g. 0.333 for 1/3) accepts any
                     prediction inside its rounding bucket; a percent or
                     labelled-list decimal gt (``6.535%``; ``...Stock:
                     0.3114, ...``) accepts any boxed value inside its
                     round-or-truncate interval, recovering truncated
                     textbook decimals and coarser-precision predictions
                     (``6.54%``); a normalised
                     literal comparison recovers plain-text variable names
                     (Big-Math keeps e.g. ``y2 < y1 < y3``) that the symbolic
                     parser mangles; a canonical-polarity comparison recovers
                     yes/no ground truths stored in the problem's original
                     language (``Выполнимо`` vs ``Yes``); a bounded symbol-
                     identification search
                     recovers predictions that substitute a prompt condition
                     the gt left general (``max(C,T)``/``C+T`` vs ``T``/``2T``
                     when the prompt said C equals T); and fixed-point numeric
                     equivalence recovers predictions that evaluate a formula
                     the gt left unevaluated (``a(1 - 10\%)^2`` vs ``0.81a``)
``code_*``           sandbox_fusion when ``sandbox_fusion_url`` is set,
                     else prime_code local execution (smoke runs only)
``logic_*``          rule verifier: extract the final answer
                     (<answer> tags -> \\boxed{} -> "Final Answer:" /
                     "The answer is ..." line -> last fenced code block /
                     <begin_board> block -> the bare post-</think> response) and
                     compare with the ground truth after normalisation
                     (whitespace/case folding, markdown-emphasis stripping,
                     separator-spacing/quote-insensitive text compare,
                     literal/JSON structural compare, numeric tolerance,
                     integer-grid normalisation). Enigmata arithmetic
                     (game24/countdown), maze, stack_permutation, the four
                     sliding/shift puzzles, twiddle, hamiltonian path/cycle,
                     car_painting, campsite/star_battle boards, full_crosswords,
                     zebra_logic and tic_tac_toe answers are verified by
                     ``enigmata_verifier`` instead of string match — their stored
                     answers are prose, one of many valid solutions, or the
                     puzzle's *initial* state, so text comparison scored correct
                     responses 0;
                     Reasoning Gym answers that string comparison rejects are
                     re-checked with the library's own task verifier
                     (``reasoning_gym_verifier``), which accepts the many
                     equivalent-but-different countdown / word_ladder /
                     shortest_path answers.
``stem_*``           math_verify on the \\boxed{} answer; Dr.SCI prompts
                     already request the boxed format
``if_*``             instruction-following (Nemotron-RL-instruction_following):
                     re-check every constraint in ground_truth against the
                     model response; reward 1.0 iff ALL pass (NeMo-Gym
                     ``grading_mode="binary"``).  Only the final (post-</think>)
                     response is verified — the think block is internal
                     reasoning.  ``if_verifier`` implements the full
                     ``verifiable_instructions`` checker registry (54 ids,
                     including the ``paragraphs:*``/``first_word:*``/
                     ``last_word:*``/``count:*`` families the dataset uses);
                     an id it cannot evaluate is graded as failed, so
                     ``check_reward.py`` reports per-row coverage instead of
                     echoing.  Install ``langdetect`` for the reference
                     language checks (otherwise a Latin-script target language
                     such as Spanish cannot be recognised).
===================  =====================================================

Every branch returns ``{"score": float}`` so the naive reward manager lifts
``score`` into ``reward_extra_info`` and DAPO ``filter_groups.metric=score``
(plus the hard-replay pass-rate computation) works unchanged.

Format gate
-----------

Before routing, every response must follow the Qwen3-4B thinking-template
shape — one non-empty ``<think>...</think>`` block followed by a non-empty
final response (``format_ok``). Non-compliant generations score 0 regardless
of answer correctness, and never reach the code sandbox.

Repetition penalty
------------------

After routing, every score is repetition-adjusted (``_repetition_adjusted``):
a single word 4-gram occurring more than the length-scaled threshold
``max(15, total_ngrams // 50)`` marks a degenerate repetition loop; each
occurrence past the threshold costs 0.1, penalties from several looped
n-grams add up, and the total is floored at -1, so degenerate repetition
loops cannot collect full credit when their last copy happens to be right.
The absolute floor of 15 keeps short texts calibrated (15 occurrences there
is a death loop); the ratio term scales the threshold with response length
so connective reasoning phrases that legitimately recur across a 16k-24k
token CoT are not charged as loops.
"""

from __future__ import annotations

import ast
import itertools
import json
import logging
import math
import re
from collections import Counter
from decimal import Decimal

logger = logging.getLogger(__name__)

try:
    from examples.reasoning_rl.reward.enigmata_verifier import strip_prose_prefix, verify_enigmata
    from examples.reasoning_rl.reward.reasoning_gym_verifier import seed_from_extra_info, verify_reasoning_gym
except ImportError:
    # Fallback when run as a plain module without the repo on sys.path.
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from enigmata_verifier import strip_prose_prefix, verify_enigmata
    from reasoning_gym_verifier import seed_from_extra_info, verify_reasoning_gym

_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\s*\{")
# SynLogic task prompts use heterogeneous final-answer conventions: besides
# "Final Answer: ...", many tasks (web_of_lies, cryptarithm, word_sorting_mistake)
# instruct 'The answer is $YOUR_ANSWER' — often wrapped in markdown bold.
_FINAL_ANSWER_RE = re.compile(r"(?:final answer\s*[::]|the answer is)\s*[::]?\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_FENCED_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n(.*?)```", re.DOTALL)
# Enigmata string/grid tasks (campsite, star_battle, ...) ask for the board
# wrapped in <begin_board>...</begin_board> inside the final answer.
_BOARD_RE = re.compile(r"<begin_board>(.*?)<end_board>", re.DOTALL | re.IGNORECASE)


# ---------------------------------------------------------------------------
# format gate (Qwen3-4B chat template, thinking enabled)
# ---------------------------------------------------------------------------


def format_ok(solution_str: str) -> bool:
    """Structural gate enforcing the Qwen3-4B thinking-template response shape.

    With the default Qwen3 chat template (thinking on), the rollout prompt ends
    with ``<|im_start|>assistant\\n`` and a compliant generation is exactly one
    think block followed by the final response::

        <think>
        {reasoning}
        </think>

        {final response}

    The naive reward manager decodes with ``skip_special_tokens=True``, so
    ``<|im_start|>``/``<|im_end|>`` never reach this function — the template
    contract enforceable on ``solution_str`` is the think-block shape: starts
    with ``<think>``, exactly one closing ``</think>``, non-empty reasoning
    (an empty block is the ``enable_thinking=False`` shortcut), a non-empty
    final response, and no stray think tags afterwards.
    """
    if not solution_str.startswith("<think>"):
        return False
    think_body, sep, response_body = solution_str[len("<think>") :].partition("</think>")
    if not sep:  # unterminated think block (e.g. truncated rollout)
        return False
    if not think_body.strip():  # "<think>\n\n</think>" == thinking disabled
        return False
    if not response_body.strip():
        return False
    return "<think>" not in response_body and "</think>" not in response_body


# ---------------------------------------------------------------------------
# math / stem
# ---------------------------------------------------------------------------


def _math_score(solution_str: str, ground_truth: str) -> float:
    """math_verify when available, else/also-on-failure verl's rule matcher.

    math_verify runs each comparison in a *shared* process pool
    (``verl.utils.reward_score.math_verify``).  A single sample that crashes or
    wedges one of its workers permanently breaks that pool for the rest of the
    run, and the upstream helper swallows the resulting ``BrokenProcessPool``
    into a plain ``0.0`` -- one pathological response would then zero the
    ``math_*`` **and** ``stem_*`` reward of every later sample.  Falling back to
    math_dapo (in-process, no shared state) keeps those samples scoreable
    instead of silently rewarding nothing.

    Last resort: when every strict verifier scores 0, five credit-adding
    checks for known ground-truth format noise get their say
    (``_math_last_resort``): a ground truth stored as a k-digit rounded
    decimal is matched within its rounding bucket, and percent/labelled-list
    decimal ground truths against their round-or-truncate uncertainty
    interval (``_rounded_decimal_match``/``_labeled_decimal_match`` -- a DAPO
    ground truth of 0.333 no longer zeroes the exact answer 1/3, ``6.535%``
    no longer zeroes the coarser-rounded ``6.54%``, and a truncated textbook
    value like ``...Stock: 0.3114`` no longer zeroes the correctly rounded
    0.3115); a whitespace/subscript/brace-normalised literal comparison
    recovers plain-text variable names the symbolic parser mangles
    (``_normalized_text_match`` -- a Big-Math ground truth of ``y2 < y1 <
    y3`` no longer zeroes ``y_2 < y_1 < y_3``); a canonical-polarity
    comparison recovers yes/no ground truths stored in the problem's original
    language (``_yes_no_match`` -- a Russian ``Выполнимо`` no longer zeroes
    an English ``Yes``, while a genuinely wrong ``No`` still scores 0); a
    bounded symbol-
    identification search recovers answers that substitute a prompt condition
    the ground truth left general (``_conditional_match`` -- a ground truth of
    ``max(C,T)``/``C+T`` no longer zeroes ``T``/``2T`` when the prompt said
    C equals T); and numeric equivalence at fixed sample points recovers
    answers that correctly evaluate a formula the ground truth left
    unevaluated (``_evaluated_gt_match`` -- ``a(1 - 10\%)^2`` no longer
    zeroes ``0.81a``).
    """
    ground_truth = str(ground_truth)
    try:
        from verl.utils.reward_score import math_verify

        score = float(math_verify.compute_score(solution_str, ground_truth))
        if score:
            return score
        # Strict verifier said no -- the ground-truth-noise last resorts get
        # their say before the miss is final.
        return 1.0 if _math_last_resort(solution_str, ground_truth) else 0.0
    except ImportError:
        pass
    except Exception as e:  # never let one bad sample kill the reward pass
        logger.warning("[reasoning_rl] math_verify failed (%s); falling back to math_dapo", e)
    from verl.utils.reward_score import math_dapo

    # math_dapo returns {"score": ±1.0, "acc": bool, ...}; normalise to 0/1 so
    # all four domains share the same pass-rate semantics.  Try the boxed-answer
    # rule first -- every prompt in this mix asks for ``\boxed{}`` -- then the
    # Minerva "Answer: X" rule, so the fallback stays useful for either shape.
    for strict_box in (True, False):
        res = math_dapo.compute_score(solution_str, ground_truth, strict_box_verify=strict_box)
        acc = float(res["acc"]) if isinstance(res, dict) else float(res)
        if acc:
            return acc
    return 1.0 if _math_last_resort(solution_str, ground_truth) else 0.0


# A ground truth stored as a plain decimal with a fraction part, e.g. "0.333";
# the capture group is the fraction digits (the gt's written precision).
_DECIMAL_GT_RE = re.compile(r"^[+-]?\d+\.(\d+)$")


def _math_pred_values(solution_str: str) -> list[float]:
    """Numeric values math_verify's parser extracts from the solution's boxed answer.

    Best-effort: returns [] when math-verify is not installed (math_dapo-only
    smoke environments), when the solution carries no boxed answer, or when
    nothing parses to a finite real number.
    """
    boxed = _extract_boxed(solution_str)
    if boxed is None or len(boxed) > 500:  # bound the parse on crafted input
        return []
    try:
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse
    except ImportError:
        return []
    try:
        # parsing_timeout=None: the default signal-based timeout only works in
        # the main thread, but rewards run in a worker-thread pool; the boxed
        # input is length-bounded above, so no timeout is needed here.
        try:
            extracted = parse(
                f"${boxed}$", (ExprExtractionConfig(), LatexExtractionConfig()), parsing_timeout=None
            )
        except TypeError:  # older math_verify without parsing_timeout
            extracted = parse(f"${boxed}$", (ExprExtractionConfig(), LatexExtractionConfig()))
    except Exception:
        return []
    values = []
    for expr in extracted:
        if isinstance(expr, tuple):  # some math_verify versions return (expr, str)
            expr = expr[0]
        try:
            value = float(expr)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def _rounded_decimal_match(solution_str: str, ground_truth: str) -> bool:
    """Last-resort match for ground truths stored as a k-digit rounded decimal.

    DAPO-Math-17k stores some answers rounded to a few decimal places ("0.333"
    for 1/3); math_verify's ~1e-6 numeric tolerance then rejects the exact
    answer -- a false negative no strict verifier can bridge.  When the gt
    literal is a plain decimal with k>=1 fraction digits, accept any extracted
    prediction within half a unit of the gt's last digit (the rounding bucket
    the gt author themselves specified).  An integer ground truth is treated as
    exact and never enters this path.  Additive-only: it runs after the strict
    verifiers failed, so it can turn a 0 into a 1 but never a 1 into a 0.

    Ground truths that are NOT one bare decimal -- a percent figure
    (``6.535%``) or a labelled list (``Adobe Systems Stock: 0.3114, ...``) --
    fall through to ``_labeled_decimal_match``.
    """
    candidate = _extract_boxed(ground_truth) or ground_truth.strip()
    candidate = candidate.strip().strip("$").strip()
    m = _DECIMAL_GT_RE.match(candidate)
    if m:  # bare single decimal: strict rounding-bucket semantics
        gt_value = float(candidate)  # the regex guarantees a plain decimal
        tol = 0.5 * 10 ** -len(m.group(1))
        return any(abs(pred - gt_value) <= tol for pred in _math_pred_values(solution_str))
    return _labeled_decimal_match(solution_str, candidate)


# Numeral with an optional (LaTeX-escaped) percent sign; group 1 is the number,
# group 2 the percent marker ("6.54", "6.54%", "6.54\%").
_NUMERAL_RE = re.compile(r"([+-]?(?:\d+\.\d+|\.\d+|\d+))\s*(\\?%)?")
# "30,500" is one number, but "0.3115, 0.3443" (and "0.3115,0.3443") are two:
# only a comma followed by exactly three digits is a thousands separator.
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")
# Characters allowed around the numerals: prose labels in the ground truth,
# only punctuation/LaTeX escapes in a boxed prediction.  Any maths operator
# (``= < > ( ) [ ] { } ^ * / _ | ~`` or a stray ``+``/``-``) means the string
# is not decimal-answer-shaped and stays with the symbolic verifiers.
_GT_FILLER_RE = re.compile(r"^[A-Za-z\s.,:;%$]*$")
_PRED_FILLER_RE = re.compile(r"^[\s.,:;%$\\]*$")
_DEC_MAX_VALUES = 8


def _numeral_value(token: str, pct: str | None) -> tuple[Decimal, int]:
    """Exact value and written precision (fraction digits) of a numeral token."""
    v = Decimal(token)
    k = len(token.partition(".")[2])
    if pct:  # "6.535%" == 0.06535 written at 5 fraction digits
        v /= 100
        k += 2
    return v, k


def _gt_decimal_bucket(token: str, pct: str | None) -> tuple[Decimal, Decimal, bool]:
    """Uncertainty interval ``(lo, hi, is_point)`` of a ground-truth numeral.

    Textbook ground truths are frequently *truncated* rather than rounded
    (Big-Math stores ``0.3114`` for 9500/30500 = 0.311475..., while the model's
    correctly rounded 0.3115 then scores 0), so the bucket extends a full unit
    of the last written digit upwards but only half a unit downwards:
    ``[v - 0.5*10^-k, v + 10^-k)``.  An integer numeral is an exact point.
    """
    v, k = _numeral_value(token, pct)
    if k == 0:
        return (v, v, True)
    return (v - Decimal(5).scaleb(-(k + 1)), v + Decimal(1).scaleb(-k), False)


def _pred_decimal_bucket(token: str, pct: str | None) -> tuple[Decimal, Decimal, bool]:
    """Rounding bucket ``(lo, hi, False)`` of a predicted numeral: the values
    whose correct rounding at the written precision is the numeral,
    ``[v - 0.5*10^-k, v + 0.5*10^-k)`` (an integer prediction uses k = 0)."""
    v, k = _numeral_value(token, pct)
    half = Decimal(5).scaleb(-(k + 1))
    return (v - half, v + half, False)


def _decimal_overlap(gt, pred) -> bool:
    """Non-empty intersection of two ``(lo, hi, is_point)`` buckets: some true
    value could be written as the gt numeral AND as the predicted numeral."""
    g_lo, g_hi, g_pt = gt
    p_lo, p_hi, p_pt = pred
    if g_pt and p_pt:
        return g_lo == p_lo
    if g_pt:
        return p_lo <= g_lo < p_hi
    if p_pt:
        return g_lo <= p_lo < g_hi
    return g_lo < p_hi and p_lo < g_hi


def _gt_decimal_buckets(candidate: str) -> list | None:
    """Decimal buckets for a (possibly labelled) decimal-list ground truth.

    Returns None when the candidate is not decimal-answer-shaped: numerals
    embedded in prose labels are fine, but any mathematical operator sends the
    ground truth back to the symbolic last resorts.
    """
    if len(candidate) > 500:
        return None
    cleaned = _THOUSANDS_RE.sub("", candidate)
    matches = list(_NUMERAL_RE.finditer(cleaned))
    if not matches or len(matches) > _DEC_MAX_VALUES:
        return None
    if not _GT_FILLER_RE.match(_NUMERAL_RE.sub("", cleaned)):
        return None
    return [_gt_decimal_bucket(m.group(1), m.group(2)) for m in matches]


def _point_decimal_buckets(boxed: str) -> list:
    """Exact point values math_verify extracts from a non-plain boxed answer."""
    try:
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse
    except ImportError:
        return []
    try:
        # parsing_timeout=None: rewards run in a worker-thread pool where the
        # default signal-based timeout does not work; input is length-bounded.
        try:
            extracted = parse(f"${boxed}$", (ExprExtractionConfig(), LatexExtractionConfig()), parsing_timeout=None)
        except TypeError:  # older math_verify without parsing_timeout
            extracted = parse(f"${boxed}$", (ExprExtractionConfig(), LatexExtractionConfig()))
    except Exception:
        return []
    points = []
    for expr in extracted:
        if isinstance(expr, tuple):  # some math_verify versions return (expr, str)
            expr = expr[0]
        try:
            value = float(expr)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(value):
            points.append((Decimal(str(value)),) * 2 + (True,))
    return points


def _pred_decimal_buckets(solution_str: str) -> list:
    """Decimal/point buckets for every \\boxed{...} in the final response.

    A boxed answer containing only numerals contributes its written values
    with their precision; anything else (``\\dfrac{1}{3}``) goes through
    math_verify as exact point values.  Capped at ``_DEC_MAX_VALUES``.
    """
    response = _extract_final_response(solution_str)
    buckets = []
    for content in _extract_all_boxed(response):
        if len(buckets) >= _DEC_MAX_VALUES:
            break
        content = content.strip()
        if not content or len(content) > 500:
            continue
        cleaned = _THOUSANDS_RE.sub("", content)
        matches = list(_NUMERAL_RE.finditer(cleaned))
        if matches and _PRED_FILLER_RE.match(_NUMERAL_RE.sub("", cleaned)):
            buckets.extend(_pred_decimal_bucket(m.group(1), m.group(2)) for m in matches)
        else:
            buckets.extend(_point_decimal_buckets(content))
    return buckets[: _DEC_MAX_VALUES]


def _labeled_decimal_match(solution_str: str, gt_candidate: str) -> bool:
    """Last-resort match for decimal ground truths that are not a bare decimal.

    Big-Math keeps textbook-style answers the bare-decimal regex cannot hold:
    percent figures (``6.535%``) and labelled value lists (``Adobe Systems
    Stock: 0.3114, Dow Chemical Stock: 0.3442, ...``).  On top of rounding,
    those strings show two extra noise classes: the written value may be
    *truncated* instead of rounded (9500/30500 = 0.311475... stored as 0.3114,
    so the correctly rounded 0.3115 scores 0), and the model may write the same
    value at a coarser precision (``6.54%`` for ``6.535%``).  Every gt numeral
    therefore gets its round-or-truncate uncertainty interval and every boxed
    prediction numeral its correct-rounding bucket; a single-value ground truth
    matches when ANY boxed value overlaps (the bare-decimal path's ``any``
    semantics, so a restated final answer cannot fail a count check), a
    multi-value ground truth requires an equal count and an in-order pairwise
    overlap.  The bare-decimal path above deliberately keeps its strict
    half-unit bucket, so this trunc-aware interval only ever applies to ground
    truths that previously scored 0 unconditionally.  Additive-only: it runs
    after the strict verifiers failed, so it can turn a 0 into a 1 but never a
    1 into a 0.
    """
    gt_buckets = _gt_decimal_buckets(gt_candidate)
    if not gt_buckets:
        return False
    pred_buckets = _pred_decimal_buckets(solution_str)
    if not pred_buckets:
        return False
    if len(gt_buckets) == 1:
        return any(_decimal_overlap(gt_buckets[0], p) for p in pred_buckets)
    if len(gt_buckets) != len(pred_buckets):
        return False
    return all(_decimal_overlap(g, p) for g, p in zip(gt_buckets, pred_buckets))


# Relation commands -> ASCII, longest first; short ones use a letter lookahead
# so ``\le`` never mangles ``\left``.
_REL_SUBS = [
    (re.compile(r"\\leqslant|\\leq|\\le(?![a-zA-Z])"), "<="),
    (re.compile(r"\\geqslant|\\geq|\\ge(?![a-zA-Z])"), ">="),
    (re.compile(r"\\neq|\\ne(?![a-zA-Z])"), "!="),
    (re.compile(r"\\lt(?![a-zA-Z])"), "<"),
    (re.compile(r"\\gt(?![a-zA-Z])"), ">"),
]
# Non-semantic LaTeX decoration dropped entirely: subscript braces/marker
# (``y_{2}``/``y_2`` -> ``y2``), grouping braces, inline spacing, dollars.
# ``^`` is KEPT: ``x^2`` (x squared) must not collapse into ``x2``.
_TEXT_STRIP_RE = re.compile(r"\\left|\\right|\\!|\\,|\\;|\\:|\\ |[{}_$~]")
_WS_RE = re.compile(r"\s+")


def _normalize_math_text(s: str) -> str:
    for pattern, repl in _REL_SUBS:
        s = pattern.sub(repl, s)
    s = _TEXT_STRIP_RE.sub("", s)
    return _WS_RE.sub("", s)


def _normalized_text_match(solution_str: str, ground_truth: str) -> bool:
    """Last-resort literal comparison after normalising LaTeX surface syntax.

    Big-Math ground truths store variable names in plain text (``y2 < y1 <
    y3``); math_verify parses ``y2`` as ``y*2`` while the model's natural
    ``y_2`` parses as a subscripted symbol, so a correct answer can never
    match symbolically.  Comparing whitespace/subscript/brace-normalised text
    recovers exactly this formatting-variance class.  ``^`` is preserved so
    powers cannot collapse into variable names.  Additive-only: it runs after
    the strict verifiers failed, so it can turn a 0 into a 1 but never a 1
    into a 0.
    """
    pred = _extract_boxed(solution_str)
    if pred is None:
        return False
    gt = _extract_boxed(ground_truth) or ground_truth
    pred_norm, gt_norm = _normalize_math_text(pred), _normalize_math_text(gt)
    return bool(pred_norm) and pred_norm == gt_norm


# Yes/no answers across the languages Big-Math mixes in.  The ground truth is
# stored in the problem's original language (Russian olympiad answers like
# ``Выполнимо.``), while the model answers in the prompt's language (English),
# so a correct ``Yes`` can never literally match.  Both sides are reduced to a
# canonical polarity and compared; the tables are deliberately small and
# unambiguous so nothing else can enter this path.
_YES_TOKENS = {
    "yes", "yeah", "true", "possible", "feasible", "doable",
    "да", "верно", "выполнимо", "возможно",
    "是", "对", "正确", "能", "可以",
}
_NO_TOKENS = {
    "no", "false", "impossible", "infeasible", "not possible",
    "нет", "неверно", "невыполнимо", "невозможно",
    "否", "错", "错误", "不能", "不可以", "不可能",
}
# LaTeX commands/decoration stripped before the polarity lookup.
_YESNO_STRIP_RE = re.compile(r"\\[a-zA-Z]+|[{}$*_~]|[.,!;:`'\"()<>]")


def _canon_yes_no(text: str):
    """Canonical ``True``/``False`` polarity of a short yes/no-style answer,
    or None when the text is not an unambiguous polarity token."""
    cleaned = _YESNO_STRIP_RE.sub(" ", text.lower())
    cleaned = " ".join(cleaned.split())
    if cleaned in _YES_TOKENS:
        return True
    if cleaned in _NO_TOKENS:
        return False
    return None


def _yes_no_match(solution_str: str, ground_truth: str) -> bool:
    """Last-resort match for yes/no ground truths stored in another language.

    Only fires when BOTH sides normalise to a polarity token, so a wrong
    answer (``No`` vs ``Выполнимо``) still scores 0 and non-yes/no strings
    never enter this path.  Additive-only: it runs after the strict verifiers
    failed, so it can turn a 0 into a 1 but never a 1 into a 0.
    """
    pred = _extract_boxed(solution_str)
    if pred is None:
        return False
    gt = _extract_boxed(ground_truth) or ground_truth
    pred_pol, gt_pol = _canon_yes_no(pred), _canon_yes_no(gt)
    return pred_pol is not None and pred_pol == gt_pol


# Bounds for _conditional_match: it runs only on the already-failed path, and
# bails (no credit) whenever the problem is bigger than these caps.
_COND_MAX_GT_EXPRS = 4
_COND_MAX_PRED_EXPRS = 8
_COND_MAX_GT_SYMBOLS = 3
_COND_MAX_SYMBOLS = 4


def _sympy_exprs(parsed) -> list:
    exprs = []
    for e in parsed:
        if isinstance(e, tuple):  # some math_verify versions return (expr, str)
            e = e[0]
        if hasattr(e, "free_symbols"):
            exprs.append(e)
    return exprs


def _strip_unevaluated(exprs) -> list:
    """Unwrap UnevaluatedExpr literals (math_verify wraps e.g. the 1/100 from
    ``10\%`` in one); they defeat subs/evalf and structural equality.  Once
    unwrapped, held arithmetic such as ``a*(1 - 10\%)^2`` auto-evaluates."""

    import sympy as sp  # lazy; callers already guard the ImportError

    def _strip(e):
        try:
            return e.replace(lambda x: isinstance(x, sp.UnevaluatedExpr), lambda x: x.args[0])
        except Exception:
            return e

    return [_strip(e) for e in exprs]


def _dedupe_keep_order(exprs) -> list:
    """Dedupe keeping first-occurrence order.  ``dict.fromkeys`` needs hashable
    items, but a parsed matrix answer yields sympy's unhashable
    ``MutableDenseMatrix`` -- probe those with equality instead of hashing."""
    seen, unhashable, out = set(), [], []
    for e in exprs:
        try:
            if e in seen:
                continue
            seen.add(e)
        except TypeError:
            if any(e == u for u in unhashable):
                continue
            unhashable.append(e)
        out.append(e)
    return out


def _conditional_match(solution_str: str, ground_truth: str) -> bool:
    """Last resort for ground truths left as unevaluated general formulas.

    Some prompts attach a condition ("if C exactly equals T ..."); the model
    then answers with the condition substituted (``T``, ``2T``) while the
    dataset keeps the general formula (``max(C,T)``, ``C+T``), and a strict
    comparison can never equate the two.  Credit the response when EVERY
    ground-truth expression equals some predicted expression under one
    consistent symbol identification (e.g. ``C->T``).  The reward never sees
    the prompt, so only identifications the prediction itself exhibits are
    considered, the identity map is excluded (that is the strict verifiers'
    territory), and all sizes are bounded by the ``_COND_MAX_*`` caps -- a
    narrow, cheap, additive-only fallback on the failed path.
    """
    try:
        import sympy as sp
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse
    except ImportError:
        return False
    response = _extract_final_response(solution_str)
    if not response:
        return False
    try:
        # parsing_timeout=None: signal-based timeouts only work in the main
        # thread, but rewards run in a worker-thread pool.
        gt_exprs = _sympy_exprs(parse(ground_truth, (LatexExtractionConfig(),), parsing_timeout=None))
        if not gt_exprs:  # datasets often keep bare LaTeX without $...$
            gt_exprs = _sympy_exprs(parse(f"${ground_truth}$", (LatexExtractionConfig(),), parsing_timeout=None))
        pred_exprs = _sympy_exprs(
            parse(response, (ExprExtractionConfig(), LatexExtractionConfig()), parsing_timeout=None)
        )
    except Exception:
        return False

    def _flatten(exprs):
        # A parsed answer set {a, b} offers each element as a candidate.
        out = []
        for e in exprs:
            out.extend(e.args if isinstance(e, sp.FiniteSet) else (e,))
        return out

    gt_exprs = _strip_unevaluated(_flatten(gt_exprs))
    pred_exprs = _dedupe_keep_order(_strip_unevaluated(_flatten(pred_exprs)))
    if not (0 < len(gt_exprs) <= _COND_MAX_GT_EXPRS) or not (0 < len(pred_exprs) <= _COND_MAX_PRED_EXPRS):
        return False
    gt_symbols = sorted(set().union(*(e.free_symbols for e in gt_exprs)), key=str)
    universe = sorted(set(gt_symbols) | set().union(*(e.free_symbols for e in pred_exprs)), key=str)
    if not (0 < len(gt_symbols) <= _COND_MAX_GT_SYMBOLS) or len(universe) > _COND_MAX_SYMBOLS:
        return False
    for mapping in itertools.product(universe, repeat=len(gt_symbols)):
        sigma = dict(zip(gt_symbols, mapping))
        if all(sigma[s] == s for s in gt_symbols):
            continue  # identity: the strict verifiers already had their say
        try:
            matched = all(any(g.subs(sigma) == p for p in pred_exprs) for g in gt_exprs)
        except Exception:
            # subs may rebuild held expressions, e.g. an inequality chain
            # parsed with evaluate=False goes non-real (``m < 2*I``) and
            # raises TypeError -- that mapping is simply not a witness.
            continue
        if matched:
            return True
    return False


# Fixed non-integer sample points for _evaluated_gt_match: deterministic
# (rewards must be reproducible), off integers/zeros to dodge trivial roots,
# and each round assigns distinct values to the sorted free symbols.
_EVAL_ROUNDS = ((1.7, 2.9, 3.3), (4.1, 5.7, 6.3), (7.9, 8.3, 9.1))
_EVAL_RTOL = 1e-6
_EVAL_ATOL = 1e-9


def _evaluated_gt_match(solution_str: str, ground_truth: str) -> bool:
    """Last resort for ground truths stored as unevaluated formulas.

    Big-Math keeps answers like ``a(1 - 10\\%)^2``: parsing preserves the
    unevaluated tree, the rational/Float epsilon (81/100 vs 0.81) defeats
    symbolic equality, and the free symbol ``a`` defeats math_verify's
    numeric comparison -- so the correctly evaluated answer ``0.81a`` scores
    0.  Numeric equivalence at fixed sample points (Schwartz-Zippel style)
    recovers exactly this class: every ground-truth expression must equal
    some predicted expression at >=2 of 3 deterministic non-integer points.
    Bounded by the ``_COND_MAX_*`` caps, additive-only on the failed path.
    """
    try:
        import sympy as sp
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse
    except ImportError:
        return False
    response = _extract_final_response(solution_str)
    if not response:
        return False
    try:
        # parsing_timeout=None: signal-based timeouts only work in the main
        # thread, but rewards run in a worker-thread pool.
        gt_exprs = _sympy_exprs(parse(ground_truth, (LatexExtractionConfig(),), parsing_timeout=None))
        if not gt_exprs:  # datasets often keep bare LaTeX without $...$
            gt_exprs = _sympy_exprs(parse(f"${ground_truth}$", (LatexExtractionConfig(),), parsing_timeout=None))
        pred_exprs = _sympy_exprs(
            parse(response, (ExprExtractionConfig(), LatexExtractionConfig()), parsing_timeout=None)
        )
    except Exception:
        return False
    # No Boolean filtering: relations fail float()/evalf below and self-exclude.
    gt_exprs = _strip_unevaluated(x for e in gt_exprs for x in (e.args if isinstance(e, sp.FiniteSet) else (e,)))
    pred_exprs = _dedupe_keep_order(
        _strip_unevaluated(x for e in pred_exprs for x in (e.args if isinstance(e, sp.FiniteSet) else (e,)))
    )
    if not (0 < len(gt_exprs) <= _COND_MAX_GT_EXPRS) or not (0 < len(pred_exprs) <= _COND_MAX_PRED_EXPRS):
        return False
    if len(set().union(*(e.free_symbols for e in gt_exprs + pred_exprs))) > _COND_MAX_SYMBOLS:
        return False

    def _equiv(g, p) -> bool:
        if g == p:
            return True
        syms = sorted(g.free_symbols | p.free_symbols, key=str)
        if not syms:
            try:
                return math.isclose(float(g.evalf()), float(p.evalf()), rel_tol=_EVAL_RTOL, abs_tol=_EVAL_ATOL)
            except Exception:
                return False
        hits = 0
        for point in _EVAL_ROUNDS:
            subs = dict(zip(syms, point[: len(syms)]))
            try:
                gv = float(g.subs(subs).evalf())
                pv = float(p.subs(subs).evalf())
            except Exception:
                continue  # singular point / non-numeric (e.g. relation); next round
            if not (math.isfinite(gv) and math.isfinite(pv)):
                continue
            if not math.isclose(gv, pv, rel_tol=_EVAL_RTOL, abs_tol=_EVAL_ATOL):
                return False  # provably different at a real point
            hits += 1
        return hits >= 2

    return all(any(_equiv(g, p) for p in pred_exprs) for g in gt_exprs)


def _math_last_resort(solution_str: str, ground_truth: str) -> bool:
    """Credit-adding fallbacks for known ground-truth format noise."""
    return (
        _rounded_decimal_match(solution_str, ground_truth)
        or _normalized_text_match(solution_str, ground_truth)
        or _yes_no_match(solution_str, ground_truth)
        or _conditional_match(solution_str, ground_truth)
        or _evaluated_gt_match(solution_str, ground_truth)
    )


# ---------------------------------------------------------------------------
# code
# ---------------------------------------------------------------------------


def _extract_python_solution(solution_str: str) -> str | None:
    """Mirror sandbox_fusion's code-block extraction (last ```python block)."""
    if "```python" in solution_str:
        return solution_str.split("```python")[-1].split("```")[0]
    if "```" in solution_str:
        parts = solution_str.split("```")
        if len(parts) >= 2:
            solution = parts[1]
            if "\n" in solution:
                first_line, rest = solution.split("\n", 1)
                if first_line.strip().isalpha():
                    solution = rest
            return solution
    return None


def _name_key(name: str) -> str:
    return name.replace("_", "").casefold()


def _align_callable_name(solution_str: str, ground_truth: str):
    """Rewrite ``fn_name`` when the model defined the callable under a variant name.

    The code prompt historically omitted the required function name, and models
    legitimately switch between snake_case and camelCase. Both the sandbox and
    prime_code wrappers resolve the callable by *exact* name, so a correct
    solution defining ``makeAcronym`` scores 0 when the tests expect
    ``make_acronym``.

    The rewrite is deliberately conservative: it only fires when the exact name
    is absent and either a case/underscore-insensitive match or a single
    non-``main`` top-level function exists, so an unrelated helper can never be
    silently substituted. ``class Solution`` layouts (handled natively by the
    wrapper) are left untouched. Returns ``ground_truth`` unchanged when the
    payload is not a call-based test dict.
    """
    try:
        payload = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    except (json.JSONDecodeError, TypeError):
        return ground_truth
    if not isinstance(payload, dict) or not payload.get("fn_name"):
        return ground_truth
    code = _extract_python_solution(solution_str)
    if not code:
        return ground_truth
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return ground_truth
    if any(isinstance(node, ast.ClassDef) and node.name == "Solution" for node in tree.body):
        return ground_truth
    defined = [node.name for node in tree.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)]
    fn_name = payload["fn_name"]
    if not defined or fn_name in defined:
        return ground_truth
    matches = [name for name in defined if _name_key(name) == _name_key(fn_name)]
    chosen = matches[0] if len(matches) == 1 else None
    if chosen is None:
        candidates = [name for name in defined if name != "main"]
        chosen = candidates[0] if len(candidates) == 1 else None
    if chosen is None or chosen == fn_name:
        return ground_truth
    logger.info("[reasoning_rl] fn_name %r absent from solution; scoring against %r", fn_name, chosen)
    return json.dumps({**payload, "fn_name": chosen})


def _code_score(
    solution_str: str, ground_truth: str, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb
) -> float:
    try:
        ground_truth = _align_callable_name(solution_str, ground_truth)
        if sandbox_fusion_url:
            from verl.utils.reward_score import sandbox_fusion

            res = sandbox_fusion.compute_score(
                sandbox_fusion_url,
                concurrent_semaphore,
                memory_limit_mb,
                solution_str,
                ground_truth,
                continuous=True,
            )
        else:
            from verl.utils.reward_score import prime_code

            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
        # Both return (score, metadata_list).
        return float(res[0])
    except Exception as e:
        logger.warning("[reasoning_rl] code verify error: %s", e)
        return 0.0


# ---------------------------------------------------------------------------
# logic
# ---------------------------------------------------------------------------


def _extract_boxed(text: str) -> str | None:
    """Return the content of the last \\boxed{...} with balanced braces."""
    starts = [m.end() for m in _BOXED_RE.finditer(text)]
    if not starts:
        return None
    i = starts[-1]
    depth = 1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i:j]
    return None


def _extract_all_boxed(text: str) -> list[str]:
    """Contents of every \\boxed{...} with balanced braces, in order."""
    contents = []
    for m in _BOXED_RE.finditer(text):
        i = m.end()
        depth = 1
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    contents.append(text[i:j])
                    break
    return contents


def _strip_fences(s: str) -> str:
    """Unwrap a markdown fenced code block (```lang\\n...```) around an answer.

    Several SynLogic tasks (skyscraper_puzzle, zebra_puzzle) require the final
    answer inside a ```python / ```json block; models also wrap <answer>-tag
    content in fences on their own. The fence is container, not content.
    """
    s = s.strip()
    m = _FENCED_BLOCK_RE.fullmatch(s)
    return m.group(1).strip() if m else s


def extract_logic_answer(solution_str: str) -> str | None:
    """Extract the final answer: <answer> tags, then \\boxed{}, then a
    "Final Answer: ..." / "The answer is ..." line, then the last fenced
    code block (DESIGN.md section 1 + SynLogic per-task prompt conventions).

    Last resort: the visible response itself.  Several task prompts (all of
    Reasoning Gym, plus SynLogic tasks phrased "respond with only your answer")
    ask for a bare answer, so a fully compliant generation can carry no marker
    at all: ``"<think>...</think>\\n\\n6"``.  Returning the body instead of
    ``None`` there is what keeps such answers scoreable; a body with extra prose
    still fails the comparison it is fed into, so this adds recall, not credit.
    """
    matches = _ANSWER_TAG_RE.findall(solution_str)
    if matches:
        return _strip_fences(matches[-1])
    boxed = _extract_boxed(solution_str)
    if boxed is not None:
        return boxed.strip()
    tail = solution_str[-2000:]  # the final line is what matters; bound the scan
    matches = _FINAL_ANSWER_RE.findall(tail)
    if matches:
        return _strip_fences(matches[-1].strip().rstrip("."))
    blocks = _FENCED_BLOCK_RE.findall(tail)
    if blocks:
        return blocks[-1].strip()
    boards = _BOARD_RE.findall(solution_str)
    if boards:
        return boards[-1].strip()
    body = solution_str.partition("</think>")[2].strip()
    return body or None


def _normalise_text(s: str) -> str:
    s = s.strip()
    # Strip markdown emphasis/backticks — prompts such as web_of_lies ask for
    # 'The answer is **yes, no**' and the stars are presentation, not content.
    s = s.replace("**", "").replace("__", "").replace("`", "")
    s = s.strip().strip('"').strip("'")
    s = re.sub(r"\s+", " ", s)
    return s.casefold()


def _loose_token_form(s: str) -> str:
    """Canonical form insensitive to separator spacing and inner quotes.

    Fixes two systematic false negatives against SynLogic ground truths:
      - gt "A,C,D,E" vs natural model output "A, C, D, E" (boolean_expressions,
        calcudoko row separators);
      - gt "[[WORD]]" vs prompt-example-compliant "[['WORD']]" (cipher): the
        quoted form parses, the bare-word ground truth never does, so text
        comparison is the only path and must ignore the quotes.
    Whitespace *between* tokens is preserved ("1 2 3" != "123").
    """
    s = _normalise_text(s)
    s = re.sub(r"\s*([,\[\]\(\)\{\}:;])\s*", r"\1", s)
    return s.replace('"', "").replace("'", "")


# An answer nested hundreds of levels deep (a policy emitting a long run of "[")
# parses fine but then blows the C stack inside the recursive comparison and
# repr() paths below. Refusing it at parse time keeps every downstream consumer
# iterative-safe; the flat text comparison still gets its say, so identical
# deep answers still match.
_MAX_STRUCTURE_DEPTH = 64


def _too_deep(obj, limit: int = _MAX_STRUCTURE_DEPTH) -> bool:
    """True when ``obj`` nests deeper than ``limit`` (iterative, no recursion)."""
    stack = [(obj, 1)]
    while stack:
        value, depth = stack.pop()
        if depth > limit:
            return True
        if isinstance(value, list | tuple):
            stack.extend((item, depth + 1) for item in value)
        elif isinstance(value, dict):
            stack.extend((item, depth + 1) for item in value.values())
    return False


def _parse_structured(s: str):
    """Best-effort parse of an answer into a Python object (grids, tuples,
    lists, numbers). Returns None when it only parses to a plain string —
    that case is covered by text comparison instead."""
    s = s.strip()
    for parser in (ast.literal_eval, json.loads):
        try:
            obj = parser(s)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
        if not isinstance(obj, str):
            return None if _too_deep(obj) else obj
    return None


def _structured_equal(a, b, tol: float = 1e-6) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, int | float) and isinstance(b, int | float):
        try:
            return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
        except (OverflowError, ValueError):
            # ``float()`` refuses an integer beyond ~1e308 (OverflowError). Such a
            # literal is not "un-equal", it is simply unordered by ``isclose``:
            # fall through to the text comparison so two identical huge integers
            # still match instead of killing the reward worker.
            pass
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        return len(a) == len(b) and all(_structured_equal(x, y, tol) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_structured_equal(a[k], b[k], tol) for k in a)
    return _loose_token_form(str(a)) == _loose_token_form(str(b))


# Tasks whose answer is an unordered collection of coordinates, dominos or
# (person, item) pairs -- both sides are canonicalised (recursively sorted)
# before the structural compare so collection ordering never decides the reward.
# Enigmata's hitori/kakurasu/light_up/minesweeper are coordinate sets too: the
# official verifiers compare them as Python sets, while the generators store
# them in an arbitrary order.
_UNORDERED_COLLECTION_TASKS = frozenset(
    {
        "minesweeper",
        "norinori",
        "star_placement_puzzle",
        "goods_exchange",
        "hitori",
        "kakurasu",
        "light_up",
    }
)

# SynLogic tasks whose answer is an arithmetic expression: the prompt asks for it
# wrapped in [[...]] and the generator's own spacing ("9 +6 -7 +(6 %5)") differs
# from anything a model writes ("9+6-7+(6%5)"), so whitespace is not content.
_WHITESPACE_FREE_TASKS = frozenset({"math_path"})


def _canon_unordered(obj):
    """Normalise tuples to lists and sort (bottom-up) any list whose elements
    are all lists. Applied only to _UNORDERED_COLLECTION_TASKS answers — never to
    grids, where row/column order is the answer."""
    if isinstance(obj, dict):
        return {k: _canon_unordered(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        items = [_canon_unordered(x) for x in obj]
        if items and all(isinstance(x, list) for x in items):
            items.sort(key=json.dumps)
        return items
    return obj


def _parse_grid_matrix(text: str) -> list[list[int]] | None:
    """Parse a rectangular integer grid, or return None if ``text`` is not one.

    Three answer conventions appear in the logic domain: a nested list
    (``"[[1, 2], [3, 4]]"``, the ARC-style model output), an Enigmata ground
    truth stored as space/newline separated text (``"1 2\\n3 4"``), and the
    ``"[[1 2,3 4]]"`` form the SynLogic calcudoko/futoshiki prompts *mandate*
    (cells separated by spaces, rows by commas, one outer bracket pair).  All
    three become the same matrix here so the comparison is format-agnostic;
    non-numeric grids (star_battle/star placement) return None and fall through
    to the text path.
    """
    s = text.strip()
    if not s:
        return None
    obj = _parse_structured(s)
    if obj is not None:
        if not isinstance(obj, list) or not obj:
            return None
        rows = obj if all(isinstance(r, list | tuple) for r in obj) else [obj]
        try:
            return [[int(x) for x in row] for row in rows]
        except (TypeError, ValueError, OverflowError):
            # ``ast``/``json`` decode ``1e999`` and ``Infinity`` to a non-finite
            # float, which is not a grid cell and which ``int()`` refuses with
            # OverflowError (nan raises ValueError).  Not a grid: fall through
            # to the text paths instead of killing the reward worker.
            return None
    if s.startswith("[[") and s.endswith("]]") and len(s) > 4:
        # "[[1 2 3,4 5 6]]" -> [[1, 2, 3], [4, 5, 6]]; non-numeric wrappers
        # (cipher, wordscapes) drop through to the text path.
        try:
            wrapped = [[int(c) for c in row.split()] for row in s[2:-2].split(",") if row.strip()]
        except ValueError:
            return None
        return wrapped or None
    rows = [row.strip() for row in s.splitlines() if row.strip()]
    if not rows:
        return None
    grid: list[list[int]] = []
    for row in rows:
        cells = [c for c in re.split(r"[,\s]+", row) if c]
        try:
            grid.append([int(c) for c in cells])
        except ValueError:
            return None
    return grid or None


def _unwrap_answer_container(text: str) -> str | None:
    """Extract the answer from a wrapper object some prompts mandate.

    ``buggy_tables`` requires the response to be ``{"result": [{"answer": X}]}``
    (a Markdown JSON block) while ground_truth stores the bare ``X``, so a
    perfectly correct response would otherwise never match.  Returns the inner
    answer rendered as text, or None when ``text`` is not such a container.
    """
    obj = _parse_structured(text)
    if not isinstance(obj, dict):
        return None
    if "answer" in obj:
        values = [obj["answer"]]
    elif isinstance(obj.get("result"), list):
        values = [r["answer"] for r in obj["result"] if isinstance(r, dict) and "answer" in r]
    elif isinstance(obj.get("result"), dict) and "answer" in obj["result"]:
        values = [obj["result"]["answer"]]
    else:
        return None
    if not values:
        return None
    rendered = [v if isinstance(v, str) else json.dumps(v, ensure_ascii=False) for v in values]
    return rendered[0] if len(rendered) == 1 else json.dumps(rendered, ensure_ascii=False)


def _unwrap_double_brackets(text: str) -> str | None:
    """Return the inner text of a prompt-mandated ``[[...]]`` wrapper, else None."""
    s = text.strip()
    if len(s) > 4 and s.startswith("[[") and s.endswith("]]"):
        return s[2:-2].strip()
    return None


def _wrapper_tolerant_match(prediction: str, ground_truth: str, task: str | None) -> bool:
    """Last-resort comparison against prompt-mandated answer wrappers.

    Every branch here runs only after the plain comparisons failed, so it can
    add credit but never remove it.
    """
    inner = _unwrap_answer_container(prediction)
    if inner is not None and logic_answer_match(inner, ground_truth, task, _depth=1):
        return True
    inner = _unwrap_double_brackets(prediction)
    if inner is not None:
        if _loose_token_form(inner) == _loose_token_form(ground_truth):
            return True
        if task in _WHITESPACE_FREE_TASKS and re.sub(r"\s+", "", inner) == re.sub(r"\s+", "", ground_truth):
            return True
    return False


def logic_answer_match(prediction: str | None, ground_truth: str, task: str | None = None, _depth: int = 0) -> bool:
    """Normalised comparison shared by all logic_* verifiers."""
    if prediction is None:
        return False
    pred_norm = _normalise_text(prediction)
    gt_norm = _normalise_text(ground_truth)
    if not gt_norm:
        return False
    if pred_norm == gt_norm:
        return True
    # Numeric fast path (handles "42" vs "42.0").
    try:
        return math.isclose(float(pred_norm), float(gt_norm), rel_tol=1e-6, abs_tol=1e-6)
    except (ValueError, OverflowError):
        pass
    # Structural path (grids / lists; SynLogic arc_agi answers are nested lists
    # whose ground truth we serialised with json.dumps at data prep time).  A
    # mismatch falls through: the text/matrix paths below must still get a say.
    pred_obj = _parse_structured(prediction)
    gt_obj = _parse_structured(ground_truth)
    if pred_obj is not None and gt_obj is not None:
        if task in _UNORDERED_COLLECTION_TASKS:
            if _structured_equal(_canon_unordered(pred_obj), _canon_unordered(gt_obj)):
                return True
        elif _structured_equal(pred_obj, gt_obj):
            return True
    # Grid path: a nested-list answer and a space/newline separated grid ground
    # truth ("[[1, 2], [3, 4]]" vs "1 2\n3 4") are the same answer written
    # differently.  Only a match short-circuits; a mismatch falls through so the
    # whitespace-insensitive text path can still accept flat-grid answers.
    pred_grid = _parse_grid_matrix(prediction)
    gt_grid = _parse_grid_matrix(ground_truth)
    if pred_grid is not None and gt_grid is not None and pred_grid == gt_grid:
        return True
    # Loose text path: separator-spacing / inner-quote insensitive comparison
    # ("A, C, D, E" vs "A,C,D,E"; "[['WORD']]" vs "[[WORD]]").
    if _loose_token_form(prediction) == _loose_token_form(ground_truth):
        return True
    # Prompt-mandated [[...]] / {"result": [...]} wrappers (math_path,
    # buggy_tables) are presentation, not content: peel one layer and retry.
    return _depth == 0 and _wrapper_tolerant_match(prediction, ground_truth, task)


def _logic_score(
    solution_str: str, ground_truth: str, data_source: str | None = None, extra_info: dict | None = None
) -> float:
    try:
        payload = json.loads(ground_truth)
        if isinstance(payload, dict):
            answer = payload.get("answer")
            task = payload.get("task")
            meta = payload.get("meta")
        else:
            answer, task, meta = payload, None, None
    except (ValueError, TypeError):
        # ValueError covers JSONDecodeError *and* the plain ValueError CPython
        # raises for an integer literal past ``sys.get_int_max_str_digits()``;
        # both mean "this ground truth is not the JSON payload we expect".
        answer, task, meta = ground_truth, None, None
    if answer is None:
        return 0.0
    prediction = extract_logic_answer(solution_str)
    # Enigmata tasks whose answer carries task-specific semantics (arithmetic
    # expressions, maze paths, stack simulation) are verified by the dedicated
    # module; it returns None for everything it does not own.
    if data_source is not None and data_source.startswith("logic_enigmata"):
        verdict = verify_enigmata(prediction, answer, task, meta)
        if verdict is not None:
            return 1.0 if verdict else 0.0
    # Strip the "The answer is: ..." scaffolding several Enigmata answers carry
    # before the generic normalised comparison.
    if logic_answer_match(prediction, strip_prose_prefix(str(answer)), task):
        return 1.0
    # Reasoning Gym stores one canonical answer per puzzle, but countdown /
    # word_ladder / shortest_path accept many equivalent answers; only a
    # string-match failure is worth the (regeneration) cost of asking the
    # library's own task verifier, and it can only add credit here.
    if data_source is not None and data_source.startswith("logic_reasoning_gym"):
        if verify_reasoning_gym(prediction, answer, task, seed_from_extra_info(extra_info)) is True:
            return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# instruction-following (Nemotron-RL-instruction_following)
# ---------------------------------------------------------------------------


def _extract_final_response(solution_str: str) -> str:
    """Return the user-facing response body (everything after </think>).

    The format gate already guaranteed a single closed think block, so this
    partition always succeeds.  Constraints apply to the visible answer only,
    not to internal reasoning.
    """
    _, _, body = solution_str.partition("</think>")
    return body.strip()


def _if_score(solution_str: str, ground_truth: str) -> float:
    """Verify the final response against the IF constraint list.

    ground_truth is the JSON string written by to_parquet_if.py:
      {"constraints": [{"id": "<category>:<type>", "kwargs": {...}}, ...]}
    """
    try:
        from examples.reasoning_rl.reward.if_verifier import verify_instructions
    except ImportError:
        # Fallback when run as a plain module without the repo on sys.path.
        import os
        import sys

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from if_verifier import verify_instructions

    try:
        payload = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    except (ValueError, TypeError):
        logger.warning("[reasoning_rl] if ground_truth is not valid JSON")
        return 0.0
    if isinstance(payload, dict):
        constraints = payload.get("constraints", [])
    elif isinstance(payload, list):
        constraints = payload
    else:
        constraints = []
    if not isinstance(constraints, list) or not constraints:
        return 0.0
    response = _extract_final_response(solution_str)
    if not response:
        return 0.0
    score, _ = verify_instructions(response, constraints)
    return float(score)


# ---------------------------------------------------------------------------
# repetition penalty
# ---------------------------------------------------------------------------

# Degenerate loops (the same block pasted over and over) can still carry the
# right answer -- e.g. a repeated <answer> fence whose last copy is correct --
# and would otherwise collect full credit.  A single word n-gram occurring
# more than ``_rep_threshold(len)`` times marks the trajectory as a death loop;
# every occurrence past the threshold costs ``_REP_STEP``, penalties from
# several looped n-grams add up, and ``_repetition_adjusted`` floors the
# final score at ``_REP_FLOOR``.  Applied to every returned score, including
# format-gate and verifier-crash zeros: a degenerate sample deserves the
# penalty regardless of correctness.
#
# The threshold is length-scaled: a fixed count cannot separate a death loop
# from normal discourse once the response-length curriculum reaches 16k-24k
# tokens -- connective reasoning phrases ("if the first person is", "the
# answer is") naturally occur 20-40 times across ~10k words, so a flat 15
# mis-scored verbose-but-legitimate responses negative.  The threshold is
# therefore ``max(_REP_MIN_COUNT, total // _REP_RATIO)``: short texts keep
# the original absolute floor (calibrated on death loops, not prose), long
# texts only get charged when one n-gram occupies >1/_REP_RATIO of all
# n-gram positions -- the signature of a loop, not of verbose reasoning.
_REP_NGRAM = 5
_REP_MIN_COUNT = 15
_REP_RATIO = 100
_REP_STEP = 0.1
_REP_FLOOR = -1.0


def _rep_threshold(total_ngrams: int) -> int:
    return max(_REP_MIN_COUNT, total_ngrams // _REP_RATIO)


def _repetition_penalty(text: str) -> float:
    """Death-loop penalty in points: per word n-gram, each occurrence past
    ``_rep_threshold`` costs ``_REP_STEP``; multiple looped n-grams stack."""
    words = text.split()
    if len(words) < _REP_NGRAM + _REP_MIN_COUNT:  # too short to loop past the threshold
        return 0.0
    counts = Counter(zip(*(words[i:] for i in range(_REP_NGRAM))))
    threshold = _rep_threshold(len(words) - _REP_NGRAM + 1)
    excess = sum(count - threshold for count in counts.values() if count > threshold)
    return _REP_STEP * excess


def _repetition_adjusted(solution_str: str, score: float) -> float:
    return max(score - _repetition_penalty(solution_str), _REP_FLOOR)


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
):
    """Dispatch on the data_source prefix; always returns {"score": float}.

    The format gate runs first: generations that do not follow the Qwen3-4B
    thinking-template shape (see format_ok) score 0 without calling the
    domain verifier — this also keeps malformed code out of the sandbox.
    Unknown data_sources still raise once the response is format-compliant,
    so config errors surface instead of being silently zeroed.

    Every verifier call is additionally guarded: this function runs inside the
    rollout's thread pool, where any escaping exception aborts the whole rollout
    (the ``OverflowError`` from a ``1e999`` grid was one example). A pathological
    row or a crafted answer fails closed with a logged 0 instead.

    Every returned score is repetition-adjusted (``_repetition_adjusted``): a
    degenerate, heavily self-repeating generation loses 0.1 per redundant
    n-gram past a length-scaled allowance, floored at -1 -- repetition loops
    can no longer collect full credit just because the last copy is right.
    """
    if not format_ok(solution_str):
        return {"score": _repetition_adjusted(solution_str, 0.0)}
    if not data_source.startswith(("math", "code", "logic", "stem", "if")):
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")
    try:
        if data_source.startswith("math"):
            score = _math_score(solution_str, ground_truth)
        elif data_source.startswith("code"):
            score = _code_score(solution_str, ground_truth, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb)
        elif data_source.startswith("logic"):
            score = _logic_score(solution_str, ground_truth, data_source, extra_info)
        elif data_source.startswith("stem"):
            score = _math_score(solution_str, ground_truth)
        else:  # if_*
            score = _if_score(solution_str, ground_truth)
    except Exception:
        logger.exception("[reasoning_rl] verifier crashed for %s; scoring 0", data_source)
        return {"score": _repetition_adjusted(solution_str, 0.0)}
    return {"score": _repetition_adjusted(solution_str, float(score))}
