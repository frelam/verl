"""Tool RL reward for verl — 3 dimensions, rule-based (label mode only).

Ported from slime ``examples/tool_rl/reward/reward.py``. The LLM-judge
(RM v2) mode is intentionally NOT migrated — samples without structured
ground-truth tool calls are filtered out at data preparation time
(see ``examples/tool_rl/prepare_data.py``).

Reward Dimensions
-----------------
==============  ==============================  ======  =============
Dim             Name                            Weight  Source
==============  ==============================  ======  =============
Dim 1           工具调用正确性 (Tool Correctness)  0.60    Label match
Dim 2           回复格式合规 (Format)              0.20    Verifier
Dim 3           工具调用格式 (Tool Call Format)    0.20    Verifier
==============  ==============================  ======  =============

Dim 1 — rule-based, order-independent matching against ground-truth labels:
  - Tool name match  → 0.5  (binary per label call)
  - Param content    → 0.5  (value match per label param)
  Calling a tool not declared in the prompt incurs ``-0.1`` per call on
  Dim 1. The penalty always stacks on the (possibly negative) match score —
  flooring at 0.0 would invert the ordering: an undeclared call would
  outscore a declared-but-unneeded one.

Dim 2 — Verifier (format, answer-agnostic):
  0.6 if all tool_calls after reasoning + 0.4 × count/N for think before
  each call.

Dim 3 — Verifier (tool call format vs label):
  1/N per label call completely matched (name + param names + param types);
  full score when the label has no tool calls.

No-tool behaviour shaping (``TOOL_RL_ABSTAIN_MODE=keyword``)
------------------------------------------------------------
When the label needs no tools (``ground_truth_calls == []``), Dim 1 and
Dim 3 are reshaped by the rule-based abstention classifier
(``reward/abstention.py`` — no reward model involved):

============================  ======  ======
Behaviour                     Dim 1   Dim 3
============================  ======  ======
request more info             1.0     1.0
declare no valid tools        1.0     1.0
guess a direct answer         0.0     1.0
spurious call (declared)      0.0     0.0
spurious call (undeclared)    -0.1×n  0.0
============================  ======  ======

Dim 2 is untouched (format compliance is answer-agnostic). With the
default weights this yields: clarify/declare 1.0 > guess 0.4 > spurious
call 0.2 > undeclared spurious call 0.14.

Guess-penalty exemption (two signals, either suffices): when the label
string carries the dataset's own reference response (``Reference:\n...``
— ToolACE turns) and that reference is itself a direct answer, or when
the sample is tagged ``answerable_direct`` at data-prep time (the query
is resolvable by pure computation — self-computable original tool or an
arithmetic-looking query), the sample's desired behaviour IS answering,
so a guess-class response scores Dim 1 = 1.0 like the desired classes.
Designed-abstention negatives (hammer, ``desc_replace``, ``no_tools``)
carry neither signal and keep the guess penalty.

Strict think-format gate
------------------------
A broken response layout — unclosed ``<think>`` opener, stray
``</think>`` closer (one or many), multiple think blocks,
think-then-stop, or **trailing content after the last tool call**
(wrapped ``<tool_call>`` block or bare JSON call alike — unreachable:
the harness executes the calls) — zeroes the **entire** reward (all
dims + total). Format is the basic contract: no partial credit for
tool-call quality when the response layout is malformed. The
``name_score`` / ``param_content_score`` breakdown fields stay raw as
diagnostics. A bare JSON call with no trailing text is NOT a strict
violation: Dim 2/Dim 3 zero its format credit, Dim 1 keeps content
credit (fallback).

Repetition penalty
------------------
Degenerate loops in the non-tool-call text (think + visible reply —
``<tool_call>`` blocks are stripped first) are detected with word-level
n-grams. The detector is deliberately lenient — it targets death loops
and nothing else: ``repeats`` counts only occurrences BEYOND the
``free``-th (default 4) of the same n-gram, and the first ``threshold``
repeats are tolerated as well, so a single n-gram must appear 13+ times
before anything fires. Every further repeat costs ``per_repeat``, capped
at ``max_penalty``. The penalty is subtracted from the total AFTER the
strict-format gate and may push the score negative. Knobs (env vars):
``TOOL_RL_REPEAT_NGRAM`` (4), ``TOOL_RL_REPEAT_FREE`` (4),
``TOOL_RL_REPEAT_THRESHOLD`` (8), ``TOOL_RL_REPEAT_PER`` (0.1),
``TOOL_RL_REPEAT_MAX`` (1.0).

Score range
-----------
The final ``score`` is clamped to ``[-1.0, 1.0]``. The weighted sum is
≤ 1 by construction, but undeclared calls (−0.1 each, unbounded) and
the repetition penalty can push the total below −1. In-range values
are preserved exactly; per-dimension breakdown fields stay raw
(unclamped) as diagnostics.

verl integration
----------------
Loaded via ``reward.custom_reward_function.path`` / ``.name=compute_score``.
Per-sample fields travel in the dataset's ``extra_info`` column:

- ``extra_info["tools"]``              — available tool schemas
- ``extra_info["ground_truth_calls"]`` — structured GT tool calls
  (``[]`` = label says no tools needed)
- ``extra_info["task_id"]``            — for logging

Optional knobs (env vars):
- ``TOOL_RL_REWARD_WEIGHTS`` — JSON dict overriding dimension weights,
  e.g. ``'{"tool_correctness": 0.6, "format": 0.2, "tool_call": 0.2}'``.
- ``TOOL_RL_ABSTAIN_MODE`` — ``keyword`` (default) | ``off``; the no-tool
  behaviour shaping described above. On by default so it survives env
  propagation losses in reward workers.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Make sibling / package imports work regardless of how this file is loaded
# (verl loads custom reward functions by file path via importlib).
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.tool_rl.reward.abstention import (  # noqa: E402
    ABSTENTION_NOT_APPLICABLE,
    AbstentionClass,
    abstain_mode_from_env,
    classify_abstention,
    reference_prefers_answer,
)
from examples.tool_rl.reward.verifier import (  # noqa: E402
    _check_strict_format,
    compute_verifier_scores,
    match_tool_calls_against_label,
    parse_ground_truth_calls,
    parse_qwen_tool_calls,
    repetition_penalty,
    undeclared_tool_penalty,
)

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS: dict[str, float] = {
    "tool_correctness": 0.60,
    "format": 0.20,
    "tool_call": 0.20,
}


def _get_weights() -> dict[str, float]:
    """Resolve weights from ``TOOL_RL_REWARD_WEIGHTS`` JSON, else defaults."""
    raw = os.environ.get("TOOL_RL_REWARD_WEIGHTS")
    defaults = dict(DEFAULT_WEIGHTS)
    if not raw:
        return defaults
    try:
        override = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Invalid TOOL_RL_REWARD_WEIGHTS JSON: %r", raw)
        return defaults
    if not isinstance(override, dict):
        return defaults
    for k in defaults:
        if k in override:
            defaults[k] = float(override[k])
    total = sum(defaults.values())
    if total > 0:
        defaults = {k: v / total for k, v in defaults.items()}
    return defaults


def _get_repetition_config() -> dict[str, Any]:
    """Repetition-penalty knobs (env-overridable).

    - ``TOOL_RL_REPEAT_NGRAM``     — n-gram size in word tokens (default 4)
    - ``TOOL_RL_REPEAT_FREE``      — occurrences of the same n-gram
      tolerated before counting (default 4)
    - ``TOOL_RL_REPEAT_THRESHOLD`` — repeat events tolerated (default 8)
    - ``TOOL_RL_REPEAT_PER``       — penalty per repeat beyond threshold
      (default 0.1)
    - ``TOOL_RL_REPEAT_MAX``       — cap on the total penalty (default 1.0)
    """
    return {
        "ngram": int(os.environ.get("TOOL_RL_REPEAT_NGRAM", "4")),
        "free": int(os.environ.get("TOOL_RL_REPEAT_FREE", "4")),
        "threshold": int(os.environ.get("TOOL_RL_REPEAT_THRESHOLD", "8")),
        "per_repeat": float(os.environ.get("TOOL_RL_REPEAT_PER", "0.1")),
        "max_penalty": float(os.environ.get("TOOL_RL_REPEAT_MAX", "1.0")),
    }


def _to_list(value: Any) -> list:
    """Normalise parquet/numpy containers to a plain Python list."""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _to_dict_list(value: Any) -> list[dict[str, Any]]:
    """Normalise a (possibly numpy / JSON-string) list of dicts.

    ``ground_truth_calls`` is stored in the parquet as a JSON string (mixed
    argument value types across datasets prevent a native struct column), so
    strings are parsed back here.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    result = []
    for item in _to_list(value):
        if hasattr(item, "tolist"):
            item = item.tolist()
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except json.JSONDecodeError:
                continue
        if isinstance(item, dict):
            result.append(dict(item))
    return result


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """verl custom reward entry point (label mode only).

    Args:
        data_source: Dataset source tag (unused, kept for the verl protocol).
        solution_str: Decoded model response (``<think>`` + ``<tool_call>``).
        ground_truth: Readable ground-truth label string (logging only).
        extra_info: Per-sample fields — must carry ``tools`` and
            ``ground_truth_calls`` (see module docstring).

    Returns:
        Dict with ``score`` plus per-dimension breakdown; extra keys are
        surfaced by the reward manager as ``reward_extra_info`` metrics.
    """
    extra_info = dict(extra_info or {})
    available_tools = _to_dict_list(extra_info.get("tools"))
    # Data-prep tag: negative sample whose query the model can answer by
    # pure computation (self-computable original tool / arithmetic query)
    # — a self-computed direct answer is legitimate, not a guess.
    answerable_direct = bool(extra_info.get("answerable_direct"))
    ground_truth_calls = extra_info.get("ground_truth_calls", None)
    task_id = extra_info.get("task_id", "unknown")

    # ``None`` (no label) should have been filtered at data prep; degrade
    # gracefully to an empty label so the run does not crash.
    if ground_truth_calls is None:
        logger.warning(
            "[tool_rl] %s: no ground_truth_calls in extra_info — "
            "treating as empty label (RM mode is not migrated)",
            task_id,
        )
        ground_truth_calls = []

    weights = _get_weights()

    parsed_gt = parse_ground_truth_calls(_to_dict_list(ground_truth_calls))
    expects_no_tools = len(parsed_gt) == 0

    # Wrap the single-turn response as a pseudo-trajectory for the verifier.
    trajectory = [{"turn": 0, "text": solution_str, "type": "turn"}]
    output_calls = parse_qwen_tool_calls(solution_str)

    # ── Strict think-format gate ──
    # A broken think-block layout (unclosed ``<think>`` opener, stray
    # ``</think>`` closer — one or many, multiple think blocks,
    # think-then-stop) zeroes the ENTIRE reward: format is the basic
    # contract, so there is no partial credit for tool-call quality.
    # Dim 2 / Dim 3 already self-zero via their internal strict gates;
    # the gate here additionally zeroes Dim 1 and the total.
    strict_format_ok = _check_strict_format(solution_str)

    # ── Dim 2 + Dim 3: Verifier (rule-based) ──
    verifier = compute_verifier_scores(
        trajectory,
        available_tools=available_tools,
        label_calls=parsed_gt,
        expects_no_tools=expects_no_tools,
    )
    format_score = verifier["format_compliance"]
    tool_call_score = verifier["tool_call_format"]

    # ── Dim 1: rule-based label matching ──
    name_score, param_score = match_tool_calls_against_label(output_calls, parsed_gt)
    tool_correctness = 0.5 * name_score + 0.5 * param_score

    # Dim 1 undeclared-tool penalty: -0.1 per undeclared call. It must
    # ALWAYS stack on the match score — the old conditional floor
    # (max(0.0, ...)) inverted the ordering: an undeclared call (floored
    # to 0.0) outscored a declared-but-unneeded call (-0.1).
    undeclared_penalty = undeclared_tool_penalty(output_calls, available_tools)
    tool_correctness -= undeclared_penalty

    # ── No-tool-label behaviour shaping (keyword mode) ──
    # Dim 2 stays answer-agnostic; only Dim 1 / Dim 3 are reshaped.
    abstention_class = ABSTENTION_NOT_APPLICABLE
    ref_direct = False
    if abstain_mode_from_env() == "keyword" and expects_no_tools:
        if output_calls:
            # Spurious call: Dim 1 = 0 (undeclared calls still subtract
            # 0.1 each), Dim 3 = 0.
            tool_correctness = -undeclared_penalty
            tool_call_score = 0.0
            abstention_class = AbstentionClass.SPURIOUS_CALL
        else:
            cls = classify_abstention(solution_str)
            abstention_class = cls
            # Reference-conditioned guess penalty: when the dataset's own
            # reference response is a direct answer (ToolACE chitchat /
            # general-knowledge negatives), answering directly IS the
            # demonstrated desired behaviour — no guess penalty.
            # Likewise for samples tagged answerable_direct at data-prep
            # time: the query is self-computable, so working out a value
            # manually is legitimate. Designed-abstention negatives
            # (hammer, desc_replace, no_tools) carry neither signal and
            # keep it.
            ref_direct = reference_prefers_answer(ground_truth)
            guess_exempt = ref_direct or answerable_direct
            tool_correctness = (
                0.0 if cls is AbstentionClass.GUESS and not guess_exempt else 1.0
            )

    # ── Repetition penalty (degenerate loops outside tool calls) ──
    # Applies to think + reply text (tool_call blocks excluded); stacks on
    # top of everything else and may push the score negative.
    rep_penalty, rep_repeats = repetition_penalty(
        solution_str, **_get_repetition_config(),
    )

    # ── Weighted sum (negatives allowed so blind guessing scores < 0) ──
    total = (
        weights["tool_correctness"] * tool_correctness
        + weights["format"] * format_score
        + weights["tool_call"] * tool_call_score
    )

    # Strict-format gate: zero Dim 1 and the total; name/param sub-scores
    # stay raw in the breakdown as diagnostics of what was matched.
    if not strict_format_ok:
        tool_correctness = 0.0
        total = 0.0

    total -= rep_penalty

    # ── Range guard: clamp the final reward to [-1, 1] ──
    # The weighted sum is ≤ 1 by construction, but undeclared calls
    # (-0.1 each, unbounded) and the repetition penalty can push the
    # total below -1. In-range values are preserved exactly; breakdown
    # fields above stay raw (unclamped) as diagnostics.
    total = max(-1.0, min(1.0, total))

    logger.info(
        "[tool_rl] %s: total=%.3f correctness=%.3f(name=%.3f+param=%.3f) "
        "format=%.3f tool_call=%.3f abstention=%s strict_format=%s "
        "rep_penalty=%.3f(repeats=%d) ref_direct=%s answerable_direct=%s",
        task_id, total, tool_correctness, name_score, param_score,
        format_score, tool_call_score,
        AbstentionClass(abstention_class).name
        if abstention_class != ABSTENTION_NOT_APPLICABLE else "n/a",
        strict_format_ok, rep_penalty, rep_repeats, ref_direct,
        answerable_direct,
    )

    return {
        "score": total,
        "tool_correctness": tool_correctness,
        "name_score": name_score,
        "param_content_score": param_score,
        "format_compliance": format_score,
        "tool_call_format": tool_call_score,
        "abstention_class": int(abstention_class),
        "abstention_ref_direct": float(ref_direct),
        "abstention_answerable_direct": float(answerable_direct),
        "repetition_penalty": rep_penalty,
        "repetition_repeats": rep_repeats,
    }
