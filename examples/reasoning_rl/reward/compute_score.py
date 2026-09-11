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
                     to verl's math_dapo when the package is missing
``code_*``           sandbox_fusion when ``sandbox_fusion_url`` is set,
                     else prime_code local execution (smoke runs only)
``logic_*``          rule verifier: extract the final answer
                     (<answer> tags -> \\boxed{} -> "Final Answer:" /
                     "The answer is ..." line -> last fenced code block)
                     and compare with the ground truth after normalisation
                     (whitespace/case folding, markdown-emphasis stripping,
                     separator-spacing/quote-insensitive text compare,
                     literal/JSON structural compare, numeric tolerance)
``stem_*``           math_verify on the \\boxed{} answer; Dr.SCI prompts
                     already request the boxed format
``if_*``             instruction-following (Nemotron-RL-instruction_following):
                     re-check every constraint in ground_truth against the
                     model response; reward 1.0 iff ALL pass (NeMo-Gym
                     convention).  Only the final (post-</think>) response is
                     verified — the think block is internal reasoning.
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
"""

from __future__ import annotations

import ast
import json
import logging
import math
import re

logger = logging.getLogger(__name__)

_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\s*\{")
# SynLogic task prompts use heterogeneous final-answer conventions: besides
# "Final Answer: ...", many tasks (web_of_lies, cryptarithm, word_sorting_mistake)
# instruct 'The answer is $YOUR_ANSWER' — often wrapped in markdown bold.
_FINAL_ANSWER_RE = re.compile(r"(?:final answer\s*[::]|the answer is)\s*[::]?\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_FENCED_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n(.*?)```", re.DOTALL)


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
    try:
        from verl.utils.reward_score import math_verify

        return float(math_verify.compute_score(solution_str, str(ground_truth)))
    except ImportError:
        from verl.utils.reward_score import math_dapo

        # math_dapo returns {"score": ±1.0, "acc": bool, ...}; normalise to 0/1
        # so all four domains share the same pass-rate semantics.
        res = math_dapo.compute_score(solution_str, str(ground_truth))
        return float(res["acc"]) if isinstance(res, dict) else float(res)
    except Exception as e:  # never let one bad sample kill the reward pass
        logger.warning("[reasoning_rl] math verify error: %s", e)
        return 0.0


# ---------------------------------------------------------------------------
# code
# ---------------------------------------------------------------------------


def _code_score(
    solution_str: str, ground_truth: str, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb
) -> float:
    try:
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
    code block (DESIGN.md section 1 + SynLogic per-task prompt conventions)."""
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
    return None


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
            return obj
    return None


def _structured_equal(a, b, tol: float = 1e-6) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, int | float) and isinstance(b, int | float):
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        return len(a) == len(b) and all(_structured_equal(x, y, tol) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_structured_equal(a[k], b[k], tol) for k in a)
    return _loose_token_form(str(a)) == _loose_token_form(str(b))


# SynLogic tasks whose answer is an unordered collection of coordinates or
# dominos — both sides are canonicalised (recursively sorted) before the
# structural compare so collection ordering never decides the reward.
_UNORDERED_COORD_TASKS = frozenset({"minesweeper", "norinori", "star_placement_puzzle"})


def _canon_unordered(obj):
    """Normalise tuples to lists and sort (bottom-up) any list whose elements
    are all lists. Applied only to _UNORDERED_COORD_TASKS answers — never to
    grids, where row/column order is the answer."""
    if isinstance(obj, dict):
        return {k: _canon_unordered(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        items = [_canon_unordered(x) for x in obj]
        if items and all(isinstance(x, list) for x in items):
            items.sort(key=json.dumps)
        return items
    return obj


def logic_answer_match(prediction: str | None, ground_truth: str, task: str | None = None) -> bool:
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
    # whose ground truth we serialised with json.dumps at data prep time).
    pred_obj = _parse_structured(prediction)
    gt_obj = _parse_structured(ground_truth)
    if pred_obj is not None and gt_obj is not None:
        if task in _UNORDERED_COORD_TASKS:
            return _structured_equal(_canon_unordered(pred_obj), _canon_unordered(gt_obj))
        return _structured_equal(pred_obj, gt_obj)
    # Loose text path: separator-spacing / inner-quote insensitive comparison
    # ("A, C, D, E" vs "A,C,D,E"; "[['WORD']]" vs "[[WORD]]").
    return _loose_token_form(prediction) == _loose_token_form(ground_truth)


def _logic_score(solution_str: str, ground_truth: str) -> float:
    try:
        payload = json.loads(ground_truth)
        answer = payload.get("answer") if isinstance(payload, dict) else payload
        task = payload.get("task") if isinstance(payload, dict) else None
    except (json.JSONDecodeError, TypeError):
        answer, task = ground_truth, None
    if answer is None:
        return 0.0
    prediction = extract_logic_answer(solution_str)
    return 1.0 if logic_answer_match(prediction, str(answer), task) else 0.0


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
    except (json.JSONDecodeError, TypeError):
        logger.warning("[reasoning_rl] if ground_truth is not valid JSON")
        return 0.0
    if isinstance(payload, dict):
        constraints = payload.get("constraints", [])
    elif isinstance(payload, list):
        constraints = payload
    else:
        constraints = []
    if not constraints:
        return 0.0
    response = _extract_final_response(solution_str)
    if not response:
        return 0.0
    score, _ = verify_instructions(response, constraints)
    return float(score)


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
    """
    if not format_ok(solution_str):
        return {"score": 0.0}
    if data_source.startswith("math"):
        score = _math_score(solution_str, ground_truth)
    elif data_source.startswith("code"):
        score = _code_score(solution_str, ground_truth, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb)
    elif data_source.startswith("logic"):
        score = _logic_score(solution_str, ground_truth)
    elif data_source.startswith("stem"):
        score = _math_score(solution_str, ground_truth)
    elif data_source.startswith("if"):
        score = _if_score(solution_str, ground_truth)
    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")
    return {"score": float(score)}
