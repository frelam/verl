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
"""Task-aware verifier for the Enigmata puzzle domain (``logic_enigmata``).

Why this exists
---------------
``to_parquet_logic.py`` stores the raw Enigmata ``answer`` field verbatim.  For
several tasks that field is *not* the string the model is asked to produce::

    game24            "The answer is: (12-10)*(12*1) = 24"   (expression, many solutions)
    countdown         "The answer is: 6*11-(15+13)/4 = 59"   (expression, many solutions)
    maze              "The answer is: (1,1)->(1,2)->..."     (prose prefix)
    stack_permutation "The output sequence is a valid stack permutation."  (sentence, not ops)

Comparing those strings literally against the model's answer can never succeed,
so a correct response is scored 0.  (Verified: countdown/game24/maze/
stack_permutation together are ~10.8% of the Enigmata pool.)

This module mirrors the official per-task verifiers shipped in
``BytedTsinghua-SIA/Enigmata`` (``verifiable_tasks/tasks/<task>/verifier.py``)
adapted to the reasoning_rl reward contract:

* the model answer is already extracted by ``extract_logic_answer`` (``<answer>``
  tags -> ``\\boxed{}`` -> final-answer line -> last fenced block), so the
  per-task logic operates on that candidate;
* expression tasks (game24/countdown) are evaluated arithmetically and, when
  ``meta`` carries the input numbers, checked to use exactly those numbers --
  this is what stops the policy from hacking the reward with ``24``;
* maze/stack_permutation need the original puzzle data, which
  ``to_parquet_logic.py`` now stores under ``"meta"`` in ``ground_truth``.  Rows
  built before that change (no ``meta``) still get the safe fallbacks documented
  per function below.

``verify_enigmata`` returns ``True``/``False`` when it owns the task, and
``None`` when the task is not handled here (caller falls back to the generic
matcher).
"""

from __future__ import annotations

import ast
import logging
import operator
import re

logger = logging.getLogger(__name__)

# Natural-language scaffolding that Enigmata puts in front of the real answer,
# e.g. "The answer is: ...", "The final answer is: ...", "Answer: ...".
_PROSE_PREFIX_RE = re.compile(
    r"^\s*(?:the\s+)?(?:final\s+)?answer\s*(?:is)?\s*[::]?\s*",
    re.IGNORECASE,
)

_COORD_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")
_NUMBER_RE = re.compile(r"\d+")

# Tasks whose ground truth needs semantic verification instead of string match.
HANDLED_TASKS = frozenset({"game24", "countdown", "maze", "stack_permutation"})

_ARITH_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
# Guard against a degenerate ``9**9**9``-style rollout hanging the driver.
_MAX_POW_EXPONENT = 64


def strip_prose_prefix(text: str) -> str:
    """Drop a leading "The answer is:"-style preamble from ``text``."""
    if not text:
        return text
    return _PROSE_PREFIX_RE.sub("", text, count=1).strip()


# ---------------------------------------------------------------------------
# arithmetic-expression tasks (game24, countdown)
# ---------------------------------------------------------------------------


def _safe_arith_eval(expr: str) -> float | None:
    """Evaluate a pure-arithmetic expression; ``None`` if it is not one.

    Implemented as an AST walk (no ``eval``) so model output can never reach a
    builtin.  Only numeric literals, unary +/- and + - * / // % ** are allowed.
    """
    try:
        node = ast.parse(expr, mode="eval").body
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None

    def ev(n):
        if isinstance(n, ast.Constant):
            if isinstance(n.value, bool) or not isinstance(n.value, int | float):
                raise ValueError("non-numeric literal")
            return n.value
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.UAdd | ast.USub):
            value = ev(n.operand)
            return value if isinstance(n.op, ast.UAdd) else -value
        if isinstance(n, ast.BinOp) and type(n.op) in _ARITH_BINOPS:
            left, right = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(right) > _MAX_POW_EXPONENT:
                raise ValueError("exponent too large")
            return _ARITH_BINOPS[type(n.op)](left, right)
        raise ValueError("disallowed expression node")

    try:
        return float(ev(node))
    except (ValueError, ZeroDivisionError, OverflowError, TypeError):
        return None


def _normalise_expression(text: str) -> str:
    """Canonicalise an expression candidate: strip prose, drop ``= target``,
    map the multiplication/division glyphs the Enigmata prompts allow."""
    s = strip_prose_prefix(text or "")
    s = s.split("=")[0]
    for src, dst in (("×", "*"), ("÷", "/"), ("−", "-"), ("x", "*"), ("X", "*")):
        s = s.replace(src, dst)
    return s.strip()


def _number_key(value) -> str:
    """Canonical string for a number so 4 / 4.0 / "4" compare equal."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(f)) if f.is_integer() else repr(f)


def _target_from_answer(answer: str) -> float | None:
    """Recover the target from a countdown/arithmetic ground truth.

    ``"The answer is: 6*11-(15+13)/4 = 59"`` -> ``59.0``.
    """
    s = strip_prose_prefix(str(answer))
    if "=" in s:
        s = s.rsplit("=", 1)[-1]
    match = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(match.group()) if match else None


def _verify_expression_task(prediction: str | None, answer, target: float, meta) -> bool:
    """Shared game24/countdown check: evaluate the expression and, when the
    input numbers are known, require exactly those numbers to be used."""
    answer_text = strip_prose_prefix(str(answer))
    if "cannot form" in answer_text.lower():
        return prediction is not None and "cannot form" in prediction.lower()
    if prediction is None:
        return False
    expr = _normalise_expression(prediction)
    value = _safe_arith_eval(expr)
    if value is None or abs(value - target) > 1e-4:
        return False
    numbers = (meta or {}).get("question")
    numeric_numbers = None
    if isinstance(numbers, list) and numbers:
        try:
            numeric_numbers = sorted(_number_key(n) for n in numbers)
        except (TypeError, ValueError):
            numeric_numbers = None
    used = sorted(_NUMBER_RE.findall(expr))
    if numeric_numbers is not None:
        # The input numbers are known: the expression must use exactly them, so
        # the policy cannot hack the reward by emitting the bare target.
        if used != numeric_numbers:
            return False
    elif not re.search(r"[+\-*/]", expr):
        # Legacy row without metadata: at least require a real expression.
        return False
    return True


# ---------------------------------------------------------------------------
# maze
# ---------------------------------------------------------------------------


def _extract_coords(text: str) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in _COORD_RE.findall(text or "")]


def _verify_maze(prediction: str | None, answer, meta) -> bool:
    """Official maze rule when ``meta`` carries the grid; else exact path match.

    Legacy rows (no ``meta``) fall back to comparing the coordinate sequences,
    which is correct whenever the puzzle has a unique path (the Enigmata maze
    generator emits one) and never creates false positives.
    """
    answer_text = strip_prose_prefix(str(answer))
    if "not exist" in answer_text.lower():
        return prediction is not None and "not exist" in prediction.lower()
    if prediction is None:
        return False
    coords = _extract_coords(prediction)
    if not coords:
        return False

    maze = (meta or {}).get("question")
    height = (meta or {}).get("height")
    width = (meta or {}).get("width")
    if isinstance(maze, list) and maze and height and width:
        try:
            height, width = int(height), int(width)
            if coords[0] != (1, 1) or coords[-1] != (height, width):
                return False
            for i, (row, col) in enumerate(coords):
                if not (1 <= row <= height and 1 <= col <= width):
                    return False
                if maze[row - 1][col - 1] == "B":
                    return False
                if i and abs(row - coords[i - 1][0]) + abs(col - coords[i - 1][1]) != 1:
                    return False
            return True
        except (IndexError, TypeError, ValueError):
            return False

    gt_coords = _extract_coords(answer_text)
    return bool(gt_coords) and coords == gt_coords


# ---------------------------------------------------------------------------
# stack_permutation
# ---------------------------------------------------------------------------


def _parse_stack_ops(text: str | None) -> list[str] | None:
    if not text:
        return None
    try:
        obj = ast.literal_eval(strip_prose_prefix(text).strip())
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None
    if isinstance(obj, list | tuple) and obj and all(isinstance(x, str) for x in obj):
        return list(obj)
    return None


def _simulate_stack(ops: list[str], input_sequence: list, output_sequence: list) -> bool:
    stack: list = []
    input_idx = output_idx = 0
    try:
        for item in ops:
            if item.startswith("Pop"):
                if not stack or stack.pop() != output_sequence[output_idx]:
                    return False
                output_idx += 1
            elif item.startswith("Push"):
                push_element = int(item.strip()[5:-1])
                if push_element != input_sequence[input_idx]:
                    return False
                stack.append(input_sequence[input_idx])
                input_idx += 1
            else:
                return False
    except (IndexError, ValueError):
        return False
    return input_idx == len(input_sequence) and output_idx == len(output_sequence)


_INVALID_MARKERS = ("not a valid", "not valid")


def _verify_stack_permutation(prediction: str | None, answer, meta) -> bool:
    """Official simulation when ``meta`` carries input/output sequences.

    Legacy rows have no sequences in ``ground_truth`` and the prompt is not
    available to the reward function, so a sequence prediction cannot be
    validated without risking false positives -- those rows stay strict until
    the logic parquet is rebuilt with ``meta``.
    """
    answer_text = str(answer)
    answer_invalid = any(marker in answer_text.lower() for marker in _INVALID_MARKERS)
    prediction_invalid = prediction is not None and any(marker in prediction.lower() for marker in _INVALID_MARKERS)
    if answer_invalid:
        return prediction_invalid
    if prediction_invalid:
        return False
    ops = _parse_stack_ops(prediction)
    if ops is None:
        return False
    input_sequence = (meta or {}).get("input_sequence")
    output_sequence = (meta or {}).get("output_sequence")
    if not isinstance(input_sequence, list) or not isinstance(output_sequence, list):
        return False
    return _simulate_stack(ops, input_sequence, output_sequence)


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------


def verify_enigmata(
    prediction: str | None,
    answer,
    task: str | None,
    meta=None,
) -> bool | None:
    """Return the Enigmata verdict, or ``None`` if the task is not handled here.

    Args:
        prediction: the model's extracted final answer (see ``extract_logic_answer``).
        answer: the raw Enigmata ``answer`` field stored in ``ground_truth``.
        task: the Enigmata ``task`` name stored in ``ground_truth``.
        meta: optional task metadata stored in ``ground_truth`` (numbers, maze
            grid, sequences, target).  Absent for rows built before the
            ``to_parquet_logic.py`` meta change.
    """
    if task not in HANDLED_TASKS:
        return None
    meta = meta if isinstance(meta, dict) else None

    if task == "game24":
        target = 24.0
        if meta and isinstance(meta.get("target"), int | float):
            target = float(meta["target"])
        return _verify_expression_task(prediction, answer, target, meta)
    if task == "countdown":
        target = None
        if meta and isinstance(meta.get("target"), int | float):
            target = float(meta["target"])
        else:
            target = _target_from_answer(answer)
        if target is None:
            return None
        return _verify_expression_task(prediction, answer, target, meta)
    if task == "maze":
        return _verify_maze(prediction, answer, meta)
    if task == "stack_permutation":
        return _verify_stack_permutation(prediction, answer, meta)
    return None
