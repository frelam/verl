"""``math_ci`` reward - math answer equivalence (plan §5).

``score = 1.0`` iff ``trajectory["final_answer"]`` is mathematically equivalent
to ``task.reference["final_answer"]``, else ``0.0`` (plan §5: 复用推理 RL 的
数学验证器做答案等价判断).  The prediction is read **only** from
``trajectory["final_answer"]`` (CONTRACTS C4); tool calls / env state are
recorded in ``info`` but never affect the score.

Verifier interface (aligned with, not dependent on, verl)
---------------------------------------------------------
The equivalence logic is a stdlib-only port of the verl reasoning-RL verifier
``verl/utils/reward_score/math_reward.py``: same function names
(``is_equiv`` / ``strip_string`` / ``last_boxed_only_string`` /
``remove_boxed``) and the same ``compute_score(solution_str, ground_truth)
-> float`` entry point, so the module can be swapped for the user's existing
verifier without changing call sites.  It never imports torch or verl (D1/§9).
A numeric-tolerance fallback is added after the reference string
normalisation, so ``0.5`` and ``1/2`` agree even when the LaTeX normalisation
cannot prove it.

Missing ``final_answer`` on either side scores ``0.0`` with an explicit
``info["reason"]`` (never raises).  Scores stay in the frozen ``[0, 1]``
range (CONTRACTS C4).
"""

from __future__ import annotations

import math

from . import base

TASK_TYPE = "math_ci"
MAX_SCORE = 1.0


# ---------------------------------------------------------------------------
# stdlib port of verl/utils/reward_score/math_reward.py (no torch / verl)
# ---------------------------------------------------------------------------

def last_boxed_only_string(string: str):
    """Extract the last ``\\boxed{...}`` / ``\\fbox{...}`` literal, else ``None``."""
    index = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if index < 0:
        index = string.rfind("\\fbox")
        if index < 0:
            return None

    position = index
    right_brace_index = None
    braces_open = 0
    while position < len(string):
        if string[position] == "{":
            braces_open += 1
        if string[position] == "}":
            braces_open -= 1
            if braces_open == 0:
                right_brace_index = position
                break
        position += 1

    if right_brace_index is None:
        return None
    return string[index : right_brace_index + 1]


def remove_boxed(text: str):
    """Strip the ``\\boxed{...}`` wrapper (returns the input unchanged otherwise)."""
    if text is None:
        return None
    if text.startswith("\\boxed "):
        return text[len("\\boxed ") :]
    left = "\\boxed{"
    if text.startswith(left) and text.endswith("}"):
        return text[len(left) : -1]
    return text

def fix_fracs(string: str) -> str:
    """``\\frac1b`` / ``\\frac12`` -> ``\\frac{1}{b}`` / ``\\frac{1}{2}``."""
    substrings = string.split("\\frac")
    new_string = substrings[0]
    if len(substrings) > 1:
        for substring in substrings[1:]:
            new_string += "\\frac"
            if not substring:
                return string
            if substring[0] == "{":
                new_string += substring
            else:
                if len(substring) < 2:
                    return string
                first = substring[0]
                second = substring[1]
                if second != "{":
                    if len(substring) > 2:
                        new_string += "{" + first + "}{" + second + "}" + substring[2:]
                    else:
                        new_string += "{" + first + "}{" + second + "}"
                else:
                    if len(substring) > 2:
                        new_string += "{" + first + "}" + second + substring[2:]
                    else:
                        new_string += "{" + first + "}" + second
    return new_string


def fix_a_slash_b(string: str) -> str:
    """``a/b`` (integer pair) -> ``\\frac{a}{b}``."""
    if len(string.split("/")) != 2:
        return string
    first = string.split("/")[0]
    second = string.split("/")[1]
    try:
        first_int = int(first)
        second_int = int(second)
    except (TypeError, ValueError):
        return string
    if string != "{}/{}".format(first_int, second_int):
        return string
    return "\\frac{" + str(first_int) + "}{" + str(second_int) + "}"


def remove_right_units(string: str) -> str:
    """Drop a right-hand ``\\text{ units}`` suffix."""
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return string


def fix_sqrt(string: str) -> str:
    """``\\sqrt3`` -> ``\\sqrt{3}``."""
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if not split:
            new_string += "\\sqrt"
        elif split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def strip_string(string: str) -> str:
    """Canonicalise a LaTeX-ish answer string (port of the verl helper)."""
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = remove_right_units(string)
    string = string.replace("\\\\%", "")
    string = string.replace("\\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = fix_sqrt(string)
    string = string.replace(" ", "")
    string = fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = fix_a_slash_b(string)
    return string


def _numeric_equal(left: str, right: str) -> bool:
    """Numeric fallback for answers the string normalisation cannot equate."""
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-9)
    except (TypeError, ValueError):
        return False


def is_equiv(str1, str2, verbose: bool = False) -> bool:
    """True when two answers are mathematically equivalent (verl-style helper)."""
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False
    try:
        stripped_left = strip_string(str1)
        stripped_right = strip_string(str2)
    except Exception:  # pragma: no cover - defensive against pathological input
        return str(str1) == str(str2)
    if verbose:  # interface parity with the verl helper
        return stripped_left == stripped_right
    if stripped_left == stripped_right:
        return True
    return _numeric_equal(stripped_left, stripped_right)


def extract_answer(text: str):
    """Extract the candidate answer from a final-answer string (boxed-first)."""
    if not isinstance(text, str):
        return None
    boxed = last_boxed_only_string(text)
    if boxed is not None:
        return remove_boxed(boxed)
    stripped = text.strip()
    return stripped if stripped else None


def _safe_strip(text: str) -> str:
    try:
        return strip_string(text)
    except Exception:  # pragma: no cover - defensive
        return text


def compute_score(solution_str: str, ground_truth: str) -> float:
    """verl-compatible entry point: 1.0 iff the answers are equivalent."""
    candidate = extract_answer(solution_str)
    expected = extract_answer(ground_truth) if isinstance(ground_truth, str) else None
    if candidate is None or expected is None:
        return 0.0
    return 1.0 if is_equiv(candidate, expected) else 0.0


# ---------------------------------------------------------------------------
# reward entry point
# ---------------------------------------------------------------------------

def reward(task: dict, trajectory: dict) -> dict:
    """Pure reward for one ``math_ci`` task/trajectory pair."""
    predicted_raw = base.final_answer(trajectory)
    expected_raw = base.reference(task).get("final_answer")
    info = {
        "predicted_raw": predicted_raw,
        "expected_raw": expected_raw if isinstance(expected_raw, str) else None,
        "used_boxed": bool(
            isinstance(predicted_raw, str) and last_boxed_only_string(predicted_raw) is not None
        ),
        "n_tool_calls": len(base.all_calls(trajectory)),
    }
    if predicted_raw is None:
        info["reason"] = "missing_prediction"
        return base.make_result(0.0, info, task_type=TASK_TYPE)
    if not isinstance(expected_raw, str) or not expected_raw.strip():
        info["reason"] = "missing_reference_answer"
        return base.make_result(0.0, info, task_type=TASK_TYPE)

    predicted = extract_answer(predicted_raw)
    expected = extract_answer(expected_raw)
    if predicted is None or expected is None:
        info["reason"] = "empty_extracted_answer"
        return base.make_result(0.0, info, task_type=TASK_TYPE)

    matched = is_equiv(predicted, expected)
    info.update(
        {
            "extracted_answer": predicted,
            "expected_answer": expected,
            "normalized_prediction": _safe_strip(predicted),
            "normalized_expected": _safe_strip(expected),
            "matched": matched,
        }
    )
    score = MAX_SCORE if matched else 0.0
    return base.make_result(score, info, task_type=TASK_TYPE)
