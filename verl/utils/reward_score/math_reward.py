# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

import re
from typing import Optional

import sympy
from latex2sympy2 import latex2sympy


def is_math_equal(expr1: str, expr2: str) -> bool:
    try:
        e1 = latex2sympy(expr1)
        e2 = latex2sympy(expr2)
        return sympy.simplify(e1 - e2) == 0
    except Exception:
        return False

def is_valid_ground_truth(ground_truth) -> bool:
    """Return False for corrupted labels such as '} )' or '] )'."""
    if ground_truth is None:
        return False
    gt = str(ground_truth).strip()
    if not gt:
        return False
    if _is_punctuation_only_fragment(gt):
        return False
    if _is_latex_env_noise_answer(gt):
        return False
    if len(gt) <= 2 and gt in ("}", ")", "]", "(", "{"):
        return False

    normalized = strip_string(gt)
    if not normalized:
        return False
    if _is_punctuation_only_fragment(normalized):
        return False
    if _is_latex_env_noise_answer(normalized):
        return False
    return _is_valid_normalized_answer(normalized, gt)


_ANSWER_TAIL_CHARS = 500


def _get_answer_tail(solution_str: str) -> str:
    text = solution_str
    for pattern in (
        r"(?is)<\s*response\s*>\s*(.*)$",
        r"(?is)\bresponse\s*:\s*(.*)$",
        r"(?is)\bfinal\s+answer\s*:\s*(.*)$",
    ):
        match = re.search(pattern, text)
        if match:
            text = match.group(1)
            break
    if len(text) > _ANSWER_TAIL_CHARS:
        text = text[-_ANSWER_TAIL_CHARS :]
    return text


def extract_answer_from_response(solution_str: str, ground_truth: str) -> Optional[str]:
    """Extract the model answer from boxed content or response-tail heuristics."""
    boxed = last_boxed_only_string(solution_str)
    if boxed is not None:
        try:
            return remove_boxed(boxed)
        except Exception:
            pass

    gt = str(ground_truth).strip()
    tail = _get_answer_tail(solution_str)

    if re.match(r"^\(?[A-I]\)?$", gt, re.IGNORECASE):
        for pattern in (
            r"\\boxed\{\\text\{([A-I])\}\}",
            r"\\boxed\{([A-I])\}",
            r"\b(?:answer|choice)\s*(?:is|:)?\s*\(?([A-I])\)?",
            r"\b([A-I])\s*[\.\)]\s*$",
            r"\b([A-I])\s*$",
        ):
            matches = re.findall(pattern, tail, re.IGNORECASE)
            if matches:
                return matches[-1].upper()

    for word in ("True", "False", "Yes", "No"):
        if gt.lower() == word.lower():
            matches = re.findall(rf"\b({word})\b", tail, re.IGNORECASE)
            if matches:
                return matches[-1]

    gt_normalized = strip_string(gt)
    if re.fullmatch(r"-?\d+\.?\d*", gt_normalized) or re.fullmatch(r"-?\d+\.?\d*%?", gt_normalized):
        numbers = re.findall(r"-?\d+\.?\d*", tail.replace(",", ""))
        if numbers:
            return numbers[-1]

    if re.search(r"\\frac|\\sqrt|\^|x", gt_normalized):
        math_blocks = re.findall(r"\$([^$]+)\$", tail)
        if math_blocks:
            return math_blocks[-1].strip()

    return None


def compute_score(solution_str, ground_truth) -> float:
    if not is_valid_ground_truth(ground_truth):
        return 0.0

    try:
        answer = extract_answer_from_response(solution_str, ground_truth)
        if answer is not None and is_equiv(answer, ground_truth):
            return 1.0
    except Exception:
        pass

    return 0.0


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
_LATEX_ENV_NAMES = frozenset(
    {
        "pmatrix",
        "bmatrix",
        "Bmatrix",
        "vmatrix",
        "Vmatrix",
        "matrix",
        "smallmatrix",
        "array",
        "subarray",
        "cases",
        "split",
        "aligned",
        "align",
        "gather",
        "multline",
        "eqnarray",
        "equation",
    }
)

_MATH_ENV_PATTERN = re.compile(
    r"\\begin\{(pmatrix|bmatrix|Bmatrix|vmatrix|Vmatrix|matrix|smallmatrix|cases|array|subarray|split)\*?\}",
    re.IGNORECASE,
)


def _contains_math_environment(s: str) -> bool:
    return _MATH_ENV_PATTERN.search(s) is not None


def _is_latex_env_noise_answer(normalized: str) -> bool:
    words = re.findall(r"[a-z]+", normalized.lower())
    if not words:
        return False
    return all(word in _LATEX_ENV_NAMES for word in words)


def _is_punctuation_only_fragment(s: str) -> bool:
    """True when the string has no meaningful alphanumeric answer."""
    if s is None:
        return True
    compact = re.sub(r"\s+", "", str(s))
    if not compact:
        return True
    if re.fullmatch(r"[\]\}\)>(\[\{\\.\\,;:+*/_\-]+", compact):
        return True
    return len(re.findall(r"[a-zA-Z0-9]", compact)) == 0


def _normalize_word_list(s: str) -> str:
    """Normalize comma/space-separated word lists and aligned LaTeX enumerations."""
    if not s:
        return ""
    s = s.lower()
    s = re.sub(
        r"\\begin\{(aligned|align|array|gather|equation)\*?\}(\[[^\]]*\])?",
        " ",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"\\end\{(aligned|align|array|gather|equation)\*?\}", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = s.replace("&", " ")
    s = s.replace("\\\\", "\n")
    s = re.sub(r"[{}()\[\];]", " ", s)
    s = s.replace(",", " ")
    words = re.findall(r"[a-z][a-z0-9]*", s)
    return " ".join(words)


def _word_lists_equivalent(str1: str, str2: str) -> bool:
    w1 = _normalize_word_list(str1).split()
    w2 = _normalize_word_list(str2).split()
    if len(w1) < 3 or len(w2) < 3:
        return False
    return sorted(w1) == sorted(w2)


def _looks_like_word_list(s: str) -> bool:
    if not s:
        return False
    if _contains_math_environment(s):
        return False
    if re.search(r"\\frac|\\sqrt|\\left|\\right", s):
        return False
    if re.search(r"\\begin\{(aligned|align|gather)", s, re.IGNORECASE):
        return True
    if re.search(r"\\text\{[^}]*,[^}]*\}", s):
        return True
    words = re.findall(r"[a-z]{3,}", s.lower())
    if len(words) >= 3 and ("," in s or "&" in s or "\\\\" in s):
        return True
    if len(words) >= 3 and not re.search(r"\\frac|\\sqrt|\\boxed", s):
        return True
    return False


def _is_valid_normalized_answer(normalized: str, raw_content: str) -> bool:
    if not _is_valid_boxed_content(raw_content):
        return False
    if not normalized or not normalized.strip():
        return False

    alnum_chars = re.findall(r"[a-zA-Z0-9]", normalized)
    if not alnum_chars:
        return False

    if re.match(r"^[\s\(\)\[\]\{\}>\.\\,;:+*/-]+$", normalized):
        return False

    if _is_latex_env_noise_answer(normalized):
        return False

    if len(alnum_chars) < 2:
        stripped = normalized.strip()
        if re.search(r"[\]\}>)(]", stripped) and not re.match(r"^[a-zA-Z0-9]$", stripped):
            return False
        punct_chars = re.sub(r"[a-zA-Z0-9]", "", stripped)
        if len(punct_chars) >= len(alnum_chars):
            return False

    return True


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        if _looks_like_word_list(str1) or _looks_like_word_list(str2):
            if _word_lists_equivalent(str1, str2):
                return True

        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        if ss1 == ss2:
            return True

        if _word_lists_equivalent(str1, str2):
            return True

        return is_math_equal(ss1, ss2)
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if s.startswith("\\boxed "):
        left = "\\boxed "
        return s[len(left) :].strip()

    for left in ("\\boxed{", "\\fbox{"):
        if s.startswith(left):
            assert s[-1] == "}"
            return s[len(left) : -1]

    raise ValueError(f"Unsupported boxed format: {s[:20]!r}")


def _is_valid_boxed_content(content: str) -> bool:
    """Reject fragments produced by malformed or truncated \\boxed{...} extraction."""
    if content is None:
        return False
    stripped = content.strip()
    if not stripped:
        return False
    if _is_punctuation_only_fragment(stripped):
        return False
    if re.match(r"^[\s\]\}]+$", stripped):
        return False
    if re.match(r"^\\right[\]\)]*\s*\}?\s*$", stripped):
        return False
    return True


def _is_valid_boxed_answer(content: str) -> bool:
    """Validate raw boxed content and its normalized form."""
    if not _is_valid_boxed_content(content):
        return False
    if _looks_like_word_list(content):
        word_list = _normalize_word_list(content)
        return bool(word_list) and _is_valid_normalized_answer(word_list, content)
    normalized = strip_string(content)
    return _is_valid_normalized_answer(normalized, content)


def _match_braced_segment(string: str, open_brace_idx: int):
    """Return string[open_brace_idx : closing_brace_idx + 1] for a balanced {...} segment."""
    if open_brace_idx < 0 or open_brace_idx >= len(string) or string[open_brace_idx] != "{":
        return None

    num_left_braces_open = 0
    for i in range(open_brace_idx, len(string)):
        if string[i] == "{":
            num_left_braces_open += 1
        elif string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                return string[open_brace_idx : i + 1]
    return None


def _boxed_command_at(string: str, idx: int):
    """Build a \\boxed{...} or \\boxed ... string starting at idx pointing to \\boxed."""
    if string.startswith("\\boxed{", idx):
        braced = _match_braced_segment(string, idx + len("\\boxed{") - 1)
        if braced is None:
            return None
        return string[idx : idx + len("\\boxed{") - 1 + len(braced)]

    if string.startswith("\\boxed ", idx):
        brace_idx = string.find("{", idx)
        if brace_idx < 0:
            return None
        braced = _match_braced_segment(string, brace_idx)
        if braced is None:
            return None
        return string[idx : brace_idx + len(braced)]

    if string.startswith("\\fbox{", idx):
        braced = _match_braced_segment(string, idx + len("\\fbox{") - 1)
        if braced is None:
            return None
        return string[idx : idx + len("\\fbox{") - 1 + len(braced)]

    return None


def last_boxed_only_string(string):
    """Extract the last valid \\boxed{...} (or \\fbox{...}) expression in the string."""
    search_end = len(string)
    while search_end > 0:
        idx = string.rfind("\\boxed{", 0, search_end)
        if idx >= 0:
            boxed = _boxed_command_at(string, idx)
            if boxed is not None:
                try:
                    content = remove_boxed(boxed)
                except Exception:
                    content = None
                if _is_valid_boxed_answer(content):
                    return boxed
            search_end = idx
            continue

        idx = string.rfind("\\boxed ", 0, search_end)
        if idx >= 0:
            boxed = _boxed_command_at(string, idx)
            if boxed is not None:
                try:
                    content = remove_boxed(boxed)
                except Exception:
                    content = None
                if _is_valid_boxed_answer(content):
                    return boxed
            search_end = idx
            continue

        break

    search_end = len(string)
    while search_end > 0:
        idx = string.rfind("\\fbox{", 0, search_end)
        if idx < 0:
            break
        boxed = _boxed_command_at(string, idx)
        if boxed is not None:
            try:
                content = remove_boxed(boxed)
            except Exception:
                content = None
            if _is_valid_boxed_answer(content):
                return boxed
        search_end = idx

    return None


def _finalize_extracted_answer(content: str) -> Optional[str]:
    if not _is_valid_boxed_content(content):
        return None
    if _looks_like_word_list(content):
        word_list = _normalize_word_list(content)
        if word_list and _is_valid_normalized_answer(word_list, content):
            return " ".join(sorted(word_list.split()))
    normalized = strip_string(content)
    if not _is_valid_normalized_answer(normalized, content):
        return None
    return normalized


def extract_boxed_answer(string: str):
    """Extract and normalize the answer inside the last valid \\boxed{...} in a string."""
    boxed = last_boxed_only_string(string)
    if boxed is None:
        return None
    try:
        content = remove_boxed(boxed)
    except Exception:
        return None
    return _finalize_extracted_answer(content)


def extract_math_answer(solution_str: str):
    """Extract a MATH reference answer from a model solution string."""
    answer = extract_boxed_answer(solution_str)
    if answer is not None:
        return answer

    search_end = len(solution_str)
    while search_end > 0:
        idx = solution_str.rfind("\\boxed{", 0, search_end)
        if idx < 0:
            break
        boxed = _boxed_command_at(solution_str, idx)
        if boxed is not None:
            try:
                content = remove_boxed(boxed)
            except Exception:
                content = None
            if content is not None:
                answer = _finalize_extracted_answer(content)
                if answer is not None:
                    return answer
        search_end = idx

    patterns = [
        r"(?i)(?:the answer is|answer is|answer:)\s*([^\n$]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, solution_str)
        if matches:
            candidate = matches[-1].strip()
            answer = _finalize_extracted_answer(candidate)
            if answer is not None:
                return answer
    return None


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except Exception:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")
    string = string.replace("\\,", "")
    string = string.replace("\\;", "")
    string = string.replace("\\:", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    string = re.sub(r'\\text\{([^{}]*)\}', r'\1', string)
    string = re.sub(r'\(([A-Za-z])\)', r'\1', string)

    # remove units (on the right)
    string = remove_right_units(string)

    string = re.sub(r'\\text\{([^{}]*)\}', r'\1', string)
    string = re.sub(r'\\left\(', r'(', string)
    string = re.sub(r'\\right\)', r')', string)

    # remove percentage
    string = string.replace("\\\\%", "")
    string = string.replace("\\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")
    string = re.sub(r'\((\d+[\+\-]\d+i?)\)', r'\1', string)

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1).
    # Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)
    # \frac{40}7 -> \frac{40}{7} (missing braces around denominator)
    string = re.sub(r"\\frac\{([^}]+)\}(-?\d+)", r"\\frac{\1}{\2}", string)
    string = re.sub(r'(\d+)/(\d+)', r'\\frac{\1}{\2}', string)
    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    string = string.replace(" ", "")
    if not re.match(r'^[a-zA-Z]+$', string):
       string = re.sub(r'[a-z]{2,}\^?\d*$', '', string)

    string = string.replace('{,}', ',')

    string = re.sub(r'(?<=\d),(?=\d{3}(?!\d))', '', string)

    string = re.sub(r'\\(?:mbox)\{.*', '', string)

    string = re.sub(r'_[{]?\d+[}]?', '', string)
    return string
