"""Rule-based verifier for Qwen3 tool-use RL — Dim 2 + Dim 3 + label matching.

Ported from slime ``examples/tool_rl/reward/verifier.py`` (rule-based parts
only; the LLM-judge/RM mode is intentionally not migrated).

Parses Qwen's XML tool call format:

.. code-block:: xml

    <tool_call>
    <function=function_name>
    <parameter=param_name>
    value
    </parameter>
    </function>
    </tool_call>

Dimensions
----------
- **Dim 2 (weight 0.20)**: Format compliance — rule verifier
  Scoring:
    1. All tool_calls after a single, non-empty <think> → +0.6
    2. Each tool_call preceded by a non-empty <think> → +0.4 × 1/N
    3. No calls → 1.0 when no tools are expected (label says no tools
       needed, or no tools are defined), otherwise 0.0

- **Dim 3 (weight 0.20)**: Tool call format correctness vs ground-truth label
  Scoring (N = number of label tool calls):
    1. Each label call completely matched by an output call (same tool name,
       exactly the label's param names, compatible param types) → +1/N
    2. Label has no tool calls → 1.0 (no detection)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ============================================================================
# Regex patterns — Qwen XML format
# ============================================================================

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

# A raw ``<think>`` opener — used to detect an unclosed think block.
_OPENING_RE = re.compile(r"<think>", re.IGNORECASE)


def _check_strict_format(text: str) -> bool:
    """Validate the top-level layout of a response's think block.

    Rules:
      1. ``<think>`` tags must be paired — an unclosed ``<think>`` opener
         with no ``</think>`` is invalid.
      2. More than one complete ``<think>...</think>`` block is invalid.
      3. Something (response text / tool_call) must follow the think block —
         emitting reasoning and halting is the "think-then-stop" collapse.
      4. Otherwise valid: no think block at all, or exactly one complete
         ``<think>...</think>`` block.
    """
    matches = list(_THINK_RE.finditer(text))
    if len(_OPENING_RE.findall(text)) != len(matches):
        return False
    if len(matches) > 1:
        return False
    if len(matches) == 1:
        return text[matches[0].end():].strip() != ""
    return True


# Qwen XML tool call: <tool_call>...<function=NAME>...</function>...</tool_call>
_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE,
)
_FUNCTION_NAME_RE = re.compile(r"<function=(\w[\w.]*)>")
_PARAM_RE = re.compile(
    r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", re.DOTALL,
)
# Inline JSON style: <tool_call>\n"name": NAME, "arguments": {...}\n</tool_call>
_INLINE_CALL_RE = re.compile(
    r'^\s*"name"\s*:\s*"([\w.]*)",\s*"arguments"\s*:\s*(\{.*\})\s*$',
    re.DOTALL,
)


def _extract_json_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract ``{"name": …, "arguments": {…}}`` objects from text.

    Uses bracket-depth tracking so nested parameter values are handled.
    """
    results: list[dict[str, Any]] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    obj = json.loads(text[start : i + 1])
                except (json.JSONDecodeError, TypeError):
                    start = -1
                    continue
                if isinstance(obj, dict) and "name" in obj:
                    results.append(obj)
                start = -1
    return results


# ============================================================================
# Tool call parsing — Qwen XML format
# ============================================================================


def parse_qwen_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse Qwen XML tool calls from text.

    Returns:
        List of ``{"name": str, "arguments": dict}``.
    """
    calls: list[dict[str, Any]] = []

    for tc_match in _TOOL_CALL_BLOCK_RE.finditer(text):
        block = tc_match.group(1)
        func_match = _FUNCTION_NAME_RE.search(block)
        inline_match = _INLINE_CALL_RE.search(block)
        if not func_match and not inline_match:
            continue
        func_name = func_match.group(1) if func_match else inline_match.group(1)

        args: dict[str, Any] = {}
        for pm in _PARAM_RE.finditer(block):
            pname = pm.group(1)
            pval = pm.group(2).strip()
            try:
                pval = json.loads(pval)
            except (json.JSONDecodeError, TypeError):
                pass
            args[pname] = pval

        if inline_match and not args:
            try:
                args = json.loads(inline_match.group(2))
            except (json.JSONDecodeError, TypeError):
                args = {}

        calls.append({"name": func_name, "arguments": args})

    # Fallback: JSON format
    if not calls:
        for obj in _extract_json_tool_calls(text):
            if obj not in calls:
                calls.append(obj)

    return calls


# ============================================================================
# Dim 2 — Format Compliance (weight 0.20)
# ============================================================================


def check_format_compliance(
    trajectory: list[dict[str, Any]],
    *,
    available_tools: list[dict[str, Any]] | None = None,
    expects_no_tools: bool = False,
) -> float:
    """Check <think>...<tool_call> format compliance.

    Scoring:
      1. All tool_calls after a single, non-empty <think> block → +0.6
      2. Each tool_call preceded by a non-empty <think> → +0.4 × count/N
      3. No calls → 1.0 when no tools are expected, otherwise 0.0

    Args:
        trajectory: Normalized trajectory.
        available_tools: Tool definitions. If non-empty and no calls, score 0.
        expects_no_tools: Label says no tools are needed for this task.

    Returns:
        Score in [0.0, 1.0].
    """
    all_text = _get_agent_text(trajectory)

    # Strict gate: well-formed thinking→response layout required.
    if not _check_strict_format(all_text):
        logger.debug("[dim2] Strict thinking→response format broken → 0.0")
        return 0.0

    n_calls = len(_xml_tool_call_spans(all_text))
    n_calls += len(_find_json_tool_call_spans(all_text))

    if n_calls == 0:
        if expects_no_tools or not available_tools:
            reason = "label says no tools needed" if expects_no_tools else "no tools defined"
            logger.debug("[dim2] No tool calls and %s → 1.0", reason)
            return 1.0
        logger.debug("[dim2] No tool calls but tools available → 0.0")
        return 0.0

    score = 0.0

    # Rule 1: All tool calls after a single, non-empty think → +0.6
    if _all_calls_after_think(all_text):
        score += 0.6
        logger.debug("[dim2] All calls after a single non-empty think → +0.6")
    else:
        logger.debug("[dim2] No valid single think (empty/missing/repeated) → no +0.6")

    # Rule 2: Each tool call preceded by a non-empty <think> → +0.4 × count/N
    preceded = _count_preceded_by_think(all_text, n_calls)
    if preceded > 0:
        bonus = 0.4 * preceded / n_calls
        score += bonus
        logger.debug("[dim2] %d/%d calls preceded by think → +%.3f", preceded, n_calls, bonus)

    return max(0.0, min(1.0, score))


def _get_agent_text(trajectory: list[dict[str, Any]]) -> str:
    parts = [r.get("text", "") for r in trajectory if r.get("type") != "observation"]
    return "\n".join(parts)


def _find_json_tool_call_spans(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` spans of ``{"name": …}`` objects in *text*."""
    spans: list[tuple[int, int]] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append((start, i + 1))
                start = -1
    return spans


def _xml_tool_call_spans(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` spans of *valid* ``<tool_call>`` blocks.

    A block is valid only when it declares a function via ``<function=...>``.
    """
    spans: list[tuple[int, int]] = []
    for m in _TOOL_CALL_BLOCK_RE.finditer(text):
        if _FUNCTION_NAME_RE.search(m.group(1)) or _INLINE_CALL_RE.search(m.group(1)):
            spans.append((m.start(), m.end()))
    return spans


def _all_calls_after_think(text: str) -> bool:
    """Check all tool calls are after a single, non-empty <think> block."""
    matches = list(_THINK_RE.finditer(text))
    if len(matches) != 1 or not matches[0].group(1).strip():
        return False
    think_end = matches[0].end()
    for start, _end in _xml_tool_call_spans(text):
        if start < think_end:
            return False
    for start, _end in _find_json_tool_call_spans(text):
        if start < think_end:
            return False
    return True


def _count_preceded_by_think(text: str, total: int) -> int:
    """Count how many tool calls have a non-empty <think> before them."""
    think_ends = [
        m.end() for m in _THINK_RE.finditer(text) if m.group(1).strip() != ""
    ]
    if not think_ends:
        return 0

    call_starts = [start for start, end in _xml_tool_call_spans(text)]
    call_starts.extend(start for start, end in _find_json_tool_call_spans(text))
    call_starts.sort()

    count = 0
    ti = 0
    for cs in call_starts:
        while ti < len(think_ends) - 1 and think_ends[ti + 1] < cs:
            ti += 1
        if think_ends[ti] < cs:
            count += 1
    return count


# ============================================================================
# Dim 3 — Tool Call Format Correctness (weight 0.20)
# ============================================================================


def check_tool_call_format(
    trajectory: list[dict[str, Any]],
    label_calls: list[dict[str, Any]] | None = None,
) -> float:
    """Check model tool calls against ground-truth label calls (Dim 3).

    Only param **names** and param **types** are checked — values are ignored.

    Two cases:
      1. Label provides tool calls: score = ``matched / len(label_calls)``.
      2. Label provides no tool calls: score 1.0 (no detection).

    Args:
        trajectory: Normalized trajectory.
        label_calls: Ground truth tool calls (``[{"name": …, "arguments": {…}}]``).

    Returns:
        Score in [0.0, 1.0].
    """
    all_text = _get_agent_text(trajectory)

    if not label_calls:
        logger.debug("[dim3] No label tool calls → 1.0 (no detection)")
        return 1.0

    if not _check_strict_format(all_text):
        logger.debug("[dim3] Strict thinking→response format broken → 0.0")
        return 0.0

    output_calls = parse_qwen_tool_calls(all_text)
    n = len(label_calls)
    matched = _count_label_matches(output_calls, label_calls)
    score = matched / n

    logger.debug("[dim3] label=%d matched=%d → %.3f", n, matched, score)
    return score


def _values_types_match(v1: Any, v2: Any) -> bool:
    """Check that two values have a compatible primitive type."""
    v1 = _unwrap_json_string(v1)
    v2 = _unwrap_json_string(v2)
    if v1 is None or v2 is None:
        return v1 is None and v2 is None
    if isinstance(v1, bool) != isinstance(v2, bool):
        return False
    if isinstance(v1, bool):
        return v1 == v2
    if isinstance(v1, (int, float)):
        return isinstance(v2, (int, float))
    if isinstance(v1, list):
        return isinstance(v2, list)
    if isinstance(v1, dict):
        return isinstance(v2, dict)
    return isinstance(v1, str) and isinstance(v2, str)


def _call_completely_matches(
    output_call: dict[str, Any],
    label_call: dict[str, Any],
) -> bool:
    """A call completely matches a label call when the tool name agrees, the
    output provides exactly the label's parameter names, and every parameter
    value has a compatible type. Values themselves are ignored.
    """
    if output_call.get("name", "") != label_call.get("name", ""):
        return False
    o_args = output_call.get("arguments", {}) or {}
    l_args = label_call.get("arguments", {}) or {}
    if set(o_args.keys()) != set(l_args.keys()):
        return False
    return all(_values_types_match(o_args[k], l_args[k]) for k in l_args)


def _count_label_matches(
    output_calls: list[dict[str, Any]],
    label_calls: list[dict[str, Any]],
) -> int:
    """Count label calls completely matched by an output call (one-to-one)."""
    used: set[int] = set()
    matched = 0
    for label_call in label_calls:
        for oi, out_call in enumerate(output_calls):
            if oi in used:
                continue
            if _call_completely_matches(out_call, label_call):
                used.add(oi)
                matched += 1
                break
    return matched


_TYPE_MAP = {
    "string": str, "str": str,
    "integer": int, "int": int,
    "number": (int, float), "float": float,
    "boolean": bool, "bool": bool,
    "array": list, "list": list,
    "object": dict, "dict": dict,
}


# ============================================================================
# Tool call correctness — per-call verdict (for token-level loss masking)
# ============================================================================


def _build_tool_index(
    available_tools: list[dict[str, Any]] | None,
) -> tuple[set[str], dict[str, dict[str, dict]]]:
    """Build tool name set and param index from available_tools."""
    tool_names: set[str] = set()
    tool_params: dict[str, dict[str, dict]] = {}
    for tool in (available_tools or []):
        name = tool.get("name", "")
        if not name:
            continue
        tool_names.add(name)
        params = tool.get("parameters", {})
        props = params.get("properties", params) if isinstance(params, dict) else {}
        if isinstance(props, dict):
            if props and isinstance(next(iter(props.values()), None), dict):
                tool_params[name] = props
    return tool_names, tool_params


def _is_tool_call_correct(
    call: dict[str, Any],
    tool_names: set[str],
    tool_params: dict[str, dict[str, dict]],
) -> bool:
    """Check whether a single parsed tool call is fully correct.

    A tool call is correct when ALL of:
    1. Function name exists in ``tool_names``
    2. All parameter names are declared for that function
    3. No extra/undeclared parameter names
    4. All parameter values match declared types

    If ``tool_names`` is empty (no tool definitions), returns ``True``.
    """
    cname = call.get("name", "")
    cargs = call.get("arguments", {}) or {}

    if not tool_names:
        return True

    if not cname or cname not in tool_names:
        return False

    if cname not in tool_params:
        return not cargs

    declared = tool_params[cname]
    declared_names = set(declared.keys())

    if not declared_names:
        return not cargs

    if not cargs:
        return False

    for k in cargs:
        if k not in declared_names:
            return False

    for k, v in cargs.items():
        if k not in declared:
            continue
        dtype = declared[k].get("type", "")
        expected = _TYPE_MAP.get(dtype.lower()) if dtype else None
        if expected is not None and not isinstance(v, expected):
            return False

    return True


def get_incorrect_tool_call_spans(
    text: str,
    available_tools: list[dict[str, Any]] | None = None,
) -> list[tuple[int, int]]:
    """Return ``(start_char, end_char)`` spans of incorrect tool call blocks.

    Args:
        text: Raw assistant response containing zero or more tool call blocks.
        available_tools: Tool definitions. If empty/None, all calls are
            treated as correct.

    Returns:
        List of ``(start_char, end_char)`` tuples for incorrect tool call
        blocks.
    """
    tool_names, tool_params = _build_tool_index(available_tools)

    incorrect_spans: list[tuple[int, int]] = []

    for match in _TOOL_CALL_BLOCK_RE.finditer(text):
        block_text = match.group(1)
        func_match = _FUNCTION_NAME_RE.search(block_text)

        call: dict[str, Any] = {"name": "", "arguments": {}}
        if func_match:
            call["name"] = func_match.group(1)

        for pm in _PARAM_RE.finditer(block_text):
            pname = pm.group(1)
            pval = pm.group(2).strip()
            try:
                pval = json.loads(pval)
            except (json.JSONDecodeError, TypeError):
                pass
            call["arguments"][pname] = pval

        if not _is_tool_call_correct(call, tool_names, tool_params):
            incorrect_spans.append((match.start(), match.end()))
            logger.debug(
                "[mask] Incorrect tool call: name=%r span=(%d, %d)",
                call.get("name"), match.start(), match.end(),
            )

    return incorrect_spans


# ============================================================================
# Label-based tool call correctness matching
# ============================================================================


def _values_match(v1: Any, v2: Any) -> bool:
    """Check value equality with some fuzziness for strings."""
    v1 = _unwrap_json_string(v1)
    v2 = _unwrap_json_string(v2)

    if v1 is v2 or type(v1) == type(v2) and v1 == v2:
        return True
    if v1 is None or v2 is None:
        return False
    if isinstance(v1, bool) != isinstance(v2, bool):
        return False
    if isinstance(v1, str) and isinstance(v2, str):
        return v1.strip().lower() == v2.strip().lower()
    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
        return abs(float(v1) - float(v2)) < 1e-6
    if isinstance(v1, dict) and isinstance(v2, dict):
        return v1.keys() == v2.keys() and all(_values_match(v1[k], v2[k]) for k in v1)
    if isinstance(v1, list) and isinstance(v2, list) and len(v1) == len(v2):
        return all(_values_match(a, b) for a, b in zip(v1, v2))
    return False


def _unwrap_json_string(value: Any) -> Any:
    """Decode a string that wraps a JSON array/object into the structured value."""
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
        if isinstance(decoded, (dict, list)):
            return decoded
    return value


def _param_content_score(
    label_args: dict[str, Any],
    output_args: dict[str, Any],
) -> float:
    """Score how well output parameter values match ground truth.

    Returns a score in [0, 1]:
      - Fraction of label params whose values match the output.
      - Extra output params incur a 50 % penalty on the remainder.
    """
    if not label_args and not output_args:
        return 1.0
    if not label_args:
        return 0.0

    correct = sum(
        1 for k, lv in label_args.items()
        if k in output_args and _values_match(lv, output_args[k])
    )
    extra = sum(1 for k in output_args if k not in label_args)
    penalty = 0.5 * extra / max(len(label_args) + extra, 1)
    return max(0.0, correct / len(label_args) - penalty)


def _format_call(call: dict[str, Any]) -> str:
    """Format a tool call as ``name(key=val, ...)`` for logging."""
    name = call.get("name", "?")
    args = call.get("arguments", {}) or {}
    if args:
        params = ", ".join(
            "%s=%s" % (k, json.dumps(v, ensure_ascii=False))
            for k, v in args.items()
        )
        return "%s(%s)" % (name, params)
    return name + "()"


def undeclared_tool_penalty(
    output_calls: list[dict[str, Any]],
    available_tools: list[dict[str, Any]] | None,
) -> float:
    """Return the Dim 1 penalty for calling tools not declared in the prompt.

    Each call whose name is not among ``available_tools`` incurs ``-0.1``.
    """
    declared = {t.get("name", "") for t in (available_tools or []) if t.get("name")}
    if not declared:
        return 0.0
    undeclared = sum(1 for c in output_calls if c.get("name", "") not in declared)
    return 0.1 * undeclared


def _guess_penalty(n_emitted: int) -> float:
    """Penalty for blind tool guessing: ``-0.1`` per call, capped at ``-1.0``."""
    return min(0.1 * n_emitted, 1.0)


def match_tool_calls_against_label(
    output_calls: list[dict[str, Any]],
    label_calls: list[dict[str, Any]],
) -> tuple[float, float]:
    """Order-independent matching of tool calls against ground truth labels.

    Jaccard-like normalisation so both missed tools and spurious calls are
    penalised::

        name_score  = matched / (M + N - matched)
        param_score = sum(matched_pair_scores) / (M + N - matched)

    If **no** label call is matched, both scores are set to
    ``-min(0.1 * M, 1.0)`` (blind-guessing penalty). When both are empty,
    the score is ``(1.0, 1.0)``.

    Returns:
        Tuple ``(name_score, param_score)``.
    """
    if output_calls:
        logger.info("[tool_rl] Model calls (%d):", len(output_calls))
        for i, c in enumerate(output_calls):
            logger.info("[tool_rl]   [%d] %s", i + 1, _format_call(c))
    else:
        logger.info("[tool_rl] Model calls: (none)")

    if label_calls:
        logger.info("[tool_rl] Label calls (%d):", len(label_calls))
        for i, c in enumerate(label_calls):
            logger.info("[tool_rl]   [%d] %s", i + 1, _format_call(c))
    else:
        logger.info("[tool_rl] Label calls: (none / no tools needed)")

    if not label_calls:
        if not output_calls:
            logger.info("[tool_rl] Match: both empty → 1.0")
            return (1.0, 1.0)
        penalty = _guess_penalty(len(output_calls))
        logger.info(
            "[tool_rl] Mismatch: label expects no tools, but model called %d tool(s) → penalty %.2f",
            len(output_calls), -penalty,
        )
        return (-penalty, -penalty)

    matched_indices: set[int] = set()
    pair_param_scores: list[float] = []

    for l_call in label_calls:
        l_name = l_call.get("name", "")
        l_args = l_call.get("arguments", {}) or {}
        best_param_score = -1.0
        best_idx = -1

        for oi, o_call in enumerate(output_calls):
            if oi in matched_indices:
                continue
            if o_call.get("name", "") != l_name:
                continue
            o_args = o_call.get("arguments", {}) or {}
            ps = _param_content_score(l_args, o_args)
            if ps > best_param_score:
                best_param_score = ps
                best_idx = oi

        if best_idx >= 0:
            matched_indices.add(best_idx)
            pair_param_scores.append(best_param_score)

    matched = len(pair_param_scores)

    if matched == 0:
        penalty = _guess_penalty(len(output_calls))
        logger.info(
            "[tool_rl] No label call matched (%d output vs %d label) → penalty %.2f",
            len(output_calls), len(label_calls), -penalty,
        )
        return (-penalty, -penalty)

    m = len(output_calls)
    n = len(label_calls)
    union = m + n - matched

    if union == 0:
        return (1.0, 1.0)

    name_score = matched / union
    param_score = sum(pair_param_scores) / union

    unmatched_output = [
        (i, _format_call(output_calls[i]))
        for i in range(m) if i not in matched_indices
    ]
    if unmatched_output:
        logger.info("[tool_rl] Unmatched output (%d):", len(unmatched_output))
        for idx, call_str in unmatched_output:
            logger.info("[tool_rl]   [#%d] %s", idx + 1, call_str)

    logger.info(
        "[tool_rl] Match result: name=%.3f param=%.3f (matched %d/%d label calls)",
        name_score, param_score, matched, n,
    )

    return (name_score, param_score)


def parse_ground_truth_calls(
    ground_truth: Any,
) -> list[dict[str, Any]]:
    """Normalise ground truth into ``[{"name": …, "arguments": {…}}]``.

    Handles:
      - ``list[dict]`` with "name"/"arguments" keys (canonical format).
      - A JSON string containing such a list.
      - A single dict (one tool call).
    """
    if not ground_truth:
        return []
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except (json.JSONDecodeError, TypeError):
            return []
    if isinstance(ground_truth, dict):
        if "name" in ground_truth:
            ground_truth = [ground_truth]
        else:
            return []
    if not isinstance(ground_truth, list):
        return []

    normalised: list[dict[str, Any]] = []
    for item in ground_truth:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("function", "")
        if not name:
            continue
        args = item.get("arguments") or item.get("parameters") or {}
        if not isinstance(args, dict):
            args = {}
        normalised.append({"name": str(name), "arguments": args})
    return normalised


def compute_verifier_scores(
    trajectory: list[dict[str, Any]],
    *,
    available_tools: list[dict[str, Any]] | None = None,
    label_calls: list[dict[str, Any]] | None = None,
    expects_no_tools: bool = False,
) -> dict[str, float]:
    """Compute Dim 2 + Dim 3 verifier scores.

    Returns:
        ``{"format_compliance": float, "tool_call_format": float}``.
    """
    return {
        "format_compliance": check_format_compliance(
            trajectory,
            available_tools=available_tools,
            expects_no_tools=expects_no_tools,
        ),
        "tool_call_format": check_tool_call_format(
            trajectory,
            label_calls=label_calls,
        ),
    }
