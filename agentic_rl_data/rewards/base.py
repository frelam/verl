"""Shared helpers for the ``agentic_rl_data`` reward functions.

Contract (plan §5, ``agentic_rl_data/schema.py``, ``CONTRACTS.md``)::

    reward(task: dict, trajectory: dict) -> {"score": float, "info": dict}

A reward is a **pure function**: no env access, no global mutable state,
no ``os.environ``, no clock, no randomness.  Everything it needs comes from
``task`` (ground truth) and the normalised ``trajectory``.

What this module provides
-------------------------
* defensive accessors for the schema fields;
* argument normalisation and "AST-ish" equality (:func:`values_equal`):
  JSON / Python literal strings are parsed before comparison, numbers use a
  small tolerance, strings compare case / whitespace-insensitively, booleans
  never compare equal to numbers;
* order-independent one-to-one call matching (:func:`match_calls`) via a
  maximum bipartite matching, so parallel calls may be permuted (plan §2/§5);
* structural format validation of tool calls against the declared tool
  schemas (:func:`format_report`) - the "format gate".  The raw
  ``<think>`` / ``<tool_call>`` layout belongs to the framework side and is
  not visible in the normalised trajectory, so "format" here means the
  *structure* the plan and schema can actually check: declared tool name,
  declared / required parameters with compatible types, and one
  ``tool_results`` entry per call;
* :func:`make_result`, which clamps the score into the frozen
  ``schema.SCORE_RANGE`` (``[0.0, 1.0]``, CONTRACTS C4), JSON-sanitises
  ``info`` and validates the payload through
  ``schema.validate_reward_result`` - so every reward family returns a
  conforming payload by construction.

Trajectory fields beyond the top-level ``schema`` contract (frozen v2, read
defensively; absence never raises, it only lowers the score and sets an
explicit ``info`` reason):

* ``trajectory["env_state"]`` - verbatim copy of the env's done-info
  (CONTRACTS C3).  The canonical keys are ``type``, ``impl``,
  ``goal_satisfied``, ``replay_ok``, ``failed_turn``, ``matched_turns``,
  ``gt_turns``, ``checks``, ``state``, ``rendered``, ``num_turns``,
  ``timeout``, ``code_tests`` (code_exec only), ``extras``.
* ``trajectory["env_state"]["code_tests"]`` - test summary of the last
  ``code_tests`` run (CONTRACTS C5):
  ``{"total": int, "passed": int, "failed": int, "all_passed": bool,
  "exit_code": int, "duration_s": float, "truncated": bool,
  "failures": [{"name": str, "message": str}]}``.  ``all_passed`` is
  authoritative.
* ``trajectory["env_state"]["extras"]["gt_returns_source"]`` - one of
  ``"config" | "reference" | "gt_turns" | "placeholder"``; ``"placeholder"``
  marks a degraded replay (:func:`degraded_replay`) and must **not** earn
  extra score (CONTRACTS C6).  The static ``task["meta"]["gt_returns_source"]``
  marker (schema v2.2) is honoured as well, so a task declared placeholder
  offline is treated as degraded even before an env runs.
"""

from __future__ import annotations

import ast
import json
import math
import re
import unicodedata
from collections.abc import Mapping
from typing import Any

try:  # normal package import (``python -m pytest`` from the repo root)
    from agentic_rl_data import schema
except ImportError:  # pragma: no cover - direct-file import fallback
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import schema  # type: ignore

__all__ = [
    "MISSING",
    "SCORE_MIN",
    "SCORE_MAX",
    "MAX_SCORE",
    "as_mapping",
    "as_list",
    "as_text",
    "iter_turns",
    "turn_calls",
    "turn_results",
    "normalize_call",
    "calls_equal",
    "call_repr",
    "all_calls",
    "call_groups",
    "nonempty_call_groups",
    "normalize_value",
    "normalize_text",
    "text_equals",
    "values_equal",
    "match_calls",
    "group_matches",
    "align_groups",
    "gt_call_groups",
    "flatten_gt_calls",
    "derive_turn_match",
    "env_state",
    "has_env_state",
    "env_extras",
    "degraded_replay",
    "reference",
    "final_answer",
    "final_answers",
    "last_assistant_content",
    "visible_reply",
    "is_empty_trajectory",
    "format_report",
    "format_ok",
    "clamp_score",
    "weighted_sum",
    "jsonable",
    "make_result",
]

MISSING = object()
SCORE_MIN, SCORE_MAX = schema.SCORE_RANGE
MAX_SCORE = SCORE_MAX

_WS_RE = re.compile(r"\s+")
_TRAILING_PUNCT = " \t\r\n.!?。！？"
_LITERAL_SIZE_LIMIT = 4096
_PARAM_SCHEMA_KEYS = frozenset(
    {"type", "required", "properties", "additionalProperties", "title", "description", "$schema"}
)


# ---------------------------------------------------------------------------
# defensive accessors
# ---------------------------------------------------------------------------

def as_mapping(value: Any) -> Mapping:
    """``value`` when it is a mapping, else an empty dict."""
    return value if isinstance(value, Mapping) else {}


def as_list(value: Any) -> list:
    """``value`` as a list when it is a list/tuple, else ``[]``."""
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def as_text(value: Any) -> str:
    """``value`` when it is a string, else ``""``."""
    return value if isinstance(value, str) else ""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_integral(value: Any) -> bool:
    """Integer JSON-schema check tolerant of integral floats (``3.0``), as the
    AST-ish value matcher treats them as equal."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and math.isfinite(value) and value.is_integer()


def iter_turns(trajectory: Any) -> list[Mapping]:
    """Valid turn mappings of a normalised trajectory."""
    turns = as_mapping(trajectory).get("turns")
    if not isinstance(turns, (list, tuple)):
        return []
    return [turn for turn in turns if isinstance(turn, Mapping)]


def turn_calls(turn: Any) -> list:
    """Raw ``assistant.tool_calls`` entries of one turn (may be malformed)."""
    if not isinstance(turn, Mapping):
        return []
    assistant = as_mapping(turn.get("assistant"))
    calls = assistant.get("tool_calls")
    if isinstance(calls, (list, tuple)):
        return list(calls)
    return []


def turn_results(turn: Any) -> list:
    """Raw ``tool_results`` entries of one turn."""
    if not isinstance(turn, Mapping):
        return []
    results = turn.get("tool_results")
    if isinstance(results, (list, tuple)):
        return list(results)
    return []


def normalize_call(call: Any) -> dict:
    """Normalise one tool call to ``{"name": str, "arguments": dict}``.

    Malformed entries become an empty-named call so they still count as
    emitted calls (never silently disappear) and never match ground truth.
    """
    mapping = as_mapping(call)
    name = mapping.get("name")
    name = name.strip() if isinstance(name, str) else ""
    arguments = mapping.get("arguments")
    if not isinstance(arguments, Mapping):
        arguments = {}
    return {"name": name, "arguments": dict(arguments)}


def calls_equal(left: Any, right: Any) -> bool:
    """True when two calls agree on the (exact) tool name and argument values."""
    left = normalize_call(left)
    right = normalize_call(right)
    if not left["name"] or not right["name"]:
        return False
    return left["name"] == right["name"] and values_equal(left["arguments"], right["arguments"])


def call_repr(call: Any, limit: int = 200) -> str:
    """Compact deterministic text form of a call for ``info`` diagnostics."""
    call = normalize_call(call)
    arguments = json.dumps(call["arguments"], ensure_ascii=False, sort_keys=True, default=repr)
    text = f"{call['name']}({arguments})" if call["name"] else f"<malformed>({arguments})"
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def all_calls(trajectory: Any) -> list[dict]:
    """Every normalised call of the trajectory, in turn order."""
    result: list[dict] = []
    for turn in iter_turns(trajectory):
        result.extend(normalize_call(call) for call in turn_calls(turn))
    return result


def call_groups(trajectory: Any) -> list[list[dict]]:
    """Per-turn groups of normalised calls (empty turn => empty group)."""
    return [
        [normalize_call(call) for call in turn_calls(turn)]
        for turn in iter_turns(trajectory)
    ]


def nonempty_call_groups(trajectory: Any) -> list[list[dict]]:
    """Per-turn call groups that actually contain calls."""
    return [group for group in call_groups(trajectory) if group]


# ---------------------------------------------------------------------------
# value normalisation / equality ("AST matching")
# ---------------------------------------------------------------------------

def _parse_literal(text: str) -> Any:
    """Parse a string as a JSON / Python literal, else return it unchanged."""
    stripped = text.strip()
    if not stripped or len(stripped) > _LITERAL_SIZE_LIMIT:
        return text
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, TypeError, RecursionError):
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            return text
    if isinstance(parsed, float) and not math.isfinite(parsed):
        return text
    return parsed


def normalize_value(value: Any) -> Any:
    """Recursively normalise an argument / state value.

    Strings that parse as JSON or Python literals are replaced by the parsed
    value (``"3"`` -> ``3``, ``"[1, 2]"`` -> ``[1, 2]``), so a model that
    stringifies its arguments can still match a structured label.  Plain
    strings are whitespace-stripped; booleans are never coerced to numbers.
    """
    if isinstance(value, str):
        parsed = _parse_literal(value)
        if parsed is not value:
            return normalize_value(parsed)
        return value.strip()
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, Mapping):
        return {str(key): normalize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_value(item) for item in value]
    return value


def _text_key(value: Any) -> str:
    """Whitespace-collapsed, NFKC, case-folded text used for string equality."""
    text = unicodedata.normalize("NFKC", str(value))
    return _WS_RE.sub(" ", text).strip().casefold()


def normalize_text(value: Any) -> str:
    """Whitespace-collapsed, NFKC text with trailing sentence punctuation removed."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = _WS_RE.sub(" ", text).strip()
    return text.strip(_TRAILING_PUNCT)


def text_equals(left: Any, right: Any) -> bool:
    """Normalised (case-insensitive) equality of two short textual answers."""
    return _text_key(normalize_text(left)) == _text_key(normalize_text(right))


def values_equal(left: Any, right: Any) -> bool:
    """AST-ish semantic equality for tool arguments / state snapshots.

    * literal strings are parsed first (:func:`normalize_value`);
    * numbers compare with a small tolerance (``3`` == ``3.0``);
    * strings compare case-insensitively after whitespace normalisation;
    * mappings compare key sets (case-insensitive keys) and values recursively;
    * sequences compare element-wise by length;
    * ``None`` equals only ``None``; booleans never equal numbers.
    """
    left = normalize_value(left)
    right = normalize_value(right)
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, int) and isinstance(right, int):
        # Int-int comparison is exact: the float tolerance below would make
        # e.g. 1_000_000_000 and 1_000_000_001 "equal" (review T3-06).
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-9)
    if isinstance(left, str) and isinstance(right, str):
        return _text_key(left) == _text_key(right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        left_map = {_text_key(key): item for key, item in left.items()}
        right_map = {_text_key(key): item for key, item in right.items()}
        if set(left_map) != set(right_map):
            return False
        return all(values_equal(left_map[key], right_map[key]) for key in left_map)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            values_equal(a, b) for a, b in zip(left, right)
        )
    return False


# ---------------------------------------------------------------------------
# ground truth accessors
# ---------------------------------------------------------------------------

def gt_call_groups(task: Any) -> list[list[dict]]:
    """Normalised per-turn ground-truth call groups (``[]`` = no tools needed)."""
    if not isinstance(task, Mapping):
        return []
    return [
        [normalize_call(call) for call in group]
        for group in schema.gt_call_groups(task)
    ]


def flatten_gt_calls(task: Any) -> list[dict]:
    """All ground-truth calls in turn order (single-turn flattening)."""
    return [call for group in gt_call_groups(task) for call in group]


def reference(task: Any) -> Mapping:
    """The task's ``reference`` mapping (or an empty dict)."""
    return as_mapping(as_mapping(task).get("reference"))


def final_answer(trajectory: Any) -> str | None:
    """``trajectory["final_answer"]`` when it is a non-empty string."""
    value = as_mapping(trajectory).get("final_answer")
    if isinstance(value, str) and value.strip():
        return value
    return None


def final_answers(task: Any) -> list[str]:
    """Acceptable answers for QA scoring (CONTRACTS C4/C7).

    ``reference["final_answers"]`` (non-empty list of strings) wins; otherwise
    fall back to ``reference["final_answer"]``; otherwise ``[]``.
    """
    ref = reference(task)
    answers = ref.get("final_answers")
    if isinstance(answers, (list, tuple)):
        result = [item.strip() for item in answers if isinstance(item, str) and item.strip()]
        if result:
            return result
    single = ref.get("final_answer")
    if isinstance(single, str) and single.strip():
        return [single.strip()]
    return []


def last_assistant_content(trajectory: Any) -> str | None:
    """Last non-empty assistant ``content`` string, else ``None``."""
    content: str | None = None
    for turn in iter_turns(trajectory):
        value = as_mapping(turn.get("assistant")).get("content")
        if isinstance(value, str) and value.strip():
            content = value
    return content


def visible_reply(trajectory: Any) -> str:
    """Best-effort user-visible reply text (last assistant content, else final answer)."""
    content = last_assistant_content(trajectory)
    if content is not None:
        return content.strip()
    answer = final_answer(trajectory)
    return answer.strip() if answer is not None else ""


def is_empty_trajectory(trajectory: Any) -> bool:
    """True when there is no turn and no final answer (an empty rollout)."""
    return not iter_turns(trajectory) and final_answer(trajectory) is None


# ---------------------------------------------------------------------------
# env_state accessors (CONTRACTS C3 / C5 / C6)
# ---------------------------------------------------------------------------

def env_state(trajectory: Any) -> Mapping:
    """The verbatim env done-info snapshot (empty mapping when absent)."""
    state = as_mapping(trajectory).get("env_state")
    return state if isinstance(state, Mapping) else {}


def has_env_state(trajectory: Any) -> bool:
    """True when ``trajectory["env_state"]`` is a mapping (even if all fields are null)."""
    return isinstance(as_mapping(trajectory).get("env_state"), Mapping)


def env_extras(trajectory: Any) -> Mapping:
    """``env_state["extras"]`` (env-private debug info)."""
    return as_mapping(env_state(trajectory).get("extras"))


def degraded_replay(trajectory: Any, task: Any = None) -> bool:
    """True when the replay used placeholder gt returns (CONTRACTS C6).

    Either the runtime marker (``env_state.extras["gt_returns_source"] ==
    "placeholder"``) or the static task marker (``task.meta["gt_returns_source"]
    == "placeholder"``, schema v2.2) marks the sample as degraded.  Degraded
    replay must be reported in ``info["degraded_replay"]`` and must never earn
    the result term; it never earns extra score either.
    """
    runtime_source = env_extras(trajectory).get("gt_returns_source")
    static_source = None
    if isinstance(task, Mapping):
        static_source = as_mapping(as_mapping(task).get("meta")).get("gt_returns_source")
    return runtime_source == "placeholder" or static_source == "placeholder"


# ---------------------------------------------------------------------------
# order-independent call matching
# ---------------------------------------------------------------------------

def _maximum_matching(
    outputs: list[dict],
    targets: list[dict],
    predicate,
) -> list[tuple[int, int]]:
    """Maximum bipartite matching ``(output_index, target_index)`` pairs (Kuhn)."""
    adjacency = [
        [target_index for target_index, target in enumerate(targets) if predicate(output, target)]
        for output in outputs
    ]
    matching: list[int] = [-1] * len(targets)

    def _augment(output_index: int, seen: list[bool]) -> bool:
        for target_index in adjacency[output_index]:
            if seen[target_index]:
                continue
            seen[target_index] = True
            current = matching[target_index]
            if current == -1 or _augment(current, seen):
                matching[target_index] = output_index
                return True
        return False

    for output_index in range(len(outputs)):
        _augment(output_index, [False] * len(targets))
    return [
        (matching[target_index], target_index)
        for target_index in range(len(targets))
        if matching[target_index] != -1
    ]


def match_calls(output_calls: Any, gt_calls: Any) -> dict:
    """One-to-one, order-independent match of emitted calls against gt calls.

    Returns a JSON-friendly report with ``matched`` (pair count), ``pairs``,
    ``missed_calls`` (gt not covered), ``extra_calls`` (emitted but unmatched),
    ``all_gt_matched`` / ``no_extra_calls`` and the call counts.  For an empty
    gt set the match is only complete when no call was emitted.
    """
    outputs = [normalize_call(call) for call in as_list(output_calls)]
    targets = [normalize_call(call) for call in as_list(gt_calls)]
    pairs = _maximum_matching(outputs, targets, calls_equal)
    matched = len(pairs)
    used_outputs = {output_index for output_index, _ in pairs}
    used_targets = {target_index for _, target_index in pairs}
    return {
        "matched": matched,
        "n_output": len(outputs),
        "n_gt": len(targets),
        "all_gt_matched": matched == len(targets),
        "no_extra_calls": matched == len(outputs),
        "pairs": [
            {"output": call_repr(outputs[output_index]), "gt": call_repr(targets[target_index])}
            for output_index, target_index in pairs
        ],
        "missed_calls": [
            call_repr(targets[index]) for index in range(len(targets)) if index not in used_targets
        ],
        "extra_calls": [
            call_repr(outputs[index]) for index in range(len(outputs)) if index not in used_outputs
        ],
    }


def group_matches(observed_calls: Any, gt_calls: Any) -> bool:
    """True when one observed turn group equals one gt turn group exactly.

    Parallel calls in the group are matched order-independently; the group
    sizes must be equal (one missing or one spurious call breaks the group).
    An empty gt group matches only an empty observed group.
    """
    outputs = [normalize_call(call) for call in as_list(observed_calls)]
    targets = [normalize_call(call) for call in as_list(gt_calls)]
    if not targets:
        return not outputs
    if len(outputs) != len(targets):
        return False
    return match_calls(outputs, targets)["all_gt_matched"]


def align_groups(observed_groups: Any, gt_groups: Any) -> dict:
    """Align per-turn observed call groups with per-turn gt groups (in order).

    Empty gt groups matter: a gt group ``[]`` means "this turn needs no tool
    call" and must be alignable with a pure-text assistant turn (review
    T3-03).  Observed groups that contain no call are never "spurious"; they
    are kept as diagnostics.  An observed group with calls that is not
    consumed is spurious and breaks ``full_match``.  ``full_match`` is true
    only when every gt group was matched and no *call-carrying* observed
    group is left over.
    """
    observed = [[normalize_call(call) for call in as_list(group)] for group in as_list(observed_groups)]
    targets = [[normalize_call(call) for call in as_list(group)] for group in as_list(gt_groups)]
    cursor = 0
    pairs: list[dict] = []
    consumed: set[int] = set()
    for gt_index, group in enumerate(targets):
        found: int | None = None
        for observed_index in range(cursor, len(observed)):
            if group_matches(observed[observed_index], group):
                found = observed_index
                break
        if found is not None:
            pairs.append({"gt": gt_index, "observed": found})
            consumed.add(found)
            cursor = found + 1
    leftovers = [
        {
            "observed": index,
            "has_calls": bool(observed[index]),
            "calls": [call_repr(call) for call in observed[index]],
        }
        for index in range(len(observed))
        if index not in consumed
    ]
    spurious = [item for item in leftovers if item["has_calls"]]
    matched = len(pairs)
    total = len(targets)
    if total == 0:
        fraction = 0.0 if spurious else 1.0
        full_match = not spurious
    else:
        fraction = matched / total
        full_match = matched == total and not spurious
    return {
        "matched": matched,
        "total": total,
        "fraction": fraction,
        "full_match": full_match,
        "pairs": pairs,
        "leftover_observed": leftovers,
        "spurious_observed": spurious,
    }


def derive_turn_match(trajectory: Any, task: Any) -> dict:
    """Per-turn match ratio computed from the trajectory turns vs ``gt_calls``.

    Uses all turn groups (including pure-text turns) so that an empty gt
    group (``gt_calls=[[c], []]``, review T3-03) can be aligned with a reply
    turn.  Used only as a fallback when the env did not report
    ``matched_turns`` / ``gt_turns`` (e.g. a stateful env); when the env
    reports them the reward trusts the env (CONTRACTS C4).
    """
    result = align_groups(call_groups(trajectory), gt_call_groups(task))
    return {
        "matched": result["matched"],
        "total": result["total"],
        "fraction": result["fraction"],
        "full_match": result["full_match"],
        "leftover_observed": result["leftover_observed"],
        "spurious_observed": result["spurious_observed"],
        "source": "derived:turns",
    }


# ---------------------------------------------------------------------------
# structural format gate
# ---------------------------------------------------------------------------

_TYPE_CHECKERS = {
    "string": lambda value: isinstance(value, str),
    "str": lambda value: isinstance(value, str),
    "integer": lambda value: _is_integral(value),
    "int": lambda value: _is_integral(value),
    "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
    "float": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
    "boolean": lambda value: isinstance(value, bool),
    "bool": lambda value: isinstance(value, bool),
    "array": lambda value: isinstance(value, (list, tuple)),
    "list": lambda value: isinstance(value, (list, tuple)),
    "object": lambda value: isinstance(value, Mapping),
    "dict": lambda value: isinstance(value, Mapping),
    "null": lambda value: value is None,
    "none": lambda value: value is None,
}


def _declared_types(spec: Any) -> list[str]:
    declared = spec.get("type") if isinstance(spec, Mapping) else spec
    if isinstance(declared, str):
        return [declared]
    if isinstance(declared, (list, tuple)):
        return [item for item in declared if isinstance(item, str)]
    return []


def _type_compatible(value: Any, type_name: str) -> bool:
    checker = _TYPE_CHECKERS.get(type_name.strip().lower())
    return checker(value) if checker is not None else True


def _arg_value_ok(value: Any, spec: Any) -> bool:
    types = _declared_types(spec)
    if types and not any(_type_compatible(value, name) for name in types):
        return False
    enum = spec.get("enum") if isinstance(spec, Mapping) else None
    if isinstance(enum, (list, tuple)) and enum:
        return any(values_equal(value, item) for item in enum)
    return True


def _tool_param_specs(tool: Any) -> tuple[dict, list[str], bool]:
    """``(properties, required, has_schema)`` from one OpenAI-style tool definition.

    Supports both the canonical ``{"properties": {...}, "required": [...]}``
    shape and the legacy flat ``{"param": {"type": ...}}`` shape.  ``False``
    means "no parameter schema information", which makes the gate skip argument
    checks for that tool rather than rejecting everything.
    """
    parameters = as_mapping(as_mapping(tool).get("parameters"))
    if not parameters:
        return {}, [], False
    properties = parameters.get("properties")
    required_raw = parameters.get("required")
    required = (
        [item for item in required_raw if isinstance(item, str)]
        if isinstance(required_raw, (list, tuple))
        else []
    )
    if isinstance(properties, Mapping):
        return dict(properties), required, True
    flat = {
        str(key): value
        for key, value in parameters.items()
        if str(key) not in _PARAM_SCHEMA_KEYS
    }
    return flat, required, bool(flat) or bool(required)


def _declared_tools(task: Any) -> dict[str, tuple[dict, list[str], bool]]:
    declared: dict[str, tuple[dict, list[str], bool]] = {}
    for tool in as_list(as_mapping(task).get("tools")):
        if not isinstance(tool, Mapping):
            continue
        name = tool.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        declared[name.strip()] = _tool_param_specs(tool)
    return declared


def _turn_format_issues(
    turn: Mapping,
    declared: dict[str, tuple[dict, list[str], bool]],
    index: int,
    *,
    expect_tool_results: bool = True,
) -> list[str]:
    """List of structural format violations of one turn (empty = legal)."""
    issues: list[str] = []
    assistant = turn.get("assistant")
    if not isinstance(assistant, Mapping):
        issues.append("assistant_not_mapping")
        content = ""
        raw_calls: list = []
    else:
        content_value = assistant.get("content")
        if isinstance(content_value, str):
            content = content_value
        else:
            content = ""
            issues.append("assistant_content_not_string")
        raw_calls = turn_calls(turn)
    if not raw_calls and not content.strip():
        issues.append("empty_turn")
    if not raw_calls and "<tool_call>" in content:
        # tool-call markup that did not parse into a call (bad JSON / name)
        issues.append("unparsed_tool_call")

    calls: list[dict] = []
    for call_index, raw_call in enumerate(raw_calls):
        if not isinstance(raw_call, Mapping):
            issues.append(f"call_{call_index}_not_mapping")
            calls.append({"name": "", "arguments": {}, "id": None})
            continue
        name_value = raw_call.get("name")
        name = name_value.strip() if isinstance(name_value, str) else ""
        arguments_value = raw_call.get("arguments")
        if isinstance(arguments_value, Mapping):
            arguments = dict(arguments_value)
        else:
            arguments = {}
            issues.append(f"call_{call_index}_arguments_not_mapping")
        calls.append({"name": name, "arguments": arguments, "id": raw_call.get("id")})
        if not name:
            issues.append(f"call_{call_index}_name_missing")
            continue
        if not declared or name not in declared:
            # No declared tools at all: every emitted call is undeclared
            # (review T3-05).  This is a format violation, not a silent pass.
            issues.append(f"call_{call_index}_undeclared_tool:{name}")
            continue
        specs, required, has_schema = declared.get(name, ({}, [], False))
        for required_name in required:
            if required_name not in arguments:
                issues.append(f"call_{call_index}_missing_required:{name}.{required_name}")
        if has_schema:
            for argument_name in arguments:
                key = str(argument_name)
                if key not in specs:
                    issues.append(f"call_{call_index}_unknown_arg:{name}.{key}")
                    continue
                if not _arg_value_ok(normalize_value(arguments[argument_name]), specs[key]):
                    issues.append(f"call_{call_index}_bad_arg_type:{name}.{key}")

    if not expect_tool_results:
        # Single-turn *prediction* (st_fc, env.type == "none"): the emitted
        # calls are the model's answer, never executed, so there is no
        # tool_results to pair them with.  Pairing/name/id checks are
        # meaningless here (CONTRACTS C14 / TEAM_BOARD D20); call-level checks
        # (declared tool / required args / arg types) above still apply.
        return issues

    results_value = turn.get("tool_results")
    if calls:
        if isinstance(results_value, (list, tuple)):
            results = list(results_value)
        else:
            results = []
            issues.append("tool_results_not_list")
        if len(results) != len(calls):
            issues.append(f"tool_results_count_mismatch:{len(results)}!={len(calls)}")
        call_ids = [call.get("id") for call in calls]
        result_ids = [
            result.get("id") if isinstance(result, Mapping) else None
            for result in results
        ]
        if (
            all(isinstance(item, str) and item for item in call_ids)
            and all(isinstance(item, str) and item for item in result_ids)
        ):
            if set(result_ids) != set(call_ids):
                issues.append("tool_result_ids_mismatch")
        else:
            for result_index in range(min(len(calls), len(results))):
                result = results[result_index]
                result_name = result.get("name") if isinstance(result, Mapping) else None
                if not isinstance(result_name, str) or result_name.strip() != calls[result_index]["name"]:
                    issues.append(f"tool_result_{result_index}_name_mismatch")
        for result_index, result in enumerate(results):
            if not isinstance(result, Mapping):
                issues.append(f"tool_result_{result_index}_not_mapping")
                continue
            if not isinstance(result.get("ok"), bool):
                issues.append(f"tool_result_{result_index}_ok_not_bool")
            if not isinstance(result.get("content"), str):
                issues.append(f"tool_result_{result_index}_content_not_string")
            result_name = result.get("name")
            if not isinstance(result_name, str) or not result_name.strip():
                issues.append(f"tool_result_{result_index}_name_missing")
    return issues


def format_report(
    task: Any,
    trajectory: Any,
    *,
    expect_tool_results: bool = True,
) -> tuple[float, dict]:
    """Structural format score in ``[0, 1]`` = legal turns / total turns.

    An empty trajectory scores ``0.0`` (no assistant response is not "legal
    format").  The second return value is a JSON-friendly diagnostic mapping
    (``format_score``, ``legal_turns``, ``total_turns``, capped ``issues``,
    ``declared_tools``).
    """
    turns = iter_turns(trajectory)
    declared = _declared_tools(task)
    if not turns:
        return 0.0, {
            "format_score": 0.0,
            "legal_turns": 0,
            "total_turns": 0,
            "issues": [{"turn": None, "issues": ["empty_trajectory"]}],
            "declared_tools": sorted(declared),
        }
    issues: list[dict] = []
    legal = 0
    for turn_index, turn in enumerate(turns):
        turn_issues = _turn_format_issues(
            turn,
            declared,
            turn_index,
            expect_tool_results=expect_tool_results,
        )
        if turn_issues:
            issues.append({"turn": turn_index, "issues": turn_issues[:8]})
        else:
            legal += 1
    score = legal / len(turns)
    return score, {
        "format_score": score,
        "legal_turns": legal,
        "total_turns": len(turns),
        "issues": issues[:10],
        "declared_tools": sorted(declared),
    }


def format_ok(report: dict) -> bool:
    """True when every turn is structurally legal (the st_fc format gate)."""
    return float(report.get("format_score", 0.0)) >= 1.0 - 1e-9


# ---------------------------------------------------------------------------
# result construction
# ---------------------------------------------------------------------------

def clamp_score(score: Any) -> float:
    """Clamp a score into the frozen ``schema.SCORE_RANGE`` (CONTRACTS C4)."""
    try:
        value = float(score)
    except (TypeError, ValueError):
        return SCORE_MIN
    if not math.isfinite(value):
        return SCORE_MIN
    return max(SCORE_MIN, min(SCORE_MAX, value))


def weighted_sum(parts: list[tuple[float, float]]) -> float:
    """``sum(weight * value)`` via ``math.fsum`` for stable 0.8+0.1+0.1 arithmetic."""
    return math.fsum(float(weight) * float(value) for weight, value in parts)


def jsonable(value: Any) -> Any:
    """Recursively convert ``value`` into plain JSON-serialisable data."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [jsonable(item) for item in sorted(value, key=lambda item: repr(item))]
    return repr(value)


def make_result(
    score: Any,
    info: Any = None,
    *,
    task_type: str | None = None,
    strict: bool = False,
) -> dict:
    """Build and validate the frozen reward payload ``{"score": float, "info": dict}``.

    * clamps ``score`` into ``[0, 1]`` (CONTRACTS C4) and records
      ``info["score_raw"]`` + ``info["score_clamped"]`` so an out-of-range
      score is visible instead of silently masked (review T3-04);
    * with ``strict=True`` an out-of-range / non-finite raw score raises
      ``schema.SchemaError`` instead of being clamped;
    * adds ``info["task_type"]`` when given;
    * JSON-sanitises ``info`` (deterministic for sets / tuples);
    * validates through :func:`schema.validate_reward_result`, raising
      ``schema.SchemaError`` if the payload would violate the contract.
    """
    payload_info = dict(as_mapping(info))
    if task_type is not None:
        payload_info.setdefault("task_type", task_type)

    raw_value: float | None
    try:
        raw_value = float(score)
    except (TypeError, ValueError):
        raw_value = None
    finite = raw_value is not None and math.isfinite(raw_value)
    clamped_value = clamp_score(score)
    clamped = not finite or not (SCORE_MIN <= raw_value <= SCORE_MAX)

    payload_info.setdefault(
        "score_raw", raw_value if finite else repr(score)
    )
    payload_info.setdefault("score_clamped", bool(clamped))
    if strict and clamped:
        raise schema.SchemaError(
            f"strict reward result: raw score {score!r} outside "
            f"[{SCORE_MIN}, {SCORE_MAX}]"
        )

    result = {"score": clamped_value, "info": jsonable(payload_info)}
    errors = schema.validate_reward_result(result)
    if errors:
        raise schema.SchemaError("invalid reward result: " + "; ".join(errors))
    return result
