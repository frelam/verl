"""Shared schema, validation and hashing helpers for ``agentic_rl_data``.

Single source of truth for the contracts used by loaders, reward functions,
envs and pipeline scripts.  The authoritative spec is
``agentic_rl_data_plan.md`` (§2 task schema, §5 reward interface).

Contract summary
----------------

Task (one JSONL line, plan §2)::

    {
      "id": "src-000123",
      "source": "funreason_mt",
      "task_type": "mt_fc" | "st_fc" | "irrelevance" | "search_qa"
                   | "math_ci" | "code_exec" | "abstain",
      "messages": [{"role": "system"|"user", "content": "..."}, ...],
      "tools": [{"name": str, "description": str, "parameters": {...}}, ...],
      "env": {"type": "none"|"mock_fc"|"search"|"python"|"code_tests",
              "config": {...}},
      "reference": {
          "final_answer": str | None,           # 单答案（展示 / 打分的兜底）
          "final_answers": [str, ...] | None,   # 多可接受答案（A²Search 等），优先用于打分
          "gt_calls": [[{"name": str, "arguments": dict}, ...], ...],  # per-turn
          "gt_turns": list | None,              # 旧版逐轮结构（可选）
          "tool_returns": [[{"name": str, "content": str, "ok": bool}], ...] | None,
          "unit_tests": str | None,
          "final_state": dict | None,           # mock_fc stateful 的 GT 终态快照
      },
      "meta": {"max_turns": int, "difficulty": null | float,
               "pass_rate": null | float, "license": null | str,
               "expected_turns": null | int,   # optional: SFT trajectory length
               "group_id": null | str},        # optional: session key for group-aware split
    }

Missing optional values MUST be explicit ``null`` (never silently dropped,
plan §2).  ``gt_calls`` is a list of *turn groups*; parallel calls live in the
same group and are order-insensitive at evaluation time.  ``[]`` means
"label says no tool call is needed"; ``None`` is *not* allowed (it means the
source could not tell, which is a data-prep bug).

Trajectory (normalised form handed to reward functions)::

    {
      "turns": [
          {"step": 1,
           "assistant": {"content": str,
                         "tool_calls": [{"id": str|None, "name": str,
                                         "arguments": dict}]},
           "tool_results": [{"id": str|None, "name": str,
                             "content": str, "ok": bool}]},
          ...,
      ],
      "final_answer": str | None,
      "num_turns": int,
      "done": bool,
      "env_state": dict | None,   # optional snapshot for stateful envs
    }

Reward interface (plan §5)::

    def reward(task: dict, trajectory: dict) -> {"score": float, "info": dict}

Scores are frozen to ``[0.0, 1.0]`` for every task type (plan §5/§10): the
result term dominates (e.g. ``mt_fc`` = 0.8 result + 0.1 per-turn match +
0.1 format).  Partial credit for timeouts is allowed but must stay in range.

Trajectory ``env_state`` (v2, see ``CONTRACTS.md`` C3): the rollout adapter
copies ``info["env_state"]`` **verbatim** into ``trajectory["env_state"]``.
Canonical keys: ``type``, ``impl``, ``goal_satisfied``, ``replay_ok``,
``failed_turn``, ``matched_turns``, ``gt_turns``, ``checks``, ``state``,
``rendered``, ``num_turns``, ``timeout``, ``code_tests`` (dict, code_exec
only), ``extras``.  Rewards read only from ``trajectory``.

Loaders: ``agentic_rl_data/loaders/<source>.py`` must expose ``SOURCE: str``
and ``load_tasks(cfg: dict | None = None) -> Iterator[dict]`` yielding *schema
valid* tasks; raw download / parsing stays private to the module.

This module is stdlib-only so it can be imported without torch / datasets.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterator, Mapping

TASK_TYPES = frozenset(
    {"mt_fc", "st_fc", "irrelevance", "search_qa", "math_ci", "code_exec", "abstain"}
)
ENV_TYPES = frozenset({"none", "mock_fc", "search", "python", "code_tests"})
MESSAGE_ROLES = frozenset({"system", "user"})
REQUIRED_TASK_KEYS = (
    "id",
    "source",
    "task_type",
    "messages",
    "tools",
    "env",
    "reference",
    "meta",
)
REWARD_KEYS = frozenset({"score", "info"})
DEFAULT_MAX_TURNS = 15

#: Frozen reward range for every task type (plan §5): 1/0 or 0.8+0.1+0.1 style,
#: never negative (the old tool_rl reward allowed negatives; this plan does not).
SCORE_RANGE: tuple[float, float] = (0.0, 1.0)

#: Keys that MUST be present in ``reference`` (values may be ``null``).
REQUIRED_REFERENCE_KEYS = (
    "final_answer",
    "final_answers",
    "gt_calls",
    "gt_turns",
    "tool_returns",
    "unit_tests",
    "final_state",
)

#: Keys that MUST be present in ``meta`` (values may be ``null`` except max_turns).
REQUIRED_META_KEYS = ("max_turns", "difficulty", "pass_rate", "license")

#: Static replay-quality declaration (CONTRACTS C6): where the env will find
#: ground-truth tool returns for this task.
GT_RETURNS_SOURCES = frozenset({"config", "reference", "gt_turns", "placeholder"})

#: Allowed env types per task type (plan §5 + §4; mirror of the reward table).
ENV_FOR_TASK_TYPE: dict[str, frozenset[str]] = {
    "mt_fc": frozenset({"mock_fc"}),
    "st_fc": frozenset({"none", "mock_fc"}),
    "irrelevance": frozenset({"none"}),
    "search_qa": frozenset({"search"}),
    "math_ci": frozenset({"python"}),
    "code_exec": frozenset({"code_tests"}),
    "abstain": frozenset({"none"}),
}


class SchemaError(ValueError):
    """Raised when a task / trajectory / reward payload breaks the contract."""


# ---------------------------------------------------------------------------
# small type helpers
# ---------------------------------------------------------------------------

def _is_str(value: Any) -> bool:
    return isinstance(value, str)


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def validate_task(task: Any) -> list[str]:
    """Return a list of human readable contract violations (empty = valid)."""
    errors: list[str] = []
    if not _is_mapping(task):
        return [f"task must be a mapping, got {type(task).__name__}"]

    for key in REQUIRED_TASK_KEYS:
        if key not in task:
            errors.append(f"missing required key {key!r}")

    tid = task.get("id")
    if not _is_str(tid) or not tid.strip():
        errors.append("id must be a non-empty string")

    source = task.get("source")
    if not _is_str(source) or not source.strip():
        errors.append("source must be a non-empty string")

    task_type = task.get("task_type")
    if task_type not in TASK_TYPES:
        errors.append(f"task_type {task_type!r} not in {sorted(TASK_TYPES)}")

    messages = task.get("messages")
    if not isinstance(messages, list) or not messages:
        errors.append("messages must be a non-empty list")
    else:
        for i, msg in enumerate(messages):
            if not _is_mapping(msg):
                errors.append(f"messages[{i}] must be a mapping")
            elif msg.get("role") not in MESSAGE_ROLES:
                errors.append(
                    f"messages[{i}].role {msg.get('role')!r} not in {sorted(MESSAGE_ROLES)}"
                )
            elif not _is_str(msg.get("content")) or not msg["content"]:
                errors.append(f"messages[{i}].content must be a non-empty string")

    tools = task.get("tools")
    if not isinstance(tools, list):
        errors.append("tools must be a list (use [] when no tool is declared)")
    else:
        seen: set[str] = set()
        for i, tool in enumerate(tools):
            if not _is_mapping(tool):
                errors.append(f"tools[{i}] must be a mapping")
                continue
            name = tool.get("name")
            if not _is_str(name) or not name.strip():
                errors.append(f"tools[{i}].name must be a non-empty string")
            elif name in seen:
                errors.append(f"duplicate tool name {name!r}")
            else:
                seen.add(name)
            if not _is_mapping(tool.get("parameters")):
                errors.append(f"tools[{i}].parameters must be a mapping")
            description = tool.get("description")
            if description is not None and not _is_str(description):
                errors.append(f"tools[{i}].description must be a string when present")

    env = task.get("env")
    if not _is_mapping(env):
        errors.append("env must be a mapping")
    else:
        env_type = env.get("type")
        if env_type not in ENV_TYPES:
            errors.append(f"env.type {env_type!r} not in {sorted(ENV_TYPES)}")
        elif task_type in ENV_FOR_TASK_TYPE and env_type not in ENV_FOR_TASK_TYPE[task_type]:
            errors.append(
                f"env.type {env_type!r} is not allowed for task_type {task_type!r} "
                f"(allowed: {sorted(ENV_FOR_TASK_TYPE[task_type])})"
            )
        if not _is_mapping(env.get("config", {})):
            errors.append("env.config must be a mapping when present")

    reference = task.get("reference")
    if not _is_mapping(reference):
        errors.append("reference must be a mapping")
    else:
        for key in REQUIRED_REFERENCE_KEYS:
            if key not in reference:
                errors.append(
                    f"reference missing key {key!r} (use explicit null when not applicable)"
                )
        gt_calls = reference.get("gt_calls")
        if gt_calls is None:
            errors.append("reference.gt_calls must be a list (use [] for 'no tools needed')")
        elif not isinstance(gt_calls, list):
            errors.append("reference.gt_calls must be a list of turn groups")
        else:
            for gi, group in enumerate(gt_calls):
                if not isinstance(group, list):
                    errors.append(f"reference.gt_calls[{gi}] must be a list (turn group)")
                    continue
                for ci, call in enumerate(group):
                    if not _is_mapping(call):
                        errors.append(f"reference.gt_calls[{gi}][{ci}] must be a mapping")
                        continue
                    if not _is_str(call.get("name")) or not call["name"].strip():
                        errors.append(f"reference.gt_calls[{gi}][{ci}].name must be a non-empty string")
                    if not _is_mapping(call.get("arguments", {})):
                        errors.append(f"reference.gt_calls[{gi}][{ci}].arguments must be a mapping")
        for key in ("final_answer", "unit_tests"):
            value = reference.get(key)
            if value is not None and not _is_str(value):
                errors.append(f"reference.{key} must be a string or null")

        final_answers = reference.get("final_answers")
        if final_answers is not None:
            if not isinstance(final_answers, list) or not final_answers:
                errors.append("reference.final_answers must be a non-empty list or null")
            else:
                for i, answer in enumerate(final_answers):
                    if not _is_str(answer) or not answer.strip():
                        errors.append(f"reference.final_answers[{i}] must be a non-empty string")

        final_state = reference.get("final_state")
        if final_state is not None and not _is_mapping(final_state):
            errors.append("reference.final_state must be a mapping or null")

        gt_turns = reference.get("gt_turns")
        if gt_turns is not None:
            if not isinstance(gt_turns, list):
                errors.append("reference.gt_turns must be a list or null")
            else:
                for i, turn in enumerate(gt_turns):
                    if not _is_mapping(turn):
                        errors.append(f"reference.gt_turns[{i}] must be a mapping")

        tool_returns = reference.get("tool_returns")
        if tool_returns is not None:
            if not isinstance(tool_returns, list):
                errors.append("reference.tool_returns must be a list of per-turn groups or null")
            else:
                for gi, group in enumerate(tool_returns):
                    if not isinstance(group, list):
                        errors.append(f"reference.tool_returns[{gi}] must be a list (turn group)")
                        continue
                    for ri, ret in enumerate(group):
                        if not _is_mapping(ret):
                            errors.append(f"reference.tool_returns[{gi}][{ri}] must be a mapping")
                            continue
                        if not _is_str(ret.get("name")) or not str(ret["name"]).strip():
                            errors.append(
                                f"reference.tool_returns[{gi}][{ri}].name must be a non-empty string"
                            )
                        content = ret.get("content")
                        if content is not None and not _is_str(content):
                            errors.append(
                                f"reference.tool_returns[{gi}][{ri}].content must be a string or null"
                            )

    meta = task.get("meta")
    if not _is_mapping(meta):
        errors.append("meta must be a mapping")
    else:
        for key in REQUIRED_META_KEYS:
            if key not in meta:
                errors.append(f"meta missing key {key!r} (use explicit null when not applicable)")
        max_turns = meta.get("max_turns")
        if not _is_int(max_turns) or max_turns <= 0:
            errors.append("meta.max_turns must be a positive int")
        pass_rate = meta.get("pass_rate")
        if pass_rate is not None and (not _is_number(pass_rate) or not (0.0 <= pass_rate <= 1.0)):
            errors.append("meta.pass_rate must be null or a float in [0, 1]")
        difficulty = meta.get("difficulty")
        if difficulty is not None and not _is_number(difficulty):
            errors.append("meta.difficulty must be null or a number")
        license_name = meta.get("license")
        if license_name is not None and not _is_str(license_name):
            errors.append("meta.license must be null or a string")
        expected_turns = meta.get("expected_turns")
        if expected_turns is not None and (not _is_int(expected_turns) or expected_turns <= 0):
            errors.append("meta.expected_turns must be null or a positive int")
        gt_returns_source = meta.get("gt_returns_source")
        if gt_returns_source is not None and gt_returns_source not in GT_RETURNS_SOURCES:
            errors.append(
                f"meta.gt_returns_source {gt_returns_source!r} not in "
                f"{sorted(GT_RETURNS_SOURCES)}"
            )
        if task_type == "mt_fc" and gt_returns_source is None:
            errors.append(
                "meta.gt_returns_source is required for task_type 'mt_fc' "
                f"(one of {sorted(GT_RETURNS_SOURCES)})"
            )
        group_id = meta.get("group_id")
        if group_id is not None and not _is_str(group_id):
            errors.append(
                "meta.group_id must be null or a string (session/group key for split)"
            )

    return errors


def assert_valid_task(task: Any) -> None:
    """Raise :class:`SchemaError` listing every violation, else return None."""
    errors = validate_task(task)
    if errors:
        raise SchemaError("invalid task:\n  - " + "\n  - ".join(errors))


def validate_trajectory(trajectory: Any) -> list[str]:
    """Return contract violations for a normalised trajectory (empty = valid)."""
    errors: list[str] = []
    if not _is_mapping(trajectory):
        return [f"trajectory must be a mapping, got {type(trajectory).__name__}"]

    turns = trajectory.get("turns")
    if not isinstance(turns, list):
        errors.append("turns must be a list")
    else:
        for i, turn in enumerate(turns):
            if not _is_mapping(turn):
                errors.append(f"turns[{i}] must be a mapping")
                continue
            assistant = turn.get("assistant")
            if not _is_mapping(assistant):
                errors.append(f"turns[{i}].assistant must be a mapping")
            else:
                if not _is_str(assistant.get("content", "")):
                    errors.append(f"turns[{i}].assistant.content must be a string")
                calls = assistant.get("tool_calls", [])
                if not isinstance(calls, list):
                    errors.append(f"turns[{i}].assistant.tool_calls must be a list")
                else:
                    for ci, call in enumerate(calls):
                        if not _is_mapping(call):
                            errors.append(f"turns[{i}].assistant.tool_calls[{ci}] must be a mapping")
                            continue
                        if not _is_str(call.get("name")) or not call["name"].strip():
                            errors.append(f"turns[{i}].assistant.tool_calls[{ci}].name must be a non-empty string")
                        if not _is_mapping(call.get("arguments", {})):
                            errors.append(f"turns[{i}].assistant.tool_calls[{ci}].arguments must be a mapping")
            results = turn.get("tool_results", [])
            if not isinstance(results, list):
                errors.append(f"turns[{i}].tool_results must be a list")
            else:
                for ri, tool_result in enumerate(results):
                    if not _is_mapping(tool_result):
                        errors.append(f"turns[{i}].tool_results[{ri}] must be a mapping")
                        continue
                    if not _is_str(tool_result.get("name")) or not tool_result["name"].strip():
                        errors.append(
                            f"turns[{i}].tool_results[{ri}].name must be a non-empty string"
                        )
                    if not _is_str(tool_result.get("content", "")):
                        errors.append(f"turns[{i}].tool_results[{ri}].content must be a string")

    num_turns = trajectory.get("num_turns", len(turns) if isinstance(turns, list) else None)
    if not _is_int(num_turns) or num_turns < 0:
        errors.append("num_turns must be a non-negative int")

    final_answer = trajectory.get("final_answer")
    if final_answer is not None and not _is_str(final_answer):
        errors.append("final_answer must be a string or null")

    done = trajectory.get("done", False)
    if not isinstance(done, bool):
        errors.append("done must be a bool")
    return errors


def assert_valid_trajectory(trajectory: Any) -> None:
    errors = validate_trajectory(trajectory)
    if errors:
        raise SchemaError("invalid trajectory:\n  - " + "\n  - ".join(errors))


def validate_reward_result(
    result: Any,
    *,
    min_score: float = SCORE_RANGE[0],
    max_score: float = SCORE_RANGE[1],
) -> list[str]:
    """Return contract violations for the ``{"score": float, "info": dict}`` payload."""
    errors: list[str] = []
    if not _is_mapping(result):
        return [f"reward result must be a mapping, got {type(result).__name__}"]
    for key in REWARD_KEYS:
        if key not in result:
            errors.append(f"missing required key {key!r}")
    score = result.get("score")
    if not _is_number(score):
        errors.append("score must be a finite number")
    elif isinstance(score, float) and not math.isfinite(score):
        errors.append("score must be finite")
    elif not (min_score <= float(score) <= max_score):
        errors.append(f"score {score!r} outside frozen range [{min_score}, {max_score}]")
    info = result.get("info")
    if not _is_mapping(info):
        errors.append("info must be a mapping (never null)")
    return errors


# ---------------------------------------------------------------------------
# JSONL I/O (deterministic on purpose: plan §8 acceptance #1)
# ---------------------------------------------------------------------------

def load_jsonl(path: str | Path) -> Iterator[dict]:
    """Yield parsed objects from a JSONL file, with line numbers in errors."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise SchemaError(f"{path}:{lineno}: invalid JSON: {exc}") from exc


def dump_jsonl(
    tasks: Iterator[dict] | list[dict],
    path: str | Path,
    *,
    sort_keys: bool = True,
) -> int:
    """Write tasks as JSONL (one object per line), creating parent dirs.

    ``sort_keys`` + ``ensure_ascii=False`` keep repeated runs byte-identical
    (acceptance criterion "重复运行结果一致").
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for task in tasks:
            fh.write(
                json.dumps(task, ensure_ascii=False, sort_keys=sort_keys, default=_json_default)
            )
            fh.write("\n")
            count += 1
    return count


def _json_default(value: Any) -> Any:
    """Serialise numpy-ish values / paths that leak in from parquet sources."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


# ---------------------------------------------------------------------------
# normalisation + dedup keys (plan §3 step 3)
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
_TRAILING_PUNCT = " \t\r\n.!?。！？"


def normalize_query(text: Any) -> str:
    """Normalise user text for exact dedup / decontamination matching."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).strip().lower()
    text = _WS_RE.sub(" ", text)
    return text.strip(_TRAILING_PUNCT)


def canonical_json(obj: Any) -> str:
    """Stable JSON encoding (sorted keys, compact separators)."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_json_default)


def _param_signature(parameters: Any) -> list[dict[str, Any]]:
    parameters = parameters if _is_mapping(parameters) else {}
    properties = parameters.get("properties") or {}
    required = list(parameters.get("required") or [])
    signature: list[dict[str, Any]] = []
    for name in sorted(properties):
        spec = properties[name] if _is_mapping(properties[name]) else {}
        signature.append(
            {"name": name, "type": str(spec.get("type", "any")), "required": name in required}
        )
    return signature


def tools_signature(tools: Any) -> str:
    """Canonical signature of a tool list (names + parameter names/types).

    Sorted by tool name on purpose: the same tool set in a different order
    must hash identically (exact dedup, plan §3 step 3).
    """
    signature = []
    candidates = [tool for tool in (tools if isinstance(tools, list) else []) if _is_mapping(tool)]
    for tool in sorted(candidates, key=lambda item: str(item.get("name"))):
        if not _is_mapping(tool):
            continue
        signature.append(
            {"name": tool.get("name"), "params": _param_signature(tool.get("parameters"))}
        )
    return canonical_json(signature)


def task_dedup_key(task: Mapping) -> str:
    """Exact-dedup hash: normalised user queries + tools signature (plan §3)."""
    queries = "\n".join(
        normalize_query(message.get("content"))
        for message in task.get("messages") or []
        if _is_mapping(message) and message.get("role") == "user"
    )
    payload = f"{queries}\x1f{tools_signature(task.get('tools'))}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def text_shingles(text: Any, k: int = 5) -> set[int]:
    """Hashed k-shingles of normalised text — input for MinHash / Jaccard."""
    norm = normalize_query(text)
    if not norm:
        return set()
    k = min(max(1, k), len(norm))
    return {
        int.from_bytes(
            hashlib.blake2b(norm[i : i + k].encode("utf-8"), digest_size=4).digest(), "big"
        )
        for i in range(len(norm) - k + 1)
    }


# ---------------------------------------------------------------------------
# ground-truth helpers (used by rewards + replay checker)
# ---------------------------------------------------------------------------

def gt_call_groups(task: Mapping) -> list[list[dict[str, Any]]]:
    """Normalised per-turn ground-truth call groups (``[]`` = no tools needed)."""
    reference = task.get("reference") if _is_mapping(task.get("reference")) else {}
    groups: list[list[dict[str, Any]]] = []
    for group in reference.get("gt_calls") or []:
        if not isinstance(group, list):
            continue
        calls = []
        for call in group:
            if not _is_mapping(call):
                continue
            calls.append(
                {
                    "name": call.get("name"),
                    "arguments": dict(call.get("arguments") or {}),
                }
            )
        groups.append(calls)
    return groups


def flatten_gt_calls(task: Mapping) -> list[dict[str, Any]]:
    """All ground-truth calls in turn order (for single-turn checks / replay)."""
    return [call for group in gt_call_groups(task) for call in group]
