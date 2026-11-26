"""``mock_fc`` multi-turn environments (plan §4.2).

Two implementations share ``env.type == "mock_fc"`` and the same
``BaseEnv`` contract; ``env.config["impl"]`` selects one (``build_env`` uses
"auto": a ``domain`` / ``goal`` config means stateful, otherwise replay).

A. Stateful (preferred, BFCL v3 style)
    A config-driven state machine per domain (filesystem / social / trading)
    plus canned responses for external data tools.  Terminal state is
    verified by ``env.config.goal`` assertions; the raw final state snapshot
    is written to ``env_state["state"]`` so the reward can fall back to
    ``reference["final_state"]`` deep comparison (CONTRACTS.md C3/C4).

B. Replay (fallback)
    Each turn's tool calls are matched against ``reference.gt_calls`` (per
    turn group, order-insensitive, loose value comparison by default).  On a
    match the ground-truth return is injected and the episode continues; on
    a mismatch the episode terminates with ``replay_ok=False``.  The verdict
    for a *fully* matched trajectory is exposed as ``env.replay_ok`` and
    ``env_state["replay_ok"]`` (CONTRACTS.md C4, mt_fc B).

Ground-truth returns (CONTRACTS.md C6), priority order:

    1. ``task.env.config["gt_returns"]``      -> source "config"
    2. ``reference["tool_returns"]``          -> source "reference"
    3. ``reference["gt_turns"][*]``           -> source "gt_turns"
    4. synthetic ``{"result": "ok"}``         -> source "placeholder"

``env_state.extras["gt_returns_source"]`` always records which branch was used,
so degraded replay data is never silent.
"""

from __future__ import annotations

import copy
import json
import math
import re
from datetime import date
from typing import Any, Mapping

from ..schema import canonical_json, gt_call_groups, normalize_query
from .base import BaseEnv, EnvError, make_tool_result, truncate_text
from .dialogue import DialogueController

#: Built-in stateful domains (BFCL v3 style subset; extend by subclassing).
DOMAINS: dict[str, type] = {}

#: GT-return sources in priority order (CONTRACTS.md C6).
GT_RETURNS_SOURCES = ("config", "reference", "gt_turns", "placeholder")

#: Transient error returned by an injected fault (env.config.faults).
DEFAULT_FAULT_MESSAGE = (
    "Error: 503 Service Unavailable - the backend timed out, nothing was changed. "
    "Please retry the call."
)

_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_NUMBER_RE = re.compile(r"^[-+]?\$?\s*\d[\d,]*(?:\.\d+)?%?$")


def _as_number(text: str) -> float | None:
    value = text.strip()
    if not _NUMBER_RE.match(value):
        return None
    try:
        return float(value.replace("$", "").replace(",", "").replace("%", "").strip())
    except ValueError:
        return None


def answer_tag_matches(actual: Any, expected: Any) -> bool:
    """Strict final-answer check: last ``<answer>...</answer>`` must equal ``expected``.

    Numbers compare with a 0.005 absolute tolerance (money / counts), text
    with :func:`normalize_query`.  Requiring the tag (instead of searching the
    whole reply) blocks the "dump every candidate value" hack.
    """
    if not isinstance(actual, str) or expected is None:
        return False
    matches = _ANSWER_TAG_RE.findall(actual)
    if not matches:
        return False
    predicted = matches[-1].strip()
    gold = str(expected).strip()
    if not predicted:
        return False
    predicted_number, gold_number = _as_number(predicted), _as_number(gold)
    if predicted_number is not None and gold_number is not None:
        return abs(predicted_number - gold_number) <= 0.005
    return normalize_query(predicted) == normalize_query(gold)


#: Content prefix for a GT tool return that could not be injected (T4-02).
#: The replay env returns ``ok=False`` and records the count instead of silently
#: fabricating a success payload; replay_check treats these as degraded, not as
#: tool-execution errors.
MISSING_RETURN_PREFIX = "Error: ground-truth return missing for tool"


# ---------------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------------


def _safe_json(value: Any) -> str:
    try:
        return canonical_json(value)
    except (TypeError, ValueError):
        return str(value)


def _get_arg(arguments: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    """First present argument among ``names``."""
    for name in names:
        if name in arguments:
            return arguments[name]
    return default


def _dir_entries(state: Mapping[str, Any], directory: str) -> list[str]:
    """List immediate children of ``directory`` from a filesystem snapshot."""
    files = state.get("files") if isinstance(state.get("files"), Mapping) else {}
    dirs = state.get("dirs") if isinstance(state.get("dirs"), list) else []
    entries: set[str] = set()
    for path in list(files.keys()) + [d for d in dirs if isinstance(d, str)]:
        path = str(path)
        if directory != "/" and not path.startswith(directory + "/"):
            continue
        remainder = path[len(directory) + 1 :] if directory != "/" else path.lstrip("/")
        if not remainder:
            continue
        entries.add(remainder.split("/")[0])
    return sorted(entries)


def _norm_path(path: Any) -> str:
    """Normalise a filesystem path used by the filesystem domain."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a non-empty string")
    value = path.strip().replace("\\", "/")
    while "//" in value:
        value = value.replace("//", "/")
    if value.startswith("./"):
        value = value[2:]
    if not value.startswith("/"):
        value = "/" + value
    if len(value) > 1 and value.endswith("/"):
        value = value.rstrip("/")
    return value or "/"


def _to_number(arguments: Mapping[str, Any], *names: str, default: Any = None) -> float:
    raw = _get_arg(arguments, *names, default=default)
    if isinstance(raw, bool) or raw is None:
        raise ValueError(f"missing numeric argument ({'/'.join(names)})")
    if isinstance(raw, (int, float)):
        return float(raw)
    try:
        return float(str(raw).strip())
    except ValueError as exc:
        raise ValueError(f"argument {names[0]!r} must be numeric, got {raw!r}") from exc


def _to_amount(arguments: Mapping[str, Any], *names: str, default: Any = None) -> int:
    value = _to_number(arguments, *names, default=default)
    if value <= 0 or int(value) != value:
        raise ValueError(f"amount must be a positive integer, got {value!r}")
    return int(value)


def deep_equal(expected: Any, actual: Any, *, loose: bool = True) -> bool:
    """Structural equality used by replay matching and ``final_state`` checks.

    ``loose`` (default): strings are stripped/lower-cased, numbers compared
    with a 1e-6 relative tolerance, and dict keys must match exactly (no extra
    keys tolerated).  ``loose=False`` is exact canonical-JSON equality.
    """
    if not loose:
        return _safe_json(expected) == _safe_json(actual)
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.strip().lower() == actual.strip().lower()
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected == actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(expected) - float(actual)) <= 1e-6 * max(1.0, abs(float(expected)))
    if expected is None or actual is None:
        return expected is None and actual is None
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        if set(expected.keys()) != set(actual.keys()):
            return False
        return all(deep_equal(expected[key], actual[key], loose=True) for key in expected)
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            return False
        return all(
            deep_equal(a, b, loose=True) for a, b in zip(list(expected), list(actual))
        )
    return expected == actual


def match_call_group(
    output_calls: list[Mapping[str, Any]],
    gt_calls: list[Mapping[str, Any]],
    *,
    mode: str = "loose",
) -> tuple[bool, str]:
    """Order-insensitive multiset match of one turn's calls against GT.

    Parallel calls in the same turn group are order-insensitive (plan §2).
    Names use exact string equality; arguments use :func:`deep_equal`
    (``mode == "exact"`` for byte-stable comparison).
    """
    if len(output_calls) != len(gt_calls):
        return False, f"expected {len(gt_calls)} call(s), got {len(output_calls)}"
    remaining = list(range(len(gt_calls)))
    for call in output_calls:
        name = str(call.get("name"))
        arguments = call.get("arguments")
        arguments = dict(arguments) if isinstance(arguments, Mapping) else {}
        matched: int | None = None
        for gt_index in remaining:
            gt_call = gt_calls[gt_index]
            if str(gt_call.get("name")) != name:
                continue
            gt_arguments = gt_call.get("arguments")
            gt_arguments = dict(gt_arguments) if isinstance(gt_arguments, Mapping) else {}
            if deep_equal(gt_arguments, arguments, loose=(mode != "exact")):
                matched = gt_index
                break
        if matched is None:
            return False, f"no ground-truth match for call {name!r} arguments={arguments!r}"
        remaining.remove(matched)
    if remaining:
        return False, f"{len(remaining)} ground-truth call(s) not matched"
    return True, "ok"


# ---------------------------------------------------------------------------
# GT-return normalisation (CONTRACTS.md C6)
# ---------------------------------------------------------------------------


def _normalise_return_entry(entry: Any) -> dict[str, Any]:
    """Normalise one return record to ``{"name", "content", "ok"}``."""
    if entry is None:
        return {"name": None, "content": "", "ok": True}
    if isinstance(entry, str):
        return {"name": None, "content": entry, "ok": True}
    if isinstance(entry, Mapping):
        name = entry.get("name")
        content = entry.get("content")
        for key in ("return", "result", "text", "output"):
            if content is None and entry.get(key) is not None:
                content = entry.get(key)
        if content is None:
            content = ""
        if not isinstance(content, str):
            content = _safe_json(content)
        return {
            "name": name if isinstance(name, str) and name.strip() else None,
            "content": content,
            "ok": bool(entry.get("ok", True)),
        }
    if isinstance(entry, (list, tuple)):
        return {"name": None, "content": _safe_json(entry), "ok": True}
    return {"name": None, "content": str(entry), "ok": True}


def _normalise_return_group(group: Any) -> list[dict[str, Any]]:
    """Normalise one per-turn return group (list, name->content map or scalar)."""
    if group is None:
        return []
    if isinstance(group, Mapping):
        entries: list[dict[str, Any]] = []
        for name, value in group.items():
            if isinstance(value, Mapping):
                entry = dict(value)
                entry.setdefault("name", name)
                entries.append(_normalise_return_entry(entry))
            else:
                entries.append(
                    {
                        "name": str(name),
                        "content": value if isinstance(value, str) else _safe_json(value),
                        "ok": True,
                    }
                )
        return entries
    if isinstance(group, str):
        return [_normalise_return_entry(group)]
    if isinstance(group, (list, tuple)):
        return [_normalise_return_entry(item) for item in group]
    return [_normalise_return_entry(group)]


def resolve_gt_returns(
    task: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> tuple[list[list[dict[str, Any]]], str]:
    """Resolve per-turn GT tool returns; return ``(groups, source)``.

    Source is one of :data:`GT_RETURNS_SOURCES` (CONTRACTS.md C6).
    """
    config = config if isinstance(config, Mapping) else {}
    raw = config.get("gt_returns")
    if raw:
        return [_normalise_return_group(group) for group in raw], "config"

    reference = task.get("reference") if isinstance(task.get("reference"), Mapping) else {}
    raw = reference.get("tool_returns")
    if raw:
        return [_normalise_return_group(group) for group in raw], "reference"

    gt_turns = reference.get("gt_turns")
    if isinstance(gt_turns, list) and gt_turns:
        groups: list[list[dict[str, Any]]] = []
        for turn in gt_turns:
            group: Any = None
            if isinstance(turn, Mapping):
                for key in ("tool_results", "results", "returns", "return", "tool_returns"):
                    if turn.get(key) is not None:
                        group = turn.get(key)
                        break
            else:
                group = turn
            groups.append(_normalise_return_group(group))
        if any(groups):
            return groups, "gt_turns"

    return [], "placeholder"


# ---------------------------------------------------------------------------
# goal assertions (stateful env, plan §4.2 A)
# ---------------------------------------------------------------------------


def _resolve_path(root: Any, path: Any) -> tuple[Any, bool]:
    """Resolve a dotted path (optional leading ``state.``) inside a snapshot."""
    if not isinstance(path, str) or not path.strip():
        return None, False
    parts = path.strip().split(".")
    if parts and parts[0] == "state":
        parts = parts[1:]
    current = root
    for part in parts:
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        elif isinstance(current, (list, tuple)) and part.isdigit():
            index = int(part)
            if 0 <= index < len(current):
                current = current[index]
            else:
                return None, False
        else:
            return None, False
    return current, True


def _compare_value(actual: Any, expected: Any, mode: str) -> tuple[bool, str]:
    if mode in ("eq", "exact", "equals", ""):
        ok = actual == expected
        return ok, f"actual={_short(actual)!r} expected={_short(expected)!r}"
    if mode == "approx":
        try:
            ok = abs(float(actual) - float(expected)) <= 1e-6 * max(1.0, abs(float(expected)))
        except (TypeError, ValueError):
            return False, f"non-numeric actual={_short(actual)!r}"
        return ok, f"actual={_short(actual)!r} expected≈{_short(expected)!r}"
    if mode in ("gte", ">="):
        try:
            return float(actual) >= float(expected), f"actual={actual!r} expected>={expected!r}"
        except (TypeError, ValueError):
            return False, f"non-numeric actual={_short(actual)!r}"
    if mode in ("lte", "<="):
        try:
            return float(actual) <= float(expected), f"actual={actual!r} expected<={expected!r}"
        except (TypeError, ValueError):
            return False, f"non-numeric actual={_short(actual)!r}"
    if mode == "contains":
        if isinstance(actual, str):
            return str(expected) in actual, f"actual={_short(actual)!r} expected-substring={_short(expected)!r}"
        if isinstance(actual, (list, tuple)):
            return expected in list(actual), f"actual={_short(actual)!r} expected-member={_short(expected)!r}"
        if isinstance(actual, Mapping):
            return expected in actual, f"actual-keys={list(actual)!r} expected-key={_short(expected)!r}"
        return actual == expected, f"actual={_short(actual)!r}"
    if mode == "length":
        try:
            ok = len(actual) == int(expected)
        except (TypeError, ValueError):
            return False, f"actual={_short(actual)!r} not sized"
        return ok, f"actual-length={len(actual) if hasattr(actual, '__len__') else '?'} expected={expected!r}"
    if mode == "regex":
        try:
            return re.search(str(expected), str(actual)) is not None, f"actual={_short(actual)!r} pattern={expected!r}"
        except re.error as exc:
            return False, f"bad regex {expected!r}: {exc}"
    raise ValueError(f"unknown comparison mode {mode!r}")


def _short(value: Any, limit: int = 120) -> str:
    text = value if isinstance(value, str) else _safe_json(value)
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _run_check(
    kind: str,
    state: Mapping[str, Any],
    path: Any,
    expected: Any,
    mode: str,
    check: Mapping[str, Any],
) -> tuple[bool, str]:
    if kind in ("any_of", "all_of"):
        nested = check.get("checks") or check.get(kind) or []
        if not isinstance(nested, list) or not nested:
            return False, "nested check list is empty"
        results = evaluate_goal(nested, state)
        if kind == "any_of":
            ok = any(item["ok"] for item in results)
        else:
            ok = all(item["ok"] for item in results)
        details = "; ".join(f"{item['name']}={item['ok']}" for item in results)
        return ok, details

    if kind in ("value", "equals", "state_value"):
        actual, resolved = _resolve_path(state, path)
        if not resolved:
            return False, f"path {path!r} not found"
        return _compare_value(actual, expected, mode)

    if kind in ("exists", "not_null"):
        actual, resolved = _resolve_path(state, path)
        return resolved and actual is not None, f"path {path!r} present={resolved}"

    if kind in ("absent", "not_exists"):
        actual, resolved = _resolve_path(state, path)
        return (not resolved) or actual is None, f"path {path!r} present={resolved}"

    if kind == "count":
        actual, resolved = _resolve_path(state, path)
        if not resolved:
            return False, f"path {path!r} not found"
        try:
            ok = len(actual) == int(expected)
        except (TypeError, ValueError):
            return False, f"value at {path!r} is not sized"
        return ok, f"length={len(actual)} expected={expected!r}"

    files = state.get("files") if isinstance(state.get("files"), Mapping) else {}
    dirs = state.get("dirs") if isinstance(state.get("dirs"), list) else []
    dir_set = {str(item) for item in dirs if isinstance(item, str)}

    if kind == "file_exists":
        target = _norm_path(path)
        ok = target in files and target not in dir_set
        return ok, f"file {target!r} exists={ok}"
    if kind == "file_not_exists":
        target = _norm_path(path)
        ok = target not in files
        return ok, f"file {target!r} absent={ok}"
    if kind == "dir_exists":
        target = _norm_path(path)
        return target in dir_set, f"dir {target!r} exists={target in dir_set}"
    if kind == "dir_not_exists":
        target = _norm_path(path)
        return target not in dir_set, f"dir {target!r} absent={target not in dir_set}"
    if kind == "file_content":
        target = _norm_path(path)
        if target not in files:
            return False, f"file {target!r} is missing"
        return _compare_value(files[target], expected, mode)
    if kind == "entries":
        target = _norm_path(path)
        if target not in dir_set and target != "/":
            return False, f"dir {target!r} is missing"
        actual_entries = _dir_entries(state, target)
        if mode == "contains":
            wanted = expected if isinstance(expected, list) else [expected]
            missing = [item for item in wanted if item not in actual_entries]
            ok = not missing
            return ok, f"entries={actual_entries!r} missing={missing!r}"
        if mode == "length":
            try:
                ok = len(actual_entries) == int(expected)
            except (TypeError, ValueError):
                return False, "expected is not an int"
            return ok, f"entries={actual_entries!r} expected-length={expected!r}"
        wanted = expected if isinstance(expected, list) else [expected]
        ok = actual_entries == sorted(str(item) for item in wanted)
        return ok, f"entries={actual_entries!r} expected={sorted(str(item) for item in wanted)!r}"

    raise ValueError(f"unknown goal check kind {kind!r}")


def evaluate_goal(
    checks: list[Mapping[str, Any]] | None,
    state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate terminal-state assertions against a domain state snapshot."""
    results: list[dict[str, Any]] = []
    for index, check in enumerate(checks or []):
        if not isinstance(check, Mapping):
            results.append(
                {"name": f"check[{index}]", "ok": False, "detail": "check must be a mapping"}
            )
            continue
        kind = str(check.get("check") or check.get("type") or "value").strip().lower()
        path = check.get("path")
        if "expected" in check:
            expected = check.get("expected")
        elif "equals" in check:
            expected = check.get("equals")
        elif "value" in check:
            expected = check.get("value")
        else:
            expected = None
        mode = str(check.get("mode") or check.get("compare") or "eq").strip().lower()
        name = str(check.get("name") or f"{kind}:{path}")
        try:
            ok, detail = _run_check(kind, state, path, expected, mode, check)
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        results.append({"name": name, "ok": bool(ok), "detail": str(detail)})
    return results


# ---------------------------------------------------------------------------
# stateful domains
# ---------------------------------------------------------------------------


def _deep_update(target: dict[str, Any], patch: Mapping[str, Any]) -> None:
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _safe_format(template: str, arguments: Mapping[str, Any]) -> str:
    try:
        return template.format(**{str(k): v for k, v in arguments.items()})
    except (KeyError, IndexError, ValueError):
        return template


class BaseDomain:
    """A config-driven tool state machine (BFCL v3 style).

    ``initial_state`` is deep-merged over :meth:`default_state`;
    ``responses`` provides canned outputs for external data tools (the state
    machine cannot know, e.g., today's weather).  Tool names declared in the
    task that the domain does not handle fall back to ``responses`` and then
    to a recoverable ``ok=False`` error.
    """

    name: str = "base"
    HANDLERS: dict[str, str] = {}
    TOOL_DESCRIPTIONS: dict[str, str] = {}

    def __init__(
        self,
        initial_state: Mapping[str, Any] | None = None,
        responses: Mapping[str, Any] | None = None,
    ) -> None:
        self.responses: dict[str, Any] = {}
        self.state: dict[str, Any] = {}
        self.reset(initial_state=initial_state, responses=responses)

    def default_state(self) -> dict[str, Any]:
        return {}

    def reset(
        self,
        *,
        initial_state: Mapping[str, Any] | None = None,
        responses: Mapping[str, Any] | None = None,
    ) -> None:
        self.state = copy.deepcopy(self.default_state())
        if isinstance(initial_state, Mapping) and initial_state:
            _deep_update(self.state, initial_state)
        if responses is not None:
            self.responses = dict(responses)

    def _handlers(self) -> dict[str, Any]:
        return {name: getattr(self, attribute) for name, attribute in self.HANDLERS.items()}

    def handled_tools(self) -> set[str]:
        return set(self.HANDLERS) | set(self.responses)

    def call(self, name: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Execute one tool call; returns ``{"ok": bool, "content": str}``."""
        arguments = dict(arguments) if isinstance(arguments, Mapping) else {}
        handlers = self._handlers()
        if name in handlers:
            try:
                outcome = handlers[name](arguments)
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                return self._error(f"Error: {type(exc).__name__}: {exc}")
            return self._normalise_outcome(outcome)
        if name in self.responses:
            return self._canned(name, arguments)
        return self._error(f"Error: unknown tool {name!r} for domain {self.name!r}")

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self.state)

    def render(self, max_chars: int = 4096) -> str:
        return truncate_text(
            json.dumps(self.snapshot(), ensure_ascii=False, sort_keys=True, indent=2),
            max_chars,
        )[0]

    # -- result helpers ---------------------------------------------------

    @staticmethod
    def _ok(content: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "content": content if isinstance(content, str) else _safe_json(content),
        }

    @staticmethod
    def _error(message: str) -> dict[str, Any]:
        return {"ok": False, "content": message}

    def _canned(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        raw = self.responses.get(name)
        if isinstance(raw, Mapping):
            content: Any = raw.get("content", raw.get("result", raw.get("return", "")))
            ok = bool(raw.get("ok", True))
        else:
            content, ok = raw, True
        if isinstance(content, str):
            content = _safe_format(content, arguments)
        elif content is not None:
            content = _safe_json(content)
        else:
            content = ""
        return {"ok": ok, "content": content}

    def _normalise_outcome(self, outcome: Any) -> dict[str, Any]:
        if isinstance(outcome, Mapping):
            ok = bool(outcome.get("ok", True))
            content = outcome.get("content", outcome.get("result", ""))
            if not isinstance(content, str):
                content = _safe_json(content)
            return {"ok": ok, "content": content}
        if isinstance(outcome, tuple) and len(outcome) == 2:
            return {"ok": bool(outcome[1]), "content": str(outcome[0])}
        return self._ok(outcome)

    def _require_file(self, path: Any) -> str:
        target = _norm_path(path)
        files = self.state.setdefault("files", {})
        if target not in files:
            raise ValueError(f"file not found: {target}")
        return target

    def _require_dir(self, path: Any) -> str:
        target = _norm_path(path)
        dirs = self.state.setdefault("dirs", ["/"])
        if target not in dirs:
            raise ValueError(f"directory not found: {target}")
        return target

    def _require_parent(self, path: str) -> None:
        parent = path.rsplit("/", 1)[0] or "/"
        dirs = self.state.setdefault("dirs", ["/"])
        if parent not in dirs:
            raise ValueError(f"parent directory does not exist: {parent}")


class FilesystemDomain(BaseDomain):
    """Filesystem state machine (``files`` contents + ``dirs`` list)."""

    name = "filesystem"
    HANDLERS = {
        "mkdir": "_mkdir",
        "touch": "_touch",
        "write_file": "_write_file",
        "read_file": "_read_file",
        "list_dir": "_list_dir",
        "delete_file": "_delete_file",
        "delete_dir": "_delete_dir",
        "move_file": "_move_file",
        "copy_file": "_copy_file",
    }

    def default_state(self) -> dict[str, Any]:
        return {"files": {}, "dirs": ["/"]}

    def reset(
        self,
        *,
        initial_state: Mapping[str, Any] | None = None,
        responses: Mapping[str, Any] | None = None,
    ) -> None:
        super().reset(initial_state=initial_state, responses=responses)
        if not isinstance(self.state.get("files"), dict):
            self.state["files"] = {}
        dirs = self.state.get("dirs")
        if not isinstance(dirs, list):
            dirs = ["/"]
            self.state["dirs"] = dirs
        if "/" not in dirs:
            dirs.insert(0, "/")

    # -- handlers ---------------------------------------------------------

    def _mkdir(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        path = _norm_path(_get_arg(arguments, "path", "directory", "dir"))
        self._require_parent(path)
        if path in self.state["dirs"] or path in self.state["files"]:
            raise ValueError(f"path already exists: {path}")
        self.state["dirs"].append(path)
        return self._ok(f"Directory created: {path}")

    def _touch(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        path = _norm_path(_get_arg(arguments, "path", "file", "filename"))
        self._require_parent(path)
        if path in self.state["dirs"]:
            raise ValueError(f"path is a directory: {path}")
        if path in self.state["files"]:
            raise ValueError(f"file already exists: {path}")
        self.state["files"][path] = ""
        return self._ok(f"File created: {path}")

    def _write_file(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        path = _norm_path(_get_arg(arguments, "path", "file", "filename"))
        content = _get_arg(arguments, "content", "text", "data", default="")
        if not isinstance(content, str):
            content = _safe_json(content)
        self._require_parent(path)
        if path in self.state["dirs"]:
            raise ValueError(f"path is a directory: {path}")
        self.state["files"][path] = content
        return self._ok(f"File written: {path} ({len(content)} chars)")

    def _read_file(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        path = _norm_path(_get_arg(arguments, "path", "file", "filename"))
        if path in self.state["dirs"]:
            raise ValueError(f"path is a directory, not a file: {path}")
        if path not in self.state["files"]:
            raise ValueError(f"file not found: {path}")
        return self._ok(self.state["files"][path])

    def _list_dir(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        path = _norm_path(_get_arg(arguments, "path", "directory", "dir", default="/"))
        if path not in self.state["dirs"]:
            raise ValueError(f"directory not found: {path}")
        entries = _dir_entries(self.state, path)
        return self._ok("\n".join(entries) if entries else "(empty)")

    def _delete_file(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        path = _norm_path(_get_arg(arguments, "path", "file", "filename"))
        if path not in self.state["files"]:
            raise ValueError(f"file not found: {path}")
        del self.state["files"][path]
        return self._ok(f"File deleted: {path}")

    def _delete_dir(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        path = _norm_path(_get_arg(arguments, "path", "directory", "dir"))
        if path in ("/", ""):
            raise ValueError("cannot delete root directory")
        if path not in self.state["dirs"]:
            raise ValueError(f"directory not found: {path}")
        children = _dir_entries(self.state, path)
        if children:
            raise ValueError(f"directory not empty: {path} ({len(children)} entries)")
        self.state["dirs"].remove(path)
        return self._ok(f"Directory deleted: {path}")

    def _move_file(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        source = _norm_path(_get_arg(arguments, "source", "src", "path", "from"))
        destination = _norm_path(_get_arg(arguments, "destination", "dest", "to", "target"))
        if source not in self.state["files"]:
            raise ValueError(f"source file not found: {source}")
        if destination in self.state["dirs"] or destination in self.state["files"]:
            raise ValueError(f"destination already exists: {destination}")
        self._require_parent(destination)
        self.state["files"][destination] = self.state["files"].pop(source)
        return self._ok(f"File moved: {source} -> {destination}")

    def _copy_file(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        source = _norm_path(_get_arg(arguments, "source", "src", "path", "from"))
        destination = _norm_path(_get_arg(arguments, "destination", "dest", "to", "target"))
        if source not in self.state["files"]:
            raise ValueError(f"source file not found: {source}")
        if destination in self.state["dirs"] or destination in self.state["files"]:
            raise ValueError(f"destination already exists: {destination}")
        self._require_parent(destination)
        self.state["files"][destination] = self.state["files"][source]
        return self._ok(f"File copied: {source} -> {destination}")


class SocialDomain(BaseDomain):
    """Minimal social state machine (posts / follows / likes / comments)."""

    name = "social"
    HANDLERS = {
        "post": "_post",
        "follow": "_follow",
        "unfollow": "_unfollow",
        "get_following": "_get_following",
        "like": "_like",
        "comment": "_comment",
        "get_post": "_get_post",
        "get_timeline": "_get_timeline",
    }

    def default_state(self) -> dict[str, Any]:
        return {"following": [], "posts": [], "next_post_id": 1}

    def _find_post(self, post_id: Any) -> dict[str, Any]:
        posts = self.state.setdefault("posts", [])
        for post in posts:
            if isinstance(post, Mapping) and str(post.get("id")) == str(post_id):
                return post  # type: ignore[return-value]
        raise ValueError(f"post not found: {post_id!r}")

    def _post(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        content = _get_arg(arguments, "content", "text", "body", default="")
        if not isinstance(content, str):
            content = _safe_json(content)
        if not content.strip():
            raise ValueError("post content must be non-empty")
        post = {
            "id": f"post_{self.state.get('next_post_id', 1)}",
            "content": content,
            "likes": 0,
            "comments": [],
        }
        self.state.setdefault("posts", []).append(post)
        self.state["next_post_id"] = int(self.state.get("next_post_id", 1)) + 1
        return self._ok(post)

    def _follow(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        user = _get_arg(arguments, "user", "username", "name")
        if not isinstance(user, str) or not user.strip():
            raise ValueError("user must be a non-empty string")
        following = self.state.setdefault("following", [])
        if user not in following:
            following.append(user.strip())
        return self._ok(f"Following {user.strip()}")

    def _unfollow(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        user = _get_arg(arguments, "user", "username", "name")
        following = self.state.setdefault("following", [])
        if not isinstance(user, str) or user.strip() not in following:
            raise ValueError(f"not following user: {user!r}")
        following.remove(user.strip())
        return self._ok(f"Unfollowed {user.strip()}")

    def _get_following(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        return self._ok({"following": list(self.state.get("following", []))})

    def _like(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        post = self._find_post(_get_arg(arguments, "post_id", "id", "post"))
        post["likes"] = int(post.get("likes", 0)) + 1
        return self._ok({"post_id": post["id"], "likes": post["likes"]})

    def _comment(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        post = self._find_post(_get_arg(arguments, "post_id", "id", "post"))
        text = _get_arg(arguments, "comment", "text", "content", default="")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("comment must be a non-empty string")
        comments = post.setdefault("comments", [])
        if not isinstance(comments, list):
            comments = []
            post["comments"] = comments
        comments.append(text)
        return self._ok({"post_id": post["id"], "comments": len(comments)})

    def _get_post(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        return self._ok(self._find_post(_get_arg(arguments, "post_id", "id", "post")))

    def _get_timeline(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        return self._ok({"posts": list(self.state.get("posts", []))})


class TradingDomain(BaseDomain):
    """Minimal trading state machine (balance / holdings / prices / history)."""

    name = "trading"
    HANDLERS = {
        "get_balance": "_get_balance",
        "get_holdings": "_get_holdings",
        "get_price": "_get_price",
        "buy": "_buy",
        "sell": "_sell",
        "transfer": "_transfer",
        "deposit": "_deposit",
        "get_history": "_get_history",
    }

    def default_state(self) -> dict[str, Any]:
        return {"balance": 0.0, "holdings": {}, "prices": {}, "history": []}

    def _price(self, symbol: Any) -> tuple[str, float]:
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        symbol = symbol.strip().upper()
        prices = self.state.setdefault("prices", {})
        if symbol not in prices:
            raise ValueError(f"no price available for symbol: {symbol}")
        return symbol, float(prices[symbol])

    def _record(self, action: str, details: Mapping[str, Any]) -> None:
        history = self.state.setdefault("history", [])
        if isinstance(history, list):
            history.append({"action": action, **dict(details)})

    def _get_balance(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        return self._ok({"balance": self.state.get("balance", 0.0)})

    def _get_holdings(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        return self._ok({"holdings": dict(self.state.get("holdings", {}))})

    def _get_price(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        symbol, price = self._price(_get_arg(arguments, "symbol", "ticker", "stock"))
        return self._ok({"symbol": symbol, "price": price})

    def _buy(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        symbol, price = self._price(_get_arg(arguments, "symbol", "ticker", "stock"))
        quantity = _to_amount(arguments, "quantity", "amount", "shares")
        cost = price * quantity
        balance = float(self.state.get("balance", 0.0))
        if cost > balance + 1e-9:
            raise ValueError(
                f"insufficient funds: need {cost:.2f}, have {balance:.2f}"
            )
        self.state["balance"] = balance - cost
        holdings = self.state.setdefault("holdings", {})
        holdings[symbol] = int(holdings.get(symbol, 0)) + quantity
        self._record("buy", {"symbol": symbol, "quantity": quantity, "cost": cost})
        return self._ok(
            {"symbol": symbol, "quantity": quantity, "cost": cost, "balance": self.state["balance"]}
        )

    def _sell(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        symbol, price = self._price(_get_arg(arguments, "symbol", "ticker", "stock"))
        quantity = _to_amount(arguments, "quantity", "amount", "shares")
        holdings = self.state.setdefault("holdings", {})
        held = int(holdings.get(symbol, 0))
        if quantity > held:
            raise ValueError(f"insufficient shares: need {quantity}, hold {held}")
        proceeds = price * quantity
        holdings[symbol] = held - quantity
        self.state["balance"] = float(self.state.get("balance", 0.0)) + proceeds
        self._record("sell", {"symbol": symbol, "quantity": quantity, "proceeds": proceeds})
        return self._ok(
            {
                "symbol": symbol,
                "quantity": quantity,
                "proceeds": proceeds,
                "balance": self.state["balance"],
            }
        )

    def _transfer(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        recipient = _get_arg(arguments, "recipient", "to", "user", "account")
        amount = _to_number(arguments, "amount", "value")
        if amount <= 0:
            raise ValueError(f"transfer amount must be positive, got {amount!r}")
        balance = float(self.state.get("balance", 0.0))
        if amount > balance + 1e-9:
            raise ValueError(f"insufficient funds: need {amount:.2f}, have {balance:.2f}")
        self.state["balance"] = balance - amount
        self._record("transfer", {"recipient": str(recipient), "amount": amount})
        return self._ok({"recipient": str(recipient), "amount": amount, "balance": self.state["balance"]})

    def _deposit(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        amount = _to_number(arguments, "amount", "value")
        if amount <= 0:
            raise ValueError(f"deposit amount must be positive, got {amount!r}")
        self.state["balance"] = float(self.state.get("balance", 0.0)) + amount
        self._record("deposit", {"amount": amount})
        return self._ok({"balance": self.state["balance"]})

    def _get_history(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        history = self.state.get("history", [])
        return self._ok({"history": list(history)[-10:]})


class RetailDomain(BaseDomain):
    """tau-bench style retail back office (users / orders / products).

    Lookup-before-write is enforced by the data, not the handlers: order ids
    and item ids are only discoverable through ``find_user_by_email`` ->
    ``get_user_details`` -> ``get_order_details`` / ``get_product_details``,
    so a policy has to chain reads before it can issue the right write.
    """

    name = "retail"
    HANDLERS = {
        "find_user_by_email": "_find_user_by_email",
        "get_user_details": "_get_user_details",
        "get_order_details": "_get_order_details",
        "get_product_details": "_get_product_details",
        "cancel_pending_order": "_cancel_pending_order",
        "modify_order_address": "_modify_order_address",
        "return_delivered_items": "_return_delivered_items",
        "exchange_delivered_items": "_exchange_delivered_items",
    }

    def default_state(self) -> dict[str, Any]:
        return {"users": {}, "orders": {}, "products": {}}

    def _user(self, user_id: Any) -> dict[str, Any]:
        users = self.state.setdefault("users", {})
        if not isinstance(user_id, str) or user_id not in users:
            raise ValueError(f"user not found: {user_id!r}")
        return users[user_id]

    def _order(self, order_id: Any) -> dict[str, Any]:
        orders = self.state.setdefault("orders", {})
        if not isinstance(order_id, str) or order_id not in orders:
            raise ValueError(f"order not found: {order_id!r}")
        return orders[order_id]

    def _variant(self, item_id: Any) -> tuple[str, dict[str, Any]]:
        for product_id, product in self.state.setdefault("products", {}).items():
            variants = product.get("variants") or {}
            if isinstance(item_id, str) and item_id in variants:
                return product_id, variants[item_id]
        raise ValueError(f"item not found: {item_id!r}")

    @staticmethod
    def _item_ids(arguments: Mapping[str, Any]) -> list[str]:
        raw = _get_arg(arguments, "item_ids", "items")
        if not isinstance(raw, list) or not raw or not all(isinstance(i, str) for i in raw):
            raise ValueError("item_ids must be a non-empty list of item id strings")
        return list(raw)

    def _find_user_by_email(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        email = _get_arg(arguments, "email")
        if not isinstance(email, str) or not email.strip():
            raise ValueError("email must be a non-empty string")
        for user_id, user in self.state.setdefault("users", {}).items():
            if str(user.get("email", "")).lower() == email.strip().lower():
                return self._ok({"user_id": user_id})
        raise ValueError(f"no user with email {email!r}")

    def _get_user_details(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        user = self._user(_get_arg(arguments, "user_id"))
        return self._ok(
            {
                "name": user.get("name"),
                "email": user.get("email"),
                "address": user.get("address"),
                "orders": list(user.get("orders", [])),
            }
        )

    def _get_order_details(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        order_id = _get_arg(arguments, "order_id")
        order = self._order(order_id)
        return self._ok({"order_id": order_id, **copy.deepcopy(order)})

    def _get_product_details(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        product_id = _get_arg(arguments, "product_id")
        products = self.state.setdefault("products", {})
        if not isinstance(product_id, str) or product_id not in products:
            raise ValueError(f"product not found: {product_id!r}")
        return self._ok({"product_id": product_id, **copy.deepcopy(products[product_id])})

    def _cancel_pending_order(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        order_id = _get_arg(arguments, "order_id")
        order = self._order(order_id)
        reason = _get_arg(arguments, "reason")
        if reason not in ("no longer needed", "ordered by mistake"):
            raise ValueError("reason must be 'no longer needed' or 'ordered by mistake'")
        if order.get("status") != "pending":
            raise ValueError(f"order {order_id} is {order.get('status')!r}, only pending orders can be cancelled")
        order["status"] = "cancelled"
        order["cancel_reason"] = reason
        return self._ok({"order_id": order_id, "status": "cancelled"})

    def _modify_order_address(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        order_id = _get_arg(arguments, "order_id")
        order = self._order(order_id)
        address = _get_arg(arguments, "address")
        if not isinstance(address, str) or not address.strip():
            raise ValueError("address must be a non-empty string")
        if order.get("status") != "pending":
            raise ValueError(f"order {order_id} is {order.get('status')!r}, only pending orders can be modified")
        order["address"] = address.strip()
        return self._ok({"order_id": order_id, "address": order["address"]})

    def _check_delivered_items(self, order_id: Any, item_ids: list[str]) -> dict[str, Any]:
        order = self._order(order_id)
        if order.get("status") != "delivered":
            raise ValueError(f"order {order_id} is {order.get('status')!r}, only delivered orders accept this")
        owned = [item.get("item_id") for item in order.get("items", [])]
        missing = [item for item in item_ids if item not in owned]
        if missing:
            raise ValueError(f"items {missing} are not part of order {order_id}")
        return order

    def _return_delivered_items(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        order_id = _get_arg(arguments, "order_id")
        item_ids = self._item_ids(arguments)
        order = self._check_delivered_items(order_id, item_ids)
        refund = round(
            sum(float(i.get("price", 0)) for i in order.get("items", []) if i.get("item_id") in item_ids), 2
        )
        order["status"] = "return requested"
        order["return_items"] = sorted(item_ids)
        return self._ok({"order_id": order_id, "status": order["status"], "refund": refund})

    def _exchange_delivered_items(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        order_id = _get_arg(arguments, "order_id")
        item_ids = self._item_ids(arguments)
        new_ids = _get_arg(arguments, "new_item_ids")
        if not isinstance(new_ids, list) or len(new_ids) != len(item_ids):
            raise ValueError("new_item_ids must be a list with one new item per exchanged item")
        order = self._check_delivered_items(order_id, item_ids)
        for old, new in zip(item_ids, new_ids):
            old_product, _ = self._variant(old)
            new_product, variant = self._variant(new)
            if old_product != new_product:
                raise ValueError(f"{new} is not a variant of the same product as {old}")
            if not variant.get("available", False):
                raise ValueError(f"item {new} is out of stock")
        order["status"] = "exchange requested"
        order["exchange"] = {old: new for old, new in zip(item_ids, new_ids)}
        return self._ok({"order_id": order_id, "status": order["status"]})


def _text_arg(
    arguments: Mapping[str, Any],
    *names: str,
    default: Any = None,
    max_chars: int | None = None,
) -> str:
    raw = _get_arg(arguments, *names, default=default)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{names[0]} must be a non-empty string")
    text = raw.strip()
    if max_chars is not None and len(text) > max_chars:
        raise ValueError(f"{names[0]} exceeds the {max_chars}-character limit")
    return text


def _finite_number(arguments: Mapping[str, Any], *names: str, default: Any = None) -> float:
    value = _to_number(arguments, *names, default=default)
    if not math.isfinite(value):
        raise ValueError(f"{names[0]} must be finite")
    return value


def _iso_date(value: Any, field: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO date string")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO date string") from exc


class VehicleDomain(BaseDomain):
    """Metric-backed vehicle controls with explicit unit conversion."""

    name = "vehicle"
    HANDLERS = {
        "check_drive_panel": "_check_drive_panel",
        "measure_speed": "_measure_speed",
        "dial_road_speed": "_dial_road_speed",
        "nudge_road_speed": "_nudge_road_speed",
        "check_gear": "_check_gear",
        "select_gear": "_select_gear",
        "check_lights": "_check_lights",
        "set_light_mode": "_set_light_mode",
        "check_climate": "_check_climate",
        "configure_climate": "_configure_climate",
    }
    TOOL_DESCRIPTIONS = {
        "check_drive_panel": "Summarize current motion, lighting, and cabin controls.",
        "measure_speed": "Report road speed in kilometres per hour or miles per hour.",
        "dial_road_speed": "Set an absolute road speed after validating units and gear.",
        "nudge_road_speed": "Apply a signed speed delta without changing the selected gear.",
        "check_gear": "Read the currently engaged transmission position.",
        "select_gear": "Engage P, R, N, or D while the vehicle is stationary.",
        "check_lights": "Read the exterior lighting mode.",
        "set_light_mode": "Choose off, low, high, or automatic exterior lighting.",
        "check_climate": "Read cabin temperature, fan level, and air-conditioning state.",
        "configure_climate": "Update one or more cabin climate settings atomically.",
    }

    def default_state(self) -> dict[str, Any]:
        return {
            "speed_kmh": 0.0,
            "gear": "P",
            "lights": "off",
            "climate": {"temperature_c": 22.0, "fan": 2, "ac_on": False},
        }

    @staticmethod
    def _speed_unit(arguments: Mapping[str, Any]) -> str:
        raw = arguments.get("unit", "km/h")
        if not isinstance(raw, str):
            raise ValueError("unit must be 'km/h' or 'mph'")
        unit = {"kmh": "km/h", "kph": "km/h", "km/h": "km/h", "mph": "mph"}.get(
            raw.strip().lower()
        )
        if unit is None:
            raise ValueError("unit must be 'km/h' or 'mph'")
        return unit

    @staticmethod
    def _to_kmh(value: float, unit: str) -> float:
        return value * 1.609344 if unit == "mph" else value

    def _speed_payload(self, unit: str) -> dict[str, Any]:
        speed_kmh = float(self.state.get("speed_kmh", 0.0))
        speed = speed_kmh / 1.609344 if unit == "mph" else speed_kmh
        return {"speed": round(speed, 2), "unit": unit, "speed_kmh": speed_kmh}

    def _check_drive_panel(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        return self._ok(
            {
                "speed_kmh": round(float(self.state["speed_kmh"]), 2),
                "speed_mph": round(float(self.state["speed_kmh"]) / 1.609344, 2),
                "gear": self.state["gear"],
                "lights": self.state["lights"],
                "climate": copy.deepcopy(self.state["climate"]),
            }
        )

    def _measure_speed(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        return self._ok(self._speed_payload(self._speed_unit(arguments)))

    def _dial_road_speed(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        unit = self._speed_unit(arguments)
        value = _finite_number(arguments, "speed", "value")
        speed_kmh = self._to_kmh(value, unit)
        if not 0.0 <= speed_kmh <= 240.0:
            raise ValueError("speed must be between 0 and 240 km/h")
        if speed_kmh > 0.0 and self.state.get("gear") != "D":
            raise ValueError("positive speed requires gear D")
        self.state["speed_kmh"] = round(speed_kmh, 6)
        return self._ok(self._speed_payload(unit))

    def _nudge_road_speed(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        unit = self._speed_unit(arguments)
        delta = _finite_number(arguments, "delta", "change")
        speed_kmh = float(self.state.get("speed_kmh", 0.0)) + self._to_kmh(delta, unit)
        if not 0.0 <= speed_kmh <= 240.0:
            raise ValueError("adjusted speed must be between 0 and 240 km/h")
        if speed_kmh > 0.0 and self.state.get("gear") != "D":
            raise ValueError("positive speed requires gear D")
        self.state["speed_kmh"] = round(speed_kmh, 6)
        return self._ok(self._speed_payload(unit))

    def _check_gear(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        return self._ok({"gear": self.state["gear"]})

    def _select_gear(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        gear = _text_arg(arguments, "gear").upper()
        if gear not in {"P", "R", "N", "D"}:
            raise ValueError("gear must be one of P, R, N, D")
        if gear != self.state.get("gear") and float(self.state.get("speed_kmh", 0.0)) > 0.0:
            raise ValueError("stop the vehicle before changing gear")
        self.state["gear"] = gear
        return self._ok({"gear": gear})

    def _check_lights(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        return self._ok({"lights": self.state["lights"]})

    def _set_light_mode(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        mode = _text_arg(arguments, "mode", "lights").lower()
        if mode not in {"off", "low", "high", "auto"}:
            raise ValueError("light mode must be off, low, high, or auto")
        self.state["lights"] = mode
        return self._ok({"lights": mode})

    def _check_climate(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        return self._ok(copy.deepcopy(self.state["climate"]))

    def _configure_climate(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        has_temperature = "temperature_c" in arguments or "temperature" in arguments
        has_fan = "fan" in arguments or "fan_speed" in arguments
        has_ac = "ac_on" in arguments
        if not (has_temperature or has_fan or has_ac):
            raise ValueError("provide at least one climate setting")
        updated = copy.deepcopy(self.state["climate"])
        if has_temperature:
            temperature = _finite_number(arguments, "temperature_c", "temperature")
            if not 16.0 <= temperature <= 30.0:
                raise ValueError("temperature_c must be between 16 and 30")
            updated["temperature_c"] = temperature
        if has_fan:
            fan = _finite_number(arguments, "fan", "fan_speed")
            if fan < 0 or fan > 7 or int(fan) != fan:
                raise ValueError("fan must be an integer between 0 and 7")
            updated["fan"] = int(fan)
        if has_ac:
            ac_on = arguments["ac_on"]
            if not isinstance(ac_on, bool):
                raise ValueError("ac_on must be a boolean")
            updated["ac_on"] = ac_on
        self.state["climate"] = updated
        return self._ok(copy.deepcopy(updated))


class TravelDomain(BaseDomain):
    """Flight and lodging inventory with dated, budget-aware reservations."""

    name = "travel"
    HANDLERS = {
        "scan_journeys": "_scan_journeys",
        "show_journey": "_show_journey",
        "scan_lodging": "_scan_lodging",
        "show_lodging": "_show_lodging",
        "exchange_trip_money": "_exchange_trip_money",
        "hold_flight_seats": "_hold_flight_seats",
        "hold_hotel_rooms": "_hold_hotel_rooms",
        "review_trip_bookings": "_review_trip_bookings",
        "show_trip_booking": "_show_trip_booking",
        "release_trip_booking": "_release_trip_booking",
    }
    TOOL_DESCRIPTIONS = {
        "scan_journeys": "Find dated flight options with enough seats for a party.",
        "show_journey": "Inspect one flight without reserving its inventory.",
        "scan_lodging": "Find rooms for a city and date range, including a price estimate.",
        "show_lodging": "Inspect one property's rates and room inventory.",
        "exchange_trip_money": "Convert a positive amount using the stored travel rates.",
        "hold_flight_seats": "Reserve a dated flight for a party within an optional budget.",
        "hold_hotel_rooms": "Reserve enough rooms for guests and nights within a budget.",
        "review_trip_bookings": "List reservations, optionally narrowed to one traveller.",
        "show_trip_booking": "Inspect the current state of one reservation.",
        "release_trip_booking": "Cancel one active reservation and restore only its inventory.",
    }

    def default_state(self) -> dict[str, Any]:
        return {
            "flights": {
                "F100": {
                    "origin": "SEA",
                    "destination": "SFO",
                    "departure_date": "2026-12-01",
                    "price_usd": 180.0,
                    "seats_available": 8,
                },
                "F200": {
                    "origin": "SEA",
                    "destination": "LAX",
                    "departure_date": "2026-12-01",
                    "price_usd": 140.0,
                    "seats_available": 5,
                },
            },
            "hotels": {
                "H10": {
                    "city": "San Francisco",
                    "nightly_usd": 160.0,
                    "room_capacity": 2,
                    "rooms_available": 3,
                },
                "H20": {
                    "city": "Los Angeles",
                    "nightly_usd": 120.0,
                    "room_capacity": 2,
                    "rooms_available": 4,
                },
            },
            "exchange_rates": {"USD": 1.0, "EUR": 1.1, "CNY": 0.14},
            "bookings": {},
            "next_booking_id": 1,
        }

    def _flight(self, flight_id: Any) -> tuple[str, dict[str, Any]]:
        key = str(flight_id).strip().upper()
        flight = self.state["flights"].get(key)
        if not isinstance(flight, dict):
            raise ValueError(f"flight not found: {flight_id!r}")
        return key, flight

    def _hotel(self, hotel_id: Any) -> tuple[str, dict[str, Any]]:
        key = str(hotel_id).strip().upper()
        hotel = self.state["hotels"].get(key)
        if not isinstance(hotel, dict):
            raise ValueError(f"hotel not found: {hotel_id!r}")
        return key, hotel

    def _booking(self, booking_id: Any) -> tuple[str, dict[str, Any]]:
        key = str(booking_id).strip().upper()
        booking = self.state["bookings"].get(key)
        if not isinstance(booking, dict):
            raise ValueError(f"booking not found: {booking_id!r}")
        return key, booking

    def _rate(self, currency: Any) -> tuple[str, float]:
        if not isinstance(currency, str) or not currency.strip():
            raise ValueError("currency must be a non-empty code")
        code = currency.strip().upper()
        raw = self.state["exchange_rates"].get(code)
        if isinstance(raw, bool) or raw is None:
            raise ValueError(f"unsupported currency: {code}")
        try:
            rate = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid exchange rate for {code}") from exc
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError(f"invalid exchange rate for {code}")
        return code, rate

    def _budget_quote(
        self, arguments: Mapping[str, Any], total_usd: float
    ) -> tuple[float, str, float | None]:
        currency, rate = self._rate(arguments.get("currency", "USD"))
        raw_budget = _get_arg(arguments, "budget", "max_budget")
        budget = None
        if raw_budget is not None:
            budget = _finite_number(arguments, "budget", "max_budget")
            if budget <= 0:
                raise ValueError("budget must be positive")
        cost = total_usd / rate
        if budget is not None and cost > budget + 1e-9:
            raise ValueError(
                f"booking cost {cost:.2f} {currency} exceeds budget {budget:.2f} {currency}"
            )
        return round(cost, 2), currency, budget

    def _new_booking_id(self) -> str:
        return f"B-{int(self.state['next_booking_id'])}"

    def _scan_journeys(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        origin = _text_arg(arguments, "origin").upper()
        destination = _text_arg(arguments, "destination").upper()
        departure_date = _iso_date(arguments.get("departure_date"), "departure_date").isoformat()
        passengers = _to_amount(arguments, "passengers", default=1)
        matches = []
        for flight_id, flight in self.state["flights"].items():
            if (
                flight["origin"] == origin
                and flight["destination"] == destination
                and flight["departure_date"] == departure_date
                and int(flight["seats_available"]) >= passengers
            ):
                matches.append({"flight_id": flight_id, **copy.deepcopy(flight)})
        return self._ok(matches)

    def _show_journey(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        flight_id, flight = self._flight(arguments.get("flight_id"))
        return self._ok({"flight_id": flight_id, **copy.deepcopy(flight)})

    def _scan_lodging(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        city = _text_arg(arguments, "city").casefold()
        check_in = _iso_date(arguments.get("check_in"), "check_in")
        check_out = _iso_date(arguments.get("check_out"), "check_out")
        if check_out <= check_in:
            raise ValueError("check_out must be after check_in")
        guests = _to_amount(arguments, "guests", default=1)
        nights = (check_out - check_in).days
        matches = []
        for hotel_id, hotel in self.state["hotels"].items():
            rooms = (guests + int(hotel["room_capacity"]) - 1) // int(hotel["room_capacity"])
            if hotel["city"].casefold() == city and int(hotel["rooms_available"]) >= rooms:
                matches.append(
                    {
                        "hotel_id": hotel_id,
                        **copy.deepcopy(hotel),
                        "nights": nights,
                        "rooms_needed": rooms,
                        "estimated_total_usd": round(float(hotel["nightly_usd"]) * nights * rooms, 2),
                    }
                )
        return self._ok(matches)

    def _show_lodging(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        hotel_id, hotel = self._hotel(arguments.get("hotel_id"))
        return self._ok({"hotel_id": hotel_id, **copy.deepcopy(hotel)})

    def _exchange_trip_money(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        amount = _finite_number(arguments, "amount")
        if amount <= 0:
            raise ValueError("amount must be positive")
        source, source_rate = self._rate(arguments.get("from_currency", "USD"))
        target, target_rate = self._rate(arguments.get("to_currency", "USD"))
        converted = amount * source_rate / target_rate
        return self._ok(
            {
                "from_currency": source,
                "to_currency": target,
                "amount": amount,
                "converted": round(converted, 2),
                "rate": round(source_rate / target_rate, 6),
            }
        )

    def _hold_flight_seats(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        flight_id, flight = self._flight(arguments.get("flight_id"))
        booked_by = _text_arg(arguments, "booked_by", "traveller", "traveler")
        passengers = _to_amount(arguments, "passengers")
        departure_date = _iso_date(arguments.get("departure_date"), "departure_date").isoformat()
        if departure_date != flight["departure_date"]:
            raise ValueError("departure_date does not match the selected flight")
        if passengers > int(flight["seats_available"]):
            raise ValueError("not enough seats remain on the selected flight")
        total_usd = round(float(flight["price_usd"]) * passengers, 2)
        cost, currency, budget = self._budget_quote(arguments, total_usd)
        booking_id = self._new_booking_id()
        booking = {
            "kind": "flight",
            "flight_id": flight_id,
            "booked_by": booked_by,
            "passengers": passengers,
            "departure_date": departure_date,
            "total_usd": total_usd,
            "cost": cost,
            "currency": currency,
            "budget": budget,
            "status": "confirmed",
        }
        flight["seats_available"] = int(flight["seats_available"]) - passengers
        self.state["bookings"][booking_id] = booking
        self.state["next_booking_id"] = int(self.state["next_booking_id"]) + 1
        return self._ok({"booking_id": booking_id, **copy.deepcopy(booking)})

    def _hold_hotel_rooms(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        hotel_id, hotel = self._hotel(arguments.get("hotel_id"))
        booked_by = _text_arg(arguments, "booked_by", "guest")
        guests = _to_amount(arguments, "guests")
        check_in = _iso_date(arguments.get("check_in"), "check_in")
        check_out = _iso_date(arguments.get("check_out"), "check_out")
        if check_out <= check_in:
            raise ValueError("check_out must be after check_in")
        nights = (check_out - check_in).days
        rooms = (guests + int(hotel["room_capacity"]) - 1) // int(hotel["room_capacity"])
        if rooms > int(hotel["rooms_available"]):
            raise ValueError("not enough rooms remain for this party")
        total_usd = round(float(hotel["nightly_usd"]) * nights * rooms, 2)
        cost, currency, budget = self._budget_quote(arguments, total_usd)
        booking_id = self._new_booking_id()
        booking = {
            "kind": "hotel",
            "hotel_id": hotel_id,
            "booked_by": booked_by,
            "guests": guests,
            "rooms": rooms,
            "check_in": check_in.isoformat(),
            "check_out": check_out.isoformat(),
            "nights": nights,
            "total_usd": total_usd,
            "cost": cost,
            "currency": currency,
            "budget": budget,
            "status": "confirmed",
        }
        hotel["rooms_available"] = int(hotel["rooms_available"]) - rooms
        self.state["bookings"][booking_id] = booking
        self.state["next_booking_id"] = int(self.state["next_booking_id"]) + 1
        return self._ok({"booking_id": booking_id, **copy.deepcopy(booking)})

    def _review_trip_bookings(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        traveller = arguments.get("booked_by", arguments.get("traveller"))
        if traveller is not None and (not isinstance(traveller, str) or not traveller.strip()):
            raise ValueError("booked_by must be a non-empty string")
        matches = []
        for booking_id, booking in self.state["bookings"].items():
            if traveller is not None and booking["booked_by"].casefold() != traveller.strip().casefold():
                continue
            matches.append({"booking_id": booking_id, **copy.deepcopy(booking)})
        return self._ok(matches)

    def _show_trip_booking(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        booking_id, booking = self._booking(arguments.get("booking_id"))
        return self._ok({"booking_id": booking_id, **copy.deepcopy(booking)})

    def _release_trip_booking(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        booking_id, booking = self._booking(arguments.get("booking_id"))
        if booking["status"] == "cancelled":
            raise ValueError(f"booking {booking_id} is already cancelled")
        if booking["kind"] == "flight":
            _, inventory = self._flight(booking["flight_id"])
            inventory["seats_available"] = int(inventory["seats_available"]) + int(booking["passengers"])
        elif booking["kind"] == "hotel":
            _, inventory = self._hotel(booking["hotel_id"])
            inventory["rooms_available"] = int(inventory["rooms_available"]) + int(booking["rooms"])
        else:
            raise ValueError(f"unsupported booking kind: {booking['kind']!r}")
        booking["status"] = "cancelled"
        return self._ok({"booking_id": booking_id, "status": "cancelled"})


class MicroblogDomain(BaseDomain):
    """Authenticated drafts, publishing, reactions, and subscriptions."""

    name = "microblog"
    HANDLERS = {
        "open_author_session": "_open_author_session",
        "close_author_session": "_close_author_session",
        "current_author": "_current_author",
        "stash_post_draft": "_stash_post_draft",
        "list_author_drafts": "_list_author_drafts",
        "rewrite_post_draft": "_rewrite_post_draft",
        "discard_post_draft": "_discard_post_draft",
        "publish_stashed_post": "_publish_stashed_post",
        "appreciate_micropost": "_appreciate_micropost",
        "subscribe_to_author": "_subscribe_to_author",
        "read_microblog_feed": "_read_microblog_feed",
    }
    TOOL_DESCRIPTIONS = {
        "open_author_session": "Authenticate one author before using account-scoped actions.",
        "close_author_session": "End the active author session.",
        "current_author": "Report whether an author session is active.",
        "stash_post_draft": "Save a short private draft for the authenticated author.",
        "list_author_drafts": "List only drafts owned by the authenticated author.",
        "rewrite_post_draft": "Replace the text of an owned, unpublished draft.",
        "discard_post_draft": "Delete one owned draft without publishing it.",
        "publish_stashed_post": "Turn an owned draft into a public micropost.",
        "appreciate_micropost": "Add the current author's single appreciation to a post.",
        "subscribe_to_author": "Follow another known author without changing their account.",
        "read_microblog_feed": "Read public posts, optionally filtered to one author.",
    }

    def default_state(self) -> dict[str, Any]:
        return {
            "accounts": {
                "alice": {"auth_key": "orchid", "following": []},
                "bob": {"auth_key": "cobalt", "following": []},
            },
            "active_user": None,
            "drafts": {},
            "posts": {
                "M-1": {"author": "bob", "text": "Harbour sketches", "liked_by": []},
                "M-2": {"author": "bob", "text": "Weekend train notes", "liked_by": []},
            },
            "next_draft_id": 1,
            "next_post_id": 3,
        }

    @staticmethod
    def _handle(value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("handle must be a non-empty string")
        return value.strip().lower()

    def _active_author(self) -> str:
        active = self.state.get("active_user")
        if active not in self.state["accounts"]:
            raise ValueError("authentication is required")
        return str(active)

    def _draft_for_author(self, draft_id: Any, author: str) -> tuple[str, dict[str, Any]]:
        key = str(draft_id).strip().upper()
        draft = self.state["drafts"].get(key)
        if not isinstance(draft, dict):
            raise ValueError(f"draft not found: {draft_id!r}")
        if draft["author"] != author:
            raise ValueError("draft belongs to another author")
        return key, draft

    def _post(self, post_id: Any) -> tuple[str, dict[str, Any]]:
        key = str(post_id).strip().upper()
        post = self.state["posts"].get(key)
        if not isinstance(post, dict):
            raise ValueError(f"micropost not found: {post_id!r}")
        return key, post

    def _open_author_session(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        handle = self._handle(arguments.get("handle", arguments.get("username")))
        auth_key = _text_arg(arguments, "auth_key", "token")
        if self.state.get("active_user") is not None:
            raise ValueError("close the current author session before signing in again")
        account = self.state["accounts"].get(handle)
        if not isinstance(account, dict) or account.get("auth_key") != auth_key:
            raise ValueError("invalid author credentials")
        self.state["active_user"] = handle
        return self._ok({"authenticated": True, "handle": handle})

    def _close_author_session(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        handle = self._active_author()
        self.state["active_user"] = None
        return self._ok({"authenticated": False, "handle": handle})

    def _current_author(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        active = self.state.get("active_user")
        return self._ok({"authenticated": active is not None, "handle": active})

    def _stash_post_draft(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        author = self._active_author()
        text = _text_arg(arguments, "text", "content", max_chars=280)
        draft_id = f"D-{int(self.state['next_draft_id'])}"
        self.state["drafts"][draft_id] = {"author": author, "text": text}
        self.state["next_draft_id"] = int(self.state["next_draft_id"]) + 1
        return self._ok({"draft_id": draft_id, "author": author, "text": text})

    def _list_author_drafts(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        author = self._active_author()
        drafts = [
            {"draft_id": draft_id, **copy.deepcopy(draft)}
            for draft_id, draft in self.state["drafts"].items()
            if draft["author"] == author
        ]
        return self._ok(drafts)

    def _rewrite_post_draft(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        author = self._active_author()
        draft_id, draft = self._draft_for_author(arguments.get("draft_id"), author)
        text = _text_arg(arguments, "text", "content", max_chars=280)
        draft["text"] = text
        return self._ok({"draft_id": draft_id, "author": author, "text": text})

    def _discard_post_draft(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        author = self._active_author()
        draft_id, _ = self._draft_for_author(arguments.get("draft_id"), author)
        del self.state["drafts"][draft_id]
        return self._ok({"draft_id": draft_id, "discarded": True})

    def _publish_stashed_post(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        author = self._active_author()
        draft_id, draft = self._draft_for_author(arguments.get("draft_id"), author)
        post_id = f"M-{int(self.state['next_post_id'])}"
        post = {"author": author, "text": draft["text"], "liked_by": []}
        self.state["posts"][post_id] = post
        del self.state["drafts"][draft_id]
        self.state["next_post_id"] = int(self.state["next_post_id"]) + 1
        return self._ok({"post_id": post_id, **copy.deepcopy(post)})

    def _appreciate_micropost(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        author = self._active_author()
        post_id, post = self._post(arguments.get("post_id"))
        if author in post["liked_by"]:
            raise ValueError("author already appreciated this micropost")
        post["liked_by"].append(author)
        return self._ok({"post_id": post_id, "appreciations": len(post["liked_by"])})

    def _subscribe_to_author(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        author = self._active_author()
        target = self._handle(arguments.get("handle", "target"))
        if target == author:
            raise ValueError("an author cannot follow their own account")
        if target not in self.state["accounts"]:
            raise ValueError(f"author not found: {target}")
        following = self.state["accounts"][author]["following"]
        if target in following:
            raise ValueError(f"already following {target}")
        following.append(target)
        return self._ok({"handle": author, "following": target})

    def _read_microblog_feed(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        self._active_author()
        requested = arguments.get("author")
        author = self._handle(requested) if requested is not None else None
        if author is not None and author not in self.state["accounts"]:
            raise ValueError(f"author not found: {author}")
        posts = [
            {"post_id": post_id, **copy.deepcopy(post)}
            for post_id, post in self.state["posts"].items()
            if author is None or post["author"] == author
        ]
        return self._ok(posts)


class MessagingDomain(BaseDomain):
    """Login-scoped contacts and private message delivery/read state."""

    name = "messaging"
    HANDLERS = {
        "enter_chat_account": "_enter_chat_account",
        "leave_chat_account": "_leave_chat_account",
        "chat_session_owner": "_chat_session_owner",
        "find_chat_people": "_find_chat_people",
        "send_chat_note": "_send_chat_note",
        "fetch_chat_inbox": "_fetch_chat_inbox",
        "fetch_chat_outbox": "_fetch_chat_outbox",
        "mark_chat_note_read": "_mark_chat_note_read",
        "show_chat_note": "_show_chat_note",
    }
    TOOL_DESCRIPTIONS = {
        "enter_chat_account": "Open a messaging session with an account-specific passcode.",
        "leave_chat_account": "Close the current messaging session.",
        "chat_session_owner": "Report the account attached to the current session.",
        "find_chat_people": "Search only contacts visible to the signed-in account.",
        "send_chat_note": "Deliver private text to a known contact.",
        "fetch_chat_inbox": "Receive messages addressed to the signed-in account.",
        "fetch_chat_outbox": "Review messages sent by the signed-in account.",
        "mark_chat_note_read": "Mark one received message read; senders cannot mark it.",
        "show_chat_note": "Inspect a message only when the current account is a participant.",
    }

    def default_state(self) -> dict[str, Any]:
        return {
            "accounts": {
                "alice": {"passcode": "maple", "contacts": ["bob", "carol"]},
                "bob": {"passcode": "river", "contacts": ["alice", "carol"]},
                "carol": {"passcode": "ember", "contacts": ["alice", "bob"]},
            },
            "active_user": None,
            "messages": {
                "M-1": {
                    "sender": "bob",
                    "recipient": "alice",
                    "body": "Draft is ready",
                    "read": False,
                },
                "M-2": {
                    "sender": "carol",
                    "recipient": "bob",
                    "body": "Platform booking confirmed",
                    "read": False,
                },
            },
            "next_message_id": 3,
        }

    @staticmethod
    def _username(value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("username must be a non-empty string")
        return value.strip().lower()

    def _active_user(self) -> str:
        active = self.state.get("active_user")
        if active not in self.state["accounts"]:
            raise ValueError("login is required")
        return str(active)

    def _message(self, message_id: Any) -> tuple[str, dict[str, Any]]:
        key = str(message_id).strip().upper()
        message = self.state["messages"].get(key)
        if not isinstance(message, dict):
            raise ValueError(f"message not found: {message_id!r}")
        return key, message

    def _enter_chat_account(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        username = self._username(arguments.get("username", "handle"))
        passcode = _text_arg(arguments, "passcode", "password")
        if self.state.get("active_user") is not None:
            raise ValueError("leave the current chat session before logging in again")
        account = self.state["accounts"].get(username)
        if not isinstance(account, dict) or account.get("passcode") != passcode:
            raise ValueError("invalid messaging credentials")
        self.state["active_user"] = username
        return self._ok({"logged_in": True, "username": username})

    def _leave_chat_account(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        username = self._active_user()
        self.state["active_user"] = None
        return self._ok({"logged_in": False, "username": username})

    def _chat_session_owner(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        active = self.state.get("active_user")
        return self._ok({"logged_in": active is not None, "username": active})

    def _find_chat_people(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        username = self._active_user()
        query = arguments.get("query", "")
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        needle = query.strip().casefold()
        contacts = [
            contact
            for contact in self.state["accounts"][username]["contacts"]
            if needle in contact.casefold()
        ]
        return self._ok(contacts)

    def _send_chat_note(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        sender = self._active_user()
        recipient = self._username(arguments.get("recipient", "to"))
        body = _text_arg(arguments, "body", "text", max_chars=1000)
        if recipient not in self.state["accounts"]:
            raise ValueError(f"recipient not found: {recipient}")
        if recipient not in self.state["accounts"][sender]["contacts"]:
            raise ValueError(f"recipient is not in {sender}'s contacts")
        message_id = f"M-{int(self.state['next_message_id'])}"
        message = {"sender": sender, "recipient": recipient, "body": body, "read": False}
        self.state["messages"][message_id] = message
        self.state["next_message_id"] = int(self.state["next_message_id"]) + 1
        return self._ok({"message_id": message_id, **copy.deepcopy(message)})

    def _message_list(self, username: str, field: str, unread_only: bool = False) -> list[dict[str, Any]]:
        return [
            {"message_id": message_id, **copy.deepcopy(message)}
            for message_id, message in self.state["messages"].items()
            if message[field] == username and (not unread_only or not message["read"])
        ]

    @staticmethod
    def _unread_flag(arguments: Mapping[str, Any]) -> bool:
        value = arguments.get("unread_only", False)
        if not isinstance(value, bool):
            raise ValueError("unread_only must be a boolean")
        return value

    def _fetch_chat_inbox(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        username = self._active_user()
        unread_only = self._unread_flag(arguments)
        return self._ok(self._message_list(username, "recipient", unread_only))

    def _fetch_chat_outbox(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        username = self._active_user()
        return self._ok(self._message_list(username, "sender"))

    def _mark_chat_note_read(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        username = self._active_user()
        message_id, message = self._message(arguments.get("message_id"))
        if message["recipient"] != username:
            raise ValueError("only the recipient can mark a message read")
        already_read = bool(message["read"])
        message["read"] = True
        return self._ok({"message_id": message_id, "read": True, "already_read": already_read})

    def _show_chat_note(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        username = self._active_user()
        message_id, message = self._message(arguments.get("message_id"))
        if username not in {message["sender"], message["recipient"]}:
            raise ValueError("current account is not a participant in this message")
        return self._ok({"message_id": message_id, **copy.deepcopy(message)})


class TicketingDomain(BaseDomain):
    """Support ticket lifecycle with validated edits and resolution metadata."""

    name = "ticketing"
    HANDLERS = {
        "file_support_ticket": "_file_support_ticket",
        "show_support_ticket": "_show_support_ticket",
        "search_support_tickets": "_search_support_tickets",
        "revise_support_ticket": "_revise_support_ticket",
        "assign_support_ticket": "_assign_support_ticket",
        "append_ticket_note": "_append_ticket_note",
        "close_support_ticket": "_close_support_ticket",
        "reopen_support_ticket": "_reopen_support_ticket",
    }
    TOOL_DESCRIPTIONS = {
        "file_support_ticket": "Create a structured support request with an initial priority.",
        "show_support_ticket": "Inspect every public field of one support request.",
        "search_support_tickets": "Find requests by exact workflow state or requester.",
        "revise_support_ticket": "Atomically edit fields on an unresolved request.",
        "assign_support_ticket": "Assign an open request to a known support agent.",
        "append_ticket_note": "Add an attributed note without altering other requests.",
        "close_support_ticket": "Resolve a request with an agent and resolution summary.",
        "reopen_support_ticket": "Return a resolved request to open and clear its resolution.",
    }

    def default_state(self) -> dict[str, Any]:
        return {
            "agents": ["sam", "riley"],
            "tickets": {
                "T-1000": {
                    "title": "Printer offline",
                    "details": "The second-floor printer does not respond.",
                    "requester": "alice",
                    "priority": "normal",
                    "status": "open",
                    "assignee": None,
                    "resolution": None,
                    "resolved_by": None,
                    "notes": [],
                }
            },
            "next_ticket_id": 1001,
        }

    def _ticket(self, ticket_id: Any) -> tuple[str, dict[str, Any]]:
        key = str(ticket_id).strip().upper()
        ticket = self.state["tickets"].get(key)
        if not isinstance(ticket, dict):
            raise ValueError(f"ticket not found: {ticket_id!r}")
        return key, ticket

    @staticmethod
    def _priority(value: Any) -> str:
        if not isinstance(value, str) or value.strip().lower() not in {"low", "normal", "high", "urgent"}:
            raise ValueError("priority must be low, normal, high, or urgent")
        return value.strip().lower()

    def _file_support_ticket(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        title = _text_arg(arguments, "title", max_chars=120)
        details = _text_arg(arguments, "details", "description", max_chars=2000)
        requester = _text_arg(arguments, "requester")
        priority = self._priority(arguments.get("priority", "normal"))
        ticket_id = f"T-{int(self.state['next_ticket_id'])}"
        ticket = {
            "title": title,
            "details": details,
            "requester": requester,
            "priority": priority,
            "status": "open",
            "assignee": None,
            "resolution": None,
            "resolved_by": None,
            "notes": [],
        }
        self.state["tickets"][ticket_id] = ticket
        self.state["next_ticket_id"] = int(self.state["next_ticket_id"]) + 1
        return self._ok({"ticket_id": ticket_id, **copy.deepcopy(ticket)})

    def _show_support_ticket(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        ticket_id, ticket = self._ticket(arguments.get("ticket_id"))
        return self._ok({"ticket_id": ticket_id, **copy.deepcopy(ticket)})

    def _search_support_tickets(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        status = arguments.get("status")
        if status is not None:
            if not isinstance(status, str) or status.strip().lower() not in {"open", "in_progress", "resolved"}:
                raise ValueError("status must be open, in_progress, or resolved")
            status = status.strip().lower()
        requester = arguments.get("requester")
        if requester is not None:
            requester = _text_arg(arguments, "requester").casefold()
        tickets = [
            {"ticket_id": ticket_id, **copy.deepcopy(ticket)}
            for ticket_id, ticket in self.state["tickets"].items()
            if (status is None or ticket["status"] == status)
            and (requester is None or ticket["requester"].casefold() == requester)
        ]
        return self._ok(tickets)

    def _revise_support_ticket(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        ticket_id, ticket = self._ticket(arguments.get("ticket_id"))
        if ticket["status"] == "resolved":
            raise ValueError("reopen a resolved ticket before editing it")
        updates: dict[str, Any] = {}
        if "title" in arguments:
            updates["title"] = _text_arg(arguments, "title", max_chars=120)
        if "details" in arguments or "description" in arguments:
            updates["details"] = _text_arg(arguments, "details", "description", max_chars=2000)
        if "priority" in arguments:
            updates["priority"] = self._priority(arguments["priority"])
        if not updates:
            raise ValueError("provide title, details, or priority to edit")
        ticket.update(updates)
        return self._ok({"ticket_id": ticket_id, **copy.deepcopy(ticket)})

    def _assign_support_ticket(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        ticket_id, ticket = self._ticket(arguments.get("ticket_id"))
        if ticket["status"] == "resolved":
            raise ValueError("a resolved ticket cannot be reassigned")
        agent = _text_arg(arguments, "agent", "assignee").lower()
        if agent not in self.state["agents"]:
            raise ValueError(f"unknown support agent: {agent}")
        ticket["assignee"] = agent
        ticket["status"] = "in_progress"
        return self._ok({"ticket_id": ticket_id, "status": "in_progress", "assignee": agent})

    def _append_ticket_note(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        ticket_id, ticket = self._ticket(arguments.get("ticket_id"))
        if ticket["status"] == "resolved":
            raise ValueError("reopen a resolved ticket before adding notes")
        author = _text_arg(arguments, "author")
        note = _text_arg(arguments, "note", "text", max_chars=1000)
        ticket["notes"].append({"author": author, "text": note})
        return self._ok({"ticket_id": ticket_id, "note_count": len(ticket["notes"])})

    def _close_support_ticket(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        ticket_id, ticket = self._ticket(arguments.get("ticket_id"))
        if ticket["status"] == "resolved":
            raise ValueError(f"ticket {ticket_id} is already resolved")
        agent = _text_arg(arguments, "agent", "resolved_by").lower()
        if agent not in self.state["agents"]:
            raise ValueError(f"unknown support agent: {agent}")
        resolution = _text_arg(arguments, "resolution", max_chars=1000)
        ticket["status"] = "resolved"
        ticket["resolution"] = resolution
        ticket["resolved_by"] = agent
        return self._ok(
            {
                "ticket_id": ticket_id,
                "status": "resolved",
                "resolution": resolution,
                "resolved_by": agent,
            }
        )

    def _reopen_support_ticket(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        ticket_id, ticket = self._ticket(arguments.get("ticket_id"))
        if ticket["status"] != "resolved":
            raise ValueError("only a resolved ticket can be reopened")
        ticket["status"] = "open"
        ticket["resolution"] = None
        ticket["resolved_by"] = None
        return self._ok({"ticket_id": ticket_id, "status": "open", "resolution": None})


DOMAINS = {
    FilesystemDomain.name: FilesystemDomain,
    SocialDomain.name: SocialDomain,
    TradingDomain.name: TradingDomain,
    RetailDomain.name: RetailDomain,
    VehicleDomain.name: VehicleDomain,
    TravelDomain.name: TravelDomain,
    MicroblogDomain.name: MicroblogDomain,
    MessagingDomain.name: MessagingDomain,
    TicketingDomain.name: TicketingDomain,
}


# ---------------------------------------------------------------------------
# A. stateful env
# ---------------------------------------------------------------------------


class MockFCEnv(BaseEnv):
    """Stateful BFCL-v3-style mock environment (plan §4.2 A).

    ``env.config`` keys::

        {"impl": "stateful",
         "domain": one of DOMAINS,
         "initial_state": {...},          # deep-merged over domain defaults
         "responses": {tool_name: str | {"content": ..., "ok": bool}},
         "goal": [{"check": ..., "path": ..., "expected": ..., "mode": ...}],
         "tool_aliases": {declared_name: domain_handler_name},
         "param_aliases": {declared_name: {public_arg: canonical_arg}},
         "param_wrappers": {declared_name: [nested_object_arg]},
         "required_parallel": [[call, ...]],  # each group must complete in one turn
         "dialogue": {"stages": [...]},      # interleaved-user stage controller
         "final_answer_mode": "none" | "normalized" | "contains" | "exact" | "regex",
         "domain_obj": BaseDomain}       # test/injection hook (deep-copied)

    ``env_state`` carries ``goal_satisfied`` (all checks pass, or ``None``
    without a goal spec), ``checks`` (per-assertion detail), ``state`` (raw
    domain snapshot — comparable with ``reference.final_state``) and
    ``rendered``.
    """

    env_type = "mock_fc"
    impl = "stateful"

    def _setup(self, task: Mapping[str, Any], config: dict[str, Any]) -> None:
        domain = config.get("domain_obj")
        if domain is not None:
            if not isinstance(domain, BaseDomain):
                raise EnvError("env.config.domain_obj must be a BaseDomain instance")
            # Deep-copy so two envs built from the same injected object cannot
            # share mutable state (concurrency safety).
            domain = copy.deepcopy(domain)
            domain.reset(
                initial_state=config.get("initial_state") or None,
                responses=config.get("responses") or None,
            )
        else:
            name = config.get("domain")
            if not isinstance(name, str) or name.strip().lower() not in DOMAINS:
                raise EnvError(
                    "mock_fc stateful env requires env.config.domain in "
                    f"{sorted(DOMAINS)}"
                )
            domain_class = DOMAINS[name.strip().lower()]
            domain = domain_class(
                initial_state=config.get("initial_state") or None,
                responses=config.get("responses") or None,
            )
        self._domain: BaseDomain = domain

        goal = config.get("goal") or config.get("goal_checks") or []
        if isinstance(goal, Mapping):
            goal = [goal]
        if not isinstance(goal, list):
            raise EnvError("env.config.goal must be a list of assertion mappings")
        self._goal: list[Mapping[str, Any]] = list(goal)

        aliases = config.get("tool_aliases") or {}
        if not isinstance(aliases, Mapping):
            raise EnvError("env.config.tool_aliases must be a mapping")
        self._aliases: dict[str, str] = {
            str(key): str(value) for key, value in aliases.items()
        }
        self._param_aliases = config.get("param_aliases") or {}
        self._param_wrappers = config.get("param_wrappers") or {}
        if not isinstance(self._param_aliases, Mapping) or not isinstance(self._param_wrappers, Mapping):
            raise EnvError("env.config.param_aliases / param_wrappers must be mappings")

        tool_names = set()
        tools = task.get("tools")
        for tool in tools if isinstance(tools, list) else []:
            if isinstance(tool, Mapping) and isinstance(tool.get("name"), str):
                tool_names.add(tool["name"])
        self._declared_tools: set[str] = tool_names

        reference = task.get("reference") if isinstance(task.get("reference"), Mapping) else {}
        self._reference_final_answer: str | None = _first_final_answer(reference)
        self._final_answer_mode = str(
            config.get("final_answer_mode", "none")
        ).strip().lower()
        if self._final_answer_mode in ("", "off"):
            self._final_answer_mode = "none"
        # require_final_answer: the terminal verdict additionally needs the
        # reported answer to match reference.final_answer (state-only goals
        # cannot score "look it up and tell me" sub-tasks).
        self._require_final_answer = bool(config.get("require_final_answer", False))
        if self._require_final_answer and self._final_answer_mode == "none":
            self._final_answer_mode = "answer_tag"
        self._tool_call_count = 0

        # Fault injection (error-recovery training): the first ``count`` calls
        # to ``tool`` fail with a transient error and do NOT touch the state,
        # so the only way to the goal is to notice the error and retry.
        self._faults: dict[str, dict[str, Any]] = {}
        raw_faults = config.get("faults") or []
        if isinstance(raw_faults, Mapping):
            raw_faults = [raw_faults]
        if not isinstance(raw_faults, list):
            raise EnvError("env.config.faults must be a list of {tool, count, message}")
        for fault in raw_faults:
            if not isinstance(fault, Mapping) or not isinstance(fault.get("tool"), str):
                raise EnvError("each env.config.faults entry needs a string 'tool'")
            count = fault.get("count", 1)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise EnvError("env.config.faults[*].count must be a non-negative int")
            self._faults[fault["tool"]] = {
                "remaining": count,
                "message": str(fault.get("message") or DEFAULT_FAULT_MESSAGE),
            }
        self._faults_injected = 0
        self._tool_errors = 0
        self._invalid_calls = 0
        # tools that failed at least once and later succeeded (recovery signal)
        self._failed_tools: set[str] = set()
        self._recovered_tools: set[str] = set()

        raw_parallel = config.get("required_parallel") or []
        if not isinstance(raw_parallel, list) or not all(isinstance(group, list) for group in raw_parallel):
            raise EnvError("env.config.required_parallel must be a list of call groups")
        self._required_parallel: list[list[dict[str, Any]]] = [
            [dict(call) for call in group] for group in raw_parallel
        ]
        self._parallel_matched = [False] * len(self._required_parallel)

        dialogue = config.get("dialogue")
        self._dialogue: DialogueController | None = None
        if dialogue is not None:
            if not isinstance(dialogue, Mapping):
                raise EnvError("env.config.dialogue must be a mapping")
            try:
                self._dialogue = DialogueController(
                    dialogue,
                    initial_goal=self._goal,
                    initial_state=self._domain.state,
                    evaluate=evaluate_goal,
                    reference_answer=self._reference_final_answer,
                )
            except ValueError as exc:
                raise EnvError(str(exc)) from exc
            self._goal = self._dialogue.goal

    def _handle_action(
        self,
        action: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], bool, dict[str, Any], str | None]:
        if self._dialogue is None:
            return super()._handle_action(action)
        if action["tool_calls"]:
            results, info = self._handle_tool_calls(action["tool_calls"])
            return results, False, info, None

        advanced, next_user, detail = self._dialogue.advance(action["content"], self._domain.state)
        info: dict[str, Any] = {"dialogue": self._dialogue.snapshot(), "detail": detail}
        if not advanced:
            return [], True, info, action["content"]
        self._goal = self._dialogue.goal
        if self._dialogue.complete:
            return [], True, info, action["content"]
        info["next_user_message"] = next_user
        return [], False, info, None

    def _adapt_arguments(self, declared: str, handler: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Translate public parameter names / simple object wrappers to canonical args."""
        adapted = dict(arguments)
        wrappers = self._param_wrappers.get(declared, self._param_wrappers.get(handler, []))
        if isinstance(wrappers, str):
            wrappers = [wrappers]
        if isinstance(wrappers, Mapping):
            wrappers = list(wrappers)
        for wrapper in wrappers if isinstance(wrappers, list) else []:
            nested = adapted.pop(str(wrapper), None)
            if isinstance(nested, Mapping):
                adapted.update(nested)
            elif nested is not None:
                adapted[str(wrapper)] = nested
        aliases = self._param_aliases.get(declared, self._param_aliases.get(handler, {}))
        if isinstance(aliases, Mapping):
            adapted = {str(aliases.get(key, key)): value for key, value in adapted.items()}
        return adapted

    def _handle_tool_calls(
        self,
        calls: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        results: list[dict[str, Any]] = []
        call_info: list[dict[str, Any]] = []
        for call in calls:
            declared = call["name"]
            fault = self._faults.get(declared)
            if self._declared_tools and declared not in self._declared_tools:
                outcome = {
                    "ok": False,
                    "content": f"Error: tool {declared!r} is not declared in this task",
                }
                self._invalid_calls += 1
            elif fault is not None and fault["remaining"] > 0:
                fault["remaining"] -= 1
                self._faults_injected += 1
                outcome = {"ok": False, "content": fault["message"]}
            else:
                handler_name = self._aliases.get(declared, declared)
                arguments = self._adapt_arguments(declared, handler_name, call["arguments"])
                outcome = self._domain.call(handler_name, arguments)
            if outcome["ok"]:
                if declared in self._failed_tools:
                    self._recovered_tools.add(declared)
            else:
                self._tool_errors += 1
                self._failed_tools.add(declared)
            results.append(
                make_tool_result(
                    declared,
                    outcome["content"],
                    ok=bool(outcome["ok"]),
                    call_id=call["id"],
                )
            )
            call_info.append({"name": declared, "ok": bool(outcome["ok"])})
        self._tool_call_count += len(calls)
        if calls and all(item["ok"] for item in call_info):
            for index, expected in enumerate(self._required_parallel):
                if not self._parallel_matched[index] and match_call_group(calls, expected)[0]:
                    self._parallel_matched[index] = True
        if self._dialogue is not None:
            self._dialogue.on_tool_calls(call_info, self._domain.state)
        return results, {"tool_calls": call_info, "tool_call_count": self._tool_call_count}

    def _env_state_fields(self) -> dict[str, Any]:
        checks = evaluate_goal(self._goal, self._domain.state)
        final_answer_matches = self._match_final_answer()
        if self._require_final_answer:
            checks.append(
                {
                    "name": "final_answer",
                    "ok": final_answer_matches is True,
                    "detail": f"mode={self._final_answer_mode} matched={final_answer_matches}",
                }
            )
        if self._required_parallel:
            checks.append(
                {
                    "name": "required_parallel",
                    "ok": all(self._parallel_matched),
                    "detail": f"matched={self._parallel_matched}",
                }
            )
        if self._dialogue is not None:
            checks.extend(self._dialogue.terminal_checks(self._domain.state))
        goal_satisfied = all(item["ok"] for item in checks) if checks else None
        if self._timeout:
            # T4-01: a max_turns timeout is a failure; terminal assertions may
            # coincidentally pass, but must never award the result term.
            goal_satisfied = False
        return {
            "goal_satisfied": goal_satisfied,
            "checks": checks,
            "state": self._domain.snapshot(),
            "rendered": self._domain.render(self.max_result_chars),
            "num_turns": len(self._turns),
            "timeout": bool(self._timeout),
            "extras": {
                "domain": self._domain.name,
                "final_answer_matches": final_answer_matches,
                "final_answer_mode": self._final_answer_mode,
                "require_final_answer": self._require_final_answer,
                "tool_call_count": self._tool_call_count,
                "tool_errors": self._tool_errors,
                "invalid_calls": self._invalid_calls,
                "faults_injected": self._faults_injected,
                "recovered_tools": sorted(self._recovered_tools),
                "parallel_matched": list(self._parallel_matched),
                "dialogue": self._dialogue.snapshot() if self._dialogue is not None else None,
            },
        }

    def _match_final_answer(self) -> bool | None:
        mode = self._final_answer_mode
        if mode == "none" or self._final_answer is None:
            return None
        expected = self._reference_final_answer
        if expected is None:
            return None
        actual = self._final_answer
        if mode == "normalized":
            return normalize_query(actual) == normalize_query(expected)
        if mode == "contains":
            return normalize_query(expected) in normalize_query(actual)
        if mode == "exact":
            return actual.strip() == expected.strip()
        if mode == "regex":
            try:
                return re.search(expected, actual) is not None
            except re.error:
                return None
        if mode == "answer_tag":
            return answer_tag_matches(actual, expected)
        return None


def _first_final_answer(reference: Mapping[str, Any]) -> str | None:
    answer = reference.get("final_answer")
    if isinstance(answer, str) and answer.strip():
        return answer
    answers = reference.get("final_answers")
    if isinstance(answers, list):
        for item in answers:
            if isinstance(item, str) and item.strip():
                return item
    return None


# ---------------------------------------------------------------------------
# B. replay env
# ---------------------------------------------------------------------------


class ReplayMockFCEnv(BaseEnv):
    """Replay environment: match each turn against ``reference.gt_calls``.

    Matched turns get the ground-truth return injected (source recorded in
    ``env_state.extras["gt_returns_source"]``); the first mismatch terminates
    the episode with ``replay_ok=False``.  A trajectory is fully matched only
    when every GT turn group was consumed and the episode ends with a no-call
    final answer.  This is the mt_fc B result signal (CONTRACTS.md C4).
    """

    env_type = "mock_fc"
    impl = "replay"

    def _setup(self, task: Mapping[str, Any], config: dict[str, Any]) -> None:
        self._gt_groups: list[list[dict[str, Any]]] = gt_call_groups(task)
        self._returns, self._returns_source = resolve_gt_returns(task, config)
        if self._returns_source not in GT_RETURNS_SOURCES:
            raise EnvError(
                f"gt_returns_source {self._returns_source!r} is not one of "
                f"{GT_RETURNS_SOURCES}"
            )
        self._match_mode = str(config.get("match_mode", "loose")).strip().lower()
        if self._match_mode not in ("loose", "exact"):
            raise EnvError(
                f"env.config.match_mode must be 'loose' or 'exact', got {self._match_mode!r}"
            )
        self._cursor = 0
        self._replay_ok: bool | None = None
        self._failed_turn: int | None = None
        self._replay_info: dict[str, Any] = {}
        # T4-02: missing GT returns must never be silently fabricated.
        self._missing_returns: int = 0
        self._injected_calls: int = 0
        raw_ratio = config.get(
            "max_missing_ratio", config.get("max_gt_returns_missing_ratio", 0.5)
        )
        try:
            self._max_missing_ratio = float(raw_ratio)
        except (TypeError, ValueError) as exc:
            raise EnvError(
                f"env.config.max_missing_ratio must be a float in [0, 1], got {raw_ratio!r}"
            ) from exc
        if not 0.0 <= self._max_missing_ratio <= 1.0:
            raise EnvError(
                "env.config.max_missing_ratio must be in [0, 1], "
                f"got {self._max_missing_ratio!r}"
            )
        reference = task.get("reference") if isinstance(task.get("reference"), Mapping) else {}
        self._reference_final_answer = _first_final_answer(reference)
        meta = task.get("meta") if isinstance(task.get("meta"), Mapping) else {}
        self._declared_returns_source = meta.get("gt_returns_source")

    # -- protocol views ---------------------------------------------------

    @property
    def replay_ok(self) -> bool | None:
        return self._replay_ok

    def _gt_turn_count(self) -> int:
        return sum(1 for group in self._gt_groups if group)

    def _missing_ratio(self) -> float:
        if self._injected_calls <= 0:
            return 0.0
        return self._missing_returns / self._injected_calls

    def _effective_returns_source(self) -> str:
        """Runtime source, downgraded to "placeholder" when GT returns are missing.

        The static ``meta["gt_returns_source"]`` cannot know how many per-call
        returns were actually present, so a task declared as ``reference`` whose
        returns are missing is downgraded here (T4-02) — reward / T2 then see a
        degraded replay instead of a silent quality loss.
        """
        if self._returns_source == "placeholder":
            return "placeholder"
        if self._missing_returns and self._missing_ratio() > self._max_missing_ratio:
            return "placeholder"
        return self._returns_source

    def _degraded(self) -> bool:
        """True when replay data is degraded (placeholder or missing returns)."""
        return (
            self._effective_returns_source() == "placeholder"
            and self._gt_turn_count() > 0
        )

    def _handle_action(
        self,
        action: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], bool, dict[str, Any], str | None]:
        calls = action["tool_calls"]
        if not calls:
            remaining = self._gt_groups[self._cursor :]
            ok = all(len(group) == 0 for group in remaining)
            self._replay_ok = ok
            reason = (
                "final answer after all ground-truth turns were matched"
                if ok
                else "final answer before ground-truth calls were consumed"
            )
            self._replay_info = {
                "ok": ok,
                "matched_turns": self._cursor,
                "gt_turns": self._gt_turn_count(),
                "reason": reason,
            }
            info: dict[str, Any] = {"replay": dict(self._replay_info)}
            if self._degraded():
                info["degraded_replay"] = True
            return [], True, info, action["content"]

        if self._cursor >= len(self._gt_groups):
            self._replay_ok = False
            self._failed_turn = self._cursor
            reason = "tool call emitted after ground-truth calls were exhausted"
            self._replay_info = {
                "ok": False,
                "failed_turn": self._failed_turn,
                "matched_turns": self._cursor,
                "gt_turns": self._gt_turn_count(),
                "reason": reason,
            }
            results = [
                make_tool_result(
                    call["name"], f"Error: {reason}", ok=False, call_id=call["id"]
                )
                for call in calls
            ]
            info = {"replay": dict(self._replay_info)}
            if self._degraded():
                info["degraded_replay"] = True
            return results, True, info, None

        expected = self._gt_groups[self._cursor]
        turn_index = self._cursor
        matched, message = match_call_group(calls, expected, mode=self._match_mode)
        if not matched:
            self._replay_ok = False
            self._failed_turn = turn_index
            self._replay_info = {
                "ok": False,
                "failed_turn": turn_index,
                "matched_turns": self._cursor,
                "gt_turns": self._gt_turn_count(),
                "reason": message if expected else "ground-truth expects no tool call this turn",
                "expected": [dict(call) for call in expected],
                "actual": [dict(call) for call in calls],
            }
            results = [
                make_tool_result(
                    call["name"],
                    f"Error: replay mismatch at turn {turn_index}: "
                    f"{_short(self._replay_info['reason'])}",
                    ok=False,
                    call_id=call["id"],
                )
                for call in calls
            ]
            info = {"replay": dict(self._replay_info)}
            if self._degraded():
                info["degraded_replay"] = True
            return results, True, info, None

        self._cursor += 1
        results = self._inject_returns(calls, turn_index)
        self._replay_info = {
            "ok": True,
            "turn": turn_index,
            "matched_turns": self._cursor,
            "gt_turns": self._gt_turn_count(),
            "gt_returns_source": self._effective_returns_source(),
        }
        info = {"replay": dict(self._replay_info)}
        if self._degraded():
            info["degraded_replay"] = True
        return results, False, info, None

    # -- internals --------------------------------------------------------

    def _inject_returns(
        self,
        calls: list[dict[str, Any]],
        turn_index: int,
    ) -> list[dict[str, Any]]:
        """Inject per-call GT returns; missing entries become ``ok=False``.

        A return is used only when its ``name`` matches the call (unnamed
        legacy records are consumed positionally).  Anything else increments
        ``extras["gt_returns_missing"]`` and yields a failed tool result, so a
        missing GT return is visible to the policy, reward and reports instead
        of being silently replaced by a fake success (T4-02).
        """
        entries = self._returns[turn_index] if turn_index < len(self._returns) else []
        used: set[int] = set()
        results: list[dict[str, Any]] = []
        for call in calls:
            self._injected_calls += 1
            entry_index: int | None = None
            unnamed_candidate: int | None = None
            for index, entry in enumerate(entries):
                if index in used:
                    continue
                entry_name = entry.get("name")
                if isinstance(entry_name, str) and entry_name.strip():
                    if entry_name == call["name"]:
                        entry_index = index
                        break
                elif unnamed_candidate is None:
                    unnamed_candidate = index
            if entry_index is None:
                entry_index = unnamed_candidate
            if entry_index is None:
                self._missing_returns += 1
                results.append(
                    make_tool_result(
                        call["name"],
                        f"{MISSING_RETURN_PREFIX} {call['name']!r} "
                        f"(gt_returns_source={self._returns_source})",
                        ok=False,
                        call_id=call["id"],
                    )
                )
                continue
            used.add(entry_index)
            entry = entries[entry_index]
            content = entry.get("content", "")
            ok = bool(entry.get("ok", True))
            results.append(
                make_tool_result(call["name"], content, ok=ok, call_id=call["id"])
            )
        return results

    def _env_state_fields(self) -> dict[str, Any]:
        effective_source = self._effective_returns_source()
        missing_ratio = round(self._missing_ratio(), 4)
        replay_ok = self._replay_ok
        if self._timeout:
            # A timeout is a failed episode; never report a matching replay.
            replay_ok = False
        rendered = (
            f"replay matched_turns={self._cursor}/"
            f"{self._gt_turn_count()} replay_ok={replay_ok}"
            f" gt_returns_source={effective_source}"
            f" gt_returns_missing={self._missing_returns}/{self._injected_calls}"
        )
        degraded = (
            effective_source == "placeholder" and self._gt_turn_count() > 0
        )
        return {
            "replay_ok": replay_ok,
            "failed_turn": self._failed_turn,
            "matched_turns": self._cursor,
            "gt_turns": self._gt_turn_count(),
            "rendered": rendered,
            "num_turns": len(self._turns),
            "timeout": bool(self._timeout),
            "extras": {
                "match_mode": self._match_mode,
                # Runtime truth: downgraded to "placeholder" when GT returns are
                # missing beyond the configured threshold (T4-02 / CONTRACTS C6).
                "gt_returns_source": effective_source,
                "gt_returns_declared_source": self._returns_source,
                "gt_returns_missing": self._missing_returns,
                "gt_returns_missing_ratio": missing_ratio,
                "max_missing_ratio": self._max_missing_ratio,
                "gt_returns_injected_calls": self._injected_calls,
                "degraded_replay": degraded,
                "gt_returns_groups": len(self._returns),
                "meta_gt_returns_source": self._declared_returns_source,
                "last_replay": dict(self._replay_info),
                "final_answer_expected": self._reference_final_answer is not None,
            },
        }


__all__ = [
    "BaseDomain",
    "DOMAINS",
    "FilesystemDomain",
    "GT_RETURNS_SOURCES",
    "MISSING_RETURN_PREFIX",
    "MessagingDomain",
    "MicroblogDomain",
    "MockFCEnv",
    "ReplayMockFCEnv",
    "RetailDomain",
    "SocialDomain",
    "TicketingDomain",
    "TradingDomain",
    "TravelDomain",
    "VehicleDomain",
    "answer_tag_matches",
    "deep_equal",
    "evaluate_goal",
    "match_call_group",
    "resolve_gt_returns",
]
