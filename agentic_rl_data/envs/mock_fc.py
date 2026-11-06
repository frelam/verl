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
import re
from typing import Any, Mapping

from ..schema import canonical_json, gt_call_groups, normalize_query
from .base import BaseEnv, EnvError, make_tool_result, truncate_text

#: Built-in stateful domains (BFCL v3 style subset; extend by subclassing).
DOMAINS: dict[str, type] = {}

#: GT-return sources in priority order (CONTRACTS.md C6).
GT_RETURNS_SOURCES = ("config", "reference", "gt_turns", "placeholder")

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


def _to_amount(arguments: Mapping[str, Any], *names: str) -> int:
    value = _to_number(arguments, *names)
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


DOMAINS = {
    FilesystemDomain.name: FilesystemDomain,
    SocialDomain.name: SocialDomain,
    TradingDomain.name: TradingDomain,
}


# ---------------------------------------------------------------------------
# A. stateful env
# ---------------------------------------------------------------------------


class MockFCEnv(BaseEnv):
    """Stateful BFCL-v3-style mock environment (plan §4.2 A).

    ``env.config`` keys::

        {"impl": "stateful",
         "domain": "filesystem" | "social" | "trading",
         "initial_state": {...},          # deep-merged over domain defaults
         "responses": {tool_name: str | {"content": ..., "ok": bool}},
         "goal": [{"check": ..., "path": ..., "expected": ..., "mode": ...}],
         "tool_aliases": {declared_name: domain_handler_name},
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
        self._tool_call_count = 0

    def _handle_tool_calls(
        self,
        calls: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        results: list[dict[str, Any]] = []
        call_info: list[dict[str, Any]] = []
        for call in calls:
            declared = call["name"]
            if self._declared_tools and declared not in self._declared_tools:
                outcome = {
                    "ok": False,
                    "content": f"Error: tool {declared!r} is not declared in this task",
                }
            else:
                handler_name = self._aliases.get(declared, declared)
                outcome = self._domain.call(handler_name, call["arguments"])
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
        return results, {"tool_calls": call_info, "tool_call_count": self._tool_call_count}

    def _env_state_fields(self) -> dict[str, Any]:
        checks = evaluate_goal(self._goal, self._domain.state)
        goal_satisfied = all(item["ok"] for item in checks) if checks else None
        if self._timeout:
            # T4-01: a max_turns timeout is a failure; terminal assertions may
            # coincidentally pass, but must never award the result term.
            goal_satisfied = False
        final_answer_matches = self._match_final_answer()
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
                "tool_call_count": self._tool_call_count,
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
    "MockFCEnv",
    "ReplayMockFCEnv",
    "SocialDomain",
    "TradingDomain",
    "deep_equal",
    "evaluate_goal",
    "match_call_group",
    "resolve_gt_returns",
]
