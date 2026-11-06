"""Sandboxed unit-test execution for ``code_exec`` (plan §4.4, CONTRACTS.md C5).

The environment writes the candidate solution to a fresh temporary directory,
runs the task's test file there in the same subprocess sandbox used by
``python_env`` (timeout / rlimits / cleared env / output truncation), and
returns a **summary only**:

``trajectory["env_state"]["code_tests"]`` (authoritative, C5)::

    {"total": int, "passed": int, "failed": int, "all_passed": bool,
     "infra_error": bool,          # harness broke, NOT a wrong answer
     "exit_code": int, "duration_s": float, "truncated": bool,
     "failures": [{"name": str, "message": str}]}   # message <= 200 chars

``infra_error`` is True when the *harness* failed (no tests configured, the
runner is missing, the runner crashed with a usage error, or the run collected
zero tests).  ``all_passed`` stays False in that case so the verdict is never
inflated, but rewards / reports can drop infra failures instead of scoring
them as a wrong answer (T4 P1, Lead裁决).

Safety rules:

* the standard-answer code is never present in the task schema for ``code_exec``
  (there is no such field), and the test file body is **never** echoed into an
  observation — only pytest/unittest pass/fail lines and truncated assertion
  messages leave the sandbox;
* ``env.config["redact"]`` accepts regex / literal patterns that are replaced
  with ``[redacted]`` in every failure message;
* the candidate code is not echoed back either (the policy wrote it).

The test file is expected to import the candidate from the ``solution`` module
(``from solution import solve``), which is how loaders must normalise APPS /
CodeContests / TACO style unit tests.  ``info["tests"]`` carries the same
summary for framework logs only (D6: rewards must read ``env_state``).
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..schema import canonical_json
from .base import BaseEnv, EnvError, make_tool_result, truncate_text
from .python_env import (
    SandboxConfig,
    build_sandbox_config,
    run_process,
)

DEFAULT_TEST_TIMEOUT_S = 10.0
DEFAULT_MAX_FAILURE_MESSAGE = 200
DEFAULT_TEST_FILENAME = "test_solution.py"
DEFAULT_SOLUTION_FILENAME = "solution.py"

#: Tool names accepted as "run the tests on this code".
EXEC_TOOL_ALIASES = (
    "run_tests",
    "submit_code",
    "execute_tests",
    "code_tests",
    "run_code",
    "check_solution",
)

#: Argument keys accepted as the candidate code payload.
CODE_ARG_KEYS = ("code", "solution", "answer", "source", "implementation", "program")

_FENCE_RE = re.compile(r"```(?:[A-Za-z0-9_+.-]*)?\s*\n(.*?)```", re.DOTALL)
_PYTEST_PASSED_RE = re.compile(r"(\d+)\s+passed")
_PYTEST_FAILED_RE = re.compile(r"(\d+)\s+failed")
_PYTEST_ERROR_RE = re.compile(r"(\d+)\s+error")
_PYTEST_COLLECTED_RE = re.compile(r"collected\s+(\d+)\s+items?")
_UNITTEST_RAN_RE = re.compile(r"Ran\s+(\d+)\s+tests?")
_UNITTEST_FAILED_RE = re.compile(r"FAILED\s*\(([^)]*)\)")
_UNITTEST_FAILURES_RE = re.compile(r"failures=(\d+)")
_UNITTEST_ERRORS_RE = re.compile(r"errors=(\d+)")

_PYTEST_RUNNER = """\
import sys
sys.path.insert(0, sys.argv[1])
import pytest
sys.exit(pytest.main(["-q", "--tb=line", "-p", "no:cacheprovider", sys.argv[2]]))
"""

_UNITTEST_RUNNER = """\
import sys
import unittest
sys.path.insert(0, sys.argv[1])
loader = unittest.TestLoader()
suite = loader.discover(sys.argv[1], pattern=sys.argv[2])
runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
result = runner.run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
"""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _argument(arguments: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in arguments:
            return arguments[name]
    return default


def _match_exec_tool(name: str, aliases: Sequence[str]) -> bool:
    candidate = str(name).strip().lower()
    if candidate in {alias.lower() for alias in aliases}:
        return True
    return any(token in candidate for token in ("test", "solution", "execute"))


def extract_code(text: Any) -> str:
    """Extract candidate code from a final answer (Markdown fenced block first)."""
    if not isinstance(text, str):
        return ""
    blocks = _FENCE_RE.findall(text)
    if blocks:
        # Longest block is the most likely implementation.
        return max(blocks, key=len).strip()
    return text.strip()


def _redact(value: str, patterns: Sequence[str]) -> str:
    for pattern in patterns:
        if not pattern:
            continue
        try:
            value = re.sub(str(pattern), "[redacted]", value)
        except re.error:
            value = value.replace(str(pattern), "[redacted]")
    return value


def parse_test_output(output: str, *, runner: str) -> dict[str, int]:
    """Parse pass/fail counts from pytest or unittest stdout."""
    counts = {"total": 0, "passed": 0, "failed": 0}
    if not isinstance(output, str):
        return counts
    if runner == "unittest":
        ran = _UNITTEST_RAN_RE.search(output)
        if ran:
            counts["total"] = int(ran.group(1))
        failed_block = _UNITTEST_FAILED_RE.search(output)
        failures = 0
        errors = 0
        if failed_block:
            match = _UNITTEST_FAILURES_RE.search(failed_block.group(1))
            failures = int(match.group(1)) if match else 0
            match = _UNITTEST_ERRORS_RE.search(failed_block.group(1))
            errors = int(match.group(1)) if match else 0
        else:
            # "FAILED (failures=N)" variants are covered above; fall back to
            # explicit "FAILED:" count when the summary block is missing.
            failures = len(re.findall(r"^FAIL:", output, re.MULTILINE))
        counts["failed"] = failures + errors
        if counts["total"] > 0:
            counts["passed"] = max(0, counts["total"] - counts["failed"])
    else:
        passed = _PYTEST_PASSED_RE.search(output)
        failed = _PYTEST_FAILED_RE.search(output)
        errors = _PYTEST_ERROR_RE.search(output)
        collected = _PYTEST_COLLECTED_RE.search(output)
        counts["passed"] = int(passed.group(1)) if passed else 0
        counts["failed"] = int(failed.group(1)) if failed else 0
        counts["failed"] += int(errors.group(1)) if errors else 0
        counts["total"] = int(collected.group(1)) if collected else 0
    if counts["total"] <= 0:
        counts["total"] = counts["passed"] + counts["failed"]
    if counts["passed"] > counts["total"]:
        counts["passed"] = counts["total"]
    return counts


def _collect_failures(
    output: str,
    *,
    runner: str,
    redact_patterns: Sequence[str],
    max_failures: int = 8,
) -> list[dict[str, str]]:
    """Extract ``[{"name", "message"}]`` failure summaries (truncated, redacted)."""
    failures: list[dict[str, str]] = []
    if not isinstance(output, str):
        return failures
    for line in output.splitlines():
        stripped = line.strip()
        name = ""
        message = ""
        if stripped.startswith("FAILED "):
            rest = stripped[len("FAILED ") :]
            name, separator, message = rest.partition(" - ")
            if not separator:
                message = ""
        elif stripped.startswith("FAIL:"):
            rest = stripped[len("FAIL:") :].strip()
            name = rest.split("(")[0].strip()
        elif stripped.startswith("ERROR:"):
            rest = stripped[len("ERROR:") :].strip()
            name = rest.split("(")[0].strip()
        elif stripped.startswith("E ") or stripped.startswith("E   "):
            # pytest --tb=line continuation lines (assertion expression).
            if failures and not failures[-1].get("message"):
                message = stripped[1:].strip()
            else:
                continue
        else:
            continue
        if not name:
            continue
        message = _redact(message, redact_patterns)
        failures.append(
            {
                "name": truncate_text(name, DEFAULT_MAX_FAILURE_MESSAGE)[0],
                "message": truncate_text(message, DEFAULT_MAX_FAILURE_MESSAGE)[0],
            }
        )
        if len(failures) >= max_failures:
            break
    # Attach bare assertion lines from --tb=line to the first failure if
    # nothing useful was captured (pytest prints "path:line: AssertionError").
    if failures and all(not item["message"] for item in failures):
        for line in output.splitlines():
            stripped = line.strip()
            if ": AssertionError" in stripped or stripped.startswith("assert "):
                failures[0]["message"] = truncate_text(
                    _redact(stripped, redact_patterns), DEFAULT_MAX_FAILURE_MESSAGE
                )[0]
                break
    return failures


def _empty_tests(
    *,
    exit_code: int = -1,
    duration_s: float = 0.0,
    truncated: bool = False,
    infra_error: bool = False,
) -> dict[str, Any]:
    return {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "all_passed": False,
        "infra_error": bool(infra_error),
        "exit_code": int(exit_code),
        "duration_s": round(float(duration_s), 3),
        "truncated": bool(truncated),
        "failures": [],
    }


# ---------------------------------------------------------------------------
# CodeTestsEnv
# ---------------------------------------------------------------------------


def _looks_like_final_code(code: str) -> bool:
    """True when the final-answer text plausibly *is* a program.

    A plain answer such as ``"done"`` must not be executed as code: the last
    test verdict is kept instead (D22).  Heuristic on common code markers;
    models that paste real code (``def``/``import``/fenced blocks) still run.
    """
    text = code.strip()
    if not text:
        return False
    markers = ("def ", "class ", "import ", "from ", "```", "print(", "return ")
    return any(marker in text for marker in markers)


class CodeTestsEnv(BaseEnv):
    """``code_exec`` environment: write code -> run unit tests -> see summary.

    ``env.config`` keys::

        {"runner": "auto" | "pytest" | "unittest",
         "timeout_s": 10, "memory_mb": 1024, "max_output_chars": 4096,
         "test_file": "test_solution.py", "solution_file": "solution.py",
         "unit_tests": "<override reference.unit_tests>",
         "redact": ["regex-or-literal", ...],
         "tools": ["run_tests", ...]}

    Turn semantics: a tool call whose name is in ``tools`` runs the tests for
    the code argument; a no-call turn is treated as the final answer and the
    final answer text (fenced code extracted from Markdown) is tested once
    before the episode ends.  The authoritative verdict lives in
    ``env_state["code_tests"]["all_passed"]`` (C5); ``info["tests"]`` duplicates
    it for framework logs only.

    ``code_tests["infra_error"]`` separates a broken test harness (missing
    runner, no tests collected, runner crash / usage error) from a wrong
    answer: rewards/reports can exclude infra failures instead of silently
    scoring them as 0 (T4 P1).
    """

    env_type = "code_tests"
    impl = "local"

    def _setup(self, task: Mapping[str, Any], config: dict[str, Any]) -> None:
        reference = task.get("reference") if isinstance(task.get("reference"), Mapping) else {}
        unit_tests = config.get("unit_tests")
        if not isinstance(unit_tests, str):
            unit_tests = reference.get("unit_tests")
        self._unit_tests: str | None = unit_tests if isinstance(unit_tests, str) else None
        self._runner = self._resolve_runner(config)
        sandbox_config = dict(config)
        sandbox_config.setdefault("memory_mb", 1024)
        self._sandbox: SandboxConfig = build_sandbox_config(
            sandbox_config, timeout_s=float(config.get("timeout_s", DEFAULT_TEST_TIMEOUT_S))
        )
        self._test_filename = str(config.get("test_file") or DEFAULT_TEST_FILENAME)
        self._solution_filename = str(config.get("solution_file") or DEFAULT_SOLUTION_FILENAME)
        self._redact_patterns = [
            str(pattern) for pattern in (config.get("redact") or []) if str(pattern)
        ]
        tools = config.get("tools") or EXEC_TOOL_ALIASES
        if isinstance(tools, str):
            tools = (tools,)
        self._tool_aliases: tuple[str, ...] = tuple(str(tool) for tool in tools)
        self._last_tests: dict[str, Any] | None = None
        self._last_diagnostics: dict[str, Any] = {}

    # -- turn semantics ---------------------------------------------------

    def _handle_action(
        self,
        action: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], bool, dict[str, Any], str | None]:
        calls = action["tool_calls"]
        if calls:
            results, info = self._handle_tool_calls(calls)
            return results, False, info, None
        code = extract_code(action["content"])
        if not code.strip():
            tests = _empty_tests(exit_code=-1, infra_error=False)
            self._last_tests = tests
            self._last_diagnostics = {
                "error": "no code submitted in the final answer",
                "infra_error": False,
            }
            return [], True, {"tests": dict(tests)}, action["content"]
        if not _looks_like_final_code(code):
            # Plain final answer with no code: keep the previous test verdict
            # instead of re-running the text as a program (D22).  Running it
            # would turn a correct solution into a spurious infra_error.
            tests = self._last_tests or _empty_tests(exit_code=-1, infra_error=False)
            self._last_diagnostics = {
                "error": "final answer without code; previous test verdict kept",
                "infra_error": False,
            }
            return [], True, {"tests": dict(tests)}, action["content"]
        tests = self._run_tests(code)
        info: dict[str, Any] = {"tests": dict(tests)}
        if self._last_diagnostics.get("runner_error"):
            info["runner_error"] = True
        return [], True, info, action["content"]

    def _handle_tool_calls(
        self,
        calls: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for call in calls:
            if not _match_exec_tool(call["name"], self._tool_aliases):
                results.append(
                    make_tool_result(
                        call["name"],
                        f"Error: unknown code_exec tool {call['name']!r}; "
                        f"available: {sorted(set(self._tool_aliases))}",
                        ok=False,
                        call_id=call["id"],
                    )
                )
                continue
            code = _argument(call["arguments"], *CODE_ARG_KEYS)
            if not isinstance(code, str) or not code.strip():
                tests = _empty_tests(exit_code=-1, infra_error=False)
                self._last_tests = tests
                self._last_diagnostics = {
                    "error": "tool call carried no code argument",
                    "infra_error": False,
                }
                results.append(
                    make_tool_result(
                        call["name"],
                        "Error: code argument is required "
                        f"({'/'.join(CODE_ARG_KEYS[:3])})",
                        ok=False,
                        call_id=call["id"],
                    )
                )
                continue
            tests = self._run_tests(code)
            results.append(
                make_tool_result(
                    call["name"],
                    canonical_json(tests),
                    ok=bool(tests["all_passed"]),
                    call_id=call["id"],
                )
            )
        info: dict[str, Any] = {"tests": dict(self._last_tests or _empty_tests())}
        return results, info

    # -- test execution ---------------------------------------------------

    def _resolve_runner(self, config: Mapping[str, Any]) -> str:
        runner = str(config.get("runner", "auto")).strip().lower()
        if runner in ("pytest", "unittest"):
            return runner
        try:
            return "pytest" if importlib.util.find_spec("pytest") is not None else "unittest"
        except (ImportError, ValueError):  # pragma: no cover - defensive
            return "unittest"

    def _run_tests(self, code: str) -> dict[str, Any]:
        if self._unit_tests is None:
            self._last_tests = _empty_tests(exit_code=-1, infra_error=True)
            self._last_diagnostics = {
                "error": "no unit_tests configured (reference.unit_tests is null)",
                "infra_error": True,
            }
            return dict(self._last_tests)

        temporary_directory = tempfile.mkdtemp(prefix="ard_code_")
        try:
            workspace = Path(temporary_directory)
            (workspace / self._solution_filename).write_text(code, encoding="utf-8")
            (workspace / self._test_filename).write_text(
                self._unit_tests, encoding="utf-8"
            )
            runner = self._runner
            try:
                tests = self._execute_runner(workspace, runner)
                if runner == "pytest" and self._last_diagnostics.get("pytest_missing"):
                    tests = self._execute_runner(workspace, "unittest")
                return tests
            except Exception as exc:  # infra failure: never crash the rollout
                tests = _empty_tests(exit_code=-1, infra_error=True)
                self._last_tests = tests
                self._last_diagnostics = {
                    "error": f"test runner crashed: {type(exc).__name__}: {exc}",
                    "infra_error": True,
                }
                return dict(tests)
        finally:
            shutil.rmtree(temporary_directory, ignore_errors=True)

    def _execute_runner(self, workspace: Path, runner: str) -> dict[str, Any]:
        script_path = workspace / f"_runner_{runner}.py"
        script_path.write_text(
            _PYTEST_RUNNER if runner == "pytest" else _UNITTEST_RUNNER,
            encoding="utf-8",
        )
        if runner == "pytest":
            argv = [
                sys.executable or "python",
                "-I",
                "-B",
                str(script_path),
                str(workspace),
                self._test_filename,
            ]
        else:
            argv = [
                sys.executable or "python",
                "-I",
                "-B",
                str(script_path),
                str(workspace),
                self._test_filename,
            ]
        execution = run_process(argv, cwd=workspace, config=self._sandbox)
        stdout = execution["stdout"]
        stderr = execution["stderr"]
        runner_error = (
            "No module named 'pytest'" in stderr
            or "No module named pytest" in stderr
        )
        if runner_error and runner == "pytest":
            self._last_diagnostics = {
                "runner_error": "pytest not importable in the sandbox",
                "pytest_missing": True,
                "infra_error": True,
            }
            return _empty_tests(
                exit_code=execution["exit_code"],
                duration_s=execution["duration_s"],
                truncated=execution["truncated"],
                infra_error=True,
            )

        counts = parse_test_output(stdout, runner=runner)
        failures = _collect_failures(
            stdout, runner=runner, redact_patterns=self._redact_patterns
        )
        total = counts["total"]
        passed = counts["passed"]
        failed = counts["failed"]
        no_tests_ran = any(
            marker in stdout
            for marker in ("no tests ran", "Ran 0 tests", "collected 0 items")
        )
        # Infra failures (runner crashed / wrong CLI / empty collection) must be
        # distinguishable from a wrong answer (T4 P1): a timeout or a failing
        # assertion stays a policy failure, not infra.
        infra_error = bool(
            not execution["timed_out"]
            and (execution["exit_code"] >= 2 or no_tests_ran)
        )
        all_passed = bool(
            total > 0
            and passed == total
            and not execution["timed_out"]
            and execution["exit_code"] == 0
        )
        tests = {
            "total": int(total),
            "passed": int(passed),
            "failed": int(failed),
            "all_passed": all_passed,
            "infra_error": infra_error,
            "exit_code": int(execution["exit_code"]),
            "duration_s": round(float(execution["duration_s"]), 3),
            "truncated": bool(execution["truncated"]),
            "failures": failures,
        }
        if execution["timed_out"]:
            # A timeout is a failure; keep counts as parsed (may be partial).
            tests["all_passed"] = False
            tests["exit_code"] = -1
        self._last_tests = tests
        self._last_diagnostics = {
            "runner": runner,
            "stdout_tail": truncate_text(stdout, self._sandbox.max_output_chars)[0],
            "stderr_tail": truncate_text(stderr, 512)[0],
            "timed_out": bool(execution["timed_out"]),
            "infra_error": infra_error,
            "network_isolated": bool(execution["network_isolated"]),
            "sandbox": execution["sandbox"],
        }
        return dict(tests)

    # -- env state --------------------------------------------------------

    def _env_state_fields(self) -> dict[str, Any]:
        return {
            "num_turns": len(self._turns),
            "timeout": bool(self._timeout),
            "code_tests": dict(self._last_tests) if self._last_tests else None,
            "rendered": self._render_tests(),
            "extras": {
                "runner": self._last_diagnostics.get("runner", self._runner),
                "tests_configured": self._unit_tests is not None,
                "infra_error": bool(
                    self._last_diagnostics.get("infra_error", False)
                ),
                "diagnostics": dict(self._last_diagnostics),
            },
        }

    def _render_tests(self) -> str:
        if not self._last_tests:
            return "code_tests: not run"
        tests = self._last_tests
        return (
            f"code_tests: total={tests['total']} passed={tests['passed']} "
            f"failed={tests['failed']} all_passed={tests['all_passed']} "
            f"infra_error={tests['infra_error']} "
            f"exit_code={tests['exit_code']}"
        )

    def _initial_content(self) -> str:
        return ""


def run_unit_tests(
    code: str,
    unit_tests: str,
    *,
    runner: str = "auto",
    timeout_s: float = DEFAULT_TEST_TIMEOUT_S,
    redact: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Standalone helper (used by tests / replay checks) returning the C5 dict."""
    if not isinstance(unit_tests, str) or not unit_tests.strip():
        raise EnvError("run_unit_tests() requires non-empty unit_tests source")
    task: dict[str, Any] = {
        "reference": {"unit_tests": unit_tests},
        "env": {"config": {"runner": runner, "timeout_s": timeout_s}},
        "tools": [],
        "meta": {"max_turns": 1},
    }
    config = {
        "runner": runner,
        "timeout_s": timeout_s,
        "redact": list(redact or []),
    }
    env = CodeTestsEnv(task=task, config=config)  # type: ignore[arg-type]
    return env._run_tests(code)  # noqa: SLF001 - intentional standalone helper


__all__ = [
    "CODE_ARG_KEYS",
    "CodeTestsEnv",
    "DEFAULT_MAX_FAILURE_MESSAGE",
    "DEFAULT_SOLUTION_FILENAME",
    "DEFAULT_TEST_FILENAME",
    "DEFAULT_TEST_TIMEOUT_S",
    "EXEC_TOOL_ALIASES",
    "extract_code",
    "parse_test_output",
    "run_unit_tests",
]
