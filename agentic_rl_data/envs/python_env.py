"""Sandboxed Python execution (plan §4.4) and the ``math_ci`` PythonEnv.

The sandbox is a **subprocess** with:

* a hard wall-clock timeout (process group killed on expiry);
* POSIX resource limits via ``preexec_fn`` (CPU seconds, address space,
  file size, open files, processes) — ``resource`` is imported guarded and is
  absent on Windows, where only the timeout applies;
* a fresh temporary working directory per execution (removed afterwards);
* a cleared environment (allowlist only), ``python -I -B`` so per-user site
  packages / ``PYTHONPATH`` / ``PYTHONSTARTUP`` cannot leak in;
* fixed-length output truncation (``max_output_chars``).

**Network isolation — honest limits.** A plain subprocess cannot isolate the
network with stdlib APIs.  Real isolation requires an external sandbox:
``bwrap --unshare-net``, ``firejail --net=none``, or a network-namespaced
container.  ``SandboxConfig.sandbox_prefix`` accepts such a wrapper
(e.g. ``("bwrap", "--unshare-net", ...)``); when it is not configured,
``result["network_isolated"]`` is ``False`` and the report states this
explicitly.  File-system isolation is likewise only "fresh cwd + rlimits";
a script can still read host files unless a real sandbox wrapper is used.
See ``reports/env_setup.md`` for the recommended wrapper commands.

``PythonEnv`` (``env.type == "python"``, used by ``math_ci``) exposes a
``python(code)`` tool (aliases accepted) and returns executables' stdout /
stderr / exit_code as canonical JSON in the tool result.  This module is
stdlib-only at import time.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..schema import canonical_json
from .base import BaseEnv, EnvError, make_tool_result, truncate_text

try:  # POSIX resource limits (absent on Windows)
    import resource as _resource  # type: ignore
except ImportError:  # pragma: no cover - Windows
    _resource = None  # type: ignore

DEFAULT_TIMEOUT_S = 5.0
DEFAULT_MAX_OUTPUT_CHARS = 4096
DEFAULT_MEMORY_MB = 512
DEFAULT_MAX_FILE_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_OPEN_FILES = 128
DEFAULT_MAX_PROCESSES = 32

#: Environment variables kept in the child process (everything else dropped).
_ENV_ALLOWLIST = (
    "PATH",
    "HOME",
    "USERPROFILE",
    "SYSTEMROOT",
    "WINDIR",
    "TEMP",
    "TMP",
    "LANG",
    "LC_ALL",
    "TZ",
    "PYTHONIOENCODING",
    "PYTHONHASHSEED",
)

#: Python tool names accepted by :class:`PythonEnv`.
PYTHON_TOOL_ALIASES = (
    "python",
    "execute_python",
    "run_python",
    "python_interpreter",
    "code_interpreter",
)

#: Argument keys accepted as the code payload.
PYTHON_CODE_KEYS = ("code", "input", "source", "script", "python", "statement")


@dataclass
class SandboxConfig:
    """Resource / timeout / isolation settings for one sandboxed execution."""

    timeout_s: float = DEFAULT_TIMEOUT_S
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS
    memory_mb: int = DEFAULT_MEMORY_MB
    cpu_seconds: int | None = None
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    max_open_files: int = DEFAULT_MAX_OPEN_FILES
    max_processes: int = DEFAULT_MAX_PROCESSES
    #: External sandbox wrapper prepended to the argv (e.g. bwrap / firejail).
    sandbox_prefix: tuple[str, ...] = field(default_factory=tuple)
    #: True only when a wrapper was configured to block network access.
    network_isolated: bool = False
    clear_env: bool = True

    def resolved_timeout(self) -> float:
        return max(0.1, float(self.timeout_s))

    def resolved_cpu_seconds(self) -> int:
        if self.cpu_seconds is not None:
            return max(1, int(self.cpu_seconds))
        return max(1, int(self.timeout_s) + 1)


def build_sandbox_config(
    config: Mapping[str, Any] | None = None,
    *,
    timeout_s: float | None = None,
) -> SandboxConfig:
    """Build a :class:`SandboxConfig` from ``env.config`` style keys."""
    config = config if isinstance(config, Mapping) else {}

    def _positive_int(key: str, default: int) -> int:
        raw = config.get(key)
        if isinstance(raw, bool) or raw is None:
            return default
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return default
        return value if value > 0 else default

    prefix = config.get("sandbox_prefix") or ()
    if isinstance(prefix, str):
        prefix = (prefix,)
    prefix = tuple(str(part) for part in prefix if str(part).strip())
    network_isolated = bool(config.get("network_isolated", False)) and bool(prefix)

    return SandboxConfig(
        timeout_s=(
            float(timeout_s)
            if timeout_s is not None
            else float(config.get("timeout_s", DEFAULT_TIMEOUT_S))
        ),
        max_output_chars=_positive_int("max_output_chars", DEFAULT_MAX_OUTPUT_CHARS),
        memory_mb=_positive_int("memory_mb", DEFAULT_MEMORY_MB),
        cpu_seconds=(
            _positive_int("cpu_seconds", 0) or None
        ),
        max_file_bytes=_positive_int("max_file_bytes", DEFAULT_MAX_FILE_BYTES),
        max_open_files=_positive_int("max_open_files", DEFAULT_MAX_OPEN_FILES),
        max_processes=_positive_int("max_processes", DEFAULT_MAX_PROCESSES),
        sandbox_prefix=prefix,
        network_isolated=network_isolated,
        clear_env=bool(config.get("clear_env", True)),
    )


# ---------------------------------------------------------------------------
# process execution
# ---------------------------------------------------------------------------


def _child_environment(config: SandboxConfig, extra: Mapping[str, str] | None) -> dict[str, str]:
    if not config.clear_env:
        environment = dict(os.environ)
    else:
        environment = {
            name: os.environ[name]
            for name in _ENV_ALLOWLIST
            if name in os.environ
        }
    environment.setdefault("PYTHONIOENCODING", "utf-8")
    environment.setdefault("PYTHONHASHSEED", "0")
    if extra:
        environment.update({str(key): str(value) for key, value in extra.items()})
    return environment


def _preexec_fn(config: SandboxConfig):
    """Return the POSIX rlimit hook (``None`` on Windows / no resource module)."""
    if os.name == "nt" or _resource is None:
        return None

    def _limit() -> None:
        limits: list[tuple[int, int]] = []
        try:
            cpu = config.resolved_cpu_seconds()
            limits.append((_resource.RLIMIT_CPU, cpu))
            if config.memory_mb > 0:
                limits.append((_resource.RLIMIT_AS, int(config.memory_mb) * 1024 * 1024))
            if config.max_file_bytes > 0:
                limits.append((_resource.RLIMIT_FSIZE, int(config.max_file_bytes)))
            if config.max_open_files > 0:
                limits.append((_resource.RLIMIT_NOFILE, int(config.max_open_files)))
            if config.max_processes > 0 and hasattr(_resource, "RLIMIT_NPROC"):
                limits.append((_resource.RLIMIT_NPROC, int(config.max_processes)))
        except (ValueError, OSError):  # pragma: no cover - platform specific
            return
        for resource_name, value in limits:
            try:
                _resource.setrlimit(resource_name, (value, value))
            except (ValueError, OSError):  # pragma: no cover - platform specific
                continue

    return _limit


def _popen_kwargs(config: SandboxConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if os.name != "nt":
        kwargs["start_new_session"] = True
        preexec = _preexec_fn(config)
        if preexec is not None:
            kwargs["preexec_fn"] = preexec
    else:  # pragma: no cover - Windows
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        if creationflags:
            kwargs["creationflags"] = creationflags
    return kwargs


def _terminate(process: subprocess.Popen[bytes]) -> None:
    """Kill the whole process group (best effort, cross-platform)."""
    try:
        if os.name != "nt":
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        else:  # pragma: no cover - Windows
            process.kill()
    except (ProcessLookupError, PermissionError, OSError, AttributeError):
        try:
            process.kill()
        except Exception:  # pragma: no cover - defensive
            pass


def run_process(
    argv: Sequence[str],
    *,
    cwd: str | Path | None = None,
    config: SandboxConfig | None = None,
    stdin_text: str | None = None,
    extra_env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run ``argv`` inside the sandbox and capture bounded output.

    Returns ``{"stdout", "stderr", "exit_code", "timed_out", "duration_s",
    "truncated", "sandbox", "network_isolated"}``.  ``exit_code`` is ``-1``
    when the process was killed for timing out (documented contract), otherwise
    the real exit code.
    """
    config = config or SandboxConfig()
    timeout_s = config.resolved_timeout()
    if not argv:
        raise EnvError("run_process() requires a non-empty argv")
    command = [str(part) for part in argv]
    if config.sandbox_prefix:
        command = list(config.sandbox_prefix) + command
    sandbox = "wrapper" if config.sandbox_prefix else "subprocess"
    if os.name == "nt":
        sandbox = "subprocess-win"

    environment = _child_environment(config, extra_env)
    process: subprocess.Popen[bytes] | None = None
    started = time.monotonic()
    timed_out = False
    stdout_bytes: bytes = b""
    stderr_bytes: bytes = b""
    exit_code: int = -1
    try:
        process = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd is not None else None,
            stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            **_popen_kwargs(config),
        )
        stdin_bytes = stdin_text.encode("utf-8") if stdin_text is not None else None
        try:
            stdout_bytes, stderr_bytes = process.communicate(
                input=stdin_bytes, timeout=timeout_s
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate(process)
            try:
                stdout_bytes, stderr_bytes = process.communicate(timeout=2.0)
            except subprocess.TimeoutExpired:  # pragma: no cover - unkillable
                stdout_bytes, stderr_bytes = b"", b""
            exit_code = -1
        else:
            exit_code = process.returncode if process.returncode is not None else -1
    finally:
        if process is not None and process.poll() is None:
            _terminate(process)

    duration = time.monotonic() - started
    stdout, stdout_truncated = truncate_text(
        stdout_bytes.decode("utf-8", "replace"), config.max_output_chars
    )
    stderr, stderr_truncated = truncate_text(
        stderr_bytes.decode("utf-8", "replace"), config.max_output_chars
    )
    return {
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": int(exit_code),
        "timed_out": bool(timed_out),
        "duration_s": round(duration, 3),
        "truncated": bool(stdout_truncated or stderr_truncated),
        "sandbox": sandbox,
        "network_isolated": bool(config.network_isolated),
    }


def run_python(
    code: str,
    *,
    config: SandboxConfig | None = None,
    timeout_s: float | None = None,
    argv_prefix: Sequence[str] | None = None,
    extra_env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Execute ``code`` in a fresh temporary directory with the sandbox limits."""
    if not isinstance(code, str) or not code.strip():
        raise EnvError("run_python() requires non-empty source code")
    config = config or SandboxConfig()
    if timeout_s is not None:
        config = SandboxConfig(**{**config.__dict__, "timeout_s": float(timeout_s)})
    temporary_directory = tempfile.mkdtemp(prefix="ard_python_")
    script_path = Path(temporary_directory) / "script.py"
    try:
        script_path.write_text(code, encoding="utf-8")
        if os.name != "nt":
            try:
                os.chmod(script_path, 0o600)
            except OSError:  # pragma: no cover - defensive
                pass
        prefix = list(argv_prefix) if argv_prefix else []
        argv = prefix + [sys.executable or "python", "-I", "-B", str(script_path)]
        result = run_process(
            argv,
            cwd=temporary_directory,
            config=config,
            extra_env=extra_env,
        )
        result["script"] = str(script_path)
        return result
    finally:
        shutil.rmtree(temporary_directory, ignore_errors=True)


# ---------------------------------------------------------------------------
# PythonEnv (math_ci)
# ---------------------------------------------------------------------------


def _argument(arguments: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in arguments:
            return arguments[name]
    return default


def _match_python_tool(name: str, aliases: Sequence[str]) -> bool:
    candidate = str(name).strip().lower()
    if candidate in {alias.lower() for alias in aliases}:
        return True
    return any(alias in candidate for alias in ("python", "interpreter", "execute"))


class PythonEnv(BaseEnv):
    """``math_ci`` environment: one sandboxed ``python(code)`` tool.

    ``env.config`` keys::

        {"timeout_s": 5, "max_output_chars": 4096, "memory_mb": 512,
         "cpu_seconds": null, "max_file_bytes": ..., "max_open_files": ...,
         "max_processes": ..., "sandbox_prefix": [...],
         "network_isolated": false, "tools": ["python", ...]}

    A call to an undeclared / unknown tool is a recoverable ``ok=False`` result;
    a script that exceeds the timeout returns ``timed_out=True`` with
    ``exit_code=-1``.  The environment itself never raises for user code.
    """

    env_type = "python"
    impl = "local"

    def _setup(self, task: Mapping[str, Any], config: dict[str, Any]) -> None:
        self._sandbox = build_sandbox_config(config)
        tools = config.get("tools") or PYTHON_TOOL_ALIASES
        if isinstance(tools, str):
            tools = (tools,)
        self._tool_aliases: tuple[str, ...] = tuple(str(tool) for tool in tools)
        self._executions: list[dict[str, Any]] = []

    def _handle_tool_calls(
        self,
        calls: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for call in calls:
            if not _match_python_tool(call["name"], self._tool_aliases):
                results.append(
                    make_tool_result(
                        call["name"],
                        f"Error: unknown python tool {call['name']!r}",
                        ok=False,
                        call_id=call["id"],
                    )
                )
                continue
            code = _argument(call["arguments"], *PYTHON_CODE_KEYS)
            if not isinstance(code, str) or not code.strip():
                results.append(
                    make_tool_result(
                        call["name"],
                        "Error: python tool requires non-empty string argument "
                        f"({'/'.join(PYTHON_CODE_KEYS[:3])})",
                        ok=False,
                        call_id=call["id"],
                    )
                )
                continue
            try:
                execution = run_python(code, config=self._sandbox)
            except EnvError as exc:  # protocol-level, still recoverable for the policy
                results.append(
                    make_tool_result(
                        call["name"], f"Error: {exc}", ok=False, call_id=call["id"]
                    )
                )
                continue
            except Exception as exc:  # defensive: host failure must not kill rollout
                results.append(
                    make_tool_result(
                        call["name"],
                        f"Error: sandbox failure: {type(exc).__name__}: {exc}",
                        ok=False,
                        call_id=call["id"],
                    )
                )
                continue
            success = (not execution["timed_out"]) and execution["exit_code"] == 0
            summary = {
                "stdout": execution["stdout"],
                "stderr": execution["stderr"],
                "exit_code": execution["exit_code"],
                "timed_out": execution["timed_out"],
                "duration_s": execution["duration_s"],
                "truncated": execution["truncated"],
            }
            self._executions.append(summary)
            results.append(
                make_tool_result(
                    call["name"],
                    canonical_json(summary),
                    ok=success,
                    call_id=call["id"],
                )
            )
        info: dict[str, Any] = {
            "executions": [
                dict(execution) for execution in self._executions[-len(calls) :]
            ]
            if calls
            else [],
            "sandbox": self._sandbox.__dict__.copy(),
        }
        return results, info

    def _env_state_fields(self) -> dict[str, Any]:
        return {
            "num_turns": len(self._turns),
            "timeout": bool(self._timeout),
            "rendered": (
                f"python executions={len(self._executions)} "
                f"timeouts={sum(1 for item in self._executions if item['timed_out'])}"
            ),
            "extras": {
                "executions": [dict(item) for item in self._executions],
                "sandbox": {
                    "timeout_s": self._sandbox.timeout_s,
                    "memory_mb": self._sandbox.memory_mb,
                    "max_output_chars": self._sandbox.max_output_chars,
                    "network_isolated": self._sandbox.network_isolated,
                    "sandbox_prefix": list(self._sandbox.sandbox_prefix),
                },
                "network_isolated": self._sandbox.network_isolated,
            },
        }


__all__ = [
    "DEFAULT_MAX_OUTPUT_CHARS",
    "DEFAULT_MEMORY_MB",
    "DEFAULT_TIMEOUT_S",
    "PYTHON_CODE_KEYS",
    "PYTHON_TOOL_ALIASES",
    "PythonEnv",
    "SandboxConfig",
    "build_sandbox_config",
    "run_process",
    "run_python",
]
