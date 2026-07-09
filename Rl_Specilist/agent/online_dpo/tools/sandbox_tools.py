# Copyright 2025 Individual Contributor
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Sandbox tool implementations for Online DPO.

Each tool runs in an isolated workspace shared across tools within the
same trajectory.  Workspaces are keyed by ``agent_data.request_id``.

Workspace lifecycle::

    first tool call → mkdir /tmp/verl_sandbox/<request_id>
    all tool calls → execute in that workspace
    cleanup        → rmtree when trajectory completes

Backends (configured via tool ``config.backend``):

    ``subprocess`` (default)
        Run commands directly on the host via asyncio subprocess.
    ``docker``
        Run commands inside a Docker container (requires image).
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import Any, Optional

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

# ---------------------------------------------------------------------------
# Shared workspace registry
# ---------------------------------------------------------------------------
# Maps agent_data.request_id → workspace directory.
# Each trajectory gets one workspace; all tools within the trajectory share it.

_workspaces: dict[str, Path] = {}


def _ensure_workspace(agent_data: Any, base_dir: str = "/tmp/verl_sandbox") -> Path:
    """Get or create the workspace directory for a trajectory."""
    rid = agent_data.request_id
    if rid not in _workspaces:
        ws = Path(base_dir) / rid
        ws.mkdir(parents=True, exist_ok=True)
        _workspaces[rid] = ws
    return _workspaces[rid]


def _cleanup_workspace(request_id: str) -> None:
    """Remove the workspace directory."""
    ws = _workspaces.pop(request_id, None)
    if ws and ws.exists():
        shutil.rmtree(ws, ignore_errors=True)


# ---------------------------------------------------------------------------
# SandboxBashTool
# ---------------------------------------------------------------------------


class SandboxBashTool(BaseTool):
    """Execute bash commands in an isolated workspace."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._timeout = int(config.get("timeout", 30))
        self._max_output = int(config.get("max_output", 4000))
        self._workspace_base = config.get("workspace_base", "/tmp/verl_sandbox")
        self._backend = config.get("backend", "subprocess")

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        """Run a bash command in the trajectory workspace.

        Args:
            parameters: Must contain ``command`` (str).
        """
        agent_data = kwargs.get("agent_data")
        command = parameters.get("command", "")

        if not command:
            return ToolResponse(text="Error: 'command' parameter is required."), 0.0, {}

        ws = _ensure_workspace(agent_data, self._workspace_base)

        try:
            output = await _run_bash(
                command=command,
                cwd=str(ws),
                timeout=self._timeout,
                backend=self._backend,
            )
            if len(output) > self._max_output:
                output = output[: self._max_output] + "\n...[output truncated]"
            if not output.strip():
                output = "(command completed successfully with no output)"
        except asyncio.TimeoutError:
            output = (
                f"Command timed out after {self._timeout}s: {command[:200]}"
            )
        except Exception as exc:
            output = f"Error executing command: {exc}"

        return ToolResponse(text=output), 0.0, {}


# ---------------------------------------------------------------------------
# SandboxReadTool
# ---------------------------------------------------------------------------


class SandboxReadTool(BaseTool):
    """Read a file from the isolated workspace."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._max_lines = int(config.get("max_lines", 500))
        self._workspace_base = config.get("workspace_base", "/tmp/verl_sandbox")

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs.get("agent_data")
        file_path = parameters.get("file_path", "")

        if not file_path:
            return ToolResponse(text="Error: 'file_path' parameter is required."), 0.0, {}

        ws = _ensure_workspace(agent_data, self._workspace_base)

        # Resolve path — refuse to escape the workspace
        target = _resolve_safe_path(ws, file_path)
        if target is None:
            return ToolResponse(
                text=f"Error: path '{file_path}' is outside the workspace."
            ), 0.0, {}

        if not target.exists():
            return ToolResponse(text=f"Error: file not found: {file_path}"), 0.0, {}

        if target.is_dir():
            try:
                listing = "\n".join(
                    sorted(p.name for p in target.iterdir())
                )
                return ToolResponse(
                    text=f"Directory listing of '{file_path}':\n{listing}"
                ), 0.0, {}
            except PermissionError:
                return ToolResponse(
                    text=f"Error: permission denied reading directory '{file_path}'"
                ), 0.0, {}

        try:
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
            total = len(lines)
            if total > self._max_lines:
                lines = lines[: self._max_lines]
                lines.append(f"...({total - self._max_lines} more lines)")
            content = "\n".join(lines)
        except Exception as exc:
            return ToolResponse(text=f"Error reading file: {exc}"), 0.0, {}

        return ToolResponse(text=content), 0.0, {}


# ---------------------------------------------------------------------------
# SandboxWriteTool
# ---------------------------------------------------------------------------


class SandboxWriteTool(BaseTool):
    """Write content to a file in the isolated workspace."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._max_size = int(config.get("max_size", 100_000))
        self._workspace_base = config.get("workspace_base", "/tmp/verl_sandbox")

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs.get("agent_data")
        file_path = parameters.get("file_path", "")
        content = parameters.get("content", "")

        if not file_path:
            return ToolResponse(text="Error: 'file_path' parameter is required."), 0.0, {}
        if len(content) > self._max_size:
            return ToolResponse(
                text=f"Error: content too large ({len(content)} chars, max {self._max_size})"
            ), 0.0, {}

        ws = _ensure_workspace(agent_data, self._workspace_base)
        target = _resolve_safe_path(ws, file_path)
        if target is None:
            return ToolResponse(
                text=f"Error: path '{file_path}' is outside the workspace."
            ), 0.0, {}

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except Exception as exc:
            return ToolResponse(text=f"Error writing file: {exc}"), 0.0, {}

        return ToolResponse(
            text=f"Successfully wrote {len(content)} chars to '{file_path}'."
        ), 0.0, {}


# ---------------------------------------------------------------------------
# SandboxSubmitTool
# ---------------------------------------------------------------------------


class SandboxSubmitTool(BaseTool):
    """Submit the final answer.  No-op — serves as a termination signal.

    ToolAgentLoop detects ``submit`` / ``finish`` tool names and treats
    them as the end of the trajectory.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        answer = parameters.get("answer", "")
        return ToolResponse(
            text=f"Answer submitted: {answer[:500]}"
        ), 0.0, {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_safe_path(workspace: Path, user_path: str) -> Optional[Path]:
    """Resolve a user-supplied path safely within the workspace.

    Returns ``None`` if the path escapes the workspace.
    """
    p = (workspace / user_path).resolve()
    try:
        p.relative_to(workspace.resolve())
    except ValueError:
        return None
    return p


async def _run_bash(
    command: str, cwd: str, timeout: int, backend: str = "subprocess",
) -> str:
    """Execute a bash command in the given backend environment.

    Backends:
        ``subprocess`` (default) — bare ``bash -c`` on host.
        ``claude`` — run inside a Claude Code-managed workspace.  Sets
            ``CLAUDE_CODE_*`` env vars and runs via ``claude`` CLI context.
        ``hermes`` — run inside a Hermes-managed workspace.  Sets
            ``HERMES_HOME`` and loads hermes config.
    """
    env = os.environ.copy()

    if backend == "claude":
        # Use Claude Code's node/npm environment
        env["CLAUDE_CODE_WORKSPACE"] = cwd
        # Don't use `claude` for each command — that would be slow.
        # Instead, provide the same env that Claude Code would.
        npm_bin = os.path.dirname(shutil.which("claude") or "")
        if npm_bin:
            env["PATH"] = f"{npm_bin}:{env.get('PATH', '')}"
    elif backend == "hermes":
        # Use Hermes's home/config
        hermes_home = os.environ.get(
            "HERMES_HOME",
            os.path.expanduser("~/.hermes/hermes-agent"),
        )
        env["HERMES_HOME"] = hermes_home

    proc = await asyncio.wait_for(
        asyncio.create_subprocess_exec(
            "bash", "-c", command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
            env=env,
        ),
        timeout=10,
    )
    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    return stdout.decode("utf-8", errors="replace")
