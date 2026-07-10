"""Smoke tests for Hermes Gateway agent entrypoint and runner.

Usage:
    pytest Rl_Specilist/agent/online_dpo/tests/test_hermes_entrypoint.py -v
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure the Rl_Specialist package is importable
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

HERMES_ENTRYPOINT = (
    REPO_ROOT / "Rl_Specilist/agent/online_dpo/hermes_entrypoint.py"
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _import_entrypoint():
    """Import hermes_entrypoint as a module for testing."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "hermes_entrypoint", HERMES_ENTRYPOINT
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Test: Hermes tool-call parsing ──────────────────────────────────────────


class TestParseHermesToolCalls:
    """Tests for parse_hermes_tool_calls."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.mod = _import_entrypoint()

    def test_single_tool_call(self):
        """Parse a single valid Hermes tool call."""
        tools = [
            {"function": {"name": "execute_bash"}},
            {"function": {"name": "submit_answer"}},
        ]
        text = 'Let me check the files.\n<tool_call>\n{"name": "execute_bash", "arguments": {"command": "ls"}}\n</tool_call>'
        content, calls = self.mod.parse_hermes_tool_calls(text, tools)
        assert content == "Let me check the files."
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "execute_bash"
        assert json.loads(calls[0]["function"]["arguments"]) == {"command": "ls"}

    def test_multiple_tool_calls(self):
        """Parse multiple Hermes tool calls in one response."""
        tools = [
            {"function": {"name": "execute_bash"}},
            {"function": {"name": "read_file"}},
        ]
        text = (
            '<tool_call>\n{"name": "execute_bash", "arguments": {"command": "echo hi"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "read_file", "arguments": {"file_path": "out.txt"}}\n</tool_call>'
        )
        content, calls = self.mod.parse_hermes_tool_calls(text, tools)
        assert len(calls) == 2
        assert calls[0]["function"]["name"] == "execute_bash"
        assert calls[1]["function"]["name"] == "read_file"

    def test_no_tool_call(self):
        """Plain text without tool calls returns empty list."""
        tools = [{"function": {"name": "execute_bash"}}]
        text = "The task is complete. The answer is 42."
        content, calls = self.mod.parse_hermes_tool_calls(text, tools)
        assert content == text
        assert calls == []

    def test_unknown_tool_skipped(self):
        """Unknown tool names are skipped."""
        tools = [{"function": {"name": "execute_bash"}}]
        text = '<tool_call>\n{"name": "nonexistent", "arguments": {}}\n</tool_call>'
        content, calls = self.mod.parse_hermes_tool_calls(text, tools)
        assert calls == []

    def test_invalid_json_skipped(self):
        """Malformed JSON in tool call is skipped."""
        tools = [{"function": {"name": "execute_bash"}}]
        text = "<tool_call>\nnot valid json\n</tool_call>"
        content, calls = self.mod.parse_hermes_tool_calls(text, tools)
        assert calls == []

    def test_no_content_before_tool_call(self):
        """When tool call starts the response, content is empty."""
        tools = [{"function": {"name": "submit_answer"}}]
        text = '<tool_call>\n{"name": "submit_answer", "arguments": {"answer": "done"}}\n</tool_call>'
        content, calls = self.mod.parse_hermes_tool_calls(text, tools)
        assert content == ""
        assert len(calls) == 1

    def test_dict_arguments_preserved(self):
        """Arguments as dict objects are JSON-serialized."""
        tools = [{"function": {"name": "execute_bash"}}]
        text = '<tool_call>\n{"name": "execute_bash", "arguments": {"command": "ls -la"}}\n</tool_call>'
        _content, calls = self.mod.parse_hermes_tool_calls(text, tools)
        args = json.loads(calls[0]["function"]["arguments"])
        assert args == {"command": "ls -la"}


# ── Test: Tool execution ────────────────────────────────────────────────────


class TestExecuteTool:
    """Tests for execute_tool in workspace."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.mod = _import_entrypoint()
        self.workspace = str(tmp_path)

    def test_submit_answer(self):
        """submit_answer returns finished status."""
        result = self.mod.execute_tool(
            "submit_answer", json.dumps({"answer": "done"}), self.workspace
        )
        parsed = json.loads(result)
        assert parsed["status"] == "finished"
        assert parsed["message"] == "done"

    def test_bash_echo(self):
        """execute_bash runs a bash command."""
        result = self.mod.execute_tool(
            "execute_bash",
            json.dumps({"command": "echo hello world"}),
            self.workspace,
        )
        assert "hello world" in result

    def test_bash_no_command(self):
        """execute_bash with no command returns error."""
        result = self.mod.execute_tool(
            "execute_bash", json.dumps({}), self.workspace
        )
        assert "Error" in result

    def test_write_and_read_file(self):
        """write_file + read_file round-trip."""
        self.mod.execute_tool(
            "write_file",
            json.dumps({"file_path": "test.txt", "content": "hello content"}),
            self.workspace,
        )
        result = self.mod.execute_tool(
            "read_file",
            json.dumps({"file_path": "test.txt"}),
            self.workspace,
        )
        assert "hello content" in result

    def test_read_nonexistent_file(self):
        """read_file for nonexistent file returns error."""
        result = self.mod.execute_tool(
            "read_file",
            json.dumps({"file_path": "nonexistent.txt"}),
            self.workspace,
        )
        assert "Error" in result.lower() or "not found" in result.lower()

    def test_path_sandbox(self):
        """read_file cannot escape workspace."""
        result = self.mod.execute_tool(
            "read_file",
            json.dumps({"file_path": "../../etc/passwd"}),
            self.workspace,
        )
        assert "outside" in result.lower() or "error" in result.lower()

    def test_unknown_tool(self):
        """Unknown tool returns error."""
        result = self.mod.execute_tool(
            "unknown_tool", "{}", self.workspace
        )
        assert "Error" in result


# ── Test: Path validation ───────────────────────────────────────────────────


class TestValidatePath:
    """Tests for _validate_path."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.mod = _import_entrypoint()
        self.workspace = str(tmp_path)

    def test_valid_path(self):
        """A path within workspace is accepted."""
        result = self.mod._validate_path("subdir/file.txt", self.workspace)
        assert result.endswith("subdir/file.txt")

    def test_path_escape(self):
        """A path escaping workspace raises ValueError."""
        with pytest.raises(ValueError, match="outside"):
            self.mod._validate_path("../../etc/passwd", self.workspace)

    def test_absolute_path_anchored(self):
        """An absolute path is anchored to workspace."""
        with pytest.raises(ValueError, match="outside"):
            self.mod._validate_path("/etc/passwd", self.workspace)


# ── Test: Task extraction ───────────────────────────────────────────────────


class TestTaskExtraction:
    """Tests for extract_task and build_hermes_task in the runner."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from Rl_Specilist.agent.online_dpo.custom_hermes_runner import (
            build_hermes_task,
            extract_task,
        )

        self.extract_task = extract_task
        self.build_hermes_task = build_hermes_task

    def test_extract_string(self):
        """Bare string prompt is returned unchanged."""
        assert self.extract_task("do something") == "do something"

    def test_extract_messages(self):
        """OpenAI-format message list extracts user content."""
        prompt = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "run the tests"},
        ]
        assert self.extract_task(prompt) == "run the tests"

    def test_extract_messages_multimodal(self):
        """Multi-modal content blocks extract first text."""
        prompt = [
            {"role": "user", "content": [
                {"type": "text", "text": "analyze this"},
                {"type": "image_url", "image_url": {"url": "http://..."}},
            ]},
        ]
        assert self.extract_task(prompt) == "analyze this"

    def test_build_task_prebuilt(self):
        """Pre-built task in tools_kwargs is used."""
        result = self.build_hermes_task(
            "ignored", {"task": "prebuilt task"}
        )
        assert result == "prebuilt task"

    def test_build_task_fallback(self):
        """Without prebuilt task, raw prompt is used."""
        result = self.build_hermes_task("bare task", {})
        assert result == "bare task"


# ── Integration: end-to-end with entrypoint subprocess ──────────────────────


class TestEntrypointSubprocess:
    """Launch hermes_entrypoint.py as a subprocess and verify it starts."""

    def test_entrypoint_requires_env(self):
        """Entrypoint fails gracefully when env vars are missing."""
        import asyncio

        async def _run():
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                str(HERMES_ENTRYPOINT),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await proc.communicate()
            return proc.returncode, stdout.decode()

        rc, stdout = asyncio.run(_run())
        assert rc != 0
        assert "HERMES_TASK" in stdout

    def test_entrypoint_help_info(self):
        """Entrypoint module docstring is accessible."""
        mod = _import_entrypoint()
        assert mod.__doc__ is not None
        assert "Hermes-format" in mod.__doc__


# ── Test: Runner SessionHandle fallback ─────────────────────────────────────


class TestSessionHandleFallback:
    """When uni-agent is not installed, the runner provides a fallback."""

    def test_fallback_dataclass(self):
        """SessionHandle fallback is a functional dataclass."""
        # Simulate ImportError by importing directly from the runner
        import importlib

        runner_mod = importlib.import_module(
            "Rl_Specilist.agent.online_dpo.custom_hermes_runner"
        )
        # If uni-agent is installed, SessionHandle comes from there;
        # either way it should be a dataclass with the right fields.
        sh = runner_mod.SessionHandle(
            session_id="test-123",
            base_url="http://localhost:8000/sessions/test-123/v1",
            reward_info_url="http://localhost:8000/sessions/test-123/reward_info",
        )
        assert sh.session_id == "test-123"
        assert sh.base_url is not None
        assert sh.reward_info_url is not None
