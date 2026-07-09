#!/usr/bin/env python3
"""Smoke tests for Online DPO sandbox tools and utilities.

Usage:
    pytest Rl_Specilist/agent/online_dpo/runners/test_smoke.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Test: SandboxBashTool
# ---------------------------------------------------------------------------

class TestSandboxBashTool:
    def test_execute_basic(self):
        import asyncio
        from verl.tools.schemas import OpenAIFunctionToolSchema
        from Rl_Specilist.agent.online_dpo.tools.sandbox_tools import (
            SandboxBashTool, _cleanup_workspace,
        )

        schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "bash",
                "description": "execute bash",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
            },
        })
        tool = SandboxBashTool(
            config={"type": "native", "timeout": 10, "workspace_base": "/tmp/test_sandbox"},
            tool_schema=schema,
        )

        agent_data = MagicMock()
        agent_data.request_id = "test_bash_001"

        async def _run():
            resp, reward, meta = await tool.execute(
                "inst_1", {"command": "echo hello"}, agent_data=agent_data
            )
            assert "hello" in resp.text
            _cleanup_workspace(agent_data.request_id)

        asyncio.run(_run())

    def test_workspace_persistence(self):
        """Multiple tool calls share the same workspace."""
        import asyncio
        from verl.tools.schemas import OpenAIFunctionToolSchema
        from Rl_Specilist.agent.online_dpo.tools.sandbox_tools import (
            SandboxBashTool, _cleanup_workspace,
        )

        schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "bash",
                "description": "bash",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
            },
        })
        tool = SandboxBashTool(
            config={"type": "native", "timeout": 10, "workspace_base": "/tmp/test_sandbox"},
            tool_schema=schema,
        )
        agent_data = MagicMock()
        agent_data.request_id = "test_persist_001"

        async def _run():
            # Write a file
            await tool.execute("inst", {"command": "echo hello > test.txt"}, agent_data=agent_data)
            # Read it back
            resp, _, _ = await tool.execute("inst", {"command": "cat test.txt"}, agent_data=agent_data)
            assert "hello" in resp.text
            _cleanup_workspace(agent_data.request_id)

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test: SandboxReadTool
# ---------------------------------------------------------------------------

class TestSandboxReadTool:
    def test_read_file(self):
        import asyncio
        from verl.tools.schemas import OpenAIFunctionToolSchema
        from Rl_Specilist.agent.online_dpo.tools.sandbox_tools import (
            SandboxBashTool, SandboxReadTool, _cleanup_workspace,
        )

        bash_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function", "function": {
                "name": "bash", "description": "...",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
            },
        })
        read_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function", "function": {
                "name": "read_file", "description": "...",
                "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]},
            },
        })
        bash = SandboxBashTool(
            config={"type": "native", "timeout": 10, "workspace_base": "/tmp/test_sandbox"},
            tool_schema=bash_schema,
        )
        read = SandboxReadTool(
            config={"type": "native", "workspace_base": "/tmp/test_sandbox"},
            tool_schema=read_schema,
        )

        agent_data = MagicMock()
        agent_data.request_id = "test_read_001"

        async def _run():
            await bash.execute("inst", {"command": "echo hello > readme.txt"}, agent_data=agent_data)
            resp, _, _ = await read.execute("inst", {"file_path": "readme.txt"}, agent_data=agent_data)
            assert "hello" in resp.text
            _cleanup_workspace(agent_data.request_id)

        asyncio.run(_run())

    def test_path_sandbox(self):
        """Cannot escape workspace with ../../etc/passwd."""
        import asyncio
        from verl.tools.schemas import OpenAIFunctionToolSchema
        from Rl_Specilist.agent.online_dpo.tools.sandbox_tools import (
            SandboxReadTool, _cleanup_workspace,
        )

        schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function", "function": {
                "name": "read_file", "description": "...",
                "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]},
            },
        })
        read = SandboxReadTool(
            config={"type": "native", "workspace_base": "/tmp/test_sandbox"},
            tool_schema=schema,
        )
        agent_data = MagicMock()
        agent_data.request_id = "test_sandbox_001"

        async def _run():
            resp, _, _ = await read.execute("inst", {"file_path": "../../etc/passwd"}, agent_data=agent_data)
            assert "outside" in resp.text.lower()
            _cleanup_workspace(agent_data.request_id)

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test: SandboxWriteTool
# ---------------------------------------------------------------------------

class TestSandboxWriteTool:
    def test_write_and_read(self):
        import asyncio
        from verl.tools.schemas import OpenAIFunctionToolSchema
        from Rl_Specilist.agent.online_dpo.tools.sandbox_tools import (
            SandboxWriteTool, SandboxReadTool, _cleanup_workspace,
        )

        write_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function", "function": {
                "name": "write_file", "description": "...",
                "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}, "required": ["file_path", "content"]},
            },
        })
        read_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function", "function": {
                "name": "read_file", "description": "...",
                "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]},
            },
        })
        write = SandboxWriteTool(
            config={"type": "native", "workspace_base": "/tmp/test_sandbox"},
            tool_schema=write_schema,
        )
        read = SandboxReadTool(
            config={"type": "native", "workspace_base": "/tmp/test_sandbox"},
            tool_schema=read_schema,
        )
        agent_data = MagicMock()
        agent_data.request_id = "test_write_001"

        async def _run():
            await write.execute("inst", {"file_path": "subdir/out.txt", "content": "hello world"}, agent_data=agent_data)
            resp, _, _ = await read.execute("inst", {"file_path": "subdir/out.txt"}, agent_data=agent_data)
            assert "hello world" in resp.text
            _cleanup_workspace(agent_data.request_id)

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test: SandboxSubmitTool
# ---------------------------------------------------------------------------

class TestSandboxSubmitTool:
    def test_submit(self):
        import asyncio
        from verl.tools.schemas import OpenAIFunctionToolSchema
        from Rl_Specilist.agent.online_dpo.tools.sandbox_tools import SandboxSubmitTool

        schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function", "function": {
                "name": "submit_answer", "description": "...",
                "parameters": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
            },
        })
        tool = SandboxSubmitTool(config={"type": "native"}, tool_schema=schema)

        async def _run():
            resp, _, _ = await tool.execute("inst", {"answer": "42"})
            assert "42" in resp.text

        asyncio.run(_run())


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
