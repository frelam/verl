# Copyright 2025 Individual Contributor
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Terminal sandbox tool for TerminalTraj dataset.

Executes bash commands inside a Docker container (image: reject-sft-terminal:latest).
Each tool instance creates a fresh container so trajectories are isolated.

The tool is registered as ``bash`` — the standard terminal agent tool name.
"""

from __future__ import annotations

import shutil
import subprocess
import textwrap
from typing import Any, Dict, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

# Docker image name (built by setup/install_env.sh)
DEFAULT_IMAGE = "reject-sft-terminal:latest"
# Max command execution time (seconds)
DEFAULT_TIMEOUT = 30
# Max output length (characters)
MAX_OUTPUT = 4000


class TerminalSandboxTool(BaseTool):
    """Execute bash commands in an isolated Docker container.

    Each ``create()`` call starts a new container; ``execute()`` runs commands
    via ``docker exec``; ``release()`` stops and removes the container.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._image = config.get("image", DEFAULT_IMAGE)
        self._timeout = config.get("timeout", DEFAULT_TIMEOUT)
        self._instance_dict: Dict[str, dict] = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())

        container_name = f"reject-sft-{instance_id[:12]}"

        # Start a detached container
        try:
            subprocess.run(
                ["docker", "run", "-d", "--name", container_name,
                 "-w", "/workspace", self._image, "sleep", "3600"],
                capture_output=True, text=True, timeout=60,
                check=True,
            )
            self._instance_dict[instance_id] = {
                "container_name": container_name,
                "call_count": 0,
                "commands": [],
                "exit_codes": [],
            }
            return instance_id, ToolResponse(text="Terminal environment ready.")
        except subprocess.CalledProcessError as e:
            return instance_id, ToolResponse(
                text=f"Error starting container: {e.stderr or e.stdout}"
            )
        except Exception as e:
            return instance_id, ToolResponse(text=f"Error: {type(e).__name__}: {e}")

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        state = self._instance_dict.get(instance_id)
        if not state:
            return (
                ToolResponse(text="Error: terminal not initialised. Call create() first."),
                -0.1,
                {"error": "not_initialised"},
            )

        command = parameters.get("command", parameters.get("cmd", ""))
        if not isinstance(command, str) or not command.strip():
            return (
                ToolResponse(text="Error: 'command' parameter is required."),
                -0.05,
                {"error": "empty_command"},
            )

        state["call_count"] += 1
        state["commands"].append(command)

        container = state["container_name"]
        try:
            result = subprocess.run(
                ["docker", "exec", "-w", "/workspace", container, "bash", "-c", command],
                capture_output=True, text=True, timeout=self._timeout,
            )
            exit_code = result.returncode
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}" if output else result.stderr
            if not output:
                output = "(no output)"

            # Truncate
            if len(output) > MAX_OUTPUT:
                output = output[:MAX_OUTPUT] + "\n...[truncated]"

            state["exit_codes"].append(exit_code)

            # Reward: exit code 0 → small positive, non-zero → small negative
            reward = 0.05 if exit_code == 0 else -0.02

            return (
                ToolResponse(text=f"$ {command}\n{output}\n[exit code: {exit_code}]"),
                reward,
                {"exit_code": exit_code, "call_count": state["call_count"]},
            )
        except subprocess.TimeoutExpired:
            state["exit_codes"].append(-1)
            return (
                ToolResponse(text=f"$ {command}\nError: command timed out after {self._timeout}s"),
                -0.05,
                {"error": "timeout", "call_count": state["call_count"]},
            )
        except Exception as e:
            state["exit_codes"].append(-2)
            return (
                ToolResponse(text=f"$ {command}\nError: {type(e).__name__}: {e}"),
                -0.05,
                {"error": str(e), "call_count": state["call_count"]},
            )

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        state = self._instance_dict.get(instance_id, {})
        exit_codes = state.get("exit_codes", [])
        if not exit_codes:
            return 0.0
        # Fraction of successful commands
        success_rate = sum(1 for c in exit_codes if c == 0) / len(exit_codes)
        return success_rate * 0.1

    async def release(self, instance_id: str, **kwargs) -> None:
        state = self._instance_dict.pop(instance_id, None)
        if not state:
            return
        container = state.get("container_name")
        if container:
            # Stop and remove container (best-effort, don't block on errors)
            subprocess.run(
                ["docker", "rm", "-f", container],
                capture_output=True, timeout=30,
            )
