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
"""SWE-bench interaction for Open-SWE-Traces / SWE-Zero datasets.

Implements ``BaseInteraction`` to provide a real code-repo environment:
  - ``start_interaction``: clone the repo at base_commit in a Docker container
  - ``generate_response``: execute agent's bash/editor commands, return observation
  - ``calculate_score``: run the test suite, return pass rate
  - ``finalize_interaction``: clean up the container

This is the heaviest environment — each sample needs its own container with
the correct repo + commit checked out.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Any, Optional
from uuid import uuid4

from verl.interactions.base import BaseInteraction

DEFAULT_IMAGE = "reject-sft-swe:latest"
DEFAULT_TIMEOUT = 120  # seconds for test execution
MAX_OUTPUT = 6000


class SWEBenchInteraction(BaseInteraction):
    """A SWE-bench style interaction environment.

    Config options (in ``config`` dict):
      - image: Docker image to use (default: reject-sft-swe:latest)
      - timeout: Test execution timeout (default: 120s)
      - workdir: Working directory inside container (default: /workspace)
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._image = config.get("image", DEFAULT_IMAGE)
        self._timeout = config.get("timeout", DEFAULT_TIMEOUT)
        self._workdir = config.get("workdir", "/workspace")
        self._instances: dict[str, dict] = {}

    async def start_interaction(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """Start a Docker container and clone the repo at base_commit.

        Expected kwargs:
          - repo: GitHub repo URL or "owner/repo" shorthand
          - base_commit: Git commit SHA to checkout
          - test_cmd: Command to run tests (optional, defaults to pytest)
        """
        if instance_id is None:
            instance_id = str(uuid4())

        repo = kwargs.get("repo", "")
        base_commit = kwargs.get("base_commit", "")
        test_cmd = kwargs.get("test_cmd", "python -m pytest -x -q")

        if not repo:
            # No repo specified — start an empty container (judge-only mode)
            container_name = f"swe-{instance_id[:12]}"
            try:
                subprocess.run(
                    ["docker", "run", "-d", "--name", container_name,
                     "-w", self._workdir, self._image, "sleep", "7200"],
                    capture_output=True, text=True, timeout=120, check=True,
                )
            except Exception:
                pass
            self._instances[instance_id] = {
                "container_name": container_name,
                "repo": repo,
                "base_commit": base_commit,
                "test_cmd": test_cmd,
                "turn_count": 0,
                "commands_run": [],
                "test_result": None,
            }
            return instance_id

        container_name = f"swe-{instance_id[:12]}"

        # Normalise repo URL
        if not repo.startswith("http"):
            repo_url = f"https://github.com/{repo}.git"
        else:
            repo_url = repo if repo.endswith(".git") else repo + ".git"

        repo_name = repo_url.rstrip("/").replace(".git", "").split("/")[-1]

        try:
            # Start container
            subprocess.run(
                ["docker", "run", "-d", "--name", container_name,
                 "-w", self._workdir, self._image, "sleep", "7200"],
                capture_output=True, text=True, timeout=120, check=True,
            )

            # Clone repo (shallow if no specific commit, full clone if commit needed)
            clone_cmd = f"git clone {repo_url} {self._workdir}/{repo_name}"
            self._exec(container_name, clone_cmd, timeout=300)

            # Checkout base commit if specified
            if base_commit:
                self._exec(
                    container_name,
                    f"cd {self._workdir}/{repo_name} && git checkout {base_commit}",
                    timeout=60,
                )

            # Install repo dependencies (best-effort)
            self._exec(
                container_name,
                f"cd {self._workdir}/{repo_name} && "
                "(pip install -e . -q 2>/dev/null || pip install -r requirements.txt -q 2>/dev/null || true)",
                timeout=180,
            )

        except Exception as e:
            # Container creation failed — record but don't crash
            print(f"[SWEBenchInteraction] Failed to start container for {instance_id}: {e}")

        self._instances[instance_id] = {
            "container_name": container_name,
            "repo": repo,
            "repo_name": repo_name,
            "base_commit": base_commit,
            "test_cmd": test_cmd,
            "turn_count": 0,
            "commands_run": [],
            "test_result": None,
        }
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict[str, Any]]:
        """Execute the assistant's command and return the observation.

        The assistant message content is expected to contain bash commands
        (possibly wrapped in tool_call tags). We extract and execute them.
        """
        state = self._instances.get(instance_id)
        if not state:
            return True, "Error: interaction not started.", 0.0, {}

        state["turn_count"] += 1
        container = state["container_name"]
        repo_name = state.get("repo_name", "")
        workdir = f"{self._workdir}/{repo_name}" if repo_name else self._workdir

        # Extract command from the last assistant message
        last_msg = messages[-1] if messages else {}
        content = last_msg.get("content", "")

        # Try to extract bash command from tool_call or code block
        command = self._extract_command(content)

        if not command:
            return False, "No command found in assistant message.", 0.0, {"turn": state["turn_count"]}

        state["commands_run"].append(command)

        # Execute in container
        try:
            result = subprocess.run(
                ["docker", "exec", "-w", workdir, container, "bash", "-c", command],
                capture_output=True, text=True, timeout=self._timeout,
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}" if output else result.stderr
            if not output:
                output = "(no output)"
            if len(output) > MAX_OUTPUT:
                output = output[:MAX_OUTPUT] + "\n...[truncated]"

            exit_code = result.returncode
            reward = 0.05 if exit_code == 0 else -0.02

            # Check if agent indicated task completion
            should_terminate = self._is_task_complete(content)

            return (
                should_terminate,
                f"$ {command}\n{output}\n[exit code: {exit_code}]",
                reward,
                {"exit_code": exit_code, "turn": state["turn_count"]},
            )
        except subprocess.TimeoutExpired:
            return False, f"$ {command}\nError: timed out after {self._timeout}s", -0.05, {"timeout": True}
        except Exception as e:
            return False, f"$ {command}\nError: {e}", -0.05, {"error": str(e)}

    async def calculate_score(self) -> float:
        """Run the test suite and return pass rate (0.0 - 1.0).

        This is called at the end of the interaction to verify correctness.
        """
        # Note: this method doesn't have instance_id in the signature,
        # so we use the last active instance. This matches BaseInteraction's design.
        if not self._instances:
            return 0.0

        # Use the most recent instance
        instance_id = list(self._instances.keys())[-1]
        state = self._instances[instance_id]
        container = state.get("container_name")
        test_cmd = state.get("test_cmd", "python -m pytest -x -q")
        repo_name = state.get("repo_name", "")
        workdir = f"{self._workdir}/{repo_name}" if repo_name else self._workdir

        if not container:
            return 0.0

        try:
            result = subprocess.run(
                ["docker", "exec", "-w", workdir, container, "bash", "-c", test_cmd],
                capture_output=True, text=True, timeout=self._timeout * 2,
            )
            output = result.stdout + result.stderr

            # Parse pytest output for pass/fail counts
            passed, failed = self._parse_pytest_output(output)
            total = passed + failed
            pass_rate = passed / total if total > 0 else 0.0

            state["test_result"] = {
                "passed": passed,
                "failed": failed,
                "pass_rate": pass_rate,
                "exit_code": result.returncode,
            }
            return pass_rate
        except Exception as e:
            state["test_result"] = {"error": str(e)}
            return 0.0

    async def finalize_interaction(self) -> None:
        """Clean up all Docker containers."""
        for state in self._instances.values():
            container = state.get("container_name")
            if container:
                subprocess.run(
                    ["docker", "rm", "-f", container],
                    capture_output=True, timeout=30,
                )
        self._instances.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _exec(self, container: str, command: str, timeout: int = 60) -> tuple[int, str]:
        """Run a command in the container, return (exit_code, output)."""
        result = subprocess.run(
            ["docker", "exec", "-w", self._workdir, container, "bash", "-c", command],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode, result.stdout + result.stderr

    def _extract_command(self, content: str) -> str:
        """Extract a bash command from assistant message content.

        Handles:
        - <tool_call>...</tool_call> with bash function
        - ```bash ... ``` code blocks
        - Raw command text
        """
        if not content:
            return ""

        # Try <tool_call> format
        tool_call_match = re.search(
            r'<tool_call>\s*.*?"name"\s*:\s*"?(?:bash|terminal|execute)?"?\s*,?\s*'
            r'"arguments"\s*:\s*\{[^}]*"command"\s*:\s*"([^"]+)"',
            content, re.DOTALL | re.IGNORECASE,
        )
        if tool_call_match:
            return tool_call_match.group(1).replace("\\n", "\n").replace('\\"', '"')

        # Try ```bash code block
        bash_match = re.search(r"```(?:bash|sh|shell)?\s*\n(.*?)```", content, re.DOTALL)
        if bash_match:
            return bash_match.group(1).strip()

        # Try JSON tool_call
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "arguments" in data:
                args = data["arguments"]
                if isinstance(args, dict) and "command" in args:
                    return args["command"]
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback: treat entire content as command if it looks like one
        lines = content.strip().split("\n")
        # Skip <think> blocks
        cmd_lines = [l for l in lines if not l.strip().startswith("<")]
        if cmd_lines and len(cmd_lines) <= 5 and all(len(l) < 500 for l in cmd_lines):
            return "\n".join(cmd_lines).strip()

        return ""

    def _is_task_complete(self, content: str) -> bool:
        """Check if the agent indicated task completion."""
        content_lower = content.lower()
        completion_markers = [
            "task complete", "task completed", "i'm done", "i am done",
            "finished", "all tests pass", "submission complete",
            "<submit_answer>", "<finish>",
        ]
        return any(marker in content_lower for marker in completion_markers)

    def _parse_pytest_output(self, output: str) -> tuple[int, int]:
        """Parse pytest output for passed/failed counts."""
        passed = 0
        failed = 0

        # Match "X passed, Y failed" pattern
        match = re.search(r"(\d+)\s+passed", output)
        if match:
            passed = int(match.group(1))

        match = re.search(r"(\d+)\s+failed", output)
        if match:
            failed = int(match.group(1))

        # Also check for "===== X passed in Ys =====" format
        if passed == 0 and failed == 0:
            match = re.search(r"=\s*(\d+)\s+passed", output)
            if match:
                passed = int(match.group(1))

        return passed, failed
