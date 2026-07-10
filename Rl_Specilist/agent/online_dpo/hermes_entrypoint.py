"""Hermes-format agent entrypoint — runs in a local workspace.

Drives a tool-use conversation loop against the Uni-Agent Gateway
(OpenAI-compatible ``/v1/chat/completions``), parsing Hermes-format tool
calls (``<tool_call>{"name": ..., "arguments": ...}</tool_call>``) and
executing them via subprocess inside the workspace.

No dependencies beyond Python stdlib — runs with the system Python.

Usage (from a runner):
    HERMES_TASK="do something" \\
    HERMES_BASE_URL="http://127.0.0.1:8765/sessions/abc/v1" \\
    HERMES_WORKSPACE="/tmp/verl_hermes/session-0-0" \\
    AGENT_MAX_TURNS=100 \\
    python hermes_entrypoint.py

Environment variables:
    HERMES_TASK        — the user task / prompt (required)
    HERMES_BASE_URL    — Gateway session base URL (required)
    HERMES_WORKSPACE   — workspace directory (default: /tmp/verl_hermes/default)
    AGENT_MAX_TURNS    — max conversation turns (default: 100)
    HERMES_MODEL       — model name sent to the Gateway (default: "default")

The Gateway handles tool-call parsing server-side (tool_parser: hermes), so
the Gateway decodes tool calls into OpenAI format before reaching this
entrypoint.  As a fallback, the entrypoint also parses Hermes-format tool
calls from the content string.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [hermes] %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("hermes_entrypoint")

# ── Hermes tool-call format ─────────────────────────────────────────────────

_HERMES_START = "<tool_call>"
_HERMES_END = "</tool_call>"
_HERMES_PATTERN = re.compile(
    rf"{re.escape(_HERMES_START)}\s*(.*?)\s*{re.escape(_HERMES_END)}",
    re.DOTALL,
)

FINISH_TOOLS = frozenset({"finish", "submit", "submit_answer", "stop", "exit"})


def parse_hermes_tool_calls(text: str, known_tools: list[dict]) -> tuple[str, list[dict]]:
    """Extract Hermes-format tool calls from model output.

    Returns:
        (assistant_content, list[OpenAI-style tool_calls dicts])
    """
    tool_name_set = {t["function"]["name"] for t in known_tools}
    tool_calls = []
    for idx, match in enumerate(_HERMES_PATTERN.finditer(text)):
        raw = match.group(1).strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse Hermes tool call JSON: %.200s", raw)
            continue
        name = parsed.get("name", "")
        if name not in tool_name_set:
            logger.warning(
                "Unknown tool '%s', skipping. Known: %s",
                name,
                sorted(tool_name_set),
            )
            continue
        arguments = parsed.get("arguments", {})
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments)
        tool_calls.append(
            {
                "id": f"call_{idx}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    # Return text before first tool call as assistant content
    content = text
    if tool_calls:
        first_start = text.find(_HERMES_START)
        if first_start >= 0:
            content = text[:first_start].strip()
    return content, tool_calls


# ── Tool execution ──────────────────────────────────────────────────────────


def execute_tool(name: str, arguments: str, workspace: str) -> str:
    """Execute a tool in the workspace and return the observation string."""
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.JSONDecodeError:
        args = {}

    if name in FINISH_TOOLS:
        return json.dumps({
            "status": "finished",
            "message": args.get("message", args.get("answer", "")),
        })

    if name == "execute_bash":
        command = args.get("command", "")
        if not command:
            return "Error: no command provided"
        timeout = int(args.get("timeout", 300))
        return _run_bash(command, cwd=workspace, timeout=timeout)

    if name == "read_file":
        file_path = args.get("file_path", "")
        if not file_path:
            return "Error: 'file_path' parameter is required"
        return _read_file(file_path, workspace)

    if name == "write_file":
        file_path = args.get("file_path", "")
        content = args.get("content", "")
        if not file_path:
            return "Error: 'file_path' parameter is required"
        return _write_file(file_path, content, workspace)

    logger.warning("Unknown tool: %s", name)
    return f"Error: unknown tool '{name}'"


def _validate_path(path: str, workspace: str) -> str:
    """Resolve and validate that *path* is within the workspace.

    Returns the resolved absolute path on success, or raises ``ValueError``
    if the path escapes the workspace.
    """
    cwd = os.path.realpath(workspace)
    resolved = os.path.realpath(os.path.join(cwd, path))
    if not resolved.startswith(cwd + os.sep) and resolved != cwd:
        raise ValueError(
            f"Path {path!r} resolves to {resolved!r}, which is outside {cwd!r}"
        )
    return resolved


def _run_bash(command: str, cwd: str, timeout: int = 300) -> str:
    """Execute a bash command and return stdout + stderr.

    Uses ``shell=True`` — security is provided by the workspace boundary.
    """
    logger.info("bash: %s", command[:200])
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"


def _read_file(file_path: str, workspace: str) -> str:
    """Read a file from the workspace."""
    try:
        safe_path = _validate_path(file_path, workspace)
        with open(safe_path, encoding="utf-8", errors="replace") as f:
            content = f.read()
        if not content:
            return "(empty file)"
        # Truncate long files
        if len(content) > 8000:
            content = content[:8000] + "\n...[truncated]"
        return content
    except ValueError as e:
        return f"Error: {e}"
    except FileNotFoundError:
        return f"Error: file not found: {file_path}"
    except Exception as e:
        return f"Error reading {file_path}: {e}"


def _write_file(file_path: str, content: str, workspace: str) -> str:
    """Write content to a file in the workspace."""
    try:
        safe_path = _validate_path(file_path, workspace)
        os.makedirs(os.path.dirname(safe_path) or ".", exist_ok=True)
        with open(safe_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} chars to '{file_path}'"
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error writing {file_path}: {e}"


# ── Gateway API client ──────────────────────────────────────────────────────


def chat_completions(
    base_url: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    max_tokens: int = 4096,
    temperature: float = 1.0,
    api_key: str = "",
) -> dict:
    """Call the Gateway's OpenAI-compatible /v1/chat/completions endpoint."""
    url = f"{base_url}/chat/completions"
    body: dict = {
        "model": os.environ.get("HERMES_MODEL", "default"),
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        body["tools"] = tools

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        logger.error("Gateway HTTP %s: %s", e.code, e.read()[:500])
        raise


# ── Main loop ───────────────────────────────────────────────────────────────


def build_system_prompt() -> str:
    """Build a minimal system prompt for command-line agent tasks."""
    return (
        "You are an AI agent completing tasks in a Linux command-line environment.\n\n"
        "You have access to the following tools:\n"
        "- execute_bash: Run bash commands in the workspace.\n"
        "- read_file: Read the contents of a file.\n"
        "- write_file: Write content to a file.\n"
        "- submit_answer: Submit your final answer when the task is complete.\n\n"
        "Use the Hermes tool-call format to invoke tools:\n"
        "<tool_call>\n"
        '{"name": "<tool_name>", "arguments": {<args_dict>}}\n'
        "</tool_call>\n\n"
        "After each tool call, you will receive the tool output. "
        "Continue until the task is complete, then call submit_answer."
    )


def run_agent(
    base_url: str,
    task: str,
    tools: list[dict],
    max_turns: int = 100,
    api_key: str = "",
) -> None:
    """Run the full Hermes agent loop against the Gateway."""
    messages: list[dict] = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": task},
    ]

    for turn in range(max_turns):
        logger.info("Turn %d/%d — calling Gateway", turn + 1, max_turns)
        response = chat_completions(base_url, messages, tools, api_key=api_key)
        choice = response.get("choices", [{}])[0]
        finish_reason = choice.get("finish_reason", "stop")
        msg = choice.get("message", {})

        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls") or []

        # If Gateway decoded tool calls directly (OpenAI format)
        if not tool_calls and _HERMES_START in content:
            content, tool_calls = parse_hermes_tool_calls(content, tools)

        if tool_calls:
            assistant_msg: dict = {
                "role": "assistant",
                "content": content or None,
                "tool_calls": tool_calls,
            }
            messages.append(assistant_msg)

            for tc in tool_calls:
                name = tc["function"]["name"]
                arguments = tc["function"]["arguments"]
                observation = execute_tool(
                    name, arguments, workspace=_get_workspace()
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": observation,
                    }
                )
                if name in FINISH_TOOLS:
                    logger.info(
                        "Agent finished via %s at turn %d", name, turn + 1
                    )
                    return
        else:
            # No tool calls — agent is done
            messages.append({"role": "assistant", "content": content})
            logger.info(
                "Agent stopped (finish_reason=%s) at turn %d",
                finish_reason,
                turn + 1,
            )
            return

    logger.warning("Agent reached max_turns=%d without finishing", max_turns)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _get_workspace() -> str:
    """Get workspace directory from env or default."""
    ws = os.environ.get("HERMES_WORKSPACE", "/tmp/verl_hermes/default")
    os.makedirs(ws, exist_ok=True)
    return ws


# ── Entry point ─────────────────────────────────────────────────────────────


def main() -> None:
    task = os.environ.get("HERMES_TASK", "")
    if not task:
        logger.error("HERMES_TASK environment variable is required")
        sys.exit(1)

    base_url = os.environ.get("HERMES_BASE_URL", "")
    if not base_url:
        logger.error("HERMES_BASE_URL environment variable is required")
        sys.exit(1)

    max_turns = int(os.environ.get("AGENT_MAX_TURNS", "100"))
    model = os.environ.get("HERMES_MODEL", "default")

    # Unset proxy vars so sandbox-internal tunnel is not bypassed
    for var in (
        "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
        "NO_PROXY", "no_proxy",
    ):
        os.environ.pop(var, None)

    logger.info(
        "Hermes agent starting: model=%s, max_turns=%d, base_url=%s",
        model, max_turns, base_url,
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "execute_bash",
                "description": (
                    "Execute a bash command in the workspace. "
                    "Returns stdout+stderr. Use for running scripts, "
                    "installing packages, file operations, etc."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The bash command to run.",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds (default: 300).",
                        },
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": (
                    "Read the contents of a file in the workspace. "
                    "Returns the file text (truncated at 8000 chars)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file, relative to the workspace.",
                        },
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": (
                    "Write content to a file in the workspace. "
                    "Creates parent directories if needed. Overwrites existing files."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file, relative to the workspace.",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file.",
                        },
                    },
                    "required": ["file_path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit_answer",
                "description": (
                    "Submit your final answer after completing the task. "
                    "Call this when you have finished all required work."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "Your final answer or summary of work done.",
                        },
                    },
                    "required": ["answer"],
                },
            },
        },
    ]

    started_at = time.time()
    try:
        run_agent(base_url, task, tools, max_turns=max_turns)
    except Exception as exc:
        logger.error("Agent loop failed: %s", exc, exc_info=True)
        sys.exit(1)
    elapsed = time.time() - started_at
    logger.info("Hermes agent finished in %.1fs", elapsed)


if __name__ == "__main__":
    main()
