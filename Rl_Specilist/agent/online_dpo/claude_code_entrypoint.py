"""Claude-Code-style agent entrypoint — runs in a local workspace.

Drives a software-engineering tool-use conversation loop against the
Uni-Agent Gateway (OpenAI-compatible ``/v1/chat/completions``).  The
*model* is the one being trained (Qwen3-4B via Gateway → vLLM); the
*agent environment* provides a Claude-Code-inspired tool set and system
prompt for code-focused tasks.

No dependencies beyond Python stdlib — runs with the system Python.

Usage (from a runner)::

    CLAUDE_TASK="fix the login bug in this repo" \\
    CLAUDE_BASE_URL="http://127.0.0.1:8765/sessions/abc/v1" \\
    CLAUDE_WORKSPACE="/tmp/verl_claude/session-0-0" \\
    AGENT_MAX_TURNS=100 \\
    python claude_code_entrypoint.py

Environment variables:
    CLAUDE_TASK         — the user task / prompt (required)
    CLAUDE_BASE_URL     — Gateway session base URL (required)
    CLAUDE_WORKSPACE    — workspace directory (default: /tmp/verl_claude/default)
    AGENT_MAX_TURNS     — max conversation turns (default: 100)
    CLAUDE_MODEL        — model name sent to Gateway (default: "default")
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
import urllib.request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [claude-code] %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("claude_code_entrypoint")

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
                "Unknown tool '%s', skipping. Known: %s", name, sorted(tool_name_set),
            )
            continue
        arguments = parsed.get("arguments", {})
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments)
        tool_calls.append({
            "id": f"call_{idx}",
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        })
    content = text
    if tool_calls:
        first_start = text.find(_HERMES_START)
        if first_start >= 0:
            content = text[:first_start].strip()
    return content, tool_calls


# ── Tool execution ──────────────────────────────────────────────────────────


_CWD: str | None = None


def _get_workspace() -> str:
    """Get workspace directory from env or default."""
    global _CWD
    if _CWD is None:
        _CWD = os.environ.get("CLAUDE_WORKSPACE", "/tmp/verl_claude/default")
        os.makedirs(_CWD, exist_ok=True)
    return _CWD


def _validate_path(path: str, workspace: str) -> str:
    """Resolve and validate that *path* is within the workspace."""
    cwd = os.path.realpath(workspace)
    resolved = os.path.realpath(os.path.join(cwd, path))
    if not resolved.startswith(cwd + os.sep) and resolved != cwd:
        raise ValueError(
            f"Path {path!r} resolves to {resolved!r}, which is outside {cwd!r}"
        )
    return resolved


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

    if name == "edit_file":
        file_path = args.get("file_path", "")
        old_string = args.get("old_string", "")
        new_string = args.get("new_string", "")
        if not file_path:
            return "Error: 'file_path' parameter is required"
        return _edit_file(file_path, old_string, new_string, workspace)

    if name == "search_code":
        pattern = args.get("pattern", "")
        directory = args.get("directory", ".")
        if not pattern:
            return "Error: 'pattern' parameter is required"
        return _search_code(pattern, directory, workspace)

    if name == "list_files":
        directory = args.get("directory", ".")
        depth = int(args.get("depth", 2))
        pattern = args.get("pattern", "*")
        return _list_files(directory, depth, pattern, workspace)

    if name == "run_tests":
        command = args.get("command", "")
        if not command:
            return "Error: 'command' parameter is required (e.g. 'pytest')"
        timeout = int(args.get("timeout", 600))
        return _run_tests(command, cwd=workspace, timeout=timeout)

    logger.warning("Unknown tool: %s", name)
    return f"Error: unknown tool '{name}'"


# ── Tool implementations ────────────────────────────────────────────────────


def _run_bash(command: str, cwd: str, timeout: int = 300) -> str:
    """Execute a bash command and return stdout + stderr."""
    logger.info("bash: %s", command[:200])
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=cwd,
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
    """Read a file from the workspace with line numbers (like `cat -n`)."""
    try:
        safe_path = _validate_path(file_path, workspace)
        with open(safe_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if not lines:
            return "(empty file)"
        if len(lines) > 500:
            head = "".join(lines[:200])
            tail = "".join(lines[-200:])
            return (
                f"File: {file_path} ({len(lines)} lines total)\n\n"
                f"[Lines 1-200]\n{head}\n"
                f"...[{len(lines) - 400} lines omitted]...\n\n"
                f"[Lines {len(lines) - 199}-{len(lines)}]\n{tail}"
            )
        numbered = "".join(
            f"{i+1:>6}\t{line}" for i, line in enumerate(lines)
        )
        return f"File: {file_path} ({len(lines)} lines)\n\n{numbered}"
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
        lines = content.count("\n") + 1
        return f"Successfully wrote {len(content)} chars ({lines} lines) to '{file_path}'"
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error writing {file_path}: {e}"


def _edit_file(
    file_path: str, old_string: str, new_string: str, workspace: str
) -> str:
    """Edit a file by exact-string replacement (single occurrence).

    Returns an error if old_string is not found or appears more than once.
    """
    try:
        safe_path = _validate_path(file_path, workspace)
        with open(safe_path, encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        return f"Error: file not found: {file_path}"
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error reading {file_path}: {e}"

    # Support for create-if-empty mode
    if not content.strip() and not old_string:
        # Creating new content in empty file
        try:
            with open(safe_path, "w", encoding="utf-8") as f:
                f.write(new_string)
            return f"Created '{file_path}' with {len(new_string)} chars"
        except Exception as e:
            return f"Error writing {file_path}: {e}"

    count = content.count(old_string)
    if count == 0:
        return (
            f"Error: old_string not found in '{file_path}'. "
            f"The file has {len(content)} chars. "
            f"Tip: use read_file first to see exact content."
        )
    if count > 1:
        return (
            f"Error: old_string appears {count} times in '{file_path}'. "
            f"Make it more specific to uniquely identify the target location."
        )

    new_content = content.replace(old_string, new_string, 1)
    try:
        with open(safe_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return (
            f"Successfully edited '{file_path}': "
            f"replaced 1 occurrence ({len(old_string)}→{len(new_string)} chars)"
        )
    except Exception as e:
        return f"Error writing {file_path}: {e}"


def _search_code(pattern: str, directory: str, workspace: str) -> str:
    """Search code in workspace using grep."""
    try:
        safe_dir = _validate_path(directory, workspace)
    except ValueError as e:
        return f"Error: {e}"

    try:
        result = subprocess.run(
            ["grep", "-rn", "--include=*.py", "--include=*.sh", "--include=*.js",
             "--include=*.ts", "--include=*.java", "--include=*.cpp", "--include=*.c",
             "--include=*.h", "--include=*.rs", "--include=*.go", "--include=*.md",
             "--include=*.txt", "--include=*.yaml", "--include=*.yml", "--include=*.json",
             "--include=*.toml", "--include=*.cfg", "--include=*.ini",
             "-I", pattern, safe_dir],
            capture_output=True, text=True, timeout=30,
        )
        output = result.stdout.strip()
        if not output:
            return f"No matches found for pattern '{pattern}' in {directory}"
        lines = output.split("\n")
        if len(lines) > 50:
            output = "\n".join(lines[:50])
            output += f"\n...({len(lines) - 50} more matches truncated)"
        return output
    except subprocess.TimeoutExpired:
        return "Error: search timed out"
    except FileNotFoundError:
        # Fallback: pure Python grep
        return _search_code_python(pattern, safe_dir)


def _search_code_python(pattern: str, directory: str) -> str:
    """Fallback search using pure Python (no grep available)."""
    import fnmatch
    results = []
    code_exts = {".py", ".sh", ".js", ".ts", ".java", ".cpp", ".c", ".h",
                 ".rs", ".go", ".md", ".txt", ".yaml", ".yml", ".json",
                 ".toml", ".cfg", ".ini"}
    try:
        compiled = re.compile(pattern.encode())
    except re.error:
        compiled = re.compile(re.escape(pattern).encode())

    for root, dirs, files in os.walk(directory):
        # Skip hidden and .git
        dirs[:] = [d for d in dirs if not d.startswith(".") or d == ".git"]
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in code_exts:
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "rb") as f:
                    for lineno, line in enumerate(f, 1):
                        if compiled.search(line):
                            rel = os.path.relpath(fpath, directory)
                            results.append(
                                f"{rel}:{lineno}: {line.decode('utf-8', errors='replace').rstrip()}"
                            )
                            if len(results) >= 50:
                                break
            except Exception:
                continue
            if len(results) >= 50:
                break
        if len(results) >= 50:
            break

    if not results:
        return f"No matches found for pattern '{pattern}' in {directory}"
    output = "\n".join(results[:50])
    if len(results) >= 50:
        output += f"\n...(results truncated at 50)"
    return output


def _list_files(
    directory: str, depth: int, pattern: str, workspace: str
) -> str:
    """List files in workspace using `find` or pure Python."""
    try:
        safe_dir = _validate_path(directory, workspace)
    except ValueError as e:
        return f"Error: {e}"

    depth = min(max(depth, 1), 5)
    try:
        result = subprocess.run(
            ["find", safe_dir, "-maxdepth", str(depth),
             "-name", pattern, "-not", "-path", "*/.git/*",
             "-not", "-path", "*/node_modules/*",
             "-not", "-path", "*/__pycache__/*",
             "-not", "-name", "*.pyc"],
            capture_output=True, text=True, timeout=15,
        )
        output = result.stdout.strip()
        if not output:
            return f"No files matching '{pattern}' found in {directory}"
        lines = output.split("\n")
        if len(lines) > 100:
            output = "\n".join(lines[:100])
            output += f"\n...({len(lines) - 100} more files truncated)"
        return output
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return _list_files_python(safe_dir, depth, pattern)


def _list_files_python(directory: str, depth: int, pattern: str) -> str:
    """Fallback file listing using pure Python."""
    import fnmatch
    results = []
    for root, dirs, files in os.walk(directory):
        current_depth = root[len(directory):].count(os.sep)
        if current_depth >= depth:
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fname in fnmatch.filter(files, pattern):
            results.append(os.path.relpath(os.path.join(root, fname), directory))
            if len(results) >= 100:
                break
        if len(results) >= 100:
            break
    if not results:
        return f"No files matching '{pattern}' found in {directory}"
    output = "\n".join(results[:100])
    if len(results) >= 100:
        output += f"\n...(results truncated at 100)"
    return output


def _run_tests(command: str, cwd: str, timeout: int = 600) -> str:
    """Run a test command and return formatted results."""
    logger.info("Running tests: %s", command[:200])
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=cwd,
        )
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        # Summarize test results
        rc = result.returncode
        status = "PASSED" if rc == 0 else f"FAILED (exit code {rc})"
        summary = f"\n[Tests {status}]"
        return (output.strip() or "(no output)") + summary
    except subprocess.TimeoutExpired:
        return f"Error: tests timed out after {timeout}s"


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
        "model": os.environ.get("CLAUDE_MODEL", "default"),
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


# ── System prompt ───────────────────────────────────────────────────────────


def build_system_prompt() -> str:
    """Build a software-engineering system prompt (Claude Code style).

    Key differences from the vanilla hermes prompt:
    - Emphasis on understanding code before editing
    - Preference for targeted edits over full rewrites
    - Test-driven workflow
    - Repository exploration best practices
    """
    return """You are a world-class software engineer working in a Linux environment.

## Your Capabilities
You have access to the following tools:
- **execute_bash**: Run shell commands (git, python, pytest, npm, etc.)
- **read_file**: Read file contents with line numbers
- **write_file**: Create or overwrite a file
- **edit_file**: Make targeted edits via exact-string replacement (PREFERRED for changes)
- **search_code**: Search for patterns across the codebase (grep)
- **list_files**: List files in a directory tree
- **submit_answer**: Submit your final answer when the task is complete

## Workflow Best Practices

### 1. Understand First, Edit Later
- Read relevant files before making changes
- Use search_code to find where things are defined/used
- Use list_files to understand project structure

### 2. Make Targeted Edits
- PREFER edit_file over write_file — only change what needs changing
- Make each edit self-contained and verifiable
- If a file needs many changes spread across it, use write_file instead

### 3. Test After Changes
- Run tests (pytest, npm test, etc.) after each significant change
- If tests fail, read the failure output carefully and fix the root cause
- Don't skip tests — they are your safety net

### 4. Iterate
- If an edit doesn't work, read the file again and try a different approach
- If you're stuck, step back and think about the problem differently
- Use bash to explore, experiment, and validate assumptions

## Tool Call Format
Use the Hermes tool-call format exactly as shown:
<tool_call>
{"name": "tool_name", "arguments": {"arg1": "value1", ...}}
</tool_call>

IMPORTANT:
- Put the tool call on its own line, wrapped in <tool_call> tags
- Use valid JSON inside the tags
- After each tool call, you will receive the tool output
- Continue until the task is complete, then call submit_answer
- Be thorough — verify your work, don't just assume it's correct"""


# ── Main loop ───────────────────────────────────────────────────────────────


def run_agent(
    base_url: str,
    task: str,
    tools: list[dict],
    max_turns: int = 100,
    api_key: str = "",
) -> None:
    """Run the full software-engineering agent loop against the Gateway.

    The Gateway routes requests to vLLM (Qwen3-4B), which is the model
    being trained.  The Gateway captures full token-level trajectories.
    """
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
                    name, arguments, workspace=_get_workspace(),
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": observation,
                })
                if name in FINISH_TOOLS:
                    logger.info(
                        "Agent finished via %s at turn %d", name, turn + 1,
                    )
                    return
        else:
            # No tool calls — agent is done (text-only response)
            messages.append({"role": "assistant", "content": content})
            logger.info(
                "Agent stopped (finish_reason=%s) at turn %d",
                finish_reason, turn + 1,
            )
            return

    logger.warning("Agent reached max_turns=%d without finishing", max_turns)


# ── Entry point ─────────────────────────────────────────────────────────────


def main() -> None:
    task = os.environ.get("CLAUDE_TASK", "")
    if not task:
        logger.error("CLAUDE_TASK environment variable is required")
        sys.exit(1)

    base_url = os.environ.get("CLAUDE_BASE_URL", "")
    if not base_url:
        logger.error("CLAUDE_BASE_URL environment variable is required")
        sys.exit(1)

    max_turns = int(os.environ.get("AGENT_MAX_TURNS", "100"))
    model = os.environ.get("CLAUDE_MODEL", "default")

    # Unset proxy vars so sandbox-internal tunnel is not bypassed
    for var in (
        "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
        "NO_PROXY", "no_proxy",
    ):
        os.environ.pop(var, None)

    logger.info(
        "Claude-Code-style agent starting: model=%s, max_turns=%d, base_url=%s",
        model, max_turns, base_url,
    )

    # ── Tool definitions ─────────────────────────────────────────────────
    # These are sent to the Gateway so the model knows what tools are available.
    # The Gateway enriches them with multi-turn format handling.
    tools = [
        {
            "type": "function",
            "function": {
                "name": "execute_bash",
                "description": (
                    "Execute a bash command in the workspace. "
                    "Returns stdout+stderr. Use for git operations, running "
                    "scripts, installing packages, compiling code, etc."
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
                    "Read a file from the workspace with line numbers. "
                    "Long files are truncated with head and tail shown. "
                    "ALWAYS read a file before editing it."
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
                    "Write content to a file, overwriting if it exists. "
                    "Creates parent directories if needed. "
                    "Use for creating new files or rewriting entire files. "
                    "PREFER edit_file for targeted changes to existing files."
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
                            "description": "Full content to write to the file.",
                        },
                    },
                    "required": ["file_path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": (
                    "Make a targeted edit to a file by replacing one occurrence "
                    "of old_string with new_string. The old_string must match "
                    "EXACTLY (including whitespace) and appear exactly once. "
                    "Use read_file first to see the exact content. "
                    "PREFERRED over write_file for small, targeted changes."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file, relative to the workspace.",
                        },
                        "old_string": {
                            "type": "string",
                            "description": "The exact text to replace (must appear exactly once).",
                        },
                        "new_string": {
                            "type": "string",
                            "description": "The replacement text.",
                        },
                    },
                    "required": ["file_path", "old_string", "new_string"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_code",
                "description": (
                    "Search for a regex pattern across code files in a directory. "
                    "Returns matching lines with file paths and line numbers. "
                    "Use to find definitions, usages, or patterns."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Regex pattern to search for.",
                        },
                        "directory": {
                            "type": "string",
                            "description": "Directory to search in (default: '.').",
                        },
                    },
                    "required": ["pattern"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": (
                    "List files in a directory tree, optionally filtering by pattern. "
                    "Use to understand project structure before making changes."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "description": "Directory to list (default: '.').",
                        },
                        "depth": {
                            "type": "integer",
                            "description": "Max directory depth (1-5, default: 2).",
                        },
                        "pattern": {
                            "type": "string",
                            "description": "File pattern, e.g. '*.py' (default: '*').",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_tests",
                "description": (
                    "Run a test command (e.g. 'pytest', 'npm test') and return "
                    "results with pass/fail status. Use after making code changes."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Test command, e.g. 'pytest test_auth.py -v'.",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds (default: 600).",
                        },
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit_answer",
                "description": (
                    "Submit your final answer. Call this when you have completed "
                    "the task and verified your work. Include a summary of what "
                    "you did and why."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "Summary of what was done, key changes, and results.",
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
    logger.info("Claude-Code-style agent finished in %.1fs", elapsed)


if __name__ == "__main__":
    main()
