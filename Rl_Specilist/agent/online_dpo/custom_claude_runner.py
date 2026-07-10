"""Custom Claude-Code-style runner for software-engineering agent tasks.

Runner contract for the Uni-Agent ``AgentFramework``.  One runner invocation
handles a single sample: it builds a task, creates an isolated workspace
(optionally cloning a repo for SWE tasks), launches a Claude-Code-style
entrypoint against the Gateway, evaluates the reward, and posts
``reward_info`` to the session.

Architecture::

    Runner (this file)
      ├─ Build task from raw_prompt + tools_kwargs
      ├─ Create workspace /tmp/verl_claude/<session_id>
      ├─ Clone repo if tools_kwargs specifies one (SWE tasks)
      ├─ Launch claude_code_entrypoint.py via subprocess
      │     └─ Agent → Gateway /v1/chat/completions → vLLM (Qwen3-4B)
      │     └─ Agent ← Gateway ← vLLM assistant reply
      │     └─ Agent → execute tools in workspace → observation
      │     └─ ... loop until submit_answer or max_turns ...
      ├─ Evaluate reward (judge_single from reward/llm_judge.py)
      └─ POST reward_info → Gateway session

Key differences from the Hermes runner:

* Richer tool set: edit_file, search_code, list_files, run_tests are
  available in addition to execute_bash, read_file, write_file.
* Software-engineering system prompt emphasising read-before-edit,
  targeted edits, and test-driven workflow.
* SWE-specific workspace setup: repos are cloned into the workspace
  before the agent starts.
* The inference model is still the one being trained (Qwen3-4B via
  Gateway → vLLM) — this runner only changes the *agent environment*.

Environment variables:
    CLAUDE_WORKSPACE_ROOT — workspace base dir (default: /tmp/verl_claude)
    CLAUDE_TIMEOUT        — max seconds per agent run (default: 3600)
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

# Gateway session handle — imported at runtime from uni-agent.
try:
    from uni_agent.gateway.session import SessionHandle
except ImportError:  # pragma: no cover
    from dataclasses import dataclass

    @dataclass
    class SessionHandle:  # type: ignore[no-redef]
        session_id: str
        base_url: str | None = None
        reward_info_url: str | None = None


# ── Task builder ────────────────────────────────────────────────────────────


def extract_task(raw_prompt: Any) -> str:
    """Extract the user task string from a dataset prompt.

    Handles both bare strings and OpenAI-format message lists.
    """
    if isinstance(raw_prompt, str):
        return raw_prompt
    for msg in raw_prompt:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        return block.get("text", "")
    return str(raw_prompt)


def build_claude_task(
    raw_prompt: Any,
    tools_kwargs: dict[str, Any] | None = None,
) -> str:
    """Build the task string for the Claude-Code-style agent.

    Enriches SWE-style tasks with repo context when available.
    """
    tools_kwargs = tools_kwargs or {}
    task = extract_task(raw_prompt)

    # Pre-formatted task from tools_kwargs takes priority
    prebuilt = tools_kwargs.get("task")
    if prebuilt and isinstance(prebuilt, str):
        return prebuilt

    # Enrich with repo context for SWE tasks
    repo = tools_kwargs.get("repo") or tools_kwargs.get("repo_url", "")
    base_commit = tools_kwargs.get("base_commit") or tools_kwargs.get("commit", "")
    instance_id = tools_kwargs.get("instance_id", "")

    if repo:
        context_parts = [task.strip(), "", "## Repository Context"]
        context_parts.append(f"- Repository: {repo}")
        if base_commit:
            context_parts.append(f"- Base commit: {base_commit}")
        if instance_id:
            context_parts.append(f"- Instance ID: {instance_id}")
        context_parts.append(
            "The repository has been cloned into `./repo/` in the current "
            "workspace.  Use list_files to explore the code, search_code to "
            "find relevant code, read_file to understand it, edit_file to "
            "make changes, and run_tests or execute_bash to verify."
        )
        return "\n".join(context_parts)

    return task.strip()


# ── Workspace management ────────────────────────────────────────────────────


def setup_workspace(
    session_id: str,
    tools_kwargs: dict[str, Any] | None = None,
) -> str:
    """Create an isolated workspace directory.

    If ``tools_kwargs`` specifies a repo URL, it is cloned into
    ``<workspace>/repo/``.  The URL is validated against a safelist of
    known code-hosting domains to prevent SSRF.
    """
    tools_kwargs = tools_kwargs or {}
    workspace_root = os.environ.get(
        "CLAUDE_WORKSPACE_ROOT", "/tmp/verl_claude"
    )
    workspace = os.path.join(workspace_root, session_id)
    os.makedirs(workspace, exist_ok=True)

    repo_url = tools_kwargs.get("repo") or tools_kwargs.get("repo_url")
    if repo_url:
        _clone_repo(
            repo_url=str(repo_url),
            workspace=workspace,
            base_commit=(
                tools_kwargs.get("base_commit")
                or tools_kwargs.get("commit")
            ),
        )

    return workspace


# Known code-hosting domains — only these are allowed for repo cloning.
_ALLOWED_REPO_DOMAINS = frozenset({
    "github.com", "gitlab.com", "bitbucket.org",
    "huggingface.co", "hf.co",
})


def _validate_repo_url(url: str) -> bool:
    """Check that a repo URL points to a known, trusted code host.

    Uses ``urllib.parse.urlparse`` for proper URL parsing (avoids regex
    bypasses with userinfo, fragments, etc.).
    """
    if not url:
        return False

    # Shorthand "owner/repo" — accept only if it looks like a sane path
    if not url.startswith("https://") and not url.startswith("http://") and not url.startswith("git@"):
        return bool(re.match(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$", url))

    # git@github.com:owner/repo.git form
    if url.startswith("git@"):
        m = re.match(r"git@([^:]+)", url)
        if not m:
            return False
        hostname = m.group(1).lower().rstrip(".")
        return hostname.split(":")[0] in _ALLOWED_REPO_DOMAINS

    # https:// or http:// — use urlparse for robust hostname extraction
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower().rstrip(".")
    return hostname in _ALLOWED_REPO_DOMAINS


def _clone_repo(
    repo_url: str,
    workspace: str,
    base_commit: str | None = None,
) -> None:
    """Clone a git repository into ``<workspace>/repo/``.

    The URL is validated before cloning.  Errors are logged but not raised
    — a missing or invalid repo shouldn't block the agent from attempting
    the task.
    """
    if not _validate_repo_url(repo_url):
        logger.warning(
            "Repo URL %r does not match allowed domains %s; skipping clone.",
            repo_url, sorted(_ALLOWED_REPO_DOMAINS),
        )
        return

    clone_dir = os.path.join(workspace, "repo")
    if os.path.exists(clone_dir):
        logger.info("Repo already exists at %s, skipping clone.", clone_dir)
        return

    logger.info("Cloning %s → %s", repo_url, clone_dir)
    try:
        subprocess.run(
            ["git", "clone", "--depth=1", repo_url, clone_dir],
            capture_output=True, text=True, timeout=180,
            check=True,
        )
        if base_commit:
            subprocess.run(
                ["git", "fetch", "--depth=1", "origin", base_commit],
                cwd=clone_dir,
                capture_output=True, text=True, timeout=60,
                check=False,
            )
            subprocess.run(
                ["git", "checkout", "-q", base_commit],
                cwd=clone_dir,
                capture_output=True, text=True, timeout=30,
                check=False,
            )
        logger.info("Repo cloned successfully: %s", clone_dir)
    except subprocess.CalledProcessError as exc:
        logger.warning(
            "Failed to clone repo (exit=%d): %.300s",
            exc.returncode, getattr(exc, "stderr", b"")[:300],
        )
    except subprocess.TimeoutExpired:
        logger.warning("Timeout cloning repo %s", repo_url)


# ── Agent subprocess management ─────────────────────────────────────────────


async def _launch_agent(
    *,
    task: str,
    base_url: str,
    workspace: str,
    max_turns: int = 100,
    agent_timeout: int = 3600,
) -> tuple[int, str, float]:
    """Launch ``claude_code_entrypoint.py`` as a subprocess.

    The entrypoint calls the Gateway for model inference (vLLM → Qwen3-4B)
    and executes tools in the workspace.  The Gateway captures full
    token-level trajectories for DPO training.

    Returns:
        (exit_code, stdout_tail, elapsed_seconds)
    """
    entrypoint = Path(__file__).resolve().parent / "claude_code_entrypoint.py"

    env = os.environ.copy()
    env.update({
        "CLAUDE_TASK": task,
        "CLAUDE_BASE_URL": base_url,
        "CLAUDE_WORKSPACE": workspace,
        "AGENT_MAX_TURNS": str(max_turns),
    })
    # Redact sensitive values from the subprocess environment
    for sensitive in ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY", "DEEPSEEK_API_KEY",
                       "JUDGE_API_KEY", "HF_TOKEN"):
        env.pop(sensitive, None)
    # Unset proxy vars that might interfere with local Gateway traffic
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
                 "NO_PROXY", "no_proxy"):
        env.pop(var, None)

    logger.info("Launching agent: python %s", entrypoint)

    started_at = time.perf_counter()
    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                sys.executable,
                str(entrypoint),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
                cwd=workspace,
            ),
            timeout=10,
        )
        stdout_bytes, _ = await asyncio.wait_for(
            proc.communicate(),
            timeout=agent_timeout,
        )
    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - started_at
        return (-1, "(agent timed out)", elapsed)

    elapsed = time.perf_counter() - started_at
    stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
    tail = stdout[-8000:] if len(stdout) > 8000 else stdout
    return (proc.returncode or 0, tail, elapsed)


# ── Reward evaluation ───────────────────────────────────────────────────────
# (Same logic as custom_hermes_runner._evaluate_reward)


async def _evaluate_reward(
    *,
    task: str,
    agent_stdout: str,
    agent_exit_code: int,
    tools_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate reward for the completed agent run.

    Two modes (controlled by ``tools_kwargs.use_batch_judge``):

    **Batch judge mode** (``use_batch_judge=True``):
        Skips the per-sample LLM judge call and posts raw agent output
        + task + data_source to the Gateway.  The actual scoring happens
        later via ``judge_batch`` in the framework.

    **Inline judge mode** (default):
        Calls ``judge_single`` with a dataset-specific rubric loaded
        from ``prompts/{data_source}_judge.txt`` when available, falling
        back to the generic rubric.

    Returns a dict to be posted as ``reward_info``.
    """
    tools_kwargs = tools_kwargs or {}
    reward_cfg = tools_kwargs.get("reward", {})
    data_source = tools_kwargs.get("data_source", "")
    use_batch_judge = tools_kwargs.get("use_batch_judge", False) or reward_cfg.get(
        "use_batch_judge", False
    )

    # ── Batch judge mode: post raw data, skip inline scoring ──────────────
    if use_batch_judge:
        logger.info(
            "Batch judge mode: deferring scoring for data_source=%r", data_source
        )
        return {
            "task": task[:20000],
            "agent_output": agent_stdout[:20000],
            "data_source": data_source,
            "agent_exit_code": agent_exit_code,
        }

    # ── Inline judge mode ─────────────────────────────────────────────────
    judge_api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get(
        "JUDGE_API_KEY"
    )
    if judge_api_key and reward_cfg.get("use_judge", True):
        try:
            from Rl_Specilist.agent.online_dpo.reward.llm_judge import (
                judge_single,
                load_judge_prompt,
            )

            rubric = load_judge_prompt(data_source) or reward_cfg.get("rubric")
            result = await judge_single(
                task=task,
                agent_output=agent_stdout,
                rubric=rubric,
            )
            return {
                **result,
                "agent_exit_code": agent_exit_code,
            }
        except Exception:
            logger.warning(
                "LLM Judge failed; falling back to basic reward", exc_info=True
            )

    # Basic reward: succeeded if agent exited cleanly
    success = agent_exit_code == 0
    return {
        "reward_score": 1.0 if success else 0.0,
        "agent_exit_code": agent_exit_code,
        "success": success,
    }


# ── Main runner ─────────────────────────────────────────────────────────────


async def custom_claude_runner(
    *,
    raw_prompt: Any,
    session: SessionHandle,
    sample_index: int,
    tools_kwargs: dict[str, Any] | None = None,
    agent_max_turns: int = 100,
    agent_timeout: int = 3600,
    **kwargs: Any,
) -> None:
    """Run Claude-Code-style agent with local workspace.

    This is the entry point wired in the training config under
    ``agent_runners.custom_claude.runner_fqn``.  The ``AgentFramework``
    calls it once per rollout sample.

    Flow:
        1. Build task from raw_prompt + tools_kwargs.
        2. Create isolated workspace ``/tmp/verl_claude/<session_id>``,
           optionally cloning the target repo for SWE tasks.
        3. Launch ``claude_code_entrypoint.py`` pointing at
           ``session.base_url`` (Gateway → vLLM → Qwen3-4B).
        4. Wait for agent process to complete.
        5. Evaluate reward (LLM Judge or basic).
        6. POST ``reward_info`` to ``session.reward_info_url``.
    """
    tools_kwargs = tools_kwargs or {}
    logger.info(
        "custom_claude_runner called: sample=%d session=%s",
        sample_index,
        session.session_id,
    )

    # ---- 1. Build task ----
    task = build_claude_task(raw_prompt, tools_kwargs)
    logger.info("Sample %d task: %.200s", sample_index, task)

    # ---- 2. Create workspace ----
    workspace = setup_workspace(session.session_id, tools_kwargs)

    # ---- 3. Validate session ----
    base_url = session.base_url
    if not base_url:
        raise ValueError(
            f"session.base_url is empty for session {session.session_id}"
        )
    reward_info_url = session.reward_info_url
    if not reward_info_url:
        raise ValueError(
            f"session.reward_info_url is empty for session {session.session_id}"
        )

    # ---- 4. Launch agent ----
    max_turns = int(
        os.environ.get("AGENT_MAX_TURNS", str(agent_max_turns))
    )
    exit_code, agent_stdout, elapsed = await _launch_agent(
        task=task,
        base_url=base_url,
        workspace=workspace,
        max_turns=max_turns,
        agent_timeout=agent_timeout,
    )
    logger.info(
        "Sample %d agent finished: exit_code=%d elapsed=%.1fs",
        sample_index,
        exit_code,
        elapsed,
    )

    # ---- 5. Evaluate reward ----
    reward_info = await _evaluate_reward(
        task=task,
        agent_stdout=agent_stdout,
        agent_exit_code=exit_code,
        tools_kwargs=tools_kwargs,
    )
    reward_info["elapsed_seconds"] = elapsed
    reward_info["agent_type"] = "claude_code"
    logger.info(
        "Sample %d reward: score=%.2f",
        sample_index,
        reward_info.get("reward_score", 0.0),
    )

    # ---- 6. Post reward_info to Gateway ----
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            reward_info_url,
            json={"reward_info": reward_info},
        )
        response.raise_for_status()
    logger.info("Sample %d reward_info posted", sample_index)

    # ---- 7. Cleanup workspace ----
    try:
        shutil.rmtree(workspace, ignore_errors=True)
    except Exception:
        logger.warning(
            "Failed to clean workspace: %s", workspace, exc_info=True
        )
