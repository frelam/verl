"""Custom Hermes runner for command-line agent tasks.

Runner contract for the Uni-Agent ``AgentFramework``.  One runner invocation
handles a single sample: it builds a task, creates an isolated workspace,
launches the Hermes-format agent entrypoint against the Gateway, evaluates
the reward, and posts ``reward_info`` to the session.

Architecture::

    Runner (this file)
      ├─ Build task from raw_prompt + tools_kwargs
      ├─ Create workspace /tmp/verl_hermes/<session_id>
      ├─ Launch hermes_entrypoint.py via subprocess
      │     └─ Agent → Gateway /v1/chat/completions → vLLM (Qwen3-4B)
      │     └─ Agent ← Gateway ← vLLM assistant reply
      │     └─ Agent → execute tools in workspace → observation
      │     └─ ... loop until submit_answer or max_turns ...
      ├─ Evaluate reward (judge_single from reward/llm_judge.py)
      └─ POST reward_info → Gateway session
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Gateway session handle — imported at runtime from uni-agent.
# Defined inline as a fallback for development / type checking.
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
    # OpenAI-format message list
    for msg in raw_prompt:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # Multi-modal content blocks — take the first text block
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        return block.get("text", "")
    return str(raw_prompt)


def build_hermes_task(
    raw_prompt: Any,
    tools_kwargs: dict[str, Any] | None = None,
) -> str:
    """Build the task string for the Hermes agent entrypoint.

    Supports two modes:

    1. **Bare prompt** — ``raw_prompt`` is a plain task string.
       Sent directly to the agent.

    2. **Structured prompt** — ``tools_kwargs`` carries metadata
       (e.g. ground-truth answer, expected output).  The task string
       is enriched with context from ``tools_kwargs``.
    """
    tools_kwargs = tools_kwargs or {}
    task = extract_task(raw_prompt)

    # If the dataset provides a pre-formatted task in tools_kwargs, use it
    prebuilt = tools_kwargs.get("task")
    if prebuilt and isinstance(prebuilt, str):
        return prebuilt

    # Otherwise use the raw task as-is
    return task.strip()


# ── Agent subprocess management ─────────────────────────────────────────────


async def _launch_agent(
    *,
    task: str,
    base_url: str,
    workspace: str,
    max_turns: int = 100,
    agent_timeout: int = 3600,
    model: str = "default",
) -> tuple[int, str, float]:
    """Launch ``hermes_entrypoint.py`` as a subprocess and wait for completion.

    Returns:
        (exit_code, stdout_tail, elapsed_seconds)
    """
    entrypoint = Path(__file__).resolve().parent / "hermes_entrypoint.py"

    env = os.environ.copy()
    env.update(
        {
            "HERMES_TASK": task,
            "HERMES_BASE_URL": base_url,
            "HERMES_WORKSPACE": workspace,
            "AGENT_MAX_TURNS": str(max_turns),
            "HERMES_MODEL": model,
        }
    )
    # Unset proxy vars that might interfere with local Gateway traffic
    for var in (
        "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
        "NO_PROXY", "no_proxy",
    ):
        env.pop(var, None)

    cmd = [shlex.quote(str(entrypoint))]
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
    tail = stdout[-4000:] if len(stdout) > 4000 else stdout
    return (proc.returncode or 0, tail, elapsed)


# ── Reward evaluation ───────────────────────────────────────────────────────


async def _evaluate_reward(
    *,
    task: str,
    agent_stdout: str,
    agent_exit_code: int,
    tools_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate reward for the completed agent run (inline LLM judge).

    Calls ``judge_single`` with a dataset-specific rubric loaded from
    ``prompts/{data_source}_judge.txt`` when available, falling back to
    the generic rubric.

    Returns a dict to be posted as ``reward_info``.
    """
    tools_kwargs = tools_kwargs or {}
    reward_cfg = tools_kwargs.get("reward", {})
    data_source = tools_kwargs.get("data_source", "")

    # ── Inline judge mode ─────────────────────────────────────────────────
    judge_api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("JUDGE_API_KEY")
    if judge_api_key and reward_cfg.get("use_judge", True):
        try:
            from uni_agent.reward.llm_judge import (
                judge_single,
                load_judge_prompt,
            )

            # Load dataset-specific rubric when available
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
            logger.warning("LLM Judge failed; falling back to basic reward", exc_info=True)

    # Basic reward: succeeded if agent exited cleanly
    success = agent_exit_code == 0
    return {
        "reward_score": 1.0 if success else 0.0,
        "agent_exit_code": agent_exit_code,
        "success": success,
    }


# ── Main runner ─────────────────────────────────────────────────────────────


async def custom_hermes_runner(
    *,
    raw_prompt: Any,
    session: SessionHandle,
    sample_index: int,
    tools_kwargs: dict[str, Any] | None = None,
    agent_max_turns: int = 100,
    agent_timeout: int = 3600,
    **kwargs: Any,
) -> None:
    """Run Hermes-format agent with local workspace.

    This is the entry point wired in the training config under
    ``agent_runners.custom_hermes.runner_fqn``.  The ``AgentFramework``
    calls it once per rollout sample.

    Flow:
        1. Build task from raw_prompt + tools_kwargs.
        2. Create isolated workspace ``/tmp/verl_hermes/<session_id>``.
        3. Launch ``hermes_entrypoint.py`` pointing at ``session.base_url``.
        4. Wait for agent process to complete.
        5. Evaluate reward (LLM Judge or basic).
        6. POST ``reward_info`` to ``session.reward_info_url``.
    """
    tools_kwargs = tools_kwargs or {}
    logger.info(
        "custom_hermes_runner called: sample=%d session=%s",
        sample_index,
        session.session_id,
    )

    # ---- 1. Build task ----
    task = build_hermes_task(raw_prompt, tools_kwargs)
    logger.info("Sample %d task: %.200s", sample_index, task)

    # ---- 2. Create workspace ----
    workspace_root = os.environ.get(
        "HERMES_WORKSPACE_ROOT", "/tmp/verl_hermes"
    )
    workspace = os.path.join(workspace_root, session.session_id)
    os.makedirs(workspace, exist_ok=True)

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
        logger.warning("Failed to clean workspace: %s", workspace, exc_info=True)
