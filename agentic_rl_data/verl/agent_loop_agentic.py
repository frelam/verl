"""verl async agent-loop adapter for ``agentic_rl_data`` (registered ``agentic_agent``).

What this loop does
-------------------
One rollout drives one sample through the sample's own environment:

1. rebuild the canonical task from ``extra_info["agentic_task"]`` (JSON string);
2. ``build_env(task, max_turns=meta.max_turns)`` + ``reset``;
3. loop: generate -> parse tool calls with the configured verl ``ToolParser``
   -> ``env.step({content, tool_calls})`` -> append ``obs["tool_results"]`` as
   chat ``tool`` messages -> accumulate tokens exactly like
   ``verl.experimental.agent_loop.tool_agent_loop.ToolAgentLoop``;
4. assemble ``trajectory`` (CONTRACTS C3/C4), dispatch the rule reward
   **inside the loop** and return ``AgentLoopOutput(reward_score=...)`` so
   verl's ``_postprocess`` builds ``rm_scores`` directly (the configured
   reward manager then short-circuits, see
   ``verl/workers/reward_manager/abstract.py``).

Per-sample tools
----------------
The dataset ``tools`` column travels through ``kwargs["tools"]``.  The
exporter stores ``{name, description, parameters(JSON string)}`` (pyarrow
struct); :func:`agentic_rl_data.verl.trajectory.normalize_tools_for_template`
wraps it into the OpenAI form used by chat templates and tool parsers.  No
global ``load_all_tools`` registry is involved (unlike ``ToolAgentLoop``).

Wiring (see ``agentic_rl_data/reports/verl_agent_loop.md``)::

    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.agent.default_agent_loop=agentic_agent
    actor_rollout_ref.rollout.agent.agent_loop_config_path=agentic_rl_data/verl/agent_loop_config.yaml
    actor_rollout_ref.rollout.multi_turn.format=hermes   # parser matching the served model

Fail-closed behaviour: reward exceptions, env build/step failures and missing
``agentic_task`` never raise out of this loop; they degrade to a single-turn
trajectory and/or ``reward_score=0.0`` with diagnostics in
``extra_fields["agentic/env_error"]`` / ``extra_fields["reward_extra_info"]``.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

# Make ``agentic_rl_data`` importable when the module is loaded from an
# arbitrary cwd (hydra resolves ``_target_`` by import path, but tests / direct
# file execution may not have the repo root on sys.path).
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ..envs.base import build_env  # noqa: E402  (see _REPO_ROOT note)
from .trajectory import (  # noqa: E402
    MISSING_TASK_REWARD_ERROR,
    as_mapping,
    build_calls,
    build_trajectory,
    fail_closed_reward,
    final_answer_from_turns,
    make_turn,
    normalize_reward_info,
    normalize_tools_for_template,
    parse_agentic_task,
    resolve_max_turns,
    task_env_type,
    task_has_env,
    task_task_type,
    tool_results_to_messages,
    tool_results_to_trajectory,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _as_int(value: Any, default: int = 0) -> int:
    """Coerce a ``non_tensor_batch`` cell (possibly ``np.int64``) to ``int``."""
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_text(value: Any) -> str:
    """Coerce an optional scalar cell (e.g. NumPy string) to ``str``."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return str(item())
        except (TypeError, ValueError):  # pragma: no cover - exotic dtypes
            return ""
    return str(value)


@register("agentic_agent")
class AgenticAgentLoop(AgentLoopBase):
    """Multi-turn agent loop backed by ``agentic_rl_data`` environments."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.full_determinism = bool(getattr(self.rollout_config, "full_determinism", False))

        # The tool parser must match the served model's tool-call format; the
        # agentic task types only need the parser (per-sample tools are parsed
        # against the schemas from the dataset row, not a global registry).
        parser_name = "hermes"
        multi_turn_config = getattr(self.rollout_config, "multi_turn", None)
        configured = getattr(multi_turn_config, "format", None) if multi_turn_config is not None else None
        if isinstance(configured, str) and configured.strip():
            parser_name = configured.strip()
        try:
            self.tool_parser = ToolParser.get_tool_parser(parser_name, self.tokenizer)
            self.tool_parser_name = parser_name
        except ValueError:
            logger.warning(
                "[agentic_agent] unknown multi_turn.format=%r, falling back to 'hermes'", parser_name
            )
            self.tool_parser = ToolParser.get_tool_parser("hermes", self.tokenizer)
            self.tool_parser_name = "hermes"

    # -- verl entry point ---------------------------------------------------

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs: Any) -> AgentLoopOutput:
        priority = _as_int(priority, 0)
        metrics: dict[str, Any] = {
            "generate_sequences": 0.0,
            "tool_calls": 0.0,
            "compute_score": 0.0,
            "num_preempted": -1,
        }
        # ``request_id`` is reused across turns (sticky session); under
        # full_determinism it stays stable across runs too (see tool_rl_agent_loop).
        request_id = f"det-{priority}" if self.full_determinism else uuid4().hex

        messages = list(kwargs["raw_prompt"])
        sample_tools = normalize_tools_for_template(kwargs.get("tools"))
        parser_tools = self._parser_tools(sample_tools)
        extra_info = as_mapping(kwargs.get("extra_info"))

        task = parse_agentic_task(extra_info.get("agentic_task"))
        task_type = task_task_type(task) or _as_text(extra_info.get("task_type"))
        task_id = _as_text(extra_info.get("task_id"))
        if not task_id and isinstance(task, dict):
            task_id = _as_text(task.get("id"))
        max_turns = resolve_max_turns(task)
        env_type = task_env_type(task)
        requires_env = task is not None and task_has_env(task)

        # ---- env construction (degrade instead of crashing the worker) ----
        env: Any = None
        env_error = ""
        if requires_env:
            try:
                env = await self.loop.run_in_executor(
                    None, lambda: build_env(task, max_turns=max_turns)
                )
                await self.loop.run_in_executor(None, lambda: env.reset(task, max_turns=max_turns))
            except Exception as exc:  # noqa: BLE001 - env backend problems must not kill rollout
                env_error = f"env_setup_failed: {type(exc).__name__}: {exc}"
                logger.warning(
                    "[agentic_agent] task=%s env_type=%s: %s", task_id or "<unknown>", env_type, env_error
                )
                if env is not None:
                    await self._close_env(env)
                    env = None
        elif task is None:
            env_error = MISSING_TASK_REWARD_ERROR
            logger.warning(
                "[agentic_agent] row without extra_info['agentic_task'] (task_id=%r); single-turn fallback",
                task_id,
            )

        single_turn = env is None  # no env -> exactly one generation (see task card)

        prompt_ids = await self.apply_chat_template(messages, tools=sample_tools or None)
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        turns: list[dict[str, Any]] = []
        last_routed_experts: Any = None
        final_answer: Optional[str] = None
        env_state: Optional[dict[str, Any]] = None
        env_done = False
        timeout = False
        assistant_turns = 0

        try:
            while True:
                if len(response_mask) >= self.response_length:
                    break
                if assistant_turns >= max_turns:
                    break
                if single_turn and assistant_turns >= 1:
                    break
                assistant_turns += 1

                output: TokenOutput = await self._generate(
                    request_id, prompt_ids, sampling_params, priority, metrics
                )
                token_ids = list(output.token_ids)
                prompt_ids.extend(token_ids)
                response_mask.extend([1] * len(token_ids))
                if output.log_probs:
                    response_logprobs.extend(output.log_probs)
                if output.routed_experts is not None:
                    last_routed_experts = output.routed_experts

                content, tool_calls = await self._extract_tool_calls(token_ids, parser_tools)
                action_calls, trajectory_calls, call_errors = build_calls(tool_calls)
                if call_errors:
                    logger.warning(
                        "[agentic_agent] task=%s turn=%d malformed tool call(s): %s",
                        task_id or "<unknown>",
                        assistant_turns,
                        "; ".join(call_errors),
                    )

                if single_turn:
                    # No env: one assistant turn is the whole episode.  The
                    # reward is computed from this single-turn trajectory with
                    # env_state=None (CONTRACTS C3 handles the missing state).
                    turns.append(make_turn(assistant_turns, content, trajectory_calls, []))
                    final_answer = content
                    env_done = not requires_env
                    break

                if len(response_mask) >= self.response_length:
                    # Token budget exhausted; the env never saw this turn.
                    turns.append(make_turn(assistant_turns, content, trajectory_calls, []))
                    final_answer = final_answer_from_turns(turns)
                    break

                action = {"content": content, "tool_calls": action_calls}
                try:
                    with simple_timer("tool_calls", metrics):
                        obs, done, info = await self.loop.run_in_executor(None, env.step, action)
                except Exception as exc:  # noqa: BLE001 - env contract failure is recoverable
                    env_error = f"env_step_failed: {type(exc).__name__}: {exc}"
                    logger.warning(
                        "[agentic_agent] task=%s turn=%d: %s",
                        task_id or "<unknown>",
                        assistant_turns,
                        env_error,
                    )
                    turns.append(make_turn(assistant_turns, content, trajectory_calls, []))
                    final_answer = final_answer_from_turns(turns)
                    break

                tool_results = obs.get("tool_results") if isinstance(obs, dict) else None
                turns.append(
                    make_turn(
                        assistant_turns,
                        content,
                        trajectory_calls,
                        tool_results_to_trajectory(tool_results),
                    )
                )

                if done:
                    env_done = True
                    final_answer = info.get("final_answer") if isinstance(info, dict) else None
                    if not isinstance(final_answer, str):
                        final_answer = final_answer_from_turns(turns)
                    env_state = info.get("env_state") if isinstance(info, dict) else None
                    if not isinstance(env_state, dict):
                        env_state = None
                    timeout = bool(info.get("timeout")) if isinstance(info, dict) else False
                    if env_state is not None:
                        timeout = timeout or bool(env_state.get("timeout"))
                    break

                # Encode the tool results as the next prompt segment (same
                # pattern as ToolAgentLoop: remove_system_prompt + separator).
                tool_messages = tool_results_to_messages(tool_results)
                if not tool_messages:
                    logger.warning(
                        "[agentic_agent] task=%s turn=%d: env returned no tool_results, stopping",
                        task_id or "<unknown>",
                        assistant_turns,
                    )
                    final_answer = final_answer_from_turns(turns)
                    break
                tool_tokens = await self.apply_chat_template(tool_messages, remove_system_prompt=True)
                tool_tokens = list(self.turn_separator) + list(tool_tokens)
                if len(response_mask) + len(tool_tokens) >= self.response_length:
                    # Tool-response tokens do not fit in the response budget.
                    break
                prompt_ids.extend(tool_tokens)
                response_mask.extend([0] * len(tool_tokens))
                if response_logprobs:
                    response_logprobs.extend([0.0] * len(tool_tokens))
        finally:
            if env is not None:
                await self._close_env(env)

        # ---- split accumulated tokens into prompt / response ----
        if response_mask:
            split_at = len(prompt_ids) - len(response_mask)
            base_prompt = prompt_ids[:split_at]
            response_ids = prompt_ids[split_at:]
        else:
            base_prompt = list(prompt_ids)
            response_ids = []
        response_ids = response_ids[: self.response_length]
        response_mask = response_mask[: self.response_length]
        response_logprobs_out = response_logprobs[: self.response_length] if response_logprobs else None
        routed_experts = None
        if last_routed_experts is not None:
            limit = len(base_prompt) + self.response_length
            try:
                routed_experts = last_routed_experts[:limit]
            except Exception:  # pragma: no cover - defensive (non-sliceable object)
                routed_experts = last_routed_experts

        trajectory = build_trajectory(
            turns,
            final_answer=final_answer,
            # ``done`` is True only when the episode actually finished: the env
            # reported done, or the task declares no env at all (single turn is
            # the whole episode).  An env build/step failure or a token-budget
            # truncation leaves done=False and env_state=None (CONTRACTS C3).
            done=bool(env_done or (single_turn and not requires_env)),
            env_state=env_state,
        )

        # ---- in-loop reward (the "result term" truth source is env_state) ----
        reward_score: Optional[float]
        reward_info: dict[str, Any]
        if task is None:
            # Cannot dispatch without a task; let verl's reward manager score it
            # from reward_model.ground_truth (mixed/legacy datasets).
            reward_score = None
            reward_info = {"reward_error": MISSING_TASK_REWARD_ERROR}
        else:
            with simple_timer("compute_score", metrics):
                try:
                    reward_score, reward_info = await self.loop.run_in_executor(
                        None, lambda: fail_closed_reward(task, trajectory)
                    )
                except Exception as exc:  # pragma: no cover - executor failure only
                    reward_score = 0.0
                    reward_info = {"reward_error": f"reward executor failed: {type(exc).__name__}: {exc}"}

        base_reward_error = reward_info.get("reward_error", "") if isinstance(reward_info, dict) else ""
        extra_fields: dict[str, Any] = {
            # Fixed-key wrapper: verl's _postprocess indexes every sample of a
            # worker chunk with the first sample's key set (mixed task types
            # would otherwise raise KeyError).
            "reward_extra_info": normalize_reward_info(
                reward_info,
                task_type=task_type,
                score=reward_score if reward_score is not None else 0.0,
                num_turns=trajectory["num_turns"],
                done=trajectory["done"],
                timeout=timeout,
                reward_error=base_reward_error,
            ),
            "env_state": env_state,
            "num_assistant_turns": trajectory["num_turns"],
            "agentic/task_id": task_id,
            "agentic/task_type": task_type,
            "agentic/env_type": env_type,
            "agentic/env_error": env_error,
        }
        # Keep the same extras schema as ToolAgentLoop (turn-level shaping hooks
        # are unused here, but downstream batch concat expects the keys).
        extra_fields.update({"turn_scores": [], "tool_rewards": []})

        return AgentLoopOutput(
            prompt_ids=base_prompt,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs_out,
            routed_experts=routed_experts,
            reward_score=reward_score,
            num_turns=len(turns) + 1,
            metrics=AgentLoopMetrics(**metrics),
            extra_fields=extra_fields,
        )

    # -- helpers ------------------------------------------------------------

    def _parser_tools(self, sample_tools: list[dict[str, Any]]) -> list[Any] | None:
        """Tool schemas for ``ToolParser.extract_tool_calls``.

        Most parsers ignore the argument, but ``qwen3_coder`` reads the
        parameter types and expects ``OpenAIFunctionToolSchema`` instances.
        Pydantic validation is best-effort: on failure fall back to raw OpenAI
        dicts so hermes-style parsers keep working.
        """
        if not sample_tools:
            return None
        try:
            return [OpenAIFunctionToolSchema(**tool) for tool in sample_tools]
        except Exception:  # noqa: BLE001 - schema shapes vary across data sources
            logger.debug("[agentic_agent] tool schema is not OpenAIFunctionToolSchema-valid, using dicts")
            return list(sample_tools)

    def _stop_token_sampling_params(self, sampling_params: dict[str, Any]) -> dict[str, Any]:
        """Add the parser's stop tokens so generation halts after a tool call."""
        stop_token_ids = list(self.tool_parser.stop_token_ids or [])
        if not stop_token_ids:
            return sampling_params
        merged = list(set(list(sampling_params.get("stop_token_ids") or []) + stop_token_ids))
        return {**sampling_params, "stop_token_ids": merged}

    async def _generate(
        self,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        priority: int,
        metrics: dict[str, Any],
    ) -> TokenOutput:
        sampling_params = self._stop_token_sampling_params(dict(sampling_params))
        with simple_timer("generate_sequences", metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                priority=priority,
            )
        num_preempted = output.num_preempted if output.num_preempted is not None else 0
        if metrics.get("num_preempted") in (None, -1):
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        else:
            metrics["num_preempted"] += num_preempted
        return output

    async def _extract_tool_calls(
        self, token_ids: list[int], parser_tools: list[Any] | None
    ) -> tuple[str, list[Any]]:
        """Parse tool calls from one generated turn, never raising."""
        try:
            content, tool_calls = await self.tool_parser.extract_tool_calls(token_ids, parser_tools)
        except Exception as exc:  # noqa: BLE001 - a parser bug must not kill rollout
            logger.warning("[agentic_agent] tool parser %r failed: %s", self.tool_parser_name, exc)
            try:
                content = await self.loop.run_in_executor(
                    None, lambda: self.tokenizer.decode(token_ids, skip_special_tokens=True)
                )
            except Exception:  # pragma: no cover - defensive
                content = ""
            return content, []
        return content, list(tool_calls or [])

    async def _close_env(self, env: Any) -> None:
        try:
            await self.loop.run_in_executor(None, env.close)
        except Exception as exc:  # noqa: BLE001 - close is best effort
            logger.warning("[agentic_agent] env.close() failed: %s", exc)


__all__ = ["AgenticAgentLoop"]
