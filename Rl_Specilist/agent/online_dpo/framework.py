"""Batch-judge-aware AgentFramework for online DPO Gateway flow.

Extends ``OpenAICompatibleAgentFramework`` to support deferred batch-level
LLM judge scoring with dataset-specific prompts.  Zero changes to uni_agent
— the subclass is wired via ``framework_class_fqn`` in the Hydra config.

Architecture::

    _run_batch_to_tq  (overridden)
      ├─ For each prompt: run all N sessions concurrently
      │     └─ _run_session  (overridden — skips _score_trajectories when
      │           _defer_reward is set, returns placeholder scores)
      ├─ Collect all session trajectories into a flat list
      ├─ _compute_batch_scores()
      │     ├─ Group by data_source
      │     ├─ Load dataset-specific prompt for each group
      │     ├─ Call judge_batch() (relative scoring within each group)
      │     └─ Assign scores back to each trajectory
      └─ Write scored trajectories to TransferQueue
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import replace
from typing import Any

from uni_agent.framework.framework import (
    OpenAICompatibleAgentFramework,
    _short_failure_reason,
)
from verl.utils.transferqueue_utils import tq

logger = logging.getLogger(__name__)

# ── Prompt resolution (shared with runner via llm_judge.load_judge_prompt) ───

_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")


# ── BatchJudgeAgentFramework ─────────────────────────────────────────────────


class BatchJudgeAgentFramework(OpenAICompatibleAgentFramework):
    """Framework subclass that defers per-session scoring to batch-level.

    When ``_defer_reward`` is True (controlled by
    ``reward.use_batch_judge`` in the Hydra config):

    * ``_run_session`` returns placeholder scores (0.0) instead of
      calling ``_score_trajectories`` per session.
    * ``_run_batch_to_tq`` accumulates all trajectories across prompts
      and sessions, calls ``_compute_batch_scores`` once, then writes
      scored trajectories to the TransferQueue.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._defer_reward: bool = False

    @classmethod
    def from_config(
        cls,
        *,
        config: Any,
        gateway_manager: Any,
        processor: Any = None,
        reward_loop_worker_handles: Any = None,
    ) -> BatchJudgeAgentFramework:
        """Create from Hydra config, reading ``use_batch_judge`` flag."""
        from omegaconf import OmegaConf

        instance: BatchJudgeAgentFramework = super().from_config(
            config=config,
            gateway_manager=gateway_manager,
            processor=processor,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )
        reward_config = OmegaConf.select(config, "reward", default={}) or {}
        instance._defer_reward = bool(reward_config.get("use_batch_judge", False))
        if instance._defer_reward:
            logger.info(
                "BatchJudgeAgentFramework: deferred batch judge mode ENABLED"
            )
        return instance

    # ── Override: skip per-session scoring when deferred ──────────────────

    async def _run_session(
        self,
        *,
        sample_fields: dict[str, object],
        sample_index: int,
        session_index: int,
        runner_name: str,
        runner_config: Any,
    ) -> tuple[list[Any], dict[str, object]]:
        """Run one session; skip ``_score_trajectories`` when deferred."""
        if not self._defer_reward:
            return await super()._run_session(
                sample_fields=sample_fields,
                sample_index=sample_index,
                session_index=session_index,
                runner_name=runner_name,
                runner_config=runner_config,
            )

        # Inject batch-judge flag into tools_kwargs so the runner
        # knows to skip inline scoring and post raw agent output.
        tools_kwargs = dict(sample_fields.get("tools_kwargs") or {})
        tools_kwargs["use_batch_judge"] = True
        sample_fields = {**sample_fields, "tools_kwargs": tools_kwargs}

        # Run the session normally (runner → gateway → finalize)
        trajectories, session_sample_fields = await super()._run_session(
            sample_fields=sample_fields,
            sample_index=sample_index,
            session_index=session_index,
            runner_name=runner_name,
            runner_config=runner_config,
        )

        # Replace placeholder scores with deferred marker.
        # Real scores are assigned later by _compute_batch_scores.
        scored = []
        for traj in trajectories:
            scored.append(
                replace(
                    traj,
                    reward_score=0.0,
                    extra_fields={
                        **traj.extra_fields,
                        "reward_extra_info": {"deferred": True},
                    },
                )
            )
        return scored, session_sample_fields

    # ── Override: collect all trajectories → batch judge → write TQ ───────

    async def _run_batch_to_tq(
        self,
        prompts: Any,
        *,
        global_steps: int,
        partition_id: str,
        num_sessions: int = 1,
    ) -> dict[str, Any]:
        """Run all prompts; when deferred, batch-score before TQ write."""
        if not self._defer_reward:
            return await super()._run_batch_to_tq(
                prompts,
                global_steps=global_steps,
                partition_id=partition_id,
                num_sessions=num_sessions,
            )

        import numpy as np

        assert len(prompts) > 0, "generate_sequences requires a non-empty batch"
        if num_sessions <= 0:
            raise ValueError(f"num_sessions must be positive, got {num_sessions}")

        # ── Phase 1: run all sessions (no scoring) ────────────────────────
        # Accumulator: list of (trajectories, sample_fields, uid, session_index)
        buffer: list[dict[str, Any]] = []
        failure_reasons: list[str] = []
        stats: dict[str, int] = {
            "num_input_prompts": len(prompts),
            "num_success_sessions": 0,
            "num_failed_sessions": 0,
            "num_success_outputs": 0,
            "num_failed_uids": 0,
            "failure_reasons": [],
        }
        stats["failure_reasons"] = failure_reasons  # type: ignore[assignment]

        for sample_index in range(len(prompts)):
            sample_fields = self._extract_sample_fields(
                prompts=prompts, sample_index=sample_index
            )
            uid = str(sample_fields.get("uid", ""))
            if not uid:
                raise ValueError("uid is required in prompts")

            tasks = [
                self._run_session_with_concurrency_limit(
                    sample_fields=sample_fields,
                    sample_index=sample_index,
                    session_index=session_index,
                )
                for session_index in range(num_sessions)
            ]
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)

            prompt_has_success = False
            for session_index, outcome in enumerate(outcomes):
                if isinstance(outcome, Exception):
                    stats["num_failed_sessions"] += 1
                    failure_reasons.append(
                        _short_failure_reason(outcome)[:512]
                    )
                    continue
                if isinstance(outcome, BaseException):
                    raise outcome  # CancelledError etc.

                trajectories, session_sample_fields = outcome
                if not trajectories:
                    stats["num_failed_sessions"] += 1
                    failure_reasons.append(
                        f"empty trajectories for uid={uid}"
                    )
                    continue

                stats["num_success_sessions"] += 1
                prompt_has_success = True
                buffer.append({
                    "trajectories": trajectories,
                    "sample_fields": session_sample_fields,
                    "session_index": session_index,
                    "uid": uid,
                })

            if prompt_has_success:
                stats["num_success_outputs"] += num_sessions
                await tq.async_kv_put(
                    key=uid,
                    partition_id=partition_id,
                    tag={"status": "finished"},
                )
            else:
                stats["num_failed_uids"] += 1
                await tq.async_kv_put(
                    key=uid,
                    partition_id=partition_id,
                    tag={"status": "failure"},
                )

        # ── Phase 2: batch judge ──────────────────────────────────────────
        await self._compute_batch_scores(buffer)

        # ── Phase 3: write scored trajectories to TQ ──────────────────────
        for entry in buffer:
            try:
                await self._write_session_trajectories_to_tq(
                    uid=entry["uid"],
                    session_index=entry["session_index"],
                    trajectories=entry["trajectories"],
                    sample_fields=entry["sample_fields"],
                    global_steps=global_steps,
                    partition_id=partition_id,
                )
            except Exception:
                logger.warning(
                    "Failed to write trajectories to TQ for uid=%s session=%s",
                    entry["uid"],
                    entry["session_index"],
                    exc_info=True,
                )

        if stats["num_success_outputs"] == 0:
            raise RuntimeError(
                f"All batch rollouts failed at global_steps={global_steps}. "
                f"failures={stats['num_failed_uids']}/{stats['num_input_prompts']}"
            )

        return stats

    # ── Batch scoring ─────────────────────────────────────────────────────

    async def _compute_batch_scores(
        self, buffer: list[dict[str, Any]]
    ) -> None:
        """Score all accumulated trajectories via batch LLM judge.

        Groups trajectories by ``data_source``, loads the appropriate
        dataset-specific prompt for each group, and calls ``judge_batch``
        for relative scoring within each group.
        """
        if not buffer:
            return

        from Rl_Specilist.agent.online_dpo.reward.llm_judge import (
            judge_batch,
            load_judge_prompt,
        )

        # Flatten: each (buffer_idx, traj_idx) points into buffer
        all_tasks: list[str] = []
        all_outputs: list[str] = []
        all_data_sources: list[str] = []
        refs: list[tuple[int, int]] = []  # (buffer_idx, traj_idx)

        for buf_idx, entry in enumerate(buffer):
            for traj_idx, traj in enumerate(entry["trajectories"]):
                rinfo = traj.reward_info or {}
                all_tasks.append(str(rinfo.get("task", "")))
                all_outputs.append(str(rinfo.get("agent_output", "")))
                ds = str(
                    rinfo.get("data_source")
                    or entry["sample_fields"].get("data_source", "unknown")
                )
                all_data_sources.append(ds.strip().lower())
                refs.append((buf_idx, traj_idx))

        if not all_tasks:
            logger.warning("_compute_batch_scores: no tasks to score")
            return

        logger.info(
            "_compute_batch_scores: scoring %d trajectories across %d sessions",
            len(all_tasks),
            len(buffer),
        )

        # Group by data_source for dataset-specific prompts
        groups: dict[str, list[int]] = {}
        for i, ds in enumerate(all_data_sources):
            groups.setdefault(ds, []).append(i)

        total_scored = 0
        for ds, indices in groups.items():
            prompt = load_judge_prompt(ds)
            group_tasks = [all_tasks[i] for i in indices]
            group_outputs = [all_outputs[i] for i in indices]

            try:
                scores = await judge_batch(
                    tasks=group_tasks,
                    outputs=group_outputs,
                    rubric=prompt,
                )
            except Exception:
                logger.warning(
                    "Batch judge failed for data_source=%r; "
                    "assigning default scores",
                    ds,
                    exc_info=True,
                )
                scores = [
                    {"reward_score": 0.5, "judge_reason": "batch judge error"}
                    for _ in group_tasks
                ]

            # Write scores back to trajectory objects
            for local_idx, score_dict in zip(indices, scores):
                buf_idx, traj_idx = refs[local_idx]
                traj = buffer[buf_idx]["trajectories"][traj_idx]
                traj.reward_score = float(score_dict.get("reward_score", 0.0))
                reason = str(score_dict.get("judge_reason", ""))
                traj.extra_fields["reward_extra_info"] = {
                    "judge_reason": reason,
                    "data_source": ds,
                    "scoring_mode": "batch_relative",
                }

            total_scored += len(indices)
            logger.info(
                "Batch judge: scored %d trajectories for data_source=%r "
                "with prompt=%s",
                len(indices),
                ds,
                "custom" if prompt else "default",
            )

        logger.info("_compute_batch_scores: done (%d scored)", total_scored)
