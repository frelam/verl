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
"""Batch-level LLM Judge reward manager for the reward loop infrastructure.

This reward manager implements ``run_batch`` which sends ALL samples in a
chunk to the LLM judge API in a single call, enabling **relative scoring**
where the judge compares trajectories against each other.

Unlike ``NaiveRewardManager`` which scores samples independently, this manager
produces scores that are relative within each batch, providing a stronger
training signal for DPO.

Usage:
    In verl config:
    ```yaml
    reward:
      reward_manager:
        source: register
        name: llm_judge
      custom_reward_function:
        path: verl.utils.reward_score.llm_judge_reward
        name: compute_score
    ```
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from typing import Any

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score.llm_judge_reward import (
    _call_judge_batch,
    _call_judge_single,
    _extract_trajectory_text,
    _get_judge_config,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("llm_judge")
class LLMJudgeRewardManager(RewardManagerBase):
    """Reward manager that uses an LLM judge to score trajectories in batch.

    Supports both:
    - ``run_batch``: Scores all samples together (relative mode, default).
      The judge sees all trajectories and assigns relative scores.
    - ``run_single``: Scores a single sample (absolute mode, fallback).
      Used when there's only one trajectory in a group.

    The judge API is called with an OpenAI-compatible interface.
    Configuration is read from environment variables (see
    ``llm_judge_reward._get_judge_config`` for details).
    """

    def __init__(
        self,
        config,
        tokenizer,
        compute_score=None,
        reward_router_address=None,
        reward_model_tokenizer=None,
    ):
        super().__init__(config, tokenizer, compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer
        # compute_score from config is the batch-level function; we also
        # accept a direct callable override.
        self.custom_score_fn = compute_score

    async def run_batch(self, data: DataProto) -> list[dict]:
        """Score all trajectories in ``data`` together via the LLM judge.

        Sends the full batch to the judge API, which compares trajectories
        relative to each other and returns per-sample scores.

        Args:
            data: DataProto containing all samples in this chunk.

        Returns:
            List of dicts, each containing ``reward_score`` and
            ``reward_extra_info``.
        """
        n = len(data)
        if n == 0:
            return []

        # If only one sample, fall back to single scoring
        if n == 1:
            return [await self.run_single(data)]

        # Extract trajectories and metadata for each sample
        trajectories = []
        tasks = []
        ground_truths = []
        data_sources = []
        extra_infos = []

        for i in range(n):
            data_item = data[i]
            response_ids = data_item.batch["responses"]
            response_length = response_ids.shape[-1]
            valid_response_length = data_item.batch["attention_mask"][
                -response_length:
            ].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # Decode response
            response_str = await self.loop.run_in_executor(
                None,
                lambda ids=valid_response_ids: self.tokenizer.decode(
                    ids, skip_special_tokens=True
                ),
            )

            data_source = data_item.non_tensor_batch.get("data_source", "unknown")
            ground_truth = data_item.non_tensor_batch.get("reward_model", {}).get(
                "ground_truth", ""
            )
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            tool_extra_fields = data_item.non_tensor_batch.get(
                "tool_extra_fields", None
            )
            if tool_extra_fields is not None:
                extra_info.update(tool_extra_fields.items())

            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            data_sources.append(str(data_source))
            ground_truths.append(ground_truth)
            extra_infos.append(extra_info)

            # Try to get the full multi-turn messages from extra_info
            traj = _extract_trajectory_text(
                {"response_str": response_str}, extra_info
            )
            trajectories.append(traj)

            # Extract task
            task = (
                extra_info.get("question")
                or extra_info.get("task")
                or str(data_source)
            )
            tasks.append(str(task))

        # Get judge config from first sample's extra_info
        config = _get_judge_config(extra_infos[0] if extra_infos else None)
        scoring_mode = config.get("scoring_mode", "relative")

        # Call judge
        if scoring_mode == "relative":
            scores = _call_judge_batch(
                trajectories, tasks, ground_truths, config
            )
        else:
            # Absolute mode: score each trajectory independently
            scores = []
            for i in range(n):
                s = _call_judge_single(
                    trajectories[i], tasks[i], ground_truths[i],
                    config, extra_infos[i],
                )
                s["index"] = i
                scores.append(s)

        # Build return format compatible with NaiveRewardManager's output
        results = []
        for i in range(n):
            score_entry = scores[i] if i < len(scores) else {"score": 0.0}
            score = float(score_entry.get("score", 0.0))

            reward_extra_info = {}
            for key, value in score_entry.items():
                if key not in ("score", "index"):
                    reward_extra_info[key] = value
            reward_extra_info["judge_source"] = config.get("model", "unknown")
            reward_extra_info["scoring_mode"] = scoring_mode

            results.append({
                "reward_score": score,
                "reward_extra_info": reward_extra_info,
            })

        return results

    async def run_single(self, data: DataProto) -> dict:
        """Score a single trajectory via the LLM judge.

        Used as fallback when there's only one sample in a chunk.

        Args:
            data: DataProto containing a single sample.

        Returns:
            Dict with ``reward_score`` and ``reward_extra_info``.
        """
        data = data[-1:]  # Only use the last sequence
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][
            -response_length:
        ].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch.get("data_source", "unknown")
        ground_truth = data_item.non_tensor_batch.get("reward_model", {}).get(
            "ground_truth", ""
        )
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        tool_extra_fields = data_item.non_tensor_batch.get(
            "tool_extra_fields", None
        )
        if tool_extra_fields is not None:
            extra_info.update(tool_extra_fields.items())

        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        extra_info["num_turns"] = num_turns

        response_str = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=True
            ),
        )

        traj = _extract_trajectory_text(
            {"response_str": response_str}, extra_info
        )
        task = (
            extra_info.get("question")
            or extra_info.get("task")
            or str(data_source)
        )
        gt_str = str(ground_truth) if ground_truth else ""

        config = _get_judge_config(extra_info)
        score_entry = _call_judge_single(traj, task, gt_str, config, extra_info)

        score = float(score_entry.get("score", 0.0))
        reward_extra_info = {}
        for key, value in score_entry.items():
            if key not in ("score", "index"):
                reward_extra_info[key] = value
        reward_extra_info["judge_source"] = config.get("model", "unknown")

        return {
            "reward_score": score,
            "reward_extra_info": reward_extra_info,
        }
