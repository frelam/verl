# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Answer submission tool with calibration support.

This tool lets the agent submit a final answer together with a self-reported
confidence score. It implements the "Know -> Search -> Verify -> Revise ->
Answer" chain described in the reward design doc:

* The agent must call ``submit_answer`` at least once to finish the task.
* Each submission returns whether the answer is correct, so the agent can
  revise if needed (verify + reflection reward).
* The confidence is stored so the reward function can compute a Brier-score
  calibration reward.

This covers capability 6 (failure recovery / reflection), 7 (calibration),
and the epistemic-awareness reward module.
"""

import logging
import os
from typing import Any, Dict, Optional
from uuid import uuid4

from verl.utils.reward_score import gsm8k as gsm8k_score
from verl.utils.reward_score import math_reward as math_reward_score
from verl.utils.reward_score import search_r1_like_qa_em as qa_em_score
from verl.utils.rollout_trace import rollout_trace_op

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _check_answer(answer: str, ground_truth: str, task_type: str) -> bool:
    """Check whether *answer* matches *ground_truth* for the given task type."""
    if task_type == "math":
        try:
            return math_reward_score.compute_score(answer, ground_truth) > 0
        except Exception:
            return False
    if task_type == "gsm8k":
        try:
            return gsm8k_score.compute_score(answer, ground_truth, method="flexible") > 0
        except Exception:
            return False
    if task_type == "qa":
        try:
            gt = {"target": [ground_truth]} if isinstance(ground_truth, str) else {"target": ground_truth}
            return qa_em_score.compute_score(answer, gt) > 0
        except Exception:
            return False
    # Fallback: exact match after normalisation
    return str(answer).strip().lower() == str(ground_truth).strip().lower()


class AnswerSubmitTool(BaseTool):
    """Tool for submitting a final answer with a confidence score.

    The tool tracks the best (highest-reward) submission so far and returns
    a small step reward that encourages improvement across revisions:

    * First correct submission  -> +0.3 step reward
    * Improved submission       -> +0.1 step reward
    * No improvement            -> -0.05 step reward (discourage spam)
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: Dict[str, dict] = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
        task_type: str = "math",
        **kwargs,
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        if ground_truth is None:
            ground_truth = kwargs.get("create_kwargs", {}).get("ground_truth", "")
        task_type = kwargs.get("create_kwargs", {}).get("task_type", task_type)
        self._instance_dict[instance_id] = {
            "ground_truth": ground_truth,
            "task_type": task_type,
            "best_reward": 0.0,
            "best_answer": None,
            "best_confidence": None,
            "submission_count": 0,
            "first_correct": False,
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        answer = parameters.get("answer", "")
        confidence = parameters.get("confidence", 0.5)
        if not isinstance(answer, str):
            answer = str(answer)
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.5

        state = self._instance_dict[instance_id]
        state["submission_count"] += 1

        is_correct = _check_answer(answer, state["ground_truth"], state["task_type"])
        current_reward = 1.0 if is_correct else 0.0

        # Step reward: encourage improvement, discourage spamming
        if current_reward > state["best_reward"]:
            if state["submission_count"] == 1 and is_correct:
                step_reward = 0.3  # first-try correct
            elif is_correct and not state["first_correct"]:
                step_reward = 0.2  # corrected after revision
            else:
                step_reward = 0.1  # general improvement
            state["best_reward"] = current_reward
            state["best_answer"] = answer
            state["best_confidence"] = confidence
            if is_correct:
                state["first_correct"] = True
        else:
            step_reward = -0.05  # no improvement

        feedback = "correct" if is_correct else "incorrect"
        return (
            ToolResponse(
                text=(
                    f"Your answer '{answer}' is {feedback}. "
                    f"Confidence received: {confidence:.2f}. "
                    f"You have submitted {state['submission_count']} time(s). "
                    f"{'You may revise and resubmit.' if not is_correct else 'This is the correct answer.'}"
                )
            ),
            step_reward,
            {
                "is_correct": is_correct,
                "confidence": confidence,
                "submission_count": state["submission_count"],
            },
        )

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return self._instance_dict[instance_id]["best_reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
