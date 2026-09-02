# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Single-turn tool-use agent loop for verl.

Ported from slime ``examples/tool_rl/generate.py`` (single-turn rollout part).
Compared with verl's built-in ``single_turn_agent``:

1. **Per-sample tools** — tool schemas travel in the dataset's ``tools``
   column and are passed to ``apply_chat_template(messages, tools=...)`` so
   each sample's system prompt declares exactly its own tool set.
2. **Optional failed-tool-call masking** — when
   ``TOOL_RL_MASK_FAILED_CALLS=1``, tokens belonging to *incorrect* tool call
   blocks (undeclared function / undeclared params / wrong value types, judged
   rule-based against the sample's tool schemas) get ``response_mask = 0`` so
   they are excluded from the policy gradient loss.

   This is the static (unconditional) variant of slime's tool-call masking.
   Slime's advantage-conditioned variant is implemented via slime's custom
   TIS hook in the loss path, which has no equivalent extension point in
   verl; it is therefore not migrated.

Registered as ``tool_rl_agent`` via ``agent_loop_config.yaml``::

    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_rl_agent
    actor_rollout_ref.rollout.agent.agent_loop_config_path=examples/tool_rl/agent_loop_config.yaml
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

# Make sibling imports work regardless of how this file is loaded (the module
# may be imported as ``examples.tool_rl.tool_rl_agent_loop`` via hydra).
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.tool_rl.reward.verifier import get_incorrect_tool_call_spans  # noqa: E402

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _to_dict_list(value: Any) -> list[dict[str, Any]]:
    """Normalise a (possibly numpy) list of tool-schema dicts from parquet."""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return []
    result = []
    for item in value:
        if hasattr(item, "tolist"):
            item = item.tolist()
        if isinstance(item, dict):
            result.append(dict(item))
    return result


@register("tool_rl_agent")
class ToolRLAgentLoop(AgentLoopBase):
    """Single-turn tool-use agent loop with per-sample tool schemas."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        # Static failed-tool-call masking (see module docstring).
        self.mask_failed_tool_calls = os.environ.get("TOOL_RL_MASK_FAILED_CALLS", "0") == "1"

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput:
        # priority may arrive as np.int64 from non_tensor_batch; normalize to Python int.
        priority = int(priority)
        messages = list(kwargs["raw_prompt"])

        # Per-sample tool schemas from the dataset's ``tools`` column.
        sample_tools = _to_dict_list(kwargs.get("tools"))

        # 1. apply chat template (with this sample's tools) and tokenize
        prompt_ids = await self.apply_chat_template(messages, tools=sample_tools or None)

        # 2. generate sequences (single turn; stops at EOS <|im_end|>)
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            request_id = f"det-{priority}" if getattr(self.rollout_config, "full_determinism", False) else uuid4().hex
            output: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                priority=priority,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

        response_ids = output.token_ids
        response_logprobs = output.log_probs

        # 3. response mask: 1 everywhere; optionally 0 on incorrect tool call tokens
        if self.mask_failed_tool_calls and response_ids:
            response_mask = self._build_tool_aware_mask(response_ids, sample_tools)
        else:
            response_mask = [1] * len(response_ids)

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )

        # keeping the schema consistent with tool_agent_loop
        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})

        return output

    def _build_tool_aware_mask(
        self,
        response_ids: list[int],
        available_tools: list[dict[str, Any]],
    ) -> list[int]:
        """Binary loss mask that zeros out incorrect tool call tokens.

        Char-level spans of incorrect tool call blocks (rule-based verdict
        against the sample's tool schemas) are mapped to token offsets via
        the tokenizer's offset mapping. On any tokeniser round-trip mismatch
        the mask falls back to all-ones (no masking).
        """
        response_len = len(response_ids)
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)

        spans = get_incorrect_tool_call_spans(response_text, available_tools)
        if not spans:
            return [1] * response_len

        enc = self.tokenizer(response_text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc["offset_mapping"]
        input_ids = enc["input_ids"]

        if len(input_ids) != response_len:
            logger.warning(
                "[tool_rl] Token count mismatch (tokenizer=%d, engine=%d) — skipping mask",
                len(input_ids),
                response_len,
            )
            return [1] * response_len

        mask = [1] * response_len
        for start, end in spans:
            for i, (s, e) in enumerate(offsets):
                if s < end and e > start:
                    mask[i] = 0

        n_masked = response_len - sum(mask)
        logger.info(
            "[tool_rl] Masked %d/%d tokens (%d incorrect tool call block(s))",
            n_masked,
            response_len,
            len(spans),
        )
        return mask
