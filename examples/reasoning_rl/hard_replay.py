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
"""Tiered hard-sample replay for reasoning_rl — reuses tool_rl's implementation.

The tool_rl ``hard_replay.py`` module is fully domain-agnostic (all knobs come
from ``sampler_kwargs``; no env vars are read inside it), so this file is a
thin re-export that lets the trainer mount it under the reasoning_rl path
while BOTH sides share the same underlying module in ``sys.modules`` — which
is what keeps the ``HardReplayPool`` singleton shared between the sampler
(trainer driver) and the dataset (``reasoning_rl_dataset.py``)::

    trainer.v1.sampler.custom_sampler.path=examples/reasoning_rl/hard_replay.py
    trainer.v1.sampler.custom_sampler.name=HardReplaySampler

See examples/tool_rl/hard_replay.py for the full design doc. Per DESIGN.md
section 8 the reasoning_rl side is simpler than tool_rl: no hint re-draw —
replayed rows carry their original ``raw_prompt + ground_truth`` unchanged.
"""

from examples.tool_rl.hard_replay import (
    HARD_REPLAY_TAG,
    TIER_HARD,
    TIER_MEDIUM,
    HardReplayPool,
    HardReplaySampler,
    PoolEntry,
    entry_to_row_dict,
    get_hard_pool,
    reset_hard_pool,
)

__all__ = [
    "HARD_REPLAY_TAG",
    "TIER_HARD",
    "TIER_MEDIUM",
    "HardReplayPool",
    "HardReplaySampler",
    "PoolEntry",
    "entry_to_row_dict",
    "get_hard_pool",
    "reset_hard_pool",
]
