#!/usr/bin/env python3
"""RLHFDataset subclass with hard-sample replay re-entry for reasoning_rl.

Registered via verl's custom dataset mechanism (no verl core changes)::

    data.custom_cls.path=examples/reasoning_rl/reasoning_rl_dataset.py
    data.custom_cls.name=ReasoningRLDataset

Hard-sample replay
------------------
Each ``__getitem__`` may return a pooled low-pass-rate prompt that is DUE for
replay (the pool decides due-ness from per-tier step intervals;
``REASONING_RL_REPLAY_RATIO``, default 1.0, is just a throttle on how eagerly
due entries are drawn) instead of the indexed sample, so it re-enters rollout
as a fresh group (see ``hard_replay.py`` and examples/tool_rl/hard_replay.py).

The pool is filled by ``HardReplaySampler`` in the trainer driver process, so
this only works with ``data.dataloader_num_workers=0``. Replayed rows keep
their original ``raw_prompt + ground_truth`` untouched (DESIGN.md section 8 —
simpler than tool_rl: no hint re-draw). Never applied to val files, which
would corrupt val metrics.

Env vars:
    REASONING_RL_HARD_REPLAY   "1"/"on" to enable (default off)
    REASONING_RL_REPLAY_RATIO  per-fetch draw throttle, default 1.0
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

from examples.reasoning_rl.hard_replay import entry_to_row_dict, get_hard_pool
from verl.utils.dataset.rl_dataset import RLHFDataset

logger = logging.getLogger(__name__)


class ReasoningRLDataset(RLHFDataset):
    """RLHFDataset that replays due pooled hard samples on the training split."""

    def __init__(self, data_files, tokenizer, processor, config, max_samples: int = -1):
        super().__init__(
            data_files=data_files,
            tokenizer=tokenizer,
            processor=processor,
            config=config,
            max_samples=max_samples,
        )
        self._replay_ratio = float(os.environ.get("REASONING_RL_REPLAY_RATIO", "1.0"))
        self._replay_enabled = (
            os.environ.get("REASONING_RL_HARD_REPLAY", "off") != "off" and not self._looks_like_val_files()
        )
        # Distinct RNG streams per process (dataloader workers fork after init).
        self._replay_rng = random.Random(f"reasoning-rl-replay:{os.getpid()}")
        if self._replay_enabled:
            logger.info(
                "[reasoning_rl] Hard replay enabled: ratio=%.2f files=%s",
                self._replay_ratio,
                self.data_files,
            )

    def _looks_like_val_files(self) -> bool:
        names = [Path(f).stem.lower() for f in self.data_files]
        return bool(names) and all("val" in n or "test" in n for n in names)

    def __getitem__(self, item):
        if self._replay_enabled:
            entry = get_hard_pool().maybe_take(self._replay_rng, self._replay_ratio)
            if entry is not None:
                return entry_to_row_dict(entry)
        return super().__getitem__(item)
