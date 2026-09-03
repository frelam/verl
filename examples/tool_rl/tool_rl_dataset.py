#!/usr/bin/env python3
"""RLHFDataset subclass with randomized system-prompt hint injection.

Registered via verl's custom dataset mechanism (no verl core changes)::

    data.custom_cls.path=examples/tool_rl/tool_rl_dataset.py
    data.custom_cls.name=ToolRLHintDataset

Behaviour is controlled by ``TOOL_RL_HINT_*`` env vars (see
``hint_injection.py``).  With the default ``TOOL_RL_HINT_MODE=off`` this
class is a transparent pass-through.

Per-epoch re-randomisation & GRPO group consistency
---------------------------------------------------
verl calls ``__getitem__`` once per sample per epoch and only *then*
repeats the batch ``rollout.n`` times, so a variant drawn here is shared
by all rollouts of a GRPO group but re-drawn every epoch.

The dataset dataframe lives in Arrow and every ``__getitem__`` returns
freshly materialised Python objects, so mutating the returned row is safe
and never corrupts the stored data.

Validation determinism
----------------------
A dataset whose files all look like validation/test files (basename
contains ``val`` or ``test``) is forced into ``fixed`` mode so val metrics
stay comparable across eval runs.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

from verl.utils.dataset.rl_dataset import RLHFDataset

from examples.tool_rl.hint_injection import apply_hint_to_row, hint_config_from_env

logger = logging.getLogger(__name__)


class ToolRLHintDataset(RLHFDataset):
    """RLHFDataset that injects a random hint variant into the system prompt."""

    def __init__(self, data_files, tokenizer, processor, config, max_samples: int = -1):
        super().__init__(
            data_files=data_files,
            tokenizer=tokenizer,
            processor=processor,
            config=config,
            max_samples=max_samples,
        )
        cfg = hint_config_from_env()
        self._hint_mode = cfg["mode"]
        self._hint_empty_prob = cfg["empty_prob"]
        self._hint_seed = cfg["seed"]
        # Distinct RNG streams per process (dataloader workers fork after init).
        # random.Random only accepts None/int/float/str/bytes seeds, so fold
        # the pid into a string instead of passing a (seed, pid) tuple.
        self._hint_rng = random.Random(f"{self._hint_seed}:{os.getpid()}")

        if self._hint_mode == "random" and self._looks_like_val_files():
            self._hint_mode = "fixed"

        if self._hint_mode != "off":
            logger.info(
                "[tool_rl] Hint injection enabled: mode=%s empty_prob=%.2f seed=%d files=%s",
                self._hint_mode,
                self._hint_empty_prob,
                self._hint_seed,
                self.data_files,
            )

    def _looks_like_val_files(self) -> bool:
        names = [Path(f).stem.lower() for f in self.data_files]
        return bool(names) and all("val" in n or "test" in n for n in names)

    def __getitem__(self, item):
        row_dict = super().__getitem__(item)
        return apply_hint_to_row(
            row_dict,
            mode=self._hint_mode,
            rng=self._hint_rng,
            seed=self._hint_seed,
            empty_prob=self._hint_empty_prob,
        )
