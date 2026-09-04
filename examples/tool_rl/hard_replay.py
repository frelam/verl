#!/usr/bin/env python3
"""Tiered hard-sample replay for tool_rl on the V1 trainer.

Two cooperating pieces, both living in the trainer driver process:

1. ``HardReplaySampler`` — a ``ReplayBuffer`` subclass registered via
   ``trainer.v1.sampler.custom_sampler.{path,name}``.  It keeps the DAPO
   group-filtering semantics of the built-in buffer (uniform-reward groups
   are evicted and refilled with fresh prompts), and additionally tracks the
   *pass rate* (fraction of trajectories with metric > ``pass_threshold``)
   of every finished group.  Groups below ``medium_threshold`` are exported
   to an in-process pool for later replay.

2. ``HardReplayPool`` — the pool.  Entries are tiered by pass rate and
   become due for replay on a per-tier global-step interval:

   - "medium" tier (``zero_threshold`` < pass rate < ``medium_threshold``):
     mixed groups with gradient signal — trained on normally AND replayed
     every ``medium_interval`` steps (default 10).
   - "hard" tier (pass rate <= ``zero_threshold``, incl. all-zero groups):
     no gradient signal — filtered out AND replayed every ``hard_interval``
     steps (default 20), so the model gets another shot once it improves.
   - pass rate >= ``medium_threshold`` (incl. all-one): never pooled.

   The dataset (see ``tool_rl_dataset.py``) draws due entries from the pool,
   so pooled prompts re-enter rollout as brand-new groups.  When a replayed
   group finishes, its pass rate is recomputed: still below the medium
   threshold -> re-armed with the NEW tier (hard can become medium);
   otherwise -> graduated out of the pool.  ``max_replays`` > 0 caps how
   many times one sample may be replayed (0 = no cap).

Both sides run in the trainer driver process, so a module-level singleton
is sufficient — but ONLY with ``data.dataloader_num_workers=0``: dataloader
worker processes fork before the pool is populated and would see a stale
copy.

Why the dataset side replays (not the sampler): ``ReplayBuffer.sample()``
only selects terminal groups whose trajectories are already generated, so
re-selecting an all-zero group would just retrain on zero-advantage
trajectories.  A true replay must re-dispatch the prompt through
``generate_sequences``, and the only prompt channel available is the
training dataloader.

Config example (the framework does NOT inject ``algorithm.filter_groups``
into custom samplers, so filtering is owned by this class and its knobs are
passed via ``sampler_kwargs``)::

    trainer.v1.sampler.custom_sampler.path=examples/tool_rl/hard_replay.py
    trainer.v1.sampler.custom_sampler.name=HardReplaySampler
    # the `+` prefix is required: the yaml default `sampler_kwargs: {}` is an
    # empty dict, which OmegaConf marks as struct and rejects key merges
    +trainer.v1.sampler.sampler_kwargs={filter_metric: score, train_batch_size: 256, medium_interval: 10,
        hard_interval: 20}
    data.dataloader_num_workers=0
    # dataset side env: TOOL_RL_HARD_REPLAY=1, TOOL_RL_REPLAY_RATIO=1.0
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import transfer_queue as tq
from omegaconf import DictConfig, OmegaConf

from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# extra_info key tagging a dataset row as a replay of a pooled hard sample.
HARD_REPLAY_TAG = "hard_replay"

TIER_HARD = "hard"
TIER_MEDIUM = "medium"

# Fields read back from one trajectory of an exported group; enough for the
# dataset to rebuild an equivalent row (see ``entry_to_row_dict``).
ROW_FIELDS = ["raw_prompt", "tools", "data_source", "reward_model", "extra_info"]


@dataclass
class PoolEntry:
    """One pooled sample awaiting (re-)replay."""

    key: str
    row: dict[str, Any]
    tier: str  # TIER_HARD | TIER_MEDIUM
    pass_rate: float  # pass rate of the group that (last) produced this entry
    last_replay_step: int  # global step when exported / last dispatched for replay
    state: str = "available"  # "available" | "inflight" (dispatched for re-rollout)
    replay_count: int = 0
    inflight_passes: int = 0


class HardReplayPool:
    """In-process pool of low-pass-rate groups awaiting replay.

    Written by ``HardReplaySampler`` (add / resolve_replay / current_step)
    and read by the training dataset (``maybe_take``).  An entry is
    ``available`` until the dataset dispatches it for re-rollout
    (``inflight``), and is resolved once the replayed group finishes: still
    below ``medium_threshold`` -> re-armed with its new tier (or dropped
    after ``max_replays`` attempts when > 0); otherwise -> graduated out of
    the pool.
    """

    def __init__(
        self,
        medium_interval: int = 10,
        hard_interval: int = 20,
        medium_threshold: float = 0.5,
        zero_threshold: float = 0.01,
        max_replays: int = 0,
        recycle_after_passes: int = 8,
        max_per_step: int = 0,
    ):
        self.medium_interval = medium_interval
        self.hard_interval = hard_interval
        self.medium_threshold = medium_threshold
        self.zero_threshold = zero_threshold
        self.max_replays = max_replays  # 0 = replay forever
        self.recycle_after_passes = recycle_after_passes
        self.max_per_step = max_per_step  # 0 = unlimited; hard cap on replays dispatched per training step
        self.current_step = 0  # maintained by the sampler via sample(global_steps)
        self.entries: dict[str, PoolEntry] = {}
        self._dispatched_step = -1  # step the dispatch counter below belongs to
        self._dispatched_count = 0

    def configure(self, **kwargs) -> None:
        """Update config knobs in place (used when the singleton pre-exists)."""
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ValueError(f"Unknown HardReplayPool config knob: {key!r}")
            setattr(self, key, value)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def num_available(self) -> int:
        return sum(1 for e in self.entries.values() if e.state == "available")

    def tier_for(self, pass_rate: float) -> str | None:
        """Tier a group belongs to, or None when it should not be pooled."""
        if pass_rate <= self.zero_threshold:
            return TIER_HARD
        if pass_rate < self.medium_threshold:
            return TIER_MEDIUM
        return None

    def interval_of(self, tier: str) -> int:
        return self.hard_interval if tier == TIER_HARD else self.medium_interval

    def add(self, key: str, row: dict[str, Any], pass_rate: float) -> bool:
        """Pool a fresh sample; returns False when the key is already pooled (tier updated)."""
        tier = self.tier_for(pass_rate)
        if tier is None:
            return False
        existing = self.entries.get(key)
        if existing is not None:
            # Same prompt observed again (dataset epochs): keep the original
            # row but re-tier from the latest pass rate.
            existing.tier = tier
            existing.pass_rate = pass_rate
            return False
        self.entries[key] = PoolEntry(
            key=key,
            row=copy.deepcopy(row),
            tier=tier,
            pass_rate=pass_rate,
            last_replay_step=self.current_step,
        )
        logger.info(
            "[tool_rl] hard-replay: pooled %s tier=%s pass_rate=%.3f (pool=%d)", key, tier, pass_rate, len(self)
        )
        return True

    def maybe_take(self, rng, ratio: float) -> PoolEntry | None:
        """With probability ``ratio``, pick a due available entry and mark it inflight.

        An entry is due once ``current_step - last_replay_step`` reaches its
        tier interval, so e.g. medium entries replay every ~medium_interval
        steps and hard entries every ~hard_interval steps.  At most
        ``max_per_step`` entries are dispatched within one training step
        (0 = unlimited), capping the replay fraction of each batch.
        """
        if ratio <= 0.0 or rng.random() >= ratio:
            return None
        if self.current_step != self._dispatched_step:
            self._dispatched_step = self.current_step
            self._dispatched_count = 0
        if self.max_per_step and self._dispatched_count >= self.max_per_step:
            return None
        due = [
            e
            for e in self.entries.values()
            if e.state == "available" and self.current_step - e.last_replay_step >= self.interval_of(e.tier)
        ]
        if not due:
            return None
        entry = rng.choice(due)
        entry.state = "inflight"
        entry.inflight_passes = 0
        entry.last_replay_step = self.current_step
        self._dispatched_count += 1
        return entry

    def resolve_replay(self, key: str, pass_rate: float) -> None:
        """Resolve an inflight entry from the pass rate of its replayed group."""
        entry = self.entries.get(key)
        if entry is None:
            return
        tier = self.tier_for(pass_rate)
        if tier is None:
            # Improved past the medium threshold (or solved): graduate.
            del self.entries[key]
            logger.info(
                "[tool_rl] hard-replay: graduated %s pass_rate=%.3f after %d replays (pool=%d)",
                key,
                pass_rate,
                entry.replay_count,
                len(self.entries),
            )
            return
        entry.replay_count += 1
        entry.tier = tier
        entry.pass_rate = pass_rate
        if self.max_replays and entry.replay_count >= self.max_replays:
            del self.entries[key]
            logger.info(
                "[tool_rl] hard-replay: gave up on %s after %d replays (pool=%d)",
                key,
                entry.replay_count,
                len(self.entries),
            )
        else:
            entry.state = "available"
            entry.inflight_passes = 0

    def end_of_pass(self) -> None:
        """Recycle inflight entries whose replayed group never showed up finished.

        Safety valve for rollout-failure leaks: a replayed prompt whose group
        dies without reaching ``finished`` would otherwise stay inflight
        forever and never be replayed again.
        """
        for entry in self.entries.values():
            if entry.state != "inflight":
                continue
            entry.inflight_passes += 1
            if entry.inflight_passes > self.recycle_after_passes:
                entry.state = "available"
                entry.inflight_passes = 0


_POOL: HardReplayPool | None = None


def get_hard_pool(**kwargs) -> HardReplayPool:
    """Process-local pool singleton; ``kwargs`` configure it on every call."""
    global _POOL
    if _POOL is None:
        _POOL = HardReplayPool(**kwargs)
    elif kwargs:
        _POOL.configure(**kwargs)
    return _POOL


def reset_hard_pool() -> None:
    """Drop the singleton (test isolation)."""
    global _POOL
    _POOL = None


def _unwrap(value):
    """Normalise a TransferQueue non-tensor field element to plain Python objects."""
    value = getattr(value, "data", value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return value


def _tq_fetch_rows(partition_id: str, keys: list[str], fields: list[str]) -> list[dict[str, Any]]:
    """Fetch ``fields`` for ``keys`` and return one plain dict per key."""
    if not keys:
        return []
    data = tq.kv_batch_get(keys=keys, partition_id=partition_id, select_fields=fields)
    columns = {f: list(data[f]) for f in fields}
    return [{f: _unwrap(columns[f][i]) for f in fields} for i in range(len(keys))]


def _row_key(row: dict[str, Any]) -> str:
    """Stable dedup key for a pooled sample: task_id when present, else a prompt hash."""
    extra_info = row.get("extra_info") or {}
    task_id = extra_info.get("task_id")
    if task_id:
        return str(task_id)
    digest = hashlib.sha1(json.dumps(row.get("raw_prompt"), sort_keys=True, default=str).encode()).hexdigest()[:16]
    return f"hash:{digest}"


def entry_to_row_dict(entry: PoolEntry) -> dict:
    """Rebuild a dataset row from a pool entry, mirroring ``RLHFDataset.__getitem__``.

    The stored ``raw_prompt`` is returned as-is (it already contains any hint
    variant drawn in its original epoch), and the row is tagged so the
    sampler can resolve this entry once the replayed group finishes.
    """
    row = copy.deepcopy(entry.row)
    extra_info = dict(row.get("extra_info") or {})
    extra_info[HARD_REPLAY_TAG] = entry.key
    row["extra_info"] = extra_info
    row["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)
    row["index"] = extra_info.get("index", 0)
    row["tools_kwargs"] = extra_info.get("tools_kwargs", {})
    row["interaction_kwargs"] = extra_info.get("interaction_kwargs", {})
    return row


class HardReplaySampler(ReplayBuffer):
    """Sync ReplayBuffer with DAPO filtering plus pass-rate-tiered replay export.

    The framework passes custom samplers only the base constructor kwargs
    (filtering semantics are owned by the custom class), so all knobs come in
    through ``sampler_kwargs``:

    - ``filter_metric`` (str, required): reward metric used for uniform-group
      classification AND pass-rate computation, read from
      ``extra_fields.reward_extra_info``.
    - ``train_batch_size`` (int, required): prompts per training batch, used
      to bound Sync DAPO in-flight generation.
    - ``gen_batch_size`` (int, default 1) / ``max_inflight_gen_batches`` (int,
      default 1): Sync DAPO refill granularity / concurrency bound.
    - ``pass_threshold`` (float, default 0.0): a trajectory counts as passed
      when its metric exceeds this value.
    - ``medium_threshold`` (float, default 0.5) / ``zero_threshold`` (float,
      default 0.01): pass-rate band edges; groups below ``medium_threshold``
      are pooled, tier "hard" at or below ``zero_threshold``.
    - ``medium_interval`` (int, default 10) / ``hard_interval`` (int,
      default 20): global-step interval between replays of each tier.
    - ``max_replays`` (int, default 0): give up on a pooled sample after this
      many replays; 0 replays forever.
    - ``recycle_after_passes`` (int, default 8): re-arm an inflight entry
      whose replayed group never finishes (rollout-failure safety valve).
    - ``max_replay_fraction`` (float, default 0.2): hard cap on replays per
      training step, as a fraction of ``train_batch_size`` (at least 1); 0
      means unlimited.
    """

    def __init__(
        self,
        trainer_mode: str,
        trainer_config: DictConfig,
        max_off_policy_threshold: int,
        max_off_policy_strategy: str,
        sampler_kwargs,
        refill_fn=None,
        poll_interval: float = 2.0,
    ):
        if isinstance(sampler_kwargs, DictConfig):
            kwargs = dict(OmegaConf.to_container(sampler_kwargs, resolve=True) or {})
        else:
            kwargs = dict(sampler_kwargs or {})
        filter_metric = kwargs.pop("filter_metric", None)
        train_batch_size = kwargs.pop("train_batch_size", None)
        gen_batch_size = int(kwargs.pop("gen_batch_size", 1))
        max_inflight_gen_batches = int(kwargs.pop("max_inflight_gen_batches", 1))
        self.pass_threshold = float(kwargs.pop("pass_threshold", 0.0))
        max_replay_fraction = float(kwargs.pop("max_replay_fraction", 0.2))
        max_per_step = 0
        if max_replay_fraction > 0.0 and train_batch_size is not None:
            max_per_step = max(1, int(int(train_batch_size) * max_replay_fraction))
        pool_config = {
            "medium_interval": int(kwargs.pop("medium_interval", 10)),
            "hard_interval": int(kwargs.pop("hard_interval", 20)),
            "medium_threshold": float(kwargs.pop("medium_threshold", 0.5)),
            "zero_threshold": float(kwargs.pop("zero_threshold", 0.01)),
            "max_replays": int(kwargs.pop("max_replays", 0)),
            "recycle_after_passes": int(kwargs.pop("recycle_after_passes", 8)),
            "max_per_step": max_per_step,
        }
        if trainer_mode != "sync":
            raise ValueError(f"HardReplaySampler only supports the sync trainer mode, got {trainer_mode!r}.")
        if filter_metric is not None and train_batch_size is None:
            raise ValueError(
                "HardReplaySampler requires sampler_kwargs.train_batch_size when filter_metric is set "
                "(the framework does not inject data.train_batch_size into custom samplers)."
            )
        super().__init__(
            trainer_mode=trainer_mode,
            trainer_config=trainer_config,
            max_off_policy_threshold=max_off_policy_threshold,
            max_off_policy_strategy=max_off_policy_strategy,
            sampler_kwargs=kwargs,
            poll_interval=poll_interval,
            refill_fn=refill_fn,
            filter_groups_metric=filter_metric,
            train_batch_size=train_batch_size,
            gen_batch_size=gen_batch_size,
            max_inflight_gen_batches=max_inflight_gen_batches,
        )
        self.pool = get_hard_pool(**pool_config)
        logger.info(
            "[tool_rl] HardReplaySampler: filter_metric=%s train_batch_size=%s pass_threshold=%.3f pool=%s",
            filter_metric,
            train_batch_size,
            self.pass_threshold,
            pool_config,
        )

    def sample(self, global_steps: int, partition_id: str, batch_size: int):
        """Track the training step so the pool can schedule replays by interval."""
        if partition_id != "val":
            self.pool.current_step = global_steps
        return super().sample(global_steps=global_steps, partition_id=partition_id, batch_size=batch_size)

    def _dapo_filtered_keys(self, partition_id: str):
        """Classify uniform-reward groups (base semantics) and sync the replay pool.

        Reimplements the base method's classification loop so the per-trajectory
        metric values it fetches also feed pass-rate computation in the SAME
        TransferQueue read (the base method only exposes the uniform / mixed
        verdict, not the values).  Keep the classification semantics identical:
        groups with a single trajectory count as mixed, missing metrics raise.
        """
        if partition_id == "val" or self.filter_groups_metric is None:
            return super()._dapo_filtered_keys(partition_id)

        finished_uids = self.finished_keys[partition_id]
        cache = self._dapo_classification_cache[partition_id]
        for uid in cache.keys() - finished_uids:
            del cache[uid]

        new_finished_uids = finished_uids - cache.keys()
        trajectory_keys = [key for key in self.partitions[partition_id] if key.split("_")[0] in new_finished_uids]
        metrics_by_uid: dict[str, list[float]] = defaultdict(list)
        extra_info_by_uid: dict[str, dict] = {}
        missing_metric_uids = new_finished_uids - {key.split("_")[0] for key in trajectory_keys}

        if trajectory_keys:
            data = tq.kv_batch_get(
                keys=trajectory_keys,
                partition_id=partition_id,
                select_fields=["extra_fields", "extra_info"],
            )
            extra_fields_list = list(data["extra_fields"])
            extra_info_list = list(data["extra_info"])
        else:
            extra_fields_list, extra_info_list = [], []

        for key, extra_fields, extra_info in zip(trajectory_keys, extra_fields_list, extra_info_list, strict=True):
            uid = key.split("_")[0]
            if uid not in extra_info_by_uid:
                extra_info_by_uid[uid] = _unwrap(extra_info) or {}
            extra_fields = getattr(extra_fields, "data", extra_fields)
            reward_extra_info = extra_fields.get("reward_extra_info", {}) if isinstance(extra_fields, dict) else {}
            if self.filter_groups_metric not in reward_extra_info:
                missing_metric_uids.add(uid)
            else:
                metrics_by_uid[uid].append(float(reward_extra_info[self.filter_groups_metric]))

        if missing_metric_uids:
            raise RuntimeError(
                f"Finished groups are missing DAPO metric {self.filter_groups_metric!r}: "
                f"{sorted(missing_metric_uids)[:5]}"
            )

        exports: list[tuple[str, float]] = []
        for uid in new_finished_uids:
            values = metrics_by_uid[uid]
            cache[uid] = float(values[0]) if len(values) > 1 and float(np.std(values)) == 0.0 else None
            pass_rate = sum(1 for v in values if v > self.pass_threshold) / len(values)
            replay_key = (extra_info_by_uid.get(uid) or {}).get(HARD_REPLAY_TAG)
            if replay_key is not None:
                # A replayed prompt whose group just finished.
                self.pool.resolve_replay(replay_key, pass_rate)
            elif self.pool.tier_for(pass_rate) is not None:
                exports.append((uid, pass_rate))

        if exports:
            first_key_by_uid = {
                uid: next(key for key in trajectory_keys if key.split("_")[0] == uid) for uid, _ in exports
            }
            rows = _tq_fetch_rows(partition_id, [first_key_by_uid[uid] for uid, _ in exports], ROW_FIELDS)
            for (_, pass_rate), row in zip(exports, rows, strict=True):
                self.pool.add(_row_key(row), row, pass_rate)

        self.pool.end_of_pass()

        filtered_rewards = {uid: reward for uid, reward in cache.items() if reward is not None}
        return set(filtered_rewards), Counter(filtered_rewards.values())
