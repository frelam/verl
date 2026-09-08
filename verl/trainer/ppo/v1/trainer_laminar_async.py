# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
import logging
import os

import ray
import transfer_queue as tq
from omegaconf import DictConfig

from verl.trainer.ppo.utils import need_reward_model
from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer
from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync
from verl.utils.debug import marked_timer
from verl.workers.rollout.rollout_manager import LaminarRolloutManagerActor

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@register_trainer("laminar_async")
class PPOTrainerLaminarAsync(PPOTrainerSeparateAsync):
    """Laminar-style trajectory-level asynchronous PPO trainer (https://arxiv.org/abs/2510.12633).

    Differences from ``separate_async``:

    1. **Publish/pull weight service**: every step the trainer only *publishes* weights to
       the store (``CheckpointEngineManager.publish_weights``) and notifies the
       ``LaminarRolloutManager``; each rollout replica independently drains, pulls and
       rejoins. There is no global abort/rendezvous weight-sync point, so training never
       blocks on rollout weight updates (except the startup barrier in ``on_init_end``).
    2. **Trajectory-level staleness**: the replay buffer measures staleness from the
       per-trajectory ``min_global_steps`` version tag stamped by the generating server
       (see ``ReplayBuffer._staleness_span``), not from the prompt's dispatch step --
       with lazy per-replica pulls, the dispatch step says nothing about the version that
       actually produced the tokens.
    3. **In-flight trajectory migration (simplified dynamic repack)**: a draining replica
       whose in-flight work outlives ``drain_timeout_seconds`` has its requests aborted;
       callers resume on non-draining replicas with ``prompt + partial tokens`` (duplicate
       prefill, no KV-cache transfer) via ``FullyAsyncLLMServerClient``'s retry loop.
    4. **Trainer-GPU hybrid rollout engines are kept asleep**: all generation runs on the
       standalone rollout pool (Laminar's fully-decoupled rollout), so trainer GPUs never
       serve generation. The hybrid replicas are still launched (they share the actor
       worker processes) but are never registered in the load balancer.

    Partial-rollout resume across versions means a migrated trajectory may span two weight
    versions; its ``min_global_steps``/``max_global_steps`` tags record the span and the
    staleness drop policy keys off the oldest one. For off-policy correction, enable
    ``algorithm.rollout_correction`` (e.g. ``bypass_mode=true`` with ``rollout_is=token``)
    so each trajectory's own rollout log-probs act as the behavior policy.
    """

    def __init__(self, config: DictConfig):
        laminar_config = config.trainer.v1.laminar_async
        assert config.actor_rollout_ref.rollout.checkpoint_engine.backend == "mooncake_store", (
            f"laminar_async requires checkpoint_engine.backend='mooncake_store' (publish/pull weight "
            f"store), got {config.actor_rollout_ref.rollout.checkpoint_engine.backend!r}"
        )
        assert laminar_config.get("parameter_sync_step", 1) == 1, (
            "laminar_async publishes weights every trainer step; trainer.v1.laminar_async.parameter_sync_step must be 1"
        )
        assert config.actor_rollout_ref.rollout.nnodes > 0, "nnodes must be > 0 in laminar async training"
        assert config.actor_rollout_ref.rollout.n_gpus_per_node > 0, (
            "n_gpus_per_node must be > 0 in laminar async training"
        )
        max_inflight = laminar_config.get("max_inflight_prompts", None)
        if max_inflight is not None:
            assert max_inflight >= config.data.train_batch_size, (
                f"trainer.v1.laminar_async.max_inflight_prompts ({max_inflight}) must be >= "
                f"data.train_batch_size ({config.data.train_batch_size}), otherwise the replay "
                "buffer can never assemble a training batch and the trainer deadlocks"
            )
        rollout_corr_config = config.algorithm.get("rollout_correction", None)
        if rollout_corr_config and rollout_corr_config.get("bypass_mode", False):
            assert config.actor_rollout_ref.rollout.calculate_log_probs, (
                "algorithm.rollout_correction.bypass_mode requires "
                "actor_rollout_ref.rollout.calculate_log_probs=true so each trajectory's own "
                "rollout log-probs can serve as the behavior policy"
            )
        if need_reward_model(config):
            assert config.reward.reward_model.enable_resource_pool, (
                "Colocate reward model (reward.reward_model.enable_resource_pool=False) is not supported "
                "in laminar async mode, because the standalone rollout never pauses to free GPU memory. "
                "Use standalone mode (reward.reward_model.enable_resource_pool=True) instead."
            )

        # Bypass PPOTrainerSeparateAsync.__init__: its
        # ``train_batch_size == parameter_sync_step * ppo_mini_batch_size`` constraint is
        # Decoupled-PPO-specific and does not apply here.
        PPOTrainer.__init__(self, config)

    def _setup(self):
        super()._setup()

        laminar_config = self.config.trainer.v1.laminar_async
        self.laminar_manager = LaminarRolloutManagerActor.remote(
            replicas=self.standalone_server_manager.get_replicas(),
            load_balancer_handle=self.standalone_server_manager.global_load_balancer,
            poll_interval_seconds=laminar_config.get("poll_interval_seconds", 1.0),
            max_concurrent_pulls=laminar_config.get("max_concurrent_pulls", 4),
            drain_timeout_seconds=laminar_config.get("drain_timeout_seconds", 600.0),
        )
        # Fire-and-forget: the control loop runs for the whole training job. With
        # ``max_concurrency > 1`` on the actor, on_publish/wait_all_replicas/get_metrics
        # calls are served concurrently with the loop.
        self.laminar_manager.start.remote()

    # ------------------------------------------------------------------ hybrid replicas

    def add_replicas_to_balancer(self):
        # Trainer-GPU hybrid rollout engines stay asleep for the whole run (base _setup
        # already slept them); generation runs exclusively on the standalone pool.
        return

    def remove_replicas_from_balancer(self):
        return

    # ------------------------------------------------------------------ hooks

    def on_init_end(self):
        # Initial weights go through the same publish/pull path as every other version.
        # This is the only point where training waits for rollout weight updates.
        self.standalone_checkpoint_manager.publish_weights(self.global_steps)
        ray.get(self.laminar_manager.on_publish.remote(self.global_steps))
        init_timeout = self.config.trainer.v1.laminar_async.get("init_pull_timeout_seconds", 3600.0)
        ray.get(self.laminar_manager.wait_all_replicas.remote(self.global_steps, init_timeout))
        logger.info(f"All standalone rollout replicas pulled initial weights (v{self.global_steps})")

    def on_step_end(self):
        with marked_timer("publish_weights", self.timing_raw, color="red"):
            self._pending_sync_metrics = self.standalone_checkpoint_manager.publish_weights(self.global_steps)
        # Fire-and-forget: per-replica drain/pull/rejoin happens asynchronously.
        self.laminar_manager.on_publish.remote(self.global_steps)
        manager_metrics = ray.get(self.laminar_manager.get_metrics.remote())
        self._pending_sync_metrics.update({f"laminar/{key}": value for key, value in manager_metrics.items()})

    def on_train_end(self):
        ray.get(self.laminar_manager.stop.remote())

    # ------------------------------------------------------------------ training pipeline

    def _compute_old_log_prob(self, batch, metrics):
        # parameter_sync_step == 1 means the Decoupled-PPO save/restore dance in
        # PPOTrainerSeparateAsync._compute_old_log_prob is pure overhead here; the base
        # implementation already handles both bypass and recompute modes by config.
        return PPOTrainer._compute_old_log_prob(self, batch, metrics)

    def _add_batch_to_generate(self):
        """Dispatch one training batch, unless the in-flight prompt cap is reached.

        Bounding in-flight prompts bounds version skew (a prompt dispatched now is
        generated by whatever version its replica happens to serve) and prevents fast
        short samples from flooding the buffer while long samples are still running.
        """
        max_inflight = self.config.trainer.v1.laminar_async.get("max_inflight_prompts", None)
        if max_inflight is not None and self._inflight_prompt_count() >= max_inflight:
            logger.info(f"In-flight prompt cap reached ({max_inflight}), skipping prompt dispatch this step")
            return
        super()._add_batch_to_generate()

    @staticmethod
    def _inflight_prompt_count() -> int:
        data = tq.kv_list("train") or {}
        items = data.get("train", {})
        return sum(
            1 for tag in items.values() if tag.get("is_prompt", False) and tag.get("status") in ("pending", "running")
        )
