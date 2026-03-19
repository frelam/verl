# Copyright 2025 Meituan Ltd. and/or its affiliates
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

import asyncio
import multiprocessing
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pprint import pformat

import numpy as np
import ray
import torch

from verl.experimental.fully_async_policy.detach_utils import (
    RolloutSample,
    ValidateMetrics,
    prepare_single_generation_data,
    safe_create_task,
)
from verl.experimental.fully_async_policy.message_queue import MessageQueueClient
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.profiler import marked_timer
from verl.utils.tracking import ValidationGenerationsLogger


@ray.remote(num_cpus=10, max_concurrency=100)
class FullyAsyncRollouter(SeparateRayPPOTrainer):
    """
    Asynchronous sample generator, responsible for continuously generating training samples
    and putting them into MessageQueue
    Based on the mature implementation improvements of OneStepOffRayTrainer
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        device_name=None,
    ):
        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine

        assert not self.hybrid_engine
        assert self.config.data.train_batch_size == 0, "train_batch_size must be zero"
        assert self.config.data.gen_batch_size == 1, "gen_batch_size must be one"
        assert self.config.async_training.staleness_threshold >= 0, "staleness_threshold must larger than 0"
        assert self.config.async_training.trigger_parameter_sync_step >= 1, (
            "trigger_parameter_sync_step must larger or equal than 1"
        )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = False

        self.use_rm = False

        self.use_critic = False
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        self.ref_in_actor = False
        self.kl_ctrl_in_reward = False

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        # ==================== fully async config ====================

        print("[FullyAsyncRollouter] Creating datasets...")
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        self._validate_config()
        if self.config.async_training.use_trainer_do_validate:
            rollout_gpus = config.rollout.nnodes * config.rollout.n_gpus_per_node
            train_gpus = config.trainer.nnodes * config.trainer.n_gpus_per_node
            total_gpus = rollout_gpus + train_gpus
            print(f"[FullyAsyncRollouter] split before val_dataset total len: {len(val_dataset)}")
            split_dataset = val_dataset.split(total_gpus)
            rollout_val_dataset0 = split_dataset[:rollout_gpus]
            from torch.utils.data import ConcatDataset

            val_dataset = ConcatDataset(rollout_val_dataset0)
            print(f"[FullyAsyncRollouter] split after val_dataset total len: {len(val_dataset)}")
        print(f"[FullyAsyncRollouter] Rollouter _create_dataloader...\n{train_dataset}\n{val_dataset}")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        self.total_rollout_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.rollout.total_rollout_steps is not None:
            self.total_rollout_steps = min(self.config.rollout.total_rollout_steps, self.total_rollout_steps)
        print(f"[FullyAsyncRollouter] Total rollout steps: {self.total_rollout_steps}")
        self.total_train_steps = None

        # Rollouter parameter configuration
        self.message_queue_client = None

        # Worker groups: rollout_wg is same to actor_rollout_wg
        self.rollout_wg = None
        self.actor_rollout_wg = None
        self.async_rollout_manager = None

        # Config
        self.staleness_threshold: float = config.async_training.get("staleness_threshold", 1)
        self.staleness_rebalance_threshold: int = config.async_training.get("staleness_rebalance_threshold", 3)
        self.max_weight_versions: int = config.async_training.get("max_weight_versions", 5)
        self.require_batches = config.async_training.require_batches
        self.required_samples = config.actor_rollout_ref.actor.ppo_mini_batch_size * self.require_batches
        self.max_required_samples = None
        self.max_concurrent_samples = None
        self.max_queue_size = None

        self.total_generated_samples = 0
        self.staleness_samples = 0
        self.dropped_stale_samples = 0
        self.processed_sample_count = 0
        self.global_steps = 1
        self.idle_start_time = time.time()
        self.step_start_time = time.time()

        self.paused = False
        self.running = True

        self.dataloader_lock = asyncio.Lock()

        self.pending_queue = asyncio.Queue(maxsize=128)
        self.active_tasks = set()

        self.sample_staleness_tracker: "dict[str, dict]" = {}
        self.current_param_version = 0
        self.worker_sample_mapping: "dict[int, set[str]]" = defaultdict(set)
        self.worker_idle_status: "dict[int, bool]" = {}

        cpu_cores = multiprocessing.cpu_count()
        self.validate_executor = ThreadPoolExecutor(max_workers=cpu_cores)
        self.validate_task = None

    def _init_async_objects(self):
        self.condition = asyncio.Condition()
        self.lock = self.condition._lock
        num_workers = len(self.async_rollout_manager.server_handles) if self.async_rollout_manager else 0
        for worker_id in range(num_workers):
            self.worker_idle_status[worker_id] = True
            self.worker_sample_mapping[worker_id] = set()

    async def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        """Set message queue client"""
        async with self.lock:
            self.message_queue_client = message_queue_client

    async def set_max_required_samples(self):
        async with self.lock:
            self.max_required_samples = int(
                self.required_samples
                * (self.staleness_threshold + 1)
                * self.config.async_training.trigger_parameter_sync_step
            )
            self.total_train_steps = int(
                self.total_rollout_steps
                / (self.required_samples * self.config.async_training.trigger_parameter_sync_step)
            )

            self.max_concurrent_samples = len(self.async_rollout_manager.server_handles) * 16
            self.max_concurrent_samples = min(self.max_concurrent_samples, self.max_required_samples)
            self.max_queue_size = self.max_required_samples

            print(
                f"[FullyAsyncRollouter] required_samples : {self.required_samples} "
                f"max_required_samples: {self.max_required_samples} "
                f"max_queue_size: {self.max_queue_size} "
                f"total_train_steps: {self.total_train_steps} "
                f"total_rollout_steps: {self.total_rollout_steps} "
                f"max_concurrent_samples: {self.max_concurrent_samples} "
            )

    def get_rollout_wg(self):
        """Get rollout worker group"""
        return self.rollout_wg

    def get_replicas(self):
        """Get rollout worker group"""
        return self.async_rollout_manager.rollout_replicas

    def get_max_queue_size(self):
        return self.max_queue_size

    def get_total_train_steps(self):
        return self.total_train_steps

    async def reset_staleness(self, new_param_version: int = None):
        """
        Reset staleness samples after parameter update.
        Returns timing_raw dictionary for metrics.
        """
        async with self.lock:
            if new_param_version is not None:
                self.current_param_version = new_param_version
            else:
                self.current_param_version += 1

            self.paused = False
            self.condition.notify_all()
            self.staleness_samples = len(self.active_tasks) + await self.message_queue_client.get_queue_size()

            for sample_id in list(self.sample_staleness_tracker.keys()):
                if sample_id not in self.active_tasks:
                    del self.sample_staleness_tracker[sample_id]

            timing_raw = {}
            rollout_active_time = self.idle_start_time - self.step_start_time
            rollout_version_time = time.time() - self.step_start_time
            idle_ratio = 1 - rollout_active_time / rollout_version_time
            timing_raw["fully_async/rollouter/active_time"] = rollout_active_time
            timing_raw["fully_async/rollouter/version_time"] = rollout_version_time
            timing_raw["fully_async/rollouter/idle_ratio"] = idle_ratio

            print(
                f"[FullyAsyncRollouter][Public][reset_staleness] "
                f"reset staleness_samples to: {self.staleness_samples} "
                f"idle_ratio: {timing_raw['fully_async/rollouter/idle_ratio']:.4f} "
                f"current_param_version: {self.current_param_version}"
            )
            self.step_start_time = time.time()
        return timing_raw

    def do_validate(self) -> ValidateMetrics:
        """Run validation and return metrics"""
        timing_raw = {}
        with marked_timer("rollouter/validate_time", timing_raw, color="green"):
            val_metrics: dict = self._validate()
        return ValidateMetrics(timing_raw=timing_raw, metrics=val_metrics)

    async def save_checkpoint(self, local_global_step_folder: str):
        # WARNING!: Due to the asynchronous nature, there are some in-flight samples
        # (pending/cancel/result queue and message queue).
        # Therefore, directly saving the state of the dataloader will result in losing these
        # samples when resuming training.
        # TODO: Implement dataloader recovery without losing in-flight samples.
        from verl.utils.fs import local_mkdir_safe

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        async with self.dataloader_lock:
            dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)
        print(f"[FullyAsyncRollouter] Saved dataloader checkpoint to {dataloader_local_path}")

    def load_checkpoint(self):
        """Load checkpoint including dataloader state based on resume mode"""

        if self.config.trainer.resume_mode == "disable":
            print("[FullyAsyncRollouter] Resume mode is disabled, starting from scratch")
            return 0

        # Determine checkpoint folder path
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("[FullyAsyncRollouter] Load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)

            global_step_folder = find_latest_ckpt_path(checkpoint_folder)

        # Find and validate global_step_folder based on resume mode
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("[FullyAsyncRollouter] Training from scratch (no checkpoint found)")
                return 0
        elif self.config.trainer.resume_mode == "resume_path":
            assert isinstance(self.config.trainer.resume_from_path, str), (
                "[FullyAsyncRollouter] resume_from_path must be str type"
            )
            assert "global_step_" in self.config.trainer.resume_from_path, (
                "[FullyAsyncRollouter] resume_from_path must specify the global_steps"
            )
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                working_dir = os.getcwd()
                global_step_folder = os.path.join(working_dir, global_step_folder)
        else:
            raise ValueError(f"[FullyAsyncRollouter] Unknown resume_mode: {self.config.trainer.resume_mode}")

        print(f"[FullyAsyncRollouter] Loading checkpoint from: {global_step_folder}")

        # Extract and set global step
        trainer_global_steps = int(global_step_folder.split("global_step_")[-1])
        self.global_steps = (
            trainer_global_steps * self.required_samples * self.config.async_training.trigger_parameter_sync_step + 1
        )
        print(f"[FullyAsyncRollouter] Setting global_steps to {self.global_steps}")

        # Load dataloader state
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
            print(f"[FullyAsyncRollouter] Loaded dataloader state from {dataloader_local_path}")
        else:
            print(
                f"[FullyAsyncRollouter] Warning: No dataloader state found at {dataloader_local_path}, "
                f"will start from scratch"
            )

    def _validate_config(self):
        # Validate asynchronous training configuration
        if not hasattr(self.config, "async_training"):
            raise ValueError("[FullyAsyncRollouter] Missing async_training configuration")
        assert self.config.actor_rollout_ref.rollout.calculate_log_probs, "must rollout calculate log_probs"

    async def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self._init_async_objects()
        self._create_worker_classes()
        self._init_reward_loop()
        await self._init_async_rollout_manager()

    def _create_actor_rollout_classes(self):
        # Skip rollout creation and let agentloop handle it
        pass

    def _init_models(self):
        self.rollout_wg = self.all_wg[str(Role.Rollout)]
        self.rollout_wg.init_model()
        self.actor_rollout_wg = self.rollout_wg

    def _create_continuous_iterator(self):
        """
        Create a continuous data iterator across epoch
        """
        for epoch in range(self.config.trainer.total_epochs):
            iterator = iter(self.train_dataloader)
            for batch_dict in iterator:
                yield epoch, batch_dict

    async def _init_async_rollout_manager(self):
        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None

        # create async rollout manager and request scheduler
        assert self.config.actor_rollout_ref.rollout.mode == "async"
        from verl.experimental.fully_async_policy.agent_loop import FullyAsyncAgentLoopManager

        self.async_rollout_mode = True
        self.async_rollout_manager = await FullyAsyncAgentLoopManager.create(
            config=self.config, worker_group=self.rollout_wg, reward_loop_worker_handles=reward_loop_worker_handles
        )

    # Add samples to the pending_queue
    async def _feed_samples(self):
        continuous_iterator = self._create_continuous_iterator()

        for epoch, batch_dict in continuous_iterator:
            # Similar to _prepare_generate_batch: Separate data
            full_batch = prepare_single_generation_data(batch_dict, self.config)

            sample_id = f"sample_{epoch}_{self.global_steps}"

            rollout_sample = RolloutSample(
                full_batch=full_batch,
                sample_id=sample_id,
                epoch=epoch,
                rollout_status={},
            )

            await self.pending_queue.put(rollout_sample)

            # Check if have reached the last step
            if self.global_steps >= self.total_rollout_steps:
                print(
                    f"[FullyAsyncRollouter][Feed] "
                    f"Maximum count has been reached, stop adding new samples: "
                    f"{self.global_steps} >= {self.total_rollout_steps}"
                )
                break

            self.global_steps += 1

        # End signal
        await self.pending_queue.put(None)
        print(f"[FullyAsyncRollouter][Feed] Sample addition is complete, {self.global_steps} samples have been added")

    def _get_stale_samples(self) -> list[tuple[str, dict]]:
        """Get samples that have staleness exceeding the threshold."""
        stale_samples = []
        current_version = self.current_param_version

        for sample_id, info in self.sample_staleness_tracker.items():
            staleness = current_version - info["start_version"]
            if staleness >= self.staleness_rebalance_threshold:
                stale_samples.append((sample_id, info))

        return sorted(stale_samples, key=lambda x: x[1]["start_version"])

    def _find_idle_workers(self) -> list[int]:
        """Find workers that are currently idle."""
        idle_workers = []
        for worker_id, is_idle in self.worker_idle_status.items():
            if is_idle and len(self.worker_sample_mapping[worker_id]) == 0:
                idle_workers.append(worker_id)
        return idle_workers

    def _select_worker_for_stale_sample(self, stale_info: dict) -> int | None:
        """Select an idle worker to help process a stale sample."""
        idle_workers = self._find_idle_workers()
        if not idle_workers:
            return None

        min_load_worker = min(idle_workers, key=lambda w: len(self.worker_sample_mapping[w]))
        return min_load_worker

    async def _rebalance_stale_samples(self):
        """
        Rebalance stale samples to idle workers.
        This is called when some samples have been processing for too long.
        """
        stale_samples = self._get_stale_samples()
        if not stale_samples:
            return

        rebalanced_count = 0
        for sample_id, stale_info in stale_samples:
            idle_worker = self._select_worker_for_stale_sample(stale_info)
            if idle_worker is not None:
                print(
                    f"[FullyAsyncRollouter][Rebalance] Sample {sample_id} with staleness "
                    f"{self.current_param_version - stale_info['start_version']} assigned to idle worker {idle_worker}"
                )
                rebalanced_count += 1

        if rebalanced_count > 0:
            print(
                f"[FullyAsyncRollouter][Rebalance] Rebalanced {rebalanced_count} stale samples to idle workers"
            )

    async def _processor_worker(self):
        """
        Streaming worker coroutines, a sample is submitted for processing without waiting for batches.
        Implements staleness-aware scheduling to prevent sample bias.
        """
        while True:
            if self.paused or await self._should_pause_generation():
                print(
                    "[FullyAsyncRollouter][Processor] Received pause signal, waiting for remaining tasks to return..."
                )
                async with self.lock:
                    self.paused = True
                while self.active_tasks:
                    async with self.lock:
                        if self.active_tasks:
                            done_tasks, self.active_tasks = await asyncio.wait(
                                self.active_tasks, return_when=asyncio.FIRST_COMPLETED
                            )
                            for task in done_tasks:
                                await task

                async with self.lock:
                    while self.paused:
                        self.idle_start_time = time.time()
                        await self.condition.wait()
                continue

            rollout_sample = await self.pending_queue.get()
            self.pending_queue.task_done()
            self.staleness_samples += 1

            if rollout_sample is None:
                print(
                    "[FullyAsyncRollouter][Processor] Received end signal, waiting for remaining tasks to complete..."
                )
                while self.active_tasks:
                    async with self.lock:
                        if self.active_tasks:
                            done_tasks, self.active_tasks = await asyncio.wait(
                                self.active_tasks, return_when=asyncio.FIRST_COMPLETED
                            )
                            for task in done_tasks:
                                await task
                break

            while len(self.active_tasks) >= self.max_concurrent_samples:
                async with self.lock:
                    if self.active_tasks:
                        done_tasks, self.active_tasks = await asyncio.wait(
                            self.active_tasks, return_when=asyncio.FIRST_COMPLETED
                        )
                        for task in done_tasks:
                            await task

            async with self.lock:
                while self.paused:
                    await self.condition.wait()

                worker_id = self._select_best_worker_for_sample()
                task = safe_create_task(
                    self._process_single_sample_streaming(rollout_sample, worker_id=worker_id),
                    name=rollout_sample.sample_id,
                    task_set=self.active_tasks,
                )

    def _select_best_worker_for_sample(self) -> int | None:
        """
        Select the best worker for a new sample.
        Implements staleness-aware scheduling:
        1. If there are stale samples and idle workers, idle workers should not take new tasks
        2. Otherwise, use round-robin load balancing
        """
        stale_samples = self._get_stale_samples()
        idle_workers = self._find_idle_workers()

        if stale_samples and idle_workers:
            return None

        if not self.worker_idle_status:
            return None

        worker_ids = list(self.worker_idle_status.keys())
        if not hasattr(self, "_worker_rr_index"):
            self._worker_rr_index = 0

        worker_id = worker_ids[self._worker_rr_index]
        self._worker_rr_index = (self._worker_rr_index + 1) % len(worker_ids)
        return worker_id

    async def _process_single_sample_streaming(self, rollout_sample: RolloutSample, worker_id: int = None):
        """Process a single sample streamingly"""
        sample_id = rollout_sample.sample_id
        start_param_version = self.current_param_version

        async with self.lock:
            self.sample_staleness_tracker[sample_id] = {
                "start_version": start_param_version,
                "worker_id": worker_id,
                "start_time": time.time(),
            }
            if worker_id is not None:
                self.worker_sample_mapping[worker_id].add(sample_id)
                self.worker_idle_status[worker_id] = False

        ret = await self.async_rollout_manager.generate_sequences_single(
            rollout_sample.full_batch,
            worker_id=worker_id,
            required_param_version=start_param_version
        )
        rollout_sample.full_batch = ret
        rollout_sample.full_batch.non_tensor_batch["uid"] = np.array(
            [f"uid_{rollout_sample.sample_id}"] * len(rollout_sample.full_batch), dtype=object
        )
        rollout_sample.rollout_status = await self.get_statistics()

        success = await self.message_queue_client.put_sample(
            sample=ray.cloudpickle.dumps(rollout_sample),
        )
        if success:
            self.total_generated_samples += 1
        else:
            self.dropped_stale_samples += 1
        self.processed_sample_count += 1

        async with self.lock:
            if sample_id in self.sample_staleness_tracker:
                del self.sample_staleness_tracker[sample_id]
            if worker_id is not None:
                self.worker_sample_mapping[worker_id].discard(sample_id)
                if len(self.worker_sample_mapping[worker_id]) == 0:
                    self.worker_idle_status[worker_id] = True

    async def _streaming_generation_main(self):
        """The main entry method for stream processing"""

        if self.async_rollout_manager is None:
            await self._init_async_rollout_manager()

        # Start the streaming loop
        print(f"[FullyAsyncRollouter] Start streaming mode, maximum concurrent samples: {self.max_concurrent_samples}")

        # Start sample feed coroutine, streaming process coroutine
        self.feed_task = safe_create_task(self._feed_samples(), name="feed_task")
        self.processor_task = safe_create_task(self._processor_worker(), name="processor_task")

        try:
            # Wait for sample feed to complete
            # Use asyncio.wait to monitor all tasks. If processor exits early,
            # detect it instead of blocking on feed_task (it might be stuck on a full queue).
            done, pending = await asyncio.wait(
                [self.feed_task, self.processor_task], return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                if task.exception():
                    raise task.exception()

            if self.feed_task not in done:
                raise RuntimeError("Processor task exited prematurely")

            print("[FullyAsyncRollouter] Sample feed completed")

            # Wait for streaming to complete
            await self.processor_task
            print("[FullyAsyncRollouter] Streaming process completed")

            await self.pending_queue.join()
            print("[FullyAsyncRollouter] pending_queue joined")

        except Exception as e:
            print(f"[FullyAsyncRollouter] Streaming process exception: {e}")
            raise e

        finally:
            if self.feed_task and not self.feed_task.done():
                self.feed_task.cancel()
                await asyncio.gather(self.feed_task, return_exceptions=True)

            if self.processor_task and not self.processor_task.done():
                self.processor_task.cancel()
                await asyncio.gather(self.processor_task, return_exceptions=True)

            self.feed_task = None
            self.processor_task = None

            # Send a finish signal
            await self.message_queue_client.put_sample(sample=None)

        async with self.lock:
            self.running = False

    async def fit(self):
        """
        Start the async rollouter - entry point that sets up and runs async tasks
        Main async fit method that coordinates all coroutines
        """

        print("[FullyAsyncRollouter] Starting FullyAsyncRollouter...")

        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set. Call set_message_queue_client() first.")

        # Set the running status flag
        async with self.lock:
            self.paused = False
            self.running = True

        # Create the main asynchronous task
        generation_task = safe_create_task(self._streaming_generation_main(), name="generation_task")
        monitor_task = safe_create_task(self._async_monitor_loop(), name="monitor_task")

        try:
            # Run build and monitoring tasks concurrently
            await asyncio.gather(generation_task, monitor_task, return_exceptions=True)
        except Exception as e:
            print(f"[FullyAsyncRollouter] Asynchronous task execution error: {e}")
        finally:
            if not generation_task.done():
                generation_task.cancel()
            if not monitor_task.done():
                monitor_task.cancel()

            # Wait for the task to complete
            await asyncio.gather(generation_task, monitor_task, return_exceptions=True)

        print("[FullyAsyncRollouter] Rollouter fit completed")

    async def _async_monitor_loop(self):
        """
        Async coroutine for monitoring:
        Function 1: Log information output
        Function 2: Trigger rollout recovery
        Function 3: Staleness rebalance check
        """
        last_stats_time = time.time()
        stats_interval = 60.0
        check_interval = 10.0
        rebalance_check_interval = 30.0
        last_rebalance_check = time.time()

        while True:
            async with self.lock:
                if not self.running:
                    break
            await asyncio.sleep(check_interval)
            current_time = time.time()
            if current_time - last_stats_time >= stats_interval:
                stats = await self.get_statistics()
                print(f"[FullyAsyncRollouter][MonitorLoop][Statistics] {pformat(stats)}")
                last_stats_time = current_time

            if self.paused and not await self._should_pause_generation():
                async with self.lock:
                    self.paused = False
                    print("[FullyAsyncRollouter][ShouldPause] notify all wait tasks.")
                    self.condition.notify_all()

            if current_time - last_rebalance_check >= rebalance_check_interval:
                await self._check_and_rebalance_staleness()
                last_rebalance_check = current_time

    async def _check_and_rebalance_staleness(self):
        """Check staleness and trigger rebalance if needed."""
        stale_samples = self._get_stale_samples()
        if not stale_samples:
            return

        idle_workers = self._find_idle_workers()
        if not idle_workers:
            return

        print(
            f"[FullyAsyncRollouter][StalenessCheck] Found {len(stale_samples)} stale samples "
            f"and {len(idle_workers)} idle workers"
        )

        await self._rebalance_stale_samples()

    async def _should_pause_generation(self) -> bool:
        """Determine whether the build should be paused"""
        queue_stats = self.message_queue_client.get_statistics_sync()
        queue_size = queue_stats["queue_size"]

        if queue_size >= self.max_queue_size:
            if not self.paused:
                print(
                    f"[FullyAsyncRollouter][ShouldPause]  "
                    f"due to full queue: size={queue_size}, max={self.max_queue_size}"
                )
            return True

        if self.staleness_samples >= self.max_required_samples:
            if not self.paused:
                print(
                    "[FullyAsyncRollouter][ShouldPause] "
                    f"due to "
                    f"staleness_samples {self.staleness_samples} >= max_required_samples {self.max_required_samples} "
                )
            return True

        return False

    async def get_statistics(self) -> dict:
        queue_stats = self.message_queue_client.get_statistics_sync()

        stale_sample_count = 0
        max_staleness = 0
        for sample_id, info in self.sample_staleness_tracker.items():
            staleness = self.current_param_version - info["start_version"]
            if staleness >= self.staleness_rebalance_threshold:
                stale_sample_count += 1
            max_staleness = max(max_staleness, staleness)

        idle_worker_count = sum(1 for is_idle in self.worker_idle_status.values() if is_idle)

        stats = {
            "monitor/active_tasks_size": len(self.active_tasks),
            "monitor/queue/pending_queue_size": self.pending_queue.qsize(),
            "monitor/queue/mq_queue_size": queue_stats["queue_size"],
            "count/total_generated_samples": self.total_generated_samples,
            "count/staleness_samples": self.staleness_samples,
            "count/dropped_stale_samples": self.dropped_stale_samples,
            "count/stale_sample_count": stale_sample_count,
            "count/max_staleness": max_staleness,
            "count/idle_worker_count": idle_worker_count,
            "count/current_param_version": self.current_param_version,
            "static/max_required_samples": self.max_required_samples,
            "static/required_samples": self.required_samples,
            "static/staleness_threshold": self.staleness_threshold,
            "static/staleness_rebalance_threshold": self.staleness_rebalance_threshold,
            "static/max_queue_size": self.max_queue_size,
            "static/max_concurrent_samples": self.max_concurrent_samples,
        }

        return stats
