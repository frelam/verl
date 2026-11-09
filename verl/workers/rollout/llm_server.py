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
"""
Utility classes for manage and request LLM servers:
- LLMServerManager: manage life-cycle of LLM servers, including launch, tear-down replicas.
- LLMServerClient: proxy client to request LLM servers, used by AgentLoopWorker.
"""

import asyncio
import logging
import os
import random
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import ray
from omegaconf import DictConfig, OmegaConf

from verl.single_controller.ray.base import RayResourcePool, RayWorkerGroup
from verl.utils import normalize_token_ids
from verl.utils.ray_utils import auto_await
from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.tracking import RLInsightLogger
from verl.workers.rollout.replica import (
    RolloutReplica,
    TokenOutput,
    TrajectoryAbortedError,
    get_rollout_replica_class,
)
from verl.workers.rollout.router import GlobalRequestLoadBalancer, ServerBusyError  # noqa: F401
from verl.workers.rollout.utils import update_prometheus_config

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class LLMServerClient:
    """
    A class to manage multiple OpenAI compatible LLM servers. This class provides
    - Load balance: least in-flight requests load balancing via global coordination
    - Sticky session: send multi-turn chat completions to same server for automatic prefix caching
    """

    def __init__(
        self,
        config: DictConfig,
        load_balancer_handle: ray.actor.ActorHandle = None,
        **kwargs,
    ):
        """Initialize the LLMServerClient.

        Args:
            config (DictConfig): whole config for main entrypoint.
            load_balancer_handle (ray.actor.ActorHandle): shared global load balancer actor
                that also holds the server-handle registry. Optional; subclasses that
                manage server routing externally can pass None.
        """
        self.config = config
        self._load_balancer = load_balancer_handle
        # Balancer field declarations, queried lazily once.
        self._lb_require_acquire_fields: list[str] | None = None
        self._lb_require_release_fields: list[str] | None = None
        self._lease_context: ContextVar[dict[str, Any] | None] = ContextVar(
            f"trajectory_lease_{id(self)}", default=None
        )
        trainer_v1 = getattr(getattr(config, "trainer", None), "v1", None)
        separate_config = getattr(trainer_v1, "separate_async", None)
        weight_sync = getattr(separate_config, "weight_sync", None)
        self._per_replica_pull = getattr(weight_sync, "mode", "global_push") == "per_replica_pull"
        async_trajectory = getattr(trainer_v1, "async_trajectory", None)
        repack = getattr(async_trajectory, "repack", None)
        if trainer_v1 is None:
            repack = getattr(getattr(config, "async_training", None), "repack", None)
        migration = getattr(repack, "migration", None)
        self._migration_enabled = getattr(migration, "enable", False)
        self._migration_resume_info: dict[str, dict[str, Any]] = {}

    @staticmethod
    async def _admission_sleep(attempt: int) -> None:
        delay = min(1.0, 0.05 * (2 ** min(attempt, 5))) * random.uniform(0.5, 1.0)
        await asyncio.sleep(delay)

    async def _acquire_server(self, request_id: str, **extra) -> tuple[str, ray.actor.ActorHandle]:
        # Atomic acquire: returns (server_id, handle) in one Ray RPC.
        # Only the declared fields are serialized.
        if self._lb_require_acquire_fields is None:
            acquire_fields, release_fields = await asyncio.gather(
                self._load_balancer.require_acquire_fields.remote(),
                self._load_balancer.require_release_fields.remote(),
            )
            self._lb_require_acquire_fields = list(acquire_fields)
            self._lb_require_release_fields = list(release_fields)
        fields = {name: extra[name] for name in self._lb_require_acquire_fields if name in extra}
        attempt = 0
        wait_start = time.monotonic()
        while True:
            try:
                result = await self._load_balancer.acquire_server.remote(request_id=request_id, **fields)
                if attempt:
                    self._load_balancer.record_admission_wait.remote(time.monotonic() - wait_start)
                if self._migration_enabled:
                    resume_info = await self._load_balancer.consume_migration_resume.remote(request_id)
                    if resume_info is not None:
                        self._migration_resume_info[request_id] = resume_info
                return result
            except ServerBusyError:
                # The LB actor never sleeps. Cancellation propagates from this client-side await.
                await self._admission_sleep(attempt)
                attempt += 1

    @asynccontextmanager
    async def trajectory_lease(self, uid: str, session_id: int):
        """Hold replica/version affinity across every turn of one trajectory."""
        if not self._per_replica_pull:
            yield
            return
        trajectory_id = f"{uid}:{session_id}"
        attempt = 0
        while True:
            try:
                lease = await self._load_balancer.acquire_trajectory_lease.remote(trajectory_id)
                break
            except ServerBusyError:
                await self._admission_sleep(attempt)
                attempt += 1
        token = self._lease_context.set(lease)
        try:
            yield lease
        finally:
            self._lease_context.reset(token)
            await self._load_balancer.release_trajectory_lease.remote(trajectory_id, lease["token"])

    def _release_server(self, server_id: str, request_id: str | None = None) -> None:
        # Fire-and-forget: release is just a counter decrement, no need to await.
        # Awaiting here risks blocking the finally clause if the LB actor is unresponsive.
        pool = {"request_id": request_id}
        fields = {name: pool[name] for name in self._lb_require_release_fields if name in pool}
        self._load_balancer.release_server.remote(server_id=server_id, **fields)

    def _vllm_request_id(self, request_id: str) -> str:
        # request_id passed to vLLM. Default: a fresh uuid per turn so each turn
        # is an independent vLLM request. Under full_determinism the caller's
        # request_id is passed straight through so vLLM sees a stable id across runs.
        if getattr(self.config.actor_rollout_ref.rollout, "full_determinism", False):
            return request_id
        return uuid4().hex

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> TokenOutput:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.

        Returns:
            TokenOutput | DiffusionOutput: token or diffusion output
        """
        lease = self._lease_context.get()
        if lease is not None:
            kwargs.update(trajectory_id=lease["trajectory_id"], lease_token=lease["token"])
        engine_request_id = self._vllm_request_id(request_id)
        server_id, server = await self._acquire_server(
            request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            mm_processor_kwargs=mm_processor_kwargs,
            engine_request_id=engine_request_id,
            **kwargs,
        )
        for routing_key in ("target_server_id", "trajectory_id", "lease_token", "engine_request_id", "attempt"):
            kwargs.pop(routing_key, None)
        output = None
        try:
            multimodal_kwargs = {}
            if audio_data is not None:
                multimodal_kwargs["audio_data"] = audio_data
            if mm_processor_kwargs:
                multimodal_kwargs["mm_processor_kwargs"] = mm_processor_kwargs
            # priority is only supported by vLLM rollout server.
            priority = kwargs.pop("priority", 0)
            priority_kwargs = (
                {"priority": priority} if priority != 0 and self.config.actor_rollout_ref.rollout.name == "vllm" else {}
            )
            output: TokenOutput = await server.generate.remote(
                request_id=engine_request_id,  # use new request_id for each turn
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                **multimodal_kwargs,
                **priority_kwargs,
                **kwargs,
            )
            global_steps = output.extra_fields.get("global_steps")
            output.extra_fields.setdefault("min_global_steps", global_steps)
            output.extra_fields.setdefault("max_global_steps", global_steps)
            return output
        finally:
            self._release_server(
                server_id,
                request_id=request_id,
            )
            resume_info = self._migration_resume_info.pop(request_id, None)
            if resume_info is not None:
                self._load_balancer.record_migration_result.remote(
                    success=output is not None,
                    reprefill_tokens=len(prompt_ids),
                    duration_seconds=max(0.0, time.monotonic() - resume_info["started_at"]),
                )


class FullyAsyncLLMServerClient(LLMServerClient):
    """FullyLLMServerClient supports resume generation on partial rollout, making rollout interruption
    invisible to the AgentLoop.
    """

    def __init__(
        self,
        config: DictConfig,
        load_balancer_handle: ray.actor.ActorHandle = None,
        only_hybrid: bool = False,
        **kwargs,
    ):
        """Initialize the FullyAsyncLLMServerClient.

        Args:
            config (DictConfig): whole config for main entrypoint.
            load_balancer_handle (ray.actor.ActorHandle): shared global load balancer actor
                that also holds the server-handle registry.
            only_hybrid (bool): When ``True``, hybrid replicas are the *only* rollout
                resource.  If the load balancer is temporarily empty (e.g. during
                weight synchronisation) :meth:`_acquire_server` will keep retrying
                every 1 second instead of raising immediately.
        """
        super().__init__(config=config, load_balancer_handle=load_balancer_handle, **kwargs)
        self._only_hybrid = only_hybrid
        trainer_v1 = getattr(getattr(config, "trainer", None), "v1", None)
        async_trajectory = getattr(trainer_v1, "async_trajectory", None)
        self._single_version_per_trajectory = getattr(async_trajectory, "single_version_per_trajectory", False)

    async def _acquire_server(self, request_id: str, **extra) -> tuple[str, ray.actor.ActorHandle]:
        # Atomic acquire: returns (server_id, handle) in one Ray RPC.
        # When only_hybrid is True, hybrid replicas are the sole rollout resource and
        # the LB may be temporarily empty during weight sync / scaling transitions.
        # In that case keep retrying every 1 s until a server becomes available.
        # Otherwise raise immediately so callers see the error right away.
        while True:
            try:
                return await super()._acquire_server(request_id, **extra)
            except RuntimeError as e:
                if "No available servers in load balancer" in str(e) and self._only_hybrid:
                    await asyncio.sleep(1)
                else:
                    raise

    def _configured_response_length(self) -> Optional[int]:
        """Per-response token budget from the rollout config, or ``None`` when unavailable.

        Tests and lightweight callers may pass a config stub without the rollout section; in that
        case the resume loop keeps its previous behaviour of deferring to the server default.
        """
        rollout_config = getattr(getattr(self.config, "actor_rollout_ref", None), "rollout", None)
        response_length = getattr(rollout_config, "response_length", None)
        if isinstance(response_length, int) and response_length > 0:
            return response_length
        return None

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> TokenOutput:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.
            image_data (Optional[List[Any]]): Image data for the chat completion.
            video_data (Optional[List[Any]]): Video data for the chat completion.
            audio_data (Optional[List[Any]]): Audio data for the chat completion.
            mm_processor_kwargs (Optional[Dict[str, Any]]): Multimodal processor kwargs.

        Returns:
            TokenOutput: token output
        """
        prompt_ids = normalize_token_ids(prompt_ids)

        limit_key = None
        if "max_tokens" in sampling_params:
            limit_key = "max_tokens"
        elif "max_new_tokens" in sampling_params:
            limit_key = "max_new_tokens"
        original_max_tokens = sampling_params.get(limit_key) if limit_key else None

        # The budget below is rewritten on every attempt, and the caller reuses its dict across
        # turns, so never mutate the caller's copy.
        sampling_params = dict(sampling_params)

        if original_max_tokens is None:
            # Without an explicit limit each attempt falls back to the server-side default, which is
            # derived from len(prompt_ids) and is only correct on the first attempt: a resume passes
            # prompt + tokens generated so far, so the default charges generated tokens against the
            # *prompt* budget and re-permits close to a full response_length every time. Pin the
            # cumulative budget here instead, which also makes the bookkeeping in step 3 effective.
            response_length = self._configured_response_length()
            if response_length is not None:
                limit_key = "max_tokens"
                original_max_tokens = response_length
                sampling_params[limit_key] = response_length

        final_output = TokenOutput(
            token_ids=[],
            log_probs=[],
            num_preempted=0,
        )
        min_global_steps, max_global_steps = None, None
        # Prefix-cache hits are reported per prefill. The base client returns the
        # server's TokenOutput directly (so sync mode surfaces num_cached_tokens),
        # but here we rebuild a fresh TokenOutput across resume iterations, so we
        # must carry it forward explicitly or the consumer sees 0. Take the first
        # (initial-prompt) prefill's hit count, matching single-prefill semantics.
        num_cached_tokens = None

        while True:
            # 1. generate tokens
            output = await super().generate(
                request_id=request_id,
                prompt_ids=prompt_ids + final_output.token_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                audio_data=audio_data,
                mm_processor_kwargs=mm_processor_kwargs,
                **kwargs,
            )

            abort_reason = output.extra_fields.get("abort_reason", "weight_sync")
            if (
                self._single_version_per_trajectory
                and output.stop_reason in ("aborted", "abort")
                and abort_reason != "migration"
            ):
                raise TrajectoryAbortedError(
                    request_id=request_id,
                    generated_tokens=len(final_output.token_ids) + len(output.token_ids),
                    abort_reason=abort_reason,
                )

            # 2. merge output into final_output
            final_output.token_ids.extend(output.token_ids)
            if output.log_probs is not None:
                final_output.log_probs.extend(output.log_probs)
            # On partial rollout resume the model version may differ, so keep
            # existing routing and only append routing for newly generated tokens.
            if output.routed_experts is not None and len(output.token_ids) > 0:
                if final_output.routed_experts is None:
                    final_output.routed_experts = output.routed_experts
                else:
                    final_output.routed_experts = np.concatenate(
                        [final_output.routed_experts, output.routed_experts[-len(output.token_ids) :]]
                    )
            if output.num_preempted is not None:
                final_output.num_preempted += output.num_preempted
            final_output.stop_reason = output.stop_reason
            for key in ("abort_reason", "target_server_id", "migration_id"):
                if key in output.extra_fields:
                    final_output.extra_fields[key] = output.extra_fields[key]

            # carry the initial prefill's prefix-cache hit count forward
            if num_cached_tokens is None:
                num_cached_tokens = output.extra_fields.get("num_cached_tokens")

            # update model weights version
            global_steps = output.extra_fields.get("global_steps", None)
            if min_global_steps is None:
                min_global_steps = global_steps
            max_global_steps = global_steps

            # 3. update max_new_tokens
            if original_max_tokens is not None:
                sampling_params[limit_key] = original_max_tokens - len(final_output.token_ids)
                if len(final_output.token_ids) >= original_max_tokens:
                    final_output.stop_reason = "length"
                    break

            # 4. check stop reason
            # If partial rollout not enable, aborted samples will be dropped.
            # For v1 trainer, should_retry is always True. Since self.config.async_training is not exist.
            should_retry = True
            if hasattr(self.config, "async_training") and not self.config.async_training.partial_rollout:
                should_retry = False
            if output.stop_reason not in ("aborted", "abort") or not should_retry:
                break

            await asyncio.sleep(1)

        if final_output.log_probs and len(final_output.log_probs) != len(final_output.token_ids):
            raise RuntimeError("Partial rollout produced response logprobs misaligned with response tokens")
        final_output.extra_fields["global_steps"] = global_steps
        final_output.extra_fields["min_global_steps"] = min_global_steps
        final_output.extra_fields["max_global_steps"] = max_global_steps
        final_output.extra_fields["num_cached_tokens"] = num_cached_tokens
        return final_output


class LLMServerManager:
    """LLMServerManager is responsible for:
    - Launch server replicas
    - Launch global load balancer
    - Elastic launch/tear-down new replicas

    Args:
        config (DictConfig): Config for the trainer entrypoint.
        worker_group (RayWorkerGroup): Worker group for the server replicas. If not none, init hybrid server,
            else init standalone server with a new resource pool.
        rollout_resource_pool (RayResourcePool): Resource pool for the server replicas, only needed for TensorRT-LLM.
        start_rank (int): First ``replica_rank`` to assign.  Defaults to 0.
        load_balancer_cls: Optional subclass of the default router strategy's
            load balancer to use as the routing actor (wrapped with
            ``ray.remote`` at instantiation). When given, it takes precedence
            over ``rollout.router_config_path``. Pass a subclass that
            overrides :meth:`acquire_server` to take full control of routing.
    """

    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        rollout_resource_pool: RayResourcePool = None,
        start_rank: int = 0,
        load_balancer_cls: type | None = None,
    ):
        self.config = config
        self.rollout_config = config.actor_rollout_ref.rollout
        self.model_config = config.actor_rollout_ref.model
        self.worker_group = worker_group
        self.rollout_resource_pool = rollout_resource_pool
        self.start_rank = start_rank
        self._load_balancer_cls = load_balancer_cls
        self._standalone = worker_group is None
        self.repack_coordinator = None
        self.weight_sync_coordinator = None
        self._laminar_metrics_enabled = False
        self._laminar_counter_baseline: dict[str, float] = {}

        assert worker_group is not None or self.rollout_config.nnodes > 0, "nnodes must be > 0 in standalone mode"

        # for recipe to change
        if not hasattr(self, "rollout_replica_class"):
            self.rollout_replica_class = get_rollout_replica_class(
                self.rollout_config.name,
                disaggregation_enabled=self.rollout_config.disaggregation.enabled,
            )

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        """Create the LLMServerManager."""
        instance = cls(*args, **kwargs)
        await instance._initialize_llm_servers()
        await instance._init_global_load_balancer()
        return instance

    async def _initialize_llm_servers(self, start_rank: int = None):
        """Initialize the LLM server replicas.

        Args:
            start_rank: First ``replica_rank`` to assign.  Defaults to ``self.start_rank``
                so standalone replicas can avoid Ray named-actor collisions with hybrid
                replicas (which start at 0) when both coexist (e.g. separate async).
        """
        if start_rank is None:
            start_rank = self.start_rank
        rollout_world_size = (
            self.rollout_config.tensor_model_parallel_size
            * self.rollout_config.data_parallel_size
            * self.rollout_config.pipeline_model_parallel_size
        )
        # PD inflates per-replica footprint; miss this and init_hybrid slices
        # past worker_group → empty workers on replica_rank>=1.
        disagg = getattr(self.rollout_config, "disaggregation", None)
        if disagg is not None and getattr(disagg, "enabled", False):
            prefill_tp = self.rollout_config.tensor_model_parallel_size
            # Inline decode_tp default: OmegaConf/Ray serialization drops dataclass methods.
            decode_tp = (
                disagg.decode_tensor_model_parallel_size
                if disagg.decode_tensor_model_parallel_size is not None
                else prefill_tp
            )
            rollout_world_size = (
                (prefill_tp * disagg.prefill_replicas + decode_tp * disagg.decode_replicas)
                * self.rollout_config.data_parallel_size
                * self.rollout_config.pipeline_model_parallel_size
            )
        world_size = (
            self.worker_group.world_size
            if self.worker_group
            else self.rollout_config.n_gpus_per_node * self.rollout_config.nnodes
        )
        num_replicas = world_size // rollout_world_size

        self.rollout_replicas = [
            self.rollout_replica_class(
                replica_rank=start_rank + replica_rank,
                config=self.rollout_config,
                model_config=self.model_config,
                gpus_per_node=self.rollout_config.n_gpus_per_node,
            )
            for replica_rank in range(num_replicas)
        ]

        if self.worker_group and self.rollout_config.name != "trtllm":
            await asyncio.gather(*[server.init_hybrid(self.worker_group) for server in self.rollout_replicas])
        # TODO: unify trtllm to init_hybrid
        elif self.worker_group and self.rollout_config.name == "trtllm":
            await asyncio.gather(
                *[
                    server.init_hybrid_colocated(self.worker_group, self.rollout_resource_pool)
                    for server in self.rollout_replicas
                ]
            )
        else:
            await asyncio.gather(*[server.init_standalone() for server in self.rollout_replicas])

        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]
        print(f"LLMServerManager: {self.server_addresses}")

        # Update Prometheus / rl-insight metrics with server addresses
        needs_metrics = (
            self.rollout_config.prometheus.enable
            or RLInsightLogger.enabled()
            or getattr(self.rollout_config, "enable_stats", False)
        )
        if self.rollout_config.disable_log_stats:
            if needs_metrics:
                raise ValueError("Metrics monitoring requires disable_log_stats=False, but it is currently True.")
        if not self.rollout_config.disable_log_stats:
            if self.rollout_config.prometheus.enable:
                update_prometheus_config(
                    self.rollout_config.prometheus, self.server_addresses, self.rollout_config.name
                )
            if RLInsightLogger.enabled():
                RLInsightLogger.register_rollout_metrics(
                    self.server_addresses,
                    self.rollout_config.name,
                    labels=[{"replica": server.replica_rank} for server in self.rollout_replicas],
                )

    async def _init_global_load_balancer(self) -> None:
        from verl.workers.rollout.router import get_router_handle

        trainer_v1 = getattr(getattr(getattr(self, "config", None), "trainer", None), "v1", None)
        async_trajectory = getattr(trainer_v1, "async_trajectory", None)
        repack = getattr(async_trajectory, "repack", None)
        if trainer_v1 is None:
            repack = getattr(getattr(getattr(self, "config", None), "async_training", None), "repack", None)
        migration = getattr(repack, "migration", None)
        separate = getattr(trainer_v1, "separate_async", None)
        weight_sync = getattr(separate, "weight_sync", None)
        required_capabilities = []
        if getattr(weight_sync, "mode", "global_push") == "per_replica_pull":
            required_capabilities.extend(["version_registry", "trajectory_lease"])
        if getattr(migration, "enable", False):
            required_capabilities.append("migration")
        max_inflight = getattr(self.rollout_config, "max_inflight_requests_per_replica", None)
        if max_inflight is not None and getattr(self.rollout_config, "full_determinism", False):
            raise ValueError("V1 full_determinism does not support per-replica admission control")
        self.global_load_balancer = get_router_handle(
            servers=dict(zip(self.server_addresses, self.server_handles, strict=True)),
            router_config_path=getattr(self.rollout_config, "router_config_path", None),
            full_determinism=getattr(self.rollout_config, "full_determinism", False),
            load_balancer_cls=self._load_balancer_cls,
            max_inflight_requests_per_replica=max_inflight,
            required_capabilities=required_capabilities,
        )
        repack_enabled = getattr(repack, "enable", False)
        self._laminar_metrics_enabled = bool(max_inflight is not None or required_capabilities or repack_enabled)
        if repack_enabled and self._standalone:
            from verl.workers.rollout.repack_coordinator import RepackCoordinatorActor

            self.repack_coordinator = RepackCoordinatorActor.remote(
                servers=dict(zip(self.server_addresses, self.server_handles, strict=True)),
                load_balancer=self.global_load_balancer,
                repack_config=OmegaConf.to_container(repack, resolve=True),
                request_budget=max_inflight or self.rollout_config.max_num_seqs,
                stats_interval_s=getattr(self.rollout_config, "stats_interval_s", 5),
            )
            await self.repack_coordinator.start.remote()

    def get_client(self, client_cls: type[LLMServerClient] | None = None, **kwargs) -> LLMServerClient:
        """Get the LLMServerClient to request LLM server replicas.

        Args:
            client_cls: The client class to instantiate. Defaults to
                :class:`LLMServerClient`. Pass a subclass to customize
                request-id handling (e.g. a deterministic client that forwards
                the caller's ``request_id`` straight to vLLM), or
                :class:`FullyAsyncLLMServerClient` for abort-resume support.
            **kwargs: Forwarded to the client constructor.
        """
        client_cls = client_cls or LLMServerClient
        return client_cls(
            config=self.config,
            load_balancer_handle=self.global_load_balancer,
            **kwargs,
        )

    def get_addresses(self) -> list[str]:
        """Get the OpenAI chat completion API http addresses of the LLM server replicas."""
        return self.server_addresses

    def get_replicas(self) -> list[RolloutReplica]:
        """Get the LLM server replicas."""
        return self.rollout_replicas

    @auto_await
    async def start_per_replica_pull(self, checkpoint_engine_config, actor_wg) -> None:
        """Start standalone drain/pull updates without sharing a temporary event loop."""
        if self.weight_sync_coordinator is not None:
            return
        if not self._standalone:
            raise ValueError("per_replica_pull requires a standalone rollout pool")
        from verl.workers.rollout.weight_sync_coordinator import WeightUpdateCoordinatorActor

        trainer_v1 = self.config.trainer.v1
        weight_sync = trainer_v1.separate_async.weight_sync
        self.weight_sync_coordinator = WeightUpdateCoordinatorActor.remote(
            checkpoint_engine_config=checkpoint_engine_config,
            actor_wg=actor_wg,
            replica_workers={replica._server_address: replica.workers for replica in self.rollout_replicas},
            load_balancer=self.global_load_balancer,
            retry_interval_s=weight_sync.retry_interval_s,
        )
        await self.weight_sync_coordinator.start.remote()
        self._laminar_metrics_enabled = True

    @auto_await
    async def publish_weight_version(self, version: int) -> dict[str, Any]:
        if self.weight_sync_coordinator is None:
            raise RuntimeError("per-replica weight coordinator is not running")
        return await self.weight_sync_coordinator.publish.remote(version)

    @auto_await
    async def stop_repack(self) -> None:
        if self.repack_coordinator is not None:
            await self.repack_coordinator.stop.remote()
            self.repack_coordinator = None
        if self.weight_sync_coordinator is not None:
            await self.weight_sync_coordinator.stop.remote()
            self.weight_sync_coordinator = None

    def set_weight_syncing(self, active: bool) -> None:
        if self.repack_coordinator is not None:
            self.repack_coordinator.set_weight_syncing.remote(active)

    def _counter_delta(self, name: str, value: float) -> float:
        previous = self._laminar_counter_baseline.get(name, 0.0)
        self._laminar_counter_baseline[name] = value
        return max(0.0, value - previous)

    @auto_await
    async def get_laminar_metrics(self) -> dict[str, float]:
        """Aggregate gauges, interval counters, and duration sum/count values."""
        if not self._laminar_metrics_enabled:
            return {}
        if self.repack_coordinator is not None:
            data = await self.repack_coordinator.get_metrics.remote()
            lb = data["load_balancer"]
        else:
            data = {}
            lb = await self.global_load_balancer.get_metrics.remote()
        metrics: dict[str, float] = {}
        for name, value in lb["counters"].items():
            metric_name = (
                f"rollout/migration/{name.removeprefix('migration_')}"
                if name.startswith("migration_")
                else (f"rollout/{name}")
            )
            metrics[metric_name] = self._counter_delta(metric_name, float(value))
        for server_id, value in lb["gauges"]["inflight_requests"].items():
            metrics[f"rollout/inflight_requests/{server_id}"] = float(value)
        for name in ("draining_servers", "updating_servers", "active_trajectory_leases"):
            metrics[f"rollout/{name}"] = float(lb["gauges"][name])
        lags = list(lb["gauges"]["version_lag"].values())
        if lags:
            metrics["rollout/version_lag/mean"] = float(np.mean(lags))
            metrics["rollout/version_lag/max"] = float(max(lags))
        if self.weight_sync_coordinator is not None:
            update_metrics = await self.weight_sync_coordinator.get_metrics.remote()
            for name, value in update_metrics.items():
                metric_name = f"rollout/{name}"
                metrics[metric_name] = self._counter_delta(metric_name, float(value))

        snapshots = data.get("snapshots", {})
        complete = [snapshot for snapshot in snapshots.values() if snapshot is not None]
        if complete:
            kv_values = [snapshot.kv_cache_usage for snapshot in complete]
            metrics["rollout/kv_cache_util/mean"] = float(np.mean(kv_values))
            metrics["rollout/kv_cache_util/max"] = float(max(kv_values))
            metrics["rollout/num_running_reqs"] = float(sum(s.num_running_reqs for s in complete))
            metrics["rollout/num_waiting_reqs"] = float(sum(s.num_waiting_reqs for s in complete))
            metrics["rollout/stats_age_seconds"] = max(0.0, time.time() - min(s.timestamp for s in complete))
        idle = set(data.get("idle_replicas", []))
        metrics["rollout/idle_replicas"] = float(len(idle))
        idle_snapshots = [snapshots[server_id] for server_id in idle if snapshots.get(server_id) is not None]
        if idle_snapshots:
            metrics["rollout/idle_kv_cache_util"] = float(np.mean([s.kv_cache_usage for s in idle_snapshots]))
            metrics["rollout/idle_req_count"] = float(sum(s.num_running_reqs for s in idle_snapshots))

        coordinator = data.get("coordinator")
        if coordinator:
            direct_names = (
                "trigger_count",
                "skipped_overlap",
                "plan_empty_count",
                "backoff_active",
                "planned_sources",
                "planned_destinations",
                "planned_moves",
                "estimated_released_replicas",
                "migration_attempts",
                "migration_successes",
                "migration_failures",
                "last_run_age_s",
            )
            for name in direct_names:
                value = coordinator.get(name)
                if value is not None:
                    metric_name = "plan_empty" if name == "plan_empty_count" else name
                    metric_name = "last_run_age_seconds" if name == "last_run_age_s" else metric_name
                    key = f"rollout/repack/{metric_name}"
                    metrics[key] = (
                        float(value)
                        if name in {"backoff_active", "last_run_age_s"}
                        else self._counter_delta(key, float(value))
                    )
            for stage in ("collect", "plan", "migration", "tick"):
                duration = coordinator[f"{stage}_duration"]
                sum_key = f"rollout/repack/{stage}_duration_seconds/sum"
                count_key = f"rollout/repack/{stage}_duration_seconds/count"
                metrics[sum_key] = self._counter_delta(sum_key, duration["total_s"])
                metrics[count_key] = self._counter_delta(count_key, duration["count"])
                metrics[f"rollout/repack/{stage}_duration_seconds/last"] = duration["last_s"]
            metrics["rollout/repack/plan_duration_ms"] = coordinator["plan_duration"]["last_s"] * 1000
        return metrics

    @auto_await
    async def get_repack_snapshot(self) -> dict[str, Any]:
        if self.repack_coordinator is None:
            return {}
        return await self.repack_coordinator.get_metrics.remote()

    @auto_await
    async def start_profile(self, **kwargs):
        """Start profiling on all rollout replicas."""
        await asyncio.gather(*[replica.start_profile(**kwargs) for replica in self.rollout_replicas])

    @auto_await
    async def stop_profile(self):
        """Stop profiling on all rollout replicas."""
        await asyncio.gather(*[replica.stop_profile() for replica in self.rollout_replicas])
