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

import logging
from dataclasses import is_dataclass
from typing import Any, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf

__all__ = ["omega_conf_to_dataclass", "validate_config", "validate_laminar_config"]

logger = logging.getLogger(__name__)


def omega_conf_to_dataclass(config: DictConfig | dict, dataclass_type: Optional[type[Any]] = None) -> Any:
    """
    Convert an OmegaConf DictConfig to a dataclass.

    Args:
        config: The OmegaConf DictConfig or dict to convert.
        dataclass_type: The dataclass type to convert to. When dataclass_type is None,
            the DictConfig must contain _target_ to be instantiated via hydra.instantiate API.

    Returns:
        The dataclass instance.
    """
    # Got an empty config
    if not config:
        return dataclass_type if dataclass_type is None else dataclass_type()
    # Got an object
    if not isinstance(config, DictConfig | ListConfig | dict | list):
        return config

    if dataclass_type is None:
        assert "_target_" in config, (
            "When dataclass_type is not provided, config must contain _target_. "
            "See trainer/config/ppo_trainer.yaml algorithm section for an example. "
            f"Got config: {config}"
        )
        from hydra.utils import instantiate

        return instantiate(config, _convert_="partial")

    if not is_dataclass(dataclass_type):
        raise ValueError(f"{dataclass_type} must be a dataclass")
    cfg = OmegaConf.create(config)  # in case it's a dict
    # pop _target_ to avoid hydra instantiate error, as most dataclass do not have _target_
    # Updated (vermouth1992) We add _target_ to BaseConfig so that it is compatible.
    # Otherwise, this code path can't support recursive instantiation.
    # if "_target_" in cfg:
    #     cfg.pop("_target_")
    cfg_from_dataclass = OmegaConf.structured(dataclass_type)
    # let cfg override the existing vals in `cfg_from_dataclass`
    cfg_merged = OmegaConf.merge(cfg_from_dataclass, cfg)
    # now convert to `dataclass_type`
    config_object = OmegaConf.to_object(cfg_merged)
    return config_object


def update_dict_with_config(dictionary: dict, config: DictConfig):
    for key in dictionary:
        if hasattr(config, key):
            dictionary[key] = getattr(config, key)


def validate_laminar_config(config: DictConfig) -> None:
    """Fail fast on unsupported Laminar feature combinations before actors start."""
    trainer_v1 = config.trainer.get("v1", None)
    if trainer_v1 is None:
        return
    mode = trainer_v1.get("trainer_mode", "sync")
    async_cfg = trainer_v1.get("async_trajectory", {})
    sampler = trainer_v1.get("sampler", {})
    rollout = config.actor_rollout_ref.rollout
    full_determinism = rollout.get("full_determinism", False)

    strategy = sampler.get("max_off_policy_strategy", "drop")
    if strategy not in {"drop", "wait", "observe"}:
        raise ValueError("trainer.v1.sampler.max_off_policy_strategy must be drop, wait, or observe")
    if strategy == "observe" and mode == "sync":
        logger.warning("max_off_policy_strategy=observe has no effect in sync trainer mode")

    max_outstanding = async_cfg.get("max_outstanding_groups", None)
    if max_outstanding is not None:
        if not isinstance(max_outstanding, int) or max_outstanding <= 0:
            raise ValueError("max_outstanding_groups must be null or a positive integer")
        if mode == "sync":
            raise ValueError("max_outstanding_groups is only valid for asynchronous trainer modes")
        if max_outstanding < config.data.train_batch_size:
            raise ValueError("max_outstanding_groups must be at least data.train_batch_size")
    if async_cfg.get("capacity_policy", "hybrid") not in {"hybrid", "backpressure", "drop_oldest"}:
        raise ValueError("Unknown async_trajectory.capacity_policy")
    high = async_cfg.get("high_watermark_ratio", 0.8)
    low = async_cfg.get("low_watermark_ratio", 0.5)
    if not 0 <= low < high <= 1:
        raise ValueError("Require 0 <= low_watermark_ratio < high_watermark_ratio <= 1")

    custom = sampler.get("custom_sampler", {}) or {}
    if custom.get("path") == "verl.trainer.ppo.v1.samplers":
        name = custom.get("name")
        if name not in {"FIFO", "RewardVariance", "Mixed", "Length"}:
            raise ValueError(f"Unknown built-in replay sampler: {name}")
        kwargs = sampler.get("sampler_kwargs", {})
        fifo_ratio = kwargs.get("fifo_ratio", 0.5)
        age_bonus = kwargs.get("age_bonus", 0.1)
        if not 0 <= fifo_ratio <= 1:
            raise ValueError("sampler fifo_ratio must be in [0, 1]")
        if age_bonus < 0:
            raise ValueError("sampler age_bonus must be non-negative")
        if name in {"RewardVariance", "Mixed"}:
            reward_model = config.reward.reward_model
            if reward_model.enable and not reward_model.enable_resource_pool:
                raise ValueError(f"{name} requires rule-based reward or an independent reward resource pool")

    admission_cap = rollout.get("max_inflight_requests_per_replica", None)
    if admission_cap is not None and (not isinstance(admission_cap, int) or admission_cap <= 0):
        raise ValueError("max_inflight_requests_per_replica must be null or a positive integer")

    separate = trainer_v1.get("separate_async", {})
    weight_sync = separate.get("weight_sync", {})
    weight_mode = weight_sync.get("mode", "global_push")
    if weight_mode not in {"global_push", "per_replica_pull"}:
        raise ValueError("weight_sync.mode must be global_push or per_replica_pull")
    if weight_sync.get("cache_versions", 2) < 2:
        raise ValueError("weight_sync.cache_versions must be at least 2")
    if weight_sync.get("retry_interval_s", 5) <= 0:
        raise ValueError("weight_sync.retry_interval_s must be positive")
    per_replica_pull = weight_mode == "per_replica_pull"
    if per_replica_pull:
        if mode != "separate_async":
            raise ValueError("per_replica_pull is only supported by separate_async standalone rollout")
        if rollout.checkpoint_engine.backend not in {"nixl", "kimi_ckpt_engine"}:
            raise ValueError("per_replica_pull requires the NIXL or KIMI checkpoint backend")
        if rollout.name not in {"vllm", "sglang"}:
            raise ValueError("per_replica_pull requires a validated vLLM or SGLang rollout adapter")
        if separate.get("hybrid_rollout", {}).get("enable_switch", False):
            raise ValueError("per_replica_pull cannot be combined with hybrid_rollout.enable_switch")

    repack = async_cfg.get("repack", {})
    migration = repack.get("migration", {})
    migration_enabled = migration.get("enable", False)
    if migration_enabled and not repack.get("enable", False):
        raise ValueError("repack.migration.enable requires repack.enable=true")
    if repack.get("enable", False):
        if mode == "colocate_async":
            logger.warning("repack is a no-op in colocate_async mode")
        elif mode != "separate_async":
            raise ValueError("V1 repack is only supported by separate_async")
        if not rollout.get("enable_stats", False):
            raise ValueError("repack requires actor_rollout_ref.rollout.enable_stats=true")
    if migration_enabled and rollout.name == "trtllm":
        raise ValueError("TRT-LLM does not support per-request trajectory migration")
    if migration_enabled and rollout.name not in {"vllm", "sglang"}:
        raise ValueError("Trajectory migration requires a validated vLLM or SGLang adapter")
    if full_determinism and (admission_cap is not None or per_replica_pull or migration_enabled):
        raise ValueError("V1 full_determinism cannot be combined with admission caps, per-replica pull, or migration")
    if rollout.get("enable_stats", False):
        if rollout.get("stats_interval_s", 5) <= 0:
            raise ValueError("rollout.stats_interval_s must be positive")
        if rollout.get("disable_log_stats", True):
            raise ValueError("rollout.enable_stats requires disable_log_stats=false")
    if async_cfg.get("single_version_per_trajectory", False) and mode != "sync" and not per_replica_pull:
        logger.warning(
            "single_version_per_trajectory with global push discards aborted trajectories; expect resampling cost"
        )


def validate_config(
    config: DictConfig,
    use_reference_policy: bool,
    use_critic: bool,
) -> None:
    """Validate an OmegaConf DictConfig.

    Args:
        config (DictConfig): The OmegaConf DictConfig to validate.
        use_reference_policy (bool): is ref policy needed
        use_critic (bool): is critic needed
    """
    validate_laminar_config(config)

    # number of GPUs total
    n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

    if not config.actor_rollout_ref.actor.use_dynamic_bsz:
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
            ), (
                f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            )
            megatron_dp = n_gpus // (
                model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

    # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
    # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
    def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options.

        Ensures that users don't set both deprecated micro_batch_size and
        the new micro_batch_size_per_gpu parameters simultaneously.

        Args:
            mbs: Deprecated micro batch size parameter value.
            mbs_per_gpu: New micro batch size per GPU parameter value.
            name (str): Configuration section name for error messages.

        Raises:
            ValueError: If both parameters are set or neither is set.
        """
        settings = {
            "actor_rollout_ref.ref": "log_prob_micro_batch_size",
            "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
        }

        if name in settings:
            param = settings[name]
            param_per_gpu = f"{param}_per_gpu"

            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(
                    f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                    f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                )

    # Actor validation done in ActorConfig.__post_init__ and validate()
    actor_config = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    actor_config.validate(n_gpus, config.data.train_batch_size, config.actor_rollout_ref.model)

    if not config.actor_rollout_ref.actor.use_dynamic_bsz:
        if use_reference_policy:
            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.ref",
            )

        #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
        check_mutually_exclusive(
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
            "actor_rollout_ref.rollout",
        )

    if config.algorithm.get("use_kl_in_reward", False) and config.actor_rollout_ref.actor.use_kl_loss:
        print("NOTICE: You have both enabled in-reward kl and kl loss.")

    # critic
    if use_critic:
        critic_config = omega_conf_to_dataclass(config.critic)
        critic_config.validate(n_gpus, config.data.train_batch_size)

    if config.data.get("val_batch_size", None) is not None:
        print(
            "WARNING: val_batch_size is deprecated."
            + " Validation datasets are sent to inference engines as a whole batch,"
            + " which will schedule the memory themselves."
        )

    # check eval config
    if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
        assert config.actor_rollout_ref.rollout.temperature > 0, (
            "validation gen temperature should be greater than 0 when enabling do_sample"
        )

    # check LoRA rank in vLLM
    lora_config = config.actor_rollout_ref.model.get("lora", {})
    lora_rank = lora_config.get("rank", 0)
    if lora_rank <= 0:
        lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
    if lora_config.get("merge", False):
        lora_rank = 0
    if lora_rank > 0 and config.actor_rollout_ref.rollout.name == "vllm":
        from verl.workers.rollout.vllm_rollout.utils import get_vllm_max_lora_rank

        get_vllm_max_lora_rank(lora_rank)

    print("[validate_config] All configuration checks passed successfully!")
