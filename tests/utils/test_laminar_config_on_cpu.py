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

import pytest
from omegaconf import OmegaConf

from verl.utils.config import validate_laminar_config


def _config(mode="separate_async"):
    return OmegaConf.create(
        {
            "data": {"train_batch_size": 8},
            "reward": {"reward_model": {"enable": False, "enable_resource_pool": False}},
            "actor_rollout_ref": {
                "rollout": {
                    "name": "vllm",
                    "full_determinism": False,
                    "max_inflight_requests_per_replica": None,
                    "enable_stats": False,
                    "disable_log_stats": True,
                    "stats_interval_s": 5,
                    "checkpoint_engine": {"backend": "nixl"},
                }
            },
            "trainer": {
                "v1": {
                    "trainer_mode": mode,
                    "sampler": {
                        "max_off_policy_strategy": "drop",
                        "custom_sampler": {"path": None, "name": None},
                        "sampler_kwargs": {"fifo_ratio": 0.5, "age_bonus": 0.1},
                    },
                    "async_trajectory": {
                        "single_version_per_trajectory": False,
                        "max_outstanding_groups": None,
                        "capacity_policy": "hybrid",
                        "high_watermark_ratio": 0.8,
                        "low_watermark_ratio": 0.5,
                        "repack": {
                            "enable": False,
                            "migration": {"enable": False},
                        },
                    },
                    "separate_async": {
                        "weight_sync": {
                            "mode": "global_push",
                            "cache_versions": 2,
                            "retry_interval_s": 5,
                        },
                        "hybrid_rollout": {"enable_switch": False},
                    },
                }
            },
        }
    )


def test_default_laminar_config_is_compatible():
    validate_laminar_config(_config())


def test_observe_and_builtin_sampler_are_valid():
    config = _config()
    config.trainer.v1.sampler.max_off_policy_strategy = "observe"
    config.trainer.v1.sampler.custom_sampler = {
        "path": "verl.trainer.ppo.v1.samplers",
        "name": "Length",
    }
    validate_laminar_config(config)


@pytest.mark.parametrize("maximum", [0, 4, -1])
def test_outstanding_cap_must_cover_train_batch(maximum):
    config = _config()
    config.trainer.v1.async_trajectory.max_outstanding_groups = maximum
    with pytest.raises(ValueError, match="max_outstanding_groups"):
        validate_laminar_config(config)


def test_sync_rejects_async_capacity():
    config = _config(mode="sync")
    config.trainer.v1.async_trajectory.max_outstanding_groups = 8
    with pytest.raises(ValueError, match="only valid for asynchronous"):
        validate_laminar_config(config)


def test_full_determinism_rejects_dynamic_features():
    config = _config()
    config.actor_rollout_ref.rollout.full_determinism = True
    config.actor_rollout_ref.rollout.max_inflight_requests_per_replica = 4
    with pytest.raises(ValueError, match="full_determinism"):
        validate_laminar_config(config)


def test_pull_rejects_hybrid_switch_and_unsupported_backend():
    config = _config()
    config.trainer.v1.separate_async.weight_sync.mode = "per_replica_pull"
    config.trainer.v1.separate_async.hybrid_rollout.enable_switch = True
    with pytest.raises(ValueError, match="hybrid_rollout"):
        validate_laminar_config(config)
    config.trainer.v1.separate_async.hybrid_rollout.enable_switch = False
    config.actor_rollout_ref.rollout.checkpoint_engine.backend = "mooncake"
    with pytest.raises(ValueError, match="NIXL or KIMI"):
        validate_laminar_config(config)
    config.actor_rollout_ref.rollout.checkpoint_engine.backend = "nixl"
    config.actor_rollout_ref.rollout.name = "trtllm"
    with pytest.raises(ValueError, match="validated vLLM or SGLang"):
        validate_laminar_config(config)


def test_repack_requires_stats_and_supported_migration_engine():
    config = _config()
    config.trainer.v1.async_trajectory.repack.enable = True
    with pytest.raises(ValueError, match="enable_stats"):
        validate_laminar_config(config)
    config.actor_rollout_ref.rollout.enable_stats = True
    config.actor_rollout_ref.rollout.disable_log_stats = False
    config.trainer.v1.async_trajectory.repack.migration.enable = True
    config.actor_rollout_ref.rollout.name = "trtllm"
    with pytest.raises(ValueError, match="TRT-LLM"):
        validate_laminar_config(config)


def test_reward_variance_rejects_colocated_reward_model():
    config = _config()
    config.reward.reward_model.enable = True
    config.reward.reward_model.enable_resource_pool = False
    config.trainer.v1.sampler.custom_sampler = {
        "path": "verl.trainer.ppo.v1.samplers",
        "name": "RewardVariance",
    }
    with pytest.raises(ValueError, match="independent reward resource pool"):
        validate_laminar_config(config)
