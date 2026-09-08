#!/usr/bin/env bash
# GRPO | Qwen3-4B | FSDP training + Laminar-style trajectory-level async rollout
# (https://arxiv.org/abs/2510.12633)
#
# Architecture:
#   - trainer nodes run FSDP actor/ref (hybrid rollout engines stay asleep);
#   - a dedicated rollout node pool runs vLLM replicas behind a global load balancer;
#   - every trainer step only *publishes* weights to the store (mooncake_store backend);
#     each replica independently drains -> pulls -> rejoins (no global sync point);
#   - long-tail draining replicas past DRAIN_TIMEOUT_SECONDS get their in-flight
#     requests aborted; callers resume on non-draining replicas with
#     prompt + partial tokens (duplicate prefill, no KV-cache transfer);
#   - staleness is measured from each trajectory's min_global_steps version tag and
#     bounded by trainer.v1.sampler.max_off_policy_threshold (drop strategy).
#
# Requires vllm>=0.18.0 (verl v0.9.0 dependency; reference image vllm==0.24.0).

set -xeuo pipefail

# ---- user-adjustable ----
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
TRAIN_FILE=${TRAIN_FILE:-$HOME/data/gsm8k/train.parquet}
TEST_FILE=${TEST_FILE:-$HOME/data/gsm8k/test.parquet}

TRAIN_NNODES=${TRAIN_NNODES:-1}          # FSDP trainer nodes
ROLLOUT_NNODES=${ROLLOUT_NNODES:-1}      # dedicated rollout nodes (must be > 0)
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}      # GPUs per node (both pools)

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-256}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}

ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}
ROLLOUT_N=${ROLLOUT_N:-5}

# Laminar knobs
MAX_OFF_POLICY_THRESHOLD=${MAX_OFF_POLICY_THRESHOLD:-8}     # max weight-version span per trajectory
MAX_INFLIGHT_PROMPTS=${MAX_INFLIGHT_PROMPTS:-$((2 * TRAIN_BATCH_SIZE))}  # >= train_batch_size
DRAIN_TIMEOUT_SECONDS=${DRAIN_TIMEOUT_SECONDS:-600}         # 0 disables migration
MAX_CONCURRENT_PULLS=${MAX_CONCURRENT_PULLS:-4}
KEEP_VERSIONS=${KEEP_VERSIONS:-2}                           # weight-store GC horizon

# Off-policy correction (recommended for laminar): each trajectory's own rollout
# log-probs serve as the behavior policy (bypass mode, token-level IS).
ROLLOUT_IS=${ROLLOUT_IS:-token}
ROLLOUT_IS_THRESHOLD=${ROLLOUT_IS_THRESHOLD:-2.0}

# DAPO group filtering (group-based algorithm support); set ENABLE_FILTER_GROUPS=0 to disable
ENABLE_FILTER_GROUPS=${ENABLE_FILTER_GROUPS:-1}

PROJECT_NAME=${PROJECT_NAME:-verl_laminar_example}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_grpo_laminar_async}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
# ---- end user-adjustable ----

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    data.train_files=${TRAIN_FILE}
    data.val_files=${TEST_FILE}
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.dataloader_num_workers=0
    algorithm.use_kl_in_reward=False
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3000
    actor_rollout_ref.actor.use_dynamic_bsz=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.nnodes=${ROLLOUT_NNODES}
    actor_rollout_ref.rollout.n_gpus_per_node=${NGPUS_PER_NODE}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    # behavior-policy log-probs for bypass-mode off-policy correction
    actor_rollout_ref.rollout.calculate_log_probs=True
    # publish/pull weight store (Laminar parameter service)
    actor_rollout_ref.rollout.checkpoint_engine.backend=mooncake_store
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096
    actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.mooncake_store.store_backend=mooncake
    actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.mooncake_store.keep_versions=${KEEP_VERSIONS}
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192
)

TRAINER=(
    trainer.use_v1=True
    trainer.v1.trainer_mode=laminar_async
    trainer.v1.laminar_async.num_warmup_batches=1
    trainer.v1.laminar_async.max_inflight_prompts=${MAX_INFLIGHT_PROMPTS}
    trainer.v1.laminar_async.drain_timeout_seconds=${DRAIN_TIMEOUT_SECONDS}
    trainer.v1.laminar_async.max_concurrent_pulls=${MAX_CONCURRENT_PULLS}
    trainer.v1.sampler.max_off_policy_threshold=${MAX_OFF_POLICY_THRESHOLD}
    trainer.v1.sampler.max_off_policy_strategy=drop
    transfer_queue.enable=True
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${TRAIN_NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
)

CORRECTION=(
    algorithm.rollout_correction.bypass_mode=True
    algorithm.rollout_correction.rollout_is=${ROLLOUT_IS}
    algorithm.rollout_correction.rollout_is_threshold=${ROLLOUT_IS_THRESHOLD}
)

EXTRA=(
)

if [ "${ENABLE_FILTER_GROUPS}" = "1" ]; then
    EXTRA+=(
        algorithm.filter_groups.enable=True
        algorithm.filter_groups.metric=acc
        reward.reward_manager.name=dapo
    )
fi

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${CORRECTION[@]}" \
    "${EXTRA[@]}" \
    "$@"
