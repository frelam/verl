#!/usr/bin/env bash
# Reasoning RL (math/code/logic/STEM, rule-verifiable rewards) | Qwen3-4B | DAPO | FSDP | vLLM
#
# Data preparation (one-time, see examples/reasoning_rl/README.md):
#   python examples/reasoning_rl/scripts/to_parquet_math.py  --local_save_dir $DATA_DIR_RAW/math
#   python examples/reasoning_rl/scripts/to_parquet_code.py  --local_save_dir $DATA_DIR_RAW/code
#   python examples/reasoning_rl/scripts/to_parquet_logic.py --local_save_dir $DATA_DIR_RAW/logic
#   python examples/reasoning_rl/scripts/to_parquet_stem.py  --local_save_dir $DATA_DIR_RAW/stem
#   python examples/reasoning_rl/scripts/mix.py --input_dir $DATA_DIR_RAW -o $DATA_DIR/train.parquet
#
# DAPO switches (DESIGN.md section 8):
#   - clip-higher (clip_ratio_high=0.28)              ON  (keeps low-prob exploration tokens)
#   - dynamic sampling / filter_groups (metric=score) ON  (owned by the hard-replay sampler)
#   - token-level policy gradient loss                ON  (loss_agg_mode=token-mean)
#   - overlong reward shaping                         OFF (this stage)
#   - length penalty                                  OFF (recall first)
#   - rollout n=16, temperature=1.0; response length curriculum 8k -> 16k -> 24k
#     (raise MAX_RESPONSE_LENGTH between runs)
#
# Optional env knobs:
#   REASONING_RL_HARD_REPLAY=1           tiered hard-sample replay on the V1 trainer
#                                        (default on): uniform-reward groups are filtered
#                                        (DAPO) and groups with pass rate < 0.5 are pooled
#                                        and re-rolled on per-tier intervals (medium ~10
#                                        steps, hard ~20), graduating at pass rate >= 0.5.
#   REASONING_RL_REPLAY_RATIO=1.0        per-fetch throttle on drawing a DUE pooled sample
#   REASONING_RL_REPLAY_MAX_FRACTION=0.2 hard cap on replays per training step, as a
#                                        fraction of train_batch_size (0 = no cap)
#   REASONING_RL_REPLAY_MEDIUM_INTERVAL=10 / REASONING_RL_REPLAY_HARD_INTERVAL=20
#   REASONING_RL_REPLAY_MAX=0            give up on a pooled sample after this many
#                                        replays (0 = replay forever)
#   SANDBOX_FUSION_URL=                  sandbox-fusion endpoint for code rewards,
#                                        e.g. http://<host>/run_code. REQUIRED for real
#                                        training runs with code data (DESIGN.md section 6);
#                                        empty falls back to prime_code local execution
#                                        (smoke runs only).

set -xeuo pipefail

########################### user-adjustable ###########################
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

DATA_DIR=${DATA_DIR:-$HOME/data/reasoning_rl/final}
train_files="['$DATA_DIR/train.parquet']"
val_files="['$DATA_DIR/val.parquet']"

train_batch_size=${TRAIN_BATCH_SIZE:-256}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-64}
max_prompt_length=${MAX_PROMPT_LENGTH:-4096}   # code prompts need up to 4k (DESIGN.md section 8)
max_response_length=${MAX_RESPONSE_LENGTH:-8192} # curriculum: 8192 -> 16384 -> 24576
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}

actor_lr=${ACTOR_LR:-1e-6}
entropy_coeff=${ENTROPY_COEFF:-0}
# clip-higher: keep epsilon_low at 0.2, raise epsilon_high (DAPO).
clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}

rollout_tp=${ROLLOUT_TP:-1}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.65}
rollout_n=${ROLLOUT_N:-16}
temperature=${TEMPERATURE:-1.0}

total_epochs=${TOTAL_EPOCHS:-10}
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-10}

# V1 trainer + tiered hard-sample replay (owns DAPO filter_groups filtering).
use_v1=${REASONING_RL_USE_V1:-1}
hard_replay=${REASONING_RL_HARD_REPLAY:-1}
replay_ratio=${REASONING_RL_REPLAY_RATIO:-1.0}
replay_max=${REASONING_RL_REPLAY_MAX:-0}
replay_max_fraction=${REASONING_RL_REPLAY_MAX_FRACTION:-0.2}
replay_medium_interval=${REASONING_RL_REPLAY_MEDIUM_INTERVAL:-10}
replay_hard_interval=${REASONING_RL_REPLAY_HARD_INTERVAL:-20}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PROJECT_NAME=${PROJECT_NAME:-verl_reasoning_rl}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_reasoning_rl_dapo_$(date +%Y%m%d_%H%M)}
########################### end user-adjustable ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$train_files"
    data.val_files="$val_files"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.truncation='left'
    data.filter_overlong_prompts=True
    # Custom dataset: hard-replay re-entry (transparent pass-through unless
    # REASONING_RL_HARD_REPLAY is set).
    data.custom_cls.path="$REPO_ROOT/examples/reasoning_rl/reasoning_rl_dataset.py"
    data.custom_cls.name=ReasoningRLDataset
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    # clip-higher (DAPO): epsilon_low fixed, epsilon_high raised.
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
    actor_rollout_ref.actor.clip_ratio_c=10.0
    # token-level policy gradient loss (DAPO).
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=sync
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.temperature=${temperature}
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
)

REWARD=(
    reward.custom_reward_function.path="$REPO_ROOT/examples/reasoning_rl/reward/compute_score.py"
    reward.custom_reward_function.name=compute_score
    reward.reward_manager.name=naive
)

# sandbox-fusion URL rides through reward_kwargs (merged into every
# compute_score call by verl's _call_with_kwargs wrapper).
if [ -n "${SANDBOX_FUSION_URL:-}" ]; then
    REWARD+=(
        "+reward.custom_reward_function.reward_kwargs.sandbox_fusion_url=${SANDBOX_FUSION_URL}"
    )
fi

TRAINER=(
    trainer.use_v1=$([ "$use_v1" = "1" ] && echo true || echo false)
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=True
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
)

# Tiered hard-sample replay: the custom sampler owns DAPO filtering (uniform
# reward groups are dropped and refilled) AND exports low-pass-rate groups to
# the tiered replay pool. The framework does not inject algorithm.filter_groups
# into custom samplers, so the knobs go through sampler_kwargs. The dataset
# side (ReasoningRLDataset) reads REASONING_RL_HARD_REPLAY /
# REASONING_RL_REPLAY_RATIO and needs the pool in-process, hence
# dataloader_num_workers=0.
if [ "$hard_replay" = "1" ] && [ "$use_v1" != "1" ]; then
    echo "REASONING_RL_HARD_REPLAY=1 requires the V1 trainer; set REASONING_RL_USE_V1=1 (or disable replay)." >&2
    exit 1
fi
if [ "$hard_replay" = "1" ]; then
    export REASONING_RL_HARD_REPLAY=1
    export REASONING_RL_REPLAY_RATIO=${replay_ratio}
    DATA+=(
        trainer.v1.sampler.custom_sampler.path="$REPO_ROOT/examples/reasoning_rl/hard_replay.py"
        trainer.v1.sampler.custom_sampler.name=HardReplaySampler
        # NOTE: `sampler_kwargs: {}` in ppo_trainer.yaml is an empty dict, which
        # OmegaConf marks as struct; merging new keys into it without the `+`
        # prefix fails with "Key ... is not in struct". `+` force-adds them.
        "+trainer.v1.sampler.sampler_kwargs={filter_metric: score, train_batch_size: ${train_batch_size}, medium_interval: ${replay_medium_interval}, hard_interval: ${replay_hard_interval}, max_replays: ${replay_max}, max_replay_fraction: ${replay_max_fraction}}"
        data.dataloader_num_workers=0
        # The custom sampler owns DAPO filtering, so the framework's automatic
        # "exact refill => gen_batch_size=1" override (which only triggers on
        # algorithm.filter_groups.enable) does not apply. Force single-prompt
        # dataloader fetches here or refill_fn(k) crashes for k % 256 != 0.
        data.gen_batch_size=1
        # Multi-node: forward the dataset-side toggles through the job runtime
        # env (values are coerced to str in main_ppo before ray.init).
        "+ray_kwargs.ray_init.runtime_env.env_vars.REASONING_RL_HARD_REPLAY=1"
        "+ray_kwargs.ray_init.runtime_env.env_vars.REASONING_RL_REPLAY_RATIO=${replay_ratio}"
    )
else
    DATA+=(
        algorithm.filter_groups.enable=true
        algorithm.filter_groups.metric=score
    )
fi

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "$@"
