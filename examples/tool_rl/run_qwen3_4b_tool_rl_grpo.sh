#!/usr/bin/env bash
# Tool-Use RL (single-turn, rule-based reward) | Qwen3-4B | GRPO | FSDP | vLLM
#
# Migrated from slime examples/tool_rl. Differences vs the slime original:
#   - rollout backend: vLLM (async server mode) instead of SGLang
#   - reward: label/rule mode only (LLM-judge RM mode is NOT migrated)
#   - tool-call masking: static variant via response_mask
#     (export TOOL_RL_MASK_FAILED_CALLS=1); slime's advantage-conditioned
#     TIS variant has no hook in verl's loss path and is not migrated
#
# Data preparation (one-time):
#   python examples/tool_rl/prepare_data.py -o "$HOME/data/tool_rl"
#
# Optional env knobs:
#   TOOL_RL_MASK_FAILED_CALLS=1      mask tokens of incorrect tool calls
#   TOOL_RL_REWARD_WEIGHTS='{"tool_correctness":0.6,"format":0.2,"tool_call":0.2}'
#   TOOL_RL_ABSTAIN_MODE=keyword     off (default) | keyword
#                                    rule-based shaping for no-tool samples:
#                                    request-more-info / no-valid-tools call 1.0,
#                                    guess & spurious calls 0 (Dim 1/Dim 3).
#                                    On by default; set TOOL_RL_ABSTAIN_MODE=off
#                                    to restore the legacy behaviour.
#   TOOL_RL_HINT_MODE=random         off (default) | random | fixed
#                                    inject a random system-prompt hint variant
#                                    per sample per epoch (fixed = deterministic
#                                    per sample; val files always fixed)
#   TOOL_RL_HINT_EMPTY_PROB=0.25     probability of the empty (no-hint) variant
#   TOOL_RL_HINT_SEED=42             base seed for hint sampling

set -xeuo pipefail

########################### user-adjustable ###########################
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-16}

DATA_DIR=${DATA_DIR:-$HOME/data/tool_rl}
train_files="['$DATA_DIR/train.parquet']"
val_files="['$DATA_DIR/val.parquet']"

train_batch_size=${TRAIN_BATCH_SIZE:-256}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-64}
max_prompt_length=${MAX_PROMPT_LENGTH:-4096}
max_response_length=${MAX_RESPONSE_LENGTH:-4096}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}

actor_lr=${ACTOR_LR:-1e-6}
entropy_coeff=${ENTROPY_COEFF:-0}

rollout_tp=${ROLLOUT_TP:-1}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.65}
rollout_n=${ROLLOUT_N:-16}

total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-5}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PROJECT_NAME=${PROJECT_NAME:-verl_tool_rl}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_tool_rl_grpo_vllm_fsdp_$(date +%Y%m%d_%H%M)}
########################### end user-adjustable ###########################

# Rule-based abstention shaping (see reward/abstention.py). On by default;
# the reward function reads this at compute time, so exporting it before
# launching the trainer is sufficient.
export TOOL_RL_ABSTAIN_MODE=${TOOL_RL_ABSTAIN_MODE:-keyword}

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    # Sync/on-policy: reuse the rollout generation-time log probs directly as
    # old_log_probs, skipping the extra actor forward pass (remember rollout
    # temperature is 1.0 so the two match). Disable if you switch to async /
    # stale-checkpoint rollouts.
    algorithm.rollout_correction.bypass_mode=True
    data.train_files="$train_files"
    data.val_files="$val_files"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    # Per-sample tool schemas are injected at rollout time and are not counted
    # by the dataset-side length filter, so keep filtering off here.
    data.filter_overlong_prompts=False
    # Custom dataset: randomized system-prompt hint injection per epoch
    # (transparent pass-through unless TOOL_RL_HINT_MODE is set).
    data.custom_cls.path="$REPO_ROOT/examples/tool_rl/tool_rl_dataset.py"
    data.custom_cls.name=ToolRLHintDataset
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
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    # async server mode is required for agent loops
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    # Store the generation-time log probs in the batch so bypass mode can use
    # them directly as old_log_probs (see algorithm.rollout_correction.bypass_mode).
    actor_rollout_ref.rollout.calculate_log_probs=True
    # custom single-turn tool agent loop (per-sample tools + optional masking)
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_rl_agent
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$REPO_ROOT/examples/tool_rl/agent_loop_config.yaml"
)

REWARD=(
    reward.custom_reward_function.path="$REPO_ROOT/examples/tool_rl/reward/tool_rl_reward.py"
    reward.custom_reward_function.name=compute_score
    reward.reward_manager.name=naive
)

TRAINER=(
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

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "$@"
