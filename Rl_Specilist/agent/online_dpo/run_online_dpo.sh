#!/usr/bin/env bash
# Online DPO training: generate trajectories, judge with DeepSeek API, build
# chosen/rejected pairs (best_vs_worst), and update the actor with DPO loss.
#
# This script runs verl's main_ppo with algorithm.use_dpo=True and
# actor.policy_loss.loss_mode=dpo. It reuses the reject_sampling tooling
# (tool_config, swe_interaction, judge_reward) but performs real weight updates.
#
# Usage:
#   bash run_online_dpo.sh <dataset> <nproc_per_node> <save_path> [extra_configs...]
#
# Examples:
#   # ToolMind, 8 GPUs, Qwen3-4B
#   bash run_online_dpo.sh toolmind 8 ~/data/online_dpo/ckpt
#
#   # TerminalTraj, 4 GPUs, override beta and lr
#   bash run_online_dpo.sh terminaltraj 4 ~/data/online_dpo/ckpt \
#       actor_rollout_ref.actor.policy_loss.dpo_beta=0.05 \
#       actor_rollout_ref.actor.optim.lr=5e-7
#
# Environment variables:
#   MODEL_PATH       (default: \$HOME/models/Qwen3-4B)
#   DATA_DIR         (default: \$HOME/data/reject_sampling)
#   DEEPSEEK_API_KEY (required for judge)
#   N_SAMPLES        (default: 8, samples per prompt)
#   TEMPERATURE      (default: 0.7)
#   DPO_MODE         (auto-set to 1; tells judge_reward to skip saving)

set -xeuo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: run_online_dpo.sh <dataset> <nproc_per_node> <save_path> [extra_configs...]"
    echo ""
    echo "Datasets: toolmind | terminaltraj | open_swe_traces | swe_zero"
    echo ""
    echo "Environment variables:"
    echo "  MODEL_PATH       (default: \$HOME/models/Qwen3-4B)"
    echo "  DATA_DIR         (default: \$HOME/data/reject_sampling)"
    echo "  DEEPSEEK_API_KEY (required for judge)"
    echo "  N_SAMPLES        (default: 8, samples per prompt)"
    echo "  TEMPERATURE      (default: 0.7)"
    exit 1
fi

dataset=$1
nproc_per_node=$2
save_path=$3
shift 3 2>/dev/null || shift $#

# ---- Load env (reuse reject_sampling env.sh for paths/keys) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REJECT_DIR="$SCRIPT_DIR/../reject_sampling"
# shellcheck disable=SC1091
source "$REJECT_DIR/setup/env.sh" 2>/dev/null || true

# ---- User-adjustable ----
MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3-4B}"
DATA_DIR="${DATA_DIR:-$HOME/data/reject_sampling}"
N_SAMPLES="${N_SAMPLES:-8}"
TEMPERATURE="${TEMPERATURE:-0.7}"
PROJECT_NAME="${PROJECT_NAME:-online-dpo}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-dpo-$dataset-$(date '+%m%d-%H%M')}"

# DPO mode: tell judge_reward.compute_score to skip trajectory saving
export DPO_MODE=1

# Resolve paths (reuse reject_sampling tooling)
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_PATH="$SCRIPT_DIR/config"
TOOL_CONFIG="$REJECT_DIR/tools/tool_config.yaml"
REWARD_PATH="Rl_Specilist.agent.reject_sampling.reward.judge_reward"

# Dataset-specific prompt file (same as reject_sampling)
TRAIN_FILE="${DATA_DIR}/prompts/${dataset}.parquet"
if [ ! -f "$TRAIN_FILE" ]; then
    TRAIN_FILE="${DATA_DIR}/prompts/train.parquet"
fi
VAL_FILE="$TRAIN_FILE  # val_before_train uses same file

# ---- Verify prerequisites ----
if [ ! -f "$TRAIN_FILE" ]; then
    echo "ERROR: Prompt file not found: $TRAIN_FILE"
    echo "Run data preparation first:"
    echo "  bash $REJECT_DIR/setup/download_datasets.sh"
    exit 1
fi

if [ ! -f "$TOOL_CONFIG" ]; then
    echo "ERROR: Tool config not found: $TOOL_CONFIG"
    exit 1
fi

if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    echo "ERROR: DEEPSEEK_API_KEY not set"
    echo "  export DEEPSEEK_API_KEY=sk-xxxxx"
    exit 1
fi

echo "========================================"
echo " Online DPO Training"
echo "========================================"
echo " Dataset:       $dataset"
echo " Prompt file:   $TRAIN_FILE"
echo " Model:         $MODEL_PATH"
echo " Tool config:   $TOOL_CONFIG"
echo " Reward:        $REWARD_PATH (DPO_MODE=$DPO_MODE)"
echo " N_SAMPLES:     $N_SAMPLES (best_vs_worst -> 1 pair/prompt)"
echo " Temperature:   $TEMPERATURE"
echo " GPUs:          $nproc_per_node"
echo " Save path:     $save_path"
echo "========================================"

ulimit -n 65535 2>/dev/null || true

# ---- Dataset-specific tool config (same as reject_sampling) ----
INTERACTION_ARGS=()
case "$dataset" in
    toolmind)
        # ToolMind uses generic tools (calculator, search, code_runner, submit_answer)
        ;;
    terminaltraj)
        # TerminalTraj uses the bash tool
        ;;
    open_swe_traces|swe_zero)
        # SWE datasets use bash + repo interaction
        INTERACTION_ARGS+=(
            "actor_rollout_ref.rollout.multi_turn.interaction_config_path=$REJECT_DIR/tools/swe_interaction_config.yaml"
        )
        ;;
    *)
        echo "Unknown dataset: $dataset (choose: toolmind, terminaltraj, open_swe_traces, swe_zero)"
        exit 1
        ;;
esac

# ---- Launch Online DPO training ----
python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='online_dpo' \
    algorithm.adv_estimator=grpo \
    algorithm.use_dpo=True \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=15 \
    actor_rollout_ref.rollout.n="$N_SAMPLES" \
    actor_rollout_ref.rollout.temperature="$TEMPERATURE" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.actor.policy_loss.loss_mode=dpo \
    actor_rollout_ref.actor.policy_loss.dpo_beta="${DPO_BETA:-0.1}" \
    actor_rollout_ref.actor.optim.lr="${DPO_LR:-1e-6}" \
    reward.custom_reward_function.path="$REWARD_PATH" \
    reward.custom_reward_function.name=compute_score \
    trainer.default_local_dir="$save_path" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node="$nproc_per_node" \
    trainer.nnodes=1 \
    "${INTERACTION_ARGS[@]}" \
    "$@"

echo ""
echo "========================================"
echo " Online DPO training complete!"
echo "========================================"
echo " Checkpoints saved to: $save_path"
