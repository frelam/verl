#!/usr/bin/env bash
# Reject sampling rollout: generate trajectories with Qwen3-4B in real environments.
#
# This script runs verl's main_ppo with lr=0, so it only does rollout (no training).
# Trajectories are judged by DeepSeek API and saved to disk for SFT.
#
# Usage:
#   bash run_reject_sampling.sh <dataset> <nproc_per_node> <save_path> [extra_configs...]
#
# Examples:
#   # ToolMind, 8 GPUs
#   bash run_reject_sampling.sh toolmind 8 ~/data/reject_sampling/ckpt
#
#   # TerminalTraj, 4 GPUs, override sampling
#   bash run_reject_sampling.sh terminaltraj 4 ~/data/reject_sampling/ckpt \
#       actor_rollout_ref.rollout.n=16
#
#   # SWE dataset with custom model
#   MODEL_PATH=/path/to/model bash run_reject_sampling.sh open_swe_traces 8 ~/data/ckpt

set -xeuo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: run_reject_sampling.sh <dataset> <nproc_per_node> <save_path> [extra_configs...]"
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

# ---- Load env ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/setup/env.sh" 2>/dev/null || true

# ---- User-adjustable ----
MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3-4B}"
DATA_DIR="${DATA_DIR:-$HOME/data/reject_sampling}"
N_SAMPLES="${N_SAMPLES:-8}"
TEMPERATURE="${TEMPERATURE:-0.7}"
PROJECT_NAME="${PROJECT_NAME:-reject-sampling}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-reject-$dataset-$(date '+%m%d-%H%M')}"

# Resolve paths
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_PATH="$SCRIPT_DIR/config"
TOOL_CONFIG="$SCRIPT_DIR/tools/tool_config.yaml"
REWARD_PATH="Rl_Specilist.agent.reject_sampling.reward.judge_reward"

# Dataset-specific prompt file
TRAIN_FILE="${DATA_DIR}/prompts/${dataset}.parquet"
if [ ! -f "$TRAIN_FILE" ]; then
    # Fall back to combined train.parquet
    TRAIN_FILE="${DATA_DIR}/prompts/train.parquet"
fi

# Val file (not used for reject sampling, but main_ppo may require it)
VAL_FILE="$TRAIN_FILE"  # Use same file; val_before_train=False

# ---- Verify prerequisites ----
if [ ! -f "$TRAIN_FILE" ]; then
    echo "ERROR: Prompt file not found: $TRAIN_FILE"
    echo "Run data preparation first:"
    echo "  bash $SCRIPT_DIR/setup/download_datasets.sh"
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
echo " Reject Sampling Rollout"
echo "========================================"
echo " Dataset:       $dataset"
echo " Prompt file:   $TRAIN_FILE"
echo " Model:         $MODEL_PATH"
echo " Tool config:   $TOOL_CONFIG"
echo " Reward:        $REWARD_PATH"
echo " N_SAMPLES:     $N_SAMPLES"
echo " Temperature:   $TEMPERATURE"
echo " GPUs:          $nproc_per_node"
echo " Save path:     $save_path"
echo " Trajectory:    ${TRAJECTORY_FILE:-$DATA_DIR/collected_trajectories.jsonl}"
echo "========================================"

ulimit -n 65535 2>/dev/null || true

# ---- Dataset-specific tool config ----
# For SWE datasets, we need the interaction config in addition to tools
INTERACTION_ARGS=()
case "$dataset" in
    toolmind)
        # ToolMind uses generic tools (calculator, search, code_runner, submit_answer)
        # tool_config.yaml already has these
        ;;
    terminaltraj)
        # TerminalTraj uses the bash tool
        # tool_config.yaml already has bash
        ;;
    open_swe_traces|swe_zero)
        # SWE datasets use bash + repo interaction
        # The swe_bench_interaction.py is loaded via interaction_config
        INTERACTION_ARGS+=(
            "actor_rollout_ref.rollout.multi_turn.interaction_config_path=$SCRIPT_DIR/tools/swe_interaction_config.yaml"
        )
        ;;
    *)
        echo "Unknown dataset: $dataset (choose: toolmind, terminaltraj, open_swe_traces, swe_zero)"
        exit 1
        ;;
esac

# ---- Launch rollout ----
python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='reject_sampling' \
    algorithm.adv_estimator=grpo \
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
    actor_rollout_ref.actor.optim.lr=0 \
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
echo " Rollout complete!"
echo "========================================"
echo " Trajectories saved to: ${TRAJECTORY_FILE:-$DATA_DIR/collected_trajectories.jsonl}"
echo ""
echo "Next steps:"
echo "  1. Filter + convert to SFT format:"
echo "     python -m Rl_Specilist.agent.reject_sampling.data_preprocess.convert_to_sft"
echo ""
echo "  2. Run SFT training:"
echo "     bash $SCRIPT_DIR/run_sft_from_reject.sh $nproc_per_node ~/data/sft_ckpt"
