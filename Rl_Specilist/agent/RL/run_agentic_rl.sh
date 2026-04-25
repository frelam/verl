#!/usr/bin/env bash
# Agentic RL | Multi-turn tool calling | GRPO | Qwen3
#
# This script launches a full agentic RL training run using verl's
# ToolAgentLoop. The agent learns to:
#   - reason inside <think>...</think>
#   - call the `calculator` tool for precise arithmetic
#   - call `submit_answer` with a confidence score
#   - revise after feedback (failure recovery)
#
# Prerequisites:
#   1. Prepare data:  python -m Rl_Specilist.agent.RL.data_preprocess.prepare_math_multiturn
#   2. (Optional) Prepare QA data: python -m Rl_Specilist.agent.RL.data_preprocess.prepare_qa_search
#
# Usage:
#   bash run_agentic_rl.sh <nproc_per_node> <save_path> [dataset] [extra_configs...]
#
# Examples:
#   # GSM8K only, 8 GPUs
#   bash run_agentic_rl.sh 8 ./ckpt/agentic_rl gsm8k
#
#   # MATH only, 4 GPUs
#   bash run_agentic_rl.sh 4 ./ckpt/agentic_rl math
#
#   # Override config
#   bash run_agentic_rl.sh 8 ./ckpt/agentic_rl gsm8k trainer.total_epochs=5

set -xeuo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: run_agentic_rl.sh <nproc_per_node> <save_path> [dataset=gsm8k] [extra_configs...]"
    echo "  dataset: gsm8k | math | qa  (default: gsm8k)"
    exit 1
fi

nproc_per_node=$1
save_path=$2
dataset=${3:-gsm8k}
shift 3 2>/dev/null || shift $#

# ---- user-adjustable ----
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-1.7B}
PROJECT_NAME=${PROJECT_NAME:-agentic-rl}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-agentic-grpo-$(date '+%m%d-%H%M')}
DATA_ROOT=${DATA_ROOT:-$HOME/data}

# Resolve dataset paths
case "$dataset" in
    gsm8k)
        TRAIN_FILE="${DATA_ROOT}/agentic_math/gsm8k/train.parquet"
        VAL_FILE="${DATA_ROOT}/agentic_math/gsm8k/test.parquet"
        ;;
    math)
        TRAIN_FILE="${DATA_ROOT}/agentic_math/math/train.parquet"
        VAL_FILE="${DATA_ROOT}/agentic_math/math/test.parquet"
        ;;
    qa)
        TRAIN_FILE="${DATA_ROOT}/agentic_qa/train.parquet"
        VAL_FILE="${DATA_ROOT}/agentic_qa/test.parquet"
        ;;
    *)
        echo "Unknown dataset: $dataset (choose: gsm8k, math, qa)"
        exit 1
        ;;
esac

# Resolve paths relative to project root
PROJECT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
CONFIG_PATH="$PROJECT_DIR/Rl_Specilist/agent/RL/config"
TOOL_CONFIG="$PROJECT_DIR/Rl_Specilist/agent/RL/tools/tool_config.yaml"
REWARD_PATH="$PROJECT_DIR/Rl_Specilist/agent/RL/reward/agentic_reward.py"
# ---- end user-adjustable ----

echo "========================================"
echo " Dataset:      $dataset"
echo " Train file:   $TRAIN_FILE"
echo " Val file:     $VAL_FILE"
echo " Model:        $MODEL_PATH"
echo " Tool config:  $TOOL_CONFIG"
echo " Reward:       $REWARD_PATH"
echo " Save path:    $save_path"
echo " GPUs:         $nproc_per_node"
echo "========================================"

# Verify files exist
if [ ! -f "$TRAIN_FILE" ]; then
    echo "ERROR: Train file not found: $TRAIN_FILE"
    echo "Run data preparation first:"
    echo "  python -m Rl_Specilist.agent.RL.data_preprocess.prepare_math_multiturn"
    exit 1
fi
if [ ! -f "$TOOL_CONFIG" ]; then
    echo "ERROR: Tool config not found: $TOOL_CONFIG"
    exit 1
fi

ulimit -n 65535

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='agentic_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=6 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=8 \
    reward.custom_reward_function.path="$REWARD_PATH" \
    reward.custom_reward_function.name=compute_score \
    trainer.default_local_dir="$save_path" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node="$nproc_per_node" \
    trainer.nnodes=1 \
    "$@"
