#!/usr/bin/env bash
# Agent Online DPO/GDPO Training with Configurable LLM Judge
# ===========================================================================
#
# This script runs Online DPO training for agent tasks with an LLM judge
# providing relative batch-level scores. It supports:
#
# 1. DPO mode (use_dpo=True): best-vs-worst pairing + DPO loss
# 2. GDPO mode (adv_estimator=gdpo): multi-dimensional decoupled normalization
#
# Judge Configuration (via environment variables):
#   JUDGE_MODEL           - Judge model name (default: deepseek-chat)
#   JUDGE_BASE_URL        - Judge API base URL (default: https://api.deepseek.com)
#   JUDGE_API_KEY         - API key for the judge service
#   JUDGE_SYSTEM_PROMPT   - "default", "gdpo", path to prompt file, or inline prompt
#   JUDGE_SCORING_MODE    - "relative" (batch comparison) or "absolute"
#   JUDGE_DIMENSIONS      - JSON array of dimension names for GDPO
#   JUDGE_MAX_RETRIES     - Max API retries (default: 3)
#   JUDGE_API_TIMEOUT     - API timeout seconds (default: 120)
#
# Rollout Control:
#   MAX_ASSISTANT_TURNS   - Max model generation turns per trajectory (default: 10)
#   MAX_USER_TURNS        - Max tool execution turns per trajectory (default: 10)
#   N_SAMPLES             - Number of rollouts per prompt (default: 4)
#
# Usage:
#   # Basic DPO with DeepSeek judge (relative scoring)
#   JUDGE_API_KEY=sk-xxx bash run_agent_dpo.sh toolmind 8 ~/ckpt/dpo
#
#   # GDPO with multi-dimensional scoring
#   JUDGE_API_KEY=sk-xxx JUDGE_SYSTEM_PROMPT=gdpo bash run_agent_dpo.sh toolmind 8 ~/ckpt/gdpo
#
#   # DPO with custom system prompt file
#   JUDGE_API_KEY=sk-xxx JUDGE_SYSTEM_PROMPT=./my_prompt.txt bash run_agent_dpo.sh terminaltraj 4 ~/ckpt/dpo
#
#   # DPO with local vLLM judge (localhost)
#   JUDGE_BASE_URL=http://localhost:8000/v1 JUDGE_MODEL=Qwen3-32B bash run_agent_dpo.sh toolmind 8 ~/ckpt/dpo
#
#   # Absolute scoring (no batch comparison, per-trajectory)
#   JUDGE_SCORING_MODE=absolute bash run_agent_dpo.sh toolmind 8 ~/ckpt/dpo
#
#   # GDPO with custom system prompt (focus on specific dimensions)
#   JUDGE_API_KEY=sk-xxx bash run_agent_dpo.sh toolmind 8 ~/ckpt/gdpo \
#       '+algorithm.gdpo_reward_keys=["accuracy_reward","format_reward","efficiency_reward"]' \
#       '+algorithm.gdpo_reward_weights=[0.5,0.2,0.3]'

set -xeuo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: run_agent_dpo.sh <dataset> <nproc_per_node> <save_path> [mode] [extra_configs...]"
    echo ""
    echo "Datasets: toolmind | terminaltraj | open_swe_traces | swe_zero"
    echo "Mode: dpo (default) | gdpo"
    echo ""
    echo "=== Judge Configuration (env vars) ==="
    echo "  JUDGE_MODEL          - Judge model (default: deepseek-chat)"
    echo "  JUDGE_BASE_URL       - API base URL (default: https://api.deepseek.com)"
    echo "  JUDGE_API_KEY        - API key (falls back to DEEPSEEK_API_KEY)"
    echo "  JUDGE_SYSTEM_PROMPT  - 'default', 'gdpo', file path, or inline text"
    echo "  JUDGE_SCORING_MODE   - 'relative' or 'absolute'"
    echo "  JUDGE_DIMENSIONS     - JSON array, e.g. '[\"accuracy_reward\",\"format_reward\"]'"
    echo ""
    echo "=== Rollout Control (env vars) ==="
    echo "  MAX_ASSISTANT_TURNS  - Max model generation turns (default: 10)"
    echo "  MAX_USER_TURNS       - Max tool execution turns (default: 10)"
    echo "  N_SAMPLES            - Rollouts per prompt (default: 4)"
    exit 1
fi

dataset=$1
nproc_per_node=$2
save_path=$3
mode="${4:-dpo}"  # dpo or gdpo
shift 3 2>/dev/null || shift $#

# ---- Environment Setup ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REJECT_DIR="$SCRIPT_DIR/../reject_sampling"
# shellcheck disable=SC1091
source "$REJECT_DIR/setup/env.sh" 2>/dev/null || true

# ---- Judge Configuration ----
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-chat}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://api.deepseek.com}"
JUDGE_API_KEY="${JUDGE_API_KEY:-${DEEPSEEK_API_KEY:-}}"
JUDGE_SYSTEM_PROMPT="${JUDGE_SYSTEM_PROMPT:-default}"
JUDGE_SCORING_MODE="${JUDGE_SCORING_MODE:-relative}"
JUDGE_DIMENSIONS="${JUDGE_DIMENSIONS:-}"
JUDGE_MAX_RETRIES="${JUDGE_MAX_RETRIES:-3}"
JUDGE_API_TIMEOUT="${JUDGE_API_TIMEOUT:-120}"

# ---- Rollout Control ----
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-10}"
MAX_USER_TURNS="${MAX_USER_TURNS:-10}"
N_SAMPLES="${N_SAMPLES:-4}"
TEMPERATURE="${TEMPERATURE:-0.7}"
DPO_BETA="${DPO_BETA:-0.1}"
DPO_LR="${DPO_LR:-1e-6}"

# ---- Model & Data ----
MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3-4B}"
DATA_DIR="${DATA_DIR:-$HOME/data/reject_sampling}"
PROJECT_NAME="${PROJECT_NAME:-agent-dpo-judge}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${mode}-${dataset}-$(date '+%m%d-%H%M')}"

# Resolve paths
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TOOL_CONFIG="$REJECT_DIR/tools/tool_config.yaml"
REWARD_PATH="verl.utils.reward_score.llm_judge_reward"

# Dataset-specific prompt file
TRAIN_FILE="${DATA_DIR}/prompts/${dataset}.parquet"
if [ ! -f "$TRAIN_FILE" ]; then
    TRAIN_FILE="${DATA_DIR}/prompts/train.parquet"
fi

# ---- Validation ----
if [ ! -f "$TRAIN_FILE" ]; then
    echo "ERROR: Prompt file not found: $TRAIN_FILE"
    echo "Run data preparation first."
    exit 1
fi

if [ ! -f "$TOOL_CONFIG" ]; then
    echo "ERROR: Tool config not found: $TOOL_CONFIG"
    exit 1
fi

if [ -z "$JUDGE_API_KEY" ]; then
    echo "ERROR: JUDGE_API_KEY or DEEPSEEK_API_KEY must be set"
    exit 1
fi

echo "========================================"
echo " Agent ${mode^^} with LLM Judge"
echo "========================================"
echo " Dataset:             $dataset"
echo " Prompt file:         $TRAIN_FILE"
echo " Model:               $MODEL_PATH"
echo " Mode:                $mode"
echo " ---"
echo " Judge model:         $JUDGE_MODEL"
echo " Judge base URL:      $JUDGE_BASE_URL"
echo " Judge system prompt: $JUDGE_SYSTEM_PROMPT"
echo " Judge scoring mode:  $JUDGE_SCORING_MODE"
echo " Judge dimensions:    ${JUDGE_DIMENSIONS:-'(auto)'}"
echo " ---"
echo " Max assistant turns: $MAX_ASSISTANT_TURNS"
echo " Max user turns:      $MAX_USER_TURNS"
echo " N samples/prompt:    $N_SAMPLES"
echo " Temperature:         $TEMPERATURE"
echo " DPO beta:            $DPO_BETA"
echo " DPO lr:              $DPO_LR"
echo " ---"
echo " GPUs:                $nproc_per_node"
echo " Save path:           $save_path"
echo "========================================"

# Export judge config for llm_judge_reward.py
export JUDGE_MODEL
export JUDGE_BASE_URL
export JUDGE_API_KEY
export JUDGE_SYSTEM_PROMPT
export JUDGE_SCORING_MODE
export JUDGE_DIMENSIONS
export JUDGE_MAX_RETRIES
export JUDGE_API_TIMEOUT

ulimit -n 65535 2>/dev/null || true

# ---- Config selection ----
if [ "$mode" == "gdpo" ]; then
    CONFIG_NAME="agent_gdpo_judge"
else
    CONFIG_NAME="agent_dpo_judge"
fi

# ---- Dataset-specific tool config ----
INTERACTION_ARGS=()
case "$dataset" in
    toolmind)
        ;;
    terminaltraj)
        ;;
    open_swe_traces|swe_zero)
        INTERACTION_ARGS+=(
            "actor_rollout_ref.rollout.multi_turn.interaction_config_path=$REJECT_DIR/tools/swe_interaction_config.yaml"
        )
        ;;
    *)
        echo "Unknown dataset: $dataset"
        exit 1
        ;;
esac

# ---- Launch ----
python3 -m verl.trainer.main_ppo \
    --config-path="$SCRIPT_DIR/config" \
    --config-name="$CONFIG_NAME" \
    algorithm.adv_estimator="${ADV_ESTIMATOR:-grpo}" \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$TRAIN_FILE" \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_ASSISTANT_TURNS" \
    actor_rollout_ref.rollout.multi_turn.max_user_turns="$MAX_USER_TURNS" \
    actor_rollout_ref.rollout.n="$N_SAMPLES" \
    actor_rollout_ref.rollout.temperature="$TEMPERATURE" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.actor.optim.lr="$DPO_LR" \
    reward.custom_reward_function.path="$REWARD_PATH" \
    reward.custom_reward_function.name=compute_score \
    reward.reward_manager.name=llm_judge \
    reward.use_batch_judge=True \
    trainer.default_local_dir="$save_path" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node="$nproc_per_node" \
    trainer.nnodes=1 \
    "${INTERACTION_ARGS[@]}" \
    "$@"

echo ""
echo "========================================"
echo " Agent ${mode^^} training complete!"
echo " Checkpoints saved to: $save_path"
echo "========================================"
