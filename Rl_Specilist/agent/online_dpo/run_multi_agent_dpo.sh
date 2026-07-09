#!/usr/bin/env bash
# Online DPO Training with Sandbox Tools
# =============================================================================
#
# Verl 模型 (Qwen3-4B) 充当 assistant，sandbox 工具充当执行环境。
# 模型生成 tool_calls → 工具在隔离 workspace 执行 → 返回 observation →
# 模型继续生成 → ... → Judge 打分 → DPO 更新权重 → 下一轮 rollout。
#
# 每步更新后，下一轮 rollout 使用最新的模型权重。
#
# =============================================================================
# Usage
# =============================================================================
#
#   # 默认 sandbox 工具（bash, read_file, write_file, submit_answer）
#   bash run_multi_agent_dpo.sh toolmind 8 ~/ckpt
#
#   # 自定义工具配置
#   bash run_multi_agent_dpo.sh terminaltraj 8 ~/ckpt \
#       --tool-config config/tool_config_sandbox.yaml
#
# =============================================================================
# Configuration
# =============================================================================
#
# --- Model & Data ---
#   MODEL_PATH           (default: $HOME/models/Qwen3-4B)
#   DATA_DIR             (default: $HOME/data/online_dpo)
#   N_SAMPLES            Rollouts per prompt (default: 4)
#
# --- Judge ---
#   DEEPSEEK_API_KEY     (required)
#   JUDGE_MODEL          (default: deepseek-chat)
#   JUDGE_BASE_URL       (default: https://api.deepseek.com)
#
# --- Sandbox ---
#   SANDBOX_WORKSPACE    Workspace base dir (default: /tmp/verl_sandbox)

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REJECT_DIR="$SCRIPT_DIR/../reject_sampling"

# shellcheck disable=SC1091
source "$REJECT_DIR/setup/env.sh" 2>/dev/null || true

# ---- CLI args ----
dataset="${1:?Usage: run_multi_agent_dpo.sh <dataset> <nproc> <save_path> [--tool-config <yaml>] [extra...]}"
nproc_per_node="${2:?}"
save_path="${3:?}"
shift 3 2>/dev/null || shift $#

tool_config=""
extra_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tool-config) tool_config="${2:?}"; shift 2 ;;
        *) extra_args+=("$1"); shift ;;
    esac
done

# ---- Config ----
MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3-4B}"
DATA_DIR="${DATA_DIR:-$HOME/data/online_dpo}"
N_SAMPLES="${N_SAMPLES:-4}"
TEMPERATURE="${TEMPERATURE:-0.7}"
DPO_BETA="${DPO_BETA:-0.1}"
DPO_LR="${DPO_LR:-1e-6}"
MAX_TURNS="${MAX_TURNS:-15}"
PROJECT_NAME="${PROJECT_NAME:-online-dpo-sandbox}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-dpo-${dataset}-$(date '+%m%d-%H%M')}"

# Judge
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-chat}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://api.deepseek.com}"
JUDGE_API_KEY="${JUDGE_API_KEY:-${DEEPSEEK_API_KEY:-}}"

# Sandbox
SANDBOX_WORKSPACE="${SANDBOX_WORKSPACE:-/tmp/verl_sandbox}"

# ---- Resolve paths ----
if [ -z "$tool_config" ]; then
    tool_config="$SCRIPT_DIR/config/tool_config_sandbox.yaml"
fi

TRAIN_FILE="${DATA_DIR}/prompts/${dataset}.parquet"
if [ ! -f "$TRAIN_FILE" ]; then
    TRAIN_FILE="${DATA_DIR}/prompts/train.parquet"
fi
REWARD_PATH="verl.utils.reward_score.llm_judge_reward"

# ---- Validation ----
if [ ! -f "$TRAIN_FILE" ]; then
    echo "ERROR: Prompt file not found: $TRAIN_FILE"
    exit 1
fi
if [ ! -f "$tool_config" ]; then
    echo "ERROR: Tool config not found: $tool_config"
    exit 1
fi
if [ -z "$JUDGE_API_KEY" ]; then
    echo "ERROR: DEEPSEEK_API_KEY must be set"
    exit 1
fi

# ---- Print ----
echo "========================================"
echo " Online DPO — Sandbox Tools"
echo "========================================"
echo " Dataset:      $dataset"
echo " Model:        $MODEL_PATH"
echo " Tool config:  $tool_config"
echo " Workspace:    $SANDBOX_WORKSPACE"
echo " Max turns:    $MAX_TURNS"
echo " N samples:    $N_SAMPLES"
echo " Temperature:  $TEMPERATURE"
echo " DPO beta:     $DPO_BETA"
echo " DPO lr:       $DPO_LR"
echo " GPUs:         $nproc_per_node"
echo " Save path:    $save_path"
echo "========================================"

export JUDGE_MODEL JUDGE_BASE_URL JUDGE_API_KEY
mkdir -p "$SANDBOX_WORKSPACE"
ulimit -n 65535 2>/dev/null || true

# ---- Launch ----
python3 -m verl.trainer.main_ppo \
    --config-path="$SCRIPT_DIR/config" \
    --config-name='agent_dpo_judge' \
    algorithm.adv_estimator=grpo \
    algorithm.use_dpo=True \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$TRAIN_FILE" \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$tool_config" \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_TURNS" \
    actor_rollout_ref.rollout.multi_turn.max_user_turns="$MAX_TURNS" \
    actor_rollout_ref.rollout.n="$N_SAMPLES" \
    actor_rollout_ref.rollout.temperature="$TEMPERATURE" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.actor.policy_loss.loss_mode=dpo \
    actor_rollout_ref.actor.policy_loss.dpo_beta="$DPO_BETA" \
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
    "${extra_args[@]}"

echo ""
echo "========================================"
echo " Online DPO complete!"
echo " Checkpoints: $save_path"
echo "========================================"
