#!/usr/bin/env bash
# Hermes Gateway DPO Training
# =============================================================================
#
# Gateway-based blackbox agent training: the Hermes-format agent runs in a
# local workspace, calls the Gateway for model inference, and the Gateway
# captures full token-level trajectories.
#
# Architecture:
#   Agent (hermes_entrypoint.py) → Gateway (FastAPI) → vLLM (Qwen3-4B)
#   Agent executes tools in workspace → loop until complete
#   Gateway captures trajectories → Judge scores → DPO update
#
# =============================================================================
# Usage
# =============================================================================
#
#   export DEEPSEEK_API_KEY=sk-xxx
#   bash run_hermes_gateway_dpo.sh <dataset> 8 ~/ckpt/hermes-gateway
#
# =============================================================================
# Configuration (all overridable via env vars)
# =============================================================================
#
# --- Model & Data ---
#   MODEL_PATH           (default: $HOME/models/Qwen3-4B)
#   TRAIN_DATA           (default: $HOME/data/online_dpo/prompts/train.parquet)
#   VAL_DATA             (default: $HOME/data/online_dpo/prompts/val.parquet)
#   N_SAMPLES            Rollouts per prompt (default: 4)
#
# --- Judge ---
#   DEEPSEEK_API_KEY     (required)
#   JUDGE_MODEL          (default: deepseek-chat)
#   JUDGE_BASE_URL       (default: https://api.deepseek.com)
#
# --- Agent ---
#   AGENT_MAX_TURNS      Max conversation turns (default: 100)
#   AGENT_TIMEOUT        Max seconds per agent run (default: 3600)
#   HERMES_WORKSPACE_ROOT Workspace base dir (default: /tmp/verl_hermes)
#
# --- GPU ---
#   Default: single 8-GPU node, 4 trainer + 4 rollout (separate_async).
#   Override with NNODES, N_GPUS_PER_NODE, ROLLOUT_NGPUS_PER_NODE.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${REPO_ROOT}"

# ---- CLI args ----
dataset="${1:?Usage: run_hermes_gateway_dpo.sh <dataset> <nproc> <save_path> [extra...]}"
nproc_per_node="${2:?}"
save_path="${3:?}"
shift 3 2>/dev/null || shift $#

# ---- Model & data ----
MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3-4B}"
TRAIN_DATA="${TRAIN_DATA:-$HOME/data/online_dpo/prompts/${dataset}.parquet}"
VAL_DATA="${VAL_DATA:-$HOME/data/online_dpo/prompts/val.parquet}"
N_SAMPLES="${N_SAMPLES:-4}"
TEMPERATURE="${TEMPERATURE:-0.7}"
DPO_BETA="${DPO_BETA:-0.1}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-20480}"
PROMPT_LENGTH="${PROMPT_LENGTH:-4096}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-16384}"

# ---- Trainer ----
TRAINER_MODE="${TRAINER_MODE:-separate_async}"
NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES:-1}"
SEPARATE_NUM_WARMUP_BATCHES="${SEPARATE_NUM_WARMUP_BATCHES:-1}"
PARAMETER_SYNC_STEP="${PARAMETER_SYNC_STEP:-4}"

# ---- Hardware ----
NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
ROLLOUT_NGPUS_PER_NODE="${ROLLOUT_NGPUS_PER_NODE:-4}"
GEN_TP="${GEN_TP:-${ROLLOUT_NGPUS_PER_NODE}}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.7}"
UPDATE_WEIGHTS_BUCKET_MB="${UPDATE_WEIGHTS_BUCKET_MB:-2048}"

# ---- Algorithm ----
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${PPO_MINI_BATCH_SIZE}}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"

# ---- Agent ----
AGENT_MAX_TURNS="${AGENT_MAX_TURNS:-100}"
AGENT_TIMEOUT="${AGENT_TIMEOUT:-3600}"
HERMES_WORKSPACE_ROOT="${HERMES_WORKSPACE_ROOT:-/tmp/verl_hermes}"
GATEWAY_COUNT="${GATEWAY_COUNT:-1}"
MAX_CONCURRENT_SESSIONS="${MAX_CONCURRENT_SESSIONS:-32}"
NUM_AGENT_WORKERS="${NUM_AGENT_WORKERS:-8}"

# ---- Judge ----
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-chat}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://api.deepseek.com}"
JUDGE_API_KEY="${JUDGE_API_KEY:-${DEEPSEEK_API_KEY:-}}"

# ---- Logging ----
PROJECT_NAME="${PROJECT_NAME:-hermes-gateway-dpo}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-hermes-dpo-${dataset}-$(date '+%m%d-%H%M')}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-10}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-10}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
CKPTS_DIR="${CKPTS_DIR:-${save_path}}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:--1}"

# ---- Validation ----
if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: Train data not found: $TRAIN_DATA"
    exit 1
fi
if [ -z "$JUDGE_API_KEY" ]; then
    echo "ERROR: DEEPSEEK_API_KEY or JUDGE_API_KEY must be set"
    exit 1
fi

# ---- Environment ----
export DEEPSEEK_API_KEY="${JUDGE_API_KEY}"
export JUDGE_MODEL JUDGE_BASE_URL
export AGENT_MAX_TURNS AGENT_TIMEOUT
export HERMES_WORKSPACE_ROOT
export GATEWAY_COUNT

# Add uni-agent + verl to PYTHONPATH
UNI_AGENT_ROOT="${UNI_AGENT_ROOT:-$HOME/workspace/uni-agent}"
export PYTHONPATH="${REPO_ROOT}:${UNI_AGENT_ROOT}:${UNI_AGENT_ROOT}/verl:${PYTHONPATH:-}"

mkdir -p "$HERMES_WORKSPACE_ROOT"
mkdir -p "$CKPTS_DIR"
ulimit -n 65535 2>/dev/null || true

# ---- Print ----
echo "========================================"
echo " Hermes Gateway DPO Training"
echo "========================================"
echo " Dataset:      $dataset ($TRAIN_DATA)"
echo " Model:        $MODEL_PATH"
echo " Engine:       vllm (gen_tp=$GEN_TP)"
echo " Workspace:    $HERMES_WORKSPACE_ROOT"
echo " Max turns:    $AGENT_MAX_TURNS"
echo " N samples:    $N_SAMPLES"
echo " Temperature:  $TEMPERATURE"
echo " DPO beta:     $DPO_BETA"
echo " Actor lr:     $ACTOR_LR"
echo " Batch:        n=$N_SAMPLES, mini_bsz=$PPO_MINI_BATCH_SIZE"
echo " Sequence:     prompt=$PROMPT_LENGTH, response=$RESPONSE_LENGTH"
echo " Trainer:      V1 $TRAINER_MODE"
echo " Resources:    trainer=${NNODES}x${N_GPUS_PER_NODE}, rollout=${ROLLOUT_NGPUS_PER_NODE}"
echo " Save path:    $CKPTS_DIR"
echo "========================================"

# ---- Launch ----
CONFIG_DIR="${SCRIPT_DIR}/config"
CONFIG_NAME="${CONFIG_NAME:-agent_hermes_gateway}"

# Compute total GPUs for Ray
if [[ "${TRAINER_MODE}" == "separate_async" ]]; then
    TOTAL_GPUS=$(( NNODES * N_GPUS_PER_NODE + NNODES * ROLLOUT_NGPUS_PER_NODE ))
else
    TOTAL_GPUS=$(( NNODES * N_GPUS_PER_NODE ))
fi

# Start Ray if not running
if ! timeout 5 ray status &>/dev/null; then
    echo "Starting Ray cluster (${TOTAL_GPUS} GPUs)..."
    ray start --head --num-gpus="${TOTAL_GPUS}" --disable-usage-stats
else
    echo "Ray cluster already running."
fi

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_DIR" \
    --config-name="$CONFIG_NAME" \
    hydra.searchpath="[pkg://verl.trainer.config]" \
    +ray_kwargs.ray_init.address="auto" \
    trainer.use_v1=true \
    "trainer.v1.trainer_mode=${TRAINER_MODE}" \
    "trainer.v1.colocate_async.num_warmup_batches=${NUM_WARMUP_BATCHES}" \
    "trainer.v1.separate_async.num_warmup_batches=${SEPARATE_NUM_WARMUP_BATCHES}" \
    "trainer.v1.separate_async.parameter_sync_step=${PARAMETER_SYNC_STEP}" \
    transfer_queue.enable=true \
    "actor_rollout_ref.model.path=${MODEL_PATH}" \
    "data.train_files=['${TRAIN_DATA}']" \
    "data.val_files=['${VAL_DATA}']" \
    "data.train_max_samples=${TRAIN_MAX_SAMPLES}" \
    "data.val_max_samples=${VAL_MAX_SAMPLES}" \
    "data.train_batch_size=${TRAIN_BATCH_SIZE}" \
    "data.val_batch_size=${VAL_BATCH_SIZE}" \
    "data.max_prompt_length=${PROMPT_LENGTH}" \
    "data.max_response_length=${RESPONSE_LENGTH}" \
    "actor_rollout_ref.rollout.n=${N_SAMPLES}" \
    "actor_rollout_ref.rollout.name=vllm" \
    "actor_rollout_ref.rollout.prompt_length=${PROMPT_LENGTH}" \
    "actor_rollout_ref.rollout.response_length=${RESPONSE_LENGTH}" \
    "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}" \
    "actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_MODEL_LEN}" \
    "actor_rollout_ref.rollout.temperature=${TEMPERATURE}" \
    "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${UPDATE_WEIGHTS_BUCKET_MB}" \
    "actor_rollout_ref.rollout.nnodes=${NNODES}" \
    "actor_rollout_ref.rollout.n_gpus_per_node=${ROLLOUT_NGPUS_PER_NODE}" \
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}" \
    "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}" \
    "actor_rollout_ref.rollout.agent.num_workers=${NUM_AGENT_WORKERS}" \
    "actor_rollout_ref.rollout.custom.agent_framework.gateway_count=${GATEWAY_COUNT}" \
    "actor_rollout_ref.rollout.custom.agent_framework.agent_runners.custom_hermes.max_concurrent_sessions=${MAX_CONCURRENT_SESSIONS}" \
    "actor_rollout_ref.rollout.custom.agent_framework.agent_runners.custom_hermes.runner_kwargs.agent_max_turns=${AGENT_MAX_TURNS}" \
    "actor_rollout_ref.rollout.custom.agent_framework.agent_runners.custom_hermes.runner_kwargs.agent_timeout=${AGENT_TIMEOUT}" \
    "actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}" \
    "actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}" \
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}" \
    "actor_rollout_ref.actor.optim.lr=${ACTOR_LR}" \
    "actor_rollout_ref.actor.policy_loss.loss_mode=dpo" \
    "actor_rollout_ref.actor.policy_loss.dpo_beta=${DPO_BETA}" \
    "algorithm.adv_estimator=grpo" \
    "algorithm.use_dpo=true" \
    "trainer.project_name=${PROJECT_NAME}" \
    "trainer.experiment_name=${EXPERIMENT_NAME}" \
    "trainer.total_epochs=${TOTAL_EPOCHS}" \
    "trainer.val_before_train=${VAL_BEFORE_TRAIN}" \
    "trainer.save_freq=${SAVE_FREQ}" \
    "trainer.test_freq=${TEST_FREQ}" \
    "trainer.default_local_dir=${CKPTS_DIR}" \
    "trainer.nnodes=${NNODES}" \
    "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}" \
    "$@"

echo ""
echo "========================================"
echo " Hermes Gateway DPO training complete!"
echo " Checkpoints: $CKPTS_DIR"
echo "========================================"
