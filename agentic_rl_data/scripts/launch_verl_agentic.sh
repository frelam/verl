#!/usr/bin/env bash
# Agentic RL launch for **verl** (multi-turn tool calling / search / python / code).
#
# This is the verl counterpart of the data plan's §9 (the plan was written for
# slime; this repo trains with verl, so the framework mapping lives here).
#
# Pipeline:
#   agentic_rl_data/data/verl/{train,heldout}.parquet  (RLHFDataset rows)
#   -> agent loop "agentic_agent" (agentic_rl_data.verl.agent_loop_agentic)
#      drives envs (mock_fc / search / python / code_tests), records the
#      trajectory, computes the rule reward INSIDE the loop (reward_score),
#   -> GRPO update.
#
# Single-turn task types (st_fc/irrelevance/abstain/search_qa/math_ci) are scored
# by the fallback custom reward function (verl_reward_bridge.py); mt_fc /
# code_exec need env_state and are always scored in the loop.
#
# Data preparation (one-time):
#   python agentic_rl_data/scripts/build_data.ps1 ...        (build mixture)
#   python -m agentic_rl_data.scripts.export_verl_parquet --mixture-dir agentic_rl_data/data/mixture_v1
#
# Optional env knobs:
#   MODEL_PATH, TRAIN_FILES, VAL_FILES
#   TOOL_RL_VLLM=1            use vLLM (default; required for async agent loop here)
#   MAX_TURNS=15              multi-turn caps (also read per-sample from the task)
#   FORMAT=hermes             tool parser format (hermes / qwen3_coder / ...)
#   SMOKE=1                   tiny run: 1 epoch, small batches, short responses
#   TRAIN_BATCH_SIZE, ROLLOUT_N, VAL_N, TOTAL_EPOCHS, LR
#   USE_V1=1                  V1 trainer (DAPO filter_groups / hard replay hooks)
#   WANDB=1                   enable wandb logging
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
TRAIN_FILES="${TRAIN_FILES:-agentic_rl_data/data/verl/train.parquet}"
VAL_FILES="${VAL_FILES:-agentic_rl_data/data/verl/heldout.parquet}"
AGENT_LOOP_CONFIG="${AGENT_LOOP_CONFIG:-agentic_rl_data/verl/agent_loop_config.yaml}"
REWARD_BRIDGE="${REWARD_BRIDGE:-agentic_rl_data/scripts/verl_reward_bridge.py}"

SMOKE="${SMOKE:-0}"
if [[ "$SMOKE" == "1" ]]; then
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
  ROLLOUT_N="${ROLLOUT_N:-4}"
  VAL_N="${VAL_N:-4}"
  TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
  MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
  MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
  SAVE_FREQ="${SAVE_FREQ:-1000}"
  TEST_FREQ="${TEST_FREQ:-1000}"
else
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
  ROLLOUT_N="${ROLLOUT_N:-16}"
  VAL_N="${VAL_N:-16}"
  TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
  MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
  MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
  SAVE_FREQ="${SAVE_FREQ:-50}"
  TEST_FREQ="${TEST_FREQ:-10}"
fi

MAX_TURNS="${MAX_TURNS:-15}"
FORMAT="${FORMAT:-hermes}"
USE_V1="${USE_V1:-0}"   # V0 async by default: the V1/TQ extra_fields path is unverified
LR="${LR:-1e-6}"
PROJECT_NAME="${PROJECT_NAME:-verl_agentic_rl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_4b_agentic_rl_$(date +%Y%m%d_%H%M)}"
LOGGERS='["console"]'
if [[ "${WANDB:-0}" == "1" ]]; then
  LOGGERS='["console","wandb"]'
fi

# ---- pre-flight -----------------------------------------------------------
for required in "$TRAIN_FILES" "$VAL_FILES" "$AGENT_LOOP_CONFIG" "$REWARD_BRIDGE"; do
  if [[ ! -f "$required" ]]; then
    echo "[launch] ERROR: missing $required" >&2
    echo "[launch] Build + export data first:" >&2
    echo "  python agentic_rl_data/scripts/build_data.py ... (or build_data.ps1)" >&2
    echo "  python -m agentic_rl_data.scripts.export_verl_parquet --mixture-dir agentic_rl_data/data/mixture_v1" >&2
    exit 1
  fi
done
if [[ ! -f agentic_rl_data/verl/agent_loop_agentic.py ]]; then
  echo "[launch] ERROR: agentic_rl_data/verl/agent_loop_agentic.py not found." >&2
  echo "[launch] The multi-turn agent loop is required for mt_fc/search/python/code_exec." >&2
  exit 1
fi

echo "[launch] model=$MODEL_PATH"
echo "[launch] train=$TRAIN_FILES val=$VAL_FILES"
echo "[launch] agent_loop=agentic_agent config=$AGENT_LOOP_CONFIG format=$FORMAT max_turns=$MAX_TURNS"
echo "[launch] reward: in-loop for env tasks; bridge for single-turn: $REWARD_BRIDGE"

# ---- hydra overrides ------------------------------------------------------
DATA=(
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False
  data.train_files="$TRAIN_FILES"
  data.val_files="$VAL_FILES"
  data.train_batch_size="$TRAIN_BATCH_SIZE"
  data.max_prompt_length="$MAX_PROMPT_LENGTH"
  data.max_response_length="$MAX_RESPONSE_LENGTH"
  # per-sample tool schemas are injected at rollout time (see export notes)
  data.filter_overlong_prompts=False
  data.truncation=error
)

MODEL=(
  actor_rollout_ref.model.path="$MODEL_PATH"
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
  actor_rollout_ref.actor.optim.lr="$LR"
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-$((TRAIN_BATCH_SIZE / 4))}"
  actor_rollout_ref.actor.policy_loss.loss_mode=vanilla
)

ROLLOUT=(
  actor_rollout_ref.rollout.name=vllm
  # async server mode is required for agent loops
  actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.n="$ROLLOUT_N"
  actor_rollout_ref.rollout.val_kwargs.n="$VAL_N"
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP:-1}"
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL:-0.65}"
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.rollout.calculate_log_probs=True
  # multi-turn agent loop: env-driven rollout, reward computed in-loop
  actor_rollout_ref.rollout.agent.default_agent_loop=agentic_agent
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$REPO_ROOT/$AGENT_LOOP_CONFIG"
  actor_rollout_ref.rollout.multi_turn.enable=True
  actor_rollout_ref.rollout.multi_turn.format="$FORMAT"
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_TURNS"
  actor_rollout_ref.rollout.multi_turn.max_user_turns="$((MAX_TURNS + 1))"
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls="${MAX_PARALLEL_CALLS:-4}"
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length="${MAX_TOOL_RESPONSE_LENGTH:-1024}"
)

REWARD=(
  # single-turn fallback; env tasks set reward_score inside the agent loop
  reward.custom_reward_function.path="$REPO_ROOT/$REWARD_BRIDGE"
  reward.custom_reward_function.name=compute_score
  reward.reward_manager.name=naive
)

TRAINER=(
  trainer.use_v1="$([ "$USE_V1" = "1" ] && echo true || echo false)"
  trainer.balance_batch=True
  trainer.logger="$LOGGERS"
  trainer.project_name="$PROJECT_NAME"
  trainer.experiment_name="$EXPERIMENT_NAME"
  trainer.n_gpus_per_node="${NGPUS_PER_NODE:-8}"
  trainer.nnodes="${NNODES:-1}"
  trainer.val_before_train=True
  trainer.save_freq="$SAVE_FREQ"
  trainer.test_freq="$TEST_FREQ"
  trainer.log_val_generations="${LOG_VAL_GENERATIONS:-5}"
  trainer.total_epochs="$TOTAL_EPOCHS"
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
