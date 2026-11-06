#!/usr/bin/env bash
# Agentic RL v2 | Qwen3-4B | GRPO + DAPO filtering | FSDP | vLLM async multi-turn
#
# Data:  bash agentic_rl_data/scripts/build_data_v2.sh   (-> data/verl_v2/*.parquet)
# Design and numbers: agentic_rl_data/reports/AGENTIC_RL_V2.md
#
# Curriculum (short -> long, one launch per stage, each resumes the previous
# checkpoint through trainer.default_local_dir + resume_mode=auto):
#   STAGE=1  train_max_turns8.parquet   MAX_TURNS=8   response 12k   ~40 steps
#   STAGE=2  train_max_turns12.parquet  MAX_TURNS=12  response 20k   ~50 steps
#   STAGE=3  train_max_turns15.parquet  MAX_TURNS=15  response 24k   until held-out plateaus
#   STAGE=all  the full mixture at MAX_TURNS=15 (no curriculum; ablation)
#
# Stability choices (why each knob is set; all overridable via env):
#   * GRPO n=8 + DAPO group filter on reward score: all-0 / all-1 groups have
#     zero advantage and only add noise; the V1 replay buffer refills them.
#   * token-mean loss aggregation + clip 0.2/0.28 (DAPO clip-higher): long
#     multi-turn responses do not dominate, entropy does not collapse.
#   * low-variance KL to the reference at 0.001: weak anchor against format
#     drift over 10-15 turns without blocking learning.
#   * lr 1e-6, no warm restart, grad clip 1.0; ppo_epochs 1 (on-policy).
#   * tool / env tokens are masked out of the loss (response_mask=0 in the
#     agent loop); only assistant tokens get gradients.
#   * per-turn generation cap (AGENTIC_MAX_TOKENS_PER_TURN) so one runaway
#     <think> cannot consume the whole 10-15 turn budget.
#   * rollout_correction token-TIS (threshold 2): vLLM vs FSDP logprob
#     mismatch grows with sequence length; truncated IS keeps it bounded.
#   * thinking ON: reflection / planning live in <think>; the stage-0 tool_rl
#     format reward already taught think -> tool_call layout.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
DATA_DIR="${DATA_DIR:-agentic_rl_data/data/verl_v2}"
STAGE="${STAGE:-1}"
NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"

case "$STAGE" in
  1)   TRAIN_FILE="$DATA_DIR/curriculum/stage8.parquet";  MAX_TURNS="${MAX_TURNS:-8}";  RESP_LEN="${MAX_RESPONSE_LENGTH:-12288}"; STEPS="${TOTAL_STEPS:-40}" ;;
  2)   TRAIN_FILE="$DATA_DIR/curriculum/stage12.parquet"; MAX_TURNS="${MAX_TURNS:-12}"; RESP_LEN="${MAX_RESPONSE_LENGTH:-20480}"; STEPS="${TOTAL_STEPS:-90}" ;;
  3)   TRAIN_FILE="$DATA_DIR/curriculum/stage15.parquet"; MAX_TURNS="${MAX_TURNS:-15}"; RESP_LEN="${MAX_RESPONSE_LENGTH:-24576}"; STEPS="${TOTAL_STEPS:-160}" ;;
  all) TRAIN_FILE="$DATA_DIR/train.parquet";              MAX_TURNS="${MAX_TURNS:-15}"; RESP_LEN="${MAX_RESPONSE_LENGTH:-24576}"; STEPS="${TOTAL_STEPS:-150}" ;;
  *) echo "STAGE must be 1|2|3|all" >&2; exit 2 ;;
esac
VAL_FILE="${VAL_FILE:-$DATA_DIR/heldout.parquet}"
# total_training_steps is cumulative across stages (resume continues the count)

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"      # prompts per step
ROLLOUT_N="${ROLLOUT_N:-8}"                     # 512 trajectories / step
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}" # 4 updates / step
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-6144}"   # system + tool schemas + task
PPO_MAX_TOKEN_LEN="${PPO_MAX_TOKEN_LEN_PER_GPU:-$(( (MAX_PROMPT_LENGTH + RESP_LEN) ))}"
LR="${LR:-1e-6}"
KL_COEF="${KL_COEF:-0.001}"
CLIP_LOW="${CLIP_RATIO_LOW:-0.2}"
CLIP_HIGH="${CLIP_RATIO_HIGH:-0.28}"
TEMPERATURE="${TEMPERATURE:-1.0}"
ENABLE_THINKING="${ENABLE_THINKING:-True}"
export AGENTIC_MAX_TOKENS_PER_TURN="${AGENTIC_MAX_TOKENS_PER_TURN:-3072}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
GPU_MEM="${ROLLOUT_GPU_MEM_UTIL:-0.6}"
SMOKE="${SMOKE:-0}"
if [[ "$SMOKE" == "1" ]]; then
  TRAIN_BATCH_SIZE=8; ROLLOUT_N=4; PPO_MINI_BATCH_SIZE=8; STEPS=3; RESP_LEN=8192
  PPO_MAX_TOKEN_LEN=$(( MAX_PROMPT_LENGTH + RESP_LEN ))
fi

PROJECT_NAME="${PROJECT_NAME:-agentic_rl_v2}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_4b_agentic_v2}"
CKPT_DIR="${CKPT_DIR:-checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME}"
LOGGERS="${LOGGERS:-[\"console\",\"wandb\"]}"

for f in "$TRAIN_FILE" "$VAL_FILE"; do
  [[ -f "$f" ]] || { echo "[run_v2] missing $f; run agentic_rl_data/scripts/build_data_v2.sh" >&2; exit 1; }
done
echo "[run_v2] stage=$STAGE train=$TRAIN_FILE max_turns=$MAX_TURNS response=$RESP_LEN steps<=$STEPS"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=True \
  algorithm.use_kl_in_reward=False \
  algorithm.filter_groups.enable=True \
  algorithm.filter_groups.metric=agentic/reward_score \
  algorithm.rollout_correction.rollout_is=token \
  algorithm.rollout_correction.rollout_is_threshold=2.0 \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$RESP_LEN" \
  data.filter_overlong_prompts=False \
  data.truncation=error \
  data.dataloader_num_workers=0 \
  +data.apply_chat_template_kwargs.enable_thinking="$ENABLE_THINKING" \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="$LR" \
  actor_rollout_ref.actor.optim.weight_decay=0.0 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN" \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.clip_ratio_low="$CLIP_LOW" \
  actor_rollout_ref.actor.clip_ratio_high="$CLIP_HIGH" \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="$KL_COEF" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.temperature="$TEMPERATURE" \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM" \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN" \
  actor_rollout_ref.rollout.val_kwargs.n=4 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.agent.default_agent_loop=agentic_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$REPO_ROOT/agentic_rl_data/verl/agent_loop_config.yaml" \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_TURNS" \
  actor_rollout_ref.rollout.multi_turn.max_user_turns="$((MAX_TURNS + 1))" \
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=4 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
  reward.custom_reward_function.path="$REPO_ROOT/agentic_rl_data/scripts/verl_reward_bridge.py" \
  reward.custom_reward_function.name=compute_score \
  reward.reward_manager.name=naive \
  trainer.use_v1=true \
  trainer.balance_batch=True \
  trainer.logger="$LOGGERS" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.default_local_dir="$CKPT_DIR" \
  trainer.resume_mode=auto \
  trainer.n_gpus_per_node="$NGPUS_PER_NODE" \
  trainer.nnodes="$NNODES" \
  trainer.val_before_train=True \
  trainer.test_freq="${TEST_FREQ:-10}" \
  trainer.save_freq="${SAVE_FREQ:-10}" \
  trainer.total_training_steps="$STEPS" \
  trainer.total_epochs=100 \
  trainer.log_val_generations=8 \
  "$@"
