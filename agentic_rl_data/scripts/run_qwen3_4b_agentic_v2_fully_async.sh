#!/usr/bin/env bash
# Agentic RL v2 | Qwen3-4B | GRPO | FSDP2 | vLLM async multi-turn | FULLY ASYNC
#
# Entry point: verl.experimental.fully_async_policy.fully_async_main (config
# fully_async_ppo_trainer.yaml).  Trainer and Rollouter live in separate Ray
# resource pools and train / generate concurrently; weights are broadcast over
# HCCL by the checkpoint engine every `trigger_parameter_sync_step` rounds.
#
# Data (build first):
#   bash agentic_rl_data/scripts/build_data_v2.sh          -> agentic_rl_data/data/verl_v2
#   -> data/verl_v2/{train,heldout}.parquet
#   -> data/verl_v2/curriculum/{stage8,stage12,stage15}.parquet
# Design/numbers: agentic_rl_data/reports/AGENTIC_RL_V2.md
# Rollout wiring: agentic_rl_data/reports/VERL_INTEGRATION.md
#
# Curriculum (same schedule as the sync recipe, one launch per stage; each stage
# resumes the previous checkpoint through trainer.resume_mode=resume_path):
#   STAGE=1  train_max_turns8.parquet   MAX_TURNS=8   response 32k
#   STAGE=2  train_max_turns12.parquet  MAX_TURNS=12  response 32k
#   STAGE=3  train_max_turns15.parquet  MAX_TURNS=15  response 32k
#   STAGE=all the full mixture (no curriculum; default)
#
# Context budget (longer than the reasoning-RL recipe on purpose: 10-15 turns
# x up to AGENTIC_MAX_TOKENS_PER_TURN of thinking + tool observations):
#   MAX_PROMPT_LENGTH 8192 + MAX_RESPONSE_LENGTH 32768 = 40960 tokens, which is
#   exactly Qwen3-4B's native max_position_embeddings -- no YaRN needed.  If you
#   raise it further, point MODEL_PATH at a YaRN-extended twin (rope factor 8,
#   max_position 262144) or vLLM silently clamps / the trainer position ids
#   overflow.
#   ppo_max_token_len_per_gpu * SP must be >= prompt+response (checked below);
#   SP=4 packs 4 x 12800 = 51200, i.e. one full 40960 sequence fits per group.
#
# Fully-async knobs and why:
#   * data.train_batch_size=0 / gen_batch_size=1 are asserted by the Rollouter:
#     prompts are streamed one at a time into the message queue.
#   * staleness_threshold=0.5 + partial_rollout=True: up to 50% of a fetched
#     batch may predate the current weight version (the async speed/quality
#     trade-off).  STALENESS_THRESHOLD=0 PARTIAL_ROLLOUT=False reproduces a
#     synchronous stream when debugging a regression.
#   * old_log_prob comes from the ROLLOUTER (use_rollout_log_probs=True +
#     algorithm.rollout_correction.bypass_mode=True), so the PPO ratio is an IS
#     ratio pi_theta / pi_rollout.  Rollout-rejection-sampling masks sequences
#     whose per-sequence log-ratio drifted too far.
#   * NOT available in the streaming pipeline: DAPO filter_groups and tiered
#     hard replay (the sync recipe's all-0/all-1 filtering).  Uniform groups are
#     trained on once; set FILTER_GROUPS via the sync recipe if you need it.
#   * tool / env observation tokens are masked out of the loss (response_mask=0
#     in the agent loop); per-turn generation is capped by
#     AGENTIC_MAX_TOKENS_PER_TURN so one runaway <think> cannot eat the budget.
#   * reward: mt_fc / code_exec need env_state, so the agent loop computes
#     reward_score itself and verl's _postprocess turns it into rm_scores
#     (reward manager short-circuits).  Single-turn task types without env get
#     the same treatment; reward.custom_reward_function is the fail-closed
#     bridge for rows the loop could not build a task for.
#
# Usage:
#   bash agentic_rl_data/scripts/run_qwen3_4b_agentic_v2_fully_async.sh [hydra overrides]
#   STAGE=2 RESUME_PATH=<stage1 ckpt dir> bash ...
#
# Everything below is overridable from the environment.
set -euo pipefail

# ---------------------------------------------------------------- environment
# vLLM V1 engine is required for async (server-mode) rollout.
export VLLM_USE_V1=1
# Ascend / HCCL runtime knobs (single node, disaggregated pools).
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-3600}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-3600}
export HCCL_ASYNC_ERROR_HANDLING=${HCCL_ASYNC_ERROR_HANDLING:-0}
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-auto}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-auto}
# Ray must not overwrite the NPU visibility set by the launcher.
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES:-1}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
# torch.compile / ACL graph are CUDA-oriented and break BiShengHIR on Ascend.
export TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE:-1}
export VLLM_TORCH_COMPILE=${VLLM_TORCH_COMPILE:-0}
export NPU_GRAPH_CAPTURE_DISABLE=${NPU_GRAPH_CAPTURE_DISABLE:-1}
export VLLM_ASCEND_ENABLE_NZ=${VLLM_ASCEND_ENABLE_NZ:-0}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_NO_SUBCLASS_INSPECT=${VLLM_NO_SUBCLASS_INSPECT:-1}
export TOKENIZERS_PARALLELISM=false
ulimit -s unlimited 2>/dev/null || true

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Interpreter: on an Ascend image torch/torch_npu/vllm-ascend may live in the
# system python, while a bare `python3` in a login shell can resolve to a venv
# without them.  Override with PYTHON_BIN if your environment differs.
if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x /usr/local/python3.12.13/bin/python ]; then
    PYTHON_BIN=/usr/local/python3.12.13/bin/python
  else
    PYTHON_BIN=python3
  fi
fi

# ------------------------------------------------------------------- knobs
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
DATA_DIR=${DATA_DIR:-$REPO_ROOT/agentic_rl_data/data/verl_v2}
STAGE=${STAGE:-all}

NNODES_TRAIN=${NNODES_TRAIN:-1}
NNODES_ROLLOUT=${NNODES_ROLLOUT:-1}
NGPUS_TRAIN=${NGPUS_TRAIN:-8}
NGPUS_ROLLOUT=${NGPUS_ROLLOUT:-8}

case "$STAGE" in
  1)   TRAIN_FILE="$DATA_DIR/curriculum/stage8.parquet";  MAX_TURNS="${MAX_TURNS:-8}"  ;;
  2)   TRAIN_FILE="$DATA_DIR/curriculum/stage12.parquet"; MAX_TURNS="${MAX_TURNS:-12}" ;;
  3)   TRAIN_FILE="$DATA_DIR/curriculum/stage15.parquet"; MAX_TURNS="${MAX_TURNS:-15}" ;;
  all) TRAIN_FILE="$DATA_DIR/train.parquet";              MAX_TURNS="${MAX_TURNS:-15}" ;;
  *) echo "STAGE must be 1|2|3|all" >&2; exit 2 ;;
esac
VAL_FILE=${VAL_FILE:-$DATA_DIR/heldout.parquet}

max_prompt_length=${MAX_PROMPT_LENGTH:-8192}
max_response_length=${MAX_RESPONSE_LENGTH:-32768}
sp_size=${SP_SIZE:-4}
# dynamic-bsz packing budget per NPU; must cover one (prompt+response) sequence
# per SP group: ppo_max_token_len_per_gpu * sp_size >= prompt+response.
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-12800}

rollout_n=${ROLLOUT_N:-8}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-16}
require_batches=${REQUIRE_BATCHES:-4}
trigger_parameter_sync_step=${TRIGGER_PARAMETER_SYNC_STEP:-1}
staleness_threshold=${STALENESS_THRESHOLD:-0.5}
partial_rollout=${PARTIAL_ROLLOUT:-True}

actor_lr=${ACTOR_LR:-1e-6}
weight_decay=${WEIGHT_DECAY:-0.0}
clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}
entropy_coeff=${ENTROPY_COEFF:-0}
use_kl_loss=${USE_KL_LOSS:-0}
kl_loss_coef=${KL_LOSS_COEF:-0.001}

rollout_tp=${ROLLOUT_TP:-1}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.8}
max_num_seqs=${MAX_NUM_SEQS:-32}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-$((max_prompt_length + max_response_length))}
enforce_eager=${ENFORCE_EAGER:-False}
nccl_timeout=${NCCL_TIMEOUT:-7200}
temperature=${TEMPERATURE:-1.0}
val_n=${VAL_N:-1}

# Prompt budget for the whole run (fully-async counts PROMPTS, not steps).
# 6000-row mixture x ROLLOUT_N=8 -> 48000 trajectories per epoch.
total_rollout_steps=${TOTAL_ROLLOUT_STEPS:-9000}
total_epochs=${TOTAL_EPOCHS:-10}
save_freq=${SAVE_FREQ:-5}
test_freq=${TEST_FREQ:-5}
log_val_generations=${LOG_VAL_GENERATIONS:-4}

# Per-assistant-turn generation cap (the agent loop reads this from the env of
# its Ray worker; it is forwarded through ray_kwargs.runtime_env below).
export AGENTIC_MAX_TOKENS_PER_TURN=${AGENTIC_MAX_TOKENS_PER_TURN:-3072}
enable_thinking=${ENABLE_THINKING:-True}
tool_format=${FORMAT:-hermes}

RESUME_MODE=${RESUME_MODE:-disable}
RESUME_PATH=${RESUME_PATH:-}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
LOGGER=${LOGGER:-'["console"]'}
PROJECT_NAME=${PROJECT_NAME:-agentic_rl_v2}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_agentic_v2_fully_async_$(date +%Y%m%d_%H%M)}
CKPT_DIR=${CKPT_DIR:-$REPO_ROOT/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME}

agent_loop_config="$REPO_ROOT/agentic_rl_data/verl/agent_loop_config.yaml"
reward_bridge="$REPO_ROOT/agentic_rl_data/scripts/verl_reward_bridge.py"

# --------------------------------------------------------------- pre-flight
for f in "$TRAIN_FILE" "$VAL_FILE"; do
  [ -f "$f" ] || { echo "ERROR: missing $f; run agentic_rl_data/scripts/build_data_v2.sh" >&2; exit 1; }
done
[ -d "$MODEL_PATH" ] || { echo "ERROR: MODEL_PATH=$MODEL_PATH is not a directory" >&2; exit 1; }
[ -f "$agent_loop_config" ] || { echo "ERROR: missing $agent_loop_config" >&2; exit 1; }
[ -f "$reward_bridge" ] || { echo "ERROR: missing $reward_bridge" >&2; exit 1; }
phase_tokens=$((max_prompt_length + max_response_length))
pack_tokens=$((ppo_max_token_len_per_gpu * sp_size))
if [ "$pack_tokens" -lt "$phase_tokens" ]; then
  echo "ERROR: ppo_max_token_len_per_gpu($ppo_max_token_len_per_gpu) * SP($sp_size) = $pack_tokens" >&2
  echo "       < prompt+response = $phase_tokens; raise the budget or SP, or shorten the context." >&2
  exit 1
fi
if [ "$RESUME_MODE" = "resume_path" ] && [ -z "$RESUME_PATH" ]; then
  echo "ERROR: RESUME_MODE=resume_path needs RESUME_PATH=<global_step_N dir>" >&2
  exit 1
fi
mkdir -p "$CKPT_DIR"

echo "[agentic-async] stage=$STAGE train=$TRAIN_FILE max_turns=$MAX_TURNS"
echo "[agentic-async] context: prompt=$max_prompt_length response=$max_response_length total=$phase_tokens sp=$sp_size pack=$pack_tokens"
echo "[agentic-async] pools: trainer=${NGPUS_TRAIN}NPU rollouter=${NGPUS_ROLLOUT}NPU n=$rollout_n mini=$ppo_mini_batch_size x $require_batches"
echo "[agentic-async] budget: total_rollout_steps=$total_rollout_steps prompts, ckpt=$CKPT_DIR"

# ------------------------------------------------------------------ config
DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    # In fully async the loss reads actor.policy_loss.rollout_correction; this
    # flag only drives the trainer-side data bypass (no old_log_prob recompute).
    algorithm.rollout_correction.bypass_mode=True
    data.train_files="$TRAIN_FILE"
    data.val_files="$VAL_FILE"
    # Fully async: train_batch_size is inert (must be 0), prompts stream in
    # one at a time.
    data.train_batch_size=0
    data.gen_batch_size=1
    # AgentLoop server mode consumes raw chat and applies the template itself
    # (with the per-sample tool schemas from the parquet `tools` column).
    data.return_raw_chat=True
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.truncation='left'
    # Per-sample tool schemas are injected at rollout time and are not counted
    # by the dataset-side length filter, so keep filtering off.
    data.filter_overlong_prompts=False
    data.dataloader_num_workers=0
    +data.apply_chat_template_kwargs.enable_thinking=${enable_thinking}
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay}
    actor_rollout_ref.actor.grad_clip=1.0
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    # Fully async: two policies (pi_rollout = pi_old, pi_theta).
    actor_rollout_ref.actor.policy_loss.loss_mode=bypass_mode
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_rs=token_k1
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_rs_threshold=${ROLLOUT_RS_THRESHOLD:-0.6_1.6}
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIMIZER_OFFLOAD:-False}
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False
    actor_rollout_ref.nccl_timeout=${nccl_timeout}
    # Weights + bookkeeping only (no optimizer state).
    actor_rollout_ref.actor.checkpoint.save_contents='["model","extra"]'
    # Fully async: disaggregated pools, log-probs come from the Rollouter.
    actor_rollout_ref.hybrid_engine=False
    actor_rollout_ref.actor.use_rollout_log_probs=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.temperature=${temperature}
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.rollout.max_num_seqs=${max_num_seqs}
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens}
    actor_rollout_ref.rollout.enforce_eager=${enforce_eager}
    +actor_rollout_ref.rollout.enable_sleep_mode=False
    # REQUIRED by fully async (asserted in FullyAsyncRollouter._validate_config).
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=${val_n}
    # Multi-turn agent loop: per-sample tools, env-backed reward.
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.format=${tool_format}
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_TURNS}
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$((MAX_TURNS + 1))
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=4
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048
    actor_rollout_ref.rollout.agent.default_agent_loop=agentic_agent
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$agent_loop_config"
)

REWARD=(
    reward.custom_reward_function.path="$reward_bridge"
    reward.custom_reward_function.name=compute_score
    reward.reward_manager.name=naive
)

TRAINER=(
    trainer.device=npu
    trainer.balance_batch=True
    trainer.logger=${LOGGER}
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_TRAIN}
    trainer.nnodes=${NNODES_TRAIN}
    trainer.val_before_train=${VAL_BEFORE_TRAIN}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.log_val_generations=${log_val_generations}
    trainer.total_epochs=${total_epochs}
    ++trainer.default_local_dir=${CKPT_DIR}
    trainer.resume_mode=${RESUME_MODE}
)

ASYNC=(
    rollout.nnodes=${NNODES_ROLLOUT}
    rollout.n_gpus_per_node=${NGPUS_ROLLOUT}
    rollout.total_rollout_steps=${total_rollout_steps}
    async_training.staleness_threshold=${staleness_threshold}
    async_training.trigger_parameter_sync_step=${trigger_parameter_sync_step}
    async_training.require_batches=${require_batches}
    async_training.partial_rollout=${partial_rollout}
    async_training.use_trainer_do_validate=False
    # The agent loop reads AGENTIC_MAX_TOKENS_PER_TURN from its worker env; a
    # shell export does not reach Ray actors on a pre-started cluster.
    +ray_kwargs.ray_init.runtime_env.env_vars.AGENTIC_MAX_TOKENS_PER_TURN="${AGENTIC_MAX_TOKENS_PER_TURN}"
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="$PYTHONPATH"
    +ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
    +ray_kwargs.ray_init.runtime_env.env_vars.TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
)

if [ -n "$RESUME_PATH" ]; then
  TRAINER+=( trainer.resume_from_path="$RESUME_PATH" )
fi

if [ "${AGENTIC_FILTER_GROUPS:-0}" = "1" ]; then
  echo "ERROR: DAPO filter_groups / hard replay are not supported by the fully-async stream." >&2
  echo "       Use agentic_rl_data/scripts/run_qwen3_4b_agentic_v2.sh (sync) for that." >&2
  exit 1
fi

# ------------------------------------------------------------------ launch
"$PYTHON_BIN" -m verl.experimental.fully_async_policy.fully_async_main \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${ASYNC[@]}" \
    "$@"
