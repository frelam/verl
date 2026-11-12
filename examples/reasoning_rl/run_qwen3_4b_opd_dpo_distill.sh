#!/usr/bin/env bash
# OPD (on-policy distillation) | Qwen3-4B | DPO checkpoint -> its YaRN 8x twin
# FSDP2 + vLLM | single node 16 NPU (8 student : 8 teacher)
#
# Why this run
#   Pipeline: stage-3 reasoning RL -> reasoning-efficiency DPO (run_qwen3_4b_dpo.sh)
#   -> YaRN 8x context extension (32k -> 256k). YaRN buys long-context reach at
#   the cost of short-context reasoning quality, and the DPO stage has already
#   trimmed the reasoning traces. This script distills the PRE-YaRN DPO
#   checkpoint (TEACHER, the "original" DPO model) into its YaRN-extended twin
#   (STUDENT) on the stage-3 prompts, so the student keeps the 256k window but
#   stays distributionally close to the DPO model on the states it visits.
#
#   TEACHER and STUDENT must come from the SAME DPO lineage (same weights up to
#   the rope_scaling change / long-context continued training) and share a
#   tokenizer; this is checked below.
#
# Objective: sampled token + reverse KL ("PG OPD", Thinking-Machines style)
#   - Rollout: the STUDENT samples y ~ pi_theta(. | x) on stage-3 prompts.
#   - Teacher: only scores the tokens the student actually sampled. The teacher
#     is queried with prompt_logprobs=0 (a pure forward pass over
#     prompt + response), so it returns exactly ONE logprob per position.
#   - Per-token reverse-KL estimate (k1):
#         kl_t = log pi_theta(y_t | s_t) - log pi_teacher(y_t | s_t)
#     used as a reward with stop-grad, advantage_t = -kl_t, and a PPO-clipped
#     policy-gradient update:
#         loss_mode=k1, use_policy_gradient=True, policy_loss_mode=vanilla
#   - Task rewards are OFF (use_task_rewards=False): the teacher distribution is
#     the ONLY supervision. The reward function is still configured because the
#     trainer computes it for logging, but it never enters the loss.
#
# Hardware split (single node, 16 NPU = TOTAL_NPUS)
#   student pool = actor (FSDP2, dp=8) + rollout (vLLM TP=1, colocated):    8 NPU
#   teacher pool = vLLM prompt-logprob servers (TP=2, 4 replicas):          8 NPU
#   Both pools live in one Ray cluster: NGPUS_STUDENT + TEACHER_WORLD_SIZE
#   must equal TOTAL_NPUS, and each pool must be divisible by its TP (checked
#   before launch). Other valid 16-NPU splits: 12+4 (2 teacher replicas) or
#   6+10 (5 teacher replicas, if teacher prefill is the bottleneck).
#
# Prerequisites
#   1) STUDENT_MODEL_PATH: the YaRN-extended 256k checkpoint of the DPO model
#      (its config.json carries rope_scaling).
#   2) TEACHER_MODEL_PATH: the SAME DPO checkpoint BEFORE YaRN (native 32k
#      context, no rope_scaling). Defaults to the latest global_step_* written
#      by dpo_trainer.py under DPO_OUTPUT_DIR.
#   3) stage-3 data: $DATA_DIR/{train,val}.parquet, built by
#      build_hallucination_stage3_data.sh.
#   4) torch_npu + vllm-ascend environment (same as the stage-3 RL runs).
#
# Usage:
#   STUDENT_MODEL_PATH=<dpo_ckpt_yarn_256k> \
#   TEACHER_MODEL_PATH=<dpo_ckpt_no_yarn> \
#   bash run_qwen3_4b_opd_dpo_distill.sh
#
#   # or let the teacher be auto-resolved from the DPO run directory
#   STUDENT_MODEL_PATH=<dpo_ckpt_yarn_256k> DPO_OUTPUT_DIR=$HOME/ckpt/dpo_qwen3_4b_xxx \
#   bash run_qwen3_4b_opd_dpo_distill.sh
#
# Optional env knobs:
#   DPO_OUTPUT_DIR=$HOME/ckpt/dpo_qwen3_4b   teacher auto-pick root (latest global_step_*)
#   DATA_DIR=$HOME/data/reasoning_rl/final_halluc_stage3   train/val parquet dir
#   TRAIN_FILES=/VAL_FILES=          explicit parquet overrides (Hydra list syntax)
#   OUTPUT_DIR=$HOME/ckpt/opd_qwen3_4b_dpo_<timestamp>
#   NGPUS_STUDENT=8 TEACHER_WORLD_SIZE=8                    NPU split (sum = TOTAL_NPUS)
#   TOTAL_NPUS=16                   NPU count on the node (also sizes ASCEND_RT_VISIBLE_DEVICES)
#   TRAIN_BATCH_SIZE=128             prompts per training step
#   PPO_MINI_BATCH_SIZE=64           prompts per PPO mini-batch
#   MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=12288        stage-3 curriculum (12k)
#                                    The DPO model is shorter-winded, but the
#                                    student keeps a 12k cap so long rollouts are
#                                    still covered by the OPD state distribution.
#   TEACHER_MAX_MODEL_LEN=           default prompt+response+1; must stay <= the
#                                    teacher's native max_position_embeddings
#   ACTOR_LR=1e-6 WEIGHT_DECAY=0.01
#   DISTILL_CLIP_LOW=0.2 DISTILL_CLIP_HIGH=0.28             PPO clip for PG OPD
#   ROLLOUT_TP=1 ROLLOUT_GPU_MEM_UTIL=0.4                   student vLLM (colocated)
#   TEACHER_TP=2 TEACHER_GPU_MEM_UTIL=0.6                   teacher vLLM (dedicated)
#   MAX_NUM_SEQS=128 ENFORCE_EAGER=False                    vllm-ascend tuning
#   NCCL_TIMEOUT=7200                                       HCCL collective timeout (s)
#   REASONING_RL_SYSTEM_PROMPT=      MUST match the value used by stage-3 RL / DPO data
#   TOTAL_EPOCHS=3 SAVE_FREQ=20 TEST_FREQ=20
#   LOGGER='["console","wandb"]' RESUME_MODE=disable
#   PROJECT_NAME= RUN_NAME=
#   SANDBOX_FUSION_URL=              forwarded to the reward kwargs (logging only)

set -xeuo pipefail

# vLLM V1 engine + Ascend/HCCL runtime knobs (same values as the stage-3 recipe).
export VLLM_USE_V1=1
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-3600}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-3600}
export HCCL_ASYNC_ERROR_HANDLING=${HCCL_ASYNC_ERROR_HANDLING:-0}
# Ray must not overwrite the NPU visibility set by the launcher.
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES:-1}
# Total NPUs on this node. The student pool (actor + colocated rollout) and the
# teacher pool are carved out of it and must sum to exactly this value.
export TOTAL_NPUS=${TOTAL_NPUS:-16}
# All NPUs on the node by default (16 = 8 student + 8 teacher).
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-$(seq -s, 0 $((TOTAL_NPUS - 1)))}
# Hard replay is a stage-3 RL sampling trick; never enable it for distillation.
export REASONING_RL_HARD_REPLAY=off

########################### user-adjustable ###########################
# REQUIRED: student = the YaRN-extended DPO checkpoint.
STUDENT_MODEL_PATH=${STUDENT_MODEL_PATH:-}
# Teacher = the SAME DPO checkpoint BEFORE YaRN. If unset, fall back to the
# latest global_step_* saved by dpo_trainer.py under DPO_OUTPUT_DIR.
DPO_OUTPUT_DIR=${DPO_OUTPUT_DIR:-$HOME/ckpt/dpo_qwen3_4b}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-}
if [ -z "$TEACHER_MODEL_PATH" ] && [ -d "$DPO_OUTPUT_DIR" ]; then
    TEACHER_MODEL_PATH=$(find "$DPO_OUTPUT_DIR" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1)
fi

DATA_DIR=${DATA_DIR:-$HOME/data/reasoning_rl/final_halluc_stage3}
train_files=${TRAIN_FILES:-"['$DATA_DIR/train.parquet']"}
val_files=${VAL_FILES:-"['$DATA_DIR/val.parquet']"}
OUTPUT_DIR=${OUTPUT_DIR:-$HOME/ckpt/opd_qwen3_4b_dpo_$(date +%Y%m%d_%H%M)}

# Resource split over the node's TOTAL_NPUS (default 16 = 8 + 8).
#   student pool = FSDP2 actor + colocated vLLM rollout, MUST satisfy
#                  NGPUS_STUDENT % ROLLOUT_TP == 0 (the actor DP degree).
#   teacher pool = dedicated vLLM prompt-logprob servers, MUST satisfy
#                  TEACHER_WORLD_SIZE % TEACHER_TP == 0; replicas are derived as
#                  TEACHER_WORLD_SIZE / TEACHER_TP by verl itself.
# Alternatives on 16 NPU: 12 + 4 (2 teacher replicas, TP=2) if training
# throughput matters more, or 6 + 10 (5 teacher replicas) if the teacher prefill
# is the bottleneck. 8 + 8 is the balanced default.
NGPUS_STUDENT=${NGPUS_STUDENT:-8}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-8}

# Stage-3 length curriculum (prompts ~2k, responses capped at 12k).
max_prompt_length=${MAX_PROMPT_LENGTH:-4096}
max_response_length=${MAX_RESPONSE_LENGTH:-12288}
max_num_tokens=$((max_prompt_length + max_response_length + 1))
# Teacher context: prompt + response + 1 generated dummy token is all OPD needs.
# The teacher is the NATIVE-context DPO model, so keep this <= its 32k limit.
teacher_max_model_len=${TEACHER_MAX_MODEL_LEN:-$max_num_tokens}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-$((max_prompt_length + max_response_length))}

train_batch_size=${TRAIN_BATCH_SIZE:-128}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-64}

actor_lr=${ACTOR_LR:-1e-6}
weight_decay=${WEIGHT_DECAY:-0.01}
# PG OPD clips the importance ratio pi_theta / pi_old around the sampled token.
distill_clip_low=${DISTILL_CLIP_LOW:-0.2}
distill_clip_high=${DISTILL_CLIP_HIGH:-0.28}

# Student rollout runs colocated with FSDP training -> keep vLLM's share modest.
rollout_tp=${ROLLOUT_TP:-1}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.4}
rollout_n=${ROLLOUT_N:-1}
temperature=${TEMPERATURE:-1.0}
# Teacher pool is dedicated (no training on it) -> it can take a larger share.
teacher_tp=${TEACHER_TP:-2}
teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-0.6}

max_num_seqs=${MAX_NUM_SEQS:-128}
enforce_eager=${ENFORCE_EAGER:-False}
nccl_timeout=${NCCL_TIMEOUT:-7200}

total_epochs=${TOTAL_EPOCHS:-3}
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-20}
log_val_generations=${LOG_VAL_GENERATIONS:-5}

# MUST match the stage-3 RL run (the dataset has no system turn; the custom
# dataset class injects it at load time, and it is forwarded into the Ray
# runtime env because the dataset is built inside the remote TaskRunner).
system_prompt=${REASONING_RL_SYSTEM_PROMPT:-}

LOGGER=${LOGGER:-'["console","wandb"]'}
RESUME_MODE=${RESUME_MODE:-disable}
PROJECT_NAME=${PROJECT_NAME:-verl_reasoning_rl}
RUN_NAME=${RUN_NAME:-opd_qwen3_4b_dpo_yarn_$(date +%Y%m%d_%H%M)}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Reward only feeds accuracy metrics under use_task_rewards=False; reuse the
# stage-3 dispatcher so halluc_* rows are scoreable.
reward_path=${REWARD_PATH:-"$REPO_ROOT/examples/reasoning_rl/reward/hallucination_compute_score.py"}
########################### end user-adjustable ###########################

########################### pre-flight checks ###########################
if [ -z "$STUDENT_MODEL_PATH" ]; then
    echo "ERROR: STUDENT_MODEL_PATH is required (the YaRN-extended DPO checkpoint)." >&2
    exit 1
fi
if [ -z "$TEACHER_MODEL_PATH" ]; then
    echo "ERROR: TEACHER_MODEL_PATH is required (the pre-YaRN DPO checkpoint)." >&2
    echo "       Set it explicitly or point DPO_OUTPUT_DIR at the DPO run directory." >&2
    exit 1
fi
if [ ! -d "$STUDENT_MODEL_PATH" ]; then
    echo "ERROR: STUDENT_MODEL_PATH does not exist: $STUDENT_MODEL_PATH" >&2
    exit 1
fi
if [ ! -d "$TEACHER_MODEL_PATH" ]; then
    echo "ERROR: TEACHER_MODEL_PATH does not exist: $TEACHER_MODEL_PATH" >&2
    exit 1
fi
if [ ! -f "$STUDENT_MODEL_PATH/config.json" ]; then
    echo "ERROR: $STUDENT_MODEL_PATH/config.json missing (not a HF checkpoint)." >&2
    exit 1
fi
if [ ! -f "$TEACHER_MODEL_PATH/config.json" ]; then
    echo "ERROR: $TEACHER_MODEL_PATH/config.json missing (not a HF checkpoint)." >&2
    exit 1
fi
if [ ! -f "$TEACHER_MODEL_PATH/tokenizer_config.json" ]; then
    echo "WARNING: $TEACHER_MODEL_PATH/tokenizer_config.json missing; dpo_trainer.py normally saves the tokenizer next to the weights." >&2
fi
if grep -q '"rope_scaling"' "$TEACHER_MODEL_PATH/config.json"; then
    echo "WARNING: teacher config.json carries \"rope_scaling\"; the teacher should be the pre-YaRN DPO checkpoint." >&2
fi
if ! grep -q '"rope_scaling"' "$STUDENT_MODEL_PATH/config.json" 2>/dev/null; then
    echo "WARNING: student config.json has no \"rope_scaling\"; is STUDENT_MODEL_PATH really the YaRN-extended checkpoint?" >&2
fi

# Teacher and student must share the same vocabulary (same DPO lineage + tokenizer),
# otherwise the per-token KL is between two different tokenisations.
if command -v python3 >/dev/null 2>&1; then
    teacher_vocab=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('vocab_size') or 0)" "$TEACHER_MODEL_PATH/config.json")
    student_vocab=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('vocab_size') or 0)" "$STUDENT_MODEL_PATH/config.json")
    if [ "$teacher_vocab" != "$student_vocab" ]; then
        echo "ERROR: vocab_size mismatch (teacher=$teacher_vocab student=$student_vocab); OPD needs identical tokenizers." >&2
        exit 1
    fi
    # The teacher is the native-context model; never ask vLLM for more than it has.
    teacher_native_ctx=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('max_position_embeddings') or 0)" "$TEACHER_MODEL_PATH/config.json")
    if [ "$teacher_native_ctx" -gt 0 ] && [ "$teacher_max_model_len" -gt "$teacher_native_ctx" ]; then
        echo "ERROR: teacher_max_model_len=$teacher_max_model_len exceeds the teacher's native max_position_embeddings=$teacher_native_ctx." >&2
        echo "       Lower MAX_PROMPT_LENGTH / MAX_RESPONSE_LENGTH / TEACHER_MAX_MODEL_LEN." >&2
        exit 1
    fi
else
    echo "WARNING: python3 not found; skipped the vocab_size / context-length pre-flight checks." >&2
fi

if [ -z "${TRAIN_FILES:-}" ] && [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "ERROR: $DATA_DIR/train.parquet missing; run build_hallucination_stage3_data.sh first" >&2
    exit 1
fi
if [ -z "${VAL_FILES:-}" ] && [ ! -f "$DATA_DIR/val.parquet" ]; then
    echo "ERROR: $DATA_DIR/val.parquet missing; run build_hallucination_stage3_data.sh first" >&2
    exit 1
fi
if [ ! -f "$reward_path" ]; then
    echo "ERROR: reward file missing: $reward_path" >&2
    exit 1
fi

# NPU allocation: both pools are created inside one Ray cluster, so the sum must
# equal the node's TOTAL_NPUS and each pool must be internally divisible.
if [ $((NGPUS_STUDENT + TEACHER_WORLD_SIZE)) -ne "$TOTAL_NPUS" ]; then
    echo "ERROR: NGPUS_STUDENT($NGPUS_STUDENT) + TEACHER_WORLD_SIZE($TEACHER_WORLD_SIZE) != TOTAL_NPUS($TOTAL_NPUS)." >&2
    exit 1
fi
if [ $((TEACHER_WORLD_SIZE % teacher_tp)) -ne 0 ]; then
    echo "ERROR: TEACHER_WORLD_SIZE($TEACHER_WORLD_SIZE) is not divisible by TEACHER_TP($teacher_tp)." >&2
    exit 1
fi
if [ $((NGPUS_STUDENT % rollout_tp)) -ne 0 ]; then
    echo "ERROR: NGPUS_STUDENT($NGPUS_STUDENT) is not divisible by ROLLOUT_TP($rollout_tp)." >&2
    exit 1
fi
teacher_replicas=$((TEACHER_WORLD_SIZE / teacher_tp))
echo "[resources] TOTAL_NPUS=$TOTAL_NPUS | student pool=$NGPUS_STUDENT NPU (FSDP2 dp=$NGPUS_STUDENT, vLLM TP=$rollout_tp)" >&2
echo "[resources] teacher pool=$TEACHER_WORLD_SIZE NPU ($teacher_replicas vLLM replicas x TP$teacher_tp)" >&2

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$train_files"
    data.val_files="$val_files"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.truncation='left'
    data.filter_overlong_prompts=True
    # AgentLoop server mode consumes raw chat and applies the template itself.
    data.return_raw_chat=True
    # Stage-3 prompts carry no system turn; this subclass injects
    # REASONING_RL_SYSTEM_PROMPT (pass-through when unset).
    data.custom_cls.path="$REPO_ROOT/examples/reasoning_rl/reasoning_rl_dataset.py"
    data.custom_cls.name=ReasoningRLDataset
)

MODEL=(
    actor_rollout_ref.model.path="$STUDENT_MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    # FSDP2 is the validated Ascend strategy (geo3k / stage-3 NPU recipes).
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    # No reference anchor: the teacher distribution is the only regularizer.
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.calculate_entropy=False
    # The trainer forces calculate_entropy=True for the old_log_prob pass; chunk
    # the logits->entropy reduction so a 12k-token micro-batch does not
    # materialise [tokens, 152k vocab] entropy intermediates in one shot.
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.entropy_from_logits_chunk_size=2048
    # Config default is True (CUDA-oriented); keep torch.compile off on Ascend.
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True
    # Weights + bookkeeping only (halves checkpoint disk).
    actor_rollout_ref.actor.checkpoint.save_contents='["model","extra"]'
    # HCCL init/collectives are slower to time out under load than NCCL.
    actor_rollout_ref.nccl_timeout=${nccl_timeout}
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    # One on-policy rollout per prompt is enough; the teacher provides the
    # dense per-token signal.
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.temperature=${temperature}
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length))
    actor_rollout_ref.rollout.max_num_seqs=${max_num_seqs}
    actor_rollout_ref.rollout.enforce_eager=${enforce_eager}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    # Validation uses the same sampling settings as the stage-3 RL runs.
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
)

# OPD: dedicated teacher pool + sampled-token reverse KL (k1, PG mode).
# The teacher is the pre-YaRN DPO checkpoint (no rope_scaling, native context).
DISTILL=(
    distillation.enabled=True
    distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE}
    distillation.nnodes=1
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL_PATH"
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${teacher_tp}
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    distillation.teacher_models.teacher_model.inference.max_model_len=${teacher_max_model_len}
    distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=$((max_prompt_length + max_response_length))
    distillation.teacher_models.teacher_model.inference.max_num_seqs=${max_num_seqs}
    # Dedicated pool: never sleep/wake (vllm-ascend sleep mode is version-dependent).
    +distillation.teacher_models.teacher_model.inference.enable_sleep_mode=False
    # Sampled-token reverse KL. k1 requires PG mode (its gradient through the
    # teacher logprob is zero, so direct backprop is meaningless).
    distillation.distillation_loss.loss_mode=k1
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=True
    distillation.distillation_loss.policy_loss_mode=vanilla
    distillation.distillation_loss.clip_ratio_low=${distill_clip_low}
    distillation.distillation_loss.clip_ratio_high=${distill_clip_high}
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

REWARD=(
    reward.custom_reward_function.path="$reward_path"
    reward.custom_reward_function.name=compute_score
    reward.reward_manager.name=naive
)
if [ -n "${SANDBOX_FUSION_URL:-}" ]; then
    REWARD+=(
        "+reward.custom_reward_function.reward_kwargs.sandbox_fusion_url=${SANDBOX_FUSION_URL}"
    )
fi

TRAINER=(
    trainer.device=npu
    trainer.logger="${LOGGER}"
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${RUN_NAME}
    # Student pool: actor + rollout share these NPUs (hybrid engine).
    trainer.n_gpus_per_node=${NGPUS_STUDENT}
    trainer.nnodes=1
    trainer.val_before_train=False
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.log_val_generations=${log_val_generations}
    trainer.total_epochs=${total_epochs}
    trainer.default_local_dir=${OUTPUT_DIR}
    trainer.resume_mode=${RESUME_MODE}
)

mkdir -p "$OUTPUT_DIR"

# System prompt injection rides the Ray runtime env: the dataset is constructed
# inside the remote TaskRunner, which does not inherit shell exports.
if [ -n "$system_prompt" ]; then
    export REASONING_RL_SYSTEM_PROMPT="$system_prompt"
    DATA+=(
        "+ray_kwargs.ray_init.runtime_env.env_vars.REASONING_RL_SYSTEM_PROMPT=\"$system_prompt\""
    )
fi

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${DISTILL[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "$@"
