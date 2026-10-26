#!/usr/bin/env bash
# Reasoning-efficiency DPO | Qwen3-4B | FSDP | dataset C (A rejected vs B chosen)
#
# Trains on top of the RL-finished stage-3 checkpoint.  The preference pairs are
# built by the three data scripts under examples/reasoning_rl/scripts/distill:
#
#   1) collect_dataset_a.py  stage-3 prompts -> vLLM (n samples) -> reward==1 -> A
#   2) build_dataset_b.py    A -> LLM judge selects the best correct rollout and
#                            condenses its <think> block, re-verified with the RL
#                            reward -> B
#   3) build_dpo_pairs.py    A + B -> dpo_train.parquet / dpo_val.parquet
#
# This script mirrors run_qwen3_4b_hallucination_stage3_fully_async.sh where it
# applies: bf16 + gradient checkpointing + the same prompt/response length
# curriculum + AdamW/cosine.  There is no rollouter here (DPO needs no
# generation), so it is a plain torchrun job; the reference policy is another
# FSDP-sharded copy of the same checkpoint (REF_FREE=1 drops it).
#
# Usage:
#   MODEL_PATH=<rl_hf_checkpoint> bash run_qwen3_4b_dpo.sh
#
# Optional env knobs:
#   DATA_DIR=$HOME/data/reasoning_rl/dpo   contains dpo_train.parquet / dpo_val.parquet
#   OUTPUT_DIR=$HOME/ckpt/dpo_qwen3_4b_<timestamp>
#   NGPUS=8                 training devices (single node)
#   BETA=0.1                DPO temperature
#   LR=1e-6                 same as the stage-3 actor LR; 5e-7 is safer for long runs
#   EPOCHS=1
#   PER_DEVICE_BATCH_SIZE=1 preference pairs per rank per micro-batch
#   GRAD_ACCUM=8
#   MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=8192  (stage-3 curriculum)
#   MAX_TOTAL_LENGTH=0      0 = prompt + response
#   LENGTH_NORMALIZE=0      1 = length-normalised log-probs (removes DPO's length bias)
#   REF_FREE=0              1 = reference-free DPO (no ref model, half the memory)
#   REF_MODEL_PATH=         default = MODEL_PATH
#   SAVE_STEPS=200 EVAL_STEPS=100 SAVE_TOTAL_LIMIT=3
#   WANDB_PROJECT= RUN_NAME=
#   MASTER_PORT=29500 NUM_WORKERS=4

set -xeuo pipefail

# Ascend/HCCL runtime knobs (same values as the stage-3 recipe).  No-ops on CUDA.
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-3600}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-3600}
export HCCL_ASYNC_ERROR_HANDLING=${HCCL_ASYNC_ERROR_HANDLING:-0}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

########################### user-adjustable ###########################
# REQUIRED: the RL-finished checkpoint in HuggingFace format (the same directory
# the vLLM service for dataset A was started from).
MODEL_PATH=${MODEL_PATH:-}
TOKENIZER_PATH=${TOKENIZER_PATH:-$MODEL_PATH}
DATA_DIR=${DATA_DIR:-$HOME/data/reasoning_rl/dpo}
TRAIN_FILES=${TRAIN_FILES:-$DATA_DIR/dpo_train.parquet}
VAL_FILES=${VAL_FILES:-$DATA_DIR/dpo_val.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-$HOME/ckpt/dpo_qwen3_4b_$(date +%Y%m%d_%H%M)}

NGPUS=${NGPUS:-8}
MASTER_PORT=${MASTER_PORT:-29500}

BETA=${BETA:-0.1}
LR=${LR:-1e-6}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
EPOCHS=${EPOCHS:-1}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-8}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
LR_SCHEDULER=${LR_SCHEDULER:-cosine}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
MAX_TOTAL_LENGTH=${MAX_TOTAL_LENGTH:-0}
LENGTH_NORMALIZE=${LENGTH_NORMALIZE:-0}
REF_FREE=${REF_FREE:-0}
REF_MODEL_PATH=${REF_MODEL_PATH:-}

SAVE_STEPS=${SAVE_STEPS:-200}
EVAL_STEPS=${EVAL_STEPS:-100}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-3}
LOG_STEPS=${LOG_STEPS:-1}
NUM_WORKERS=${NUM_WORKERS:-4}

WANDB_PROJECT=${WANDB_PROJECT:-}
RUN_NAME=${RUN_NAME:-}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
FSDP=${FSDP:-1}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
########################### end user-adjustable ###########################

########################### pre-flight checks ###########################
if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: MODEL_PATH is required (the RL-finished HuggingFace checkpoint)." >&2
    exit 1
fi
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: MODEL_PATH does not exist: $MODEL_PATH" >&2
    exit 1
fi
if [ ! -f "$TRAIN_FILES" ]; then
    echo "ERROR: $TRAIN_FILES missing; run examples/reasoning_rl/scripts/distill/build_dpo_pairs.py first" >&2
    exit 1
fi
if ! command -v torchrun >/dev/null 2>&1; then
    echo "ERROR: torchrun not found on PATH (needed for the FSDP launch)." >&2
    exit 1
fi

ARGS=(
    --model "$MODEL_PATH"
    --tokenizer "$TOKENIZER_PATH"
    --train_files "$TRAIN_FILES"
    --output_dir "$OUTPUT_DIR"
    --beta "$BETA"
    --lr "$LR"
    --weight_decay "$WEIGHT_DECAY"
    --epochs "$EPOCHS"
    --per_device_batch_size "$PER_DEVICE_BATCH_SIZE"
    --grad_accum "$GRAD_ACCUM"
    --max_grad_norm "$MAX_GRAD_NORM"
    --warmup_ratio "$WARMUP_RATIO"
    --lr_scheduler "$LR_SCHEDULER"
    --max_prompt_length "$MAX_PROMPT_LENGTH"
    --max_response_length "$MAX_RESPONSE_LENGTH"
    --max_total_length "$MAX_TOTAL_LENGTH"
    --save_steps "$SAVE_STEPS"
    --eval_steps "$EVAL_STEPS"
    --save_total_limit "$SAVE_TOTAL_LIMIT"
    --log_steps "$LOG_STEPS"
    --num_workers "$NUM_WORKERS"
    --attn_implementation "$ATTN_IMPLEMENTATION"
)

# The val split is optional; a short run without it still trains (and warns).
if [ -f "$VAL_FILES" ]; then
    ARGS+=(--val_files "$VAL_FILES")
else
    echo "WARNING: $VAL_FILES not found; training without validation" >&2
fi

if [ "$LENGTH_NORMALIZE" = "1" ]; then
    ARGS+=(--length_normalize)
fi
if [ "$REF_FREE" = "1" ]; then
    ARGS+=(--ref_free)
else
    ARGS+=(--ref_model "${REF_MODEL_PATH:-$MODEL_PATH}")
fi
if [ "$FSDP" != "1" ]; then
    ARGS+=(--no-fsdp)
fi
if [ -n "$WANDB_PROJECT" ]; then
    ARGS+=(--wandb_project "$WANDB_PROJECT" --run_name "${RUN_NAME:-$(basename "$OUTPUT_DIR")}")
fi

mkdir -p "$OUTPUT_DIR"

########################### launch ###########################
torchrun \
    --nnodes=1 \
    --nproc_per_node="$NGPUS" \
    --master_addr=127.0.0.1 \
    --master_port="$MASTER_PORT" \
    "$REPO_ROOT/examples/reasoning_rl/dpo_trainer.py" \
    "${ARGS[@]}" \
    "$@"
