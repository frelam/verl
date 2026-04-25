#!/usr/bin/env bash
# SFT | ShareGPT + AgentInstruct | FSDP | Qwen3-0.6B
#
# Prerequisite: preprocess mixed dataset
#   python examples/data_preprocess/mix_sharegpt_agentinstruct_sft.py \
#       --local_save_dir ~/data/sharegpt_agentinstruct_sft
#
# Usage:
#   # Single GPU
#   bash examples/sft/sharegpt_agentinstruct/run_qwen3_0_6b_sft.sh 1 /tmp/qwen3-0.6b-sft
#
#   # 4 GPUs + Ulysses SP=2
#   SP_SIZE=2 bash examples/sft/sharegpt_agentinstruct/run_qwen3_0_6b_sft.sh 4 /tmp/qwen3-0.6b-sft
#
#   # LoRA fine-tuning
#   USE_PEFT=1 bash examples/sft/sharegpt_agentinstruct/run_qwen3_0_6b_sft.sh 1 /tmp/qwen3-0.6b-sft-lora
#
#   # Override hydra configs
#   bash examples/sft/sharegpt_agentinstruct/run_qwen3_0_6b_sft.sh 1 /tmp/sft \
#       trainer.total_epochs=5 optim.lr=5e-5 data.sharegpt_ratio=0.5

set -xeuo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen3_0_6b_sft.sh <nproc_per_node> <save_path> [hydra_overrides...]"
    echo ""
    echo "Environment variables:"
    echo "  MODEL_PATH              (default: Qwen/Qwen3-0.6B)"
    echo "  DATA_DIR                (default: ~/data/sharegpt_agentinstruct_sft)"
    echo "  SP_SIZE                 Ulysses sequence parallel size (default: 1)"
    echo "  USE_PEFT                1 to enable LoRA (default: 0)"
    echo "  MICRO_BATCH_SIZE_PER_GPU (default: 4)"
    echo "  LR                      (default: 1e-4)"
    echo "  TOTAL_EPOCHS            (default: 3)"
    exit 1
fi

nproc_per_node=$1
save_path=$2
shift 2

# ---- user-adjustable ----
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-0.6B}
DATA_DIR=${DATA_DIR:-$HOME/data/sharegpt_agentinstruct_sft}
SP_SIZE=${SP_SIZE:-1}
USE_PEFT=${USE_PEFT:-0}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_TARGETS=${LORA_TARGETS:-all-linear}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
LR=${LR:-1e-4}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
MAX_LENGTH=${MAX_LENGTH:-4096}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-16384}
PROJECT_NAME=${PROJECT_NAME:-sharegpt-agentinstruct-sft}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3-0.6b-sharegpt-agentinstruct}
# ---- end user-adjustable ----

train_files="${DATA_DIR}/train.parquet"
val_files="${DATA_DIR}/test.parquet"

if [ ! -f "${train_files}" ]; then
    echo "Error: train parquet not found: ${train_files}"
    echo "Run data preprocessing first:"
    echo "  python examples/data_preprocess/mix_sharegpt_agentinstruct_sft.py \\"
    echo "      --local_save_dir ${DATA_DIR}"
    exit 1
fi

extra_args=()
if [ "${USE_PEFT}" = "1" ]; then
    extra_args+=(
        "model.lora_rank=${LORA_RANK}"
        "model.lora_alpha=${LORA_ALPHA}"
        "model.target_modules=${LORA_TARGETS}"
    )
fi

val_args=()
if [ -f "${val_files}" ]; then
    val_args=("data.val_files=${val_files}")
else
    echo "Warning: val parquet not found (${val_files}), training without validation."
fi

torchrun --standalone --nnodes=1 --nproc_per_node="${nproc_per_node}" \
    -m verl.trainer.sft_trainer \
    data.train_files="${train_files}" \
    "${val_args[@]}" \
    data.messages_key=messages \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
    data.pad_mode=no_padding \
    data.max_length="${MAX_LENGTH}" \
    data.truncation=left \
    data.use_dynamic_bsz=true \
    data.max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
    data.ignore_input_ids_mismatch=true \
    optim.lr="${LR}" \
    engine=fsdp \
    engine.ulysses_sequence_parallel_size="${SP_SIZE}" \
    model.path="${MODEL_PATH}" \
    model.use_remove_padding=true \
    trainer.default_local_dir="${save_path}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger='["console","wandb"]' \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.save_freq=after_each_epoch \
    trainer.test_freq=after_each_epoch \
    "${extra_args[@]}" "$@"
