#!/usr/bin/env bash
# SFT | AgentInstruct | FSDP engine | NVIDIA GPUs
# Train Qwen3-0.6B on the AgentInstruct dataset using verl SFT trainer.
#
# Usage:
#   # Single GPU
#   bash run_qwen3_0_6b_agent_instruct.sh 1 /tmp/sft-ckpt
#
#   # Multi-GPU (e.g. 4 GPUs)
#   bash run_qwen3_0_6b_agent_instruct.sh 4 /tmp/sft-ckpt
#
#   # With LoRA
#   USE_PEFT=1 bash run_qwen3_0_6b_agent_instruct.sh 4 /tmp/sft-ckpt
#
#   # Override specific configs
#   bash run_qwen3_0_6b_agent_instruct.sh 4 /tmp/sft-ckpt \
#       trainer.total_epochs=5 optim.lr=5e-5

set -xeuo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen3_0_6b_agent_instruct.sh <nproc_per_node> <save_path> [other_configs...]"
    echo "  Env: SP_SIZE (default 1), USE_PEFT (0|1, default 0)"
    exit 1
fi

nproc_per_node=$1
save_path=$2
shift 2

# ---- user-adjustable ----
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-0.6B}
SP_SIZE=${SP_SIZE:-1}
USE_PEFT=${USE_PEFT:-0}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_TARGETS=${LORA_TARGETS:-all-linear}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-4}
LR=${LR:-1e-4}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
PROJECT_NAME=${PROJECT_NAME:-agent-instruct-sft}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-agent-instruct-sft-qwen3-0.6b}
DATA_DIR=${DATA_DIR:-$HOME/data/agent_instruct}
# Format-only SFT: only train on format tokens (tool call markers, XML/JSON
# structure, reasoning tags). Content tokens are masked out.
FORMAT_ONLY=${FORMAT_ONLY:-1}
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
FORMAT_DATASET_PATH="${SCRIPT_DIR}/format_only_sft_dataset.py"
# ---- end user-adjustable ----

extra_args=()
if [ "${USE_PEFT}" = "1" ]; then
    extra_args+=(
        "model.lora_rank=${LORA_RANK}"
        "model.lora_alpha=${LORA_ALPHA}"
        "model.target_modules=${LORA_TARGETS}"
    )
fi

if [ "${FORMAT_ONLY}" = "1" ]; then
    extra_args+=(
        "data.custom_cls.path=${FORMAT_DATASET_PATH}"
        "data.custom_cls.name=FormatOnlySFTDataset"
        "data.format_only=true"
    )
fi

torchrun --standalone --nnodes=1 --nproc_per_node=${nproc_per_node} \
    -m verl.trainer.sft_trainer \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.messages_key=messages \
    data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU} \
    data.pad_mode=no_padding \
    data.max_length=4096 \
    data.truncation=left \
    data.use_dynamic_bsz=true \
    data.max_token_len_per_gpu=16384 \
    optim.lr=${LR} \
    engine=fsdp \
    engine.ulysses_sequence_parallel_size=${SP_SIZE} \
    model.path="${MODEL_PATH}" \
    model.use_remove_padding=true \
    trainer.default_local_dir="${save_path}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger='["console","wandb"]' \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.save_freq=after_each_epoch \
    trainer.test_freq=after_each_epoch \
    "${extra_args[@]}" "$@"
