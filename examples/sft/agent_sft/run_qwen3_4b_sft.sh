#!/usr/bin/env bash
# SFT | Agent trajectories | FSDP | Qwen3-4B
#
# Trains a Qwen3-4B model on agent trajectory datasets (ToolMind, Open-SWE-Traces,
# SWE-Zero, TerminalTraj, OpenResearcher) using the verl SFT trainer.
#
# Prerequisite: run data preprocessing first
#   python examples/data_preprocess/prepare_agent_sft_data.py \
#       --output_dir ~/data/agent_sft \
#       --toolmind_n 10000 \
#       --open_swe_traces_n 20000 \
#       --swe_zero_n 20000 \
#       --terminaltraj_n 5000 \
#       --openresearcher_n 10000
#
# Usage:
#   # Single GPU
#   bash examples/sft/agent_sft/run_qwen3_4b_sft.sh 1 /tmp/qwen3-4b-agent-sft
#
#   # 8 GPUs + Ulysses SP=4
#   SP_SIZE=4 bash examples/sft/agent_sft/run_qwen3_4b_sft.sh 8 /tmp/qwen3-4b-agent-sft
#
#   # Override hydra configs
#   bash examples/sft/agent_sft/run_qwen3_4b_sft.sh 8 /tmp/sft \
#       optim.lr=5e-5 trainer.total_epochs=2

set -xeuo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen3_4b_sft.sh <nproc_per_node> <save_path> [hydra_overrides...]"
    echo ""
    echo "Environment variables:"
    echo "  MODEL_PATH                (default: /home/charles/workspace/qwen3-4b-gdpo-step850)"
    echo "  DATA_DIR                  (default: ~/data/agent_sft)"
    echo "  SP_SIZE                   Ulysses sequence parallel size (default: 1)"
    echo "  MICRO_BATCH_SIZE_PER_GPU  (default: 2)"
    echo "  TRAIN_BATCH_SIZE          (default: 64)"
    echo "  LR                        (default: 1e-5)"
    echo "  TOTAL_EPOCHS              (default: 3)"
    echo "  MAX_LENGTH                (default: 32768)"
    echo "  MAX_TOKEN_LEN_PER_GPU     (default: 32768)"
    echo "  USE_PEFT                  1 to enable LoRA (default: 0)"
    exit 1
fi

nproc_per_node=$1
save_path=$2
shift 2

# ---- user-adjustable ----
MODEL_PATH=${MODEL_PATH:-/home/charles/workspace/qwen3-4b-gdpo-step850}
DATA_DIR=${DATA_DIR:-$HOME/data/agent_sft}
SP_SIZE=${SP_SIZE:-1}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-2}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
LR=${LR:-1e-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
MAX_LENGTH=${MAX_LENGTH:-32768}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-32768}
PROJECT_NAME=${PROJECT_NAME:-agent-sft}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3-4b-agent-sft}
USE_PEFT=${USE_PEFT:-0}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_TARGETS=${LORA_TARGETS:-all-linear}
# ---- end user-adjustable ----

train_files="${DATA_DIR}/train.parquet"
val_files="${DATA_DIR}/val.parquet"

if [ ! -f "${train_files}" ]; then
    echo "Error: train parquet not found: ${train_files}"
    echo "Run data preprocessing first:"
    echo "  python examples/data_preprocess/prepare_agent_sft_data.py \\"
    echo "      --output_dir ${DATA_DIR}"
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
    data.num_workers=4 \
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
