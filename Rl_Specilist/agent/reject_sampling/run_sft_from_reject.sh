#!/usr/bin/env bash
# SFT training on reject-sampled trajectories.
#
# This is a thin wrapper around examples/sft/agent_sft/run_qwen3_4b_sft.sh
# that points DATA_DIR to the reject sampling output.
#
# Usage:
#   bash run_sft_from_reject.sh <nproc_per_node> <save_path> [extra_configs...]
#
# Examples:
#   bash run_sft_from_reject.sh 8 ~/data/sft_ckpt
#   bash run_sft_from_reject.sh 4 ~/data/sft_ckpt trainer.total_epochs=5 optim.lr=5e-5

set -xeuo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: run_sft_from_reject.sh <nproc_per_node> <save_path> [extra_configs...]"
    echo ""
    echo "Environment variables:"
    echo "  MODEL_PATH  (default: \$HOME/models/Qwen3-4B)"
    echo "  DATA_DIR    (default: \$HOME/data/reject_sampling_sft)"
    exit 1
fi

nproc_per_node=$1
save_path=$2
shift 2 2>/dev/null || shift $#

# ---- Load env ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/setup/env.sh" 2>/dev/null || true

# ---- User-adjustable ----
MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3-4B}"
DATA_DIR="${SFT_DATA_DIR:-$HOME/data/reject_sampling_sft}"
PROJECT_NAME="${PROJECT_NAME:-reject-sft}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-reject-sft-qwen3-4b-$(date '+%m%d-%H%M')}"
LR="${LR:-1e-5}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
MAX_LENGTH="${MAX_LENGTH:-32768}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-2}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-32768}"

# ---- Verify data exists ----
TRAIN_FILE="$DATA_DIR/train.parquet"
if [ ! -f "$TRAIN_FILE" ]; then
    echo "ERROR: SFT train file not found: $TRAIN_FILE"
    echo "Run convert_to_sft first:"
    echo "  python -m Rl_Specilist.agent.reject_sampling.data_preprocess.convert_to_sft"
    exit 1
fi

# ---- Resolve verl project dir ----
VERL_DIR="${VERL_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
SFT_SCRIPT="$VERL_DIR/examples/sft/agent_sft/run_qwen3_4b_sft.sh"

echo "========================================"
echo " Reject Sampling SFT Training"
echo "========================================"
echo " Model:       $MODEL_PATH"
echo " Data dir:    $DATA_DIR"
echo " Train file:  $TRAIN_FILE"
echo " LR:          $LR"
echo " Epochs:      $TOTAL_EPOCHS"
echo " Max length:  $MAX_LENGTH"
echo " GPUs:        $nproc_per_node"
echo " Save path:   $save_path"
echo "========================================"

# ---- Check if verl's SFT script exists ----
if [ -f "$SFT_SCRIPT" ]; then
    # Use verl's official SFT script with our data
    DATA_DIR="$DATA_DIR" \
    MODEL_PATH="$MODEL_PATH" \
    LR="$LR" \
    TOTAL_EPOCHS="$TOTAL_EPOCHS" \
    MAX_LENGTH="$MAX_LENGTH" \
    MICRO_BATCH_SIZE_PER_GPU="$MICRO_BATCH_SIZE_PER_GPU" \
    MAX_TOKEN_LEN_PER_GPU="$MAX_TOKEN_LEN_PER_GPU" \
    PROJECT_NAME="$PROJECT_NAME" \
    EXPERIMENT_NAME="$EXPERIMENT_NAME" \
    bash "$SFT_SCRIPT" "$nproc_per_node" "$save_path" "$@"
else
    # Fallback: run sft_trainer directly
    echo "WARN: verl SFT script not found at $SFT_SCRIPT, running directly..."

    VAL_FILE="$DATA_DIR/val.parquet"
    VAL_ARGS=()
    if [ -f "$VAL_FILE" ]; then
        VAL_ARGS=("data.val_files=$VAL_FILE")
    fi

    torchrun --standalone --nnodes=1 --nproc_per_node="$nproc_per_node" \
        -m verl.trainer.sft_trainer \
        data.train_files="$TRAIN_FILE" \
        "${VAL_ARGS[@]}" \
        data.messages_key=messages \
        data.micro_batch_size_per_gpu="$MICRO_BATCH_SIZE_PER_GPU" \
        data.pad_mode=no_padding \
        data.max_length="$MAX_LENGTH" \
        data.truncation=left \
        data.use_dynamic_bsz=true \
        data.max_token_len_per_gpu="$MAX_TOKEN_LEN_PER_GPU" \
        data.ignore_input_ids_mismatch=true \
        data.num_workers=4 \
        optim.lr="$LR" \
        engine=fsdp \
        model.path="$MODEL_PATH" \
        model.use_remove_padding=true \
        trainer.default_local_dir="$save_path" \
        trainer.project_name="$PROJECT_NAME" \
        trainer.experiment_name="$EXPERIMENT_NAME" \
        trainer.logger='["console","wandb"]' \
        trainer.total_epochs="$TOTAL_EPOCHS" \
        trainer.save_freq=after_each_epoch \
        trainer.test_freq=after_each_epoch \
        "$@"
fi

echo ""
echo "========================================"
echo " SFT Training Complete!"
echo "========================================"
echo " Checkpoint: $save_path"
