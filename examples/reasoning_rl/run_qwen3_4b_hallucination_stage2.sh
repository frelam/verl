#!/usr/bin/env bash
# Hallucination-resistance stage 2 (HALLUCINATION_RL_DESIGN.md): DATA BUILD + TRAINING.
#
# One entry point for the whole experiment:
#   Phase 1 (build, idempotent — SKIP_BUILD=1 to skip):
#     fetch_raw.py                    -> $DATA_ROOT/halluc/raw/<source>/   (per-source raw files)
#     treecut-math clone              -> $DATA_ROOT/halluc/raw/treecut/repo (generator repo, not on HF)
#     <source>_adapter.py  x 8        -> $DATA_ROOT/halluc/built/<source>.parquet
#     distractor_synth.py (D17)       -> $DATA_ROOT/halluc/built/d17.parquet
#     mix_halluc.py                   -> $STAGE2_DIR/{train,val}.parquet + mix_stats.json
#   Phase 2 (train):
#     delegates to run_qwen3_4b_reasoning_rl_fully_async.sh with the stage-2 mix and
#     REWARD_PATH pointed at hallucination_compute_score.py (halluc_* rows scored by
#     the abstention dispatcher; every other row is delegated back to compute_score.py,
#     so the mixed parquet keeps the stage-1 domains scoring unchanged).
#
# Usage:
#   RESUME_PATH=<stage1_ckpt_dir> bash run_qwen3_4b_hallucination_stage2.sh
#   SKIP_BUILD=1 RESUME_PATH=<stage1_ckpt_dir> bash run_qwen3_4b_hallucination_stage2.sh
#
# Env knobs (this script):
#   SKIP_BUILD=0          skip phase 1 (parquets already built)
#   RUN_TESTS=0           run the hallucination unit tests before training
#   PYTHON_BIN=python3    interpreter used by the build scripts
#   DATA_ROOT             default ~/data/reasoning_rl
#   STAGE1_DATA_DIR       stage-1 mix dir with train.parquet / val.parquet
#                         (default $DATA_ROOT/final); used by distractor_synth and
#                         mix_halluc for old-domain mixing + near-dedup anchors
#   STAGE2_DIR            stage-2 output dir (default $DATA_ROOT/final_halluc)
#   HALLUC_TOTAL=20000    hallucination-domain row quota (mix_halluc --halluc_total)
#   HALLUC_UNSOLVABLE_RATIO=0.6   unsolvable share (mix_halluc --halluc_unsolvable_ratio)
#   OLD_DOMAIN_RATIO=0.7  stage-1 share of the stage-2 mix
#   VAL_SIZE=256          hallucination-domain val rows
#   SEED=42
#   RESUME_MODE / RESUME_PATH   resume knobs, forwarded to the recipe. RESUME_PATH
#                         (a stage-1 checkpoint) is REQUIRED for the normal
#                         mid-training hand-off.
#   Every other env var (MODEL_PATH, TOTAL_STEPS, WEIGHT_DECAY, STALENESS_THRESHOLD,
#   SANDBOX_FUSION_URL, ...) passes through to the fully-async recipe untouched.

set -xeuo pipefail

########################### build configuration ###########################
PYTHON_BIN=${PYTHON_BIN:-python3}
DATA_ROOT=${DATA_ROOT:-$HOME/data/reasoning_rl}
STAGE1_DATA_DIR=${STAGE1_DATA_DIR:-$DATA_ROOT/final}
STAGE2_DIR=${STAGE2_DIR:-$DATA_ROOT/final_halluc}
RAW_ROOT=$DATA_ROOT/halluc/raw
BUILT_DIR=$DATA_ROOT/halluc/built

HALLUC_TOTAL=${HALLUC_TOTAL:-20000}
HALLUC_UNSOLVABLE_RATIO=${HALLUC_UNSOLVABLE_RATIO:-0.6}
OLD_DOMAIN_RATIO=${OLD_DOMAIN_RATIO:-0.7}
VAL_SIZE=${VAL_SIZE:-256}
SEED=${SEED:-42}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HALLUC_SCRIPTS=$REPO_ROOT/examples/reasoning_rl/scripts/hallucination
REWARD_FILE=$REPO_ROOT/examples/reasoning_rl/reward/hallucination_compute_score.py
RECIPE=$REPO_ROOT/examples/reasoning_rl/run_qwen3_4b_reasoning_rl_fully_async.sh

########################### phase 1: data build ###########################
if [ "${SKIP_BUILD:-0}" != "1" ]; then
    # 1a. Raw downloads (idempotent; completed files are cached and sha256-manifested).
    $PYTHON_BIN "$HALLUC_SCRIPTS/fetch_raw.py" --dest "$RAW_ROOT"

    # 1b. TreeCut is a generator repo, not a file: fetch_raw does not cover it
    #     (PENDING_SUBDIRS). The adapter runs <raw_dir>/repo/treecut in-process.
    if [ ! -d "$RAW_ROOT/treecut/repo/treecut" ]; then
        mkdir -p "$RAW_ROOT/treecut"
        git clone https://github.com/j-bagel/treecut-math "$RAW_ROOT/treecut/repo"
    fi

    # 1c. Per-source adapters. --raw-dir / --out are passed explicitly because
    #     several adapters hard-code a /home/<user>/... default. Output names
    #     must keep the <source>.parquet layout mix_halluc reads the directory for.
    mkdir -p "$BUILT_DIR"
    $PYTHON_BIN "$HALLUC_SCRIPTS/kk_adapter.py"        --raw-dir "$RAW_ROOT/kk"     --out "$BUILT_DIR/kk.parquet"
    $PYTHON_BIN "$HALLUC_SCRIPTS/mip_adapter.py"       --raw-dir "$RAW_ROOT/mip"    --out "$BUILT_DIR/mip.parquet"
    $PYTHON_BIN "$HALLUC_SCRIPTS/falseqa_adapter.py"   --raw-dir "$RAW_ROOT/falseqa" --out "$BUILT_DIR/falseqa.parquet"
    $PYTHON_BIN "$HALLUC_SCRIPTS/gsm_ic_adapter.py"    --raw-dir "$RAW_ROOT/gsmic"  --out "$BUILT_DIR/gsmic.parquet"
    $PYTHON_BIN "$HALLUC_SCRIPTS/sum_adapter.py"       --raw-dir "$RAW_ROOT/sum"    --out "$BUILT_DIR/sum.parquet"
    $PYTHON_BIN "$HALLUC_SCRIPTS/umwp_adapter.py"      --raw-dir "$RAW_ROOT/umwp"   --out "$BUILT_DIR/umwp.parquet"
    $PYTHON_BIN "$HALLUC_SCRIPTS/treecut_adapter.py"   --raw-dir "$RAW_ROOT/treecut" --out "$BUILT_DIR/treecut.parquet"
    $PYTHON_BIN "$HALLUC_SCRIPTS/crepe_adapter.py"     --raw-dir "$RAW_ROOT/crepe"  --out "$BUILT_DIR/crepe.parquet"

    # 1d. D17 synthesised distractors. The main-pool slice (100 rows of table B)
    #     needs a stage-1 math parquet as its question source; without one the
    #     cell is left to mix_halluc's shortfall redistribution.
    DISTRACTOR_ARGS=(--raw-dir "$RAW_ROOT" --out "$BUILT_DIR/d17.parquet")
    if [ -f "$STAGE1_DATA_DIR/train.parquet" ]; then
        DISTRACTOR_ARGS+=(--stage1-path "$STAGE1_DATA_DIR/train.parquet")
    else
        echo "WARNING: $STAGE1_DATA_DIR/train.parquet not found; building distractors without the stage-1 main pool" >&2
        DISTRACTOR_ARGS+=(--main-quota 0)
    fi
    $PYTHON_BIN "$HALLUC_SCRIPTS/distractor_synth.py" "${DISTRACTOR_ARGS[@]}"

    # 1e. Stage-2 mix. --stage1_path points at the stage-1 output DIRECTORY: the
    #     mixer splits train.parquet / val.parquet, keeps the stage-1 val rows in
    #     the stage-2 val (old-domain regression signal) and runs the section 7.2
    #     MinHash near-dedup against the stage-1 train rows.
    $PYTHON_BIN "$HALLUC_SCRIPTS/mix_halluc.py" \
        --halluc_dir "$BUILT_DIR" \
        --stage1_path "$STAGE1_DATA_DIR" \
        --output_dir "$STAGE2_DIR" \
        --halluc_total "$HALLUC_TOTAL" \
        --halluc_unsolvable_ratio "$HALLUC_UNSOLVABLE_RATIO" \
        --old_domain_ratio "$OLD_DOMAIN_RATIO" \
        --val_size "$VAL_SIZE" \
        --seed "$SEED"
fi

########################### pre-flight checks ###########################
for f in "$STAGE2_DIR/train.parquet" "$STAGE2_DIR/val.parquet" "$REWARD_FILE"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: required artifact missing: $f (build with SKIP_BUILD unset, or fix STAGE2_DIR)" >&2
        exit 1
    fi
done

if [ "${RUN_TESTS:-0}" = "1" ]; then
    $PYTHON_BIN -m pytest "$HALLUC_SCRIPTS" "$REPO_ROOT/examples/reasoning_rl/reward/test_hallucination_compute_score.py"
fi

########################### phase 2: training ###########################
# Stage 2 resumes from the stage-1 checkpoint. Starting fresh is possible
# (RESUME_MODE=disable) but then the stage-1 training never happened.
RESUME_MODE=${RESUME_MODE:-resume_path}
if [ "$RESUME_MODE" = "resume_path" ] && [ -z "${RESUME_PATH:-}" ]; then
    echo "ERROR: stage 2 resumes from a stage-1 checkpoint; set RESUME_PATH=<ckpt_dir>" >&2
    exit 1
fi

export TRAIN_FILES="['$STAGE2_DIR/train.parquet']"
export VAL_FILES="['$STAGE2_DIR/val.parquet']"
export REWARD_PATH="$REWARD_FILE"
export RESUME_MODE
if [ -n "${RESUME_PATH:-}" ]; then
    export RESUME_PATH
fi

cd "$REPO_ROOT/examples/reasoning_rl"
bash run_qwen3_4b_reasoning_rl_fully_async.sh "$@"
