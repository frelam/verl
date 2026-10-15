#!/usr/bin/env bash
# Hallucination-resistance stage 3 (HALLUCINATION_RL_DESIGN.md D29): DATA BUILD ONLY.
#
# Stage 3 owns the D28 three-layer repair contract that used to live in stage 2:
# the four-tier diagnosis rows ask for \boxed{UNSOLVABLE: <选项ID>: <修复后的答案>}
# and the reward AND-gates refusal + correct pointer + correct post-repair
# solution.  Stage 2 keeps the two-layer pointer contract (see
# build_hallucination_stage2_data.sh); this script builds the repair variant.
#
# Idempotent build pipeline (completed artifacts are cached / re-used):
#   fetch_raw.py                    -> $DATA_ROOT/halluc/raw/<source>/   (falseqa + treecut inputs)
#   treecut-math clone              -> $DATA_ROOT/halluc/raw/treecut/repo (generator repo, not on HF)
#   falseqa_adapter.py  --repair-contract -> $REPAIR_BUILT_DIR/falseqa.parquet
#   treecut_adapter.py  --repair-contract -> $REPAIR_BUILT_DIR/treecut.parquet
#   mix_halluc.py --repair_mode     -> $STAGE3_DIR/{train,val}.parquet + mix_stats.json
#
# Training is a separate step: run_qwen3_4b_hallucination_stage3_fully_async.sh.
#
# Usage:
#   bash build_hallucination_stage3_data.sh
#
# Env knobs:
#   RUN_TESTS=0           run the hallucination unit tests after the build
#   PYTHON_BIN=python3    interpreter used by the build scripts
#   DATA_ROOT             default ~/data/reasoning_rl
#   STAGE1_DATA_DIR       stage-1 mix dir with train.parquet / val.parquet
#                         (default $DATA_ROOT/final); used by mix_halluc for
#                         old-domain mixing + near-dedup anchors
#   STAGE3_DIR            stage-3 output dir (default $DATA_ROOT/final_halluc_repair)
#   REPAIR_TOTAL=5835     repair-domain row quota (mix_halluc --halluc_total)
#   OLD_DOMAIN_RATIO=0.7  stage-1 share of the stage-3 mix
#   VAL_SIZE=256          repair-domain val rows
#   SEED=42

set -xeuo pipefail

########################### build configuration ###########################
PYTHON_BIN=${PYTHON_BIN:-python3}
DATA_ROOT=${DATA_ROOT:-$HOME/data/reasoning_rl}
STAGE1_DATA_DIR=${STAGE1_DATA_DIR:-$DATA_ROOT/final}
STAGE3_DIR=${STAGE3_DIR:-$DATA_ROOT/final_halluc_repair}
RAW_ROOT=$DATA_ROOT/halluc/raw
REPAIR_BUILT_DIR=$DATA_ROOT/halluc/built_repair

REPAIR_TOTAL=${REPAIR_TOTAL:-5835}
OLD_DOMAIN_RATIO=${OLD_DOMAIN_RATIO:-0.7}
VAL_SIZE=${VAL_SIZE:-256}
SEED=${SEED:-42}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HALLUC_SCRIPTS=$REPO_ROOT/examples/reasoning_rl/scripts/hallucination

# 1a. Raw downloads (idempotent; completed files are cached and sha256-manifested).
$PYTHON_BIN "$HALLUC_SCRIPTS/fetch_raw.py" --dest "$RAW_ROOT"

# 1b. TreeCut is a generator repo, not a file: fetch_raw does not cover it
#     (PENDING_SUBDIRS). The adapter runs <raw_dir>/repo/treecut in-process.
if [ ! -d "$RAW_ROOT/treecut/repo/treecut" ]; then
    mkdir -p "$RAW_ROOT/treecut"
    git clone https://github.com/j-bagel/treecut-math "$RAW_ROOT/treecut/repo"
fi

# 1c. The two four-tier sources, in the stage-3 repair contract.  Only these two
#     carry the third reward layer; the other sources have no repair variant.
mkdir -p "$REPAIR_BUILT_DIR"
$PYTHON_BIN "$HALLUC_SCRIPTS/falseqa_adapter.py"   --raw-dir "$RAW_ROOT/falseqa" --out "$REPAIR_BUILT_DIR/falseqa.parquet" --repair-contract
$PYTHON_BIN "$HALLUC_SCRIPTS/treecut_adapter.py"   --raw-dir "$RAW_ROOT/treecut" --out "$REPAIR_BUILT_DIR/treecut.parquet" --repair-contract

# 1d. Stage-3 mix.  --repair_mode quotas only the two four-tier repair cells
#     (928:4907, rescaled to REPAIR_TOTAL); the solvable rows sharing the repair
#     parquets are not quota'd.  --stage1_path adds the old-domain replay and
#     the near-dedup anchors exactly like the stage-2 mix.
$PYTHON_BIN "$HALLUC_SCRIPTS/mix_halluc.py" \
    --repair_mode \
    --halluc_dir "$REPAIR_BUILT_DIR" \
    --stage1_path "$STAGE1_DATA_DIR" \
    --output_dir "$STAGE3_DIR" \
    --halluc_total "$REPAIR_TOTAL" \
    --old_domain_ratio "$OLD_DOMAIN_RATIO" \
    --val_size "$VAL_SIZE" \
    --seed "$SEED"

########################### post-build checks ###########################
for f in "$STAGE3_DIR/train.parquet" "$STAGE3_DIR/val.parquet"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: required artifact missing after build: $f" >&2
        exit 1
    fi
done

if [ "${RUN_TESTS:-0}" = "1" ]; then
    $PYTHON_BIN -m pytest "$HALLUC_SCRIPTS" "$REPO_ROOT/examples/reasoning_rl/reward/test_hallucination_compute_score.py"
fi

echo "Stage-3 data ready: $STAGE3_DIR/{train,val}.parquet"
echo "Next: RESUME_PATH=<stage2_ckpt_dir> bash run_qwen3_4b_hallucination_stage3_fully_async.sh"
