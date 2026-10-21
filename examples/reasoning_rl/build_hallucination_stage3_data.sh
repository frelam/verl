#!/usr/bin/env bash
# Hallucination-resistance stage 3 (HALLUCINATION_RL_DESIGN.md D30): DATA BUILD ONLY.
#
# Stage 3 resumes from a stage-2 checkpoint and trains the domain-balanced
# unsolvable mix.  The target composition is fixed in two steps:
#   * hallucination domain = 25% solvable / 75% unsolvable by side weight, with
#     domain weights logic 20% / commonsense 20% / math 60% (row counts);
#   * old domain = 60% of the final train set, and the stage-1 domain is
#     solvable-only, so the whole artifact is 70% solvable / 30% unsolvable.
# The logic unsolvable side needs the D30 synthesised K&K rows
# (kk_adapter --emit-unsolvable --kkcut-per-group 0).  The D28 repair layer is
# retired: stage-3 rows are the plain stage-2 contracts, and the four-tier
# unsolvable rows score on the stage-2 pointer alone -- so the adapters run
# WITHOUT --repair-contract and mix_halluc runs --stage3_mix instead of the
# retired --repair_mode (see build_hallucination_stage2_data.sh).
#
# Idempotent build pipeline (completed artifacts are cached / re-used):
#   fetch_raw.py                    -> $DATA_ROOT/halluc/raw/<source>/   (per-source raw files)
#   treecut-math clone              -> $DATA_ROOT/halluc/raw/treecut/repo (generator repo, not on HF)
#   <source>_adapter.py  x 8        -> $DATA_ROOT/halluc/built_stage3/<source>.parquet
#                                      (kk with --emit-unsolvable --kkcut-per-group 0,
#                                      crepe with --quota-overrides, the rest at their defaults)
#   distractor_synth.py (D17)       -> $DATA_ROOT/halluc/built_stage3/d17.parquet
#   mix_halluc.py --stage3_mix      -> $STAGE3_DIR/{train,val}.parquet + mix_stats.json
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
#                         (default $DATA_ROOT/final); used by distractor_synth and
#                         mix_halluc for old-domain mixing + near-dedup anchors
#   STAGE3_DIR            stage-3 output dir (default $DATA_ROOT/final_halluc_stage3)
#   HALLUC_TOTAL=14000    hallucination-domain row quota, train + val (mix_halluc
#                         --halluc_total; --stage3_mix rescales the D30 table on
#                         this total, so 14000 / 0.75 reproduces it to the row)
#   OLD_DOMAIN_RATIO=0.6  stage-1 share of the stage-3 train mix (with the
#                         hallucination domain's 25% solvable this makes the
#                         whole artifact 70/30)
#   CODE_SHARE=0.24       share of the whole train set that must be ability
#                         "code". The hallucination domain has no code source,
#                         so the stage-1 replay is stratified: code is drawn to
#                         this share and the remaining abilities keep their
#                         stage-1 proportions (uniform sampling would dilute
#                         code to ~0.6 x its stage-1 share, i.e. ~15%)
#   VAL_SIZE=700          hallucination-domain val rows, carved out of the filled
#                         quota proportionally per cell (5% of HALLUC_TOTAL)
#   SEED=43               mix seed, the kkcut deletion-order seed (only read
#                         on the --emit-unsolvable path) AND crepe's quota
#                         subsampling seed (43 != stage 2's default 0, so the
#                         stage-3 crepe rows do not re-draw the stage-2 rows)

set -xeuo pipefail

########################### build configuration ###########################
PYTHON_BIN=${PYTHON_BIN:-python3}
DATA_ROOT=${DATA_ROOT:-$HOME/data/reasoning_rl}
STAGE1_DATA_DIR=${STAGE1_DATA_DIR:-$DATA_ROOT/final}
STAGE3_DIR=${STAGE3_DIR:-$DATA_ROOT/final_halluc_stage3}
RAW_ROOT=$DATA_ROOT/halluc/raw
BUILT_DIR=$DATA_ROOT/halluc/built_stage3

HALLUC_TOTAL=${HALLUC_TOTAL:-14000}
OLD_DOMAIN_RATIO=${OLD_DOMAIN_RATIO:-0.6}
CODE_SHARE=${CODE_SHARE:-0.24}
VAL_SIZE=${VAL_SIZE:-700}
SEED=${SEED:-43}

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

# 1c. Per-source adapters, in the plain stage-2 contracts.  Two differences
#     from the stage-2 build:
#     * kk_adapter gets --emit-unsolvable --kkcut-per-group 0 (D30): every
#       certified-unique parent group emits *every* admissible statement
#       deletion (the first-only cap yielded 490 rows, short of the 2,100 the
#       quota needs), as kkcut:<family>:<N>ppl:<index>:<deleted> bare-refusal
#       rows in the same parquet.  $SEED fixes the deletion order.
#     * crepe_adapter gets --quota-overrides (D18 table B, stage-3 rescale):
#       400 solvable-judge / 1280 unsolvable-bare instead of the 250/400 the
#       stage-2 table fixes, because commonsense now owns 20% of the pool.
#       --seed "$SEED" re-keys its quota subsampling away from the stage-2
#       default, so the stage-3 sample is not a superset of the stage-2 rows.
mkdir -p "$BUILT_DIR"
$PYTHON_BIN "$HALLUC_SCRIPTS/kk_adapter.py"        --raw-dir "$RAW_ROOT/kk"     --out "$BUILT_DIR/kk.parquet"     --emit-unsolvable --kkcut-per-group 0 --seed "$SEED"
$PYTHON_BIN "$HALLUC_SCRIPTS/mip_adapter.py"       --raw-dir "$RAW_ROOT/mip"    --out "$BUILT_DIR/mip.parquet"
$PYTHON_BIN "$HALLUC_SCRIPTS/falseqa_adapter.py"   --raw-dir "$RAW_ROOT/falseqa" --out "$BUILT_DIR/falseqa.parquet"
$PYTHON_BIN "$HALLUC_SCRIPTS/gsm_ic_adapter.py"    --raw-dir "$RAW_ROOT/gsmic"  --out "$BUILT_DIR/gsmic.parquet"
$PYTHON_BIN "$HALLUC_SCRIPTS/sum_adapter.py"       --raw-dir "$RAW_ROOT/sum"    --out "$BUILT_DIR/sum.parquet"
$PYTHON_BIN "$HALLUC_SCRIPTS/umwp_adapter.py"      --raw-dir "$RAW_ROOT/umwp"   --out "$BUILT_DIR/umwp.parquet"
$PYTHON_BIN "$HALLUC_SCRIPTS/treecut_adapter.py"   --raw-dir "$RAW_ROOT/treecut" --out "$BUILT_DIR/treecut.parquet"
$PYTHON_BIN "$HALLUC_SCRIPTS/crepe_adapter.py"     --raw-dir "$RAW_ROOT/crepe"  --out "$BUILT_DIR/crepe.parquet" --quota-overrides solvable_judge=400,unsolvable_bare=1280 --seed "$SEED"

# 1d. D17 synthesised distractors (the main-pool slice of the stage-3 table).
#     Same fallback as the stage-2 build: without a stage-1 math parquet the
#     cell is left to mix_halluc's shortfall redistribution.
DISTRACTOR_ARGS=(--raw-dir "$RAW_ROOT" --out "$BUILT_DIR/d17.parquet")
if [ -f "$STAGE1_DATA_DIR/train.parquet" ]; then
    DISTRACTOR_ARGS+=(--stage1-path "$STAGE1_DATA_DIR/train.parquet")
else
    echo "WARNING: $STAGE1_DATA_DIR/train.parquet not found; building distractors without the stage-1 main pool" >&2
    DISTRACTOR_ARGS+=(--main-quota 0)
fi
$PYTHON_BIN "$HALLUC_SCRIPTS/distractor_synth.py" "${DISTRACTOR_ARGS[@]}"

# 1e. Stage-3 mix. --stage3_mix swaps the quota base to the D30 table (logic
#     20% / commonsense 20% / math 60% by rows, 25/75 solvable/unsolvable by
#     edge weight inside the hallucination domain; with old_domain_ratio 0.6 the
#     whole artifact is 70/30).  The other knobs are shared with the stage-2
#     mixer (old-domain replay, near-dedup anchors), except that stage 3 runs
#     the dedup passes before the fill and carves val out of the filled rows.
#     --strict_cells turns a short quota cell into a hard error: if the kkcut
#     supply cannot reach the logic cell, the build stops here and names the
#     gap instead of silently shifting it onto another data source.
$PYTHON_BIN "$HALLUC_SCRIPTS/mix_halluc.py" \
    --stage3_mix \
    --strict_cells \
    --halluc_dir "$BUILT_DIR" \
    --stage1_path "$STAGE1_DATA_DIR" \
    --output_dir "$STAGE3_DIR" \
    --halluc_total "$HALLUC_TOTAL" \
    --halluc_unsolvable_ratio 0.75 \
    --old_domain_ratio "$OLD_DOMAIN_RATIO" \
    --code_share "$CODE_SHARE" \
    --val_size "$VAL_SIZE" \
    --seed "$SEED"

########################### post-build checks ###########################
for f in "$STAGE3_DIR/train.parquet" "$STAGE3_DIR/val.parquet"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: required artifact missing after build: $f" >&2
        exit 1
    fi
done

# The D30 rows are audited against the raw files: parent uniqueness, the
# >= 2-solutions certificate and the one-sentence surgery are re-derived by
# verify_kk's own enumerator -- a build that mints a bad cut row fails here,
# not in training.
$PYTHON_BIN "$HALLUC_SCRIPTS/verify_kk.py" --rows "$BUILT_DIR/kk.parquet" --raw-dir "$RAW_ROOT/kk"

if [ "${RUN_TESTS:-0}" = "1" ]; then
    $PYTHON_BIN -m pytest "$HALLUC_SCRIPTS" "$REPO_ROOT/examples/reasoning_rl/reward/test_hallucination_compute_score.py"
fi

echo "Stage-3 data ready: $STAGE3_DIR/{train,val}.parquet"
echo "Next: RESUME_PATH=<stage2_ckpt_dir> bash run_qwen3_4b_hallucination_stage3_fully_async.sh"
