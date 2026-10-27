#!/usr/bin/env bash
# Hallucination-resistance stage 4 (HALLUCINATION_RL_DESIGN.md D31 + D32): DATA BUILD ONLY.
#
# Stage 4 = the stage-3 domain-balanced mix PLUS two additive domains:
#
#   * instruction-following (IF) at ~15% of the final train set, built by
#     if_adapter.py from nvidia/Nemotron-RL-instruction_following (WildChat
#     prompts + IFBench constraints), a source stages 1-3 never trained on;
#   * a BBH-gap skill-repair domain at ~10% of the final train set, built by
#     bbh_adapter.py from *fresh* same-ability generators (D32).  It targets the
#     five BIG-Bench-Hard subtasks the stage-3 checkpoint is weakest on:
#       dyck_languages (40%), disambiguation_qa (64%), causal_judgement (64.7%),
#       movie_recommendation (66.4%), geometric_shapes (74%).
#     No BBH/BIG-Bench test row is read: Dyck / geometry / disambiguation /
#     causal rows are regenerated from scratch, and movie_recommendation uses
#     the raw MovieLens latest-small ratings (the public source the BIG-Bench
#     task itself used), not the benchmark queries.
#
# The additive share formula over the fixed non-additive part N (halluc train +
# old-domain replay) is:
#     if  = IF_RATIO  / (1 - IF_RATIO  - BBH_RATIO) * N
#     bbh = BBH_RATIO / (1 - IF_RATIO  - BBH_RATIO) * N
# so the final train set has exactly the requested shares and the stage-3 cells
# keep their absolute size.
#
# The stage-3 hallucination table is otherwise unchanged: `if_ratio` / `bbh_ratio`
# are *additive* shares of the final train set.
#
# Idempotent build pipeline:
#   fetch_raw.py                    -> $DATA_ROOT/halluc/raw/<source>/
#   treecut-math clone              -> $DATA_ROOT/halluc/raw/treecut/repo
#   <source>_adapter.py  x 8        -> $DATA_ROOT/halluc/built_stage4/<source>.parquet
#                                      (kk with --emit-unsolvable --kkcut-per-group 0,
#                                      crepe with --quota-overrides; SEED=44 rotates
#                                      the kkcut order / CREPE subsample vs stage 3)
#   if_adapter.py                   -> $DATA_ROOT/halluc/built_stage4/if.parquet
#   bbh_adapter.py                  -> $DATA_ROOT/halluc/built_stage4/bbh.parquet
#                                      (Dyck/geometry/disambiguation/causal are
#                                      generated; MovieLens is downloaded into
#                                      $DATA_ROOT/halluc/raw/bbh/ if missing)
#   distractor_synth.py (D17)       -> $DATA_ROOT/halluc/built_stage4/d17.parquet
#   mix_halluc.py --stage3_mix      -> $STAGE4_DIR/{train,val}.parquet + mix_stats.json
#        + --if_ratio/--if_path + --bbh_ratio/--bbh_path/--bbh_weights
#        + --old_pool_path/--exclude_path
#
# Training is a separate step: use run_qwen3_4b_hallucination_stage4_fully_async.sh
# with RESUME_PATH=<stage-3 ckpt> (the stage-3 recipe with DATA_DIR=$STAGE4_DIR).
#
# Usage:
#   bash build_hallucination_stage4_data.sh
#
# Env knobs:
#   RUN_TESTS=0           run the hallucination + BBH unit tests after the build
#   PYTHON_BIN=python3    interpreter used by the build scripts
#   DATA_ROOT             default ~/data/reasoning_rl
#   STAGE1_DATA_DIR       stage-1 mix dir with train.parquet / val.parquet
#                         (default $DATA_ROOT/final); supplies the old-domain val
#                         rows and the D17 distractor base pool
#   STAGE4_DIR            stage-4 output dir (default $DATA_ROOT/final_halluc_stage4)
#   HALLUC_TOTAL=14000    hallucination-domain row quota, train + val
#   OLD_DOMAIN_RATIO=0.6  stage-1 share of the *non-additive* part (keeps the
#                         stage-3 70/30 solvable/unsolvable artifact inside it)
#   CODE_SHARE=0.24       share of the FINAL train set that must be ability "code"
#   VAL_SIZE=700          hallucination-domain val rows
#   IF_RATIO=0.15         instruction-following share of the final train set
#   IF_GRADING_MODE=binary  IF reward mode written into the ground truth
#                           (binary = NeMo-Gym all-or-nothing; fraction = passed/total)
#   IF_QUOTA=0            cap on IF rows in the pool (0 = all admissible rows)
#   IF_MINHASH_THRESHOLD=0  optional in-pool near-dedup for the IF pool
#   IF_PATH               local jsonl/parquet override for the IF source
#                           (default: HF repo; a cached raw file is used if present)
#   BBH_RATIO=0.10        BBH-gap share of the final train set
#   BBH_TASKS             comma-separated adapter tasks (default all five)
#   BBH_QUOTA             comma-separated task=row pool quotas
#   BBH_WEIGHTS           per-source shares inside the BBH domain
#   MOVIELENS_DIR         local MovieLens latest-small dir (movies.csv/ratings.csv);
#                         default $RAW_ROOT/bbh, auto-downloaded when missing
#   BBH_MIN_MOVIE_RATINGS=20  drop movies with fewer ratings than this (lower it
#                         if the local MovieLens slice is too small)
#   INCLUDE_OLD_POOLS=1   draw the old-domain replay from the raw pools below and
#                         exclude every previously trained artifact. Set to 0 to
#                         replay from $STAGE1_DATA_DIR only (falls back to the
#                         stage-2 behaviour: no freshness).
#   OLD_POOL_PATHS        raw domain pool files/dirs (default the four standard
#                         train_*.parquet under $DATA_ROOT)
#   EXCLUDE_PATHS         previously trained artifacts (default $DATA_ROOT/final,
#                         final_halluc, final_halluc_stage3 — whichever exist)
#   SEED=44               mix seed, K&K deletion-order seed, CREPE subsample seed,
#                         IF quota-draw seed and BBH generator seed (44 != 43)

set -xeuo pipefail

########################### build configuration ###########################
PYTHON_BIN=${PYTHON_BIN:-python3}
DATA_ROOT=${DATA_ROOT:-$HOME/data/reasoning_rl}
STAGE1_DATA_DIR=${STAGE1_DATA_DIR:-$DATA_ROOT/final}
STAGE4_DIR=${STAGE4_DIR:-$DATA_ROOT/final_halluc_stage4}
RAW_ROOT=$DATA_ROOT/halluc/raw
BUILT_DIR=$DATA_ROOT/halluc/built_stage4

HALLUC_TOTAL=${HALLUC_TOTAL:-14000}
OLD_DOMAIN_RATIO=${OLD_DOMAIN_RATIO:-0.6}
CODE_SHARE=${CODE_SHARE:-0.24}
VAL_SIZE=${VAL_SIZE:-700}
SEED=${SEED:-44}

IF_RATIO=${IF_RATIO:-0.15}
IF_GRADING_MODE=${IF_GRADING_MODE:-binary}
IF_QUOTA=${IF_QUOTA:-0}
IF_MINHASH_THRESHOLD=${IF_MINHASH_THRESHOLD:-0}
IF_PATH=${IF_PATH:-}

# BBH-gap domain (D32): fresh same-ability generators, never the benchmark rows.
BBH_RATIO=${BBH_RATIO:-0.10}
BBH_TASKS=${BBH_TASKS:-dyck_languages,geometric_shapes,disambiguation_qa,causal_judgement,movie_recommendation}
BBH_QUOTA=${BBH_QUOTA:-dyck_languages=3000,geometric_shapes=2000,disambiguation_qa=2000,causal_judgement=2000,movie_recommendation=1500}
BBH_WEIGHTS=${BBH_WEIGHTS:-dyck=0.30,geometric=0.20,disambiguation=0.20,causal=0.15,movie=0.15}
MOVIELENS_DIR=${MOVIELENS_DIR:-$RAW_ROOT/bbh}
BBH_MIN_MOVIE_RATINGS=${BBH_MIN_MOVIE_RATINGS:-20}

INCLUDE_OLD_POOLS=${INCLUDE_OLD_POOLS:-1}
OLD_POOL_PATHS=${OLD_POOL_PATHS:-"$DATA_ROOT/math/train_math.parquet $DATA_ROOT/code/train_code.parquet $DATA_ROOT/logic/train_logic.parquet $DATA_ROOT/stem/train_stem.parquet"}
EXCLUDE_PATHS=${EXCLUDE_PATHS:-"$DATA_ROOT/final $DATA_ROOT/final_halluc $DATA_ROOT/final_halluc_stage3"}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HALLUC_SCRIPTS=$REPO_ROOT/examples/reasoning_rl/scripts/hallucination

# 1a. Raw downloads (idempotent; completed files are cached and sha256-manifested).
$PYTHON_BIN "$HALLUC_SCRIPTS/fetch_raw.py" --dest "$RAW_ROOT"

# 1b. TreeCut is a generator repo, not a file (PENDING_SUBDIRS in fetch_raw).
if [ ! -d "$RAW_ROOT/treecut/repo/treecut" ]; then
    mkdir -p "$RAW_ROOT/treecut"
    git clone https://github.com/j-bagel/treecut-math "$RAW_ROOT/treecut/repo"
fi

# 1c. Per-source adapters, in the same plain stage-2 contracts the stage-3 build
#     uses.  SEED rotates the kkcut deletion order and the CREPE quota subsample
#     away from stage 3 (43), so stage 4 does not re-draw exactly the same rows.
mkdir -p "$BUILT_DIR"
$PYTHON_BIN "$HALLUC_SCRIPTS/kk_adapter.py"        --raw-dir "$RAW_ROOT/kk"     --out "$BUILT_DIR/kk.parquet"     --emit-unsolvable --kkcut-per-group 0 --seed "$SEED"
$PYTHON_BIN -c "
import json, sys
r = json.load(open('$BUILT_DIR/kk_report.json'))['kkcut']
print('[kk] kkcut emitted %d from %d groups (cap %s), skipped %s' % (
    r['emitted'], r['groups_attempted'], r['per_group_cap'], r['skipped']))
sys.exit(0 if r['emitted'] else 1)
" || { echo "ERROR: kk_adapter emitted no kkcut rows; the stage-4 logic-unsolvable cell cannot be filled." >&2; exit 1; }
$PYTHON_BIN "$HALLUC_SCRIPTS/mip_adapter.py"       --raw-dir "$RAW_ROOT/mip"    --out "$BUILT_DIR/mip.parquet"
$PYTHON_BIN "$HALLUC_SCRIPTS/falseqa_adapter.py"   --raw-dir "$RAW_ROOT/falseqa" --out "$BUILT_DIR/falseqa.parquet"
$PYTHON_BIN "$HALLUC_SCRIPTS/gsm_ic_adapter.py"    --raw-dir "$RAW_ROOT/gsmic"  --out "$BUILT_DIR/gsmic.parquet"
$PYTHON_BIN "$HALLUC_SCRIPTS/sum_adapter.py"       --raw-dir "$RAW_ROOT/sum"    --out "$BUILT_DIR/sum.parquet"
$PYTHON_BIN "$HALLUC_SCRIPTS/umwp_adapter.py"      --raw-dir "$RAW_ROOT/umwp"   --out "$BUILT_DIR/umwp.parquet"
$PYTHON_BIN "$HALLUC_SCRIPTS/treecut_adapter.py"   --raw-dir "$RAW_ROOT/treecut" --out "$BUILT_DIR/treecut.parquet"
$PYTHON_BIN "$HALLUC_SCRIPTS/crepe_adapter.py"     --raw-dir "$RAW_ROOT/crepe"  --out "$BUILT_DIR/crepe.parquet" --quota-overrides solvable_judge=400,unsolvable_bare=1280 --seed "$SEED"

# 1d. D17 synthesised distractors (same as stage 3).
DISTRACTOR_ARGS=(--raw-dir "$RAW_ROOT" --out "$BUILT_DIR/d17.parquet")
if [ -d "$STAGE1_DATA_DIR" ]; then
    DISTRACTOR_ARGS+=(--stage1-path "$STAGE1_DATA_DIR")
else
    echo "WARNING: $STAGE1_DATA_DIR not found; building distractors without the stage-1 main pool" >&2
    DISTRACTOR_ARGS+=(--main-quota 0)
fi
$PYTHON_BIN "$HALLUC_SCRIPTS/distractor_synth.py" "${DISTRACTOR_ARGS[@]}"

# 1e. Exclusion set: every artifact the model has already been trained on.  It
#     is applied to the old-domain replay pool AND to the IF pool, so neither
#     re-trains a prompt that stages 1-3 already saw.
EXCLUDE_ARGS=()
for path in $EXCLUDE_PATHS; do
    if [ -e "$path" ]; then
        EXCLUDE_ARGS+=(--exclude_path "$path")
    fi
done
if [ ${#EXCLUDE_ARGS[@]} -eq 0 ]; then
    echo "WARNING: none of EXCLUDE_PATHS exist; stage 4 will not filter previously trained rows" >&2
fi

# 1f. Instruction-following pool (the new domain).  Local raw file if IF_PATH is
#     given, otherwise the HF repo (a cached file under $RAW_ROOT/if wins).
IF_ARGS=(--raw-dir "$RAW_ROOT/if" --out "$BUILT_DIR/if.parquet" --seed "$SEED" --grading-mode "$IF_GRADING_MODE" --minhash-threshold "$IF_MINHASH_THRESHOLD")
if [ ${#EXCLUDE_ARGS[@]} -gt 0 ]; then
    IF_ARGS+=("${EXCLUDE_ARGS[@]}")
fi
if [ -n "$IF_PATH" ]; then
    IF_ARGS+=(--if-path "$IF_PATH")
fi
if [ "$IF_QUOTA" -gt 0 ]; then
    IF_ARGS+=(--quota "$IF_QUOTA")
fi
$PYTHON_BIN "$HALLUC_SCRIPTS/if_adapter.py" "${IF_ARGS[@]}"

# 1g. BBH-gap pool (D32).  This adapter deliberately does NOT read any
#     BBH/BIG-Bench row: Dyck, geometric-shape, disambiguation and causal rows
#     are regenerated from fresh same-ability generators, while the movie rows
#     are built from raw MovieLens latest-small ratings (auto-downloaded into
#     $MOVIELENS_DIR when movies.csv/ratings.csv are absent).  The task list and
#     per-source weights are written into the report next to the parquet.
BBH_ARGS=(
    --raw-dir "$RAW_ROOT/bbh"
    --out "$BUILT_DIR/bbh.parquet"
    --tasks "$BBH_TASKS"
    --quota "$BBH_QUOTA"
    --seed "$SEED"
    --movielens-dir "$MOVIELENS_DIR"
    --min-movie-ratings "$BBH_MIN_MOVIE_RATINGS"
)
if [ ${#EXCLUDE_ARGS[@]} -gt 0 ]; then
    BBH_ARGS+=("${EXCLUDE_ARGS[@]}")
fi
$PYTHON_BIN "$HALLUC_SCRIPTS/bbh_adapter.py" "${BBH_ARGS[@]}"

# 1h. Old-domain freshness: raw pools + exclusion.  Without the raw pools the
#     replay would come from $STAGE1_DATA_DIR, i.e. the stage-1 mix the previous
#     stages already trained on.
OLD_POOL_ARGS=()
if [ "$INCLUDE_OLD_POOLS" = "1" ]; then
    for path in $OLD_POOL_PATHS; do
        if [ -e "$path" ]; then
            OLD_POOL_ARGS+=(--old_pool_path "$path")
        fi
    done
    if [ ${#OLD_POOL_ARGS[@]} -eq 0 ]; then
        echo "ERROR: INCLUDE_OLD_POOLS=1 but none of OLD_POOL_PATHS exist:" >&2
        echo "       $OLD_POOL_PATHS" >&2
        echo "       build the domain pools first, or set INCLUDE_OLD_POOLS=0 to replay from $STAGE1_DATA_DIR" >&2
        exit 1
    fi
fi

# 1i. Stage-4 mix.  --stage3_mix keeps the stage-3 hallucination table; the IF
#     and BBH domains are additive (--if_ratio / --bbh_ratio), so the stage-3
#     cells keep their absolute size and only the total grows.  --strict_cells
#     stays on: a short halluc cell must fail the build rather than silently
#     shift the domain weights.
MIX_ARGS=(
    --stage3_mix
    --strict_cells
    --halluc_dir "$BUILT_DIR"
    --stage1_path "$STAGE1_DATA_DIR"
    --if_path "$BUILT_DIR/if.parquet"
    --if_ratio "$IF_RATIO"
    --bbh_path "$BUILT_DIR/bbh.parquet"
    --bbh_ratio "$BBH_RATIO"
    --bbh_weights "$BBH_WEIGHTS"
    --output_dir "$STAGE4_DIR"
    --halluc_total "$HALLUC_TOTAL"
    --halluc_unsolvable_ratio 0.75
    --old_domain_ratio "$OLD_DOMAIN_RATIO"
    --code_share "$CODE_SHARE"
    --val_size "$VAL_SIZE"
    --seed "$SEED"
)
if [ ${#OLD_POOL_ARGS[@]} -gt 0 ]; then
    MIX_ARGS+=("${OLD_POOL_ARGS[@]}")
fi
if [ ${#EXCLUDE_ARGS[@]} -gt 0 ]; then
    MIX_ARGS+=("${EXCLUDE_ARGS[@]}")
fi
$PYTHON_BIN "$HALLUC_SCRIPTS/mix_halluc.py" "${MIX_ARGS[@]}"

########################### post-build checks ###########################
for f in "$STAGE4_DIR/train.parquet" "$STAGE4_DIR/val.parquet" "$BUILT_DIR/if.parquet" "$BUILT_DIR/bbh.parquet"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: required artifact missing after build: $f" >&2
        exit 1
    fi
done

# Fail here, not in training, if either additive domain missed its requested
# share (a short pool or an over-aggressive exclusion is the usual cause).
$PYTHON_BIN -c "
import json, sys
s = json.load(open('$STAGE4_DIR/mix_stats.json'))
for name, share_key, want_key, rows_key in (
    ('IF', 'if_share_of_train', 'if_ratio', 'if_rows_in_train'),
    ('BBH', 'bbh_share_of_train', 'bbh_ratio', 'bbh_rows_in_train'),
):
    share = s['achieved_ratios'][share_key]
    want = s['config'][want_key]
    print('[stage4] %s share of train: %.4f (requested %.4f), rows %d/%d' % (
        name, share, want, s[rows_key], s['total_train']))
    if share < want - 0.01:
        print('[stage4] ERROR: %s share is more than 1pt below the request' % name, file=sys.stderr)
        sys.exit(1)
print('[stage4] BBH rows by source:', s['bbh_by_source'])
print('[stage4] BBH train/val:', s['bbh_rows_in_train'], s['bbh_rows_in_val'])
"

# The D30 kkcut rows are audited against the raw files.
$PYTHON_BIN "$HALLUC_SCRIPTS/verify_kk.py" --rows "$BUILT_DIR/kk.parquet" --raw-dir "$RAW_ROOT/kk"

if [ "${RUN_TESTS:-0}" = "1" ]; then
    $PYTHON_BIN -m pytest "$HALLUC_SCRIPTS" \
        "$REPO_ROOT/examples/reasoning_rl/reward/test_hallucination_compute_score.py" \
        "$REPO_ROOT/examples/reasoning_rl/reward/test_if_verifier.py" \
        "$REPO_ROOT/examples/reasoning_rl/reward/test_bbh_verifier.py"
fi

echo "Stage-4 data ready: $STAGE4_DIR/{train,val}.parquet"
echo "Next: RESUME_PATH=<stage3_ckpt_dir> bash run_qwen3_4b_hallucination_stage4_fully_async.sh"
