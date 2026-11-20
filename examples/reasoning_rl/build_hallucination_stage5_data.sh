#!/usr/bin/env bash
# Hallucination-resistance stage 5 (D35): DATA BUILD ONLY.
#
# Stage 5 keeps the stage-4 data stack (same adapters, reward, trainer and
# freshness rules) and rebalances the mix for a final math/code/logic push:
#
#     hallucination 5% + old-domain replay 35% + hard math 26%
#       + instruction-following 10% + BBH-gap 10% + logic 10% + stem 4%
#
# Compared with stage 4:
#   * the hallucination domain drops from 30% to 5% of the final train set.
#     Stages 2-4 already installed the abstention / pointer / judgment
#     behaviour, so stage 5 keeps only a small maintenance slice; the slice is
#     re-weighted 20/20/60 (math/logic/commonsense) so commonsense gets 3% of
#     the final train instead of 1%;
#   * hard math drops from 30% to 26% (still far above stage 4's 15%): the D34
#     pool is rebuilt by hard_math_adapter.py and re-difficulty-filtered on the
#     stage-4 starting checkpoint; the candidate quota stays at 55k and the
#     pass@k filter now generates with max_tokens=20000 so it sees the same
#     response budget as the stage-5 rollout;
#   * code rises from 18% to 30% via --code_share 0.30, drawn from the fresh
#     old-domain replay pool (stages 1-4 artifacts are excluded);
#   * IF drops from 15% to 10% and the old replay from 40% to 35%; the freed
#     shares go to the new additive logic (10%) and stem (4%) domains, drawn
#     from the raw logic/stem pools and exact-deduped against the replay rows;
#   * BBH stays at 10%.
#
# Because the ability target is deliberately different from stage 3, this build
# turns --match_stage3_abilities OFF.  --code_share pins code at 30% of the
# final train set and the rest of the replay follows the raw-pool ability mix
# (math 45% / logic 15% / stem 15%).  The build fails closed when the code pool
# cannot supply the requested share (FAIL_ON_CODE_SHORTFALL=1); a silent
# under-fill would quietly undo the stage-5 rebalance.
#
# At the default knobs (train total ~44.3k):
#   hallucination 2,217 (5.0%) + old replay 15,519 (35.0%)
#     + hard math 11,528 (26.0%) + IF 4,434 (10.0%) + BBH 4,434 (10.0%)
#     + logic 4,434 (10.0%) + stem 1,774 (4.0%)
#
# Post-build assertions: additive shares (including logic/stem), old-domain
# share, code share and the 5% hallucination share must all land within 1pt.
#
# Usage:
#   HARD_MATH_DIFFICULTY_BASE_URL=http://127.0.0.1:8000/v1 \
#   HARD_MATH_DIFFICULTY_MODEL=<served stage-4 start ckpt> \
#       bash build_hallucination_stage5_data.sh
#
#   SKIP_HARD_MATH_DIFFICULTY=1 bash build_hallucination_stage5_data.sh   # debug only
#
# This is a thin wrapper around build_hallucination_stage4_data.sh; every other
# stage-4 knob (DECONTAMINATE_PATHS, BBH_QUOTA, HARD_MATH_WEIGHTS, ...) is still
# available and can be overridden from the environment.
#
# Environment defaults set here:
#   BUILT_DIR=$DATA_ROOT/halluc/built_stage5
#   STAGE5_DIR=$DATA_ROOT/final_halluc_stage5
#   HALLUC_TOTAL=2917      hallucination train 2,217 + val 700
#   VAL_SIZE=700
#   OLD_DOMAIN_RATIO=0.875  old replay = 35% of the final train set
#   CODE_SHARE=0.30        code = 30% of the final train set
#   FAIL_ON_CODE_SHORTFALL=1
#   MATCH_STAGE3_ABILITIES=0
#   HARD_MATH_RATIO=0.26
#   HARD_MATH_CANDIDATES=55000
#   HARD_MATH_DIFFICULTY_MAX_TOKENS=20000  pass@k filter response budget
#   IF_RATIO=0.10
#   BBH_RATIO=0.10
#   LOGIC_RATIO=0.10
#   LOGIC_PATH=$DATA_ROOT/logic/train_logic.parquet
#   STEM_RATIO=0.04
#   STEM_PATH=$DATA_ROOT/stem/train_stem.parquet
#   HALLUC_DOMAIN_WEIGHTS=math=0.2,logic=0.2,commonsense=0.6
#   SEED=45
#   EXCLUDE_PATHS=final final_halluc final_halluc_stage3 final_halluc_stage4
#                 final_halluc_stage5

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT=${DATA_ROOT:-$HOME/data/reasoning_rl}
STAGE5_DIR=${STAGE5_DIR:-$DATA_ROOT/final_halluc_stage5}
PYTHON_BIN=${PYTHON_BIN:-python3}

export DATA_ROOT
export PYTHON_BIN
export STAGE4_DIR=${STAGE4_DIR:-$STAGE5_DIR}
export BUILT_DIR=${BUILT_DIR:-$DATA_ROOT/halluc/built_stage5}

export HALLUC_TOTAL=${HALLUC_TOTAL:-2917}
export VAL_SIZE=${VAL_SIZE:-700}
export OLD_DOMAIN_RATIO=${OLD_DOMAIN_RATIO:-0.875}
export CODE_SHARE=${CODE_SHARE:-0.30}
export FAIL_ON_CODE_SHORTFALL=${FAIL_ON_CODE_SHORTFALL:-1}
export MATCH_STAGE3_ABILITIES=${MATCH_STAGE3_ABILITIES:-0}
export HARD_MATH_RATIO=${HARD_MATH_RATIO:-0.26}
export HARD_MATH_CANDIDATES=${HARD_MATH_CANDIDATES:-55000}
export HARD_MATH_DIFFICULTY_MAX_TOKENS=${HARD_MATH_DIFFICULTY_MAX_TOKENS:-20000}
export IF_RATIO=${IF_RATIO:-0.10}
export BBH_RATIO=${BBH_RATIO:-0.10}
export LOGIC_RATIO=${LOGIC_RATIO:-0.10}
export LOGIC_PATH=${LOGIC_PATH:-$DATA_ROOT/logic/train_logic.parquet}
export STEM_RATIO=${STEM_RATIO:-0.04}
export STEM_PATH=${STEM_PATH:-$DATA_ROOT/stem/train_stem.parquet}
export HALLUC_DOMAIN_WEIGHTS=${HALLUC_DOMAIN_WEIGHTS:-math=0.2,logic=0.2,commonsense=0.6}
export SEED=${SEED:-45}
export EXCLUDE_PATHS=${EXCLUDE_PATHS:-"$DATA_ROOT/final $DATA_ROOT/final_halluc $DATA_ROOT/final_halluc_stage3 $DATA_ROOT/final_halluc_stage4 $DATA_ROOT/final_halluc_stage5"}

echo "Stage-5 mix: halluc 5% (20/20/60) / old 35% / hard math 26% / IF 10% / BBH 10% / logic 10% / stem 4% (code 30%)"
bash "$REPO_ROOT/examples/reasoning_rl/build_hallucination_stage4_data.sh" "$@"
# Stage 5's defining constraint: the hallucination slice must stay at 5% of the
# final train set.  The stage-4 builder asserts the additive, old-domain and
# code shares; this last check covers the slice that stage 5 deliberately shrank.
"$PYTHON_BIN" -c "
import json, sys
s = json.load(open('$STAGE5_DIR/mix_stats.json'))
want_rows = s['config']['halluc_total'] - s['config']['val_size']
got_rows = s['halluc_rows_in_train']
total = s['total_train']
share = got_rows / total if total else 0.0
target = want_rows / total if total else 0.0
print('[stage5] hallucination share of train: %.4f (target %.4f), rows %d/%d' % (
    share, target, got_rows, total))
if abs(share - target) > 0.01:
    print('[stage5] ERROR: hallucination share is more than 1pt off the target', file=sys.stderr)
    sys.exit(1)
"
echo "Stage-5 data ready: $STAGE5_DIR/{train,val}.parquet"
