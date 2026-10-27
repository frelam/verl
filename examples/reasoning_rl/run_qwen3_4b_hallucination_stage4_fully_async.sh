#!/usr/bin/env bash
# Hallucination-resistance stage 4 (HALLUCINATION_RL_DESIGN.md D31 + D32) | Qwen3-4B
#
# Thin stage-4 wrapper around the stage-3 fully-async recipe.  Stage 4 uses the
# same trainer / rollout / reward stack; only the data directory changes:
#
#   * stage-3 domain-balanced mix (hallucination + old-domain replay),
#   * + instruction-following domain (D31, ~15%),
#   * + BBH-gap skill-repair domain (D32, ~10%; built by bbh_adapter.py from
#     fresh same-ability generators, never from BBH rows),
#   * + fresh old-domain replay (--exclude_path).
#
# The reward is still reward/hallucination_compute_score.py: `halluc_*` rows use
# the abstention dispatcher, `if_*` rows delegate to the IF verifier, and
# `bbh_*` rows delegate to reward/bbh_verifier.py (benchmark-native bracket /
# option-letter / yes-no answers).
#
# Build the data first:
#   bash build_hallucination_stage4_data.sh
#
# Usage:
#   RESUME_PATH=<stage3_ckpt_dir> bash run_qwen3_4b_hallucination_stage4_fully_async.sh
#
# All knobs of the stage-3 recipe (TOTAL_STEPS, TRAIN_FILES, VAL_FILES,
# STALENESS_THRESHOLD, ...) are still available; DATA_DIR / EXPERIMENT_NAME
# default to the stage-4 values and can be overridden from the environment.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE4_SCRIPT="$REPO_ROOT/examples/reasoning_rl/run_qwen3_4b_hallucination_stage3_fully_async.sh"

DATA_DIR=${DATA_DIR:-$HOME/data/reasoning_rl/final_halluc_stage4}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_hallucination_stage4_$(date +%Y%m%d_%H%M)}
export DATA_DIR EXPERIMENT_NAME

if [ -z "${RESUME_PATH:-}" ] && [ "${RESUME_MODE:-resume_path}" = "resume_path" ]; then
    echo "ERROR: stage 4 resumes from the stage-3 checkpoint; set RESUME_PATH=<ckpt_dir>" >&2
    exit 1
fi
if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "ERROR: $DATA_DIR/train.parquet missing; run build_hallucination_stage4_data.sh first" >&2
    exit 1
fi

exec bash "$STAGE4_SCRIPT" "$@"
