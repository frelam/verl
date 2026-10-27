#!/usr/bin/env bash
# Hallucination-resistance stage 4 (HALLUCINATION_RL_DESIGN.md D31 + D32 + D34) | Qwen3-4B
#
# Thin stage-4 wrapper around the stage-3 fully-async recipe.  Stage 4 uses the
# same trainer / rollout / reward stack; only the data directory changes:
#
#   * stage-3 domain-balanced mix (hallucination 30% + old-domain replay 30%),
#   * + instruction-following domain (D31, ~15%),
#   * + BBH-gap skill-repair domain (D32, ~10%; built by bbh_adapter.py from
#     fresh same-ability generators, never from BBH rows),
#   * + hard-math domain (D34, ~15%; AceReason / DeepScaleR / DeepMath level>=6,
#     answer-format filtered, AIME-decontaminated, pass-rate filtered to 0.1-0.9),
#   * + fresh old-domain replay (--exclude_path).
#
# The reward is still reward/hallucination_compute_score.py: `halluc_*` rows use
# the abstention dispatcher, `if_*` rows delegate to the IF verifier, `bbh_*`
# rows delegate to reward/bbh_verifier.py, and `math_hard_*` rows use the stage-1
# math path (compute_score.py) with the D34 exact-rational literal fallback.
#
# Build the data first:
#   bash build_hallucination_stage4_data.sh
#
# Usage:
#   RESUME_PATH=<dpo_ckpt_dir> bash run_qwen3_4b_hallucination_stage4_fully_async.sh
#
# All knobs of the stage-3 recipe (TOTAL_STEPS, TRAIN_FILES, VAL_FILES,
# STALENESS_THRESHOLD, ...) are still available; DATA_DIR / EXPERIMENT_NAME
# default to the stage-4 values and can be overridden from the environment.
#
# Stage-4 length penalty (HALLUCINATION_RL_DESIGN.md D33)
# -------------------------------------------------------
# Stage 4 starts from the DPO-trimmed checkpoint, so its concise chain of thought
# is the behaviour to preserve.  Measure the checkpoint's own per-subtask average
# response length on this mix, then feed the targets back here:
#
#   python examples/reasoning_rl/scripts/distill/measure_length_targets.py \
#       --input $DATA_DIR/train.parquet --output_dir <targets_dir> \
#       --base_url http://127.0.0.1:8000/v1 --model <dpo_served_name>
#   source <targets_dir>/length_targets.env
#
# `length_targets.env` exports the three knobs below; the wrapper forwards them as
# `reward.custom_reward_function.reward_kwargs.*` (the reward consumes them and
# never forwards them to the stage-1 dispatcher).  Without a targets file the
# reward is byte-identical to stage 3.
#   LENGTH_PENALTY_TARGETS_PATH   length_targets.json emitted by the stats script
#                                 (must be readable on the rollouter node; empty
#                                 = no penalty)
#   LENGTH_PENALTY_COEF=0.2       penalty slope: coef * (len/target - 1)
#   LENGTH_PENALTY_MAX=0.5        penalty cap in points (< 1 keeps correct-long
#                                 above wrong; applied to positively scored rows only)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE4_SCRIPT="$REPO_ROOT/examples/reasoning_rl/run_qwen3_4b_hallucination_stage3_fully_async.sh"

DATA_DIR=${DATA_DIR:-$HOME/data/reasoning_rl/final_halluc_stage4}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_hallucination_stage4_$(date +%Y%m%d_%H%M)}
export DATA_DIR EXPERIMENT_NAME

LENGTH_PENALTY_TARGETS_PATH=${LENGTH_PENALTY_TARGETS_PATH:-}
LENGTH_PENALTY_COEF=${LENGTH_PENALTY_COEF:-0.2}
LENGTH_PENALTY_MAX=${LENGTH_PENALTY_MAX:-0.5}

if [ -z "${RESUME_PATH:-}" ] && [ "${RESUME_MODE:-resume_path}" = "resume_path" ]; then
    echo "ERROR: stage 4 resumes from the DPO-trimmed checkpoint; set RESUME_PATH=<ckpt_dir>" >&2
    exit 1
fi
if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "ERROR: $DATA_DIR/train.parquet missing; run build_hallucination_stage4_data.sh first" >&2
    exit 1
fi

LENGTH_OVERRIDES=()
if [ -n "$LENGTH_PENALTY_TARGETS_PATH" ]; then
    if [ ! -f "$LENGTH_PENALTY_TARGETS_PATH" ]; then
        echo "ERROR: LENGTH_PENALTY_TARGETS_PATH=$LENGTH_PENALTY_TARGETS_PATH does not exist" >&2
        exit 1
    fi
    # The reward worker may run with a different cwd; hand it an absolute path.
    case "$LENGTH_PENALTY_TARGETS_PATH" in
        /*) ;;
        *) LENGTH_PENALTY_TARGETS_PATH="$PWD/$LENGTH_PENALTY_TARGETS_PATH" ;;
    esac
    echo "Stage-4 length penalty: targets=$LENGTH_PENALTY_TARGETS_PATH coef=$LENGTH_PENALTY_COEF max=$LENGTH_PENALTY_MAX"
    LENGTH_OVERRIDES+=(
        "++reward.custom_reward_function.reward_kwargs.length_penalty_targets_path=$LENGTH_PENALTY_TARGETS_PATH"
        "++reward.custom_reward_function.reward_kwargs.length_penalty_coef=$LENGTH_PENALTY_COEF"
        "++reward.custom_reward_function.reward_kwargs.length_penalty_max=$LENGTH_PENALTY_MAX"
    )
fi

# ${arr[@]+"${arr[@]}"} keeps `set -u` happy when the array is empty.
exec bash "$STAGE4_SCRIPT" ${LENGTH_OVERRIDES[@]+"${LENGTH_OVERRIDES[@]}"} "$@"
