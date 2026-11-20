#!/usr/bin/env bash
# Hallucination-resistance stage 5 (D35) | Qwen3-4B
#
# Thin stage-5 wrapper around the stage-3 fully-async recipe.  Stage 5 uses the
# same trainer / rollout / reward stack as stage 4; only the data directory and
# the default knobs change:
#
#   * stage-5 mix (built by build_hallucination_stage5_data.sh):
#       hallucination 5% (20/20/60 math/logic/commonsense)
#       + old-domain replay 35% + hard math 26%
#       + instruction-following 10% + BBH-gap 10% + logic 10% + stem 4%
#   * the hallucination slice is now a small maintenance set (stages 2-4
#     installed the abstention behaviour) but is weighted toward commonsense;
#     hard math and code carry the capability push, and the new additive
#     logic/stem domains raise those abilities to ~12% / ~5%;
#   * the reward is still reward/hallucination_compute_score.py: `halluc_*` rows
#     use the abstention dispatcher, `if_*` rows delegate to the IF verifier,
#     `bbh_*` rows delegate to reward/bbh_verifier.py, `logic_*` rows use the
#     stage-1 logic path, and `math_hard_*` rows use the stage-1 math path
#     (compute_score.py) with the D34 exact-rational literal fallback.
#
# Build the data first:
#   bash build_hallucination_stage5_data.sh
#
# Usage:
#   RESUME_PATH=<stage-4 ckpt_dir> bash run_qwen3_4b_hallucination_stage5_fully_async.sh
#
# All knobs of the stage-3 recipe (TOTAL_STEPS, TRAIN_FILES, VAL_FILES,
# STALENESS_THRESHOLD, ...) are still available; DATA_DIR / EXPERIMENT_NAME
# default to the stage-5 values and can be overridden from the environment.
# ROLLOUT_N defaults to 8 here (the stage-3 recipe uses 16).  Batch accounting is
# prompt-based in fully async:
#   one mini-batch   = 256 prompts -> PPO_MINI_BATCH_SIZE=256
#   one param version/batch = 4 mini-batches = 1024 prompts (require_batches=4,
#   the stage-3 default).  In sequences that is 2048 and 8192 respectively.
# Trainer sequence parallelism is fixed at SP_SIZE=4 (Ulysses) for the actor and
# the optional ref policy; the vLLM rollout TP is unchanged.
#
# Stage-5 length penalty (HALLUCINATION_RL_DESIGN.md D33)
# -------------------------------------------------------
# Stage 5 starts from the stage-4 checkpoint.  Measure that checkpoint's own
# per-subtask average response length on this mix, then feed the targets back:
#
#   python examples/reasoning_rl/scripts/distill/measure_length_targets.py \
#       --input $DATA_DIR/train.parquet --output_dir $DATA_DIR \
#       --max_tokens 20000 \
#       --base_url http://127.0.0.1:8000/v1 --model <stage-4_served_name>
#   source $DATA_DIR/length_targets.env
#
# `--max_tokens 20000` matches the stage-5 rollout budget; the script default
# (8192) would truncate long hard-math/code answers and understate their length
# targets.
#
# `--target_scale` (stats-script default 0.6) multiplies the measured per-source
# mean before it becomes the target, i.e. the reward asks for a further 40% cut
# below the stage-4 checkpoint's own length; pass 1.0 to anchor at that length.
#
# Long-CoT sources (`math_hard_*`, `code_*`) are exempt from that mean: the stats
# script gives them a *ceiling* by default (`--long_cot_metric p90`,
# `--long_cot_scale 1.0`).  A mean target keeps charging the naturally long
# correct answers from step 0, and in all-correct groups the penalty becomes the
# only signal, so RL shortens the derivation until accuracy drops there.
# Override with `--long_cot_prefixes ""` to apply the mean uniformly again.
#
# `length_targets.env` exports the knobs below; the wrapper forwards them as
# `reward.custom_reward_function.reward_kwargs.*` (the reward consumes them and
# never forwards them to the stage-1 dispatcher).
#   LENGTH_PENALTY_ENABLED=1      the penalty is ON BY DEFAULT.  Set to 0 only for
#                                 a stage-4-parity ablation; the reward is then
#                                 byte-identical to stage 4.
#   LENGTH_PENALTY_TARGETS_PATH   length_targets.json emitted by the stats script;
#                                 defaults to $DATA_DIR/length_targets.json (must
#                                 be readable on the rollouter node).  An exported
#                                 path -- e.g. from `source length_targets.env` --
#                                 always wins.  A missing file aborts the launch
#                                 (fail-closed: never train silently without it).
#   LENGTH_PENALTY_COEF=0.2       penalty slope: coef * (len/target - 1)
#   LENGTH_PENALTY_MAX=0.5        penalty cap in points (< 1 keeps correct-long
#                                 above wrong; applied to positively scored rows only)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE3_SCRIPT="$REPO_ROOT/examples/reasoning_rl/run_qwen3_4b_hallucination_stage3_fully_async.sh"

DATA_DIR=${DATA_DIR:-$HOME/data/reasoning_rl/final_halluc_stage5}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_hallucination_stage5_$(date +%Y%m%d_%H%M)}
# Stage-5 response budget: 20k tokens for the rollout, matching the data-side
# hard-math pass@k filter (HARD_MATH_DIFFICULTY_MAX_TOKENS=20000).  The stage-3
# recipe derives ppo_max_token_len_per_gpu and rollout.max_num_batched_tokens
# from max_prompt_length + max_response_length, so no further override is needed.
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-20000}
# Stage-5 rollout group size: 8 samples per prompt (stage-3 default is 16).
ROLLOUT_N=${ROLLOUT_N:-8}
# Prompt-level mini-batch: 256 prompts (trainer multiplies it by ROLLOUT_N, i.e.
# 2048 sequences); require_batches=4 (stage-3 default) then gives 1024 prompts =
# 8192 sequences per parameter version.
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-256}
# Trainer sequence parallelism (Ulysses). Stage 5 uses SP=4 so the 20k-token
# response fits the per-GPU activation budget; the optional KL anchor gets the
# same SP setting. This is the trainer-side SP, not the vLLM rollout TP.
SP_SIZE=${SP_SIZE:-4}
export DATA_DIR EXPERIMENT_NAME MAX_RESPONSE_LENGTH ROLLOUT_N PPO_MINI_BATCH_SIZE SP_SIZE

SP_OVERRIDES=(
    "actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP_SIZE"
    "actor_rollout_ref.ref.ulysses_sequence_parallel_size=$SP_SIZE"
)

LENGTH_PENALTY_ENABLED=${LENGTH_PENALTY_ENABLED:-1}
# Default-on (D33): the target file produced by measure_length_targets.py lives
# next to the stage-5 data.  An exported LENGTH_PENALTY_TARGETS_PATH (e.g. from
# `source length_targets.env`) still takes precedence.
LENGTH_PENALTY_TARGETS_PATH=${LENGTH_PENALTY_TARGETS_PATH:-$DATA_DIR/length_targets.json}
LENGTH_PENALTY_COEF=${LENGTH_PENALTY_COEF:-0.2}
LENGTH_PENALTY_MAX=${LENGTH_PENALTY_MAX:-0.5}

if [ -z "${RESUME_PATH:-}" ] && [ "${RESUME_MODE:-resume_path}" = "resume_path" ]; then
    echo "ERROR: stage 5 resumes from the stage-4 checkpoint; set RESUME_PATH=<ckpt_dir>" >&2
    exit 1
fi
if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "ERROR: $DATA_DIR/train.parquet missing; run build_hallucination_stage5_data.sh first" >&2
    exit 1
fi

LENGTH_OVERRIDES=()
if [ "$LENGTH_PENALTY_ENABLED" != "0" ]; then
    if [ ! -f "$LENGTH_PENALTY_TARGETS_PATH" ]; then
        echo "ERROR: length penalty is on by default but $LENGTH_PENALTY_TARGETS_PATH is missing." >&2
        echo "       Measure the stage-4 checkpoint's lengths first (D33), then rerun:" >&2
        echo "         python examples/reasoning_rl/scripts/distill/measure_length_targets.py \\" >&2
        echo "             --input $DATA_DIR/train.parquet --output_dir $DATA_DIR \\" >&2
        echo "             --base_url http://127.0.0.1:8000/v1 --model <stage-4_served_name>" >&2
        echo "       Set LENGTH_PENALTY_ENABLED=0 for a stage-4-parity run without the penalty." >&2
        exit 1
    fi
    # The reward worker may run with a different cwd; hand it an absolute path.
    case "$LENGTH_PENALTY_TARGETS_PATH" in
        /*) ;;
        *) LENGTH_PENALTY_TARGETS_PATH="$PWD/$LENGTH_PENALTY_TARGETS_PATH" ;;
    esac
    echo "Stage-5 length penalty: targets=$LENGTH_PENALTY_TARGETS_PATH coef=$LENGTH_PENALTY_COEF max=$LENGTH_PENALTY_MAX"
    LENGTH_OVERRIDES+=(
        "++reward.custom_reward_function.reward_kwargs.length_penalty_targets_path=$LENGTH_PENALTY_TARGETS_PATH"
        "++reward.custom_reward_function.reward_kwargs.length_penalty_coef=$LENGTH_PENALTY_COEF"
        "++reward.custom_reward_function.reward_kwargs.length_penalty_max=$LENGTH_PENALTY_MAX"
    )
fi

# ${arr[@]+"${arr[@]}"} keeps `set -u` happy when an array is empty.
exec bash "$STAGE3_SCRIPT" "${SP_OVERRIDES[@]}" ${LENGTH_OVERRIDES[@]+"${LENGTH_OVERRIDES[@]}"} "$@"
