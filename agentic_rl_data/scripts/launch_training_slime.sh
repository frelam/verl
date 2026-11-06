#!/usr/bin/env bash
# DEPRECATED (2026-11-06): this project trains with **verl**, not slime.
# Use `agentic_rl_data/scripts/launch_verl_agentic.sh` instead; see
# `agentic_rl_data/reports/VERL_INTEGRATION.md` for the verl interface mapping.
# The script is kept only as a historical reference for a slime-based setup.
#
# Launch agentic-RL training on slime (plan §9, slime variant).
#
# The plan leaves the four framework interfaces as TODO.  This script therefore
# does NOT guess them: it validates the data artifacts, exports them via env
# vars, and refuses to run until you provide the launch command.  Fill in:
#
#   SLIME_LAUNCH_CMD   the command that starts your slime training job, e.g.
#                        "python train.py --config configs/agentic_rl.yaml"
#                      it must read the dataset from AGENTIC_RL_DATA_TRAIN.
#   AGENTIC_RL_DATA_DIR (optional, default: agentic_rl_data/data/mixture_v1)
#
# §9 TODO checklist (implement on the slime side first, then use this script):
#   1. prompt / label data format       -> how a task JSONL row maps to samples
#   2. custom multi-turn rollout        -> env reset/step wiring, max_turns
#   3. custom reward interface          -> agentic_rl_data/rewards/get_reward
#   4. model-native tool-call format    -> chat template for `tools`
#
# Acceptance gates before launching (run with --check-only):
#   python -m pytest agentic_rl_data/tests -q
#   python -m agentic_rl_data.scripts.report --strict --allow-missing-steps passrate
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATA_DIR="${AGENTIC_RL_DATA_DIR:-agentic_rl_data/data/mixture_v1}"
TRAIN="${AGENTIC_RL_DATA_TRAIN:-$DATA_DIR/train.jsonl}"
HELDOUT="${AGENTIC_RL_DATA_HELDOUT:-$DATA_DIR/heldout.jsonl}"

if [[ ! -f "$TRAIN" ]]; then
  echo "[launch] ERROR: missing $TRAIN" >&2
  echo "[launch] Build it first: bash agentic_rl_data/scripts/build_data.sh" >&2
  echo "[launch] Any step skipped? read agentic_rl_data/data/_pipeline_manifest.json" >&2
  exit 1
fi
if [[ ! -f "$HELDOUT" ]]; then
  echo "[launch] ERROR: missing $HELDOUT (acceptance wants a real held-out split)" >&2
  exit 1
fi

export AGENTIC_RL_DATA_DIR="$DATA_DIR"
export AGENTIC_RL_DATA_TRAIN="$TRAIN"
export AGENTIC_RL_DATA_HELDOUT="$HELDOUT"

echo "[launch] data: train=$TRAIN heldout=$HELDOUT"
echo "[launch] plan §9 interfaces are user-provided; exporting AGENTIC_RL_DATA_{DIR,TRAIN,HELDOUT}"

if [[ "${1:-}" == "--check-only" ]]; then
  echo "[launch] check-only: data artifacts present; not starting training"
  exit 0
fi

if [[ -z "${SLIME_LAUNCH_CMD:-}" ]]; then
  cat >&2 <<'EOF'
[launch] ERROR: SLIME_LAUNCH_CMD is not set (plan §9 TODO).
[launch] Provide the training command, for example:
[launch]   SLIME_LAUNCH_CMD="python train.py --config configs/agentic_rl.yaml" bash agentic_rl_data/scripts/launch_training_slime.sh
[launch] The command must read the dataset path from $AGENTIC_RL_DATA_TRAIN.
EOF
  exit 1
fi

echo "[launch] starting: $SLIME_LAUNCH_CMD"
exec bash -c "$SLIME_LAUNCH_CMD"
