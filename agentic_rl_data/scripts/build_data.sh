#!/usr/bin/env bash
# Build the Agentic RL dataset end to end (plan §3 steps 1-8).
#
# Usage:
#   bash agentic_rl_data/scripts/build_data.sh                     # default config
#   bash agentic_rl_data/scripts/build_data.sh --dry-run           # print plan only
#   bash agentic_rl_data/scripts/build_data.sh --strict-report     # plan §8 gate
#   CONFIG=path/to/pipeline.json bash agentic_rl_data/scripts/build_data.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-agentic_rl_data/configs/pipeline.json}"
PYTHON="${PYTHON_EXE:-python}"

exec "$PYTHON" -m agentic_rl_data.scripts.run_pipeline --config "$CONFIG" "$@"
