#!/usr/bin/env bash
# One-command data build for the v2 agentic RL recipe (Qwen3-4B, up to 24 turns).
#
#   raw sources -> data/raw_unified/*.jsonl          (build_raw, loaders)
#   procedural stateful tasks (generated, verified)  (loader: procedural_mt)
#   multi-hop QA with per-task corpora               (loader: multihop_local)
#   -> data/mixture_v2/{train,heldout}.jsonl + curriculum/  (build_mixture v2)
#   -> data/verl_v2/{train,heldout}.parquet (+ curriculum stages)
#
# Inputs (local paths or HF ids, see configs/raw_sources_v2.example.json):
#   RAW_CFG=agentic_rl_data/configs/raw_sources_v2.json
#   N_PROCEDURAL=4000      legacy generated stateful tasks (>= 2x its quota)
#   N_*                    new generated pools default to >= 2x their quotas
#   TOTAL_TRAIN=12000      mixture size
#   SMOKE=1                tiny build (120 train rows + reduced generated pools)
#
# Only the stdlib + pyarrow (parquet export) are needed; `datasets` only if a
# source is given as an hf_dataset id.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

RAW_CFG="${RAW_CFG:-agentic_rl_data/configs/raw_sources_v2.json}"
RAW_DIR="${RAW_DIR:-agentic_rl_data/data/raw_unified}"
MIX_CONFIG="${MIX_CONFIG:-agentic_rl_data/configs/mixture_v2.yaml}"
MIX_DIR="${MIX_DIR:-agentic_rl_data/data/mixture_v2}"
VERL_DIR="${VERL_DIR:-agentic_rl_data/data/verl_v2}"
N_PROCEDURAL="${N_PROCEDURAL:-4000}"
N_INTERLEAVED="${N_INTERLEAVED:-1800}"
N_ABSTAIN="${N_ABSTAIN:-2400}"
N_VEHICLE_TRAVEL="${N_VEHICLE_TRAVEL:-1800}"
N_SOCIAL_OPS="${N_SOCIAL_OPS:-1800}"
N_PARALLEL="${N_PARALLEL:-600}"
N_PROCEDURAL_ST="${N_PROCEDURAL_ST:-2000}"
TOTAL_TRAIN="${TOTAL_TRAIN:-12000}"
SEED="${SEED:-20251106}"
SMOKE="${SMOKE:-0}"
if [[ "$SMOKE" == "1" ]]; then
  N_PROCEDURAL=200
  N_INTERLEAVED=200
  N_ABSTAIN=200
  N_VEHICLE_TRAVEL=200
  N_SOCIAL_OPS=200
  N_PARALLEL=100
  N_PROCEDURAL_ST=200
  TOTAL_TRAIN=120
fi

echo "[build_v2] 1/4 generate + verify $N_PROCEDURAL procedural stateful tasks"
python -m agentic_rl_data.scripts.build_raw --source procedural_mt \
  --cfg "{\"n\": $N_PROCEDURAL, \"seed\": $SEED, \"fault_prob\": 0.3}" --out-dir "$RAW_DIR"

echo "[build_v2] 1/4 generate + verify $N_INTERLEAVED interleaved multi-user tasks"
python -m agentic_rl_data.scripts.build_raw --source interleaved_mt \
  --cfg "{\"n\": $N_INTERLEAVED, \"seed\": $SEED}" --out-dir "$RAW_DIR"

echo "[build_v2] 1/4 generate + verify $N_ABSTAIN stateful abstention tasks"
python -m agentic_rl_data.scripts.build_raw --source stateful_abstain \
  --cfg "{\"n\": $N_ABSTAIN, \"seed\": $SEED}" --out-dir "$RAW_DIR"

echo "[build_v2] 1/4 generate + verify $N_VEHICLE_TRAVEL vehicle/travel tasks"
python -m agentic_rl_data.scripts.build_raw --source vehicle_travel_mt \
  --cfg "{\"n\": $N_VEHICLE_TRAVEL, \"seed\": $SEED}" --out-dir "$RAW_DIR"

echo "[build_v2] 1/4 generate + verify $N_SOCIAL_OPS social/operations tasks"
python -m agentic_rl_data.scripts.build_raw --source social_ops_mt \
  --cfg "{\"n\": $N_SOCIAL_OPS, \"seed\": $SEED}" --out-dir "$RAW_DIR"

echo "[build_v2] 1/4 generate + verify $N_PARALLEL parallel multi-turn tasks"
python -m agentic_rl_data.scripts.build_raw --source parallel_mt \
  --cfg "{\"n\": $N_PARALLEL, \"seed\": $SEED}" --out-dir "$RAW_DIR"

echo "[build_v2] 1/4 generate + verify $N_PROCEDURAL_ST procedural single-turn tasks"
python -m agentic_rl_data.scripts.build_raw --source procedural_st \
  --cfg "{\"n\": $N_PROCEDURAL_ST, \"seed\": $SEED}" --out-dir "$RAW_DIR"

echo "[build_v2] 2/4 load raw sources from $RAW_CFG"
if [[ -f "$RAW_CFG" ]]; then
  SOURCES=$(python -c "import json,sys; print(' '.join('--source '+k for k in json.load(open(sys.argv[1])) if not k.startswith('_')))" "$RAW_CFG")
  # shellcheck disable=SC2086
  python -m agentic_rl_data.scripts.build_raw $SOURCES --cfg-json "$RAW_CFG" --out-dir "$RAW_DIR"
else
  echo "[build_v2] WARNING: $RAW_CFG not found; only generated sources are available." >&2
  echo "           copy configs/raw_sources_v2.example.json and point it at your local dumps." >&2
fi

echo "[build_v2] 3/4 mixture ($MIX_CONFIG, total_train=$TOTAL_TRAIN)"
python -m agentic_rl_data.scripts.build_mixture --config "$MIX_CONFIG" \
  --raw-dir "$RAW_DIR" --out-dir "$MIX_DIR" --total-train "$TOTAL_TRAIN" --emit-curriculum

echo "[build_v2] 4/4 export verl parquet -> $VERL_DIR"
python -m agentic_rl_data.scripts.export_verl_parquet --mixture-dir "$MIX_DIR" --out-dir "$VERL_DIR"
if [[ -d "$MIX_DIR/curriculum" ]]; then
  python -m agentic_rl_data.scripts.export_verl_parquet --mixture-dir "$MIX_DIR/curriculum" \
    --out-dir "$VERL_DIR/curriculum" \
    --splits "stage8:train_max_turns8.jsonl,stage16:train_max_turns16.jsonl,stage24:train_max_turns24.jsonl"
fi
echo "[build_v2] done: $VERL_DIR/{train,heldout}.parquet"
