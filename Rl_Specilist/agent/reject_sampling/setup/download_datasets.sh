#!/usr/bin/env bash
# 一键下载 reject sampling 所需的 4 个数据集 + Qwen3-4B 模型 + 预提取 prompt
#
# Usage:
#   bash download_datasets.sh
#   MAX_SAMPLES=500 bash download_datasets.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

MAX_SAMPLES="${MAX_SAMPLES:-500}"
DATASETS_TO_DOWNLOAD="${DATASETS_TO_DOWNLOAD:-toolmind terminaltraj open_swe_traces swe_zero}"

mkdir -p "$DATA_DIR/raw" "$DATA_DIR/prompts" "$HOME/models"

echo "========================================"
echo " 下载数据集 + 模型"
echo "========================================"
echo " DATA_DIR:      $DATA_DIR"
echo " MODEL_PATH:    $MODEL_PATH"
echo " MAX_SAMPLES:   $MAX_SAMPLES"
echo " Datasets:      $DATASETS_TO_DOWNLOAD"
echo "========================================"

# ========== 1. 下载 4 个数据集 ==========
echo ""
echo "[1/3] 下载 HuggingFace 数据集..."

python3 - <<'PYEOF'
import os
import sys
from huggingface_hub import snapshot_download

datasets = {
    "Nanbeige/ToolMind": "toolmind",
    "nvidia/Open-SWE-Traces": "open_swe_traces",
    "nvidia/SWE-Zero-openhands-trajectories": "swe_zero",
    "m-a-p/TerminalTraj": "terminaltraj",
}
data_dir = os.path.expanduser(os.environ.get("DATA_DIR", "~/data/reject_sampling") + "/raw")
datasets_to_download = os.environ.get("DATASETS_TO_DOWNLOAD", " ".join(datasets.values())).split()

for repo_id, name in datasets.items():
    if name not in datasets_to_download:
        continue
    local_dir = os.path.join(data_dir, name)
    if os.path.exists(local_dir) and os.listdir(local_dir):
        print(f"  [skip] {name} already at {local_dir}")
        continue
    print(f"  Downloading {repo_id} -> {local_dir} ...")
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
        )
        print(f"  [ok] {name}")
    except Exception as e:
        print(f"  [FAIL] {name}: {e}", file=sys.stderr)

print("Done.")
PYEOF

# ========== 2. 下载 Qwen3-4B 模型 ==========
echo ""
echo "[2/3] 下载 Qwen3-4B 模型..."

python3 - <<'PYEOF'
import os
from huggingface_hub import snapshot_download

model_path = os.path.expanduser(os.environ.get("MODEL_PATH", "~/models/Qwen3-4B"))
if os.path.exists(model_path) and any(f.endswith(".safetensors") for f in os.listdir(model_path)):
    print(f"  [skip] model already at {model_path}")
else:
    print(f"  Downloading Qwen/Qwen3-4B -> {model_path} ...")
    snapshot_download(
        repo_id="Qwen/Qwen3-4B",
        local_dir=model_path,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*", "merges*"],
    )
    print(f"  [ok] model downloaded")
PYEOF

# ========== 3. 预提取 prompt ==========
echo ""
echo "[3/3] 提取 prompt (max_samples=$MAX_SAMPLES)..."

python3 -m Rl_Specilist.agent.reject_sampling.data_preprocess.extract_prompts \
    --raw_dir "$DATA_DIR/raw" \
    --output_dir "$DATA_DIR/prompts" \
    --max_samples "$MAX_SAMPLES" \
    --datasets $DATASETS_TO_DOWNLOAD

echo ""
echo "========================================"
echo "✅ 数据集下载 + prompt 提取完成"
echo "========================================"
echo " Prompts: $DATA_DIR/prompts/"
echo " Model:   $MODEL_PATH"
echo ""
echo "下一步：bash $REJECT_SFT_DIR/run_reject_sampling.sh toolmind 8 ~/data/reject_sampling/ckpt"
