#!/usr/bin/env bash
# =============================================================================
# Online DPO 数据集下载 + 预处理
# =============================================================================
#
# 下载中低难度 agent 数据集，提取 prompt，产出 parquet 文件。
#
# 数据集：
#   toolmind        L1 通用工具调用（calculator, search, code_runner）
#   terminaltraj    L3 Docker 终端 bash 命令
#   swe_zero        L2 代码修复轨迹
#
# Usage:
#   bash download_data.sh
#   DATA_DIR=~/my_data bash download_data.sh
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONLINE_DPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- Config ----
DATA_DIR="${DATA_DIR:-$HOME/data/online_dpo}"
HF_TOKEN="${HF_TOKEN:-}"
MAX_SAMPLES="${MAX_SAMPLES:-500}"
INCLUDE_HARD="${INCLUDE_HARD:-0}"

echo "========================================"
echo " 数据集下载 + 预处理"
echo "========================================"
echo " DATA_DIR:     $DATA_DIR"
echo " MAX_SAMPLES:  $MAX_SAMPLES"
echo "========================================"

mkdir -p "$DATA_DIR/raw" "$DATA_DIR/prompts"

# ---- 数据集定义 ----
# repo_id : short_name : difficulty
DATASETS=(
    "Nanbeige/ToolMind:toolmind:L1"
    "m-a-p/TerminalTraj:terminaltraj:L3"
    "nvidia/SWE-Zero-openhands-trajectories:swe_zero:L2"
)
if [ "$INCLUDE_HARD" -eq 1 ]; then
    DATASETS+=("nvidia/Open-SWE-Traces:open_swe_traces:L4")
fi

# ---- 1. 下载原始数据 ----
echo ""
echo "[1/4] 下载 HuggingFace 数据集..."

$PYTHON_BIN -c "
import os, sys
from huggingface_hub import snapshot_download

data_dir = os.path.expanduser('$DATA_DIR') + '/raw'
datasets = dict(item.split(':')[:2] for item in '''${DATASETS[*]}'''.split())

for repo_id, name in datasets.items():
    local_dir = os.path.join(data_dir, name)
    if os.path.exists(local_dir) and os.listdir(local_dir):
        print(f'  [skip] {name}')
        continue
    print(f'  Downloading {repo_id} -> {local_dir} ...')
    try:
        snapshot_download(repo_id=repo_id, repo_type='dataset', local_dir=local_dir)
        print(f'  [ok] {name}')
    except Exception as e:
        print(f'  [FAIL] {name}: {e}', file=sys.stderr)
print('Done.')
" ${HF_TOKEN:+\--hf_token "$HF_TOKEN"}

# ---- 2. 提取 prompt ----
echo ""
echo "[2/4] 提取 prompt (max_samples=$MAX_SAMPLES)..."

$PYTHON_BIN -m Rl_Specilist.agent.reject_sampling.data_preprocess.extract_prompts \
    --raw_dir "$DATA_DIR/raw" \
    --output_dir "$DATA_DIR/prompts" \
    --max_samples "$MAX_SAMPLES" \
    --datasets $(echo "${DATASETS[@]}" | tr ' ' '\n' | cut -d: -f2 | tr '\n' ' ')

# ---- 3. 完成 ----
echo ""
echo "[3/3] 验证数据..."

$PYTHON_BIN -c "
import pandas as pd, os, glob

data_dir = '$DATA_DIR/prompts'
files = sorted(glob.glob(f'{data_dir}/*.parquet'))
if not files:
    print('  ⚠️ 未找到 parquet 文件')
    exit(1)

for f in files:
    df = pd.read_parquet(f)
    name = os.path.basename(f)
    print(f'  {name}: {len(df)} samples')
print(f'\\n✅ 共 {len(files)} 个 parquet 文件')
"

echo ""
echo "========================================"
echo " ✅ 数据准备完成"
echo "========================================"
echo ""
echo "可用数据集:"
ls -1 "$DATA_DIR/prompts/"*.parquet 2>/dev/null | while read -r f; do
    count=$($PYTHON_BIN -c "import pandas as pd; print(len(pd.read_parquet('$f')))" 2>/dev/null || echo "?")
    echo "  $(basename "$f")  ($count samples)"
done
echo ""
echo "启动训练:"
echo "  bash $ONLINE_DPO_DIR/run_multi_agent_dpo.sh toolmind 8 $DATA_DIR/ckpt"
