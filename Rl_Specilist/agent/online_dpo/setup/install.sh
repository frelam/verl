#!/usr/bin/env bash
# =============================================================================
# Online DPO Multi-Agent 一键安装脚本
# =============================================================================
#
# 在全新环境上一键安装所有依赖：
#   - verl + Python 环境
#   - hermes CLI（本地 agent）
#   - claude CLI（Claude Code npm 包）
#   - 训练模型 (Qwen3-4B)
#   - 数据集 + prompt 提取
#
# Usage:
#   # 完整安装
#   bash install.sh
#
#   # 跳过已有组件
#   bash install.sh --skip-hermes --skip-claude --skip-model
#
#   # 仅安装特定组件
#   bash install.sh --only hermes,claude
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONLINE_DPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VERL_DIR="$(cd "$ONLINE_DPO_DIR/../../.." && pwd)"

# ---- Config ----
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONDA_ENV="${CONDA_ENV:-verl-dpo}"
PIP_EXTRA="${PIP_EXTRA:-}"
DATA_DIR="${DATA_DIR:-$HOME/data/online_dpo}"
MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3-4B}"
HF_TOKEN="${HF_TOKEN:-}"
DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"

# ---- CLI flags ----
SKIP_HERMES=0
SKIP_CLAUDE=0
SKIP_MODEL=0
SKIP_DATA=0
ONLY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-hermes) SKIP_HERMES=1; shift ;;
        --skip-claude) SKIP_CLAUDE=1; shift ;;
        --skip-model)  SKIP_MODEL=1; shift ;;
        --skip-data)   SKIP_DATA=1; shift ;;
        --only)
            ONLY="$2"
            SKIP_HERMES=1; SKIP_CLAUDE=1; SKIP_MODEL=1; SKIP_DATA=1
            for c in ${ONLY//,/ }; do
                case "$c" in
                    hermes) SKIP_HERMES=0 ;;
                    claude) SKIP_CLAUDE=0 ;;
                    model)  SKIP_MODEL=0 ;;
                    data)   SKIP_DATA=0 ;;
                esac
            done
            shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo " Online DPO Multi-Agent 安装"
echo "========================================"
echo " VERL_DIR:    $VERL_DIR"
echo " DATA_DIR:    $DATA_DIR"
echo " MODEL_PATH:  $MODEL_PATH"
echo " PYTHON:      $PYTHON_BIN"
echo "========================================"

# ---- Step 1: Python 环境 ----
echo ""
echo "[1/6] 安装 Python 环境..."

# Create venv if needed
if [ -n "${CONDA_PREFIX:-}" ] || python3 -c "import verl" 2>/dev/null; then
    echo "  Python 环境已就绪（verl 可导入）"
else
    $PYTHON_BIN -m venv "$HOME/.venv/verl-dpo" || true
    source "$HOME/.venv/verl-dpo/bin/activate" 2>/dev/null || true
fi

# Install verl + deps
$PYTHON_BIN -m pip install -e "$VERL_DIR" $PIP_EXTRA -q 2>&1 | tail -3
$PYTHON_BIN -m pip install aiohttp hydra-core pyarrow pandas -q 2>&1 | tail -3

echo "  ✅ Python 环境就绪"

# ---- Step 2: Hermes CLI ----
echo ""
echo "[2/6] 安装 Hermes CLI..."

if [ "$SKIP_HERMES" -eq 1 ]; then
    echo "  ⏭ 跳过"
elif command -v hermes &>/dev/null; then
    echo "  ✅ Hermes 已安装: $(hermes --version 2>&1 | head -1)"
else
    # Hermes is installed via pip from PyPI or GitHub
    echo "  安装 hermes-agent..."
    $PYTHON_BIN -m pip install hermes-agent -q 2>&1 | tail -5 || {
        echo "  pip install 失败，尝试从 GitHub 安装..."
        $PYTHON_BIN -m pip install git+https://github.com/nousresearch/hermes-agent.git -q 2>&1 | tail -5
    }
    # Verify
    if command -v hermes &>/dev/null; then
        echo "  ✅ Hermes 安装成功"
    else
        echo "  ⚠️ Hermes 安装后未找到 hermes 命令，请检查 PATH"
    fi
fi

# ---- Step 3: Claude Code CLI ----
echo ""
echo "[3/6] 安装 Claude Code CLI..."

if [ "$SKIP_CLAUDE" -eq 1 ]; then
    echo "  ⏭ 跳过"
elif command -v claude &>/dev/null; then
    echo "  ✅ Claude Code 已安装: $(claude --version 2>&1)"
else
    echo "  通过 npm 安装 @anthropic-ai/claude-code..."
    if ! command -v npm &>/dev/null; then
        echo "  npm 未安装，正在安装 Node.js..."
        curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - 2>/dev/null || true
        sudo apt-get install -y nodejs 2>/dev/null || {
            # 备选：nvm
            curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash
            export NVM_DIR="$HOME/.nvm"
            [ -s "$NVM_DIR/nvm.sh" ] && source "$NVM_DIR/nvm.sh"
            nvm install 22
        }
    fi
    npm install -g @anthropic-ai/claude-code 2>&1 | tail -5
    echo "  ✅ Claude Code 安装成功"
fi

# ---- Step 4: 下载模型 ----
echo ""
echo "[4/6] 下载训练模型 Qwen3-4B..."

if [ "$SKIP_MODEL" -eq 1 ]; then
    echo "  ⏭ 跳过"
elif [ -d "$MODEL_PATH" ] && ls "$MODEL_PATH"/*.safetensors &>/dev/null; then
    echo "  ✅ 模型已存在: $MODEL_PATH"
else
    echo "  从 HuggingFace 下载 Qwen/Qwen3-4B ..."
    mkdir -p "$MODEL_PATH"
    $PYTHON_BIN -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-4B', local_dir='$MODEL_PATH',
    allow_patterns=['*.json','*.safetensors','*.txt','tokenizer*','merges*'])
" 2>&1 | tail -5
    echo "  ✅ 模型下载完成"
fi

# ---- Step 5: 下载数据集 ----
echo ""
echo "[5/6] 下载数据集..."

if [ "$SKIP_DATA" -eq 1 ]; then
    echo "  ⏭ 跳过"
else
    bash "$SCRIPT_DIR/download_data.sh"
fi

# ---- Step 6: 验证 ----
echo ""
echo "[6/6] 验证安装..."

PASS=0
FAIL=0

check() {
    local name="$1"; shift
    if "$@" &>/dev/null; then
        echo "  ✅ $name"
        PASS=$((PASS + 1))
    else
        echo "  ❌ $name — 请检查安装"
        FAIL=$((FAIL + 1))
    fi
}

check "python3 + verl"   $PYTHON_BIN -c "import verl" 2>/dev/null
check "pandas"           $PYTHON_BIN -c "import pandas" 2>/dev/null
check "aiohttp"          $PYTHON_BIN -c "import aiohttp" 2>/dev/null
check "hermes CLI"       hermes --version 2>/dev/null
check "claude CLI"       claude --version 2>/dev/null
check "Qwen3-4B model"   test -f "$MODEL_PATH/config.json" 2>/dev/null
check "prompt data"      ls "$DATA_DIR"/prompts/*.parquet 2>/dev/null

echo ""
echo "========================================"
echo " 安装完成: $PASS 通过, $FAIL 失败"
echo "========================================"
echo ""
echo "下一步："
echo "  1. 配置 API keys:"
echo "     export DEEPSEEK_API_KEY=sk-xxx      # Judge 打分"
echo ""
echo "  2. 启动训练:"
echo "     bash $ONLINE_DPO_DIR/run_multi_agent_dpo.sh toolmind 8 $DATA_DIR/ckpt"
