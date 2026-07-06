#!/usr/bin/env bash
# 一键安装 reject sampling SFT 所需的完整环境
# 幂等：重复执行不报错，已存在的组件会跳过
#
# Usage:
#   bash install_env.sh
#   ENV_NAME=myenv VERL_DIR=/path/to/verl CUDA_VERSION=12.1 bash install_env.sh
set -euo pipefail

# ========== 0. 前置条件检查 ==========
echo "[0/8] 检查前置条件..."

if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. NVIDIA driver is required." >&2
    exit 1
fi
nvidia-smi >/dev/null 2>&1 || { echo "ERROR: nvidia-smi failed." >&2; exit 1; }

if ! command -v git &>/dev/null; then
    echo "ERROR: git not found." >&2
    exit 1
fi

echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# ========== 1. 系统依赖 ==========
echo "[1/8] 安装系统依赖..."

if ! command -v docker &>/dev/null; then
    echo "  Installing docker..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq docker.io docker-compose-plugin >/dev/null
fi

# 启动 docker（兼容 systemd 和非 systemd）
sudo systemctl start docker 2>/dev/null || sudo service docker start 2>/dev/null || true

# 加入 docker 组（免 sudo）
if ! groups | grep -qw docker; then
    sudo usermod -aG docker "$USER" 2>/dev/null || true
    echo "  NOTE: 用户已加入 docker 组，需重新登录或 'newgrp docker' 生效"
fi

# ========== 2. conda 环境 ==========
echo "[2/8] 配置 conda 环境..."

if ! command -v conda &>/dev/null; then
    echo "  Installing miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
    conda init bash >/dev/null 2>&1 || true
fi

# 激活 conda
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
# shellcheck disable=SC1090
source "$CONDA_BASE/etc/profile.d/conda.sh"

ENV_NAME="${ENV_NAME:-reject_sft}"
if conda env list | grep -qw "$ENV_NAME"; then
    echo "  conda env '$ENV_NAME' already exists"
else
    echo "  Creating conda env '$ENV_NAME'..."
    conda create -y -n "$ENV_NAME" python=3.10 >/dev/null
fi
conda activate "$ENV_NAME"

# ========== 3. PyTorch ==========
echo "[3/8] 安装 PyTorch..."

CUDA_VERSION="${CUDA_VERSION:-12.1}"
# 去掉小数点，如 12.1 -> 121
CUDA_TAG="cu${CUDA_VERSION//./}"

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  Installing torch for CUDA ${CUDA_VERSION}..."
    pip install -q torch==2.4.0 --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
fi
echo "  torch: $(python -c 'import torch; print(torch.__version__)')"

# ========== 4. 克隆 verl ==========
echo "[4/8] 准备 verl 仓库..."

VERL_DIR="${VERL_DIR:-$HOME/workspace/verl}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 如果脚本本身就在 verl 仓库里，自动推断 VERL_DIR
if [ -f "$SCRIPT_DIR/../../../pyproject.toml" ]; then
    VERL_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi

mkdir -p "$(dirname "$VERL_DIR")"
if [ ! -d "$VERL_DIR/.git" ]; then
    echo "  Cloning verl to $VERL_DIR..."
    git clone --depth=1 https://github.com/volcengine/verl.git "$VERL_DIR"
    # 把当前 reject_sampling 目录拷过去（因为是从 fork 拉的，可能没有 Rl_Specilist）
    cp -r "$SCRIPT_DIR" "$VERL_DIR/Rl_Specilist/agent/reject_sampling" 2>/dev/null || true
else
    echo "  verl already exists at $VERL_DIR"
fi
cd "$VERL_DIR"

# ========== 5. 安装 verl + Python 依赖 ==========
echo "[5/8] 安装 verl + Python 依赖..."

if ! python -c "import verl" 2>/dev/null; then
    echo "  Installing verl (editable)..."
    pip install -e . -q
fi

# 核心依赖（幂等：pip install 会跳过已安装的）
pip install -q -r requirements.txt 2>/dev/null || true

# vllm
if ! python -c "import vllm" 2>/dev/null; then
    echo "  Installing vLLM..."
    pip install -q "vllm>=0.6.0"
fi
echo "  vllm: $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 'not installed')"

# flash-attn（编译耗时较长，单独处理）
if ! python -c "import flash_attn" 2>/dev/null; then
    echo "  Installing flash-attn (this may take ~10 min)..."
    pip install flash-attn --no-build-isolation -q 2>/dev/null || echo "  WARN: flash-attn install failed, skipping"
fi

# reject sampling 额外依赖
pip install -q "openai>=1.0.0"    # DeepSeek API（兼容 OpenAI SDK）
pip install -q docker             # Python Docker SDK（TerminalTraj）
pip install -q aiohttp            # 异步 HTTP
pip install -q huggingface_hub    # 数据集下载

# ========== 6. 构建 agent 环境 Docker 镜像 ==========
echo "[6/8] 构建 agent Docker 镜像..."

SETUP_DIR="$VERL_DIR/Rl_Specilist/agent/reject_sampling/setup"
DOCKER_DIR="$SETUP_DIR/docker"

# TerminalTraj 终端沙箱镜像
if ! docker image inspect reject-sft-terminal:latest >/dev/null 2>&1; then
    if [ -f "$DOCKER_DIR/Dockerfile.terminal" ]; then
        echo "  Building reject-sft-terminal:latest..."
        docker build -t reject-sft-terminal:latest -f "$DOCKER_DIR/Dockerfile.terminal" "$DOCKER_DIR" 2>&1 | tail -3
    else
        echo "  WARN: Dockerfile.terminal not found, skipping"
    fi
else
    echo "  reject-sft-terminal:latest already exists"
fi

# SWE-bench 评测镜像（可选，失败不中断）
if ! docker image inspect reject-sft-swe:latest >/dev/null 2>&1; then
    if [ -f "$DOCKER_DIR/Dockerfile.swe_bench" ]; then
        echo "  Building reject-sft-swe:latest (optional, may fail)..."
        docker build -t reject-sft-swe:latest -f "$DOCKER_DIR/Dockerfile.swe_bench" "$DOCKER_DIR" 2>&1 | tail -3 \
            || echo "  WARN: SWE image build failed, you can build it later"
    else
        echo "  WARN: Dockerfile.swe_bench not found, skipping"
    fi
else
    echo "  reject-sft-swe:latest already exists"
fi

# ========== 7. 配置环境变量 ==========
echo "[7/8] 配置环境变量..."

ENV_FILE="$SETUP_DIR/env.sh"
# 用实际路径替换占位符
sed -i "s|__VERL_DIR__|$VERL_DIR|g" "$ENV_FILE" 2>/dev/null || true

# 写入 ~/.bashrc（幂等：先检查是否已存在）
BASHRC_LINE="source \"$ENV_FILE\""
if ! grep -qF "$BASHRC_LINE" "$HOME/.bashrc" 2>/dev/null; then
    echo "" >> "$HOME/.bashrc"
    echo "# verl reject sampling env" >> "$HOME/.bashrc"
    echo "$BASHRC_LINE" >> "$HOME/.bashrc"
fi

echo "  env.sh: $ENV_FILE"
echo "  (已添加到 ~/.bashrc)"

# ========== 8. 自检 ==========
echo "[8/8] 运行自检..."
bash "$SETUP_DIR/verify_env.sh" || echo "  (自检有警告，请按提示修复)"

echo ""
echo "========================================"
echo "✅ 环境安装完成！"
echo "========================================"
echo ""
echo "下一步："
echo "  1. 设置 API key:"
echo "     export DEEPSEEK_API_KEY=sk-xxxxx"
echo "     export HF_TOKEN=hf_xxxxx"
echo ""
echo "  2. 下载数据集 + 模型:"
echo "     bash $SETUP_DIR/download_datasets.sh"
echo ""
echo "  3. 跑 reject sampling:"
echo "     bash $VERL_DIR/Rl_Specilist/agent/reject_sampling/run_reject_sampling.sh toolmind 8 ~/data/reject_sampling/ckpt"
