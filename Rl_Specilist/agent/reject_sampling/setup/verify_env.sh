#!/usr/bin/env bash
# 环境自检脚本 — 检查 reject sampling 所需的所有组件是否就绪
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh" 2>/dev/null || echo "  (env.sh not sourced, using defaults)"

WARN_COUNT=0
warn() { echo "  ⚠️  $1"; WARN_COUNT=$((WARN_COUNT + 1)); }
ok()   { echo "  ✅  $1"; }

echo "=== Reject Sampling 环境自检 ==="
echo ""

# 1. Python + 关键包
echo "[1/5] Python 依赖:"
python3 -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA available={torch.cuda.is_available()}')" 2>/dev/null && ok "torch" || warn "torch not installed"
python3 -c "import verl; print(f'  verl ok')" 2>/dev/null && ok "verl" || warn "verl not installed (run: pip install -e .)"
python3 -c "import vllm; print(f'  vLLM {vllm.__version__}')" 2>/dev/null && ok "vllm" || warn "vllm not installed"
python3 -c "import openai; print(f'  openai {openai.__version__}')" 2>/dev/null && ok "openai" || warn "openai not installed"
python3 -c "import docker; print(f'  docker SDK ok')" 2>/dev/null && ok "docker SDK" || warn "docker SDK not installed"
python3 -c "import datasets; print(f'  datasets ok')" 2>/dev/null && ok "datasets" || warn "datasets not installed"
python3 -c "import huggingface_hub; print(f'  huggingface_hub ok')" 2>/dev/null && ok "huggingface_hub" || warn "huggingface_hub not installed"

# 2. GPU
echo ""
echo "[2/5] GPU:"
if command -v nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo "  GPUs: $GPU_COUNT × $GPU_NAME (${GPU_MEM}MiB each)"
    ok "GPU"
else
    warn "nvidia-smi not found"
fi

# 3. Docker + 镜像
echo ""
echo "[3/5] Docker:"
if docker info >/dev/null 2>&1; then
    ok "docker daemon"
else
    warn "docker daemon not running (sudo systemctl start docker)"
fi

if docker image inspect reject-sft-terminal:latest >/dev/null 2>&1; then
    ok "reject-sft-terminal:latest"
else
    warn "reject-sft-terminal:latest not built (for TerminalTraj)"
fi

if docker image inspect reject-sft-swe:latest >/dev/null 2>&1; then
    ok "reject-sft-swe:latest"
else
    warn "reject-sft-swe:latest not built (for SWE-bench, optional)"
fi

# 4. API 连通性
echo ""
echo "[4/5] DeepSeek API:"

if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    warn "DEEPSEEK_API_KEY not set (export DEEPSEEK_API_KEY=sk-xxxxx)"
else
    python3 - <<'PYEOF' 2>/dev/null && ok "DeepSeek API" || warn "DeepSeek API connection failed"
import os
from openai import OpenAI
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)
r = client.chat.completions.create(
    model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
    messages=[{"role": "user", "content": "ping"}],
    max_tokens=5,
)
print(f"  response: {r.choices[0].message.content!r}")
PYEOF
fi

# 5. 数据目录
echo ""
echo "[5/5] 数据集:"
for d in toolmind terminaltraj open_swe_traces swe_zero; do
    if [ -d "$DATA_DIR/raw/$d" ] && [ -n "$(ls -A "$DATA_DIR/raw/$d" 2>/dev/null)" ]; then
        ok "dataset $d"
    else
        warn "dataset $d not downloaded (run: download_datasets.sh)"
    fi
done

# Prompt parquet
if [ -d "$DATA_DIR/prompts" ] && ls "$DATA_DIR/prompts/"*.parquet >/dev/null 2>&1; then
    ok "prompts extracted ($DATA_DIR/prompts/)"
else
    warn "prompts not extracted (run: extract_prompts.py)"
fi

# 模型
if [ -n "$(ls "$MODEL_PATH"/*.safetensors 2>/dev/null)" ]; then
    ok "model at $MODEL_PATH"
else
    warn "model not downloaded at $MODEL_PATH"
fi

echo ""
echo "=== 自检完成: $WARN_COUNT 个警告 ==="
if [ "$WARN_COUNT" -gt 0 ]; then
    exit 1
fi
