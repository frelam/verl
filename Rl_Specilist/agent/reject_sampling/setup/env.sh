# verl + reject sampling 环境变量
# 本文件由 install_env.sh 自动生成，请勿手动编辑

export VERL_DIR=/home/charles/workspace/verl
export REJECT_SFT_DIR="$VERL_DIR/Rl_Specilist/agent/reject_sampling"
export DATA_DIR="${DATA_DIR:-$HOME/data/reject_sampling}"
export SFT_DATA_DIR="${SFT_DATA_DIR:-$HOME/data/reject_sampling_sft}"
export MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3-4B}"

# DeepSeek API（用于 reject sampling judge）
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"
export DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-https://api.deepseek.com}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-chat}"

# HuggingFace
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_TOKEN="${HF_TOKEN:-}"

# 轨迹落盘路径
export TRAJECTORY_FILE="${TRAJECTORY_FILE:-$DATA_DIR/collected_trajectories.jsonl}"
