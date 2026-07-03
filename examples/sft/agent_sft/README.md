# Agent SFT Training on verl

使用 5 个 HuggingFace 智能体轨迹数据集，在 verl 框架上对 Qwen3-4B 进行 SFT（监督微调）训练。

## 数据集概览

| 数据集 | HuggingFace Repo | 规模 | 特点 |
|--------|-----------------|------|------|
| ToolMind | `Nanbeige/ToolMind` | 369k | 大规模工具调用 + 推理增强 |
| Open-SWE-Traces | `nvidia/Open-SWE-Traces` | 207k | SWE 智能体轨迹 (OpenHands/SWE-agent) |
| SWE-Zero | `nvidia/SWE-Zero-openhands-trajectories` | 318k | 无执行 SWE 智能体轨迹 |
| TerminalTraj | `m-a-p/TerminalTraj` | 20k+ | 终端智能体轨迹 (Docker 环境) |
| OpenResearcher | `OpenResearcher/OpenResearcher-Dataset` | 96k | 长程深度研究轨迹 |

## 文件结构

```
examples/
├── data_preprocess/
│   └── prepare_agent_sft_data.py    # 数据下载与处理脚本
└── sft/agent_sft/
    ├── run_qwen3_4b_sft.sh          # 训练启动脚本
    └── README.md                     # 本文档
```

## 1. 环境准备

确保已安装 verl 及其依赖：

```bash
cd /home/charles/workspace/verl
pip install -e .
pip install datasets pandas pyarrow
```

## 2. 数据下载与处理

### 基本用法

```bash
# 处理所有数据集（使用完整数据）
python examples/data_preprocess/prepare_agent_sft_data.py \
    --output_dir ~/data/agent_sft
```

### 动态配置每个数据集的样本数

核心功能 —— 通过命令行参数独立控制每个数据集的采样数量（`-1` 表示使用完整数据集）：

```bash
python examples/data_preprocess/prepare_agent_sft_data.py \
    --output_dir ~/data/agent_sft \
    --toolmind_n 10000 \
    --open_swe_traces_n 20000 \
    --swe_zero_n 20000 \
    --terminaltraj_n 5000 \
    --openresearcher_n 10000 \
    --val_ratio 0.02
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output_dir` | `~/data/agent_sft` | 输出目录 |
| `--datasets` | 全部 | 选择处理的数据集（可选: toolmind, open_swe_traces, swe_zero, terminaltraj, openresearcher） |
| `--toolmind_n` | -1 | ToolMind 采样数（-1 = 全部） |
| `--open_swe_traces_n` | -1 | Open-SWE-Traces 采样数 |
| `--swe_zero_n` | -1 | SWE-Zero 采样数 |
| `--terminaltraj_n` | -1 | TerminalTraj 采样数 |
| `--openresearcher_n` | -1 | OpenResearcher 采样数 |
| `--val_ratio` | 0.02 | 验证集比例 |
| `--no_merge` | False | 不合并各数据集为统一的 train/val.parquet |

### 输出文件

处理完成后，`output_dir` 下会生成：

```
~/data/agent_sft/
├── toolmind_train.parquet          # 各数据集单独的 train 文件
├── toolmind_val.parquet
├── open_swe_traces_train.parquet
├── open_swe_traces_val.parquet
├── swe_zero_train.parquet
├── swe_zero_val.parquet
├── terminaltraj_train.parquet
├── terminaltraj_val.parquet
├── openresearcher_train.parquet
├── openresearcher_val.parquet
├── train.parquet                   # 合并后的训练集（打乱顺序）
└── val.parquet                     # 合并后的验证集
```

### 只处理部分数据集

```bash
python examples/data_preprocess/prepare_agent_sft_data.py \
    --output_dir ~/data/agent_sft \
    --datasets toolmind terminaltraj \
    --toolmind_n 5000 \
    --terminaltraj_n 3000
```

## 3. 数据格式

所有数据集统一转换为 verl 的 `MultiTurnSFTDataset` 所需格式：

- **`messages`** 列：`list[dict]`，每个 dict 包含 `role`（system/user/assistant/tool）和 `content` 字段
- **`tools`** 列（可选）：`list[dict]`，工具定义列表

该格式直接兼容模型 `chat_template.jinja`，训练时由 `MultiTurnSFTDataset` 自动应用 chat template。

### 各数据集转换逻辑

| 数据集 | 原始字段 | 转换 |
|--------|---------|------|
| ToolMind | `conversations`, `tools` | `conversations` → `messages`，保留 `tool_calls` |
| Open-SWE-Traces | `trajectory`, `tools` (list[str]) | `trajectory` → `messages`，解析 `tools` JSON 字符串 |
| SWE-Zero | `trajectory` | `trajectory` → `messages` |
| TerminalTraj | `messages` | 已兼容，仅做规范化 |
| OpenResearcher | `messages` (GPT-OSS 格式) | 将 content blocks 展开为标准 {role, content} 格式 |

## 4. 启动训练

### 基本用法

```bash
# 单卡训练
bash examples/sft/agent_sft/run_qwen3_4b_sft.sh 1 /tmp/qwen3-4b-agent-sft

# 8 卡训练 + Ulysses 序列并行 SP=4
SP_SIZE=4 bash examples/sft/agent_sft/run_qwen3_4b_sft.sh 8 /tmp/qwen3-4b-agent-sft
```

### 可配置的环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `/home/charles/workspace/qwen3-4b-gdpo-step850` | 模型路径 |
| `DATA_DIR` | `~/data/agent_sft` | 数据目录 |
| `SP_SIZE` | 1 | Ulysses 序列并行大小 |
| `MICRO_BATCH_SIZE_PER_GPU` | 2 | 每卡微批次 |
| `TRAIN_BATCH_SIZE` | 64 | 全局训练批次 |
| `LR` | 1e-5 | 学习率 |
| `TOTAL_EPOCHS` | 3 | 训练轮数 |
| `MAX_LENGTH` | 32768 | 最大序列长度 |
| `MAX_TOKEN_LEN_PER_GPU` | 32768 | 每卡最大 token 数 |
| `USE_PEFT` | 0 | 是否使用 LoRA |

### 训练示例

```bash
# 8卡全参数微调
SP_SIZE=4 \
MICRO_BATCH_SIZE_PER_GPU=2 \
TRAIN_BATCH_SIZE=64 \
LR=1e-5 \
TOTAL_EPOCHS=3 \
MAX_LENGTH=32768 \
bash examples/sft/agent_sft/run_qwen3_4b_sft.sh 8 /tmp/qwen3-4b-agent-sft

# LoRA 微调（节省显存）
USE_PEFT=1 \
LORA_RANK=32 \
bash examples/sft/agent_sft/run_qwen3_4b_sft.sh 4 /tmp/qwen3-4b-agent-sft-lora

# 覆盖 hydra 配置
bash examples/sft/agent_sft/run_qwen3_4b_sft.sh 8 /tmp/sft \
    optim.lr=5e-5 \
    trainer.total_epochs=2 \
    data.max_length=16384
```

## 5. 完整流程示例

```bash
# Step 1: 数据处理（各取 1 万条，2% 验证集）
python examples/data_preprocess/prepare_agent_sft_data.py \
    --output_dir ~/data/agent_sft \
    --toolmind_n 10000 \
    --open_swe_traces_n 10000 \
    --swe_zero_n 10000 \
    --terminaltraj_n 5000 \
    --openresearcher_n 10000 \
    --val_ratio 0.02

# Step 2: 8卡训练
SP_SIZE=4 bash examples/sft/agent_sft/run_qwen3_4b_sft.sh 8 /tmp/qwen3-4b-agent-sft
```

## 6. 注意事项

1. **显存**：`MAX_LENGTH=32768` 需要较大显存。如 OOM，降低 `MAX_LENGTH`、`MICRO_BATCH_SIZE_PER_GPU` 或 `MAX_TOKEN_LEN_PER_GPU`。
2. **数据下载**：首次运行需要从 HuggingFace 下载全部数据集，需要网络连接和足够的磁盘空间（建议 100GB+）。
3. **OpenResearcher 格式**：该数据集使用 GPT-OSS 原生格式（带 channel 的 content blocks），脚本会自动转换为标准格式。
4. **`ignore_input_ids_mismatch=true`**：训练脚本默认启用此选项，因为 Qwen3 thinking 模型的 chat template 在逐 turn 处理时可能与整体处理存在细微差异。
5. **工具调用格式**：模型的 `chat_template.jinja` 使用 XML 风格的 `<tool_call>` 标签渲染 `tool_calls`，与 ToolMind、Open-SWE-Traces 等数据集的 `tool_calls` 字段兼容。
