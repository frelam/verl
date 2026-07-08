# Agent SFT 数据预处理

本目录包含 Agent SFT 训练的数据预处理脚本。

---

## 脚本列表

### 1. `agent_instruct.py` — AgentInstruct 数据集转换为 Parquet

将 AgentInstruct 格式的 JSONL 数据转换为 verl 框架可用的 Parquet 格式。

```bash
python Rl_Specilist/agent/SFT/data_preprocess/agent_instruct.py \
    --input_file /path/to/agent_instruct.jsonl \
    --output_dir ~/data/agent_sft
```

### 2. `mix_sharegpt_agentinstruct_sft.py` — 混合 ShareGPT + AgentInstruct

将 ShareGPT 对话数据与 AgentInstruct 工具调用数据混合，生成统一的 SFT 数据集。

```bash
python Rl_Specilist/agent/SFT/data_preprocess/mix_sharegpt_agentinstruct_sft.py \
    --sharegpt_path /path/to/sharegpt \
    --agent_instruct_path /path/to/agent_instruct \
    --output_dir ~/data/mixed_agent_sft \
    --mixing_ratio 0.5
```

### 3. `prepare_agent_sft_data.py` — Agent SFT 数据准备

完整的数据准备流程，包括：
- 从 Hugging Face Hub 下载数据集
- 格式转换（ChatML / ShareGPT / AgentInstruct → verl 格式）
- 训练/验证分割
- 数据质量检查

```bash
python Rl_Specilist/agent/SFT/data_preprocess/prepare_agent_sft_data.py \
    --dataset_path agentinstruct \
    --output_dir ~/data/agent_sft_prepared
```

### 4. `format_only_sft_dataset.py` — 仅格式训练的 SFT 数据

生成仅用于格式训练的数据集（不关心内容正确性，只关心格式遵循度）。位于上级目录 `Rl_Specilist/agent/SFT/`。

---

## 数据格式转换流程

```
原始数据源                  预处理                      verl 训练
──────────────────────────────────────────────────────────────────
AgentInstruct (JSONL)  ──→ agent_instruct.py ──→  train.parquet
ShareGPT (JSON)       ──→ mix_*.py           ──→  train.parquet
HuggingFace Hub       ──→ prepare_agent_sft   ──→  train.parquet

                                                  ┌→ format_only_sft_dataset.py
                                                  │   (仅格式训练)
                                                  │
train.parquet ────────────────────────────────────┼→ SFT 训练
                                                  │
                                                  └→ Reject Sampling 的 rollout
```

---

## 相关文档

- **训练总图**: `Rl_Specilist/agent/SFT/doc/dataset_and_train_way.md`
- **Reward 设计原则**: `Rl_Specilist/agent/SFT/doc/reward_priciple.md`
- **训练脚本**: `Rl_Specilist/agent/SFT/run_qwen3_0_6b_agent_instruct.sh`
- **examples 目录**: `examples/sft/agent_sft/` 中的 `run_qwen3_4b_sft.sh`
