# Math Specialist Training Recipes

数学专精模型的多阶段强化学习训练方案。从单数据集 GSM8K 开始，逐步扩展到混合多数据集 + 多维奖励 + 长度惩罚 + max-effort 提示。

## 训练路线图

```
Stage 1: GSM8K DAPO (580 steps)
  │  纯数学，单数据集，基础 DAPO/GRPO
  │  奖励: accuracy only
  │
  ▼
Stage 2: Mixed Math GDPO
  │  混合多数学数据集（GSM8K + MATH + OpenR1 + NuminaMath）
  │  多维奖励: accuracy + format + repetition + wait
  │  引入 GDPO（组内解耦归一化）
  │
  ▼
Stage 3: Mixed Math + Length Penalty
  │  在 Stage 2 基础上增加长度惩罚维度
  │  奖励: accuracy + format + wait + len
  │  鼓励简洁推理
  │
  ▼
Stage 4: Mixed Math + Length Penalty + Max Effort
  │  在 Stage 3 基础上引入 max_effort 数据源
  │  对困难题使用 "maximum effort" 提示模板
  │  奖励: accuracy + format + wait + len (+ max_effort bonus)
  │
  ▼
Stage 5: Mixed Logic + Math Reasoning
     混合逻辑推理（GPQA）与数学推理
     引入 prompt_type 区分不同数据集
     最终全能力数学推理模型
```

## 各阶段详情

### Stage 1 — GSM8K DAPO

**目标**：在单一数学任务上建立基础推理能力。

| 项目 | 配置 |
|------|------|
| 算法 | GRPO (DAPO reward manager) |
| 数据集 | GSM8K (train + test) |
| 奖励 | `accuracy_reward` only（基于 `math_reward.compute_score`） |
| Rollout n | 32 |
| max_prompt_length | 512 |
| max_response_length | 800 |
| max_model_len | 1400 |

**文件**：
- `stage 1 - gsm8k 580 step/gsm8k_dapo_train_recipe.sh`
- `stage 1 - gsm8k 580 step/gsm8k_reward.py`

### Stage 2 — Mixed Math GDPO

**目标**：扩展数据分布，引入多维奖励和 GDPO 算法。

| 项目 | 配置 |
|------|------|
| 算法 | GDPO (Group reward-Decoupled normalization) |
| 数据集 | GSM8K + MATH-lighteval + OpenR1-Math-220k + NuminaMath-CoT |
| 奖励维度 | accuracy (0.7) + format (0.1) + repetition (0.1) + wait (0.1) |
| Rollout n | 16 |
| max_prompt_length | 1700 |
| max_response_length | 4000 |
| max_model_len | 6000 |

**GDPO 核心思想**：对多个奖励维度在组内**独立归一化**后再加权聚合，避免主导信号（如准确率）淹没弱信号（如格式）。

**文件**：
- `stage 2 - mixed math dataset/mixed_math_gdpo_train_recipe.sh`
- `stage 2 - mixed math dataset/mixed_math_gdpo_reward.py`
- `stage 2 - mixed math dataset/mixed_math_gdpo_training_design.md` ← 详细设计文档

### Stage 3 — Length Penalty

**目标**：鼓励模型产出简洁推理，减少冗余 token。

与 Stage 2 的主要区别：
- 新增 `len_reward` 维度，惩罚过长响应（仅对正确答案生效）
- 奖励权重：accuracy (0.7) + format (0.1) + wait (0.1) + **len (0.1)**
- 使用 Muon + AdamW 混合优化器

**文件**：
- `stage 3 - mixed math dataset + len penalty/mixed_math_gdpo_train_recipe.sh`
- `stage 3 - mixed math dataset + len penalty/mixed_math_gdpo_reward.py`

### Stage 4 — Max Effort

**目标**：对困难问题使用 "maximum effort" 提示，让模型在复杂问题上投入更多推理资源。

与 Stage 3 的主要区别：
- 训练数据中包含 `data_source == "max_effort"` 的样本
- 奖励函数识别 `data_source` 字段，对 max_effort 样本给予额外的长度奖励空间
- 非 max_effort 样本：长度惩罚；max_effort 样本：不惩罚长度或给予 bonus

**文件**：
- `stage 4 - mixed dataset + len penalty + max effort/mixed_math_gdpo_train_recipe.sh`
- `stage 4 - mixed dataset + len penalty + max effort/mixed_math_gdpo_reward_max_effort.py`

### Stage 5 — Mixed Logic + Math

**目标**：最终阶段，混合逻辑推理与数学推理，构建全能推理模型。

| 项目 | 配置 |
|------|------|
| 数据集 | GPQA（逻辑推理）+ MATH + GSM8K + ... |
| 提示策略 | 按 `prompt_type` 区分：普通题用 standard prompt，max_effort 题用 intense prompt |
| 奖励 | 统一的多维奖励，按 `prompt_type` 微调 |

**数据生成脚本**：
- `stage 4 - mixed stage 5 - mixed_logic_and_math/mixed_reasoning_dataset_with_prompt_types.py`
- `stage 4 - mixed stage 5 - mixed_logic_and_math/reward.py`

## 数据集生成

### 生成混合数学数据集

```bash
# 使用 examples/ 下的创建脚本
python examples/data_preprocess/create_mixed_data.py --output_dir ./data/mixed_math

# 或者直接运行
python examples/data_preprocess/mixed_math_dataset.py
```

### 生成混合推理数据集（含 GPQA）

```bash
python "Rl_Specilist/math/stage 4 - mixed stage 5 - mixed_logic_and_math/mixed_reasoning_dataset_with_prompt_types.py" \
    --output_dir ./data/mixed_reasoning
```

更多细节见 `examples/data_preprocess/README_mixed_math.md`。

## GDPO vs GRPO

| 特性 | GRPO | GDPO |
|------|------|------|
| 奖励处理 | 先求和再归一化 | 每个维度独立归一化后再加权 |
| 多维奖励 | 主导信号会淹没弱信号 | 各维度公平竞争，权重可控 |
| 奖励函数返回 | 单一 float | dict（含 `score` + 各维度子奖励） |
| 适用场景 | 单一奖励或维度间量级相近 | 多维奖励且量级差异大 |
| 论文 | [GRPO](https://arxiv.org/abs/2402.03300) | [GDPO](https://arxiv.org/abs/2601.05242) |

## 可用的稳定性技术

训练过程中可以组合使用以下技术，详见各独立文档：

| 技术 | 文档 | 说明 |
|------|------|------|
| Dynamic Sampling | `../dapo_dynamic_sampling_guide.md` | 组重采样、过滤、reward 重采样 |
| Off-Policy Seq Mask | `../offpolicy_seq_mask_usage.md` | DeepSeek-V3.2 off-policy 序列掩码 |
| Unbiased KL Estimate | `../unbiased_kl_estimate.md` | 无偏 KL 估计 |
| Keep Sampling Mask | `../keep_sampling_mask.md` | 训练-推理采样一致性 |
| Optimizer State Reset | `../optimizer_state_reset.md` | 每次迭代重置优化器状态 |
| Muon Optimizer | `../muon/muon.md` | Muon + AdamW 混合优化器 |
| Repetition Penalty | `../penalty_features.md` | rollout 重复惩罚 + 长度惩罚 |

## 快速参考

| Stage | 启动命令 |
|-------|---------|
| 1 | `bash "stage 1 - gsm8k 580 step/gsm8k_dapo_train_recipe.sh"` |
| 2 | `bash "stage 2 - mixed math dataset/mixed_math_gdpo_train_recipe.sh"` |
| 3 | `bash "stage 3 - mixed math dataset + len penalty/mixed_math_gdpo_train_recipe.sh"` |
| 4 | `bash "stage 4 - mixed dataset + len penalty + max effort/mixed_math_gdpo_train_recipe.sh"` |

> **注意**：所有训练脚本中 `PATH_TO_TRAIN_PARQUET`、`PATH_TO_TEST_PARQUET`、`PATH_TO_MODEL` 需要替换为实际路径。
