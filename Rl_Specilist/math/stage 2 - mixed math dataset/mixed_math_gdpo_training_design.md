# Mixed Math GDPO 训练策略说明

## 一、算法概述

本方案采用 **GDPO (Group reward-Decoupled Normalization Policy Optimization)** 算法对数学推理模型进行强化学习训练。GDPO 的核心思想是：对多个奖励维度在组内**独立归一化**后再加权聚合，避免某个主导性奖励信号（如准确率）淹没较弱的奖励信号（如格式、重复惩罚等）。

**GDPO 三步计算流程：**

1. **组内解耦归一化**：对每个奖励维度 k，在组 g 内独立做 GRPO 归一化：
   `A_k = (r_k - μ_group(r_k)) / (σ_group(r_k) + ε)`

2. **加权聚合**：按配置权重求和：
   `A_sum = Σ_k w_k · A_k`

3. **批次级白化**：对聚合结果做 masked_whiten 得到最终优势值

> 论文参考：[GDPO](https://arxiv.org/abs/2601.05242)

---

## 二、多维奖励设计

奖励函数由四个独立维度组成，最终通过 GDPO 解耦归一化后按权重聚合：

| 奖励维度 | 权重 | 说明 |
|---|---|---|
| `accuracy_reward` | 0.7 | 答案正确性奖励，基于 math_reward 的基础得分 |
| `format_reward` | 0.1 | 思考格式奖励，检查是否包含且仅包含一个 `⋘...⋫` 思考块 |
| `repetition_reward` | 0.1 | 无重复奖励，检测 3~5-gram 是否出现 ≥5 次重复 |
| `wait_reward` | 0.1 | wait 用词奖励，限制 "wait" 出现次数（≤10次）和占比（≤1%） |

### 奖励计算逻辑

```
base_score = math_reward.compute_score(solution, ground_truth)
if base_score == 0: base_score = -0.1    # 错误答案给予轻微负奖励

accuracy_reward = base_score
format_reward   = base_score > 0 且 恰好1个思考块 ? 1.0 : 0.0
repetition_reward = base_score > 0 且 无重复 ? 1.0 : 0.0
wait_reward     = base_score > 0 且 wait不超标 ? 1.0 : 0.0

score = 0.7 * accuracy_reward + 0.1 * format_reward + 0.1 * repetition_reward + 0.1 * wait_reward
```

**关键设计**：格式/重复/wait 奖励仅在答案正确（`base_score > 0`）时才生效，避免模型通过"格式正确但答案错误"的方式获取奖励。

---

## 三、训练配置要点

### 数据配置
- 训练/验证集为 Parquet 格式，`max_prompt_length=1700`，`max_response_length=4000`
- 超长 prompt 会被过滤，截断策略设为 `error`（超长直接报错而非静默截断）

### Actor 配置
- 学习率 `1e-6`，权重衰减 `0.1`，梯度裁剪 `1.0`
- 使用动态 batch size，PPO mini batch = 32，micro batch per GPU = 1
- Clip ratio 范围 `[0.2, 0.28]`，clip_ratio_c = 10.0（GDPO 推荐的较窄裁剪范围）
- 损失聚合模式 `token-mean`
- FSDP 并行度 = 2，不开启参数/优化器 offload

### Rollout 配置
- 使用 vLLM 引擎，每个 prompt 采样 **n=16** 个响应（组采样）
- 温度 `1.0`，repetition_penalty `1.0`（不额外惩罚，由奖励函数处理）
- `max_model_len=6000`，GPU 显存利用率 0.7
- 不启用 chunked prefill

### 训练流程
- 8 GPU 单节点训练，总 epochs = 15
- 每 10 步验证一次，每 20 步保存检查点
- 不使用 critic warmup（`critic_warmup=0`）

---

## 四、GDPO vs GRPO 的区别

| 特性 | GRPO | GDPO |
|---|---|---|
| 奖励处理 | 先求和再归一化 | 每个维度独立归一化后再加权 |
| 多维奖励 | 主导信号会淹没弱信号 | 各维度公平竞争，权重可控 |
| 奖励函数返回 | 单一 float | dict（含 score + 各维度子奖励） |
| 适用场景 | 单一奖励或维度间量级相近 | 多维奖励且量级差异大 |

---

## 五、文件说明

| 文件 | 说明 |
|---|---|
| `mixed_math_gdpo_train_recipe.sh` | 训练启动脚本，配置 GDPO 算法参数、模型/数据/训练流程 |
| `mixed_math_gdpo_reward.py` | 奖励函数实现，返回 dict 格式（含 `score` + 四个子奖励维度） |

### 奖励函数与训练脚本的对应关系

训练脚本中的关键配置：
```bash
custom_reward_function.path='mixed_math_gdpo_reward.py'   # 指向奖励函数文件
custom_reward_function.name='compute_score'                # 入口函数名
+algorithm.gdpo_reward_keys='["accuracy_reward", "format_reward", "repetition_reward", "wait_reward"]'
+algorithm.gdpo_reward_weights='[0.7, 0.1, 0.1, 0.1]'
reward_model.reward_manager=gdpo                           # 使用 GDPO 奖励管理器
```

`mixed_math_gdpo_reward.py` 的 `compute_score` 返回的 dict 键名必须与 `gdpo_reward_keys` 一一对应，`gdpo_reward_weights` 定义各维度的聚合权重。GDPO 奖励管理器会从返回的 dict 中提取各维度子奖励，分别归一化后再加权聚合。

---

## 六、注意事项

1. **奖励函数必须返回 dict**：GDPO 的 `reward_manager` 要求 `compute_score` 返回包含 `score` 键和各维度子奖励键的字典，而非单一 float。当前 `mixed_math_gdpo_reward.py` 已正确实现此格式。

2. **`mixed_math_custom_reward.py` 不适用于 GDPO**：该文件返回 float 而非 dict，且存在 `matches` 未定义、`flag` 拼写错误等 bug，与 GDPO 训练配置不匹配。请使用 `mixed_math_gdpo_reward.py`。
