# DAPO 动态采样功能使用指南

本文档介绍 verl 中与 DAPO/GRPO 相关的三种动态采样功能，帮助你根据训练场景选择和组合使用。

---

## 功能概览

| 功能 | 作用 | 对全0/全1组的处理 | 额外开销 |
|------|------|-------------------|---------|
| `group_resample` | 组内重新生成 responses | 对同一 prompt 重新生成，尝试修复 | 额外 rollout 计算 |
| `filter_groups` | 过滤无效组 | 直接丢弃，换新 prompt | 需要更多 prompt 数据 |
| `reward_resample` | 低 reward 组增加采样权重 | 增加采样概率 | 无额外计算 |

---

## 1. group_resample（组内动态采样）

### 原理

GRPO 算法中，当同一 prompt 的所有 response 的 reward 全为 0 或全为 1 时，组内 advantage 计算结果为 0，**没有梯度信号**，训练无效。

`group_resample` 检测这些"退化组"，对同一个 prompt 重新生成 responses，替换掉退化的 responses，直到组内有梯度信号或达到最大重试次数。

```
Prompt A 生成 4 个 response → reward: [0, 0, 0, 0]  ← 全0，无梯度
  ↓ 触发 group_resample
Prompt A 重新生成 4 个 response → reward: [0, 1, 0, 1]  ← 有梯度了！
  ↓ 替换
Prompt A 最终使用: [0, 1, 0, 1]
```

### 配置参数

```yaml
algorithm:
  group_resample:
    enable: true                # 是否开启
    metric: seq_reward          # 检测指标
    max_resample_rounds: 3      # 最大重试轮数
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable` | bool | false | 是否开启组内动态采样 |
| `metric` | str | "seq_reward" | 检测退化组的指标，见下方说明 |
| `max_resample_rounds` | int | 3 | 每步最大重采样轮数，每轮对退化组重新生成 responses |

### metric 选项

| metric | 数据来源 | 适用场景 |
|--------|----------|----------|
| `seq_reward` | `token_level_rewards` 求和 | **推荐**，通用场景 |
| `acc` | `reward_extra_info["acc"]` | 数学/代码等有明确对错的任务 |
| `score` | `token_level_scores` 求和 | 使用 reward model 打分的场景 |
| `seq_final_reward` | 最后一个 token 的 reward | 稀疏奖励场景 |

### 监控指标

训练日志中会输出以下指标：

```
group_resample/total_groups        # 总组数
group_resample/degenerate_groups   # 退化组数
group_resample/degenerate_ratio    # 退化组比例
group_resample/remaining_degenerate # 重采样后仍退化的组数
```

控制台输出示例：

```
group_resample round 1/3: 5 degenerate groups, 20 samples to regenerate
group_resample round 2/3: 2 degenerate groups, 8 samples to regenerate
group_resample done: 0 degenerate groups remain after 2 rounds
```

### 调参建议

| `max_resample_rounds` | 效果 | 适用场景 |
|-----------------------|------|----------|
| 1 | 只重试一次，开销最小 | 退化率低（<10%） |
| 3 | **推荐**，平衡效果和开销 | 大多数场景 |
| 5+ | 更积极修复，但开销大 | 退化率高（>30%）或 prompt 珍贵时 |

---

## 2. filter_groups（DAPO 动态采样过滤）

### 原理

DAPO 论文提出的动态采样策略：直接丢弃全0/全1的组，用新的 prompt 重新生成，保证训练 batch 中所有组都有有效梯度。

```
Prompt A: reward [0, 0, 0, 0] → ❌ 丢弃
Prompt B: reward [1, 0, 1, 0] → ✅ 保留
Prompt C: reward [1, 1, 1, 1] → ❌ 丢弃
  ↓ 不够？用新 prompt 重新生成
Prompt D: reward [0, 1, 0, 1] → ✅ 保留
```

### 配置参数

```yaml
data:
  gen_batch_size: 1536          # 每轮生成的 prompt 数量
  train_batch_size: 512         # 目标有效 prompt 数量

algorithm:
  filter_groups:
    enable: true                # 是否开启
    metric: seq_reward          # 过滤指标
    max_num_gen_batches: 10     # 最大重复生成轮数（0 或负数表示无上限）
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable` | bool | false | 是否开启过滤 |
| `metric` | str | None | 过滤指标，选项同 group_resample |
| `max_num_gen_batches` | int | 0 | 最大重复生成轮数，防止死循环 |

### 注意事项

- `gen_batch_size` 应 ≥ `train_batch_size`，否则一轮生成无法满足训练需求
- `max_num_gen_batches` 设为 0 或负数表示无上限，但数据质量差时可能导致死循环
- 当前 verl 核心代码实现了过滤逻辑，"重复生成"循环需在外部训练脚本中实现

---

## 3. reward_resample（低 reward 组加权重采样）

### 原理

在有效组中，让低 reward 的组被采样的概率更高，使模型更多从困难样本中学习。

```
Group A: mean_reward = 0.8 → 采样权重低
Group B: mean_reward = 0.2 → 采样权重高  ← 模型多学这个
Group C: mean_reward = 0.5 → 采样权重中
```

### 配置参数

```yaml
algorithm:
  reward_resample:
    enable: true                # 是否开启
    reweight_method: inverse_pow # 权重计算方法
    weight_pow: 2.0             # 权重指数
    temperature: 1.0            # 温度（softmax_inverse 方法用）
    score_key: token_level_scores # 分数来源
    group_key: uid              # 组级别重采样（设为 null 则为样本级别）
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable` | bool | false | 是否开启 |
| `reweight_method` | str | "inverse_pow" | 权重方法，见下方说明 |
| `weight_pow` | float | 2.0 | `inverse_pow` 方法的指数 |
| `temperature` | float | 1.0 | `softmax_inverse` 方法的温度 |
| `score_key` | str | "token_level_scores" | 分数来源 key |
| `group_key` | str | "uid" | 组 key，设为 null 则样本级别 |

### reweight_method 选项

| 方法 | 公式 | 效果 |
|------|------|------|
| `inverse_pow` | `(max_score - score + eps) ^ weight_pow` | **推荐**，低 reward 组权重高 |
| `softmax_inverse` | `softmax(-score / temperature)` | 低 reward 组权重高，温度控制锐度 |
| `pow` | `|score| ^ weight_pow` | 高 reward 组权重高（同 PF-PPO） |

### weight_pow 调参

| weight_pow | 效果 | 适用场景 |
|------------|------|----------|
| 1.0 | 线性反向加权，温和 | reward 差异不大 |
| 2.0 | **推荐**，平方反向加权 | 大多数场景 |
| 3.0+ | 更强偏向低 reward 组 | reward 差异很大 |

---

## 4. 功能组合使用

### 推荐组合：三个功能全开

```yaml
data:
  gen_batch_size: 1536
  train_batch_size: 512

algorithm:
  adv_estimator: grpo

  # 第1步：尝试修复退化组（先治）
  group_resample:
    enable: true
    metric: seq_reward
    max_resample_rounds: 3

  # 第2步：过滤仍退化的组（后筛）
  filter_groups:
    enable: true
    metric: seq_reward
    max_num_gen_batches: 10

  # 第3步：低 reward 组增加采样权重（多学困难样本）
  reward_resample:
    enable: true
    reweight_method: inverse_pow
    weight_pow: 2.0
    group_key: uid
```

### 执行顺序

```
训练循环:
  1. 生成 responses
  2. 计算 reward
  3. 【group_resample】对全0/全1组重新生成 responses，尝试修复
  4. 【filter_groups】过滤仍全0/全1的组（兜底清理）
  5. 计算 GRPO advantage
  6. 【reward_resample】低 reward 组增加采样权重
  7. 更新 actor/critic
```

### 不同场景的推荐配置

#### 场景 A：数学推理（reward 为 0/1 的准确率）

```yaml
algorithm:
  group_resample:
    enable: true
    metric: acc              # 用准确率检测
    max_resample_rounds: 3
  filter_groups:
    enable: true
    metric: acc
    max_num_gen_batches: 10
  reward_resample:
    enable: true
    reweight_method: inverse_pow
    weight_pow: 2.0
```

#### 场景 B：通用 RLHF（reward model 打分）

```yaml
algorithm:
  group_resample:
    enable: true
    metric: seq_reward       # 用 reward 求和检测
    max_resample_rounds: 2
  filter_groups:
    enable: false            # reward model 打分很少全0/全1，不需要
  reward_resample:
    enable: true
    reweight_method: inverse_pow
    weight_pow: 1.5          # reward 差异通常不大，用较小的指数
```

#### 场景 C：只关心困难样本（prompt 充足，不需要修复）

```yaml
algorithm:
  group_resample:
    enable: false            # 不需要修复，直接丢弃
  filter_groups:
    enable: true
    metric: seq_reward
    max_num_gen_batches: 10
  reward_resample:
    enable: true
    reweight_method: inverse_pow
    weight_pow: 2.0
```

#### 场景 D：prompt 珍贵，不想丢弃任何数据

```yaml
algorithm:
  group_resample:
    enable: true
    metric: seq_reward
    max_resample_rounds: 5   # 更积极修复
  filter_groups:
    enable: false            # 不丢弃
  reward_resample:
    enable: true
    reweight_method: inverse_pow
    weight_pow: 2.0
```

---

## 5. 完整启动命令示例

```bash
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/math/train.parquet \
    data.val_files=$HOME/data/math/test.parquet \
    data.train_batch_size=512 \
    data.gen_batch_size=1536 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B \
    actor_rollout_ref.rollout.n=4 \
    +algorithm.group_resample.enable=true \
    +algorithm.group_resample.metric=seq_reward \
    +algorithm.group_resample.max_resample_rounds=3 \
    +algorithm.filter_groups.enable=true \
    +algorithm.filter_groups.metric=seq_reward \
    +algorithm.filter_groups.max_num_gen_batches=10 \
    +algorithm.reward_resample.enable=true \
    +algorithm.reward_resample.reweight_method=inverse_pow \
    +algorithm.reward_resample.weight_pow=2.0 \
    +algorithm.reward_resample.group_key=uid
```

---

## 6. 常见问题

### Q1: group_resample 和 filter_groups 同时开启，会不会重复处理？

不会。执行顺序是先 `group_resample`（修复），后 `filter_groups`（过滤）。修复成功的组不会被 filter_groups 重复处理。

### Q2: group_resample 会增加多少训练时间？

取决于退化组比例。每轮重采样需要对退化组重新做 rollout + reward 计算。如果退化率为 20%，每轮额外开销约为正常步的 20%。`max_resample_rounds=3` 时，最坏情况额外开销约 60%。

### Q3: 退化率一直很高怎么办？

- 检查 reward 函数是否正确
- 检查 `metric` 选择是否匹配任务（数学任务用 `acc`，通用任务用 `seq_reward`）
- 适当增大 `rollout.n`（每个 prompt 生成更多 response，降低全0/全1概率）
- 检查模型是否已经收敛到极端策略

### Q4: reward_resample 的 weight_pow 怎么调？

- 观察训练日志中 `group_resample/degenerate_ratio`
- 如果退化率低（<10%），`weight_pow=1.0` 即可
- 如果 reward 分布差异大，用 `weight_pow=2.0` 或更高
- 如果训练不稳定，降低 `weight_pow`

### Q5: 可以只开 reward_resample 不开另外两个吗？

可以。三个功能完全独立。`reward_resample` 不处理全0/全1组，它只是在有效组中调整采样权重。如果只关心"让模型多学困难样本"而不关心全0/全1问题，只开 `reward_resample` 即可。
