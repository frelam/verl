# Off-Policy Sequence Masking 使用文档

## 一、背景与原理

### 1.1 问题

在异步 RL 训练中（如 verl 的 rollout-training 解耦模式），存在以下 off-policy 问题：

1. **Rollout 数据来自旧策略**：大批量 rollout 数据由旧策略 π_old 生成，然后切分为多个 mini-batch 进行多次梯度更新，引入 off-policy 行为
2. **推理-训练框架不一致**：推理引擎（如 vLLM）和训练引擎（如 FSDP）在数值精度、采样逻辑等实现细节上存在差异，进一步加剧 off-policy 程度

当 π_old >> π_θ（旧策略对某个 token 赋予的概率远高于当前策略）且 advantage < 0 时，当前策略会因为一个"它已经不会再做出的行为"而受到惩罚。这种梯度信号是噪声，会误导优化过程，导致训练不稳定甚至崩溃。

### 1.2 解决方案

DeepSeek-V3.2 论文提出了 **Off-Policy Sequence Masking**，引入二值掩码 M：

```
M_{i,t} = 0   if (A_{i,t} < 0) AND (KL(π_old || π_θ)_{i,t} > δ)
M_{i,t} = 1   otherwise
```

其中：
- `A_{i,t}` 是 token 级别的 advantage
- `KL(π_old || π_θ)_{i,t} = log π_old(a_t|s_t) - log π_θ(a_t|s_t)` 是近似的 KL 散度
- `δ` 是控制策略偏离阈值的超参数

**直觉**：只从"自己的错误"中学习，屏蔽那些高度 off-policy 的负样本，因为它们可能只是噪声。

**关键设计**：
- 只屏蔽 **负 advantage + 高 KL** 的 token，正 advantage 的 token 永远不被屏蔽
- 支持两种粒度：token 级别和 sequence 级别

### 1.3 参考文献

> DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models
> Section 3.1 "Scaling GRPO" — Off-Policy Sequence Masking
> https://arxiv.org/abs/2512.02556

---

## 二、实现位置

| 文件 | 修改内容 |
|---|---|
| `verl/trainer/ppo/core_algos.py` | 新增 `compute_offpolicy_seq_mask()` 函数；修改 `compute_policy_loss_vanilla()` 集成 mask |
| `verl/workers/config/actor.py` | `PolicyLossConfig` 新增 3 个配置字段 |

---

## 三、配置参数

在 `PolicyLossConfig` 中新增以下参数：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `use_offpolicy_seq_mask` | bool | `False` | 是否启用 Off-Policy Sequence Masking |
| `offpolicy_seq_mask_kl_threshold` | float | `3.0` | KL 散度阈值 δ，仅当 KL > δ 且 advantage < 0 时屏蔽 token。典型范围 1.0–5.0 |
| `offpolicy_seq_mask_granularity` | str | `"token"` | 屏蔽粒度：`"token"` 逐 token 屏蔽；`"sequence"` 若序列中任一 token 触发屏蔽，则整条序列屏蔽 |

---

## 四、使用方式

### 4.1 YAML 配置

在训练的 YAML 配置文件中，在 `actor_rollout_ref.actor.policy_loss` 下添加：

```yaml
actor_rollout_ref:
  actor:
    policy_loss:
      use_offpolicy_seq_mask: true
      offpolicy_seq_mask_kl_threshold: 3.0
      offpolicy_seq_mask_granularity: token
```

### 4.2 Python 配置

```python
from verl.workers.config.actor import PolicyLossConfig

policy_loss_config = PolicyLossConfig(
    use_offpolicy_seq_mask=True,
    offpolicy_seq_mask_kl_threshold=3.0,
    offpolicy_seq_mask_granularity="token",
)
```

### 4.3 命令行覆盖

```bash
python -m verl.trainer.main_ppo \
    +actor_rollout_ref.actor.policy_loss.use_offpolicy_seq_mask=true \
    +actor_rollout_ref.actor.policy_loss.offpolicy_seq_mask_kl_threshold=3.0 \
    +actor_rollout_ref.actor.policy_loss.offpolicy_seq_mask_granularity=token
```

---

## 五、监控指标

启用后，训练日志中会新增以下指标（前缀 `offpolicy_seq_mask/`）：

| 指标 | 含义 |
|---|---|
| `offpolicy_seq_mask/masked_fraction` | 被 mask 的 token 占所有有效 token 的比例 |
| `offpolicy_seq_mask/neg_adv_fraction` | 负 advantage token 占比 |
| `offpolicy_seq_mask/high_kl_fraction` | 高 KL token 占比 |
| `offpolicy_seq_mask/kl_threshold` | 当前使用的 KL 阈值 δ |
| `offpolicy_seq_mask/mean_kl` | 平均 KL 散度 |
| `offpolicy_seq_mask/max_kl` | 最大 KL 散度 |
| `offpolicy_seq_mask/seq_masked_fraction` | 包含被 mask token 的序列占比 |

---

## 六、参数调优建议

### 6.1 KL 阈值 δ 的选择

| δ 值 | 效果 | 适用场景 |
|---|---|---|
| 1.0 | 激进屏蔽，大量负 advantage token 被 mask | 训练极不稳定、频繁崩溃 |
| 3.0 | **推荐默认值**，平衡屏蔽与学习 | 一般异步训练场景 |
| 5.0 | 保守屏蔽，仅屏蔽极端 off-policy 样本 | 训练较稳定，只需微调 |

### 6.2 屏蔽粒度的选择

| 粒度 | 效果 | 适用场景 |
|---|---|---|
| `token` | 精细控制，仅屏蔽问题 token | **推荐默认值**，大多数场景 |
| `sequence` | 粗粒度，整条序列要么全用要么全不用 | 序列级一致性要求高的场景 |

### 6.3 与其他稳定性技术的配合

Off-Policy Sequence Masking 可以与以下技术组合使用：

- **Unbiased KL Estimate**（`use_unbiased_kl=True`）：修正 KL 估计偏差
- **Rollout Correction**（IS weights + Rejection Sampling）：处理推理-训练框架不一致
- **Keep Sampling Mask**（`use_keep_sampling_mask=True`）：保持采样掩码一致性
- **PPO Dual-Clip**（`clip_ratio_c`）：限制负 advantage 下的 ratio 下界

---

## 七、算法细节

### 7.1 KL 近似

使用 `approx_kl = log π_old - log π_θ` 作为 KL(π_old || π_θ) 的近似。这是 PPO 中常用的近似 KL 估计器，计算高效且在训练循环中已可用（无需额外的前向传播）。

### 7.2 Mask 的梯度阻断

`offpolicy_mask` 在返回前通过 `.detach()` 阻断梯度，确保 mask 不会影响策略梯度的计算。mask 仅作为 `response_mask` 的乘法因子，改变哪些 token 参与损失聚合。

### 7.3 与 PPO Dual-Clip 的关系

Off-Policy Sequence Masking 和 PPO Dual-Clip 都处理负 advantage 下的稳定性问题，但机制不同：

- **Dual-Clip**：通过 `clip(ratio, max=clip_ratio_c)` 限制 ratio 的上界，防止负 advantage 下 ratio 过大导致梯度爆炸
- **Off-Policy Sequence Masking**：直接屏蔽高 KL + 负 advantage 的 token，从根源上消除噪声梯度

两者可以同时使用，提供双重保护。
