# Unbiased KL Estimate（无偏 KL 估计）

## 背景

在 RL 训练中，样本来自旧策略 π_old（rollout 阶段），但训练优化的是当前策略 π_θ。传统的 KL 估计器（如 k1、k3）虽然在旧策略下是无偏的，但其梯度相对于当前策略是有偏的。当重要性采样比率 π_θ(a|s) / π_old(a|s) 较大时，梯度会不稳定。

## 原理

通过重要性采样比率 r = exp(log_prob - old_log_prob) 对 KL 项进行重新加权，使 KL 估计在当前策略下无偏：

```
KL_unbiased = r · KL_old,  where r = exp(log_prob - old_log_prob).detach()
```

## 配置方式

在 actor 配置中启用：

```yaml
actor_rollout_ref:
  actor:
    use_kl_loss: true
    use_unbiased_kl: true
    kl_loss_type: low_var_kl   # 支持: "kl"(k1), "abs", "mse"(k2), "low_var_kl"(k3), "k3+" 等
    kl_loss_coef: 0.001
```

如果使用 KL reward penalty 模式（`use_kl_in_reward: true`），同样生效：

```yaml
algorithm:
  use_kl_in_reward: true
  kl_penalty: kl

actor_rollout_ref:
  actor:
    use_unbiased_kl: true
```

## 适用场景

- 训练过程中策略更新幅度较大（clip_ratio 较高或 PPO epochs 较多）
- 使用 off-policy 数据（如 decoupled rollout 模式）
- KL loss 梯度不稳定或出现 NaN

## 注意事项

- 默认关闭，需要显式启用
- IS 比率使用 `.detach()` 和 `.clamp(-20, 20)` 保证数值稳定性
