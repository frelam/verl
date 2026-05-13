# Optimizer State Reset Per Iteration

## 功能概述

在默认情况下，优化器（如 AdamW）在训练过程中会持续累积状态信息，包括：
- **一阶矩（momentum / exp_avg）**：梯度的指数移动平均
- **二阶矩（exp_avg_sq）**：梯度平方的指数移动平均
- **步数计数（step）**：用于偏差校正的步数

这些状态在训练的每次迭代（iteration）之间会延续，即上一次迭代结束时的优化器状态会作为下一次迭代的初始状态。这在大多数情况下是期望的行为，但在某些场景下，你可能希望每次迭代都从"干净"的优化器状态开始。

`reset_optimizer_state_per_iter` 功能允许你在每次训练迭代开始时重置优化器状态，使得每次参数更新不受之前迭代累积的动量和二阶统计量的影响。

## 适用场景

- **PPO/GSPO 训练**：每次迭代使用不同的 rollout 数据，可能不希望优化器动量跨迭代累积
- **非平稳数据分布**：当训练数据分布在不同迭代间变化较大时
- **调试与分析**：需要隔离每次迭代的影响，分析单步更新的效果
- **实验对比**：对比有/无动量延续对训练效果的影响

## 配置方式

### 方式一：通过 YAML 配置文件

在优化器配置中添加 `reset_optimizer_state_per_iter: true`：

```yaml
# FSDP 优化器配置示例 (verl/trainer/config/optim/fsdp.yaml)
actor_rollout_ref:
  actor:
    optim:
      lr: 1e-6
      weight_decay: 0.1
      reset_optimizer_state_per_iter: true   # 启用每次迭代重置优化器状态
```

对于 critic 模型，同样可以配置：

```yaml
critic:
  optim:
    lr: 1e-5
    reset_optimizer_state_per_iter: true   # 启用每次迭代重置优化器状态
```

### 方式二：通过命令行参数

在启动脚本中通过 Hydra 的命令行覆盖语法添加：

```bash
# 同时重置 actor 和 critic 的优化器状态
python -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.optim.reset_optimizer_state_per_iter=True \
    critic.optim.reset_optimizer_state_per_iter=True \
    ...
```

在 shell 脚本中的使用示例：

```bash
ACTOR_CONFIG=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=10
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.optim.reset_optimizer_state_per_iter=True
)

CRITIC_CONFIG=(
    critic.optim.lr=1e-5
    critic.optim.reset_optimizer_state_per_iter=True
)
```

### 方式三：仅重置 Actor 或仅重置 Critic

你可以独立控制 actor 和 critic 的优化器状态重置：

```bash
# 仅重置 actor 的优化器状态
python -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.optim.reset_optimizer_state_per_iter=True \
    ...
```

## SFT 训练中的使用

SFT 训练同样支持此功能：

```bash
python -m verl.trainer.sft_trainer \
    optim.reset_optimizer_state_per_iter=True \
    ...
```

## 支持的优化器配置

所有优化器配置类型均支持此功能：

| 配置类型 | YAML 文件 | 说明 |
|---------|----------|------|
| `FSDPOptimizerConfig` | `optim/fsdp.yaml` | FSDP 策略 |
| `McoreOptimizerConfig` | `optim/megatron.yaml` | Megatron 策略 |
| `TorchtitanOptimizerConfig` | `optim/torchtitan.yaml` | TorchTitan 策略 |
| `VeOmniOptimizerConfig` | `optim/veomni.yaml` | VeOmni 策略 |
| `AutomodelOptimizerConfig` | `optim/automodel.yaml` | Automodel 策略 |
| `MuonOptimizerConfig` | `optim/muon_fsdp.yaml` | Muon + AdamW 混合优化器 |

## 工作原理

当 `reset_optimizer_state_per_iter=True` 时，在每次训练迭代开始时（actor/critic 更新之前），系统会：

1. 遍历优化器中所有参数的状态字典
2. 将所有张量类型的状态（如 `exp_avg`、`exp_avg_sq`）清零
3. 将所有数值类型的状态（如 `step` 计数器）重置为 0
4. 保持学习率、权重衰减等超参数不变

**注意**：此操作不会影响学习率调度器（LR Scheduler）的状态，学习率会按照既定策略正常变化。

## 性能影响

- 重置操作本身的开销很小，仅涉及对优化器状态张量的 `zero_()` 操作
- 由于动量被重置，训练可能需要更多的迭代才能收敛
- 建议在实验中对比开启和关闭此功能的效果，选择最适合你任务的配置

## 默认值

`reset_optimizer_state_per_iter` 默认为 `false`，即保持原有行为（优化器状态跨迭代延续）。
