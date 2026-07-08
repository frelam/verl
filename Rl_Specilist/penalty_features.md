# Repetition Penalty & Length Penalty

训练中的两类惩罚机制：Rollout 阶段的重复惩罚和 Reward 阶段的长度惩罚。

---

## 一、Repetition Penalty（重复惩罚）

### 原理

在 rollout 推理阶段，对已出现的 token 施加惩罚，降低其被再次采样的概率，防止模型陷入重复循环。

```
logit_penalized = logit_original - repetition_penalty * 1(token_already_generated)
```

### 配置方式

在 rollout 配置中设置：

```yaml
actor_rollout_ref:
  rollout:
    repetition_penalty: 1.0   # 1.0 = 关闭；>1.0 = 惩罚重复
```

或通过命令行：

```bash
python -m verl.trainer.main_ppo \
    actor_rollout_ref.rollout.repetition_penalty=1.05 \
    ...
```

### 参数说明

| repetition_penalty | 效果 |
|--------------------|------|
| `1.0`（默认） | 关闭重复惩罚（math specialist 训练中通常设为 1.0） |
| `1.02 - 1.05` | 轻微惩罚，减少偶发重复 |
| `1.05 - 1.1` | 中等惩罚，适用于容易重复的小模型 |
| `> 1.1` | 强惩罚，可能影响生成多样性 |

### 适用场景

- 模型容易陷入 token 级别重复循环
- 长文本生成中对重复敏感的任务
- 数学推理中通常不需要（默认 `1.0`），因为重复往往是模型在反思和迭代，而非无意义循环

### 注意事项

- 值过大（>1.2）会严重损害生成质量
- 在数学 specialist 训练中，建议配合 `repetition_reward` 维度使用（在 reward 层面处理重复问题），而非在 rollout 侧惩罚
- 与 `keep_sampling_mask` 配合使用时不冲突，两者作用于不同阶段

---

## 二、Length Penalty（长度惩罚）

### 原理

在 reward 计算中引入长度维度，对过长的推理给予负奖励（仅对正确答案生效），鼓励模型生成简洁的推理过程。

```
len_reward = base_score > 0 且 len < threshold ?  1.0  :  0.0
            base_score > 0 且 len > threshold ? -0.5  :  0.0  (严格版)
```

### 设计原则

1. **仅对正确答案惩罚长度**：错误答案不惩罚长度，避免模型通过"缩短错误推理"获取奖励
2. **阈值灵活**：可根据任务难度设置不同阈值，简单题要求更短，困难题允许更长
3. **与 max_effort 配合**：标记为 max_effort 的数据源可豁免长度惩罚

### 在 GDPO 中的使用

在 GDPO 训练中，`len_reward` 作为独立维度参与组内归一化：

```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gdpo \
    +algorithm.gdpo_reward_keys='["accuracy_reward", "format_reward", "wait_reward", "len_reward"]' \
    +algorithm.gdpo_reward_weights='[0.7, 0.1, 0.1, 0.1]' \
    reward_model.reward_manager=gdpo \
    ...
```

### 奖励函数实现

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    # ... 计算 base_score ...
    
    # 长度惩罚（仅对正确答案）
    tokens = len(solution_str.split())
    is_correct = base_score > 0
    
    if is_correct:
        if tokens <= MAX_IDEAL_LEN:
            len_reward = 1.0   # 简洁正确 → 满分
        elif tokens <= MAX_ACCEPTABLE_LEN:
            len_reward = 0.5   # 稍长 → 半分
        else:
            len_reward = -0.5  # 过长 → 负分
    else:
        len_reward = 0.0       # 错误答案不惩罚长度
    
    return {
        "score": w_acc * accuracy_reward + w_len * len_reward + ...,
        "accuracy_reward": accuracy_reward,
        "len_reward": len_reward,
        ...
    }
```

### Max Effort 豁免

对于标记为 `data_source == "max_effort"` 的样本，长度惩罚可以豁免或放宽：

```python
if extra_info.get("data_source") == "max_effort":
    # max_effort 样本：长度惩罚减半，或完全豁免
    len_reward = max(0.0, len_reward)
```

这使得模型在困难问题上可以"放心"投入更多 token 进行推理，而在简单问题上仍然保持简洁。

### 调参建议

| 场景 | 推荐阈值（token 数） | 惩罚幅度 |
|------|---------------------|---------|
| GSM8K (简单题) | 200-400 | 温和 (-0.1) |
| MATH (中等) | 500-1000 | 中等 (-0.3) |
| GPQA (困难) | 豁免 (max_effort) | 不限 |
| 混合数据 | 按 prompt_type 区分 | 按难度分级 |

---

## 三、两种惩罚的配合

在数学 specialist 训练中，通常的配置策略是：

```
Rollout 侧: repetition_penalty = 1.0  (不惩罚，让模型自由探索)
Reward 侧:  len_reward + repetition_reward  (在 reward 层面引导简洁 + 无重复)
```

这样设计的原因：
- **Rollout 侧惩罚会限制探索空间**，可能阻止模型在推理过程中进行重复验证（合理的重复）
- **Reward 侧引导更灵活**，可以区分"有用的重复"（验证步骤）和"无用的重复"（死循环）

### 完整配置示例

```yaml
# rollout — 不限制生成
actor_rollout_ref:
  rollout:
    repetition_penalty: 1.0   # 关闭
    temperature: 1.0           # 鼓励多样性

# reward — GDPO 多维引导
algorithm:
  adv_estimator: gdpo
  gdpo_reward_keys: ["accuracy_reward", "format_reward", "repetition_reward", "wait_reward", "len_reward"]
  gdpo_reward_weights: [0.7, 0.1, 0.1, 0.1, 0.1]
```
