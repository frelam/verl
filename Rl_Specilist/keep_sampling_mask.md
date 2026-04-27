# Keep Sampling Mask（保持采样掩码）

## 背景

Rollout 推理时使用 top-k / top-p 采样，会将 logits 截断为前 N 个概率最大的 token，然后在这 N 个 token 中做 softmax 得到采样概率。但训练时计算 log_prob 是在完整词表上做 softmax，两者不一致。如果模型为被截断的 token 分配高概率，训练时的 log_prob 会偏小，导致梯度偏差。

## 原理

在 rollout 推理时，记录每个 token 位置的候选 token indices（即 top-k/top-p 采样后保留的 token 集合）。这些 indices 会随数据流同步到训练侧。训练时，使用这些 indices 将 logits 中不在候选集合的 token 设为 `-inf`，然后做 `log_softmax`。这等价于只在候选 token 中做 softmax，确保训练和推理的一致性。

数据流与 routing replay 特性类似：

```
Rollout (vLLM/SGLang)
  |-- enable_keep_sampling_mask=True, keep_sampling_mask_num_tokens=N
  |-- vLLM: logprobs=N 返回 top-N token 的 indices
  |-- SGLang: top_logprobs_num=N 返回 top-N token 的 indices
  |-- TokenOutput.sampling_token_indices: list[list[int]]
  |
Agent Loop
  |-- AgentLoopOutput.sampling_token_indices (原始, 变长)
  |-- _InternalAgentLoopOutput.sampling_token_indices (padding 后, [1, response_length, num_candidates])
  |-- Batched: torch.cat -> TensorDict["sampling_token_indices"] shape [bsz, response_length, num_candidates]
  |
Training (dp_actor)
  |-- micro_batch["sampling_token_indices"] shape (..., num_candidates)
  |-- _apply_sampling_mask_to_logits(logits, sampling_token_indices)
  |-- mask.scatter_(-1, sampling_token_indices, True) -> logits.masked_fill(~mask, -inf)
  |-- logprobs_from_logits(masked_logits, labels) -> 在候选 token 中 softmax 得到的 log_prob
```

## 配置方式

需要同时配置 rollout 和 actor：

```yaml
actor_rollout_ref:
  actor:
    use_keep_sampling_mask: true

  rollout:
    enable_keep_sampling_mask: true
    keep_sampling_mask_num_tokens: 50   # 记录每个 token 位置的候选 token 数量，应 >= top_k
    top_k: 50                           # rollout 的 top_k
    top_p: 0.9                          # rollout 的 top_p
```

> **重要**：`keep_sampling_mask_num_tokens` 应 >= rollout 的 `top_k` 值，以确保所有候选 token 都被记录。如果同时使用 top-p，建议设置为与 top_k 相同的值，因为 top-p 是在 top-k 基础上的进一步过滤。

## 适用场景

- Rollout 使用了 top-k 或 top-p 采样（非 greedy）
- 训练时 log_prob 与 rollout 时的概率分布不一致
- 模型词表较大，被截断 token 的概率总和不可忽略

## 注意事项

- 默认关闭，需要显式启用
- 需要同时配置 rollout 和 actor 两侧
- 目前仅支持 FSDP 后端（dp_actor），Megatron 后端暂不支持
- 会增加少量内存开销（存储候选 token indices），但对训练速度影响很小
- 候选 token indices 来自 rollout 推理时的实际采样结果，而非训练侧重新计算，确保了训练-推理的一致性
