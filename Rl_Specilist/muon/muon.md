# Muon Optimizer in verl

Last updated: 04/28/2026.

Muon (**M**oment**U**m **O**rthogonalized by **N**ewton-**S**chulz) is an optimizer for 2D weight matrices in neural network hidden layers. It orthogonalizes the SGD-momentum update via an efficient Newton-Schulz iteration, which amplifies rare gradient directions and improves training sample efficiency.

verl implements Muon based on the [MoonshotAI ZeRO-1 distributed scheme](https://github.com/MoonshotAI/Moonlight), which has been validated at scale up to 1.5B parameters. Muon is fully integrated with verl's FSDP training backend and supports multi-GPU distributed training out of the box.

> [!IMPORTANT]
> Muon only optimizes **2D hidden-layer weight matrices**. All other parameters (embeddings, output heads, biases, LayerNorm) are automatically optimized with AdamW. This mixed strategy is handled by the `MuonWithAdamW` optimizer class.

## How It Works

For each 2D weight matrix, Muon performs the following update at each step:

1. **Momentum accumulation**: `B_t = μ · B_{t-1} + G_t` (Nesterov-style)
2. **Orthogonalization**: `O_t = NewtonSchulz5(B_t)` — approximately replaces the update with the nearest semi-orthogonal matrix
3. **RMS scaling**: `O_t *= 0.2 × √max(m, n)` — normalizes update magnitude across different matrix shapes
4. **Parameter update**: `W_t = W_{t-1} - η · O_t`

The Newton-Schulz iteration runs in bfloat16 and only involves matrix multiplications, adding less than 1% FLOP overhead in typical LM training scenarios.

## Quick Start

### Command Line

Add the following flags to your existing training script:

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.optim._target_=verl.workers.config.MuonOptimizerConfig \
    actor_rollout_ref.actor.optim.lr=2e-3 \
    actor_rollout_ref.actor.optim.momentum=0.95 \
    actor_rollout_ref.actor.optim.ns_steps=5 \
    actor_rollout_ref.actor.optim.rms_scale=0.2 \
    actor_rollout_ref.actor.optim.muon_param_filter=hidden \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    ...
```

### YAML Configuration

Alternatively, use the provided config template:

```yaml
# In your training config YAML
actor_rollout_ref:
  actor:
    optim:
      _target_: verl.workers.config.MuonOptimizerConfig
      lr: 2e-3
      momentum: 0.95
      ns_steps: 5
      rms_scale: 0.2
      muon_param_filter: hidden
      weight_decay: 0.01
```

## Configuration Reference

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | float | `2e-3` | Learning rate for Muon (2D hidden weights) |
| `momentum` | float | `0.95` | Momentum factor (Nesterov-style) |
| `ns_steps` | int | `5` | Number of Newton-Schulz iteration steps |
| `ns_eps` | float | `1e-7` | Epsilon for Newton-Schulz normalization |
| `rms_scale` | float | `0.2` | RMS scaling factor. Set to `0` to disable |
| `weight_decay` | float | `0.01` | Decoupled weight decay (applied to both Muon and AdamW) |

### AdamW Parameters (for non-Muon params)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adamw_lr` | float\|null | `null` | Learning rate for AdamW. If `null`, uses `lr` |
| `adamw_betas` | tuple | `(0.9, 0.999)` | Betas for AdamW |
| `adamw_eps` | float | `1e-8` | Epsilon for AdamW |

### Parameter Filtering

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `muon_param_filter` | str | `"hidden"` | How to select 2D params for Muon. See below |

**`muon_param_filter` options:**

- `"hidden"` (recommended): Muon is applied to 2D hidden-layer weights only. Parameters whose names contain `embed`, `lm_head`, `norm`, or `bias` are excluded and optimized with AdamW instead.
- `"all_2d"`: Muon is applied to all 2D parameters. Embeddings and output heads will also use Muon, which may hurt performance.

### Inherited FSDP Parameters

`MuonOptimizerConfig` inherits all parameters from `FSDPOptimizerConfig`, including:

- `clip_grad`: Gradient clipping norm (default `1.0`)
- `lr_scheduler_type`: `"constant"` or `"cosine"`
- `lr_warmup_steps` / `lr_warmup_steps_ratio`: Learning rate warmup
- `min_lr_ratio`: Minimum LR ratio for cosine schedule

## Distributed Training

Muon supports distributed training via ZeRO-1 style data parallelism:

1. **Gradient aggregation**: Reduce-scatter across the DP group (handled by FSDP backward pass)
2. **Momentum update**: Each rank updates its local momentum shard
3. **All-gather momentum**: Full momentum matrix is reconstructed across the DP group
4. **Local orthogonalization**: Newton-Schulz iteration runs on the full matrix
5. **Shard extraction**: Only the local shard of the orthogonalized update is kept

The communication overhead is approximately **1.25×** that of standard ZeRO-1 AdamW (one extra all-gather per step).

> [!NOTE]
> Distributed Muon requires **FSDP2** or FSDP1 with `use_orig_params=True`, because FSDP1's default flat parameter mode loses the 2D matrix structure needed for orthogonalization. verl's `fsdp2` strategy satisfies this requirement by default.

## Example: GRPO with Muon

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B \
    actor_rollout_ref.actor.optim._target_=verl.workers.config.MuonOptimizerConfig \
    actor_rollout_ref.actor.optim.lr=2e-3 \
    actor_rollout_ref.actor.optim.momentum=0.95 \
    actor_rollout_ref.actor.optim.ns_steps=5 \
    actor_rollout_ref.actor.optim.rms_scale=0.2 \
    actor_rollout_ref.actor.optim.muon_param_filter=hidden \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.n=5 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1
```

## Python API

You can also use Muon directly in PyTorch code:

```python
from verl.utils.muon import Muon, MuonWithAdamW

# Option 1: Muon only (for 2D parameters)
optimizer = Muon(
    model.parameters(),
    lr=2e-3,
    momentum=0.95,
    ns_steps=5,
    rms_scale=0.2,
    weight_decay=0.01,
)

# Option 2: Mixed Muon + AdamW (recommended)
muon_params = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed" not in n and "lm_head" not in n]
adamw_params = [p for n, p in model.named_parameters() if p not in muon_params]

optimizer = MuonWithAdamW(
    muon_params=muon_params,
    adamw_params=adamw_params,
    lr=2e-3,
    momentum=0.95,
    adamw_lr=2e-3,
)

# Standard training loop
optimizer.zero_grad()
loss = model(x)
loss.backward()
optimizer.step()
```

## Tuning Guide

### Learning Rate

Muon typically requires a **higher learning rate** than AdamW. Recommended starting points:

| Model Size | Muon LR | AdamW LR (for non-2D params) |
|------------|---------|-------------------------------|
| 7B | 2e-3 | 2e-3 |
| 14B | 2e-3 | 2e-3 |
| 72B | 1e-3 | 1e-3 |

If using `adamw_lr=null` (same as Muon LR), start with `2e-3` and adjust.

### Momentum

The default `momentum=0.95` works well for most cases. Values between `0.9` and `0.99` are reasonable.

### RMS Scale

`rms_scale=0.2` (from MoonshotAI) normalizes the update magnitude so that Muon's effective update RMS matches AdamW's. This is important when mixing Muon and AdamW in the same training run. Set to `0` only if you want to tune the learning rate manually for each matrix shape.

### Newton-Schulz Steps

`ns_steps=5` is sufficient for convergence in all tested scenarios. Increasing to 7-10 provides marginal improvement at the cost of extra computation. Decreasing to 3 may work for small models but risks insufficient orthogonalization.

## Limitations

- **FSDP1 flat parameter mode**: Not supported. Use `strategy=fsdp2` or FSDP1 with `use_orig_params=True`.
- **Megatron backend**: Not yet supported. Muon currently works with the FSDP backend only.
- **Optimizer offloading**: CPU offloading of Muon's momentum buffer is not yet supported.
- **Gradient accumulation**: Works correctly with gradient accumulation (momentum is only updated when gradients are present).

## References

- [Muon blog post](https://kellerjordan.github.io/posts/muon/) by Keller Jordan
- [Moonlight (MoonshotAI)](https://github.com/MoonshotAI/Moonlight) — Muon scaling to 1.5B parameters
- [Newton-Schulz iteration](https://arxiv.org/abs/2409.20325) — Bernstein & Newhouse, 2024
