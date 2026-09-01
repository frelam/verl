# MOPD full-vocab 蒸馏 — 现状、问题与待优化点

> 本文档供后续 code agent 实现参考。目标 PR 为 `verl-project/verl` 的
> `#7375 (add opd full-vocab)`，本仓库当前 **尚未合入** 任何 full-vocab 相关代码
> （下文的“现状”均基于该 PR diff / 接口推断，实现时需以最新 upstream 为准）。

## 背景与目标

给 OPD（Offline Policy Distillation）增加 **全词表（full-vocab）KL** 蒸馏能力，并最终
支持 **MOPD（多教师 + 多数据集路由）**。

- 现有 top-k 方案：teacher 只产出 topk/1 个 logprob，KL 在 top-k 子集上近似，有截断与失真。
- full-vocab 方案：teacher 导出 **pre-lm_head 的 hidden states** 到 TransferQueue（TQ）；
  学生端（Megatron 引擎）用教师 **frozen lm_head** 现场重建完整词表 logits，算全词表 KL。
  精度高但开销大，需按 chunk 分片。

## 一、当前现状（PR 已打通的部分）

### 1. Teacher 端（vLLM）导出 hidden states
- 文件：`verl/experimental/teacher_loop/teacher_manager.py`（新增 `compute_teacher_full_vocab_single`）、
  `verl/workers/rollout/vllm_rollout/vllm_async_server.py`（`generate()` 新增 `full_vocab` 参数）。
- prefill-only 前向：`max_tokens=1, prompt_logprobs=0, temperature=1.0`，触发 hidden-state 捕获。
- TQ key：`{teacher_name}/step={step}/sample={uid}`，按 `step` 分组到 partition（便于 `clear_step` 清理）。
- 服务端把 hidden 写 TQ，**只返回 artifact 元数据 dict** 放在 `extra_fields`。

### 2. agent_loop 编排
- 文件：`verl/experimental/agent_loop/agent_loop.py`。
- 显式传递 `global_steps`（agent_loop 路径取每样本 column / 经典 trainer 路径取 `batch.meta_info`）。
- `_resolve_full_vocab_step_uid`：解析 `(step, uid)`，把 `session_id`(rollout-n) 折进 uid 避免同 prompt 不同 rollout 撞 key。
- 每样本有 `routing_key`（来自 dataset 列）：`compute_teacher_full_vocab_single` 用 `_resolve_teacher_key`
  选对应教师导出隐藏态，TQ key 带 `teacher_name`。
- 把 artifact 从 `extra_fields` **pop 到 batch 顶层字段**（`teacher_full_vocab_artifact`），供 Megatron loss 读取。

### 3. 学生端（Megatron）全词表 KL
- 文件：`verl/trainer/distillation/losses.py`、`verl/workers/engine/megatron/transformer_impl.py`。
- logits_processor 重构成方法 `_lm_head_logits_processor`（用 `functools.partial` 绑定），
  `_full_vocab_distillation_outputs` 调 `compute_full_vocab_loss`。
- `_get_teacher_lm_head_shards`：**每个教师**都加载一份 `[V/tp, H]` 的 lm_head 分片，缓存为
  `self._teacher_lm_head_shards: dict[teacher_key, tensor]`。
- 新增 loss：`forward_kl_full_vocab`、`reverse_kl_full_vocab`；top-k 公共收尾抽成
  `_finalize_forward_kl_losses`；加 teacher hidden norm / coverage 诊断指标。
- 限制：**仅支持 megatron 引擎**；**必须 `use_fused_kernels=False`**（fused 绕过 logits processor）。

### 4. 配置
- 文件：`verl/workers/config/distillation.py`。
- `DistillationLossSettings` 新增 `use_full_vocab`（与 `use_topk`/`use_estimator` 三选一互斥校验）。
- 新增：`use_chunked_topk`、`chunked_topk_chunk_size`、`full_vocab_chunk_tokens`、
  `full_vocab_lm_head_checkpoint`、`full_vocab_lm_head_layer`、`full_vocab_experiment_name`。
- 附带：`prefill_context_parallel_size`（PCP）纳入 `per_replica_world_size`、DP 本地实例、replica 对齐校验。

## 二、核心问题（重点，需新 agent 解决）

### P1. 多教师 lm_head 一次性全量加载、不卸载（显存问题）
- **现状**：`_get_teacher_lm_head_shards` 引擎启动时就遍历 `distillation_config.teacher_models`，
  把 **每个教师** 的 `[V/tp,H]` 分片全部加载并常驻 `self._teacher_lm_head_shards`，无 offload。
- **量级**（每 TP rank）：`V·H·2 / tp`。例如 V=152K、H=4096、fp16、tp=8 ⇒
  152000×4096×2/8 ≈ **156MB/教师/rank**，× N 个教师线性增长；尚未计学生全词表 logits 与 chunk log-softmax。
- **期望**：按需加载 + 释放（on-demand / offload），避免 N 个教师时显存爆炸。

### P2. 多数据集路由缺失（MOPD 的灵魂，尚未落地）
- **现状**：路由只依赖 dataset 携带一列 routing key（agent_loop 中 `routing_value`），然后走
  `_resolve_teacher_key`。**数据管线侧**（多个数据集各自声明教师）、**路由信息如何落到 batch meta
  并传到引擎 loss** 均未真正实现。
- **期望**：完整的多数据集路由——数据按数据集分配教师，引擎侧逐样本/逐 token 归一到对应教师参与聚合。
- 该能力对 MOPD 至关重要，是当前最薄的一环。

### P3. 逐 token 的多教师 KL 聚合算法未合入（依赖项）
- 被 import 但 **不在本 PR**、也 **不存在于本仓库** 的配套模块：
  - `verl.trainer.distillation.megatron.full_vocab_kl`（KL 内核：hidden×shard、逐 token 归属、跨 TP 聚合）
  - `verl.workers.rollout.vllm_rollout.full_vocab_hidden_export`（hidden 捕获/导出）
  - `verl.trainer.distillation.full_vocab_tq`（TQ partition 前缀 `resolve_partition_prefix`）
- **期望**：实现/跟进上述模块；`trust_remote_code` 被硬编码 `True` 需复核安全性。

## 三、待优化点（建议的分治实现顺序）

### O1. 教师 lm_head 按需加载 / LRU / 显式释放
- 改造 `_get_teacher_lm_head_shards`：从“启动全量预载”改为“首次访问某教师时加载 + 可配置释放策略
  （LRU / 按步释放 / 显式 `del` + `torch.cuda.empty_cache`）”。
- 建议做成**可配置开关**（如 `full_vocab_offload` / `full_vocab_max_teachers_resident`），不要写死不换。
- 需保证按教师切换时不破坏跨 TP 一致性（各 rank 同步切换）。

### O2. 按教师分 pass 计算（配合 O1 的显存优化）
- 方案 A（重排批）：按路由 index 重排 batch 样本 → 每次只对该教师的子集加载 lm_head 计算。
- 方案 B（更贴合现状，推荐）：**保留 batch 原序**，遍历每个教师，用 mask 标记归属该教师的
  样本/位置，只算该子集的 KL 贡献并写回对应位置，N 个教师累积 N 轮；利用 TQ 已按 `teacher_name`
  分区的特性，逐个教师拉取其 partition 的 hidden。
- 权衡：多 pass 吞吐略降换显存；若 N 小或分片小，全量缓存仍更优——用开关支持两种。

### O3. 多数据集路由落地
- 数据侧：为每个数据集声明教师（`teacher_key`），生成 routing key 列；支持数据混合(MoE/Mixture)式分配。
- 传输侧：确保 routing 信息从数据/agent_loop 进入 batch meta，最终传到 Megatron loss。
- 引擎侧：`full_vocab_kl` 按每 token 的教师归属聚合（forward + reverse KL）。

### O4. 正确性 & 诊断
- 保留/增强 teacher hidden norm、coverage 指标，检出 hidden 导出错位（export misalignment）。
- 复核 `V/tp` 分片维度与 TQ 中 hidden 的 TP 划分是否与学生端一致（避免 gather/allreduce 维度不匹配）。

### O5. 安全/健壮性
- `trust_remote_code=True` 硬编码：合入前需显式确认或改为可配置（涉及 HF bridge 与 vLLM 启动参数）。
- full-vocab 与 `use_fused_kernels=False`、仅 megatron 的限制，启动时主动报错（已有部分）并补充文档。

## 四、参考接口 / 改动落点

| 关注点 | 主要文件 |
| --- | --- |
| 教师导出 hidden | `verl/experimental/teacher_loop/teacher_manager.py`、`teacher_model.py`、`verl/workers/rollout/vllm_rollout/vllm_async_server.py`、`vllm_rollout.py`(replica) |
| agent_loop 编排/路由 | `verl/experimental/agent_loop/agent_loop.py` |
| 学生端优化 | `verl/trainer/distillation/losses.py`、`verl/workers/engine/megatron/transformer_impl.py`、（需）`megatron/full_vocab_kl.py` |
| TQ 分区 | `verl/single_controller/ray/base.py`、（需）`full_vocab_tq.py` |
| 配置 | `verl/workers/config/distillation.py`、`verl/trainer/config/distillation/distillation.yaml` |
| 隐藏态捕获 | （需）`vllm_rollout/full_vocab_hidden_export.py` |

## 五、实现顺序建议
1. 先补 `full_vocab_kl` / `full_vocab_hidden_export` / `full_vocab_tq` 三个缺失模块（P3，前置依赖）。
2. 落地 **O1 + O2**（多教师 lm_head 内存与按教师分 pass），这是当前最明确的显存瓶颈。
3. 落地 **O3 多数据集路由**（P2，MOPD 核心）。
4. 收尾 O4/O5（诊断、trust_remote_code）。