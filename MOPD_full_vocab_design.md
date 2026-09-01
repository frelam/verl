# 全词表 OPD 蒸馏 与 MOPD 多教师蒸馏 — 完整设计文档

> 本文档是全词表（full-vocab）OPD（On-Policy Distillation）与 MOPD（Multi-teacher OPD）
> 在 verl 中的实现设计。设计参考并整合了两个业界 PR 的方案：
>
> - `verl-project/verl#7375`（add opd full-vocab）：teacher 导出 pre-lm_head hidden states
>   到 TransferQueue，学生端 Megatron 引擎用冻结的 teacher lm_head 现场重建全词表 logits。
> - `verl-project/verl#6194`（Nitrobrew）：online-softmax 分块 KL 内核，不物化 `[N, V]`
>   teacher logits，峰值额外显存从 O(N·V) 降到 O(N·C)。
>
> 本设计在此之上补齐：FSDP 后端、reverse KL（含 Megatron TP）、多教师 lm_head 的
> 按需加载 / offload、按教师分批计算、token 数上限控制，并做到 GPU/NPU 硬件无关。

## 0. 背景与目标

### 0.1 问题

- 现有 top-k OPD：teacher 只产出 top-k 个 logprob，KL 在 top-k 子集上近似，存在截断与
  失真（top-k 之外的概率质量被丢弃，KL 可能为负，需要 clamp）。
- 全词表 OPD：对完整词表 V（典型 150K+）计算精确 KL。直接物化 teacher logits
  `[N, V]` 不可行：N=16K tokens、V=152K、fp32 时需 ~9.3GB，且还要乘 log_softmax 等
  中间副本。
- MOPD：多个 teacher 各自服务不同数据集/域名，学生按样本路由到对应 teacher 蒸馏。
  多教师意味着多份 lm_head（每份 V·H·2B，如 152K×4096×2 ≈ 1.2GB），不能全部常驻显存。

### 0.2 目标

1. **全词表 OPD**：
   - 支持 forward KL（`KL(p_T‖p_S)`）与 reverse KL（`KL(p_S‖p_T)`）。
   - 支持 Megatron 与 FSDP 两种训练后端。
   - teacher 侧只导出/存储 **pre-lm_head hidden states**（`[S, H]`），不传输/存储完整
     logits（`H≪V`，传输存储开销降为 H/V ≈ 1/37）。
   - 学生侧在算 KL loss 时从存储读取 hidden states，**动态加载 teacher lm_head**（按
     学生的词表并行方式切分），现场重建 teacher logits 并计算 KL。
   - teacher logits 重建是 stop-gradient 计算，不保留中间状态，及时释放显存。
2. **MOPD 多教师**：
   - 复用 verl 现有的 rollout→teacher 路由（dataset 列 + `teacher_key`）。
   - 学生侧按路由教师把 batch 分组，按教师分批计算 KL。
   - 教师 lm_head 常驻 CPU（pinned memory），按需加载到 GPU/NPU，算完即释放；
     同时常驻 GPU 的教师数可配置（默认 1）。
   - 每次 KL 计算的 token 数有上限（复用 dynamic-bsz / `max_token_len_per_gpu`
     思路，loss 内部再按 `full_vocab_chunk_tokens` 流式分块）。
3. **硬件无关**：不直接使用 `torch.cuda.*`，统一走 `verl.utils.device` 抽象
   （`get_torch_device()` / `get_device_name()`），GPU/NPU 均可运行。

## 1. 总体架构

```
┌──────────────────────────── Rollout 阶段（教师推理）────────────────────────────┐
│ dataset 列 (teacher_key, 默认 data_source)                                      │
│      │                                                                          │
│      ▼                                                                          │
│ agent_loop._compute_teacher_logprobs                                            │
│      │  routing_key = sample[teacher_key]                                       │
│      ▼                                                                          │
│ AsyncTeacherLLMServerManager.compute_teacher_full_vocab_single                  │
│      │  解析 teacher_key → 选教师 client；解析 (step, uid)                       │
│      ▼                                                                          │
│ vLLM teacher server: prefill-only 前向 (max_tokens=1, prompt_logprobs=0)        │
│      │  FullVocabHiddenWorkerExtension: start_hidden_capture → 捕获              │
│      │  pre-lm_head hidden [S, H] → fetch_captured_hidden                       │
│      ▼                                                                          │
│ export_hidden_to_tq: hidden → TransferQueue                                     │
│   partition = full_vocab_hidden_{prefix}_{teacher}_step_{step}                  │
│   key       = {teacher}/step={step}/sample={uid}                                │
│      │  仅 artifact 元数据 dict 随样本返回 batch 顶层                             │
│      ▼                                                                          │
│ batch.non_tensor_batch["teacher_full_vocab_artifact"] = per-sample dict         │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼ 训练阶段（学生引擎，logits processor 内）
┌──────────────────────────── Student engine forward ────────────────────────────┐
│ student_logits [B, T_local, V_shard]  (Megatron: V/tp；FSDP: 全 V，SP 切序列)    │
│      │                                                                          │
│      ▼  compute_full_vocab_loss（losses.py 分发，按 strategy）                   │
│  1. 读 micro-batch 每样本 artifact → 按 teacher_key 分组（MOPD 调度）            │
│  2. for teacher in sorted(groups):              # 所有 rank 顺序一致             │
│       a. TQ 拉取该组样本 hidden → nested [b, j, H]                              │
│       b. lm_head_store.acquire(teacher): CPU pinned → device（按需）             │
│       c. hidden 按学生相同规则做 CP/SP 切分 → 与 student_logits 对齐             │
│       d. 按 token 上限切 sub-batch；每个 sub-batch 内：                          │
│          for token_chunk in split(N, full_vocab_chunk_tokens):                  │
│             online-softmax 分块 KL（vocab 维按 chunk 流式，lm_head 逐 chunk 上卡）│
│       e. 该教师算完 → lm_head_store.release(teacher)（按策略释放 + empty_cache） │
│  3. KL 写回各样本原位 → distillation_losses [B, T_local]                        │
│      ▼                                                                          │
│ no_padding_2_padding → response_mask 聚合 → distillation_loss                   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼ 一步训练完成后
        driver: full_vocab_tq.clear_step(prefix, teacher, step) 释放 TQ 分区
```

### 1.1 关键对齐约定（正确性根基）

- vLLM prefill 的 hidden 位置 `t` 是"看到前缀 `x_≤t` 后、用于预测 `x_{t+1}`"的表示；
  学生 logits 位置 `t` 同样是预测 `x_{t+1}` 的分布。因此 **hidden 与 student logits 逐位
  对齐，无需 shift**（与现有 top-k 路径的 `teacher_logprobs` 对齐方式一致；
  `no_padding_2_padding` 取 `[seq_offset-resp_len-1 : seq_offset-1]` 的左移已由该函数
  统一处理）。
- 每个样本最后一个位置的 hidden/logits 没有监督目标（结构性的最后一行），
  loss 由 `response_mask` 天然屏蔽。
- 教师与学生必须同 tokenizer / 同词表（OPD 前提）。V_real 到学生 padded vocab
  （Megatron `make_vocab_size_divisible_by`）之间的 pad 行，teacher lm_head 补零行
  （`z_t=0`），保证 KL 有界（reverse KL 不允许 `log p_T = -inf` 而 `p_S>0`）。

## 2. 数学与 KL 内核设计

### 2.1 分块 online-softmax（不物化 `[N, V]`）

记 token 数 N，词表 V，词表块大小 C。teacher logit `z_t(v) = h · W[v]`，
student logit `z_s(v)`。所有计算在 fp32 累加器中进行（数值稳定），输入可为 bf16。

**Forward KL**：`KL(p_T‖p_S) = Σ_v p_T·(log p_T − log p_S)`
按词表块流式维护（单遍 online softmax）：

```
m   = running max of z_t
S0  = Σ e^{z_t − m}
T1  = Σ e^{z_t − m} · z_t
U   = Σ e^{z_t − m} · z_s          （可选 log_prob_min_clamp 下 clamp z_s）
lse_T = m + log S0
KL  = T1/S0 − lse_T − U/S0 + lse_S
```

**Reverse KL**：`KL(p_S‖p_T) = Σ_v p_S·(log p_S − log p_T)`

```
先算 lse_S（student 全词表 logsumexp，Megatron 下走 TP allreduce，见 §2.2）
m, S0 同上（teacher 的 online softmax）
E   = Σ p_S · z_s
U'  = Σ p_S · z_t
KL  = E − U' − lse_S + lse_T
```

**反向（闭式梯度，stop-gradient：z、W 无梯度，只对 student logits 回传）**：

```
forward KL:  ∂KL/∂z_s(v) = p_S(v) − p_T(v)
reverse KL:  ∂KL/∂z_s(v) = p_S(v)·( log p_S(v) − log p_T(v) − KL )
```

反向时**重新分块计算** `z_t = h·W`（第二遍重计算），不保存任何 `[N, V]` 中间量；
autograd 只需保存 `h [N, H]`（小）、`t_lse / s_lse / KL [N]`、以及 lm_head 的
**CPU 引用**（反向逐 chunk 重新上卡）。这满足"stop gradient、不保留中间状态、
及时释放"的要求。

**数值与稳定性**：
- 全 fp32 累加；输出 KL 再 cast 回 student dtype。
- `log_prob_min_clamp`（沿用现有配置）对 student log-prob 下钳，防止 `p→0` 时
  `log q − log p` 爆炸；clamp 后 KL 可能略负，最终在 loss finalize 统一 `clamp_min(0)`。
- `kd_temperature`（新增配置）对两侧 logits 同除温度。

### 2.2 分布式聚合

**Megatron（vocab-parallel，TP 切词表）**：
- 每 TP rank 持有 student logits 的 `[N, V/tp]` 与 teacher lm_head 分片 `[V/tp, H]`
  （切分区间与学生 `VocabUtility.vocab_range_from_per_partition_vocab_size` 一致，
  天然分布式，无需跨 rank 交换 logits）。
- `lse_S`：标准 Megatron vocab-parallel logsumexp（max 做 `all_reduce(MAX)`、
  sum-exp 做 `all_reduce(SUM)`）。
- teacher online-softmax 累加器 `(m, S0, T1, U[, E, U'])`：各 rank 本地分块累加后，
  用 all-gather 精确合并：`m_g = max_r m_r`，`Sx_g = Σ_r Sx_r · e^{m_r − m_g}`。
  （reverse KL 的 `E/U'` 是加性量，`all_reduce(SUM)`；`lse_T` 由合并后的 `m_g/S0_g` 得。）
- KL 每 rank 得到相同的全局标量序列 `[N]`，梯度只回传本地 `z_s` 分片，通信量 O(N·tp)。

**FSDP（不切词表；Ulysses SP 切序列）**：
- 每 rank 持有全 V 的 student logits（SP>1 时为本有序列片段）。
- teacher hidden 与学生相同的 SP 规则切序列（`slice_input_tensor(dim=1)`）。
- lm_head 为全 `[V, H]`：常驻 CPU pinned，**逐词表块 `W[v:v+C].to(device)`** 上卡，
  GPU 峰值只有 O(C·H)。SP 组内无需额外通信（各 rank 序列段独立）。

**CP（Megatron context parallel）**：teacher hidden 复用
`preprocess_thd_engine` / `preprocess_bshd_engine`，按与 `input_ids` 完全相同的
zigzag/contiguous 规则切分，保证与学生 logits 的 token 对齐。

### 2.3 融合算子（可选性能项）

首版内核为 PyTorch 实现，逐 chunk 的更新函数用 `@torch.compile`（环境不支持时自动
退化为 eager）。接口（`_fwd_chunk_update` / `_bwd_chunk` 等）已按"纯张量进、
纯张量出"组织，后续可无缝替换为 Triton 融合内核（单 kernel 完成
`z@W.T → online-softmax 累加`），属性能优化而非功能依赖。

## 3. Teacher 侧设计（hidden state 获取与存储）

### 3.1 捕获（vLLM teacher server）

- 新增 `verl/workers/rollout/vllm_rollout/full_vocab_hidden_export.py`：
  - `FullVocabHiddenWorkerExtension(vLLMColocateWorkerExtension)`：为每个 vLLM worker
    增加两个 RPC：
    - `start_hidden_capture()`：在模型的 `LogitsProcessor` 上注册 forward pre-hook，
      捕获其 `hidden_states` 输入（pre-lm_head hidden，`detach()` 后挂到模块属性）。
      捕获点是 LogitsProcessor 而非具体模型层，因此对模型结构通用。
    - `fetch_captured_hidden()`：取回 buffer 并清空。
  - `unwrap_captured_hidden(rpc_result)`：`collective_rpc` 返回多 worker 结果，
    取第一个非 None（TP 下各 rank 捕获的是同一份完整 hidden；`max_num_seqs=1`
    保证一个 engine 一次只有一个请求）。
  - `export_hidden_to_tq(hidden, seq_len, teacher_name, step, uid, prefix)`：
    校验长度后写入 TQ，返回 artifact dict。
- `vllm_async_server.generate()` 新增 `full_vocab: dict | None` 参数：
  - 仅当 server 以 `full_vocab_export_config={"enabled", "prefix"}` 启动（teacher）
    才接受；否则报错。
  - 捕获→前向→取回全程由 `asyncio.Lock` 串行化（捕获 buffer 挂在模型上、全 server
    共享；引擎本身 `max_num_seqs=1`）。
  - full-vocab 请求跳过 `extract_prompt_logprobs`（监督信号在 hidden，不在 logprobs）。
- 教师采样参数：`max_tokens=1, temperature=1.0, prompt_logprobs=0`——
  `prompt_logprobs=0` 足以让 vLLM 对每个 prompt 位置计算 logits，从而触发捕获。
- 教师 inference 强校验（配置期）：`max_num_seqs=1`、`enable_chunked_prefill=False`、
  `max_num_batched_tokens >= max_model_len`（v1 调度器仍可能切分超长 prefill，
  捕获只保留最后一次前向，切片会静默丢 hidden）。

### 3.2 存储（TransferQueue）

新增 `verl/trainer/distillation/full_vocab_tq.py`：

- 分区命名：`full_vocab_hidden_{prefix}_{teacher}_step_{step}`；key：
  `{teacher}/step={step}/sample={uid}`。`prefix` 隔离共享 TQ 集群的不同实验
  （`full_vocab_experiment_name` → env `VERL_FULL_VOCAB_EXPERIMENT_NAME` →
  `"default_exp"`）。
- hidden 以 CPU 张量（bf16）存入 TQ（TQ storage 在 host 内存/盘，不占 GPU）。
- artifact dict（随 batch 走的轻量元数据）：
  `{teacher_name, step, uid, key, partition_id, seq_len, shape, dtype}`。
- API：
  - `put_hidden(...)` / `fetch_hidden(keys, partition_id) -> list[Tensor]`（学生侧按需拉取）；
  - `clear_step(prefix, teacher_name, step)`：训练步结束后 driver 清理该步分区
    （`kv_list` → `kv_clear`），best-effort，失败仅告警；
  - `resolve_partition_prefix(name)`。
- `transfer_queue` 为可选依赖：模块顶层不 import，函数内延迟 import，缺失时报
  配置错误（full-vocab 模式下 TQ 是硬性依赖）。

### 3.3 编排（agent_loop / teacher_manager）

- `teacher_manager.AsyncTeacherLLMServerManager.compute_teacher_full_vocab_single(
  sequence_ids, *, step, uid, routing_key, multi_modal_data, mm_processor_kwargs)`：
  解析教师→发 prefill-only 请求（带 `full_vocab={"teacher_name","step","uid"}`）→
  返回 artifact；无 artifact 直接 raise（静默丢失会静默关掉该样本的蒸馏）。
  `step=None` 时回退 manager 内单调计数器（v0 路径无 global_steps）；
  `uid` 内部拼随机后缀防同 key 覆盖。
- `agent_loop`：
  - `global_steps` 显式透传（不混入 `**kwargs`，避免具体 agent loop 拒收未知参数）；
    `_resolve_full_vocab_step_uid` 把 `session_id`（rollout-n）折进 uid 防撞 key。
  - `loss_settings.use_full_vocab` 时走 `compute_teacher_full_vocab_single`，
    artifact 写入 `extra_fields["teacher_full_vocab_artifact"]`，`as_dict` 时 pop 到
    batch 顶层（成为 NonTensorStack 列，随 micro-batch 切分）。
  - **routing 复用**：`routing_key = sample[teacher_key]`（现有机制原样保留），
    并随 artifact 进入引擎——这就是 MOPD 引擎侧分组的依据。
- TQ 清理：trainer（`ray_trainer.py` legacy 与 `trainer_base.py` v1）在
  `_update_actor` 完成后调 `clear_step`。

## 4. Student 侧设计（teacher logits 重计算）

### 4.1 lm_head 加载与切分（与学生并行方式一致）

新增 `verl/trainer/distillation/full_vocab_kl.py`（公共部分）：

- `load_teacher_lm_head(checkpoint_path, layer="auto") -> CPU Tensor [V, H]`：
  直读 safetensors（`model.safetensors[.index.json]`），避免为读一个矩阵构建完整
  HF 模型；`layer="auto"` 按 `tie_word_embeddings` 选 `embed_tokens` 或 `lm_head`；
  支持 `full_vocab_lm_head_checkpoint` 覆盖（默认用教师 `model_path`）；
  路径经 `copy_to_local`（HDFS 友好）。
- `shard_lm_head(weight, vocab_size_padded, tp_rank, tp_size) -> [V_padded/tp, H]`：
  按学生 Megatron 的词表区间切行，不足补零行（见 §1.1）。FSDP 时 `tp_size=1`。
- `TeacherLmHeadStore`（每引擎进程一个）：
  - 初始化（lazy，首次用时）：遍历 `teacher_models`，读 safetensors → 切分 →
    按常驻策略存放。**所有教师的 lm_head 分片默认常驻 CPU pinned memory**
    （满足"多教师 lm_head 存放于内存，需要时才加载到 GPU/NPU"）。
  - 常驻策略 `full_vocab_lm_head_residency`：
    - `"cpu"`（默认，MOPD 推荐）：分片常驻 CPU pinned；前向/反向逐词表块
      `.to(device, non_blocking=True)`，GPU 峰值 O(C·H)。
    - `"gpu"`：分片常驻 GPU（单教师/显存充裕时最快速度）；受
      `full_vocab_max_resident_teachers`（默认 1）限制，超出时 LRU 逐出回 CPU。
  - `acquire(teacher_key, device)` / `release(teacher_key)`：
    上下文管理；release 时按策略把 GPU 副本 `del` 并
    `get_torch_device().empty_cache()`（跨 TP rank 调用顺序一致，无挂起风险）。
  - 反向支持：autograd 保存的是 store 中 CPU 分片的**引用**，反向重新逐 chunk
    上卡（重计算），不占 GPU 常驻。

### 4.2 KL 计算入口与多教师调度（MOPD 核心）

`compute_full_vocab_loss(config, distillation_config, data, student_logits,
data_format, lm_head_store)`（`losses.py`，按 `config.strategy` 分发到
`megatron/full_vocab_kl.py` 或 `fsdp/full_vocab_kl.py`）：

```
artifacts = data["teacher_full_vocab_artifact"]           # per-sample dict
groups = 按 artifact["teacher_name"] 分组样本索引          # 保持 batch 原序
losses = empty[B, T_local]
for teacher_key in sorted(groups):                        # 所有 rank 遍历顺序一致
    idx = groups[teacher_key]
    hidden = full_vocab_tq.fetch_hidden(...)              # → nested [len(idx), j, H]
    hidden = 按学生规则做 CP/SP 切分
    with lm_head_store.acquire(teacher_key, device):
        for sub in 按 full_vocab_max_tokens_per_pass 切 token 子批:
            losses[idx[sub]] = FullVocabKL.apply(student_logits[sub], hidden[sub], ...)
    # release：多教师时逐教师释放 GPU 副本
losses[无 artifact 或 validate 样本] = 0
```

- **按教师分 pass**（需求 2.1/2.2 的落地）：保留 batch 原序、按 mask 写回（而不是
  重排 batch），避免破坏 megatron 的 micro-batch 结构；每个 pass 只有 1 个教师的
  lm_head 在 GPU（`full_vocab_max_resident_teachers=1`），算完即释放。
- **token 上限**（需求 2.3）：两级控制——
  1. micro-batch 级：沿用现有 dynamic-bsz（`ppo_max_token_len_per_gpu`）；
  2. loss 内部：展平 `[B·T]` 后按 `full_vocab_chunk_tokens` 流式分块（默认 4096），
     峰值 `[chunk, C]` 缓冲与 batch 长度解耦。
- **单教师退化为一次 pass**，与 PR#7375 行为一致。
- 显存估算（V=152K, H=4096, bf16, tp=8, chunk_tokens=4096, C=8192）：
  - lm_head：GPU 常驻 156MB/教师（`gpu` 模式）或 CPU pinned 156MB/教师 +
    GPU chunk `8192×4096×4B=128MB`（`cpu` 模式）；
  - online-softmax 中间量：约 `3 × chunk_tokens × C × 4B ≈ 400MB`；
  - hidden：`chunk_tokens×4096×2B = 32MB`；
  - 反向重计算与正向同量级，无 `[N, V]` 级别分配。

### 4.3 引擎集成点

- **Megatron**（`engine/megatron/transformer_impl.py`）：
  `_lm_head_logits_processor` 增加 `distillation_use_full_vocab` 分支，
  调 `compute_full_vocab_loss`；store 由引擎 lazy 构建并缓存
  （`self._teacher_lm_head_store`）。`use_fused_kernels=True` 时直接报错
  （fused 绕过 logits processor）。
- **FSDP**（`engine/fsdp/transformer_impl.py`）：`prepare_model_outputs` 两个
  padding 分支在现有 `distillation_use_topk` 处理处增加 full-vocab 分支；
  SP>1 时 hidden 同步切序列。store 同样由引擎持有。
- `distillation_only`（跳过 policy log_probs 省显存）判定从 `use_topk` 扩展到
  `use_topk or use_full_vocab`。

### 4.4 loss 注册与 finalize（`trainer/distillation/losses.py`）

- `DistillationLossSettings` 新增 `use_full_vocab`，与 `use_topk/use_estimator`
  三选一互斥。
- 注册 `forward_kl_full_vocab`、`reverse_kl_full_vocab`（`use_full_vocab=True`）：
  finalize 复用 top-k 的公共收尾（padding 还原、response mask、`clamp_min(0)`），
  并输出诊断指标：
  - `distillation/teacher_hidden_norm(_min/_max)`：response 区 teacher hidden 范数，
    用于发现 hidden 导出错位/缺失；
  - `distillation/teacher_hidden_coverage`：范数非零位置占比（健康运行为 1）。
- `compute_topk_loss` 重命名泛化为 logits-processor 分发入口（保留 top-k 行为不变）。

## 5. 配置设计

`DistillationLossConfig` 新增（`verl/workers/config/distillation.py` +
`trainer/config/distillation/distillation.yaml`）：

| 配置 | 默认 | 说明 |
| --- | --- | --- |
| `loss_mode` | — | 新增 `forward_kl_full_vocab` / `reverse_kl_full_vocab` |
| `full_vocab_chunk_tokens` | 4096 | loss 内 token 维流式块大小（峰值显存主控） |
| `full_vocab_chunk_vocab` | 8192 | 词表维流式块大小 C |
| `full_vocab_lm_head_checkpoint` | null | lm_head 来源 ckpt；null=教师 `model_path` |
| `full_vocab_lm_head_layer` | `"auto"` | 权重名；auto 按 tie_word_embeddings 推断 |
| `full_vocab_experiment_name` | null | TQ 分区前缀（env 回退，再 `default_exp`） |
| `full_vocab_lm_head_residency` | `"cpu"` | `cpu`（pinned，按需上卡）/ `gpu`（常驻） |
| `full_vocab_max_resident_teachers` | 1 | `gpu` 模式下同时常驻 GPU 的教师数（LRU） |
| `full_vocab_max_tokens_per_pass` | null | 单个教师单次 pass 的最大 token 数；null=不限制（由 micro-batch 与 chunk_tokens 控制） |
| `kd_temperature` | 1.0 | KL 温度（两侧 logits 同除） |
| `log_prob_min_clamp` | -10.0 | （已有）student log-prob 下钳 |

校验：
- `use_full_vocab` 时 `full_vocab_chunk_tokens>0`、`full_vocab_chunk_vocab>0`；
- 每个教师 inference：`max_num_seqs=1`、`enable_chunked_prefill=False`、
  `max_num_batched_tokens>=max_model_len`（见 §3.1）；
- 引擎侧：full-vocab + `use_fused_kernels=True` 启动即报错；
- 教师后端：hidden 导出仅实现 vLLM（其他后端启动即报错）。

## 6. 文件清单

**新增**：
| 文件 | 内容 |
| --- | --- |
| `verl/trainer/distillation/full_vocab_tq.py` | TQ 分区命名、put/fetch/clear_step、artifact |
| `verl/trainer/distillation/full_vocab_kl.py` | 公共：online-softmax KL 内核（fwd/rev，autograd）、lm_head safetensors 加载/切分、`TeacherLmHeadStore` |
| `verl/trainer/distillation/megatron/full_vocab_kl.py` | TP 精确合并、CP 切分、多教师分 pass 调度（Megatron 入口） |
| `verl/trainer/distillation/fsdp/full_vocab_kl.py` | SP 切分、多教师分 pass 调度（FSDP 入口） |
| `verl/workers/rollout/vllm_rollout/full_vocab_hidden_export.py` | vLLM 捕获扩展 + TQ 导出 |
| `tests/trainer/distillation/test_full_vocab_kl_on_cpu.py` | CPU 正确性：分块 vs 朴素 fwd/rev、反向梯度、TP 合并（gloo 双进程）、多教师调度、lm_head 加载/store |

**修改**：
| 文件 | 改动 |
| --- | --- |
| `verl/trainer/distillation/losses.py` | `use_full_vocab`、注册两个 loss、`compute_full_vocab_loss` 分发、公共 finalize |
| `verl/workers/config/distillation.py` | §5 配置与校验 |
| `verl/trainer/config/distillation/distillation.yaml` | 配置项与注释 |
| `verl/experimental/teacher_loop/teacher_manager.py` | `compute_teacher_full_vocab_single`、full-vocab 采样参数 |
| `verl/experimental/teacher_loop/teacher_model.py` | `full_vocab_export_config` 派生与透传 |
| `verl/experimental/agent_loop/agent_loop.py` | global_steps 透传、`_resolve_full_vocab_step_uid`、full-vocab 分支、artifact 置顶 |
| `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | `full_vocab` 参数、捕获串行化、`_export_full_vocab_hidden`、worker extension 切换 |
| `verl/workers/rollout/vllm_rollout/utils.py` | `extract_prompt_logprobs` 的 None 防护 |
| `verl/workers/rollout/replica.py` | `full_vocab_export_config` 透传 |
| `verl/workers/engine/megatron/transformer_impl.py` | logits processor full-vocab 分支、store 挂载、fused 报错 |
| `verl/workers/engine/fsdp/transformer_impl.py` | 两个 padding 分支的 full-vocab 处理、store 挂载 |
| `verl/workers/engine_workers.py` | full-vocab 策略检查、loss_fn 透传 |
| `verl/trainer/ppo/ray_trainer.py`、`verl/trainer/ppo/v1/trainer_base.py` | `distillation_use_full_vocab` extra_info、`distillation_only` 扩展、TQ `clear_step` |

## 7. 正确性验证计划

1. **内核数值**：分块 online-softmax fwd/rev KL vs 物化 `[N,V]` 的朴素实现
   （atol 1e-5）；温度 ≠1 同测；反向梯度 vs autograd 朴素参考。
2. **TP 合并**：gloo 双进程模拟 vocab 切分，合并结果 == 单进程全词表。
3. **多教师调度**：两组样本混合 batch，分组计算结果 == 逐教师单独计算；
   无 artifact 样本 loss=0；教师遍历顺序跨 rank 一致。
4. **lm_head store**：加载（tied/untied/分片 ckpt）、切分零行填充、acquire/release
   后 GPU 副本释放、LRU 上限。
5. **对齐**：构造已知分布的 toy 样本，验证 hidden[t] 与 logits[t] 对齐
   （错位一位时 KL 显著变大，作为回归哨兵）。
6. 静态检查：`ruff` + 受影响文件编译。

## 8. 限制与约束

- 教师/学生必须同 tokenizer；词表不同的多教师组合不支持（启动校验）。
- 教师 hidden 导出仅支持 vLLM 后端；训练后端支持 Megatron / FSDP（VeOmni 继承
  FSDP 路径）。
- Megatron 下要求 `use_fused_kernels=False`；`pad_to_length` 与蒸馏互斥（已有约束）。
- 教师推理要求 `max_num_seqs=1`、关闭 chunked prefill、`max_num_batched_tokens >=
  max_model_len`。
- 教师侧多副本/GPU 共享（PR#7375 的 `share_gpu_group`/sleep controller）不在本期
  范围；MOPD 的显存优化聚焦学生侧 lm_head 调度（本设计 §4）。
- hidden states 每步生命周期为"rollout 导出 → 该步训练消费 → clear_step"，
  多 epoch（`ppo_epochs>1`）时同一批 hidden 在前向/反向中被重复消费，TQ 清理
  发生在整步结束之后，语义安全。

## 9. 硬件无关性

- 设备上下文一律取 `tensor.device`；清缓存用
  `verl.utils.device.get_torch_device().empty_cache()`；不引用 `torch.cuda`。
- pinned memory 通过 `tensor.pin_memory()`（NPU 上 PyTorch 同样支持；不支持时
  退化为普通 CPU 张量，仅损失拷贝性能）。
- 分布式通信只使用 `torch.distributed` 标准原语与 Megatron `mpu` 的 process
  group（NCCL/HCCL 由上层初始化决定）。
