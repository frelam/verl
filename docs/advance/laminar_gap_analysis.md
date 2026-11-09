# Laminar 论文与当前代码实现差距分析

> 论文：Laminar: A Scalable Asynchronous RL Post-Training Framework
> （arXiv:2510.12633，EuroSys '26，https://arxiv.org/abs/2510.12633）
>
> 代码基线：verl V1 async trainer（`trainer.v1.trainer_mode` = `sync` / `colocate_async` / `separate_async`）
>
> 文档目的：逐条列出论文设计点与当前实现的差距，供逐条决策，作为后续实现的输入。

---

## 1. 结论摘要

当前代码实现了 Laminar 的**轨迹级异步与数据面解耦骨架**，但只完成了一部分：

| 论文机制 | 当前代码 | 完成度 |
| --- | --- | --- |
| 轨迹级异步（§2.4、§3） | TransferQueue + `ReplayBufferAsync` + `FullyAsyncLLMServerClient` | 基本完成 |
| 分层权重同步 / relay workers（§4） | `verl/checkpoint_engine/` 的 relay ranks、NIXL/Mooncake ring | 部分完成 |
| Dynamic repack（§5） | 无 | 未实现 |
| 单轨迹单策略版本（§2.3、§9） | 允许 partial rollout 跨版本续跑 | 已决策实现，见 A1 |
| 不设显式 staleness bound（§6） | `max_off_policy_threshold=8` + `drop`/`wait` | 保留硬阈值，新增 `observe` 不截断模式 |
| 容错与恢复（§3.3） | TQ checkpoint + 重新下发 prompt；无心跳/租约 | 部分完成 |

因此更准确的定位是 **Laminar-inspired**，而非 Laminar 的完整复刻。

---

## 2. 论文机制与代码锚点

| 论文位置 | 机制 | 代码锚点 |
| --- | --- | --- |
| §3.1 | Rollout Manager + Rollouts | `LLMServerManager`、`GlobalRequestLoadBalancer`、`AgentLoopManagerTQ` |
| §3.1 | Prompt Pool | dataloader `_next_train_batch` + TQ 的 `{uid}` key |
| §3.1 | Partial Response Pool | 无中心化实现；在飞 token 存在 worker 的 `AgentData` 内存中 |
| §3.1 | Experience Buffer（writer + sampler） | TQ 轨迹 key + `_agent_loop_postprocess`（writer）+ `ReplayBuffer`（sampler） |
| §3.1、§4 | Relay Workers | `verl/checkpoint_engine/`；`CheckpointEngineWithCache` 注释直接引用 Laminar |
| §3.2 ①~⑧ | 训练工作流 | 见下方调用链 |
| §5 | Dynamic Repack | 无对应实现 |
| §6 | 固有 staleness `M - K` | `min/max_global_steps` → `training/off_policy/trajectory_staleness/*` |

论文 §3.2 的八步工作流与代码的对应关系：

| 论文步骤 | 代码位置 |
| --- | --- |
| ① rollout 取 prompt | `_next_train_batch` → `_submit_batch_to_rollout`（TQ 写 `pending`）→ `generate_sequences` |
| ② 在飞轨迹写入 partial response pool | 缺失；等价物是 `AgentData.response_ids` 在 worker 内存中累积 |
| ③ 完成轨迹进 experience buffer | `_agent_loop_postprocess` → `tq.async_kv_batch_put`，打 `global_steps` / `min_global_steps` / `max_global_steps` |
| ④ trainer 从 buffer 采样 | `ReplayBufferAsync.sample` → `wait_for_sampleable` → 淘汰 / 补充 / 选择 |
| ⑤ actor 推权重给 master relay 后继续训练 | `checkpoint_manager.update_weights` → `CheckpointEngineWorker.update_weights`（`blocking=False`） |
| ⑥ master relay 链式广播 | `NCCLCheckpointEngine._relay_weights`；NIXL/Mooncake 的 ring p2p |
| ⑦ rollout 从本地 relay 取权重 | `receive_weights` + `server_adapter.update_weights`（cuda ipc） |
| ⑧ repack 长尾轨迹 | 缺失 |

---

## 3. 差距总览

| ID | 差距 | 论文 | 优先级 | 工作量 | 建议决策 |
| --- | --- | --- | --- | --- | --- |
| A1 | 单轨迹跨多个策略版本（partial rollout） | §2.3、§9 | P0 | M | 实现 |
| A2 | 有显式 staleness bound，论文主张不设 | §6 | P0 | S | 实现 |
| A3 | 无下发预算与异步深度上限（容量无界） | §3.1 | P1 | M | 实现 |
| A4 | prompt 由 trainer push，而非 rollout 主动 pull | §3.1、§3.2① | P2 | L | 延后 |
| A5 | 无 per-rollout 并发上限 | §8 A.2 | P1 | M | 实现 |
| A6 | 采样策略仅 FIFO（可插拔已支持） | §3.1、§6 | P2 | M | 实现 |
| B1 | 无 CPU relay（pinned shm）+ 链式广播 | §4.2 | P2 | L | 延后 |
| B2 | 无 anytime pull（完成 batch 即拉最新权重） | §4.2⑦ | P1 | L | 实现 |
| B3 | 无 master 选举 / 链重建 / 无超时故障检测 | §4.3 | P2 | L | 延后 |
| B4 | `CheckpointEngineWithCache.get_weights` 是空接口 | §4.2 | P1 | M | 实现 |
| B5 | 无全局权重版本注册表 | §4.2、§6 | P1 | S | 实现 |
| C1 | 无 per-replica KVCache 利用率采集 | §5.2 | P1 | M | 实现 |
| C2 | 无 idleness 判定 | §5.2 | P1 | S | 实现 |
| C3 | 无 Best-Fit 装箱（Algorithm 1） | §5.2 | P1 | M | 实现 |
| C4 | 无轨迹迁移执行（中止 + re-prefill） | §5.1③ | P2 | XL | 实现 |
| C5 | 无 repack 触发时机 | §5.1 | P1 | S | 实现 |
| D1 | 无中心化 partial response pool | §3.1、§3.2② | P2 | XL | 延后 |
| D2 | `running` 无租约/超时，故障会永久卡住 | §3.3 前置 | P0 | S | 延后 |
| D3 | 故障 rollout 的在飞轨迹不重定向 | §3.3 | P2 | L | 延后 |
| D4 | relay 故障不检测/不恢复 | §4.3 | P2 | L | 延后 |
| D5 | elastic add/remove 与 LB/TQ 一致性 | §3.3 | P1 | M | 延后 |
| E1 | 缺 per-rollout KV / idle / repack 指标 | §8.4 | P1 | S | 实现 |
| E2 | 收敛性保护（rollout correction） | §8.2 | P1 | S | 确认即可 |
| F1 | 跨引擎（vLLM / SGLang / TRT-LLM）约束未定义 | §7 | P0 | S | 实现 |
| F2 | 配置项与默认值缺失 | §5.1、§6 | P0 | S | 实现 |
| F3 | 无 repack / relay 单测与 benchmark | §8.4 | P1 | M | 实现 |

工作量说明：S（< 1 天）、M（1~3 天）、L（1~2 周）、XL（跨模块，需引擎支持）。

---

## 4. 详细条目

### A. 数据面与调度

#### A1. 单轨迹单版本（关闭 partial rollout resume）

- **论文依据**：§2.3 指出 partial rollout 有重 prefill 开销和单轨迹混合策略版本两个代价；§9 明确 Laminar 保证 "each trajectory is generated with a single, consistent policy version"。
- **现状**：`FullyAsyncLLMServerClient.generate` 对 V1 恒 `should_retry=True`，abort 后 resume；`trajectory_spans = max - min + 1` 可大于 1。
- **差距**：代码允许跨版本，论文要求单版本。
- **结论**：该项目标明确要求"不能跨版本"，本项按"实现"处理。
- **实现方案（已决策）**：
  1. 配置：新增 `trainer.v1.async_trajectory.single_version_per_trajectory`，默认 `False` 以保持对旧配置的兼容。
  2. 判定位置：`FullyAsyncLLMServerClient.generate` 在一次 `super().generate()` 返回后检查 `stop_reason in ("aborted", "abort")`。开关为 `True` 时立即抛出 `TrajectoryAbortedError`，不合并部分结果、不进入下一轮 resume。
  3. 异常类型：在 `verl/workers/rollout/replica.py` 新增 `TrajectoryAbortedError`，携带 `request_id` 与已生成 token 数，便于日志和指标。
  4. 失败传播：`AgentLoopWorkerTQ._run_prompt` 识别该异常为预期终止，整组标 `failure`；`ReplayBufferAsync` 的 failure 淘汰路径会 evict 并 refill，保证训练 batch 规模稳定。
  5. 兼容旧模式：旧 `fully_async_policy` 的配置没有 `trainer.v1` 节，读取不到时回退 `False`，继续由 `async_training.partial_rollout` 控制。
  6. 多轮轨迹：任一 turn 被 abort 即整条轨迹失败，避免出现"前几轮旧版本、后几轮新版本"的拼接。
  7. 观测：`training/off_policy/trajectory_spans/max` 应恒为 1；可选新增 `rollout/aborted_discarded` 计数。
- **影响面**：`verl/workers/rollout/llm_server.py`、`verl/workers/rollout/replica.py`、`verl/trainer/ppo/v1/agent_loop_tq.py`、`verl/trainer/config/ppo_trainer.yaml`。
- **风险与代价**：`colocate_async` 必须在采样后 abort 以释放显存，开启该开关后长尾轨迹会被整条丢弃并重采，样本利用率下降；需要 A5（并发上限）与 A3/C 系列补偿。`separate_async` 的 standalone rollout 不受 abort 影响，代价较小。
- **验收标准**：
  - `colocate_async` 下 `training/off_policy/trajectory_spans/max == 1`。
  - 人为触发一次 abort，日志出现 `Discarded aborted trajectory`，对应 group 状态变为 `failure` 并触发 refill。
  - 关闭开关时行为与当前实现完全一致。
- **决策**：
  - [x] 实现
  - [ ] 跳过
  - [ ] 延后

#### A2. staleness 控制：保留硬阈值，新增不截断模式

- **论文依据**：§6 "does not require an explicitly configured staleness bound"，实测 staleness 通常小于 3、最大 4。
- **现状**：`max_off_policy_threshold=8` + `max_off_policy_strategy=drop|wait`，超过即淘汰或阻塞。
- **差距**：论文让 staleness 自然涌现，代码用硬阈值截断。
- **结论**：硬阈值（`drop`/`wait`）可以接受并保留；同时新增一个不做硬阈值截断的模式，供需要贴近论文行为的场景使用。
- **实现方案（已决策）**：
  1. 扩展现有 `trainer.v1.sampler.max_off_policy_strategy`，取值增加 `observe`；默认仍为 `drop`，`drop`/`wait` 行为完全不变。
  2. `observe` 的语义：
     - 不做 staleness 淘汰：`_stale_terminal_keys` 返回空集，staleness 相关的 `evicted_samples` 不再增长；
     - 不做 staleness 阻塞：`_has_enough_samples` 的阻塞逻辑只对 `wait` 生效；
     - DAPO 过滤与 failure 淘汰不受影响，仍会 evict + refill。
  3. `max_off_policy_threshold` 在 `observe` 下退化为"观测阈值"：继续用于 `trajectory_staleness` 指标，并新增超阈值计数（如 `training/off_policy/over_threshold_groups`）与告警，不参与淘汰决策。
  4. `ReplayBufferAsync.sample` 中"选中样本 span <= threshold"的断言保持只在 `drop` 下执行（现有实现已按此门控）。
  5. 配置校验放宽为 `max_off_policy_strategy in {"drop", "wait", "observe"}`。
- **影响面**：`verl/trainer/ppo/v1/replay_buffer.py` 的 `__init__` 校验、`_stale_terminal_keys`、`_has_enough_samples`、`sample` 指标；`verl/trainer/config/ppo_trainer.yaml` 的注释说明。
- **风险与代价**：`observe` 下 staleness 无上界，可能影响收敛；用 `over_threshold_groups` 告警 + A3（容量淘汰）+ rollout correction 兜底。
- **验收标准**：
  - `observe` 下 staleness 相关 `evicted_samples == 0`，DAPO/failure 淘汰仍正常触发；
  - `trajectory_staleness` 可以超过 `max_off_policy_threshold`，且 `over_threshold_groups` 计数正确；
  - `drop`/`wait` 行为与当前实现完全一致。
- **决策**：
  - [x] 实现
  - [ ] 跳过
  - [ ] 延后

#### A3. 下发预算与容量溢出（异步深度控制）

- **论文依据**：§3.1 "eviction strategy for removing old experiences when buffer capacity is not enough"；§3.2① rollout 自主从 prompt pool 取数据，缓冲区需要独立的容量边界。
- **现状**：下发量由 trainer 的 step 速率决定，而不是 staleness：`prepare_step()` 每步下发 `train_batch_size` 个 prompt，`step()` 每步消费同量，稳态 outstanding 约为 `B × (1 + num_warmup_batches)`。`max_off_policy_threshold` 只控制淘汰，不控制异步深度。未来实现 A4 或主动加深管道后，TQ 可能无界增长。
- **差距**：没有显式的异步深度上限，也没有容量溢出策略；staleness 不是下发量的控制器。
- **结论**：新增独立的“下发预算 + 容量溢出”机制，与 staleness 解耦；`observe` 模式下它是唯一流控，`drop`/`wait` 下是安全网。
- **实现方案（已决策）**：
  1. 新增 `trainer.v1.async_trajectory.max_outstanding_groups`，默认 `null` 表示不限制；统计 `pending + running + finished` 的 prompt group 数（train 分区）。
  2. 软背压（默认策略）：`high_watermark_ratio`（默认 0.8）与 `low_watermark_ratio`（默认 0.5）。高于高水位时 `prepare_step()` **非阻塞跳过下发**，且不推进 dataloader；低于低水位后恢复。禁止在 `prepare_step` 阻塞等待，因为容量只能由本 step 后续 `sample()` 释放，会死锁。
     - 在高水位以下时，允许按 `max_outstanding_groups` 与当前 outstanding 的缺口**一次下发多个 batch**，用于维持管道深度；这也是 A4 延后情况下的替代方案。
  3. 硬上限（安全阀）：当已在飞的组完成后使 outstanding 超过 `max_outstanding_groups`，按 `prompt_global_steps` 淘汰最老的 terminal group；只丢完整 prompt group，不拆 GRPO 组；**不立即 refill**，避免边丢边补。
  4. `capacity_policy` 可选 `hybrid`（默认，软背压 + 硬上限丢最老）、`backpressure`（只停下发，不丢）、`drop_oldest`（只丢最老，不停下发）。
  5. 与现有淘汰的关系：staleness 的 `drop`/`wait`、DAPO 过滤、failure 淘汰逻辑保持不变；refill 仍是 1:1 替换，不额外消耗预算。
  6. 指标：`rollout/outstanding_groups`、`rollout/dispatch_skipped_steps`、`rollout/capacity_dropped_groups`、`rollout/watermark/{high,low}`。
- **影响面**：`verl/trainer/ppo/v1/trainer_base.py`（`prepare_step` / `_add_batch_to_generate` / `_add_prompts_to_generate`）、`verl/trainer/ppo/v1/replay_buffer.py`（outstanding 统计与硬上限淘汰）、`verl/trainer/config/ppo_trainer.yaml`。
- **风险与代价**：上限过低会让 rollout 空转、异步深度变浅；过高会增加显存/内存与 staleness。滞后水位需要调参。默认 `null` 完全保持现状。
- **验收标准**：
  - 设置上限后，outstanding 不超过 `max + 在飞完成造成的瞬时超额`，且 `dispatch_skipped_steps` 在高水位时增长。
  - `prepare_step` 永不阻塞，无死锁。
  - `observe` 模式下长时间运行，outstanding 稳定在 `[low, max]`。
  - 硬上限只丢完整 prompt group，且不触发立即 refill。
  - `max_outstanding_groups=null` 时行为与当前一致。
- **决策**：
  - [x] 实现
  - [ ] 跳过
  - [ ] 延后

#### A4. Rollout 主动 pull prompt（prompt pool 模型）—— 延后

- **论文依据**：§3.1 "each rollout... after generating its own batch sampled from the prompt pool"；§3.2① rollout 主动从 pool 取。
- **现状**：trainer 的 `_submit_batch_to_rollout` 把 batch 推给 worker。
- **差距**：调度中心化在 trainer，rollout 无自主取任务能力。
- **结论**：A4 不影响正确性，rollout 能正常拿到样本，轨迹级异步的执行语义也不受影响。当前 push 模型的限制是“拿不满”：`AgentLoopManagerTQ.generate_sequences` 把 batch 静态切成 worker 数份，快 worker 干完只能等下一轮，不能从慢 worker 偷活；且每个 global step 只下发一个 batch，管道深度受 `num_warmup_batches` 限制。A4 是负载均衡优化项，不是必选项。
- **延后理由**：
  1. A3 的水位机制可以扩展为“在高水位以下一次下发多个 batch”，在不改 TQ schema 的前提下维持管道深度；
  2. A4 的主要收益是 rollout 速度异构时的 work-stealing，需要 profiling 证明快慢 worker 的空闲时间差异显著；
  3. A4 与 A5、D5 强耦合，改动大、收益不确定。
- **重新评估条件**：profiling 显示快慢 worker 的 idle 时间差异明显，或需要 elastic / 异构 rollout 时。
- **未来实现方案**：TQ 新增 `prompt_pool` 分区，worker 循环 `kv_get` 抢占式取 prompt；trainer 只负责补 pool。
- **影响面**：`verl/trainer/ppo/v1/agent_loop_tq.py`、`verl/trainer/ppo/v1/trainer_base.py`、TQ schema。
- **风险**：与 A5、D5 强耦合，改动大。
- **决策**：
  - [ ] 实现
  - [ ] 跳过
  - [x] 延后

#### A5. Per-replica 并发准入上限

- **论文依据**：§8 A.2 "maximum concurrency to 1024 trajectories per rollout replica"（收敛实验为 256）。
- **现状（分三层看）**：
  1. **Server 级负载均衡没有丢**：`GlobalRequestLoadBalancer` 仍做 sticky session + least-inflight，`LLMServerClient.generate` 在 `_acquire_server()` 后于 `finally` 释放。
  2. **Worker 级分发退化**：旧 `fully_async_policy` 的 `_select_best_worker()` 是逐样本 round-robin；V1 的 `AgentLoopManagerTQ.generate_sequences` 改成 `prompts.chunk(len(workers))` 的静态切分。该问题归入 A4。
  3. **调度层 per-replica 准入上限确实丢失**：旧 `fully_async_policy` 有 `async_training.concurrent_samples_per_replica`（默认 16）和 `max_concurrent_samples = active_server_count × concurrent_samples_per_replica`，由 Rollouter 的 `active_tasks` 门控。V1 里 `AgentLoopWorkerTQ` 无条件 `create_task`，LB 也不检查单 server 容量。
  4. **引擎级上限仍在**：`rollout.max_num_seqs`（YAML 默认 256，`RolloutConfig` 默认 1024）会传给 vLLM 的 `max_num_seqs` / SGLang 的 `max_running_requests`。但它限制的是单次 decode 的并发序列数，不是调度层准入：超量请求会在引擎内部排队，可能造成排队延迟、KV 抢占和显存压力，也无法约束 agent 侧同时在飞的请求数（多轮轨迹、`rollout.n` 条并行 session 都不计入）。
- **差距**：有引擎级 decode 上限，没有调度层 per-replica 准入上限。
- **结论**：旧 `concurrent_samples_per_replica` 在 V1 迁移中丢失，需要补回；实现位置选 LB 侧，因为只有 LB 能看到所有 worker 发往同一 replica 的全局在飞数。
- **实现方案（已决策）**：
  1. 新增 `rollout.max_inflight_requests_per_replica`，默认 `null` 表示不限制，保持现状。
  2. `GlobalRequestLoadBalancer.acquire_server()` 增加容量检查：目标 server 的 `_inflight_requests[server_id] >= cap` 时返回 busy；客户端 `_acquire_server` 收到 busy 后 `asyncio.sleep` 重试，**不能阻塞 LB actor**。
  3. 单位是 **generate 请求数**（≈ 轨迹轮次），不是 prompt group 数；一个 prompt 会展开成 `rollout.n` 个 session，多轮轨迹每轮一次请求。
  4. sticky session 命中但目标 server 已满时同样走 busy 重试，避免绕过上限。
  5. 建议默认值可比 `max_num_seqs` 略小，为 prefill 和控制请求留余量；未显式配置时仍用 `null`。
  6. 指标：`rollout/inflight_requests/{server}`、`rollout/admission_wait_seconds`。
- **影响面**：`verl/workers/rollout/router.py`、`verl/workers/rollout/llm_server.py`、`verl/trainer/config/rollout/rollout.yaml`。
- **风险与代价**：上限过低会让 rollout 空转；过高仍会在引擎侧排队。sticky 与容量冲突时需要触发重选 server，注意不要把热路径变成忙等。
- **验收标准**：
  - 设置上限后，每个 server 的 in-flight 请求数不超过配置值。
  - 满载时客户端等待并重试，而不是报错；LB actor 不被阻塞。
  - `max_inflight_requests_per_replica=null` 时行为与当前一致。
- **决策**：
  - [x] 实现
  - [ ] 跳过
  - [ ] 延后

#### A6. 采样策略：内置 custom_sampler

- **论文依据**：§6 把 priority-based sampling 列为 future work，并明确它是 orthogonal 的；本项作为可选的训练信号优化实现。
- **现状**：默认 `_select_prompt_uids` 按 `prompt_global_steps` 升序（FIFO）；`custom_sampler` 只提供扩展点，没有内置实现。
- **概念澄清**：A6 指的是“group 级采样”，即轨迹进入 experience buffer 后，trainer 挑选哪些 prompt group 组成下一个训练 batch。它不同于 group 内选择（GRPO 用全部 n 条，DAPO 过滤零方差组），也不同于 PPO mini-batch 的内部切分。
- **结论**：实现一组内置 sampler，用户通过 `custom_sampler.path/name` 选择，默认仍是 FIFO。
- **内置 sampler（建议）**：
  1. `ReplayBufferFIFO`：最老优先，等价当前默认，用于显式选择和对照。
  2. `ReplayBufferRewardVariance`：按组内 reward 标准差降序，优先高学习信号组；用 `age_penalty` 防止饥饿。
  3. `ReplayBufferMixed`：按 `fifo_ratio` 混合 FIFO 与 priority，推荐作为 priority 的默认用法。
  4. `ReplayBufferLength`：按 `response_len` 排序，支持 `short_first`（吞吐优先）与 `long_first`（长尾覆盖）。
- **实现方案（已决策）**：
  1. 新增 `verl/trainer/ppo/v1/samplers.py`，均继承 `ReplayBufferAsync`（或 `ReplayBuffer`），只覆盖 `_select_prompt_uids`。
  2. 可复用的数据：`prompt_global_steps`（age）、TQ tag 中的 `response_len` / `seq_len`（长度）、与 `_dapo_filtered_keys` 同源的 reward 指标（方差）。
  3. Reward 方差需要 `kv_batch_get`，按 `finished` 组不可变做缓存，复用 `_dapo_classification_cache` 的思路，避免每次 poll 重复拉取。
  4. 优先级分数统一形式：`score = signal - age_penalty × age`；`ReplayBufferMixed` 先取 FIFO 前 `fifo_ratio × batch_size` 个，再按分数补满。
  5. 所有 sampler 只改变选择顺序，不绕过 staleness / DAPO / failure 淘汰，也不绕过 A3 的容量与 A5 的准入限制。
  6. 配置示例：
     ```yaml
     trainer.v1.sampler:
       custom_sampler:
         path: verl.trainer.ppo.v1.samplers
         name: ReplayBufferMixed
       sampler_kwargs:
         fifo_ratio: 0.5
         age_penalty: 0.1
     ```
- **影响面**：新增 `verl/trainer/ppo/v1/samplers.py`；`replay_buffer.py` 暴露可复用的排序/缓存 hook；`trainer_base.py` 的 `_build_replay_buffer` 已支持加载外部类，无需改动。
- **风险与代价**：priority 可能饥饿老样本、抬高 staleness；reward 方差读取增加 IO；不同 sampler 会改变训练数据分布，需要实验对比。
- **验收标准**：
  - 四个 sampler 均可通过 `custom_sampler` 加载，并在 sync / async 模式下跑通。
  - `ReplayBufferFIFO` 的选择结果与当前默认完全一致。
  - priority 模式下不会选中超过 `max_off_policy_threshold` 的组（`drop` 模式），且 `over_threshold_groups` 可观测。
  - 单测覆盖排序、饥饿保护与缓存命中。
- **决策**：
  - [x] 实现
  - [ ] 跳过
  - [ ] 延后

---

### B. 参数面（relay / weight sync）

#### B1. CPU relay + pinned shared memory + 链式广播 —— 延后

- **论文依据**：§4.2 每个 rollout 机器一个 CPU relay，master relay 收权重后 RDMA 链式广播，rollout 通过 PCIe 从本地 relay 取。
- **现状**：参数同步功能已经可用，且多种 backend 已覆盖论文的部分能力：
  - `nccl` / `hccl`：all-gather + broadcast，固定集群可用；
  - `nixl`：ring p2p，使用 pinned buffer（`pin_memory=True`），支持弹性拓扑；
  - `mooncake`：ring p2p；
  - `kimi_ckpt_engine`：权重先 offload 到 CPU，再 p2p + broadcast；
  - `delta_checkpoint_engine`：pinned-CPU 快照 + 只传增量。
- **差距**：缺少论文那种统一的 master relay + 本地 relay 常驻 + rollout 按需 pull 的抽象。属于架构与效率优化，不影响功能正确性；`separate_async` 只要求 backend 不是 `naive`。
- **结论**：当前先保证“能同步”，B1 延后；现有 backend 作为功能基线。
- **重新评估条件**：
  1. profiling 显示 `timing_s/param_sync` 占比高，或 rollout wait 时间显著；
  2. 集群规模上来后，NCCL 广播成为 actor stall 的瓶颈；
  3. 若 B2 采用 NIXL/Mooncake P2P + B4 本地缓存即可，B1 不是前置；只有需要统一 CPU relay 或 P2P 后端不足时才做 B1。
- **未来实现方案**：新增 `relay` backend：master relay 进程持有 pinned CPU 张量，worker 通过 UCX/RDMA 链式拉取，暴露 `get_weights(version)`。
- **影响面**：`verl/checkpoint_engine/` 新 backend 与配置。
- **风险**：高，需要真实 RDMA 环境验证。
- **决策**：
  - [ ] 实现
  - [ ] 跳过
  - [x] 延后

#### B2. Per-replica anytime pull（仅 separate_async）

- **论文依据**：§4.2⑦ rollout 完成一批生成后随时从本地 relay 拉最新权重，不等全局同步点。
- **现状**：V1 是 step 边界全局 push：`CheckpointEngineManager.update_weights` 对同一 pool 的所有 replica 执行 abort → 建全局通信组 → 同步 → resume，trainer 用 `ray.get` 阻塞等待。
- **差距**：没有“replica 空闲且版本落后时单独更新”的能力；所有 rollout 一起停。
- **范围**：只支持 `separate_async`。`colocate_async` 保留现有全局 push（rollout 与 trainer 共卡，无法做到“边推理边等更新”），这也符合 Laminar 的分离式前提。
- **结论**：实现 per-replica pull；前置为 B4、B5，复用 NIXL/Mooncake 的 P2P/ring，不要求 B1 的完整 CPU relay。
- **实现方案（已决策）**：
  1. 版本发布：trainer 每个 global step 结束后只更新 `latest_weight_version`，并把该版本权重写入共享权重缓存（master cache）；不再对 standalone pool 调全局 `update_weights`。
  2. replica 生命周期：`LLMServerManager` / LB 为每个 server 维护 `local_version` 与 `state ∈ {serving, draining, updating}`。
  3. 触发条件：`state=serving` 且 `inflight_requests == 0` 且 `local_version < latest_weight_version` 时，置为 `draining`（LB 停止路由），然后执行更新。
  4. 更新方式：replica 从共享缓存 `get_weights(latest_version)` 拉取（B4），经 NIXL/Mooncake P2P 写入引擎；不触发 actor 重新导出权重。
  5. 更新完成后置为 `serving`，LB 恢复路由；TQ 侧继续用 `global_steps` / `min_global_steps` / `max_global_steps` 记录轨迹版本。
  6. 在飞请求处理：只 drain、不主动 abort，等请求自然结束；长尾交给 repack（C）或后续可选的 `max_drain_wait_s`，本阶段不引入 abort。
  7. 一致性：一个 prompt group 只会被路由到一个 `serving` replica，且该 replica 在 group 执行期间不会进入 `draining`，与 A1（单轨迹单版本）一致。
  8. 指标：`rollout/version/{server}`、`rollout/version_lag`、`rollout/drain_wait_seconds`、`rollout/update_duration_seconds`、`rollout/draining_servers`。
- **影响面**：`verl/workers/rollout/router.py`（drain/activate/version）、`verl/workers/rollout/llm_server.py`（版本与状态）、`verl/checkpoint_engine/base.py`（per-replica update）、`verl/trainer/ppo/v1/trainer_separate_async.py`（去掉 step 边界全量更新）、配置。
- **风险与代价**：replica 版本会短暂不一致，LB 需要按版本/状态路由；长尾 replica 可能长期 `draining`，降低可用容量；共享权重缓存需要额外 CPU 内存。`colocate_async` 不受益。
- **验收标准**：
  - `separate_async` 下，step 边界不再 abort 全部 replica；每个 replica 在自己空闲且落后时单独更新。
  - 任意时刻 LB 不会把新请求路由到 `draining/updating` 的 replica。
  - 轨迹的 `min_global_steps == max_global_steps`（配合 A1）。
  - `colocate_async` 行为与当前一致。
- **决策**：
  - [x] 实现
  - [ ] 跳过
  - [ ] 延后

#### B3. Master relay 选举 + 链重建 + 无超时故障检测 —— 延后

- **论文依据**：§4.3 master 故障后选新 master，链式拓扑 O(1) 重建，故障检测不依赖超时。
- **现状**：NCCL 引擎有静态 relay 角色；NIXL README 声称可动态调整 ring topology，但代码中没有 master 选举与故障检测。
- **结论**：延后。理由：
  1. B2 已改为 per-replica pull + 共享权重缓存，不再依赖单一 master 做全局广播，master 选举与链重建的收益下降；
  2. NIXL/Mooncake 已提供 ring/P2P，且声称可动态调整拓扑；真正缺失的是故障检测，属于规模化后的可靠性需求；
  3. 该能力需要真实多机 RDMA 故障注入验证，成本高，不影响当前功能正确性；
  4. 与 D4（relay 故障检测/恢复）高度重叠，应与 D4 一起设计，避免两套故障检测机制。
- **重新评估条件**：
  1. 集群规模达到数百 replica，relay/缓存节点故障概率不可忽略；
  2. 线上观测到 relay/缓存节点故障导致权重同步长时间卡住；
  3. 实施 B1 的统一 CPU relay 时，master 选举成为必需能力。
- **未来实现方案**：在 `CheckpointEngineManager` 增加 `health_check` 与 `rebuild_topology`；为共享权重缓存增加 master 选举；优先在 NIXL backend 实现；与 D4 统一故障检测接口。
- **影响面**：`verl/checkpoint_engine/base.py`、`verl/checkpoint_engine/nixl_checkpoint_engine.py`。
- **风险**：高，依赖真实多机 RDMA 环境验证。
- **决策**：
  - [ ] 实现
  - [ ] 跳过
  - [x] 延后

#### B4. `CheckpointEngineWithCache.get_weights` 落地

- **论文依据**：§4.2 本地 relay 缓存。
- **现状**：`verl/checkpoint_engine/base.py` 定义了 `CheckpointEngineWithCache` 抽象类并引用 Laminar，但没有子类实现，也没有调用方。
- **结论**：作为 B2 的前置实现；不需要完整 CPU relay（B1）。
- **实现方案（已决策）**：
  1. 在 NIXL（优先）与 KIMI backend 上实现 `get_weights(version)`：从本地/共享缓存读取该版本的权重分片。
  2. 权重缓存放在 CPU pinned memory 或本地 shm；版本目录由 B5 的注册表索引。
  3. `CheckpointEngineManager` 增加 `get_weights(version)` 的 per-replica 调用路径，与现有 `update_weights` 并存。
  4. 缓存淘汰：保留最近 N 个版本（配置 `weight_cache_versions`），避免无限占用 CPU 内存。
- **影响面**：`verl/checkpoint_engine/base.py`、`nixl_checkpoint_engine.py`、`kimi_checkpoint_engine.py`、配置。
- **风险与代价**：缓存一致性与内存占用；多版本并发读取需要引用计数或不可变快照。
- **验收标准**：同一版本可被多个 replica 重复读取；读取不触发 actor 重新导出；旧版本按配置淘汰。
- **决策**：
  - [x] 实现
  - [ ] 跳过
  - [ ] 延后

#### B5. 全局权重版本注册表

- **论文依据**：§4.2、§6 每条轨迹记录生成时的版本 K，staleness = M - K。
- **现状**：只有 `global_steps` 随请求透传，没有“当前最新版本”和“每个 replica 的本地版本”。
- **结论**：作为 B2 的前置实现。
- **实现方案（已决策）**：
  1. 在 `GlobalRequestLoadBalancer`（或独立的 `WeightVersionRegistry`）维护 `latest_weight_version` 与 `server_id -> {local_version, state}`。
  2. trainer 每次产出新权重后更新 `latest_weight_version`，不直接触发全量同步。
  3. `acquire_server` 只返回 `serving` 的 server；可选按版本亲和路由（同版本优先），为后续 repack/版本组做准备。
  4. API 与指标：`get_versions()`、`set_latest_version()`、`mark_draining()` / `mark_serving()`；`rollout/version_lag`。
- **影响面**：`verl/workers/rollout/router.py`、`verl/workers/rollout/llm_server.py`、`verl/trainer/ppo/v1/trainer_separate_async.py`。
- **风险与代价**：版本注册表在热路径上，需保证原子性（Ray actor 天然串行）。
- **验收标准**：任意时刻可查询 latest 与每个 replica 的版本；`draining/updating` 的 replica 不会被路由。
- **决策**：
  - [x] 实现
  - [ ] 跳过
  - [ ] 延后

---

### C. Repack（长尾消除）

#### C1. Per-replica KVCache 利用率采集

- **论文依据**：§5.2 用 KVCache utilization 作为 idleness 指标，需要 `C_used`、`C_prev`、`N_reqs`。
- **现状**：vLLM/SGLang 只在启用 Prometheus / `enable_metrics` 时暴露 `/metrics`，manager 没有采集入口；`router.py` 只维护请求计数，不知道 KV 使用率。
- **结论**：实现统一的 per-replica 指标采集，供 C2/C3/C5 与监控使用。
- **实现方案（已决策）**：
  1. 在 server adapter 上新增 `get_metrics_snapshot()` actor RPC（沿用现有 `get_server_address` / `get_rollout_config` 的 handler-getter 模式），返回结构化 `ReplicaMetrics`：
     - `kv_cache_usage`（0~1）
     - `num_running_reqs` / `num_waiting_reqs`
     - `num_running_seqs`（可选）
     - `weight_version`（来自 B5）
     - `role`（PD 分离时的 prefill/decode）
     - `timestamp`
  2. vLLM：从 `AsyncLLM` 的 engine stats 或 `/metrics` 读取 `vllm:gpu_cache_usage_perc`、`vllm:num_requests_running`、`vllm:num_requests_waiting`。
  3. SGLang：从 `tokenizer_manager` 的 stats 或 `/metrics` 读取对应指标（如 token usage、running/waiting 请求数）。
  4. `LLMServerManager` 增加 `collect_stats()`，按 `rollout.stats_interval_s`（默认 5s）并发拉取所有 replica；PD 分离时分别保留 prefill/decode 指标。
  5. 维护最近一次快照与短历史（至少 2 个点，供 C2 的 `C_prev`）；指标缺失的 engine 返回 `None` 并降级，不阻塞主流程。
  6. 输出指标：`rollout/kv_cache_util/{mean,max}`、`rollout/num_running_reqs`、`rollout/num_waiting_reqs`、`rollout/stats_age_seconds`。
- **影响面**：`verl/workers/rollout/vllm_rollout/vllm_async_server.py`、`verl/workers/rollout/sglang_rollout/async_sglang_server.py`、`verl/workers/rollout/replica.py`、`verl/workers/rollout/llm_server.py`、`verl/workers/rollout/router.py`。
- **风险与代价**：不同 engine 的指标名与含义不一致；PD 分离需要区分角色；轮询本身有开销，需限频并允许关闭。
- **验收标准**：
  - vLLM 与 SGLang 都能返回 `kv_cache_usage` 与 running/waiting 请求数。
  - 未启用 metrics 时返回 `None`，不影响训练。
  - 任意时刻可查询每个 replica 的最新快照，且保留至少一个历史点供 C2 使用。
  - `stats_interval_s` 可配置，关闭时不产生轮询开销。
- **决策**：
  - [x] 实现
  - [ ] 跳过
  - [ ] 延后

#### C2. Idleness 判定（复用现有 dynamic_schedule）

- **论文依据**：Algorithm 1 Line 3：
  `S = {r | r.C_used < min(C_max, r.C_prev) and r.N_reqs < B}`。
- **现有实现（可复用）**：
  - `DefaultDynamicSchedulePolicy` / `FixedRatioDynamicSchedulePolicy`：基于 wait samples 与 cost/benefit 决定 activate/deactivate hybrid replica；
  - `DynamicResourceController`：`STANDALONE_ONLY <-> HYBRID_ACTIVE` 状态机；
  - `FullyAsyncRollouter._compute_rollout_resource_utilization`：基于 `active_tasks / max_concurrent_samples` 的利用率；
  - `separate_async` 的 `_step_had_idle` / `switch_threshold`：基于 sampleable 数量。
  - 这些都用“计数/等待时间”，没有使用 KVCache 利用率。
- **结论**：不新起一套调度器，直接在现有 `DynamicSchedulePolicyBase` 框架里加入 KV-based idleness 信号，用 C1 的快照作为输入。
- **实现方案（已决策）**：
  1. C1 提供 `ReplicaMetrics` 与至少 2 个历史点；C2 使用 `current` 与 `previous`。
  2. 新增纯函数 `select_idle_replicas(stats, kv_cache_threshold, roofline_batch_size) -> set[server_id]`：
     - `C_used = kv_cache_usage`；
     - `C_prev = 上一快照的 kv_cache_usage`；
     - `C_max = kv_cache_threshold`（默认 0.99）；
     - `N_reqs = num_running_reqs`，并要求 `num_waiting_reqs == 0`（避免把仍有排队工作的 replica 判为空闲）；
     - 条件：`C_used < min(C_max, C_prev)` 且 `N_reqs < B`，其中 `B = roofline_batch_size or max_num_seqs`。
  3. 把结果放进 `DynamicScheduleContext`（新增 `idle_replicas` / `replica_metrics`），新增 `@register_policy("kv_idle")` 的 policy，复用 `DynamicResourceController` 的 activate/deactivate 流程。
  4. `separate_async` 侧通过 `LLMServerManager.get_idle_replicas()` 暴露同一函数结果，供 C3/C4/C5 使用。
  5. 缺失指标时回退到现有计数逻辑（`wait_samples` / `sampleable_count`），不把缺数据当成 idle。
  6. 指标：`rollout/idle_replicas`、`rollout/idle_kv_cache_util`、`rollout/idle_req_count`。
- **影响面**：`dynamic_schedule/base.py`、`dynamic_schedule/default_policy.py`（可选继承）、新增 `dynamic_schedule/idleness.py`、`fully_async_trainer.py` / `dynamic_resource_controller.py`、`separate_async` 的 switch 逻辑、`router.py`。
- **风险与代价**：`C_prev` 抖动可能导致误判；PD 分离需要按角色分别判断；指标缺失时要可靠降级。
- **验收标准**：
  - 构造 C1 快照序列，纯函数输出符合 Algorithm 1。
  - `kv_idle` policy 能在 `DynamicResourceController` 中触发 deactivate/activate。
  - 缺失指标时回退到计数逻辑，不误判。
  - `separate_async` 能拿到同一份 idle 集合结果。
- **决策**：
  - [x] 实现
  - [ ] 跳过
  - [ ] 延后

#### C3. Best-Fit 装箱（Algorithm 1）

- **论文依据**：Algorithm 1。源集合 `S = {r | r.C_used < min(C_max, r.C_prev) and r.N_reqs < B}`；`CanFit` 检查 `C_load + s.C_used <= C_max` 且 `N_load + s.N_reqs <= B`；选择装箱后最满的 destination，目标是最大化被释放的 replica 数。
- **现有可复用部分**：直接采用论文的 Best-Fit 启发式，不引入新的装箱算法；输入复用 C1 的 `ReplicaMetrics` 与 C2 的 `select_idle_replicas()`，调度侧复用现有 `DynamicSchedulePolicyBase` 框架，不另起调度器。
- **结论**：实现纯函数规划器，先支持 dry-run；C4 负责真正迁移，C5 负责触发。
- **实现方案（已决策）**：
  1. 新增 `verl/workers/rollout/repack.py`，提供纯函数 `plan_repack(stats_by_server, idle_set, c_max, b) -> RepackPlan`，复杂度 `O(|S|^2)`。
  2. 版本分组：先按 `weight_version` 分组，只在**同版本** replica 之间 repack，避免跨版本迁移违反 A1（单轨迹单版本）。
  3. 源集合 `S` = `idle_set` ∩ 同版本组，按 `C_used` 升序排序（优先释放小负载，符合论文 Line 4）。
  4. `CanFit(d, s, P)` 与论文一致：destination 的 `C_used`、`N_reqs` 要累加当前计划里已分配给它的 source，再判断是否超过 `C_max` / `B`。
  5. 对每个 source，从合法 destination 中选 `argmax(C_used + 已分配 source 的 C_used)`，即装箱后最满的一个；记录 `(s, d*)` 并标记 `s` 已释放。
  6. 输出 `RepackPlan`：`moves`、`released_replicas`、`estimated_destination_kv_util`、`estimated_moves`。支持 `dry_run=True`：只出计划、不执行，用于 C4 未实现时观测潜在收益。
  7. 对接：`LLMServerManager.plan_repack()` 拉取 C1 快照并调用 C2 得到 idle 集合；C5 触发；可选在 `DynamicSchedulePolicyBase.request_rebalance()` 中复用同一计划。
  8. 指标：`rollout/repack/planned_sources`、`planned_destinations`、`planned_moves`、`estimated_released_replicas`、`plan_duration_ms`。
- **影响面**：新增 `verl/workers/rollout/repack.py`；`verl/workers/rollout/llm_server.py`、`router.py`、`verl/experimental/fully_async_policy/dynamic_schedule/`、配置。
- **风险与代价**：C1 快照抖动会让计划不稳定；按版本分组后 `S` 可能很小，收益受限；Best-Fit 假设迁移开销可忽略，C4 真实迁移可能打破该假设。
- **验收标准**：
  - 合成 `ReplicaMetrics` 单测覆盖 Algorithm 1 的各种情况（可释放、不可释放、同版本分组）。
  - 任意计划都不超过 `C_max` 与 `B`，且不跨版本。
  - `dry_run=True` 无副作用，指标正确输出。
  - 复杂度为 `O(|S|^2)`，大规模下可接受。
- **决策**：
  - [x] 实现
  - [ ] 跳过
  - [ ] 延后

#### C4. 轨迹迁移执行（中止 + 重新 prefill）

- **论文依据**：§5.1③ 将未完成轨迹搬到 destination rollout，迁移开销 0.69s 且不影响延迟。
- **迁移定义**：不搬运 KV cache，而是“源 replica 中止当前请求 → 收集 prompt token + 已生成 response token（以及 logprobs 等元数据）→ 发给目的 replica 重新 prefill 并继续推理”。
- **现有可复用能力**：
  - vLLM 已有 `abort_request(request_id)`（`vllm_async_server.py`），SGLang 已有 `http_server_engine.abort_request(rid)`；
  - `FullyAsyncLLMServerClient.generate` 已有“prompt + 已生成 token 重新请求”的 resume 循环，C4 复用它，只是把“随机重试”改成“定向迁移到 C3 计划的目的 replica”。
- **关键缺口**：
  1. vLLM 当前 `generate` 只保留最后一个流式输出；abort 时返回空 `outputs`，会丢失已生成 token。需要在整个流式循环中累积 token/logprobs，abort 时返回部分结果。
  2. 需要区分 abort 原因：权重同步中止（A1 下丢弃或重采）与迁移中止（保留部分结果并在目的 replica 续跑）。
- **实现方案（已决策）**：
  1. 服务端：在 vLLM/SGLang 的 `generate` 流式循环中累积 `token_ids`、`log_probs`、`routed_experts`；`abort_request` 支持携带 `reason="migration"`，被中止的请求返回部分 `TokenOutput`，并设置 `stop_reason="migrated"` / `extra_fields["abort_reason"]="migration"`。
  2. 迁移协调器：由 `LLMServerManager` 执行 C3 的 `RepackPlan`：
     - 将 source 置为 `draining`，LB 停止新路由；
     - 从 LB 取该 source 的在飞 `request_id` 列表；
     - 逐个调用 `src.abort_request(request_id, reason="migration")`；
     - 请求所属的 `FullyAsyncLLMServerClient.generate` 收到部分结果后，带 `target_server_id=dst` 重新发起，目的 replica 用 `prompt + partial tokens` 重新 prefill 并继续解码。
  3. 目标路由：LB 的 `acquire_server` 支持 `target_server_id` / `preferred_servers` 提示；迁移完成后把 sticky session 从 source 重绑到 destination，避免后续请求回到 source。
  4. 版本约束：目的 replica 必须与 source 同 `weight_version`（C3 已按版本分组），迁移不改变策略版本，满足 A1。
  5. 多轮轨迹：迁移发生在当前 turn 的 `generate` 调用内；工具调用等 AgentLoop 状态仍在 worker 内存中，不参与迁移；下一 turn 按正常路由。
  6. 竞态处理：若 abort 时请求已完成，返回 `aborted=False` 并跳过迁移；若目的 replica 不可用，回退到 LB 选路或原地重试。
  7. 指标：`rollout/migration/requests_total`、`success`、`failed`、`reprefill_tokens`、`duration_seconds`、`fallback_to_lb`。
- **影响面**：`verl/workers/rollout/vllm_rollout/vllm_async_server.py`、`verl/workers/rollout/sglang_rollout/async_sglang_server.py`、`verl/workers/rollout/llm_server.py`、`verl/workers/rollout/router.py`、`verl/workers/rollout/repack.py`（协调器）、配置。
- **风险与代价**：需要修改流式输出累积逻辑，否则 abort 会丢 token；不同引擎的 abort/部分输出语义不一致；迁移会触发 re-prefill，长 prompt 有额外开销；sticky 重绑与并发 abort 的竞态需要测试。
- **验收标准**：
  - 在生成中途 abort，能拿到完整的前缀 token 与 logprobs；迁移到目的 replica 后输出连续、无重复 token。
  - 训练侧看到的轨迹 `min_global_steps == max_global_steps`（同版本）。
  - abort 与完成竞态下不丢请求，必要时回退 LB。
  - vLLM 与 SGLang 都能跑通；TRT-LLM 视其 per-request abort 能力可选支持。
  - 指标正确记录迁移次数、re-prefill token 数与耗时。
- **决策**：
  - [x] 实现
  - [ ] 跳过
  - [ ] 延后

#### C5. Repack 触发时机（定时触发）

- **论文依据**：§5.1 周期检查（如 5 秒）+ trainer 完成权重更新后立即触发。
- **结论**：本轮只做**定时触发**；权重更新后的立即触发作为可选后续，不纳入本轮。
- **范围**：`separate_async`（以及实验性的 fully_async）生效；`colocate_async` 直接 no-op。
- **实现方案（已决策）**：
  1. `LLMServerManager` 启动一个后台协程，按 `repack.interval_s`（默认 5s）周期执行。
  2. 每个 tick 的顺序：C1 `collect_stats()` → C2 `select_idle_replicas()` → C3 `plan_repack()` → 有非空计划时 C4 执行迁移。
  3. 单飞（single-flight）：上一轮 repack 未结束时跳过本 tick，避免计划重叠；指标记录 `skipped_overlap`。
  4. 以下情况跳过本 tick：`repack.enable=False`、正在做权重同步/全局 abort、没有 idle 候选、或没有 `separate_async` 的 standalone replica。
  5. 失败退避：迁移连续失败时按指数或固定倍数退避，避免每个 tick 都重试同一批 source。
  6. 与 B2 的 `draining/updating` 状态互斥：这些 replica 不作为 repack 的 source 或 destination。
  7. 配置：复用 `repack.enable` 与 `repack.interval_s`；不新增“权重更新后立即触发”开关。
  8. 指标：`rollout/repack/trigger_count`、`last_run_age_seconds`、`skipped_overlap`、`plan_empty`、`backoff_active`。
- **影响面**：`verl/workers/rollout/llm_server.py`、`router.py`、`verl/workers/rollout/repack.py`、`verl/trainer/ppo/v1/trainer_separate_async.py`、配置。
- **风险与代价**：周期过短会增加 C1 采集与规划开销；周期过长会错过短暂的空闲窗口。单飞与退避是必须的，否则容易和 B2 的 drain/update 冲突。
- **验收标准**：
  - `repack.enable=True` 时按 `interval_s` 稳定触发；`enable=False` 时完全不运行。
  - 同一时刻最多一个 repack 在执行，`skipped_overlap` 正确计数。
  - 权重同步期间不触发 repack。
  - `colocate_async` 下 no-op，不影响现有行为。
- **决策**：
  - [x] 实现
  - [ ] 跳过
  - [ ] 延后

---

### D. 容错与恢复（本轮整体延后）

> D 组整体先不做：本轮聚焦核心调度与性能，等 A/B/C 稳定后再评估容错。
> 注意：D2 是 liveness 风险（worker 崩溃后 `running` 永不终态，`wait` 策略会永久阻塞），暂不做期间需要监控 `running` 数量并在卡死时人工重启。

#### D1. 中心化 partial response pool

- **论文依据**：§3.1、§3.2② 在飞轨迹流式写入 partial response pool，故障后可续。
- **现状**：在飞 token 只在 worker 的 `AgentData` 内存中；TQ checkpoint 只保存 prompt，不保存中间 token。
- **差距**：rollout 机器故障时已生成 token 全部丢失。
- **建议方案**：每 N 个 token 将 `response_ids` / `logprobs` / `mask` 写入 TQ；恢复时从最近快照续跑。与 A1 互斥。
- **风险**：写放大明显，需采样与限流。
- **结论**：本轮不做（D 组整体延后）。
- **决策**：
  - [ ] 实现
  - [ ] 跳过
  - [x] 延后

#### D2. Running 租约 / 超时（防死锁）

- **论文依据**：§3.3 故障检测与快速恢复的前置条件。
- **现状**：TQ 中 `status=running` 没有 TTL；worker 崩溃后该 group 永不进入终态，`wait` 策略会永久阻塞，`drop` 策略也不会清理。
- **建议方案**：prompt tag 增加 `lease_deadline`，`ReplayBuffer` 每轮将超时的 `running` 转为 `pending` 重新下发（或转为 `failure` 淘汰）。
- **影响面**：`verl/trainer/ppo/v1/agent_loop_tq.py`、`verl/trainer/ppo/v1/replay_buffer.py`、`verl/trainer/ppo/v1/trainer_base.py`。
- **优先级说明**：P0，这是当前实现的真实死锁风险，不依赖任何论文特性。
- **验收标准**：kill 一个 worker 后，训练在 N 秒内继续。
- **结论**：本轮不做（D 组整体延后）。实现前需监控 `running` 数量，卡死时人工重启。
- **决策**：
  - [ ] 实现
  - [ ] 跳过
  - [x] 延后

#### D3. 故障 rollout 的在飞轨迹重定向

- **论文依据**：§3.3 "redirects interrupted trajectories to healthy rollouts with the same weight version"。
- **建议方案**：依赖 D1 与 B5（同版本路由），把轨迹从故障 replica 迁移到同版本健康 replica。
- **结论**：本轮不做（D 组整体延后）。
- **决策**：
  - [ ] 实现
  - [ ] 跳过
  - [x] 延后

#### D4. Relay 故障检测 / 恢复

- **论文依据**：§4.3 无超时检测、秒级链重建。
- **建议方案**：与 B3 合并实现。
- **结论**：本轮不做（D 组整体延后）。
- **决策**：
  - [ ] 实现
  - [ ] 跳过
  - [x] 延后

#### D5. Elastic add/remove 与 LB/TQ 一致性

- **论文依据**：§3.3 动态增删 rollout 不中断训练。
- **现状**：`CheckpointEngineManager.add_replicas/remove_replicas` 会重建通信组，`GlobalRequestLoadBalancer` 有 `add_servers/remove_servers`；但移除时在飞请求的处理与 TQ 中 `running` 的归属没有闭环。
- **建议方案**：remove 前先 `abort`，并把该 replica 的 running group 置回 `pending`；与 D2 共用租约机制。
- **结论**：本轮不做（D 组整体延后）。
- **决策**：
  - [ ] 实现
  - [ ] 跳过
  - [x] 延后

---

### E. 可观测性

#### E1. Per-rollout KV / idle / repack 指标

- **论文依据**：§8.4 报告 average KVCache util（82.2% vs 71.6%）、repack overhead 0.69s、生成吞吐 +26%。
- **现状**：只有 `trajectory_spans` 与 `trajectory_staleness`。
- **结论**：实现统一指标聚合，覆盖本轮已决策的 A/B/C 特性。
- **指标清单（已决策）**：
  - C1：`rollout/kv_cache_util/{mean,max}`、`rollout/num_running_reqs`、`rollout/num_waiting_reqs`、`rollout/stats_age_seconds`；
  - C2：`rollout/idle_replicas`、`rollout/idle_kv_cache_util`、`rollout/idle_req_count`；
  - C3：`rollout/repack/{planned_sources,planned_destinations,planned_moves,estimated_released_replicas,plan_duration_ms}`；
  - C4：`rollout/migration/{requests_total,success,failed,reprefill_tokens,duration_seconds,fallback_to_lb}`；
  - B2/B5：`rollout/version_lag/{mean,max}`、`rollout/draining_servers`、`rollout/update_duration_seconds`；
  - A2/A3：`training/off_policy/over_threshold_groups`、`rollout/outstanding_groups`、`rollout/dispatch_skipped_steps`；
  - A5：`rollout/inflight_requests/{server}`、`rollout/admission_wait_seconds`。
- **采集与上报**：`LLMServerManager.collect_stats()` 每个 step 聚合一次，trainer 合并进现有 metrics dict；避免 per-token 开销；指标缺失时跳过对应 key，不影响训练。
- **影响面**：`verl/workers/rollout/llm_server.py`、`router.py`、`verl/trainer/ppo/v1/replay_buffer.py`、`trainer_base.py`、`trainer_separate_async.py`。
- **验收标准**：
  - 上述指标在训练日志中可见，且数值与 C1/C2/C3/C4 的中间结果一致。
  - 指标采集不在 generate 热路径上，不增加逐 token 开销。
  - 引擎不提供某项指标时降级为缺失，不报错。
- **决策**：
  - [x] 实现
  - [ ] 跳过
  - [ ] 延后

#### E2. 收敛性保护（rollout correction）—— 确认即可

- **论文依据**：§8.2 最小 staleness、不引入额外 bias。
- **现状**：已有 `algorithm.rollout_correction` 与 `bypass_mode`；`separate_async` 实现了 Decoupled PPO 的 `pi_old` 保存/恢复。
- **差距**：基本对齐，不需要新代码，只需在 A1/A2/B2 变更后确认配置组合。
- **确认项**：
  1. A1 开启后，轨迹 `min_global_steps == max_global_steps`，`rollout_correction` 仍能正确处理。
  2. A2 `observe` 模式下不再淘汰过期样本，确认 correction 覆盖高 staleness 区间。
  3. B2 per-replica 版本下，不同 replica 的 `global_steps` 可能不同；确认 `rollout_correction` 使用每条轨迹自己的版本标签，而不是全局最新版本。
  4. `separate_async` 的 Decoupled PPO `pi_old` 保存/恢复与 `parameter_sync_step` 的组合不变。
  5. F3 增加一组配置组合测试，避免后续改动破坏上述假设。
- **决策**：
  - [x] 确认即可
  - [ ] 跳过

---

### F. 工程与配置

#### F1. 跨引擎约束

- **现状**：`separate_async` 已断言 checkpoint backend 不能为 `naive`；repack 与轨迹迁移需要引擎支持。
- **结论**：实现“文档支持矩阵 + 启动时 fail-fast 校验”，避免用户开启不支持的特性组合后到运行时才失败。
- **支持矩阵（已决策）**：

  | 特性 | vLLM | SGLang | TRT-LLM |
  | --- | --- | --- | --- |
  | A1 单轨迹单版本 | 支持 | 支持 | 支持 |
  | A5 per-replica 准入 | 支持（LB 侧，引擎无关） | 支持 | 支持 |
  | B2 per-replica pull | 支持（NIXL/KIMI） | 支持（NIXL/KIMI） | 支持（NIXL/KIMI） |
  | C1 指标采集 | 支持（engine stats / `/metrics`） | 支持（`/metrics`） | 部分（stats API） |
  | C3 Best-Fit 规划 | 引擎无关 | 引擎无关 | 引擎无关 |
  | C4 轨迹迁移 | 支持（`abort_request`） | 支持（`abort_request`） | 暂不支持（无 per-request abort） |

- **实现方案（已决策）**：
  1. 在 `docs/advance/` 补充上述支持矩阵。
  2. 在 `PPOTrainer.__init__` / `LLMServerManager` 集中做校验：如 `repack.migration.enable=True` 且 `rollout.name=trtllm` 时直接报错，并提示改用 vLLM/SGLang。
  3. C1 指标缺失的引擎只告警、不阻断；C4 不支持的引擎禁用迁移并回退到“只规划不执行”（dry-run）。
  4. 校验逻辑集中在一处，避免散落 assert。
- **影响面**：`docs/advance/`、`verl/trainer/ppo/v1/trainer_base.py`、`trainer_separate_async.py`、`verl/workers/rollout/llm_server.py`。
- **验收标准**：不支持的组合启动即报错且信息明确；支持的组合可正常跑通。
- **决策**：
  - [x] 实现
  - [ ] 跳过
  - [ ] 延后

#### F2. 配置项设计

建议新增/扩展的配置：

```yaml
# 已有配置节：扩展 max_off_policy_strategy 的取值
trainer.v1.sampler:
  # drop | wait | observe
  # observe: 不做 staleness 淘汰与阻塞，只统计与告警
  max_off_policy_strategy: drop
  max_off_policy_threshold: 8
  # 内置 sampler 选择：FIFO / RewardVariance / Mixed / Length
  custom_sampler:
    path: null            # 例如 verl.trainer.ppo.v1.samplers
    name: null            # 例如 ReplayBufferMixed
  sampler_kwargs:
    fifo_ratio: 0.5
    age_penalty: 0.1

# 新增配置节：轨迹级行为
trainer.v1.async_trajectory:
  # 每条轨迹只允许一个策略版本
  single_version_per_trajectory: false
  # 异步深度上限：pending + running + finished 的 prompt group 数；null 表示不限制
  max_outstanding_groups: null
  # hybrid | backpressure | drop_oldest
  capacity_policy: hybrid
  high_watermark_ratio: 0.8
  low_watermark_ratio: 0.5
  repack:
    enable: false
    interval_s: 5
    kv_cache_threshold: 0.99
    roofline_batch_size: null
    # C4：中止源请求后在目的 replica 重新 prefill 续跑
    migration:
      enable: true
      timeout_s: 30
      fallback_to_lb: true

# rollout 引擎/准入配置
rollout:
  max_num_seqs: 256
  # 调度层 per-replica 准入上限（generate 请求数）；null 表示不限制
  max_inflight_requests_per_replica: null
  # per-replica KV/请求指标采集
  enable_stats: true
  stats_interval_s: 5
```

- **结论**：按上面的配置落地，统一默认值；同步更新 4 个 `_generated_ppo_*_trainer.yaml` 参考配置；在 `PPOTrainer.__init__` / `_build_replay_buffer` 增加取值范围与组合校验。
- **决策**：
  - [x] 实现
  - [ ] 跳过
  - [ ] 延后

#### F3. 测试与 benchmark

- **结论**：按本轮已决策范围实现测试；D 组与 B1 相关测试随对应功能延后。
- **测试范围（已决策）**：
  1. **单测**：A2 `observe` 模式；A6 四个 sampler；C2 `select_idle_replicas()`；C3 `plan_repack()`；A3 水位与硬上限；A5 LB 准入；C1 指标解析（mock engine）。
  2. **集成测试**：A1 下 `trajectory_spans == 1`；C4 abort → resume 竞态（含 abort 与完成同时发生）；E2 的配置组合。
  3. **Benchmark**：C3 dry-run 规划延迟；B2 per-replica 更新耗时（有环境时）；E1 指标采集开销。
  4. **不做**：D2 故障注入、B1 relay benchmark（随对应功能延后）。
- **决策**：
  - [x] 实现
  - [ ] 跳过
  - [ ] 延后

---

## 5. 依赖与互斥关系

- **A1 与 D1 互斥**：选择"单轨迹单版本"后，不需要 partial response pool；反之亦然。
- **C4 依赖 C3 + B5 + per-request abort/部分输出**：迁移只做同版本、re-prefill（不搬 KV cache），需要目的 replica 与 source 同 `weight_version`。
- **B2 依赖 B4 + B5，且只支持 separate_async**：per-replica pull 需要本地缓存接口与版本注册表，复用 NIXL/Mooncake P2P；B1 不阻塞。
- **D 组整体延后**：D2 仍是 liveness 风险（`running` 无 TTL），暂不做期间需监控 `running` 数量并在卡死时重启；A1 与 D1 的互斥关系保持不变。
- **A3、A5 是 A1/A2 的配套**：A3 提供异步深度与容量上限，A5 提供单 replica 并发上限；开启 `observe` 后尤其依赖 A3 防止无界增长。
- **A6 只改选择顺序**：内置 sampler 依赖 `_select_prompt_uids` 与 reward/长度元数据，不能绕过 A1/A2/A3 的淘汰、容量与背压语义。

---

## 6. 建议实施顺序

1. **P0（配置与正确性开关）**：F2 → F1（支持矩阵与校验）→ A1（开关形式）→ A2（`observe` 模式）
2. **P1（吞吐与可控性）**：A5 → A3 → A6 → B5 → B4 → B2 → C1/C2/C3/C5 → E1
3. **P2（论文完整形态）**：C4
4. **暂缓/不做**：A4（rollout 主动 pull）、B1（CPU relay + 链式广播）、B3（master 选举/链重建）；D1–D5（容错整体不做，D2 需监控 `running`）
5. **横向**：E2 确认、F3 测试与每一步并行

---

## 7. 决策记录

| ID | 决策 | 备注 |
| --- | --- | --- |
| A1 | 实现 | 明确不能跨版本；以开关 `single_version_per_trajectory` 落地，默认关闭保持兼容 |
| A2 | 实现 | 保留 `drop`/`wait` 硬阈值；新增 `max_off_policy_strategy=observe`，不做 staleness 淘汰/阻塞，只统计与告警 |
| A3 | 实现 | 新增 `max_outstanding_groups` 独立控制异步深度；默认 `hybrid`：高水位非阻塞停下发，硬上限丢最老且不立即 refill |
| A4 | 延后 | 不影响正确性；先用 A3 的多 batch 下发维持管道深度，等 profiling 显示 worker 负载不均再实现 |
| A5 | 实现 | 补回旧 `concurrent_samples_per_replica` 的等价能力：新增 `rollout.max_inflight_requests_per_replica`，在 LB 侧做全局准入，默认 null 保持现状 |
| A6 | 实现 | 内置 4 个 sampler：FIFO、RewardVariance、Mixed、Length；通过 `custom_sampler` 选择，默认仍 FIFO |
| B1 | 延后 | 参数同步功能已由 nccl/nixl/mooncake/kimi 覆盖；统一的 CPU relay 属效率优化，等 param_sync 成为瓶颈或 P2P 后端不足时再实现 |
| B2 | 实现 | 仅 `separate_async`：replica 空闲且版本落后时 drain → per-replica pull → 重新 serving；前置为 B4 + B5，不要求 B1 |
| B3 | 延后 | B2 改为 per-replica pull 后 master 选举收益下降；与 D4 重叠，等规模化或实施 B1 时一起做 |
| B4 | 实现 | 在 NIXL/KIMI backend 落地 `get_weights(version)` 与本地/共享权重缓存，作为 B2 前置 |
| B5 | 实现 | 维护 `latest_weight_version` 与每个 replica 的 `local_version/state`，作为 B2 前置 |
| C1 | 实现 | 新增 `get_metrics_snapshot()` RPC 与 `LLMServerManager.collect_stats()`，采集 KV 使用率、running/waiting 请求数和版本；供 C2/C3/C5 使用 |
| C2 | 实现 | 复用现有 `DynamicSchedulePolicyBase` / `DynamicResourceController`，新增 `select_idle_replicas()` 与 `kv_idle` policy，输入来自 C1 快照；缺失指标时回退计数逻辑 |
| C3 | 实现 | 采用论文 Algorithm 1 的 Best-Fit，新增 `plan_repack()` 纯函数；输入来自 C1/C2，按 `weight_version` 分组，支持 dry-run |
| C4 | 实现 | 中止 source 请求 → 收集 prompt + partial tokens → 定向到同版本 destination 重新 prefill；复用现有 per-request abort 与 client resume，需补流式 token 累积 |
| C5 | 实现 | 只做定时触发：`LLMServerManager` 后台协程按 `repack.interval_s` 执行 C1→C2→C3→C4；单飞、失败退避、权重同步期间跳过；`colocate_async` no-op |
| D1 | 延后 | D 组整体不做；与 A1 互斥，等容错需求出现再评估 |
| D2 | 延后 | D 组整体不做；但它是 liveness 风险，暂不做期间需监控 `running` 数量并人工重启 |
| D3 | 延后 | D 组整体不做；依赖 D1 与 B5 |
| D4 | 延后 | D 组整体不做；与 B3 合并评估 |
| D5 | 延后 | D 组整体不做；与 D2 的租约机制一起设计 |
| E1 | 实现 | 统一指标：KV、idle、repack、migration、version lag、outstanding、admission 等，按 step 聚合，缺失降级 |
| E2 | 确认即可 | 已有 rollout correction / bypass_mode / Decoupled PPO；确认 A1/A2/B2 变更后的配置组合 |
| F1 | 实现 | 文档支持矩阵 + 启动时 fail-fast 校验；TRT-LLM 暂不支持 C4 |
| F2 | 实现 | 统一配置默认值，更新 4 个生成参考配置，并在 init 增加范围/组合校验 |
| F3 | 实现 | 单测覆盖 A2/A6/C1/C2/C3/A3/A5；集成覆盖 A1/C4/E2；D2/B1 测试随功能延后 |

---

## 8. 参考

- Laminar: A Scalable Asynchronous RL Post-Training Framework，arXiv:2510.12633，https://arxiv.org/abs/2510.12633
- V1 Async Trainer 现有文档：[v1_async_trainer.md](./v1_async_trainer.md)
- 代码锚点：
  - `verl/trainer/ppo/v1/agent_loop_tq.py`
  - `verl/trainer/ppo/v1/replay_buffer.py`
  - `verl/trainer/ppo/v1/trainer_base.py`
  - `verl/trainer/ppo/v1/trainer_colocate_async.py`
  - `verl/trainer/ppo/v1/trainer_separate_async.py`
  - `verl/workers/rollout/llm_server.py`
  - `verl/workers/rollout/router.py`
  - `verl/checkpoint_engine/base.py`
  - `verl/checkpoint_engine/nccl_checkpoint_engine.py`
  - `verl/checkpoint_engine/kimi_checkpoint_engine.py`
  - `verl/experimental/agent_loop/tool_agent_loop.py`
