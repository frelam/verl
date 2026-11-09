# Laminar 改进最终实施计划

Last updated: 11/06/2026.

> 输入文档：[laminar_gap_analysis.md](./laminar_gap_analysis.md)
> 代码基线：当前工作树，commit `89dad2d7`
> 目标：在不破坏 V1 `sync` / `colocate_async` / `separate_async` 现有路径的前提下，交付已决策的 Laminar-inspired 能力。
> 本文是实施依据；原差距文档中的实现顺序、接口假设或验收口径与本文冲突时，以本文为准。

## 1. 总体结论

本轮需求**整体可行，但不能按原方案直接顺序编码**。A2、A5、C3、E1、F1–F3 有较直接的实现路径；A1、A3、A6、B2、C1/C2/C5 需要修正设计；B4 与 C4 必须先完成底层能力改造，不能只增加表层接口。

实施必须遵守以下原则：

1. **默认兼容**：所有新行为默认关闭或沿用旧值；默认配置下的 dispatch、partial rollout resume、全局 push、FIFO、淘汰和指标语义不变。
2. **小步可回滚**：每个里程碑独立测试、独立交付，先只读观测或 dry-run，再开放会改变训练数据或请求生命周期的功能。
3. **先正确性后性能**：不以复现论文吞吐数字作为正确性验收的替代，也不预设一定获得论文中的收益。
4. **显式能力边界**：不支持的引擎、router、reward 路径和 trainer 组合在启动时 fail-fast，禁止静默降级成另一种训练语义。
5. **不改变延期决策**：A4、B1、B3、D1–D5 仍延后；但 B4 所需的不可变快照存储属于 B2 的必要子集，不等同于实现完整 CPU relay、链式广播或 master 选举。

## 2. 最终范围

| 范围 | ID | 最终交付 |
| --- | --- | --- |
| 数据面 | A1、A2、A3、A5、A6 | 单版本开关、`observe`、有界下发、LB 准入、四种内置 group sampler |
| 参数面 | B2、B4、B5 | 版本注册、不可变快照、standalone replica 的 drain 后独立 pull |
| Repack | C1、C2、C3、C4、C5 | 指标、idle 判定、Best-Fit、持久调度器、同版本迁移 |
| 可观测性 | E1 | 覆盖 A/B/C 的 gauge、counter、duration 与迁移指标 |
| 收敛兼容 | E2 | 验证 behavior logprob、rollout correction 与 Decoupled PPO，不改算法公式 |
| 工程 | F1、F2、F3 | 支持矩阵、配置与校验、单测/集成/合同测试/benchmark |
| 延后 | A4、B1、B3、D1–D5 | 不纳入本轮完成标准；保留重新评估条件 |

## 3. 可行性复核与必须修正的设计

### 3.1 逐项结论

| ID | 可行性 | 代码复核结论 | 最终处理 |
| --- | --- | --- | --- |
| A1 | 可行，需调整 | abort 可经 session 异常传播为整组 `failure`；但 `separate_async` 的全局 push 也会 abort，并非“不受影响”。sample 内即时 1:1 refill 在 colocate 下可能形成补充后马上再次 abort 的浪费 | abort 原因分类；单版本判断在 `length` 覆盖前执行；abort-discard 与普通 failure 的补偿分开，按缺口补发，禁止补偿风暴 |
| A2 | 可直接实现 | 现有代码已经只在 `drop` 淘汰/断言、只在 `wait` 阻塞 | 扩展校验与指标即可；明确 `over_threshold_groups` 使用 prompt 下发步龄口径 |
| A3 | 原方案需重做计数与触发点 | 状态还有 `failure`，必须计入 outstanding；pending→running→terminal 不改变组数，因此“完成导致超额”不成立。选组时 prompt 被清除并释放调度容量，但 trajectory 到 step 末才清理 | 在 dispatch/refill 前 enforcement；统计四种状态；保留多 batch prefetch并补 epoch/checkpoint 账本；明确它是调度深度上限，不是 TQ 物理内存硬上限 |
| A5 | 可行，需调整 | sticky 命中满载 server 后原地 busy 重试会产生头阻塞，即使其他 replica 空闲；但版本租约不能随意重选，V1 也不能把容量路由宣称为完整 `full_determinism` | 未租约请求满载时在同版本可用集合中重选；租约请求原地等待，迁移走原子租约移交；仅全部可选 replica 满载时由 client 退避重试；与 V1 `full_determinism=true` 的组合首版 fail-fast |
| A6 | 原继承方案不可直接落地 | `custom_sampler` 当前替换整个 buffer，且外部 sampler 不自动继承内置 DAPO/failure 参数；单一基类不能同时保持 sync/async 语义。colocated RM 在 sample 后才计算 reward | 内置实现改为 selection-only mixin，按 trainer mode 组合原基类；保留外部 sampler 合同；reward 不可用时对 RewardVariance/Mixed fail-fast |
| B4 | 高复杂度，需先建基础设施 | 当前 `get_weights()` 无版本参数且直接抛 `NotImplementedError`；NIXL/Mooncake 是临时 bucket 链式转发，KIMI 有 group barrier/broadcast，都不是可重复读取的多版本缓存 | 新增不可变 `WeightSnapshotStore`、引用计数与完整发布屏障；NIXL 建 store→replica 直连，KIMI 只复用 CPU offload并新增非 collective pull |
| B5 | 可行 | adapter 已能把实际权重步写入输出；但 prompt 的 `global_steps` 是下发步，二者不是同一语义 | latest/local/state 与版本标签统一；保留下发步龄和实际权重版本两套字段，禁止混用 |
| B2 | 原“anytime hot pull”假设不成立 | 当前每请求 sticky，没有跨 turn 的版本租约；多轮 turn 间隙会出现 inflight=0。现有 adapter 原地换权重并清 KV，不能让旧请求继续使用旧版本 | 改为“随时发布快照，drain 后空闲安装”；引入覆盖单轨迹全部 turn 的 trajectory lease；只支持 standalone pool |
| C1 | 可行，需能力探测 | vLLM/SGLang 可接 stats 或 `/metrics`，但默认 stats 可能关闭；TRT-LLM 当前没有对应适配 | 缺指标、超时、过期均返回 `None`；执行规划要求完整且新鲜的快照；TRT-LLM 不预称支持 |
| C2 | 纯函数可行，集成需解耦 | 旧 experimental dynamic scheduler 服务 hybrid activate/deactivate，不应成为 V1 separate_async 的依赖，但其 `kv_idle` 适配仍是已决策交付范围 | 纯函数放 rollout/repack；V1 直接使用；另交付旧 DynamicResourceController 适配器。迁移执行禁止用计数推断替代缺失的 KV 指标 |
| C3 | 可行 | Best-Fit 可独立实现和测试 | 先 dry-run；排除自身、非 serving、跨版本、未定义 PD 组合和过期快照；执行前再按真实 token/请求数复核 |
| C4 | 当前底层能力阻塞，改造后可行 | vLLM abort 空输出会丢前缀；SGLang 主 async adapter 未证明 per-request abort + partial output，且可能清空不对齐的 logprob；TRT-LLM 只有全局 cancel 后的部分输出，没有 per-request abort | 先定义引擎合同测试，再 vLLM、后 SGLang；增加定向 acquire、请求身份注册表、sticky CAS 与 trajectory lease 原子移交；TRT-LLM 禁用迁移 |
| C5 | 可行，需改生命周期 | `auto_await` 的临时 `asyncio.run` 返回后事件循环关闭，不能直接承载常驻后台协程 | 使用有 start/stop 的独立 Ray actor 或持久循环；V1 与 experimental fully_async 共用 tick/规划实现；先 dry-run，C4 完成后才开放执行 |
| E1 | 可行 | 现有 step 聚合不能把所有值都按样本加权平均 | counter 报区间 delta，状态报最后 gauge，duration 报 sum/count 或合适的分布值 |
| E2 | 可确认，但原验收描述需修正 | rollout correction 消费逐 token 的 `rollout_log_probs` 与 `old_log_probs`，不读取版本标签 | 验证混合版本批次中 logprob/IS/RS 与 π_old 保存恢复不变；不把版本标签描述成 correction 的输入，也不把测试等同于收敛保证 |

### 3.2 A3 的最终容量语义

1. `outstanding = pending + running + finished + failure`，仅统计 train 分区。
2. 每次决策前先同步 TQ 元数据；所有新 prompt 下发都经过统一的 budget-aware dispatch 入口，包括 prepare、warmup、普通 refill 和 DAPO refill。checkpoint reissue 不新增 group，不重复消耗 dataloader 预算。
3. `max_outstanding_groups=null` 走原有快速路径；设置后必须满足 `max_outstanding_groups >= data.train_batch_size`。
4. 下发分为两类：
   - **可选 prefetch**：prepare/warmup 为维持管道深度而多发数据，受高低水位、硬上限和 epoch 预算约束；
   - **当前 batch 必需补发**：淘汰后 `sampleable + pending + running` 已不足以组成当前 batch 时，为 A1/failure/DAPO 等补足缺口。它豁免高低水位，否则高水位与样本不足会形成死等；但仍受硬上限约束，空间不足时等待在飞完成或按 policy 淘汰可丢 terminal group。
5. 水位只根据**本次可选下发前**的 outstanding 判断：达到高水位进入暂停，降到低水位恢复。未暂停时可按硬上限缺口一次下发多个完整 batch，因此 `cap=B` 的首批也能下发，不能用 `outstanding + B > high_watermark` 阻止首批。
6. 三种 policy：
   - `backpressure`：可选 prefetch 使用滞后水位，不做容量淘汰；
   - `drop_oldest`：可选 prefetch 不使用滞后水位，空间不足时先淘汰最老的完整 terminal group；
   - `hybrid`：可选 prefetch 使用滞后水位，并在硬上限不足时淘汰最老 terminal group 为正式 dispatch 腾空间。
   所有 policy 的多 batch 数量都取 `max - outstanding` 与 epoch 可预取量的较小值，并按 dataloader fetch 粒度向下取整；必需补发则按实际缺口取整，任何情况下都不得超过硬上限。
7. 多 batch prefetch 必须新增 dispatch ledger：记录每个训练 epoch 的已下发、已训练、预取和预借数量。可选 prefetch 不跨 dataloader epoch 回卷；必需补发保留原有自动回卷能力，但记为下一 epoch 的预借并等量扣减其可选 prefetch。checkpoint 同时保存 dataloader 状态、ledger 和 TQ outstanding，恢复后先重建账本再 reissue，避免重复消费或样本跳过。
8. 容量淘汰只处理完整 group，不拆 `rollout.n`；容量淘汰自身不触发 1:1 refill。后续可选 prefetch 或当前 batch 必需补发独立决策，避免边丢边补循环。已有足够潜在样本时不得继续处理 A1 abort-discard credit。
9. staleness、DAPO、普通 failure 的既有规则保持；默认 `max_outstanding_groups=null` 时仍走原来的 1:1 refill。只有启用新容量/A1 语义后，超出当前 batch 缺口的补偿才转为可选 prefetch并受水位控制。
10. “非阻塞”指**预算检查不新增等待**，不承诺改变 `separate_async.prepare_step()` 已有的 sampleable/switch 等待，也不把 `ReplayBuffer.sample()` 改成非阻塞。

### 3.3 A5 的最终准入语义

1. 未持有版本租约的请求，sticky server 满载后在同版本、 serving 且未满的 replica 中重选并更新 sticky；全部可选 replica 满载时才返回 busy。
2. 持有 trajectory lease 的请求不得因普通负载均衡改绑 replica；满载时等待绑定 replica 释放容量。只有 C4 迁移可以通过原子 lease transfer 改绑。
3. 本计划不扩展 V1 的完整确定性保证。A5 cap、per-replica pull 和 migration 都会引入动态时序或改绑，首版与 V1 `full_determinism=true` 的组合统一 fail-fast；不得把“hash/sticky 原地等待”描述为完整确定性。
4. busy 重试统一放在通用 client，使用有界 sleep + jitter，处理 cancellation、server removal 和 acquire/release 配对；LB actor 内不得 sleep。

### 3.4 A6 的最终排序语义

1. FIFO 使用 `(prompt_global_steps, uid)` 确定性排序，与现有最老优先一致。
2. RewardVariance 使用采样前已存在的 canonical reward 或配置的 reward metric，建立独立 variance cache；不能复用只保存“零方差值或 None”的 DAPO 分类 cache。
3. 原公式 `signal - age_penalty × age` 会让越老样本优先级越低，**必须改为**：

   ```text
   score = signal + age_bonus × age
   age = current_global_step - prompt_global_steps
   ```

4. Mixed 先取 FIFO 前 `floor(fifo_ratio × batch_size)` 个，再按 RewardVariance 分数补满，去重并以 uid 打破同分。
5. Length 直接使用 trajectory tag 的 `response_len`，支持 `short_first` / `long_first`，同长度按 FIFO。
6. 所有内置 sampler 只覆盖选择顺序，staleness、DAPO、failure、A3 和 A5 仍由公共路径执行。
7. RewardVariance/Mixed 要求 rule-based reward 或独立 reward resource pool；colocated RM 场景启动即报错。Length 不依赖 reward。

### 3.5 B2/B5 的最终版本与租约语义

1. **两种步龄分离**：
   - `prompt_global_steps`：prompt 下发时的 trainer step，用于 A2/A3/A6；
   - `min/max_global_steps`：replica 实际生成时使用的权重版本，用于轨迹观测与单版本验收。
2. trainer 仍在原 `on_step_end` 同步点产出权重，保持 `parameter_sync_step` 和 Decoupled PPO 节奏；变化仅在 standalone 的分发方式。
3. 快照完整提交后才推进 `latest_weight_version`。发布失败的版本不可见；每个版本只从 actor 导出一次。
4. 租约采用**单轨迹**而非整个 prompt group：`trajectory_id=(uid, session_id)`。在 `_run_agent_loop` 首次 generate 前获取 lease，将 token 贯穿后续 turn 的路由参数，并在该 session 的 `finally` 中释放。同一轨迹固定 replica/version；同一 GRPO group 的不同 session 可以分布到不同 replica，A1 不要求组内同版本，E2 必须覆盖这种混合版本 batch。
5. 状态机在 LB actor 内原子转换：
   - `serving`：可接受新 trajectory lease；
   - 发现 `local_version < latest_weight_version` 后先置 `draining`，立即停止新 lease，但允许已租约轨迹完成；
   - trajectory lease 与 inflight 都归零后进入 `updating`，拉取一个确定的不可变快照；
   - 安装成功后更新 `local_version` 并回到 `serving`；失败则保留旧版本、按间隔重试，不能谎报新版本。
6. 这不是引擎内的双版本 hot swap。真正的“请求不排空、旧新权重同时服务”需要引擎提供双版本权重/KV API，不在本轮承诺范围内。
7. 首版 `per_replica_pull` 仅支持 standalone pool；`colocate_async` 和 hybrid switch 继续使用原全局 push/abort。首版对 `per_replica_pull + hybrid_rollout.enable_switch=true` fail-fast，避免两套版本生命周期混用。

### 3.6 C4/C5 的最终安全合同

1. abort 必须带原因：`weight_sync`、`migration` 等不能混为一个 `stop_reason`。A1 只丢弃不允许续跑的 abort；migration 在同版本条件下保留前缀续跑。
2. LB 增加 `request_id → engine_request_id/server/attempt/version/trajectory_id` 注册表、`target_server_id` 定向 acquire、`list_inflight(server_id)` 和基于 attempt 的 sticky 原子重绑。
3. 先在 destination 预留准入容量并复核真实 token/请求预算，再 abort source；不能在 source 已中止后才寻找 destination。
4. 对需要续跑的迁移，必须在同一个 LB 原子操作中完成 sticky 重绑和 attempt 递增，后续 turn 只能进入 destination。per-replica pull 下持有 trajectory lease 时，还必须同步完成 `lease: source → destination`；global push 下没有 lease，只执行 sticky/attempt CAS，并由 C5 与全局权重同步互斥保证同版本。若请求已完整结束则不迁移、不改绑；fallback 到另一同版本 replica 时遵循同一规则。
5. abort 与完成竞态中不得丢请求；destination 失效时只允许回退到**同版本 serving** replica。A1 开启时禁止回退到其他版本。
6. migration 不得全局清空 prefix cache。非空 response logprob 必须与 response token 等长；logprob 未启用时允许为 `None`。routed experts 按各引擎现有 schema 校验其与 `prompt + response` 或 response 区段的对齐，不能简单要求它与 response token 等长。所需元数据无法保留时不开放迁移。
7. `repack.enable=false`、`migration.enable=false`、`fallback_to_lb=false` 为安全默认。`migration.enable=false` 即 C5 的 dry-run，不再增加重复的 dry-run 配置开关。
8. 缺指标、快照过期、权重同步、状态冲突、上一轮未结束或连续失败退避期间，本 tick 均 no-op。

## 4. 最终配置设计

以下是逻辑配置树；实际 rollout 配置位于 `actor_rollout_ref.rollout`。各配置随对应里程碑加入，未实现的非默认组合必须报错，不能先接受配置但静默无效。

```yaml
trainer:
  v1:
    sampler:
      max_off_policy_strategy: drop  # drop | wait | observe
      max_off_policy_threshold: 8
      custom_sampler:
        path: null                   # 例如 verl.trainer.ppo.v1.samplers
        name: null                   # FIFO/RewardVariance/Mixed/Length 对应类名
      sampler_kwargs:
        fifo_ratio: 0.5
        age_bonus: 0.1
        short_first: true

    async_trajectory:
      single_version_per_trajectory: false
      max_outstanding_groups: null
      capacity_policy: hybrid        # hybrid | backpressure | drop_oldest
      high_watermark_ratio: 0.8
      low_watermark_ratio: 0.5
      repack:
        enable: false
        interval_s: 5
        kv_cache_threshold: 0.99
        roofline_batch_size: null
        migration:
          enable: false
          timeout_s: 30
          fallback_to_lb: false

    separate_async:
      weight_sync:
        mode: global_push            # global_push | per_replica_pull
        cache_versions: 2
        retry_interval_s: 5

actor_rollout_ref:
  rollout:
    max_inflight_requests_per_replica: null
    enable_stats: false
    stats_interval_s: 5
```

集中校验规则：

- `0 <= low_watermark_ratio < high_watermark_ratio <= 1`；上限为正整数且不小于一个 train batch；`max_inflight_requests_per_replica` 为 `null` 或正整数。
- V1 `full_determinism=true` 与 A5 cap、per-replica pull 或 migration 的非默认组合首版统一 fail-fast；本计划不声称扩展 V1 的完整确定性支持。自定义 router 必须声明并实现所开启功能所需的 lease/version/migration 能力。
- `fifo_ratio ∈ [0,1]`、`age_bonus >= 0`、`cache_versions >= 2`，避免发布新版本时淘汰仍被读取的旧快照；被引用版本无论如何不得淘汰。
- `observe` 只对 async 有实际意义；A3 上限在 sync 下显式配置时报错，避免误以为 sync 也有异步管道。
- pull 仅允许 `separate_async` 和已验证的 backend/adapter；首版禁止与 hybrid switch 同时开启。
- stats 开启时校验引擎能力与 `disable_log_stats` 组合；repack 必须有时效内完整指标。仅监控时缺指标可告警降级，执行 repack 时不可降级。
- migration 只允许通过合同测试的引擎与内置/兼容 router；TRT-LLM 首版直接报错。`colocate_async` 的 repack 为明确的一次性告警 no-op。
- A1 + 全局 push 合法但给出高丢弃率告警；B2 完成前不得宣称 separate_async 不受全局 abort 影响。

## 5. 分阶段实施计划

### M0：基线、合同与校验骨架（F1/F2/F3）

**改动**

- 固化三种 trainer mode 的默认行为回归用例和现有关键指标。
- 建立集中式 feature combination validator；自定义 router 按所需能力声明接口，缺失时只在相关功能开启后报错。
- 为 vLLM/SGLang/TRT-LLM 建 abort、partial output、logprob、metrics 能力合同测试；区分“现成可用、需改造、未验证”，不预先写成已支持。
- 配置变更同时覆盖 YAML、dataclass、生成参考配置和文档；参考配置由现有脚本生成，不手改。

**退出条件**：默认配置行为不变；不支持组合能在启动阶段失败；引擎能力有明确测试证据。

### M1：轨迹版本与 staleness（A1/A2）

**改动**

- A2 增加 `observe`、超阈值 gauge 和 drop/wait 回归。
- A1 增加 `TrajectoryAbortedError`，在 `length` 覆盖前识别 abort，保留旧开关下的 resume。
- 将 abort-discard 与普通 failure 分开记录；prepare 优先处理补偿，sample 只在潜在样本不足时按缺口补发。
- 记录 `aborted_discarded` 与整组 failure/refill；多轮任一非迁移 abort 使整组失败。

**验收**：两种 async 模式连续两步 abort/refill 无死等、无补偿风暴；`n>1` sibling 全部结算；A1 开启时 `trajectory_spans/max == 1`，关闭时 resume 与当前一致；`observe` 不做 staleness 淘汰/阻塞，DAPO/failure 仍工作。

### M2：准入与有界下发（A5/A3）

**改动**

- A5 在 router factory 注入 cap；未租约 sticky 请求满载先重选，全部可选 replica 满载才返回 busy；通用 client 使用有界 sleep + jitter 并正确处理 cancellation/release。预留后续 trajectory lease 的路由参数与接口；V1 `full_determinism=true` 与 cap 的组合启动即报错。
- A3 统一 dispatch 入口，统计四种状态；实现三种 policy、硬上限前检查、完整 group 淘汰、指标和 checkpoint/warmup/refill 集成。
- 实现高/低水位以下的多 batch prefetch；当前 batch 必需补发只受硬上限约束。新增 dispatch ledger，处理可选 prefetch 不跨 epoch、必需补发跨 epoch 预借、checkpoint 保存与恢复。
- 验证默认 `null` 时不启用 ledger 驱动的新下发路径，保持原单 batch prepare 与 1:1 refill。

**验收**：sticky 满载但邻节点空闲时未租约请求不头阻塞；全满时 LB actor 不 sleep、不阻塞；cap 不被突破，`null` 完全兼容。A3 覆盖 cap=B 首批、多 batch 水位、必需补发豁免水位、failure 占容量、sample 释放、跨 epoch 预借、checkpoint 恢复、capacity drop 不立即 refill、长时间 `observe` 不增长无界；高水位且样本不足不会死等。

### M3：内置采样与收敛兼容（A6/E2）

**改动**

- 新增 selection-only 内置 sampler 及 sync/async 基类组合；外部 custom sampler 的构造合同不变。
- 实现 FIFO、RewardVariance、Mixed、Length，独立 reward stats cache 与缺失元数据校验。
- E2 增加不同版本标签但相同 logprob 时 correction 结果不变、高 staleness、A1 单版本、`bypass_mode` 两种取值及 π_old 调用序列测试。

**验收**：三种 trainer mode 均可加载适用 sampler；显式 FIFO 与默认结果一致；DAPO/failure/A3 语义不回归；排序、年龄奖励、缓存命中和 reward 不可用 fail-fast 有测试。

### M4：版本注册与不可变权重快照（B5/B4）

**改动**

- B5 先以只读方式接入现有 global push，注册 latest/local/state，核对 adapter 实际版本标签。
- 实现 `WeightSnapshotStore`：不可变版本、完整发布屏障、并发读、校验和、引用计数、保留最近 N 版且不可淘汰在读版本。
- `CheckpointEngineWithCache.get_weights(version)` 落地为版本化接口。
- NIXL 建 store→replica 直连拉取；KIMI 复用 CPU offload，但新增无 group barrier/broadcast 的单 replica pull。Mooncake 不在本轮 pull 支持列表。
- 本阶段仍由配置强制使用 global push。

**验收**：同版本并发重复读取一致；actor 每版本只导出一次；失败发布不可见；读者持有时不淘汰；NIXL/KIMI 在对应 GPU/RDMA 环境通过真实权重校验。未通过这些合同前不得进入 B2 开放测试。

### M5：Standalone per-replica drain-pull（B2）

**改动**

- 在 `_run_agent_loop` 获取并在 `finally` 释放 trajectory lease，将 lease token 贯穿所有 AgentLoop/内置路由参数；实现版本亲和路由和 `serving→draining→updating→serving` 原子状态机。
- A5 对租约请求只允许绑定 replica 等待，普通负载重选只适用于未租约请求。
- trainer 发布快照后不再对 standalone pool 做全局 abort/push；各 replica 独立 drain、pull、安装和恢复 serving。
- hybrid/colocate 保持原路径；首版禁止 pull 与 hybrid switch 组合。
- 接入 version lag、drain wait、update duration、serving/draining 数量和失败重试指标。

**验收**：两 replica 环境中一个更新时另一个持续生成；新 trajectory lease 不进入 draining/updating；同一轨迹多轮及 turn 间隙始终同 replica/version，同组不同 session 可落在不同版本；acquire/drain 竞态无穿透；A1 下每条轨迹 `min_global_steps == max_global_steps`；global push 默认路径和 colocate 回归通过。

### M6：Repack 观测与 dry-run（C1/C2/C3/C5/E1）

**改动**

- C1 实现结构化快照、短历史、能力探测、超时与新鲜度检查。
- C2/C3 在 rollout/repack 中实现共享纯函数；V1 separate_async 直接调用，不依赖 experimental rollouter。
- 另行交付 `kv_idle` DynamicSchedulePolicy 和 DynamicResourceController 适配，验证 activate/deactivate；计数回退只用于旧资源切换，不得驱动 repack 迁移。
- C5 使用独立持久 coordinator，具备 start/stop、single-flight、失败退避和退出清理；V1 与 experimental fully_async 复用同一 tick/规划实现，本阶段 `migration.enable=false`，只收集与规划。
- E1 按 gauge/delta/duration 聚合，不在 generate/token 热路径采集；增加 running group 数与最老 running age，作为 D2 延期期间的监控。

**验收**：mock engine 指标解析、缺失/过期、idle 序列、Best-Fit 容量/版本/PD、dry-run 无副作用、single-flight、提前退出全部通过；`kv_idle` 能触发 DynamicResourceController activate/deactivate；V1 separate_async 与 experimental fully_async 都能定时生成 dry-run 计划；colocate no-op；默认 stats 关闭时无新增轮询 RPC。

### M7：真实迁移与执行开放（C4/C5）

**改动**

- 先完成 vLLM partial token/logprob 保留、per-request migration abort、定向 resume，再完成 SGLang 同等合同。
- 实现请求身份注册表、目标容量预留、sticky CAS、同版本 fallback 和 abort/complete 竞态处理；per-replica pull 下同步移交 trajectory lease，global push 下只做 sticky/attempt CAS 并与全局权重同步互斥。
- V1 与 experimental fully_async 的 coordinator 都在 `migration.enable=true` 时执行计划；不满足安全合同仍 no-op/fail-fast，不能自动跨版本续跑。
- TRT-LLM 保持迁移禁用；未验证的自定义 router/backend 不开放。

**验收**：生成中途迁移后 token 连续、无重复；非空 response logprob 与 response token 等长，routed experts 按引擎 schema 与最终序列正确对齐；lease/sticky 改绑后后续 turn 不回源；目的端满/下线、abort 与完成同时发生、fallback、A1 开启均有测试；V1 separate_async 与 experimental fully_async 都能定时执行真实迁移，并保持 single-flight 与退出清理；vLLM/SGLang GPU 集成通过后，才将支持矩阵标为“支持”。

### M8：系统验收与发布

- 跑默认关闭的全量回归，以及各 feature 的独立开启矩阵和合法组合矩阵。
- 用相同数据、模型、资源和种子对比基线：tokens/s、trainer 等待、丢弃率、outstanding、admission wait、staleness、version lag、drain/update、KV util、迁移成功率和 re-prefill 开销。
- benchmark 结果只报告实测值；功能默认 off 合并不以复现论文的 26% 或 0.69s 为门槛，生产默认开启则需另行依据测量评审。
- 更新 [v1_async_trainer.md](./v1_async_trainer.md) 的配置、支持矩阵、限制、指标和回滚说明，完成 F1/F2/F3 闭环。

## 6. 主要代码落点

| 模块 | 主要文件 |
| --- | --- |
| A1/A2/A3/A6 | [replay_buffer.py](../../verl/trainer/ppo/v1/replay_buffer.py)、[trainer_base.py](../../verl/trainer/ppo/v1/trainer_base.py)、[agent_loop_tq.py](../../verl/trainer/ppo/v1/agent_loop_tq.py)、[trainer_colocate_async.py](../../verl/trainer/ppo/v1/trainer_colocate_async.py)、[trainer_separate_async.py](../../verl/trainer/ppo/v1/trainer_separate_async.py)，新增 `verl/trainer/ppo/v1/samplers.py` |
| A5/B5/C4 | [router.py](../../verl/workers/rollout/router.py)、[llm_server.py](../../verl/workers/rollout/llm_server.py)、[replica.py](../../verl/workers/rollout/replica.py) |
| B2/B4 | [base.py](../../verl/checkpoint_engine/base.py)、[nixl_checkpoint_engine.py](../../verl/checkpoint_engine/nixl_checkpoint_engine.py)、[kimi_checkpoint_engine.py](../../verl/checkpoint_engine/kimi_checkpoint_engine.py)，新增快照存储实现 |
| C1–C5 | [vllm_async_server.py](../../verl/workers/rollout/vllm_rollout/vllm_async_server.py)、[async_sglang_server.py](../../verl/workers/rollout/sglang_rollout/async_sglang_server.py)，新增 `verl/workers/rollout/repack.py` 与持久 coordinator |
| 配置 | [ppo_trainer.yaml](../../verl/trainer/config/ppo_trainer.yaml)、[rollout.yaml](../../verl/trainer/config/rollout/rollout.yaml)、[rollout.py](../../verl/workers/config/rollout.py)、必要的 trainer dataclass |
| 测试 | [test_replay_buffer_on_cpu.py](../../tests/trainer/ppo/v1/test_replay_buffer_on_cpu.py)、[test_agent_loop_tq_on_cpu.py](../../tests/trainer/ppo/v1/test_agent_loop_tq_on_cpu.py)、[test_router_on_cpu.py](../../tests/workers/rollout/test_router_on_cpu.py)、[test_global_steps_on_cpu.py](../../tests/checkpoint_engine/test_global_steps_on_cpu.py) 及各引擎合同测试 |

## 7. 测试与验证矩阵

| 层级 | 必测内容 |
| --- | --- |
| CPU 单测 | A2/A3/A6、A5 路由与重试、版本/lease 状态机、快照发布与引用计数、C1 解析、C2/C3、指标聚合、single-flight |
| 并发合同 | acquire/release 配对、sticky 满载重选、full_determinism 非法组合 fail-fast、trajectory lease 跨 turn 与原子移交、drain 竞态、目标预留、sticky CAS、abort/complete、取消与退出 |
| Trainer 集成 | 三种模式默认兼容；A1 两种 async；A3 多 batch/warmup/refill/checkpoint/epoch；E2 两种 bypass；B2 两 replica；C5 colocate no-op |
| 引擎 GPU 合同 | vLLM/SGLang/TRT-LLM 能力分别验证；迁移只对通过合同的引擎开放；NIXL/KIMI 使用真实后端环境，不能用 mock 结果宣称支持 |
| Benchmark | 采集开销、Best-Fit 规划延迟、权重导出次数、单副本更新、版本 lag、迁移/re-prefill、端到端 tokens/s 与训练质量 |

核心回归命令按变更范围执行：

```bash
uv run pytest -q tests/trainer/ppo/v1/test_replay_buffer_on_cpu.py
uv run pytest -q tests/trainer/ppo/v1/test_agent_loop_tq_on_cpu.py
uv run pytest -q tests/trainer/ppo/v1/test_trainer_base_on_cpu.py
uv run pytest -q tests/trainer/ppo/v1/test_separate_async_step_switch_on_cpu.py
uv run pytest -q tests/workers/rollout/test_router_on_cpu.py
uv run pytest -q tests/checkpoint_engine/test_global_steps_on_cpu.py
bash scripts/generate_trainer_config.sh
pre-commit run --all-files
```

新增测试优先复用上述文件；sampler、repack、snapshot store 等独立模块可新增对应 `*_on_cpu.py`。真实 GPU/RDMA 测试必须在具备依赖的环境运行并记录命令、环境和结果。

## 8. 上线与回滚顺序

1. **默认关闭合并**：先合入兼容性与测试，不改变线上行为。
2. **数据面灰度**：先 A3/A5 观测与有界保护，再 A2 `observe`；A1 在 global push 下开启前必须评估丢弃率。
3. **sampler 灰度**：先显式 FIFO，再 Length，最后 RewardVariance/Mixed；对比数据分布、staleness、reward 与 KL。
4. **权重同步灰度**：global push 下验证 B5/B4 只读与快照正确性，再对 standalone 小池开放 pull；异常时停止新发布并回退到 global push 配置，回退过程按全局同步规程处理在飞轨迹。
5. **Repack 灰度**：stats → idle/plan dry-run → 单 vLLM replica 对迁移 → SGLang；任何合同失败都关闭 `migration.enable`，保留观测。
6. **回滚单位**：`single_version_per_trajectory=false`、`max_outstanding_groups=null`、`max_inflight_requests_per_replica=null`、默认 FIFO、`weight_sync.mode=global_push`、`repack.enable=false`、`migration.enable=false`、`enable_stats=false`。

## 9. 风险与延期项边界

- **D2 仍是真实 liveness 风险**：`running` 无租约，worker 崩溃可能使 `wait` 永久阻塞。本轮不实现 TTL；E1 必须提供 running 数和最老年龄，运行手册保留人工重启流程。
- **A1 有样本利用率代价**：global push 下 colocate 与 separate standalone 都会 abort。A3/A5 能控制容量，但不能消除重采成本；B2 是降低 standalone 丢弃的关键后续。
- **A3 不等于物理内存配额**：prompt 元数据在选组时释放，trajectory fields 在 step 末清理；需要单独监控 TQ 内存。
- **A4 仍只有延后价值**：A3 的多 batch prefetch 能维持管道深度，但不能解决 batch 静态切分后的快慢 worker 不均与 work-stealing；只有 profiling 证明这些差异显著时才重新评估 A4。
- **B4 与 B1 有存储重叠但不扩大范围**：本轮只实现 B2 必需的快照与直连 pull，不实现 relay 链、master 选举、拓扑重建。
- **observe 没有 staleness 上界**：A3 只能限制调度深度，不能证明收敛；必须保留阈值告警、训练质量监控和可回滚配置。
- **Repack 规划是估计**：KV utilization 不精确等于待迁移请求的真实 KV 占用，执行前必须二次核验并保留目的地安全边际。

## 10. 完成定义

本轮完成必须同时满足：

- 第 2 节“最终交付”中的功能均有实现、配置、文档和对应测试，不以仅添加配置或接口占位计为完成。
- 默认配置下三种 trainer mode 的原有行为与测试通过；所有非法组合 fail-fast，所有安全降级不改变训练语义。
- A1 单版本、A2 observe、A3 有界调度、A5 per-replica 准入、A6 四种 sampler 有端到端证据。
- B4 真实快照与 NIXL/KIMI 合同通过，B2 两 replica drain-pull 通过；否则不得将 B2/B4 标记完成。
- C1–C3/C5 dry-run 与 C4 vLLM/SGLang 真实迁移分别通过；V1 separate_async 和 experimental fully_async 均验证定时执行、single-flight 与退出清理；TRT-LLM 和未验证 router 保持明确禁用。
- E1 指标数值与内部状态一致，E2 的 correction/Decoupled PPO 兼容测试通过。
- A4、B1、B3、D1–D5 未被意外扩大进本轮；D2 的监控与人工恢复说明已交付。
