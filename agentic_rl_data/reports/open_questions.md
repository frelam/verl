# 开放问题清单 — agentic_rl_data（按席位分组）

> Owner: critic（T5）。每条都必须是**可裁决**的问题：有 owner、有建议裁决、有"不解决会怎样"。
> 依据：plan = `C:\Users\Administrator\Downloads\agentic_rl_data_plan.md`；CT = `agentic_rl_data/CONTRACTS.md`（Lead v1，2026-11-05 21:20）。
> 优先级：P0 = 会算错分/污染评测/崩溃；P1 = 验收缺口/不可复现；P2 = 文档/口径。
> 本清单与 `reports/acceptance_matrix.md` 的编号（A/S/C/R/K）互相引用。

## 覆盖自检（任务卡要求的 11 个必答项 → 对应问题）

| 必答项 | 问题 |
|---|---|
| 回放式奖励诱发调用准确性偏好 | Q-18（+ 矩阵 K2、R3） |
| N=8 通过率估计噪声/保留区间 | Q-11 |
| FunReason-MT / ToolMind 与 BFCL 污染、去污染 0 命中 | Q-01 |
| license 合规 | Q-02 |
| 单轮占比伤害多轮规划 | Q-12 |
| search 吞吐/超时拖垮训练 | Q-27 |
| code_exec 单测质量与答案泄漏 | Q-28 |
| memory 第一期边界 | Q-31 |
| held-out 样本量 | Q-13 |
| 采样字节级可复现 | Q-14 |
| `[CONFIRM]` 未核实项对配比的影响 | Q-03 |

---

## A 组 — data-researcher（T1，数据源与 loaders）

### Q-01 FunReason-MT / ToolMind 与 BFCL 同源污染：去污染必须真的是 0 命中
- **优先级**：P0
- **owner**：data-researcher（T1）；critic 复核（T5）
- **为什么重要**：plan L229 明确两者"与 BFCL 同源或相近"；L127 要求对 BFCL **query 与工具定义**做 n-gram + 近似匹配，命中即丢并记录；验收 A2（L208）要求命中数为 0。如果只做 query 字面匹配、或只对 BFCL 的一个 split 检查，报告会给出"假 0"。
- **建议裁决**：①去污染语料必须覆盖 BFCL v3 multi-turn + v4 agentic（web search、memory）+ Live/Non-Live/Hallucination（plan L201-204）；②匹配同时走"规范化 query 精确 + 3~5-gram Jaccard + tools 定义签名/近似"三路；③FunReason-MT 用最严阈值（命中即全丢，不做阈值豁免）；④报告输出 `bfcl_hits`、`dropped_by_source`，并保留每条命中的 `(task_id, bfcl_id, match_kind, score)`；⑤stage 内先 decontam 后 dedup 的样本顺序要记录（避免去重把污染样本"洗白"）。**验收前的自查**：随机抽 30 条被判"未命中"的 FunReason-MT，人工/规则比对 BFCL 同任务。
- **不解决的后果**：BFCL 分数虚高，整个训练结论不可信；A2 无法判定为通过。
- **关联**：A2、S4、K4；TEAM_BOARD `[Q1]`。

### Q-02 各数据集 license 是否允许训练用途
- **优先级**：P1
- **owner**：data-researcher（T1）
- **为什么重要**：plan L230 要求逐个核许可证并记录 `meta.license`；`[CONFIRM]` 标注的数据集（FunReason-MT、APIGen-MT、Tau-bench、Toucan、ToolMind、Search-R1、ASearcher、ReTool、When2Call、APPS/CodeContests）来源不一，部分可能仅研究用途/禁止再分发。
- **建议裁决**：产出一张 `reports/data_sources.md` 许可表：source / HF ID / license 版本 / 是否允许训练 / 是否允许再分发 / 核实时点 / 依据链接；不允许训练的源直接从配比剔除并在 `data_report.md` 记一行；`meta.license` 写归一化短码（如 `apache-2.0`）。
- **不解决的后果**：训练产物无法对外说明合法性；报告无法回答"这些数据能用吗"；如果事后剔除，配比要重跑。
- **关联**：C11、S1。

### Q-03 `[CONFIRM]` 未核实项对配比的影响（数据集规模/字段/轮数）
- **优先级**：P1
- **owner**：data-researcher（T1）→ mixture-researcher（T2）据此更新 caps
- **为什么重要**：plan 的配比（L176-184）按类别给占比，但每类来源的实际可用条数决定能不能填满。例如 FunReason-MT "约 16k"、ToolMind "约 11 万"、APIGen-MT 5k、Toucan 1.5M 都带 `[CONFIRM]`；若某源 ID 不存在/字段变了/过 15 轮过滤后只剩几百，多轮 30% 与搜索 25% 可能填不满，只能靠重复采样或借别的源。
- **建议裁决**：①T1 对每个 `[CONFIRM]` 项给"已核实/未核实/不存在"三态与依据（HF card/API 截图/URL）；②T2 的 `mixture.yaml` 里每源写 `cap` 与 `fallback`，构建脚本在"某类不足"时打印 `shortfall` 而不是静默重复采样；③报告列出每类 `available vs sampled`。
- **不解决的后果**：mixture 名义配比 ≠ 实际配比（重复采样会让某源权重超限）；A1/A4 的"最终配比"统计失真。
- **关联**：C4、S7、A4；TEAM_BOARD L55-56。

### Q-04 现有 5 源的条数与 plan 表格是否一致（配比分母）
- **优先级**：P1
- **owner**：data-researcher（T1）
- **为什么重要**：plan L13-23 给 apigen 4900 / sealtools 4873 / toolace 4033 / hammer 2980 / apibank 161。这些是"处理后的 train 条数"还是原始规模未说明；`examples/tool_rl/prepare_data.py` 已经做过清洗（DESIGN.md §2.1/2.3），实际可用条数可能不同。单轮 20%（L180）的分母依赖这个数。
- **建议裁决**：T1 复用 `examples/tool_rl/prepare_data*.py` 的口径，输出 `reports/data_sources.md` 表：原始 count / 清洗后 count / 可用 count（过轮数与去重后），并注明各自命令；T2 用"可用 count"做 cap。
- **不解决的后果**：20% 单轮配比实际可能 30%+，压垮多轮规划（与 plan L23 观察到的现象直接冲突）。
- **关联**：S1、S7。

### Q-05 `reference.gt_turns` 的元素 shape 与回放返回（与 T4 联调）
- **优先级**：P0
- **owner**：Lead（冻结）+ T1（写入）+ T4（解析）
- **为什么重要**：CT C6（L84-85）规定回放来源优先级：`env.config["gt_returns"] → reference.gt_turns[*].tool_results/results/return → placeholder`。但 `schema.py:21-26` 只写 `gt_turns: list | None`，没有元素字段定义；T1 写出的 dict 与 T4 解析器不一致时，全部样本会落到 `placeholder`，S5 重放校验失去意义。
- **建议裁决**：Lead 在 CT C6 补一段 JSON 示例并把 `gt_turns` 元素 shape 冻结为（例如）`{"assistant": {...}, "tool_results": [{id,name,content,ok}]}`；T1 的 loader 单测断言写入 shape；T4 replay env 单测断言能解析该 shape。三方用同一份合成 fixture 交叉验证。
- **不解决的后果**：S5 全变 placeholder；reward 拿不到 `replay_ok` 真值；A3 重放满分不可信。
- **关联**：C7、K7、S5；review F-01。

### Q-06 BFCL 测试数据的来源与版本固定
- **优先级**：P1
- **owner**：data-researcher（T1）核实；Lead 决策
- **为什么重要**：plan L202 提到 `bfcl-eval==2025.10.27.1` 并标 `[CONFIRM]`。**T1 已核：PyPI 最新是 `bfcl-eval==2026.3.23`，文档版本已过时**。去污染语料必须与评测时用的数据是同一快照，否则"命中 0"与真实评测集对不上。TEAM_BOARD `[Q1]` 仍未定：pip 包内置数据 vs gorilla 仓库目录。
- **建议裁决**：选定一个来源并 pin 版本（建议 `bfcl-eval==2026.3.23`，除非用户评测环境固定旧版）+ 记录 `sha256`/commit hash 到报告；脚本支持"包内 data 目录 / 用户给路径"双入口（T1 已按此设计）；把 v3 multi-turn 与 v4 agentic 的 data 路径落表；版本不可得时用 gorilla 仓库某 commit 快照，报告写明。
- **不解决的后果**：A2 永远无法判定；版本漂移后训练集里可能重新引入污染；"0 命中"不可复现。
- **关联**：A2、S4。

### Q-07 近似去重阈值与 ToolMind↔xlam/ToolACE 的删除量
- **优先级**：P1
- **owner**：data-researcher（T1）
- **为什么重要**：plan L126 要求近似去重并"重点处理 ToolMind 与 xlam/ToolACE 的重叠"，但没给阈值；阈值太松丢难负样本（Seal-Tools 孪生工具场景，DESIGN.md L19-23），太紧污染去不掉。
- **建议裁决**：MinHash(5-shingle)+Jaccard 默认阈值 0.8，可配；embedding 分支离线环境不可用则显式声明"未实现"而不是假装；报告给每对来源的删除数与抽样 50 对人工核对结论；保留"被删样本 id 清单"便于回滚。
- **不解决的后果**：ToolMind 与已有 5 源重复样本进入训练 → 变相提高某些源权重；或误删导致多轮类不足（连锁 Q-03）。
- **关联**：C13、C14、S3。

### Q-08 数学 prompt 对推理评测集去污染需要用户提供评测集清单
- **优先级**：P1
- **owner**：Lead/用户提供清单；T1 实现
- **为什么重要**：plan L127 后半句要求"对你的推理评测集（如 AIME 等）对数学 prompt 做去污染"。仓库里没有 AIME 等清单时，该分支只能空转。
- **建议裁决**：用户给现有推理评测集路径/名称（AIME 2024/2025、MATH-500 等）；T1 做同样的 query 精确 + n-gram 去污染，命中写入报告；`math_ci` 池复用用户 reasoning RL 的 prompt 池（plan L59）。
- **不解决的后果**：数学/推理基准虚高，掩盖策略偏移（plan L194 的监控失效）。
- **关联**：S4、R5。

### Q-09 ASearcher 数据位置与"预期交互轮数"元数据
- **优先级**：P1
- **owner**：data-researcher（T1）
- **为什么重要**：plan L52 要求 ASearcher 过滤到 ≤15 轮（原作可到 40+）；L123 要求"gt 轮数**或预期交互**"两种口径。如果 loader 只记录 `gt_calls` 轮数，会低估搜索轨迹的交互量。
- **建议裁决**：loader 在 `reference.gt_turns` 或 env.config 记录原始轨迹轮数与 token 长度；过滤以 `max(gt 轮数, 原始轨迹轮数)` 为准；ASearcher 单独路径与过滤计数写报告。
- **不解决的后果**：S2 过滤后仍有 >15 轮任务进训练，超限即失败 → 大量无效样本与零梯度。
- **关联**：S2、Q-05。

### Q-10 search 训练环境与 BFCL web search 的工具 schema 是否对齐
- **优先级**：P1
- **owner**：env-builder（T4）主，data-researcher（T1）供语料
- **为什么重要**：plan L55 要求训练用本地 Wikipedia + E5、BFCL 风格 `search(query)` + `fetch(url_or_doc_id)`；而 BFCL v4 web search 评测用搜索 API（L203 提到 SerpAPI key）。两边工具名/参数/返回格式不一致时，训练学到的调用方式在评测上不迁移。
- **建议裁决**：T4 冻结工具 schema（名称、必选参数、返回字段 `{title, snippet, doc_id}`），T1 的 Search-R1/ASearcher loader 把 gold 检索文档映射到同一 doc_id 空间；报告写明"训练工具 vs BFCL 工具"差异表（允许差异但必须列明）。
- **不解决的后果**：web search 得分低且无法归因（是检索质量还是调用格式）；A6 对比表失去解释力。
- **关联**：S1、S4、A6。

---

## B 组 — mixture-researcher（T2，配比、通过率、构建）

### Q-11 N=8 通过率估计的噪声与保留区间
- **优先级**：P0（直接影响"哪些任务能进训练"）
- **owner**：mixture-researcher（T2）；Lead 决策
- **为什么重要**：plan L129 建议 N=8 采样，保留通过率 ∈(0,1)。8 次二项采样的可观测值只有 0/8…8/8：p̂=0.5 时 σ≈0.177；真实通过率 0.1 的任务被采成 0/8 的概率 ≈0.9^8≈43%（被误删）；真实 0.9 的任务 8/8 概率 ≈43%（被误删）。若把 1/8 当作"太难"再删，实际删掉的可能只是噪声。
- **建议裁决**：①记录 `(pass_rate, n=8, checkpoint)` 三列；②保留区间按**观测值** `[1/8, 7/8]`（而不是任意小数）并在报告写明；③困难类（mt_fc/search）用 N=16 复核边界样本；④把"全 0/全 1"的样本标 `hard/easy` 保留在 side file 供 hard_replay（DESIGN.md §3.2）而不是直接销毁；⑤消融：N=8 vs N=16 的 mixture 分布差异。
- **不解决的后果**：mixture 组成随采样噪声波动、不可复现地偏向"中等难度"；难样本和易样本都被系统性丢弃，训练信号变窄。
- **关联**：S6、A4。

### Q-12 单轮数据占比上限（20%）与监控钩子
- **优先级**：P1
- **owner**：mixture-researcher（T2）；Lead 决策
- **为什么重要**：plan L23 观察到"单轮数据过多会让多轮规划变差、过度关注调用准确性"，L180 给约 20%；L193 要求指标下降时优先下调单轮。这个"20%"必须有硬上限与监控，否则各源 cap 叠加后实际可能超过。
- **建议裁决**：`mixture.yaml` 写 `st_plus_abstain_ratio: 0.20` 并逐类校验（构建后打印实际占比，超过阈值则 `--strict` 报错退出）；Hammer irrelevance 必须保留（L180）；报告给"实际单轮占比"与"st_fc/irrelevance/abstain 各自条数"。
- **不解决的后果**：多轮指标下降且无法归因；回到 plan L23 的原始问题。
- **关联**：S7、A4、R10。

### Q-13 held-out 的数量口径与切分时点
- **优先级**：P1
- **owner**：mixture-researcher（T2）
- **为什么重要**：plan L130 说"每类别至少 200 条，不参与训练"，但"类别"没定义（§6 的 5 类？7 个 task_type？每 source？）；且切分时点若在混合之后，可能把同会话/同源近重复切进验证集（DESIGN.md §2.5 的会话级泄漏教训）。
- **建议裁决**：按 **7 个 task_type × ≥200 = ≥1400** 定 held-out（与 schema 枚举对齐，机检方便）；每 source 额外 ≥20；**先 dedup/decontam、按会话组/来源去重，再按 id hash 稳定切分**（同一会话整组只进一侧）；`build_mixture` 断言 `set(train_ids) ∩ set(heldout_ids) == ∅`。
- **不解决的后果**：验证分数虚高或方差过大；早期选 checkpoint 的决策不可信。
- **关联**：S7、A4。

### Q-14 采样/构建的字节级可复现
- **优先级**：P1
- **owner**：mixture-researcher（T2）
- **为什么重要**：A1（plan L207）要求重复运行结果一致；这不仅是 `dump_jsonl(sort_keys=True)`（schema.py:345-366），还取决于任务迭代顺序、`random`/numpy 种子、集合迭代、`meta.pass_rate` 浮点。
- **建议裁决**：①固定 `seed` 写入 config；②输出任务按 `id` 升序排序后写出；③随机采样用 `random.Random(seed)` 或 `numpy.random.Generator(PCG64(seed))`，禁止裸 `random` / `np.random` 全局；④报告中写 `manifest_sha256`；⑤重复运行两次对比 sha256，加入 `[shell]` 验收命令。
- **不解决的后果**：A1 必失败；两次训练 run 的数据不可比，消融结论失真。
- **关联**：A1、C12、S7。

### Q-15 通过率过滤的输入接口（rollout 耦合 vs 文件）
- **优先级**：P1
- **owner**：mixture-researcher（T2）+ Lead（D2 已定默认文件接口）
- **为什么重要**：TEAM_BOARD L70-71 已说"默认按文件接口实现，rollout 耦合留 TODO"。接口若不定，T3/T4 的单测与用户侧 slime 接入会各写一套。
- **建议裁决**：冻结文件 schema，例如 JSONL `{"id": str, "pass_rate": float, "n": int, "checkpoint": str}`（一行一 task）；`filter_by_passrate.py --pass-rates <path>` 读入、写回 `meta.pass_rate`，`--n` 与 pass_rate 一致性校验（pass_rate 必须等于 k/n 或允许浮点？建议严格 k/n 并记 k）；slime rollout 导出脚本留 TODO 与占位命令。
- **不解决的后果**：S6 无法执行；用户接入时二次返工。
- **关联**：S6、Q-11。

### Q-16 mixture 总量、类别口径与 epoch 预算
- **优先级**：P1
- **owner**：Lead + mixture-researcher（T2）
- **为什么重要**：plan 只给占比，没有总量；L195 建议"每类 ~500 条"先跑通再放大。总量决定 RL steps/epoch 与 held-out 比例，也影响 A6 的对比成本。
- **建议裁决**：给出两档配置：`smoke`（每 task_type ~500，总量 ~2-3k）与 `v1`（建议 ~15-20k 或按用户算力定）；config 显式写 `total_tasks` 与每类目标条数（占比 × total 取整后的余数规则要固定，例如向下取整后按 id hash 补足）；报告打印每类 target vs actual。
- **不解决的后果**：占比取整误差（每类几百条时 1% 误差 = 十几条）导致实际配比与声明不符；A4 的"最终配比"对不上。
- **关联**：S7、A1、A4。

### Q-17 placeholder 回放样本在 mixture 里的处理
- **优先级**：P1
- **owner**：mixture-researcher（T2）+ data-researcher（T1）
- **为什么重要**：CT C6（L88-91）要求 placeholder 样本显式标记、统计比例、`mt_fc` 过滤或降权，并在报告给过滤前后条数。placeholder 意味着 reward 只能看"调用匹配"而无真实返回，进一步放大调用准确性偏好（Q-18）。
- **建议裁决**：mixture 默认 `drop placeholder`（可 config 切换为 `keep_with_weight=0.5`）；报告输出 `placeholder_total / placeholder_dropped / placeholder_kept`；`mt_fc` 若保留 placeholder 必须 `info.degraded_replay=True` 且 reward 不给 0.8 结果项。
- **不解决的后果**：降质样本污染多轮奖励信号，且报告看不到降质比例。
- **关联**：K4、Q-18、A4。

---

## C 组 — reward-writer（T3，奖励函数）

### Q-18 回放式 mt_fc 奖励与"调用准确性偏好"（plan §10 点名风险）
- **优先级**：P0
- **owner**：reward-writer（T3）+ env-builder（T4）；Lead 冻结权重
- **为什么重要**：plan L144 指出 B 方案"逐调用匹配的稠密信号容易把策略拉向调用准确性"，L233 再次强调"务必让结果项占主导"。CT C4（L67-68）冻结 `0.8 结果 + 0.1 逐轮匹配 + 0.1 格式`，但**结果项 `replay_ok` 的定义**决定是否真的占主导：若 `replay_ok` 只是"gt_calls 全部按顺序匹配"，那 0.8 本质仍是调用准确性，只是换了个名字。
- **建议裁决**：①`replay_ok` 必须由 env 终态/最终结果判定（B 方案在 mock_fc 里至少要校验终态变量或 final_answer，而不是只看调用名/参数）；②`0.1 逐轮匹配` 上限锁死，禁止把 `matched_turns/gt_turns` 乘到 0.8；③单测三条硬断言：仅逐轮匹配但终态失败 → 总分 < 0.2；仅结果对但调用参数有小差异 → 仍能拿 ≥0.8；全错 → 0；④A/B 两方案各跑 10 条人工轨迹对比分数排序；⑤`info` 暴露 `terminal/match/format` 三个分量用于 Goodhart 监控。
- **不解决的后果**：策略退化成"背诵 gt 调用序列"，BFCL 多轮与 agentic 分数上不去（plan L23 的已知现象重演）。
- **关联**：R3、K2、Q-17、A3。

### Q-19 reward 的"双真相源"审计（只读 trajectory）
- **优先级**：P0
- **owner**：reward-writer（T3）；critic 逐文件审计（T5）
- **为什么重要**：CT C3（L51-52）明确 reward 只读 `trajectory`，不得访问 env 实例/私有缓存。若某 reward 为了"终态"偷偷 import env 或读模块级单例，会出现训练时算分与离线复算不一致（可复现性崩溃）。
- **建议裁决**：所有 reward 的签名固定为 `reward(task, trajectory)`；源码不允许 import `agentic_rl_data.envs.*`（可加 CI grep 断言）；env 的所有判定结果必须经 `env_state` 传过来（CT C3 的 `goal_satisfied/replay_ok/checks/extras.tests`）；单测：先跑 env 得到 trajectory，再清空/复用 env 实例跑 reward，分数不变。
- **不解决的后果**：reward 不可离线复算 → DAPO 过滤/报告数字对不上；P0。
- **关联**：K1、R8、CT C3。

### Q-20 reward 分数范围（含是否允许负分）
- **优先级**：P1
- **owner**：Lead（schema 冻结）+ T3
- **为什么重要**：`schema.py:308-324` 只要求有限数；plan §5 的 1/0 与 0.8+0.1+0.1 隐含 `[0,1]`，但 `examples/tool_rl/DESIGN.md:11` 的既有实现刻意允许负分来压制乱猜。两套语义混用会让 GRPO 组内归一化/报告口径不一致，且 `validate_reward_result` 无法在验收时发现越界。
- **建议裁决**：建议冻结 `[0,1]`（plan §5 是"SFT→RL 初期、偏移要小"，负分惩罚不是规格要求），越界视为契约错误；若将来要负分，必须 Lead 改 CT/schema 并广播。给 `validate_reward_result(result, *, min_score=0.0, max_score=1.0)` 或单独 `validate_score_range`。同时明确 `info` 里的 `invalid`/`timeout`/`degraded_replay` 布尔标记（不算分但要统计）。
- **不解决的后果**：T2 的 DAPO/std 过滤与 reward 尺度不匹配；A3 的"满分"口径不定。
- **关联**：R10、C16、K5。

### Q-21 "gt 重放 reward 满分"的浮点容差
- **优先级**：P2
- **owner**：reward-writer（T3）+ T4
- **为什么重要**：0.8+0.1+0.1 的浮点求和依赖具体分组方式，`== 1.0` 断言可能偶发红；A3 要求"重放 reward 为满分"，没有容差定义就没法判定。
- **建议裁决**：每个 reward 模块暴露 `MAX_SCORE: float`（mt_fc=1.0），测试统一 `math.isclose(score, MAX_SCORE, abs_tol=1e-9)`；`validate_reward_result` 不强制等于 1。
- **不解决的后果**：A3 验收在 CI 上抖动；不同机器（x87/FMA）上偶发失败。
- **关联**：A3、R3。

### Q-22 `abstain` 与 `irrelevance` 如何区分
- **优先级**：P1
- **owner**：reward-writer（T3）+ data-researcher（T1）
- **为什么重要**：plan L163（irrelevance：未调用任何工具=1）与 L168（abstain：正确追问/拒答=1）在"不调用工具"时行为重叠；`gt_calls=[]` 在 schema 里同时表示两者（schema.py:33-35）。若 abstain 的正确答案是"向用户反问"，irrelevance 的正确答案也可能是"不调用"，奖励会互相冲突。
- **建议裁决**：定义判分规则：`irrelevance` 只对"工具列表已声明但都不该用"的场景，score=1 当 `tool_calls==[]`；`abstain` 需要额外标注（如 `reference.gt_turns` 里含用户期望的反问文本，或 `meta.abstain_kind ∈ {ask, refuse}`），score=1 当最终回复匹配"信息不足/请求更多信息"的规则分类；两个 task_type 的测试用例必须区分。
- **不解决的后果**：模型学到"永远不调用"，或两个类型的分数互相抵消；Q-12 的单轮占比统计也失真。
- **关联**：R2、R7、C6。

### Q-23 `search_qa` 多答案的表示（A²Search 多答案 F1）
- **优先级**：P1
- **owner**：Lead（schema 字段决策）+ T3
- **为什么重要**：plan L165 允许 A²Search 多答案 F1；`reference.final_answer` 是单字符串（schema.py:21-26），装不下多答案；用 JSON 字符串编码是隐性约定，reward/loader/报告三方都可能解析不一致。
- **建议裁决**：Lead 裁决：①新增 `reference.gold_answers: list[str] | None`（推荐，显式）；或②在 CT 里冻结 `final_answer` 为 JSON list 的约定并给 loader/reward 共用解析函数。T3 用 `gold_answers` 存在时走多答案 F1，否则单答案 EM/F1。
- **不解决的后果**：多答案样本只能按单答案判分 → 假阴性，搜索类奖励噪声大。
- **关联**：R4、C6。

### Q-24 reward 分布记录（防 Goodhart）落到哪
- **优先级**：P1
- **owner**：reward-writer（T3）+ T2（report）+ Lead（训练侧）
- **为什么重要**：plan L172 要求记录各项奖励分布以定位"被钻空子"；没有固定 info 字段，就无法在离线数据上聚合，训练侧更难。
- **建议裁决**：冻结 `info` 的最小公共键：`score_components: {terminal, match, format, tests, abstain}`、`invalid: bool`、`timeout: bool`、`degraded_replay: bool`；`reports/data_report.md` 加"奖励分量分布"节（histogram/均值/方差，离线用 gold trajectory 跑一遍）；训练侧采样分布留 TODO。
- **不解决的后果**：无法发现"只拿格式分/只拿匹配分"的策略；奖励 bug 靠肉眼。
- **关联**：R9、K4、K5。

### Q-25 `code_exec` reward 对 infra 失败（超时/沙箱挂）的语义
- **优先级**：P1
- **owner**：reward-writer（T3）+ env-builder（T4）
- **为什么重要**：plan L167 说"单测全部通过=1，否则 0"，但沙箱超时/启动失败不是"答错"；若一律 0，会教模型"避免执行代码"（错误信号）。CT C5 的 `extras.tests` 有 `ok/exit_code/duration_s/truncated`，但 infra 失败与测试失败需区分。
- **建议裁决**：`extras.tests.ok=False` 且 `exit_code` 为超时/信号类（或 `invalid=True`）时，reward 返回 `score=0` 但 `info.invalid=True`，训练侧应从 batch 剔除（不是学 0）；`message` 截断且不含答案代码（CT C5 L79）；T3/T4 各写一条用例。
- **不解决的后果**：沙箱抖动直接变成负样本；A3 桩数据不稳定。
- **关联**：R6、K3、K5。

### Q-26 timeout 样本不得拿 0.8 结果项（与 T4 联动）
- **优先级**：P0
- **owner**：env-builder（T4）+ reward-writer（T3）
- **为什么重要**：CT C1（L20-21）说 timeout"不额外扣分"，只由 reward 决定"不给 0.8 主项"。若 env 在超限时仍保留 `goal_satisfied=True`（例如最后一步已达成目标但超轮数），或 reward 把"null 当 false 但仍有 0.1+0.1"，会给超限轨迹高分。plan L139 明确"超限视为失败"。
- **建议裁决**：env 在 `timeout=True` 时强制 `goal_satisfied=False` / `replay_ok=False`（或 null，reward 侧 null→false），`extras` 保留原始状态供审计；reward 单测：timeout 且 checks 全过 → 总分 ≤0.2（理想 0，因为"超限视为失败"）；T4 单测：超 max_turns 一步后 `done=True, timeout=True` 且终态布尔为 false。
- **不解决的后果**：模型学会"拖到最后一轮"；A3 重放满分样例中混入超限样本。
- **关联**：K5、R3、A3。

---

## D 组 — env-builder（T4，环境）

### Q-27 search 环境吞吐/超时是否拖垮训练
- **优先级**：P0
- **owner**：env-builder（T4）
- **为什么重要**：plan L148 要求本地检索服务支持高并发、超时与重试，L232 强调"并发和超时直接影响 RL 吞吐，需压测"。检索是 rollout 的同步阻塞点，p95 延迟高会直接拉长 step 时间；超时策略不当会把大量轨迹判失败。
- **建议裁决**：冻结 SLO：单次 `search/fetch` p95 ≤ 300ms（本地 E5+索引可做到），任务级总超时 60-90s；超时/失败返回统一 degradation obs（`ok=false` + 简短 error）而不是抛异常；并发压测脚本输出 QPS/p95 到 `reports/env_setup.md`；训练侧配置 `max_concurrent` 与 `retry=1`。报告给"超时率/重试率"两个指标。
- **不解决的后果**：search 类 25% 数据 rollout 大量超时 → 组内全 0 被 DAPO 过滤 → 搜索能力学不到；训练 step 变慢。
- **关联**：S1、A6。

### Q-28 code_exec 单测质量与答案泄漏
- **优先级**：P0（污染/奖励作弊）
- **owner**：env-builder（T4）主，data-researcher（T1）供题
- **为什么重要**：plan L64 标 `[SUGGEST]` 未验证，L152 要求"不把标准答案代码暴露给策略"。APPS/CodeContests 部分题带官方解答且测试用例少（可能 1-2 个弱测试），模型可以通过"猜测试"或"打印答案"骗分；`env.config["redact"]`（CT C5 L79）只脱敏日志，不解决题面里的解法信息。
- **建议裁决**：①入池条件：测试用例 ≥5 且含至少 2 个非常规输入；题目描述不得含官方解法/题解链接；②执行输出截断 + 正则脱敏（`redact` 配置），`failures.message` 不返回期望值之外的任何源码；③对"打印 hardcode 输出"类作弊跑一轮 canary（隐藏测试）；④报告写明每来源单测数分布与 canary 结果。
- **不解决的后果**：code_exec 10% 数据教模型作弊；A3 重放满分可能只是测试太弱。
- **关联**：R6、K3、Q-25。

### Q-29 mock_fc A 方案状态机的范围（对齐 BFCL v3 multi-turn 类别）
- **优先级**：P1
- **owner**：env-builder（T4）
- **为什么重要**：plan L143 要求"参照 BFCL v3 multi-turn 的有状态 API 类（文件系统、社交、交易等）实现，终态可验证"。BFCL 的类别多、状态机复杂，若只实现一个玩具域，训练/评测域不匹配，"终态可验证"也会变成自定义断言。
- **建议裁决**：首期实现 2-3 个与 BFCL 对齐的域（建议 filesystem+social 或 trading+vehicle），每个域给状态 schema 与终态断言函数；`env.config` 冻结域定义 shape；报告列"已实现域 vs BFCL 类别"映射，未实现类别写明由 B 方案兜底。
- **不解决的后果**：mt_fc A 方案覆盖率低，多数多轮样本只能回放（进一步放大 Q-18 倾向）。
- **关联**：R3、S5、CT C1-C3。

### Q-30 placeholder 比例与 env_state 形状的机检
- **优先级**：P1
- **owner**：env-builder（T4）
- **为什么重要**：CT C3（L35-49）列了 13 个 env_state 键，CT C6 要求 `gt_returns_source` 三态显式；目前 schema 的 `env_state` 是任意 dict（schema.py:52），形状错了 reward 会 KeyError 或在验收时才发现。
- **建议裁决**：T4 在 `tests/test_envs_*.py` 加"done 时 env_state 键集合 == CT C3 键集合（未用为 null）"断言；`placeholder` 比例加阈值告警（>5% 在报告里红字，构建脚本可 `--fail-on-placeholder`）；reward 侧对缺键给明确 SchemaError 而不是静默 0。
- **不解决的后果**：S5 静默降质；A3 表面通过。
- **关联**：K6、K4、K7。

---

## E 组 — Lead / 跨席位

### Q-31 memory 第一期边界
- **优先级**：P2（但必须显式）
- **owner**：Lead
- **为什么重要**：plan L70-72 明确 memory 是第二期可选，且"没有成规模的公开训练集"。BFCL v4 memory 占 agentic 40% 的一部分，容易被误认为必须覆盖。
- **建议裁决**：Phase 1 明确 out-of-scope：不实现 memory 工具、不造合成数据；只在 `reports/` 记录"评测跟踪项 + 第二期合成方案（memory_write/read/update 三工具）"；训练配比不含 memory 类。
- **不解决的后果**：seats 各自造 memory 数据/环境，范围失控且无验收口径。
- **关联**：§2 无 memory task_type（schema 未含）。

### Q-32 slime 四接口（§9 TODO）必须用户填
- **优先级**：P1（阻塞 A5）
- **owner**：用户（Lead 转达）
- **为什么重要**：plan L216-223 明确 prompt/label、rollout、reward、chat template 四项为 `TODO`，并要求"不要猜接口"。没有这四项，A5 smoke 与 A6 评测无法实现。
- **建议裁决**：用户提供 slime 仓库路径/接口文档/现有 reasoning-RL 配置样例；Lead 在 Phase 2 任务卡中按接口实现适配层（`agentic_rl_data/integrations/slime.py` 或明确 out-of-scope）。
- **不解决的后果**：A5 永远阻塞；reward/env 接口可能与训练框架错配，返工。
- **关联**：A5、A6、TEAM_BOARD D1。

### Q-33 评测版本/API key/额度与耗时预算
- **优先级**：P1（阻塞 A6）
- **owner**：用户 + Lead
- **为什么重要**：plan L202-203：BFCL 版本要确认；web search 评测需要 SerpAPI key 与额度（否则全零）；memory 评测预填充耗时。没有这些，A6 无法执行。
- **建议裁决**：用户确认 ①`bfcl-eval` pin 版本；②SerpAPI 账号与预算；③memory/agentic 评测可接受时长；Lead 把评测命令与成本写进 `reports/eval_plan.md`。
- **不解决的后果**：A6 不可判定；web search 分数会被误读为模型能力。
- **关联**：A6。

### Q-34 held-out 的去污染与 train/held-out 同源泄漏
- **优先级**：P1
- **owner**：data-researcher（T1）+ mixture-researcher（T2）
- **为什么重要**：plan 要求 held-out "不参与训练"，但没说 held-out 是否也要过去污染。若 held-out 里留着 BFCL 同源样本，验证分虚高；若 train 与 held-out 来自同一会话/同一 API 文档，近重复泄漏（DESIGN.md §2.5）。
- **建议裁决**：held-out 与 train 走**同一套** dedup/decontam；切分按来源会话组；报告给 held-out 的去污染命中数与 `train/held-out` 近似重复对数（应为 0）。
- **不解决的后果**：选 checkpoint 依据失真；A6/A4 数字不可信。
- **关联**：S7、A4、Q-13。

---

## 补充事实（T1 核实，2026-11-05 21:40，Lead 转发到 TEAM_BOARD L58-64）

以下事实改变若干问题的答案，必须在 T2/Lead 决策时使用：

1. **BFCL 版本**：PyPI 最新 `bfcl-eval==2026.3.23`，plan 写的 `2025.10.27.1` 已过时 → 去污染必须 pin 一个具体版本（Q-06）。HF/BFCL 数据在 T1 沙箱不可达，**A2 保持 `[ext]` 阻塞**。
2. **Search-R1 真实数据**：HF `RUC-NLPIR/FlashRAG_datasets` config=nq（plan 的 `PeterJinGo/search-r1` 数据集 ID 不准）→ loader ID 必须可配置、默认用核实值（Q-03/Q-10）。
3. **ASearcher**：仓库 `inclusionAI/ASearcher-train-data`，原作轨迹 **128 轮** → ≤15 轮过滤是硬门槛，不是"重点过滤"而是"大批量丢弃"，T2 的 search 25% 配额要按过滤后可用量重算（Q-09/Q-03）。
4. **Toucan-1.5M** 是首期 `mt_fc` **真实工具返回的主力**；FunReason-MT 等多为 placeholder → 若按 `placeholder_gt_returns: drop` 过滤，multi-turn 30% 配额可能严重不足（**配比 vs 回放质量的直接冲突，见 Q-38**）。
5. **tau-bench** 无现成轨迹（needs-synthesis）→ 不能当作 mt_fc 数据源计数；When2Call test 由 BFCL v2 Live 派生、**绝不入训练**。
6. **HF 卡片 license/条数一律 `[UNVERIFIED]`**（沙箱不可达）→ Q-02 的结论只能是"待用户核实"，不能写"已合规"；A2/A6 的 `[ext]` 判定不变。

---

## 新增问题（由 T1 核实事实 / 配置产物触发）

### Q-35 held-out 会话级切分依赖 `meta.group_id`，但该字段未入契约
- **优先级**：P1
- **owner**：Lead（schema/CT）+ T1（loader 写）+ T2（切分）
- **为什么重要**：`configs/mixture.yaml:90-91` 的 `group_aware: true` 声称"whole sessions (`meta.group_id`, or id when absent) land on one side only"，并引用 DESIGN.md §2.5。但 schema v2.1 的 `meta` 必填键只有 4 个（schema.py:124-125），`CONTRACTS.md` 也没有 `group_id`；loader 没有字段可写，T2 会**静默退化为按 task id 切分**，ToolACE/API-Bank 同会话多 turn 可能跨 train/held-out。
- **建议裁决**：Lead 在 schema/CONTRACTS 增可选 `meta.group_id: str|null`（loader 填会话 key，如原始 sample id 去 turn 后缀）；T1 在各 loader 填；T2 单测断言"同 group 只落一侧"；无 group_id 的来源在报告标 `[NO-GROUP]` 并说明泄漏风险。
- **不解决的后果**：held-out 指标虚高，A6 对比表与 checkpoint 选择失真；T2 的 `group_aware` 承诺无法验证。
- **关联**：K9、Q-13、Q-34、S7。

### Q-36 数据侧 `meta.gt_returns_source` 静态标记没有机检
- **优先级**：P1
- **owner**：Lead（schema 可选校验）+ T1（loader 写）+ T2（离线过滤）
- **为什么重要**：CONTRACTS C6 v2.1（L99-102）与 D11 要求 loader 在 `meta["gt_returns_source"]` ∈ {config,reference,gt_turns,placeholder} 静态声明，T2 不运行 env 就能过滤占位样本；但 schema 不校验该键、loader/T2 产物未出 → 目前只是文字承诺。且 `configs/mixture.yaml:141-150` 的 `placeholder_gt_returns: drop` 尚无可执行脚本。
- **建议裁决**：schema 对 `mt_fc` 增加 `meta.gt_returns_source` 枚举校验（或至少 CONTRACTS 冻结 loader 必填）；T2 的 build 脚本读取静态标记并在报告给 `placeholder_meta` / `placeholder_runtime_mismatch` 计数；T5 契约测试用合成 task 覆盖四值。
- **不解决的后果**：placeholder 样本静默进训练，`mt_fc` 的 0.8 结果项失真（Q-18），报告无法解释。
- **关联**：K8、Q-17、A4。

### Q-37 `abstain` 类在 mixture 里没有启用任何来源（与 plan §6 "单轮 + 拒答 20%"冲突）
- **优先级**：P1
- **owner**：mixture-researcher（T2）+ Lead 决策
- **为什么重要**：`configs/mixture.yaml` 的 `single_turn` 类只含 apigen/sealtools/toolace/apibank/hammer（L357-384），`when2call` 被 `enabled: false`（L336-343）→ 最终 mixture 中 `task_type=abstain` 的条数很可能为 0，但 plan L180 的 20% 明确含"拒答"，且 BFCL hallucination 10%（L28）依赖弃权行为。
- **建议裁决**：二选一并在 config/report 写明：(a) 启用 When2Call train（已核不可入 test）并给 single_turn 内 abstain 权重（如 5-10%，从 st_fc 权重挪）；(b) 明确"首期拒答由 hammer irrelevance 承担、abstain task_type 只做 reward/env 单测、不进训练"，并把该决定写进 TEAM_BOARD 决策记录。**不接受"配置里禁用但报告里假装覆盖"**。
- **不解决的后果**：训练集零拒答样本，BFCL hallucination 与"信息不足追问"能力无训练信号；A3 的 abstain 单测变成"有 reward 没数据"。
- **关联**：R7、Q-22、S7、A4。

### Q-38 配比 vs 回放质量：多轮 30% 配额可能被 placeholder 过滤打穿
- **优先级**：P0（配比可能不可达）
- **owner**：T2（重算配额）+ T1（报告 placeholder 率）+ Lead（裁决过滤策略）
- **为什么重要**：T1 核实 Toucan 是首期 `mt_fc` 真实返回的主力，FunReason-MT/APIGen-MT 多为 SFT 轨迹（工具返回是否可映射到 `reference.tool_returns` 未验证）。若 FunReason-MT（config 权重 45%、cap 1400）大面积 placeholder 且 `placeholder_gt_returns: drop`，multi_turn 30% 只能靠 Toucan 子采样（config cap 500）+ tau（300）+ apigen_mt（900）——上限约 1700 < 2400 目标；而 `on_unavailable: fail` 会让构建直接失败。
- **建议裁决**：①T1 首跑后把每源 `placeholder 比例 + 可映射真实返回条数` 写进 `reports/data_sources.md`；②T2 用"可用真实返回量"重算 multi_turn cap；③若不足，按优先级降权而非 fail：Toucan 可扩子采样（需真实返回）、FunReason-MT 允许降权保留（如 0.5×0.1 匹配项而非全 drop）并在报告给条数；④Lead 在 TEAM_BOARD 记"30% 是目标而非硬约束"的裁决。
- **不解决的后果**：A1 构建直接 fail 或名义 30% 实际靠重复采样；`mt_fc` 训练信号以调用匹配为主（Q-18 风险叠加）。
- **关联**：Q-17、Q-18、Q-03、S7、A1。

### Q-39 search_qa 的 loader env.config 与 SearchEnv 契约不一致（跨席位 P0）
- **优先级**：P0（25% 数据无法 rollout）
- **owner**：data-researcher（T1，推荐改 loader）/ env-builder（T4，备选改 env）
- **为什么重要**：`loaders/common.py:608-614` 的 `env_search()` 产出 `{"backend": "local_wikipedia_e5", "retriever": {"url":..., "topk":3}}`；`envs/search.py:1144-1209` 的 `_resolve_backend` 只接受 `inmemory|http|e5`、http 分支读 `url`/`top_k`。`build_env()` 会 raise `EnvError`，`search_r1`/`asearcher` 全部 rollout 失败。
- **建议裁决**：T1 改 `env_search()` 为 `{"backend":"http","url":retriever["url"],"top_k":retriever.get("topk",3),"max_turns":...}`；T4 可选兼容 `local_wikipedia_e5`→`e5` + `retriever` 嵌套键。契约测试 `test_search_loader_env_config_is_constructible` 已覆盖（当前红）。
- **不解决的后果**：search 类 25% 数据不可训练；A6 的 web search 对比无法解释。
- **关联**：S1、A6、Q-10、Q-27。

---

## 裁决跟踪

| 问题 | 需要谁在何时定 | 截止（建议） |
|---|---|---|
| Q-05 gt_turns shape | Lead（改 CT C6 + schema 注释） | ✅ 已由 schema v2.1 收窄（`tool_returns` 为规范来源，见 K7） |
| Q-18 replay_ok 语义 | Lead + T4 | 立即（T3 写 reward 前） |
| Q-26 timeout 0.8 | T4 + T3 | T4 写 mock_fc 前 |
| Q-19 双真相源 | T3 | 写第一个 reward 前（rewards/base.py 已建，audit 待做） |
| Q-20 score 范围 | Lead | ✅ 已修：`[0,1]` 强制（schema v2.1） |
| Q-11 N=8 区间 | Lead + T2 | ✅ T2 已定 band [0.125,0.875]（configs/mixture.yaml:64-70），等 A1 执行 |
| Q-01 去污染语料 | T1 | 写 decontam 前 |
| Q-31 memory | Lead | 本轮结束前（写入 TEAM_BOARD） |
| Q-35 meta.group_id | Lead + T1 | **立即**（T2 的 held-out 切分已按它设计） |
| Q-36 meta.gt_returns_source 机检 | Lead + T1 + T2 | 本轮结束前（loader 落地前） |
| Q-37 abstain 配比 | T2 + Lead | 本轮结束前（A4 报告前） |
| Q-38 placeholder vs 30% | T1（首跑数据）+ Lead | 用户首跑后第一轮 |
| Q-39 search env 契约不匹配 | T1/T4 | **立即**（search 数据不可训练，P0） |

> 本清单随 T1-T4 产物落地上调/合并；任何"已裁决"的条目在 `reports/review_r1.md` 里回填证据（文件:行）。critic 不修改他人实现。
