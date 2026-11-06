# 验收矩阵 — agentic_rl_data（plan §2 / §3 / §5 / §8 → 可判定行）

> Owner: critic（T5）。只做判定口径与缺口，不改实现。
> 权威依据：`C:\Users\Administrator\Downloads\agentic_rl_data_plan.md`（下称 plan，行号即该文件行号）。
> 契约依据：`agentic_rl_data/schema.py`（**v2.2**，Lead 独占）、`agentic_rl_data/CONTRACTS.md`（**v2** + v2.1 增补，Lead 维护）、`agentic_rl_data/README.md`。
> 快照时间：2026-11-06 09:35 (+08:00)；A4/S8 于本轮（report.py 交付后）复核，其余行维持 2026-11-05 21:45 快照。**本机 pwsh 不可用（0xC0000142），本矩阵全部「执行类」判定均为待执行，不代表已通过。**
> v1→v2.2 变更记录（critic 提议、Lead 已采纳并修入 schema.py）：F-01 `reference.tool_returns`、F-02 reference 7 键必填、score 冻结 `[0,1]`、`tools_signature` 排序、`meta` 4 键必填、`expected_turns`、`tools[].description` 校验、`tool_results` 子结构校验、`info` 不得为 null；**v2.2：`gt_turns` 元素必须 mapping、`meta.gt_returns_source` 对 mt_fc 强制四值枚举**。

## 0. 图例与判定原则

| 标记 | 含义 |
|---|---|
| ✅ 已有 | 产物已存在且 critic 已静态核对通过；不代表运行时通过 |
| 🟡 待产出 | T1–T4 尚未产出文件，或已产出但未核对 |
| 🔴 阻塞 | 本轮明确不做（D1/Phase 1）或依赖外部资源（slime/GPU/API key/BFCL 数据），当前无法判定 |
| ⚪ 延期 | plan 标注第二期/可选（如 memory），只跟踪不做 |
| ❌ 缺口 | 契约级缺陷：即使产物齐了按现有实现也过不了该条 |

**环境标记**（每条都标，防止把"静态能读"误报成"已验收"）：

- `[静]`：**无 shell 也能验** —— 文件存在性、代码走查、schema 调用、报告段落核对、行号引用。
- `[shell]`：**必须在有 shell 环境执行** —— pytest、脚本运行、sha256 对比、数据下载。
- `[ext]`：还需要外部资源 —— HF/tau-bench 数据、BFCL 测试数据（pip `bfcl-eval` 或 gorilla 仓库）、搜索 API key、GPU/slime。

**判定可复现原则**：每行的"判定方式"都是具体命令或具体文件/行断言；凡"看起来实现了"一律不算通过。

**当前总体状态（一句话）**：schema 已到 **v2.2**，F-01/F-02 + score 范围 + tools 排序 + `gt_turns` shape + mt_fc 静态 marker 全部修入（待静态复核）；T1/T2/T3/T4 主体文件已陆续产出（loaders 全家族 + rewards 7 类 + envs base/mock_fc/search/python/code_tests + scripts 多数），但本轮新增 **2 个跨席位 P0**（见 review_r1 §1/§6）：①search_qa loader 产出的 env.config 与 SearchEnv 契约不匹配（25% 数据无法 rollout）；②timeout 仍可拿 0.8 结果项（env+reward 双侧漏洞）。**§8 六条验收当前可判定通过 = 0 条（A1/A3 结构可静态核，A2/A5/A6 需 shell/外部资源）。**

---

## 1. plan §8 六条验收标准（硬验收）

| ID | 条目（plan 行号） | 判定方式 | 依赖席位 | 当前状态 | 缺口（具体、可修） | 验证环境 |
|---|---|---|---|---|---|---|
| **A1** | `scripts/build_mixture.py` 一条命令从原始数据产出 `mixture_v1`，**重复运行结果一致**（plan L207） | `[静]` ①`scripts/build_mixture.py` 存在且是单一入口；②`configs/mixture.yaml` 含 `seed`、五类占比、各源上限（plan L186）；③采样 RNG 显式 seed（不接受模块级 `random` 默认状态）；④输出全部经 `schema.dump_jsonl`。`[shell]` `python -m agentic_rl_data.scripts.build_mixture --config agentic_rl_data/configs/mixture.yaml --out data/mixture_v1` 跑两次 → `sha256sum data/mixture_v1/train.jsonl` 与 `heldout.jsonl` 两次相同 | T2 主，T1（loaders） | 🟡 待产出 | 未看到任何脚本。字节一致不靠 `dump_jsonl` 单独保证：任务迭代顺序（建议按 `id` 排序）、集合/字典迭代、浮点 `meta.pass_rate` 写入都必须确定；否则 A1 必失败 | `[静]+[shell]` |
| **A2** | 去污染报告：对 BFCL 测试集**命中数为 0**（命中的已剔除并有记录）（plan L208） | `[静]` ①`scripts/decontam.py` 存在；②对 **query** 与 **tools 定义**都做匹配（plan L127 明确两者）；③n-gram + 近似匹配两路，阈值可配置；④FunReason-MT 用最严档（plan L229）；⑤每条命中写 (task_id, matched_bfcl_id, score, 处理=drop)；⑥报告口径是「命中数=0 或 命中已 drop」，不是「未检查」。`[shell+ext]` 固定版本 BFCL 数据（`pip install bfcl-eval==<pin>` 或 gorilla 仓库快照，`[CONFIRM]` plan L202），跑 decontam 后报告 `bfcl_hits == 0` 且 drop 记录可逐条追溯 | T1 主，T5 复核 | 🟡 + `[ext]` | BFCL 数据来源 Q1 未定（TEAM_BOARD L93）；**bfcl-eval PyPI 最新 2026.3.23 ≠ plan 的 2025.10.27.1（T1 已核）** → 必须 pin 版本/快照，否则"0 命中"不可复现；只用一个 split 会给出「假 0」——至少覆盖 v4 的 web search / memory（agentic 40%）与 v3 multi-turn（30%）。**T1 的 decontam 已产出但仍有缺口（review_r1 §6.3）**：空索引会导致"假 0"（无 num_records==0 防护）、只做 n-gram 未做 config 声明的 Jaccard 0.90 近似、短 query 5-char n-gram + min_hits=1 易误杀；本沙箱 HF 不可达，BFCL 数据未落地前此条无法判定 | `[静]+[shell+ext]` |
| **A3** | 每个 `task_type` 的环境与 reward 都有单测，**gt 轨迹重放 reward 为满分**（plan L209） | `[静]` ①7 个 task_type × (reward 单测 + env 单测) 文件/用例存在，且含正例/反例/边界（plan L171）；②`tests/test_acceptance_contract.py` 覆盖非网络部分；③重放满分断言用 `math.isclose(score, max_score)` 而非 `== 1.0`。`[shell]` `python -m pytest agentic_rl_data/tests agentic_rl_data/rewards agentic_rl_data/envs -q` 全绿；`python -m agentic_rl_data.scripts.replay_check --limit 10` 输出 `passed=10/10` | T3 + T4 主，T1 replay 脚本 | 🟡 待产出 | ①「满分」定义缺失：`0.8+0.1+0.1` 的浮点求和未必精确等于 1.0，必须给 `MAX_SCORE` + 容差；②B 方案重放缺逐轮工具返回字段（见 §2 C6 与 review_r1 F-01），**A3 的重放部分在 schema 不变的情况下不可实现**；③env 单测要断言 `reset/step` 无状态泄漏（同一 task 两次 reset 输出一致） | `[静]+[shell]` |
| **A4** | `reports/data_report.md` 生成，含 §3 要求的**所有统计**（plan L210, L131） | `[静]` `scripts/report.py`（本轮交付）结构已核：§2 每来源加载/丢弃、§3 各步丢弃（round/dedup/decontam/passrate/replay）、§4 最终配比、§5 轮数/工具数/通过率/`gt_returns_source`/license/group 分布、§6 一致性检查、§7 [NOT RUN]/strict、§8 复现命令；分布全部从 `train/heldout.jsonl` 重算（`report.py:179-309`），缺失报告渲染 `[NOT RUN]` 不编数字（`report.py:680-785`）；`--strict` 规则见 `STRICT_RULES`（`report.py:70-76`）。`[shell]` 跑完整流水线后 `data_report.md` 自动生成，分布数字可与 JSONL 行数复核。 | T1（report.py）+ T2（配比统计） | 🟢 **实现已交付（静态复核通过；strict 有假通过向量，执行未验）** | 数字可追溯性 OK；但 strict 的 A2 语义错误（把 `total_dropped` 当 residual hits：正确剔除命中会失败、残缺 BFCL 语料可假通过，R2-01/P0）；strict 未要求 heldout/上游步骤/`filters_applied`（R2-02/P1）；replay `skipped/reward_used` 未查（R2-03/P1）；默认目录发现漏 `data/{decontaminated,deduped,filtered_rounds}`（R2-04/P1）；`effective_turns` 与 build_mixture 不一致（R2-05/P1）；stale report/sha 缺失/group overlap 假通过（R2-06..R2-08/P1）；3 个上游字段名不匹配（R2-09/P2）；详见 review_r1 §6.8 | `[静]+[shell]` |
| **A5** | 小规模 smoke-test 配置能在 **slime** 中跑通 rollout → reward → 更新一步（plan L211） | `[静]` 只可检查 smoke 配置模板/命令是否存在。`[shell+ext]` 需 slime 仓库、GPU、训练侧配置 | Lead + 用户；plan §9（L216-223）接口全是 `TODO` | 🔴 阻塞 | TEAM_BOARD D1（L62）明确 Phase 1 不接 slime；§9 的 prompt/label、rollout、reward、chat template 四个接口未定义。**本轮不应宣称通过；交付物只能给「接口清单 + 用户侧接入步骤」** | `[ext]` |
| **A6** | 训练后在评测点同时给出 BFCL v3 multi-turn、v4 agentic 与推理基准对比表（plan L212, L201-204） | `[静]` 只能检查评测脚本/表格模板是否存在。`[shell+ext]` 需训练 run + `bfcl-eval` 固定版本 + web search 的 SerpAPI key/额度 + memory 预填充（耗时） | 用户（训练负责人）+ Lead | 🔴 阻塞 | 无 checkpoint、无 BFCL 数据、无 API key；本轮不可判定。建议 Phase 1 产出「评测点清单 + 每列含义 + 命令模板」放 reports/eval_plan.md | `[ext]` |

---

## 2. plan §3 八个流水线步骤

| ID | 步骤（plan 行号） | 判定方式 | 依赖席位 | 当前状态 | 缺口 | 验证环境 |
|---|---|---|---|---|---|---|
| **S1** | Load：各来源 loader → 统一 schema → `data/raw_unified/{source}.jsonl`（plan L122） | `[静]` ①每个来源一个 `loaders/<source>.py`；②模块暴露 `SOURCE: str` + `load_tasks(cfg=None)`（schema.py:59-61）；③只产出过 `schema.validate_task` 的 task（合成 fixture 可静态验）；④loader 注册表可枚举（T2 需要）；⑤无网络也能跑 `--dry-run`/fixture 模式。`[shell]` `python -m agentic_rl_data.scripts.build_raw --config ...` 产出 `data/raw_unified/` 且每行 `validate_task` 为空 | T1 | 🟡 待产出 | 5 个现有源应复用 `examples/tool_rl/prepare_data.py` 的解析逻辑（TEAM_BOARD L53-54）；新增源（FunReason-MT/APIGen-MT/Tau/Toucan/ToolMind/Search-R1/ASearcher/ReTool 等）逐个落 loader；loader registry 接口未冻结（T2 软依赖） | `[静]+[shell]` |
| **S2** | 轮数过滤：丢弃 gt 轮数或预期交互 >15；ASearcher 重点过滤（plan L123） | `[静]` `scripts/filter_rounds.py` 存在；谓词读 `reference.gt_turns` 或轨迹长度；ASearcher 走独立阈值；丢弃原因写入 report。`[shell]` 跑过滤后所有 task 的 `len(gt_calls) <= meta.max_turns <= 15` | T1 | 🟡 待产出 | schema 的 `reference.gt_turns` 未校验（schema.py:204-242）；SFT 轨迹的"预期交互轮数"不在 schema 内，loader 必须自行记录（否则 ≤15 过滤只用 gt_calls 会低估）。**S2 与 §2 C6 的字段缺口是同一根问题** | `[静]+[shell]` |
| **S3** | 去重：来源内/来源间精确去重（规范化 query + tools 签名 hash）；近似去重（MinHash/embedding），重点 ToolMind vs xlam/ToolACE 与已有 5 源（plan L124-126） | `[静]` `scripts/dedup.py` 用 `schema.task_dedup_key` + `schema.text_shingles`；近似阈值/算法可配置并写进 report；报告给出各对来源的删除数。`[shell]` 跑后全库 `task_dedup_key` 无重复；抽样 200 对近似候选人工核对 | T1 | 🟡 待产出 | ①近似去重若无 embedding（离线环境）只能用 MinHash/Jaccard，阈值 plan 未给 → 需 T1 定默认+报告；（review P2：`tools_signature` 对工具顺序敏感会漏去重）；②去重后仍必须保留去污染（顺序 S3→S4，注意 S3 可能先合并掉污染样本，见 Q-15） | `[静]+[shell]` |
| **S4** | 去污染：BFCL v3/v4 query + 工具定义 n-gram/近似匹配；FunReason-MT 尤其严格；数学 prompt 对推理评测集去污染（plan L127） | 同 A2，并额外检查数学 prompt 的推理评测集去污染分支 | T1 | 🟡 + `[ext]` | 见 A2；数学 prompt 侧需要用户提供推理评测集清单（AIME 等）否则无法跑该分支 | `[静]+[shell+ext]` |
| **S5** | 工具可执行性校验：mock_fc 里 gt 轨迹必须完整重放并通过 reward（reward 满分），否则丢弃（plan L128） | `[静]` `scripts/replay_check.py` 存在；mock_fc env 实现重放；样例 task 断言 `score == full`（容差）。`[shell]` 对 `data/raw_unified/*.jsonl` 跑 replay_check，报告 pass/fail 计数与丢弃原因 | T4 主 + T3 + T1 | 🔴 契约阻塞 | **B 方案需要逐轮工具返回，schema 未提供字段**（review_r1 F-01）；A 方案需要 mock_fc 状态机与 API 定义（env.config 形状未冻结）。**S5 在 schema 不变情况下无法实现，需 Lead 先裁决 C6** | `[静]+[shell]` |
| **S6** | 通过率过滤：当前 checkpoint 每 prompt 采样 N=8（`[SUGGEST]`），保留 (0,1)，记录 `meta.pass_rate`（plan L129） | `[静]` ①`scripts/filter_by_passrate.py` 存在；②输入接口支持 `--pass-rates` 文件（D2/Lead 决策，TEAM_BOARD L70-71），rollout 耦合留 TODO 不猜；③写回 `meta.pass_rate`；④保留带宽可配。`[shell]` 用用户提供的 pass-rate 文件跑过滤，输出 `meta.pass_rate` ∈(0,1) | T2 主 | 🟡 待产出 | T2 已在 `configs/mixture.yaml:57-79` 定 band `[0.125, 0.875]`、`n_samples=8`、`missing=keep`、`smoothed=false` —— 与我们 Q-11 建议一致；但 `missing=keep` 意味着当前无 rollout 后端时不过滤，报告必须显式标注"未执行通过率过滤"，否则 A1/A4 的 "最终配比" 会被误读为已过滤 | `[静]+[shell]` |
| **S7** | 混合：按 §6 配比采样 → `data/mixture_v1/train.jsonl`；held-out 每类至少 200 条不参与训练（plan L130） | `[静]` ①`configs/mixture.yaml` 五类占比 20/30/25/15/10（plan L176-184）；②`build_mixture` 先切 held-out 再做训练采样并断言 id 不相交；③held-out 与 train 走同一去重/去污染。`[shell]` 输出两份 JSONL；用脚本统计 `train ∩ heldout = ∅`、每类 heldout ≥200 | T2 主 + T1 | 🟡 待产出 | T2 已在 config 定 `heldout.per_category=300`（5 个配比类 × 300 = 1500，占 total_train 8000 的 18.8%）、`group_aware=true`（configs/mixture.yaml:85-96）。**缺口**：①"每类"仍是 5 个配比类，不是 7 个 task_type（plan L130 未定义，见 Q-13）；②`group_aware` 依赖 `meta.group_id`，但 schema v2.1 与 CONTRACTS 都**没有该字段**（只允许额外键，loader 不会自动填）→ 实际会退化为按 `id` 切分，ToolACE/API-Bank 同会话多 turn 仍可能跨 train/held-out 泄漏（见 K9）；③`oversample: 2` 与 caps 的交互无单测 | `[静]+[shell]` |
| **S8** | 报告：`reports/data_report.md` 含每来源加载数、各步丢弃数与原因、最终配比、轮数/工具数分布、通过率分布（plan L131） | 同 A4（A4 是 S8 的验收口径）；`scripts/report.py` 已交付，渲染 8 个小节并给出每节来源文件 | T1 + T2 | 🟢 **实现已交付（静态复核通过；strict 假通过项待修）** | 同 A4；字段级修法见 review_r1 §6.8 R2-01..R2-12；`[shell]` 执行仍待用户 | `[静]+[shell]` |

---

## 3. plan §2 统一 schema 要求

> 全部以 `schema.py` 函数行为为准；`[静]` 判定 = 读实现 + 用合成 task 调 `validate_task`，不需要 shell。

| ID | 要求（plan 行号） | 判定方式（schema.py 行号） | 当前状态 | 缺口 | 验证环境 |
|---|---|---|---|---|---|
| **C1** | 一行一 JSONL task；8 个固定键 `id/source/task_type/messages/tools/env/reference/meta`（plan L78-96） | `validate_task` 必填键检查（schema.py:81-90, 140-143） | ✅ | 不拒绝未知键（plan L78 说"键固定"；影响小，P2） | `[静]` |
| **C2** | `task_type ∈ {mt_fc, st_fc, irrelevance, search_qa, math_ci, code_exec, abstain}`（plan L84） | `TASK_TYPES` 枚举（schema.py:76-78, 152-154） | ✅ | — | `[静]` |
| **C3** | `messages: [{role: system\|user, content}]`，非空（plan L85） | schema.py:156-168 | ✅ | 未要求 system 至多一条/置首（plan 未强制，P2 可选） | `[静]` |
| **C4** | `tools` 为 OpenAI 风格 JSON schema，含 `name/description/parameters`（plan L99） | schema.py:203-223（name 非空/唯一、parameters mapping、description str\|null） | ✅ v2.1 已修（待静态复核） | 缺 description 仍合法（`None` 允许，plan 未强制必填）；可接受 | `[静]` |
| **C5** | `gt_calls` 按轮分组（list of list），并行同组、顺序无关；`[]`=无需工具，**禁止 None**（plan L100, schema.py:31-35） | schema.py:208-225；`gt_call_groups` schema.py:456-474 | ✅ | — | `[静]` |
| **C6** | **缺失字段显式置 null，禁止静默丢样本**（plan L101） | schema.py:113-122（`REQUIRED_REFERENCE_KEYS`）+ 244-248 强制 7 键存在；`final_answers/final_state/gt_turns/tool_returns` 类型均已校验（272-310） | ✅ v2.1 已修（待静态复核） | `gt_turns` 只校验"list"，元素 shape 仍未冻结（K7）；`final_answer` 非 search/math 类可 null，按 task_type 的语义必填仍未强制（由 loader 单测承担） | `[静]` |
| **C7** | 回放式 mock_fc（B 方案）所需的逐轮工具返回（plan L144, L128） | schema.py:26（`tool_returns: [[{name,content,ok}], ...] | None`）+ 289-311 校验；CONTRACTS C6 v2 定为回放规范来源（`config → reference.tool_returns → gt_turns → placeholder`） | ✅ v2.1 已修（待静态复核） | 元素 `ok` 类型未校验（只校验 name/content）；conftest 无；`tool_returns` 与 `gt_calls` 的轮次对齐（同长）未校验 → 建议 loader 单测断言 `len(tool_returns)==len(gt_calls)` | `[静]` |
| **C8** | `env.type ∈ {none, mock_fc, search, python, code_tests}` 且与 task_type 匹配（plan L87；schema.py:94-103） | `ENV_FOR_TASK_TYPE`（schema.py:95-103, 189-202） | ✅ | 映射理由未写进 README（如 st_fc 为何允许 mock_fc）；`irrelevance` 仅允许 `none` 是否过严需 T3 确认 | `[静]` |
| **C9** | `meta.max_turns` 正整数；课程 5→10→15（plan L94, L192） | schema.py:319-321；`meta.expected_turns` 可选正整数（331-333）；CT C9 冻结 `effective_turns = max(len(gt_calls), expected_turns or 0, len(gt_turns or []))` | ✅ v2.1 已修（待静态复核） | schema 仍未校验 `effective_turns <= max_turns`（CT C9 是 S2 脚本职责）；loader 必须填 `expected_turns`/`gt_turns` 否则过滤低估（Q-05/Q-09） | `[静]` |
| **C10** | `meta.pass_rate: null\|[0,1]`（plan L94） | schema.py:238-240 | ✅ | — | `[静]` |
| **C11** | `meta.license`、`meta.difficulty` 记录（plan L94, L230） | schema.py:124-125（`REQUIRED_META_KEYS`）+ 316-330 强制 4 键存在并校验类型（`difficulty` number\|null、`license` str\|null） | ✅ v2.1 已修（待静态复核） | `license` 语义未归一化（短码约定属 T1，CT C10 要求），验收只能查"键在、类型对、非 null 或标 UNVERIFIED" | `[静]` |
| **C12** | 重复运行字节一致（plan L186, §8 A1） | `dump_jsonl(sort_keys=True, ensure_ascii=False)`（schema.py:456-477）；T2 config 用 `sha256(seed\x1f salt\x1f task_id)` 排名选择（configs/mixture.yaml:26-30） | ✅ v2.1（工具）+ 🟡（调用方） | 仍需 `build_mixture` 按 `id` 排序输出并写 `manifest_sha256`；`dump_jsonl` 单独不保证任务顺序 | `[静]` |
| **C13** | 精确去重 key = 规范化 query + tools 签名（plan L125） | `task_dedup_key`（schema.py:543-551）；`tools_signature` 已按 name 排序（526-540） | ✅ v2.1 已修（待静态复核） | tools 重排已不再产生不同 hash；但 dedup 不区分 task_type/source（同 query+tools 的 st_fc 与 mt_fc 会被视为重复——可能是期望，需 T1 在报告说明） | `[静]` |
| **C14** | 近似去重可用（MinHash/embedding）（plan L126） | `text_shingles`（schema.py:438-449） | ✅（原料） | 只给 shingles，无 MinHash/Jaccard/阈值；T1 需实现并给默认阈值（P1 验收缺口） | `[静]` |
| **C15** | 轨迹归一化结构 `turns/final_answer/num_turns/done/env_state`（schema.py:41-57） | `validate_trajectory`（schema.py:345-403）：turns/assistant/tool_calls/tool_results 子结构（381-390） | ✅ v2.1 已修（待静态复核） | `tool_result.ok`（bool）与 `id` 类型未校验（390 只查 name/content）；`env_state` 形状由 CONTRACTS C3 管，schema 不校验（K6） | `[静]` |
| **C16** | reward 返回 `{"score": float, "info": {...}}`（plan L171） | `validate_reward_result(result, *, min_score, max_score)`（schema.py:412-435）：score 有限 + 冻结 `[0,1]`，`info` 必须 mapping | ✅ v2.1 已修（待静态复核） | 若 reward 原始分越界，`rewards.base.make_result` 会**静默 clamp** 后再校验（rewards/base.py:819-827, 850-866）→ 越界 bug不会报警（P2，见 review R-06） | `[静]` |
| **C17** | 每个来源 loader 只吐 schema-valid task（README L34-35） | 各 loader + 合成 fixture 调 `validate_task` | 🟡 待产出 | 无 loader 产物；注册表接口未冻结 | `[静]`（无网络 fixture） |

---

## 4. plan §5 奖励要求（每 task_type 一行）

| ID | task_type / 要求（plan 行号） | 判定方式 | 依赖席位 | 当前状态 | 缺口 | 验证环境 |
|---|---|---|---|---|---|---|
| **R1** | `st_fc`：函数名 + 参数 AST 匹配（并行顺序无关）1/0，格式合法门控（plan L162） | `[静]` 单测覆盖：精确匹配/参数类型错/并行调用顺序置换/空参数/多余调用；实现不做字符串全文匹配。`[shell]` pytest | T3 | 🟡 待产出 | "AST 匹配"的数值/字符串/嵌套语义未定义 → 必须由单测冻结行为；格式门控是否乘到 score 未说明 | `[静]+[shell]` |
| **R2** | `irrelevance`：未调用任何工具=1，否则 0（plan L163） | 单测：无调用=1、调用=0、有 tool_results 但无 assistant call、tools 列表为空 | T3 | 🟡 待产出 | 与 `abstain` 的区分（正确追问 vs 不调用）需定义；`irrelevance` env 强制 `none` 时调用工具是否算 0 需固定 | `[静]+[shell]` |
| **R3** | `mt_fc`：终态/结果正确 0.8；逐轮匹配 0.1；格式合法 0.1（plan L164） | `[静]` ①权重常量可查且和=1.0±1e-9；②终态项**不是**"调用匹配"的别名（走 env 终态或 final_answer），否则违反 plan L158/L233；③单测：仅终态对/仅逐轮对/全错。`[shell]` replay 满分 | T3 + T4 | 🟡 待产出 | **B 方案缺 tool_returns（F-01）**；A 方案终态判定依赖 mock_fc state 接口；回放式容易退化成功利"调用准确性"，需消融证据（Q-01） | `[静]+[shell]` |
| **R4** | `search_qa`：规范化 EM/F1 对比参考答案（A²Search 可用多答案 F1）（plan L165） | 单测：EM/F1 边界（空答案、单 token、多答案、冠词/大小写/标点规范化）；多答案输入路径 | T3 | 🟡 待产出 | schema 的 `reference.final_answer` 是**单字符串**，多答案需约定编码（JSON list 字符串或新增字段，Q-18） | `[静]+[shell]` |
| **R5** | `math_ci`：复用已有推理 RL 数学验证器（答案等价判断）1/0（plan L166） | `[静]` reward 只调用注入的 verifier（不复制 `\boxed{}` 逻辑），带合成等价/不等价用例 | T3 + 用户（提供 verifier 接口） | 🟡 待产出 | verifier 在用户现有 pipeline，接口未定（§9 同样不猜）→ 需用户给 callable/路径；本轮只能写适配层 + TODO | `[静]+[shell+ext]` |
| **R6** | `code_exec`：单测全部通过=1，否则 0（plan L167；§4.4 不暴露答案代码 L152） | 单测：全过/部分过/编译错/超时/infra 失败（infra 失败不应算 0 而应标记 invalid）；reward 只读 env 的结果字段 | T3 + T4 | 🟡 待产出 | env 结果字段 shapes 未冻结（`passed/total/error/timeout`）；泄漏要求需 env 侧保证（Q-07） | `[静]+[shell]` |
| **R7** | `abstain`：正确追问/拒答=1（plan L168） | 单测：追问/声明信息不足/拒答/直接乱答；判定规则（关键词 vs 标注 vs LLM-judge）需固定 | T3 | 🟡 待产出 | 判定规则未定义且不可用 LLM-judge（DESIGN.md:9 已否决）；`gt_calls=[]` 无法区分 irrelevance/abstain，需额外标注或规则（Q-17） | `[静]+[shell]` |
| **R8** | 每个 reward 是纯函数 `reward(task, trajectory) -> {...}`，附正/反/边界单测（plan L171） | `[静]` 模块级无可变全局状态（无缓存/计数器）；`validate_reward_result` 通过；7 个类型各 ≥3 用例；`[shell]` pytest | T3 | 🟡 待产出 | 无产物 | `[静]+[shell]` |
| **R9** | 记录各项奖励分布，防奖励被钻空子（plan L172） | `[静]` reward `info` 暴露可聚合的分量（如 `terminal/format/call_match`），`reports/data_report.md` 或训练日志有分布段 | T3 + T1（report）/Lead（训练侧） | 🟡 待产出 | 分布记录机制无归属；训练侧（slime）才能真采分布 → 本轮只能约定 info 字段结构 | `[静]+[shell]` |
| **R10** | 总原则：结果为主，格式/调用合法性只占小权重（plan L158, L233） | `[静]` 跨 reward 检查权重：`mt_fc` 0.8 结果项 > 0.1+0.1 辅助；`st_fc` 无"调用合法性"主导；score 取值范围与 schema 一致 | T3 + Lead（schema 冻结） | 🟡 待产出 | **score 范围未冻结（Q-12）**；若沿用 `examples/tool_rl` 的负分方案，schema 与 plan 都要写清楚，否则 T3 与 T2/DAPO 的假设不一致 | `[静]` |

---

## 5. 汇总：哪些"无 shell 也能验"，哪些必须有 shell

### 5.1 无 shell 可验（本轮必须先自查，不能拿"环境限制"当借口）

1. **schema 契约**：用合成 task 逐条调 `validate_task` / `validate_reward_result` / `validate_trajectory`（tests/test_acceptance_contract.py 已写，未执行）。
2. **代码走查类**：loader 暴露 `SOURCE`/`load_tasks`；reward 无全局状态；env `reset/step` 无模块级 state；decontam 同时匹配 query 与 tools；build_mixture seed 显式；`dump_jsonl` 使用；丢弃原因落报告。
3. **报告段落**：data_report.md 6 项统计有无；license 表是否逐源；数字是否有 `[EST]`/出处。
4. **本轮已核对项**：`schema.py` 的 C1/C2/C3/C5/C8/C9/C10/C12/C13/C15/C16；详见 review_r1.md 的"已核对项"。

### 5.2 必须在有 shell 环境执行（用户侧命令，待流水线文件落地后按实际 CLI 定稿）

```powershell
cd D:\lhy\verl\verl-toolRL
# 1) 契约测试（先跑这个，任何红都阻断后续）
python -m pytest agentic_rl_data/tests/test_acceptance_contract.py -q
# 2) 全量单测
python -m pytest agentic_rl_data -q
# 3) S1 Load（CLI 以 T1 最终实现为准）
python -m agentic_rl_data.scripts.build_raw --config agentic_rl_data/configs/mixture.yaml
# 4) S2-S6 流水线，再跑 S7 两次并对比字节
python -m agentic_rl_data.scripts.build_mixture --config agentic_rl_data/configs/mixture.yaml --out data/mixture_v1
Get-FileHash data/mixture_v1\train.jsonl -Algorithm SHA256
Get-FileHash data/mixture_v1\train.jsonl -Algorithm SHA256   # 第二次构建后再取一次，必须相同
# 5) A2 去污染 0 命中（需先落地 BFCL 数据）
python -m agentic_rl_data.scripts.decontam --bfcl-dir <bfcl-data> --in data/raw_unified --report reports/contamination_report.json
# 6) A3 重放满分
python -m agentic_rl_data.scripts.replay_check --limit 10
# 7) A4 报告
python -m agentic_rl_data.scripts.report --out reports/data_report.md
```

### 5.3 本轮不可能通过（🔴），不要出"已验收"结论

- **A5**（slime smoke）与 **A6**（训练后评测对比）：依赖 §9 接口、GPU、BFCL eval 数据、SerpAPI key。本轮交付只能是「接入清单 + 命令模板」。
- **A2** 的"0 命中"在 BFCL 数据未落地前不可判定；静态只能验匹配逻辑存在。
- **A3** 的重放满分部分在 C7（tool_returns）未裁决前不可实现。

---

## 6. CONTRACTS v1 跨席位核对项（Lead 2026-11-05 21:20 冻结）

> 依据 `agentic_rl_data/CONTRACTS.md`（下称 CT，行号即该文件行号）。这些是**跨席位**约束，任何一条不满足即算验收失败；T3/T4/T1/T2 都受影响。

| ID | 条目（CT 行号） | 判定方式 | 依赖席位 | 当前状态 | 缺口 | 验证环境 |
|---|---|---|---|---|---|---|
| **K1** | reward **只读** `trajectory`，不得访问 env 实例 / env 私有缓存 / 全局可变状态（CT C3 L51-52；双真相源禁令） | `[静]` 逐个 reward 审查 import：不得 `from agentic_rl_data.envs...` 取模块级对象；不得接受 env 参数；不得读写模块级全局；`[shell]` 构造一个 trajectory 后修改原 env 实例状态，再调用 reward，结果不变（隔离测试） | T3 主（T4 提供 env_state） | 🟡 待产出 | 目前 reward 未产出；**审计必须逐个文件做**，不能抽样 | `[静]+[shell]` |
| **K2** | `mt_fc` 的 `0.8 × 结果项 + 0.1 × 逐轮匹配 + 0.1 × 格式`，**结果项严格占主导**（CT C4 L67-68；plan §10 L233） | `[静]` 权重常量可查；断言 `w_result == 0.8` 且 `w_result > w_match + w_format`；结果项读取 `env_state.goal_satisfied`（A）或 `env_state.replay_ok`（B），**不是**逐调用匹配的别名；`[shell]` 单测：仅逐轮全匹配但终态失败 → 总分 < 0.2（即拿不到 0.8） | T3 + T4 | 🟡 待产出 | 若 B 方案把 `replay_ok` 等同于"整条轨迹匹配"再叠加 0.8，实际上仍是调用准确性主导——replay_ok 必须由 env 终态/结果定义，不能只看 gt_calls 逐条匹配。需 Lead 明确 replay_ok 语义（Q-01/Q-21） | `[静]+[shell]` |
| **K3** | `code_exec` reward **唯一**读 `trajectory.env_state["code_tests"]`；`info["tests"]` 仅供框架日志，reward 不得读（CT v2 C5 L74-84；D9） | `[静]` grep reward 源码：无 `info["tests"]` / `info.get("tests")` 访问；`all_passed` 与 `passed==total` 一致；`[shell]` 构造矛盾输入（code_tests 全过、info.tests 全挂）→ 分数跟 code_tests 走 | T3 + T4 | 🟡 待产出 | CT v1 曾写 `extras.tests`，TEAM_BOARD D6（L75）仍留旧口径，D9 已迁 `env_state["code_tests"]` → **文档内部不一致**（P2，Lead 需改 D6）；T4 尚未产出 code_tests env 与单测 | `[静]+[shell]` |
| **K4** | 回放降质必须显式：`extras["gt_returns_source"] ∈ {config, reference, gt_turns, placeholder}`；placeholder 需 `info.degraded_replay=True` 且**不加分**；T1/T2 统计 placeholder 比例，`mt_fc` 过滤或降权，报告给过滤前后条数（CT C6 L86-102、D11） | `[静]` ①env 四分支在代码中（envs/mock_fc.py:239-273）；②reward 读 placeholder 并在 info 标记；③T2 报告 `placeholder_before/after`；④placeholder 的 0.8 结果项**不得**发放；`[shell]` 构造 placeholder task + 全匹配轨迹 → 分数 ≤0.2 | T4（写）/ T3（读标记）/ T1+T2（统计+过滤） | ⚠️ 部分已有，**发现 P1** | env 四分支已实现；**但 rewards/mt_fc.py:184 只把 `degraded_replay` 写进 info，仍按 `replay_ok` 发 0.8**（review T3-02，P1 → 建议 placeholder 结果项强制 0 或封顶 0.2）；T2 的 `gt_returns_source()` 未读 `meta` 静态标记（review T2-02，P1） | `[静]+[shell]` |
| **K5** | 超限（timeout）`done=True, info["timeout"]=True`，**不额外扣分**，但 reward 侧只决定"不给 0.8 主项"（CT C1 L20-21；plan L139"超限视为失败"） | `[静]` reward 在 `env_state.timeout == True` 时结果项**不可能**贡献 0.8；env 在 timeout 时 `goal_satisfied/replay_ok` 必须为 false/null。`[shell]` timeout 但其余匹配良好 → score ≤0.2 | T3 + T4 | ⚠️ 发现 P0 漏洞 | **双侧都没做**：`envs/mock_fc.py:1050-1067`（stateful）在 timeout 后仍按 goal checks 报 `goal_satisfied`；`envs/base.py:414-437` 只置 timeout 标记；`rewards/mt_fc.py:144-160` 完全不读 `timeout` → timeout 轨迹可拿 0.8（review T3-02/T4-01，P0） | `[静]+[shell]` |
| **K6** | `trajectory.env_state` 形状符合 CT C3（type/impl/goal_satisfied/replay_ok/failed_turn/matched_turns/gt_turns/checks/state/rendered/num_turns/timeout/code_tests/extras 全键在，未用为 null） | `[静]` `make_env_state`（envs/base.py:250-283）键齐全；各 env `_env_state_fields` 填充；`[shell]` 7 类 env done 快照各 1 例断言键集合 | T4 | 🟡 部分已有 | `envs/base.py` 与 `envs/mock_fc.py` 已产；search/python/code_tests 未产；schema 不校验 env_state（任意 dict），需 T4 单测断言键集合（契约测试已写，未执行） | `[静]+[shell]` |
| **K7** | 回放返回来源优先级 `env.config["gt_returns"] → reference.tool_returns → reference.gt_turns[*].tool_results/results/return → placeholder`（CT C6 L86-95） | `[静]` env 代码按此优先级（envs/mock_fc.py:239-273）；`gt_turns` 元素必须 mapping（schema.py:289-297）；CONTRACTS C6 补 `{"step","assistant","tool_results"}` 形状示例 | T4（解析）/ T1（写入） | ✅ v2.2 已修（待静态复核） | `reference.tool_returns` 与 `gt_calls` 轮数对齐未校验（loader 单测承担）；`mock_fc._inject_returns` 缺条目时静默 `{"result":"ok"}`（review T4-03，P1） | `[静]` |
| **K8** | 数据侧静态声明 `meta["gt_returns_source"]` ∈ {config, reference, gt_turns, placeholder}；`mt_fc` 强制必填；与 env runtime `extras["gt_returns_source"]` 双标记，runtime 为准、不一致写 warning（CT C6 v2.1 L99-102、D11） | `[静]` schema.py:129 枚举 + 343-352（mt_fc 必填校验）；loader 写入（loaders/common.py:997-1001）；T2 离线过滤必须**先读 meta** 并统计不一致。`[shell]` 构造 meta=reference 但 env 落 placeholder → stats mismatch 计数 | T1（loader 写）+ T2（离线过滤）+ T4（runtime 写） | ⚠️ 契约 v2.2 已修，**实现仍红** | schema 已强制，但 `loaders/common.py:999` 写的是复数 `"references"`（不在枚举内）→ 所有 mt_fc 任务被 `validate_task` 拒绝，`load_mt_tasks_from_records` 用 `except Exception: logger.warning; continue` 静默吞掉（loaders/common.py:1019-1021）→ **loader 产出 0 条 mt_fc**（P0，review T1-01）。T2 的 `build_mixture.gt_returns_source()`（scripts/build_mixture.py:763-781）不读 meta 静态标记（P1，T2 修改中） | `[静]+[shell]` |
| **K9** | held-out 的会话级切分依赖 `meta.group_id`（configs/mixture.yaml:90-91 注释 "meta.group_id, or id when absent"） | `[静]` schema.py 的 `meta` 没有 `group_id` 字段（REQUIRED_META_KEYS 只有 4 键，额外键不校验）；T1 loader 是否写 `meta.group_id` 无契约；T2 的 `group_aware` 会静默退化为按 task `id` 切 | T2 + T1 + Lead（契约） | ❌ 契约缺口（P1） | 没有 `meta.group_id` 的冻结定义，T2 的"整组落一侧"承诺无法验证；ToolACE/API-Bank 同会话多 turn（DESIGN.md §2.5）可能同时出现在 train 与 held-out → held-out 分数虚高（Q-34/Q-13）。修法：Lead 在 CT/schema 增可选 `meta.group_id: str|null`（loader 填写会话 key），T2 单测断言同 group 只落一侧 | `[静]+[shell]` |

---

## 7. 本矩阵的维护规则

- T1–T4 每产出一个文件，critic 在 `reports/review_r1.md` 追加对应行判决，并更新本矩阵对应行的状态（`🟡→✅/❌`）。
- 状态升级为 ✅ 必须同时满足：`[静]` 判定全过，且给出将执行的 `[shell]` 命令与预期断言；否则最多 `🟡`。
- 任何"外部位子"（A5/A6/A2-BFCL）不得由内部席位自行标 ✅。
