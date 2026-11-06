# Agentic RL 首期数据配比分析（mixture_analysis）

> 席位：mixture-researcher（T2）· 权威规格：`C:\Users\Administrator\Downloads\agentic_rl_data_plan.md`（§3/§5/§6/§7/§8）
> 上游契约：`agentic_rl_data/schema.py` v2、`agentic_rl_data/CONTRACTS.md` v2（C6/D11、C9、C10）
> 产物：`agentic_rl_data/configs/mixture.yaml`、`agentic_rl_data/scripts/build_mixture.py`、
> `agentic_rl_data/scripts/filter_by_passrate.py`、`agentic_rl_data/tests/test_mixture.py`
>
> 状态声明：**本机 pwsh 不可用（0xC0000142），python/pytest 未执行**。文中所有数字分三类：
> `[PLAN]`=权威规格给出的值；`[REPO]`=仓库既有证据；`[EST]`=估算（估算口径写在括号里）；
> 未核实的数据集条数/许可证标 `[CONFIRM]`，由 T1 `reports/data_sources.md` 为唯一真相源，
> 本文按 T1 2026-11-05 的核实结论（见 §1.2）更新；T1 的结论变化时本文件与 config 同步改。

---

## 0. 结论摘要（TL;DR）

1. **五类配比（plan §6）20/30/25/15/10 不是拍脑袋**：它把 BFCL v4 的权重面和"已有数据现状"
   做了显式对齐——单轮类**故意欠配**（0.75 份数据 / 1 份 BFCL 权重），多轮 1:1，搜索类**超配**
   （1.25 倍，补 memory 空缺与工具格式冷启动），数学/代码 25% 作为推理保锚与多轮闭环迁移（§2）。
2. **首期多轮 30% 大概率保不住**：T1 核实 FunReason-MT / APIGen-MT 以 placeholder 为主、tau-bench
   无现成轨迹，真实返回主力是 Toucan 子集。`placeholder_gt_returns=drop` 时多轮可用上限约
   **900–2325 条**（取决于 placeholder 比例与 T1 首跑统计），不足 2400 → 缺口被其他类吸收
   （§4 敏感性表）。config 给出 `drop` / `keep + placeholder_keep_share=0.25 + 降权 0.5` 两套配额。
3. **单轮数据过多会伤多轮规划**有机制解释与量纲：按 token 计单轮只占约 **6.0%** 梯度 token，
   但按"样本/提示组"计占 20%；GRPO 组内归一化下，1 轮样本与 6 轮样本同权，等价于把
   "一步命中"先验放大到与完整交互同等地位（§3）。仓库证据：`examples/tool_rl/DESIGN.md`
   §1/§2.2/§2.4/§3.1 的弃权塑造、25% 负样本、逐步澄清 hint，都是在对抗单轮化。
4. **通过率过滤**：硬规则 (0,1)，推荐带 `[1/N, 1−1/N]`；N=8 是"全对/全错探测器"而非难度估计器
   （真实 p=5% 的任务有 **66.3%** 概率因 k=0 被误删），难度分层请用 N=16/32 或二阶段复筛（§6）。
5. **held-out 口径 = 5 个配比类 × 300 = 1500 条**（不是 7 个 task_type × 200）；单类 p=0.5 时
   95% CI ±5.7pt，全量 ±2.5pt。会话级切分依赖 `meta.group_id/group_size`，二者**尚未进 schema**
   → 标 **[NO-GROUP]**，当前为 best-effort（§7）。

---

## 1. 数据盘点（数字依据）

### 1.1 已有单轮数据（plan §0，精确值）

| source | 原始数据 | 条数 [PLAN] | 本配比可用的 train 上限（cap） | 说明 |
|---|---|---:|---:|---|
| apigen | Salesforce/xlam-function-calling-60k | 4900 | 550 | 单轮主力，下采样到 9–12% |
| sealtools | Seal-Tools | 4873 | 550 | 含孪生工具，奖励侧 any-of |
| toolace | Team-ACE/ToolACE | 4033 | 400 | 会话多 turn，需按会话切分 |
| apibank | API-Bank + 5 干扰工具 | 161 | 100 | 池子最小，cap 100 > 72 目标 |
| hammer | xlam-irrelevance-7.5k | 2980 | 500 | 只贡献 irrelevance，占单轮类 25% |
| **合计** | | **16947** | | 只取 1600（**9.4%**）+ held 300 |

### 1.2 新增来源（T1 核实结论，2026-11-05）

| 来源 | 类别 | 条数 / 可用性 | 许可证 | 首期结论 |
|---|---|---|---|---|
| FunReason-MT | multi | ~16k [PLAN]，`gt_returns_source` 以 placeholder 为主 [T1] | [UNVERIFIED] | 保留但依赖 placeholder 策略 |
| APIGen-MT-5k | multi | 5k [PLAN]，多为 placeholder [T1] | [UNVERIFIED] | 同上 |
| Toucan-1.5M | multi | 1.5M [PLAN]；**真实返回主力** [T1] | [UNVERIFIED] | 子采样 4000，cap 900 |
| tau-bench train | multi | **无现成轨迹，需合成** [T1] | — | **首期不进配比**（config 中 disabled） |
| ToolMind | multi | ~110k [PLAN]，与 xlam/ToolACE 重叠 [PLAN] | [UNVERIFIED] | 首期不进 v1（去重/去污染后再议） |
| Search-R1 | search | 实际数据是 HF `RUC-NLPIR/FlashRAG_datasets` config=nq [T1]；NQ+HotpotQA 训练集约 17 万 [EST, 文档 79k+90k] | [UNVERIFIED] | search 主力，≤8 轮 |
| ASearcher | search | 官方 `inclusionAI/ASearcher-train-data`；qa 对无逐题轮数标注 [T1]；原 RL 轨迹 128 轮 [T1] | [UNVERIFIED] | 静态 ≤15 过滤在数据侧**不可判定**（无 expected_turns）；由 runtime search env 8 轮截断 → 权重降到 0.25、cap 600 |
| A²Search | search | 可选，需多答案 F1 | [UNVERIFIED] | 首期 disabled |
| 已有推理数学池 | math_ci | 用户现有池，条数 [CONFIRM] | 用户侧 | 只挂 python 工具，分布不变 |
| SimpleRL Math3-5 | math_ci | [EST] 未核实 | [UNVERIFIED] | 权重 10% |
| ReTool-SFT | math_ci | ~2k [PLAN] | [UNVERIFIED] | 仅冷启动/格式，**不进 RL 主力**（disabled） |
| code_exec | apps / codecontests / taco | [EST] 未核实；单测质量 [CONFIRM] | [UNVERIFIED] | [SUGGEST] 整类；T1 单一 loader `code_exec`（kind 选择数据）；Q4 未决 |
| When2Call | abstain | 只有 train 可用；test 由 BFCL v2 Live 派生、**禁入训练** [T1] | [UNVERIFIED] | 首期 disabled（见 §5） |
| BFCL v3/v4 | 评测 | `bfcl-eval==2026.3.23`（T1 核实，文档版本过时） | 评测用 | 只做去污染，不训练 |

> 条数/许可需用户在有 shell + HF 可达的环境复核（命令见 §10）：
> `python -c "from datasets import load_dataset; ds=load_dataset('Bingguang/FunReason-MT', split='train'); print(len(ds))"`
> 等。T1 结论若变化，`sources.*.cap/required` 随之改，**不需要改脚本**。

---

## 2. BFCL v4 权重分解 → 配比依据（核心数字）

BFCL v4 总分 = Agentic 40%（web search + memory，按 20/20 拆）+ Multi-Turn 30% + Live 10% + Non-Live 10% + Hallucination 10%（`[PLAN]` §0）。

| BFCL 维度 | 权重 | 本配比覆盖类别 | 该类占 train | 占 train 折算 | "每 1 份权重配多少训练数据" | 判断 |
|---|---:|---|---:|---:|---:|---|
| Multi-Turn | 30% | `multi_turn` | 30% | 30% | **1.00** | 1:1，缺数据所以不给折扣 |
| Agentic: web search | 20% [EST 拆分] | `search_qa` | 25% | 25% | **1.25** | 超配：补 memory 空缺 + search 工具 schema 冷启动 |
| Agentic: memory | 20% [EST 拆分] | 无（§1.6 第二期） | 0% | 0% | **0.00** | 首期明确不训，只在评测跟踪 |
| Live + Non-Live | 10% + 10% | `single_turn` 中的 st_fc（1600×75%≈1200） | 15% | 15% | **0.75** | 故意欠配：已有 16947 条单轮已过饱和 |
| Hallucination | 10% | hammer irrelevance（1600×25%=400） | 5% | 5% | **0.50** | 欠配：只保"该弃权"行为不退化 |
| （支撑项） | 0 | `math_ci` + `code_exec` | 15% + 10% | 25% | — | 推理保锚 + 写-跑-改闭环迁移 |

读法：
- **单轮类 0.75**：每 1 个 BFCL 单轮权重点只配 0.75 份数据；因为单轮能力已被 16947 条旧数据
  覆盖 2.1 倍（16947 / 1600 = 10.6 倍压缩后仍覆盖全部单轮评测面），继续堆只会加重多轮伤
  （plan §0 已观察现象 + §7 "若多轮下降先降单轮占比"）。
- **多轮类 1.00**：BFCL 30% 权重是目标主战场，训练数据从 0 起步 → 必须 1:1 起配。
- **搜索类 1.25**：web search 占 Agentic 20%，但（a）memory 20% 首期空白（无数据）；
  （b）search 任务要学 `search/fetch` 工具 schema + 多跳停止判断，比单轮更吃样本；
  （c）Search-R1 1–2 跳任务简单（plan §1.2），**质量低 → 数量补**，故 25% 名义超配 25%。
- 数学/代码 25% 是**保锚项**：不直接换 BFCL 分，但（a）plan §7 要求监控推理基准不回退；
  （b）`code_exec` 的"写→跑→看报错→改"与 agentic 多轮同构，是低成本的多轮闭环迁移；
  （c）math_ci 在支撑 AIME/常识推理的同时学"何时调用 python"，避免工具化损害推理策略。
- 覆盖合计：80% BFCL 权重有训练数据（100% − memory 20%）；memory 是明确的 phase-2 缺口，不做假配比。

**为什么不是凭感觉**：上面的"每 1 份权重配多少数据"列，就是配置权重的直接来源；
`total_train=8000` 时 20/30/25/15/10 对应 1600/2400/2000/1200/800，每一类的来源 cap
都 ≥ 该类目标 + held-out（见 §2.1），保证"配比可执行"而不只是名义。

### 2.1 目标条数与来源上限（baseline total_train=8000）

| 配比类 | 目标 train | 来源（weight → 目标条数；cap） | held-out（300×weight） |
|---|---:|---|---|
| single_turn | 1600 | apigen .27→432(cap550)；sealtools .27→432(550)；toolace .165→264(400)；apibank .045→72(100)；hammer .25→400(500) | 81/81/49/14/75 =300 |
| multi_turn | 2400 | funreason .40→960(cap1200)；apigen_mt .25→600(700)；toucan .35→840(900) | 120/75/105 =300 |
| search_qa | 2000 | search_r1 .75→1500(cap1500)；asearcher .25→500(600) | 225/75 =300 |
| math_ci | 1200 | reasoning_math .90→1080(cap1400)；simplerl .10→120(300) | 270/30 =300 |
| code_exec | 800 | code_exec（T1 单 loader 聚合 APPS/CodeContests/TACO，`kind` 选择数据）→800(cap800) | 300 |
| 合计 | 8000 | | **1500** |

- cap 为 baseline `total_train=8000` 下的绝对值；**build_mixture 按 `total_train` 线性缩放**
  （critic T2-01），stats 同时记录 `cap_declared` 与 `cap_scaled`。例如 2500 smoke run 时
  apigen cap 550→172、类目标 1600→500，来源占比不变；32k 全量时等比放大，不会"小源饿死"。
- apibank 只有 161 条，train 72 + held 14 = 86（53% 使用率），是唯一高使用率来源，正常。
- tau-bench / toolmind / a2search / retool / when2call 在 config 中 `enabled:false`，
  不引用进 category，因此不影响权重和（独占校验会报错）。

---

## 3. 单轮数据过多为什么伤害多轮规划（机制 + 量纲）

### 3.1 机制（5 条，可观测）

1. **样本权 vs 交互权**：GRPO 是"每个 prompt 组归一化优势"。1 轮样本和 6 轮样本在 seq-mean
   归一化下权重相同 → 20% 的单轮样本拿到约 20% 的策略梯度质量，却只覆盖 1/6 的交互深度；
   策略先学会"一次调用 + 收尾"，多轮所需的"再观察、再决策"没有正梯度来源。
2. **归一化口径差异**：若用 token-mean（按总 token 归一），单轮 token 只占 ~6%（见 3.2），
   单轮样本反而被稀释；但一旦 rollout 预算按"样本条数"分配，单轮会以 1/6 成本占据
   20% 的采样槽与 20% 的 prompt 组 → **token 占比与组占比不一致**，取决于训练侧归一化
   （偏移风险点，需在消融里固定口径）。
3. **奖励密度与早期主导**：单轮 reward 在 1 步内 1/0，梯度即时、方差低；多轮终态 reward
   要到 4–8 步才出现（`[REPO]` mt_fc 0.8 结果项 + 0.1 逐轮 + 0.1 格式）。早期训练单轮
   advantage 更稳定 → 更快塑形 → 把"调用准确性"写成更强先验（plan §10 的 replay 放大风险同源）。
4. **数据量级失衡**：旧池 16947 条单轮 vs 新池 0 条多轮。如果不 cap，按比例采样时单轮
   占绝对多数；CAP 到 9.4% 是"压缩 10.6 倍"后的结果。
5. **DAPO 过滤的时序效应**：单轮旧数据被现有 checkpoint 练过 → pass 率高 → 组内零方差
   被 `filter_groups` 过滤（`[REPO]` DESIGN.md §3.3）；但过滤前它们仍占用 rollout 预算，
   且"全对组"的样本在早期 step 里提供格式/格式合法性梯度，强化"一步到位"格式先验。

**仓库第一手证据**（`examples/tool_rl/DESIGN.md`，只读参考）：
- §1 奖励 Dim1 工具正确性权重 0.60，配合弃权塑造 → 明确偏向"调用准"。
- §2.2 首条 user 注入 `<instruction>`：鼓励多轮澄清、别一步到位、别臆测 → 作者已在对抗单轮化。
- §2.4 负样本目标 25%（`trim_negatives`）→ 本配比单轮类 hammer 25% 的直接数字出处。
- §3.1 hint 注入同义改写池 + 25% 无 hint 对照 → 证明固定措辞/固定行为会被过拟合。

### 3.2 Token 量纲估算（[EST]，用于判断"梯度 token 占比"）

假设（写死在报告里以便复核）：单轮 1 轮 ≈1.3k token；多轮 6 轮 ≈6k；搜索 5 轮 ≈5k；
math_ci 2 轮 ≈2k；code_exec 4 轮 ≈7k。按 §2.1 条数：

| 类 | 条数 | 每条约 token | 合计 token | token 占比 | 样本占比 |
|---|---:|---:|---:|---:|---:|
| single_turn | 1600 | 1.3k | 2.08M | **6.0%** | 20% |
| multi_turn | 2400 | 6k | 14.4M | 41.8% | 30% |
| search_qa | 2000 | 5k | 10.0M | 29.0% | 25% |
| math_ci | 1200 | 2k | 2.4M | 7.0% | 15% |
| code_exec | 800 | 7k | 5.6M | 16.2% | 10% |
| 合计 | 8000 | | 34.5M | 100% | 100% |

结论：token 口径下单轮仅 6%，多轮+搜索 70.8%——**从 token 角度配比是"重多轮"的**；
但样本/组口径下单轮仍 20%。因此验收时要同时看两个指标：`token_share` 与 `group_share`，
并在消融中锁定训练框架的 loss 归一化口径，避免"配比数字对、梯度效果错"。

---

## 4. Placeholder 敏感性（T1 C6/D11；critic 要求的两套配额）

T1 结论：`mt_fc` 真实返回主力 = Toucan 子集；FunReason-MT / APIGen-MT 多为 placeholder。
`build_mixture` 读 **静态** `meta["gt_returns_source"]∈{config, reference, gt_turns, placeholder}`
（schema v2.2 对 mt_fc 强制必填；缺失会 schema-invalid 被 drop/error 记录），未知时才按
C6 优先级推断；stats 分开记 `placeholder_static` 与 `placeholder_inferred_fallback`（critic T2-02）。

参数化假设（[EST]）：Toucan 干净池 900（cap），FunReason 1200 / APIGen 700（cap），
二者 placeholder 比例 r；多轮目标 2400。`keep` 预算 = floor(2400×0.25)=600 条。

| placeholder 比例 r | `drop`（placeholder 全删）：多轮实际 / 占比 | `keep`（预算 600，降权 0.5）：多轮实际 / 占比 / placeholder 占比 |
|---:|---|---|
| 0% | 2400 / 30.0% | 2400 / 30.0% / 0% |
| 25% | 2325 / 29.1%（缺 75） | 2400 / 30.0% / 3.1% |
| 50% | 1850 / 23.1% | 2400 / 30.0% / 22.9% |
| 75% | 1375 / 17.2% | 1975 / 24.7% / 30.4%（预算打满，仍缺 425） |
| 100% | 900 / 11.3% | 1500 / 18.8% / 40.0%（预算打满，缺 900） |

- **drop（config 默认，T1 要求 placeholder 不进训练）**：多轮缺口由 build 的 deficit 吸收机制
  均摊到其他有容量的类（stats 记 `deficit_initial/remaining/absorbed_deficit` 与
  `share_effective`），实际配比会偏离 §6；报告/验收必须按 `share_effective` 看，不能只看 config。
- **keep + share=0.25 + weight=0.5**：可保住 30% 配额到 r≈50%，代价是 22.9% 的多轮样本是
  降质回放（B 分支 0.8 主项形同虚设的风险），必须由 reward 侧按 `placeholder_weight=0.5`
  降权（trainer 侧决策，config 只记录该数值）并在报告给"过滤前后条数"。
- 敏感性结论：**若首跑 r≤25% 用 drop；若 r 在 25%–50%，用 keep(0.25)/drop 两套跑消融，
  用 held-out 的 mt_fc 结果项（不是 replay 匹配）决定**。r>50% 时首期多轮 30% 不可兑现，
  建议提高 Toucan 子采样/cap 并接受有效占比 <30%，同时把 BFCL multi-turn 指标作为独立门槛
  （别用训练配比数字自欺）。
- 多轮 held-out 只用非 placeholder 样本（build 已实现：placeholder 只进 train 预算，不进 held-out），
  否则 eval 本身会被降质回放污染。

---

## 5. Abstain / 拒答 20% 的落地（critic T2-03 答复）

**选择 (b)**：首期声明"单轮+拒答 20%"由 **hammer irrelevance 承担**（400 条 = 总 train 5%、
单轮类 25%，与 `[REPO]` DESIGN.md §2.4 的负样本 25% 目标同源）；`abstain` task_type 与
When2Call **只做 reward/env 单元测试，不进训练**。理由：
1. T1 核实 When2Call 只有 train 可用、test 由 BFCL v2 Live 派生 → 混入风险高、许可证未核；
2. 首期目标是"工具调用+多轮规划"，拒答可由 irrelevance 的 1/0 奖励充分塑造；
3. 引入 abstain 需要额外 reward/env 工期，且 5–10% 权重会从已饱和的 st_fc 挪出，收益不确定。
若后续需要 (a)，config 已预留：启用 when2call，weight 0.10 从 st_fc 各源等比扣除
（apigen .24/sealtools .24/toolace .145/apibank .025 + hammer .25=1.0），并新增 abstain 配额。
已答复 Lead 记录 TEAM_BOARD。

---

## 6. 通过率过滤：N=8 的数学与建议区间

规则（plan §3 步骤 6）：保留 `pass_rate ∈ (0,1)`；全对/全错剔除；`meta.pass_rate` 写回
（raw k/N；可选 Jeffreys 平滑另存）；config 补充推荐带 `[0.125,0.875]`、N=8。

### 6.1 噪声表（`filter_by_passrate --noise-table` 同源）

| N | 网格 1/N | 最大标准误 (p=.5) | 95% CI 半宽（最坏） | 误删率 p=.05 | p=.10 | p=.20 |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | .250 | .2500 | ±.49 | .8145 | .6561 | .4096 |
| 8 | .125 | .1768 | ±.346 | **.6634** | **.4305** | **.1678** |
| 16 | .0625 | .1250 | ±.245 | .4401 | .1853 | .0281 |
| 32 | .03125 | .0884 | ±.173 | .1937 | .0343 | .00079 |
| 64 | .015625 | .0625 | ±.1225 | .0375 | .00118 | ~0 |
| 100 | .01 | .0500 | ±.098 | .00592 | ~0 | ~0 |

误删率 = `(1−p)^N + p^N`（全错或全对即删）。**关键结论**：
- N=8 无法估计难度（p=.5 时 SE 17.7pt、CI ±35pt）；它只可靠地区分 p≈0/p≈1。
- N=8 + "保留(0,1)"必然误删大量 p=5% 的题（66%）；这是**有意代价**（GRPO 低通过率组大多
  零方差），但会形成"有效难度下限 ≈ 12%"，下游别把 pass_rate 当真实难度。
- 想 SE≤5pt 需 N≥100；难度分层建议 N=16/32；二阶段方案：N=8 初筛 → 对 k∈{1,N−1} 的边界样本
  再采 24 次（p=.5 时边界样本约 6.25%，额外 rollout ≈ 2 条/题，成本可控）。
- **保留区间建议**：正式训练用 (0,1) 硬规则 + 记录 pass_rate；若做难度课程，
  `[0.25,0.75]`（k∈{2..6} at N=8）；`[0.125,0.875]` 仅作为"显式声明不做全对/全错"的默认，
  在 N=8 网格上等价于 (0,1)。config 的两项都已可配（band、n_samples），消融见 §8。
- 采样预算量级：若 4 万 prompt × N=8 = 32 万 rollout；按 5 轮、每轮约 500 token 估算
  约 8 亿生成 token（[EST]），过滤本身要计入成本；这就是"先用 N=8 粗筛再对边界复筛"的原因。

### 6.2 N 与 rollout 成本的关系（选型建议）

- 首期：N=8（plan 默认），只做全对/全错剔除 + 记录；
- 难度区间消融：N=32 重采样 2–4k 任务（约 8–12.8 万 rollout），验证"中难度区间"假设；
- 若策略 pass 率整体上移（模型变强），N 需同步上调，否则 (0,1) 规则会把新变简单的题大量删掉
   → 建议每次换 checkpoint 重新 rollout，而不是复用旧 rates 文件。

---

## 7. Held-out 样本量数学与口径

- **口径**：5 个配比类 × 300 = 1500 条（不是 7 个 task_type × 200）；plan 原文"每类别至少 200 条"
  中"类别"按 §6 五类解释。config `heldout.per_category: 300` 独立于各类 `heldout`（可单类覆盖）。
- 二项比例 95% CI（最坏 p=.5）：
  | n | SE | 95% CI |
  |---:|---:|---:|
  | 200 | .0354 | ±6.9pt |
  | 300 | .0289 | ±5.7pt |
  | 1500 | .0129 | ±2.5pt |
- 两臂比较（80% power, α=.05）可检出的绝对差：n=200 → 14.0pt；n=300 → 11.4pt；n=1500 → 5.1pt。
  → **单类 300 只够看粗方向，消融结论用全量 1500 或 BFCL 评测（其样本量更大）**；
  别用单类 held-out 的 2–3pt 差异宣称"配比有效"。
- **评估噪声叠加**：held-out pass rate 若每题再采 N_eval=8，估计量的方差 = 任务级方差 +
  rollout 级方差/8；报告需给 N_eval 与 CI，不能只报一个均值。建议 N_eval=8 与训练一致，
  并固定随机种子/温度（evaluation-only）。
- **[NO-GROUP] 风险**：会话级切分需要 `meta.group_id`（同 session 多 turn 同组）与
  `meta.group_size`（判 partial group），**两者未进 schema v2**（仅作为额外 meta 键被容忍）。
  当前 build 实现：有 group_id 则按组切（组整体进 train 或 held）；group_size 缺失时无法识别
  被 top-K 截断的 partial group，只能 best-effort 并报 `partial_groups`。
  建议 Lead 在 schema v3 加 `meta.group_id/group_size`（或 loader 侧保证 group 完整）——
  否则同一会话可能跨 train/held，held-out 分数虚高（`[REPO]` DESIGN.md §2.5 已有前车之鉴）。
- **held-out 固定性**：held-out 用独立 salt 固定选择、跨 build 不变，保证消融可比；
  换 seed 会换 held-out（对比实验时注意固定 seed）。

---

## 8. 课程（plan §7）与消融清单

### 8.1 课程结构（config `curriculum`，`[SUGGEST]` 需消融）

| 阶段 | 训练进度 | max_turns | 数据子集（估算） | 观察指标 / 晋级条件 |
|---|---:|---:|---|---|
| 1 | 0–20% | 5 | st_fc+irrelevance 1600、math_ci 1200、code_exec 800、搜索 ≤5 轮子集、多轮 ≤5 轮子集（`[EST]` 约 55–65% of train） | stage 子集 held-out pass>0.6 且 3 次 eval 增益 <3pt；timeout 率 <10% |
| 2 | 20–50% | 10 | 追加多轮 ≤10、搜索全部（env 硬上限 8 轮）→ `[EST]` ~90% | 同上；多轮结果项（goal_satisfied/replay_ok 非 placeholder）通过率 |
| 3 | 50–100% | 15 | 全部 train（实际数字由 `build_mixture --emit-curriculum` 产出，stats 记每阶段 count） | BFCL v3 multi-turn + v4 agentic web search 不降；推理基准不降 |

- Search env 由 env-builder 冻结：`max_rounds=8` 硬上限、`top_k=3`、单次 5s 超时 + 2 次重试
  （0.5/1.0s 退避）、snippet ≤1000 字符、obs ≤4096 字符；因此 stage1 的 search 实际上限=5，
  stage2/3 被 8 卡住（课程 10/15 对 search 无效）。config `search_qa.turn_cap: 8` 已在 build 端
  把 >8 轮样本过滤并计数（`over_max_turns`），报告给轮数分布。
- 架构建议：stage 文件由 `--emit-curriculum` 生成，trainer 按进度切分；不把 max_turns 写进
  task.meta（保持数据静态可复现，课程是训练侧调度）。
- 偏移控制（plan §7）：每评测点同时跑推理基准；KL to base checkpoint 超阈值则回退，
  **回退顺序是先降单轮占比，不是加多轮**。

### 8.2 消融清单（每项：预期效应 + 观察指标 + 优先级）

| # | 变量 | 变体 | 预期效应 | 观察指标 | 优先级 |
|---|---|---|---|---|---|
| A1 | 单轮占比 | 20% → 15% / 30% | 15%: 多轮↑、单轮/Live 略↓；30%: 反向 | BFCL MT、Non-Live/Live、AIME、KL | P0 |
| A2 | 单轮内 irrelevance 比例 | 25% → 15% / 35% | 推理/过度弃权的 trade-off | BFCL Hallucination、误拒率（abstain 奖励分布） | P1 |
| A3 | 多轮占比 | 30% → 20% / 40% | 与 BFCL MT 单调相关，但受 placeholder 比例影响 | BFCL MT、replay_ok 通过率、timeout | P0 |
| A4 | 搜索占比 | 25% → 15% / 35% | 15%: web search↓但推理更稳；35%: 相反 | BFCL web search、AIME、search QPS/超时率 | P0 |
| A5 | placeholder 策略 | drop / keep(0.25, w=.5) / keep(0.5,w=.3) | keep 保住 30% 配额但混入降质样本 | mt_fc 结果项（非 replay 匹配）、BFCL MT | P0 |
| A6 | 数学+CI | 15% → 5% / 25% | 降→推理回退风险；升→多轮/搜索配额被挤 | AIME/MATH、BFCL agentic | P1 |
| A7 | code_exec | 10% → 0% / 20% | 0% 少一类多轮闭环；20% 依赖单测质量 | BFCL MT、代码基准、单测通过率 | P1（等 Q4 许可证） |
| A8 | 通过率 N | 8 → 16/32（只对边界样本） | 难度估计更准、中难度样本保留更多 | 保留率、pass_rate 分布、训练曲线 | P1 |
| A9 | 通过率带 | (0,1) → [0.25,0.75] / [0.125,0.875] | 窄带=只训"可学"题，可能过拟合中难度 | 零方差组比例、held-out 提升 | P2 |
| A10 | 课程 | 开 / 关；阶段边界 20/50% vs 30/70% | 开：早期稳、长程慢；关：直接 15 轮更省事 | timeout 率、MT 学习速度、最终分 | P0 |
| A11 | 种子 | seed ±1 | 选择漂移的方差上界 | 两次 build 的 id 重合率、指标方差 | P2（保证可复现即可） |
| A12 | 去重 | exact on/off；近似阈值 .85/.9 | off → 重复样本虚高训练信号 | 去重条数、held-out 泛化 | P0 |
| A13 | 去污染 | 严格 hit=drop vs 只报告 | 严格→0 命中（验收 #2）；宽松→BFCL 虚高 | BFCL 测试集命中数（必须 0） | P0（硬验收） |
| A14 | ToolMind 入列（去重后） | 0 vs 5–10% 多轮 | 若重叠小→多轮多样性↑；否则污染/重复 | 近似重复率、BFCL MT | P2 |
| A15 | top_k（配合搜索吞吐） | 3 → 2 | 检索噪声↓、QPS↑；可能损失召回 | web search、QPS/超时 | P2（env 侧） |

> A1/A3/A5/A10 优先：它们直接决定"单轮不挤多轮"与"30% 多轮是否兑现"。
> 消融记录模板：{配置 hash、seed、steps、held-out 各类 pass_rate（N_eval=8）、BFCL 各子集、
> AIME/MATH、KL、timeout 率、token_share/group_share}。

---

## 9. 构建与过滤的可复现性（验收 §8-1/2/4）

- `build_mixture.py` 复现机制：`seed` + `sha256(seed, salt, task_id)` 稳定哈希排序（不依赖
  Python `random` 实现），同 rank 以 task_id 破平（不依赖文件行序）；最终按 (source, id) 排序；
  `schema.dump_jsonl`（sort_keys + 非 ASCII 保留）；stats 无时间戳、含 config/输出 sha256。
  → 同一 config + 同一输入，两次运行 `train/heldout/mixture_stats` 字节一致（测试覆盖）。
- 输入来源：优先 `data/raw_unified/{source}.jsonl`（T1 步骤 1 产物），文件缺失时可用
  `--use-loaders` 走 T1 `loaders.load_source(loader_name, cfg)`（软依赖；config 的
  `sources.*.loader` / `loader_cfg` 可映射"配比源名 → T1 registry 名"，例如
  `apigen_mt`/`toucan` 已按 T1 registry 改名，`code_exec` 用单一 loader）；两者都没有则显式报错
  （required 源）或记录 `missing_optional`（可选源）。**缺源/空源不会静默**。
- 本地守门（build 端）：schema 校验（drop + 记录 vs error）、C9 轮数口径
  `max(len(gt_calls), expected_turns, len(gt_turns))` ≤ min(15, category.turn_cap)、
  通过率硬规则、placeholder 静态过滤、精确去重（含 train/held 冲突以 held 优先）、
  held-out 组感知、cap 缩放、缺额吸收与 effective share 记录。
- T1 脚本负责且本脚本**不重复实现**的：近似去重（MinHash/阈值）、去污染（BFCL v3/v4）、
  轮数过滤的原始统计、license 核实；这些开关在 config `filters.*` 中唯一声明，build 在 stats
  `switches` 里落档（`report_required` 可在正式运行改 true 硬卡）。
- 验收 #4 的 `reports/data_report.md` 由 T1/Lead 汇总；build 的 `mixture_stats.json`
  提供它需要的：每源加载/各步丢弃计数与原因、最终配比（target/effective）、轮数分布、
  通过率分布、placeholder 前后条数与占比、curriculum 各阶段条数、输出 sha256。
  另含显式执行标记 `filters_applied`（`passrate=False`、`decontam=False` = **build 本身
  未执行这两步**；`data_report.md` 必须对未执行项标 `[NOT RUN]`，禁止把未执行当已验收，
  critic T2-04）。

---

## 10. 验证命令（用户在有 shell 的环境执行；本机未执行）

```bash
# 0) 环境（仓库根目录）
cd D:/lhy/verl/verl-toolRL
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # 若需要 PyYAML: pip install pyyaml（可选）

# 1) 单测（本机未执行；覆盖确定性/配比容差/无重叠/缺源/通过率/placeholder）
python -m pytest agentic_rl_data/tests/test_mixture.py -v

# 2) 通过率过滤（先在 T1 生成了 raw_unified 后）
python -m agentic_rl_data.scripts.filter_by_passrate \
  --tasks data/raw_unified/funreason_mt.jsonl \
  --rates data/passrate/funreason_mt.jsonl \
  --out data/raw_unified/funreason_mt.filtered.jsonl \
  --report data/passrate/funreason_mt.report.json \
  --band 0.125,0.875 --n-samples 8 --missing keep
python -m agentic_rl_data.scripts.filter_by_passrate --noise-table   # N 噪声表

# 3) 一条命令构建 mixture_v1（plan §8-1）
python -m agentic_rl_data.scripts.build_mixture \
  --config agentic_rl_data/configs/mixture.yaml \
  --raw-dir data/raw_unified --out-dir data/mixture_v1

# 4) 字节级复现校验（两次运行 sha256 相同；计划 §8-1）
sha256sum data/mixture_v1/train.jsonl data/mixture_v1/heldout.jsonl data/mixture_v1/mixture_stats.json
# Windows PowerShell:
# Get-FileHash data/mixture_v1/train.jsonl, data/mixture_v1/heldout.jsonl -Algorithm SHA256

# 5) smoke 规模（plan §7 "每类 ~500 先跑通"；cap 自动线性缩放）
python -m agentic_rl_data.scripts.build_mixture --total-train 2500 --out-dir data/mixture_v1_smoke

# 6) 课程子集
python -m agentic_rl_data.scripts.build_mixture --emit-curriculum --out-dir data/mixture_v1
```

预期：`train=8000（数据齐时）/ heldout=1500`，`mixture_stats.json` 无警告或仅
placeholder/deficit 相关警告；两次构建 sha256 一致；若存在 placeholder 比例高，
stats 的 `categories.multi_turn.share_effective` 会 <0.30（这是预期，不是 bug）。

---

## 11. 风险与未决

| # | 风险 / 未决项 | 影响 | 现状与建议 |
|---|---|---|---|
| R1 | placeholder 比例未知（T1 首跑才知） | 多轮 30% 可能落空 | §4 两套配额 + 敏感性表；A5 消融 |
| R2 | `[CONFIRM]` 条数/许可未核实（HF 本沙箱不可达） | cap 可能偏大/偏小、许可不可训练 | 全部标 [EST]/[UNVERIFIED]；用户按 §10 复核；T1 data_sources.md 为准 |
| R3 | asearcher 无逐题轮数标注（T1），≤15 静态过滤不可判定；runtime 8 轮截断 | search 25% 质量下降（长轨迹被截断） | 权重 0.25/cap 600；监控 timeout/截断率，>20% 再降权；Search-R1 承担主力 |
| R4 | code_exec 许可证/单测质量（Q4） | 10% 配额不可用 | config `on_unavailable: redistribute`；待 T1 结论，确认后改 fail |
| R5 | memory 20% BFCL 权重无训练数据（plan §1.6 第二期） | 首期 BFCL v4 上限受限 | 明确不训，只跟踪；报告不装作覆盖 |
| R6 | `meta.group_id/group_size` 未进 schema（[NO-GROUP]） | 会话级泄漏、held-out 虚高 | build best-effort + partial_groups 计数；建议 schema v3 补齐 |
| R7 | 训练侧 loss 归一化口径未定（token-mean vs seq-mean） | 同一配比的单轮实际影响不同 | §3.2 双指标（token_share/group_share）；消融锁定口径 |
| R8 | 通过率 rates 文件后端未接（TEAM_BOARD Q2） | filter 只能吃离线文件 | 本脚本已留 CLI 文件接口；rollout 耦合留 TODO（不猜 slime） |
| R9 | 搜索 env 吞吐未实测（[SUGGEST]） | 25% 搜索可能拖慢 rollout | env-builder 参数：8 轮/ top_k3/ 5s/ retry2/ QPS 目标≥64；不达标先降 top_k 或 search 占比，不放宽 8 轮 |
| R10 | bfcl-eval 版本漂移（T1 核实 2026.3.23） | 去污染/评测口径与文档不一致 | 固定 2026.3.23 并记录；去污染报告必须 0 命中 |

---

## 附：文件与责任人

| 文件 | 内容 | owner |
|---|---|---|
| `agentic_rl_data/configs/mixture.yaml` | 配比/来源上限/seed/heldout/课程/通过率/去重去污染/placeholder 策略 | T2 |
| `agentic_rl_data/scripts/build_mixture.py` | 单命令确定性构建 + stats + curriculum + 守门 | T2 |
| `agentic_rl_data/scripts/filter_by_passrate.py` | 通过率过滤纯函数 + CLI + N 噪声表 | T2 |
| `agentic_rl_data/tests/test_mixture.py` | 确定性/配比容差/无重叠/缺源/placeholder/通过率 | T2（未执行） |
| `agentic_rl_data/reports/data_sources.md` | 数据源条数/license/[CONFIRM] | T1 |
| `agentic_rl_data/reports/data_report.md` | plan §3 步骤 8 汇总报告 | T1 + Lead |
| `agentic_rl_data/CONTRACTS.md` / `schema.py` | 跨席位契约（v2/v2.2） | Lead |
