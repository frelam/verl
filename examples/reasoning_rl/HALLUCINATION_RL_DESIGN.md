# 幻觉抵制 Reasoning RL —— 实施设计（工作文档）

> 状态：**草稿 / 待逐条确认**。本文档是这份方案的唯一事实来源与施工图，会随确认过程逐步修改。
> 与已有的 `DESIGN.md`（Qwen3-4B reasoning RL 主方案）**并行存在、互不覆盖**：本方案只在训练后程
> 以「换数据 mix + 换 reward 文件」的方式接入，不修改阶段一任何行为。

原始方案输入：`hallucination_resistant_reasoning_rl_spec.md`
（sha256 `05fbab17…06a752`，只读副本在 `~/.dsh/attachments/v1/files/05/05fbab177faa00102b3aa78cca8c24687a0a818cdc949207382b62a76106a752/`）。
本文档不复述该 spec 全文，只记录**落到本仓库后的具体实现决策、接口与开放问题**。

---

## 0. 已锁定决策

| # | 项 | 决定 | 来源 |
|---|---|---|---|
| D1 | 接入方式 | 两段式：阶段 1 完全不动；阶段 2 从 ckpt `RESUME_MODE=resume_path` 续训，换新 mix parquet + 新 reward 文件 | 用户确认 |
| D2 | 首批数据源 | 垂直切片四源：K&K（logic）+ MiP（math）+ FalseQA（commonsense）+ GSM-IC（可解-干扰对照） | 用户确认 |
| D3 | 答案契约 | 可解 `\boxed{答案}`；不可解 `\boxed{UNSOLVABLE: B}`；无诊断标签源 `\boxed{UNSOLVABLE}` | 用户确认 |
| D4 | 一致性辅助项（spec §3.2） | **不做**。reward 只保留 spec §3.1 主 reward | 用户确认 |
| D5 | 归一化（spec §3.3） | 随 D4 失效；退化组屏蔽沿用现有 DAPO `filter_groups`（hard-replay sampler 自持） | 推论 |
| D6 | 冷启动 SFT（spec §4） | **不在范围**。只交付数据 + reward，由用户在训练后程自行挂载 | 用户确认 |
| D7 | 配比 | 固定比例，不做 step 退火；脚本暴露参数供手动配置 | 用户确认 |
| D8 | 覆盖约束 | 不改 `reward/compute_score.py`、不改 `scripts/*`、不改 `DESIGN.md`、不改现有 run 脚本的数据/reward 语义 | 用户确认 |
| D9 | AbstentionBench | **移出训练源**，降级为评估套件：聚合体已含 FalseQA/UMWP（重复），且含 GPQA/GSM8K/MMLU 等评估集（污染） | 用户确认 |
| D10 | UMWP | **本阶段不进训练**，登记为不可解供给的替补（v0.6 更新触发条件：不再是"MiP 配对失败"，而是"约 **0.93k** 的不可解池被判定不够"，见 §12 Q4）；其原生 2600 对配对可直接支撑 diff | 用户确认 |
| D11 | K&K 定位 | 实测 48,076 行**全部唯一可解、零弃答/零诊断数据** → K&K 降级为**可解锚点**：不生成选项、不进四档 reward；按 `(人数, index)` 组内去重、只用 N≥4、批内占比 4–6% | 实测确认（§4.1） |
| D12 | MiP 定位 | 实测：可解侧有原题答案（客观可校验），但不可解侧的选项集**必然泄题**（选"题面里唯一找不到的那句"命中 **95.6%**，抹掉数字后仍 95.6%）→ MiP **不做诊断**，`has_diagnosis_label=false`，不可解侧只判"是否拒答"（bare `\boxed{UNSOLVABLE}`，prompt 不附选项块）。可入池 **严格口径 276 / 宽口径 299** | 用户确认（§4.3） |
| D13 | FalseQA 定位 | 实测有 `label` 列且三份 split 严格 50:50，**并且是按索引对齐的对照对**（第 k 条假前提 = 第 k 条真前提的局部改写）。→ **保留四档诊断**，但 gold **不是** `answer`（自由文本 rebuttal / test 里是 3 条并列参考），而是**配对 diff 出的假前提片段**；选项**只用本题面内等长跨度**（k=3，见 D15）。可入池 **train 657 / valid 259 / test 388**（§4.4） | 本轮实测确认 |
| D14 | FalseQA 可解侧 | `label=0`（1,187）**入池**，但契约不是"作答"而是**"判断该题是否可答"**：gold = `label`，期望输出 `\boxed{SOLVABLE}`；**误拒 → −1**（与"该拒答却编造 → −1"对称）。提示词在 FalseQA 源内**两个 label 完全统一**（都附选项块、都先问"前提是否成立"），否则"有选项块 ⟺ 不可解"会成为 100% 的模板泄漏。可入池 **851**（train） | 用户提议（§4.4/D14） |
| D15 | 选项数 k | **k = 3**（1 正确 + 2 干扰，A/B/C）。依据：同题 + 等长下 k=3 与 k=4 的表层启发式**都 ≈ 随机**——**k 不是抗蒙参数，而是覆盖率参数**（§4.4 L3）；而不可解池是本方案最紧的资源（≈0.93k），k=4 只会把四档从 657 砍到 473（−19.8pp，差额全部来自"等长候选池恰好 = 2"的 184 条）。**两个 label 必须同 k**（D14 的模板同构要求：选项个数本身就是一条捷径）。代价：猜测下限 25% → 33.3%，评测须对该基线读数（§10） | 用户确认 |
| D16 | 不可解侧扩源（v0.10） | 实测后纳入：**SUM**（MIT，train 36,480 → 四档 9,832 / 三档 6,050）、**UMWP**（CC-BY-SA-4.0，5,200 → 四档 1,449 / 三档 440）、**KUQ**（MIT，6,884，unknown 3,437 / known 3,447）、**TreeCut**（Apache-2.0，生成器可无限产，L1 构造即证明但 L3 需先修）、**CREPE**（BSD，HF 镜像；**只能进判断型**——其 `presuppositions` 实测只有 **1.1%** 逐字出现在题面里，是转述不是 span）。**排除** SQuAD 2.0（判"不可解"需 NLI → L1 过不了）与 GSM-DC（仅 6,300 样例、合成风格生硬）。可用性实测见 §4.9 表 A | 用户确认 |
| D17 | 干扰项数据（题面有干扰、答案不变） | **不引新源**，**规则复用 GSM-IC 的 242 条 `sentence_template`**（全部由 `{role}`/`{number}` 参数化）施加到 SUM/UMWP 的可解原题、K&K 与主池数学题上：答案不变、gold 仍是原答案、reward 走现有 `_math_score`，不新增判分逻辑。难度梯度直接用 GSM-IC 自带标注——`sentence_label`（in_topic 45% / out_topic 55%）、`role_label`（overlapped 50% / nonoverlapped 50%）、`number_label`（in_range / out_range 50:50），**三类天然配平 → 无标签泄漏**。零 LLM、零新依赖 | 用户提议（§4.9 表 C） |
| D18 | 幻觉域规模与类型平衡 | 幻觉域目标 **20,000 条**：不可解 **12,000（60.0%）** / 可解 8,000。不可解侧按缺陷类型配平——缺前提 35% / 关键信息歧义 25% / 不现实前提 11.7% / 假前提 10.5% / 问题缺失 9.5% / 无关实体 8.3%。**硬约束**：模板 A（带选项块）与模板 B（不带选项块）**各自都必须同时含可解与不可解**，否则"有没有选项块"本身就成了捷径 | 用户确认 |

---

## 1. 目标与非目标

**目标**：让模型在可解题上给出答案；在不可解题上**拒答而不是强行编造**。拒答本身必须可校验——
即"正确识别不可解"才得分。**细粒度诊断（指出缺了/矛盾的是哪一条）只在"选项语义可靠、无表层捷径"
的源上启用**：本阶段四源中只有 FalseQA 满足（v0.6/D13：把诊断实现为**指向题面中那一处假前提片段**，
而不是复述"缺了哪条条件"；选项只取同题等长跨度），MiP 有不可解数据但选项集必然泄题，故按 D12 退化为
"只判拒答"（§4.3）。这样"拒答"这一侧不再是靠猜选项得分，而是可验证的行为。
**可解侧同样要有"别乱拒答"的压力**（v0.7/D14）：FalseQA 的真前提对照以"判断该题是否可答"的形式入池
（gold = `label`，误拒给 −1），否则常识题型在训练里只剩"假前提"一个方向。

**非目标（本阶段）**：
- 不做数据生成（IGC-MWP 式生成器、TreeCut 式树生成）——全部用现成数据集。
- 不做 SFT / 冷启动示范轨迹（D6）。
- 不做一致性辅助项（D4）。
- 不做 UMWP、不做 AbstentionBench 训练数据（D9/D10，理由与替补关系见 §4.7）。
- 不做 embedding 近邻干扰项挖掘的正式启用（接口预留，见 §4.6）。
- 不做 spec §3.3 的"两分量分别组内归一化"（D5）。

---

## 2. 与现有 reasoning_rl 的关系

### 2.1 完全不动（只读复用）

| 现有资产 | 复用方式 |
|---|---|
| `reward/compute_score.py` | 新文件按前缀委托：非 `halluc_*` 的 sample 原样转调它的 `compute_score`，一行不改 |
| `reward/` 下其它 verifier（`enigmata_verifier.py` / `if_verifier.py` / `reasoning_gym_verifier.py`） | 同上，经 `compute_score` 间接复用 |
| `scripts/mix.py`、`scripts/to_parquet_*.py`、`scripts/dedup.py`、`scripts/decontaminate.py` | 只调用其函数/模式，不改文件 |
| `run_qwen3_4b_reasoning_rl_dapo.sh` | 复用其 `TRAIN_FILES`/`VAL_FILES` + `RESUME_MODE`/`RESUME_PATH` 热插拔接缝；见 §7.3 关于 reward 路径切换的待确认项 |
| `reasoning_rl_dataset.py` / `hard_replay.py` | 阶段二沿用（system prompt 注入、hard replay 均正交） |

关键复用点（已核对源码）：

- `compute_score()` 按 `data_source` 前缀分发 `math_* / code_* / logic_* / stem_* / if_*`，
  未知前缀在格式合规后抛 `NotImplementedError`。→ 新增前缀必须由新文件自己的 dispatcher 处理。
- `format_ok()` 是通用结构门：要求 `<think>…</think>` 且 reasoning/final response 均非空。
  → `halluc_*` 走同一道门（阶段二基座是同一 Qwen3 thinking 模板）。
- `_extract_boxed()` 已实现"取最后一个配平花括号的 `\boxed{}`"。
  → 新 reward 直接复用它做终局抽取，不重造。
- run 脚本已有 `train_files=${TRAIN_FILES:-…}` / `val_files=${VAL_FILES:-…}` 与
  `RESUME_MODE`/`RESUME_PATH`；`mix.py --if_ratio` 是"后程混入新域"的现成先例。

### 2.2 新增（全部是新文件，见 §8）

- `reward/hallucination_compute_score.py` + 测试
- `scripts/hallucination/` 下一整套 schema / adapter / spike / mix / 测试

---

## 3. 统一数据 Schema 与 verl parquet 映射

spec §1 的 JSON schema 不直接进训练；落盘统一转成现有 reasoning_rl 的 parquet 约定
（与 `DESIGN.md` §1 同构），映射如下：

| spec 字段 | 落盘位置 | 说明 |
|---|---|---|
| `id` | `extra_info.task_id` | 全局唯一，hard replay 的 dedup key 也用它 |
| `domain` | `ability` + `extra_info.domain` | `logic` / `math` / `commonsense` |
| `source_dataset` | `extra_info.source` | 如 `K-and-K`、`MiP-SVAMP`、`FalseQA`、`GSM-IC` |
| `prompt`（含选项） | `prompt` | chat 格式 `[{"role":"user","content":…}]`，选项块在 §5 模板里渲染 |
| `solvable` | `reward_model.ground_truth` JSON | 见下 |
| `ground_truth_answer` | 同上 | `solvable=true` 必填 |
| `role_words` | 同上 | ⚠️ **K&K 专用**：该行自己的表面角色词 `[真话者词, 说谎者词]`（取自 `knight_knave` 映射）。答案解析必须按它归一化，否则 reward 反向，见 §5.1 |
| `missing_condition_options` | `extra_info.options` + prompt 渲染 | reward 侧用 `correct_option_id` 判分；options 文本进 `extra_info` 便于审计。**本阶段只有 FalseQA 有选项**（MiP 按 D12 无选项，§4.3）。v0.6 后选项语义 = **题面内的跨度**（不是跨题 rebuttal 句，§4.4/D13） |
| `correct_option_id` | `reward_model.ground_truth` JSON | `solvable=false` 且 `has_diagnosis_label=true` 时必填；MiP 侧为 `null` |
| `has_diagnosis_label` | `reward_model.ground_truth` JSON | 决定四档 / 三档。v0.5 后**不再是 `solvable=false` 的同义词**：MiP 是 `false` + `false`（§4.8 结论 1） |
| `judgment_only`（v0.7/D14） | `reward_model.ground_truth` JSON | `true` 时该条**只判"是否可答"**（期望 `\boxed{SOLVABLE}`），不做数值匹配；本阶段仅 FalseQA-real |
| `perturbation_type` | `extra_info.perturbation_type` | `missing_condition / contradictory_condition / distracting_condition / null` |
| `difficulty_tag` | `extra_info.difficulty` | 来源原生难度，供后续分桶 |
| `metadata.paired_original_text` | `extra_info.paired_original_text` | 审计用（可裁掉以控体积） |
| `metadata.perturbed_entity_text` | `extra_info.perturbed_entity_text` | 本阶段只做审计/监控，D4 后 reward 不用它 |

`reward_model`：

```json
{
  "style": "rule",
  "ground_truth": "{\"solvable\": false, \"answer\": null, \"correct_option_id\": \"B\", \"has_diagnosis_label\": true, \"perturbation_type\": \"missing_condition\"}"
}
```

K&K 行示例（可解、无选项、带角色词表）：

```json
{
  "style": "rule",
  "ground_truth": "{\"solvable\": true, \"answer\": \"angel devil devil\", \"role_words\": [\"angel\", \"devil\"], \"has_diagnosis_label\": false, \"perturbation_type\": null}"
}
```

> 约定：`ground_truth` 是 **JSON 字符串**（parquet 列里存字符串，reward 侧 `json.loads`）；
> 这样新增字段不需要动 parquet schema。解析失败 → reward 记 0 并 log（fail closed，与现有
> `compute_score` 的异常兜底一致）。

`data_source` 命名（= reward 路由键）：

| data_source | 含义 | 走哪个 scorer |
|---|---|---|
| `halluc_logic_kk` | K&K clean + 扰动 | 新逻辑（§6） |
| `halluc_math_mip` | MiP 系列（不可解，缺条件） | 新逻辑（§6） |
| `halluc_commonsense_falseqa` | FalseQA 假前提 | 新逻辑（§6） |
| `halluc_math_gsmic` | GSM-IC（**solvable=true**，含干扰条件） | 委托现有 `_math_score` |

---

## 4. 数据源与 adapter（垂直切片）

数据根目录约定（与现有 `DATA_DIR` 并列，不混入）：

```
~/data/reasoning_rl/halluc/
├── raw/          # 各源原始下载
├── built/        # 各 adapter 产出的统一 schema parquet（每源一份）
└── final/        # mix_halluc.py 产出的阶段二 train.parquet / val.parquet
```

### 4.1 K&K 核查结论（已完成；结论与 v0.3 设想相反）

spec §6.1 要求"先验证、不得假设开箱即用"，核查已完成，**推翻了原先"K&K 会产出无解样本"的假设**。

**核查 1 — 无解样本数：0。** 对全部 26 个 parquet（`K-and-K/knights-and-knaves` 的 7,000 条 clean
+ `K-and-K/perturbed-knights-and-knaves` 的 41,176 条扰动）共 **48,076 行**逐条还原约束系统并穷举
所有赋值：

```
0 解 0 条    唯一解 48,076 条    多解 0 条    枚举解与 solution 字段一致 48,076 条
```

→ **K&K 不提供任何不可解/弃答/诊断数据。** 原 `n_solutions == 0 → contradictory_condition` /
`n_solutions >= 2 → missing_condition` 的筛法**筛不出任何东西**。

**核查 2 — 原因。** 论文（arXiv 2410.23123）§2.2 原文：Perturber "…replacing a statement or a
leaf node in a statement with a newly sampled one, **and ensures the perturbation has a different
solution**"。数学级扰动在生成时就被强制要求"仍有（不同的）解"，无解候选直接被丢弃。

**核查 3 — 六种扰动的真实性质。**

| 扰动（train+test） | 层级 | 抽象 `statements` | 角色词 | 备注 |
|---|---|---|---|---|
| `perturbed_statement` 6,194+700 | 数学级 | 改变 | 不变 | 无解候选已被生成器筛掉 |
| `perturbed_leaf` 6,021+661 | 数学级 | 改变 | 不变 | 同上 |
| `reorder_statement` 6,200+700 | 语言级 | **逐字节同 clean** | 不变 | 仅调语序 |
| `uncommon_name` 6,200+700 | 语言级 | **逐字节同 clean** | 不变 | 仅换人名 |
| `flip_role` 6,200+700 | 语言级 | **逐字节同 clean** | **互换** | `knight_knave` 映射交换 |
| `random_pair` 6,200+700 | 语言级 | **逐字节同 clean** | **换词对** | angel/devil、sage/fool 等 6 组 |

**核查 4 — 题型只有一种。** 48,076 条 prompt 全部是 `So who is a knight and who is a knave?`
（句式统计 1000/1000 同尾句；角色词正则 48,076/48,076 匹配，0 未匹配）。对 `quiz` 列做关键词扫描：
`assume / assumption / suppose / hypothes / correct / valid / consistent / verify / is it true /
determine whether / which of the following` **全部 0 命中**。
"假设…判断…"的观感来自 **`cot_repeat_steps` 列**（`assume` 在该列出现 **362,925** 次），那是数据集
自带的合成 CoT（`Assume X is a knight. No contradiction is found…`），**是推理过程、不是题型**。
本阶段不做 SFT（D6）故不使用，但可留作 eval 参考与将来拒绝采样的材料。

**核查 5 — 重复度 7×（影响采样）。** 语料 = **约 6,900 个抽象题 × 7 个变体**（clean + 6 扰动）；
7 × 6,900 = 48,300，实际 48,076，差 224 行来自 `perturbed_statement`（缺 6）与
`perturbed_leaf`（缺 218）的重采样缺口。
其中四个语言级扰动与 clean 的**数学结构逐字节相同** → 从 48,076 行均匀采样 = 同一道题被喂 7 遍。
→ 必须按 `(len(names), index)` 分组去重（§4.2）。注意 `index` 单独**不唯一**（各人数档复用 100–1099）。
实测该键对 clean↔6 类扰动的联接覆盖率 **100%**。

**核查 6 — 角色词陷阱（会导致 reward 反向，见 §5.1）。** `solution` 是**规范语义**（`True` = 真话者），
表面词由该行的 `knight_knave` 字典给出。只有 `flip_role` 与 `random_pair` 真的换词：

```
pert_train_flip_role    knight_knave = {'knight': 'knave', 'knave': 'knight', 'a_knight': 'a knave', …}
    quiz     : … inhabited only by knaves and knights. Knaves always tell the truth, and knights always lie. …
               … So who is a knave and who is a knight?
    solution : [True, True]      solution_text: 'Oliver is a knave, and Ethan is a knave.'

pert_train_random_pair  knight_knave = {'knight': 'angel', 'knave': 'devil', 'a_knight': 'an angel', …}
```

两族合计 **13,800 行 = 全语料 28.7%**。硬编码 `K = knight` 会在这 28.7% 上**完全反向**。

**K&K 的新定位（D11）：小型可解锚点。**
四源里唯一的纯形式化可验证源，提供"硬逻辑 + 抗措辞扰动"信号；**不提供弃答信号**。

- 只用 **N ≥ 4**：2ppl（train 200 / test 100）瞎猜 25%，3ppl 12.5%，两档占 clean train 的 **19%**。
- 批内占比 **4–6%**（= 幻觉域 0.3 中的 15–20%）；`train_batch_size=256` 时约 **10–15 prompt/step**。
- 组内采样：每组只取 ≤1 个变体，clean : perturbed ≈ 1:1（否则 perturbed:clean 天然 6:1）。
- 不生成诊断选项、不进四档 reward（`has_diagnosis_label=false`）。

**License**：两个数据集经 HF API 核实均为 **cc-by-nc-sa-4.0（禁止商用）**。

**核查脚本**（未追踪，不入库）：`scratch/halluc_survey/probe_kk_cases*.py`、`kk_roles.py`、
`dump_kk_text.py`。

### 4.2 `kk_adapter.py`

- 输入：clean（`K-and-K/knights-and-knaves`）+ perturbed（`K-and-K/perturbed-knights-and-knaves`）
  两处，共 26 个 parquet（或等价的 jsonl 原始文件）。
- 产出：**全部 `solvable=true`**，无选项块、无诊断字段
  （`has_diagnosis_label=false`、`correct_option_id=null`）。
- **答案**：`ground_truth_answer` = 按 `names` 顺序的**表面角色词序列**（如 `angel devil devil`），
  同时写 `role_words = (knight_knave['knight'], knight_knave['knave'])`；
  规范 `solution` 布尔列表只留在 `extra_info` 供审计。
- **组内去重**：按 `(len(names), index)` 分组（每组 ≤7 条变体），每组只取 ≤1 条；
  同一组的成员**不得同时进 train 或 val 的同一侧**（按组划分 split，避免变体泄漏）。
- **筛除**：N < 4 的档位不进训练集（可留作 val 的简单档）。
- `difficulty_tag` = `len(names)`；`extra_info.perturbation_family` ∈
  `{clean, perturbed_statement, perturbed_leaf, reorder_statement, random_pair, uncommon_name, flip_role}`
  仅作监控分桶，不参与 reward。
- **不需要生成数据**（数据自带唯一解）；但保留一个枚举器作 gold 自检与 §9 的测试 oracle ——
  这是免费的、可随时重算的 ground truth。
- **必须**读 `knight_knave` 建 `role_words`。漏掉这一步 = §11 风险 9。

### 4.3 MiP 核查结论 + `mip_adapter.py`（**结论：不做诊断，退三档**）

- 来源：`github.com/tianyi-lab/MiP-Overthinking` 的 `data/{gsm8k,svamp,math,formula}.json`
  （**不是 HF 数据集**，仓库里没有 MiP-MMLU 文件；`data/` 之外全是模型 response 记录）。
- **实测规模**：合计仅 **984 条** —— gsm8k 582 / svamp 300 / math 52 / formula 50。
  构造方式是从原题**明确删掉或抽象掉一个数值前提**：`question` = 可解原题（带 `answer`/`solution`），
  `insufficient_question` = 残缺版 → **一行同时供给可解与不可解两版**，不需要另配对照。
- **Spike 2 的实际范围比原设想小得多**：984 条里 **634 条（64.4%）文件内自带原题**，不必回题库匹配 ——
  gsm8k 582 + math 52 都成对；svamp 300 只有残缺版 + `answer`（需回 SVAMP 公开题库匹配，
  且 `answer` 语义未验证）；formula 50 无 `answer` 无原题 → 废弃。

**核查：这 634 条能支撑"可校验的四档诊断"吗？→ 不能。三层审计全为否。**

| 审计层 | 检查 | 实测结果 |
|---|---|---|
| L1 标签证书 | 被删段重插回原题 == 原题 | 565/626 = **90.3%**（其余含 replace/insert，非纯删除） |
| L1 标签证书 | 被删数值在**原始答案推导链**里真的被用到（gsm8k `<<expr=val>>` / math `solution`） | 折算百分数后 gsm8k **90.0%** / math **86.7%**；仍查不到 **23 条（8.1%）** → 待复核清单，不是自动通过 |
| L2 gold 唯一性 | 删除段是否只含 **1 个**数值（= 只缺一个前提） | 209 条含 ≥2 个数值、55 条不含数值、37 条多段改动、26 条被删数字在残缺题面里仍可见 → 都不能当单选 |
| L3 反作弊 | 选项集能否被表层启发式绕过 | **H1「选"题面里唯一找不到的那一句"」命中 95.6%**；把数字抹掉后再这么选仍 **95.6%**；换跨题干扰项后用主题词重叠猜 → **83.7%** |

**L3 是结构性的，不是数据问题**：正确答案按定义**不在**题面里，干扰项按定义**都在**题面里 →
只要选项是"文本"、gold 是"被删掉的那句"，任何词面匹配（哪怕先抹掉数字）都能 ~96% 蒙对；
换成跨题干扰项，又退化成"哪个选项和题干更同主题"（84%）。**RL 一定会找到这条路。**

> 附带结论：也不存在"让模型自由写出缺失条件再匹配"的救法 —— 被删的**数值**在题面里已不可见
> （85.7%），模型无从得知，要求它写出数值 = 要求它猜；只能降到"缺的是哪个量"的槽位匹配，
> 那是另一条工程线（需改写 gold 与选项语义），本阶段不做。

**因此的决定（D12）：MiP 不做诊断，`has_diagnosis_label=false`，不可解侧只判"是否拒答"。**

- 输出 bare `\boxed{UNSOLVABLE}`，**prompt 不附选项块** → 与题面无关，**零泄漏**。
- 可解侧用**原题自带 `answer`** 做 exact-match（客观、无捷径）→ 可解/不可解两侧都变成可校验的。
- 这也把 §5.1 的**三档分支从"永不触发的死代码"变成有真实数据的分支**（§4.8 结论 1 随之改写）。
- 与 spec 的关系：spec §1 明写"若来源数据不支持细粒度诊断，`has_diagnosis_label=false`，
  退化为三档 reward"（spec 第 43 行）→ **是 spec 自己留的口子，不是偏离**。

**可入池漏斗（实测）**：984 → 634 有配对 → 626 题面真不同 → **325** 单值前提被移除
（311 纯删除 + 14 数值→占位词，如 `220`→`many`）→ **299** 该数值确实从可见题面消失 →
**262** 通过 L1 必要性检查（另 **23** 待复核、**14** 占位词型无需链校验）
→ **严口径 276 / 宽口径 299**。默认取**严口径 276**；209 多值 + 55 无数值 + 26 数字仍可见 + 37 多段 → 丢弃。
⚠️ v0.3/v0.4 记的"MiP 634 条带诊断标签"应读作"634 条可 diff"，真正可用 **276**（§4.8）。

**其余实现要点**：

- SVAMP 配对（仅 svamp 300 需要）成功 → 文本 diff 定位被删条件，写 `extra_info.deleted_condition_text`
  与 `metadata.paired_original_text`；**D12 后这两项只作审计，不进 reward**。
- 配对失败 → 直接丢弃该条（不伪造粗粒度选项）。
- `solvable` 两版各出一行：可解行 `ground_truth_answer` = 原题答案、`perturbation_type=null`；
  不可解行 `has_diagnosis_label=false`、`correct_option_id=null`、`perturbation_type=missing_condition`。
- **MiP 可解侧（原题 634 条）默认不进池**：与 Big-Math 主池同源（GSM8K/MATH）、量小、去重成本高于收益（§12 Q20）。
- 必须交付 `verify_mip.py`：把 L1/L2 做成断言（证书 + 必要性 + 唯一性），并对**任何将来启用的选项集**
  把 L3 的 H1/H2/H4 做成断言（必须落回接近随机），见 §12 Q21。

**核查脚本**（未追踪，不入库）：`scratch/halluc_survey/probe_mip_options.py`（L1+L2）、
`probe_mip_leak.py`（L3 三种启发式）、`probe_mip_funnel2.py`（漏斗）、`probe_mip_necessary.py`（必要性）。

### 4.4 `falseqa_adapter.py`（**v0.6 重写：gold 从 `answer` 换成配对 diff 片段**）

**源事实（实测 `fq_train.csv` / `fq_valid.csv` / `fq_test.csv`）**

- 列只有 `question, answer, label`。`label=1` = 假前提，`label=0` = 真前提，三份 split 都**严格 50:50**
  （train 1,187/1,187，valid 491/491，test 687/687）。
  → **可答性是有标注的**；"看起来没有标注"应该来自下一行。
- **`answer` 列不能当 reward gold**：`label=0` 侧 67.8% 是自由文本短答（`Because cats are much larger
  than mice.`），只有 ~4% 是 yes/no 或数值；`label=1` 侧是自由文本 rebuttal，test split 更是
  **每条 3 个并列参考答案**（`['…','…','…']`）——连"唯一正确字符串"都不存在。
- **配对是索引对齐的**：第 k 条 `label=1` = 第 k 条 `label=0` 的局部改写，词级 Jaccard ≥0.5 占
  **89.1% / 88.6% / 90.2%**（train/valid/test）。**这是本源自带的"配对证书"**，不需要再去挖 MUS。

**为什么它反而是四档诊断的唯一出路（与 MiP 的对照）**

| | MiP（§4.3） | FalseQA |
|---|---|---|
| 假前提怎么造的 | 把条件**删掉** | 把片段**替换/插入** |
| 题面里还能不能看见 gold | ❌ 被删数值 **85.7% 不可见** | ✅ 假前提片段**就在题面里** |
| pointer gold 是否可知 | ❌ 不可知（只能退到槽位匹配） | ✅ 由配对 diff 唯一确定 |
| 结论 | 退三档（D12） | **保留四档（D13）** |

**三层审计（与 §4.3 同一套框架）**

| 层 | 检查 | 实测（train / valid / test） |
|---|---|---|
| **L1 证书** | fake 侧差异区域是否恰好一个连续片段（word-level `difflib`，`autojunk=False`）；gold 是否含内容词 | 单一区域 **928（78.2%）** / 377（76.8%）/ 536（78.0%）；含内容词 **98.9%**；gold 词数 median **1**、单 token **66.7%**；gold 首词在题面唯一出现 96.3% |
| **L1 旁证** | 数据集自己的 rebuttal 是否**指名**该片段（独立佐证，只做旗标、不当 gold） | 仅 **56.8%** / 56.2% / 85.3% → **审计旗标**：按此字段做一层抽样人工复核（§9） |
| **L2 唯一性** | 同题能否造出**等长**干扰跨度、凑够 **k=3（D15）** | **657（55.3% 原始 / 70.8% 的 pointer gold）** / 259 / 388 |
| **L3 反作弊** | 见下表 | 同题选项把 MiP 的 H1 **结构性堵死**（0.4% ≈ 0）；跨题选项 **99–100% 可蒙** |

**L3 反作弊实测（train，928 条 pointer gold；选项位置已按 seed 打乱）**

| 表层启发式 | 跨题干扰项（v0.5 设想：其他条的 rebuttal 句） | 同题·**等长**·k=3（**D15 锁定**） | 同题·**等长**·k=4 |
|---|---|---|---|
| "选题面里**唯一找不到**的那一项" | —（都来自别的题，不适用） | **≈0%** | **≈0%** |
| "选与题干**主题重叠最大**的那一项" | **99.1%** | — | — |
| "选最长" | — | 30.9%（随机 33.3%） | 24.1%（随机 25.0%） |
| "选唯一带大写 / 带数字的" | — | 30.9% / 34.1% | 24.9% / 27.9% |
| 题面词袋 NB 5-fold CV（判"假前提/真前提" AUC） | — | 0.172 / 0.828 | — |
| **可构造的样本数** | — | **657（70.8%）** | **473（51.0%）** |

- **跨题干扰项 = 完全不可用**：99.1%（train）/ 99.5%（valid）/ **100%（test）** 可被"主题重叠最大"
  直接蒙对，比 MiP 那条 83.7% 更糟。→ **选项必须全部取自本题面**（D13），
  v0.5 §4.6 的"同模板近邻"方案在 FalseQA 上**禁用**。
- 同题选项把 MiP 的致命捷径（"选题面里唯一找不到的那一句"95.6%）**结构性消灭**（≈0）：
  所有选项都出现在题面里，不存在"看谁不在题面"这条路。**这是四档能成立的核心原因。**
- **真正在做反作弊的是"等长"这个约束，不是 k**：同题 + 等长下，k=3 与 k=4 的表层启发式**都 ≈ 随机**。
  于是 k 的选择**纯粹是覆盖率问题**（657 vs 473），不是抗蒙问题。
  ⚠️ v0.6 的表把"**近似等长**·k=4"与"**严格等长**·k=3"并排，同时改了两个变量，
  造成"k 越大越容易被蒙"的错误读数；v0.7 已更正（近似等长时"选最长"确实到 38.8%，
  但那是**没强制等长**的锅，降 k 只贡献很小一部分）。
- **k 一大池子就断崖的原因**：题面 median 只有 **5** 个内容词，扣掉 gold 后"等长且不与 gold 重叠"
  的候选池 median 只有 **3**（mean 2.66）。所以"要 2 个"与"要 3 个"是在小整数分布上跨阈值：
  P(池≥2) = 70.8%、P(池≥3) = 51.0%，**差的 19.8pp 全部来自"池恰好 = 2"的那 184 条**。
  按 gold 词数拆：gold=1 词（619 条）池 median 3 / P(≥3) 73.5%；**gold=2 词（201 条）池 median 1 /
  P(≥3) 9.0%**；gold=3 词（52 条）池 median 0。
- 若再叠加"同形态（大写/数字也要一致）"：k=3 → 551（59.4%）、k=4 → 358（38.6%），基线同样≈随机。
  所以本阶段**只强制等长，不强制同形态**（D15）。
- 词袋 NB 判"假前提"AUC 只有 0.172（判"真前提"0.828）：**不存在"看题面文风就能判假"的捷径**，
  那点弱信号方向与任务一致（"正常问句 → 照常作答"），且给不出 gold 片段。

**漏斗**：train 1,187 假前提 → **928**（78.2%，单一差异区域 = pointer gold）→ **657**（55.3%，
k=3 等长选项可构造）；valid 491 → 377 → 259；test 687 → 536 → 388（评估用）。

**可解侧（`label=0`）怎么用 —— D14：以"是否可答"的判断入池**

`answer` 判不了分（67.8% 自由文本），所以**不让模型作答，而是让模型判断"该题前提是否成立/是否可答"，
用 `label` 当 gold**。这样可解侧立刻变得可机检，并且直接补上 §11 风险 12 的过度拒答洞。

| | `label=0`（真前提） | `label=1`（假前提） |
|---|---|---|
| 期望输出 | `\boxed{SOLVABLE}` | `\boxed{UNSOLVABLE: <选项ID>}` |
| gold | `label=0` | `correct_option_id` |
| 得分 | 对 → **+1**；输出 UNSOLVABLE → **−1**（误拒）；其它 → 0 | 选中 gold → **+1**；编造答案 → **−1**；其它 → 0 |
| 可入池（train） | **851**（选项块可构造 71.7%） | **657** |

- **⚠️ 模板必须在本源内完全统一（否则 100% 泄漏）**：两个 label 的 prompt **都**先问"前提是否成立"、
  **都**附选项块。若只有 `label=1` 附选项块，模型可以直接学"看到选项块 ⇒ 输出 UNSOLVABLE"，
  判断那一半就白送了（而 MiP 侧没有选项块，这个捷径在 MiP 上还不成立 → 迁移更糟）。
- `label=0` 的选项块同样用**同题等长跨度**构造：锚点取该条的**配对 real 侧 diff 片段**的词数/形态，
  再按 §4.6 的规则抽 3 个（`label=0` 侧没有"正确项"，选项块只为保持模板同构）。
- **误拒给 −1 是关键**：它让 `-1` 从"只在不可解侧出现"变成**双向对称**（编造 / 误拒各一种），
  正好回应 §11 风险 3（负奖励量纲与拒答塌缩）。
- **⚠️ 判断本身有相当一部分是"主题记忆"，不是文风破绽**（实测，5-fold CV）：
  题面词袋 NB 最优阈值 balanced-acc = **0.765 / 0.833 / 0.807**（train/valid/test）。
  拆开看：**只用虚词 → 0.569 / 0.601**（几乎没有文风信号），**只用实词 → 0.757 / 0.802**。
  → 信号来自"哪些实体/主题在这份数据里偏假"（space / Confucius / Mars / abalone…），
  是**闭集内的主题记忆**，不是可迁移的推理。所以：
  (a) 判断这一半**便宜且可机检，但外推性存疑**，不要把它当作能力指标；
  (b) 真正难的是**四档诊断**（要指向题面里那一处片段），主题记忆给不出片段；
  (c) §10 必须**分开**看这两个指标（§10 已拆）。
- `label=1` 侧没用上的 530 条（多差异区域 259 + 凑不满等长选项 271）本阶段**丢弃**；
  将来若愿意把模板再扩一档（"没有合适选项就输出 bare `\boxed{UNSOLVABLE}`"）可以收回来，见 §12 Q25。

**交付约束（写进 `falseqa_adapter.py`）**

- gold = **配对 diff 片段**（fake 侧、恰好一个连续区域）——不是 `answer` 字段。
- 选项 = **同题**内容跨度（unigram/bigram），**强制与 gold 等长**，**k=3（D15）**，位置按 seed 打乱，写进 `extra_info.options`。
- 选项凑不满 3 个 → **该条直接丢弃**，禁止用跨题/同模板兜底。
- `correct_option_id` 指向 gold 段，`has_diagnosis_label=true`，`perturbation_type=contradictory_condition`。
- `label=0` 侧（D14）：`has_diagnosis_label=false` + `judgment_only=true`，**两个 label 共用同一套模板**，
  gold 就是 `label`；选项块按配对 real 侧片段的词数/形态构造，仅为模板同构。
- 交付前必须跑 §9 的 L3 断言（`verify_falseqa.py`）并在报告里留数（§12 Q21）。

### 4.5 `gsm_ic_adapter.py`

- 来源：**原始 repo** `github.com/google-research-datasets/GSM-IC` 的
  `GSM-IC_2step.json` + `GSM-IC_mstep.json`。
  ⚠️ 修正：HF 上的 `voidful/GSM-IC` **只有 1,000 行**（一个小子集），不能当作全量；
  两份原始文件实测为 **2step 34,220 + mstep 23,832 = 58,052 条**。
- 字段：`original_question` / `answer` / `new_question`（插入无关句后的题面）+
  `role` / `number` / `sentence_template` / `role_label` / `number_label` / `sentence_label`
  （标注插入的那句无关句属于 `in_topic` 还是 `out_of_topic` 等）。
- 性质：**可解但含无关干扰条件**，`solvable=true`，`perturbation_type=distracting_condition`。
- 用途：与不可解数据互补，训练"不被无关信息带偏"。
- 附注：它的 `sentence_label` 是一份**干扰句的真值标注**，理论上可支撑"指出哪句是无关条件"
  这类辅助任务；但**不是**不可解诊断，本阶段不启用（§4.8 表内计为 0）。
- reward 走现有数学匹配（`halluc_math_gsmic` → 委托 `_math_score`），不新增判分逻辑。
- **v0.10：GSM-IC 封顶 2,000 条，但它的 `sentence_template` 被抽出来当干扰项合成引擎（D17）**。
  实测 58,052 行里共 **242 个不同模板**，前 4 个就覆盖 61%：
  `The shoe size of {role} is {number}.` / `{role} bought {number} tomatoes from the grocery store.` /
  `The height of {role} is {number} feet.` / `{role} is {number} years old.`。
  全部只由 `{role}`（272 个不同取值）与 `{number}`（55 个）参数化 → **纯规则可重放，不需要 GSM-IC 的生成代码
  （该 repo 只发布了两个 JSON，没有 generator）**。施加到其他池的可解原题上时答案不变，
  这正好构成"**多一句无关条件（仍可解）**"与"**少一条必要前提（不可解）**"的结构性对照对。
  难度梯度直接沿用 GSM-IC 自带的三组标注，且天然配平（§4.9 表 C）。

### 4.6 干扰项挖掘（`distractor_mining.py`）—— **v0.6 起只做同题跨度**

- **跨题方案全部禁用**：v0.5 设想的"同模板/同源词近邻"（取其他假前提条目的 `answer` 当干扰项）
  实测 **99.1% / 99.5% / 100%** 可被"选与题干主题重叠最大的那项"蒙对（§4.4 L3）。
  spec §2.4 预留的 embedding 近邻**即使打开也是同一个洞**（同主题就更容易被主题匹配区分），
  除非干扰项与题干**同主题**——那就等于又变成同题候选。→ 本阶段**不启用**，不新增依赖。
- 本阶段实现（只服务 FalseQA，D13）：
  1. 同题跨度池 = 题面内容词 unigram + bigram，排除与 gold 重叠的 token；
  2. **过滤 `len(span) == len(gold)`**（等长是反作弊的关键约束，不是"优化项"，§4.4）；
  3. 按 seed 抽 k−1 = 2 个，与 gold 一起打乱位置。
- **等长约束不是"优化项"，是反作弊的主约束（v0.7 定性）**：若只按 `|len−|gold||` 近似等长，
  单词干扰会挤掉多词干扰，"选最长"命中就到 38.8%；强制等长后所有表层启发式≈随机。
  v0.6 把它误记成"k 从 4 降 3 的功劳"，v0.7 更正（§4.4 L3）。
- 统一产出 `list[{id,text}]`（**k=3，D15**：1 正确 + 2 干扰）；凑不满直接丢弃该条，**不做跨题兜底**。
  `label=0`（D14）侧同样出一份 **k=3（D15）** 等长选项块（锚点取配对 real 侧片段），只为**保持模板同构**——
  ⚠️ **两侧 k 必须相同（D15）**：若一侧 3 个、一侧 4 个，"选项个数"本身就是一条一眼可见的捷径。
- K&K / GSM-IC 全可解、MiP 按 D12 无选项 → **本阶段唯一消费方是 FalseQA**（两个 label）。
- ⚠️ 交付时必须附带 §4.4 的 L3 反作弊断言（阈值：随机 + 10pt；§12 Q21）。

### 4.7 被移除 / 登记为替补的数据源（D9 / D10）

spec §2.2 的「UMWP 兜底」与 §2.3 的「AbstentionBench 聚合入口」在落地前做了核查，
结论是**两者都不进本阶段训练源**，但原因不同。

**AbstentionBench：移出训练源，降级为评估套件。**
它是 Meta 的聚合 benchmark，20 个子集里**直接包含 `FalseQA` 与 `UMWP`**——与我们已选的源重复；
同时包含 `GPQA` / `GSM8K` / `MMLU Math`，其中 `GSM8K` 子集是它**从 `openai/gsm8k` 的 test split
现场构造**的（删掉 context 句造不可解变体，保留的 answerable 变体就是 GSM8K test 原题+答案），
`MMLU Math` / `GPQA` 同理是 benchmark 本体 → **训练它会污染 `DESIGN.md` §7 的固定评估协议**。
其余子集基本只有 `should_abstain` 布尔标签（三档 reward），而本阶段（D4 后）没有三档数据的位置。
license 为 cc-by-nc-4.0（非商用）。→ 保留它**作为评估套件**；将来若要扩三档不可解覆盖，
逐个挑原始源（KUQ / CoCoNot / QAQA / SQuAD2 / Musique…）并逐源查污染与 license，不训聚合体。

**UMWP：不是重复，但与 MiP 抢同一生态位，本阶段登记为替补。**

- **可配对性优于 spec 假设**：UMWP 原生保证配对——answerable 与 unanswerable 各 2600 条，
  `i ↔ 2600+i` 来自同一道原题，且每行带 `relevant_ids`。所以 spec §2.2 说的
  「找不到配对就退化成 5 类固定选项」在 UMWP 上**基本不成立**，它同样能做 diff 生成精确诊断选项。
  → 一旦 MiP spike 2 配对率不达标，UMWP 是首选替补（且可能比 MiP 更容易建）。
- **但诊断语义只有约 1/3 契合**：不可解类别构成为 Key Information Missing 32% /
  Ambiguous Key Information 49% / Unrealistic Conditions 11% / Unrelated Object 4% /
  Question Missing 5%。后四类不是"缺一条具体条件"，需要重新定义"正确选项"的语义。
- **底题与 MiP / 现有主池重叠**：answerable 侧来源为 SVAMP 500 / MultiArith 300 /
  **GSM8K 1700** / ASDiv 100，与 MiP（SVAMP/GSM8K/MATH/MMLU）共享 GSM8K/SVAMP 底题，
  且这批底题也大量在 Big-Math 主池里；变体不同、底题同源，靠题面去重会砍掉可观数量。
- license 为 CC-BY-SA-4.0，论文 ethics 明确禁止商用（MiP 的 license 在 spike 时一并确认）。

**替补触发条件：已于 v0.10 触发（D16），UMWP 正式进训练源。** v0.9 之前卡住的两点现在都解了：
① 「其余四类选项语义没设计清楚」→ §4.4/§4.6 的**同题等长跨度**方案对所有类别通用，
且实测 UMWP 的四类可见缺陷（cat2–cat5，合计 1,760 条）产出的表层启发式全部≈随机（§4.9 表 A）；
② 「不可解池 0.93k 不够」→ SUM/TreeCut/KUQ/CREPE 一起补上，池子到 ~19k。
**唯一保留的类别约束**：cat1（Key Information Missing，840 条）缺陷在题面里不可见 →
**只能走三档 bare**，不做四档（与 MiP/D12 同理）。

参考：[AbstentionBench HF](https://huggingface.co/datasets/facebook/AbstentionBench)、
[AbstentionBench GSM8K loader](https://github.com/facebookresearch/abstentionbench/blob/main/recipe/abstention_datasets/gsm8k.py)、
[UMWP 论文](https://ar5iv.labs.arxiv.org/html/2403.03558)、[UMWP repo](https://github.com/Yuki-Asuuna/UMWP)。

### 4.8 诊断标签覆盖（已实测，回答"带诊断标签的到底有多少"）

口径：**"带诊断标签"= 该条不可解样本能给出可 exact-match 的"具体哪一项缺失/矛盾"选项 id**
（即 `has_diagnosis_label=true`，走四档 reward）。只是"知道它不可解"不算。

**各源实测（原始计数，未去重、未过滤）**

| 源 | 原始量 | 带诊断标签的量 | 占比 | 备注 |
|---|---|---|---|---|
| K&K perturbed | 37,015 (train) / 4,161 (test) | **0** | 0% | ⚠️ 实测 41,176 行**全部唯一可解**：生成器强制"扰动后仍有不同的解"（§4.1 核查 1/2） |
| K&K clean | 6,200 (train) / 700 (test) | 0 | — | 可解；与 perturbed 合计 **48,076 行零弃答数据** |
| MiP 合计 | **984** | **0**（主动放弃，D12） | 0% | 可 diff 634 → 三层审计后可用 **276**，但 L3 反作弊失败（95.6% 可蒙）→ 退三档，§4.3 |
| FalseQA | train 2,374 / valid 982 / test 1,374，各 50:50 | **657**（train，D13 审计后；原始假前提侧 1,187） | 审计后 55.3% / 原始 50% | gold 改为配对 diff 片段、选项改同题等长跨度后可保四档，§4.4。另：**可解侧 851 条**以"是否可答"的判断入池（D14） |
| GSM-IC | **58,052**（2step 34,220 + mstep 23,832） | 0 | 0% | 全部可解；`sentence_label` 是干扰句标注，不是不可解诊断 |

**结论 1（结构性；v0.6 微调）**：**四源里 `has_diagnosis_label=true` 只剩 FalseQA 一家，且它从"有风险"变成"可机检"。**
- K&K / GSM-IC 侧本来就没有不可解数据；MiP 有不可解数据但**主动放弃诊断**（D12）→
  `solvable=false` 与 `has_diagnosis_label=false` **不再等价**（v0.4 的结论 1 作废）。
- 三档分支（bare `\boxed{UNSOLVABLE}`）有真实数据（MiP 不可解侧 276 条）；
  四档分支有 **657 条单一来源**（D13），且选项已过 L3 反作弊（§4.4）——v0.5 记的"该源自身有表层捷径风险（洞 1）"**已解决**。
- → Q12（FalseQA 是否也降三档）**目的已消失**：v0.6 关闭该问题，四档保留。
- **v0.7 补**：FalseQA 的 `label=0` 侧以**判断型可解**入池（D14，851 条），
  于是契约里多出第五种输出形态 `\boxed{SOLVABLE}`；它同时也是**四档诊断的对照组**
  （同一模板、同一选项块外观，只有"前提成立与否"不同）。

**结论 2（mix 层面的分支占比）**：`train_batch_size=256`，默认 `--old_domain_ratio 0.7`
+ `--halluc_solvable_ratio 0.7` + `--kk_share 0.15` + `--falseqa_judge_share 0.15`
+ 不可解侧按池内比例（MiP : FalseQA ≈ 30 : 70，即 `--mip_share 0.30`）：

| 契约分支 | 批内占比 | prompt/step |
|---|---|---|
| 老四域（委托现有 reward） | 70.0% | ~179 |
| 幻觉域·可解·数值（GSM-IC） | 12.0% | ~31 |
| 幻觉域·可解·角色词序列（K&K，D11） | 4.5% | ~12 |
| **幻觉域·可解·判断型（FalseQA-real，`\boxed{SOLVABLE}`，D14）** | **4.5%** | **~12** |
| **幻觉域·不可解·三档 bare（MiP，D12）** | **2.7%** | **~7** |
| **幻觉域·不可解·四档诊断（FalseQA-fake，D13）** | **6.3%** | **~16** |

（推导：幻觉域 = 256 × 0.3 = 76.8；不可解 = 76.8 × 0.3 = 23.0，其中三档 = 23.0 × 0.30 = 6.9、
四档 = 23.0 × 0.70 = 16.1；K&K 与 FalseQA-judge 各取幻觉域的 0.15 = 11.5，可解预算的余量 30.8 给 GSM-IC。
v0.5 表里的 1.7% / 7.3% 用的是 `--mip_share 0.19` 与未经审计的 FalseQA 池。）

抬高两个比例可把不可解侧推到 15%/25%，但 **0.93k** 的池子会被更快耗尽（结论 3）。
**v0.10 起本表只作历史记录**：幻觉域规模与分支配额改由 **D18 / §4.9.3 表 B** 定义（20,000 条、
不可解 12,000、6 个来源），`--old_domain_ratio` / `--halluc_solvable_ratio` 两个旋钮的语义
跟着改成"对 20,000 条的切分"，不再按 0.93k 的池子反推。

**结论 3（供给侧上限）**：

| 侧 | 需求（= 默认设置下批内占比） | 供给 | 结论 |
|---|---|---|---|
| 可解（数值/角色词） | 16.5% | **≈66k–73k**（GSM-IC 58,052 + K&K 组内去重后 ≈6,900 组） | 宽裕 |
| 可解（判断型，D14） | 4.5% | **851**（FalseQA-real） | 够用；但注意它是**主题记忆型**信号（§4.4） |
| 不可解 | 9.0% | **≈0.93k**（MiP 276 + FalseQA-fake 657） | ⚠️ **本方案最紧的约束** |

（K&K 不做组内去重时是 48,076 行、可解池 107.3k；v0.4 写的 72.3k 是笔误。
v0.5 写的不可解池 ≈1.46k 用的是 FalseQA **未经审计的原始侧 1,187**；D13 审计后只剩 657 → **0.93k**。）

> **v0.10 重算（D16/D18）**：本节的"0.93k 是最紧约束"**已作废**——SUM + UMWP + TreeCut + KUQ + CREPE
> 把不可解池推到 **≈19.4k**（§4.9 表 A），幻觉域按 D18 定为 **20,000 条（不可解 12,000 / 可解 8,000）**，
> 详见 §4.9.3。同时修正本节两处口径：
> ① **K&K 的训练池是 5,000 组，不是 6,200**——§4.2 明确筛除 `N<4` 的档位，而 `clean_train` 的
> 2ppl 200 + 3ppl 1,000 正好是被筛掉的 1,200 条（test 同理 700 → 500）；
> ② 因此"可解池 ≈66k–73k"应改为 **≈63.9k**（GSM-IC 58,052 + K&K 5,000 + FalseQA-real 851）。

**由此暴露的设计洞（v0.7 改写）**：
1. ~~**FalseQA 的"选项"语义有表层捷径**（跨题 rebuttal 干扰项可被主题匹配蒙对）~~ →
   **v0.6 已解**：把选项源从"跨题 rebuttal 句"换成"**同题等长跨度**"后，跨题捷径（99.1%）消失，
   同题内部唯一残余的"选最长"（38.8%）也**由等长约束**压回随机（30.9% vs 随机 33.3%；k 不是关键变量，
   v0.7 更正）。实测见 §4.4 L3。
2. ~~三档分支无真实数据~~ → **v0.5 已解**：MiP 退三档后，bare 分支有 276 条真实数据。
3. ~~**可解侧整体缺位**（常识题型在训练里只有"假前提"一个方向）~~ → **v0.7 已解**：
   按 D14 把 `label=0` 以"是否可答"的判断形式入池（851 条，gold = `label`），
   误拒给 −1 → 模型有明确的"别乱拒答"压力。**代价**：这个判断本身有 76–83% 的
   主题记忆成分（§4.4 H6），外推性存疑 → 已升级为 §11 风险 12 的新形态。


---

### 4.9 不可解侧扩源与干扰项合成（**v0.10 新增；D16 / D17 / D18**）

本节只收录**已实际下载并逐条量过**的源。每个源给下载方式、license、实测行数、配对性、
L1 证书、L3 反作弊结论。**L3 一律用同一把尺**：题面词袋 Naive Bayes 五折 balanced accuracy
（随机 = 0.5）。对照组是 §4.4 的 FalseQA H6：**0.765 / 0.833 / 0.807**。

#### 4.9.1 候选源可用性（表 A）

| 源 | 取数方式（已验证） | license | 实测规模 | 配对 | L1 证书 | L3（实测 BoW NB） | 定档 |
|---|---|---|---|---|---|---|---|
| **SUM** | HF `lime-nlp/Synthetic_Unanswerable_Math` 单 parquet | MIT | train **36,480**（+test 284） | ✅ 同行 `answerable_question` | ⚠️ o3-mini 生成 + 专家复核；**无机器证书**（只有最终 `ground_truth`，无推导链） | **0.502 / 0.503** ✅ | 四档 9,832 / 三档 6,050 |
| **UMWP** | GitHub raw `data/StandardDataset.jsonl` | CC-BY-SA-4.0 | **5,200**（2,600 / 2,600） | ✅ **100%**（`relevant_ids`） | ⚠️ 人工构造 | **0.500 / 0.500** ✅ | 四档 1,449 / 三档 440 |
| **KUQ** | HF `amayuelas/KUQ` → `knowns_unknowns.jsonl` | MIT | **6,884**（unknown 3,437 / known 3,447） | ✅ 同文件双类 | ⚠️ 众包 + 六类标注 | 未测 | 三档 / 判断型 |
| **TreeCut** | 生成器 repo（纯 Python 无依赖）+ HF `jouyang/treecut-math` 21,000 样例 | Apache-2.0 | **可无限生成** | ✅ 同参数生成对照 | ✅✅ **树结构保证必要性**，`proof` 自带凭证 | ❌ **0.755**（长度 0.726） | 三档（**须先修 L3**） |
| **CREPE** | HF 镜像 `tasksource/CREPE`（**官方 Google Drive 链接已 404**） | BSD | 8,466（3,462 / 2,000 / 3,004） | ❌ | ⚠️ 人工标注，但 `presuppositions` 仅 **1.1%** 是题面逐字 span | 未测 | **仅判断型** |
| CoCoNot | HF `allenai/coconot`（`original` / `contrast` config） | 待核 | train ~11,477 | — | ⚠️ | 未测 | **本轮不纳入** |
| GSM-DC | HF `YMinglai/GSM-DC-Dataset-Sample` | 未标注 | **仅 6,300 样例** | 符号图 | ⚠️ 合成风格生硬 | 未测 | **不纳入** |
| SQuAD 2.0 | HF `rajpurkar/squad_v2` | CC-BY-SA-4.0 | train 不可解 43,498 | ❌ | ❌ 判"不可解"需 NLI | — | **排除** |

参考：[SUM](https://huggingface.co/datasets/lime-nlp/Synthetic_Unanswerable_Math)、
[UMWP](https://github.com/Yuki-Asuuna/UMWP)、[KUQ](https://huggingface.co/datasets/amayuelas/KUQ)、
[TreeCut 论文](https://arxiv.org/abs/2502.13442)、[TreeCut repo](https://github.com/j-bagel/treecut-math)、
[CREPE 镜像](https://huggingface.co/datasets/tasksource/CREPE)、
[AbstentionBench](https://ar5iv.labs.arxiv.org/html/2506.09038)。

#### 4.9.2 三个必须记录的坑（"确保数据处理正确"）

1. **CREPE 的标签串是 `'false presupposition'`（空格）**，而 README 写的是 `'false_presupposition'`（下划线）。
   按 README 过滤会得到 **0 行**。实测 train 分布：`('normal',)` 2,535 / `('false presupposition',)` 907 / 双标 20
   → **真值取 927 条**。adapter 里必须按实测串匹配并加断言（§9）。
2. **CREPE 的 `presuppositions` 不是 span 而是转述**：1,295 个片段里只有 **14 个（1.1%）** 逐字出现在
   `question` 中（其余是改写，如 Q "Does eating spicy food destroy…" → pres "Eating spicy food could
   potentially destroy…"）。**exact-match 指针 gold 因此不成立** → CREPE 只能进判断型。
   这是它与 FalseQA 的本质区别（FalseQA 的同题 diff 片段是**逐字**的）。
3. **TreeCut 的可解/不可解两类天然长度不同**：`gen_data.py` 删掉一整句，实测可解 351.6 字符 vs
   不可解 303.6。于是长度启发式 **0.726**、BoW NB **0.755**——与 FalseQA 的 H6 同级，属于**结构性泄漏**。
   **修法**：负类不取"完整题"，改取"**剪掉一条非必要边**"的版本（句子数、变量数、长度分布全对齐，
   可解性不变；`cut` 只在 `ans_upstream` 上取，非必要边的集合由 `get_all_edges()` 减去 `ans_upstream` 即得）。
   **修完必须重跑 L3，落到 ≈0.5 才允许入池**（§9 硬门槛）。

**TreeCut 的 L1 为什么值钱**：`cut = ans_upstream[cutDepth - 1]` → 剪掉的边**必在根→答案路径上**；
`all_edges = [e for e in edges if e[1] != cut]` → 删掉所有以 cut 为头的边；`gen_disproof` 输出的
`proof` 就是保留边集 + "N variables but M formulas"。**不可解性是构造即证明**——这正是 MiP 缺的东西
（MiP 只有 90.3% 重建证书、8.1% 必要性查不到，§4.3）。

**同源污染加重**：SUM 源自 DeepScaleR、UMWP 源自 GSM8K/SVAMP/MultiArith/ASDiv、TreeCut 为合成数学题，
全部与 Big-Math 主池重叠 → **§7.2 的池内近重去重从"可选"升级为硬前置**（§11 风险 2）。

#### 4.9.3 20,000 条目标构成（表 B；D18）

按**契约分支**分配（这是 reward 直接消费的轴）：

| # | 分支 | 模板 | gold | 来源与配额 | 小计 |
|---|---|---|---|---|---|
| 1 | 可解·数值 + 干扰项 | B | `\boxed{答案}` | GSM-IC 2,000 + **模板合成干扰 2,400**（D17） | **4,400** |
| 2 | 可解·角色词序列 | B | `\boxed{角色词 …}` | K&K 2,000 | **2,000** |
| 3 | 可解·判断（**带**选项块） | A | `\boxed{SOLVABLE}` | UMWP-answerable 550 + SUM-answerable 350 | **900** |
| 4 | 可解·判断（**不带**选项块） | B | `\boxed{SOLVABLE}` | CREPE-normal 500 + KUQ-known 200 | **700** |
| 5 | **不可解·四档诊断** | A | `\boxed{UNSOLVABLE: <选项ID>}` | SUM-visible 5,094 + UMWP-visible 1,449 + FalseQA-fake 657 | **7,200** |
| 6 | **不可解·三档 bare** | B | `\boxed{UNSOLVABLE}` | SUM-del 2,000 + TreeCut 1,084 + UMWP-del 840 + CREPE-FP 400 + MiP 276 + KUQ(FA+CF) 200 | **4,800** |
| | | | | **合计** | **20,000**（不可解 **12,000 = 60.0%**） |

按**缺陷类型**配平（可解侧 8,000 之外，仅列不可解 12,000）：

| 缺陷类型 | 档位 | 目标 | 占比 | 承载源 |
|---|---|---:|---:|---|
| 缺一条必要条件（题面**不可见**） | 三档 | 4,200 | 35.0% | SUM-del 2,000 / TreeCut 1,084 / UMWP-cat1 840 / MiP 276 |
| 关键信息歧义 | 四档 | 3,000 | 25.0% | SUM-visible ≈1,960 / UMWP-cat2 1,040 |
| 前提不现实·自相矛盾 | 四档 | 1,400 | 11.7% | SUM-visible ≈1,174 / UMWP-cat3 226 |
| 假前提（题面**可指认**） | 四档 | 657 | 5.5% | FalseQA-fake 657 |
| 假前提（**不可指认**） | 三档 | 600 | 5.0% | CREPE-FP 400 / KUQ(FA+CF) 200 |
| 问题缺失 | 四档 | 1,140 | 9.5% | SUM-visible ≈1,042 / UMWP-cat5 98 |
| 无关·未定义实体 | 四档 | 1,003 | 8.4% | SUM-visible ≈915 / UMWP-cat4 85 |
| **合计** | | **12,000** | **100%** | |

- **SUM 没有类型标注**（只有 3 列：`answerable_question` / `unanswerable_question` / `ground_truth`）。
  表里的类型分配靠**规则 diff 分类器**（沿用 §4.4 的 `difflib` opcode 投影）：纯删除→缺前提；
  替换成模糊量词→歧义；插入负数/不可能值→不现实；插入未定义实体→无关实体；删掉问句→问题缺失。
  **该分类器只服务配平与监控，不参与 reward**；它自身要过 §9 的抽检（N=100）。
- **同一缺陷类型跨档位是正常的**（假前提既有可指认的四档、也有不可指认的三档），
  这反而是好事：同类型跨两个模板 → 削弱"选项块 ⟺ 不可解"的关联。

#### 4.9.4 干扰项合成（表 C；D17）

GSM-IC 的 242 个 `sentence_template` 全部只由 `{role}` / `{number}` 参数化，**纯规则可重放**，
施加到其他池的可解原题上时答案不变。难度梯度直接用 GSM-IC 自带的三组标注，且天然配平：

| 维度 | 标注字段 | 实测取值分布 | 配平 |
|---|---|---|---|
| 干扰句是否切题 | `sentence_label` | `out_topic` 18,816 / `in_topic` 15,404 | 55 : 45 |
| 角色是否与题干人物重合 | `role_label` | `nonoverlapped` 16,990 / `overlapped` 16,902 / `n/a` 328 | ≈50 : 50 |
| 数值是否落在题干数值范围内 | `number_label` | `in_range` 17,080 / `out_range` 17,080 / `n/a` 60 | ≈50 : 50 |

- **施加对象**（表 B 第 1 行的 2,400 条）：SUM-answerable 1,000 / UMWP-answerable 600 / K&K 400 / 主池数学题 400。
- **答案与 reward 完全不变**：gold 仍是原答案，走 `_math_score`；不新增判分逻辑、不新增数据源依赖。
- **成对性**：干扰项（**多一句无关条件，仍可解**）与 SUM-del/TreeCut/UMWP-cat1/MiP（**少一条必要前提，不可解**）
  构成本方案最想要的对照对。**硬约束**：两者的底题必须来自同一分布（同一 source split），
  否则"是哪个底题库"会变成捷径。

#### 4.9.5 规模与步数换算

幻觉域 20,000 条在 `--old_domain_ratio 0.7` 下 → 阶段二 parquet `N = 20,000 / 0.3 = 66,667`：

| `total_epochs` | 步数 | 每条曝光 |
|---:|---:|---:|
| 1 | 260 | 1 |
| 3 | 781 | 3 |
| 10（现默认） | **2,604** | 10 |

**20,000 是数据量目标，步数由 `total_epochs` 决定**——池子够大之后不再需要靠重复入池来凑比例，
所以每条曝光 = `total_epochs`，与配比无关（v0.9 那张 d=3 / 30× 曝光的表随之作废）。
建议阶段二用 `TOTAL_EPOCHS=3`（781 步）。

---

## 5. 统一答案契约与 Prompt 模板

### 5.1 契约（D3）

| 情形 | 模型应输出 |
|---|---|
| 可解（数值/表达式：MiP 可解侧、GSM-IC） | 最后一行以 `\boxed{<数值/表达式>}` 结尾 |
| 可解但**只判可答性**（D14；FalseQA-real，prompt 显式要求判断） | `\boxed{SOLVABLE}` |
| 可解（角色序列：K&K） | `\boxed{<角色词> <角色词> …}`，按题面 `names` 顺序，**用题面自己的角色词** |
| 不可解且有诊断标签（**v0.6 后仅 FalseQA-fake**） | `\boxed{UNSOLVABLE: <选项ID>}`（如 `\boxed{UNSOLVABLE: B}`）；选项ID 指向**题面里那一处假前提片段**（§4.4/D13） |
| 不可解且无诊断标签（**MiP 不可解侧**、将来的三档源） | `\boxed{UNSOLVABLE}` |

- **只约束最终落点，不约束推理过程/位置**（spec 原则 2）：reward 只在"最后一个 `\boxed{}`"上判，
  推理中途出现"矛盾/缺少"等字样不扣分、也不加分（D4 后无一致性项）。
- **四源里只有 FalseQA 走四档诊断**（K&K / GSM-IC 全可解；MiP 按 D12 放弃诊断）→
  **MiP 不可解侧走三档 bare**，该分支已有真实数据（276 条，§4.8 结论 1）；
  四档占比与单源依赖风险见 §4.8 结论 2 / 洞 1，**Q12 已在 v0.6 关闭（选项改同题跨度后捷径消失）**。

**⚠️ K&K 角色词归一化（强制；漏掉则 reward 反向）**

`solution` 是**规范语义**（`True` = 真话者）；题面用的表面词由该行的 `knight_knave` 映射给出。
`flip_role`（映射互换）与 `random_pair`（angel/devil、sage/fool…）真的换词，两族合计
**13,800 行 = 全语料 28.7%**（§4.1 核查 6）。因此：

```
role_words = (knight_knave['knight'], knight_knave['knave'])   # 该行自己的 [真话者词, 说谎者词]
verifier   : 先把模型答案按 role_words 映射回规范 bool, 再与 solution 比对
```

- 例（`flip_role`）：题面写作 "Knaves always tell the truth, and knights always lie"，
  正确回答 `knaves knaves` → 映射回规范 `[True, True]`。若硬编码 `K = knight`，
  这条会被判**错**、而错误答案会被判**对**。
- `role_words` 必须由 adapter 写进 `reward_model.ground_truth`（§3），
  reward 侧不得从题面文本里猜。
- §9 必须为 `flip_role` / `random_pair` 各写一条单测。

**⚠️ MiP 不附选项块（D12；否则 L3 泄题复现）**

- MiP 的不可解行 `has_diagnosis_label=false` → **prompt 里不得出现选项块**（用 §5.2 模板 B）。
  一旦附上选项，§4.3 L3 的"选"题面里唯一找不到的那一句""就在 95.6% 的样本上直接命中。
- 若仍然收到 `\boxed{UNSOLVABLE: A}` 这种带选项 ID 的输出，按 spec §3.1"只要正确识别不可解
  就给满分"处理 → **同样 +1**（选项 ID 对它无意义，见 §6）。这样不会因格式抖动把正确拒答判成 0。
- 反过来说：**MiP 侧的正确拒答不依赖任何选项**，是"模型自己判断信息不足"，这才是有价值的信号。

**⚠️ FalseQA 的选项是"题面片段"，不是"解释句"（D13）**

- 选项块里放的是**从该题面里切出来的词/短语**（如 `men` / `in space` / `Confucius`），
  其中一个正是让前提为假的那一处；**不要**放 rebuttal 解释句或来自别的题的文本。
- 理由：跨题文本作干扰项时，模型可靠"哪一项和题干同主题"蒙对（实测 99.1%，§4.4 L3）；
  换成同题片段后这条捷径**结构性消失**（"选题面里找不到的"命中 0.4% ≈ 0）。
- 选项**必须与 gold 等长**、位置打乱、**k=3（D15）**；不足 3 个则该条不入池（§4.6）。
- 若模型输出 `\boxed{UNSOLVABLE}`（bare，不带选项 ID）：按 §6 的判定链，FalseQA-fake 属
  `has_diagnosis_label=true`，**bare 落 `else` 分支只得 0**（识别对了但没给诊断）。
  这与 MiP 侧（`has_diagnosis_label=false` → bare 拿 +1）**恰好相反**，是 §6 最容易写错的一处。

### 5.2 Prompt 模板（追加在题面之后；按源二选一）

**模板 A（附选项块 + 显式判断）—— FalseQA 的**两个 label 共用同一套**（D13/D14）：**

```
<题面…>

请先判断该题的前提是否成立、是否可以作答，并只在最后一行给出结论：

- 若前提成立、题目可以作答，最后一行输出：\boxed{SOLVABLE}
- 若前提不成立 / 条件相互矛盾，最后一行输出：\boxed{UNSOLVABLE: <选项ID>}
  其中 <选项ID> 是下面选项中"让题目前提为假"的那一项。

选项：
A. …
B. …
C. …
```

- 模板 A 里的 A./B./C. 是**从本题面切出的词或短语**（D13，§5.1 告示），三个选项**等长**、
  位置已打乱；reward 只比选项 ID，不比文本。
- **`label=0` 与 `label=1` 必须共用这一套模板**（D14）：若只有假前提侧附选项块，
  "看到选项块 ⇒ 输出 UNSOLVABLE"就是一条 100% 捷径，D14 想补的判断那一半会被白送掉。
- 选项数固定 **k=3（D15）**；凑不满 3 个的样本在 adapter 阶段就丢弃（§4.6）。
  两个 label **同 k、同模板外观**，只有选项文本与题面不同（D14）。
- 本阶段**不给 bare 兜底**（"选项都不合适 → `\boxed{UNSOLVABLE}`"）：那要求给这 530 条
  不可解样本补一套 bare gold，属 Q25 的扩展，现在不开口子。

**模板 B（不附选项块）—— MiP 不可解侧 + 所有可解侧（MiP 可解 / GSM-IC / K&K）：**

```
<题面…>

若题目给出的信息不足以确定唯一答案，或条件相互矛盾，请在最后一行输出：
\boxed{UNSOLVABLE}

否则，请 step by step 推理，并在最后一行输出 \boxed{你的最终答案}。
```

- **MiP 走模板 B**（D12）：没有选项块，所以不存在"选哪个"的捷径（§5.1 的 MiP 告示）。
- `solvable=true` 的数据也可带干扰条件（GSM-IC），同样不附选项块。
- K&K 属可解但**不附选项块**，答案是**题面里的角色词序列**（如 `\boxed{angel devil devil}`）；
  模板 B 需加一句："用题目中出现的角色名称作答（若题目称其为 saint/sinner，就用这两个词），
  顺序与题面列出的居民一致。"——不能写死 knight/knave（§4.1 核查 6）。
- 数学类沿用现有 prompt 收尾约定（`\boxed{}`），与 `DESIGN.md` §1 一致，避免额外解析路径。
- 模板归属由 `ground_truth.has_diagnosis_label` 决定（而非 `data_source`），adapter 落盘时就固定进
  `prompt`，reward 侧不再判模板。

---

## 6. Reward 设计（只保留 spec §3.1）

新文件 `reward/hallucination_compute_score.py`，入口函数名仍为 `compute_score`
（verl 的 `reward.custom_reward_function.name=compute_score`）。

```
compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if not data_source.startswith("halluc_"):
        return base_compute_score(...)            # 委托现有文件，逐字节兼容老四域/if
    if not format_ok(solution_str):
        return {"score": 0.0}
    if data_source.startswith("halluc_math_gsmic"):
        return {"score": _math_score(solution_str, gt.answer)}     # solvable + 干扰项
    gt = json.loads(ground_truth)                # 解析失败 → log + 0（fail closed）
    status, answer, option = extract_final(solution_str)
        # status ∈ {"answer", "solvable_marker", "unsolvable_option", "unsolvable_bare", "none"}
        # 复用现有 _extract_boxed() 取最后一个 \boxed{}，再匹配 UNSOLVABLE / SOLVABLE 形态

    if gt.solvable:
        if gt.get("judgment_only"):       # D14: FalseQA-real，只判"是否可答"
            if status == "solvable_marker":                      return 1.0
            if status in {"unsolvable_bare", "unsolvable_option"}: return -1.0   # 误拒
            return 0.0                    # 没判断（给了普通答案/无 \boxed{}）
        if gt.get("role_words"):          # K&K: 角色词序列, 必须先做词表归一化(§5.1)
            return 1.0 if kk_match(answer, gt["role_words"], gt["answer"]) else 0.0
        return 1.0 if (status == "answer" and math_match(answer, gt.answer)) else 0.0
        # 答错数值 / 误判为不可解 都落这里
    else:
        if status == "answer":            return -1.0   # 应该拒答却编造
        if not gt.has_diagnosis_label:    return 1.0 if status in {"unsolvable_bare","unsolvable_option"} else 0.0
        return 1.0 if (status == "unsolvable_option" and option == gt.correct_option_id) else 0.0
        # 识别出不可解但诊断错误/未给诊断 → 0
```

要点：

- **主 reward 取值 `{+1, 0, -1}`**，与 spec §3.1 一致。
  ⚠️ 已知量纲风险：现有其它域是 `{0,1}`，`-1` 在组内 std 归一化下可能被放大；退化组由现有
  DAPO `filter_groups` 屏蔽（D5）。上线时按 §10 重点观察负 advantage 的分布。
- `math_match` 复用现有数学匹配路径（`_math_score` / math_verify + math_dapo fallback），
  不新写比较逻辑。
- `kk_match`（K&K 专用）：把模型答案里的角色词按该行 `role_words` 映射回规范 bool，再与
  `answer` 逐位比对；词表外的 token、位数不符、空答案 → 0。**不要**在 K&K 这条路上复用
  `math_match`。判定依据是 `ground_truth` 里是否存在 `role_words` 键，而不是 `data_source`
  （便于单测直接构造）。
- 抽取全部集中在 `extract_final`，只取最后一个 `\boxed{}`；无 `\boxed{}` → `status="none"` → 0。
- **MiP 走 bare 分支（D12）**：MiP 不可解行 `has_diagnosis_label=false` → 命中
  `status ∈ {"unsolvable_bare","unsolvable_option"}` 即 **+1**；`\boxed{UNSOLVABLE: A}` 同样 +1
  （其 prompt 里没有选项，选项 ID 无意义，spec §3.1「识别出不可解即满分」）。
  → **不需要为 D12 改这段逻辑**，它本来就是 `has_diagnosis_label=false` 的既有语义。
- **FalseQA 走 `else` 分支（D13）**：`has_diagnosis_label=true` → 只有
  `status == "unsolvable_option"` **且** `option == gt.correct_option_id` 才 +1；
  bare `\boxed{UNSOLVABLE}` → **0**（识别对了但没给诊断，符合 spec §2.2 的细粒度诊断要求）。
  `gt.correct_option_id` 指的是**题面片段选项**的 ID（§4.4/D13），reward 侧不比文本。
- **FalseQA-real 走 `judgment_only` 分支（D14）**：`solvable=true` 但 `judgment_only=true` →
  只认 `\boxed{SOLVABLE}`。**误拒给 −1**，于是 `-1` 在本方案里**双向对称**：
  不可解侧"编造"→ −1、可解侧"误拒"→ −1（回应 §11 风险 3 的拒答塌缩）。
  走这条分支的条目**不参与 `math_match`**，其 `answer` 字段只留作审计。
- 任何异常（JSON 坏、抽取越界等）→ log + `{"score": 0.0}`，不抛出（与现有 `compute_score`
  在同一线程池里 fail-closed 的约定一致）。

---

## 7. 混合与阶段二接入

### 7.1 `mix_halluc.py`

输入：阶段一最终 mix 的 `train.parquet` / `val.parquet`（原四域/if 行原样保留）+ 四个 adapter 的 built parquet。
输出：阶段二 `final_halluc/train.parquet` + `val.parquet` + `mix_stats.json`。

暴露四个独立旋钮（都手动配，不退火，D7）：

- `--old_domain_ratio`：原四域（+ 可选 if）在总量中的占比；幻觉域拿 `1 - old_domain_ratio`。
- `--halluc_solvable_ratio`：幻觉域内 `solvable : unsolvable` 的占比。
- `--kk_share`：K&K 在**幻觉域内**的占比（K&K 只落在可解侧，故从可解预算里扣，其余给
  GSM-IC；FalseQA-real 按 D13 不入池）；默认 **0.15** → 批内 K&K ≈ 4.5%，D11 建议区间 0.15–0.20。
  K&K 的组内去重（每组 ≤1 条变体、clean:perturbed 1:1、N≥4）在 adapter 里完成（§4.2），
  mix 只按比例抽。
- `--mip_share`（v0.5 新增，v0.6 改默认值）：**不可解侧** MiP 的占比，其余给 FalseQA-fake；
  默认 **0.30**（= 276/(276+657)，即按**审计后**的池内比例；v0.5 写的 0.19 用的是 FalseQA
  未经审计的 1,187）。因为 MiP 已无诊断（D12），这个旋钮同时就是
  **"三档 bare : 四档诊断"的比例旋钮** —— 调低它 → 四档占比上升、三档占比下降。
- `--falseqa_judge_share`（v0.7 新增，D14）：**幻觉域内** FalseQA-real 判断型样本的占比；
  默认 **0.15**（与 `--kk_share` 同量级）→ 批内 ≈4.5%。它给模型"别对真前提乱拒答"的对照信号。
  调高它会挤掉 GSM-IC 的预算（GSM-IC 是可解数值侧的余量项）。

默认建议 `--old_domain_ratio 0.7`、`--halluc_solvable_ratio 0.7`（两个 7:3）、`--kk_share 0.15`、
`--falseqa_judge_share 0.15`、`--mip_share 0.30`。⚠️ 用户说的"7:3"确切指哪个，见 §12 Q1。
该默认下批内各分支占比见 §4.8 结论 2（判断型 4.5% / 三档 bare 2.7% / 四档诊断 6.3%）。

> **v0.10 覆盖（D16/D18）**：上面的默认值已被 **§4.9.3 表 B** 的 20,000 条配额取代。
> 旋钮语义随之简化为**三个纯配额旋钮**（不再需要从池子反推）：
> `--halluc_total 20000`、`--halluc_unsolvable_ratio 0.6`、`--halluc_four_tier_ratio 0.6`
> （不可解内部四档:三档）。K&K / 干扰项 / 判断型的配额写成 adapter 常量（§12 Q28）而不是旋钮，
> 因为它们的量已经被 D18 固定。`--mip_share` 退役为"三档内部来源配比"的调试开关。
**不可解侧供给上限只有 ≈0.93k**（MiP 276 + FalseQA-fake 657，§4.8 结论 3），
是实际可配比例的主要约束。

实现细节沿用 `mix.py` 的做法：**先切 val 再采 train**、按比例缩放、`total_size` 不足则
无放回/告警、`rng.shuffle`、重排 `extra_info.index`、写 `mix_stats.json`（按 ability/source 分桶）。
`extra_info.split` 在 val 行写 `"val"`。

### 7.2 去重与去污染

- 新数据与阶段一训练 mix 做**池内近重去重**（复用 `dedup.py` 的 MinHash 思路），
  因为 MiP 源自 GSM8K/SVAMP/MATH，与 Big-Math 主池高度同源
  （v0.5 后 MiP 可解侧不进池，**需要过这一步的只剩它的 276 条不可解行**）。
- 对评估集（MATH-500 / AIME / GPQA…）的 n-gram + embedding 去污染沿用
  `decontaminate.py`；若验证集 parquet 尚未就绪，至少先跑 n-gram 一级并在报告里标注。

### 7.3 阶段二启动方式

- 数据：`TRAIN_FILES="['…/final_halluc/train.parquet']" VAL_FILES="['…/final_halluc/val.parquet']"`。
- 续训：`RESUME_MODE=resume_path RESUME_PATH=<阶段一 ckpt 目录>`。
- reward：现有 run 脚本把 `reward.custom_reward_function.path` 硬编码为 `compute_score.py`，
  **切换需要额外一小步**。候选方案见 §12 Q8：
  (a) 新增一份薄 wrapper run 脚本；(b) 给现有脚本加一个默认值不变的 `REWARD_PATH` env 开关；
  (c) 只写进 README 让你手动改配置。D8 只约束"不改现有数据/reward 语义"，(b) 属默认保持的加法。

---

## 8. 文件清单（全部新增，零覆盖）

```
examples/reasoning_rl/
├── HALLUCINATION_RL_DESIGN.md                     # 本文档
├── reward/
│   ├── hallucination_compute_score.py             # 新 dispatcher（老前缀委托）
│   └── test_hallucination_compute_score.py
└── scripts/hallucination/
    ├── schema.py                                  # pydantic schema + parquet 序列化/校验
    ├── verify_kk.py                               # K&K gold 自检/审计（Spike 1 已完成，见 §4.1；保留为审计工具）
    ├── verify_mip.py                              # MiP L1/L2/L3 审计断言（§4.3）：证书、必要性、唯一性、反作弊
    ├── verify_falseqa.py                          # FalseQA L1/L2/L3 审计断言（§4.4）：配对 diff 证书、等长选项、反作弊
    ├── kk_adapter.py
    ├── mip_adapter.py                             # 三档源（D12）：svamp 配对 + §4.3 漏斗筛选
    ├── falseqa_adapter.py                         # 四档源（D13）：索引对齐配对 → diff 片段 gold + 同题等长选项
    ├── gsm_ic_adapter.py
    ├── distractor_mining.py                       # v0.6 后只服务 FalseQA，且只做「同题等长跨度」（§4.6）
    ├── mix_halluc.py
    └── test_*.py                                  # 每个 adapter / mix / schema 的单元测试
```

`scripts/hallucination/__init__.py` 视导入方式决定是否添加（倾向显式模块路径导入，避免与
`examples.reasoning_rl.scripts` 现有包结构冲突）。

---

## 9. 测试策略

- **reward 单测**（`test_hallucination_compute_score.py`）：
  - 四档 / 三档全分支（含 `-1` 编造、选错选项、bare UNSOLVABLE、无 `\boxed{}`、坏 JSON、格式门不过）；
  - 老前缀委托：`math_*` / `logic_*` / `if_*` 抽样断言与直接调用 `compute_score` 结果一致；
  - `\boxed{}` 抽取：多个 boxed 取最后、嵌套花括号、`UNSOLVABLE: B` 与普通答案互不误判。
  - **K&K 角色词（防 reward 反向）**：`flip_role`（`role_words=['knave','knight']`）与
    `random_pair`（`role_words=['angel','devil']`）各一条，断言"按题面词作答 → +1"
    且"按 canonical knight/knave 作答 → 0"，见 §5.1。
  - **MiP 三档（D12）**：`has_diagnosis_label=false` 的不可解条 —— bare `\boxed{UNSOLVABLE}` → +1、
    `\boxed{UNSOLVABLE: A}` → +1（选项 ID 无意义）、编造数值得 −1、无 `\boxed{}` → 0；
    以及其 prompt 落盘时**不含选项块**（模板 B）。
  - **FalseQA 四档（D13）**：`has_diagnosis_label=true` 的不可解条 —— 选中 gold 选项 → +1、
    选中干扰项 → 0、**bare `\boxed{UNSOLVABLE}` → 0（不是 +1，与 MiP 恰好相反）**、
    编造答案 → −1、越界选项 ID（如 `\boxed{UNSOLVABLE: Z}`）→ 0；
    以及其 prompt 落盘时**含选项块且选项文本都取自本题面**（模板 A）。
  - **FalseQA 判断型（D14）**：`solvable=true, judgment_only=true` 的条目 ——
    `\boxed{SOLVABLE}` → +1、`\boxed{UNSOLVABLE}` / `\boxed{UNSOLVABLE: B}` → **−1（误拒）**、
    只给普通答案或没有 `\boxed{}` → 0；且**不允许**走 `math_match`。
  - **选项结构单测（D15）**：断言每条 FalseQA 样本的 `extra_info.options` **恰好 3 项**、
    三项与 gold **等长**、三项**都出现在该题题面里**、`correct_option_id` 落在 1..3；
    并断言 `label=0` 与 `label=1` 两个子集的**选项个数完全一致**（防止"选项个数"变成捷径）。
  - **模板同构单测（D14 的防线）**：`label=0` 与 `label=1` 的落盘 prompt 在
    "是否附选项块""是否有判断指令""选项个数"三件事上必须**完全一致**（只允许选项文本/题面不同）——
    否则"看到选项块 ⇒ 输出 UNSOLVABLE"是一条 100% 捷径（§5.2）。
  - **选项乱序不变性**：同一条数据、同一次输出，只打乱选项块顺序 + 同步改 `correct_option_id`，
    reward 结果必须不变（防止实现里偷偷依赖位置）。
- **FalseQA 审计**（`scripts/hallucination/verify_falseqa.py`，§4.4）：
  - L1/L2：断言"第 k 条 label=1 与第 k 条 label=0 是索引对齐配对"、"fake 侧差异区域恰好 1 个连续区间"、
    "gold 含内容词"、"gold 首词在题面唯一出现"；不满足 → 该条不入池（fail-closed）。
  - L1 旁证旗标：统计"rebuttal 指名 gold 片段"的占比（实测 train 仅 56.8%），
    **按此字段抽样 N=50 做人工复核**，复核不通过率 >10% 则整源暂停（§12 Q24）。
  - L3 反作弊：对**被启用的选项集**跑「选题面里唯一找不到的那项」/「选最长」/「选大写/数字」/
    「跨题干扰项 + 主题重叠最大化」，断言命中率 ≤ 随机基线 + 10pt（§12 Q21）。
    实测：跨题 99.1%；**同题 + 等长下 k=3 与 k=4 的全部表层启发式都 ≈ 随机**
    （k=3: 30.9%/30.9%/34.1% vs 随机 33.3%；k=4: 24.1%/24.9%/27.9% vs 随机 25.0%）。
    断言还必须包含两条结构性检查：**"选项全部出现在题面里"**、**"所有选项与 gold 等长"**——
    v0.7 的归因更正说明真正起作用的约束是**等长**（§4.4）。
  - **H6 判断任务审计（v0.7 新增，D14 专用）**：对 `label` 作 gold 的"是否可答"任务，
    报告题面词袋 NB 的 5-fold balanced-acc，并**拆虚实词**：全词表 / 只虚词 / 只实词。
    实测 train 0.765 / 0.569 / 0.757 → 结论是"主题记忆而非文风"，此数必须写进报告，
    作为解读 D14 指标时的上限参照（不是 fail 条件，是**解释基准**）。
- **MiP 审计**（`scripts/hallucination/verify_mip.py`，§4.3）：
  - L1/L2：对每条断言"重插回原题 == 原题"、"被删数值在原答案链中被用到（折算百分数）"、
    "删除段只含 1 个数值"、"该数值在残缺题面中不可见"；不满足 → 该条不入池（fail-closed）。
  - L3 反作弊：对**任何被启用的选项集**跑 H1「题面里唯一找不到的那一项」/ H2「抹掉数字后再选」/
    H4「跨题干扰项 + 主题词重叠」，断言命中率接近随机（默认阈值 ≤ 随机基线 + 10pt，见 Q21）。
- **新源审计（v0.10 新增，D16/D17/D18）**——每个 adapter 交付时同带一个 `verify_*.py`：
  - **SUM**（`verify_sum.py`）：① 配对非空断言（`answerable_question` / `unanswerable_question` 都非空）；
    ② 用 §4.4 同一套 `difflib` opcode 投影复算"四档 9,832 / 三档 6,050"，与实测数偏差 >2% 则报警；
    ③ **L3 硬门槛**：题面词袋 NB 5-fold balanced-acc **≤ 0.55**（实测 0.502/0.503）；
    ④ 规则类型分类器抽检 N=100（§12 Q26），不通过率 >10% 则整源降级。
  - **UMWP**（`verify_umwp.py`）：① `relevant_ids` 配对完整性必须 **= 100%**（实测 100%，2600/2600）；
    ② 按 `category` 分别跑 L3（cat1 只准进三档，断言其"可见区域 = 0"）；
    ③ 断言 cat1 的行**不生成选项块**（`has_diagnosis_label=false`）。
  - **TreeCut**（`verify_treecut.py`）：① 断言 `proof` 含"N variables but M formulas"且 `M < N`（证书非空）；
    ② 与 `answerable` 对照跑**长度启发式与 BoW NB**，**硬门槛 ≤ 随机 + 5pt**——修正前的 0.726 / 0.755
    **不通过**，必须先改生成器（§4.9.2 坑 3），改完重测达标才允许入池；
    ③ 断言生成的可解/不可解对照**共用同一 `numVars`/`ansDepth`/`theme`/`order`**。
  - **CREPE**（`verify_crepe.py`）：① 断言标签按实测串 `'false presupposition'`（**空格**）匹配，
    且命中 927 条（train）；若命中 0 说明用了 README 的下划线串，直接 fail；
    ② 断言任何 `presuppositions` 片段**逐字**出现在 `question` 里的比例 < 10%（实测 1.1%）→
    **四档指针构造必须被拒绝**（断言 `has_diagnosis_label=false`，只走判断型）。
  - **KUQ**（`verify_kuq.py`）：断言 unknown/known 各 3,437/3,447 且六类 `category` 全部非空；
    断言两个 label 共用同一模板（D14 同构约束）。
  - **干扰项合成**（`verify_distractor.py`，D17）：① **答案不变断言**——合成后的行的
    `ground_truth` 必须与底题逐字节相同（这是本项唯一的功能正确性判据，必须硬断言）；
    ② 三类标注（`sentence_label` / `role_label` / `number_label`）在成品里各自配平到 ±5pt；
    ③ 断言干扰句**不出现在底题的解题链里**（复算：干扰句的 `{number}` 不参与 `answer` 的推导）；
    ④ 底题同源断言——干扰项与不可解项必须来自同一 source split（§4.9.4 硬约束）。
- **reward 验证（v0.10 新增；对应"确保 reward 校验正确"）**：
  - **六分支 × 三类用例的矩阵单测**（`test_hallucination_compute_score.py`）。分支取自 §4.9.3 表 B
    第 1–6 行；每格三个用例：**正确输出**（应 +1）/ **错向输出**（应 −1）/ **不可解析**（应 0）。
    其中必须显式覆盖的错向对：
    | 分支 | 正确 (+1) | 错向 (−1) | 不可解析 (0) |
    |---|---|---|---|
    | 可解·数值+干扰 | `\boxed{42}`（= gold） | `\boxed{UNSOLVABLE}`（**误拒**） | 无 `\boxed{}` |
    | 可解·角色词 | `\boxed{angel devil}`（归一化后） | `\boxed{UNSOLVABLE}` | `\boxed{K N}` 字面量 |
    | 可解·判断（A/B 两侧） | `\boxed{SOLVABLE}` | `\boxed{UNSOLVABLE}` | `\boxed{UNSOLVABLE: B}` |
    | 不可解·四档 | `\boxed{UNSOLVABLE: B}`（= `correct_option_id`） | `\boxed{答案}`（**编造**） | `\boxed{UNSOLVABLE}`（**裸的不给分**） |
    | 不可解·三档 | `\boxed{UNSOLVABLE}` | `\boxed{答案}`（**编造**） | `\boxed{UNSOLVABLE: B}`（误附选项块） |
  - **模板同构断言（D14/D18 的硬约束，做成单测而不是靠人眼）**：
    ① 分别统计模板 A 子集与模板 B 子集里 `solvable` 的真值分布，
    **两侧的 `solvable=true` 占比都必须 > 0**（否则"有没有选项块 ⟺ 不可解"）；
    ② 断言 `has_option_block` 与 `solvable` 的**互信息 ≈ 0**（阈值：|corr| < 0.1）；
    ③ 断言模板 A 的两个 `label` 子集**选项个数相同**（=3，D15）。
  - **裸 `\boxed{UNSOLVABLE}` 的分支相关性**（v0.6 起就有的隐性坑，这里显式化）：
    同一个字符串在 **MiP/SUM-del/TreeCut/UMWP-cat1/CREPE/KUQ** 上是 **+1**，
    在 **FalseQA-fake / SUM-visible / UMWP-visible** 上必须是 **0**（四档源要求给 ID）。
    → 单测必须**两个方向都断言**，且断言错误消息里打印 `data_source`，避免将来加源时静默错配。
  - **只评最后一个平衡 `\boxed{}`**：断言 `\boxed{1} \boxed{2}` → 取 `2`；
    `\boxed{` 未闭合 → 0；`\boxed{ }` 空 → 0（fail-closed）。
  - **reward 与 adapter 的字段契约单测**：`reward_model.ground_truth` 必须是 JSON 字符串，
    且四档源的 `correct_option_id` 必须能在 `options` 里命中（否则该条 fail-closed 且报警——
    这条能同时抓出 adapter 的选项乱序 bug 与 reward 的索引 off-by-one）。
- **adapter 单测**：用小规模固定 fixture（几条真实样本的冻结文本）验证 schema 映射与选项生成，
  不依赖联网；真实下载走一次性脚本，不入 CI。
- **spike 脚本**：产出 JSON/MD 报告 + 断言式自检（例如枚举解数必须与求解器一致）。
- 运行方式沿用仓库现状（`pytest examples/reasoning_rl/...`）。

---

## 10. 监控指标（阶段二分域绘图，不合并）

- 可解题准确率（按 source 分开）。
- 不可解侧**按契约分支分开看**（v0.5 改 / v0.7 扩）：
  - MiP（三档 bare）：**编造率 / 正确拒答率** 两档；
  - FalseQA-fake（四档诊断）：**编造率 / 拒答但诊断错 / 诊断正确** 三档；
  - FalseQA-real（判断型，D14）：**`\boxed{SOLVABLE}` 正确率 / 误拒率（输出 UNSOLVABLE）** 两档。
- 选项分布（是否出现位置偏置）、`UNSOLVABLE` 触发率随 step 变化。
- **MiP 侧"选项式输出"比例**（`\boxed{UNSOLVABLE: X}` 的占比）：正常应≈0，异常升高说明 prompt
  误附了选项块（§5.2 模板 A/B 混用），同时也是 §5.1 那条告示的自检信号。
- **FalseQA 侧选项位置分布**：gold 落在 A/B/C 的比率应≈各 1/3；明显偏斜说明 adapter 没打乱。
- **FalseQA 真前提误拒率（v0.6 新增 / v0.7 改造；风险 12 的直接指标）**：
  D14 之后它**同时是训练指标**——FalseQA-real 的 reward 就是 1 − 误拒率。
  另在 `label=0` 的 valid 491 条上单独评一次"会不会输出 UNSOLVABLE"作为外推检查。
- **把 D14 的"判断"与"诊断"分开看（v0.7 新增）**：`label=0` 的 `\boxed{SOLVABLE}` 准确率
  与 `label=1` 的四档准确率**必须分开画**。已知判断这一半有 76–83% 的主题记忆成分（§4.4 H6），
  若它先冲到 90%+ 而四档还趴着，说明学到的只是"哪些实体在这份数据里偏假"，不是能力。
- **`\boxed{SOLVABLE}` 的跨源泄漏检查（v0.7 新增）**：统计它在**非 FalseQA 源**
  （GSM-IC / K&K / 老四域）上的出现率，正常应≈0；升高说明判断指令污染了其它模板。
- `-1` reward 的比例与对应 advantage 分布（配合 §6 的量纲风险）。
- **退化组被 `filter_groups` 丢掉的量**：不可解题每组 `rollout_n=16`，若模型一开始 16 条全部
  编造（全 `-1`）或全部拒答（全 `+1`/全 `0`），组内零方差会被屏蔽 → 冷启动阶段可能
  "该学的组恰好全被丢掉"。观察不可解组中有效组的占比是否随 step 上升。
- **K&K 按 `perturbation_family` 分桶**（clean vs 6 类扰动）看准确率：若 `flip_role` /
  `random_pair` 显著低于 clean，多半是 §5.1 的角色词归一化没生效（§11 风险 9）。
- MiP / FalseQA 各子类型也分开看，不同来源收敛速度不同。
- **FalseQA 的"诊断正确率"要对着"选项基线"读**（v0.6 新增；D15 锁定 k=3 后基线固定）：同题等长 k=3 的随机水平是 **33.3%**，
  所以我们期望的是**明显高于 33%** 但**不是一上来就 95%+**；后者先怀疑 §4.4 的 L3 有实现事故
  （选项里混进了跨题文本 / 等长约束没生效）。
- **MiP 可解侧不进池时的覆盖检查**：三档 bare 信号只来自 MiP 的 276 条不可解行，
  且它们与主池同源（GSM8K/MATH）→ 观察"不可解识别"是否只在同源数学题上生效
  （若只在 MiP 式缺条件题上有效率、在其它不可解分布上无效，说明学到的是题面特征，不是能力）。

---

## 11. 已知风险与限制

1. **不可解池只有 ≈0.93k，且四档诊断只覆盖批内 ≈6.3%、只来自 FalseQA 一家**（§4.8 实测）：
   K&K 核查后归零（§4.1）、MiP 因 L3 反作弊失败退三档（§4.3/D12）、FalseQA 经 D13 审计后从
   原始 1,187 降到 657（§4.4），池子只剩 MiP 276 + FalseQA 657。默认设置下每步约 7 个 prompt
   走三档、16 个走四档；0.93k 条在 10 epoch 内会被反复采样 → **本方案最紧的约束**。
   缓解路径：§4.7 的 UMWP 替补（触发条件见 §12 Q4）、§12 Q14（引入真三档源扩大不可解供给）、
   Q19（复核 23 条后把 MiP 严口径 276 补到 299）、Q23（k=4 只多要一个干扰项就把可构造量从 657 砍到 473——
   因为题面 median 只有 5 个内容词，池子 median 只有 3）。
2. **数据同源污染**：MiP(GSM8K/SVAMP/MATH) 与 Big-Math 主池重叠，必须去重，否则阶段二
   隐性重复放大该部分权重。（MiP 可解侧默认不进池后，这条只剩不可解侧 276 条需要过）
3. **`-1` 的量纲**：见 §6；D4 去掉了辅助项后，负奖励是唯一"比不给分更差"的信号，
   观察其是否导致拒答塌缩或过度保守。
4. **四档诊断只剩单一来源（v0.6 改写）**：`has_diagnosis_label=true` 全部来自 FalseQA。
   它自身的表层捷径已被 D13 + §4.4 L3 审计处理（见风险 5），但**单源**这件事本身没法解决：
   四档的正确率变化无法归因到"能力"还是"这一个源的题型"。**Q12 已在 v0.6 关闭**
   （选项改同题跨度后捷径消失，不再需要降级）。
5. **FalseQA 选项的表层捷径（v0.7 改写：主风险已解，归因已更正）**：
   - ~~跨题 rebuttal 干扰项可被主题匹配蒙对~~ → 已解：选项改成**同题等长跨度**后，
     "选题面里找不到的"命中 ≈0；所有表层启发式（选最长/大写/数字/首位）都回到随机水平。
   - ⚠️ **归因更正**：起作用的是**等长约束**，不是"把 k 从 4 降到 3"。同题 + 等长下 k=3 与 k=4 都≈随机；
     v0.6 那句"'选最长'从 38.8% 压到 30.3%"混淆了"近似等长 k=4"与"严格等长 k=3"（§4.4 L3）。
   - 残余：题面词袋 NB 判"真前提"仍有 AUC 0.828 的弱信号（判"假前提"只有 0.172）。
     这个信号**方向与任务一致**（"像正常问句 → 照常作答"），且**给不出 gold 片段**，
     但它降低了"识别假前提"这一半任务的难度 → §10 要盯四档正确率是否一上来就异常高。
6. **embedding 干扰项未启用**：同题候选不足时干扰项质量可能偏低，靠抽检发现。
7. **阶段二 reward 路径切换**是小改动但触及现有 run 脚本，见 §12 Q8。
8. **License**：**K&K（clean + perturbed）已核实为 cc-by-nc-sa-4.0，禁止商用**（§4.1）；
   FalseQA / GSM-IC / MiP 的 license 尚未逐个确认；UMWP（CC-BY-SA-4.0）与 AbstentionBench
   （cc-by-nc-4.0）均含非商用条款。若后续有商用诉求，K&K 必须整体替换或删除。
9. **K&K 奖励反向（静默失败）**：若 `kk_adapter` 漏写 `role_words`、或 reward 侧硬编码
   `K = knight`，则 `flip_role` + `random_pair` 的 13,800 行（28.7%）reward 会 **100% 反向**——
   表现为"loss 在降、指标在动，但学的是错的"。§5.1 的归一化 + §9 的单测是唯一防线。
10. **MiP 的取舍带选择偏差**（v0.5 新增）：634 条里 **57% 被丢弃**（209 多值 + 55 无数值 +
    37 多段 + 26 数字仍可见 + 23 未过必要性的 8.1%），保留下来的偏向"改动干净、单个数值、
    跨度短"的题型。→ 三档拒答能力可能只在"一个数被换成占位词/被删"这一种形态上生效，
    对"缺一条定性条件""多条前提同时缺"不成立。§10 的覆盖检查就是盯这个。
11. **偏离 spec §2.2 的记录**：spec 把 MiP 列为数学域细粒度诊断源，D12 后 MiP 只做三档。
    这是 spec 自己在 §1 允许的降级路径（`has_diagnosis_label=false` → 三档），
    但**需要用户明确接受"四档诊断只剩 FalseQA 一家"**（§12 Q12/Q18）。
    另外 spec §2.2 也没写"诊断对象是题面片段"——D13 把细粒度诊断实现为**pointer（指向题面中的假前提片段）**
    而不是"复述缺失条件"，这是对 spec 的实现层解释，记录在案。
12. **常识题型的可解侧信号是"主题记忆"型（v0.7 改写：洞已堵，但换成了另一种风险）**：
    v0.6 的洞——`label=0` 不入池导致"常识问句一律拒答"——已按 D14 堵上（851 条判断型样本 + 误拒 −1）。
    **新风险是它的质量**：对"是否可答"这个判断，题面词袋 NB 的 5-fold balanced-acc 就有
    **0.765 / 0.833 / 0.807**（train/valid/test），而**只用虚词只有 0.569 / 0.601**——
    也就是说信号几乎全在**实词**上，本质是"space / Confucius / Mars / abalone 这些实体在这份数据里偏假"
    的**闭集主题记忆**，不是能外推的前提检查。风险后果：
    (a) 判断指标好看 ≠ 能力，(b) 换一个假前提数据集可能立刻掉回去。
    缓解：§10 把"判断"与"四档诊断"分开画；报告里必须并列 H6 的记忆上限（§9）；
    四档诊断（要指向片段）才是本阶段真正难的那一半。
13. **FalseQA 的取舍带选择偏差（v0.6 新增 / v0.7 扩）**：假前提侧 1,187 条里 **44.7% 被丢弃**
    （21.8% 差异区域不唯一 + 等长选项凑不满），可解侧 1,187 条里 28.3% 因同样原因丢弃。
    保留下来的是"改动局部、片段短、题面里同型词多"的样本。
    → 四档诊断能力可能只在"替换一个名词/数词"这一种形态上生效，对"整句被改写""多处同时为假"不成立。
    与风险 10（MiP）是同一类问题，§10 的覆盖检查同样适用。

---

## 12. 待确认问题（Open Questions）

> 每项给出**当前默认**；用户可直接在本文件上改，或口头答复后由 agent 更新。

| # | 问题 | 当前默认 |
|---|---|---|
| Q1 | "7:3" 指的是**幻觉域内 solvable:unsolvable**，还是**原四域:幻觉域**？ | 两个都做成旋钮；默认 `old_domain_ratio=0.7` 且 `halluc_solvable_ratio=0.7` |
| Q2 | 每题选项数量？ | 3（A/B/C，1 正确 + 2 干扰）—— **已由 D15 锁定**，含"两个 label 必须同 k"的约束 |
| ~~Q3~~ | ~~K&K 扰动保留率低到多少就暂停该路径？~~ **已关闭** | 实测无解样本为 0，K&K 不再走"筛不可解"路线，改按 D11 作可解锚点（§4.1） |
| ~~Q4~~ | ~~UMWP 替补（§4.7）的触发条件？~~ **v0.10 已关闭**：触发条件（"≈0.93k 不可解池不够"）已满足，UMWP 按 D16 **正式进训练源**——四档 1,449 / 三档 440。① "其余类别选项语义没设计清楚"由 §4.6 的同题等长跨度解决并实测 ≈随机（§4.9 表 A）；② 不可解池已到 ≈19.4k。唯一保留的类别约束：**cat1（Key Information Missing，840 条）只走三档 bare**（题面不可见） | **已纳入（D16）**；cat1 走三档 |
| Q5 | 是否把原始 spec 也 vendor 进仓库（`HALLUCINATION_RL_SPEC.md`）？ | 是，便于离线查阅与版本对照（待你确认） |
| Q6 | 阶段二 val 切分规模？ | 沿用 `mix.py` 默认 `--val_size 256`（幻觉域内按比例分配） |
| Q7 | 三档 reward 分支本阶段就实现，还是等有真实三档数据再说？ | 实现 + 合成用例（契约完整）；**v0.5 后已有真实数据（MiP 不可解侧 276 条）**，不再只是合成覆盖 |
| Q8 | 阶段二怎么切 reward 路径？(a) 新薄 wrapper 脚本 / (b) 现有脚本加默认不变的 `REWARD_PATH` env / (c) 只写手动步骤 | (b)，最小且默认行为不变 |
| Q9 | 干扰项本阶段是否就上 embedding 近邻？ | 不上，只预留接口（§4.6） |
| Q10 | 数据源拉取/构建是现在就真跑（需要联网下 K&K/MiP/FalseQA/GSM-IC），还是先只写代码 + fixture？ | 先写代码 + 小规模真跑一遍验证（联网可用） |
| Q11 | 是否有商用诉求？（决定 license 约束是否需要提前介入） | 暂按研究用途处理，spike 时记录各源 license |
| ~~Q12~~ | ~~**FalseQA 是否也降为三档源**？~~ **v0.6 已关闭**：该问题存在的前提是"FalseQA 的选项语义会被表层启发式绕过"（v0.5 洞 1）。实测跨题 rebuttal 干扰项确实 99.1% 可蒙，但**换成同题等长跨度后捷径结构性消失**（§4.4 L3）→ 保留四档（D13），无需降级 | **不降级**；原"倾向降为三档"的默认作废 |
| Q13 | FalseQA 的干扰项是否投入成本做"同模板族挖掘"（跨题取文本）？ | **v0.6 关闭**：跨题方案实测 99–100% 可蒙（§4.4），一律不做；只做同题等长跨度（§4.6） |
| ~~Q14~~ | ~~是否引入一个真三档来源？~~ **v0.10 已关闭**：按 D16 纳入 **TreeCut**（生成器，三档 1,084）、**SUM 删除型**（三档 2,000）、**UMWP cat1**（三档 840）、**KUQ / CREPE**（不可指认的假前提，三档 600）。三档分支从"276 条真实数据"变成 **4,800 条、6 个来源** | **已纳入（D16）** |
| Q15 | **K&K 不提供任何弃答/诊断数据**（48,076/48,076 唯一可解，§4.1），接受它只作可解锚点、把不可解供给压力全部留给 MiP+FalseQA 吗？ | 已按 D11 处理；**v0.10 后供给压力已解除**（D16 把不可解池推到 ≈19.4k），K&K 保持纯可解锚点，且训练池按 §4.2 的 `N≥4` 筛除后是 **5,000 组**（§4.8 v0.10 注） |
| Q16 | K&K 的四类语言级扰动（数学结构逐字节同 clean）是否全保留？ | 默认**全保留**：它们正是"抗措辞扰动"信号，其中 `flip_role` / `random_pair` 还换角色词，最能打掉记忆式作答；但组内只取 1 条，clean : perturbed 目标 **1:1** |
| Q17 | K&K 答案写"表面角色词"还是"规范 K/N"？ | 默认**表面角色词**（照题面用词作答），verifier 按 `role_words` 归一化（§5.1）；不接受 `K`/`N` 字面量 |
| Q18 | **MiP 退三档**（不可解侧只判拒答、prompt 不附选项块、`has_diagnosis_label=false`），接受"四档诊断只剩 FalseQA 一家"吗？ | **已按 D12 决定**（§4.3），此处仅作记录；若不能接受，替代路径是把 276 条做成"缺哪个量"的改写选项（另一条工程线，本阶段不做） |
| Q19 | §4.3 L1 未通过的 **23 条**（被删数值折算百分数后仍在原答案推导链里找不到，8.1%）是否人工复核后补入？ | 默认**不补**（宁缺勿错，fail-closed）；若要补，复核通过后严口径 276 → 299（宽口径） |
| Q20 | MiP **可解侧**（原题 634 条，自带 `answer`）是否进池？ | 默认**不进**：与 Big-Math 主池同源（GSM8K/MATH）、仅 634 条、去重成本高于收益（§4.3）；若要用须先过 §7.2 池内近重去重 |
| Q21 | 是否把 §4.3/§4.4 的 L3 反作弊断言（H1/H4、选最长/大写/数字，命中率须接近随机）设为**硬门槛**——任何将来启用的选项集都要先过 `verify_mip.py` / `verify_falseqa.py`？ | 默认**是**，阈值"随机基线 + 10pt"；FalseQA 的现有选项集上线前也要跑一遍并留报告 |
| ~~Q22~~ | ~~**FalseQA 的 `label=0` 可解侧**如何处理？~~ **v0.7 已关闭（用户提议）**：不给模型"作答"的任务（gold 是自由文本，判不了分），而是**加一段 prompt 让它判断"该题是否可答"，用 `label` 当 gold**（D14）。两个 label 共用同一套模板，误拒给 −1 | **按 D14 实施**：`label=0` 入池 851 条，输出 `\boxed{SOLVABLE}` |
| ~~Q23~~ | ~~FalseQA 的**选项数 k**？~~ **v0.9 已关闭（用户确认）** → **D15：k=3**。依据：k **不是**抗蒙参数而是**覆盖率**参数——同题 + 等长下 k=3 与 k=4 的表层启发式都≈随机（§4.4 L3）；k=4 只会把四档从 657 砍到 473（−19.8pp，差额全来自"等长候选池恰好 =2"的 184 条）。另加约束：**两个 label 必须同 k**（D14 的模板同构） | **k=3 锁定**；若将来不可解池扩容（Q25 / UMWP），再回头评估 k=4 的更低下限（25% vs 33.3%） |
| Q24 | §4.4 的 **L1 旁证旗标**（rebuttal 未指名 gold 片段，train 占 43.2%）是否需要在入池前做人工抽检？ | 默认**要**：按该字段抽 N=50 复核，不通过率 >10% 则暂停该源并回到 §4.7 找替补 |
| ~~Q25~~ | ~~假前提侧被丢掉的 **530 条**要不要收回来当 bare 三档？~~ **v0.10 已关闭**：**不收回**。它原本的唯一理由是"不可解池不够"，而 D16 之后池子到 ≈19.4k、且三档分支已有 4,800 条配额（6 个来源），收回 530 条既无收益、又会削弱四档信号（同一 prompt 里多一条逃逸出口，理由同原判） | **不收回**；供给不足的前提已消失 |
| Q26 | **SUM 无机器证书**（只有最终 `ground_truth`，没有推导链 → 做不了 §4.3 的"必要性"检查）。它的不可解标签只经 o3-mini 生成 + 专家复核，怎么把关？ | 默认**人工/规则抽检 N=100**（分层覆盖五个退化类型），不通过率 **>10% 则该源降级为三档或弃用**；抽检清单与结论写进构建报告 |
| Q27 | **TreeCut 修 L3 之后的生成量与参数**？ | 默认按表 B 生成 **1,084 条三档 + 125 条"不现实"类**，约 1,200 条；参数在 `numVars∈{4,6,8}`、`ansDepth∈{2,4,6}`、`theme∈{food,outfit}` 上分层，保证每格 ≥100 条。若 L3 修完仍 >0.55，改用 SUM-del 补位 |
| Q28 | 干扰项合成（D17）的**施加对象配额**是否要独立旋钮？ | 默认**不加旋钮**，按 §4.9.4 固定（SUM-answerable 1,000 / UMWP-answerable 600 / K&K 400 / 主池 400）；若要调，改 adapter 常量并重跑 `mix_stats.json` |

---

## 13. 变更记录

| 版本 | 日期 | 变更 |
|---|---|---|
| v0.1 | — | 初稿：锁定 D1–D8，四源垂直切片，reward 只保留 §3.1，列出 open questions |
| v0.2 | — | 数据源核查后新增 D9/D10：AbstentionBench 移出训练源（降级为评估套件）、UMWP 登记为 MiP 替补；新增 §4.7；更新 §1/§11/§12 |
| v0.3 | — | 实测四源规模与诊断标签覆盖，新增 §4.8；修正三处事实错误（GSM-IC 全量 58,052 而非 HF 的 1,000、MiP 实际仅 984 条且 634 条文件内自带原题、K&K 自带机器可读 `statements`）；§5.1 指向 §4.8；§11 风险重编号并补 FalseQA 表层启发式风险；新增 Q12–Q14 |
| v0.4 | — | **K&K 全量核查完成（48,076 行枚举：0 无解、全部唯一可解），推翻 v0.3 的 K&K 设想**。新增 D11（K&K 降为可解锚点：组内去重、N≥4、批内 4–6%、不进四档）；重写 §4.1（六项核查：无解为 0 / 生成器强制有解 / 六种扰动性质 / 题型唯一且 `assume` 只在 `cot_repeat_steps` / 7× 重复度 / 角色词陷阱）与 §4.2；§3 增 `role_words` 字段与 K&K 的 `ground_truth` 示例；§5.1 增角色词归一化契约；§6 增 `kk_match`；§9 增防反向单测；§4.8/§11/§7.1 同步修正不可解池（≈1.8k）与 K&K license（cc-by-nc-sa-4.0）；§10 增按扰动族分桶；Q3 关闭，新增 Q15–Q17 |
| v0.5 | — | **MiP 核查完成：三层审计（标签证书 / gold 唯一性 / 反作弊）全部为否，选项集"选'题面里唯一找不到的那一句'"命中 95.6%** → 新增 **D12**：MiP 不做诊断、退三档（`has_diagnosis_label=false`、prompt 不附选项块、可解侧用原题 `answer` 校验），可入池 **严 276 / 宽 299**；重写 §4.3（审计表 + 漏斗 + 6 处实现要点）；§5.2 拆成**模板 A（附选项，仅 FalseQA）/ 模板 B（不附选项，MiP 及所有可解侧）**；§5.1 契约表与告示更新；§4.8 结论 1 改写（`has_diagnosis_label=true` 只剩 FalseQA 一家，三档分支不再是无数据的死代码）、结论 2 换成**分支占比表**（三档 1.7% / 四档 7.3%）、结论 3 修正（不可解池 ≈1.46k；可解池 72.3k 系笔误 → 66–73k）；§4.6 适用范围收窄到 FalseQA；§7.1 新增第四个旋钮 `--mip_share`；§6/§9/§10/§11 同步（新增风险 10 选择偏差、风险 11 偏离 spec 记录）；新增 Q18–Q21，Q7/Q12/Q15 更新 |
| v0.6 | — | **FalseQA 核查完成，结论与 v0.5 的直觉相反**：① 它**有**可答性标注（`label`，三份 split 严格 50:50），且是**索引对齐的对照对**（89–90% 词级 Jaccard ≥0.5）；② 但 `answer` 列不能当 gold（自由文本短答/rebuttal，test 侧每条 3 个并列参考答案）；③ 与 MiP 相反，假前提是**被替换**而非被删，**片段就在题面里** → pointer gold 可知。→ 新增 **D13**：FalseQA 保留四档，gold 改为**配对 diff 片段**、选项改为**同题等长跨度（k=3）**，可入池 **train 657 / valid 259 / test 388**。重写 §4.4（源事实 + 与 MiP 的对照表 + L1/L2/L3 审计表 + 反作弊实测 + 漏斗 + 6 条交付约束）、§4.6（跨题方案禁用，只做同题等长跨度）；§5.1/§5.2/§6 增 FalseQA 告示与 `else` 分支说明（**bare `\boxed{UNSOLVABLE}` 在 FalseQA 上是 0，在 MiP 上是 +1**）；§4.8 结论 1/3 与洞 1/洞 3 改写（不可解池 1.46k → **0.93k**，四档 1,187 → 657）、结论 2 分支占比重算（三档 2.7% / 四档 6.3%）；§7.1 `--mip_share` 默认 0.19 → **0.30**；§8/§9/§10 增 `verify_falseqa.py`、选项乱序不变性单测、真前提误拒率等监控；§11 风险 1/4/5 改写并新增**风险 12（可解侧缺位→过度拒答）**与风险 13（FalseQA 选择偏差）；Q12/Q13 关闭，新增 **Q22（label=0 是否入池）/ Q23（k）/ Q24（L1 旁证抽检）** |
| v0.7 | — | **可解侧按用户提议改造（D14）**：`label=0` 不再要求"作答"（gold 是自由文本、判不了分），而是**加 prompt 让模型判断"该题是否可答"，用 `label` 当 gold**，期望输出 `\boxed{SOLVABLE}`，**误拒给 −1**（使 `-1` 双向对称）。关键约束：**两个 label 必须共用同一套模板**（否则"有选项块 ⟺ 不可解"是 100% 捷径）；`label=0` 侧同样构造同题等长选项块，可入池 **851**。新增 **H6 判断任务审计**：题面词袋 NB 5-fold balanced-acc = 0.765/0.833/0.807，**只虚词仅 0.569/0.601** → 判断信号是"实词=主题记忆"而非文风破绽，写进 §4.4/§9/§10/§11 风险 12 作为解释上限。§4.8 结论 2 重算（新增判断型 4.5% 一行，GSM-IC 16.5% → 12.0%）、结论 3 加"判断型"供给行；§5.1 增第 5 种输出形态；§5.2 模板 A 改为**两 label 统一**；§6 增 `judgment_only` 分支与 `solvable_marker` 状态；§7.1 新增第五个旋钮 `--falseqa_judge_share`（默认 0.15）；§9 增判断分支单测、模板同构单测、H6 断言；§10 增"判断 vs 诊断分开画"与 `\boxed{SOLVABLE}` 跨源泄漏检查；Q22 关闭，新增 **Q25（被丢的 530 条要不要收回当 bare 三档）** |
| v0.8 | — | **k 的归因更正（用户追问触发）**：v0.6/v0.7 把"表层启发式回到随机"记成"等长 + 把 k 从 4 降到 3"的功劳，**是错的归因**——那张对比表把"**近似等长**·k=4"与"**严格等长**·k=3"并排，**同时改了两个变量**。固定等长后重测：k=3 的（选最长/大写/数字/首位）= 30.9/30.9/34.1/30.9%（随机 33.3%），k=4 = 24.1/24.9/27.9/24.1%（随机 25.0%）——**两者都≈随机，k 不是抗蒙参数**。真正的代价是覆盖率：657 → 473（−19.8pp）。并补上池子证据：题面 median 仅 **5** 个内容词，扣掉 gold 后等长候选池 median 仅 **3**（mean 2.66），P(池≥2)=70.8% / P(池≥3)=51.0%，**差额 19.8pp 全部来自"池恰好 =2"的 184 条**；按 gold 词数拆，gold=2 词（201 条）池 median 掉到 **1**、P(≥3) 仅 9.0%。同步更正 §4.4 L3 表与结论、§4.6、§9、§11 风险 1/5、Q23 的论述 |
| v0.9 | — | **k=3 锁定（用户确认）→ 新增 D15**：k 是**覆盖率参数**不是抗蒙参数；不可解池是最紧资源，k=4 砍掉 184 条而不换来抗蒙收益。D15 同时写入一条此前只是隐含的硬约束：**两个 label 必须同 k**（否则"选项个数"本身就是 D14 模板同构的一条捷径）。§4.4 L2 行/交付约束、§4.6、§5.2、§9 全部改引 D15；§9 新增**选项结构单测**（恰 3 项、与 gold 等长、都在题面里、`correct_option_id` ∈ 1..3、两子集个数一致），模板同构单测扩为"选项块/判断指令/**选项个数**"三项；Q2 标注已由 D15 锁定，Q23 关闭 |
| v0.10 | — | **不可解侧扩源 + 干扰项合成 + 20,000 目标（用户确认）→ 新增 D16/D17/D18，新增 §4.9**。逐源实测可用性（表 A）：**SUM**（MIT，train 36,480 → 四档 9,832 / 三档 6,050，L3 BoW NB **0.502/0.503** ✅）、**UMWP**（CC-BY-SA-4.0，5,200，配对 100%，四档 1,449 / 三档 440，**0.500/0.500** ✅）、**KUQ**（MIT，6,884，六类标注）、**TreeCut**（Apache-2.0，生成器可无限产，**L1 构造即证明**但 L3 **0.755** 需先修）、**CREPE**（BSD，HF 镜像 `tasksource/CREPE`——**官方 Google Drive 已 404**；`presuppositions` 仅 **1.1%** 是逐字 span → **只能进判断型**）；**排除** SQuAD 2.0（需 NLI）与 GSM-DC（仅 6,300 样例）。记录三个数据处理坑（CREPE 标签是**空格**不是下划线 / CREPE 无 span / TreeCut 长度泄漏 0.726）与修法。**D17**：不引新源，**规则复用 GSM-IC 的 242 条 `sentence_template`**（`{role}`/`{number}` 参数化）合成"有干扰项但答案不变"的 2,400 条，难度梯度用 GSM-IC 自带三组天然配平的标注。**D18**：幻觉域定 **20,000 条（不可解 12,000 = 60% / 可解 8,000）**，按契约分支给配额表、按缺陷类型给配平表（缺前提 35% / 歧义 25% / 不现实 11.7% / 假前提 10.5% / 问题缺失 9.5% / 无关实体 8.4%），并加硬约束"模板 A 与模板 B 各自都要含可解与不可解"。§4.5 增模板抽取说明、§4.7 把 UMWP 从替补改为正式纳入、§4.8 结论 3 全部作废并修正 K&K 训练池 6,200 → **5,000**（`N≥4` 筛除）与可解池 66–73k → **≈63.9k**；Q4/Q14/Q25 关闭，新增 Q26（SUM 抽检 N=100）/ Q27（TreeCut 生成量与参数）/ Q28（干扰项配额） |
