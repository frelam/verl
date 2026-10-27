# 幻觉抵制 Reasoning RL —— 实施设计

> 状态：**实施口径已定稿（D1–D34）**。本文档是这份方案的唯一事实来源与施工图。
> 与已有的 `DESIGN.md`（Qwen3-4B reasoning RL 主方案）**并行存在、互不覆盖**：本方案只在训练后程
> 以「换数据 mix + 换 reward 文件」的方式接入，不修改阶段一任何行为。

原始方案输入：`hallucination_resistant_reasoning_rl_spec.md`（sha256 `05fbab17…06a752`）。
本文档不复述该 spec 全文，只记录**落到本仓库后的具体实现决策、接口与开放问题**。

---

## 0. 已锁定决策

| # | 项 | 决定 |
|---|---|---|
| D1 | 接入方式 | 两段式：阶段 1 完全不动；阶段 2 从 ckpt `RESUME_MODE=resume_path` 续训，换新 mix parquet + 新 reward 文件 |
| D2 | 数据源 | 垂直切片四源：K&K（logic）+ MiP（math）+ FalseQA（commonsense）+ GSM-IC（可解-干扰对照）；后经 D16 扩充 SUM / UMWP / KUQ / TreeCut / CREPE（KUQ 后被 D25 移出） |
| D3 | 答案契约 | 可解 `\boxed{答案}`；不可解 `\boxed{UNSOLVABLE}`（无诊断标签）或 `\boxed{UNSOLVABLE: <选项ID>}`（有诊断标签，**stage 2**，D29）；**stage 3** 修复行 `\boxed{UNSOLVABLE: <选项ID>: <修复后的答案>}`（D28，经 D29 迁至独立阶段）；判断型可解 `\boxed{SOLVABLE}`；SUM 配对 `\boxed{<A或B>: <答案>}`（D23）；UMWP-answerable / FalseQA-answerable 两层 `\boxed{答案}`（D24/D27） |
| D4 | 一致性辅助项（spec §3.2） | **不做**。reward 只保留 spec §3.1 主 reward |
| D5 | 归一化（spec §3.3） | 随 D4 失效；退化组屏蔽沿用现有 DAPO `filter_groups` |
| D6 | 冷启动 SFT（spec §4） | **不在范围**。只交付数据 + reward |
| D7 | 配比 | 固定比例，不做 step 退火；脚本暴露参数供手动配置 |
| D8 | 覆盖约束 | 不改 `reward/compute_score.py`、不改 `scripts/*`、不改 `DESIGN.md`、不改现有 run 脚本的数据/reward 语义 |
| D9 | AbstentionBench | **移出训练源**，降级为评估套件：聚合体已含 FalseQA/UMWP（重复），且含 GPQA/GSM8K/MMLU 等评估集（污染），cc-by-nc-4.0 |
| D10 | UMWP | 初登记为替补；**已被 D16 取代**（正式入池） |
| D11 | K&K 定位 | 实测 48,076 行**全部唯一可解** → 纯**可解锚点**：不生成选项、不进四档；按 `(人数, index)` 组内去重、只用 N≥4；训练池 **≈5,000 组** |
| D12 | MiP 定位 | 不可解侧选项集**必然泄题**（"选题面里唯一找不到的那句"命中 95.6%）→ **不做诊断**，`has_diagnosis_label=false`，只判拒答（bare `\boxed{UNSOLVABLE}`，prompt 不附选项块）；可解侧不进池；入池 **276** 条不可解行 |
| D13 | FalseQA 定位 | **保留四档诊断**：gold 由索引对齐的配对 diff 提取（假前提片段就在题面里，pointer gold 可知）。选项语义后被 **D21** 改写为「替换对」 |
| D14 | FalseQA 可解侧判断入池 | **已被 D22 作废、D27 部分复活**（label=0 以两层 reward 入池，非纯判断型；原判断 +0.5 / 答案 +0.5 的部分分已改为 **AND 门控**：两层都对 +1，否则 0） |
| D15 | 选项数 k | **k = 3**（1 正确 + 2 干扰，A/B/C）。k 是覆盖率参数不是抗蒙参数；同一模板内各子集**必须同 k** |
| D16 | 不可解侧扩源 | 纳入 **SUM**（MIT，train 36,480）、**UMWP**（CC-BY-SA-4.0，5,200，配对 100%）、**KUQ**（MIT，6,884；后被 **D25** 移出）、**TreeCut**（Apache-2.0，生成器可无限产，L1 构造即证明但 L3 需先修；D26 后正样本也入池）、**CREPE**（BSD，HF 镜像；只能进判断型——`presuppositions` 仅 1.1% 是题面逐字 span）。**排除** SQuAD 2.0（判不可解需 NLI）与 GSM-DC（仅 6,300 样例） |
| D17 | 干扰项合成 | **不引新源**：规则复用 GSM-IC 的 242 条 `sentence_template`（全部由 `{role}`/`{number}` 参数化）施加到可解原题，答案不变、gold 不变、走 `_math_score`；难度梯度沿用 GSM-IC 自带三组天然配平标注 |
| D18 | 幻觉域规模 | 目标 **20,000 条**：不可解 **12,000（60%）** / 可解 8,000。**硬约束**：模板 A（带选项块）与模板 B（不带选项块）各自都必须同时含可解与不可解。构成表随 D21–D26 重排，见 §4.8 |
| D19 | K&K 答案契约 | 输出为**逐人「人名: 角色词」对**（`\boxed{Oliver: angel, Ethan: devil}`，顺序不限）；gold 存 **name→表面角色词 mapping**（adapter 由 `names` + `solution` + `knight_knave` 合成），reward 按 `role_words` 归一化**逐人比对**，人名集合须与 gold 恰好相等；纯角色词序列（无人名）判 0 |
| D20 | K&K clean | **不入池**：只用 `K-and-K/perturbed-knights-and-knaves`（41,176 行，6 扰动族）；组内去重保留（每组候选变体 ≤6）；训练池按组计仍 **≈5,000 组** |
| D21 | FalseQA 选项契约 | 选项为**「假前提片段 → 替换词」替换对**：gold = `假前提片段 -> 配对真前提片段`（由索引对齐配对 diff 提取）；2 个干扰对**左项固定为同一假前提片段、右项为题面外同类等长词**（k=3 沿用 D15）。左项无信息、区分度全在右项；三个右项**全部题面外** → "选右项不在题面内的那对"等表层捷径对三项均等命中 ≈ 随机 |
| D22 | FalseQA 可解侧 | **已被 D27 作废**（label=0 复活入池）。原口径：只用不可解侧（label=1 入池，label=0 不入池）；模板 A 的可解行改由 UMWP-answerable 承担 |
| D23 | SUM 配对判断任务 | SUM 不拆四档/三档；每行 = 一对题：**prompt 同含 answerable / unanswerable 两问（A/B 顺序行级随机 50:50）**，模型先判哪个可解再解答，输出 `\boxed{<ID>: <答案>}`；**两层 reward（AND 门控）**：判断与答案都对 +1；判断错、答案错或未解答均为 0（原判断 +0.5 / 答案 +0.5 的部分分已废弃——判断层随机下限 50%，部分分会奖励瞎猜，见 §11 风险 8）。依据：SUM 题面词袋 NB 五折 balanced-acc ≈ **0.50**，判断无表层捷径 |
| D24 | UMWP 全部两层化 | UMWP 只消费 `question / answer / answerable` 三字段，统一两层 reward（AND 门控）：**answerable 侧**输出 `\boxed{答案}`——给出答案即判定"可解"，答案也对才 +1（判对答错 / 拒答均 0 分）；**unanswerable 侧**三档 bare——`\boxed{UNSOLVABLE}` 即 +1。**放弃可见缺陷类四档诊断**（同题等长跨度选项整体废弃），四档诊断源只剩 FalseQA 928 条（**D26** 后 TreeCut 负样本四档化，四档合计 5,835）；cat 分类只用于缺陷类型统计，不再决定契约分支 |
| D25 | KUQ 移出 | **KUQ 整体不入池**：无可验证答案、只能判断是否 unknown，判断信号缺答案层约束、模型容易瞎猜 |
| D26 | TreeCut 正负双侧入池 + 负样本四档化 | TreeCut **正样本（可解）入池**：输出 `\boxed{答案}`，prompt 带**占位选项块**（题面条件中随机抽 k=3 条换变量名/变量值生成，无正确项），误拒 → 0（与普通可解行一致）；**负样本升四档诊断**：gold = 生成器剪掉的边（`cut` 已知），选项 = 生成器顺带产出的候选缺失条件（k=3，全部题面外），`\boxed{UNSOLVABLE: <ID>}` 命中才 +1。**正负 prompt 同带选项块**（模板 A 外观同构，同 k=3，D15），消除"有选项块 ⇒ UNSOLVABLE"捷径；负样本选项集须过 MiP 同款泄题测试（§9）。配额：正 1,000（GSM-IC 合成干扰让渡 1,400→400；**D27 后再让渡至 500**）/ 负 4,907（三档→四档，不可解侧不变） |
| D27 | FalseQA 双侧入池（label=0 复活） | **作废 D22 的"label=0 不入池"**：answerable 侧 928 条入池，与 label=1 索引对齐配对、**按对划分 split**（同对不进异侧）；走**两层 reward（AND 门控**，gt 键 `solvable_answer`）：`\boxed{答案}` 与 gold（label=0 answer）`norm_match` 通过才 +1（判对答错 → 0）；`\boxed{SOLVABLE}` → 0；误拒（`UNSOLVABLE` 任意形态）→ 0。label=0 answer 67.8% 为自由文本短答 → 答案层用归一化精确匹配（大小写/标点/冠词），噪声抽检见 Q17。**answerable 侧同样带占位替换对选项块**（题面内左项 + 题面外同类等长右项，k=3 无正确项），模板 A 外观与不可解侧同构，防"无选项块 ⇒ SOLVABLE"格式记忆。配额：行 3 扩为 UMWP 550 + FalseQA-ans 928 = 1,478；让渡 TreeCut 正 1,000→500、GSM-IC 1,200→772（§4.8） |
| D28 | 四档不可解行三层 AND-gate | **（D29 后：仅 stage 3；stage 2 已由 D29 回退为两段式，见下）** 四档不可解行（FalseQA-fake / TreeCut 负）的输出契约由 `\boxed{UNSOLVABLE: <选项ID>}` 扩为 `\boxed{UNSOLVABLE: <选项ID>: <修复后的答案>}`；reward 在原「拒答 + 指针命中」两层之上**加第三层「补条件后的正确解」**：三层全对 +1，否则 0。gold 侧新增 `repaired_answer`（必有）与 `repaired_free_text`（FalseQA 置 true 走 `norm_match`；TreeCut 数值走 `math_match`）。来源：FalseQA 用配对真题的 source answer（构造即证书——应用 gold 替换对恰好重放真题题面）；TreeCut 用 `certify_repaired_answer`（恢复 `deleted_sentence` 重跑 `certify_answer`，失败丢整对）。兼容性：缺 `repaired_answer` 的旧 parquet 回退旧两层口径 |
| D29 | 三层修复契约拆为独立 stage 3 | **动机**：D28 把"判不可解 + 指对选项 + 解出修复后的题"绑进 stage 2 同一个 reward，四档行第三层长期 ≈0 时整行零方差被 `filter_groups` 屏蔽，反而污染 stage 2 的"拒答 + 诊断"信号。**拆分**：① stage 2 四档行回退两段式 `\boxed{UNSOLVABLE: <选项ID>}`（拒答 + 指针命中即 +1；收到三段式输出时第三段被解析但忽略，不奖不罚）；② stage 3 原样保留 D28 三段式契约与三层 AND-gate，由 gt 键 `repair_task=true` 路由（`repair_task` 与模板 `A_repair` 1:1 绑定，schema 强校验；带 `repaired_answer` 无 `repair_task` 的 pre-D29 artifact 走 legacy 回退仍按三层门评分）；③ stage 3 数据 = 纯修复行（FalseQA-repair 928 + TreeCut-repair 4,907，`REPAIR_QUOTA`）+ 旧域回放（`--old_domain_ratio 0.7`），由 `mix_halluc.py --repair_mode` 构建（跳过 `enforce_pair_atomicity`，`A_repair` 模板平衡结构性豁免）；④ adapter 侧 `--repair-contract` 旗标控制是否产出 `repair_task`/`repaired_answer`（stage 2 默认不产出，TreeCut 顺带跳过 `certify_repaired_answer` 的整对丢弃）；⑤ stage 3 训练从 stage-2 ckpt 续训（`run_qwen3_4b_hallucination_stage3_fully_async.sh`），reward 文件与 stage 2 共用 `hallucination_compute_score.py`（路由依据是 ground_truth 的键，不是 data_source） |
| D30 | stage 3 域平衡重构（kkcut 合成 + 新配比） | **动机**：用户要求全局可解/不可解 = 70/30、老域/幻觉域 = 60/40（可解含两域），且合并后 math/code/logic 比例贴近老域；K&K 无原生不可解 → 合成。**内容**：① `kk_adapter.py --emit-unsolvable` 合成 **kkcut 行**——对已证唯一解的谜题删除一条陈述（`statements` 元组与 `quiz` 文本同步手术），仅当剩余子系统 2^N 全量枚举 **≥2 解** 时保留（删约束只能增解 ⇒ 被删陈述必要，枚举即证书；`verify_kk.py` 新增 `kkcut:` 独立审计通道，独立重推导不复用 adapter 代码）；契约 = 三档裸拒（模板 B、无 `role_words`，MiP 式），审计字段 `kkcut_deleted_index` / `kkcut_deleted_statement` / `kkcut_solutions_after`；`--kkcut-per-group 0` 允许同一父组枚举多条可删陈述，`kkcut:` task_id 增加第 5 段保证唯一；② stage 3 幻觉侧换**域平衡表 `STAGE3_QUOTA`**（14 cell，合计 14,000，§4.8）：logic 20%（2,800 = kk 700 可解 + kkcut 2,100 不可解）/ commonsense 20%（2,800 = FalseQA 300+820 + CREPE 400 判断 + 1,280 裸拒）/ math 60%（8,400 = GSM-IC 340 + UMWP 155/170/1,925 + 主池 15 + TreeCut 320/3,030 + SUM 2,200 + MiP 245），幻觉域内按**边权**（SUM 配对 0.5/0.5）精确 25/75，老域占 train 60% ⇒ 全局 70/30；老域按 **ability 分层抽样**（`--code_share`，stage-3 默认 0.24）：幻觉域没有 code 源，若均匀回放 code 会被稀释到全量的 ~15%，故先取 0.24×train_total 条 code，其余按池内供给比例分给 math/logic/stem，使合并后 math/code/logic ≈ 老域的 45/25/15；旧 `--repair_mode` 与 `REPAIR_QUOTA` 删除（D29 修复层已退役，stage 3 不再消费 `--repair-contract` 产物，配对原子性/去重/模板平衡检查与 stage 2 同一套原样生效）；③ CREPE 配额经 `--quota-overrides solvable_judge=400,unsolvable_bare=1280` 扩到 1,680；④ 构建入口 `build_hallucination_stage3_data.sh`（`HALLUC_TOTAL=14000`、`OLD_DOMAIN_RATIO=0.6`、`VAL_SIZE=700`，kk 带 `--emit-unsolvable --kkcut-per-group 0`，构建后跑 `verify_kk.py` 把坏 cut 行挡在构建期）→ `halluc/built_stage3/` → `mix_halluc.py --stage3_mix --strict_cells` → `final_halluc_stage3/`；**stage 2 表 B 不变** |
| D31 | stage 4：指令遵从域（15%）+ 老域新鲜化 | **动机**：stage 1–3 的数据模型已训练过一轮，继续在同一批题上压熵会过拟合；同时 stage 3 只优化拒答/判断，缺少显式格式与内容约束的指令遵从信号。**内容**：① 新增 **`if_adapter.py`**，把 `nvidia/Nemotron-RL-instruction_following`（46,391 行，stage 1–3 未使用）构造成 stage-1 schema 的 `if_nemotron` 池——逐行过滤 reward 无法评估的 instruction id（`if_verifier.SUPPORTED_INSTRUCTION_IDS`），prompt 精确去重，`--exclude-path` 剔除已训练过的 prompt，`--quota/--seed` 决定子采样，ground truth = `{"constraints":[…],"grading_mode":"binary\|fraction"}`；② reward 复用 `compute_score.py` 的 `if_*` 分支（`_if_score`），新增 **`grading_mode="fraction"`**（通过约束数 / 总数）；缺省/未知值仍是 NeMo-Gym 的 `binary`（全部通过才 1.0），且约束只校验 `</think>` 之后的可见回答；③ `mix_halluc.py` 新增**正交**旋钮：`--if_ratio`（IF 占**最终** train 的比例，`if = ratio/(1-ratio) × 非 IF 总量`，因此 stage-3 幻觉域绝对规模不变）、`--if_path`、`--if_val_size`（先从 IF 池切 val 再抽 train）；`--exclude_path`（stage 1/2/3 的 train parquet，精确题面命中即从老域回放池与 IF 池剔除）、`--old_pool_path`（把 `$DATA_ROOT/{math,code,logic,stem}` 原始池并入回放供给，从而抽到 stage-1 mix 从未采过的尾部；池内 `val.parquet` 跳过）；④ 构建入口 **`build_hallucination_stage4_data.sh`**（`IF_RATIO=0.15`、`SEED=44` 轮换 kkcut 顺序与 CREPE 子采样、`INCLUDE_OLD_POOLS=1` 默认开启新鲜化，`EXCLUDE_PATHS` 默认 `final`/`final_halluc`/`final_halluc_stage3`）→ `halluc/built_stage4/` → `mix_halluc.py --stage3_mix --strict_cells --if_ratio 0.15` → `final_halluc_stage4/`；构建后断言 IF 实际占比不低于请求值 1pt；⑤ 训练沿用 `run_qwen3_4b_hallucination_stage3_fully_async.sh`（`DATA_DIR=$STAGE4_DIR`、`RESUME_PATH=<stage-3 ckpt>`），reward 文件不变。默认口径下 stage 4 train ≈ 33,250（非 IF）+ 5,868（IF）= 39,118，IF = 15.0%；val ≈ 700（幻觉）+ 255（stage-1 val）+ 105（IF） |
| D33 | stage 4：按子任务长度惩罚（维持 DPO 精简习惯） | **动机**：stage 4 从 DPO 精简过思维链的 ckpt 起步，RL 会沿奖励坡度重新拉长 CoT；需要把"正确且简短"的行为锚住。**内容**：① 新统计脚本 `scripts/distill/measure_length_targets.py`——对 DPO ckpt 的 OpenAI 兼容服务（vLLM 必须带 `--reasoning-parser qwen3`）在 stage-4 train mix 上逐 prompt 生成，按 `data_source` 聚合**响应字符数**（默认 `mean`，可选 `p50/p90`，默认丢弃 `finish_reason=length` 的截断样本），产出 `length_stats.json` / `length_targets.json` / `length_targets.env`；样本数 < `--min_samples` 的源依次回退到 `extra_info.domain` 均值与全局 `__default__`；② reward 侧新增 4 个 reward_kwargs（`length_penalty_targets_path` / `length_penalty_coef` / `length_penalty_max`，以及便于单测的内联 `length_penalty_targets`），在分支打分（含委派给 stage-1 的 `math/code/logic/stem/if/bbh` 行）之后对**正分**行扣 `min(MAX, COEF × (len/target − 1))`，目标按 `data_source → extra_info.domain → __default__` 解析；0 分与唯一负分（CREPE 误拒 −1）不罚，`MAX < 1` 保证"正确但长"仍高于"错误但短"；③ stage-4 薄封装读取 `LENGTH_PENALTY_TARGETS_PATH` / `LENGTH_PENALTY_COEF` / `LENGTH_PENALTY_MAX` 三个环境变量并以 `++reward.custom_reward_function.reward_kwargs.*` 注入；不设 targets 时 reward 与 stage 3 完全一致；④ 惩罚开启时每行固定返回 `length_chars` / `length_target` / `length_penalty` 三个 aux 字段——reward-loop 以 batch 首样本的键集合为准（`reward_loop.py:366`），逐样本增删会直接 KeyError |
| D34 | stage 4：高难度数学域（15%）+ 老域降到 30% | **动机**：benchmark 报告显示高难数学是当前最弱项，而 stage 1–3 的数学回放以 Big-Math/DAPO 为主（难度低于 AIME）；stage 4 要在不扩大总量的前提下补齐这部分能力。**内容**：① 新增 **`hard_math_adapter.py`**，从 `nvidia/AceReason-Math`（≈50k，AMC–AIME）、`agentica-org/DeepScaleR-Preview-Dataset`（≈40k，AIME 1984-2023 + AMC + Omni-MATH + Still）、`zwhe99/DeepMath-103K`（难度 ≥ 6 子集）构建 `math_hard_*` 池（`dapo` 作为可选第四源，默认权重 0，因为它就是老域数学池）；② **答案格式统一**：只保留整数 / 简单分数 / 短小数（可带 `\%`），复杂符号答案默认丢弃（`--keep-symbolic` 可保留）；图片依赖题（`<image>` / Asymptote）始终丢弃；ground truth 只去 `$...$` / `\boxed{}` 包装，不改写数值；③ **去重**：池内 exact + MinHash（跨源重叠明显），可选 `--anchor-path` 对老域数学池做 MinHash 去近重，`--exclude-path` 按精确题面剔除已训练行；④ **去污染**：`--decontaminate-path`（默认 `$DATA_ROOT/val/aime_2024_2025.parquet` 等四个候选）做 word 8-gram + MinHash，未提供时**构建失败**（`--allow-no-decontamination` 仅调试）；⑤ **难度过滤**（不可省）：在 stage-4 起始 ckpt 的 vLLM OpenAI 服务上每题采样 `--difficulty-k 8`（可到 16），用训练 reward 打分，只保留通过率 ∈ `[0.1, 0.9]` 的题（全对/全错都丢弃），通过率写入 `extra_info.pass_rate`；结果按 `task_id` 缓存在 `<out>_difficulty.jsonl`，可断点续跑与分片；reward 不可用（缺 verl/math_verify）时 preflight 直接报错，不会把整池误判成全错；⑥ `mix_halluc.py` 新增第三个增量域 `--hard_math_ratio`（占最终 train 的 15%）、`--hard_math_path`、`--hard_math_weights`、`--hard_math_val_size`，三个增量域共用分母 `1 - if - bbh - hard`；`--old_domain_ratio` 由 0.6 降到 **0.5**，老域因此占最终 train 的 **30%**，最终构成 **幻觉 30% / 老域 30% / hard math 15% / IF 15% / BBH 10%**，train 总量仍约 **44.3k**；老域内部按 stage-3 的 `train_by_ability` 逐项匹配（`--match_stage3_abilities`，hard math 计入 math 预算并等量替换老域 math），老域总数仍 13,300，`--code_share` 退化为该模式关闭时的回退旋钮（0.16）；⑦ reward 侧 `compute_score.py` 的 math 分支新增 **`_numeric_literal_match`** 兜底层（`_math_last_resort` 首位）：用 `Fraction` 精确比较 AIME 风格字面量（`042 == 42`、`1,234 == 1234`、`\dfrac{1}{2} == 0.5`），但 `0.333 != 1/3`，避免把四舍五入误判为精确相等；`math_hard_*` 前缀仍路由到同一 stage-1 数学匹配链；⑧ 构建入口 `build_hallucination_stage4_data.sh`（`HARD_MATH_RATIO=0.15`、`HARD_MATH_DIFFICULTY_BASE_URL`/`MODEL` 必填，`SKIP_HARD_MATH_DIFFICULTY=1` 仅调试），构建后断言 hard math 实际占比 ≥ 请求值 −1pt 且 pass-rate 数据非空 |

---

## 1. 目标与非目标

**目标**：让模型在可解题上给出答案；在不可解题上**拒答而不是强行编造**。拒答本身必须可校验——
"正确识别不可解"才得分。**细粒度诊断（指出缺了/矛盾的是哪一处）只在选项语义可靠、无表层捷径的源上
启用**：FalseQA（替换对选项，D21）。
MiP 有不可解数据但选项集必然泄题，按 D12 退化为"只判拒答"；UMWP 可见缺陷类的同题等长跨度选项
已随 D24 废弃。
**可解侧同样有"别乱拒答"的压力**：判断型可解行（CREPE-normal）误拒给 −1；UMWP 可解侧（D24）与
SUM 配对任务（D23）把"识别可解"与"解出答案"绑定在同一行里训练（两层 reward）。

**非目标（本阶段）**：
- 不做数据生成（IGC-MWP 式生成器）；TreeCut 只用其现成生成器产对照数据。
- 不做 SFT / 冷启动示范轨迹（D6）。
- 不做一致性辅助项（D4）。
- 不做 AbstentionBench 训练数据（D9）。
- 不做 embedding 近邻干扰项挖掘（跨题干扰项一律禁用，§5.1 告示）。
- 不做 spec §3.3 的"两分量分别组内归一化"（D5）。

---

## 2. 与现有 reasoning_rl 的关系

### 2.1 完全不动（只读复用）

| 现有资产 | 复用方式 |
|---|---|
| `reward/compute_score.py` | 新文件按前缀委托：非 `halluc_*` 的 sample 原样转调它的 `compute_score`，一行不改 |
| `reward/` 下其它 verifier | 同上，经 `compute_score` 间接复用 |
| `scripts/mix.py`、`scripts/to_parquet_*.py`、`scripts/dedup.py`、`scripts/decontaminate.py` | 只调用其函数/模式，不改文件 |
| `run_qwen3_4b_reasoning_rl_dapo.sh` | 复用 `TRAIN_FILES`/`VAL_FILES` + `RESUME_MODE`/`RESUME_PATH` 热插拔接缝；reward 路径切换见 §12 Q3 |
| `reasoning_rl_dataset.py` / `hard_replay.py` | 阶段二沿用（system prompt 注入、hard replay 均正交） |

关键复用点（已核对源码）：

- `compute_score()` 按 `data_source` 前缀分发，未知前缀在格式合规后抛 `NotImplementedError`
  → 新增前缀必须由新文件自己的 dispatcher 处理。
- `format_ok()` 是通用结构门：要求 `<think>…</think>` 且 reasoning/final response 均非空
  → `halluc_*` 走同一道门。
- `_extract_boxed()` 已实现"取最后一个配平花括号的 `\boxed{}`" → 新 reward 直接复用，不重造。
- run 脚本已有 `train_files=${TRAIN_FILES:-…}` / `val_files=${VAL_FILES:-…}` 与
  `RESUME_MODE`/`RESUME_PATH`；`mix.py --if_ratio` 是"后程混入新域"的现成先例。

### 2.2 新增（全部是新文件，见 §8）

- `reward/hallucination_compute_score.py` + 测试
- `scripts/hallucination/` 下一整套 schema / adapter / verify / mix / 测试

---

## 3. 统一数据 Schema 与 verl parquet 映射

落盘统一转成现有 reasoning_rl 的 parquet 约定（与 `DESIGN.md` §1 同构）：

| 字段 | 落盘位置 | 说明 |
|---|---|---|
| `id` | `extra_info.task_id` | 全局唯一，hard replay 的 dedup key |
| `domain` | `ability` + `extra_info.domain` | `logic` / `math` / `commonsense` |
| `source_dataset` | `extra_info.source` | 如 `K-and-K`、`MiP-SVAMP`、`FalseQA`、`GSM-IC`、`SUM`、`UMWP`、`TreeCut`、`CREPE` |
| `prompt`（含选项块） | `prompt` | chat 格式 `[{"role":"user","content":…}]`，按 §5.2 模板渲染 |
| `solvable` | `reward_model.ground_truth` JSON | 见下 |
| `ground_truth_answer` | 同上 | `solvable=true` 必填 |
| `role_words` | 同上 | ⚠️ **K&K 专用**：该行自己的表面角色词 `[真话者词, 说谎者词]`（取自 `knight_knave` 映射）。答案解析必须按它归一化，否则 reward 反向，见 §5.1 |
| `correct_option_id` | 同上 | `solvable=false` 且 `has_diagnosis_label=true` 时必填 |
| `has_diagnosis_label` | 同上 | 决定四档 / 三档；**不是 `solvable=false` 的同义词**（MiP/UMWP/CREPE 等为 `false`；TreeCut 负样本为 `true`，D26） |
| `judgment_only` | 同上 | `true` 时该条**只判"是否可答"**（期望 `\boxed{SOLVABLE}`），不做数值匹配；仅用于 CREPE-normal |
| `two_layer` | 同上 | ⚠️ **UMWP-answerable 专用（D24）**：`true` 时走两层 reward（AND 门控）——给出答案即判定"可解"，答案也对才 +1，否则 0 |
| `solvable_answer` | 同上 | ⚠️ **FalseQA-answerable 专用（D27）**：`true` 时走两层 reward（AND 门控）——给出答案即判定"可解"，与 `answer` 归一化精确匹配也对才 +1；`\boxed{SOLVABLE}` → 0 |
| `pair_task` + `answerable_id` | 同上 | ⚠️ **SUM 专用（D23）**：`pair_task=true`，`answerable_id ∈ {"A","B"}` 标记哪个问题可解；A/B 顺序由 adapter 行级随机 |
| `options` | `extra_info.options` + prompt 渲染 | 选项文本（FalseQA 为替换对），reward 侧只比 `correct_option_id` |
| `perturbation_type` | `extra_info.perturbation_type` | `missing_condition / contradictory_condition / distracting_condition / null` |
| `difficulty_tag` | `extra_info.difficulty` | 来源原生难度，供分桶监控 |
| `metadata.paired_original_text` | `extra_info.paired_original_text` | 审计用（可裁掉以控体积） |
| `pair_id` | `extra_info.pair_id` | ⚠️ **FalseQA 专用（D27）**：同一索引配对（label=1 / label=0）两条共用的稳定键；mix 用它保证同对不跨 train/val 两侧（两侧行落在不同 `(branch, source)` cell，按 cell 切分会裂开），reward 不读 |
| `repaired_answer` | `reward_model.ground_truth` JSON | ⚠️ **stage-3 修复行专用（D28/D29）**：补上 gold 条件后题面的正确解。`repair_task=true` 时必填（schema 强校验）；**stage-2 四档行不得携带**（两段式契约，D29）。FalseQA 来自配对真题的 source answer（自由文本）；TreeCut 来自 `certify_repaired_answer` 恢复 `deleted_sentence` 后重跑 `certify_answer`（整数）。**2026-10-15 起仅作审计字段：reward 不再读取，修复层从评分中退役** |
| `repaired_free_text` | 同上 | ⚠️ **stage-3 修复行专用（D28/D29）**：原为匹配器路由键（`true` → `norm_match`，缺省 → `math_match`）。**2026-10-15 起仅作审计标注，reward 不再读取** |
| `repair_task` | 同上 | ⚠️ **stage-3 模板绑定键（D29）**：`true` 时该四档行必须使用模板 `A_repair`（1:1 绑定，schema 强校验）。**2026-10-15 起不再改变奖励行为**：`repair_task` 行与 stage-2 行同走指针契约（指针命中即 +1），不再路由三层 AND-gate；带 `repaired_answer` 而无 `repair_task` 的 pre-D29 artifact 同样指针契约。**D30 起 stage-3 构建不再产出该键**（`--repair-contract` 废除，见 §4.8），schema 校验保留供 pre-D29 artifact 审计 |

`reward_model` 通用示例：

```json
{
  "style": "rule",
  "ground_truth": "{\"solvable\": false, \"answer\": null, \"correct_option_id\": \"B\", \"has_diagnosis_label\": true, \"perturbation_type\": \"contradictory_condition\"}"
}
```

K&K 行示例（可解、无选项、带角色词表；`answer` 为 **name→表面角色词 mapping**，D19）：

```json
{
  "style": "rule",
  "ground_truth": "{\"solvable\": true, \"answer\": {\"Quinn\": \"angel\", \"Ava\": \"devil\", \"Jack\": \"devil\"}, \"role_words\": [\"angel\", \"devil\"], \"has_diagnosis_label\": false, \"perturbation_type\": null}"
}
```

SUM 行示例（配对判断任务，D23；A/B 顺序已随机，本例 answerable 在 B）：

```json
{
  "style": "rule",
  "ground_truth": "{\"solvable\": true, \"pair_task\": true, \"answerable_id\": \"B\", \"answer\": 42, \"has_diagnosis_label\": false, \"perturbation_type\": null}"
}
```

UMWP-answerable 行示例（两层 reward，D24）：

```json
{
  "style": "rule",
  "ground_truth": "{\"solvable\": true, \"two_layer\": true, \"answer\": 42, \"has_diagnosis_label\": false, \"perturbation_type\": null}"
}
```

FalseQA-answerable 行示例（两层 reward，D27；`answer` 为自由文本，reward 侧归一化匹配）：

```json
{
  "style": "rule",
  "ground_truth": "{\"solvable\": true, \"solvable_answer\": true, \"answer\": \"a teacher\", \"has_diagnosis_label\": false, \"perturbation_type\": null}"
}
```

四档不可解行示例（**stage 2** 两段式，D29；FalseQA-fake 侧，拒答 + 指针命中即 +1）：

```json
{
  "style": "rule",
  "ground_truth": "{\"solvable\": false, \"answer\": null, \"correct_option_id\": \"B\", \"has_diagnosis_label\": true, \"perturbation_type\": \"contradictory_condition\"}"
}
```

四档不可解行示例（**stage 3** 三层，D28/D29；FalseQA-repair 侧，`repair_task=true` + `repaired_free_text=true`）：

```json
{
  "style": "rule",
  "ground_truth": "{\"solvable\": false, \"answer\": null, \"correct_option_id\": \"B\", \"has_diagnosis_label\": true, \"perturbation_type\": \"contradictory_condition\", \"repair_task\": true, \"repaired_answer\": \"Because a child needs food.\", \"repaired_free_text\": true}"
}
```

四档不可解行示例（**stage 3** 三层，D28/D29；TreeCut-repair 负样本侧，`repaired_answer` 为链式证书认证的整数）：

```json
{
  "style": "rule",
  "ground_truth": "{\"solvable\": false, \"answer\": null, \"correct_option_id\": \"A\", \"has_diagnosis_label\": true, \"perturbation_type\": \"missing_condition\", \"repair_task\": true, \"repaired_answer\": \"42\"}"
}
```

> 约定：`ground_truth` 是 **JSON 字符串**（parquet 列里存字符串，reward 侧 `json.loads`）；
> 新增字段不需要动 parquet schema。解析失败 → reward 记 0 并 log（fail closed）。

`data_source` 命名（= reward 路由键）：

| data_source | 含义 | scorer |
|---|---|---|
| `halluc_logic_kk` | K&K 扰动（D20：clean 不入池） | `kk_match`（§6） |
| `halluc_math_mip` | MiP 不可解（缺条件，三档） | 新逻辑（§6） |
| `halluc_commonsense_falseqa` | FalseQA 双侧（label=1 四档替换对 / label=0 两层，D27） | 新逻辑（§6） |
| `halluc_math_gsmic` | GSM-IC（可解 + 干扰条件） | 委托现有 `_math_score` |
| `halluc_math_sumpair` | SUM 配对判断（D23） | 两层 reward（§6） |
| `halluc_math_umwp` | UMWP（answerable 两层 / unanswerable 三档，D24） | 新逻辑（§6） |
| `halluc_math_treecut` | TreeCut（正样本可解 + 负样本四档，D26） | 新逻辑（§6） |
| `halluc_commonsense_crepe` | CREPE（normal 判断型 / false-presupposition 三档） | 新逻辑（§6） |
| `halluc_math_main` | 阶段一主池数学题被 §4.7 抽作合成干扰底题的 100 行（D17） | 新逻辑（§6，普通数值分支） |

> **实现口径（2026-09-20 复核）**：`halluc_math_main` 是 §4.7「主池数学题 100」的落地路由键
> （本表初稿只登记 8 个源；第 9 个键是实现时才需要的——合成行必须与主池行分开记账与监控）。
> 它只承载 D17 的 100 条合成干扰行，配额计入 §4.8 行 1 的数值 cell（772+200+100=1,072）。
>
> **实现口径（2026-09-20 复核）**：`extra_info.difficulty` 沿用阶段一「字符串分桶」约定
> （`kk_adapter` 写 `"4ppl"` 这类档位串），与 §4.1「`difficulty_tag = len(names)`」的数值读法
> 差一层映射——人数就是档位串的数字部分，监控分桶等价。落盘类型以本表为准（字符串）。
>
> **实现口径（2026-09-20 复核）**：`validate_row(s)` 除本表字段外还硬校验四件事，任一不过即拒绝
> 落盘（fail closed）：① `extra_info.branch` ∈ §4.8 表 B 的 7 个契约分支键；② `extra_info.template`
> ∈ {A, A_repair, B, B_judge, C}（A_repair 随 D29 加入，与 `repair_task` 1:1 绑定；
> D30 起 stage-3 构建不产出 A_repair 行，枚举保留供 pre-D29 artifact 校验）；
> ③ `ability == extra_info.domain`（本表 `domain` 同时落两处的映射）；
> ④ `task_id` **全局唯一**（本表「hard replay 的 dedup key」），冲突按批报出。带选项块的行
> 选项数必须 = k = 3（D15）。D29 追加：repair 字段的一致性校验——`repaired_answer` 无
> `repair_task` 拒绝（stage-2 行不得携带）、`repair_task` 无 `repaired_answer` 拒绝、
> `repair_task` ⟺ 模板 `A_repair` 双向绑定、`repair_task` 仅限四档行。

---

## 4. 数据源与 adapter

数据根目录约定（与现有 `DATA_DIR` 并列，不混入）：

```
~/data/reasoning_rl/halluc/
├── raw/          # 各源原始下载
├── built/        # 各 adapter 产出的统一 schema parquet（每源一份）
└── final/        # mix_halluc.py 产出的阶段二 train.parquet / val.parquet
```

### 4.1 K&K（可解锚点；D11 / D19 / D20）

**源事实（全量实测 48,076 行 = clean 7,000 + perturbed 41,176）**：

- **零不可解数据**：逐条还原约束系统并穷举赋值——0 解 0 条、唯一解 48,076 条、枚举解与
  `solution` 字段全一致。生成器（arXiv 2410.23123 §2.2）强制"扰动后仍有不同的解"，无解候选
  生成时即被丢弃。→ K&K **不提供任何弃答/诊断信号**，定位为纯可解锚点。
- **六种扰动的性质**：`perturbed_statement` / `perturbed_leaf` 为数学级（抽象 statements 改变、
  角色词不变）；`reorder_statement` / `uncommon_name` / `flip_role` / `random_pair` 为语言级
  （抽象 statements 与 clean **逐字节相同**）。其中 `flip_role`（角色词互换）与 `random_pair`
  （angel/devil、sage/fool 等 6 组词对）**真的换角色词**，合计 **13,800 行 = 28.7%**——
  硬编码 `K = knight` 会在这 28.7% 上 reward 完全反向（风险 5）。
- **重复度 7×**：语料 = 约 6,900 个抽象题 × 7 个变体（clean + 6 扰动）→ 必须按
  `(len(names), index)` 组内去重；该键对 clean↔6 类扰动的联接覆盖率 100%。
- **License**：clean 与 perturbed 均为 **cc-by-nc-sa-4.0（禁止商用）**。

**adapter（`kk_adapter.py`）**：

- 输入：**仅 perturbed**（D20；41,176 行）。若将来恢复 clean，只需加回各组候选变体，去重键与
  split 划分不变。
- 产出：全部 `solvable=true`，无选项块（`has_diagnosis_label=false`、`correct_option_id=null`）。
- **答案（D19）**：`ground_truth_answer` = **name→表面角色词 mapping**
  （如 `{"Oliver": "angel", "Ethan": "devil"}`），由 `names` + `solution` + `knight_knave` 合成
  （`solution[i]` 为规范 bool，`True` → `role_words[0]`、`False` → `role_words[1]`，与 `names[i]`
  配对）；同时写 `role_words = (knight_knave['knight'], knight_knave['knave'])`；规范 bool 列表
  只留在 `extra_info` 供审计。
- **组内去重**：按 `(len(names), index)` 分组（每组 ≤6 条变体），每组只取 ≤1 条；同组成员
  不得同时进 train 或 val 的同一侧（按组划分 split，避免变体泄漏）。
- **筛除**：N < 4 的档位不进训练集（瞎猜下限过高：2ppl 25%、3ppl 12.5%）→ 训练池 **≈5,000 组**。
- `difficulty_tag = len(names)`；`extra_info.perturbation_family` 仅作监控分桶，不参与 reward。
  > **实现口径（2026-09-20 复核）**：落盘时 `extra_info.difficulty` 是**字符串档位**（`"4ppl"`，
  > 沿用阶段一的字符串约定，见 §3 注），数值就是 `len(names)`；`task_id` 形态
  > `kk:{family}:{N}ppl:{index}` 同时证明 N。`verify_kk.py` 现已硬断言 **N ≥ 4**（`MIN_INHABITANTS`），
  > 2ppl/3ppl 行重新混入会被审计拦下。
- 保留枚举器作 gold 自检与测试 oracle（免费的、可随时重算的 ground truth）。

### 4.2 MiP（三档不可解；D12）

- 来源：`github.com/tianyi-lab/MiP-Overthinking` 的 `data/{gsm8k,svamp,math,formula}.json`
  （非 HF 数据集）。合计仅 **984 条**（gsm8k 582 / svamp 300 / math 52 / formula 50）。
  构造方式是从原题**删掉或抽象掉一个数值前提**：`question` = 可解原题（带 `answer`/`solution`），
  `insufficient_question` = 残缺版。
- **不做诊断的原因（三层审计）**：L1 标签证书——被删段重插回原题仅 90.3% 可重建、8.1% 必要性
  查不到；L2 gold 唯一性——209 条含 ≥2 个数值、55 条不含数值、37 条多段改动；**L3 反作弊——
  正确答案按定义不在题面里、干扰项按定义都在题面里，"选题面里唯一找不到的那句"命中 95.6%
  （抹掉数字后仍 95.6%），跨题干扰项则退化成"主题重叠最大"（83.7%）**。这是结构性的，
  RL 一定会找到这条路 → 放弃诊断（spec §1 自带 `has_diagnosis_label=false` 的降级路径）。
- **入池（三档 bare）**：漏斗 984 → 634 有配对 → 626 题面真不同 → 325 单值前提被移除 →
  299 数值确实从题面消失 → 262 过必要性 + 14 占位词型 = **严口径 276**（默认）/ 宽口径 299。
  > **实现口径（2026-09-20 复核）**：`mip_adapter.py` 实现的是 fail-closed 的严口径，实测漏斗为
  > 984 → 634 → 626 → 325 → 299 → 299 → **270**（gsm8k 254 / math 16）：比文档的 276 少 6 条，
  > 差额来自"证书链缺失 2 条 + 源从未使用的占位值 3 条 + 占位值仍在题面 1 条"。按 Q6 默认
  > **不补**：mix 的 `(unsolvable_bare, MiP)` cell 仍写文档的 276，缺的 ~9 条（含 val 切分）
  > 作为 cell shortfall 进 `mix_stats.json`，不静默凑数。
  不可解行 `has_diagnosis_label=false`、`perturbation_type=missing_condition`、
  **prompt 不附选项块**（模板 B，零泄漏）。
- **可解侧不进池**：与 Big-Math 主池同源（GSM8K/MATH）、量小、去重成本高于收益（§12 Q7）。
- 交付 `verify_mip.py`：L1/L2 断言（证书 + 必要性 + 唯一性），见 §9。

### 4.3 FalseQA（双侧入池：label=1 四档诊断 + label=0 两层；D13 / D21 / D27）

**源事实（实测 `fq_train.csv` / `fq_valid.csv` / `fq_test.csv`）**：

- 列只有 `question, answer, label`。`label=1` = 假前提，`label=0` = 真前提，三份 split 严格 50:50
  （train 1,187/1,187，valid 491/491，test 687/687）。
- **`answer` 列不能当 gold**：label=0 侧 67.8% 是自由文本短答，label=1 侧是自由文本 rebuttal，
  test split 每条 3 个并列参考答案——不存在唯一正确字符串。
- **配对是索引对齐的**：第 k 条 `label=1` = 第 k 条 `label=0` 的局部改写，词级 Jaccard ≥0.5 占
  **89.1% / 88.6% / 90.2%**——本源自带的"配对证书"。
- 与 MiP 的本质区别：假前提是**被替换**而非被删，**假前提片段就在题面里**，pointer gold 可知。

**使用方式（D27 双侧入池；作废 D22）**：**label=1（不可解）928 + label=0（answerable）928
均入池**，索引对齐配对、**按对划分 split**（同对不进异侧，防"题面几乎相同"跨侧泄漏）。

- **label=1 侧（不可解·四档；输出按 stage 分两式，奖励统一指针契约）**：沿用 D21 替换对选项契约
  （下述），`has_diagnosis_label=true`。**stage 2（默认构建）**：`\boxed{UNSOLVABLE: <ID>}`——
  拒答 + 指针命中即 +1。**stage 3（D29 的 `--repair-contract` 构建；D30 起该旗标不再用于
  stage-3 构建，机制保留供未来实验）**：gt 加 `repair_task=true`、
  prompt 要求三段式 `\boxed{UNSOLVABLE: <ID>: <修复后的答案>}`——**奖励仍是 stage-2 的两层
  契约：拒答 + 指针命中即 +1，修复解段被解析但不计分**（2026-10-15 决策：`norm_match` 自由文本
  匹配是主要误判源，修复层从奖励中退役）。`repaired_answer` = 配对真题的 source `answer`
  （构造即证书：应用 gold 替换对恰好重放真题题面），仅作审计字段保留；`repaired_free_text`
  同样只作审计标注。配对真题 answer 为空 → stage 3 整对丢（`answer_empty` 计 1；未观测到）；
  stage 2 不需要 `repaired_answer`，answerable 侧（label=0）此时仍存活、diag 行单独保留。
- **label=0 侧（answerable·两层，gt 键 `solvable_answer`，AND 门控）**：`\boxed{答案}`
  与 gold（label=0 `answer`）`norm_match` 通过才 +1（判对答错 → 0）；`\boxed{SOLVABLE}`
  → 0；误拒（`UNSOLVABLE` 任意形态）→ 0。`answer` 67.8% 为自由文本短答 → 答案层用
  **归一化精确匹配**（小写/去标点/去冠词/strip）；AND 门控下同义 paraphrase 会损失整分，
  匹配噪声人工抽检 N=50（§12 Q17）。
- **answerable 侧同样带占位替换对选项块**（左项 = 题面内内容短语、右项 = 题面外同类等长词，
  k=3 无正确项）——与不可解侧模板 A 外观同构，防"无选项块 ⇒ SOLVABLE"格式记忆（D18）。

**选项契约（D21，替换对）**：

- **gold 对** = `假前提片段 -> 配对真前提片段`，由配对 diff（word-level `difflib`，`autojunk=False`）
  提取：fake 侧差异区域须恰好一个连续片段（train 928 条 = 78.2%、valid 377、test 536 满足），
  且含内容词（98.9%）。示例：`'man -> women'`、`'rainy days -> summer holiday'`。
- **干扰对（2 个，k=3 沿用 D15）**：**左项固定为同一假前提片段**，右项为**题面外同类等长词**
  （词性/实体类型与 gold 右项一致、词数相同，从题外词表规则采样）。
- **反作弊论证**：左项三项全同 → 无信息；三个右项**全部题面外** → "选右项不在题面内的那对"
  对三项均等命中 ≈ 随机（若干扰右项取自本题面，则 gold 右项是唯一的题外词，该启发式 100% 命中——
  这是必须避免的镜像捷径）。剩下的区分只能靠"哪个替换真的让前提成立"，即任务本身。
- **残余风险**：规则生成的干扰右项可能语义上也能修复前提（"也对"）→ §9 抽检 + §12 Q9 人工复核。

**reward（answerable 侧两层 / 不可解侧统一指针契约）**：不可解侧走 §6 四档分支——**stage 2（D29
两段式）**：`\boxed{UNSOLVABLE: <ID>}` 命中 gold → +1；编造答案 → 0；bare `\boxed{UNSOLVABLE}` /
指针错 → 0（收到三段式输出时第三段被解析但忽略，指针命中仍 +1）。**stage 3（`repair_task` 行）**
奖励与 stage 2 **完全一致**：ID 命中 gold → +1，修复解段无论给不给、对不对都不计分（2026-10-15
决策：`norm_match` / `math_match` 修复匹配是主要误判源，修复层从奖励中退役，`repaired_answer`
仅作审计）。answerable 侧走 §6 `solvable_answer` 分支（AND 门控：两层都对 +1，否则 0，见上）。

**交付约束（`falseqa_adapter.py`）**：

- gold = 配对 diff 的替换对（不是 `answer` 字段）；`correct_option_id` 指向 gold 对，
  `has_diagnosis_label=true`，`perturbation_type=contradictory_condition`。
- 选项恰好 3 项（D15），左项全同、右项两两不同且与 gold 右项**同类等长、题面外**，位置按 seed 打乱。
- **answerable 侧（D27）**：`solvable=true`、gt 键 `solvable_answer=true`、`answer` 键
  = label=0 `answer` 字段（原样保留，reward 侧做归一化）；选项块为**占位替换对**（左项题面内、
  右项题面外同类等长，k=3 无正确项，`correct_option_id=null`）；与 label=1 侧**按对分 split**。
- 差异区域不唯一 / 无法构造合格干扰右项 → 该条丢弃，禁止兜底。
- **stage 3 修复契约（D29；D30 起 stage-3 构建不再传该旗标，机制保留）**：`--repair-contract` 时
  diag 行 gt 加 `repair_task=true` +
  `repaired_answer`（= 配对真题 source answer）+ `repaired_free_text=true`、模板改用 `A_repair`
  （三段式 marker）；默认（stage 2）不带任何 repair 键、模板 A（两段式 marker）。answerable 侧
  两种模式恒为 stage-2 形态。配对真题 answer 为空时：stage 2 只丢 answerable 行、diag 行存活；
  repair 模式整对丢。
- 交付前必须跑 `verify_falseqa.py`（§9）并留报告。

> **实现口径（2026-09-20 复核）**：真实三 split 的产出为 train 1,757（**829 四档 + 928 两层**）、
> valid 713（336 + 377）、test 1,011（476 + 535），审计各 PASS 31/31。四档侧比 §4.8 的 928 少 99：
> 区域证书本身仍复现 928/377/535，缺的 99 条死在 D21 的**金标对**门槛上（33 条纯插入 ⇒ 没有"配对
> 真前提片段"、30 条 gold 首词不唯一、24 条修复片段仍在题面内、11 条假前提片段只有停用词、1 条
> 干扰右项不足 k−1）——即 D21 把一个"区域证书"任务收紧成"替换对证书"任务的必然代价，fail-closed
> 不兜底。混样时该 cell 的 ~99 条缺口（+val 切分）作为 shortfall 报出。answerable 侧只需区域证书 +
> 非空 answer，恰好 928。
> 另：干扰右项取自"同 split 题面词表 + 表面类型签名 + 字符长度分层"，残余提示"选最高频右项"
> 实测 train 0.46 / test 0.42（基线 1/3），`verify_falseqa.py` 打印但**不设门槛**（§9 未给该源设 L3
> 硬门槛）；若将来要压，改用文档频率带会把提示移向"选最稀有"（0.48–0.56）并损失约 4% 产出。

### 4.4 GSM-IC（可解 + 干扰对照；常规处理）

- 来源：原始 repo `github.com/google-research-datasets/GSM-IC` 的 `GSM-IC_2step.json` +
  `GSM-IC_mstep.json`（HF 的 `voidful/GSM-IC` 只有 1,000 行子集，不能用）；
  实测 **2step 34,220 + mstep 23,832 = 58,052 条**。
- 性质：**可解但含无关干扰条件**，`solvable=true`，`perturbation_type=distracting_condition`。
- **reward 校验与答案提取保持常规**：与普通数学题完全一样——gold = 原 `answer`，
  期望 `\boxed{答案}`，走现有 `_math_score`（`halluc_math_gsmic` 委托），**不新增任何判分逻辑**。
  > **实现口径（2026-09-20 复核）**：① 640 行的 gold 去掉了千分位逗号（`1,200` → `1200`）——
  > `math_verify`/`math_dapo` 对带逗号的字面量解析不稳，数值等价、字符串不同；其余行逐字节等于原
  > `answer`。② `verify_gsmic.py` 现已把「三组标注配平」做成**硬断言**（§8 承诺、本节称"天然配平"）：
  > 实测 800 行池 `in_topic/out_topic = 376/424`、`overlapped/nonoverlapped = 373/414`（13 行 `n/a`）、
  > `in_range/out_range = 385/414`（1 行 `n/a`），三轴各档均在文档目标的 ±5pt 内、`n/a` ≤5%。
  > ③ 该源的 reward 委托见 §6 注①（无 `halluc_math_gsmic` 专门分支的原因）。
- 附带资产：242 条 `sentence_template`（全部由 `{role}`/`{number}` 参数化）抽出来当干扰项
  合成引擎（D17，§4.7）；`sentence_label` / `role_label` / `number_label` 三组标注天然配平。

### 4.5 SUM（配对判断任务；D23）

**源事实**：HF `lime-nlp/Synthetic_Unanswerable_Math` 单 parquet（MIT），train **36,480**
（+ test 284）。每行三列：`answerable_question` / `unanswerable_question` / `ground_truth`
（可解题的答案）——**同行即配对**，无需另行匹配。不可解标签为 o3-mini 生成 + 专家复核，
**无机器证书**（无推导链）→ 需人工/规则抽检（§12 Q10）。题面词袋 NB 五折 balanced-acc =
**0.502 / 0.503** ≈ 随机 → "哪个可解"没有表层捷径，判断层是真实任务。

**数据构造（D23）**：

- prompt 中**同时包含** `answerable_question` 与 `unanswerable_question`，标号 A/B；
  **A/B 顺序行级随机（50:50，按行 seed）**，不能固定 A=answerable——否则模型靠位置瞎猜。
- 模型任务：**先判断哪个问题可解，再解答该问题**，输出 `\boxed{<ID>: <答案>}`（如 `\boxed{A: 42}`）。
- gold：`pair_task=true`、`answerable_id ∈ {"A","B"}`、`answer` = 该行 `ground_truth`。

**两层 reward（AND 门控）**：

| 情形 | 得分 |
|---|---|
| 判断对（ID = `answerable_id`）且答案对（`math_match` vs `ground_truth`） | **+1.0** |
| 判断对、答案错 | **0** |
| 判断错（答的是不可解题，答案层不给分、也不额外扣分——任务强制二选一） | **0** |
| 无法解析（无 `\boxed{}` / 不是 `ID: 答案` 形态） | **0** |

设计依据：判断层随机下限 50%，原 0.5+0.5 部分分给瞎猜留了 0.25 的期望（§11 风险 8）；
AND 门控下判断层单独不值分，只有完整解出（+1）有奖励，瞎猜期望降为 0——解出本身就
要求识别出可解题，答案层严格难于判断层。判断层准确率仍须对照 50% 基线监控（§10）。

**adapter（`sum_adapter.py`）**：断言两问均非空、`ground_truth` 可解析；A/B 打乱用行级 seed
（可复现）；`verify_sum.py` 断言成品中 `answerable_id` 的 A/B 分布为 50:50 ±2pt。

> **实现口径（2026-09-20 复核）**：36,480 行原始 → 36,342 可用（94 条 gold 冲突 + 44 条同 gold 重复）
> → 选 6,000 条（`random.Random(f"{seed}:order")` 全库洗牌，非 first-N）；A/B 实测 3,032/2,968 =
> 50.53% A（±2pt 内）。**L3 复核发现**：§4.5 报告的 BoW-NB 五折 balanced-acc 0.502/0.503 在本仓
> 估计器（`bow_nb_oof`，min_support=5）下不可复现——4,000 条成员上 0.5565、12,000 条成员上 **0.5927**
> （shuffled-label 对照 0.497/0.494），跨 min_support 协议读数区间 0.46–0.63。仍低于 Q8 默认门槛
> （随机 + 10pt = 0.60），且 §9 未给 SUM 设 L3 硬门槛 → `verify_sum.py` 打印为 *L3b finding*，
> 不新增文档没设的硬门槛。上线时按 §10 的判断层准确率（对照 50% 基线）盯这一点。

### 4.6 UMWP / TreeCut / CREPE（扩源；D16）

| 源 | 取数方式 | license | 实测规模 | 配对 | 定档 |
|---|---|---|---|---|---|
| **UMWP** | GitHub `data/StandardDataset.jsonl` | CC-BY-SA-4.0（论文禁商用） | 5,200（2,600/2,600） | ✅ 100%（`relevant_ids`） | 两层化（D24）：answerable 550（判断+解答）/ unanswerable 2,489 三档 |
| **TreeCut** | 生成器 repo（纯 Python 无依赖） | Apache-2.0 | 可无限生成 | ✅ 同参数生成对照 | 正 500（模板 A 可解行，占位选项块；D27 让渡后）+ 负 4,907 四档（D26；**L3 修复前不得入池**） |
| **CREPE** | HF 镜像 `tasksource/CREPE`（官方 Google Drive 已 404） | BSD | 8,466 | ❌ | normal 判断型 250 / false-presupposition 三档 400 |

各源要点：

- **UMWP**（D24 两层化）：只消费 `question / answer / answerable` 三字段，统一两层 reward
  （AND 门控）——**answerable 侧**进模板 A 的可解行（承担 D18"模板 A 必须含可解"的硬约束），
  输出 `\boxed{答案}`，给出答案即判定"可解"，答案也对才 +1（判对答错 / 拒答均 0 分）；**unanswerable 侧**全部
  三档 bare（`\boxed{UNSOLVABLE}` 即 +1，prompt 不附选项块）。可见缺陷类的同题等长跨度选项
  **整体废弃**；不可解类别（Key Information Missing 32% / Ambiguous 49% / Unrealistic 11% /
  Unrelated Object 4% / Question Missing 5%）仅用于缺陷类型统计，不再决定契约分支。
  底题源自 GSM8K/SVAMP/MultiArith/ASDiv，与主池同源 → §7.2 去重为硬前置。
  > **实现口径（2026-09-20 复核）**：全量 `StandardDataset.jsonl` 产出 **2,588 answerable / 2,588
  > unanswerable**（5,200 → 5,176 配对行），审计 15/15 PASS。**"无类别补充 200"在本快照里观测不到**：
  > 该文件给全部 2,600 条 unanswerable 都标了 cat1–5，实测 category-less = 0；适配器已去掉类别
  > 守卫（无类别行会被接收、`perturbation_type` 记 None），配额 2,489 由有类别行填满。类别分布
  > 实测 cat1 834 / cat2 1,259 / cat3 273 / cat4 103 / cat5 119（§4.8 的 840/1,040/226/85/98+200
  > 是重勘的缩放估计，非本文件计数）。L3：长度启发式 0.5044、BoW-NB(support≥5) 0.3744，均 ≤0.55
  > PASS；但同一估计器在 support≥50 读 0.5750（判别 token 是缺陷词表本身 some/several/less，
  > 不是主题记忆）——落在 Q8 的 10pt 默认内、超出 5pt，`verify_umwp.py` 无条件打印该 sweep，
  > 门槛按文档预注册的 support≥5 配置读。
- **TreeCut**（D26 正负双侧入池）：`cut = ans_upstream[cutDepth-1]` → 剪掉的边必在根→答案
  路径上，**不可解性构造即证明**（`proof` 自带"N variables but M formulas"凭证）——这正是
  MiP 缺的东西；且**被剪的边已知** → 负样本可做四档诊断（gold 选项 = 被剪条件）。
  - **负样本（4,907，四档；输出按 stage 分两式，奖励统一指针契约）**：选项 = 生成器顺带产出的
    候选缺失条件（k=3，D15）——同一参数族内候选 `cut` 集合，**全部题面外**（每条都是"题面里
    缺失的条件"形态）；`has_diagnosis_label=true`。**stage 2（默认构建）**：
    `\boxed{UNSOLVABLE: <ID>}`——拒答 + 指针命中即 +1，**不跑 `certify_repaired_answer`**
    （其失败不再丢整对）。**stage 3（D29 的 `--repair-contract` 构建；D30 起不再用于 stage-3
    构建，机制保留）**：gt 加 `repair_task=true`、
    prompt 要求三段式 `\boxed{UNSOLVABLE: <ID>: <修复后的答案>}`——**奖励与 stage 2 一致：
    指针命中即 +1，修复解段不计分**（2026-10-15 决策）；`repaired_answer` 为整数字符串，由
    `certify_repaired_answer`（恢复 `deleted_sentence` 重跑 `certify_answer`）认证，失败丢整对
    （数字仍作审计与 verify 用途）；`repaired_free_text` 缺省，同样仅审计标注。
  - **正样本（500，D27 让渡后；模板 A 可解行）**：输出 `\boxed{答案}` 走 `_math_score`；prompt 带
    **占位选项块**（从题面条件中随机抽 k=3 条，换变量名或变量值生成，无正确项）——
    正负两侧选项块外观同构，模型无法靠"有选项块 ⇒ UNSOLVABLE"蒙（D18 硬约束）；
    误拒（`UNSOLVABLE` 任意形态）→ 0（与普通可解行一致）。
  - **泄题防线**：负样本选项若含"题面内条件"，则"选题面里唯一找不到的那句"直接命中 gold
    （MiP 同款捷径，§4.2 实测 95.6%）——故选项必须**全部题面外**，且过 §9 泄题启发式
    硬门槛（≤ 随机 + 5pt）。
  **坑**：可解/不可解两类天然长度不同（351.6 vs 303.6 字符），长度启发式 0.726、BoW NB 0.755，
  属结构性泄漏。**修法**：负类不取"完整题"，改取"剪掉一条非必要边"的版本（句子数/变量数/长度
  分布对齐，可解性不变）；**修完重跑 L3 落到 ≈0.5 才允许入池**（§9 硬门槛）。
  > **实现口径（2026-09-20 复核）**：① 分层网格落地为 **6 个 `(numVars, ansDepth)` 组合 × 2 theme
  > = 12 格**，不是 Q11 写的 3×3×2；正样本实测 **41–42 条/格**、负样本 408–409 条/格。
  > 即「每格 ≥100 条」只对负样本成立——它与 D27 之后的正样本配额（500/12 ≈ 41.7）算术上不相容，
  > 属 Q11 与 D26 的内部矛盾；实现按 D26 的 500 配额走，格数/每格量记在
  > `treecut_adapter.py` 的 DEVIATIONS 3。
  > ② 修复后 L3 实测：长度启发式 **0.5211**、BoW NB **0.5040**（≤ 随机 + 5pt，§9 硬门槛通过，
  > 阈值比 Q8 的 +10pt 更严）；泄题启发式实测 **0.3269**（选项全部题面外）。
  > ③ 「修法」句（剪掉一条**非必要**边）与 D26（剪掉的边**必在**根→答案路径上，即缺必要条件的
  > 构造）读法冲突：实现取 D26 读法（缺必要条件 → 四档诊断有 gold），并在 adapter 的
  > DEVIATIONS 1 记录。
- **CREPE**：两个坑——① 标签串是 `'false presupposition'`（**空格**，README 写的下划线串会
  匹配到 0 行）；真值取 train 927 条。② `presuppositions` 仅 **1.1%** 是题面逐字 span（其余是
  转述）→ exact-match 指针 gold 不成立，**只能进判断型/三档**。
  > **实现口径（2026-09-20 复核）**：去掉 KUQ（D25）后 CREPE-only 的 650 条构建上，BoW-NB 五折
  > OOF balanced-acc = **0.6075**（+0.1075 over 随机），高于 §9 的 0.60 参考线。`verify_crepe.py`
  > 把它作为 **FINDING**（打印并计入报告、不阻断）而不是静默放宽：§9 的硬门槛针对"选项集是唯一
  > 信号"的源，CREPE 两侧都**不带选项块**，且随机标签对照 0.5002 仍是硬检查、7/7 通过。
  > 原始数据目录随 D25 从 `raw/kuq_crepe` 更名为 `raw/crepe`（`fetch_raw.py`）；本机既有的重勘
  > bundle 仍在旧名下，用 `--raw-dir` / `HALLUC_CREPE_RAW_DIR` 指向即可（或重跑 fetch_raw）。

### 4.7 干扰项合成（D17）

GSM-IC 的 242 个 `sentence_template` 全部只由 `{role}`（272 个取值）/ `{number}`（55 个）参数化
 → **纯规则可重放**（该 repo 只发布两个 JSON，无 generator）。施加到其他池的可解原题上：

> **实现口径（2026-09-20 复核）**：本节的两个数字是初勘读数，实现用的是**全量重勘**值：
> 模板库取 GSM-IC 两个文件的**并集 394 条**（含全部 242 条 2step 模板，多出的来自 `mstep`；
> 「242 条」是 2step 文件的计数），占位符取值域实测 `{role}` **457** / `{number}` **58**。
> 这不改变契约（仍只由这两个占位符参数化、仍纯规则可重放），只是可用模板比初勘更多。

- **答案与 reward 完全不变**：gold 仍是原答案，走 `_math_score`；不新增判分逻辑、不新增数据源依赖。
- **难度梯度**直接沿用 GSM-IC 自带三组标注，且天然配平：`sentence_label`（in_topic 45% /
  out_topic 55%）、`role_label`（overlapped / nonoverlapped ≈50:50）、`number_label`
  （in_range / out_range ≈50:50）。
- **施加对象**（表 B 第 1 行的 400 条；由 1,400 让渡而来——D26 让 1,000 给 TreeCut 正样本，
  D27 后 TreeCut 正降至 500、本行维持 400）：
  UMWP-answerable 200 / K&K 100 / 主池数学题 100。
- **成对性硬约束**：干扰项（多一句无关条件，仍可解）与三档缺前提源（少一条必要前提，不可解）
  构成对照对，两者底题必须来自同一分布（同一 source split），否则"是哪个底题库"会变成捷径。
- `verify_distractor.py`：**答案不变断言**（合成行 `ground_truth` 与底题逐字节相同，硬断言）、
  三类标注配平 ±5pt、干扰句的 `{number}` 不参与答案推导、底题同源断言。

### 4.8 20,000 条目标构成（D18；随 D21–D27 重排）

按**契约分支**分配（表 B，reward 直接消费的轴）：

| # | 分支 | 模板 | gold | 来源与配额 | 小计 |
|---|---|---|---|---|---|
| 1 | 可解·数值 + 干扰项 | B | `\boxed{答案}` | GSM-IC 772（D27 让渡后）+ 合成干扰 400（D17；D26 让渡后） | **1,172** |
| 2 | 可解·人名-角色对（D19） | B | `\boxed{人名: 角色, …}` | K&K（仅 perturbed，D20）1,600 | **1,600** |
| 3 | 可解·判断 + 解答（两层 reward，D24/D27） | A | `\boxed{答案}` | UMWP-answerable 550 + FalseQA-answerable 928（D27） | **1,478** |
| 4 | 可解·判断（`judgment_only`） | B | `\boxed{SOLVABLE}` | CREPE-normal 250 | **250** |
| 5 | 可解·数值 + 占位选项块（D26） | A | `\boxed{答案}` | TreeCut 正样本 500（D27 让渡后） | **500** |
| 6 | 配对判断 + 解答（两层 reward，D23） | C | `\boxed{<ID>: 答案}` | SUM 6,000 | **6,000** |
| 7 | **不可解·四档诊断（stage 2 两段式，D29）** | A | `\boxed{UNSOLVABLE: <ID>}` | FalseQA-fake 928 + TreeCut 负 4,907（D26） | **5,835** |
| 8 | **不可解·三档 bare** | B | `\boxed{UNSOLVABLE}` | UMWP-unanswerable 2,489 + CREPE-FP 400 + MiP 276 | **3,165** |
| | | | | **合计** | **20,000** |

**60/40 核算**：SUM 配对行每条同时含"判断哪条可解"与"解答"两层信号，按 **0.5/0.5** 计入两侧
（UMWP-answerable / FalseQA-answerable / TreeCut 正样本是纯可解行，整条计入可解侧，不拆分）
→ 不可解 = 3,000 + 5,835 + 3,165 = **12,000（60.0%）**；可解 = 1,172 + 1,600 + 1,478 + 250 +
500 + 3,000 = **8,000（40.0%）**。

**硬约束检查（D18）**：模板 A 含可解（行 3/5）与不可解（行 7）✅；模板 B 含可解（行 1/2/4）与
不可解（行 8）✅；模板 C 自带两侧 ✅。

**stage 3 构成（D30 域平衡版；与表 B 不同的另一份 artifact）**：stage 3 幻觉侧不再复用表 B，
改为**域平衡新配比**——logic 20%（2,800）/ commonsense 20%（2,800）/ math 60%（8,400），且
**幻觉域内部**按边权（SUM 配对 0.5/0.5 计入两侧）可解/不可解 = 25/75；老域（stage-1，全可解）
占最终 train 的 60%，于是整份 artifact 为 **70/30**（0.6×1.0 + 0.4×0.25 = 0.70）。logic 的不可解
侧由 K&K **合成**（`kk_adapter.py --emit-unsolvable --kkcut-per-group 0`，D30：删一条陈述 + 2^N
枚举证书，三档裸拒，同一父组可产出多条不同删除，审计字段与 `kkcut:` 审计通道见 §0 D30）。
按 `STAGE3_QUOTA` 的 **14 个 cell**（`mix_halluc.py --stage3_mix`，合计 14,000）：

| 域 | 分支 | 模板 | 来源与配额 | 小计 |
|---|---|---|---|---|
| logic | 可解·人名-角色对（D19） | B | K&K（仅 perturbed，D20）700 | **700** |
| logic | **不可解·三档 bare（D30 kkcut 合成）** | B | K&K `kkcut:` 2,100 | **2,100** |
| math | 可解·数值 + 干扰项 | B | GSM-IC 340 + UMWP 155 + 主池 15 | **510** |
| math | 可解·数值 + 占位选项块（D26） | A | TreeCut 正样本 320 | **320** |
| math | 可解·判断 + 解答（两层，D24） | A | UMWP-answerable 170 | **170** |
| math | 配对判断 + 解答（D23，0.5/0.5） | C | SUM 2,200 | **2,200** |
| math | **不可解·四档诊断** | A | TreeCut 负 3,030 | **3,030** |
| math | **不可解·三档 bare** | B | UMWP-unanswerable 1,925 + MiP 245 | **2,170** |
| commonsense | 可解·判断 + 解答（两层，D27） | A | FalseQA-answerable 300 | **300** |
| commonsense | 可解·判断（`judgment_only`） | B | CREPE-normal 400 | **400** |
| commonsense | **不可解·四档诊断** | A | FalseQA-fake 820 | **820** |
| commonsense | **不可解·三档 bare** | B | CREPE-FP 1,280 | **1,280** |
| | | | **合计** | **14,000** |

**幻觉域 25/75 边权核算**：可解 2,400 + SUM 2,200 的半侧 1,100 = 3,500（25%）；不可解 9,400 +
1,100 = 10,500（75%）。分域：math 2,100 / 6,300、logic 700 / 2,100、commonsense 700 / 2,100
（行数 20/20/60）。`scaled_quota(14000, 0.75, base=STAGE3_QUOTA)` 逐行复现该表。

**整份 artifact 核算**：`HALLUC_TOTAL=14,000`（train 13,300 + val 700）、`OLD_DOMAIN_RATIO=0.6`
⇒ 老域 19,950 行，train 合计 33,250：老域 60% / 幻觉 40%，可解 23,275（70%）/ 不可解 9,975（30%）。

**供给侧说明**：MiP 全部严格证书供给 270（实测上限）；kkcut 需 `--kkcut-per-group 0`（旧默认每
父组 1 条只能产 490，2,100 需多删除放量）；CREPE 需 adapter `--quota-overrides`
（`solvable_judge=400,unsolvable_bare=1280`）。构建入口 `build_hallucination_stage3_data.sh`：
8 个 adapter 全量重建（kk 带 `--emit-unsolvable --kkcut-per-group 0 --seed`、crepe 带
`--quota-overrides`、其余默认契约——D29 的 `--repair-contract` 已废除，无 `A_repair` 专用模板，
stage 3 用标准模板 A/B/C）→ `halluc/built_stage3/`，构建后跑 `verify_kk.py`（坏 cut 行在构建期
FAIL 而非训练期）→ `mix_halluc.py --stage3_mix --halluc_total 14000 --halluc_unsolvable_ratio 0.75
--old_domain_ratio 0.6 --val_size 700 --strict_cells --seed 43` → `final_halluc_stage3/`。
**配对原子性、近重去重与模板平衡检查原样生效**（FalseQA 双侧同在 stage 3 池内；模板 A 含可解
umwp 170 + treecut 320 + falseqa-ans 300 与不可解 treecut 负 3,030 + falseqa-fake 820，模板 B
两侧亦齐，D18 硬约束天然满足）。
`--halluc_total` / `--halluc_unsolvable_ratio` 旋钮在 `--stage3_mix` 下按边权对 `STAGE3_QUOTA`
重缩放，默认值（14,000 / 0.75）逐行复现该表；`--strict_cells` 让短 cell 直接报错而不是跨 cell
回填，val 在填满后按 cell 比例、以原子组（D27 孪生 / D30 kkcut 兄弟）为单位切出。

> **实现口径（§4.8 行 1 的 400 条合成干扰行）**：D17 的三段施加对象（UMWP-answerable 200 / K&K 100 /
> 主池数学题 100）全部保留，但 K&K 底题产出的行**答案契约是 D19 的「人名: 角色词」**，无法落在行 1
> （行 1 的金标是 `\boxed{答案}` 走 `_math_score`）。因此按 `(branch, source)` 记账时：K&K 的 100 条
> 计入行 2 的 `(solvable_roles, halluc_logic_kk)` cell（1,600 + 100 = **1,700**），行 1 的数值 cell 为
> GSM-IC 772 + UMWP 200 + 主池 100 = 1,072。**两侧权重完全不变**（三条都是纯可解行，权重 1），
> 所以 40/60 与 20,000 总量仍与表 B 逐行一致。
>
> **实现口径（2026-09-20 复核，分支键与行号的对应）**：表 B 有 **8 行**，但 `(branch, source)`
> 记账只有 **7 个分支键**——行 5（TreeCut 正样本）与行 1 共用 `solvable_numeric`：两者的 reward
> 完全相同（模板 A 数值行、占位选项块不进判分），差别只在 `data_source`（`halluc_math_treecut`）
> 与选项块来源，因此用 `data_source` 区分、不新增分支键。schema 的分支常量表、§9 reward 矩阵
> 与 mix 的 15 个 cell 都按这 7 键 + 来源展开。

按**缺陷类型**分布（不可解 12,000；SUM 配对按 3,000 计入"混合"）：

| 缺陷类型 | 档位 | 供给 | 占比 | 承载源 |
|---|---|---:|---:|---|
| 缺一条必要条件（可指认，生成器 `cut` 已知） | 四档 | 4,907 | 40.9% | TreeCut 负（D26） |
| 混合（SUM 配对判断层） | — | 3,000 | 25.0% | SUM 6,000 对 |
| 缺一条必要条件（题面不可见） | 三档 | 1,116 | 9.3% | UMWP-cat1 840 / MiP 276 |
| 关键信息歧义 | 三档 | 1,040 | 8.7% | UMWP-cat2 |
| 假前提（可指认，替换对） | 四档 | 928 | 7.7% | FalseQA-fake |
| 假前提（不可指认） | 三档 | 400 | 3.3% | CREPE-FP 400 |
| 前提不现实·自相矛盾 | 三档 | 226 | 1.9% | UMWP-cat3 |
| 问题缺失 | 三档 | 98 | 0.8% | UMWP-cat5 |
| 无关·未定义实体 | 三档 | 85 | 0.7% | UMWP-cat4 |
| 无类别标注（统计口径外） | 三档 | 200 | 1.7% | UMWP-unanswerable 补充 |
| **合计** | | **12,000** | **100%** | |

> UMWP-unanswerable 入池 2,489 = 有类别标注实测 2,289（cat1–cat5）+ 无类别补充 200
> （KUQ 移出后不可解侧的补位，D25）；cat 分类仅作缺陷类型统计（D24）。

⚠️ 两处结构性后果，待确认（§12 Q13）：
① **TreeCut 占不可解侧的 40.9%**（4,907/12,000；占四档的 84.1%），而其 L3 修复尚未验证
（现状 0.755 不合格）——修复失败时该配额由 SUM 配对上调补位（P: 6,000→8,000，两侧配额按
§4.8 公式重算）或总量下调；
② 类型分布偏向"缺前提"（可指认 40.9% + 不可见 9.3% = 50.2%）；四档诊断 = **FalseQA 928 +
TreeCut 负 4,907**（D24 后 UMWP 可见缺陷类退化为三档、D26 后 TreeCut 负样本四档化），
"歧义/不现实/问题缺失/无关实体"四类合计 12.1% 且均不可诊断。

**规模与步数换算**：幻觉域 20,000 条在 `--old_domain_ratio 0.7` 下 → 阶段二 parquet
`N = 20,000 / 0.3 = 66,667`：

| `total_epochs` | 步数 | 每条曝光 |
|---:|---:|---:|
| 1 | 260 | 1 |
| 3（建议） | 781 | 3 |
| 10 | 2,604 | 10 |

### 4.9 被排除 / 降级的源

- **AbstentionBench**（D9）：聚合体含 FalseQA/UMWP（重复）与 GPQA/GSM8K/MMLU（评估集污染）→
  降级为评估套件。
- **SQuAD 2.0**：判"不可解"需 NLI，L1 证书不成立 → 排除。
- **GSM-DC**：仅 6,300 样例、合成风格生硬 → 排除。
- **CoCoNot**：本轮不纳入（待核 license 与污染）。
- **MiP 可解侧**（634 条）：与主池同源、去重成本高于收益 → 不进池（§12 Q7）。
- **FalseQA `label=0`**：D22 曾排除；**D27 已复活入池**（两层 reward + 占位替换对选项块，§4.3）。
- **KUQ**（D25）：无可验证答案、只能判断是否 unknown，判断信号缺答案层约束、模型易瞎猜 →
  整体不入池。

---

## 5. 统一答案契约与 Prompt 模板

### 5.1 契约（D3 / D19 / D21 / D23 / D24 / D27 / D28 / D29）

| 情形 | 模型应输出 |
|---|---|
| 可解（数值/表达式：GSM-IC、合成干扰、主池数学题） | 最后一行 `\boxed{<数值/表达式>}` |
| 可解（K&K：逐人角色判定，D19） | `\boxed{<人名>: <角色词>, <人名>: <角色词>, …}`（如 `\boxed{Oliver: angel, Ethan: devil}`）；**顺序不限**；人名与角色词都用题面原词 |
| 可解·**判断+解答**（`two_layer`：UMWP-answerable，D24） | `\boxed{<数值答案>}`——给出答案即判定"可解" |
| 可解·**判断+解答**（`solvable_answer`：FalseQA-answerable，D27） | `\boxed{<答案>}`（自由文本，与 gold 归一化精确匹配；AND 门控，`\boxed{SOLVABLE}` → 0） |
| 可解但**只判可答性**（`judgment_only`：CREPE-normal） | `\boxed{SOLVABLE}` |
| 配对判断（SUM，D23） | `\boxed{<A或B>: <数值答案>}`（如 `\boxed{A: 42}`） |
| 不可解且有诊断标签（FalseQA-fake 替换对 / TreeCut 负·缺失条件，D26）——**stage 2**（D29） | `\boxed{UNSOLVABLE: <选项ID>}`（如 `\boxed{UNSOLVABLE: B}`；指针命中即 +1，收到三段式输出时第三段被解析但忽略） |
| 不可解且有诊断标签——**stage 3 修复行**（D29 `--repair-contract` 产物；D30 起 stage-3 构建不再产出，键保留供 pre-D29 artifact 路由） | `\boxed{UNSOLVABLE: <选项ID>: <修复后的答案>}`（如 `\boxed{UNSOLVABLE: B: 42}`；**奖励与 stage 2 一致：指针命中即 +1，修复解段解析但不计分**——2026-10-15 决策） |
| 不可解且无诊断标签（MiP / UMWP-unanswerable / CREPE-FP） | `\boxed{UNSOLVABLE}` |

- **只约束最终落点，不约束推理过程/位置**：reward 只在"最后一个 `\boxed{}`"上判。
- **bare `\boxed{UNSOLVABLE}` 的得分是分支相关的**（§6 最易写错的一处）：三档源上 +1，
  四档源上 0（识别对了但没给诊断）。
- **stage-2 四档行指针命中即满分**（D29）：`\boxed{UNSOLVABLE: <ID>}` 命中 gold → +1；
  模型若输出 stage-3 的三段式形态，第三段被 `extract_final` 解析但**不参与判分**（不奖不罚）。
- **stage-3 修复行的奖励与 stage 2 完全一致**（2026-10-15 决策，退役 D28 修复层）：
  `repair_task` 行同样指针命中即 +1——`\boxed{UNSOLVABLE: <ID>}` 与
  `\boxed{UNSOLVABLE: <ID>: <任意文本>}` 等价得分；`repaired_answer` /
  `repaired_free_text` 仅作审计字段保留。退役动机：`norm_match`（FalseQA 自由文本）/
  `math_match`（TreeCut 数值）修复匹配是主要误判（漏判）源，指针契约无此问题。

**⚠️ K&K 角色词归一化（强制；漏掉则 reward 反向）**

`solution` 是**规范语义**（`True` = 真话者）；表面词由该行 `knight_knave` 映射给出。
`flip_role` 与 `random_pair` 真的换词，合计 **13,800 行 = 28.7%**（§4.1）。因此：

```
role_words = (knight_knave['knight'], knight_knave['knave'])   # 该行自己的 [真话者词, 说谎者词]
gold       : answer = {人名: 表面角色词}      # adapter 由 names + solution + knight_knave 合成
verifier   : 把模型输出的「人名: 角色」对按 role_words 逐人映射回规范 bool, 与 gold 逐人比对
```

- 例（`flip_role`）：题面写作 "Knaves always tell the truth, and knights always lie"，
  正确回答 `\boxed{Oliver: knave, Ethan: knave}` → 逐人映射回规范 `{Oliver: True, Ethan: True}`。
  若硬编码 `K = knight`，这条会被判**错**、错误答案会被判**对**。
- `role_words` 必须由 adapter 写进 `reward_model.ground_truth`，reward 侧不得从题面文本里猜。
- 带人名后 reward 按**人名集合**校验：漏人/多人/人名不在 gold 集合/旧纯序列形态 → 0（§6）。

**⚠️ 三档源不附选项块（否则泄题复现）**

- MiP / UMWP-unanswerable / CREPE 的不可解行 `has_diagnosis_label=false` →
  **prompt 里不得出现选项块**（模板 B）。MiP 一旦附选项，"选题面里唯一找不到的那句"就在
  95.6% 的样本上直接命中（§4.2）。
- 若仍收到 `\boxed{UNSOLVABLE: A}` 这种带选项 ID 的输出：三档源按 spec §3.1"识别出不可解即满分"
  → **同样 +1**（选项 ID 无意义）；四档源则 → 0（没给对诊断）。

**⚠️ 四档源的两种选项形态 + 可解行的占位形态（D26/D27）**

- FalseQA 不可解侧（D21）：选项块放 `假前提片段 -> 替换词` 对，左项三项全同、右项全部题面外
  同类等长（§4.3）；**answerable 侧（D27）放占位替换对**（左项题面内、右项题面外同类等长，
  无正确项）——两侧外观同构。
- TreeCut 负（D26）：选项块放**候选缺失条件**（每条都是"题面里缺失的条件"形态，全部题面外），
  gold = 生成器被剪的边；正样本则带**占位选项块**（题面条件换变量名/变量值，无正确项）——
  正负选项块外观同构（§4.6）。
- UMWP 可见缺陷类的同题等长跨度选项已随 D24 废弃；跨题文本作干扰项可被"主题重叠最大"
  蒙对（FalseQA 上实测 99.1%），一律禁用。

### 5.2 Prompt 模板（追加在题面之后）

**模板 A（附选项块 + 显式判断）—— FalseQA 双侧（label=1 四档 D21 / answerable 两层 D27）/ UMWP-answerable（D24）/ TreeCut 正负（D26）：**

FalseQA 版（选项为替换对，D21；**label=0/1 双侧共用同一文案**，D27）：

```
<题面…>

请判断该问题是否可解，并在最后一行给出结论：

- 若前提成立、题目可以作答，最后一行输出：\boxed{<答案>}
- 若前提不成立 / 条件相互矛盾，最后一行输出：\boxed{UNSOLVABLE: <选项ID>}
  其中 <选项ID> 是下列"前提替换"候选中，能让该问题变为可解的那一项。

选项：
A. man -> women
B. man -> child
C. man -> teacher
```

- 模板 A 内**可解与不可解子集必须共用同一套外观**（都先问是否可解、都附选项块、选项个数同为
  k=3），否则"看到选项块 ⇒ 输出 UNSOLVABLE"是 100% 捷径（D18 硬约束）。模板 A 的不可解 marker
  是**两段式** `\boxed{UNSOLVABLE: <选项ID>}`（D29：stage 2 回退后的契约；三段式形态只属于
  `A_repair`，见下）。
  > **实现口径（2026-09-20 复核）**：为满足这条「共用同一套外观」，落盘文案把上面那行
  > `其中 <选项ID> 是下列"前提替换"候选中…` 统一写成 `其中 <选项ID> 是下面候选中，能让该问题变为可解的那一项。`
  > ——「前提替换」是 FalseQA-fake 专用的措辞，写在 UMWP/FalseQA-answerable/TreeCut 的**占位**选项块
  > 上就是错的（那里没有 premise replacement 这回事），而模板 A 必须**逐字同构**才能防格式记忆。
  > 这是本节初稿文字与 D18/D26/D27 的同构要求之间的取舍：以同构为准，措辞取两侧都成立的版本。
  > 另：`schema.question_of(row)` 按模板指令块的前缀把**题面**切出来，供 §7.2 去重使用。
- UMWP-answerable（D24）/ FalseQA-answerable（D27）行：复用模板 A 外观——选项块由 adapter
  用题面外同类等长替换对**占位**（无正确项），仅为满足 D18；可解时输出 `\boxed{答案}`，
  两层 reward 见 §6（AND 门控：输出选项 ID / `UNSOLVABLE` / `SOLVABLE` / 答错 → 0）。
- TreeCut 行（D26）：**负样本**选项块为候选缺失条件（全部题面外，gold = 被剪的边）；
  **正样本**选项块为题面条件换变量名/变量值的**占位**项（无正确项），可解时输出
  `\boxed{答案}`，误拒（`UNSOLVABLE` 任意形态）→ 0（与普通可解行一致）。正负两侧外观同构。
- 选项位置按 seed 打乱；reward 只比选项 ID，不比文本。

**模板 A_repair（D29 引入；D30 起 stage-3 构建不再使用）—— FalseQA-repair / TreeCut-repair 四档修复行：**

与模板 A 同构（同问"是否可解"、同附 k=3 选项块），唯一差异在不可解 marker 是**三段式**：

```
<题面…>

请判断该问题是否可解，并在最后一行给出结论：

- 若前提成立、题目可以作答，最后一行输出：\boxed{<答案>}
- 若前提不成立 / 条件相互矛盾，最后一行输出：\boxed{UNSOLVABLE: <选项ID>: <修复后的答案>}
  其中 <选项ID> 是下面候选中，能让该问题变为可解的那一项；<修复后的答案> 是按该项
  补足条件后，该问题的正确答案。

选项：
A. man -> women
B. man -> child
C. man -> teacher
```

- `A_repair` 与 gt 键 `repair_task=true` **1:1 绑定**（schema `validate_row` 双向强校验：
  `repair_task` 行必须用 `A_repair`，`A_repair` 行必须带 `repair_task`），且仅限四档不可解行。
- 两个 marker 互为非子串（两段式在 `>` 后即 `}`，三段式中间有 `: <修复后的答案>`），
  `verify_falseqa.py` / `verify_treecut.py` 按行模板推导预期 marker，同一脚本可审计两种
  artifact，无需新增旗标。
- stage 3 的幻觉侧只有 `A_repair` 一个模板、只有不可解行——D18 的模板平衡硬约束对它结构性
  豁免（§4.8 stage 3 构成）。**D30 起 stage 3 换域平衡表、用标准模板 A/B/C，不再产出
  `A_repair` 行**；模板与 schema 校验保留，供 pre-D29 artifact 审计与未来实验。

**模板 B（不附选项块）—— 所有三档不可解源 + 可解数值/角色词/判断型：**

```
<题面…>

若题目给出的信息不足以确定唯一答案，或条件相互矛盾，请在最后一行输出：
\boxed{UNSOLVABLE}

否则，请 step by step 推理，并在最后一行输出 \boxed{你的最终答案}。
```

- K&K 加一句："请逐人给出结论，最后一行输出 `\boxed{人名: 角色词, 人名: 角色词, …}`，
  人名须与题面拼写一致，角色词用题面原词（若题目称其为 saint/sinner，就用这两个词）。"
  ——不能写死 knight/knave（§4.1）；没有人名的纯角色词序列判 0（§6 `kk_match`）。
- 判断型（CREPE-normal）加一句："若该题可以作答，最后一行输出 `\boxed{SOLVABLE}`。"
- 数学类沿用现有 `\boxed{}` 收尾约定，与 `DESIGN.md` §1 一致。

**模板 C（配对判断 + 解答）—— SUM 专用（D23）：**

```
下面给出两个问题，其中一个可以求解，另一个因缺少条件或条件矛盾而无法求解。

问题 A：<question_1>
问题 B：<question_2>

请先判断哪个问题可解，再解答该问题。最后一行输出：
\boxed{<可解问题的编号>: <最终答案>}
例如 \boxed{A: 42}。
```

- A/B 顺序由 adapter 按行 seed 随机（answerable 落在 A 与 B 各 50%），落盘即固定。
- 模板归属由 `ground_truth` 的字段决定（`pair_task` / `has_diagnosis_label`），adapter 落盘时
  就固定进 `prompt`，reward 侧不再判模板。

---

## 6. Reward 设计（只保留 spec §3.1 主 reward）

新文件 `reward/hallucination_compute_score.py`，入口函数名仍为 `compute_score`：

```
compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if not data_source.startswith("halluc_"):
        return base_compute_score(...)            # 委托现有文件，逐字节兼容老四域/if
    if not format_ok(solution_str):
        return {"score": 0.0}
    gt = json.loads(ground_truth)                # 解析失败 → log + 0（fail closed）
    if data_source.startswith("halluc_math_gsmic"):
        return {"score": _math_score(solution_str, gt.answer)}     # GSM-IC：常规数学校验
    status, answer, option = extract_final(solution_str)
        # status ∈ {"answer", "solvable_marker", "unsolvable_option", "unsolvable_bare", "none"}
        # 复用现有 _extract_boxed() 取最后一个 \boxed{}，再匹配 UNSOLVABLE / SOLVABLE 形态
        # D28：unsolvable_option 时 boxed 内容形如 "UNSOLVABLE: B: 42"，
        # 解析为 option="B"、answer="42"（正则 ^([A-Za-z])\s*[:：]\s*(.+)$，DOTALL）

    if gt.get("pair_task"):                   # SUM 配对判断（D23）：\boxed{ID: 答案}
        sid, sans = parse_pair(answer)        # 正则 ^([AB])\s*[:：]\s*(.+)$，作用于 boxed 内容
        if sid is None or sid != gt["answerable_id"]:
            return 0.0                        # 判断层失败 → 答案层不给分
        return 1.0 if math_match(sans, gt["answer"]) else 0.0   # AND 门控：判断层单独不值分

    if gt.solvable:
        if gt.get("two_layer"):           # UMWP-answerable 两层（D24）：与 SUM pair_task 对称
            if status != "answer":                         return 0.0   # 拒答 → 判断层失败
            return 1.0 if math_match(answer, gt.answer) else 0.0        # AND 门控：判对答对才 +1
        if gt.get("solvable_answer"):     # FalseQA-answerable 两层（D27）：自由文本答案
            if status != "answer":                           return 0.0   # 拒答/误拒/SOLVABLE → 0
            return 1.0 if norm_match(answer, gt.answer) else 0.0        # AND 门控：判对答对才 +1
        if gt.get("judgment_only"):       # 判断型可解（CREPE-normal）
            if status == "solvable_marker":                      return 1.0
            if status in {"unsolvable_bare", "unsolvable_option"}: return -1.0   # 误拒
            return 0.0                    # 没判断（给了普通答案/无 \boxed{}）
        if gt.get("role_words"):          # K&K：「人名: 角色」对，先做词表归一化（§5.1，D19）
            return 1.0 if kk_match(answer, gt["role_words"], gt["answer"]) else 0.0
        return 1.0 if (status == "answer" and math_match(answer, gt.answer)) else 0.0
    else:
        if status not in {"unsolvable_bare","unsolvable_option"}:
            return 0.0                    # 编造/无 box/其它形态 → 0（不可解分支不出负分）
        if not gt.has_diagnosis_label:    return 1.0    # 三档：识别出不可解即满分
        # 四档分支：「拒答 + 指针」两层；修复层已退役（2026-10-15），repair 行同指针契约
        if status != "unsolvable_option":              return 0.0   # bare / 无 box → 0
        if option != gt.correct_option_id:             return 0.0   # 指针错 → 0
        return 1.0
        # 指针命中即 +1；三段式输出的第三段已被 extract_final 解析，此处忽略
        # （repaired_answer / repaired_free_text 仅作审计，不再路由匹配器）
        # 识别出不可解但诊断错误/未给诊断 → 0
```

要点：

- **主 reward 取值 `{+1, 0, -1}`**（SUM 配对 / UMWP-answerable / FalseQA-answerable 两层任务
  经 AND 门控后同为 `{+1, 0}`——判断层单独不值分，判断与答案都对才 +1），与 spec §3.1 一致。
  ⚠️ 量纲风险：现有其它域是 `{0,1}`，`-1` 在组内 std 归一化下可能被放大；退化组由 DAPO
  `filter_groups` 屏蔽（D5）。上线时按 §10 观察负 advantage 分布。
- `math_match` 复用现有数学匹配路径（`_math_score` / math_verify + math_dapo fallback），
  不新写比较逻辑。GSM-IC 与合成干扰项直接委托 `_math_score`（D17：gold 不变）。
- `kk_match`（K&K 专用，D19）：`gt["answer"]` 是 **name→表面角色词 mapping**。两步：
  1. **解析**：从 boxed 内容解析 `人名: 角色` 对（分隔符接受逗号/分号/换行；人名与角色词
     均大小写不敏感、去首尾空白；角色词先 strip 冠词 `a/an/the`）；
  2. **比对**：每个角色词按该行 `role_words` 映射回规范 bool（`role_words[0]` → 真话者、
     `role_words[1]` → 说谎者），与 gold 逐人比对。**人名集合必须与 gold 恰好相等**且
     全部角色一致才 +1。
  以下一律 → 0：解析不出任何 `人名: 角色` 结构（含旧纯序列形态）、人名不在 gold 集合 /
  缺人 / 重复人名、角色词在 `role_words` 词表外、空答案。
  **不要**在 K&K 这条路上复用 `math_match`。判定依据是 `ground_truth` 里是否存在
  `role_words` 键，而不是 `data_source`（便于单测直接构造）。
- `parse_pair`（SUM 专用，D23）：作用于 boxed 内容，正则 `^([AB])\s*[:：]\s*(.+)$`；
  答案部分交给 `math_match`（gold 是数值）。判定依据是 `pair_task` 键。
- 抽取全部集中在 `extract_final`，只取最后一个 `\boxed{}`；无 `\boxed{}` → `status="none"` → 0。
- **`two_layer` 分支**（UMWP-answerable，D24）：两层 reward 经 AND 门控，结构上与 SUM 的
  `pair_task` 对称——先判"可解性"（`status=="answer"` 即给出 `\boxed{答案}`；输出 `SOLVABLE` /
  `UNSOLVABLE` / 选项 ID / 无 box → 0），再解答案（`math_match` 对 → **+1**，错 → **0**；
  判断层单独不值分）。判定依据是 `two_layer` 键。
- **`solvable_answer` 分支**（FalseQA-answerable，D27）：与 `two_layer` 同构，但答案层是
  **自由文本**——`norm_match`（小写/去标点/去冠词/strip 后精确匹配；复用 gsm8k 系的归一化
  helper，不新写）对 → **+1**，错 → **0**（AND 门控）。`\boxed{SOLVABLE}` → **0**（只判断未
  解答不给分）；误拒（`UNSOLVABLE` 任意形态）→ **0**（与普通可解行一致，不引入 −1 误拒）。
  判定依据是 `solvable_answer` 键。答案层匹配噪声：AND 门控下同义 paraphrase 会丢整分 →
  入池前人工抽检 N=50，误判率 >10% 只保留短答子集（§12 Q17）。
- **三档 bare 分支**（`has_diagnosis_label=false`）：`status ∈ {"unsolvable_bare","unsolvable_option"}`
  即 +1——MiP/UMWP-unanswerable/CREPE 的 prompt 里没有选项，选项 ID 无意义，
  spec §3.1「识别出不可解即满分」。
- **四档分支**（`has_diagnosis_label=true`：FalseQA-fake / TreeCut 负，D26）：指针契约统一覆盖
  全部模板——`status == "unsolvable_option"` **且** `option == gt.correct_option_id` 才 +1，
  否则 0（bare `\boxed{UNSOLVABLE}` → **0**，识别对了但没给诊断）。**修复层已退役**
  （2026-10-15 决策）：`repair_task` 行（D29 的 `A_repair` 产物；**D30 起 stage-3 构建不再
  产出该键**）与 pre-D29 legacy artifact
  （带 `repaired_answer` 无 `repair_task`）的奖励与 stage 2 完全一致——指针命中即 +1，
  第三段修复解由 `extract_final` 解析但不参与判分；`repaired_answer` /
  `repaired_free_text` 仅作审计字段，reward 不再读取。reward 侧只比选项 ID，不比选项文本，
  也不比修复解；判定依据是 ground_truth 的键而不是 `data_source`——全部四档行共用同一个
  reward 文件的同一条分支。
- **TreeCut 正样本**走可解主分支（`status == "answer"` 且 `math_match` → +1；误拒 → 0，
  与普通可解行一致），prompt 的占位选项块不进 reward 判定（D26）。
- **`judgment_only` 分支**（CREPE-normal）：只认 `\boxed{SOLVABLE}`；**误拒给 −1**——这是
  全矩阵唯一的 −1（不可解分支已收敛为 `{+1, 0}`：编造 / 弃答 / 其它形态一律 0）。该分支不参与
  `math_match`。
- 任何异常（JSON 坏、抽取越界等）→ log + `{"score": 0.0}`，不抛出（fail-closed）。

> **实现口径（2026-09-20 复核，四则 reward 实现注记）**：
> ① **`halluc_math_gsmic` 没有专门分支**（与本文伪代码 §6:647-648 的字面写法不同）：
> `_math_score` 在缺 `math_verify` 的进程里会**静默返回 0.0**——`verl.utils.reward_score.math_verify`
> 自己吞掉 `ImportError` 并打日志，`_math_score` 等待的 `except ImportError` 因此不可达。本机实测
> `_math_score('\boxed{42}','42') = 0.0` 而通用数值分支 `math_match('42','42') = True`——若照字面
> 短路，本节整个 GSM-IC 源的 reward 会全 0 且不报错。代码走通用数值分支（`math_match` →
> `_math_score` → math_dapo → 阶段一文本比较），对 GSM-IC 的 `{+1, 0}` 语义与本节一致。
> ② **`math_match` 的三层**：`_math_score` → math_dapo（strict_box 两档）→ 阶段一
> `logic_answer_match`。第 1 层就是本节的「`_math_score` / math_verify」，第 2 层是 DESIGN.md §6
> 承诺的 math_dapo fallback；第 3 层是**只增信**的兜底（实测 `042`/`42.000`/`(42)`/`42.`/
> `2` vs `2.0` 判对，`4.20` vs `42`、`1,2` vs `12`、`12` vs `1.2` 判错），用于 `math_verify` 池
> 损坏时仍能评普通整数/表达式；`1/2` vs `0.5` 这类分数-小数等价在第 1 层不可用时判 0（假阴性，
> 宁缺勿错）。三层均只加信，故不严于本节路径。
> ③ **`pair_task` 嵌在 `if solvable:` 之内**（本文伪代码把它放在顶层）：SUM 行恒 `solvable=true`，
> 现网无差异；嵌套读法更严——即使出现 `pair_task=true` 且 `solvable=false` 的畸形行，也不会拿到
> 答案层分数，而是按不可解分支的「编造 → 0」处理。
> ④ **`norm_match` 自写标点/冠词归一化**（本节说「复用 gsm8k 系 helper」）：阶段一 `gsm8k.py`
> 并无标点/冠词 helper，实际复用其 `_normalise_text`（大小写/空白）+ 本文件的去标点/去冠词两式；
> 语义与本节「小写/去标点/去冠词/strip 后精确匹配」一致。另：`parse_ground_truth` 同时接受
> JSON 字符串（§3 约定）与已解析的 dict（便于单测），四档选项 ID 比较对大小写不敏感。

---

## 7. 混合与阶段二接入

### 7.1 `mix_halluc.py`

输入：阶段一最终 mix 的 `train.parquet` / `val.parquet`（原四域/if 行原样保留）+ 各 adapter 的
built parquet。输出：阶段二 `final_halluc/train.parquet` + `val.parquet` + `mix_stats.json`。

旋钮（都手动配，不退火，D7）：

- `--old_domain_ratio`：原四域（+ 可选 if）在总量中的占比；幻觉域拿 `1 - old_domain_ratio`，
  默认 **0.7**。
- `--halluc_total`：幻觉域总量，默认 **20000**（D18）。
- `--halluc_unsolvable_ratio`：幻觉域内不可解占比，默认 **0.6**（SUM 配对行按 0.5/0.5 计入两侧，
  §4.8）。
- `--stage3_mix`（D30，stage 3）：幻觉侧配额基表换成 **`STAGE3_QUOTA`**（域平衡 14 cell，
  §4.8 stage 3 构成）——logic 20% / commonsense 20% / math 60%，幻觉域内边权 25/75，老域占
  train 60% ⇒ 全局 70/30；该模式下 `--halluc_total` 默认 **14000**、`--halluc_unsolvable_ratio`
  默认 **0.75**、`--old_domain_ratio` 默认 **0.6**。基表含 `kkcut:` 合成 cell（消费
  `kk_adapter.py --emit-unsolvable --kkcut-per-group 0` 的产物）。旧 `--repair_mode`
  （D29）与 `REPAIR_QUOTA` 已**删除**：修复层退役后 stage 3 不再消费 `--repair-contract`
  产物，也不再有"纯不可解"强制与 `A_repair` 模板豁免——配对原子性 / 去重 / 模板平衡检查
  与 stage 2 同一套原样生效。`--strict_cells` 让短 cell 直接报错（不跨 cell 回填），val 在
  fill 之后按 cell 比例切出（`carve_val_from_filled`），`mix_stats.json` 记
  `config.stage3_mix` / `config.strict_cells` 与 `achieved_ratios`。
- 各来源配额（K&K 1,600 / SUM 6,000 / FalseQA 928×2 / …）按 §4.8 表 B 写成 adapter 常量（§12 Q12），
  不做旋钮；K&K 组内去重在 adapter 完成（§4.1），mix 只按比例抽。

实现细节沿用 `mix.py`：**先切 val 再采 train**、按比例缩放、`total_size` 不足则无放回/告警、
`rng.shuffle`、重排 `extra_info.index`、写 `mix_stats.json`（按 ability/source 分桶）。
`extra_info.split` 在 val 行写 `"val"`。

> **实现口径（2026-09-20 复核）**：`--stage1_path` 传**目录**时按 `train.parquet` / `val.parquet`
> 分开读（其余 parquet 视作 train 分片），两侧各有去向：train 行按 `--old_domain_ratio` 采样进
> 阶段二 train，**val 行原样进阶段二 val**（`mix_stats.stage1_val_rows_kept_in_val` 记账）。
> 这一条是本行「原四域/if 行原样保留」的落地方式，也修掉了递归读目录的隐患——`read_parquet_rows`
> 走 `**/*.parquet`，直接指向阶段一输出目录会把**阶段一 val 行折进阶段二 train**，正是 §7.2 要防的
> 泄漏。单文件路径视为 train-only（不产生旧域 val 行）。Q2 的 `--val_size 256` 只描述**幻觉域**
> 的 val 配额；旧域 val 行是额外追加，故 `total_val` 可能大于 256。

### 7.2 去重与去污染

- 新数据与阶段一训练 mix 做**池内近重去重**（复用 `dedup.py` 的 MinHash 思路）——**硬前置**：
  MiP（GSM8K/SVAMP/MATH）、UMWP（GSM8K/SVAMP/MultiArith/ASDiv）、SUM（源自 DeepScaleR）
  均与 Big-Math 主池同源。
- 对评估集（MATH-500 / AIME / GPQA…）的 n-gram + embedding 去污染沿用 `decontaminate.py`；
  若验证集 parquet 未就绪，至少先跑 n-gram 一级并在报告里标注。

> **实现口径（2026-09-20 复核）**：两条去重都在 `mix_halluc.py` 内落地，不再依赖调用方另行开跑。
> ① **精确文本**（`dedup_against`）与 ② **MinHash 近重**（`dedup.py:minhash_near_dedup`，
> `--minhash_threshold` 默认 **0.6**，0 关闭）都以**本次抽中的阶段一行**为锚集合：锚行先进
> 贪心遍历因而必被保留，命中的幻觉行被丢弃；比较对象**只跨池、不池内**——D27 的 FalseQA 双胞胎
> （label=0/1 只差一个替换片段，Jaccard 远高于阈值）与 UMWP 的配对行必须同时存活，所以被丢的
> 行若带 `pair_id`，它的孪生行一并退出（`mix_stats.dropped_minhash_pair_twins` 记账）。
> 相似度算在**题面**上：`schema.question_of()` 先剥掉模板指令块（同一模板所有行共享该块，留着会
> 虚高无关行的相似度、稀释真重复的相似度）；模板 C 的配对 prompt 返回整段。
> ③ 评估集去污染（`decontaminate.py` 的 n-gram/embedding 一级）**仍需调用方单独跑**——它要读
> 评估集 parquet，不在本 mix 的输入范围内；未跑时按本行要求「在报告里标注」。

### 7.3 阶段二启动方式

- 数据：`TRAIN_FILES="['…/final_halluc/train.parquet']" VAL_FILES="['…/final_halluc/val.parquet']"`。
- 续训：`RESUME_MODE=resume_path RESUME_PATH=<阶段一 ckpt 目录>`。
- reward：现有 run 脚本把 `reward.custom_reward_function.path` 硬编码为 `compute_score.py`，
  切换需要额外一小步——默认方案：给现有脚本加一个默认值不变的 `REWARD_PATH` env 开关（§12 Q3）。

### 7.4 阶段三启动方式（D30）

- 数据：`build_hallucination_stage3_data.sh` 一键构建——8 个 adapter 全量重建输出
  `halluc/built_stage3/`（kk 带 `--emit-unsolvable --kkcut-per-group 0 --seed`、crepe 带
  `--quota-overrides solvable_judge=400,unsolvable_bare=1280 --seed`，其余默认契约；D29 的
  `--repair-contract` 已废除），构建后跑 `verify_kk.py` 把坏 cut 行挡在构建期；随后
  `mix_halluc.py --stage3_mix --halluc_total 14000 --halluc_unsolvable_ratio 0.75
  --old_domain_ratio 0.6 --code_share 0.24 --val_size 700 --strict_cells --seed 43`
  产出 `final_halluc_stage3/`
  （`SEED` 同时是 mix 种子、kkcut 删除顺序种子与 crepe 配额子采样种子——43 ≠ stage-2 的
  默认 0，stage-3 的 crepe 大样本不会把 stage-2 行整批含入；`--val_size` 700 是幻觉侧的
  5%，val 从填满的 cell 里按比例切出）。
  训练脚本 `run_qwen3_4b_hallucination_stage3_fully_async.sh` 的 `DATA_DIR` 默认指向它。
- 续训：`RESUME_PATH=<阶段二 ckpt 目录>`（stage 3 不从头训：拒答 + 指针 + 判断行为由
  stage 2 建立，本阶段在域平衡配比下保持这些行为并适应 kkcut 新形态）。
- reward：**与 stage 2 共用** `reward/hallucination_compute_score.py`——按 gt 键路由（§6），
  同一文件服务两个 stage，无需切换。

### 7.5 阶段四启动方式（D31）

- 数据：`build_hallucination_stage4_data.sh` 一键构建——沿用 stage-3 的 8 个 adapter 与
  `STAGE3_QUOTA`（`SEED=44` 轮换 kkcut 删除顺序与 CREPE 子采样），另建 IF 池与新鲜化老域：
  1. `if_adapter.py --out halluc/built_stage4/if.parquet`：过滤 reward 不可评估的 instruction id、
     prompt 精确去重、`--exclude-path` 剔除 stage 1/2/3 已训练 prompt，写入
     `{"constraints":[…],"grading_mode":"binary"}`（`IF_GRADING_MODE=fraction` 可切部分分）；
  2. `EXCLUDE_PATHS`（默认 `final` / `final_halluc` / `final_halluc_stage3`）用于老域回放池与 IF 池；
  3. `INCLUDE_OLD_POOLS=1` 时 `OLD_POOL_PATHS`（默认 `math/code/logic/stem` 的 `train_*.parquet`）
     并入回放供给，`--old_pool_path` 的 `val.parquet` 跳过；
  4. `mix_halluc.py --stage3_mix --strict_cells --if_ratio 0.15 --if_path … --exclude_path …
     --old_pool_path …` → `final_halluc_stage4/`。IF 是增量域：`if = 0.15/0.85 × (幻觉 train + 老域)`，
     所以 stage-3 幻觉域绝对规模与 60/40 结构不变，`--code_share 0.24` 按**最终**总量计算；
     构建末尾断言 IF 实际占比 ≥ 请求值 −1pt。
- 规模（D31-only 口径，`BBH_RATIO=0`）：非增量基数 33,250（幻觉 13,300 + 老域 19,950）
  + IF 5,868 = **39,118**，IF **15.0%**；val ≈ 700（幻觉）+ 255（stage-1 val）+ 169（IF）≈ 1,124，
  IF 仍约 15%。`--val_size` 仍是幻觉侧配额，IF val 默认按同一公式
  `ratio/(1-ratio) × (幻觉 val + 老域 val)` 从 IF 池**先切后抽** train，因此 val 与 train 不会共用 prompt。
  D32 默认 `BBH_RATIO=0.10` 时，IF 与 BBH 从同一个非增量基数 N 计算：IF=6,650、BBH=4,433，
  最终 train **44,333**（IF 15.0% / BBH 10.0% / 非增量 75.0%），见 §7.6。
- 续训：`RESUME_PATH=<阶段三 ckpt 目录> bash run_qwen3_4b_hallucination_stage4_fully_async.sh`
  （D31+D32 薄封装，DATA_DIR 默认 `final_halluc_stage4`，其余沿用 stage-3 recipe）。
  > **D34 更新**：本节的老域/幻觉域 60/40 与 `--old_domain_ratio 0.6` 是 D31+D32 口径；
  > D34 起默认 `--old_domain_ratio 0.5`（老域 30%），并新增 hard-math 增量域，
  > 最终 total 仍是 44,333，配比见 §7.8。
- reward：仍是 `reward/hallucination_compute_score.py`；`if_nemotron` 行按 `data_source` 前缀
  委托给 `compute_score.py::_if_score`，约束只检查 `</think>` 之后的可见回答。
  **注意**：binary 模式下 IF 组的正确率通常很低（最多 5 条约束全部通过），组内奖励容易全 0；
  fully-async 不过滤零方差组，若 IF 组长期零方差，优先改用 `IF_GRADING_MODE=fraction` 或检查
  约束覆盖率报告（`built_stage4/if_report.json`），而不是直接调大 `IF_RATIO`。

### 7.6 阶段四 BBH 薄弱项补强（D32）

- 背景：stage-3 checkpoint 在 BBH 上的分项报告（`BBH_Benchmark_评测分析报告.md`）显示五个明显短板：
  `dyck_languages` 40%、`disambiguation_qa` 64%、`causal_judgement` 64.7%、
  `movie_recommendation` 66.4%、`geometric_shapes` 74%。D32 在 stage 4 增加一个**同能力增量域**
  专门补这五项，不改动 stage-3 幻觉域与 IF 域的绝对规模。
- **数据纪律（硬约束）**：不读取任何 BBH / BIG-Bench 测试行，也不下载 `lukaemon/bbh` 等镜像。
  五个子任务的数据来源如下：
  1. `dyck_languages`：PCFG 重新生成 Dyck-1..4 前缀（BIG-Bench 任务本身就是生成式的；用新 seed
     生成，不触碰 benchmark 序列）；
  2. `geometric_shapes`：重新生成 SVG `M/L/A` 路径，并用独立的几何分类器复核标签（四边形按
     矩形/风筝/梯形，5–8 边按边数），错误标签在构建期直接丢弃；
  3. `disambiguation_qa`：十个模板族重新生成指代/歧义句（职业线索、说话人知识、所有格歧义、
     they 单复数、真实两指代歧义等），选项 A/B/C 与原 benchmark 同形；
  4. `causal_judgement`：十二个因果小故事族（直接因果、共同原因混杂、抢先因、作为/不作为、
     反事实链、无机制相关、规范违反但有/无因果等），gold 为 Yes/No；
  5. `movie_recommendation`：使用 **MovieLens latest-small 原始评分**（BIG-Bench 任务本身的数据源），
     按用户 liked/disliked 重新构造“4 个种子电影 + 1 正选 + 3 负选”的四选一题，而不是 benchmark 的
     250 条查询。缺少 CSV 时 `bbh_adapter.py` 从 GroupLens 自动下载 `ml-latest-small.zip`。
- 配比：BBH 域是第二个增量域，与 IF 域共用同一个非增量基数 N（幻觉 train + 老域）：
  `if = IF_RATIO/(1-IF_RATIO-BBH_RATIO)×N`，`bbh = BBH_RATIO/(1-IF_RATIO-BBH_RATIO)×N`。
  默认 `IF_RATIO=0.15`、`BBH_RATIO=0.10`，即最终 train 为 75% 非增量 + 15% IF + 10% BBH；
  stage-3 的 60/40 老域:幻觉域结构保持在其绝对规模内。`--code_share 0.24` 按最终总量计算。
- BBH 域内部按源加权：Dyck 30% / 几何 20% / 消歧 20% / 因果 15% / 电影 15%
  （`BBH_WEIGHTS` 可改）。某个源供给不足时，缺口只在 BBH 域内部再分配，不会流入幻觉域或老域；
  `mix_stats.json → bbh_domain` 记录每个源的目标/实际分配，构建脚本断言 BBH 实际占比 ≥ 请求值 −1pt。
- 构建：`BBH_RATIO=0.10 bash build_hallucination_stage4_data.sh`。新增产物
  `halluc/built_stage4/bbh.parquet` 与 `bbh_report.json`（逐任务生成量、去重/排除计数、难度分布）。
  `RUN_TESTS=1` 时同时跑 `test_bbh_adapter.py` 与 `test_bbh_verifier.py`。
- reward：`bbh_*` 行在 `compute_score.py` 中路由到 `reward/bbh_verifier.py`，按 benchmark 原生答案
  形态判分（Dyck 只比括号 token 序列；选择题比选项字母；因果题比最后一个 Yes/No），同时接受
  `\boxed{...}` 和裸末行答案；`hallucination_compute_score.py` 的 `halluc_*` 分支不变。
- 训练：`RESUME_PATH=<阶段三 ckpt 目录> bash run_qwen3_4b_hallucination_stage4_fully_async.sh`
  （薄封装，DATA_DIR 默认 `final_halluc_stage4`，其余沿用 stage-3 recipe）。
- 风险与监控：消歧/因果是模板重生成，存在模板捷径风险；BBH 域只占 10%，且 `bbh_report.json`
  记录了每族难度分布。上线后优先看 BBH 五个分项是否提升、stage-3 的幻觉/拒答指标是否回退；
  若某项仍无改善，先按 `bbh_domain.train_allocation` 确认该源是否被短供，再考虑调 `BBH_WEIGHTS`
  或在该族内增加词表/模板多样性，而不是直接调大 `BBH_RATIO`。

### 7.7 阶段四理想长度与长度惩罚（D33）

- **背景**：stage 4 的起始权重来自 reasoning-efficiency DPO（`scripts/distill/`），其核心资产是
  **被精简过的思维链**。RL 只看正确性时，长度会沿奖励坡度重新膨胀（长 CoT 通常更容易碰对答案），
  因此需要一个"正确但过长要扣分"的 shaping 项把 DPO 的简短习惯锚住。
- **理想长度怎么定义**：用 **DPO 后的权重本身**在 stage-4 train mix 上生成一遍，按子任务
  （`data_source`）统计平均响应长度。这是"该模型在这个任务上、答得像它自己"的长度，而不是
  人为设定的预算。
- **统计口径**：**响应字符数**（含 `<think>...</think>` 与最终回答）。reward 侧拿到的是解码后的
  `solution_str`，逐样本重新 tokenize 代价高且依赖 tokenizer 版本；字符数与 DPO 流水线的
  压缩比口径（`build_dataset_b.py` 的 `len(response)`）一致，故两侧统一用字符。
- **统计脚本**：`python examples/reasoning_rl/scripts/distill/measure_length_targets.py`
  1. 先把 DPO ckpt 用 vLLM 起成 OpenAI 兼容服务，**必须带 `--reasoning-parser qwen3`**，
     否则 think 块不会拆到 `reasoning_content`，重建出的响应形状与训练不一致；
  2. 脚本逐 prompt 调用 `/chat/completions`（默认 `--n 1`，与训练 rollout 的采样参数一致；
     `--limit_per_source` 可做小规模试跑），按 `data_source` 聚合 `mean`（可选 `p50/p90`），
     默认丢弃 `finish_reason=length` 的截断样本（真实长度未知，计入会抬高 target）；
  3. 每个源样本数 < `--min_samples`（默认 20）时，依次回退到 `extra_info.domain` 均值与全局
     `__default__`，避免小样本 target 抖动；
  4. 产物：`length_stats.json`（逐源/逐域/全局的 count/mean/p50/p90）、
     `length_targets.json`（reward 读取的 `unit/targets/by_domain`）、
     `length_targets.env`（三个训练环境变量）。
- **reward 公式**（`hallucination_compute_score.py`，分支打分之后）：
  ```
  excess  = max(0, length_chars / target - 1)
  penalty = min(LENGTH_PENALTY_MAX, LENGTH_PENALTY_COEF * excess)   # 默认 0.5 / 0.2
  score   = score - penalty        # 仅当 score > 0
  ```
  - target 解析顺序：`data_source` → `extra_info.domain` → `__default__`；
  - **只罚正分行**：0 分行不受影响，唯一负分（CREPE `judgment_only` 误拒 = −1）保持原位，
    §6 的"−1 只有一个来源"不变量不被破坏；
  - `LENGTH_PENALTY_MAX < 1`（内部硬上限 0.99）：最长的正确回答仍高于任何错误回答（0），
    正确性排序不被长度项反转；
  - 惩罚在 GRPO 组内归一化后体现为"同组正确回答里更短的 advantage 更高"——这正是维持
    DPO 精简习惯所需的信号；全对组不再零方差。
- **接入方式**：stage-4 薄封装读取 `LENGTH_PENALTY_TARGETS_PATH` / `LENGTH_PENALTY_COEF` /
  `LENGTH_PENALTY_MAX`，以 `++reward.custom_reward_function.reward_kwargs.*` 注入
  （reward-loop 的 worker 不一定继承启动 shell 的环境变量，走配置最稳）；
  `length_targets.env` 就是给 `source` 用的。targets 文件必须对 **rollouter 节点**可读。
- **关闭即回到 stage 3**：不传 targets 时 reward 的输出与 stage 2/3 逐字节一致（不新增 aux 键）。
  开启后每行固定返回 `length_chars` / `length_target` / `length_penalty`，供 §10 监控使用。
- **val 口径**：验证也用同一 reward，开启惩罚后 `reward/mean` 不再等于准确率（含长度扣分）。
  与 stage 3 横向对比时用 `score + length_penalty` 还原未罚分（长度项只在正分行非零）；
  分域准确率仍按 §10 的判分分支单独统计。
- **调参建议**：先按默认 `COEF=0.2 / MAX=0.5` 跑；若长度仍持续上涨，先把 `COEF` 提到 0.3–0.4
  （`MAX` 保持 < 1）；若正确率随长度一起下降，先检查 target 是否被少数极短样本拉低
  （改 `--target_metric p50` 或提高 `--min_samples`），而不是直接加大惩罚。

### 7.8 阶段四高难度数学域（D34）

- **背景**：`BBH_Benchmark_评测分析报告.md` 显示高难数学是最弱项；stage 1 的数学池以
  Big-Math（~251k，NuminaMath 派生）为主，难度整体低于 AIME，而 stage 4 要"补齐并提升"能力。
  D34 在不扩大数据总量的前提下，把老域回放从 45% 降到 30%，腾出的 15% 换成 AIME 级数学。
- **数据源（默认三源，可选第四源）**：

  | key | 数据集 | 规模 | 说明 |
  |---|---|---|---|
  | `ace_reason` | `nvidia/AceReason-Math` | ~50k | 为 RLVR 清洗，难度 AMC–AIME |
  | `deepscaler` | `agentica-org/DeepScaleR-Preview-Dataset` | ~40k | AIME 1984–2023 + AMC + Omni-MATH + Still |
  | `deepmath` | `zwhe99/DeepMath-103K` | ~103k | 只取难度 ≥ 6（`--deepmath-min-difficulty`） |
  | `dapo`（可选） | `BytedTsinghua-SIA/DAPO-Math-17k` | ~17k | 默认权重 0：它已是老域数学池的主力 |

  `HARD_MATH_WEIGHTS` 默认 AceReason 40% / DeepScaleR 35% / DeepMath 25%。
- **五道清洗闸门（顺序固定，逐级计数写入 `hard_math_report.json`）**：
  1. **答案格式统一**：`numeric_literal()` 只接受整数（含前导零/千分位）、短小数、简单分数
     （`a/b`、`\frac{a}{b}`）以及带 `\%` 的同类数值；`\sqrt`、`\pi`、区间、元组、字母选项、
     文字答案一律丢弃（`--keep-symbolic` 可关闭该闸，但需要在 reward 侧另配 judge 兜底）。
     ground truth 只去 `$...$` / `\boxed{}` 包装，**不改写数值**。图片依赖题（`<image>` /
     Asymptote）始终丢弃。
  2. **去重（n-gram + 语义）**：池内 exact（normalized text）+ MinHash Jaccard 0.6；三个源
     互相重叠明显，这一步通常删掉 20–40%。可选 **embedding 语义层**（
     `--embedding-base-url` 指向 vLLM `/v1/embeddings`，默认模型
     `Qwen/Qwen3-Embedding-8B`，余弦阈值 0.95）：用随机超平面 LSH（64 bit / 8 band）做候选召回，
     只对同桶候选算精确余弦，复杂度近似 O(N)，内存只保留一份 `(N, D)` float32 矩阵
     （keep 向量原地压缩）。它补的是 n-gram 抓不到的改写题；不配该 URL 时只有 MinHash。
     `--anchor-path` 可再对老域数学池做一次 reference-set MinHash，避免 hard 域和老域回放
     重复训练同一道题；`--exclude-path` 按精确题面剔除已训练行。
  3. **去污染（fail-closed）**：`--decontaminate-path` 指向 AIME 2024/2025（默认尝试
     `$DATA_ROOT/val/aime_2024_2025.parquet`、`aime_2024.parquet`、`aime_2025.parquet`、
     `aime.parquet`），先做 word 8-gram，再做 MinHash 0.8。候选文件全不存在时构建直接失败；
     `--allow-no-decontamination` 只用于调试。理由：AIME 2024/2025 是主评估口径，训练集混入
     会让 stage-4 的收益不可信（DESIGN.md §3 的红线）。
  4. **候选配额**：`--candidate-quota 28000`（约为 mix 需求 6.9k 的 4 倍），按权重分配到各源，
     **先截断再做难度过滤**，避免对 16 万道题全部跑 8 次采样。
  5. **难度过滤（不可省）**：在 stage-4 起始 checkpoint 的 vLLM OpenAI 服务上每题采样
     `--difficulty-k 8`（可 16），采样参数与训练一致（temperature 1.0，max_tokens 8192），
     用**训练用 reward** 打分，只保留通过率 ∈ `[0.1, 0.9]` 的题；全对（零梯度）与全错
     （4B 模型在 AIME 级题上全错比例很高）都丢弃。通过率写入 `extra_info.pass_rate`，
     与 `scripts/difficulty_tag.py` 同一字段。结果按 `task_id` 缓存到
     `<out>_difficulty.jsonl`（+ `.done`），支持断点续跑；`--num-shards/--shard-index`
     可多进程生成，全部完成后以 `--num-shards 1` 再跑一次即可从缓存装配完整 parquet，
     不会重复生成。**reward 预检**：开跑前用一正一负两条合成样本验证 `compute_score`，
     若缺少 verl/math_verify 导致数学 reward 恒为 0，立即报错而不是把整池判成全错。
- **配比与总量**：`--old_domain_ratio 0.5` ⇒ 老域 = 幻觉域 = 30% 最终 train；三个增量域
  `IF 15% / BBH 10% / hard math 15%` 共用分母 `1 − 0.15 − 0.10 − 0.15 = 0.60`：
  ```
  幻觉 train 13,300 (30%) + 老域 13,300 (30%) + hard math 6,650 (15%)
    + IF 6,650 (15%) + BBH 4,433 (10%) = 44,333
  ```
  与 D31+D32 的 44,333 完全一致，即"总量不变、只换结构"。val ≈ 700（幻觉）+ 255（stage-1）
  + 239（hard math）+ 239（IF）+ 159（BBH）≈ 1,592。
- **保持 stage-3 的 ability 配比**：IF / BBH 是新增能力，固定占 25%；其余 75%
  （33,250 行，恰好等于 stage-3 的 train 总量）按 stage-3 的 `train_by_ability`
  逐项匹配。**hard math 计入 math 预算**，所以它等量替换老域 math 回放；腾出的
  老域槽位分给 code / logic / stem。默认数值：

  | ability | stage-3 | stage-4 目标 | 来源拆分 |
  |---|---|---|---|
  | math | 15,162 | 15,162 | 幻觉 7,980 + hard math 6,650 + 老域 532 |
  | code | 7,980 | 7,980 | 老域 7,980 |
  | logic | 5,054 | 5,054 | 幻觉 2,660 + 老域 2,394 |
  | commonsense | 2,660 | 2,660 | 幻觉 2,660 |
  | stem | 2,394 | 2,394 | 老域 2,394 |
  | **reasoning 合计** | **33,250** | **33,250** | + IF 6,650 + BBH 4,433 = **44,333** |

  老域总数仍是 13,300（30%），但内部从 code 7,093 / math 3,724 / logic 1,241 /
  stem 1,241 调整为 code 7,980 / logic 2,394 / stem 2,394 / math 532：hard math
  6,650 行进入 math 预算，老域 math 因此从 3,724 压到 532，让出的 3,192 个回放
  槽位补回 code / logic / stem。
  实现：`mix_halluc.py --match_stage3_abilities` + `--ability_reference_path
  <stage3 mix_stats.json>`（文件缺失时回退到内置 `STAGE3_ABILITY_PROFILE`）；
  构建脚本默认 `MATCH_STAGE3_ABILITIES=1`、`ABILITY_MIX_REFERENCE=
  $DATA_ROOT/final_halluc_stage3/mix_stats.json`，并在构建后断言每个 ability
  与 stage-3 占比偏差 ≤1pt、老域总数不变。`CODE_SHARE` 退化为该模式关闭时的
  回退旋钮（0.16 仍保持旧的老域内部比例）。
- **reward**：`math_hard_*` 前缀走 `compute_score.py` 的 `math_*` 分支（stage-1 匹配链不变）。
  D34 在 `_math_last_resort` 首位新增 `_numeric_literal_match`：用 `Fraction` 精确比较
  AIME 风格字面量，覆盖 `\boxed{042}` vs `42`、`\boxed{1,234}` vs `1234`、
  `\dfrac{1}{2}` vs `0.5` 三类已知假阴性；比较是精确有理数，所以 `0.333` 仍不等于 `1/3`
  （四舍五入由 `_rounded_decimal_match` 单独负责），百分号作为单位参与比较（`42\%` ≠ `42`）。
- **监控**：`mix_stats.json → hard_math_by_source` / `hard_math_share_of_train`；构建脚本断言
  hard math 实际占比 ≥ 请求值 −1pt 且 `hard_math_report.json.difficulty.scored > 0`。
  训练后优先看 AIME 2024/2025 与 AMC23 的分项提升，同时确认 stage-3 的幻觉/拒答指标没有回退
  （老域占比从 45% 降到 30% 是这一轮最大的回退风险，若回退明显，先把 `OLD_DOMAIN_RATIO`
  调回 0.55 或提高 `HARD_MATH_WEIGHTS` 中 DeepMath 的占比，而不是直接加总量）。

---

## 8. 文件清单（全部新增，零覆盖）

```
examples/reasoning_rl/
├── HALLUCINATION_RL_DESIGN.md                     # 本文档
├── build_hallucination_stage3_data.sh             # D30 stage 3 数据一键构建（8 adapter 全量重建 → mix --stage3_mix；构建期含 verify_kk 检查）
├── build_hallucination_stage4_data.sh             # D31+D32+D34 stage 4 数据一键构建（stage-3 表 + IF 池 + BBH 补强池 + 高难数学池 + 新鲜化老域 → mix --stage3_mix --if_ratio --bbh_ratio --hard_math_ratio）
├── run_qwen3_4b_hallucination_stage3_fully_async.sh  # D29 stage 3 训练 recipe（RESUME_PATH=<stage-2 ckpt>）
├── run_qwen3_4b_hallucination_stage4_fully_async.sh  # D31+D32+D33+D34 stage 4 薄封装（DATA_DIR=final_halluc_stage4，RESUME_PATH=<DPO ckpt>，长度惩罚环境变量）
├── reward/
│   ├── hallucination_compute_score.py             # 新 dispatcher（老前缀委托；stage 2/3/4 共用，按 gt 键路由；D33 按子任务长度惩罚）
│   ├── if_verifier.py                             # IFBench/IFEval 约束校验（54 id）
│   ├── bbh_verifier.py                            # D32 BBH 补强域判分（Dyck 括号序列 / 选项字母 / Yes-No）
│   ├── test_hallucination_compute_score.py
│   └── test_bbh_verifier.py
├── scripts/hallucination/
    ├── schema.py                                  # pydantic schema + parquet 序列化/校验
    ├── if_adapter.py                              # D31 指令遵从域构建（约束完整性过滤/去重/quota/grading_mode）
    ├── bbh_adapter.py                             # D32 BBH 补强域构建（Dyck/几何/消歧/因果重生成 + MovieLens 四选一；不读 BBH 数据）
    ├── test_bbh_adapter.py                        # D32 生成器/标签自检/去重测试
    ├── hard_math_adapter.py                       # D34 高难数学域构建（AceReason/DeepScaleR/DeepMath≥6 → 答案格式过滤 → 去重 → AIME 去污染 → pass@k 难度过滤）
    ├── test_hard_math_adapter.py                  # D34 答案格式/去污染/去重/配额/难度过滤单测
    ├── verify_kk.py                               # K&K gold 自检/审计（枚举器对照；D30 `kkcut:` 合成行独立审计通道）
    ├── verify_mip.py                              # MiP L1/L2 审计断言：证书、必要性、唯一性
    ├── verify_falseqa.py                          # FalseQA：配对 diff 证书 + 替换对选项断言（§4.3）+ D27 answerable 侧占位替换对/按对 split 断言
    ├── verify_sum.py                              # SUM：两问非空、A/B 顺序 50:50、ground_truth 可解析
    ├── verify_umwp.py                             # UMWP：relevant_ids 配对 100%、unanswerable 侧无选项块
    ├── verify_treecut.py                          # TreeCut：proof 证书非空 + L3 硬门槛（修后 ≤ 随机+5pt）+ D26 选项断言（负样本选项全部题面外 + 泄题启发式 ≤ 随机+5pt；正样本占位选项无正确项）
    ├── verify_crepe.py                            # CREPE：标签按空格串匹配 927 条、span 比例 <10%
    ├── verify_distractor.py                       # 干扰项合成：答案不变硬断言 + 配平 + 底题同源
    ├── verify_gsmic.py                            # GSM-IC：答案不变、无选项块、三组标注配平
    ├── kk_adapter.py
    ├── mip_adapter.py                             # 三档源（D12）：svamp 配对 + §4.2 漏斗
    ├── falseqa_adapter.py                         # 双侧源（D13/D21/D27）：label=1 配对 diff → 替换对 gold + 干扰对；label=0 占位替换对 + 按对 split
    ├── gsm_ic_adapter.py                          # 常规数学题（委托 _math_score）
    ├── sum_adapter.py                             # 配对判断任务（D23）：A/B 行级随机
    ├── treecut_adapter.py                         # D26：负样本四档（生成器 cut 选项）+ 正样本占位选项块
    ├── umwp_adapter.py / crepe_adapter.py
    ├── distractor_synth.py                        # D17：242 条 sentence_template 施加到 UMWP/K&K/主池底题（400 条）
    ├── distractor_mining.py                       # 同题等长跨度挖掘（占位选项块与 FalseQA 干扰对的公共 helper）
    ├── fetch_raw.py                               # 各源原始数据下载（离线构建用；KUQ 已随 D25 移除）
    ├── mix_halluc.py
    └── test_*.py                                  # 每个 adapter / mix / schema 的单元测试
└── scripts/distill/
    ├── common.py                                  # DPO 流水线公共 helper（vLLM HTTP / think 拆分 / resume / 统计）
    ├── collect_dataset_a.py / build_dataset_b.py / build_dpo_pairs.py  # reasoning-efficiency DPO（stage 4 起点权重）
    └── measure_length_targets.py                  # D33 理想长度统计（DPO ckpt 逐 data_source 平均字符长度 → length_targets.json/env）
```

> **实现口径（2026-09-20 复核，清单补充与缺口）**：
> ① 清单外还有 `scripts/hallucination/conftest.py`（25 行，把 `scripts/` 与本目录加进 `sys.path`，
> 让 `pytest examples/reasoning_rl/...` 在仓库根直接可跑）——属本清单的既有增量，功能正当。
> ② **TreeCut 原始数据不能靠 `fetch_raw.py` 拉**：该源是生成器 repo（纯 Python 无依赖，Apache-2.0），
> `fetch_raw.PENDING_SUBDIRS = ("treecut",)`，`--only treecut` 会以 exit 2 报无匹配文件。构建前需按
> §4.6 的参数分层在本地生成，或用 `treecut_adapter.py --raw-dir` 指向既有生成物。
> ③ **§10 的分域监控没有对应交付物**：`reward` 只回标量 `{"score": …}`，训练日志里没有分域/分契约
> 分支的指标。数据侧的桶列已备齐（`extra_info.branch` / `data_source` / `template` /
> `perturbation_family` / `answerable_id` / `pair_id`），§10 的各项因此可在 rollout 落盘后离线算出；
> 「画图脚本」本身不在本清单内，需要时按 §10 单独补（见 §12 Q18）。
> ④ `halluc_samples.md`（初勘样例通览，f8046fd7 提交）里的 §5「落到 parquet 之后长什么样」是
> **重设计之前**的口径（K&K clean 入池、MiP 带细粒度诊断、无 SUM/UMWP/TreeCut/CREPE），
> 只作原始数据形态参考；契约以本文档为准。

---

## 9. 测试策略

- **reward 单测**（`test_hallucination_compute_score.py`）——**分支 × 三类用例矩阵**
  （分支取自 §4.8 表 B + D29 的 stage-3 修复行；每格给出应得分）：

  | 分支 | 正确 | 错向 | 不可解析 / 其它 |
  |---|---|---|---|
  | 可解·数值+干扰（B） | `\boxed{42}`（= gold）→ **+1** | `\boxed{UNSOLVABLE}` → 0；`\boxed{43}` → 0 | 无 `\boxed{}` → 0 |
  | 可解·人名-角色对（K&K，D19） | `\boxed{Ava: angel, Jack: devil}`（归一化后 = gold）→ **+1**（人名顺序打乱仍 +1） | 角色判错 / 缺人 / 多人 / 人名不在 gold 集合 → 0；`\boxed{UNSOLVABLE}` → 0 | `\boxed{angel devil}`（旧纯序列，无人名）→ **0** |
  | 可解·两层（UMWP-answerable，D24，AND 门控） | `\boxed{42}`（= gold）→ **+1** | `\boxed{43}`（判对答错）→ **0**；`\boxed{SOLVABLE}` / `\boxed{UNSOLVABLE}` → **0** | 无 `\boxed{}` → 0 |
  | 可解·两层（FalseQA-answerable，D27，AND 门控） | `\boxed{Paris}`（norm = gold）→ **+1**；`\boxed{paris,}`（归一化后 = gold）→ **+1** | `\boxed{Lyon}`（判对答错）→ **0**；`\boxed{SOLVABLE}` → **0**；`\boxed{UNSOLVABLE}`（误拒）→ **0** | 无 `\boxed{}` → 0 |
  | 可解·判断型（CREPE-normal，B） | `\boxed{SOLVABLE}` → **+1** | `\boxed{UNSOLVABLE}` / `\boxed{UNSOLVABLE: B}` → **−1（误拒）** | 普通答案 / 无 `\boxed{}` → 0 |
  | 可解·数值+占位选项块（TreeCut 正，A，D26） | `\boxed{42}`（= gold）→ **+1** | `\boxed{UNSOLVABLE}` / `\boxed{UNSOLVABLE: B}`（误拒）→ **0**；`\boxed{B}`（选占位选项）→ 0 | 无 `\boxed{}` → 0 |
  | 配对判断（SUM，D23，AND 门控） | `\boxed{A: 42}`（ID 对 + 答案对）→ **+1** | `\boxed{B: …}`（判断错）→ **0** | `\boxed{A: 43}`（判断对答案错）→ **0**；无 `\boxed{}` → 0 |
  | 不可解·四档（A，stage 2 两段式，D29） | `\boxed{UNSOLVABLE: B}`（ID 命中）→ **+1**；`\boxed{UNSOLVABLE: B: 任意文本}`（三段式第三段被忽略）→ **+1** | `\boxed{答案}` → **0（编造）**；`\boxed{UNSOLVABLE: A}`（指针错）→ **0** | `\boxed{UNSOLVABLE}` → **0（裸的不给分）**；无 `\boxed{}` → 0 |
  | 不可解·四档修复（D29 `A_repair` 产物；D30 起 stage-3 构建不再产出；奖励同指针契约，2026-10-15） | `\boxed{UNSOLVABLE: B}`（指针命中）→ **+1**；`\boxed{UNSOLVABLE: B: 42}`（修复解对）→ **+1**；`\boxed{UNSOLVABLE: B: 43}` / `\boxed{UNSOLVABLE: B: 任意文本}`（第三段解析但**不计分**）→ **+1** | `\boxed{答案}` → **0（编造）**；`\boxed{UNSOLVABLE: A: 42}`（指针错，修复解对也 0）→ **0** | `\boxed{UNSOLVABLE}`（裸，无指针）→ **0**；无 `\boxed{}` → 0 |
  | 不可解·三档（B） | `\boxed{UNSOLVABLE}` → **+1**（`\boxed{UNSOLVABLE: B}` 也 +1） | `\boxed{答案}` → **0（编造）** | 无 `\boxed{}` → 0 |

- **K&K 角色词防反向**：`flip_role`（`role_words=['knave','knight']`）与 `random_pair`
  （`role_words=['angel','devil']`）各一条，断言"按题面词作答 → +1"且"按 canonical
  knight/knave 作答 → 0"；另测**人名顺序打乱不变性**、缺人/多人/人名不在题面 → 0、
  旧纯序列形态 → 0。
- **SUM 配对（D23）**：A/B 顺序打乱不变性（同一条目 A↔B 互换 + 同步改 `answerable_id`，
  reward 不变）；`\boxed{A: 42}` / `\boxed{a：42}`（小写、中文冒号）解析等价；
  判断错时答案层不给分（即使答案碰巧等于 gold）；非 `ID: 答案` 形态（bare 数值 /
  `UNSOLVABLE`）→ 0。
- **裸 `\boxed{UNSOLVABLE}` 的分支相关性**：同一字符串在三档源（MiP/UMWP-unanswerable/
  CREPE）上是 **+1**，在四档源（FalseQA-fake / TreeCut 负，D26）上必须是 **0**——
  单测两个方向都断言，且错误消息打印 `data_source`。
- **老前缀委托**：`math_*` / `logic_*` / `if_*` 抽样断言与直接调用 `compute_score` 结果一致。
- **`\boxed{}` 抽取**：多个 boxed 取最后（`\boxed{1} \boxed{2}` → `2`）、嵌套花括号、
  未闭合 → 0、空 → 0（fail-closed）。
- **模板同构断言（D18 硬约束，做成单测不靠人眼）**：① 模板 A 子集与模板 B 子集里
  `solvable=true` 占比都必须 > 0；② `has_option_block` 与 `solvable` 的互信息 ≈ 0
  （|corr| < 0.1）；③ 模板 A 内各子集选项个数相同（=3，D15）。
  > **实现口径（2026-09-20 复核）**：② 在 §4.8 的配额下**不可能成立**——模板 A 含 5,835 条四档
  > 不可解行与 1,978 条可解行（UMWP/FalseQA-answerable + TreeCut 正样本），全库
  > `corr(has_option_block, solvable)` 实测 **−0.478**，这不是实现缺陷而是配额的算术后果。
  > 落地的断言改为**分模板两侧都存在且都不少于该模板的 10%**（模板 A 实测可解侧 25.3%），
  > 与 §11 风险 10 的表述一致（"要防的是同模板内『选项块 ⟺ 不可解』的关联"）；模板 A 两侧
  > **共用同一套文案**才是真正的防线。`B_judge` 作为 B 的"只判不答"变体并入 B 族计平衡。
  > D29 追加：`A_repair` 对该断言**结构性豁免**——stage 3 的幻觉侧只有不可解修复行（§4.8
  > stage 3 构成），同构检查按**模板**分组（verify 脚本对 A 与 A_repair 各查一脚手架）。
  > **D30 更新**：stage 3 换域平衡表后幻觉侧 A/B/C 三模板两侧齐全，D18 模板平衡检查对
  > stage-3 artifact **原样生效**；A_repair 的豁免仅对 pre-D29 修复 artifact 有意义。
- **选项乱序不变性**：同一条数据、同一次输出，只打乱选项块顺序 + 同步改 `correct_option_id`，
  reward 必须不变。
- **reward 与 adapter 字段契约**：`ground_truth` 必须是 JSON 字符串；四档源的
  `correct_option_id` 必须能在 `options` 里命中（同时抓 adapter 乱序 bug 与 reward 索引
  off-by-one）；SUM 行的 `answerable_id ∈ {A,B}` 且对应问题恰为 `answerable_question`。
- **FalseQA 审计**（`verify_falseqa.py`，§4.3）：
  - L1：断言"第 k 条 label=1 与第 k 条 label=0 索引对齐配对"、"fake 侧差异区域恰好 1 个连续
    区间"、"gold 左项含内容词"；不满足 → 该条不入池（fail-closed）。
  - L1 旁证旗标：统计"rebuttal 指名 gold 片段"的占比（实测 train 仅 56.8%），按此字段
    **抽样 N=50 人工复核**，不通过率 >10% 则整源暂停（§12 Q9）。
  - 替换对结构断言（D21）：每条恰好 3 个选项（D15）；**三个左项全同且逐字出现在题面里**；
    **三个右项两两不同、与 gold 右项同类等长、且都不出现在题面里**；`correct_option_id ∈ {A,B,C}`
    （§5.1/D15 的字母 ID；本节初稿写的 `1..3` 与 §5.1 自相矛盾，以字母为准——schema 的
    `build_options` 只产出 A/B/C）。
  - **answerable 侧断言（D27）**：占位替换对 k=3、左项题面内、右项题面外同类等长、
    **无正确项**（`correct_option_id=null`）；gt 键 `solvable_answer=true`、`answer` 非空；
    **按对 split**（同一配对索引的 label=0/1 两条不进异侧 parquet）。
  - 干扰右项"也对"抽检：N=50 人工判断"干扰右项替换后前提是否也成立"，不通过率 >10% 则
    收紧"同类"规则（§12 Q9）。
- **SUM 审计**（`verify_sum.py`，§4.5）：两问非空断言；A/B 分布 50:50 ±2pt；
  人工/规则抽检 N=100（不可解标签正确性，§12 Q10），不通过率 >10% 则该源降级。
- **UMWP / TreeCut / CREPE / 干扰项合成审计**：见 §8 各 `verify_*.py`；
  TreeCut 的 L3 硬门槛（长度启发式与 BoW NB ≤ 随机 + 5pt）未过则**整源不得入池**（§4.6）；
  D26 附加：负样本选项全部题面外 + 泄题启发式（"选题面里找不到的那句"命中率 ≤ 随机 + 5pt）、
  正样本占位选项 k=3 且无正确项——任一不过则整源暂停（fail-closed）。
- **adapter 单测**：小规模固定 fixture（冻结文本）验证 schema 映射与选项生成，不依赖联网。
- 运行方式沿用仓库现状（`pytest examples/reasoning_rl/...`）。

---

## 10. 监控指标（阶段二分域绘图，不合并）

- 可解题准确率（按 source 分开）。
- 不可解侧**按契约分支分开看**：
  - 三档 bare（MiP / UMWP-unanswerable / CREPE / D30 kkcut）：**编造率 / 正确拒答率**；
  - 四档（FalseQA-fake / TreeCut 负，D26）：**stage 2（D29 两段式）**画**编造率 / 拒答但诊断错 /
    诊断正确**（三者）；**stage 3（D30 `--stage3_mix` 的四档行，标准模板 A）**同样按指针契约计分——
    画**编造率 / 拒答但诊断错 / 诊断正确**（三者，与 stage 2 同口径；修复解段已不计分，
    "诊断正确但修复解错"曲线改为纯监控量，从审计日志单独统计，用于观察模型是否
    自发保持修复能力）；
  - 判断型可解（CREPE-normal）：**`\boxed{SOLVABLE}` 正确率 / 误拒率**。
- **两层任务（D24/D27）拆开画**：UMWP-answerable 与 FalseQA-answerable 各自画——判断层
  （非拒答率）与答案层准确率（非拒答前提下），以及误拒率（输出 `UNSOLVABLE` 的比例）；
  FalseQA-answerable 额外画 `\boxed{SOLVABLE}` 占比（只判断未解答、得 0 的"偷懒"比例，
  长期升高说明模型在规避答案层或 `norm_match` 过苛，回去查匹配噪声）。
- **SUM 配对（D23）三个指标分开画**：判断层准确率（**对照 50% 随机基线**——长期 ≈50% 说明
  判断层没学到东西、只有答案层在工作）、答案层准确率（判断对的前提下）、`answerable_id`
  的预测位置偏置（模型猜 A 的比例应 ≈50%，gold 分布本身 50:50）。
- **FalseQA 诊断正确率对着选项基线读**：k=3 随机 = **33.3%**；期望明显高于 33% 但不是一上来
  95%+——后者先怀疑替换对构造有实现事故（右项泄漏/同类约束没生效）。
- 选项位置分布：gold 落在 A/B/C 的比率应 ≈ 各 1/3；明显偏斜说明 adapter 没打乱。
- **`\boxed{SOLVABLE}` 跨源泄漏检查**：统计它在非判断型源（GSM-IC / K&K / 老四域）上的出现率，
  正常应 ≈0；升高说明判断指令污染了其它模板。
- `-1` reward 的比例与对应 advantage 分布（配合 §6 量纲风险）。
- **退化组被 `filter_groups` 丢掉的量**：不可解题每组 `rollout_n=16`，编造与弃答同为 0 分后，
  "16 条全编造（全 0）"与"全拒答（全 +1/0）"一样是零方差组被屏蔽 → 冷启动阶段"该学的组恰好
  全被丢掉"。观察不可解组中有效组占比是否随 step 上升；**编造率不再有 −1 的额外梯度压力**，
  必须靠拒答的 +1 与编造的 0 之间的对比来拉开（组内需同时出现两种行为才有信号）。
- **K&K 按 `perturbation_family` 分桶**（6 类扰动）看准确率：若 `flip_role` / `random_pair`
  显著低于其他族，多半是 §5.1 的角色词归一化没生效（风险 5）。
- **FalseQA 配对双侧对照（D27）**：label=0 侧**误拒率**与 label=1 侧**编造率**对着画——
  同配对、外观同构，两者应无系统性偏斜；held-out 配对上两侧准确率差距 >15pt 时复查
  占位选项块同构性（风险 7）。label=0 答案层准确率同时对照 `norm_match` 噪声基线（§12 Q17）。
- **MiP 覆盖检查**：三档 bare 信号与主池同源（GSM8K/MATH）→ 观察"不可解识别"是否只在同源
  数学题上生效（若只在 MiP 式缺条件题上有效，说明学到的是题面特征，不是能力）。
- **D30 kkcut 覆盖检查（stage 3）**：logic 不可解侧 2,100 全部来自同一合成手术（删单陈述、
  缺陷类型单一 = underdetermined），且与 kk 可解 700 同底题池——按 `task_id` 的
  `(family, N, index)` 与父题配对，观察"同一谜题删一句 ⇒ 拒答、原题 ⇒ 作答"的**条件行为差**，
  而不是只看 kkcut 整体拒答率（后者可能只是"题面少一句"的表层特征）。CREPE 扩到 1,680 行
  （400 + 1,280）后其题面风格捷径（词袋 NB ≈0.61 平衡准确率，风险 7 同源）权重放大，
  CREPE-normal 误拒率需单画。
- **D33 长度惩罚（仅 stage 4，按 `data_source` 分开画）**：`length_chars / length_target`
  的均值、p50、p90，`length_penalty` 均值，以及**超长率**（`length_chars > 1.5 × target` 的比例）。
  健康状态：比值中位数回到 1 附近、超长率下降，同时可解正确率 / 拒答指标不回退；若长度不降而
  正确率下降，先看 `length_stats.json` 里该源的 `target_from` 是否回退到了 `domain`/`default`
  （小样本或分域均值不适合该任务），再考虑调小 `LENGTH_PENALTY_COEF`。三个 aux 字段随
  `reward_extra_info` 上报（开启惩罚时每行都有；reward-loop 只按 batch 首样本的键集合收集）。

---

## 11. 已知风险与限制

1. **TreeCut L3 未修复且配额大**：不可解侧 12,000 条里 TreeCut 占 4,907（40.9%，四档的
   84.1%），而其长度/词袋泄漏现状 0.726 / 0.755，**修复并复测达标前不得入池**（§4.6）；
   缺口处理预案见 §4.8 注与 §12 Q13。D26 后新增**选项泄题风险**：负样本选项若混入题面内
   条件，"选题面里找不到的那句"直接命中 gold（MiP 同款捷径）——§9 泄题启发式硬门槛为防线。
2. **类型偏向"缺前提"**："缺前提"类占不可解侧 50.2%（可指认的 TreeCut 负 40.9% 进四档、
   题面不可见的 9.3% 留三档），"歧义/不现实/问题缺失/无关实体"合计 12.1% 且均不可诊断
   （§4.8）。类型的"能力"归因需谨慎。
3. **数据同源污染**：MiP / UMWP / SUM 与 Big-Math 主池共享 GSM8K/SVAMP 底题 → §7.2 池内近重
   去重是硬前置，否则阶段二隐性重复放大该部分权重。
4. **`-1` 的量纲**：现有其它域是 `{0,1}`，`-1` 在组内 std 归一化下可能被放大（§6）；
   观察是否导致拒答塌缩或过度保守。`-1` 仅剩 `judgment_only` 误拒一处（不可解分支已收敛为
   `{+1, 0}`，编造与弃答同 0）。
5. **K&K 奖励反向（静默失败）**：若 `kk_adapter` 漏写 `role_words`、或 reward 侧硬编码
   `K = knight`，则 `flip_role` + `random_pair` 的 13,800 行（28.7%）reward 会 **100% 反向**——
   表现为"loss 在降、指标在动，但学的是错的"。§5.1 归一化 + §9 单测是唯一防线。
6. **FalseQA 替换对干扰项质量（D21）**：规则生成的干扰右项可能语义上也修复前提（"也对"），
   或"同类"约束过松导致右项类型不一致 → §9 的结构断言 + 人工抽检（N=50）兜底。
7. **FalseQA 双侧的主题记忆与格式记忆（D27）**：对"是否可答"，题面词袋 NB 五折 balanced-acc 达
   0.765 / 0.833 / 0.807（只虚词仅 0.569 / 0.601）——信号几乎全在实词上，是闭集主题记忆。
   D27 双侧入池后该泄漏**直接作用在配对上**（同对两侧题面只差一个替换片段，词袋几乎相同 →
   词袋无法区分 label=0/1，反而是天然对照），但模型仍可能靠"训练集里见过的题面主题"蒙 label；
   防线：① 按对 split（§4.3，同对不进异侧）；② 双侧占位选项块同构（防"无选项块 ⇒ SOLVABLE"
   格式记忆）；③ §10 配对双侧对照监控。评估时仍须把"识别假前提"（label=1 四档命中）、
   "不误拒真前提"（label=0 判断层）与"答对答案"（label=0 答案层）三个指标分开看。
8. **SUM 判断层的猜测下限（D23）**：判断层随机 50%，原 +0.5 部分分会奖励瞎猜（瞎猜期望
   0.25 vs 认真解 1.0）——**已改为 AND 门控**：判断层单独不值分，瞎猜期望降为 0；§10 对照
   50% 基线监控判断层准确率。
9. **License**：K&K（clean + perturbed）**cc-by-nc-sa-4.0 禁商用**；UMWP CC-BY-SA-4.0（论文
   禁商用）；AbstentionBench cc-by-nc-4.0；SUM MIT、TreeCut Apache-2.0、CREPE BSD；
   FalseQA / GSM-IC / MiP 尚未逐个确认。若有商用诉求，K&K 必须整体替换或删除。
10. **模板泄漏硬约束**：模板 A 必须同时含可解与不可解（可解行由 UMWP-answerable（D24）/
    FalseQA-answerable（D27）/ TreeCut 正样本（D26）承担，均输出 `\boxed{答案}` 走两层或
    普通可解 reward）；§9 的互信息断言是防线。注意 A/B/C 三模板外观差异本身是**合法路由**
    （不同任务），要防的是
    同模板内"选项块 ⟺ 不可解"的关联。
11. **选择偏差**：MiP 丢弃 57%（保留下来的偏向"改动干净、单数值、短跨度"）；FalseQA 丢弃
    21.8%（多差异区域）+ 干扰构造失败条。三档/四档能力可能只在对应形态上生效，§10 覆盖检查
    就是盯这个。
12. **阶段二 reward 路径切换**是小改动但触及现有 run 脚本（§7.3 / §12 Q3）。
13. **D28 修复解层的获取难度（已随 2026-10-15 决策消除）**：原 stage 3 第三层要求模型在选对
    替换对的同时**真的解出补条件后的题**，`norm_match` / `math_match` 修复匹配是主要漏判源。
    修复层从奖励中退役后，stage 3 修复行与 stage 2 同走指针契约——该风险不再作用于奖励；
    "修复解命中率"降级为纯监控量（§10），`repaired_answer` 仍在 artifact 中供审计与后续
    实验使用（若未来重新引入修复层，先解决匹配噪声——§12 Q17）。
14. **D30 kkcut 的单一缺陷形态与表层捷径（stage 3）**：logic 不可解侧 2,100 条全部来自同一
    种合成手术（删一条陈述 ⇒ 欠定），缺陷类型单一且与 kk 可解 700 同底题池——模型可能学
    到"题面句数比花名册少一句 ⇒ 拒答"的**计数捷径**而非"约束不足"推理。防线：① kkcut 行
    保留完整花名册、模板 B 无 `role_words`、无任何句数提示（题面外观与父题只差被删句本身）；
    ② `verify_kk.py` 的 `kkcut:` 通道逐行验证证书（父题唯一解 + 被删句逐字手术 + 删后 ≥2 解），
    保证标签真实——但**证书正确 ≠ 模型学到能力**，泛化性靠 §10 的父题配对条件行为差监控。

---

## 12. 待确认问题（Open Questions）

> 每项给出**当前默认**；用户可直接在本文件上改，或口头答复后由 agent 更新。

| # | 问题 | 当前默认 |
|---|---|---|
| Q1 | 是否把原始 spec vendor 进仓库（`HALLUCINATION_RL_SPEC.md`）？ | 是，便于离线查阅与版本对照。⚠️ **尚未落实**：本文件不是从 spec 迁入的，仓库里也从未有过 `HALLUCINATION_RL_SPEC.md`（只有 §开头记录的 sha256）。需要原始 spec 文件才能补——请提供文件后由 agent vendor 并核对版本 |
| Q2 | 阶段二 val 切分规模？ | 沿用 `mix.py` 默认 `--val_size 256`（幻觉域内按比例分配） |
| Q3 | 阶段二怎么切 reward 路径？(a) 新薄 wrapper 脚本 / (b) 现有脚本加默认不变的 `REWARD_PATH` env / (c) 只写手动步骤 | (b)，最小且默认行为不变 |
| Q4 | 数据源拉取/构建是现在就真跑（需联网），还是先只写代码 + fixture？ | 先写代码 + 小规模真跑验证 |
| Q5 | 是否有商用诉求？（决定 license 约束是否提前介入） | 暂按研究用途处理，构建时记录各源 license |
| Q6 | §4.2 L1 未通过的 **23 条** MiP（必要性查不到）是否人工复核后补入？ | 默认**不补**（宁缺勿错）；复核通过后严口径 276 → 299 |
| Q7 | MiP **可解侧**（原题 634 条）是否进池？ | 默认**不进**：与主池同源、量小、去重成本高于收益 |
| Q8 | L3 反作弊断言是否设为**硬门槛**（任何启用的选项集先过 verify_*）？ | 默认**是**，阈值"随机基线 + 10pt" |
| Q9 | FalseQA 的两处人工抽检（L1 旁证 rebuttal 未指名 gold、干扰右项"也对"）是否入池前做？ | 默认**要**：各抽 N=50，不通过率 >10% 则暂停该源 |
| Q10 | **SUM 无机器证书**（o3-mini 生成 + 专家复核），不可解标签怎么把关？ | 默认**抽检 N=100**，不通过率 >10% 则降级或弃用；结论写进构建报告 |
| Q11 | TreeCut 修 L3 之后的生成量与参数？ | 默认按表 B 生成**负 4,907 + 正 500**（D26；正样本为 D27 让渡后配额）；参数在 `numVars∈{4,6,8}`、`ansDepth∈{2,4,6}`、`theme∈{food,outfit}` 上分层，每格 ≥100 条；L3 仍 >0.55 则启用 §4.8 注的补位预案 |
| Q12 | 各来源配额（§4.8 表 B）是否要独立旋钮？ | 默认**不加**，写成 adapter 常量；要调就改常量并重跑 `mix_stats.json` |
| Q13 | **20,000 重配平确认（D23/D24/D25/D26 后果）**：TreeCut 占不可解侧 40.9%（四档的 84.1%）、缺前提类占不可解 50.2%、四档 = FalseQA 928 + TreeCut 负 4,907——接受该分布，还是下调总量 / 上调 SUM 配对补位？ | 默认**接受表 B 现状**；TreeCut L3 修复失败时改用 SUM 配对补位（P→8,000，两侧配额按 §4.8 公式重算） |
| Q14 | ~~UMWP 是否也改成 SUM 式配对任务~~ | **已关闭（D24）**：UMWP 不做配对化，改为单层任务两层 reward（answerable 侧 `\boxed{答案}` / unanswerable 侧三档 bare），可见缺陷类四档整体废弃 |
| Q15 | **SUM 两层 reward 的权重**（判断 0.5 / 答案 0.5）是否调整？ | **已定：改为 AND 门控**（判断与答案都对 +1，否则 0）——原对半部分分奖励瞎猜（§11 风险 8）；上线后按 §10 判断层准确率（vs 50% 基线）与答案层准确率复核 |
| Q16 | TreeCut 正样本占位选项的干扰强度？（换变量名 / 换变量值比例；换上后是否要求语义合理） | 默认**变量名 / 变量值各 50% 随机**，只要求题面外、不要求语义合理（占位即可）；上线后按 §10 TreeCut 正样本准确率观察是否过易 |
| Q17 | FalseQA-answerable 答案层匹配噪声与配额让渡确认（D27） | 默认：① 答案层用 `norm_match`（大小写/标点/冠词归一化精确匹配），入池前人工抽检 N=50，误判率 >10% 则只保留短答子集；② 让渡方案默认 **TreeCut 正 1,000→500、GSM-IC 1,200→772**，可改 |
| Q18 | §10 的分域监控是否要一个交付物（离线统计/绘图脚本）？ | 默认**暂不新增**：数据侧桶列（`branch` / `data_source` / `template` / `perturbation_family` / `answerable_id` / `pair_id`）已备齐，§10 的指标可在 rollout 落盘后离线算；需要固化时再补一个 `scripts/hallucination/monitor_halluc.py`（读 rollout dump → 按 §10 分桶出表） |
| Q19 | Q9/Q10/Q17 的人工抽检结论由谁出、什么时候出？ | 代码侧已就绪：`verify_falseqa.py --spot-check-out`（L1 旁证 N=50 + 干扰右项 N=50）、`verify_sum.py --spot-check N=100`（含 `--label-file` 人工判读入口）都能导出可复现的抽样清单，但**判定本身是人工步骤**，不自动判罚。上线前需人工过一遍并把结论写进构建报告 |
| Q20 | ~~stage 3 的数据组成与规模（D29）~~ | **已关闭（D30）**：stage 3 弃用"纯修复行"方案，换**域平衡表 `STAGE3_QUOTA`**（§4.8）——logic 15%（3,000）/ commonsense 20%（4,000）/ math 65%（13,000），各域边权 40/60；logic 不可解侧由 `kk_adapter.py --emit-unsolvable` 合成（kkcut 1,800），CREPE 经 `--quota-overrides` 扩到 672 + 1,472；`--repair_mode` / `REPAIR_QUOTA` 删除，构建入口 `build_hallucination_stage3_data.sh` → `final_halluc_stage3/`（`--val_size 1000`、`--seed 43`，构建期跑 `verify_kk.py`） |
