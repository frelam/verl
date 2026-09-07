# Qwen3-4B Reasoning RL 数据与训练方案

> 目标：以 Qwen3-4B 为基座，用 DAPO 算法在数学 / 代码 / 逻辑谜题 / STEM 四个域上做可验证奖励的
> reasoning RL。本文档是实施方案的唯一事实来源（single source of truth），涵盖数据获取、
> 同源合并、去污染、难度预筛、混合配比、奖励接入、评估协议与训练配置。

---

## 0. 流水线总览

```
原始下载 → 统一 parquet schema → 同源合并去重 → 对验证集去污染(n-gram + embedding，全自动)
→ 难度预筛(pass@k) → 混合配比 → 训练(DAPO + hard replay) → 固定评估协议
```

每个阶段脚本化，产出落盘（parquet + 统计报告），保证可复现、可增量更新。

目录结构：

```
examples/reasoning_rl/
├── DESIGN.md                     # 本文档
├── README.md                     # 数据处理执行手册（供 agent 按步骤执行）
├── scripts/
│   ├── dedup.py                  # 池内去重：exact + MinHash-LSH（已实现）
│   ├── to_parquet_math.py        # 数学域：Big-Math + DAPO-17k 合并转换（已实现）
│   ├── to_parquet_code.py        # 代码域：DeepCoder + CodeContests + APPS 转换（已实现）
│   ├── to_parquet_logic.py       # 逻辑域：SynLogic/Enigmata/ReasoningGym（已实现）
│   ├── to_parquet_stem.py        # STEM 域：Dr.SCI 可验证子集（已实现）
│   ├── decontaminate.py          # n-gram + embedding 两级去污染（已实现）
│   ├── difficulty_tag.py         # pass@k 难度预筛（已实现）
│   └── mix.py                    # 按比例混合生成最终训练 parquet（已实现）
├── reward/
│   ├── compute_score.py          # 四域 reward 分发（已实现）
│   └── test_compute_score.py     # reward 单元测试
├── hard_replay.py                # hard replay（re-export examples.tool_rl 实现）
├── reasoning_rl_dataset.py       # replay 重入 dataset（已实现）
├── run_qwen3_4b_reasoning_rl_dapo.sh  # DAPO + hard replay 训练脚本（已实现）
└── data/                         # 产出落盘（gitignore）
    ├── raw/
    ├── decontaminated/
    └── final/
```

---

## 1. 统一数据 Schema（所有域共用）

所有数据集最终转成 verl 标准 parquet，字段约定：

| 字段 | 内容 |
|---|---|
| `data_source` | 奖励路由键，命名 `{domain}_{dataset}`，如 `math_bigmath`、`code_deepcoder`、`logic_synlogic`、`stem_drsci` |
| `prompt` | chat 格式 `[{"role": "user", "content": ...}]` |
| `ability` | `math` / `code` / `logic` / `stem` |
| `reward_model` | `{"style": "rule", "ground_truth": ...}`，ground_truth 按域定义（见 §6） |
| `extra_info` | `{"split", "index", "task_id", "domain", "source", "difficulty", "pass_rate", "seed"(生成式必带)}` |

逻辑：`data_source` 是 reward 分发键（`verl/utils/reward_score/__init__.py` 的
`default_compute_score`，或训练配置 `reward_model.custom_reward_function.path` 挂自定义分发），
新增域只需注册 verifier，训练侧零改动。

统一 prompt 格式约束（写死在转换脚本里，避免 reward 解析路径分散）：

- 数学/STEM：`"Please reason step by step, and put your final answer within \\boxed{}."`
- 代码：`"Write a complete Python program reading from stdin. Wrap it in ```python ... ```."`
  （fn_name 型题目改用函数签名模板）
- 逻辑：各 task 模板 + 统一 `Final Answer: ...` 收尾行

---

## 2. 数据源与域内处理

### 2.1 数学

| 角色 | 数据集 | 规模 | 处理 |
|---|---|---|---|
| 训练主池 | Big-Math-RL-Verified | 25.1 万 | 已是 NuminaMath 的可验证子集，直接作主池 |
| 训练补充 | DAPO-Math-17k | 1.7 万 | verl 原生格式，整数答案，保留 |
| 弃用 | NuminaMath-1.5 原始池 | ~86 万 | Big-Math 已覆盖其可验证部分，回补的只会是不可验证噪声题 |
| 弃用 | OpenR1-Math-220k | 22 万 | 题目与主池 100% 同源（RL 只吃题面+答案，R1 解法用不上） |
| 验证 | AIME 2024+2025(+2026) | ~90 题 | 全部年份留作验证，不进训练 |
| 验证 | MATH 官方 test split | ~5k(常用 MATH-500) | 只用官方 test split |

**问题**：NuminaMath / OpenR1 / Big-Math 三者同源，混用 = 大规模隐性重复。
**方案**：单一数学池 = Big-Math-RL-Verified + DAPO-17k，池内 MinHash 近重去重（词级 5-gram
shingle，Jaccard≥0.6）+ 规范化文本精确去重，来源标签写入 `extra_info.source`。
**逻辑**：Big-Math 已替我们完成 Numina→verifiable 的筛选（proof 题、多选题、无标准答案题已被滤除），
重复建设只会引入噪声。

### 2.2 代码

| 角色 | 数据集 | 规模 | 处理 |
|---|---|---|---|
| 训练骨架 | DeepCoder 训练集 | ~2.4 万 | 已去重、已按 LCB 时间切好（2023.5–2024.7） |
| 训练补充 | CodeContests 官方 train | ~1.3 万 | 官方 split 隔离 |
| 训练补充 | APPS 官方 train | ~1 万 | 官方 split 隔离 |
| 不单独加 | TACO / PrimeIntellect verifiable-coding-problems | — | DeepCoder 训练集 ≈ TACO-verified ∪ PrimeIntellect ∪ LCB 段，重复加入会隐性重复 |
| 验证 | LiveCodeBench 2024.8+ 日期段 | 滚动 | 按提交时间切，不随机划（同题不同版本会同时出现在两边） |
| 验证 | CodeContests / APPS 官方 test | 官方切分 | 直接用 |

测试用例格式统一为 prime_code / sandbox_fusion 消费的
`{"inputs": [...], "outputs": [...] (, "fn_name": ...)}` JSON：

- CodeContests：public + private + generated tests 合并，优先级 public > private > generated；
- APPS：解析 `input_output` JSON 字符串，保留 `fn_name`；
- 每题保留 ≤20 个测试用例（超时控制），0 测试用例的题丢弃；
- 入库前做"题面 MinHash + 测试用例哈希"双重去重（不同源收录同一 Codeforces 原题很常见）。

### 2.3 逻辑/谜题

| 角色 | 数据集 | 规模 | 处理 |
|---|---|---|---|
| 训练 | SynLogic train split | ~3.3 万 | 官方切分 |
| 训练 | PuzzleClone(PC-SL-35K 90%) | ~3.2 万 | 按题目族 cluster 切分，不按行切（防同族变体两边都出现） |
| 训练 | Enigmata-Train | ~1.44 万 | 官方 strict train-eval separation |
| 训练 | Reasoning Gym 程序生成 | 理论无限 | seed 区间硬切分（见下） |
| 训练 | ARC-AGI 1/2 官方 train | 官方切分 | 需单独的网格 prompt 模板 + exact-match reward |
| 验证 | SynLogic test / Enigmata-Eval / ARC-AGI eval | — | 官方切分 |

**问题**：程序生成数据的"污染"不是文本被抓取，而是同一 seed/config 同时进 train 和 eval。
**方案**：seed 区间硬切分 + 登记制——

- train seed ∈ [0, 10⁶)，eval seed ∈ [10⁶, 1.1×10⁶)；
- eval (task, config, seed) 三元组清单落盘 `eval_seeds.json`，训练采样启动时硬过滤（非概率排除）；
- 每条生成样本 `extra_info.seed` 必填，无 seed 不允许入库。

### 2.4 STEM

| 角色 | 数据集 | 规模 | 处理 |
|---|---|---|---|
| 训练 | Dr.SCI 461K rule-verifiable 子集 | 46.1 万 | 来源是网络蒸馏语料，需最严一档去污染 |
| 验证 | GPQA / GPQA-Diamond | 448 / 198 | gated，注意题目文本不外泄 |
| 验证 | SuperGPQA | 26,529 | — |
| 验证 | SciBench | ~700 | — |
| 验证 | HLE(仅 STEM 部分) | ~2,000 | — |

**问题**：Dr.SCI 的污染形式是"转述/讨论"而非原文复制，n-gram 基本无效。
**方案**：embedding 为主 + 高区分度实体（化合物名、罕见常数、专有术语）倒排召回为辅。

---

## 3. 去污染流水线（两级，全自动）

### 一级：n-gram 重叠

8-gram 词级重叠，训练题与任一验证题命中即删。对数学域有效（AIME/MATH 原题混入）。

### 二级：embedding 相似（本地部署，无人工环节）

1. **部署**：本机 vLLM 起 embedding 服务
   ```bash
   vllm serve Qwen/Qwen3-Embedding-8B --task embed --port 8001
   ```
   或离线批量模式 `vllm.LLM(task="embed")` 直接编码。数学 ~27 万 + STEM 46 万条，单卡批编码可完成。
2. **索引**：验证集归一化 embedding 建 FAISS `IndexFlatIP`（归一化后内积 = cosine），
   训练集分批 `search(k=1)`。
3. **阈值自动标定（替代人工抽查）**：
   - 正对照：对验证题做规则化改写（换数值、换变量名、句式重组）生成"必然应删"样本对，测 cosine 分布；
   - 负对照：验证题 × 随机训练题配对的 cosine 分布；
   - 取两分布最优分离点（Youden's J）为阈值，预期落在 0.83–0.88。
4. **执行与审计**：命中即删；删除清单（训练题、命中验证题、cosine）落盘 CSV，
   报告自动汇总各源删除率。全自动但保留完整审计链。

**注意**：OpenR1/Big-Math 只对 MATH-500/AIME2024 去过污染，使用 AIME2025/2026 必须自己重跑。

---

## 4. 难度预筛（pass@k）

**问题**：原始池中大量题对当前模型要么全会要么全错（都是零梯度），DAPO dynamic sampling
虽在线过滤，但每个 step 都在为无用 rollout 付算力。

**方案**：

- 用 Qwen3-4B 基座对数学+逻辑池做 pass@8（temp=1.0），代码池 pass@4（执行贵）；
- 保留 pass_rate ∈ (0, 1)；pass_rate 写回 `extra_info` 作为难度分层依据；
- 全错题保留 5–10% 做探索（与 hard replay 机制衔接）；
- 预算不足时先随机抽 1/3 预筛。

**逻辑**：离线筛一次的成本 ≪ 训练全程反复 rollout 废题的成本，且 pass_rate 先验让
filter_groups 的浪费大幅下降。Big-Math 自带的 `llama8b_solve_rate` 也写入
`extra_info.prior_solve_rate` 作为免费难度先验。

---

## 5. 混合配比（v1 起点）

| 域 | 占比 | 来源 |
|---|---|---|
| 数学 | 45% | Big-Math + DAPO-17k |
| 代码 | 25% | DeepCoder + CC/APPS train |
| 逻辑 | 15% | SynLogic + Enigmata + Reasoning Gym |
| STEM | 15% | Dr.SCI 可验证子集 |

逻辑：数学是 reasoning 迁移性最强的域，占主导；代码推理链长、执行 reward 噪声低，次之；
逻辑/STEM 防过拟合到数学格式。每个 step 内按比例分层采样，跑 1–2 个短程实验后再调比。

---

## 6. 奖励函数映射

| data_source | verifier | ground_truth 格式 |
|---|---|---|
| `math_*` | math_verify（`pip install math-verify`），fallback math_dapo | 字符串答案 |
| `code_*` | sandbox_fusion（正式训练）；prime_code 本地执行（smoke run） | `{"inputs","outputs"(,"fn_name")}` JSON |
| `logic_synlogic` / `logic_enigmata` / `logic_puzzleclone` | 各官方 repo 的 rule verifier 封装 | per-task 结构 |
| `logic_reasoning_gym` | reasoning_gym 库的 `verify()` | 库内 answer 字段 |
| `logic_arc` | 网格 exact match | 二维数组 JSON |
| `stem_*` | MCQ 选项字母 match + 数值题走 math_verify | `\boxed{}` 内答案 |

新 data_source 统一在 `reward/compute_score.py` 扩展，训练配置用
`reward_model.custom_reward_function.path/name` 挂载，不改 verl 源码。

### sandbox-fusion 说明

- **是什么**：字节开源的远程代码执行沙箱服务（`bytedance/SandboxFusion`），verl 通过 HTTP
  调 `/run_code`，在隔离环境执行模型生成的代码并返回结果作 reward 判定。
- **能力**：20+ 语言、编译/运行超时、内存上限、stdin 注入、fn_name 自动包 wrapper
  （`verl/utils/reward_score/sandbox_fusion/utils.py` 已实现）、`max_concurrent=256` 高并发。
- **安装**：verl 环境内零新依赖（只是 `requests.post`）；但需单独部署服务——Docker 本地部署
  （官方文档 get-started → local deployment）或火山引擎 FaaS。端点填入
  `reward_model.sandbox_fusion.url='http://<host>/run_code'`。
- **为什么正式训练必须用它**：
  1. 安全——RL 生成海量不可信代码，直接在 trainer 机器执行等于开放 RCE 面；
  2. 吞吐——官方文档称 reward 阶段省 10–30% 时间；`reward_manager=prime` 可多子进程并行验证；
  3. 环境一致——依赖/超时/内存集中管理，消除 trainer 节点差异带来的 reward 噪声。
- **资源**：max_concurrent=256 对应 64–128 核 CPU 起步，无 GPU 需求。
- **退路**：不配 URL 自动 fallback 到 prime_code 本地执行，仅供 smoke run。

---

## 7. 验证集与评估协议（固定，全程不改）

| 验证集 | 协议 |
|---|---|
| AIME 2024+2025(+2026) | avg@32, temp=0.6, top_p=0.95, 32k 生成 |
| MATH-500（官方 test） | avg@4 |
| LiveCodeBench(2024.8+ 段) | pass@1, temp=0.2 |
| GPQA-Diamond | avg@8 |
| Enigmata-Eval / SynLogic test / ARC-AGI eval | pass@1 |

要点：题量小的集合（AIME ~90 题）方差极大，必须 avg@32 并把均值写进 wandb；所有 ckpt 用同一
评估脚本、同一 decoding 参数，否则曲线不可比。

---

## 8. 训练配置（DAPO + hard replay，保 recall）

**基座**：优先 Qwen3-4B-Base + thinking 模板，避免在已 RL 过的版本上二次 RL。

**DAPO 组件开关**：

| 组件 | 状态 | 说明 |
|---|---|---|
| clip-higher (ε_high=0.28) | 开 | 保住低概率探索 token，直接服务 recall |
| dynamic sampling / filter_groups | 开 | `metric=score`，过滤全对/全零组 |
| token-level policy gradient loss | 开 | DAPO 标准配置 |
| overlong reward shaping | **关** | 当前阶段不开 |
| 长度惩罚 | **关** | 先把 recall 提上来，长度问题后期再治 |

配套：rollout `n=16`、`temperature=1.0`；response 长度课程 8k → 16k → 24k；
prompt 上限数学 2k、代码 4k；`data.dataloader_num_workers=0`（hard replay pool 在 driver 进程）。

**Hard replay（移植自 tool_rl 的成熟设计）**：

- 自定义 sampler 自持 DAPO 过滤（框架不给 custom sampler 注入 `filter_groups`，旋钮走
  `sampler_kwargs`：`filter_metric=score`、`train_batch_size`、`max_replays`）；
- 全零组在 `_clear_groups` 前从 TransferQueue 导出到 driver 进程内 replay pool，
  dedup key 用 `extra_info.task_id`（题面哈希兜底）；
- replay re-entry 走 dataset 侧 `__getitem__`；比 tool_rl 更简单——无 hint 重抽逻辑，
  原始 `raw_prompt + ground_truth` 原样重放即可；
- 分层调度：成功率 ≥50% 出池；1–50% 升中档、每 10 步一放；≤1% 留池、每 20 步一放；
  每步 replay 占比 ≤ `train_batch_size × 0.2`；
- 环境变量：`REASONING_RL_HARD_REPLAY` / `REASONING_RL_REPLAY_RATIO` /
  `REASONING_RL_REPLAY_MAX_FRACTION`（或直接写进启动脚本 sampler_kwargs）。

---

## 9. 实施 Checklist（按依赖顺序）

1. 部署 sandbox-fusion 服务，用 100 条 TACO 题验证 reward 通路（未开始）；
2. 数学/代码域 `to_parquet` 转换脚本（已完成）；
3. 同源合并 + 池内 MinHash 去重（已完成，内置于转换脚本）；
4. 去污染流水线脚本（已完成；待验证集 parquet 就绪 + embedding 服务启动后真实运行）；
5. 逻辑/STEM 域 `to_parquet` 转换脚本（已完成；SynLogic easy+hard / Enigmata 36 task /
   Dr.SCI verifiable 子集已真实探查 schema 并小规模跑通）；
6. 逻辑/STEM 域 verifier 接入 `reward/compute_score.py`（已完成，含 21 条单元测试）；
7. pass@k 难度预筛脚本 + 45/25/15/15 混合脚本（已完成；预筛待 GPU 真实运行）；
8. 训练脚本 `run_qwen3_4b_reasoning_rl_dapo.sh`（已完成：DAPO + hard replay，
   sampler 复用 tool_rl 实现）；
9. 小规模 smoke run（prime_code 本地 reward）验证 reward 分布与 filter 比例，再进正式训练。
