# Agentic RL v2 设计（Qwen3-4B，10–15 轮）

目标：在 stage-0 单轮工具调用 RL 之后，强化 **工具调用准确性** 与 **10–15 轮 agentic 能力**（规划、反思、多轮格式、工具错误恢复），同时保证训练稳定、reward 不可被 hack、环境零服务依赖。

## 0. v1 的主要问题（本次改动的出发点）

| 问题 | 影响 |
|---|---|
| 多轮数据以 replay 为主（FunReason-MT / APIGen-MT / Toucan），首个偏离即终止 | 只能学"复现参考调用"，惩罚等价的其他方案；**无法学错误恢复**（出错后没有下一轮） |
| stateful mock_fc 已实现，但没有任何数据源生成 stateful 任务（`env_mock_fc(impl="stateful")` 写的 `domain="synthetic"` 不存在） | 结果导向的多轮 reward 实际从未被使用 |
| search 硬上限 8 轮，Search-R1 NQ 是 1–2 跳 | 达不到 10–15 轮；且需要 Wikipedia E5 索引 + 检索服务（几十 GB、GPU/CPU faiss） |
| search_qa：错误答案仍拿 0.1 格式分；F1 部分分可被"罗列候选答案"刷 | reward 下限抬高、GRPO 优势被压缩；答案堆砌 hack |
| agent loop：工具调用 JSON 解析失败时直接当成最终回答结束 | 学不到"调用格式错了→修正" |
| 单轮生成无上限 | 一次失控的 `<think>` 吃掉整个多轮预算 |

## 1. 数据：来源、配比、数量

配置：`configs/mixture_v2.yaml`；一键构建：`scripts/build_data_v2.sh`。

| 桶 | 占比 | 来源 | 训练什么 |
|---|---|---|---|
| multi_turn | **45%** | procedural_mt 75%（自研生成器）+ Toucan 15% + APIGen-MT 10% | 长程规划、先查后写、错误恢复、并行调用 |
| search_qa | 25% | multihop_local：HotpotQA-distractor / 2Wiki / MuSiQue-ans（2–4 跳） | 查询规划、改写、证据链、反思 |
| single_turn | 20% | APIGen 35% / ToolACE 25% / Seal-Tools 15% / Hammer 无关负样本 25% | 防遗忘：工具选择与参数准确性、该不调用时不调用 |
| math_ci | 10% | 已有推理 RL 数学池 + python 工具 | 推理保持、代码报错后修正 |

**数量**：默认 `total_train=6000`，held-out 900（每桶 200–300）。batch 64×n8 时约 94 step/epoch；配合 DAPO 过滤，有效训练约 150–250 step，对 4B 规则 reward 一般已到饱和区。所有来源是生成的（无限）或原始量 >20k，可用 `--total-train` 线性放大而不改份额。原始数据建议取需求量的约 3 倍（`raw_sources_v2.example.json`），给去重 / 去污染 / 通过率过滤留余量。

**procedural_mt 生成器**（`generators/stateful_tasks.py`，参考 τ-bench / BFCL-v3 / AgentGym，但完全本地生成）：

- 3 个域、13 个模板：filesystem（按关键词归档、备份后编辑、清理并计数、按扩展名整理并写索引）、trading（调仓、条件下单、补足差额后转账）、retail（取消、换货、退货并报退款、改地址、复合请求）。
- 难度 = 计划长度：GT 计划 4–15 步，约 26% ≥10 步；`max_turns = min(15, 步数 + 1 + slack 3 + 故障数)`。
- **故障注入**约 25%：某个写工具第一次调用返回 `503 ... Please retry`，且**不修改状态**。不重试就无法完成目标，所以学到的是"读错误→重试"，而不是去奖励"制造错误再恢复"。
- 必要信息（订单号、文件路径、价格）只存在于工具返回中，迫使模型先读后写；独立的读操作有 35% 以并行组形式出现。
- **每条任务在产出前都会过真实 env + reward 自检**（`verify_task`）：GT 计划必须得 1.0；什么都不做必须失败；需要回答的任务，不报答案必须失败；有故障的任务，去掉重试那一步必须失败。1000 条生成 0 条被拒，2 条超长丢弃。

**multihop_local**：每题自带 10–20 段候选段落（含 2–4 段 gold 和强干扰段），直接放进 `env.config.docs`，由 stdlib BM25 在 rollout worker 内检索，不需要索引和服务。为避免一次宽泛查询就把所有段落搜出来，设 `top_k=2`、`snippet_chars=240`；过滤掉 yes/no 题（不搜也有 50% 猜对）、不可回答题和长答案题。

## 2. Reward：审查、补充与防 hacking

所有 reward 范围 [0,1]，无 judge 模型。

**mt_fc（stateful，`process_term=efficiency`）**：`0.8·result + 0.1·efficiency + 0.1·format`

- result：env 终态断言全部通过，且（需要时）`<answer>` 匹配。断言同时检查目标和**非目标对象不变**（干扰文件内容、其他订单的状态和地址、余额精确值），所以"全部取消 / 全部删除"会失败。
- efficiency：**只在成功时发放**。轮数 ≤ 期望 + 2 时满分（重试和一次验证性读取不扣分），之后线性降到 `max_turns` 时为 0。不再按"和唯一 GT 计划的相似度"打分，等价的其他方案不受罚，也不奖励模仿。
- 没有任何工具调用、任务又失败时，format 项清零，避免"直接空答"凭格式分高于诚实但失败的尝试。
- 超时一律不给 result（v1 已有，保留）；未声明的工具由 env 拒绝并计入 `invalid_calls`。
- 不设"恢复奖励"：奖励恢复本身会诱导模型先制造错误。

**最终答案**（`answer_tag_matches`）：只认最后一个 `<answer>...</answer>`；数值容差 0.005，支持 `$`、千分位、`%`；文本做归一化后比较。答案写在标签外、或标签内列多个候选，都不算。

**search_qa**：`0.9·answer + 0.1·format`

- EM=1 得满分；否则部分分 = 0.8·F1。预测长度超过 max(6, 3×最长 gold) 个 token 时**不给部分分**，防止答案堆砌。
- answer 为 0 时 format 也为 0：去掉 0.1 的地板分，错误答案不再高于不回答。

**format**：新增 `unparsed_tool_call`，即出现 `<tool_call>` 标记但没解析出调用的那一轮判为格式不合法。

**单测**（`tests/test_v2_agentic.py`，17 个）覆盖以下 hack 探针：空轨迹、只调用不回答、标签外堆答案、标签内多候选、"全部取消"、调用未声明工具、刷轮数、超时、跳过重试。

## 3. 环境：轻量、零服务

| 环境 | 依赖 | 单步开销 |
|---|---|---|
| stateful mock_fc（filesystem / trading / social / **retail 新增**） | 纯 Python，状态随任务 JSON 下发 | 微秒级，无共享状态，可任意并发 |
| search（inmemory，per-task 语料） | stdlib BM25 | 约 0.1 ms |
| python（math_ci） | subprocess + rlimit | 约 50 ms |

参考的业界做法：τ-bench（retail 域的读写工具与"其他订单不变"类断言）、BFCL-v3 multi-turn（状态机 + 终态检查）、AgentGym-RL / RAGEN（短到长的课程、按结果打分）、Search-R1 / ReSearch（多跳检索 + EM），以及 HotpotQA-distractor 自带上下文、免检索服务的做法。v1 的 E5 检索服务和 code_tests 仍然可用，只是不再出现在 v2 配比中。

search 轮数上限改为可配（默认 8，硬上限 12）；multihop 任务用 10。

## 4. 训练 recipe

`scripts/run_qwen3_4b_agentic_v2.sh`（单机 8 卡），分三阶段课程，用 `resume_mode=auto` 依次接续：

| STAGE | 数据 | max_turns | response | 累计 step |
|---|---|---|---|---|
| 1 | GT ≤8 轮子集 | 8 | 12k | 40 |
| 2 | GT ≤12 轮子集 | 12 | 20k | 90 |
| 3 | 全量 | 15 | 24k | 160 |

关键设置及理由：

- GRPO，n=8，batch 64，mini-batch 16，ppo_epochs=1，lr 1e-6，grad clip 1.0。
- **DAPO 组过滤**（`filter_groups.metric=agentic/reward_score`）：全 0 / 全 1 组没有梯度信号，由 V1 replay buffer 补充新 prompt。
- `loss_agg_mode=token-mean`，clip 0.2/0.28（clip-higher，防熵塌缩）；low_var_kl 系数 0.001，作为防止格式漂移的弱锚点。
- 工具和 env 的 token 不参与 loss（`response_mask=0`）；单轮生成上限 `AGENTIC_MAX_TOKENS_PER_TURN=3072`。
- `rollout_is=token`，阈值 2.0：多轮长序列下 vLLM 与 FSDP 的 logprob 偏差会累积，用截断 IS 兜住。
- thinking 开启：反思和规划放在 `<think>` 里。
- agent loop：工具调用解析失败时，返回一条格式错误观察，让模型修正后重试，而不是直接结束 episode。

**训练中要盯的指标**：`agentic/reward_score` 均值及按 task_type 拆分；`agentic/num_turns`（不应持续上涨，否则说明在刷轮数）；`agentic/parse_errors`（应快速下降）；`agentic/reward_timeout` 比例（超过 20% 时提高 response 长度，或推迟进入下一阶段）；DAPO 过滤比例（全 1 组超过 50% 就该进入下一阶段）；actor 熵与 KL；held-out 上每个 domain/template 的成功率。

**评测**：BFCL v3 multi-turn 和 v4 agentic（已做去污染）、τ-bench retail（与 retail 生成器同构，需注意过拟合，以 airline 为 OOD）、HotpotQA / MuSiQue dev EM，以及 stage-0 单轮测试集（检查遗忘）。

## 5. 已知限制 / 待验证

- 本次改动只做了 CPU 侧验证：单测、数据端到端（生成→混合→parquet 导出）、hydra 键名核对。**尚未在 GPU 上跑过训练**，step 数、response 长度、单轮 token 上限都是经验初值，需要先 `SMOKE=1` 冒烟。
- procedural 任务模板有限（13 个），模型可能学到模板特征。缓解手段：Toucan / APIGen-MT 提供工具 schema 多样性，评测看 OOD（τ-bench airline、BFCL）；后续可增加域，模板写法参照现有生成器即可。
- multihop_local 语料很小，检索比真实 Web 容易。如需更强的检索能力，可把部分任务切回 v1 的 E5 服务（`search_r1`、`asearcher`）。
- 部分数据源 license 尚未核实（沿用 v1 的 `[CONFIRM]` 标记）。
