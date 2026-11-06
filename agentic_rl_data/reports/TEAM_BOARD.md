# TEAM BOARD — agentic_rl_data

> Lead 独占维护（其他人只读）。每次收到席位汇报、做出跨席位决策后更新。
> 最新一次更新：2026-11-05 23:10 (+08:00) · Lead · 契约版本 **schema v2.3 / CONTRACTS v2.3**（5 个任务卡全部 completed）

## 0. 目标（用户指定）

按 `C:\Users\Administrator\Downloads\agentic_rl_data_plan.md` 产出
**Agentic RL 初期训练数据**的实现：统一 schema → 多来源 loader → 轮数/去重/去污染/通过率过滤
→ 配比与 held-out → 每种 task_type 的 reward 与 env → 报告 + 单测。
目标评测面为 BFCL v3 multi-turn + BFCL v4 agentic（web search / memory）+ 单轮 + hallucination。

Phase 1（本轮）范围：**框架无关骨架 + 规则 reward + 环境接口 + 数据/配比调研结论**，
不追求真的下载全量数据（无 shell），但要给出可直接执行的脚本与验证命令。

## 1. 花名册

| 席位 | 成员 | 任务卡 | 写入范围（独占） |
|---|---|---|---|
| 协调 | `lead`（我） | — | `schema.py`、`README.md`、`__init__.py`、`reports/TEAM_BOARD.md`、最终集成 |
| 数据调研 | `data-researcher` | T1 | `loaders/**`、`scripts/{dedup,decontam,filters,report}.py`、`reports/data_sources.md` |
| 配比调研 | `mixture-researcher` | T2 | `configs/**`、`scripts/{build_mixture,filter_by_passrate}.py`、`reports/mixture_analysis.md` |
| 写 Reward | `reward-writer` | T3 | `rewards/**`、`tests/test_rewards*.py` |
| 环境构建 | `env-builder` | T4 | `envs/**`、`tests/test_envs*.py`、`reports/env_setup.md` |
| 检查/提问 | `critic` | T5 | `reports/review_r1.md`、`reports/open_questions.md`、`reports/acceptance_matrix.md`、`tests/test_acceptance_contract.py` |

## 2. 依赖图（谁等谁）

```
schema.py (Lead, 已冻结 v1)
   ├── loaders (T1) ──→ build_mixture (T2, 软依赖：接口按契约先写)
   ├── rewards (T3, 只依赖 schema，可与 T1 并行)
   └── envs    (T4, 独立)
T5 critic：不阻塞，先出「验收矩阵 + 提问清单」，随后增量审查 T1–T4 的产物
```

硬依赖只有一条：**`schema.py` 变更必须先走 Lead**，否则 T1/T3/T4 会各自漂移。
T2 的 `build_mixture.py` 对 T1 的 loader registry 是软依赖——按契约先写，最后联调。

## 3. 已冻结的契约（schema v2.2 / CONTRACTS v2.2）

1. `Task` / `Trajectory` / `reward` 返回结构以 `schema.py` 为准（含 `ENV_FOR_TASK_TYPE`）。
2. `gt_calls` 按轮分组；`[]` = 标注为"无需工具"；**禁止 `None`**（`validate_task` 会报错）。
3. 精确去重 key = `task_dedup_key`（用户消息归一化 + 工具签名 sha256）；近似去重用 `text_shingles`。
4. 所有 JSONL 写出用 `dump_jsonl`（sort_keys + 非 ASCII 保留）→ 重复运行字节一致。
5. reward 是纯函数；env `reset/step` 无全局状态；loader 只吐 schema-valid task。
6. 评测/训练侧接口：**本项目用 verl**（用户 2026-11-06 明确），plan §9 的四个接口已映射到
   RLHFDataset / 自定义 agent loop `agentic_agent` / custom reward / `multi_turn.format`，
   见 `reports/VERL_INTEGRATION.md`；slime 相关内容仅作历史记录。

## 4. 跨席位信息（Lead 转发）

- **环境限制**：本机 `pwsh` 起不来（`0xC0000142`），pytest/git/下载都跑不了。
  → 单测照写但标注"未执行"；报告里必须给用户可复制的验证命令。
- **数据现状**：仓库已有 `examples/tool_rl/prepare_data.py` / `prepare_data_v2.py`
  处理过 apigen / sealtools / toolace / hammer / apibank 五源（单轮），T1 应优先复用其解析逻辑。
- **文档中的 `[CONFIRM]` 项**（HF ID / 字段 / license / 轮数）由 T1 逐条核实并落表；
  核实结论必须同时通知 T2（影响配比上限）与 T5（影响验收矩阵）。

- **数据源核实进展（T1，2026-11-05 21:40）**：
  - `bfcl-eval` PyPI 最新 **2026.3.23**（文档写的 2025.10.27.1 已过时）→ 去污染数据源按 pin 版本读取，脚本支持"包内 data 目录 / 用户给路径"双入口。
  - Search-R1 实际数据是 HF `RUC-NLPIR/FlashRAG_datasets` config=nq（文档猜的 `PeterJinGo/nq_hotpotqa_train` 不准）；loader ID 可配置。
  - ASearcher 训练数据仓库 `inclusionAI/ASearcher-train-data`；原作轨迹 **128 轮** → ≤15 轮过滤为硬门槛，否则长轨迹拖垮配比。
  - Toucan-1.5M 是首期 `mt_fc` **真实工具返回的主力**（其余 MT 源多为 placeholder）；FunReason-MT 等 placeholder 率待用户首跑后填报告。
  - tau-bench 无现成轨迹（needs-synthesis）不进 placeholder 池；When2Call 只取 train，test 由 BFCL v2 Live 派生、绝不入训练。
  - **本沙箱 HF（huggingface.co / hf-mirror / datasets-server）不可达**：HF 卡片的 license/条数一律标 `[UNVERIFIED]` 并给用户侧核实命令；GitHub/arXiv/PyPI 可核实的标 `verified`。

## 5. 决策记录

| # | 决策 | 理由 |
|---|---|---|
| D1 | Phase 1 先做框架无关实现；**2026-11-06 修正：训练框架 = verl**（D19 补齐四个接口映射） | 最初 §9 接口 TODO、猜接口会返工；用户明确用 verl 后转为 verl 适配 |
| D2 | 写 5 个席位而不是合并配比进数据调研 | 配比需要独立消融设计与 held-out 策略 |
| D3 | 新代码全部隔离在 `agentic_rl_data/`，不碰 verl 既有代码 | 避免污染训练仓库、便于用户单独迁移到 slime |
| D4 | `tools` 用 OpenAI 风格 JSON schema 单一表示 | 唯一可跨源对齐的中间格式（plan §2） |
| D5 | env↔trajectory↔reward 字段契约 v1 冻结（`CONTRACTS.md`），不改 `schema.py` 结构 | env 的 done-info 原样进 `trajectory.env_state`；reward 只读 trajectory，避免双真相源 |
| D6 | `code_exec` 测试结果唯一真相源 = `trajectory.env_state["code_tests"]`（原 `extras.tests` 口径已由 D9 取代）；`info["tests"]` 仅日志 | 防止 reward 与日志口径漂移 |
| D7 | 回放缺 gt 返回时允许占位，但必须标 `extras.gt_returns_source`，placeholder 样本过滤/降权 | 静默降质会让 reward B 分支失去意义且指标虚高 |
| D8 | **schema v2**：`reference` 7 键必填（值可 null）；`final_answers` / `final_state` / `tool_returns` 入契约；score 冻结 `[0,1]`；`tools_signature` 按 name 排序 | critic P1/P2 + reward-writer 字段需求一次性裁决，避免各方来回对齐 |
| D9 | `code_exec` 结果键从 `env_state.extras.tests` 迁至 **`env_state["code_tests"]`**（`all_passed` 权威） | code_exec 专属命名空间，语义更窄、单个真相源 |
| D10 | `mt_fc` A 方案结果依据优先级：`goal_satisfied` → `reference.final_state` vs `env_state.state` → `final_answer` EM → `replay_ok` | env 是终态权威验证方；reward 不重复实现状态机验证 |
| D11 | 数据侧静态声明 `meta["gt_returns_source"]`（config/reference/gt_turns/placeholder），与 env runtime `extras["gt_returns_source"]` 双标记 | T2 必须在不运行 env 的前提下过滤 placeholder；runtime 为准、静态可审计 |
| D12 | search `env.config` 规范 = `{backend: e5/inmemory/http, url, top_k, max_rounds, retriever}`；T1 产规范形式，T4 过渡期兼容旧别名（`local_wikipedia_e5`→e5、`retriever{url,topk}`→url/top_k） | critic 发现 T1 loader 与 T4 SearchEnv 互不兼容会让 25% 搜索数据直接 EnvError；双端兼容降低联调失败 |
| D13 | `meta.group_id`（str|null）进 schema v2.2 可选键，供 group-aware held-out；缺失时报告标 `[NO-GROUP]` | T2 的会话级切分承诺不能依赖未定义字段 |
| D14 | `abstain` 首期**选 (b)**：由 hammer irrelevance 承担（400 条 = 单轮类 25%）；When2Call/abstain 只做 reward/env 单测、不进训练（When2Call test 由 BFCL v2 Live 派生，有泄漏风险且 license 未核）；config 已预留 (a) 调整注释 | plan §6 的 20% 含拒答；零来源会与 BFCL hallucination 目标脱节 |
| D15 | **CONTRACTS v2.3**：冻结 `env_state["code_tests"]["infra_error"]`（infra≠答错）、`extras.{gt_returns_missing, gt_returns_declared_source, gt_returns_missing_ratio}`、`env.config["task_deadline_s"]`（默认 30s） | T4 收口时的跨席位新增字段，须一次广播给 T2/T3/T5 |
| D16 | 收口补写 `scripts/report.py` + `tests/test_report.py`（Lead）：数据报告生成器，`--strict` 校验去污染 0 命中 / replay 0 失败 / 无 split 重叠 / schema 全合法 / sha 可复现；缺失上游报告渲染 `[NOT RUN]` | plan §3 step 8 / §8 A4 之前只有规划没有实现（critic 矩阵标 🟡），是最后一个验收缺口 |
| D17 | **去污染语义修正（critic R2-01）**：`decontam.py` 新增 `residual_hits`（kept 再匹配，门禁必须 0）；`total_dropped`/`hits_removed` 降为审计列；`report.py --strict` 只判 residual + BFCL 索引非空，并新增 `--allow-missing-steps` / `--require-provenance` | 原实现把"已剔除命中"当"残留命中"，会把正确剔除判失败、把残缺 BFCL 语料判通过 |
| D18 | 新增一键入口与启动脚本（Lead，2026-11-06）：`scripts/run_pipeline.py`（数据 8 步编排 + manifest + `--dry-run`）、`configs/pipeline.json`、`build_data.ps1/.sh`、`serve_search.ps1`（E5 建索引 + 起服务）、`launch_training_slime.sh`（§9 未定义接口前 fail-closed）、`smoke.ps1` | 之前只有单步脚本，没有"一条命令建数据"和"启动脚本"；slime 接口是用户侧 TODO，不能猜 |
| D19 | **目标框架修正为 verl**（用户明确）：plan §9 四项 → ①`export_verl_parquet.py`（RLHFDataset parquet，`extra_info.agentic_task` 存整条 task）②`agentic_rl_data/verl/agent_loop_agentic.py`（注册 `agentic_agent`，每样本 tools + env 驱动，loop 内算 `reward_score`）③`scripts/verl_reward_bridge.py`（单轮类型 custom reward）④`multi_turn.format`（默认 hermes）；启动 = `scripts/launch_verl_agentic.sh`（`SMOKE=1` 冒烟）。slime 脚本标记为仅历史参考 | 用户所有背景是 verl；原 §9 是 slime 口径，继续按 slime 猜接口会做错方向 |
| D20 | **st_fc 格式门修正**（verl 接入暴露）：`format_report(expect_tool_results=False)`；st_fc 是单轮预测（env.type none，无 tool_results），不再因"calls 无配对结果"判 0；call 级检查保留 | 之前 env-none 的 st_fc 只要 emit 正确调用就被 format gate 判 0，等于该类型数据在 verl 里全是负样本（CONTRACTS C14） |
| D21 | `launch_verl_agentic.sh` 默认 `USE_V1=0`（走 V0 async）：verl V1/TQ trainer 的 `extra_fields` 序列化路径未验证，先跑通 V0 再切 V1 | T6 明确 V1/TQ 路径未验证；默认值应选择已验证风险最低的组合 |

## 5b. 契约冻结：env ↔ trajectory ↔ reward（v2.2，详见 `CONTRACTS.md`）

- 所有 env 终止 `info["env_state"]`（含 `type/impl/goal_satisfied/replay_ok/failed_turn/matched_turns/gt_turns/checks/state/rendered/num_turns/timeout/extras`）**原样**写入 `trajectory["env_state"]`；未用字段为 null。
- `mt_fc`：A 方案 `goal_satisfied`（0.8 主项，env 权威）→ 否则 `reference.final_state` vs `env_state.state` 深比较 → 否则 `final_answer` EM → 否则 `replay_ok`；B 方案 `replay_ok` + `matched_turns/gt_turns`（0.8 主项 + 0.1×比例 + 0.1 格式），结果项严格占主导。
- `code_exec`：唯一测试结果源 `trajectory["env_state"]["code_tests"]`（{total,passed,failed,all_passed,failures,exit_code,duration_s,truncated}）；`info["tests"]` 仅日志。
- schema v2：`reference` 7 键必填（`final_answer/final_answers/gt_calls/gt_turns/tool_returns/unit_tests/final_state`，值可 null）；reward score 冻结 `[0,1]` 并由 `validate_reward_result` 强制；`tools_signature` 已按 name 排序（修复同工具集重排漏去重）。
- `search` 默认：`max_rounds=8`、`top_k=3`（Search-R1 设置），课程值经 `reset(max_turns=)` 取 min；超时/重试参数由 T4 给 T2 做吞吐假设。
- `placeholder` 回放：`extras.gt_returns_source == "placeholder"` 的 `mt_fc` 样本，T2 过滤或降权并在报告给过滤前后条数。

## 6. 待决问题（谁回答）

- `[Q1]` BFCL v3/v4 去污染数据来源：`bfcl-eval` pip 包内置 vs gorilla 仓库目录？（T1 核，T5 复核）
- `[Q2]` 通过率过滤的采样后端：slime rollout 还是先留 `--pass-rates` 文件接口？（T2 设计，Lead 决策）
  默认按文件接口实现，rollout 耦合留 TODO。
- `[Q3]` search env 的 Wikipedia 索引规模与 E5 模型版本？（T4 给建议，T2 据此估吞吐）
- `[Q4]` 首期是否真的需要 `code_exec`（APPS/CodeContests 许可证与测试质量待核）？（T1/T2）

## 7. 席位状态（22:20 快照）

| 席位 | 状态 | 交付 |
|---|---|---|
| T1 data-researcher | ✅ 完成（task-1） | 15 源 loaders（5 现有 + search_r1/asearcher/a2search + funreason_mt/apigen_mt/toolmind/toucan + when2call/retool/code_exec）、build_raw/filter_rounds/dedup/decontam、单测、`data_sources.md`；P0/P1 全关 |
| T2 mixture-researcher | ✅ 完成（task-2） | `configs/mixture.yaml`、`build_mixture.py`（单命令可复现）、`filter_by_passrate.py`（N 噪声表）、`mixture_analysis.md`；caps 线性缩放 + 静态 placeholder 过滤已修 |
| T3 reward-writer | 🟡 进行中 | 7 类 reward + base + 单测；timeout / placeholder / 空 gt 组已修 |
| T4 env-builder | ✅ 完成（task-4） | base/mock_fc（stateful 3 域 + replay）/search/python_env/code_tests、replay_check、env 单测、`env_setup.md`；timeout 终态、别名兼容、task deadline、infra_error 已修 |
| T5 critic | ✅ 完成（task-5） | 验收矩阵、39 条开放问题、review_r1、契约测试（未执行） |

## 8. T1 placeholder 结论（回答 T2 的配额问题）

- 已接入源**没有**"必然 placeholder"者：funreason_mt / apigen_mt / toolmind 为条件性（SFT 轨迹无 observation → placeholder）；Toucan 预期 `reference`（真实 MCP 返回，MT 真实返回主力）。
- 若三个 HF MT 源全 placeholder，真实 MT 只剩 Toucan 子集 → T2 应对 placeholder **降权而非硬删**（30% 配额硬删会不足）；比例待用户首跑 `_build_report.json`。
- 每源真实条数**无实测**（沙箱无 shell / HF 不可达）：机制已交付（`SourceStats` rows/emitted/skipped_by_reason + `_build_report.json`），用户侧一条命令可得；知识基线（非实测）5 源 = 4900 / 4873 / 4033 / 2980 / 161。
- 运行方式注意：T1 脚本按 `python agentic_rl_data/scripts/<name>.py` 运行（内部 bootstrap sys.path）；若要 `python -m agentic_rl_data.scripts.<name>`，需 Lead 补 `scripts/__init__.py`（收口时统一加）。

## 9. 未关闭问题跟踪（review_r1 @22:05，Lead 派单）

- [P0] search loader ↔ SearchEnv 不兼容 → D12 双端兼容（T1 已产规范 config，T4 加别名）— 待 T4 复核
- [P0] timeout 仍可拿 0.8 → T3/T4 已修（`mt_fc` basis=timeout；`envs/base` timeout 强制 `goal_satisfied/replay_ok=False`）
- [P1] placeholder 仍发 0.8 → T3 已修（degraded fail-closed）
- [P1] per-turn 空 gt 组 → T3 已修（含空组对齐）
- [P1] decontam 假 0 → T1 已修（空索引 raise、Jaccard 0.90、全量命中计数 + 每源 50 条样本）
- [P1] caps 绝对值 → T2 处理中；[P1] abstain 零来源 → T2 待选方案 (a) 启用 When2Call / (b) hammer irrelevance 承担
- [P1] search 无任务级 deadline / code_tests infra 失败不可区分 → T4 处理中
- [P1] `meta.group_id` 未入契约 → Lead 已补 schema v2.2（str|null）+ D13
- [P2] funreason_mt/apigen_mt/toucan docstring 复数 → T1 已改单数

**T4 残留风险（收口声明）**：① python/code_tests 仅子进程 + rlimit + 超时，无真实网络/FS 隔离（生产需 bwrap/firejail/容器）；② E5/faiss 依赖与 search 吞吐未验证，参数均 `[SUGGEST]`；③ `replay_check` 在 `rewards.get_reward` 缺失时降级为 replay/终态判定并标 `reward_used=False`；④ stateful `mock_fc` 需要 loader 写完整 `reference.final_state`（filesystem={files,dirs} 等）；⑤ BFCL 其余域（vehicle/travel/智能家居）待按 `BaseDomain` 扩展。

## 10. T2 配比结论（22:31）

- 基线 `total_train=8000`：单轮 **1600**（旧池 9.4%）/ 多轮 **2400** / 搜索 **2000** / 数学+CI **1200** / 代码 **800**；cap 按 `total_train` 线性缩放（stats 记 `cap_declared` / `cap_scaled`）。
- 搜索来源权重：`search_r1` 0.75 / `asearcher` 0.25（ASearcher 128 轮 + 无逐题轮数标注 → 降权、cap 600）。
- placeholder 敏感性（多轮目标 2400，网格 r=placeholder 比例）：`drop` → r=25% 时 2325(29.1%)、r=50% 时 1850(23.1%)、r=75% 时 1375(17.2%)；`keep(share .25, weight .5)` 可保配额但引入 22.9–24.7% 低质样本。默认 `drop`；建议 r≤25% 用 drop，25–50% 跑两套消融。
- pass-rate N=8：SE=.177、p=.05 误删 66.3%；建议 N≥100（SE≤.05）或边界任务二阶段 N=16/32；band 默认 `[0.125, 0.875]`。
- `abstain` 首期由 hammer irrelevance 承担（D14=(b)），When2Call 仅单测不进训练。
