# 数据源核实报告（T1 / data-researcher）

> 对应规格：`C:\Users\Administrator\Downloads\agentic_rl_data_plan.md` §1（数据源）、§2（schema）、§3（步骤 1–4）、§10（风险）。
> 冻结契约：`agentic_rl_data/CONTRACTS.md` C6/C9/C10；`schema.py` v2.2。
> 核实时间：2026-11-05 (+08:00)。**核实方式见 §0：HF Hub 在本沙箱不可达**，因此凡只依赖 HF 数据集卡片的信息一律标 `[UNVERIFIED]`，并给出用户侧可执行的核实命令；GitHub / arXiv / PyPI 可达，相关结论标 `verified`。

## 0. 核实方法与可信度分级

| 级别 | 含义 | 依据 |
|---|---|---|
| `verified` | 本沙箱内可读取的权威来源直接确认 | `api.github.com` / `raw.githubusercontent.com` / `arxiv.org` / `pypi.org` |
| `verified-indirect` | 官方仓库 README/代码中明确给出该 HF ID，但 HF 卡片本身未能打开 | GitHub 官方 README、官方预处理脚本 |
| `[UNVERIFIED]` | 仅来自 plan 文档或社区转述，HF Hub 不可达、无 GitHub 佐证 | 需要用户在有网环境核实 |

**环境限制（已验证）**：`huggingface.co`、`hf-mirror.com`、`datasets-server.huggingface.co` 在本会话全部 `fetch failed`；`web_search` 无 API key 不可用。可访问：GitHub（API + raw）、arXiv、PyPI。因此本报告**不默认信任 plan 里的任何 HF ID**：

- 能通过官方 GitHub README / 官方预处理脚本交叉确认的 → `verified-indirect`；
- 只有 plan 一处提到的 → `[UNVERIFIED]`；
- 仓库自身 license 用 GitHub API 的 `license.spdx_id` 核实（这是**代码/仓库 license**，数据集卡片 license 可能不同，报告里分开写）。

用户侧统一核实命令（有 shell/网络的环境执行）：

```bash
python - <<'PY'
# 1) 数据集是否存在 + 条数 + 字段 + license
from huggingface_hub import HfApi
api = HfApi()
for repo in [
    "Bingguang/FunReason-MT", "Salesforce/APIGen-MT-5k", "Agent-Ark/Toucan-1.5M",
    "Nanbeige/ToolMind", "Team-ACE/ToolACE", "MadeAgents/xlam-irrelevance-7.5k",
    "Salesforce/xlam-function-calling-60k", "PeterJinGo/nq_hotpotqa_train",
    "inclusionAI/ASearcher-train-data", "JoeYing/ReTool-SFT", "nvidia/When2Call",
    "BAAI/TACO", "codeparrot/apps", "deepmind/code_contests",
]:
    try:
        info = api.dataset_info(repo)
        print(repo, "| license:", (info.card_data or {}).get("license"), "| tags:", info.tags[:3])
    except Exception as exc:
        print(repo, "| ERROR:", exc)
PY

# 2) 条数 + 字段（逐个，按需替换 repo / config）
python - <<'PY'
from datasets import load_dataset
ds = load_dataset("Bingguang/FunReason-MT", split="train")
print(len(ds), ds.features)
print(ds[0])
PY

# 3) 已安装 bfcl-eval 的版本 / 数据目录
python -c "import importlib.metadata as m; print(m.version('bfcl-eval'))"
python -c "import bfcl_eval, pathlib; print(pathlib.Path(bfcl_eval.__path__[0]) / 'data')"
```

---

## 1. 现有 5 个单轮源（plan §0 表，已被 `examples/tool_rl/prepare_data*.py` 消费）

| source | 原始 ID | license | 原始/已用条数 | 字段（legacy 解析依赖） | 转 RL | 状态 |
|---|---|---|---|---|---|---|
| apigen | HF `Salesforce/xlam-function-calling-60k`（文件 `xlam_function_calling_60k.json`） | HF 卡片 `[UNVERIFIED]`；代码侧无仓库 | raw 60k / 已用 4900 | `id`, `query`, `tools`(JSON str), `answers`(JSON str) | 直接 → `st_fc`（`answer` 为空则不可用，需丢弃） | ID `verified-indirect`（`nvidia/When2Call` 官方 README 引用该 ID 作为训练数据来源）；license/count `[UNVERIFIED]` |
| sealtools | GitHub `fairyshine/Seal-Tools`，`Seal-Tools_Dataset/{tool,train}.jsonl` | **Apache-2.0**（GitHub API 核实） | ~4k tools / 已用 4873 | tool: `api_name`,`api_description`,`parameters`,`required`,`field`；train: `id`,`query`,`calling[{api,parameters,responses}]` | 直接 → `st_fc`；嵌套调用（参数含 `API_call_N`）必须丢弃（无执行环境） | `verified` |
| toolace | HF `Team-ACE/ToolACE`（无同名 GitHub 仓库，`Team-ACE/ToolACE` 404） | HF 卡片 `[UNVERIFIED]`（第三方 fork 有 Apache/MIT 混杂，不可作为依据） | raw 大 / 已用 4033 | `system`(内嵌 tools JSON), `conversations[{from,value}]` | 多轮对话按 user turn 拆成单轮 `st_fc`；`[func(k=v)]` 解析，无法解析的 call-shaped 行要拒跑（loader 有 >5% 守卫） | ID `verified-indirect`（legacy 代码长期使用）；license `[UNVERIFIED]` |
| hammer | HF `MadeAgents/xlam-irrelevance-7.5k` | HF 卡片 `[UNVERIFIED]` | raw 7.5k / 已用 2980 | `query`,`tools`,`answers`（空 = 负样本） | 负样本 → `irrelevance`；有答案 → `st_fc` | ID `[UNVERIFIED]`（仅 plan/legacy 代码）；license `[UNVERIFIED]` |
| apibank | GitHub `AlibabaResearch/DAMO-ConvAI`（`api-bank/`） | **MIT**（GitHub API 核实） | 已用 161（level-1 dialogues） | API 类：`description`/`input_parameters`；对话：`{"role":"User|AI|API","api_name","param_dict","result"}` | 对话按 API turn 拆单轮 `st_fc`；加 5 个干扰工具 | `verified` |

**结论**：5 个源的解析逻辑全部在 `agentic_rl_data/loaders/{apigen,sealtools,toolace,hammer,apibank}.py` 复刻（网络下载路径改成可传参：`cfg["path"]` / `cfg["hf_dataset"]`），单轮侧不新增风险；真正的任务是①license 逐源补核 ②它们与 ToolMind / When2Call 的去重。

---

## 2. 新增多轮工具调用源（plan §1.1，对应 BFCL v3 multi-turn）

| 源 | ID | license | 条数 | 字段/形态 | 直接转 RL? | 状态 |
|---|---|---|---|---|---|---|
| FunReason-MT | HF `Bingguang/FunReason-MT`；论文 arXiv:2510.24645 | HF 卡片 `[UNVERIFIED]`；论文 CC-BY-SA-4.0（**论文** license，非数据集） | ~16k（plan） | BFCL 风格合成环境的多轮 SFT 轨迹；字段未核实 | **必须去污染**（与 BFCL v3 multi-turn 同分布）；`mt_fc` 可转，但**是否有真实工具返回未知** → 可能是 placeholder | 论文 `verified`；HF ID / 字段 / 条数 / license `[UNVERIFIED]`；GitHub 无官方仓库（搜索 0 命中） |
| APIGen-MT-5k | HF `Salesforce/APIGen-MT-5k` | `[UNVERIFIED]` | 5k（plan） | 带用户交互的轨迹（messages/tools 形态） | `mt_fc`；loader 宽容解析 messages/tools；placeholder 风险同 FunReason-MT | ID `[UNVERIFIED]`（GitHub 无 `Salesforce/APIGen-MT` 仓库，404） |
| tau-bench train | GitHub `sierra-research/tau-bench` | **MIT**（GitHub API 核实） | 仓库为"任务定义 + 可执行环境"；train/test 划分条数未核实 | 任务 + 领域环境（retail/airline 等），**不是现成轨迹** | **不能直接转 RL**：需先用环境生成轨迹，再按 replay/stateful 入池。首期标 `needs-synthesis`，**不进 placeholder 池**、不占多轮配额 | repo `verified`；count/划分 `[UNVERIFIED]`（README 抓取两次超时） |
| Toucan-1.5M | GitHub `TheAgentArk/Toucan`（desc: "Synthesizing 1.5M Tool-Agentic Data from Real-World MCP Environments"）；HF `Agent-Ark/Toucan-1.5M` | **MIT**（GitHub API 核实）；HF 卡片 `[UNVERIFIED]` | 1.5M（仓库描述）；plan 要求只子采样 | 约 500 个真实 MCP server 轨迹，工具返回为真实输出，含 irrelevance 与多轮扩展 | `mt_fc` 可转；**唯一预期带真实工具返回的 MT 主力**（`reference.tool_returns` 可填，`meta.gt_returns_source="reference"`） | repo `verified`；HF ID `[UNVERIFIED]`（org 命名不一致：GitHub `TheAgentArk` vs HF `Agent-Ark`） |
| ToolMind | HF `Nanbeige/ToolMind`；arXiv:2511.15718 | `[UNVERIFIED]` | ~110k（plan） | 带推理链轨迹；与 xlam / ToolACE 重叠 | `mt_fc`；**必须与 xlam/ToolACE 做精确 + MinHash 去重**；是否含真实返回未知 | 论文 `verified`；HF ID / 字段 / 条数 / license `[UNVERIFIED]`；无官方 GitHub 仓库 |

**关键风险（plan §10）**

1. **FunReason-MT 与 BFCL v3 multi-turn 同分布**：`scripts/decontam.py` 必须对它最严格（同源风险最高）；报告要单列命中数。
2. **ToolMind ↔ xlam/ToolACE**：`scripts/dedup.py` 支持 `--against legacy.jsonl`，把 5 个单轮源组成 reference pool 后再灌 ToolMind，命中即丢。
3. **真实返回缺口**：MT 源中只有 Toucan 预期携带真实 observation；FunReason-MT / APIGen-MT / ToolMind 是 SFT 轨迹，是否含 observation 取决于发布格式 → loader 自动判定：能取到 → `tool_returns` + `gt_returns_source="reference"`；取不到 → `tool_returns=null` + `gt_returns_source="placeholder"`（不伪造）。

**给 T2 的"必然 placeholder"清单（当前证据）**

- `funreason_mt` / `apigen_mt` / `toolmind`：**条件性 placeholder**（下载后由 loader 自动统计；若发布格式是纯 messages+tool_calls 则全部 placeholder）。
- `toucan`：预期 `reference`（真实返回主力），需要首跑确认比例。
- `tau_bench`：不进池（needs-synthesis），不产生 placeholder 混入。
- 因此 30% 多轮配额的现实主力 = Toucan 子集 + FunReason-MT（去污染后）；**若 FunReason-MT placeholder 比例 >50%，T2 应降权而非丢弃**（否则配额不够，这是 plan §6 的硬约束）。

---

## 3. 搜索 / 多跳问答（plan §1.2，对应 BFCL v4 web search + 常识）

| 源 | ID / 获取 | license | 条数 | 字段 | 转 RL | 状态 |
|---|---|---|---|---|---|---|
| Search-R1 训练集 | **官方代码实际用** HF `RUC-NLPIR/FlashRAG_datasets`（config `nq`，train/test），不是 plan 猜的 `PeterJinGo/nq_hotpotqa_train`；`PeterJinGo/search-r1` collection 是模型/资源 | FlashRAG 数据集 license `[UNVERIFIED]`；Search-R1 代码 Apache-2.0（`scripts/data_process/nq_search.py` 头注释核实） | NQ train 规模 `[UNVERIFIED]`（通常 ~79k；以 FlashRAG 为准） | `question`, `golden_answers`；处理脚本另加 `prompt`/`reward_model` | 直接 → `search_qa`；本地 Wikipedia + E5（`retriever.url`、topk=3）；1–2 跳，起步难度低 | **代码/路径 `verified`**；plan 的 HF ID 与官方代码不符，已按官方实现修正；数据集 license/count `[UNVERIFIED]` |
| ASearcher 训练数据 | HF `inclusionAI/ASearcher-train-data`（官方 README 明确链接）；GitHub `inclusionAI/ASearcher` | HF 卡片 `[UNVERIFIED]`；repo license `[UNVERIFIED]`（README 未标） | README 训练配置提到 `ASearcher-LRM-35k`（~35k，`[UNVERIFIED]`） | QA pairs（question/answer 或 prompt/reward_model 形态），**无逐题轨迹/轮数标注** | `search_qa`；**数据侧无法判定 ≤15 轮**：QA 对没有 per-task 轮数标签，`expected_turns` 只在原始记录显式携带时才写；≤15 实际由 **runtime `search env.max_rounds`** 承担（本 loader 的 env config 给 `max_rounds=15`，Search-R1 基线按计划用 8）。`filter_rounds.py` 只能丢弃"已标注 expected_turns>阈值"的任务（默认 ASearcher 阈值独立可配），不能宣称已按轨迹过滤 | ID `verified-indirect`；count/license `[UNVERIFIED]`；原 RL 允许 128 轮（README 核实） |
| A²Search（可选） | GitHub `zfj1998/A2Search`，数据在 repo `dataset/a2search_musique_2wiki_nq.parquet` | **Apache-2.0**（GitHub API 核实） | 训练集 1 个 parquet（条数 `[UNVERIFIED]`） | `data_source`, `question`, `reward_model.ground_truth[{aliases,answer}]`, `extra_info.id` | 直接 → `search_qa`（`final_answers` 携带多答案，Answer-F1 有依据）；复用 Search-R1 wiki index | repo/字段 `verified`；条数 `[UNVERIFIED]` |
| 检索后端 | 本地 Wikipedia 索引 + E5（Search-R1 设置） | — | — | `search(query, top_k)` + `fetch(doc_id)` | 不依赖外网；BFCL 风格工具 schema；`env.config` 按 **CONTRACTS C11/D12** 输出 `{"backend":"e5","url":null,"top_k":3,"max_rounds":8,"retriever":...}` | `verified`（代码内） |

**注意**：plan 写的 `PeterJinGo/search-r1` + `nq_hotpotqa_train` 未经核实；本沙箱可达的官方代码用的是 FlashRAG `nq`。loader 两种都支持（`hf_dataset` + `hf_config` 可配，或直接喂本地 parquet）。

---

## 4. 数学 + 代码解释器（plan §1.3）

| 源 | ID | license | 条数 | 字段 | 转 RL | 状态 |
|---|---|---|---|---|---|---|
| 已有 reasoning-RL 数学 prompt 池 | 内部（不在本报告范围） | 内部 | 大 | 题面 + 答案 | **首选**：只挂 `python` 工具，题目分布不变、偏移最小 → `math_ci` | 设计决策 |
| SimpleRL Math3-5 | GitHub `hkust-nlp/simpleRL-reason`（MATH level 3-5 子集） | **MIT**（GitHub API 核实） | `[UNVERIFIED]` | MATH 题 + 答案 | 补充 → `math_ci` | repo `verified`；HF 数据 ID / count `[UNVERIFIED]` |
| ReTool-SFT | HF `JoeYing/ReTool-SFT`（plan 给的 ID） | `[UNVERIFIED]` | ~2k（plan） | messages + 工具调用轨迹（字段未核实） | 只做冷启动/格式参考 → `math_ci`（loader 保留最终答案，不把工具调用当 replay GT） | GitHub `JoeYing/ReTool` 404；HF ID/字段/license `[UNVERIFIED]` |

**建议**：`math_ci` 主池用内部数学 prompt（plan §1.3 首选），SimpleRL / ReTool 只做补充与消融；不要为 ReTool 单独做 replay 环境。

---

## 5. 代码执行（plan §1.4）

| 源 | ID | license | 条数 | 字段 / 测试形态 | 转 RL | 状态 |
|---|---|---|---|---|---|---|
| APPS | HF `codeparrot/apps`；repo `hendrycks/apps` | repo **MIT**（核实）；HF 卡片 `[UNVERIFIED]` | ~10k（plan/常识，`[UNVERIFIED]`） | `question`,`input_output`(JSON str: inputs/outputs),`solutions`,`difficulty` | `code_exec`（env `code_tests`），单测来自 `input_output`；标准答案不入 task | repo `verified`；HF card/count `[UNVERIFIED]` |
| CodeContests | HF `deepmind/code_contests`；repo `google-deepmind/code_contests`（已归档） | repo **Apache-2.0**（核实）；HF 卡片 `[UNVERIFIED]` | ~13k（`[UNVERIFIED]`） | `description`,`public_tests`/`private_tests`（`{"input":[],"output":[]}`） | `code_exec`；隐藏测试可做验收集 | repo `verified`；HF card/count `[UNVERIFIED]` |
| TACO | HF `BAAI/TACO`；repo `FlagOpen/TACO` | repo **Apache-2.0**（核实）；HF 卡片 `[UNVERIFIED]` | ~26k（`[UNVERIFIED]`） | `question`,`input_output`,`solutions`,`difficulty`,`source` | `code_exec` | repo `verified`；HF card/count `[UNVERIFIED]` |

**共同风险**：代码题的解可能是题面直接可算/可猜；奖励必须是"单测全过=1"，且 `unit_tests` 只放测试、不放参考代码（CONTRACTS C5）。仓库级 SWE 任务轮数超预算，本阶段不用（plan §1.4）。
`scripts/...` 之外，loader `code_exec.py` 用 `cfg["kind"]` 选 apps / taco / code_contests。

---

## 6. 拒答 / 追问（plan §1.5）

| 源 | ID | license | 条数 | 字段 | 转 RL | 状态 |
|---|---|---|---|---|---|---|
| When2Call | HF `nvidia/When2Call`（官方 README 明确）；GitHub `NVIDIA/When2Call` | **repo Apache-2.0**（核实）；HF 卡片 `[UNVERIFIED]` | train SFT 条数 `[UNVERIFIED]` | train SFT: `tools` + `messages`（assistant 调用包在 `<TOOLCALL>...</TOOLCALL>`）；test: MCQ + LLM-judge | **只取 train**：有 tool call → `st_fc`；无调用 → `abstain`（追问/拒答） | ID/字段 `verified`（README）；HF license `[UNVERIFIED]` |

**污染警告（verified）**：When2Call 的 **test** 由 BFCL v2 Live 派生（README 明示），**绝不入训练**；train 由 `Salesforce/xlam-function-calling-60k` 合成，与现有 `apigen` 同源 → 必须与 apigen 做精确/近似去重，否则等于把单轮分布再放大。

---

## 7. Memory（plan §1.6）

BFCL v4 memory 五类场景，公开训练集**没有成规模来源**（plan 已判定需合成）。首期不做，只跟踪评测；`task_type`/env 暂不新增（需 Lead 改 schema）。**状态：不做**。

---

## 8. 评测集 / 去污染数据源（plan §1/§3/§8）

| 项 | 结论 | 状态 |
|---|---|---|
| `bfcl-eval` 版本 | PyPI 最新稳定 **2026.3.23**（plan 写的 `2025.10.27.1` 已过时）；license **Apache-2.0**（PyPI metadata / 包描述核实：leaderboard data 亦 Apache-2.0）；数据随 wheel 发布 | `verified`（`https://pypi.org/pypi/bfcl-eval/json`） |
| 去污染数据路径 | 支持三种：`pip install bfcl-eval==2026.3.23` 后的包内 `bfcl_eval/data` 目录 / gorilla 仓库 BFCL 目录 / 用户导出的 JSONL；`scripts/decontam.py --bfcl-path` | 已实现 |
| BFCL v3 multi-turn / v4 agentic 测试集 | 由 bfcl-eval 提供；v4 web search 需 SerpAPI key，memory 评测有预填充耗时（plan §8） | 评测面由 T4/T5 落实 |
| 推理评测集去污染 | `decontam.py --extra-path <AIME 等 prompt 集>` 支持额外文本索引（无工具定义） | 已实现 |
| HF `gorilla-llm/Berkeley-Function-Calling-Leaderboard` | plan/legacy 使用；HF 卡片 `[UNVERIFIED]`，但 BFCL 数据 Apache-2.0 已由 PyPI 包确认 | 数据 license `verified`（Apache-2.0）；HF 镜像卡 `[UNVERIFIED]` |

---

## 9. 交付物映射（loaders / scripts / tests）

| 交付 | 文件 | 覆盖来源 |
|---|---|---|
| registry | `agentic_rl_data/loaders/__init__.py`（`SOURCE_REGISTRY` / `load_source` / `iter_sources`） | 全部 15 个 |
| 单轮 | `apigen.py` `sealtools.py` `toolace.py` `hammer.py` `apibank.py` | 现有 5 源 |
| 搜索 | `search_r1.py` `asearcher.py` `a2search.py` | §1.2 |
| 多轮 | `funreason_mt.py` `apigen_mt.py` `toolmind.py` `toucan.py` + `common.load_mt_tasks_from_records` | §1.1 |
| 其它 | `when2call.py`（abstain）、`retool.py`（math_ci）、`code_exec.py`（apps/taco/code_contests） | §1.3–1.5 |
| 流水线 | `scripts/build_raw.py`（步骤 1）、`filter_rounds.py`（步骤 2）、`dedup.py`（步骤 3）、`decontam.py`（步骤 4） | plan §3 |
| 测试 | `tests/test_loaders.py`（合成 fixture 覆盖 13 个 loader）、`tests/test_pipeline_units.py`（轮数/去重/去污染/构建纯函数） | — |

**回放契约（C6/C9）落实方式**

- `mt_fc`：真实工具返回 → `reference.tool_returns`（按轮分组，与 `gt_calls` 对齐）+ `meta.gt_returns_source="reference"`（规范来源）；`env.config["gt_returns"]` 只在 `cfg` 手工覆盖时出现。
- 只有部分轮次有返回 → `meta.gt_returns_source="gt_turns"`（`reference.gt_turns` 用 C6 形状 `{"step","assistant","tool_results"}`）。
- 完全没有真实返回 → `reference.tool_returns=null` + `meta.gt_returns_source="placeholder"`（**不伪造 JSON 返回**，env 运行时自行降质并在 reward info 标 `degraded_replay`）。
- `meta` 四键（`max_turns/difficulty/pass_rate/license`）由 `common.build_task` 统一补齐，`expected_turns` 可选；`effective_turns` 口径按 C9。

---

## 10. 未决 / TODO（交给用户或后续席位）

1. **逐源补核 license / 条数 / 字段（HF 侧）**：上表所有 `[UNVERIFIED]` 项，用 §0 命令核实后回填 `meta.license` 与 `reports/data_report.md`；`meta.license` 若仍不确定写 `null`，不要写成看似已核实。
2. **FunReason-MT / ToolMind / APIGen-MT 的真实返回比例**：下载后跑 `build_raw.py` + 统计 `meta.gt_returns_source`，把比例交 T2；这决定 30% 多轮配额够不够（见 §2）。
3. **tau-bench**：需要 env 生成轨迹（stateful mock_fc A 方案）后再入池；首期不进。
4. **scripts 作为包运行**：我没有写 `scripts/__init__.py`（不在 T1 写范围）。目前所有脚本支持 `python agentic_rl_data/scripts/<name>.py`（内部自行 bootstrap `sys.path`）；若 Lead 希望支持 `python -m agentic_rl_data.scripts.<name>`，请 Lead 加空 `__init__.py`。
5. **单测未执行**：本沙箱 `pwsh`/`python`/`pytest` 全部起不来（`0xC0000142`）。用户侧执行：
   ```bash
   python -m pytest agentic_rl_data/tests/test_loaders.py agentic_rl_data/tests/test_pipeline_units.py -q
   python agentic_rl_data/scripts/build_raw.py --all --cfg-json configs/raw_paths.json \
       --out-dir agentic_rl_data/data/raw_unified
   python agentic_rl_data/scripts/filter_rounds.py --in-dir agentic_rl_data/data/raw_unified
   python agentic_rl_data/scripts/dedup.py --in agentic_rl_data/data/raw_unified/funreason_mt.jsonl \
       --against agentic_rl_data/data/raw_unified/apigen.jsonl \
       --against agentic_rl_data/data/raw_unified/toolace.jsonl
   python agentic_rl_data/scripts/decontam.py --in-dir agentic_rl_data/data/deduped \
       --bfcl-path "$(python -c 'import bfcl_eval,pathlib;print(pathlib.Path(bfcl_eval.__path__[0])/"data")')"
   ```
6. **内存/规模**：`dedup.py` 是纯 Python MinHash（无 numpy/datasketch），1.5M Toucan 全量会慢；按 plan 只子采样（loader 默认 50k）。
7. **BFCL v4 memory**：无公开训练数据，首期不做（plan §1.6）。

---

## 11. 计数与丢弃统计（T1-02 落实）与本次修复记录

### 11.1 计数口径

- `loaders/common.py` 新增 `SourceStats`（`rows` / `emitted` / `skipped_by_reason`），
  经 `cfg["_stats"]` 传给每个 loader；loader 在能归因的丢弃点调用 `record_skip(cfg, reason)`。
- `scripts/build_raw.py` 的 `build_source()` 返回并写入 `data/raw_unified/_build_report.json`：
  `rows / loaded / skipped_invalid / skipped_total / unattributed_skips / skipped_by_reason / task_types / errors`。
  `unattributed_skips = rows - loaded - skipped_total`（保底暴露"没法归因"的静默丢弃，不会变成 0）。
- `report.py`（T2 负责）应读取 `_build_report.json` 汇总；本报告不复制运行结果，因为**本沙箱无法执行**。

### 11.2 已定义的丢弃原因（代码内可枚举）

| source | 主要 dropped reason |
|---|---|
| apigen | `missing_query`, `no_parseable_answer` |
| hammer | `missing_answers_field`（缺字段 ≠ 显式 `[]`，T1-06）, `unparseable_answers`, `missing_query` |
| toolace | `empty_conversations`, `missing_assistant_response`；call-shaped 未解析比 >5% 时整源报错（不静默） |
| sealtools | `missing_query_or_calling`, `nested_or_unresolvable` |
| apibank | `unresolved_api_turn` |
| search_r1 / asearcher / a2search | `missing_question_or_answers` |
| funreason_mt / apigen_mt / toolmind / toucan | `invalid_or_unparseable_record`（转换/校验失败；不伪造返回） |
| when2call | `missing_user_messages`, `missing_assistant_response` |
| retool | `missing_question_or_answer` |
| code_exec | `missing_question`, `no_usable_tests` |

### 11.3 "必然 placeholder"清单（T2 配额判断用）

| 源 | `meta.gt_returns_source` 预期 | 说明 |
|---|---|---|
| Toucan | `reference`（预期） | 真实 MCP 工具返回，MT 真实返回主力；需首跑确认比例 |
| FunReason-MT | `reference` 或 `placeholder`（**条件性**） | SFT 轨迹是否含 observation 未知；无 observation 时 `tool_returns=null` |
| APIGen-MT-5k | 同上 | 同上 |
| ToolMind | 同上 | 同上 |
| tau-bench | 不进池 | needs-synthesis，不产生 placeholder |
| 其余单轮/搜索源 | 不适用 | `st_fc` / `irrelevance` / `search_qa` / `math_ci` / `code_exec` |

结论：**没有"必然 placeholder"的已接入源**；三个 HF 多轮源是"条件性 placeholder"。若它们全部 placeholder，MT 真实返回只剩 Toucan 子集 → T2 应降权其余 MT 源而不是直接删除（否则 30% 配额不足）。

### 11.4 本轮修复（对应 critic review T1-01…T1-07、Lead D12）

| 项 | 修复 |
|---|---|
| T1-01 | `meta.gt_returns_source` 统一写单数 `reference`（`config|reference|gt_turns|placeholder`），docstring 同步 |
| T1-02 | `SourceStats` + `build_source()` 统计 + `_build_report.json` |
| T1-05 | `meta.max_turns = min(cfg.max_turns, 15)`；长轨迹写 `meta.expected_turns` 交 S2（C9）丢弃并计数 |
| T1-03 | 报告改为"ASearcher 数据侧 ≤15 不可判定，runtime 由 search env `max_rounds` 承担"（见 §3） |
| T1-04 | `loaders/__init__.py` 提供 `SOURCE_REGISTRY` / `available_sources()` / `load_source(source, cfg)` / `iter_sources()`；`build_raw.py` 走 `load_source` |
| T1-06 | hammer 缺 `answers` 字段 → 丢弃并计 `missing_answers_field`；显式 `[]` 才当 irrelevance |
| T1-07 | `build_task` 对 `st_fc` / `mt_fc` 拒绝空 tools 列表（raise + loader 计 skip） |
| P0（D12/C11） | `env_search()` 输出规范 config：`backend/url/top_k/max_rounds/retriever`（兼容旧别名归一化） |
| P1（去污染防假 0） | `build_index` 空索引 **raise ValueError**；新增 0.90 5-shingle Jaccard 近似通道；报告改 `hit_counts_by_source` + 每源最多 50 条样本（不再全局只留 50 条） |
| `meta.group_id` | toolace / apibank / when2call / MT 源按会话 id 填 `group_id`（可选 str） |

