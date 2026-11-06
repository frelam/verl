# 环境落地报告（T4 · env-builder）

> 权威规格：`C:\Users\Administrator\Downloads\agentic_rl_data_plan.md` §4/§3 步骤 5/§8 验收 3&5/§10；
> 契约：[`CONTRACTS.md`](../CONTRACTS.md)（v2 + v2.1/v2.2 补丁）与 [`schema.py`](../schema.py)。
> 本报告对应任务卡 **task-4**，代码范围：`envs/**`、`scripts/replay_check.py`、
> `tests/test_envs*.py`。
>
> **重要声明：本机 `pwsh` 不可用（启动即 `0xC0000142`，`cmd /c` 同样失败），本报告中的
> 所有 Python / pytest / curl / 压测命令均未在本机执行（未执行 = 0 次）。**
> 代码按静态审查交付；可执行验证命令见 §10，请在有 shell 的机器上按序执行。
> 所有性能数字均为 `[SUGGEST]` 起点，**不是实测值**。

---

## 0. 交付物与状态

| 文件 | 内容 | 状态 |
|---|---|---|
| [`envs/base.py`](../envs/base.py) | Env 协议、统一 obs、`max_turns`/timeout、`trajectory()`、`env_state` 构造与校验、`build_env` 工厂 | 静态审查通过，未运行 |
| [`envs/mock_fc.py`](../envs/mock_fc.py) | A) 有状态（filesystem / social / trading）；B) 回放式（`replay_ok`、GT 返回来源与降级标记） | 同上 |
| [`envs/search.py`](../envs/search.py) | in-memory / E5+faiss 后端、FastAPI 服务、重试客户端、SearchEnv（≤8 轮 + 任务 deadline + 兼容别名）、建索引 / 压测 CLI | 同上 |
| [`envs/python_env.py`](../envs/python_env.py) | 子进程沙箱（超时 / rlimit / 清环境 / 输出截断 / 网络隔离说明）、PythonEnv | 同上 |
| [`envs/code_tests.py`](../envs/code_tests.py) | pytest/unittest 沙箱跑单测、`env_state["code_tests"]`、`infra_error`、脱敏 | 同上 |
| [`scripts/replay_check.py`](../scripts/replay_check.py) | plan §3 步骤 5：GT 轨迹重放 + reward 满分断言 + 失败清单 + placeholder 比例统计 | 同上 |
| [`tests/test_envs.py`](../tests/test_envs.py) | 协议/终止/状态隔离/mock_fc 确定性/并发 | 未执行（pytest） |
| [`tests/test_envs_units.py`](../tests/test_envs_units.py) | search fake index/重试/deadline、python 沙箱、code_tests、replay_check | 未执行（pytest） |
| [`reports/env_setup.md`](../reports/env_setup.md) | 本报告 | — |

`envs/` 包只 eager import base；`build_env()` 按 `env.type` 懒加载具体环境，
因此导入 `agentic_rl_data.envs` 不会拉入 fastapi / faiss / torch。

### 0.1 本轮 review / 裁决修复（已静态落地，未运行验证）

| 编号 | 问题 | 修复 |
|---|---|---|
| T4-01 (P0) | timeout 时 stateful mock_fc 仍可能 `goal_satisfied=True` | base 层 + MockFCEnv 在 `timeout=True` 时强制 `goal_satisfied=False`；Replay 同理强制 `replay_ok=False`；`checks` 保留诊断。单测 `test_max_turns_timeout_forces_failed_terminal_verdict` / `test_timeout_does_not_promote_replay_ok` |
| T4-02 (P1) | `_inject_returns` 静默伪造成功返回 | 缺失/名字不符 → `ok=False` + `MISSING_RETURN_PREFIX` 文本，累计 `extras["gt_returns_missing"]`；缺失比例 > `max_missing_ratio`（默认 0.5）→ runtime `gt_returns_source` 降级 `placeholder` + `degraded_replay=True`，保留 `gt_returns_declared_source` |
| T4-06 (P2) | close 后 reset 语义 | close 幂等、保留 trajectory；close 后 step 抛 `EnvError`；`reset(task)` 可重开。单测 `test_close_keeps_trajectory_and_reset_reopens` |
| T4-07 (P2) | checks 形状 | `normalise_checks()` 统一为 `{name:str, ok:bool, detail:str≤500}`；新增 `validate_env_state()` 全字段形状校验（replay_check 也调用） |
| C11/D12 (P0) | SearchEnv 与 T1 loader 的 `local_wikipedia_e5` + `retriever{url,topk}` 不兼容 | `normalise_search_config()` 兼容别名：`local_wikipedia_e5→e5`，`retriever{url,topk}`→`url/top_k`；e5+url 且无本地索引文件时按 HTTP 服务解析；其余 retriever 键（docs_path/index_path/model/device/超时/预算）提升到顶层 |
| search P1 | 无任务级 deadline（最坏 8×5s×3≈132s/任务） | 新增 `task_deadline_s`（默认 30s，≤0 关闭）：每次调用前检查预算，耗尽后 `ok=False` + `info["deadline_exceeded"]`/`backend_error`；HTTP 客户端超时 clamp 到 deadline |
| code_tests P1 | infra 故障与答错不可分（`exit_code=-1` 一律 0 分） | `env_state["code_tests"]["infra_error"]`：无 unit_tests / runner 缺失 / runner 崩溃(exit≥2) / 收集到 0 测试 → True；超时、断言失败、候选 import 失败 → False |

---

## 1. 依赖分层与安装（Linux 训练机为准；Windows 把 `python` 换成 `py -3`）

```bash
# 0) 基础（必需）：Python 3.10+，测试用 pytest；envs 纯逻辑部分零第三方依赖
pip install "pytest>=7"

# 1) search 服务（必需，仅 HTTP 层）
pip install fastapi uvicorn

# 2) E5 检索（按 GPU/CPU 二选一）
pip install numpy sentence-transformers        # E5 编码
pip install faiss-cpu                          # CPU 检索（或按 CUDA 版本装 faiss-gpu）
# GPU 编码需先装匹配 CUDA 的 torch，再装 sentence-transformers

# 3) 可选：真正隔离网络/文件系统的包装器（见 §5.3），例如
sudo apt install firejail                      # 或使用 bwrap / docker
```

说明：`envs/search.py` 的 `E5FaissBackend` 用 `sentence-transformers` + `faiss`；
缺依赖时**不会**在 import 期报错，而是构造 env 成功、首次工具调用返回
`ok=False` 的"后端不可用"结果（见 §2.4），保证 rollout 不崩。

---

## 2. 字段级契约（reward / 适配层消费口径，冻结）

### 2.1 Env 协议（C1）

```python
env = build_env(task)                     # 每 task 一个实例
obs = env.reset(task, max_turns=...)      # 全量重建；task=None 复用上次 task
obs, done, info = env.step(action)        # 每次 = 一个 assistant 轮
env.close()                               # 幂等；close 后 trajectory 仍可读，reset(task) 可重开
```

- `action = {"content": str, "tool_calls": [{"id", "name", "arguments"}]}`；裸 `str` = 无调用。
- 生效 `max_turns = min(task.meta.max_turns, reset 课程值, env.config.max_turns, 环境自带上限)`；
  SearchEnv 自带上限 = 8（Search-R1）。
- **timeout（T4-01 修复）**：超过 `max_turns` → `done=True, info["timeout"]=True`；
  且 `trajectory["env_state"]` 中 `goal_satisfied=False`、`replay_ok=False`
  （`checks` / `state` 保留作诊断）。即"超限轨迹不可能拿终态满分"，不额外扣分
  （plan §4.1），由 reward 按 timeout 字段处理。

### 2.2 统一 obs（C2，与 `Trajectory.turns[*].tool_results` 对齐）

```json
{"step": int,
 "tool_results": [{"id": str|null, "name": str, "content": str, "ok": bool}],
 "content": str, "env_state": {...}, "info": {}}
```

`reset` 时 `tool_results=[]`；`step` 时与 `action.tool_calls` 一一对应（id 缺省时由 env 生成
`call-<step>-<i>`）。

### 2.3 `trajectory["env_state"]`（C3，所有 env 同形；未用字段 null）

| 字段 | 语义 |
|---|---|
| `type` / `impl` | `mock_fc`/`search`/`python`/`code_tests`；`stateful`/`replay`/`local` |
| `goal_satisfied` | stateful mock_fc：终态断言全过；timeout 时强制 `False`；无 goal 且非 timeout 时 `None` |
| `replay_ok` | replay mock_fc：整条 GT 轨迹全匹配；timeout 时强制 `False` |
| `failed_turn` / `matched_turns` / `gt_turns` | replay 诊断（0-based、非空 GT 轮组数） |
| `checks` | `[{"name","ok","detail"}]`，base 层统一归一化（detail ≤500 字符） |
| `state` | stateful 终态快照（域原始 state；与 `reference.final_state` 同形可深比较） |
| `code_tests` | 仅 code_exec，见 §2.5；其他 env 为 null |
| `rendered` / `num_turns` / `timeout` | 人类可读摘要 / 轮数 / 是否超轮 |
| `extras` | env 私有调试信息（下表） |

`extras` 关键键：

| env | 键 |
|---|---|
| mock_fc replay | `gt_returns_source`（runtime 真相：config/reference/gt_turns/placeholder）、`gt_returns_declared_source`、`gt_returns_missing`、`gt_returns_missing_ratio`、`max_missing_ratio`、`degraded_replay`、`meta_gt_returns_source` |
| mock_fc stateful | `domain`、`final_answer_matches`、`final_answer_mode`、`tool_call_count` |
| search | `backend`、`doc_count`、`top_k`、`max_rounds`、`task_deadline_s`、`elapsed_s`、`deadline_exceeded`、`queries`、`fetched_docs`、`backend_errors` |
| python | `executions`（每次运行 stdout/stderr/exit_code/timed_out/duration_s/truncated 摘要）、`sandbox`、`network_isolated` |
| code_tests | `runner`、`tests_configured`、`infra_error`、`diagnostics`（stdout/stderr tail，仅供框架日志） |

### 2.4 工具返回内容（policy 可见的 `tool_results[i].content`）

| env | content 格式 |
|---|---|
| search `search` | canonical JSON `{"results":[{"doc_id","title","snippet","score"}]}` |
| search `fetch` | canonical JSON `{"doc_id","title","text"}`（text ≤4096 字符） |
| python | canonical JSON `{"stdout","stderr","exit_code","timed_out","duration_s","truncated"}` |
| mock_fc stateful | 域 handler 的文本/JSON 结果或 `Error: ...` |
| mock_fc replay | GT 返回原文；缺失时为 `Error: ground-truth return missing for tool ...`（`ok=False`） |

后端/沙箱故障一律 `ok=False` + 可读错误文本，**不抛异常进训练循环**；
search 的每步 `info` 另带 `backend_error=True` / `deadline_exceeded=True` 供日志。

### 2.5 `code_tests`（C5 + T4 P1 的 `infra_error`；**唯一真相源**）

```json
{"total": int, "passed": int, "failed": int, "all_passed": bool,
 "infra_error": bool, "exit_code": int, "duration_s": float,
 "truncated": bool, "failures": [{"name": str, "message": str}]}
```

- `all_passed = (total > 0 and passed == total and exit_code == 0 and not timed_out)`，权威判分字段。
- `infra_error=True`：无 `unit_tests`、runner 缺失、runner 崩溃/用法错误（exit≥2）、
  收集到 0 个测试；此时 `all_passed` 恒 False 但**不是答错**，reward/报告可剔除或单独统计。
  超时、断言失败、候选代码 import 失败均 `infra_error=False`（属于策略失败）。
- `failures[].message` ≤200 字符；只从输出摘要抽取，绝不回显候选代码或标准答案；
  `env.config["redact"]` 可加正则/字面量脱敏。
- `info["tests"]` 是同内容副本，仅供框架日志（reward 不得读）。

### 2.6 mock_fc 回放返回来源（C6 + T4-02）

优先级：`env.config["gt_returns"]` → `reference["tool_returns"]` → `reference["gt_turns"][*].tool_results/results/return` → 占位。

- `extras["gt_returns_source"]` 是 **runtime 真相**（`config/reference/gt_turns/placeholder`，
  单数枚举）。缺失的对调用返回 `ok=False` 并累计；
  缺失比例超过 `env.config["max_missing_ratio"]`（默认 `0.5`）→ `gt_returns_source` 降级为
  `placeholder`、`degraded_replay=True`，不允许沿用 `reference`。
- `meta["gt_returns_source"]`（loader 静态声明，schema v2.2 对 mt_fc 必填）与 runtime 不一致时，
  `replay_check.py` 记 warning（runtime 为准）。
- 阈值示例：`{"max_missing_ratio": 0.0}` = 任何缺失都降级；默认 0.5 = 超过一半才降级。
  建议 T2 对 `degraded_replay`/placeholder 比例做过滤或降权（§7.3）。

---

## 3. mock_fc（§4.2）

### 3.1 A. 有状态环境（`MockFCEnv`，impl=`stateful`）

```json
{"impl":"stateful","domain":"filesystem|social|trading",
 "initial_state":{...},              // 深合并到域默认 state
 "responses":{"tool_name":"canned 文本或 {\"content\":...,\"ok\":...}"},  // 外部数据类工具
 "tool_aliases":{"declared":"domain_handler"},   // BFCL 工具名 → 域 handler
 "goal":[{"check":"file_exists","path":"/home/user/notes.txt"},
         {"check":"file_content","path":"...","expected":"hello","mode":"contains"}],
 "final_answer_mode":"none|normalized|contains|exact|regex",
 "max_turns": null, "tolerate_tool_errors": false}
```

域与内置工具（最小 BFCL v3 风格）：

- **filesystem**：`mkdir/touch/write_file/read_file/list_dir/delete_file/delete_dir/move_file/copy_file`；
  state = `{"files": {path: content}, "dirs": [path...]}`（默认 `["/"]`）。
- **social**：`post/follow/unfollow/get_following/like/comment/get_post/get_timeline`；
  state = `{"following": [...], "posts": [{"id","content","likes","comments"}], "next_post_id": int}`。
- **trading**：`get_balance/get_holdings/get_price/buy/sell/transfer/deposit/get_history`；
  state = `{"balance": float, "holdings": {sym: int}, "prices": {sym: float}, "history": [...]}`。

goal check 支持：`file_exists/file_not_exists/file_content/dir_exists/dir_not_exists/entries/value/
exists/absent/count/any_of/all_of`；比较模式 `eq/approx/gte/lte/contains/length/regex`。
未配置 goal 时 `goal_satisfied=None`，reward 按 C4 优先级回退 `reference.final_state` 深比较 →
`final_answer` EM → `replay_ok`。

> **给 loaders（重要）**：`env_state["state"]` 是域默认 state 的完整快照（如
> filesystem `{"files":..., "dirs":[...]}`）。`reference.final_state` 必须按同一 shape 写全量快照，
> 否则 reward 深比较会因缺键/多键失败。初始态用 `initial_state` 补齐（例如
> `{"dirs":["/","/home","/home/user"]}` 必须包含 `/`）。

### 3.2 B. 回放环境（`ReplayMockFCEnv`，impl=`replay`）

- 逐轮匹配 `reference.gt_calls`（同轮并行调用顺序无关；参数默认宽松比较：
  string `strip/lower`、数值 1e-6 相对容差、dict key 集合必须一致；`config.match_mode="exact"`
  可切精确）。
- 命中 → 注入对应轮 GT 返回继续；不匹配 → 立即终止，`replay_ok=False` 并带 `failed_turn`；
  GT 轮走完后的额外工具调用、提前给最终答案都判失败。
- 整条轨迹全匹配（含最终无调用收尾）才 `replay_ok=True`，供 mt_fc B 的 0.8 主项。

---

## 4. search（§1.2/§4.3/§10）

### 4.1 索引：优先复用 Search-R1 预建 E5 索引，其次自建

```bash
# 方式 1（推荐，省 GPU 时间）：Search-R1 官方 wiki-18 E5 索引 [CONFIRM 数据集名/版本]
pip install huggingface_hub
huggingface-cli download PeterJinGo/wiki-18-e5-index --repo-type dataset \
  --local-dir data/wiki/search_r1_e5
# 目录内通常有 e5_Flat.index（faiss IndexFlatIP）与 wiki-18.jsonl(.gz)
```

把官方文件转换成 `E5FaissBackend` 需要的 `(docs.jsonl, 索引)` 对（**行序必须与索引行号严格一致**）：

```bash
python - <<'PY'
import gzip, json, pathlib
src = pathlib.Path("data/wiki/search_r1_e5/wiki-18.jsonl.gz")
out = pathlib.Path("data/wiki/e5_docs.jsonl"); out.parent.mkdir(parents=True, exist_ok=True)
with gzip.open(src, "rt", encoding="utf-8") as f, out.open("w", encoding="utf-8") as g:
    for i, line in enumerate(f):
        d = json.loads(line)
        text = d.get("text") or d.get("contents") or d.get("passage") or ""
        title = d.get("title") or f"doc_{i}"
        g.write(json.dumps({"doc_id": str(i), "title": title, "text": text}, ensure_ascii=False) + "\n")

import shutil, faiss
shutil.copy("data/wiki/search_r1_e5/e5_Flat.index", "data/wiki/e5.index.faiss")
PY
```

```bash
# 方式 2（自建，适合子集冒烟或自定义语料）：docs JSONL 每行 {"doc_id","title","text"}
python -m agentic_rl_data.envs.search build-index \
  --docs data/wiki/e5_docs.jsonl \
  --out-prefix data/wiki/e5_index \
  --model intfloat/e5-base-v2 --batch-size 64 --device cuda:0
# 产物：e5_index.npz / e5_index.faiss / e5_index.docs.jsonl / e5_index.meta.json
```

> 自建 flat index 内存 O(N×d)，适合 <100 万 passage 的冒烟；全量 21M passage 请用官方
> IVF+PQ 索引或 pyserini/faiss 离线构建（[CONFIRM] 具体参数），不要用本仓库 `build-index` 跑全量。

### 4.2 起服务

```bash
# 单进程（推荐：1 个进程 + 内部 32-64 线程；E5 模型与 faiss 只加载一份）
python -m agentic_rl_data.envs.search serve \
  --backend e5 --docs data/wiki/e5_docs.jsonl --index data/wiki/e5.index.faiss \
  --model intfloat/e5-base-v2 --device cuda:0 \
  --host 0.0.0.0 --port 8899 --workers 1 \
  --request-timeout 5 --snippet-chars 1000
```

```bash
# 多 worker（uvicorn 进程池；每个 worker 各加载一份索引/模型 → 显存与内存 × workers，
# 建议 2-4，不要 8+）
export ARD_SEARCH_BACKEND=e5
export ARD_SEARCH_DOCS=data/wiki/e5_docs.jsonl
export ARD_SEARCH_INDEX=data/wiki/e5.index.faiss
export ARD_SEARCH_MODEL=intfloat/e5-base-v2
export ARD_SEARCH_DEVICE=cuda:0
export ARD_SEARCH_REQUEST_TIMEOUT=5
export ARD_SEARCH_SNIPPET_CHARS=1000
uvicorn "agentic_rl_data.envs.search:app" --factory --host 0.0.0.0 --port 8899 --workers 4
```

若暂时没有 E5/faiss，也可用 `--backend inmemory --docs <docs.jsonl>` 起一个依赖最小的服务做联调。

### 4.3 冒烟测试

```bash
curl -s http://127.0.0.1:8899/healthz
curl -s -X POST http://127.0.0.1:8899/search \
  -H 'Content-Type: application/json' -d '{"query":"capital of France","top_k":3}'
curl -s 'http://127.0.0.1:8899/fetch?doc_id=0'

# 客户端 CLI（走同一 HTTP 客户端，带超时+重试）
python -m agentic_rl_data.envs.search health  --url http://127.0.0.1:8899
python -m agentic_rl_data.envs.search search "capital of France" --url http://127.0.0.1:8899
python -m agentic_rl_data.envs.search fetch 0 --url http://127.0.0.1:8899
```

`/search` 返回 `{"results":[...]}`；`/fetch` 返回 `{"doc_id","title","text"}`；
后端超时/错误统一 HTTP 503（客户端自动重试，最终降级为 `ok=False`）。

### 4.4 压测（训练前必做；本机未执行）

```bash
# 30s 级别压测：并发 64、5000 次查询（客户端不重试，避免掩盖错误率）
python -m agentic_rl_data.envs.search bench \
  --url http://127.0.0.1:8899 --concurrency 64 --requests 5000 \
  --top-k 3 --timeout 5 --retries 0
# 输出 {"requests","errors","qps","mean_ms","p50_ms","p95_ms","p99_ms","wall_s"}
```

验收门槛 `[SUGGEST]`：`p99 < 100ms`、错误率 `< 0.5%`、单进程 `QPS ≥ 200`；不达标时依次：
① 检查 GPU 利用率/批量化；② faiss 索引改 IVF+PQ 并调 nprobe；③ 降 `top_k`（3→2）；
④ 最后才调低 search_qa 配比。**这些数字未实测，不许写进最终训练报告当结论。**

### 4.5 SearchEnv 配置（含 T1 过渡期兼容别名）

```json
{"type":"search","config":{
  "backend":"http",                      // 规范：e5|inmemory|http；兼容 local_wikipedia_e5→e5
  "url":"http://127.0.0.1:8899",         // 兼容 retriever.url
  "top_k":3,                             // 兼容 retriever.topk / retriever.top_k
  "max_rounds":8,                        // 只能 ≤8（Search-R1 硬上限）
  "task_deadline_s":30,                  // 任务级后端总预算；<=0 关闭
  "request_timeout_s":5,"max_retries":2,
  "snippet_chars":1000
}}
```

`retriever` 对象里的 `docs_path/index_path/model/device/…` 也会提升到顶层；
工具名别名：`search/web_search/search_wiki`、`fetch/fetch_page/fetch_doc`。

**deadline 语义**：每次后端调用前检查剩余预算；预算耗尽后所有调用立即返回
`ok=False("search task deadline ...")`，`info["deadline_exceeded"]=True`、
`extras["deadline_exceeded"]=True`，rollout 继续。
默认值下的最坏界 = `deadline + 单次调用最坏（(max_retries+1)×min(request_timeout,deadline)+backoff）`；
HTTP 客户端超时会被 clamp 到 `deadline`。8 轮 × 5s × 3 尝试≈132s 的旧问题因此被**任务级**封顶
（默认 30s 量级，健康后端下永不触发）。

### 4.6 给 mixture-researcher 的吞吐参数（已单独发送）

- 默认 `top_k=3`、`max_rounds=8`、`task_deadline_s=30`、`request_timeout_s=5`、`max_retries=2`；
- 规划检索 QPS ≥ 64/训练 worker、并发 32–64 在飞请求；
- 若服务能力不足，优先 `top_k 3→2` 或降 search_qa 占比，**不要放宽 8 轮上限**；
- 索引规模（TEAM_BOARD Q3）：首期 `wiki_dpr` E5 全量约 21M passages（faiss IVF+PQ 约
  40–60GB 磁盘 / 64–96GB 内存），冒烟可先 5–6M 子集；服务常驻内存 50–70GB 量级，
  多 worker 按倍数放大 → 建议 1 进程 + 64 线程。
- 上述均为 `[SUGGEST]`，压测后替换。

---

## 5. python 沙箱（§4.4）

### 5.1 默认限制（`PythonEnv` / `env.config`）

| 项 | 默认 | 配置键 |
|---|---|---|
| 墙钟超时 | 5s | `timeout_s` |
| CPU 秒（POSIX rlimit） | `timeout+1` | `cpu_seconds` |
| 内存（POSIX `RLIMIT_AS`） | 512MB | `memory_mb` |
| 单文件大小 | 16MB | `max_file_bytes` |
| 打开文件数 | 128 | `max_open_files` |
| 进程数 | 32 | `max_processes` |
| 输出截断 | 每路 4096 字符 | `max_output_chars` |
| 环境变量 | 清空，仅白名单（PATH/SYSTEMROOT/TEMP/LANG/PYTHONIOENCODING…） | `clear_env` |
| 工作目录 | 每次新建临时目录，退出即删 | — |
| 网络隔离 | **默认无**（见 5.3） | `sandbox_prefix` / `network_isolated` |

Windows 上 `resource` 模块不可用 → **只有超时限制**；生产训练机必须是 Linux。
`python -I -B` 隔离 user site / `PYTHONPATH` / `PYTHONSTARTUP`。

### 5.2 行为

`run_python(code)` 返回 `{"stdout","stderr","exit_code","timed_out","duration_s","truncated",
"sandbox","network_isolated"}`；超时被杀时 `exit_code=-1, timed_out=True`。
`PythonEnv` 工具名别名：`python/execute_python/run_python/python_interpreter/code_interpreter`；
代码参数键：`code/input/source/script/python/statement`。未知工具 / 空代码 → `ok=False`，不抛异常。

### 5.3 安全边界（如实陈述）

- **本实现（无 `sandbox_prefix` 时）不隔离网络、不隔离文件系统**：脚本仍可 `open()` 宿主文件、
  发起网络请求。仅适合可信数据/内网训练机。
- 真正隔离需外部沙箱包装器，通过 `env.config["sandbox_prefix"]` 传入 argv 前缀；示例（未测试）：

```json
{"sandbox_prefix":["firejail","--quiet","--net=none","--"],
 "network_isolated":true}
```

```json
{"sandbox_prefix":["bwrap","--unshare-net","--die-with-parent",
  "--ro-bind","/usr","/usr","--ro-bind","/lib","/lib","--ro-bind","/lib64","/lib64",
  "--dev","/dev","--proc","/proc","--bind","<每任务临时目录>","/work","--chdir","/work","--"],
 "network_isolated":true}
```

bwrap 示例需要按宿主路径调整（临时目录由 env 创建，当前 `sandbox_prefix` 是静态前缀，
**尚不支持动态插入工作目录** → 用 firejail/docker 更简单，或按需扩展；已列为未决项）。
生产建议：容器/namespace 级隔离（如 per-rollout Docker 或 `nsjail`），本仓库不附带。

---

## 6. code_tests（§4.4）

- 任务输入：`reference.unit_tests`（测试文件全文，约定 `from solution import <symbol>`），
  环境把候选代码写成 `solution.py`、测试写成 `test_solution.py`，在临时目录沙箱内运行。
- runner：`auto`（有 pytest 用 pytest，否则 unittest）→ `pytest -q --tb=line` /
  `unittest discover`；候选代码/测试均不进入 policy 可见输出。
- 输出：`env_state["code_tests"]`（C5，见 §2.5）；`info["tests"]` 仅日志。
- 时限默认 10s（`timeout_s`）、内存 1GB（`memory_mb`）；**候选代码死循环/测试 hang 归为策略失败**
  （`infra_error=False`），基础设施/收集不到测试归 `infra_error=True`。
- 数据要求：单测必须能在 ≤10s 跑完；超过该阈值的题（大输入/复杂依赖）建议过滤或提高
  `timeout_s`（会影响 code_exec 吞吐）。测试文件里**不要**放标准答案；API 不同时用
  `solution.py` 导入约定做适配。

---

## 7. `scripts/replay_check.py`（plan §3 步骤 5）

### 7.1 命令

```bash
python -m agentic_rl_data.scripts.replay_check \
  --tasks data/raw_unified/funreason_mt.jsonl data/raw_unified/apigen_mt.jsonl \
  --task-types mt_fc \
  --impl auto \
  --reward auto \
  --min-score 1.0 \
  --out reports/replay_check.json
# 退出码：0 全通过；1 有失败（失败清单会打印到 stdout）
```

可选：`--impl replay|stateful`（强制实现）、`--allow-tool-errors`（GT 调用返回 `ok=False`
记为 warning 而非失败）、`--max-failures`（报告保留条数）、`--limit`、`--quiet`。

### 7.2 判定逻辑

1. `schema.validate_task`（含 v2 reference 7 键、v2.2 `meta.gt_returns_source` 必填）失败 → `stage=schema`；
2. 构建 env（auto：有 `domain/goal` → stateful，否则 replay），重放每轮 GT 调用；
3. 检查未提前终止/未 timeout（`stage=timeout/early_termination`）；
4. `validate_trajectory` + `validate_env_state`（C3 形状，`stage=trajectory/env_state`）；
5. 终态判定（C4）：replay → `replay_ok is True`；stateful → `goal_satisfied` 优先，
   否则 `reference.final_state` 与 `env_state.state` 深比较；无任何终态断言 → warning；
6. 若 `rewards.get_reward` 可导入（或 `REWARDS/REWARD_FUNCTIONS` registry），调用 reward
   并要求 `score ≥ --min-score`（默认 1.0）；否则标记 `reward_used=False` 并仅用终态/replay 判定兜底。

### 7.3 报告与降质统计（C6/D7）

`--out` 写 JSON：`summary` 含

- `total/passed/failed/skipped/reward_used/reward_backend`；
- `gt_returns_source_counts`（runtime 来源分布）、`placeholder_count`、`placeholder_ratio`；
- `gt_returns_missing`（缺失返回调用总数）、`gt_returns_source_mismatch`（meta 声明 vs runtime 不一致数）；
- `warnings` 计数与每个 task 的 `warnings`（含 degraded replay、tool_error 容忍、无终态断言等）；
- `failures[]`：`task_id/source/task_type/stage/reason`（replay 失败带 `failed_turn/matched_turns/gt_turns`）。

**给 T1/T2 的过滤建议（`[SUGGEST]`）**：`placeholder_ratio > 5%` 或 `degraded_replay` 的 mt_fc
样本过滤/降权，并在数据报告给出过滤前后条数；`meta` 与 runtime 来源不一致的样本单独标注。

失败 `stage` 清单：`schema/config/timeout/early_termination/no_termination/env/tool_error/
trajectory/env_state/replay/goal/final_state/reward/reward_score`。

---

## 8. 并发 / 超时 / 安全 速查

| 关注点 | 实现 |
|---|---|
| 并发/无状态泄漏 | 每 task 独立 env 实例；所有可变状态在实例字段；`reset()` 全量重建；测试含双实例交错 + 6 线程并发 |
| `max_turns` | min(task, 课程 reset 值, config, 环境上限)；超限 `done=True/timeout=True`，终态判定强制 False |
| search 重试 | `request_timeout_s=5`、`max_retries=2`、指数退避 0.5→1.0s，仅 408/429/5xx/网络错误可重试 |
| search 任务预算 | `task_deadline_s`（默认 30s）检查于每次调用前；耗尽即 `ok=False` + `backend_error` 标志 |
| search 服务侧 | FastAPI 线程池 + `request_timeout` 硬超时 → HTTP 503，不挂起请求 |
| python/code_tests | 子进程组 kill、POSIX rlimit、临时目录、清环境、输出截断；无真实网络/FS 隔离（需外部 wrapper） |
| 故障不崩训练 | 所有后端/沙箱错误 → `tool_results.ok=False`；env 内部错误抛 `EnvError`（测试/数据问题可视化） |
| code_tests 安全 | 不返回候选代码/标准答案；只回 pass/fail 摘要 + ≤200 字符失败消息；`redact` 可选脱敏 |

---

## 9. 未执行 / 未验证清单（如实）

**本机（`D:\lhy\verl\verl-toolRL`）一条命令都没跑**（pwsh `0xC0000142`）。未验证项：

1. 全部 pytest（§10 的 3 条 pytest 命令）。
2. search 服务实际启动、HTTP smoke、`fastapi`/`uvicorn` 版本兼容与 503 超时行为。
3. E5 索引构建/加载、faiss 依赖、官方 wiki-18 索引的真实文件名与格式 `[CONFIRM]`。
4. 压测数字（QPS/p99/错误率）与 8 轮 × deadline 的真实吞吐。
5. python / code_tests 的 rlimit 行为（Linux）与 Windows 下无 rlimit 的行为。
6. firejail/bwrap 网络隔离示例（需要宿主机权限与路径调整）。
7. 与 slime 的 rollout/reward 适配（§9 未冻结接口，本阶段不接）。
8. 真实 FunReason-MT 等 loader 产出的 `env.config` / `final_state` / `gt_returns` 形状对齐
   （T1 规范形式与过渡期别名的端到端联调）。

未决问题：

- `[Q3]` 索引规模/E5 版本（本报告给建议，需压测定版）；
- stateful mock_fc 目前覆盖 filesystem/social/trading；BFCL v3 其余域（vehicle/travel/智能家居等）
  需按同一 `BaseDomain` 接口扩展并维护 `tool_aliases`；
- `reference.final_state` 必须是域完整 state 快照（见 §3.1 警告），loader 侧需统一；
- bwrap 动态工作目录支持（当前仅静态前缀）。

---

## 10. 一键验证命令（用户机器复制执行）

```bash
cd verl/verl-toolRL            # 或你的仓库路径

# 1) 环境纯逻辑单测（无需 fastapi/faiss）
python -m pytest agentic_rl_data/tests/test_envs.py -q

# 2) 子系统单测（fastapi/httpx 缺省时相关用例自动 skip；python 沙箱用例需要可执行 python）
python -m pytest agentic_rl_data/tests/test_envs_units.py -q

# 3) 全部环境测试
python -m pytest agentic_rl_data/tests/test_envs.py agentic_rl_data/tests/test_envs_units.py -q

# 4) search 冒烟（先起服务，见 §4.2）
python -m agentic_rl_data.envs.search health --url http://127.0.0.1:8899
python -m agentic_rl_data.envs.search search "capital of France" --url http://127.0.0.1:8899

# 5) search 压测（§4.4）
python -m agentic_rl_data.envs.search bench --url http://127.0.0.1:8899 \
  --concurrency 64 --requests 5000 --top-k 3 --timeout 5 --retries 0

# 6) plan §3 步骤 5：GT 轨迹重放 + reward 满分检查
python -m agentic_rl_data.scripts.replay_check \
  --tasks data/raw_unified/*.jsonl --task-types mt_fc \
  --reward auto --min-score 1.0 --out reports/replay_check.json

# 7) code_tests 快速验证
python -m pytest agentic_rl_data/tests/test_envs_units.py -q -k code_tests
```

预期：`1`/`2`/`3`/`7` 全绿；`4` 返回 backend/doc_count；`5` 输出 QPS/p99 摘要；
`6` 全通过退出码 0，否则打印失败清单。若 `1` 失败，优先检查 Python 版本（需 ≥3.10）
与仓库根目录是否在 `sys.path`（在仓库根运行 `python -m pytest` 即可）。
