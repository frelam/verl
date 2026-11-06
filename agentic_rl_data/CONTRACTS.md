# CONTRACTS（Lead 独占维护，其他席位只读）

> 版本 **v2.4** · 最后更新 2026-11-06 09:45 (+08:00) · 上游 `schema.py`（v2.2）
> v1→v2：`code_exec` 结果迁到 `env_state["code_tests"]`；`reference` 7 键必填 + `final_answers/final_state/tool_returns`；
> `mt_fc` 结果依据优先级；`reference.tool_returns` 为回放规范来源；reward score 冻结 `[0,1]`。
> v2.1→v2.4：环境字段（infra_error / task_deadline_s / 回放诊断键）、`meta.group_id`、**data_report 语义（C13）**。

## C1. Env 协议（env-builder 实现）

```python
class Env:
    def __init__(self, task: dict): ...
    def reset(self, task: dict | None = None, *, max_turns: int | None = None) -> dict: ...
    def step(self, action: dict) -> tuple[dict, bool, dict]: ...
    def close(self) -> None: ...
```

- `action` = 一整个 assistant 轮：`{"content": str, "tool_calls": [{"id", "name", "arguments"}]}`；裸 `str` 视为无调用。
- 每 task 一个实例；`reset` 全量重建内部状态 → 实例复用/并发无状态泄漏。
- 生效轮数上限 = `min(task.meta.max_turns, reset 传入的课程值, env.config 自带上限)`。
- 超限：`done=True`、`info["timeout"]=True`，**不额外扣分**（plan §4.1）。

## C2. 统一 observation（与 `Trajectory.turns[*].tool_results` 对齐）

```json
{"step": int, "content": str,
 "tool_results": [{"id": str|null, "name": str, "content": str, "ok": bool}],
 "env_state": {...}|null, "info": {...}}
```

`reset` 时 `tool_results=[]`；`step` 时 `tool_results` 与 `action.tool_calls` 一一对应。

## C3. `trajectory["env_state"]`（done 时必给；未用字段为 null）

```json
{"type": "mock_fc|search|python|code_tests",
 "impl": "stateful|replay|local",
 "goal_satisfied": bool|null,     // stateful：终态断言全过（mt_fc A 的 0.8 主项来源）
 "replay_ok": bool|null,          // replay：整条 gt 轨迹全匹配（mt_fc B 的 0.8 主项来源）
 "failed_turn": int|null, "matched_turns": int|null, "gt_turns": int|null,
 "checks": [{"name": str, "ok": bool, "detail": str}],
 "state": {...}|null,             // stateful 实际终态快照（reward/审计用）
 "rendered": str, "num_turns": int, "timeout": bool,
 "code_tests": {...}|null,        // 仅 code_exec：见 C5
 "extras": {}}                    // env 私有调试信息（search=queries/fetched_docs；python=executions）
```

**适配层责任**：rollout 侧把 done 时的 `info["env_state"]` **原样**写入 `trajectory["env_state"]`；
reward 只读 `trajectory`，不得访问 env 实例或 env 私有缓存。

## C4. `task_type → reward 输入` 映射（冻结）

| task_type | 主输入 | 来源 |
|---|---|---|
| `st_fc` | calls vs `reference.gt_calls`（并行同组、顺序无关，参数值/AST 比较） | `trajectory.turns[*].assistant.tool_calls` |
| `irrelevance` | 是否发生任何工具调用 | 同上 |
| `mt_fc` A | `goal_satisfied`（env 为权威验证方） | `trajectory.env_state` |
| `mt_fc` B | `replay_ok` + `matched_turns / gt_turns` | `trajectory.env_state` |
| `search_qa` | `final_answer` vs `final_answers`（多答案优先，规范化 EM/F1） | `trajectory.final_answer` |
| `math_ci` | `final_answer` vs `final_answer`（等价判断） | 同上 |
| `code_exec` | 单测结果摘要 | **唯一** `trajectory.env_state["code_tests"]` |
| `abstain` | 最终回复文本 + 是否调用工具 | `trajectory.turns` + `final_answer` |

`mt_fc` 结果依据优先级（冻结，顺序不可改）：

1. `trajectory.env_state["goal_satisfied"]` 非 null → 直接用它（env 声明了终态断言，权威）；
2. 否则 `reference["final_state"]` 与 `trajectory.env_state["state"]` 都非 null → 深度比较；
3. 否则 `reference["final_answer"]` 非 null → 用 `trajectory.final_answer` 做规范化 EM；
4. 否则视为仅回放模式 → `replay_ok` 决定结果项。

多轮权重（plan §5/§10）：`mt_fc = 0.8 × 结果项 + 0.1 × 逐轮匹配比例 + 0.1 × 格式`；结果项严格占主导。
所有 task_type 的 `score` 冻结在 `[0, 1]`（`schema.validate_reward_result` 强制校验）。

## C5. `env_state["code_tests"]` 结构（唯一真相源）

```json
{"total": int, "passed": int, "failed": int, "all_passed": bool,
 "exit_code": int, "duration_s": float, "truncated": bool,
 "failures": [{"name": str, "message": str}]}
```

- `all_passed` 是权威判分字段（= `passed == total`，两边必须一致）；`failures[].message` 截断 ≤200 字符。
- **绝不包含标准答案代码/候选代码**；`env.config["redact"]` 可加正则脱敏。
- `info["tests"]` 仅供框架日志，reward 不读（D6）。

## C6. 回放返回来源与降质标记（禁止静默）

replay env 取 gt 返回的优先级（v2）：

1. `task.env.config["gt_returns"]`（手工覆盖，测试用）
2. `task.reference["tool_returns"]`（**规范来源**，按轮分组，与 `gt_calls` 对齐）
3. `reference["gt_turns"][*].tool_results / results / return`
4. 规范化 JSON 占位

无论哪支，`env_state.extras["gt_returns_source"]` **必须** ∈ `{"config", "reference", "gt_turns", "placeholder"}`。
placeholder：匹配判定照做但原始返回缺失；reward `info.degraded_replay=True`，**不因此加分**；
T1/T2 必须统计 placeholder 比例，`mt_fc` 中应过滤或降权并在报告给过滤前后条数。

**静态声明（v2.1，数据侧）**：loader 必须在 `meta["gt_returns_source"]` 声明该任务形态
（`{"config", "reference", "gt_turns", "placeholder"}`），供 T2 **不运行 env** 就能过滤/降权；
env 运行时的 `extras["gt_returns_source"]` 是动态真相，两者不一致时以 runtime 为准并写 warning。
`reference["gt_turns"]` 元素 shape（v2.2，legacy 分支）：`{"step": int?, "assistant": {"content": str, "tool_calls": [...]}?, "tool_results": [{"name": str, "content": str, "ok": bool}]}`；env 解析时按 `tool_results` → `results` → `return` 顺序取第一组非空。元素必须是 mapping（schema v2.2 已校验）。
`meta["gt_returns_source"]` 对 `mt_fc` **必填**且枚举校验（v2.2）：`{"config", "reference", "gt_turns", "placeholder"}`（注意是单数 `reference`，不是 `references`）。

## C7. `reference` 必填键（schema v2）

7 个键必须存在，不适用时显式 `null`：
`final_answer`、`final_answers`、`gt_calls`、`gt_turns`、`tool_returns`、`unit_tests`、`final_state`。
- `final_answers`: 非空 list[str] | null（多答案，`search_qa` 优先用它；不要再手写 `|` 分隔）。
- `final_state`: dict | null（`mock_fc` stateful 的 GT 终态；env 据此生成 `goal_satisfied`）。
- `tool_returns`: `[[{name, content, ok?}], ...]` | null（按轮分组）。

## C9. 轮数口径（S2 过滤 / 课程 / 报告统一）

- `meta["max_turns"]`：允许的最大 assistant 轮数（课程 5→10→15 由此覆盖）。
- `meta["expected_turns"]`（可选，int|null）：SFT 轨迹的预期交互轮数；拿不到时用
  `max(len(reference.gt_calls), len(reference.gt_turns or []))` 兜底。
- S2 判定：`effective_turns = max(len(gt_calls), expected_turns or 0, len(gt_turns or []))`，
  `effective_turns > 15` 即丢弃（ASearcher 阈值独立可配，报告写明）。
- 数据报告必须给「轮数分布」（plan §3 步骤 8）以及每步丢弃计数。

## C10. `meta` 必填键（schema v2.1）

4 键必填（值可 null，`max_turns` 必须正整数）：`max_turns / difficulty / pass_rate / license`。
`expected_turns` 可选。流水线写回 `pass_rate` / `license` 时不得删除任何键；
license 必须逐来源核实（plan §10），不确定写 `null` 并在报告标 `[UNVERIFIED]`。

**v2.2 新增可选键**：
- `group_id`（str|null）：会话/分组键，供 group-aware held-out 切分（T2 缺省时退化为按 `id` 切，报告标 `[NO-GROUP]`；schema 已校验类型）。
- `gt_returns_source`（`mt_fc` 必填，四值枚举）。

## C11. search env.config 规范（v2.2，D12）

```json
{"backend": "e5" | "inmemory" | "http", "url": str|null,
 "top_k": int, "max_rounds": int, "retriever": {...}|null}
```

- T1 的 loader 产出**规范**形式；T4 的 `SearchEnv` 期间必须兼容旧别名
  （`backend="local_wikipedia_e5"` → `e5`；`retriever={url, topk}` → `url/top_k`），
  过渡期双端兼容，避免 rollout 直接 `EnvError`。
- 任务级 deadline 必填语义：单任务搜索阶段的累计上限（默认 8 轮 × 单次超时×重试，
  最坏 ~130s），超限按 `info["backend_error"]` 返回而不是拖垮 rollout。

## C12. v2.3 新增字段（T4 收口冻结，2026-11-05 23:05）

- `env_state["code_tests"]["infra_error"]`（bool）：**harness 故障 ≠ 答错**。infra_error=True 时
  `all_passed` 必为 False，但不得被 reward/报告当作"答错"惩罚样本；report 必须单列 infra 失败数。
- `env_state["extras"]` 回放诊断键：`gt_returns_missing`（int）、`gt_returns_declared_source`（str）、
  `gt_returns_missing_ratio`（float）；当 `gt_returns_missing_ratio > env.config["max_missing_ratio"]`
  （默认待 `reports/env_setup.md`，建议 0.25）时 `gt_returns_source` 降级为 `"placeholder"`。
- `env.config["task_deadline_s"]`（float，默认 30，`<=0` 关闭）：单任务搜索 wall-clock 预算；
  HTTP 客户端超时被 clamp 到剩余预算；超限 → `info["deadline_exceeded"]=True`、
  `info["backend_error"]=True`、tool result `ok=False`（**不得拖垮 rollout**）。

## C13. data_report 语义（v2.4，2026-11-06）

- **去污染两个数不能混**：`hits_removed`（= `decontam.total_dropped`，检测到并已剔除，审计列，可 > 0）
  与 `residual_hits`（对 **kept** 再匹配一次的残留数，**门禁列，必须为 0**，plan §8 A2）。
  `scripts/decontam.py` 负责写 `residual_hits`；缺失即视为"未验证"，`report.py --strict` 失败。
- **`--strict` 的 fail-closed 门禁**：六个上游报告齐全（可用 `--allow-missing-steps passrate` 等逐项豁免）；
  held-out 非空；去污染 residual==0 且 BFCL 索引记录数 > 0；replay `failed==0, skipped==0, reward_used>=passed`；
  train/heldout 的 id / dedup-key / group 重叠均为 0；schema 全合法；`mixture_stats.json` 存在且 sha256/行数一致。
- **轮数显示口径**：`effective_turns` 的 C9 公式不变（过滤用）；`data_report` 展示时取 `max(1, effective_turns)`，
  与 `build_mixture.turn_distribution` 对齐，避免 A4 数字与 stats 不可互证。
- **provenance**：上游报告若带 `input_sha256` 则记录；`--require-provenance` 可要求全部带（防 stale report）。

## C14. 单轮预测 vs 执行（v2.5，2026-11-06）

- `format_report(..., expect_tool_results=True)` 默认对 calls ↔ tool_results 做数量/id/名称配对检查（执行型轨迹）。
- **`st_fc` 是单轮预测**：`env.type` 可以是 `"none"`（工具不执行），模型输出的 calls 就是答案，
  没有 tool_results 属正常 → `st_fc.py` 传 `expect_tool_results=False`，跳过配对检查；
  但 call 级检查（未声明工具/必填参数/参数类型）仍然生效。
- 该缺陷由 verl 接入时暴露（`verl_reward_bridge` 与 `agentic_agent` 都不会为 env-none 的 st_fc 造 tool_results）。

## C8. 变更流程

1. 席位发现契约不足 → `send_message` 给 Lead：字段名、语义、为什么现有字段不够。
2. Lead 裁决 → 更新本文件（版本 +1）→ 在 `TEAM_BOARD.md` 记决策 → 广播给受影响席位。
3. 广播前各方按当前版本实现，**不得自行加平行字段或另起结构**。
