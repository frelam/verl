# review_r1 — 对 T1–T4 产物的增量审查（第一轮）

> Owner: critic（T5）。只读审查，不改任何席位实现；发现按 `send_message` 报 Lead 转派。
> 快照：2026-11-05 21:52 (+08:00)。**本机 pwsh 不可用（0xC0000142），所有 pytest/脚本均未执行；下文 P0/P1 全部来自静态走查与合成反例。**
> 编号约定：`T1-xx` data-researcher、`T2-xx` mixture-researcher、`T3-xx` reward-writer、`T4-xx` env-builder、`L-xx` Lead 文档/契约、`S-xx` schema 残余。

## 0. 本轮实际读到的文件（未列出的=尚未产出=待审查）

| 席位 | 已读 | 状态 |
|---|---|---|
| Lead | `schema.py`（v2.2）、`CONTRACTS.md`、`README.md`、`reports/TEAM_BOARD.md` | 已复核 v2.2 修复；见 L-01 / S-01..S-04 |
| T1 | `loaders/common.py`、`loaders/{apigen,hammer,search_r1,asearcher,funreason_mt,toucan}.py`；`loaders/{toolace,sealtools,apibank,toolmind,apigen_mt}.py`、`scripts/*`、`reports/data_sources.md` **尚未出现** | 部分 |
| T2 | `configs/mixture.yaml`、`scripts/build_mixture.py`；`scripts/filter_by_passrate.py`、`reports/mixture_analysis.md` **尚未出现** | 部分 |
| T3 | `rewards/{base,st_fc,irrelevance,mt_fc,search_qa,math_ci,code_exec,abstain,__init__}.py`；`tests/test_rewards*.py` **尚未出现** | 部分 |
| T4 | `envs/{base,mock_fc}.py`；`envs/{search,python_env,code_tests}.py`、`tests/test_envs*.py`、`reports/env_setup.md` **尚未出现** | 部分 |

> 本轮结论只覆盖"已读文件"；未产出部分在 §5 标「待审查」，不提前判过/不过。

---

## 1. P0 findings（会算错分 / 数据静默丢失 / 崩溃）

### T1-01 [P0] 多轮 loader 写错枚举值 → 所有 mt_fc 任务被静默丢弃
- **证据**：`loaders/common.py:997-1001` 写入 `meta_extra["gt_returns_source"] = "references"`；schema v2.2 的枚举是**单数** `{config, reference, gt_turns, placeholder}`（`schema.py:129`, 343-352），且对 `mt_fc` 强制必填。`build_task` 会 raise（`loaders/common.py:716-719`）；`load_mt_tasks_from_records` 用 `except Exception: logger.warning; continue` 吞掉（`loaders/common.py:1019-1021`）→ **FunReason-MT / APIGen-MT / Toucan / ToolMind 全部 0 条产出，且日志里只有 warning**。同类错误在 `loaders/funreason_mt.py:13-15`、`loaders/toucan.py:10-12` 的 docstring 里也写成复数。
- **反例输入**：任意一条 `load_mt_tasks_from_records([{...tool_calls...}], source="synthetic")` → 0 条 task。
- **修法**：`common.py:999` 改成 `"reference" if view.get("tool_returns") else "placeholder"`；三个 docstring 同步；`load_mt_tasks_from_records` 把异常计入 `stats["skipped"]` 而不是仅 warning；契约测试 `tests/test_acceptance_contract.py::test_loader_mt_tasks_schema_valid` 已覆盖（当前预期红）。
- **转派**：已在 21:52 消息中报 Lead（Lead 说已通知 T1）；**T1 改完后须重跑该测试**。

### T3-01 [P0] mt_fc reward 不看 `timeout`，超限轨迹可拿 0.8 结果项
- **证据**：`rewards/mt_fc.py:144-160` 只按 `_resolve_result` 给 `result_value=1.0`，全文无 `timeout` 判断（只在 info 里带 `"timeout": env.get("timeout")`，`mt_fc.py:183`）。CT C1（`CONTRACTS.md:21`）明确"超限…不额外扣分（reward 侧只决定**不给 0.8 主项**）"；plan L139 写"超限视为失败"。
- **反例输入**：`env_state={goal_satisfied: True, timeout: True, ...}` + 轨迹其余匹配 → 当前 score=1.0（应 ≤0.2）。
- **修法**：在 `reward()` 计算 result 前加：`if env.get("timeout") is True: result_ok = False; result_basis = "timeout"`；并在 `info` 保留 `timeout`。契约测试 `test_mt_fc_timeout_cannot_earn_result_term` 已覆盖（预期红）。

### T4-01 [P0] stateful mock_fc 在 timeout 时仍报 `goal_satisfied`
- **证据**：`envs/mock_fc.py:1050-1067` 的 `_env_state_fields` 每次都用 `evaluate_goal` 现算；`envs/base.py:414-437` 只在 `_steps >= _max_turns` 时置 `_timeout=True`，**不清零任何终态布尔**。规划中"最后一轮已满足目标但超轮数"的场景会得到 `goal_satisfied=True` + `timeout=True`，叠加 T3-01 → 0.8 结果项照发。
- **修法**：`BaseEnv._build_env_state`（`envs/base.py:620-624`）在 `self._timeout` 时强制 `fields["goal_satisfied"]=False`、`fields["replay_ok"]=False`（或子类覆盖 + 单测）；`MockFCEnv._env_state_fields` 在 timeout 时把 checks 保留但 `goal_satisfied=False`。T4 单测：超 `max_turns` 后 `done=True, timeout=True, goal_satisfied=False`。

---

## 2. P1 findings（验收缺口 / 可复现性 / 静默降质）

### T1-02 [P1] loader 静默丢样本，无丢弃计数
- **证据**：`loaders/common.py:1019-1021` `except Exception: logger.warning; continue`；各 loader 也大量 `if not ...: continue`（如 `apigen.py:56-66`、`asearcher.py:74-76`）。plan L101 要求"禁止静默丢样本，丢弃要记录原因（§3 的 report）"。
- **修法**：loader 返回值/回调携带 `stats`；`scripts/build_raw.py` 汇总每源 `rows/emitted/skipped_by_reason`；报告 §3 step 8 使用该计数。契约测试只能覆盖 schema-valid，计数只能由 T1 的 report 验收。

### T1-03 [P1] ASearcher"≤15 轮过滤"的数据前提不成立
- **证据**：`loaders/asearcher.py:9-13` 自述"raw QA file is question/answer only; the loader does not invent search trajectories"；`_expected_turns`（`asearcher.py:45-52`）只有原始 record 显式带 turn 数时才填。TEAM_BOARD L61 说"原作轨迹 **128 轮** → ≤15 过滤是硬门槛"，但 **QA-only 数据没有轨迹轮数**，`effective_turns`（`build_mixture.py:751-760`）对这类任务恒为 1 → S2 的过滤对 ASearcher 实际是 no-op。
- **修法**：报告明确"ASearcher 的 ≤15 在数据侧不可判定；实际约束由 search env `max_rounds=8`（runtime）承担"；若用户拿到带轨迹的 split 再启用数据侧过滤；§8 A3 不宣称已验 ASearcher 轮数过滤。

### T1-04 [P1] `loaders/` 没有 `__init__.py`/`load_source` 注册表 → `build_mixture --use-loaders` 不可用
- **证据**：当前 `agentic_rl_data/loaders/` 目录只有各 loader 与 `common.py`，无 `__init__.py`；`build_mixture.py:854-869` 的 `_loader_tasks` 读取 `agentic_rl_data.loaders.load_source`，缺失即 `--use-loaders` 直接抛 MixtureDataError。
- **修法**：T1 增加 `loaders/__init__.py`：`LOADERS = {SOURCE: load_tasks}` + `load_source(name, cfg=None)`，未知 source raise LoaderError；与 `scripts/build_raw.py` 共用同一注册表。

### T2-01 [P1] 配比 caps 是绝对值，`total_train` 放大后全部类别饿死（"改一个数字就能放大"是假的）
- **证据**：`configs/mixture.yaml:32-35` 注释"Scale up by editing this one number (or --total-train); shares stay fixed"；但每源 `cap` 是绝对条数（如 single_turn 550/550/400/100/500 = 2100、multi_turn 1400/900/500/300 = 3100、search 1500/1000、math 1400/300、code 500/350/250），`allocate_train_counts`（`build_mixture.py:1071-1113`）按 cap 截断后返回不足 → `total_train=20000` 时各类 target（4000/6000/5000/3000/2000）全部 > caps，实际配比严重偏离（最多 ~9k/20k）。
- **反例**：`--total-train 20000` → single_turn 实际 ≤2100（目标 4000）、multi_turn ≤3100（目标 6000）。
- **修法**：cap 改为"相对份额"（`cap = ceil(share * total_train * weight * 1.5)`）或 build 时按 `total_train` 线性缩放 absolute caps，并在 `mixture_stats.json` 记 `caps_scaled`；config 注释删除"改一个数字即可"的不实承诺。

### T2-02 [P1] build_mixture 离线过滤不读 schema v2.2 的 `meta.gt_returns_source`
- **证据**：`build_mixture.py:763-781` 的 `gt_returns_source(task)` 从 `env.config.gt_returns` / `reference.tool_returns` / `gt_turns` 推断，**完全不读 `task["meta"]["gt_returns_source"]`**；而 schema v2.2 对 mt_fc 强制该键（`schema.py:343-352`）、D11/CT C6 要求"T2 不运行 env 就能按静态标记过滤"，且 runtime 与静态不一致要告警。
- **反例**：`meta={"gt_returns_source":"placeholder"}` + `reference.tool_returns` 有值 → 当前返回 `"reference"`，placeholder 样本进训练。
- **修法**：先读 `meta["gt_returns_source"]`（四值校验），与推断值不一致时以 **runtime 为准**（构建时只能静态，故先按 meta 过滤 + 把推断值写进 `stats["mismatch"]`/warning）；契约测试 `test_mixture_offline_marker_honours_meta` 已覆盖（预期红）。Lead 已通知 T2。

### T2-03 [P1] `abstain` 类在 mixture 中零来源（与 plan §6 "单轮 + 拒答 20%"冲突）
- **证据**：`configs/mixture.yaml:336-343` `when2call.enabled: false`；`single_turn` 分类（L357-384）只有 apigen/sealtools/toolace/apibank/hammer → `task_type=abstain` 的期望条数为 0。plan L180 明确把"拒答"计入 20%；BFCL hallucination（plan L28）也依赖弃权行为。
- **修法**：二选一并写进报告/config：(a) 启用 When2Call train，从 st_fc 挪 5-10% 权重；(b) 声明"首期拒答由 hammer irrelevance 承担，abstain 只做 reward/env 单测、不进训练"并记 TEAM_BOARD 决策。见 open_questions Q-37。

### T3-02 [P1] placeholder 回放只打 info 标记，0.8 结果项照发
- **证据**：`rewards/mt_fc.py:181-184` 仅 `env.get("replay_ok")` 与 `base.degraded_replay(trajectory)` 进 info；`result_value` 仍来自 `_resolve_result`（`mt_fc.py:145,153`），没有 degraded 分支。CT C6（`CONTRACTS.md:96-97`）说 placeholder "不因此加分"；T2 侧本应过滤，但 reward 必须 fail-closed（不能依赖数据侧一定过滤干净）。
- **修法**：`if base.degraded_replay(trajectory): result_ok=False; result_basis="placeholder"`（或封顶 0.2）；契约测试 `test_mt_fc_placeholder_does_not_earn_result_term` 已覆盖（预期红）。

### T3-03 [P1] per-turn 匹配 fallback 丢掉"无调用轮"，空 gt 组永久不匹配
- **证据**：`rewards/base.py:573-588` `derive_turn_match` 用 `nonempty_call_groups(trajectory)`（`base.py:234-236`，过滤掉无调用的 assistant 轮）去对齐 `gt_call_groups`（含空组）。只要 `reference.gt_calls` 里有一个空 turn 组（例如"该轮只回答、不调用"），`align_groups`（`base.py:527-570`）就找不到可匹配的 observed 空组 → `fraction = matched/total < 1`，0.1 项被扣。
- **反例**：`gt_calls=[[call1], []]` + 轨迹 turn1=c1、turn2=无调用回答 → 当前 fraction=0.5（应 1.0）。
- **修法**：改用 `base.call_groups(trajectory)`（包含空组）做对齐；或 filter 掉的只有"轨迹末尾未出现的空组"；补单测：含空 gt 组的重放应满分。

### T3-04 [P1] `make_result` 静默 clamp 越界分数，掩盖 reward bug
- **证据**：`rewards/base.py:819-827` `clamp_score` 把任意越界/NaN 分数压回 `[0,1]`；`make_result`（850-866）先 clamp 再 `validate_reward_result`，所以验收时"分数永远合法"，root cause 不可见（例如 mt_fc 权重写错 1.1 也会变 1.0）。
- **修法**：`make_result` 保留 raw score 到 `info["score_raw"]`，越界时记 `info["score_clamped"]=True`；`strict=True` 模式直接 raise。P2 也可，但这是 P1 因为会掩盖 A3 的"满分"判定。

### T4-02 [P1] replay `_inject_returns` 缺条目时静默伪造 `{"result":"ok"}`
- **证据**：`envs/mock_fc.py:1256-1263`：当 `reference.tool_returns` 的条目数少于该轮 calls 或名字对不上时，依次 fallback 到"任意未用条目"再 fallback 到 `{"result": "ok"}`，且 `extras["gt_returns_source"]` 仍是 `"reference"`（`mock_fc.py:1273-1295` 只在 placeholder 时置 degraded）。这正是 CT C6 禁止的"静默降质"。
- **修法**：缺条目/名字不符时：`extras["gt_returns_missing"] += n`，该 result `ok=False`；若缺失比例超阈值则把 source 降级为 `placeholder`（让 reward/T2 看得见）；补单测。

### T1-05 [P1] loader 把 `meta.max_turns` 抬到本任务轮数，绕过 plan ≤15 语义
- **证据**：`loaders/common.py:983-988,1014`：`max_turns_value = max(max_turns, turn_estimate)`，128 轮轨迹的 task 会写入 `meta.max_turns=128`。S2 过滤（`build_mixture.py:1006-1008` 用 `min(cfg.max_gt_turns, category.turn_cap)`）能兜住，但 schema/报告/monitor 看到的 `max_turns` 已不是 plan 约束值；验收矩阵 C9 断言会失真。
- **修法**：`meta.max_turns` 固定 ≤15（`min(15, ...)`），长轨迹任务直接由 loader 丢弃并计数（或标记 `meta.expected_turns` 后由 S2 丢）。

---

## 3. P2 findings（口径/文档/低风险）

| ID | 位置 | 问题 | 修法 |
|---|---|---|---|
| T1-06 | `loaders/hammer.py:36-47` | `answers` 缺失被当作 `[]`（irrelevance），与 docstring"unparseable skip"语义混淆；缺失标签可能不是真负样本 | 区分 `None`（字段缺失→skip）和 `[]`（明确无调用→negative） |
| T1-07 | `loaders/apigen.py:59-66` + `rewards/base.py:718` | tools 字段解析失败 → `tools=[]` 仍可通过 validate；st_fc 的 format gate 在 `declared` 为空时**不报 undeclared tool** → 无工具声明也能满分 | loader 对 st_fc/mt_fc 要求 tools 非空；`format_report` 在 `tools==[] and calls` 时记 issue |
| T2-04 | `configs/mixture.yaml:57-79,128-139` | `passrate.missing=keep`、`require_report=false`、`decontam.report_required=false`：phase-1 允许**未执行过滤**就产出"final" mixture，stats 只记 switch 不记"已执行" | stats 增 `filters_applied: {passrate:false, decontam:false}`；`data_report.md` 用 `[NOT RUN]` 标注，禁止把未执行当已验收 |
| T2-05 | `scripts/build_mixture.py:1499-1501` | 死代码 `task for ... .get_any if False`（恒为空列表） | 删除；保留下方显式循环 |
| T2-06 | `configs/mixture.yaml:85-96` | held-out 口径是 5 个配比类 ×300=1500（非 7 个 task_type），plan L130 未定义口径；`oversample:2` 实际是 `+2` 条候选（`build_mixture.py:941-944`），注释"reserve this many targets"有歧义 | 报告写明口径；注释改成"额外候选条数"；见 Q-13 |
| T2-07 | `configs/mixture.yaml:90-91` + `build_mixture.py:809-814` | `group_aware` 依赖 `meta.group_id`，但 schema/CONTRACTS 无此字段 → 静默退化为按 id 切（K9/Q-35） | Lead 增可选 `meta.group_id`；无 group_id 的来源在报告标 `[NO-GROUP]` |
| T3-05 | `rewards/base.py:718-720` | 同 T1-07（空 declared 时调用合法性门失效） | `if (not declared and calls) or (declared and name not in declared)` |
| T3-06 | `rewards/base.py:322-323` | `rel_tol=1e-6` 对大整数会误判相等（如 10^9 与 10^9+1000） | int-int 精确比较；float 才用 isclose |
| T3-07 | `rewards/base.py:326-331` | `values_equal` 要求参数 key 集合完全一致 → 模型补可选参数（有默认值）也判错；与 DESIGN.md "参数值匹配"口径不必一致，但需在单测/文档固定 | 明确产品口径：可选参数默认值是否忽略；按工具 schema 的 `required` 决定（当前无文档） |
| T4-05 | `envs/base.py:734-760` | `build_env` 懒导入 `.search/.python_env/.code_tests` 尚未产出 → 目前调用 mock_fc 以外的 env 会 ImportError | T4 补齐后再核；单测覆盖 5 类 env 工厂 |
| T4-06 | `envs/base.py:361-390` | `close()` 后再 `reset()` 会把 `_closed=False` 复活实例（生命周期语义不清） | 关闭后 reset 应 raise，或文档明确"close 后可复用=重新打开" |
| T4-07 | `envs/base.py:250-283` | `make_env_state` 不校验 `checks` 元素形状（CT C3 要求 `{name,ok,detail}`） | 在 env 单测中断言 checks shape |
| L-01 | `reports/TEAM_BOARD.md:3,30-40` | 标题/§2/§3 仍写"已冻结 v1"（实际 schema v2.2）；T1 事实与 D6-D11 已追加但版本标注未同步 | Lead 更新版本号与"已冻结 v2.2" |
| S-01 | `schema.py:167-335` | `validate_task` 不拒绝未知键（plan L78 "键固定"） | 可选 `strict_keys`；或 README 声明允许扩展 |
| S-02 | `schema.py:377-390` | `tool_results` 只查 name/content，不查 `ok` bool 与 `id` 类型（CT C2） | 加校验或在 verify test 中覆盖 |
| S-03 | `schema.py:456-477` | `dump_jsonl` 仅保证 dict 键顺序，任务顺序由调用方决定（A1 依赖） | 已由 T2 的 `(source,id)` 排序 + stats sha256 覆盖；README 写明调用约定 |

---

## 4. 已核对项（无问题，列出防"漏审"）

1. **schema v2.2 修复复核**：reference 7 键必填（`schema.py:113-122, 244-248`）；`tool_returns` 按轮分组校验（289-320）；`meta` 4 键 + `expected_turns`（124-125, 316-333）；score `[0,1]` 强制（109-111, 412-435）；`tools_signature` 按 name 排序（526-540）；`gt_turns` 元素 mapping 校验（289-297）；mt_fc `meta.gt_returns_source` 四值强制（343-352）。F-01/F-02 关闭。
2. **reward registry**：7 个 task_type 全覆盖（`rewards/__init__.py:39-47`），`get_reward` 未知类型 raise（52-69）；7 个 reward 均只 import `.base`（`base` 只 import schema/stdlib），未发现 env 实例/全局状态访问 → 双真相源（K1）静态通过。
3. **reward 权重**：mt_fc `RESULT_WEIGHT=0.8/TURN_MATCH_WEIGHT=0.1/FORMAT_WEIGHT=0.1`（`mt_fc.py:38-40`），`_resolve_result` 的优先级与 CT C4 一致（71-105）；`_turn_match` 优先 env `matched_turns/gt_turns`（108-134）；结果项**不是**调用匹配的别名（除 replay_ok，而 replay_ok 由 env 判定）→ K2 静态通过（但 T3-02/T3-01 会破坏它）。
4. **search_qa/math_ci/reward 纯函数**：search_qa 单答案/多答案（`final_answers` 优先）与 EM/F1 实现正确（`search_qa.py:41-78`, 104-163）；math_ci 移植 verl 的 `strip_string/is_equiv` 且 `compute_score` 兼容（`math_ci.py:40-232`）。
5. **code_exec 单一真相源**：只读 `env_state["code_tests"]`（`code_exec.py:59-68`），`all_passed` 与计数一致性 fail-closed（80-99）；未发现 `info["tests"]` 读取 → K3 静态通过。
6. **env_state 形状**：`make_env_state` 14 键与 CT C3 对齐（`envs/base.py:250-283, 77-93`）；`BaseEnv.reset` 全量重建所有字段（361-401），无模块级可变状态。
7. **mock_fc 回放**：GT 返回优先级与 CT C6 一致（`envs/mock_fc.py:239-273`），每次 done 分支都写 `extras.gt_returns_source`（1273-1295），turn 内并行调用顺序无关匹配（144-178）；stateful 三域（filesystem/social/trading）实现带终态断言（635-935）。
8. **mixture 确定性结构**：`rank_of` 用 sha256(seed\x1f salt\x1f id)（`build_mixture.py:740-743`），`_keep_best` 有界堆、tie-break 按 id（903-923）；输出按 `(source,id)` 排序（1561-1592）；stats 无时间戳、含输出 sha256（1336-1368, 1527-1536）→ A1 的**结构**条件满足（执行仍需 shell）。
9. **config 内部一致性**：五类 share 和=1、每类 source weight 和=1（`parse_config` 校验 669-684）；heldout ≥200 校验（505-510, 624-630）；curriculum 连续/递增校验（571-587）；`synthetic`/YAML fallback 覆盖 config 所用语法（114-334）。**config 里的数字可复核**：1600/16947=9.4%（4900+4873+4033+2980+161=16947 正确）；300 条 held-out 的 95% CI ±5.7pt（SE≈0.0289 正确）；`0.95^8≈0.663` 正确。
10. **loaders 静态**：`apigen`/`hammer`/`search_r1`/`asearcher` 的原始字段/ID 与 TEAM_BOARD 核实结论一致（Search-R1 = FlashRAG nq；ASearcher = inclusionAI repo）；`common.extract_trajectory` 把 assistant/tool 轮折叠为 system/user + `<tool_response>`，符合 schema 的 role 限制；`tool_returns` 只在**全组**有真实返回时才填（896-910）→ 与 CT C6 placeholder 语义一致（除 T1-01 枚举问题）。

---

## 5. 待审查（文件出现后下一轮核对）

| 对象 | 核对点 |
|---|---|
| T1 `scripts/{build_raw,dedup,decontam,filter_rounds,report}.py` | S1-S4 可执行性；decontam 同时匹配 query+tools、0 命中报告；drop 计数进 report；loader 注册表 |
| T1 剩余 loaders（toolace/sealtools/apibank/toolmind/apigen_mt/tau/when2call/a2search/retool/simplerl） | 每个 loader 合成记录 → `validate_task` 空；`canonical` 字段；license 标注 `[UNVERIFIED]`；drop 统计 |
| T4 `envs/{search,python_env,code_tests}.py` + `tests/test_envs*.py` | search 超时/降级返回、并发；code_tests 不泄漏答案/`code_tests` 结构；timeout 终态 false；env_state 键集合；`build_env` 5 类工厂 |
| T3 `tests/test_rewards*.py` | 7 类正/反/边界；timeout/placeholder/degraded；st_fc 并行顺序置换；search_qa 多答案；code_exec 矛盾摘要 |
| T2 `scripts/{filter_by_passrate,build_mixture}.py`（改后）+ `mixture_analysis.md` | 静态 marker 先读 meta（T2-02）；caps 缩放（T2-01）；两次构建 sha256；held-out/train 不相交；`data_report.md` 统计出处 |
---

## 6. 二轮增量（2026-11-05 21:52–22:05 新产出，已读）

> 这一节覆盖第一轮写作后才出现的文件：T1 的 `scripts/{build_raw,filter_rounds,dedup,decontam}.py`、`loaders/__init__.py`、T4 的 `envs/{search,python_env,code_tests}.py`、T2 改后的 `build_mixture.py`、T3/T1 的 `tests/`。未读的（`scripts/report.py`、`replay_check.py` 细节、`filter_by_passrate.py`、`data_sources.md`、`env_setup.md`）仍在 §5 待审查。

### 6.1 已修复（复核通过，仅留 P2 尾巴）

- **T1-01（原 P0）已修**：`loaders/common.py:1004-1014` 现在写 `config | reference | gt_turns | placeholder` 单数枚举（`gt_source=""reference""`），并区分 manual/config 覆盖；`loaders/__init__.py` 已提供 `SOURCE_REGISTRY` + `load_source`（修复 T1-04）。**残余 P2**：`loaders/funreason_mt.py:13`、`apigen_mt.py:10`、`toucan.py:11` 的 docstring 仍写复数 `"references"`，会误导下一个读代码的人。
- **T2-02（原 P1）已修**：`build_mixture.py:796-840` 新增 `declared_gt_returns_source` / `infer_gt_returns_source` / `gt_returns_source()->(kind, origin)`，静态声明优先、并记录 `gt_returns_sources` 计数与 mismatch warning。契约测试已改为解包 tuple（`origin=="static"`）。
- **T1-02（silent skip）仍未修**：`loaders/common.py:1033-1035` 依旧 `except Exception: logger.warning; continue`，且没有把 skipped 计数写进任何 stats（`build_raw.py` 是否汇总待核）。plan L101 的"丢弃要记录原因"仍未满足。

### 6.2 新 P0：T1 loader 与 T4 SearchEnv 的 env.config 契约不匹配

- **证据**：`loaders/common.py:608-614` 的 `env_search()` 产出
  `{"type": "search", "config": {"max_turns":8, "retriever": {"url": "...", "topk": 3}, "backend": "local_wikipedia_e5"}}`；
  而 `envs/search.py:1144-1209` 的 `_resolve_backend` 只接受 `inmemory | http | e5`，且 http 分支读 `config["url"]`/`config["top_k"]`，不认 `retriever`/`topk`/`local_wikipedia_e5` → **每个 search_r1 / asearcher 任务在 `build_env()` 时 raise `EnvError("unknown search backend 'local_wikipedia_e5'")`**，search 类 25% 数据无法 rollout。
- **修法（二选一，建议 T1 改）**：T1 `env_search()` 改为
  `{"type":"search","config":{"backend":"http","url":retriever["url"],"top_k":retriever.get("topk",3),"max_turns":8}}`；
  或 T4 在 `_resolve_backend` 把 `local_wikipedia_e5` 映射为 `e5` 并接受 `retriever` 嵌套键。改完后 `tests/test_acceptance_contract.py::test_search_loader_env_config_is_constructible` 应从红转绿。
- **状态**：已报 Lead（见 22:05 消息）。

### 6.3 新 P1：decontam 的"0 命中"仍可能是假 0 / 口径不足（A2）

- **假 0**：`scripts/decontam.py:270` `index = build_index(read_any_records(args.bfcl_path))` 不检查 `index["num_records"]`；路径写错/目录为空/只给了单个 split 时，报告照样 `total_dropped=0` → "0 命中"是**未检查**而不是**无污染**。修法：`num_records==0` 直接退出码 2；报告记录 BFCL source 路径、版本/sha256、覆盖的 categories 数（至少要求 v3 multi-turn + v4 agentic web-search/memory 的 data 子目录）。
- **只有 n-gram、没有近似匹配**：`match_task`（`decontam.py:146-185`）只做 word n-gram 命中（默认 `k=13`, `min_hits=1`）；config 的 `jaccard: 0.90`（`configs/mixture.yaml:135`）在 decontam 里**完全没被使用**，plan L127 的"n-gram + 近似匹配"只落实了一半。修法：对 query/tool 描述加 5-shingle MinHash/Jaccard（复用 `schema.text_shingles`），阈值可配；报告分别给 exact/approx 命中数。
- **短文本 5-char fallback 过激**：`text_ngrams`（`decontam.py:52-70`）在 <3 token 时退化为 5 字符 n-gram，`min_hits=1` 会把 `"paris"` 这类常见词误杀；建议短文本只匹配 tool name/整串，或提高 min_hits。
- **命中记录截断**：`report["hit_details"].extend(dropped[:50])`（`decontam.py:308`）每个输入文件只留 50 条，A2 要求"命中的已被剔除并有记录" → 大规模命中时不可追溯；至少记录全量 id 摘要 + 分来源计数。
- **验收**：A2 仍 `[ext]`（需 BFCL 数据 + shell）；静态只能验逻辑存在，不能验 0 命中。

### 6.4 新 P1：search 客户端的失败延迟预算没有任务级上限（Q-27/§10 吞吐）

- **证据**：`envs/search.py:57-62` 默认 `request_timeout_s=5.0`, `max_retries=2`, backoff 0.5/1.0 → 单次失败 search 最坏 ≈5+0.5+5+1+5≈16.5s；`SearchR1` 任务最多 8 轮 → 单任务最坏 ≈132s，且 `SearchEnv` 只有轮数上限（`_turn_cap` 8），没有**任务级 deadline**，`envs/base.py` 的 timeout 也只按轮数。若 backend 抖动，rollout 线程会被拖死。
- **修法**：`env.config.task_timeout_s`（建议 60-90s）+ 每轮累计 deadline；超时后返回统一的 `ok=False` 降级 obs 并 `done=True, timeout=True`（reward 侧不给 0.8，与 K5 一致）；`bench_search`（`search.py:1239-1295`）已有压测函数，报告需给真实 p95/QPS 数字（当前 `env_setup.md` 未产出 → 待审查）。

### 6.5 新 P2/P1：code_tests 的 infra 失败与答错不可区分（Q-25）

- **证据**：`envs/code_tests.py:446-465`：`timed_out` 时 `all_passed=False`、`exit_code=-1`；`rewards/code_exec.py:80-99` 只认 `all_passed` → 超时/沙箱失败被当成答错，给 0 梯度。
- **修法**：reward 在 `exit_code==-1` 或 `truncated/infra` 时返回 `info["invalid"]=True`（score 0 但训练侧应剔除）；或在 `code_tests` 增加 `invalid: bool` 字段（改 CT C5 需 Lead）。
- **已核对（无问题）**：`code_tests` 的唯一真相源是 `env_state["code_tests"]`（`code_tests.py:479-490` 写、`rewards/code_exec.py:61-68` 读），`failures[].message` 截断到 200 且过 `redact`（`code_tests.py:127-135,176-231`）；不把候选/标准答案代码回显（`_handle_action` 只返回 summary，`code_tests.py:299-361`）。K3 静态通过。

### 6.6 新 P2：其它静态发现

| ID | 位置 | 问题 | 修法 |
|---|---|---|---|
| T1-08 | `loaders/__init__.py:54,63` | `SOURCE_TASK_TYPE` 里 `hammer: "irrelevance|st_fc"`、`when2call: "st_fc|abstain"` 是非法 task_type 字符串，若被 report/过滤当枚举用会 KeyError | 改成 tuple/两值，或独立 `produces` 列表 |
| T1-09 | `scripts/build_raw.py`（未细读） | 需核对：每源 drop 计数、`dump_jsonl` 排序、与 `loaders/__init__` 注册表一致 | 待审查 |
| T4-08 | `envs/search.py:944-970` | `max_rounds` 被硬夹到 8（`SEARCH_R1_MAX_ROUNDS`），但 `min(task.max_turns, ...)`；当课程 stage=5 时 `_turn_cap` 返回 5，OK；文档与 config turn_cap=8 一致 | 已核对通过 |
| T4-09 | `envs/search.py:1012-1015` | 模型传 `top_k` 超过 backend 支持时未夹到上限（只 `max(1, int)`），可能拉大返回体积 | 在 `_search_call` 夹 `min(top_k, 10)` |
| T2-08 | `scripts/build_mixture.py:791-793` | 用 `hasattr(schema,"GT_RETURNS_SOURCES")` 兜底，但 schema v2.2 一定有该常量；兜底掩盖导入问题 | 直接 `from agentic_rl_data.schema import GT_RETURNS_SOURCES` 或用显式 try/except 报错 |

### 6.7 二轮测试状态（仍未执行）

`tests/test_acceptance_contract.py` 现在共 4 条**预期红**：`test_mt_fc_timeout_cannot_earn_result_term`（T3-01/T4-01）、`test_mt_fc_placeholder_does_not_earn_result_term`（T3-02）、`test_search_loader_env_config_is_constructible`（6.2）、以及可能的 `test_single_turn_loader_schema_valid`（取决于 apigen tools 解析）；其余应绿。另 T1/T3 的 `tests/test_loaders.py`、`tests/test_rewards*.py` 已产，但**尚未逐条读**（下轮核对覆盖度）。

### 6.8 三轮增量：`scripts/report.py` / `tests/test_report.py` 复核（2026-11-06 09:35）

> 范围：`report.py`（992 行）、`test_report.py`（384 行）；同时对照上游报告字段：`build_raw._build_report.json`、`filter_rounds._filter_rounds_report.json`、`dedup._dedup_report.json`、`decontam._decontam_report.json`、`filter_by_passrate` report、`replay_check` report、`mixture_stats.json`、`filter_rounds.effective_turns`、`build_mixture.filters_applied`。**未执行**。

**已核对通过（正面项）**：①只读 + 无时间戳 + 排序/`json.dumps(sort_keys)` → 两次渲染字节一致（`report.py:606-905`）；②分布全部从 `train/heldout.jsonl` 重算（179-309），不依赖上游数字；③缺失报告渲染 `[NOT RUN]`，不编数字（680-785）；④8 个小节覆盖 plan §3 step 8 的 6 项统计 + license/gt_returns/group 覆盖；⑤schema 全量校验、id/dedup-key/group 重叠、sha256 重算、strict 规则可读（70-76, 445-527）；⑥replay report 字段匹配（`replay_check.py:599-620` 的 total/passed/failed/skipped/reward_used/placeholder_ratio/gt_returns_source_mismatch）；dedup 字段匹配（`dedup.py:225-228,274`）；`filter_rounds` per-source `{in,kept,dropped}` 匹配（`filter_rounds.py:98-109`）；`filters_applied` 契约存在（`build_mixture.py:1557-1566`）；`scripts/__init__.py` 存在，`from agentic_rl_data.scripts.filter_rounds import effective_turns` 可导入。

#### R2-01 [P0] strict 的 A2 语义错误：把"已剔除的命中数"当"残留命中数"
- **证据**：`report.py:458-460` `decontam_hits = decontam.get("total_dropped")`；`report.py:495-503` strict 要求 `decontam_hits == 0`。但 `decontam.py:188-211` 的 `decontaminate()` 把命中任务**丢弃**并在 `total_dropped` 计数（`decontam.py:300-311`），`total_dropped>0` 表示"发现了污染并已剔除"，这正是 plan L208 括号里"命中的已被剔除并有记录"的**合规**状态。当前实现导致：**正确剔除了 3 条污染 → strict 失败**（假阴性）；而只喂 BFCL 的一个 split/残缺语料、`bfcl_records=42` 且 0 命中 → strict 通过（假阳性）。测试 `test_strict_fails_on_contamination_hits`（`test_report.py:284-304`）把错误语义固化。
- **修法**：报告同时输出 `hits_detected/removed`（允许 >0、必须逐条记录）与 `residual_hits`（最终 train/heldout 对 BFCL 索引重新匹配后的残留，必须 0）；strict 判 `report present + bfcl_records>0 + residual_hits==0`，并校验 BFCL 覆盖清单（v3 multi-turn + v4 agentic web-search/memory 的 data 子目录）；`total_dropped` 只作审计列，不参与通过判据。同步改 `test_report.py` 的 decontam 用例。

#### R2-02 [P1] strict 可以在上游步骤未跑、held-out 缺失时通过
- **证据**：`report.py:491-527` 只对 decontam/replay/schema/overlap/sha 做 strict 失败；`build_raw`/`filter_rounds`/`dedup`/`passrate` 缺失只进 `not_run` 列表（491）与警告（529-540），不失败；`heldout.jsonl` 不存在时 `heldout=[]`（427）且无 strict 检查；`filters_applied` 只渲染（813-817），不参与判定。`test_report.py:323-335` 的 `test_strict_passes_with_complete_reports` 只给 decontam+replay、heldout 为空，却断言 `strict_failures == []` → **测试把假通过当预期**。
- **修法**：strict 要求 §3 step 1/2/3 报告齐全、`heldout_rows>0`、`filters_applied.passrate/decontam == true`（或显式 `--allow-unrun` 逃生门用于 smoke）；测试相应补一例 `heldout 缺失 → strict 失败`。

#### R2-03 [P1] replay 的 `skipped` / `reward_used` 未校验 → A3 可假通过
- **证据**：`report.py:461-462` 只取 `failed`；`report.py:504-507` 仅 `replay_failed != 0` 失败。`replay_check.py:556-562` 会 `skipped += 1; total -= 1`，summary 有 `skipped` 与 `reward_used`（`replay_check.py:599-620`）。若 env/reward 未就绪导致全部 skipped，`failed=0` → strict 通过，但 A3 "gt 轨迹重放 reward 满分" 完全未验证；`reward_used=0`（只比调用匹配、未用 reward）也通过。
- **修法**：strict 增加 `skipped == 0`、`reward_used >= passed`（或 `reward_used == total`）、`placeholder_ratio` 阈值告警；报告 §3.5 渲染这些字段。

#### R2-04 [P1] 默认报告发现目录漏掉各 step 的默认输出目录
- **证据**：`report.py:118-132` 只搜索 `mixture_dir`、`mixture_dir.parent`、`raw_dir`、`raw_dir.parent`（= `data/` 两层）；而默认输出是 `decontam.py:326` `data/decontaminated/`、`dedup.py:255` `data/deduped/_dedup_report.json`、`filter_rounds.py:136` `data/filtered_rounds/`。按 `report.py` 自己的 §8 复现命令跑完后（`report.py:896-902`），decontam 报告在 `data/decontaminated/`、dedup 在 `data/deduped/`，**report.py 默认找不到** → 全是 `[NOT RUN]`，strict 必然失败（或用户被迫手传 `--report-dir`）。另外 `report.py:897` 写 `data/rounds_filtered` 与 `filter_rounds.py:136` 的 `data/filtered_rounds` 拼写不一致。
- **修法**：`_candidate_dirs` 增加标准子目录 `data/{raw_unified,filtered_rounds,deduped,decontaminated,mixture_v1}`；§8 命令统一为实际默认值。

#### R2-05 [P1] `effective_turns` 双实现漂移：data_report 与 mixture_stats 的轮数分布对不上
- **证据**：`report.py:51` 用 `filter_rounds.effective_turns`（`filter_rounds.py:40-51`，严格按 CT C9 `max(len(gt_calls), expected_turns, len(gt_turns))`，无下限）；`build_mixture.py:799-808` 是另一份实现并多了 `, 1)` 下限（偏离 CT C9）。对 `search_qa` 无 gt_calls/expected_turns/gt_turns 的 QA 对：data_report 记 0（落 "1" 桶），mixture_stats 的 `turn_distribution` 记 1 → A4 的"轮数分布"与 `mixture_stats.json` 不可互证（违反"数字可追溯"）。
- **修法**：把 `effective_turns` 收敛到单一位置（建议 `schema.py` 或 `filter_rounds`，`build_mixture` 直接 import），并决定是否保留下限；若保留，Lead 同步改 CONTRACTS C9 并广播。`report.py` 与 `build_mixture.py` 同时引用同一函数。

#### R2-06 [P1] stale upstream report 可假通过（无输入绑定）
- **证据**：`discover_reports`（`report.py:135-171`）只要文件名匹配就采用；strict 的 sha 校验只比较"当前 JSONL vs mixture_stats"（`report.py:448-482`），两者可以来自同一次旧构建，而 decontam/replay 报告来自更早的数据版本；报告没有把上游输入路径/摘要纳入 strict 校验（`report.py:542-577` 的 `inputs` 只是渲染）。
- **修法**：上游报告写入输入文件的 `sha256`（decontam/replay/filter_by_passrate 已记录路径，补 digest），strict 对比当前 `data/deduped`/`data/raw_unified` 摘要；或引入 run-id（上游所有报告共享同一 `pipeline_run_id`，report 校验一致）。

#### R2-07 [P1] mixture_stats 缺 `outputs`/sha 时 strict 仍然通过
- **证据**：`report.py:474-482` `train_sha256_match` 在 recorded sha 为空时为 `None`；`report.py:524-527` 只对 `False` 失败；stats 文件缺失只是警告（535-536）。因此删除 `mixture_stats.json` 或删掉 `outputs` 字段 → 可复现性检查"未知"却被 strict 放过。
- **修法**：strict 要求 stats 存在、`outputs.train/heldout.sha256` 非空、且 `count == len(rows)`；`None` 视为失败。

#### R2-08 [P1] `group_overlap` 计算了但 strict 不判
- **证据**：`compute_overlap` 返回 `group_overlap`（`report.py:333`），`checks["group_overlap"]`（472）、§6 渲染（857），但 strict 分支（508-527）没有 `group_overlap`。K9/Q-34 的会话级泄漏可径直通过 strict。
- **修法**：`if checks["group_overlap"]: strict_failures.append(...)`；无 group_id 时另给 `[NO-GROUP]` 警告（当前只有渲染）。

#### R2-09 [P2] 三个上游字段名不匹配（渲染为空，不报错）
- `build_raw`：`report.py:358` 读 `payload.get("unattributed")`；上游 `SourceStats.as_dict()` 写的是 `unattributed_skips`（`loaders/common.py:121-130`）→ "unattributed" 列永远空。
- `passrate`：`report.py:391` 读 `stats.get("missing_rate")`；`filter_by_passrate.build_report` 的 stats 是 `{input_tasks, kept, kept_without_rate, dropped, drop_reasons, ...}`（`filter_by_passrate.py:399-409`）→ missing 列空。
- `decontam`：`report.py:378` 读 `hit_counts_by_source`；`decontam.py:288-311` 的 report 没有该键 → 空。
- **测试保真度问题**：`test_report.py:156-167` 的 `_good_decontam` 自造了 `hit_counts_by_source`，与上游 decontam.py 不一致 → 测试绿不能证明端到端字段对齐。修法：改 report.py 读正确键/派生；fixtures 用上游真实 shape（从 decontam.py 输出构造）。

#### R2-10 [P2] §0 摘要渲染 `None / None`
- `report.py:631-632` 对 `total_train_target/actual` 直接 f-string，stats 缺失时显示 `None / None`，与其余格用 `_fmt` 的 `—` 不一致；不影响判定，统一格式即可。

#### R2-11 [P2] 0 轮任务落进 "1" 桶 + `_turn_buckets` 语义
- `report.py:183-196`：`turns <= 1` 都进 `"1"` 桶；配合 R2-05 的 0 下限，data_report 会把 0 轮样本标成"1 轮"。建议单列 `"0"` 桶或先统一 `effective_turns` 语义。

#### R2-12 [P2] `test_report.py` 覆盖缺口（12 个用例，非 10）
- 已有：分布重算、overlap、[NOT RUN]、strict 缺报告/命中/空索引/sha/schema/overlap；缺：stale report、heldout 缺失、replay skipped/reward_used、group_overlap strict、stats 缺失/`outputs` 缺失、`filters_applied=false`、`build_raw` 字段名；`test_overlap_detects_ids_keys_and_groups`（`test_report.py:215-228`）docstring 说 keys 但只断言 id/group，未断言 `dedup_key_overlap`。`test_strict_passes_with_complete_reports`（323-335）应随 R2-02 改为"补齐上游步骤 + heldout"后才绿。

**结论（回应 Lead 的 5 个重点）**：①数字可追溯——分布/mixture 部分成立，上游字段有 3 处空值（R2-09）；②缺失报告是否真的不编数字——markdown 层成立，但 strict 未把所有缺失步骤判失败（R2-02）；③strict 假通过——**存在且不止一处**，最严重是 R2-01（A2 语义反了）+ R2-02/03/06/07/08；④与 `mixture_stats.json`/各 step 字段——大体对齐，3 处字段名漂移 + `filters_applied` 未判；⑤`effective_turns`/`task_dedup_key`——`task_dedup_key` 用法正确，`effective_turns` 双实现漂移（R2-05）。**A4/S8 状态：实现已交付、静态结构通过；strict 判定逻辑需按 R2-01..R2-08 修正后才能作为验收门禁，执行仍待 shell。**

---

## 7. 给 Lead 的优先级建议

1. **立即**：修 §6.2 的 search env 契约（否则 25% 数据不可训练）；确认 T1-01 的单数修复已进最终 loader 文件。
2. **立即**：T3/T4 修 timeout 0.8 漏洞（T3-01+T4-01，P0）与 placeholder 加分（T3-02）。
3. **本轮收尾前**：T2 修 caps 缩放（T2-01）与 abstain 零来源（T2-03）；T1 补 decontam 假 0 防护 + 近似匹配（6.3）。
4. **验收时**：A2/A3/A5/A6 仍需 `[ext]`，报告不得标"已通过"；A1 用两次构建 sha256 验。
5. 下一轮 review 重点：`scripts/report.py` + `data_report.md` 数字出处、`replay_check.py` 全量重放、`filter_by_passrate.py` 与 N=8 区间、剩余 envs 的 timeout 终态、T3 单测覆盖度。

