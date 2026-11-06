# Phase 1 交付汇总 — agentic_rl_data

> Lead 终审 · 2026-11-05 23:10 (+08:00) · 契约版本 **schema v2.3 / CONTRACTS v2.3**
> 权威规格：`C:\Users\Administrator\Downloads\agentic_rl_data_plan.md`
> 全部代码**静态交付**：本机 `pwsh` 不可用（`0xC0000142`），pytest / 脚本 / 下载 **0 次执行**。

## 1. 团队与分工（1 协调 + 5 席位）

| 角色 | 成员 | 任务卡 | 交付 |
|---|---|---|---|
| 协调 | `lead` | — | schema / CONTRACTS / README / 看板 / 终审、跨席位裁决与派单 |
| 数据调研 | `data-researcher` | task-1 ✅ | 15 源 loaders、build_raw / filter_rounds / dedup / decontam、`data_sources.md` |
| 配比调研 | `mixture-researcher` | task-2 ✅ | `mixture.yaml`、build_mixture（字节可复现）、filter_by_passrate、`mixture_analysis.md` |
| 写 Reward | `reward-writer` | task-3 ✅ | 7 类 reward + base + 2 个测试文件（~80 用例） |
| 环境构建 | `env-builder` | task-4 ✅ | base / mock_fc / search / python_env / code_tests、replay_check、env 单测、`env_setup.md` |
| 检查提问 | `critic` | task-5 ✅ | `acceptance_matrix.md`、`open_questions.md`（39 条）、`review_r1.md`、契约测试 |

协调机制：Lead 维护 `schema.py` 与 `CONTRACTS.md` 两个**单点真相**，跨席位信息一律经 `reports/TEAM_BOARD.md` 汇总并广播；每个席位有独占写范围，冲突用任务卡写范围 + 依赖图隔离。

## 2. 冻结契约（v2.3，详见 `CONTRACTS.md`）

- Task：8 键固定；`reference` 7 键必填（值可 null）：`final_answer / final_answers / gt_calls / gt_turns / tool_returns / unit_tests / final_state`。
- `meta`：4 键必填 + `expected_turns` / `group_id` / `gt_returns_source`（`mt_fc` 必填四值枚举）。
- Trajectory：`turns / final_answer / num_turns / done / env_state`；`env_state` 14 键（含 `goal_satisfied / replay_ok / checks / state / code_tests / extras`）。
- Reward：纯函数 `reward(task, trajectory) -> {"score", "info"}`，score 冻结 `[0,1]`；`mt_fc = 0.8 结果 + 0.1 逐轮 + 0.1 格式`，结果项严格主导；timeout / placeholder 回放不给结果项。
- Env：`reset/step/close`，每 task 一实例、无全局状态；超限 `done=True + info["timeout"]=True` 且终态布尔强制 False。
- 数据侧静态标记：`meta["gt_returns_source"] ∈ {config, reference, gt_turns, placeholder}`；placeholder 样本由 T2 过滤/降权。

## 3. 关键裁决（D1–D15 摘要）

D1 Phase 1 先做框架无关实现；**2026-11-06 修正：训练框架 = verl**（D19 已补齐四个接口映射，见 §11）；D8 schema v2（reference 7 键 + score 范围）；D9 code_exec 单一真相源 `env_state["code_tests"]`；D10/D11 mt_fc 结果依据优先级 + 静态/动态回放标记；D12 search config 双端兼容；D13 `group_id`；D14 abstain 首期 = hammer irrelevance；D15 v2.3 冻结 `infra_error` / 回放诊断键 / `task_deadline_s`。

## 4. 质量闭环（review_r1 → 派单 → 修复）

- critic 静态审查出 **3 个 P0**（loader 复数枚举导致 mt_fc 全丢、mt_fc 不看 timeout、stateful env 超限仍报 `goal_satisfied`）+ **~10 个 P1**（placeholder 仍发 0.8、空 gt 组、decontam 假 0、caps 绝对值、search loader/env 不兼容、search 无 task deadline、code_tests infra 不可区分 等）。
- Lead 逐条裁决并派回 owner；T1–T4 均已落盘修复（Lead 已用 grep/读码复核关键路径：`BACKEND_ALIASES`、`task_deadline_s`、`infra_error`、`gt_returns_source` 静态优先、timeout 强制 False）。
- `tests/test_acceptance_contract.py` 是跨席位契约测试（含此前预期的 3 条红：timeout / placeholder / search env），修复后预计转绿，**需用户侧执行确认**。

## 5. 用户侧验证命令（必须执行）

```powershell
cd D:\lhy\verl\verl-toolRL

# 1) 单测（含跨席位契约）
python -m pytest agentic_rl_data/tests -q

# 2) 数据构建（两次，比 sha256；plan §8 A1）
python -m agentic_rl_data.scripts.build_raw --help
python agentic_rl_data/scripts/build_raw.py --all --cfg-json <你的 {source: cfg}> --out-dir agentic_rl_data/data/raw_unified
python -m agentic_rl_data.scripts.build_mixture --config agentic_rl_data/configs/mixture.yaml --out data/mixture_v1
#   再跑一次 → train/heldout 的 sha256 必须相同（stats 里有输出 sha）

# 3) 去污染（plan §8 A2，需 bfcl-eval 数据）
python agentic_rl_data/scripts/decontam.py --bfcl-path <bfcl-data> --in-dir agentic_rl_data/data/raw_unified

# 4) GT 轨迹重放（plan §8 A3）
python agentic_rl_data/scripts/replay_check.py --limit 10
```

**预期**：全部 pytest 绿；build_mixture 字节一致；decontam 报告 `hits=0` 且 `num_records>0`；replay_check `passed=10/10`。
**已知需阅读后决定**：`filter_by_passrate.py` 需要你先提供 (task_id, pass_rate) 文件（rollout 后端未接，Q2）；`data_report.md` 需按首跑 `_build_report.json` 汇总（下一步）。

## 6. 已知限制与风险

1. 本机无 shell → 全部代码未运行；测试文件头均标注 "NOT EXECUTED"。
2. HF Hub 不可达 → 各 HF 卡片 license/条数多标 `[UNVERIFIED]`，`data_sources.md` 给用户核实命令。
3. 训练框架是 **verl**（不是 slime）：plan §9 四项已映射到 `export_verl_parquet.py` + `agentic_agent` loop + `verl_reward_bridge.py` + `multi_turn.format`，见 §11 与 [`VERL_INTEGRATION.md`](VERL_INTEGRATION.md)。
4. python/code_tests 仅子进程 + rlimit + 超时，无真实网络/文件系统隔离（生产需 bwrap/firejail/容器）。
5. search 的 E5/faiss 依赖与吞吐未压测，参数为 `[SUGGEST]`；placeholder 真实比例待首跑（决定多轮 30% 能否兑现）。
6. `irrelevance` 空轨迹=1 是 plan 字面口径（Goodhart 通道，建议数据侧要求非空回复）。
7. `replay_check` 在 registry 缺失时降级并标 `reward_used=False`。

## 7. 文件索引

| 目录 | 内容 |
|---|---|
| `schema.py` / `CONTRACTS.md` / `README.md` | 单点契约与协作说明 |
| `reports/TEAM_BOARD.md` | 花名册、依赖图、D1–D15、状态与未关问题 |
| `reports/{data_sources,mixture_analysis,env_setup}.md` | 数据源核实、配比依据与消融、环境落地 |
| `reports/{acceptance_matrix,open_questions,review_r1}.md` | 验收矩阵、39 条开放问题、P0/P1/P2 审查 |
| `loaders/`（15 源）· `envs/`（5 类）· `rewards/`（7 类）· `scripts/`（10 个，含 `report.py` / `run_pipeline.py`）· `tests/`（10 个） | 实现与测试 |

## 9. 一键数据构建与启动脚本（2026-11-06 收口）

| 文件 | 用途 |
|---|---|
| `scripts/run_pipeline.py` | plan §3 八步编排：load → rounds → dedup → decontam → replay → passrate → mix → report；`--from/--to/--skip`、`--dry-run`、`--keep-going`、`--strict-report`；结束写 `<data>/_pipeline_manifest.json` |
| `configs/pipeline.json` | 全部路径 + 各步参数；`decontam.bfcl_path` / `passrate.rates` 为 null 时对应步骤显式跳过（不静默） |
| `scripts/build_data.ps1` / `build_data.sh` | 一条命令建数据（PowerShell / bash 包装） |
| `scripts/serve_search.ps1` | 建 E5 索引（`-BuildIndex -Docs ...`）并启动本地搜索服务（`python -m agentic_rl_data.envs.search serve`） |
| `scripts/launch_verl_agentic.sh` | **verl 训练启动**（`SMOKE=1` 冒烟；agent loop=`agentic_agent`；reward bridge + loop 内 reward） |
| `scripts/launch_training_slime.sh` | 仅历史参考（已被 verl 启动脚本取代，文件头已标 DEPRECATED） |
| `scripts/smoke.ps1` | pytest + pipeline dry-run 冒烟 |
| `tests/test_run_pipeline.py` | 编排器单测（配置校验、merge/split、步骤选择、dry-run 不写文件；未执行） |

用户侧最短路径：

```powershell
.\agentic_rl_data\scripts\smoke.ps1 -SkipTests          # 先看 8 步计划
.\agentic_rl_data\scripts\build_data.ps1                # 建数据（需数据源配置 / HF 或本地文件）
.\agentic_rl_data\scripts\build_data.ps1 -Strict        # 验收门禁（去污染/replay/sha 全过才返回 0）
```

## 8. 收口补写（Lead，2026-11-06）

- `scripts/report.py` + `tests/test_report.py`：plan §3 step 8 / §8 A4 的 `data_report.md` 生成器（此前只有规划）。
  - 只读：分布从 `mixture_v1/{train,heldout}.jsonl` 重算，丢弃数读上游报告；
  - 缺失上游报告 → `[NOT RUN]`，不编造数字；
  - 去污染分**审计列**（`hits_removed`，可 > 0）与**门禁列**（`residual_hits`，必须 0；`decontam.py` v2.4 负责计算）；
  - `--strict` 强制：六个上游报告齐全（`--allow-missing-steps passrate` 可临时豁免）、held-out 非空、
    residual==0 且 BFCL 索引非空、replay `failed=skipped=0` 且 `reward_used>=passed`、
    train/heldout 的 id/dedup-key/group 重叠为 0、schema 全合法、输出 sha256 与 `mixture_stats.json` 一致
    （`--require-provenance` 可再要求上游报告带 `input_sha256` 防 stale）；任一失败返回非 0。
- 用户侧命令：`python -m agentic_rl_data.scripts.report --mixture-dir data/mixture_v1 --raw-dir agentic_rl_data/data/raw_unified --out agentic_rl_data/reports/data_report.md --strict --allow-missing-steps passrate`

## 11. verl 接入（2026-11-06，目标框架 = verl）

> 用户明确：**所有背景是 verl**，不是 slime。plan §9 四项已按 verl 落实现（详见
> [`VERL_INTEGRATION.md`](VERL_INTEGRATION.md) 与 TEAM_BOARD D19）。

| plan §9 项 | verl 实现 | 文件 |
|---|---|---|
| prompt / label 格式 | RLHFDataset parquet：`data_source/prompt/tools/reward_model/extra_info`；`extra_info["agentic_task"]` 存整条 task JSON，loop 据此重建 env/reference | `scripts/export_verl_parquet.py` → `data/verl/{train,heldout}.parquet` |
| 多轮 rollout | `AgentLoopBase` 子类，注册名 `agentic_agent`；每样本 `kwargs["tools"]` + env `reset/step`，按 `ToolAgentLoop` 的 token 累积范式；done 时组 trajectory 并**在 loop 内**算 reward（`reward_score` → `rm_scores`，reward manager 不覆盖） | `agentic_rl_data/verl/agent_loop_agentic.py` + `agent_loop_config.yaml` |
| reward 接口 | 单轮类型（st_fc/irrelevance/abstain/search_qa/math_ci）走 `compute_score` 桥接；mt_fc/code_exec 必须由 loop 算（缺 env_state 时桥接 fail-closed 为 0 + 标记） | `scripts/verl_reward_bridge.py` |
| 工具调用格式 | `actor_rollout_ref.rollout.multi_turn.format`（默认 `hermes`，Qwen3 可试 `qwen3_coder`） | 启动脚本 `FORMAT` 变量 |

启动（**smoke 优先**）：

```bash
# 1) 数据：mixture -> verl parquet（run_pipeline 的最后一步也会自动做）
python -m agentic_rl_data.scripts.export_verl_parquet \
  --mixture-dir agentic_rl_data/data/mixture_v1 --out-dir agentic_rl_data/data/verl
# 2) 冒烟：1 epoch / 小 batch / 短 response
SMOKE=1 bash agentic_rl_data/scripts/launch_verl_agentic.sh
# 3) 正式
MODEL_PATH=Qwen/Qwen3-4B NGPUS_PER_NODE=8 TOTAL_EPOCHS=15 \
  bash agentic_rl_data/scripts/launch_verl_agentic.sh
```

关键 override 已写进启动脚本：`rollout.mode=async`、`agent.default_agent_loop=agentic_agent`、
`agent_loop_config_path=agentic_rl_data/verl/agent_loop_config.yaml`、`multi_turn.enable=True`、
`multi_turn.format=$FORMAT`、`reward.custom_reward_function.path=scripts/verl_reward_bridge.py`。

**未执行**：`export_verl_parquet`、`agentic_agent` loop、启动脚本均未在本机跑过（`pwsh` 0xC0000142）。
用户侧需先 `SMOKE=1` 验证：rollout 日志里 tool call 能被解析（否则全 0 分）、env 正常推进、reward 分布合理。
风险：工具 parser 格式与 Qwen3 输出不匹配、search 服务未起（search_qa 会拿 `backend_error`）、
`tools[].parameters` 为 JSON 字符串需在 loop 侧 `tools_for_model()` 解析。
收口时另修两处：**st_fc 格式门**（D20/C14：单轮预测不要求 calls↔tool_results 配对，否则 env-none 的 st_fc 正确调用会被判 0）；
启动脚本默认 `USE_V1=0`（V1/TQ 的 `extra_fields` 序列化未验证，先跑通 V0 async）。
