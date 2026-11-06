# agentic_rl_data — Agentic RL 初期训练数据流水线

本目录实现用户提供的实施规格 [`agentic_rl_data_plan.md`](../../../../../../Users/Administrator/Downloads/agentic_rl_data_plan.md)
（文档全文在 `C:\Users\Administrator\Downloads\agentic_rl_data_plan.md`）中的
**框架无关部分**：统一 schema、各来源 loader、去重/去污染/轮数与通过率过滤、
数据配比、每种 `task_type` 的 reward、环境接口，以及报告与单测。

> plan §9 的四个接口在本项目映射到 **verl**（不是 slime）：数据 → RLHFDataset parquet、
> 多轮 rollout → 自定义 agent loop `agentic_agent`、reward → loop 内 `reward_score` + 单轮
> `custom_reward_function` 桥接、工具格式 → `multi_turn.format`。详见
> [`reports/VERL_INTEGRATION.md`](reports/VERL_INTEGRATION.md)。

> **v2（Qwen3-4B，10–15 轮 agentic RL）**：设计见 [`reports/AGENTIC_RL_V2.md`](reports/AGENTIC_RL_V2.md)。
> 数据 `bash agentic_rl_data/scripts/build_data_v2.sh`（配比 `configs/mixture_v2.yaml`，
> 自检的程序化 stateful 任务 `generators/stateful_tasks.py` + 本地语料多跳检索 `loaders/multihop_local.py`）；
> 训练 `STAGE=1|2|3 bash agentic_rl_data/scripts/run_qwen3_4b_agentic_v2.sh`。

## 目录与归属（Lead 维护，改动请走任务卡）

```
agentic_rl_data/
  schema.py                 # 契约：Task / Trajectory / reward 返回 / dedup key（Lead）
  README.md                 # 本文件（Lead）
  __init__.py               # 包标记（Lead）
  configs/mixture.yaml      # 配比、来源上限、随机种子（mixture-researcher）
  loaders/                  # 每来源一个 loader，输出统一 schema（data-researcher）
  scripts/                  # 流水线命令（见下）
  envs/                     # search / python / code_tests / mock_fc（env-builder）
  rewards/                  # 每 task_type 一个纯函数 reward（reward-writer）
  tests/                    # pytest（各席位写自己前缀的文件）
  reports/                  # 研究/配比/环境/审查报告 + TEAM_BOARD.md（Lead）
```

## 契约（必须先读 `schema.py`）

- **Task**：一行一条 JSONL；键固定为 `id / source / task_type / messages / tools / env / reference / meta`；
  `gt_calls` 按「轮」分组，并行调用同组、评测顺序无关；缺失值写 `null`，禁止静默丢样本。
- **Trajectory**：`{"turns": [...], "final_answer", "num_turns", "done", "env_state"}`；
  reward 只吃这个归一化结构，不做原始文本解析（解析/渲染属于框架侧）。
- **Reward**：`reward(task, trajectory) -> {"score": float, "info": dict}`，纯函数、无全局状态。
- **Loader**：`loaders/<source>.py` 暴露 `SOURCE: str` 与
  `load_tasks(cfg: dict | None = None) -> Iterator[dict]`，只产出通过 `schema.validate_task` 的 task。
- **Env**：`reset(task) -> obs`、`step(action) -> obs, done, info`；无全局状态、可并发。

## 流水线（对应 plan §3 八个步骤）

| 步骤 | 命令 | 负责 |
|---|---|---|
| 1 Load → 统一 schema | `python -m agentic_rl_data.scripts.build_raw` | data-researcher |
| 2 轮数过滤 | `scripts/filter_rounds.py` | data-researcher |
| 3 精确/近似去重 | `scripts/dedup.py` | data-researcher |
| 4 去污染（BFCL v3/v4 等） | `scripts/decontam.py` | data-researcher |
| 5 工具可执行性校验（mock_fc 重放） | env-builder + `scripts/replay_check.py` | env-builder |
| 6 通过率过滤 | `scripts/filter_by_passrate.py` | mixture-researcher |
| 7 混合 + held-out | `scripts/build_mixture.py` | mixture-researcher |
| 8 报告 | `scripts/report.py` → `reports/data_report.md`（`--strict` 同时校验 A2/A3/A4） | Lead（收口补写） |
| 9 导出 verl 格式 | `scripts/export_verl_parquet.py` → `data/verl/{train,heldout}.parquet` | Lead（verl 接入） |

> `report.py` 只读：数字要么从 `mixture_v1/{train,heldout}.jsonl` 重算，要么读上游 JSON 报告；
> 上游报告缺失的小节渲染为 `[NOT RUN]`（不编数字），`--strict` 下直接非 0 退出。

## 一键运行与启动脚本

| 目的 | 命令 |
|---|---|
| 建数据（全流程 1-8 步） | `.\agentic_rl_data\scripts\build_data.ps1`（`-DryRun` 先看计划；`-Strict` 验收门禁） |
| 同上（bash / Linux） | `bash agentic_rl_data/scripts/build_data.sh` |
| 起本地搜索服务（可先建 E5 索引） | `.\agentic_rl_data\scripts\serve_search.ps1 -BuildIndex -Docs data\wiki\e5_docs.jsonl` |
| 训练启动（verl，plan §9） | `bash agentic_rl_data/scripts/launch_verl_agentic.sh`（`SMOKE=1` 冒烟） |
| 导出 verl parquet | `python -m agentic_rl_data.scripts.export_verl_parquet --mixture-dir agentic_rl_data/data/mixture_v1` |
| 冒烟（测试 + pipeline dry-run） | `.\agentic_rl_data\scripts\smoke.ps1` |

- 统一入口：`python -m agentic_rl_data.scripts.run_pipeline --config agentic_rl_data/configs/pipeline.json`；
  `--from/--to/--skip` 选择步骤，`--dry-run` 不写文件，结束时写
  `<data>/_pipeline_manifest.json`（记录每步 ok/skipped/failed 与原因）。
  数据源路径写在 `build_raw.cfg_json`（格式见 [`configs/raw_sources.example.json`](configs/raw_sources.example.json)）；不填则走 loader 默认 HF id。
- 可选步骤（去污染 / 通过率）在配置缺失时**显式跳过**（`bfcl_path` / `rates` 为空），不会静默产出"最终数据集"；
  没有分开的 `[NOT RUN]` 会同步落到 `data_report.md`。
- 训练启动脚本是 **fail-closed 模板**：plan §9 的四个接口（prompt/label 格式、多轮 rollout、reward、chat template）
  尚未定义，脚本会先校验数据产物，再要求 `SLIME_LAUNCH_CMD`，不会猜框架参数。

## 环境约束（本会话重要）

- 本机 `pwsh` 不可用（进程启动即 `0xC0000142`，`cmd /c echo` 同样失败），
  **任何 Python / pytest / git 命令都跑不起来**。所有代码必须写成"可被信任的静态产物"：
  单测照写，但要在报告里标注 *未执行*，并把可执行的验证命令写清楚交给用户在有 shell 的环境运行。
- 不修改仓库既有代码：`verl/`、`examples/tool_rl/` 均只读参考。
  所有新代码在 `agentic_rl_data/` 之内。
- 只写自己任务卡里的 scope；跨 scope 的文件由 Lead 在任务卡上改派。

## 协作机制

- 任务卡：`team_task_*`；开工前 `get` 后 `claim`，完成后 `complete`。
- 信息共享：各自结论写进自己的 `reports/<scope>.md`；**跨席位**的发现、契约变更、
  阻塞项通过 `send_message` 发给 Lead，由 Lead 汇总进 `reports/TEAM_BOARD.md`
  （该文件 Lead 独占，其他人只读）。
- 依赖：`loaders` → `build_mixture`；`schema.py` 是所有人的上游契约，变更只能由 Lead 发起并广播。
