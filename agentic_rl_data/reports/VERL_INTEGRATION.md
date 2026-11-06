# verl 接入说明（plan §9 → verl）

> 目标框架是 **verl**（不是 slime）。本文件说明 `agentic_rl_data` 的数据 / reward / 多轮 rollout
> 如何映射到 verl 的接口，以及启动方式。所有代码静态交付，本机 `pwsh` 不可用（`0xC0000142`），
> 执行类验证需在有 shell 的机器上做。

## 1. 四个接口映射（替换 plan §9 的 TODO）

| plan §9 项 | verl 对应实现 | 文件 |
|---|---|---|
| prompt / label 数据格式 | RLHFDataset parquet：`data_source / prompt / tools / reward_model / extra_info` | `scripts/export_verl_parquet.py` → `data/verl/{train,heldout}.parquet` |
| 自定义多轮 rollout | `AgentLoopBase` 子类，注册名 **`agentic_agent`** | `agentic_rl_data/verl/agent_loop_agentic.py` + `agent_loop_config.yaml` |
| 自定义 reward 接口 | ① 多轮：loop 内直接算 `reward_score`（`_postprocess` 直接构造 `rm_scores`，reward manager 不覆盖）；② 单轮兜底：`compute_score` 自定义 reward | `scripts/verl_reward_bridge.py` |
| 工具调用模型原生格式 | `actor_rollout_ref.rollout.multi_turn.format`（默认 `hermes`；Qwen3 可用 `qwen3_coder`） | 启动脚本 `FORMAT` |

## 2. 数据契约（export 产物）

```text
data_source : "agentic_rl"
prompt      : [{role, content}]                      # task.messages（schema 只允许 system/user）
tools       : [{name, description, parameters}]      # parameters 是 JSON 字符串（pyarrow 统一类型需要）
reward_model: {style: "rule", ground_truth: str}     # GT 标签（仅日志 / 兼容）
extra_info  : {
  index, task_id, source, task_type,
  agentic_task        # 整条 unified task 的 JSON 字符串（loop 用它重建 env/reference）
  ground_truth_calls  # JSON 字符串（仅调试）
  gt_returns_source   # config/reference/gt_turns/placeholder（回放质量）
  max_turns, tool_selection
}
```

消费方注意：parquet 里 `tools[].parameters` 是 **JSON 字符串**；agent loop / chat template 用
`export_verl_parquet.tools_for_model()` 解析回 dict。

## 3. 多轮 loop（`agentic_agent`）

- 从 `kwargs["tools"]` 取每样本 schema（**不走全局 tool registry**，与 verl 内置 `ToolAgentLoop` 的关键差异）。
- 从 `kwargs["extra_info"]["agentic_task"]` 重建 task → `build_env(task, max_turns=meta.max_turns)`。
- 逐轮：`server_manager.generate` → `ToolParser.extract_tool_calls` → `env.step({content, tool_calls})`
  → 工具返回编码进 `prompt_ids`（`response_mask=0`）→ 继续生成；终止条件为 env done / `meta.max_turns` /
  `response_length`。
- done 时组 `trajectory`（CONTRACTS C3）→ `dispatch_reward` → `output.reward_score` + `extra_fields["reward_extra_info"]`。
- timeout：`env_state.timeout=True`，reward 压制 0.8 结果项（CONTRACTS C1）。
- reward/env 异常 → score 0 + `reward_error`，不崩 rollout。

## 4. 启动

```bash
# 1) 数据：mixture -> verl parquet
python -m agentic_rl_data.scripts.export_verl_parquet \
  --mixture-dir agentic_rl_data/data/mixture_v1 --out-dir agentic_rl_data/data/verl

# 2) 冒烟（1 epoch、小 batch）
SMOKE=1 bash agentic_rl_data/scripts/launch_verl_agentic.sh

# 3) 正式训练
MODEL_PATH=Qwen/Qwen3-4B NGPUS_PER_NODE=8 TOTAL_EPOCHS=15 \
  bash agentic_rl_data/scripts/launch_verl_agentic.sh
```

关键 override（脚本内已设）：`actor_rollout_ref.rollout.mode=async`、
`actor_rollout_ref.rollout.agent.default_agent_loop=agentic_agent`、
`agent_loop_config_path=agentic_rl_data/verl/agent_loop_config.yaml`、
`multi_turn.enable=True`、`multi_turn.format=hermes`、
`reward.custom_reward_function.path=scripts/verl_reward_bridge.py`。

## 5. 单轮 vs 多轮 reward 归属

| task_type | 谁算 reward | 说明 |
|---|---|---|
| `st_fc` / `irrelevance` / `abstain` | `verl_reward_bridge.compute_score` | 不需要 env_state；解析 `<tool_call>` 后 dispatch |
| `search_qa` / `math_ci` | 同上 | 只读 `trajectory.final_answer` |
| `mt_fc` / `code_exec` | **agent loop 内** | 需要 `env_state.goal_satisfied/replay_ok/code_tests`，loop 内 dispatch 并设 `reward_score` |

若 `mt_fc`/`code_exec` 落到 bridge（配置错误），bridge 返回 0 并标 `unsupported_single_turn`，不会静默给错分。

## 6. 已知限制（用户侧验证）

1. 未执行：`pytest`、`export_verl_parquet`、`launch_verl_agentic.sh` 都没跑过（本机无 shell）。
2. 多轮 loop 依赖 verl async 模式（`rollout.mode=async`）与工具解析器格式；Qwen3 建议 `FORMAT=qwen3_coder`（
   `hermes` 为默认，若不匹配输出格式会解析不到调用 → 全 0 分，需先看 rollout 日志）。
   `qwen3_coder/glm` 解析器需要 `OpenAIFunctionToolSchema` 实例（loop 已 best-effort 构造，失败降级 raw dict）。
3. `tools[].parameters` 是 JSON 字符串；loop 侧 `tools_for_model()`/`normalize_tools_for_template()` 负责解析。
4. env 侧 python/code_tests 的沙箱隔离、search 的 E5 索引与服务吞吐均未压测（见 `reports/env_setup.md`）。
5. `data.custom_cls` 未定制（默认 RLHFDataset）；若需要每 epoch 随机 hint（tool_rl 的机制），需扩展 dataset。
6. 多轮路径的 `multi_turn.max_user_turns` 与任务 `meta.max_turns` 双上限取小；课程 5→10→15 通过改 `MAX_TURNS`。
7. **st_fc 格式门已修正**（D20/C14）：env-none 的单轮预测不再要求 calls 与 tool_results 配对；否则正确调用会被判 0。
8. `reward_extra_info` 在 loop 内固定为 7 个 `agentic/*` 键 + 原 info 的 JSON 字符串：verl `_postprocess`
   用首样本键集合逐样本索引，混 task_type 的原生 info 会 KeyError 崩 worker。
9. 默认 `USE_V1=0`：V1/TQ trainer 的 `extra_fields` 序列化未验证；先用 V0 async 跑通。
10. `filter_overlong_prompts` 在启动脚本里关闭：per-sample tools 的长度不计入全局 filter，开启会被左截断。
