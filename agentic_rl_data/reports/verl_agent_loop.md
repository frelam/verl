# verl 多轮 agent loop 适配（`agentic_agent`）

> 席位：verl-integrator（T6）· 状态：**代码已交付，本机全部未执行**（`pwsh` 0xC0000142，无 Python/pytest）
> 平台：verl `actor_rollout_ref.rollout.mode=async` 的 AgentLoop（**不是 slime**）
> 相关：`agentic_rl_data/verl/agent_loop_agentic.py`、`agentic_rl_data/verl/trajectory.py`、`agentic_rl_data/verl/agent_loop_config.yaml`、
> `agentic_rl_data/tests/test_verl_agent_loop.py`、`agentic_rl_data/scripts/export_verl_parquet.py`

## 1. 交付物

| 文件 | 作用 |
|---|---|
| `agentic_rl_data/verl/agent_loop_agentic.py` | `@register("agentic_agent")` 的 `AgenticAgentLoop(AgentLoopBase)`：env 驱动的多轮 rollout + loop 内 reward |
| `agentic_rl_data/verl/trajectory.py` | 纯函数层（stdlib only）：tools 归一化、task 解析、tool_calls/tool_results 转换、trajectory 组装、fail-closed reward、固定键 `reward_extra_info` |
| `agentic_rl_data/verl/agent_loop_config.yaml` | 注册配置（`name: agentic_agent` / `_target_: ...AgenticAgentLoop`） |
| `agentic_rl_data/verl/__init__.py` | 轻量入口：`AgenticAgentLoop` 惰性导入（导入本包不需要 verl/torch） |
| `agentic_rl_data/tests/test_verl_agent_loop.py` | 纯函数测试 + `pytest.importorskip("verl")` 的注册测试 |

`verl/`、`examples/` 既有文件一律只读参考，未改动。

## 2. 设计

```
run(sampling_params, priority=0, **kwargs):
  raw_prompt / tools / extra_info
      │
      ├─ tools 列(导出为 {name, description, parameters(JSON str)}) → 包装成 OpenAI schema
      ├─ extra_info["agentic_task"] (JSON 字符串) → 重建 unified task
      │
      ├─ [task.env.type != "none"] build_env(task, max_turns=meta.max_turns) → reset(task, max_turns=...)
      │       └─ 失败：记录 agentic/env_error，降级为单轮
      │
      ├─ loop（最多 meta.max_turns 轮 / response_length token）:
      │     generate(request_id, prompt_ids, sampling_params, priority)
      │       → prompt_ids += token_ids；response_mask += [1]*n；logprobs += ...
      │       → ToolParser.extract_tool_calls(token_ids, per-sample schemas)
      │       → env.step({content, tool_calls})  ← run_in_executor（search HTTP 不阻塞事件循环）
      │       → obs["tool_results"] → role=tool 消息
      │       → apply_chat_template(tool_msgs, remove_system_prompt=True) + turn_separator
      │         prompt_ids += tokens；response_mask += [0]*n；logprobs += 0.0
      │       → done / token 预算 / env 异常 → 结束
      │
      ├─ 组装 trajectory {turns, final_answer, num_turns, done, env_state=info["env_state"]}
      ├─ dispatch_reward(task, trajectory)（run_in_executor + fail-closed）
      └─ AgentLoopOutput(reward_score=score, extra_fields={reward_extra_info, env_state, ...})
```

关键语义：

- **reward 在 loop 内算**：`AgentLoopOutput.reward_score` 非空时，`AgentLoopWorker._postprocess` 直接构造
  `rm_scores`（`verl/experimental/agent_loop/agent_loop.py:1086-1092`），随后 `NaiveRewardManager.__call__`
  走 `_extract_reward_from_rm_scores` 短路（`verl/workers/reward_manager/naive.py:90-96` +
  `abstract.py:47-72`），**不会覆盖** loop 内分数。这一点与任务卡一致。
- `extra_fields["reward_extra_info"]` 会写入 `non_tensor_batch`（`agent_loop.py:1100-1104`）并经
  `meta_info["reward_extra_keys"]` 原样回传给 reward manager，用于训练/验证指标。
- **token 累积照抄 `ToolAgentLoop` 范式**：assistant token `response_mask=1`；工具返回 token
  `response_mask=0`、`response_logprobs=0.0`、前缀 `self.turn_separator`、`remove_system_prompt=True`。
- **env 同步阻塞点全部包 `await self.loop.run_in_executor(None, ...)`**：`build_env`、`reset`、`step`（search/python/code_tests 可能 HTTP/子进程）、`close`、reward dispatch。
- **timeout**：env 超限时 `done=True, info["timeout"]=True`，`env_state.timeout=True` 原样进 trajectory（reward 侧压制结果项，不额外扣分）。
- **done / env_state 语义**：只有 env 真的 done（或 task 本身无 env 的单轮样本）才 `done=True`；
  token 预算截断 / env 报错时 `done=False, env_state=None`，交给 `mt_fc`/`code_exec` 的 `missing_env_state` 分支 fail-closed。
- **单轮兜底**：`extra_info` 无 `agentic_task` → 生成一次即结束、`reward_score=None`（交给 verl 配置的 reward manager，绝不瞎造 task）；`task.env.type=="none"`（st_fc / irrelevance / abstain 等）→ 生成一次、仍 dispatch reward，`env_state=None`（这几个 reward 不读 env_state）。

## 3. 数据列契约（rollout 时可见）

| 列 | 类型 | loop 里的用途 |
|---|---|---|
| `prompt` / `raw_prompt` | `list[{role, content}]` | `apply_chat_template` 初始 prompt（`raw_prompt` 由 `RLHFDataset.__getitem__` 合成） |
| `tools` | `list[{name, description, parameters(JSON string)}]`（见 `export_verl_parquet.normalize_tool_list`） | 每样本工具；locals wrap 成 OpenAI `{type,function{name,description,parameters}}` 后用于 chat template 与 `ToolParser`；**不走全局 tool registry** |
| `reward_model` | `{style: "rule", ground_truth: str}` | 仅日志/兜底 reward 用，loop 内 reward 不读 |
| `extra_info.agentic_task` | unified task 的 JSON 字符串 | 重建 task / env / reference，**reward 的唯一 ground truth 来源** |
| `extra_info.task_id / task_type / source / max_turns / gt_returns_source / tool_selection` | 标量 | 诊断字段（`task_id/task_type` 回填 `extra_fields`） |
| `data_source` | `"agentic_rl"` | 不参与 loop 内 reward |

trajectory 结构（冻结，见 `schema.py` / `CONTRACTS.md` C3/C4）：

```json
{"turns": [{"step": 1, "assistant": {"content": "", "tool_calls": [{"id", "name", "arguments": {}}]},
           "tool_results": [{"id", "name", "content", "ok"}]}],
 "final_answer": str|null, "num_turns": int, "done": bool, "env_state": {...}|null}
```

## 4. config 键（launch 时覆盖；我未执行任何训练）

```bash
actor_rollout_ref.rollout.mode=async
actor_rollout_ref.rollout.agent.default_agent_loop=agentic_agent
actor_rollout_ref.rollout.agent.agent_loop_config_path=agentic_rl_data/verl/agent_loop_config.yaml
actor_rollout_ref.rollout.multi_turn.format=hermes      # 必须与所服务模型的 tool-call 格式一致（deepseek_v4/qwen3_coder/glm/kimi 等见 tool_parser.py）
actor_rollout_ref.rollout.agent.num_workers=<N>         # 并发 rollout worker 数（env.step 在 worker 线程池执行）
actor_rollout_ref.rollout.prompt_length=<P>
actor_rollout_ref.rollout.response_length=<R>           # 多轮 token 预算（含工具返回 token）
actor_rollout_ref.data.filter_overlong_prompts=False    # 重要：按 per-sample tools 重新计长的过滤尚未实现
reward.reward_manager.name=naive                        # loop 内 reward 时会被 rm_scores 短路；仅 taskless 行回退
```

> 数据集导出：`python -m agentic_rl_data.scripts.export_verl_parquet --mixture-dir ... --out-dir agentic_rl_data/data/verl`
> 该脚本产出 `train.parquet/heldout.parquet`（含 `agentic_task` 列），本 loop 已按其 `tools` 存储形态适配。

## 5. 与 verl `ToolAgentLoop` 的差异

| 维度 | `ToolAgentLoop` | `AgenticAgentLoop` |
|---|---|---|
| 工具来源 | 全局 registry（`load_all_tools` + `ToolListWrap`） | **每样本** `kwargs["tools"]`（`extra_info.tool_selection` 不再需要） |
| 工具执行 | `BaseTool`/`FunctionTool` 生命周期 | 任务的 `env.step(action)`（mock_fc/search/python/code_tests 统一协议） |
| 轮数控制 | `multi_turn.max_assistant_turns/max_user_turns` | `task.meta.max_turns`（env 自身也按同名上限超时）+ 自身 guard |
| reward | 无（交给 reward manager） | **loop 内 dispatch_reward**，`reward_score` 直进 `rm_scores` |
| 工具返回消息 | 结构化 `ToolResponse`/多模态 | `obs["tool_results"]` → `role=tool` 纯文本（`id`→`tool_call_id`） |
| `apply_chat_template` 参数 | 全局 schemas | 每样本 schemas（`tools=sample_tools or None`） |
| `response_mask` 语义 | 1=模型 token，0=工具返回 token | 相同 |
| fail-closed | tool 异常 → `ToolResponse` 文本 | env 构造/step/reward 异常 → 降级 + `reward_error`，绝不崩 rollout |

## 6. 与任务卡/Lead 假设的偏差（必须说明）

1. **`extra_fields["reward_extra_info"]` 不是裸 `result["info"]`，而是固定键包装**。
   原因（源码事实）：`AgentLoopWorker._postprocess` 用 **第一个样本** 的 info key 集合逐样本索引
   （`agent_loop.py:1101-1104`），混合 `task_type`（本数据集 7 类）时后面的样本缺 key 会直接
   `KeyError` 崩掉整个 worker。因此固定 7 键：
   `agentic/task_type`、`agentic/reward_score`、`agentic/reward_error`、`agentic/reward_timeout`、
   `agentic/num_turns`、`agentic/done`、`agentic/reward_info`（后者是原 info 的 JSON 字符串，截断 4000 字符）。
   → **若 Lead 想按 task_type 聚合 reward 子项指标，建议在数据侧（`data_source`/`extra_info.task_type`）聚合；原始 reward info 在 `agentic/reward_info` 里可解析。**
2. **无 `agentic_task` 的行不做 reward dispatch**（`reward_score=None`）：没有 task_type/reference 时无法合规打分，
   交给 verl 配置的 `/agentic_rl_data/scripts/verl_reward_bridge.py` 或默认 `compute_score` 走单轮路径（不臆造 task/reference）。
   副作用：**同一 batch 混有 taskless 行时，`all(score is not None)` 为假 → 整批回退 reward manager，loop 内分数被覆盖**。
   请保证训练数据全部来自 `export_verl_parquet`（每行都有 `agentic_task`）。
3. **`env.type=="none"` 的样本不走 NullEnv**，直接"生成一次即结束 + 仍 dispatch reward"；这几个 task_type 的 reward 不依赖 env_state。
4. 导出脚本的 `tools.parameters` 是 **JSON 字符串**（与任务卡"OpenAI 风格 dict"不完全一致）：loop 内已做
   `{name,description,parameters(str)}` → OpenAI 包装 + `json.loads`；`apply_chat_template` 收到的是标准 OpenAI schema。

## 7. 已知限制 / 风险 / 待联调

- **未执行**：本机无 shell/Python，任何 pytest / 训练 / 导出都未运行；测试与报告均为静态审查产物。
- **混合 task_type 的 `reward_extra_info` 键一致性**已用固定键规避（见 §6.1）。
- **`response_logprobs` 对齐**：若 `calculate_log_probs=False` 且某轮有 logprobs，与 `ToolAgentLoop` 同样的
  对齐假设（通常全开或全关）；联调时请确认 `actor_rollout_ref.rollout.calculate_log_probs=True`。
- **`routed_experts`（MoE router replay）**：按 ToolAgentLoop 的方式只保留最后一轮并截断；非 MoE 模型不影响。
- **per-sample tools 的长度过滤**：`RLHFDataset` 的 `filter_overlong_prompts` 用的是全局 `tool_schemas`，
  不含每样本工具，可能让真实 prompt 超 `prompt_length`；此时 `apply_chat_template` 会左截断（可能切掉 tools 声明）。
  联调请先关掉该过滤或在数据侧按带 tools 的模板重算长度。
- **env 线程安全**：`env.step/reset` 在默认 ThreadPoolExecutor 里执行；依据 `envs/base.py` 的"每 task 一实例、
  reset 全量重建"契约可并发（不同实例），但 **search 后端（HTTP/e5/faiss）的并发上限与单任务 `task_deadline_s`
  需在联调时压测**，避免拖慢 agent loop worker。
- **`build_env` + `reset` 各做一次初始化**（与 `scripts/replay_check.py` 同模式）；search 索引若初始化很重，可考虑后续只在
  `build_env` 时初始化（需 env-builder 确认 `reset` 是廉价重建）。
- **env 超时/异常降级**：env 构造失败 → 单轮 + `done=False, env_state=None`；step 异常 → 立即结束；两者 reward 仍 fail-closed，不崩 rollout。
- **v1/TQ trainer 路径**（`verl/trainer/ppo/v1/agent_loop_tq.py` + TransferQueue）未针对本 loop 验证；
  `extra_fields` 里带 `env_state` dict 的序列化只在 V0 async `_postprocess` 路径上验证过（本适配默认走 V0）。
- **`reward_score` 语义核对**：已核对 `_postprocess` 用 `attention_mask` 找到最后一个有效 response token 写 `rm_scores`
  （`agent_loop.py:1089-1091`），不是 `as_dict()` 的 `[-1]`；空 response（0 token）时该行为边界情况，见 §7 未执行项。
- **`num_turns`**：输出 `len(turns)+1`（与 `ToolAgentLoop` 的 user+assistant+1 口径一���）。
- **指标命名带 `/`**：`agentic/*` 键会进 non_tensor_batch；wandb/验证指标聚合时请确认后端接受该命名（字符串列会被
  `process_validation_metrics` 跳过，数值列正常求均值）。

## 8. 用户侧验证命令（我未执行）

```bash
cd D:\lhy\verl\verl-toolRL

# 1) 纯函数 +（有 verl 时）注册测试
python -m pytest agentic_rl_data/tests/test_verl_agent_loop.py -q
python -m pytest agentic_rl_data/tests/test_verl_agent_loop.py -q -k "not registered"

# 2) 导出 parquet（每行必须带 extra_info.agentic_task）
python -m agentic_rl_data.scripts.export_verl_parquet \
  --mixture-dir agentic_rl_data/data/mixture_v1 --out-dir agentic_rl_data/data/verl

# 3) 单步 smoke（1 卡、小样本、关掉长度过滤；按项目实际 launch 脚本改）
python -m verl.trainer.main_ppo \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.agent.default_agent_loop=agentic_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=agentic_rl_data/verl/agent_loop_config.yaml \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.data.filter_overlong_prompts=False \
  data.train_files=agentic_rl_data/data/verl/train.parquet \
  trainer.total_epochs=0 trainer.total_training_steps=1 \
  trainer.n_gpus_per_node=1 actor_rollout_ref.rollout.n=1

# 4) 首个 smoke 通过后检查：
#    - 训练日志 agent_loop/generate_sequences、agent_loop/tool_calls 非 0（多轮确实发生）
#    - 验证集指标里 rm_scores 与 agentic/reward_score 一致（loop 内 reward 生效）
#    - 抽样 dump 的 reward_extra_info 中 agentic/reward_error 为空串（无 fail-closed 行）
#    - task_type 分布与 extra_info.task_type 一致（env 未错配）
```
