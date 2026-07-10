# Online DPO — Uni-Agent Gateway + Verl 模型训练

## 架构

**Verl 训练的模型（Qwen3-4B）是 assistant。Uni-Agent Gateway 作为推理网关，捕获完整轨迹。**

```
┌─── Agent (hermes_entrypoint.py) — local workspace ─┐
│  接收 task → 调用 Gateway → 解析 tool_calls          │
│  执行 bash 工具 → 组装 observation → 循环             │
│  直到 submit_answer 或 max_turns                     │
└────────────────────────────────────────────────────┘
           │ Hermes-format tool calls
           │ (Gateway 侧解析 tool_parser: hermes)
           ▼
┌─── Uni-Agent Gateway (FastAPI) ─────────────────────┐
│  /v1/chat/completions                               │
│  工具调用解析 + 轨迹捕获                              │
│  每个 session 独立 base_url + reward_info_url        │
└────────────────────────────────────────────────────┘
           │ chat/completions
           ▼
┌─── vLLM (Qwen3-4B) ────────────────────────────────┐
│  推理引擎，每步 DPO 更新权重                          │
└────────────────────────────────────────────────────┘
           │
           ▼
    Runner 收集 reward → Judge 打分 (batch + inline)
           │
           ▼
   best vs worst → DPO loss → 更新模型
           │
           ▼
   下一轮 rollout — 模型权重已更新 ✨
```

**关键点：**
- **Agent entrypoint** = 黑盒 agent，通过 Gateway 与 verl 模型交互
- **Gateway** = 中间层，解析 Hermes 格式的 tool call，捕获完整 token 级轨迹
- **Runner** = `custom_hermes_runner.py`，管理 session → workspace → entrypoint → reward 全流程
- **Judge** = 双重模式：runner 侧 inline 打分（单轨迹）+ BatchRewardManager 批量打分
- **Online DPO**：模型生成 → 工具执行 → 打分 → 更新 → 下次 rollout 用新权重

---

## 快速开始

### 1. 安装

```bash
bash setup/install.sh
```

### 2. 配置环境变量

```bash
export DEEPSEEK_API_KEY=sk-xxx     # Judge 打分（必须）
export HF_TOKEN=hf_xxx              # 下载模型/数据
```

### 3. 下载数据

```bash
bash setup/download_data.sh
```

### 4. 启动训练（Gateway 模式）

```bash
# 使用 Uni-Agent Gateway + Hermes entrypoint
bash run_hermes_gateway_dpo.sh <dataset> 8 ~/ckpt/hermes-gateway
```

**旧模式（sandbox 工具，已弃用）：**
```bash
# sandbox 工具模式 — 保留作为参考
bash run_multi_agent_dpo.sh toolmind 8 ~/data/online_dpo/ckpt
```

---

## 配置文件

### Gateway 训练配置（推荐）

`config/agent_hermes_gateway.yaml` — Uni-Agent Gateway DPO Hydra config

```yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      enable: true
      format: hermes                     # Hermes tool call 格式
    agent:
      num_workers: 8
      agent_loop_manager_class: uni_agent.framework.entry.AgentFrameworkRolloutAdapter
    custom:
      agent_framework:
        agent_runners:
          custom_hermes:
            runner_fqn: Rl_Specilist.agent.online_dpo.custom_hermes_runner.custom_hermes_runner
  actor:
    policy_loss:
      loss_mode: dpo
      dpo_beta: 0.1
```

### Sandbox 工具配置（旧模式）

`config/tool_config_sandbox.yaml` — BaseTool 子类注册

```yaml
tools:
  - class_name: "....SandboxBashTool"
  - class_name: "....SandboxReadTool"
  - class_name: "....SandboxWriteTool"
  - class_name: "....SandboxSubmitTool"
```

---

## 核心模块

### 1. `hermes_entrypoint.py` — Agent 入口

独立 Python 脚本（仅依赖 stdlib），在工作区中运行工具调用循环：

- 通过环境变量 `HERMES_TASK` 接收任务
- 调用 Gateway 的 `/v1/chat/completions`（OpenAI 兼容 API）
- 解析 Hermes 格式的 tool call（`<tool_call>{"name": ..., "arguments": ...}</tool_call>`）
- 通过 `subprocess` 在工作区内执行 bash 命令
- 循环直到 `submit_answer` 或达 `max_turns` 限制

```bash
HERMES_TASK="do something" \
HERMES_BASE_URL="http://127.0.0.1:8765/sessions/abc/v1" \
HERMES_WORKSPACE="/tmp/verl_hermes/session-0-0" \
AGENT_MAX_TURNS=100 \
python hermes_entrypoint.py
```

### 2. `custom_hermes_runner.py` — Runner

Runner 契约实现，对接 Uni-Agent `AgentFramework`：

```
Runner (custom_hermes_runner)
  ├─ 从 raw_prompt + tools_kwargs 构建 task
  ├─ 创建隔离 workspace /tmp/verl_hermes/<session_id>
  ├─ 启动 hermes_entrypoint.py (subprocess)
  │     └─ Agent → Gateway → vLLM (Qwen3-4B)
  │     └─ Agent ← Gateway ← assistant reply
  │     └─ Agent → 在 workspace 执行工具 → observation
  │     └─ ... 循环 ...
  ├─ 评估 reward (LLM Judge inline 或 basic)
  ├─ POST reward_info → Gateway
  └─ 清理 workspace
```

### 3. `reward/llm_judge.py` — Judge 打分

双重接口：

| 接口 | 用途 | 调用方 |
|------|------|--------|
| `judge_single(task, agent_output)` | 单轨迹 inline 打分 | `custom_hermes_runner` |
| `compute_score(data_sources, ...)` | 批量打分（BatchRewardManager） | verl reward loop |

配置：
```bash
export JUDGE_MODEL=deepseek-chat        # Judge 模型
export JUDGE_BASE_URL=https://api.deepseek.com
export DEEPSEEK_API_KEY=sk-xxx          # API Key
```

---

## 环境变量参考

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `$HOME/models/Qwen3-4B` | 训练模型路径 |
| `TRAIN_DATA` | `$HOME/data/online_dpo/prompts/<dataset>.parquet` | 训练数据 |
| `N_SAMPLES` | `4` | 每个 prompt 的 rollout 数 |
| `AGENT_MAX_TURNS` | `100` | Agent 最大对话轮数 |
| `AGENT_TIMEOUT` | `3600` | 单个 agent 运行超时（秒） |
| `HERMES_WORKSPACE_ROOT` | `/tmp/verl_hermes` | Workspace 根目录 |
| `GATEWAY_COUNT` | `1` | Gateway 实例数 |
| `MAX_CONCURRENT_SESSIONS` | `32` | 最大并发 session 数 |
| `NUM_AGENT_WORKERS` | `8` | Agent worker 数量 |
| `JUDGE_MODEL` | `deepseek-chat` | Judge 模型 |
| `JUDGE_BASE_URL` | `https://api.deepseek.com` | Judge API 地址 |
| `DEEPSEEK_API_KEY` | — | Judge API Key（必须） |

---

## 训练监控

```bash
# WandB dashboard
# 关键指标：
#   actor/loss                — DPO loss ↓
#   actor/dpo_chosen_logp     — chosen 样本 log-prob ↑
#   actor/dpo_rejected_logp   — rejected 样本 log-prob ↓
#   reward/mean_score         — 平均 judge 评分 ↑
#   agent_loop/generate_sequences/mean — rollout 平均耗时
```

---

## 目录结构

```
Rl_Specilist/agent/online_dpo/
├── config/
│   ├── agent_hermes_gateway.yaml     # Gateway DPO + Hermes 配置 ★
│   ├── agent_dpo_judge.yaml          # Sandbox DPO + batch judge（旧）
│   ├── agent_gdpo_judge.yaml         # Sandbox GDPO + batch judge（旧）
│   ├── online_dpo.yaml               # 原始 online DPO 配置（旧）
│   └── tool_config_sandbox.yaml      # Sandbox 工具注册（旧）
├── hermes_entrypoint.py              # Agent 入口（stdlib only）★
├── custom_hermes_runner.py           # Runner — session/workspace/reward ★
├── reward/
│   ├── __init__.py
│   └── llm_judge.py                  # Judge 打分（batch + inline）★
├── tests/
│   ├── __init__.py
│   └── test_hermes_entrypoint.py     # Hermes entrypoint 单元测试 ★
├── tools/
│   ├── __init__.py
│   └── sandbox_tools.py              # BaseTool 子类（旧模式）
├── runners/
│   ├── __init__.py                   # 已弃用 — 旧 agent loop runner
│   └── test_smoke.py                 # 冒烟测试
├── setup/
│   ├── install.sh                    # 一键安装
│   └── download_data.sh             # 数据下载
├── prompts/                          # Judge prompt 模板
│   ├── coding_judge.txt
│   └── math_judge.txt
├── run_hermes_gateway_dpo.sh         # Gateway 训练启动脚本 ★
├── run_multi_agent_dpo.sh            # Sandbox 模式训练（旧）
├── run_agent_dpo.sh                  # Agent DPO（旧）
├── run_online_dpo.sh                 # 原始 online DPO（旧）
└── README.md
```

★ = 新架构核心文件

---

## 新旧架构对比

| 维度 | 旧架构（Sandbox） | 新架构（Uni-Agent Gateway） |
|------|-------------------|---------------------------|
| Agent 入口 | verl `ToolAgentLoop` | `hermes_entrypoint.py`（独立脚本） |
| 推理网关 | vLLM 直接调用 | Gateway → vLLM |
| 工具执行 | `BaseTool` 子类（`tools/sandbox_tools.py`） | `subprocess` 直接执行 |
| 工具注册 | `tool_config_sandbox.yaml` | 不需要（agent 自行管理） |
| 轨迹捕获 | verl 内部 | Gateway 捕获完成 token 级轨迹 |
| Runner | `ToolAgentLoop` | `custom_hermes_runner.py`（AgentFramework） |
| Judge | 仅 batch（`BatchRewardManager`） | batch + inline 双模式 |
| 配置 | `agent_dpo_judge.yaml` | `agent_hermes_gateway.yaml` |
| 启动 | `run_multi_agent_dpo.sh` | `run_hermes_gateway_dpo.sh` |

## 注意事项

1. **Gateway 必须先启动**：训练启动前确保 Uni-Agent Gateway 已运行
2. **工具不可逃逸**：workspace 隔离，每个 session 独立目录，trajectory 结束后自动清理
3. **Judge 限流**：DeepSeek API 有速率限制，注意 `train_batch_size` 设置
4. **磁盘清理**：workspace 每 trajectory 自动清理，残留目录可通过 `rm -rf /tmp/verl_hermes/*` 手动清理
5. **Agent timeout**：`AGENT_TIMEOUT` 超时后 agent 终止，reward 标记为失败
6. **Gateway proxy**：Runner 自动 unset `HTTP_PROXY`/`HTTPS_PROXY`，避免干扰本地 Gateway 通信
