# Online DPO — Sandbox 工具 + Verl 模型训练

## 架构

**Verl 训练的模型（Qwen3-4B）是 assistant。Sandbox 工具是执行环境。**

```
┌─── verl model (Qwen3-4B) — ASSISTANT ──┐
│  每步 DPO 更新权重                       │
│  生成: "I need to search..."            │
│  生成: tool_call: bash("ls")            │
└────────────────────────────────────────┘
           │ tool_call
           ▼
┌─── Sandbox Tools — 执行环境 ────────────┐
│  bash()       → 在隔离 workspace 执行   │
│  read_file()  → 读取文件内容            │
│  write_file() → 写入文件                │
│  submit_answer() → 提交最终答案         │
└────────────────────────────────────────┘
           │ observation (stdout, file contents)
           ▼
┌─── verl model 继续生成 ────────────────┐
│  处理 observation, 决定下一步           │
│  直到 submit_answer 或 max_turns        │
└────────────────────────────────────────┘
           │
           ▼
   完整 trajectory → Judge 打分 → best vs worst → DPO loss → 更新模型
           │
           ▼
   下一轮 rollout — 模型权重已更新 ✨
```

**关键点：**
- **Assistant = verl 训练的模型**（Qwen3-4B），它决定何时调用工具、调用什么工具
- **Tool execution = sandbox**，提供隔离的 bash + 文件读写环境
- 每步 DPO 更新后，模型权重变化，下一轮 rollout 使用最新模型
- 这是真正的 Online DPO：模型生成 → 工具执行 → 打分 → 更新 → 生成更好

---

## 快速开始

### 1. 安装

```bash
bash setup/install.sh
```

### 2. 配置

```bash
export DEEPSEEK_API_KEY=sk-xxx     # Judge 打分
export HF_TOKEN=hf_xxx              # 下载模型/数据
```

### 3. 下载数据

```bash
bash setup/download_data.sh
```

### 4. 启动训练

```bash
# 使用 sandbox 工具（bash/read_file/write_file/submit_answer）
bash run_multi_agent_dpo.sh toolmind 8 ~/data/online_dpo/ckpt
```

---

## 配置文件

### 训练配置

`config/agent_dpo_judge.yaml` — DPO + batch judge Hydra config

关键参数：
```yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      enable: true
      format: hermes                    # 模型 tool call 格式
      max_assistant_turns: 15           # 模型最多生成轮数
    n: 4                                # 每个 prompt 采样数
  actor:
    policy_loss:
      loss_mode: dpo
      dpo_beta: 0.1
algorithm:
  use_dpo: true
reward:
  use_batch_judge: true                 # 批量 Judge 打分
```

### 工具配置

`config/tool_config_sandbox.yaml` — 注册 sandbox 工具

```yaml
tools:
  - class_name: "....SandboxBashTool"      # bash 命令
  - class_name: "....SandboxReadTool"      # 读文件
  - class_name: "....SandboxWriteTool"     # 写文件
  - class_name: "....SandboxSubmitTool"    # 提交答案
```

启动时通过 `--tool-config` 指定：
```bash
bash run_multi_agent_dpo.sh toolmind 8 ~/ckpt \
    --tool-config config/tool_config_sandbox.yaml
```

---

## 工具说明

| 工具 | 参数 | 说明 |
|------|------|------|
| `bash` | `command: str` | 在隔离 workspace 执行 bash 命令 |
| `read_file` | `file_path: str` | 读取 workspace 内的文件 |
| `write_file` | `file_path: str, content: str` | 写入文件到 workspace |
| `submit_answer` | `answer: str` | 提交最终答案，结束轨迹 |

**Workspace 隔离：**
- 每个 trajectory 一个独立 workspace（`/tmp/verl_sandbox/<request_id>`）
- 同一 trajectory 内所有工具共享 workspace
- `read_file` / `write_file` 拒绝访问 workspace 外的路径
- Trajectory 结束后自动清理

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
│   ├── agent_dpo_judge.yaml        # DPO + batch judge
│   ├── tool_config_sandbox.yaml    # sandbox 工具注册
│   └── online_dpo.yaml             # 原始配置
├── tools/
│   └── sandbox_tools.py            # BaseTool 子类实现
├── setup/
│   ├── install.sh                  # 一键安装
│   └── download_data.sh           # 数据下载
├── prompts/                        # Judge prompt 模板
├── run_multi_agent_dpo.sh          # 训练启动脚本
├── run_agent_dpo.sh                # (旧) agent DPO
├── run_online_dpo.sh               # (旧) 原始 online DPO
└── README.md
```

---

## 添加新工具环境

继承 `BaseTool`，注册到 `tool_config.yaml`：

```python
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import ToolResponse

class MyEnvTool(BaseTool):
    async def execute(self, instance_id, parameters, **kwargs):
        agent_data = kwargs.get("agent_data")
        # 在隔离环境执行
        result = await my_runner.run(parameters)
        return ToolResponse(text=result), 0.0, {}
```

在 `tool_config.yaml` 中注册，ToolAgentLoop 自动发现并使用。

## 注意事项

1. **模型参与 rollout**：这是 online DPO，verl 训练的 Qwen3-4B 充当 assistant。每一轮 DPO 更新后，下一轮 rollout 使用最新权重
2. **工具不可逃逸**：`read_file` / `write_file` 限制在 workspace 内
3. **Judge 限流**：DeepSeek API 有速率限制，`train_batch_size=32`
4. **磁盘清理**：sandbox workspace 每 trajectory 自动清理
