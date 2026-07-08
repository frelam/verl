# Online DPO — Agent 多轮工具调用直接偏好优化

从模型自身 rollout 的轨迹中选择 best/worst 样本对，用 DPO loss 直接更新策略，无需额外 reward model 或 value function。

## 🆕 批量 LLM Judge 打分（v2）

新增 **批量相对打分** 模式：将所有 rollout 轨迹一次性发送给 Judge 大模型（如 DeepSeek API 或本地 vLLM 部署），让 Judge 在同一批次内**比较轨迹质量**，给出相对分数。相比逐条独立打分，批量相对打分提供了更强的训练信号。

**支持的功能：**
- **批量相对打分**：Judge 同时看到所有轨迹，按相对质量排序评分
- **可配置 System Prompt**：自定义 Judge 关注哪些评估维度
- **多服务商支持**：DeepSeek API / OpenAI API / 本地 vLLM
- **多维打分（GDPO）**：返回 accuracy_reward, format_reward, efficiency_reward 等多个维度
- **滚动轮数控制**：`max_assistant_turns` / `max_user_turns` 控制 agent 最大交互轮数

**快速开始：**
```bash
# DPO 模式（单一分数，相对比较）
JUDGE_API_KEY=sk-xxx bash run_agent_dpo.sh toolmind 8 ~/ckpt/dpo

# GDPO 模式（多维打分，解耦归一化）
JUDGE_API_KEY=sk-xxx JUDGE_SYSTEM_PROMPT=gdpo \
  bash run_agent_dpo.sh toolmind 8 ~/ckpt/gdpo gdpo

# 自定义 System Prompt
JUDGE_API_KEY=sk-xxx \
  JUDGE_SYSTEM_PROMPT=./prompts/coding_judge.txt \
  bash run_agent_dpo.sh terminaltraj 4 ~/ckpt/dpo

# 本地 vLLM 部署
JUDGE_BASE_URL=http://localhost:8000/v1 \
  JUDGE_MODEL=Qwen3-32B \
  bash run_agent_dpo.sh toolmind 8 ~/ckpt/dpo
```

## 动机

Reject Sampling 流程存在以下问题：

1. **样本效率低**：Qwen3-4B 经过 Reasoning RL 后，多轮轨迹满足要求的概率很低，大部分 rollout 不合格
2. **速度慢**：需要收集大量样本后才能筛选 → 转 SFT → 训练，流程冗长
3. **SFT 遗忘风险**：SFT 容易遗忘之前的 Reasoning RL 能力

Online DPO 的优势：
- **DPO 只需相对好的信息**：不需要完美的正样本，best vs worst 即可提供有效训练信号
- **速度更快**：所需样本量少（不用采样一个 group），不用训练 value model
- **不需要精确 Reward**：用模型 judge 打分即可（容错率高）
- **在线训练**：每步采样 → judge → DPO 更新，无需离线收集数据

## 核心流程

```
数据集 prompt → Qwen3-4B 多轮 rollout (真实环境) → 每条轨迹 reward
                                                         ↓
                                    组内 best (最高分) vs worst (最低分) 配对
                                                         ↓
                                                  DPO loss 更新 actor
```

**关键设计**：
- 复用 `verl/main_ppo.py` + `ToolAgentLoop`（与 Reject Sampling 相同的工具环境）
- 设置 `algorithm.use_dpo=True` 启用 DPO 分支（跳过 advantage/critic）
- `policy_loss.loss_mode=dpo` 使用 DPO loss
- 复用 `reject_sampling` 的工具配置（`tool_config.yaml`）和 judge reward（`judge_reward.py`）
- `DPO_MODE=1` 环境变量告诉 judge 跳过轨迹保存，只返回 score

## 目录结构

```
Rl_Specilist/agent/online_dpo/
├── config/
│   └── online_dpo.yaml       # DPO 训练配置（Hydra config）
├── run_online_dpo.sh          # 训练启动脚本
└── README.md                  # 本文档
```

依赖的外部组件（复用 `reject_sampling`）：
- `Rl_Specilist/agent/reject_sampling/tools/tool_config.yaml` — 工具注册
- `Rl_Specilist/agent/reject_sampling/tools/swe_interaction_config.yaml` — SWE 交互配置
- `Rl_Specilist/agent/reject_sampling/reward/judge_reward.py` — DeepSeek judge
- `Rl_Specilist/agent/reject_sampling/setup/env.sh` — 环境变量

## 快速开始

### 前置条件

1. 已完成 Reject Sampling 的环境安装：
   ```bash
   bash Rl_Specilist/agent/reject_sampling/setup/install_env.sh
   ```

2. 已下载数据集并提取 prompt：
   ```bash
   bash Rl_Specilist/agent/reject_sampling/setup/download_datasets.sh
   ```

3. 设置 API key：
   ```bash
   export DEEPSEEK_API_KEY=sk-xxxxx
   export HF_TOKEN=hf_xxxxx
   ```

### 启动训练

```bash
bash Rl_Specilist/agent/online_dpo/run_online_dpo.sh <dataset> <nproc_per_node> <save_path> [extra_configs...]
```

**示例：**

```bash
# ToolMind 数据集，8 GPU，Qwen3-4B
bash Rl_Specilist/agent/online_dpo/run_online_dpo.sh toolmind 8 ~/data/online_dpo/ckpt

# TerminalTraj，4 GPU，自定义 beta 和 lr
bash Rl_Specilist/agent/online_dpo/run_online_dpo.sh terminaltraj 4 ~/data/online_dpo/ckpt \
    actor_rollout_ref.actor.policy_loss.dpo_beta=0.05 \
    actor_rollout_ref.actor.optim.lr=5e-7
```

### 支持的数据集

| 数据集 | 难度 | 说明 |
|--------|------|------|
| `toolmind` | L1 | 通用工具调用（calculator, search, code_runner, submit_answer） |
| `terminaltraj` | L3 | Docker 终端沙箱，bash 命令执行 |
| `open_swe_traces` | L4 | 代码仓库修复，需要 repo + 测试 sandbox |
| `swe_zero` | L2 | 代码修复轨迹 |

### 可调参数

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `MODEL_PATH` | `$HOME/models/Qwen3-4B` | 基座模型路径 |
| `DATA_DIR` | `$HOME/data/reject_sampling` | prompt 数据目录 |
| `N_SAMPLES` | `8` | 每个 prompt 采样数（best_vs_worst → 1 pair/prompt） |
| `TEMPERATURE` | `0.7` | 采样温度 |
| `DPO_BETA` | `0.1` | DPO β 参数（控制对 reference 的偏离程度） |
| `DPO_LR` | `1e-6` | DPO 学习率 |

## 配置要点

### 与 Reject Sampling 的关键区别

| 配置项 | Reject Sampling | Online DPO |
|--------|----------------|-----------|
| `hybrid_engine` | `False`（只做 rollout） | `True`（实际训练 actor） |
| `lr` | `0`（不更新） | `1e-6`（DPO 更新） |
| `policy_loss.loss_mode` | — | `dpo` |
| `algorithm.use_dpo` | — | `True` |
| `use_kl_loss` | — | `False`（DPO loss 已含 ref 项） |
| `critic.enable` | `False` | `False`（DPO 不需要 critic） |

### Best-vs-Worst 配对策略

- 对每个 prompt 采样 `n=8` 个响应
- 按 judge reward 排序，score 最高的作为 chosen，最低的作为 rejected
- 每个 prompt 产出 1 个 (chosen, rejected) pair
- `train_batch_size=32` → 每步 32 个 prompt，产出 32 个 pair

### Reward: DeepSeek API Judge

复用 `judge_reward.py` 的 `compute_score`，但在 `DPO_MODE=1` 下行为不同：
- 正常模式：打分 + 轨迹落盘 JSONL
- DPO 模式：仅打分，不保存轨迹（节省磁盘 I/O）

## Judge 配置说明

### 环境变量

Judge 行为通过环境变量控制（`run_agent_dpo.sh` 会自动设置）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `JUDGE_MODEL` | `deepseek-chat` | Judge 模型名 |
| `JUDGE_BASE_URL` | `https://api.deepseek.com` | API 地址（支持任意 OpenAI 兼容接口） |
| `JUDGE_API_KEY` | `$DEEPSEEK_API_KEY` | API 密钥 |
| `JUDGE_SYSTEM_PROMPT` | `default` | `default` / `gdpo` / 文件路径 / 内联 prompt |
| `JUDGE_SCORING_MODE` | `relative` | `relative`（批量比较）/ `absolute`（逐条独立） |
| `JUDGE_DIMENSIONS` | — | JSON 数组，如 `'["accuracy_reward","format_reward"]'` |
| `JUDGE_MAX_RETRIES` | `3` | API 调用最大重试次数 |
| `JUDGE_API_TIMEOUT` | `120` | API 超时（秒） |

### System Prompt 模板

预设模板位于 `prompts/` 目录：
- `prompts/coding_judge.txt` — 编程任务专用
- `prompts/math_judge.txt` — 数学推理任务专用
- 代码内的 `DEFAULT_SYSTEM_PROMPT` — 通用 agent 任务
- 代码内的 `GDPO_SYSTEM_PROMPT` — 多维打分（GDPO 模式）

### 滚动轮数控制

Agent 的最大交互轮数通过以下参数控制（在 YAML 或命令行中设置）：
- `actor_rollout_ref.rollout.multi_turn.max_assistant_turns`：模型最多生成多少轮回复（默认 10）
- `actor_rollout_ref.rollout.multi_turn.max_user_turns`：最多执行多少轮工具调用（默认 10）
- `actor_rollout_ref.rollout.multi_turn.max_tool_response_length`：工具返回内容的最大 token 数
- `actor_rollout_ref.rollout.response_length`：rollout 总长度限制

### 架构流程

```
Rollout (agent loop)
    │
    ├─ max_assistant_turns=10   ← 控制滚动轮数
    ├─ max_user_turns=10
    │
    ▼
收集所有轨迹 (messages, tool calls, ...)
    │
    ▼
批量 Judge (LLM API)
    │
    ├─ system_prompt + 所有轨迹 → Judge 模型
    ├─ 返回每个轨迹的相对分数
    ├─ (optional) 返回多维度分数
    │
    ▼
DPO/GDPO Loss 计算
    │
    ├─ DPO: best-vs-worst 配对
    ├─ GDPO: 每维度独立归一化 + 加权聚合
    │
    ▼
更新 Actor 权重
```

## 训练监控

训练日志中关注以下指标：

- `actor/loss` — DPO loss 值
- `actor/dpo_chosen_logp` — chosen 样本的平均 log prob
- `actor/dpo_rejected_logp` — rejected 样本的平均 log prob
- `reward/mean_score` — 平均 judge reward
- `reward/chosen_score` — chosen 样本的平均 score
- `reward/rejected_score` — rejected 样本的平均 score

## 与 Reject Sampling / Agent RL 的关系

```
Agent SFT (初始化)
    ↓
Reject Sampling (离线: rollout + judge → 筛选正确轨迹 → SFT)
    ↓                    ↓
Online DPO (在线: rollout + judge → best/worst → DPO 更新)
    ↓
Agentic RL (在线: rollout + 多维复合奖励 → GRPO 更新)
```

- **Reject Sampling**：适合初始阶段，快速积累高质量数据
- **Online DPO**：替代 Reject Sampling，更高效地从负样本中学习相对偏好
- **Agentic RL**：适合有明确多维奖励函数的场景，直接最大化复合 reward

## 注意事项

1. **API 速率限制**：`train_batch_size=32`（比 Reject Sampling 的 64 小）以避免 DeepSeek API 限流
2. **DeepSeek API Key 必须设置**：`DEEPSEEK_API_KEY` 环境变量
3. **Prompt 文件必须存在**：默认路径 `$DATA_DIR/prompts/<dataset>.parquet`
4. **Qwen3-4B 生成质量**：如果通过率太低，可提高 `N_SAMPLES` 或降低 `TEMPERATURE`
