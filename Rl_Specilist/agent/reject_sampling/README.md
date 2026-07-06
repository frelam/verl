# Reject Sampling SFT 使用指南

用 Qwen3-4B 在真实环境多轮 rollout 生成轨迹，DeepSeek-V4-Pro API 筛选正确轨迹，用于 SFT 训练。

## 目录结构

所有代码在 `Rl_Specilist/agent/reject_sampling/` 下：

```
Rl_Specilist/agent/reject_sampling/
├── setup/                              # 一键环境安装
│   ├── install_env.sh                  # 系统依赖 + conda + verl + vLLM + Docker 镜像
│   ├── download_datasets.sh            # 4 个数据集 + Qwen3-4B 模型 + prompt 提取
│   ├── verify_env.sh                   # 环境自检
│   ├── env.sh                          # 环境变量配置
│   └── docker/
│       ├── Dockerfile.terminal         # TerminalTraj 终端沙箱镜像
│       └── Dockerfile.swe_bench        # SWE-bench 评测镜像
├── data_preprocess/                    # 数据处理
│   ├── extract_prompts.py              # 从 4 个数据集提取 prompt + tools + ground_truth
│   └── convert_to_sft.py               # JSONL → 筛选 top-K → SFT parquet
├── tools/                              # 真实环境工具
│   ├── generic_tool_executor.py        # ToolMind 通用工具执行器
│   ├── terminal_sandbox_tool.py        # TerminalTraj Docker 终端沙箱
│   ├── swe_bench_interaction.py        # SWE-bench 代码仓库交互环境
│   ├── tool_config.yaml                # 工具注册配置
│   └── swe_interaction_config.yaml     # SWE interaction 配置
├── reward/                             # DeepSeek judge
│   ├── judge_reward.py                 # compute_score: rule-based 优先 → DeepSeek API → 轨迹落盘
│   └── trajectory_collector.py         # JSONL 读写 + prompt 去重 + 统计
├── config/
│   └── reject_sampling.yaml            # rollout 配置 (lr=0, n=8, temperature=0.7)
├── run_reject_sampling.sh              # rollout 启动脚本
└── run_sft_from_reject.sh              # SFT 训练脚本
```

## 数据集与难度分级

| 难度 | 数据集 | HF Repo | 验证方式 | 环境需求 |
|---|---|---|---|---|
| L1 | ToolMind | `Nanbeige/ToolMind` | 工具调用成功 + DeepSeek judge | 通用 tool executor |
| L2 | SWE-Zero | `nvidia/SWE-Zero-openhands-trajectories` | 测试通过或 DeepSeek judge | repo + 测试 sandbox |
| L3 | TerminalTraj | `m-a-p/TerminalTraj` | 命令成功率 + DeepSeek judge | Docker terminal |
| L4 | Open-SWE-Traces | `nvidia/Open-SWE-Traces` | 测试通过 | OpenHands + repo + 测试 sandbox |

OpenResearcher（L5）暂不支持。

## 核心流程

```
数据集 prompt → Qwen3-4B 多轮 rollout (真实环境) → DeepSeek judge 打分
                                                          ↓
                                              score >= 0.7 的轨迹落盘 JSONL
                                                          ↓
                                              筛选 top-K → SFT parquet → 训练
```

**关键设计**：复用 verl 的 `main_ppo.py` + `ToolAgentLoop` 做 rollout（设 `lr=0` 只生成不训练），在 `custom_reward_function` 里调 DeepSeek API 打分并落盘轨迹。

## 新机器使用流程

### 1. 一键安装环境

```bash
bash Rl_Specilist/agent/reject_sampling/setup/install_env.sh
```

可覆盖的环境变量：
- `ENV_NAME`：conda 环境名（默认 `reject_sft`）
- `VERL_DIR`：verl 安装路径（默认 `$HOME/workspace/verl`）
- `CUDA_VERSION`：CUDA 版本（默认 `12.1`）

脚本会自动完成：
- 安装系统依赖（docker, git, build-essential）
- 创建 conda 环境（Python 3.10）
- 安装 PyTorch + verl + vLLM + flash-attn
- 构建 agent 环境 Docker 镜像
- 配置环境变量到 `~/.bashrc`
- 运行自检

### 2. 设置 API key

```bash
export DEEPSEEK_API_KEY=sk-xxxxx
export HF_TOKEN=hf_xxxxx
```

### 3. 下载数据集 + 模型 + 提取 prompt

```bash
bash Rl_Specilist/agent/reject_sampling/setup/download_datasets.sh
```

默认每个数据集取 500 条样本，可通过 `MAX_SAMPLES=1000` 调整。

### 4. 跑 reject sampling

```bash
# 先跑 ToolMind（环境最轻，验证全流程）
bash Rl_Specilist/agent/reject_sampling/run_reject_sampling.sh toolmind 8 ~/data/reject_sampling/ckpt

# 然后 TerminalTraj
bash Rl_Specilist/agent/reject_sampling/run_reject_sampling.sh terminaltraj 8 ~/data/reject_sampling/ckpt

# 最后 SWE 数据集（环境最重）
bash Rl_Specilist/agent/reject_sampling/run_reject_sampling.sh open_swe_traces 8 ~/data/reject_sampling/ckpt
bash Rl_Specilist/agent/reject_sampling/run_sft_from_reject.sh swe_zero 8 ~/data/reject_sampling/ckpt
```

可覆盖的参数：
- `N_SAMPLES=16`：每 prompt 采样数（默认 8）
- `TEMPERATURE=0.5`：采样温度（默认 0.7）
- `MODEL_PATH=/path/to/model`：模型路径

### 5. 筛选 + 转 SFT 格式

```bash
python -m Rl_Specilist.agent.reject_sampling.data_preprocess.convert_to_sft \
    --top_k 2 \
    --min_score 0.7 \
    --output_dir ~/data/reject_sampling_sft
```

### 6. SFT 训练

```bash
bash Rl_Specilist/agent/reject_sampling/run_sft_from_reject.sh 8 ~/data/sft_ckpt
```

可覆盖的参数：
- `LR=5e-5`：学习率（默认 1e-5）
- `TOTAL_EPOCHS=5`：训练轮数（默认 3）
- `MAX_LENGTH=16384`：最大序列长度（默认 32768）

## 验证状态

以下已通过验证：
- 所有非 verl 模块导入正常
- 端到端流程（trajectory 保存 → 读取 → 筛选 → SFT parquet）验证通过
- verl 依赖模块（BaseTool/BaseInteraction 子类）需在 conda 环境（Python 3.10+）下运行

## 风险与应对

| 风险 | 应对 |
|---|---|
| DeepSeek API 速率限制 | rule-based 优先减少 API 调用；带重试 + 退避 |
| Qwen3-4B 生成质量低导致通过率低 | 提高 `N_SAMPLES=16`；降低 `TEMPERATURE=0.5` |
| SWE-bench 环境搭建复杂 | 先做 ToolMind + TerminalTraj，SWE 后做 |
| Docker 资源消耗大 | TerminalTraj 限制并发容器数 |
| `solution_str` 缺少 tool response 信息 | 结合 `rollout_reward_scores` 补充 |

## 复用的 verl 组件

以下 verl 文件直接复用，不修改：
- `verl/trainer/main_ppo.py` — rollout 入口
- `verl/experimental/agent_loop/tool_agent_loop.py` — 多轮 agent loop
- `verl/workers/reward_manager/naive.py` — reward manager
- `verl/trainer/sft_trainer.py` — SFT trainer
- `examples/sft/agent_sft/run_qwen3_4b_sft.sh` — SFT 启动脚本（改 DATA_DIR）
