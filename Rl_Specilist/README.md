# Rl_Specilist — RL Specialist 特性集合

基于 verl 框架的强化学习训练特性和工具集合，涵盖 Agent 训练（SFT / Reject Sampling / Online DPO / Agentic RL）、数学专精多阶段训练、以及底层训练稳定性增强技术。

## 目录总览

```
Rl_Specilist/
├── README.md                              ← 本文件
│
├── agent/                                 # Agent 训练
│   ├── agent_ability.md                   # Agent 能力定义与训练路线图
│   ├── SFT/                               # Agent SFT 数据准备与训练
│   │   ├── data_preprocess/               # 数据处理脚本
│   │   │   ├── README.md                  # 预处理脚本使用指南
│   │   │   ├── agent_instruct.py          # AgentInstruct → Parquet
│   │   │   ├── mix_sharegpt_agentinstruct_sft.py  # 混合 ShareGPT + AgentInstruct
│   │   │   └── prepare_agent_sft_data.py  # SFT 数据完整准备流程
│   │   ├── doc/
│   │   │   ├── dataset_and_train_way.md   # 能力-训练-数据-基准 总图
│   │   │   └── reward_priciple.md         # Reward 设计十大原则
│   │   ├── format_only_sft_dataset.py     # 仅格式训练的 SFT 数据生成
│   │   └── run_qwen3_0_6b_agent_instruct.sh
│   │
│   ├── reject_sampling/                   # Reject Sampling（离线轨迹筛选）
│   │   ├── README.md                      # 完整使用指南
│   │   ├── setup/                         # 一键环境安装脚本
│   │   ├── data_preprocess/               # prompt 提取 + SFT 转换
│   │   ├── tools/                         # 真实环境工具
│   │   ├── reward/                        # DeepSeek API judge
│   │   ├── config/                        # rollout 配置
│   │   ├── run_reject_sampling.sh         # rollout 启动
│   │   └── run_sft_from_reject.sh         # SFT 训练启动
│   │
│   ├── online_dpo/                        # Online DPO（在线偏好训练）
│   │   ├── README.md                      # 使用指南
│   │   ├── config/                        # 训练配置
│   │   │   ├── agent_hermes_gateway.yaml  # Gateway DPO 配置（当前）
│   │   │   ├── agent_dpo_judge.yaml       # Sandbox DPO 配置（旧）
│   │   │   ├── agent_gdpo_judge.yaml      # Sandbox GDPO 配置（旧）
│   │   │   ├── online_dpo.yaml            # 原始 DPO 配置（旧）
│   │   │   └── tool_config_sandbox.yaml   # Sandbox 工具注册（旧）
│   │   ├── hermes_entrypoint.py           # Agent 入口（当前）
│   │   ├── custom_hermes_runner.py        # Runner（当前）
│   │   ├── reward/llm_judge.py            # Judge 打分（当前）
│   │   ├── run_hermes_gateway_dpo.sh      # Gateway 训练启动（当前）
│   │   ├── run_multi_agent_dpo.sh         # Sandbox 模式（旧）
│   │   ├── run_agent_dpo.sh               # Agent DPO（旧）
│   │   └── run_online_dpo.sh              # 原始 online DPO（旧）
│   │
│   └── RL/                                # Agentic RL（多轮工具调用 GRPO）
│       ├── README.md                      # 使用指南
│       ├── config/agentic_grpo.yaml       # GRPO 训练配置
│       ├── data_preprocess/               # 多轮数据生成
│       ├── reward/agentic_reward.py       # 多维复合奖励
│       ├── tools/                         # calculator + submit_answer
│       ├── test_smoke.py                  # 冒烟测试
│       └── run_agentic_rl.sh              # 训练启动脚本
│
├── math/                                  # 数学专精多阶段训练
│   ├── README.md                          # 训练路线图（Stage 1-5 总览）
│   ├── dataset/                           # 数据集生成与处理脚本
│   │   ├── README.md                      # 数据集脚本使用指南
│   │   ├── mixed_math_dataset.py          # 混合数据集主脚本（4 数据源）
│   │   ├── create_mixed_data.py           # 从文件夹聚合数据集
│   │   ├── mixed_reasoning_dataset_with_prompt_types.py  # 带 prompt 类型分类
│   │   ├── math_dataset_with_prompt_types.py  # 数学数据集 + prompt 类型
│   │   ├── mixed_gdpo_reward.py           # GDPO 多维奖励函数（Stage 2-4）
│   │   ├── mixed_reasoning_reward.py      # 混合推理奖励函数（Stage 5）
│   │   ├── test_mixed_math_score.py       # 奖励函数单元测试
│   │   ├── run_mixed_math.sh              # 一键运行脚本
│   │   └── README_mixed_math.md           # 详细使用指南
│   ├── stage 1 - gsm8k 580 step/         # GSM8K DAPO 基础训练
│   ├── stage 2 - mixed math dataset/     # 混合数据集 + GDPO + 多维奖励
│   ├── stage 3 - mixed math dataset + len penalty/  # + 长度惩罚
│   ├── stage 4 - mixed dataset + len penalty + max effort/  # + max_effort
│   └── stage 4 - mixed stage 5 - mixed_logic_and_math/  # + 逻辑推理
│
├── muon/                                  # Muon 优化器
│   └── muon.md                            # 配置参考、调优指南
│
├── tools/                                 # 通用工具
│   ├── README.md                          # 工具使用指南
│   ├── compare_weights.py                 # SafeTensors 权重对比
│   └── compare_weight.result              # 对比结果示例
│
│   # === 独立文档（训练稳定性增强技术） ===
├── dapo_dynamic_sampling_guide.md         # 动态采样（group_resample / filter / reward_resample）
├── keep_sampling_mask.md                  # 训练-推理采样一致性
├── offpolicy_seq_mask_usage.md            # DeepSeek-V3.2 off-policy 序列掩码
├── optimizer_state_reset.md               # 每迭代优化器状态重置
├── unbiased_kl_estimate.md                # 无偏 KL 估计
└── penalty_features.md                    # Repetition Penalty + Length Penalty
```

## 特性全景图

### Agent 训练

```
训练流程: Agent SFT → Reject Sampling → Online DPO → Agentic RL
              ↓              ↓                ↓              ↓
           建立基础      离线收集      在线偏好学习    多维 Reward
           行为模式      高质量轨迹     best/worst     GRPO 全训练
```

| 阶段 | 方法 | 适用场景 | 文档 |
|------|------|----------|------|
| 初始化 | SFT | 建立基础 agent 行为模式 | `agent/SFT/doc/` |
| 数据收集 | Reject Sampling | 离线收集高质量轨迹 | `agent/reject_sampling/README.md` |
| 偏好学习 | Online DPO | 从相对偏好中高效学习 | `agent/online_dpo/README.md` |
| 强化训练 | Agentic RL | 多维复合奖励 GRPO | `agent/RL/README.md` |

### 数学专精训练

5 阶段渐进式训练：单数据 → 混合 → 长度控制 → max_effort → 全能推理

详见 `math/README.md`。

### 训练稳定性增强

| 技术 | 解决什么问题 | 文档 |
|------|-------------|------|
| Dynamic Sampling | 全 0/全 1 退化组，低 reward 样本 | `dapo_dynamic_sampling_guide.md` |
| Off-Policy Seq Mask | 异步训练中 off-policy 噪声梯度 | `offpolicy_seq_mask_usage.md` |
| Unbiased KL Estimate | KL 估计有偏导致训练不稳定 | `unbiased_kl_estimate.md` |
| Keep Sampling Mask | 训练-推理 top-k/top-p 不一致 | `keep_sampling_mask.md` |
| Optimizer State Reset | 动量跨迭代累积影响当前迭代 | `optimizer_state_reset.md` |
| Repetition Penalty | 模型生成重复 token | `penalty_features.md` |
| Length Penalty | 模型产出过长推理 | `penalty_features.md` |
| Muon Optimizer | 2D 权重矩阵优化效率 | `muon/muon.md` |

## 相关资源

### verl 框架中的对应修改

| 文件 | 特性 |
|------|------|
| `verl/trainer/ppo/core_algos.py` | DPO loss, Off-Policy Seq Mask, Dynamic Sampling |
| `verl/trainer/ppo/ray_trainer.py` | DPO training loop, filter_groups, group_resample, reward_resample |
| `verl/utils/muon/` | Muon optimizer 实现 |
| `verl/utils/reward_score/math_reward.py` | 数学答案提取与验证 |
| `verl/utils/reward_score/mixed_math.py` | 混合数学奖励 |
| `verl/workers/config/optimizer.py` | MuonOptimizerConfig |
| `verl/workers/utils/losses.py` | DPO loss 函数 |
| `verl/workers/utils/padding.py` | Keep sampling mask |
| `verl/utils/torch_functional.py` | `masked_whiten` 等工具函数 |
| `verl/trainer/config/optim/muon_fsdp.yaml` | Muon FSDP 配置模板 |
| `verl/trainer/config/rollout/rollout.yaml` | repetition_penalty 配置 |
| `verl/experimental/agent_loop/` | Agent 训练 loop (多轮/工具调用) |
| `verl/utils/chat_template.py` | Chat template 初始化与系统提示提取 |
| `verl/workers/reward_manager/dapo.py` | DAPO 动态采样 + 长度惩罚 |
| `verl/workers/reward_manager/gdpo.py` | GDPO 多维奖励管理 |

### 测试文件

| 文件 | 测试内容 |
|------|---------|
| `tests/trainer/ppo/test_dpo_loss.py` | DPO loss 正确性 |
| `tests/trainer/ppo/test_dpo_pairing.py` | best/worst 配对逻辑 |
| `tests/utils/reward_score/test_math_reward.py` | 数学奖励函数 |
| `tests/workers/config/test_muon_optimizer.py` | Muon 优化器配置 |

### 示例脚本

| 文件 | 说明 |
|------|------|
| `Rl_Specilist/math/dataset/README.md` | 数据集生成脚本完整指南 |
| `Rl_Specilist/math/dataset/run_mixed_math.sh` | 混合数学数据集一键生成 |
| `Rl_Specilist/math/dataset/README_mixed_math.md` | 混合数据集详细使用指南 |
| `Rl_Specilist/tools/README.md` | 工具使用指南（权重对比、LoRA 提取） |
| `Rl_Specilist/agent/SFT/data_preprocess/README.md` | SFT 数据预处理脚本指南 |
| `examples/sft/agent_sft/README.md` | Agent SFT 训练指南 |
| `scripts/extract_lora_from_ckpt.py` | 从 checkpoint 提取 LoRA（详见 tools/README.md） |

## 环境要求

- Python 3.10+
- verl 框架
- CUDA 12.1+
- Docker（Reject Sampling 的 TerminalTraj / SWE 环境）
- Uni-Agent Gateway（Online DPO 的推理网关）
- DeepSeek API Key（Reject Sampling / Online DPO 的 judge）
- 8× GPU（推荐）或 4× GPU（最小）

## 贡献

所有代码和文档存放在 `Rl_Specilist/` 目录下。新增特性时请遵循以下规范：

1. 按照功能分类放入对应子目录
2. 提供 `README.md` 说明使用方式和配置参数
3. 提供启动脚本（`run_*.sh`）和 Hydra 配置（`config/*.yaml`）
4. 数据处理脚本统一放在 `data_preprocess/` 子目录
