# 轻量Agentic数据集与环境接入方案

## 0. 背景与定位

当前工具调用RL阶段（APIGen / ToolACE / Hammer / BFCL 单轮工具调用预测）已完成，下一步是让4B模型进入**完整agent环境探索 + 可验证结果**的RL阶段（terminal / SWE，10-20轮）。本文档梳理的是介于两者之间的**课程学习第一级台阶**：

- 任务复杂度：≤10轮内可解
- 环境复杂度：不需要复杂sandbox，本地可轻量搭建
- 训练目标：基础多轮规划（think→act→observe→再规划）+ 真实工具调用（而非单轮预测）

---

## 1. 数据集/环境总览

| 类别 | 数据集/环境 | 典型轮数 | 环境搭建成本 | 奖励/验证方式 | 备注 |
|---|---|---|---|---|---|
| 检索式多跳QA | HotpotQA / FEVER / Bamboogle | 2-6 | 一个wiki搜索API或本地索引 | EM/F1匹配最终答案 | 工具集只有search+finish，环境最轻 |
| 检索式多跳QA | FireAct | 2-6 | 同上，且轨迹现成 | 同上 | 专为轻量agent微调设计的ReAct轨迹数据集 |
| 代码/SQL执行 | InterCode-SQL | 5-10 | 单一sqlite容器 | 最终查询结果与ground truth匹配 | "执行→报错/结果→修正"loop，结构上同构于SWE-bench，可作为terminal agent前置阶段 |
| 代码/SQL执行 | InterCode-Bash | 5-10 | 单一bash容器 | 命令输出/测试通过 | 同上，动作空间和轨迹长度都更小 |
| 文本游戏/具身 | ALFWorld | 几步-15+（多数<10） | pip安装，TextWorld底层 | 环境内置任务完成判定 | 离散动作空间，AgentTuning/ETO等已有现成SFT轨迹 |
| 文本游戏/具身 | WebShop | 5-10 | 需起模拟商品检索服务，略重 | 商品匹配得分 | 规划成分（搜索→比较→决策）强于ALFWorld |
| 合成多轮工具调用 | BFCL multi-turn/multi-step子集 | 2-8 | 无需环境，带执行检查器 | 内置state-based检查 | 可直接复用现有BFCL评测代码做reward |
| 合成多轮工具调用 | API-Bank | 2-8 | 无需环境 | 标注匹配 | 标注了是否需要澄清/多步调用 |

---

## 2. 推荐组合

**HotpotQA/FEVER（ReAct）+ InterCode-SQL + ALFWorld**

理由：
1. 三者环境搭建成本都很低（一个API/索引、一个sqlite容器、一个pip包），不涉及重型sandbox。
2. 轮数天然卡在10轮以内，符合当前课程学习阶段的目标。
3. InterCode 的"执行反馈→修正"loop 与未来 terminal/SWE 的循环结构同构，衔接自然；ALFWorld 补充离散动作规划；ReAct-QA 补充纯检索式的推理-工具调用交替能力。
4. 三者的验证信号都是确定性的（EM/F1、执行结果匹配、任务完成判定），不依赖judge模型，避免过早引入self-judging的Goodhart风险。

---

## 3. 分环境接入与Reward设计建议

### 3.1 HotpotQA / FEVER（ReAct loop）

**接入方式**：唯一工具为 `search(query)`，返回wiki片段；模型交替输出 `Thought` / `Action` / `Observation`，直到输出 `finish(answer)`。

**轨迹格式建议**：
```json
{
  "task_id": "...",
  "turns": [
    {"role": "assistant", "thought": "...", "action": "search", "action_input": "...", "step": 1},
    {"role": "tool", "observation": "...", "step": 1},
    ...
    {"role": "assistant", "action": "finish", "action_input": "...", "step": k}
  ],
  "gold_answer": "..."
}
```

**Reward映射**（沿用现有五分量框架）：
- `R_out`：最终答案EM/F1
- `R_info`：每一步search返回的片段是否被后续thought实际引用（避免"搜了但没用"）
- `R_eff`：轮数惩罚，鼓励在更少轮数内finish
- `R_kl`：常规KL约束
- 可不需要`R_cf`（这里没有复杂格式合规问题，工具集单一）

### 3.2 InterCode-SQL / InterCode-Bash

**接入方式**：单容器起sqlite或bash，模型输出命令→环境执行→返回stdout/stderr/报错→模型修正，直到提交最终答案或达到最大轮数。

**轨迹格式建议**：与terminal/SWE阶段保持同一schema（`command` / `stdout` / `stderr` / `exit_code`），便于直接复用你现有的trajectory buffer和sandbox microservice接口，只是任务规模小很多。

**Reward映射**：
- `R_out`：最终查询结果/测试是否通过
- `R_cf`：命令语法是否合规（避免连续无效命令）
- `R_eff`：命令轮数惩罚
- `R_kl`：常规约束

### 3.3 ALFWorld

**接入方式**：TextWorld底层，环境返回房间描述与可执行动作列表，模型从有限动作空间中选择。

**轨迹格式建议**：动作+环境反馈的标准TextWorld格式，可直接借用AgentTuning已发布的SFT轨迹做冷启动。

**Reward映射**：
- `R_out`：环境内置任务完成判定（二值）
- `R_eff`：步数惩罚
- 由于动作空间离散且合法性由环境保证，`R_cf`基本不需要

---

## 4. 课程学习编排建议

| 阶段 | 内容 | 轮数 | 状态 |
|---|---|---|---|
| Stage 0 | 单轮工具调用预测（APIGen/ToolACE/Hammer/BFCL） | 1 | 已完成 |
| Stage 1 | 本文三类环境（ReAct-QA / InterCode / ALFWorld） | ≤10 | 待启动 |
| Stage 2 | terminal / SWE 全agent环境 | 10-20 | 规划中 |

Stage 1 的产出（统一trajectory schema + 确定性reward）可以直接作为Stage 2的冷启动SFT数据和reward框架的验证集，减少切换阶段时的返工。

---

## 5. 待确认事项

- BFCL multi-turn子集与你已诊断出的Irrelevance/Relevance低分问题是否需要在Stage 1中专门补一批负样本（不可用工具/信息不足场景），建议一并纳入课程设计。
- InterCode环境的docker镜像是否能在你们的sandbox microservice架构下直接复用，还是需要重新适配。
