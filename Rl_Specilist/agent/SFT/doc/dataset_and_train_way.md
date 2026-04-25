针对上面我们讨论的agentic 模型需要具备的能力点， 帮我总结所有需要的能力， 并指出每个能力需要的训练方法和数据集， 画出一个图给我

我在把它整理成一张“能力—训练—数据—基准”的总图。先把最关键的能力面和对应基准对齐，再把训练方法压到一页里。

我已经把“遗漏项”并进总图了，重点会放在记忆、预算、协作、证据链这几块，因为它们最决定通用性。

已思考 27s

我把通用 agentic 体拆成 10 个能力面：格式协议、工具路由、规划分解、长程执行、动态改计划、失败恢复、事实性/校准、长期记忆、多代理协作、预算与自治控制。你前面提到的主干，已经分别能被 SWE-bench（真实 GitHub issue 代码修复）、Terminal-Bench（真实终端长流程）、τ-bench（动态用户与域内规则）、GAIA（推理+多模态+浏览+工具）、SimpleQA / BrowseComp（事实性与难检索信息）、MemoryAgentBench / LongMemEval（记忆）、MultiAgentBench（多代理协作）这些基准覆盖到不同部分。

下面这张图是我建议的总图。

通用 Agentic 体
├─ 1. 格式协议 / 指令遵从
│  ├─ 目标：按 schema 输出、按工具协议调用、少跑偏
│  ├─ 方法：SFT + 格式硬约束 + 负样本对比 + DPO
│  └─ 数据：function-call 轨迹、JSON/工具调用样本、格式打乱/修复样本
│
├─ 2. 工具路由 / 何时用工具
│  ├─ 目标：知道该不该搜、该不该算、该不该问
│  ├─ 方法：分类式监督 + hard negative mining + RL
│  └─ 数据：可直答/需检索/需计算/需浏览三分数据；GAIA；SimpleQA；BrowseComp
│
├─ 3. 规划分解 / 短程推理
│  ├─ 目标：把任务拆成可执行子目标
│  ├─ 方法：SFT on plans + process reward + outcome reward
│  └─ 数据：多步问答、表格分析、轻量研究任务、GAIA
│
├─ 4. 长程执行 / 多步探索
│  ├─ 目标：几十步到几百步持续推进
│  ├─ 方法：课程学习（短→长）+ 稀疏终局奖励 + GRPO/PPO
│  └─ 数据：SWE-bench、Terminal-Bench、仓库级修复、真实系统操作任务
│
├─ 5. 动态改计划 / 用户意图漂移
│  ├─ 目标：用户中途改需求还能继续对齐
│  ├─ 方法：多轮对话轨迹监督 + policy conflict reward
│  └─ 数据：τ-bench、改需求/加约束/撤销前提的对话轨迹
│
├─ 6. 失败恢复 / 韧性
│  ├─ 目标：报错后继续试、换工具、回滚、修正
│  ├─ 方法：failure-injection RL + recovery reward
│  └─ 数据：工具报错、404、超时、编译失败、测试失败、部分成功轨迹
│
├─ 7. 事实性 / 校准 / 诚实
│  ├─ 目标：知道不知道、该查再查、该拒答就拒答
│  ├─ 方法：置信度奖励 + abstention 奖励 + 证据约束
│  └─ 数据：SimpleQA、BrowseComp、时效性事实题、不可回答/已过期样本
│
├─ 8. 长期记忆 / 更新 / 遗忘
│  ├─ 目标：记住偏好、更新旧信息、忘掉过期信息
│  ├─ 方法：记忆写读分离 + 记忆回放 + selective forgetting 训练
│  └─ 数据：MemoryAgentBench、LongMemEval、跨 session 对话、知识更新轨迹
│
├─ 9. 多代理协作 / 委派 / 复核
│  ├─ 目标：会分工、会交接、会互相校验
│  ├─ 方法：self-play + role specialization + critique/merge
│  └─ 数据：MultiAgentBench、BattleAgentBench、协作/竞争式任务
│
└─ 10. 预算与自治控制
   ├─ 目标：在 token / tool / time / context 预算内完成任务
   ├─ 方法：budget-aware reward + tool-cost penalty + stop policy
   └─ 数据：同一任务的多预算版本、上下文压缩任务、context-folding 风格样本
这 10 个能力，建议这样训
能力	训练方法	你该重点收的数据
格式协议	SFT、硬约束、DPO	工具调用格式、JSON schema、格式纠错样本
工具路由	分类监督 + hard negatives + RL	“直答 vs 检索 vs 计算 vs 浏览”三分样本
规划分解	计划监督 + process reward	多步任务的 plan / act / reflect 轨迹
长程执行	课程学习 + 稀疏终局奖励 + GRPO/PPO	SWE-bench、Terminal-Bench、长流程操作
动态改计划	多轮轨迹监督 + 冲突奖励	τ-bench 式用户改口、补约束、反悔
失败恢复	failure injection + recovery reward	工具报错、测试失败、超时、回滚
事实性/校准	置信度训练 + abstention reward	SimpleQA、BrowseComp、时效性事实题
长期记忆	记忆写读分离 + 回放 + 遗忘惩罚	MemoryAgentBench、LongMemEval、跨会话偏好更新
多代理协作	self-play + role specialization	MultiAgentBench、协作/竞争式任务
预算与自治	budget-aware reward + stop policy	多预算版本任务、上下文压缩、工具成本控制
我会特别补的“遗漏项”

你前面的方案已经很强，但要做“通用体”，我认为最容易漏掉的是三件事：

长期记忆：不是长上下文，而是跨会话、可更新、可遗忘的记忆。MemoryAgentBench 明确把能力拆成 accurate retrieval、test-time learning、long-range understanding、selective forgetting。
预算意识：很多 agent 会做，但不会在预算内做完；这个会直接影响真实可用性。
证据驱动自检：不是“会回答”，而是“先找证据，再回答；证据不够就降置信度或拒答”。SimpleQA 和 BrowseComp 正好分别覆盖短事实和难检索信息。
推荐的训练顺序
1. 格式协议 / 工具路由
2. 短程规划 / 单工具任务
3. 动态用户 / 失败恢复
4. 长程执行 / 多工具协同
5. 事实性 / 校准 / 诚实
6. 长期记忆 / 多代理协作
7. 预算与自治控制
8. 混合式 agentic RL 总训