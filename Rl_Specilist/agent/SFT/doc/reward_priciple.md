帮我把上面我们讨论的reward 设计原则也总结一下
已思考 1 秒

可以，把我们前面讨论的 reward 设计原则压成一版更通用的框架：

1. 先奖励结果，再奖励过程。
最终正确性仍然是主目标，但不能只看最终答案；还要把“怎么得到答案”纳入奖励。

2. 不要把“不会”惩罚得太狠。
错误答案可以扣分，但“不确定 / 拒答 / 需要验证”不应该被重罚，否则模型会学成过度保守，或者为了拿分而乱猜。

3. 工具使用要奖励“合理性”，不是奖励“次数少”。
核心不是少用工具，而是该用时用，不该用时不用。
所以工具成本应该是“超过合理阈值才惩罚”，或者按工具价值分层计费，而不是一刀切。

4. 必须显式奖励 verify。
agent 里最关键的不是“查”，而是“查完后确认是否修正”。
verify 行为本身要有分，verify 后真的纠错要有更高分。

5. 校准要单独建模。
模型输出的置信度要和真实正确率对齐。
也就是让模型学会：
“我说 90% 把握时，真的大多数时候是对的；我说 30% 时，真的更可能错。”

6. 过程级奖励比只看终局更重要。
Agent 任务里，单靠最终 reward 很难分清哪一步对、哪一步错。
所以最好加 process reward / step reward / PRM，帮助模型学会中间步骤的质量。

7. 训练目标要避免“表演式客观”。
不能只奖励“看起来像在验证”“看起来很谦虚”。
要奖励真实证据一致、真实修正、真实交叉验证，而不是话术模板。

8. 奖励设计要平衡三件事：正确性、效率、可信度。

正确性：答案对不对
效率：工具是不是用得太多
可信度：模型是否知道自己知道什么、不知道什么

9. 对不同任务要分开定义 reward。
客观题、时效题、开放题，不该用同一套判分逻辑。

客观题：rule verify
时效题：真实检索结果
开放题：reward model / judge model / 人工 rubric

10. 终极目标不是“更会答”，而是“更会判断何时该答”。
真正好的 agent，不是永远强行给答案，而是能在“直接回答、调用工具、继续验证、承认不确定”之间做出正确选择。

如果把这些收成一句话，就是：

Reward 设计的核心，不是单纯逼模型更像“会答题的人”，而是让它学会“对世界保持校准、对不确定性保持敬畏、对工具保持理性使用”。

我也可以把这套原则整理成一张**“reward 设计原则图”**给你。

具体的细节也总结一下

我把前面讨论的内容整理成一个比较完整的 Agentic RL Reward Design Spec v1。这份东西其实已经接近可以直接指导训练系统设计了。

1. 设计目标

Agentic RL 的目标不是：

提高答案正确率

而是：

提高任务完成率
+
提高客观性
+
降低幻觉
+
提高工具使用能力
+
提高校准能力

因此 Reward 不应该只看：

Answer Correct?

而应该看：

Know
↓
Search
↓
Verify
↓
Revise
↓
Answer

整个轨迹。

2. Reward 总体结构

定义：

R=R
answer
	​

+R
calibration
	​

+R
verify
	​

+R
process
	​

−R
tool
	​

−R
safety
	​


其中：

模块	作用
R_answer	结果正确性
R_calibration	校准
R_verify	验证行为
R_process	过程质量
R_tool	工具成本
R_safety	安全约束
3. Answer Reward
客观题

例如：

数学
代码
逻辑题
知识题

使用：

rule verifier

例如：

if correct:
    reward = 1
else:
    reward = -0.3

注意：

不要：

wrong = -1

因为会导致：

过度拒答

推荐：

correct = +1

wrong = -0.2~-0.5

unknown = 0
时效题

例如：

今天苹果股价
最新新闻
天气

采用：

Search Tool
+
Answer Match

奖励：

match = True

reward = +1

否则：

reward = -0.3
开放题

例如：

战略分析
论文评价
商业分析

采用：

Judge Model
Reward Model
Human Preference

输出：

[-1,1]
4. Calibration Reward

核心目标：

训练：

知道自己知道什么
知道自己不知道什么

模型额外输出：

Confidence: 0~1

例如：

Answer: B
Confidence: 0.85

真实标签：

Correct=1
Wrong=0

Brier Score：

R
cal
	​

=−(p−y)
2

例如：

正确：

p=0.9

得到：

-0.01

错误：

p=0.9

得到：

-0.81

效果：

模型逐渐学会：

不知道
→降低confidence

知道
→提高confidence
5. Tool Cost Reward

目标：

不是少用工具。

而是：

合理使用工具

错误设计：

tool_count * λ

会导致：

压制Agent能力

推荐：

工具分级：

calculator = 0.01

search = 0.05

browser = 0.1

python = 0.2

human = 1.0

设置免费预算：

free_budget = 2

惩罚：

R
tool
	​

=λmax(0,cost−budget)

目的：

让模型学会：

先便宜工具
再昂贵工具
6. Verify Reward

这是降低幻觉最重要的模块之一。

错误Agent：

Search
Answer

正确Agent：

Search
Verify
Answer

奖励：

if verify:
    reward += 0.2

如果：

Verify
发现错误
修正答案

额外：

reward += 0.5

即：

R
verify
	​

=VerifyBonus+CorrectionBonus

重点：

奖励：

真正纠错

而非：

写一句
"我进行了验证"
7. Process Reward

解决：

Credit Assignment

问题。

轨迹：

Thought
Search
Verify
Answer

PRM输出：

+0.1
+0.3
+0.4

累计：

R
process
	​

=∑r
t
	​


重点关注：

Tool Selection

是否选对工具

例如：

数学题
→ calculator

知识题
→ search
Tool Usage

是否正确解析工具结果

例如：

Search得到A

Answer写B

扣分。

Reflection

是否识别失败

例如：

Search失败

继续：

换Query

加分。

Persistence

是否持续尝试

例如：

第一次失败
直接放弃

扣分。

失败
反思
重试
成功

加分。

8. Epistemic Reward

这是Agent RL中特有的重要模块。

目标：

训练：

认识知识边界

奖励：

Case1

知道自己知道

正确
高confidence

加分。

Case2

知道自己不知道

低confidence
调用工具

加分。

Case3

不知道却硬答

高confidence
错误

重罚。

Case4

应该搜索却不搜索

扣分。

Case5

已经有充分证据还继续搜索

扣分。

9. Safety Reward

Agent训练后容易出现：

奖励黑客
工具滥用
伪验证

增加：

Fake Verify

例如：

我验证了一下

实际没验证。

扣分。

Fake Citation

引用不存在证据。

扣分。

Tool Hallucination

声称：

Search结果显示

实际上没有。

扣分。

10. 最终 Reward

推荐：

R=R
answer
	​

+0.2R
cal
	​

+0.2R
verify
	​

+0.3R
process
	​

+0.2R
epistemic
	​

−0.1R
tool
	​

−R
safety
	​

11. Agentic RL 真正训练的能力

最终不是训练：

会搜索
会调用工具

而是训练下面这套能力链：

识别问题
↓
评估把握度
↓
决定是否搜索
↓
选择工具
↓
执行工具
↓
验证结果
↓
发现错误
↓
修正答案
↓
输出置信度

从能力层次看，可以总结为：

L1 任务正确性（Answer）

L2 工具使用（Tool Use）

L3 工具选择（Tool Selection）

L4 验证能力（Verify）

L5 反思纠错（Reflection）

L6 校准能力（Calibration）

L7 认识知识边界（Epistemic Awareness）

而我认为对于下一代 Agent 模型来说，L6（校准）和 L7（认识知识边界）才是 Agentic RL 相对于传统 Reasoning RL 最重要、也最容易产生能力跃迁的部分。 Reasoning RL 教模型“如何推导答案”，Agentic RL 则进一步教模型“何时不该只依赖自己的推导，而应该去获取和验证证据”。