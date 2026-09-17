# 幻觉抵制 RL —— 四源真实数据样例通览

> 配套文档：`HALLUCINATION_RL_DESIGN.md`。本文件只做一件事：**把四个数据源的真实样本摆出来**，
> 让“数据长什么样”这件事可核对。所有样例都是脚本直接读原始文件打印的，未手工改写。

| 源 | 原始文件 | 获取方式 | 总量 |
|---|---|---|---|
| K&K clean | `K-and-K/knights-and-knaves` | HF datasets（parquet） | 6,200 train + 700 test |
| K&K perturbed | `K-and-K/perturbed-knights-and-knaves` | HF datasets（parquet，6 个 split） | 37,015 train + 4,161 test |
| MiP | `github.com/tianyi-lab/MiP-Overthinking` `data/*.json` | raw.githubusercontent | 984 |
| FalseQA | `github.com/thunlp/FalseQA` `dataset/*.csv` | raw.githubusercontent | train 2,374 / valid 982 / test 1,374 |
| GSM-IC | `github.com/google-research-datasets/GSM-IC` `GSM-IC_{2step,mstep}.json` | raw.githubusercontent | 2step 34,220 + mstep 23,832 |

拉取命令（一次拉全，约 50 MB）：

```bash
# K&K：走 HF datasets-server 转好的 parquet（clean 7 split + perturbed 12 split）
curl 'https://datasets-server.huggingface.co/parquet?dataset=K-and-K%2Fknights-and-knaves'
curl 'https://datasets-server.huggingface.co/parquet?dataset=K-and-K%2Fperturbed-knights-and-knaves'
#   -> 取返回 JSON 的 parquet_files[].url 逐个下载，再用 pyarrow 读
# MiP
for f in gsm8k svamp math formula; do curl -sLO \
  https://raw.githubusercontent.com/tianyi-lab/MiP-Overthinking/main/data/$f.json; done
# FalseQA
for s in train valid test; do curl -sLO \
  https://raw.githubusercontent.com/thunlp/FalseQA/main/dataset/$s.csv; done
# GSM-IC
curl -sLO https://raw.githubusercontent.com/google-research-datasets/GSM-IC/main/GSM-IC_2step.json
curl -sLO https://raw.githubusercontent.com/google-research-datasets/GSM-IC/main/GSM-IC_mstep.json
```

---

## 1. K&K（logic 域）

### 1.1 字段结构

每条是一道“骑士与骗子”逻辑题，一行就自带**约束系统的机器可读形式**和**唯一解**：

| 字段 | 含义 |
|---|---|
| `quiz` | 题面（自然语言） |
| `names` | 人物列表，下标即约束系统里的角色编号 |
| `statements` | **字符串**（Python repr），元素 i = 人物 i 的断言，可 `ast.literal_eval` |
| `solution` | 布尔列表，`True` = 该人物是骑士 |
| `solution_text` | 解的英文句子 |
| `cot_head` / `cot_repeat_steps` / `cot_foot` | 官方 CoT（本方案不用） |
| `index` | 题目编号；**(人数, index) 是 clean↔perturbed 的联接键** |

`statements` 的算子：`telling-truth i`（i 是骑士）/ `lying i`（i 是骗子）/ `not` / `and` /
`or` / `->` / `<=>`。约束是 **「人物 i 的断言为真 ⟺ 人物 i 是骑士」**，对每个人各一条。

### 1.2 clean 一条完整样本

```
index      : 100
names      : ['Ethan', 'Abigail', 'David', 'Noah']
statements : (('or', ('lying', 2), ('lying', 1)), ('->', ('telling-truth', 1), ('telling-truth', 2)), ('and', ('telling-truth', 3), ('lying', 0)), ('<=>', ('lying', 1), ('telling-truth', 0)))
solution   : [False, True, True, True]
solution_text: Ethan is a knave, Abigail is a knight, David is a knight, and Noah is a knight.
```

**quiz（去掉公共开头）：**

```
You meet 4 inhabitants: Ethan, Abigail, David, and Noah. Ethan asserted: "David is a knave or Abigail is a knave".
Abigail was heard saying, "If Abigail is a knight then David is a knight". "Noah is a knight and Ethan is a knave" -
David. Noah remarked, "Abigail is a knave if and only if Ethan is a knight". So who is a knight and who is a knave?
```

**我们枚举出的解数：** `1`（唯一解，且与 `solution` 字段一致）

### 1.3 六种扰动 vs clean 原题（每条各 1 例，index 均为 100 的 4 人题）

这是本方案最关键的一组对照：**同一道题扰动前后的题面、约束系统、解。**

#### perturbed_statement

| | 内容 |
|---|---|
| clean 题面 | You meet 4 inhabitants: Ethan, Abigail, David, and Noah. Ethan asserted: "David is a knave or Abigail is a knave".<br>Abigail was heard saying, "If Abigail is a knight then David is a knight". "Noah is a knight and Ethan is a knave" -<br>David. Noah remarked, "Abigail is a knave if and only if Ethan is a knight". So who is a knight and who is a knave? |
| perturbed 题面 | You meet 4 inhabitants: Ethan, Abigail, David, and Noah. Ethan asserted: "David is a knave or Abigail is a knave".<br>Abigail was heard saying, "David is a knight and David is a knave". "Noah is a knight and Ethan is a knave" - David.<br>Noah remarked, "Abigail is a knave if and only if Ethan is a knight". So who is a knight and who is a knave? |
| clean 解 | Ethan is a knave, Abigail is a knight, David is a knight, and Noah is a knight. |
| perturbed 解 | Ethan is a knight, Abigail is a knave, David is a knave, and Noah is a knight. |
| statements 是否改变 | **是** |
| solution 是否改变 | **是** |
| 扰动后解数 | `1` |

#### perturbed_leaf

| | 内容 |
|---|---|
| clean 题面 | You meet 4 inhabitants: Ethan, Abigail, David, and Noah. Ethan asserted: "David is a knave or Abigail is a knave".<br>Abigail was heard saying, "If Abigail is a knight then David is a knight". "Noah is a knight and Ethan is a knave" -<br>David. Noah remarked, "Abigail is a knave if and only if Ethan is a knight". So who is a knight and who is a knave? |
| perturbed 题面 | You meet 4 inhabitants: Ethan, Abigail, David, and Noah. Ethan asserted: "David is a knave or Abigail is a knave".<br>Abigail was heard saying, "If Abigail is a knight then Noah is a knave". "Noah is a knight and Ethan is a knave" -<br>David. Noah remarked, "Abigail is a knave if and only if Ethan is a knight". So who is a knight and who is a knave? |
| clean 解 | Ethan is a knave, Abigail is a knight, David is a knight, and Noah is a knight. |
| perturbed 解 | Ethan is a knight, Abigail is a knight, David is a knave, and Noah is a knave. |
| statements 是否改变 | **是** |
| solution 是否改变 | **是** |
| 扰动后解数 | `1` |

#### random_pair

| | 内容 |
|---|---|
| clean 题面 | You meet 4 inhabitants: Ethan, Abigail, David, and Noah. Ethan asserted: "David is a knave or Abigail is a knave".<br>Abigail was heard saying, "If Abigail is a knight then David is a knight". "Noah is a knight and Ethan is a knave" -<br>David. Noah remarked, "Abigail is a knave if and only if Ethan is a knight". So who is a knight and who is a knave? |
| perturbed 题面 | A very special island is inhabited only by altruists and egoists. Altruists always tell the truth, and egoists always<br>lie. You meet 4 inhabitants: Ethan, Abigail, David, and Noah. Ethan was heard saying, "David is an egoist or Abigail<br>is an egoist". "If Abigail is an altruist then David is an altruist" - Abigail. David remarked, "Noah is an altruist<br>and Ethan is an egoist". According to Noah, "Abigail is an egoist if and only if Ethan is an altruist". So who is an<br>altruist and who is an egoist? |
| clean 解 | Ethan is a knave, Abigail is a knight, David is a knight, and Noah is a knight. |
| perturbed 解 | Ethan is an egoist, Abigail is an altruist, David is an altruist, and Noah is an altruist. |
| statements 是否改变 | 否（与 clean 完全相同） |
| solution 是否改变 | 否（与 clean 完全相同） |
| 扰动后解数 | `1` |

#### reorder_statement

| | 内容 |
|---|---|
| clean 题面 | You meet 4 inhabitants: Ethan, Abigail, David, and Noah. Ethan asserted: "David is a knave or Abigail is a knave".<br>Abigail was heard saying, "If Abigail is a knight then David is a knight". "Noah is a knight and Ethan is a knave" -<br>David. Noah remarked, "Abigail is a knave if and only if Ethan is a knight". So who is a knight and who is a knave? |
| perturbed 题面 | You meet 4 inhabitants: Ethan, Abigail, David, and Noah. Abigail was heard saying, "If Abigail is a knight then David<br>is a knight". Ethan asserted: "David is a knave or Abigail is a knave". "Noah is a knight and Ethan is a knave" -<br>David. Noah remarked, "Abigail is a knave if and only if Ethan is a knight". So who is a knight and who is a knave? |
| clean 解 | Ethan is a knave, Abigail is a knight, David is a knight, and Noah is a knight. |
| perturbed 解 | Ethan is a knave, Abigail is a knight, David is a knight, and Noah is a knight. |
| statements 是否改变 | 否（与 clean 完全相同） |
| solution 是否改变 | 否（与 clean 完全相同） |
| 扰动后解数 | `1` |

#### uncommon_name

| | 内容 |
|---|---|
| clean 题面 | You meet 4 inhabitants: Ethan, Abigail, David, and Noah. Ethan asserted: "David is a knave or Abigail is a knave".<br>Abigail was heard saying, "If Abigail is a knight then David is a knight". "Noah is a knight and Ethan is a knave" -<br>David. Noah remarked, "Abigail is a knave if and only if Ethan is a knight". So who is a knight and who is a knave? |
| perturbed 题面 | You meet 4 inhabitants: Vesper, Elodie, Thorsten, and Isolde. Vesper asserted: "Thorsten is a knave or Elodie is a<br>knave". Elodie was heard saying, "If Elodie is a knight then Thorsten is a knight". "Isolde is a knight and Vesper is<br>a knave" - Thorsten. Isolde remarked, "Elodie is a knave if and only if Vesper is a knight". So who is a knight and<br>who is a knave? |
| clean 解 | Ethan is a knave, Abigail is a knight, David is a knight, and Noah is a knight. |
| perturbed 解 | Vesper is a knave, Elodie is a knight, Thorsten is a knight, and Isolde is a knight. |
| statements 是否改变 | 否（与 clean 完全相同） |
| solution 是否改变 | 否（与 clean 完全相同） |
| 扰动后解数 | `1` |

#### flip_role

| | 内容 |
|---|---|
| clean 题面 | You meet 4 inhabitants: Ethan, Abigail, David, and Noah. Ethan asserted: "David is a knave or Abigail is a knave".<br>Abigail was heard saying, "If Abigail is a knight then David is a knight". "Noah is a knight and Ethan is a knave" -<br>David. Noah remarked, "Abigail is a knave if and only if Ethan is a knight". So who is a knight and who is a knave? |
| perturbed 题面 | A very special island is inhabited only by knaves and knights. Knaves always tell the truth, and knights always lie.<br>You meet 4 inhabitants: Ethan, Abigail, David, and Noah. Ethan asserted: "David is a knight or Abigail is a knight".<br>Abigail was heard saying, "If Abigail is a knave then David is a knave". "Noah is a knave and Ethan is a knight" -<br>David. Noah remarked, "Abigail is a knight if and only if Ethan is a knave". So who is a knave and who is a knight? |
| clean 解 | Ethan is a knave, Abigail is a knight, David is a knight, and Noah is a knight. |
| perturbed 解 | Ethan is a knight, Abigail is a knave, David is a knave, and Noah is a knave. |
| statements 是否改变 | 否（与 clean 完全相同） |
| solution 是否改变 | 否（与 clean 完全相同） |
| 扰动后解数 | `1` |

### 1.4 ⚠️ 全量枚举结果：K&K 扰动**不产生**不可解题

用上面那套 semantics，先在 clean 上自检，再对 6 个扰动 split 全量枚举 `n_solutions`：

**自检（clean）**：6,200 / 6,200 条都是「唯一解 且 与 `solution` 字段逐位一致」。

**求解器控制组**（证明枚举器能识别 0 解与多解）：

| 人造输入 | n_solutions | 符合预期 |
|---|---|---|
| 单人断言「我是骗子」 | 0 | ✅ 矛盾 |
| 两人互相断言对方是骗子 | 0 | ✅ 矛盾 |
| 两人断言同一句「0 号是骑士」 | 2 | ✅ 欠定 |
| 空约束（无人断言） | 4 = 2² | ✅ 全解 |

**扰动 split 全量枚举：**

| 扰动类型 | train 行数 | test 行数 | **0 解（矛盾）** | **1 解** | **≥2 解（缺条件）** |
|---|---|---|---|---|---|
| `perturbed_statement` | 6,194 | 700 | **0** | **6,894** | **0** |
| `perturbed_leaf` | 6,021 | 661 | **0** | **6,682** | **0** |
| `random_pair` | 6,200 | 700 | **0** | **6,900** | **0** |
| `reorder_statement` | 6,200 | 700 | **0** | **6,900** | **0** |
| `uncommon_name` | 6,200 | 700 | **0** | **6,900** | **0** |
| `flip_role` | 6,200 | 700 | **0** | **6,900** | **0** |
| **合计** | **37,015** | **4,161** | **0** | **41,176** | **0** |

结论：**41,176 条扰动样本（train 37,015 + test 4,161），100% 仍是唯一可解的**。
`perturbed_statement` / `perturbed_leaf` 改的是逻辑结构、答案随之改变、数据集也重新求解了（`statements` 与 `solution` 在 100% 的行上都与 clean 不同），但**结果是另一道唯一可解的题**，不是矛盾题、也不是欠定题。

→ 按本方案的 schema，这些行的正确标注是 `solvable=true`，**不提供任何弃答/诊断信号**。
详见设计文档 §4.1 与 §12 Q15。

---

## 2. MiP（math 域，不可解 + 可 diff 出被删条件）

仓库 `tianyi-lab/MiP-Overthinking` 的 `data/` 下 4 个文件（其余全是大模型 response 记录，不是数据）：

| 文件 | 条数 | 字段 | 能否构造诊断选项 |
|---|---|---|---|
| `gsm8k.json` | 582 | `question`（原题）/ `answer` / `insufficient_question`（删条件版） | ✅ 文件内 diff |
| `math.json` | 52 | 同上 + `solution` / `subject` / `level` / `unique_id` | ✅ 文件内 diff |
| `svamp.json` | 300 | 只有 `insufficient_question` + `answer` | ❌ 见 2.4 |
| `formula.json` | 50 | 只有 `insufficient_question` + `complexity` / `depth` | ❌ 无答案、无原题 |

### 2.1 `gsm8k.json` —— 原题 / 删条件版 / 答案

**例 1**

| | |
|---|---|
| 原题 | James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week? |
| 原答案 | `He sprints 3*3=<<3*3=9>>9 times ⏎ So he runs 9*60=<<9*60=540>>540 meters ⏎ #### 540` |
| 删条件后 | **James decides to run 3 sprints 3 times a week. How many total meters does he run a week?** |
| diff 得到的诊断选项 | 被删掉的那句数值条件 |

**例 2**

| | |
|---|---|
| 原题 | Eliza's rate per hour for the first 40 hours she works each week is $10. She also receives an overtime pay of 1.2 times her regular hourly rate. If Eliza worked for 45 hours this week, how much are her earnings for this week? |
| 原答案 | `Eliza is entitled to 45 -40 = <<45-40=5>>5 hours overtime pay. ⏎ Her hourly rate for the overtime pay is $10 x 1.2 = $<<10*1.2=12>>12. ⏎ So, Eliza will receive $12 x 5 =$<<12*5=60>>60 for overtime pay. ⏎ Her regular weekly earning is $10 x 40 = $<<10*40=400>>400. ⏎ Thus, Eliza will receive a total of $400 + $60 = $<<400+60=460>>460 for this week's work. ⏎ #### 460` |
| 删条件后 | **Eliza's rate per hour for the first 40 hours she works each week is $10. If Eliza worked for 45 hours this week, how much are her earnings for this week?** |
| diff 得到的诊断选项 | 被删掉的那句数值条件 |

**例 3**

| | |
|---|---|
| 原题 | A new program had 60 downloads in the first month. The number of downloads in the second month was three times as many as the downloads in the first month, but then reduced by 30% in the third month. How many downloads did the program have total over the three months? |
| 原答案 | `The number of downloads of the program in the second month increased to 3*60 = <<3*60=180>>180 ⏎ In the first two months, the total number of downloads of the program was 180+60 = <<180+60=240>>240 ⏎ In the third month, the number of downloads of the program reduced by 30/100*180 = <<30/100*180=54>>54 ⏎ There were 180-54 = <<180-54=126>>126 downloads in the third month. ⏎ In the three months, the total number of downloads of the program was 126+240 = <<126+240=366>>366 ⏎ #### 366` |
| 删条件后 | **The number of downloads in the second month was three times as many as the downloads in the first month, but then reduced by 30% in the third month. How many downloads did the program have total over the three months?** |
| diff 得到的诊断选项 | 被删掉的那句数值条件 |

### 2.2 `math.json`（LaTeX 题，带 `unique_id` 可回连 MATH）

| 字段 | 值 |
|---|---|
| `unique_id` | test/precalculus/927.json |
| `subject` | Precalculus |
| `level` | 4 |
| `question` | The set of points $(x,y,z)$ that satisfy<br>\[2x = 3y = -z\]is a line.<br><br>The set of points $(x,y,z)$ that satisfy<br>\[6x = -y = -4z\]is another line.<br><br>Find the angle between these lines, in degrees. |
| `insufficient_question` | The set of points $(x,y,z)$ that satisfy<br>\[2x = 3y = -z\]is a line.<br><br>There is another line in the space.<br><br>Find the angle between these lines, in degrees. |
| `answer` | 90^\circ |

### 2.3 `svamp.json` 的 300 条是**跨题拼接**，不是删条件

- `insufficient_question`: Dan had $ 3 left with him after he bought a candy bar. If he had $ 4 at the start How many bottle caps did danny have at first?
  - `answer`: `91.0`
- `insufficient_question`: Paco had 26 salty cookies and 17 sweet cookies. He ate 14 sweet cookies and 9 salty cookies. How many pencils does she have?
  - `answer`: `84.0`

实测：300/300 都是「**A 题的 Body + B 题的 Question**」拼起来的（例如上例第一条 =
SVAMP `chal-2` 的 Body + `chal-655` 的 Question），270 条答案取的是 Question 来源题的答案、
另 30 条对不上。**没有“被删掉的那个条件”这个对象**，无法生成细粒度诊断选项。
→ 见设计文档 §4.3：`svamp.json` 只能当三档（不可解但无诊断）来源，或直接丢弃。

### 2.4 `formula.json`（无答案、无原题，直接废弃）

- What is the value of $(\gamma - \arctan(\tan(\kappa) + 6 - \omega) \cdot 10^{x})$?   （complexity=3, depth=3）
- What is the value of $\min(8 - 4) \div \arcsin(\tan(u_2 + \epsilon \div 6) - I_3 \times (\lambda^1)) \cdot 1^{3}$?   （complexity=4, depth=3）

---

## 3. FalseQA（commonsense 域，假前提）

仓库 `thunlp/FalseQA` 的 `dataset/{train,valid,test}.csv`，字段只有三个：`question` / `answer` / `label`。

| split | 行数 | label=1（假前提，不可解） | label=0（同模板真前提，可解） |
|---|---|---|---|
| `train.csv` | 2374 | 1187 | 1187 |
| `valid.csv` | 982 | 491 | 491 |
| `test.csv` | 1374 | 687 | 687 |

### 3.1 同模板 假前提 / 真前提 对照

| # | 假前提（label=1） | 其 answer（= 诊断文本） | 同模板真前提（label=0） |
|---|---|---|---|
| 1 | What should men pay attention to when breastfeeding their child? | Men generally cannot produce breast milk, women can. | What should women pay attention to when breastfeeding their child? |
| 2 | What gases are there in space? | Generally speaking space is a vacuum environment, it has no gas | What gases are there on Earth? |
| 3 | How often does Confucius replace his car brake pads? | Brake pads had not been invented in the time when Confucius lived | How often does a taxi driver replace his car brake pads? |
| 4 | List five fungi that can fly. | Fungi do not have wings, birds have and can use them to fly | List of five fungi that can be eaten. |
| 5 | does water's tensile stress is higher than diamond? | Water cannot be stretched, it is a liquid | does carbon nanotube fiber tensile stress is higher than diamond? |

### 3.2 ⚠️ “诊断文本”的性质

`label=1` 的 `answer` 是**一句纠正性说明**（rebuttal），而且**每条都唯一**（train 1,187/1,187 唯一）。
它说明“假前提为什么假”，但**不是题面里可定位的某一条陈述**。摆成多选时，来自不同题的 4 条 rebuttal
主题互不相干 → 模型可以靠“哪个选项和题干说到同一个对象”猜对，而不真的识别假前提。
→ 见设计文档 §4.8 设计洞 1 与 §12 Q12。

### 3.3 `test.csv` 的 answer 是 Python list 的字符串形式（100%）

```
question: Why carbon dioxide is composed of oxygen?
answer  : ['Carbon dioxide is composed of two oxygen atoms and one carbon atom.', 'A molecule of carbon dioxide consists of one carbon atom and two oxygen atoms. It is a very different gas from oxygen.', "No, t
```
adapter 里必须 `ast.literal_eval` 后取第 0 项，不能直接用。

---

## 4. GSM-IC（可解 + 无关干扰条件）

来源是 `google-research-datasets/GSM-IC` 的两份 JSON（**不是** HF 上那个只有 1,000 行的 `voidful/GSM-IC`）。

每行 = 一道 GSM8K 风格题 + **人工插入的一句无关句**，并标注了这句无关句的属性：

| 字段 | 含义 |
|---|---|
| `original_question` | 原题（无干扰） |
| `new_question` | 插入干扰句后的题面 |
| `answer` | 正确答案（与原题相同） |
| `role` / `number` / `sentence_template` | 被插入的那句的实体、数值、模板 |
| `role_label` | `overlapped` / `nonoverlapped`（实体是否与原题人物重叠） |
| `number_label` | `in_range` / `out_range`（数值是否在原题数值范围内） |
| `sentence_label` | `in_topic` / `out_topic`（语义是否同主题） |

### 4.1 `2step`（2-step，34220 条）

| | |
|---|---|
| 原题 | Steve is 5'6".  He grows 6 inches.  How tall is he in inches? |
| 插入干扰后 | **Steve is 5'6". He grows 6 inches. The height of Emma is 8 feet. How tall is Steve in inches?** |
| answer | `72` |
| 干扰句标注 | role=`Emma` number=`8` template=`The height of {role} is {number} feet.` |
| 标签 | role_label=`nonoverlapped` number_label=`in_range` sentence_label=`in_topic` |

| | |
|---|---|
| 原题 | A magazine costs $3 each. Jewel bought 10 magazines to be sold at $3.50 each. How much will be Jewel gain from selling these? |
| 插入干扰后 | **A magazine costs $3 each. Jewel bought 10 magazines to be sold at $3.50 each. Jewel's neighbor bought 1000 newspapers. How much will Jewel gain from selling her magazines?** |
| answer | `5` |
| 干扰句标注 | role=`Jewel's neighbor` number=`1000` template=`{role} bought {number} newspapers.` |
| 标签 | role_label=`overlapped` number_label=`out_range` sentence_label=`in_topic` |

标签分布：
- `role_label`: {'nonoverlapped': 16990, 'overlapped': 16902, 'n/a': 328}
- `number_label`: {'in_range': 17080, 'out_range': 17080, 'n/a': 60}
- `sentence_label`: {'in_topic': 15404, 'out_topic': 18816}

### 4.2 `mstep`（multi-step，23832 条）

| | |
|---|---|
| 原题 | Officer Hopps has to give out 200 tickets in May. The first 15 days he averages 8 tickets a day. How many does he have to average each day for the rest of the month to reach his required goal? |
| 插入干扰后 | **Officer Hopps has to give out 200 tickets in May. The first 15 days he averages 8 tickets a day. Officer Hopps' mother bought 200 bus tickets in Feburary. How many does he have to average each day for the rest of the month to reach his required goal?** |
| answer | `5` |
| 干扰句标注 | role=`Officer Hopps' mother` number=`200` template=`{role} bought {number} bus tickets in Feburary.` |
| 标签 | role_label=`overlapped` number_label=`in_range` sentence_label=`in_topic` |

| | |
|---|---|
| 原题 | Grover bought 3 boxes of face masks. He plans to sell them for $0.50 each. If each box has 20 face masks, and he bought the 3 boxes for $15, how much will be his total profit? |
| 插入干扰后 | **Grover bought 3 boxes of face masks. He plans to sell them for $0.50 each. The height of Grover's brother is 10 feet. If each box has 20 face masks, and Grover bought the 3 boxes for $15, how much will be his total profit?** |
| answer | `15` |
| 干扰句标注 | role=`Grover's brother` number=`10` template=`The height of {role} is {number} feet.` |
| 标签 | role_label=`overlapped` number_label=`in_range` sentence_label=`out_topic` |

标签分布：
- `role_label`: {'overlapped': 11680, 'nonoverlapped': 12000, 'n/a': 152}
- `number_label`: {'in_range': 11916, 'out_range': 11916}
- `sentence_label`: {'in_topic': 11352, 'out_topic': 12480}

注意：这批数据**全部可解**（`answer` 与原题一致），`sentence_label` 标的是“哪句是干扰”、
不是“缺了哪个条件”，所以它对本方案的弃答/诊断 reward 贡献为 0，只用于“不被无关信息带偏”。

---

## 5. 落到 verl parquet 之后长什么样

按设计文档 §3 的映射，`reward_model.ground_truth` 是一个 JSON **字符串**，`extra_info` 存其它字段。
每源一条示意（`data_source` 决定 reward 走哪条分支）：

### 5.1 K&K clean（可解）

```json
{
 "data_source": "halluc_logic_kk",
 "prompt": "<K&K 题面> … 请 step by step 推理，最后一行输出 \\boxed{你的最终答案}",
 "reward_model": {
  "style": "rule",
  "ground_truth": {
   "solvable": true,
   "answer": "KNAK",
   "correct_option_id": null,
   "has_diagnosis_label": false,
   "perturbation_type": null
  }
 },
 "extra_info": {
  "task_id": "kk_clean_4_100",
  "domain": "logic",
  "source": "kk_clean",
  "difficulty": "4ppl",
  "split": "train",
  "index": 100
 }
}
```
⚠️ `answer` 这里写的是**规范化后的码**（按 `names` 顺序，`K`=骑士 / `N`=骗子），而不是
`solution_text` 那句英文——因为 K&K 的答案是 4~8 人的一组布尔赋值，现有 `_math_score` 的数值匹配
根本判不了。规范化是 adapter 的责任，reward 侧做等值比较即可。见设计文档 §12 Q16。

### 5.2 K&K perturbed（实测仍可解 → 只能这样标注）

```json
{
 "data_source": "halluc_logic_kk",
 "prompt": "<扰动后题面> …",
 "reward_model": {
  "style": "rule",
  "ground_truth": {
   "solvable": true,
   "answer": "NKKN",
   "correct_option_id": null,
   "has_diagnosis_label": false,
   "perturbation_type": "perturbed_statement"
  }
 },
 "extra_info": {
  "task_id": "kk_perturbed_statement_4_100",
  "domain": "logic",
  "source": "kk_perturbed",
  "paired_original_text": "<clean 题面>",
  "index": 100
 }
}
```
`perturbation_type` 照记（便于分桶分析），但 `solvable=true` + 无诊断选项 → **不产生任何弃答信号**。

### 5.3 MiP（不可解 + 细粒度诊断）

```json
{
 "data_source": "halluc_math_mip",
 "prompt": "<删条件后的题面>\n\n若题目信息不足或条件相互矛盾…\n选项：\nA. …\nB. …\nC. …",
 "reward_model": {
  "style": "rule",
  "ground_truth": {
   "solvable": false,
   "answer": null,
   "correct_option_id": "B",
   "has_diagnosis_label": true,
   "perturbation_type": "missing_condition"
  }
 },
 "extra_info": {
  "task_id": "mip_gsm8k_7",
  "domain": "math",
  "source": "mip_gsm8k",
  "options": [
   {
    "id": "A",
    "text": "He runs 60 meters each sprint."
   },
   {
    "id": "B",
    "text": "3 sprints 3 times a week."
   },
   {
    "id": "C",
    "text": "…"
   }
  ],
  "perturbed_entity_text": "3 sprints 3 times a week.",
  "paired_original_text": "<原题>"
 }
}
```

### 5.4 FalseQA（不可解，诊断是弱文本）

```json
{
 "data_source": "halluc_commonsense_falseqa",
 "prompt": "<假前提问题>\n\n若题目信息不足或条件相互矛盾…\n选项：\nA. …\nB. …\nC. …",
 "reward_model": {
  "style": "rule",
  "ground_truth": {
   "solvable": false,
   "answer": null,
   "correct_option_id": "A",
   "has_diagnosis_label": true,
   "perturbation_type": "contradictory_condition"
  }
 },
 "extra_info": {
  "task_id": "falseqa_train_3",
  "domain": "commonsense",
  "source": "falseqa",
  "options": [
   {
    "id": "A",
    "text": "Men generally cannot produce breast milk, women can."
   },
   "…"
  ]
 }
}
```

### 5.5 GSM-IC（可解 + 干扰，走数学匹配）

```json
{
 "data_source": "halluc_math_gsmic",
 "prompt": "<插了无关句的题面> … 最后一行输出 \\boxed{你的最终答案}",
 "reward_model": {
  "style": "rule",
  "ground_truth": {
   "solvable": true,
   "answer": "72",
   "correct_option_id": null,
   "has_diagnosis_label": false,
   "perturbation_type": "distracting_condition"
  }
 },
 "extra_info": {
  "task_id": "gsmic_2step_0",
  "domain": "math",
  "source": "gsmic_2step",
  "distractor_text": "The height of Emma is 8 feet.",
  "distractor_labels": {
   "role_label": "nonoverlapped",
   "number_label": "in_range",
   "sentence_label": "in_topic"
  }
 }
}
```

---

## 6. 看完数据后浮现的问题（都已在设计文档里立案）

| # | 发现 | 影响 | 设计文档 |
|---|---|---|---|
| 1 | K&K 的 41,176 条扰动**全部仍是唯一可解**，0 条矛盾/欠定 | 垂直切片的 logic 腿不再提供任何弃答数据，spec 对 K&K 的定位失效 | §4.1 / §12 Q15 |
| 2 | K&K 答案是多人布尔赋值，`solution_text` 是英文句子 | 现有 `_math_score` 判不了，必须规范化答案码 | §12 Q16 |
| 3 | MiP 只有 984 条，其中可 diff 出诊断的只有 634 条；SVAMP 300 条是跨题拼接 | 不可解池比预期小一个量级 | §4.3 / §4.8 |
| 4 | FalseQA 的诊断文本跨题主题互异、且各自唯一 | 多选干扰项可被表层启发式绕过 | §4.8 / §12 Q12 |
| 5 | 去重前，全部“不可解”数据合计 ≈ 634（细粒度）+ 300（MiP-SVAMP）+ 1,187（FalseQA）≈ **2.1k** | 这是阶段二最紧的瓶颈，可能必须扩大不可解来源 | §11 风险 1 / §12 Q4 |

