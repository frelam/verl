# 数学数据集生成脚本

> **注意**: 本目录为规范源（canonical source）。`examples/data_preprocess/` 中的同名脚本
> 为本目录的镜像，为方便用户从 `examples/` 入口访问。修改脚本时应同步更新两边。

本目录包含混合数学数据集的生成和处理脚本，支持多种数据源、prompt 类型分类和长度控制。

---

## 目录结构

```
dataset/
├── README.md                                      ← 本文件
├── __init__.py
│
├── mixed_math_dataset.py                          # 混合数据集主脚本（4 数据源）
├── create_mixed_data.py                           # 从已处理数据文件夹创建混合数据集
├── mixed_reasoning_dataset_with_prompt_types.py   # 带 prompt 类型分类的混合推理数据集
├── math_dataset_with_prompt_types.py              # 带 prompt 类型分类的数学数据集
│
├── mixed_gdpo_reward.py                           # GDPO 多维混合奖励函数（Stage 2）
├── mixed_reasoning_reward.py                      # 混合推理奖励函数（Stage 5）
│
├── test_mixed_math_score.py                       # 奖励函数单元测试
├── run_mixed_math.sh                              # 一键运行脚本
└── README_mixed_math.md                           # 详细使用指南
```

---

## 数据源总览

| 数据集 | 来源 | 规模 | 说明 |
|--------|------|------|------|
| OpenR1-Math-220k | `open-r1/OpenR1-Math-220k` | ~220K | DeepSeek R1 推理轨迹 |
| GSM8K | `openai/gsm8k` | 7.5K train + 1.3K test | 小学数学应用题 |
| MATH-lighteval | `DigitalLearningGmbH/MATH-lighteval` | 7.5K train + 5K test | 竞赛级数学题 |
| NuminaMath-CoT | `AI-MO/NuminaMath-CoT` | ~860K (默认采样 10K) | 数学推理链 |

---

## 脚本说明

### 1. `mixed_math_dataset.py` — 混合数据集主脚本

生成包含 4 个数据源的混合数学数据集，输出 `train.parquet` 和 `test.parquet`。

```bash
python Rl_Specilist/math/dataset/mixed_math_dataset.py \
    --local_save_dir ~/data/mixed_math

# 快速测试（限制样本数）
python Rl_Specilist/math/dataset/mixed_math_dataset.py \
    --local_save_dir ~/data/mixed_math_test \
    --openr1_max_samples 1000 \
    --numina_max_samples 500
```

对应训练阶段：**Stage 2 — 混合数学数据集**。

### 2. `create_mixed_data.py` — 从文件夹创建混合数据集

从多个已预处理的数据文件夹聚合为统一数据集，支持按比例采样和 `max_occurrences` 限制。

```bash
python Rl_Specilist/math/dataset/create_mixed_data.py \
    --data_dirs /path/to/math_data /path/to/logic_data \
    --output_dir ~/data/mixed_reasoning \
    --ratios 0.6 0.4
```

### 3. `mixed_reasoning_dataset_with_prompt_types.py` — 带 prompt 类型的推理数据集

在 mix 基础上增加 `prompt_type` 标签，区分不同类型的推理任务（math、logic、code 等），支持按照 prompt 类型设置不同的长度惩罚和 max_effort 标记。

```bash
python Rl_Specilist/math/dataset/mixed_reasoning_dataset_with_prompt_types.py \
    --local_save_dir ~/data/mixed_reasoning
```

对应训练阶段：**Stage 5 — 混合逻辑与数学**。

### 4. `math_dataset_with_prompt_types.py` — 带 prompt 类型的数学数据集

类似上述脚本，但仅针对数学数据集添加 prompt 类型标签。

### 5. `mixed_gdpo_reward.py` — GDPO 多维奖励函数

实现 GDPO 的复合奖励计算，支持 `accuracy_reward`、`format_reward`、`wait_reward`、`len_reward` 等多维度。

```python
from Rl_Specilist.math.dataset.mixed_gdpo_reward import compute_score

result = compute_score(
    data_source="openai/gsm8k",
    solution_str="...",
    ground_truth="42",
)
# 返回: {"score": 0.9, "accuracy_reward": 0.8, "format_reward": 1.0, ...}
```

对应训练阶段：**Stage 2-4**。

### 6. `mixed_reasoning_reward.py` — 混合推理奖励函数

支持多数据源（math + logic + code）的统一评分，按 `data_source` 路由到不同评分逻辑。

对应训练阶段：**Stage 5**。

### 7. `test_mixed_math_score.py` — 奖励函数测试

对 `compute_score` 函数进行单元测试，覆盖各种答案格式和边界情况。

```bash
python Rl_Specilist/math/dataset/test_mixed_math_score.py
```

### 8. `run_mixed_math.sh` — 一键运行

```bash
bash Rl_Specilist/math/dataset/run_mixed_math.sh
```

---

## 数据格式

生成的数据集每条记录包含以下字段：

```json
{
  "data_source": "openai/gsm8k",
  "prompt": [
    {"role": "user", "content": "Janet 的鸭子每天下 3 个蛋..."}
  ],
  "ability": "math",
  "reward_model": {
    "style": "rule",
    "ground_truth": "42"
  },
  "extra_info": {
    "split": "train",
    "index": 0,
    "answer": "42",
    "question": "原始问题文本...",
    "prompt_type": "math_word_problem",
    "max_effort": false
  }
}
```

### extra_info 重要字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `split` | str | `"train"` 或 `"test"` |
| `index` | int | 数据集内索引 |
| `answer` | str | 原始答案（用于 ground truth 校验） |
| `question` | str | 原始问题文本 |
| `prompt_type` | str | prompt 类型标签（如 `"math_word_problem"`、`"logic_puzzle"`） |
| `max_effort` | bool | 是否标记为 max_effort（豁免长度惩罚） |

---

## 与训练阶段的对应关系

| 数据脚本 | 奖励脚本 | 训练阶段 |
|----------|----------|----------|
| GSM8K 原始数据 | `stage 1/gsm8k_reward.py` | Stage 1: GSM8K 基础 DAPO |
| `mixed_math_dataset.py` | `mixed_gdpo_reward.py` | Stage 2: 混合数学 + GDPO |
| `mixed_math_dataset.py` | `mixed_gdpo_reward.py` (含 len_penalty) | Stage 3: + 长度惩罚 |
| `mixed_math_dataset.py` | `mixed_gdpo_reward.py` (含 max_effort) | Stage 4: + max_effort 豁免 |
| `mixed_reasoning_dataset_with_prompt_types.py` | `mixed_reasoning_reward.py` | Stage 5: + 逻辑推理 |

---

## 相关文件

- **verl 框架评分函数**: `verl/utils/reward_score/mixed_math.py` — 统一 `default_compute_score`
- **训练配方**: 各 stage 目录下的 `*_train_recipe.sh`
- **详细使用指南**: `README_mixed_math.md`
