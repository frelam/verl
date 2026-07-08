# 混合数学数据集处理脚本

> **注意**: 本目录中的脚本与 `Rl_Specilist/math/dataset/` 中的脚本为镜像关系。
> `Rl_Specilist/math/dataset/` 为规范源（canonical source），本目录为方便用户从
> `examples/` 入口访问而提供。如需修改脚本，应同步更新两边。

本脚本用于混合多个数学数据集，生成可直接用于verl框架训练的数据集。

## 包含的数据集

1. **OpenR1-Math-220k** - 高质量数学推理数据集（约22万条）
   - 来源: `open-r1/OpenR1-Math-220k`
   - 包含DeepSeek R1生成的推理轨迹
   
2. **GSM8K** - 经典小学数学问题数据集
   - 来源: `openai/gsm8k`
   - 训练集: 7,473条，测试集: 1,319条
   
3. **MATH-lighteval** - 竞赛级数学问题数据集
   - 来源: `DigitalLearningGmbH/MATH-lighteval`
   - 训练集: 7,500条，测试集: 5,000条
   
4. **NuminaMath-CoT** - 数学推理数据集
   - 来源: `AI-MO/NuminaMath-CoT`
   - 默认采样10,000条

## 使用方法

### 基本用法

```bash
python examples/data_preprocess/mixed_math_dataset.py \
    --local_save_dir ~/data/mixed_math_dataset
```

### 高级选项

```bash
python examples/data_preprocess/mixed_math_dataset.py \
    --local_save_dir ~/data/mixed_math_dataset \
    --openr1_max_samples 50000 \
    --numina_max_samples 10000 \
    --skip_numina
```

### 参数说明

- `--local_save_dir`: 数据集保存目录（默认: `~/data/mixed_math_dataset`）
- `--hdfs_dir`: HDFS目录（可选）
- `--openr1_math_path`: OpenR1-Math-220k本地路径（可选）
- `--gsm8k_path`: GSM8K本地路径（可选）
- `--math_path`: MATH-lighteval本地路径（可选）
- `--numina_math_path`: NuminaMath-CoT本地路径（可选）
- `--openr1_max_samples`: OpenR1-Math-220k最大样本数（用于调试）
- `--numina_max_samples`: NuminaMath-CoT最大样本数（默认10000）
- `--skip_openr1`: 跳过OpenR1-Math-220k数据集
- `--skip_gsm8k`: 跳过GSM8K数据集
- `--skip_math`: 跳过MATH-lighteval数据集
- `--skip_numina`: 跳过NuminaMath-CoT数据集

## 输出文件

脚本会在指定目录生成以下文件：

- `train.parquet` - 混合后的训练集
- `test.parquet` - 混合后的测试集
- `train_example.json` - 训练集示例
- `test_example.json` - 测试集示例

## 数据格式

每条数据包含以下字段：

```json
{
  "data_source": "数据集来源",
  "prompt": [
    {
      "role": "user",
      "content": "问题内容"
    }
  ],
  "ability": "math",
  "reward_model": {
    "style": "rule",
    "ground_truth": "正确答案"
  },
  "extra_info": {
    "split": "train/test",
    "index": 索引,
    "answer": "原始答案",
    "question": "原始问题"
  }
}
```

## compute_score函数

`verl/utils/reward_score/mixed_math.py` 提供了统一的评分函数，支持所有混合数据集：

```python
from verl.utils.reward_score import default_compute_score

score = default_compute_score(
    data_source="open-r1/OpenR1-Math-220k",
    solution_str="模型输出的解答",
    ground_truth="正确答案"
)
```

### 支持的数据源

- `open-r1/OpenR1-Math-220k`
- `openai/gsm8k`
- `DigitalLearningGmbH/MATH-lighteval`
- `AI-MO/NuminaMath-CoT`

### 获取详细信息

```python
from verl.utils.reward_score.mixed_math import compute_score_with_details

result = compute_score_with_details(
    data_source="openai/gsm8k",
    solution_str="Let's think step by step... #### 42",
    ground_truth="42"
)

# 返回:
# {
#     "score": 1.0,
#     "data_source": "openai/gsm8k",
#     "ground_truth": "42",
#     "extracted_answer": "42",
#     "is_correct": True
# }
```

## 注意事项

1. OpenR1-Math-220k数据集较大（约8.44GB），首次下载需要较长时间
2. 可以使用 `--openr1_max_samples` 参数限制样本数量进行快速测试
3. 数据集会自动打乱顺序（seed=42）
4. 测试集只包含GSM8K和MATH-lighteval，因为这两个数据集有标准测试集

## 示例：快速测试

```bash
python examples/data_preprocess/mixed_math_dataset.py \
    --local_save_dir ~/data/mixed_math_test \
    --openr1_max_samples 1000 \
    --numina_max_samples 1000
```

这会创建一个小型数据集用于快速测试流程。
