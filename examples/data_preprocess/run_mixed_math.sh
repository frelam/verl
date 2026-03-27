#!/bin/bash
# 混合数学数据集处理脚本示例

# 基本用法 - 下载并处理所有数据集
# python examples/data_preprocess/mixed_math_dataset.py \
#     --local_save_dir ~/data/mixed_math_dataset

# 快速测试 - 使用较少的样本
python examples/data_preprocess/mixed_math_dataset.py \
    --local_save_dir ~/data/mixed_math_test \
    --openr1_max_samples 1000 \
    --numina_max_samples 1000

# 完整处理 - 使用所有数据
# python examples/data_preprocess/mixed_math_dataset.py \
#     --local_save_dir ~/data/mixed_math_full \
#     --numina_max_samples 10000

# 使用本地数据集
# python examples/data_preprocess/mixed_math_dataset.py \
#     --local_save_dir ~/data/mixed_math_local \
#     --openr1_math_path /path/to/openr1_math \
#     --gsm8k_path /path/to/gsm8k \
#     --math_path /path/to/math

# 跳过某些数据集
# python examples/data_preprocess/mixed_math_dataset.py \
#     --local_save_dir ~/data/mixed_math_subset \
#     --skip_numina
