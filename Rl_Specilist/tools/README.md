# Tools — 通用工具集

RL Specialist 训练辅助工具集，包括权重对比、LoRA 提取等功能。

---

## 一、权重对比工具 (`compare_weights.py`)

比较两份 SafeTensors 权重文件，按 L2 差异降序输出每个权重的变化。

### 用法

```bash
python Rl_Specilist/tools/compare_weights.py \
    --original /path/to/original_model/model.safetensors \
    --trained /path/to/trained_model/model.safetensors \
    --topk 20
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `--original` | 原始模型权重文件路径（必填） |
| `--trained` | 训练后模型权重文件路径（必填） |
| `--topk` | 仅显示 top-K 差异最大的权重（可选） |

### 输出说明

输出包含以下列：
- **Rank**: 排名（按绝对 L2 差异降序）
- **Weight Name**: 权重名称
- **L2 Diff**: 绝对 L2 范数差异
- **Rel L2 Diff**: 相对 L2 范数差异
- **Shape**: 权重张量形状

### 应用场景

- 对比 LoRA 微调前后的权重变化
- 检查 SFT/RL 训练对模型各层的影响程度
- 调试模型训练中权重更新的异常（某层变化过大/过小）

### 示例输出

参见 `compare_weight.result`。

---

## 二、LoRA 提取工具 (`extract_lora_from_ckpt.py`)

从 FSDP checkpoint 中提取 LoRA 权重，保存为 PEFT 独立适配器格式。

> 脚本位置: `scripts/extract_lora_from_ckpt.py`

### 背景

verl 框架在 LoRA 训练时，每个 rank 会保存自己的 checkpoint（`model_world_size_*_rank_*.pt`）。要将 LoRA 权重用于其他框架（如 vLLM、Hugging Face PEFT），需要先将 LoRA 参数从完整 checkpoint 中提取出来，合并为独立的适配器格式。

### 用法

```bash
# 从 FSDP checkpoint 目录提取
python scripts/extract_lora_from_ckpt.py \
    --ckpt_dir /path/to/checkpoint/global_step_100

# 指定输出目录
python scripts/extract_lora_from_ckpt.py \
    --ckpt_dir /path/to/checkpoint/global_step_100 \
    --output_dir ./my_lora_adapter

# 手动指定 lora_alpha（如果 lora_train_meta.json 中未设置）
python scripts/extract_lora_from_ckpt.py \
    --ckpt_dir /path/to/checkpoint/global_step_100 \
    --lora_alpha 16
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `--ckpt_dir` | FSDP checkpoint 目录路径（必填） |
| `--output_dir` | 输出目录，默认为 `<ckpt_dir>_lora_adapter` |
| `--lora_alpha` | LoRA alpha 值，若 `lora_train_meta.json` 中未设置则使用此值（默认 16） |

### 支持的 Checkpoint 格式

1. **Safetensors 格式**（推荐）: `ckpt_dir/huggingface/*.safetensors`（支持分片）
2. **PT 格式**: `ckpt_dir/model_world_size_*_rank_*.pt`（torch.save 格式）

### 输出文件

```
<output_dir>/
├── adapter_config.json    # PEFT 适配器配置
├── adapter_model.safetensors   # 提取的 LoRA 权重
└── README.md              # 使用说明
```

### 使用提取的 LoRA 适配器

```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM

# 加载基座模型
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# 加载 LoRA 适配器
lora_model = PeftModel.from_pretrained(base_model, "./my_lora_adapter")

# 推理
output = lora_model.generate(...)
```

### 注意事项

- 需要安装 `peft` 和 `safetensors` 库
- 对于 FSDP 多 rank 的 `.pt` 格式 checkpoint，取 rank-0 即可，LoRA 参数在所有 rank 上是完全相同的（因为 FSDP 只 shard base model 参数）
- Safetensors 分片格式会自动合并所有 shard
