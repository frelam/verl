#!/usr/bin/env python3
"""
从 FSDP checkpoint 中提取 LoRA 权重，保存为 PEFT 独立适配器格式。

用法:
    # 从 FSDP checkpoint 目录提取
    python scripts/extract_lora_from_ckpt.py --ckpt_dir /path/to/global_step_100

    # 指定输出目录
    python scripts/extract_lora_from_ckpt.py --ckpt_dir /path/to/global_step_100 --output_dir ./my_lora_adapter

    # 手动指定 lora_alpha（如果 lora_train_meta.json 中未设置）
    python scripts/extract_lora_from_ckpt.py --ckpt_dir /path/to/global_step_100 --lora_alpha 16
"""

import argparse
import json
import os
import warnings
from collections import OrderedDict

import torch
from safetensors.torch import load_file, save_file


def collect_safetensors_paths(huggingface_dir: str):
    """收集 huggingface/ 目录下所有的 safetensors 文件路径。"""
    paths = []
    for fname in sorted(os.listdir(huggingface_dir)):
        if fname.endswith(".safetensors"):
            paths.append(os.path.join(huggingface_dir, fname))
    return paths


def load_state_dict(ckpt_dir: str):
    """
    从 checkpoint 目录加载完整的 state_dict。

    支持两种格式：
    1. model_world_size_*_rank_*.pt (torch.save 格式)
    2. huggingface/*.safetensors (safetensors 格式, 支持分片)
    """
    # 优先尝试 safetensors 格式
    hf_dir = os.path.join(ckpt_dir, "huggingface")
    if os.path.isdir(hf_dir):
        safetensors_paths = collect_safetensors_paths(hf_dir)
        if safetensors_paths:
            print(f"从 huggingface/ 加载 {len(safetensors_paths)} 个 safetensors 文件...")
            state_dict = {}
            for path in safetensors_paths:
                shard = load_file(path)
                state_dict.update(shard)
                print(f"  加载: {os.path.basename(path)} ({len(shard)} 个参数)")
            return state_dict

    # 退回到 .pt 文件
    pt_files = [f for f in os.listdir(ckpt_dir) if f.startswith("model_world_size_") and f.endswith(".pt")]
    if pt_files:
        path = os.path.join(ckpt_dir, pt_files[0])
        print(f"从 {pt_files[0]} 加载...")
        return torch.load(path, map_location="cpu", weights_only=False)

    raise FileNotFoundError(
        f"在 {ckpt_dir} 中未找到 model_world_size_*.pt 或 huggingface/*.safetensors 文件"
    )


def load_lora_metadata(ckpt_dir: str) -> dict | None:
    """尝试加载 lora_train_meta.json (由 SFT trainer 保存)。"""
    meta_path = os.path.join(ckpt_dir, "lora_train_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                return json.load(f)
        except Exception as e:
            warnings.warn(f"读取 lora_train_meta.json 失败: {e}")
    return None


def clean_lora_key(key: str) -> str:
    """清理 key 中的 FSDP 和 PEFT 前缀，转换为标准 PEFT 适配器格式。"""
    # 移除 FSDP 包裹前缀
    key = key.replace("_fsdp_wrapped_module.", "")
    # PEFT adapter 保存时使用 base_model.model. 前缀，需要保留
    # 但有些 key 可能没有这个前缀，保持不变
    return key


def extract_lora(state_dict: dict, lora_alpha: int | None = None, output_dir: str = "lora_adapter"):
    """
    从完整 state_dict 中提取 LoRA 权重并保存为 PEFT 适配器格式。

    Args:
        state_dict: 完整的模型 state_dict (含基础模型 + LoRA 权重)
        lora_alpha: LoRA alpha 值。如果为 None，尝试从元数据读取。
        output_dir: 输出目录
    """
    # 筛选 LoRA 相关的 key
    lora_keys = [k for k in state_dict if "lora_" in k]

    if not lora_keys:
        print("错误: 未找到任何 LoRA 权重 (key 中不包含 'lora_')")
        print("请确认 checkpoint 确实包含 LoRA 适配器。")
        return None

    print(f"找到 {len(lora_keys)} 个 LoRA 参数")

    # 清理 key 并收集 LoRA 参数
    lora_params = OrderedDict()
    target_modules = set()

    for key in lora_keys:
        clean_key = clean_lora_key(key)
        # PEFT adapter_model.safetensors 使用 .weight 而不是 .default.weight
        peft_key = clean_key.replace(".default.weight", ".weight")
        target_modules.add(peft_key.split(".")[-3])
        lora_params[peft_key] = state_dict[key]

    # 推断 LoRA rank（从最后一个 lora_B 或 lora_A 的形状）
    last_key = list(lora_params.keys())[-1]
    inferred_rank = min(lora_params[last_key].shape[0], lora_params[last_key].shape[1])
    print(f"从权重视图形状推断 LoRA rank = {inferred_rank}")

    # 打印 LoRA 权重示例，方便校验
    print(f"\nLoRA 参数示例 (前 3 个):")
    for i, k in enumerate(list(lora_params.keys())[:3]):
        print(f"  {k}: shape {lora_params[k].shape}")

    # 准备 LoRA 配置
    lora_meta = load_lora_metadata(args.ckpt_dir)

    lora_rank = inferred_rank
    lora_alpha_val = 0
    task_type = None

    if lora_meta is not None:
        meta_rank = lora_meta.get("r")
        if meta_rank is not None and meta_rank > 0:
            if meta_rank != inferred_rank:
                warnings.warn(
                    f"LoRA rank 元数据 ({meta_rank}) 与权重推断 ({inferred_rank}) 不匹配，"
                    f"使用元数据值 {meta_rank}"
                )
            lora_rank = meta_rank

        meta_alpha = lora_meta.get("lora_alpha")
        if meta_alpha is not None:
            lora_alpha_val = meta_alpha

        meta_task_type = lora_meta.get("task_type")
        if meta_task_type is not None:
            task_type = meta_task_type

    if lora_alpha is not None:
        lora_alpha_val = lora_alpha

    if lora_alpha_val == 0:
        warnings.warn(
            "lora_alpha 为 0。请通过 --lora_alpha 手动指定正确的值，"
            "否则加载适配器时 LoRA 缩放为 0 将导致模型输出为 0。"
        )

    # 构建 PEFT 配置
    try:
        from peft import LoraConfig
    except ImportError:
        print("警告: 未安装 peft 库，将手动构建 adapter_config.json")
        peft_config = {
            "r": lora_rank,
            "lora_alpha": lora_alpha_val,
            "target_modules": sorted(target_modules),
            "task_type": task_type or "CAUSAL_LM",
            "peft_type": "LORA",
            "bias": "none",
            "lora_dropout": 0.0,
            "init_lora_weights": True,
            "use_rslora": False,
            "use_dora": False,
            "layers_pattern": None,
            "layers_to_transform": None,
            "modules_to_save": None,
            "fan_in_fan_out": False,
            "enable_lora": None,
            "megatron_config": None,
            "peft_version": None,
        }
    else:
        peft_dict = {
            "r": lora_rank,
            "lora_alpha": lora_alpha_val,
            "target_modules": sorted(target_modules),
        }
        if task_type is not None:
            peft_dict["task_type"] = task_type
        peft_config_obj = LoraConfig(**peft_dict)
        peft_config = peft_config_obj.to_dict()
        peft_config["task_type"] = (
            peft_config["task_type"].value
            if hasattr(peft_config["task_type"], "value")
            else (peft_config["task_type"] or None)
        )
        peft_config["peft_type"] = (
            peft_config["peft_type"].value
            if hasattr(peft_config["peft_type"], "value")
            else (peft_config["peft_type"] or None)
        )
        peft_config["target_modules"] = sorted(peft_config["target_modules"])

    # 保存
    os.makedirs(output_dir, exist_ok=True)

    # adapter_config.json
    config_path = os.path.join(output_dir, "adapter_config.json")
    with open(config_path, "w") as f:
        json.dump(peft_config, f, ensure_ascii=False, indent=4)
    print(f"\n保存 adapter_config.json -> {config_path}")

    # adapter_model.safetensors
    model_path = os.path.join(output_dir, "adapter_model.safetensors")
    save_file(lora_params, model_path)
    print(f"保存 adapter_model.safetensors -> {model_path} ({len(lora_params)} 个参数)")

    print(f"\nLoRA 适配器已保存到: {os.path.abspath(output_dir)}")
    print(f"  target_modules: {sorted(target_modules)}")
    print(f"  rank: {lora_rank}")
    print(f"  lora_alpha: {lora_alpha_val}")
    if task_type:
        print(f"  task_type: {task_type}")

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 FSDP checkpoint 中提取 LoRA 适配器权重")
    parser.add_argument(
        "--ckpt_dir",
        required=True,
        help="checkpoint 目录路径 (包含 model_world_size_*.pt 或 huggingface/ 子目录)",
    )
    parser.add_argument(
        "--output_dir",
        default="lora_adapter",
        help="LoRA 适配器输出目录 (默认: ./lora_adapter)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=None,
        help="手动指定 lora_alpha 值 (覆盖 lora_train_meta.json 中的值)",
    )
    args = parser.parse_args()

    state_dict = load_state_dict(args.ckpt_dir)
    extract_lora(state_dict, lora_alpha=args.lora_alpha, output_dir=args.output_dir)