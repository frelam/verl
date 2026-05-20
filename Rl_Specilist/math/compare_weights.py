import argparse
import torch
from safetensors.torch import load_file


def compare_weights(original_path, trained_path, topk=None):
    """
    比较两份 SafeTensors 权重文件，按 L2 差异降序输出每个权重的变化。
    """
    print(f"Loading original weights from: {original_path}")
    orig = load_file(original_path)
    print(f"Loading trained weights from: {trained_path}")
    trained = load_file(trained_path)

    # 找出公共键和非公共键
    orig_keys = set(orig.keys())
    trained_keys = set(trained.keys())
    common_keys = orig_keys & trained_keys
    only_orig = orig_keys - trained_keys
    only_trained = trained_keys - orig_keys

    if only_orig:
        print(f"Keys only in original file: {only_orig}")
    if only_trained:
        print(f"Keys only in trained file: {only_trained}")
    if not common_keys:
        print("No common keys found. Exiting.")
        return

    results = []
    for key in sorted(common_keys):
        w1 = orig[key]
        w2 = trained[key]
        if w1.shape != w2.shape:
            print(f"Skipping '{key}' due to shape mismatch: {w1.shape} vs {w2.shape}")
            continue

        diff = w2 - w1
        l2 = torch.norm(diff).item()                     # 绝对 L2 差异
        norm_orig = torch.norm(w1).item()
        rel_l2 = l2 / (norm_orig + 1e-8)                 # 相对 L2 差异
        results.append((key, l2, rel_l2, tuple(w1.shape)))

    # 按绝对 L2 差异降序排序
    results.sort(key=lambda x: x[1], reverse=True)

    # 可选截取 top-k
    if topk is not None:
        results = results[:topk]

    # 打印表头
    print(f"\n{'Rank':<6} {'Weight Name':<60} {'L2 Diff':<14} {'Rel L2 Diff':<14} {'Shape'}")
    print("-" * 110)
    for rank, (name, l2, rel_l2, shape) in enumerate(results, 1):
        print(f"{rank:<6} {name:<60} {l2:<14.6f} {rel_l2:<14.6e} {str(shape)}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare two SafeTensors weight files and show largest weight changes (L2 norm)."
    )
    parser.add_argument("--original", required=True, help="Path to the original .safetensors file")
    parser.add_argument("--trained", required=True, help="Path to the trained .safetensors file")
    parser.add_argument("--topk", type=int, default=None, help="Only show the top K largest differences")
    args = parser.parse_args()

    compare_weights(args.original, args.trained, args.topk)
