# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
pass@k difficulty pre-screening (DESIGN.md section 4).

Runs the base model over a pool parquet with vLLM offline batch generation,
scores each rollout with the SAME reward dispatcher used in training
(examples/reasoning_rl/reward/compute_score.py), then:

  - writes the observed pass rate back to ``extra_info.pass_rate`` (difficulty
    stratification signal for training / hard replay);
  - drops all-solved rows (pass_rate == 1, zero gradient for GRPO groups);
  - keeps ``--keep_zero_frac`` of all-wrong rows for exploration (these are
    exactly what the hard-replay mechanism re-rolls later);
  - drops every row in between nothing — pass_rate in (0, 1) rows all stay.

Suggested budgets (DESIGN.md section 4): pass@8 temp=1.0 for math/logic,
pass@4 for code (execution-expensive). When the budget is short, subsample
the pool first with --subsample_frac.

Usage:
    python scripts/difficulty_tag.py \
        --input ~/data/reasoning_rl/logic/train_logic.parquet \
        --output ~/data/reasoning_rl/logic/train_logic_tagged.parquet \
        --model Qwen/Qwen3-4B --k 8
"""

import argparse
import importlib.util
import json
import os
import random
import sys


def _load_reward_fn():
    """Load reward/compute_score.py as a standalone module (no verl import needed
    at module top level of the reward file)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reward", "compute_score.py")
    spec = importlib.util.spec_from_file_location("reasoning_rl_reward", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Pool parquet to tag.")
    parser.add_argument("--output", required=True, help="Tagged parquet out path.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--k", type=int, default=8, help="Rollouts per prompt (pass@k).")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--max_prompt_length", type=int, default=4096)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument(
        "--keep_zero_frac", type=float, default=0.075, help="Fraction of all-wrong rows kept for exploration."
    )
    parser.add_argument(
        "--drop_solved", action="store_true", default=True, help="Drop pass_rate==1 rows (zero-gradient)."
    )
    parser.add_argument("--keep_solved", dest="drop_solved", action="store_false")
    parser.add_argument(
        "--subsample_frac", type=float, default=1.0, help="Tag only a random fraction of the pool (budget control)."
    )
    parser.add_argument(
        "--sandbox_fusion_url",
        default=None,
        help="Required to tag code pools (prime_code local execution is too slow/unsafe).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stats_out", default=None)
    args = parser.parse_args()

    import datasets
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    compute_score = _load_reward_fn()
    rng = random.Random(args.seed)

    ds = datasets.load_dataset("parquet", data_files=args.input, split="train")
    rows = ds.to_list()
    print(f"[tag] loaded {len(rows)} rows from {args.input}")
    if args.subsample_frac < 1.0:
        rows = rng.sample(rows, max(1, int(len(rows) * args.subsample_frac)))
        print(f"[tag] subsampled to {len(rows)} rows (frac={args.subsample_frac})")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = []
    for r in rows:
        text = tokenizer.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True)
        prompts.append(text[: args.max_prompt_length * 4])  # rough char cap; token filter is dataset-side

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_prompt_length + args.max_tokens,
    )
    sampling = SamplingParams(n=args.k, temperature=args.temperature, top_p=1.0, max_tokens=args.max_tokens)
    print(f"[tag] generating pass@{args.k} for {len(prompts)} prompts ...")
    outputs = llm.generate(prompts, sampling)

    kept, stats = [], {"all_wrong": 0, "all_right": 0, "mixed": 0, "zero_kept": 0, "verify_errors": 0}
    for row, out in zip(rows, outputs, strict=True):
        passes = 0
        for completion in out.outputs:
            try:
                res = compute_score(
                    data_source=row["data_source"],
                    solution_str=completion.text,
                    ground_truth=row["reward_model"]["ground_truth"],
                    extra_info=row.get("extra_info"),
                    sandbox_fusion_url=args.sandbox_fusion_url,
                )
                passes += float(res["score"] if isinstance(res, dict) else res) > 0.0
            except Exception as e:
                stats["verify_errors"] += 1
                if stats["verify_errors"] <= 5:
                    print(f"[tag] verify error ({row['data_source']}): {e}")
        pass_rate = passes / max(1, len(out.outputs))
        row["extra_info"]["pass_rate"] = pass_rate
        if pass_rate >= 1.0:
            stats["all_right"] += 1
            if args.drop_solved:
                continue
        elif pass_rate <= 0.0:
            stats["all_wrong"] += 1
            if rng.random() >= args.keep_zero_frac:
                continue
            stats["zero_kept"] += 1
        else:
            stats["mixed"] += 1
        kept.append(row)

    for i, r in enumerate(kept):
        r["extra_info"]["index"] = i

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    datasets.Dataset.from_list(kept).to_parquet(args.output)
    stats["kept"] = len(kept)
    stats["input"] = len(rows)
    print(f"[done] kept {len(kept)} / {len(rows)} -> {args.output}")
    print(f"[done] stats: {json.dumps(stats)}")
    if args.stats_out:
        with open(args.stats_out, "w") as f:
            json.dump(stats, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
