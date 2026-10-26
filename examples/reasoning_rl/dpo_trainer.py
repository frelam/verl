#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Standalone DPO trainer for the reasoning-efficiency pairs (dataset C).

Why a standalone trainer: this repository's verl stack has no DPO trainer, and
the preference data here is a plain parquet of ``(prompt, chosen, rejected)``
rows -- no rollout, reward model or critic is involved.  The recipe mirrors
``run_qwen3_4b_hallucination_stage3_fully_async.sh`` where it applies: FSDP2-era
full-parameter training (FSDP1 is used so the script needs neither verl nor
Ray), bf16 mixed precision, gradient checkpointing, the same prompt/response
length curriculum, AdamW with cosine warmup, and the same checkpoint layout
(a HuggingFace directory that vLLM can serve directly).

Objective (Rafailov et al., reference-implementation form)::

    margin = beta * [ (logpi_c - logpi_r) - (logref_c - logref_r) ]
    loss   = -logsigmoid(margin).mean()

Only completion tokens enter the log-probabilities; prompt tokens are masked.
``chosen`` is dataset B (the judge-condensed chain of thought), ``rejected`` is
dataset A (the original RL rollout); both were verified to score 1 with the RL
reward, so the only signal is "same answer, less redundant reasoning".

Launch (single node; 8 NPUs or GPUs)::

    torchrun --nproc_per_node=8 examples/reasoning_rl/dpo_trainer.py \
        --model <rl_hf_checkpoint> \
        --train_files ~/data/reasoning_rl/dpo/dpo_train.parquet \
        --val_files ~/data/reasoning_rl/dpo/dpo_val.parquet \
        --output_dir ~/ckpt/dpo_qwen3_4b \
        --beta 0.1 --lr 1e-6 --epochs 1

Memory notes: the policy and the (frozen) reference policy are both FSDP
sharded, so a 4B model costs roughly one model shard + one optimizer state set
per rank.  The vocabulary projection is the peak consumer; ``--logprob_chunk_size``
bounds the float32 softmax chunk, not the bf16 logits themselves.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from collections import Counter

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler

try:  # registers the "npu" device on Ascend images; no-op elsewhere
    import torch_npu  # noqa: F401
except ImportError:
    pass

logger = logging.getLogger("dpo_trainer")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="RL-finished HuggingFace checkpoint (policy init)")
    parser.add_argument("--ref_model", default="", help="reference policy (default = --model)")
    parser.add_argument("--tokenizer", default="", help="tokenizer path (default = --model)")
    parser.add_argument("--train_files", required=True)
    parser.add_argument("--val_files", default="")
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr_scheduler", choices=("cosine", "linear", "constant"), default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=1,
        help="preference pairs per rank per micro-batch",
    )
    parser.add_argument("--grad_accum", type=int, default=8)

    parser.add_argument("--max_prompt_length", type=int, default=4096)
    parser.add_argument("--max_response_length", type=int, default=8192)
    parser.add_argument("--max_total_length", type=int, default=0, help="0 = max_prompt + max_response")
    parser.add_argument("--append_eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--length_normalize",
        action="store_true",
        help="average the log-prob over completion tokens instead of summing "
        "(removes DPO's implicit length preference; leave off when the goal is conciseness)",
    )
    parser.add_argument("--logprob_chunk_size", type=int, default=1024)

    parser.add_argument("--ref_free", action="store_true", help="drop the reference policy (reference-free DPO)")
    parser.add_argument("--fsdp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fsdp_min_num_params", type=int, default=int(1e8), help="auto-wrap threshold per FSDP unit")
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn_implementation", default="sdpa", choices=("sdpa", "flash_attention_2", "eager"))
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust_remote_code", action="store_true")

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=100, help="0 disables periodic validation")
    parser.add_argument("--save_steps", type=int, default=200, help="0 disables periodic checkpoints")
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--wandb_project", default="")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--seed", type=int, default=42)
    return parser


# ---------------------------------------------------------------------------
# distributed / device setup
# ---------------------------------------------------------------------------


def _device_type() -> str:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _init_distributed():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_type = _device_type()
    backend = {"npu": "hccl", "cuda": "nccl"}.get(device_type, "gloo")
    # Bind the device BEFORE init_process_group: HCCL resolves the local device
    # when the process group is created.
    if device_type == "npu":
        torch.npu.set_device(local_rank)
        device = torch.device(f"npu:{local_rank}")
    elif device_type == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return rank, world_size, local_rank, device, device_type


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def _read_parquet_rows(path: str) -> list[dict]:
    import glob

    import pyarrow.parquet as pq

    expanded = os.path.expanduser(path)
    if os.path.isdir(expanded):
        files = sorted(glob.glob(os.path.join(expanded, "**", "*.parquet"), recursive=True))
    else:
        files = sorted(glob.glob(expanded))
    if not files:
        raise FileNotFoundError(f"no parquet files matched: {path}")
    rows: list[dict] = []
    for file in files:
        for batch in pq.ParquetFile(file).iter_batches(batch_size=4096):
            rows.extend(batch.to_pylist())
    return rows


def _encode_prompt(tokenizer, messages, enable_thinking: bool) -> list[int]:
    chat_kwargs = {} if enable_thinking else {"enable_thinking": False}
    try:
        ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
            **chat_kwargs,
        )
    except TypeError:
        # Tokenizers whose template does not accept enable_thinking.
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=False
        )
    return list(ids)


class PreferenceDataset(Dataset):
    """``(prompt, chosen, rejected)`` parquet -> tokenised preference pairs."""

    def __init__(self, path: str, tokenizer, args: argparse.Namespace):
        self.tokenizer = tokenizer
        self.args = args
        self.rows: list[dict] = []
        dropped = Counter()
        for row in _read_parquet_rows(path):
            prompt = row.get("prompt")
            chosen = row.get("chosen") or ""
            rejected = row.get("rejected") or ""
            if not prompt or not chosen.strip() or not rejected.strip():
                dropped["missing_field"] += 1
                continue
            if chosen.strip() == rejected.strip():
                dropped["identical"] += 1
                continue
            try:
                prompt_ids = _encode_prompt(tokenizer, prompt, args.enable_thinking)
            except Exception as exc:  # noqa: BLE001 - one bad prompt must not kill the run
                logger.warning("[data] chat template failed for task %r: %s", row.get("task_id"), exc)
                dropped["prompt_template_error"] += 1
                continue
            if not prompt_ids:
                dropped["empty_prompt"] += 1
                continue
            # Only the raw row is cached: tokenised prompt ids are ~36 bytes per
            # token in Python and would cost gigabytes across a full data set.
            self.rows.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "task_id": row.get("task_id", ""),
                }
            )
        logger.info("[data] %s: %d usable pairs (dropped %s)", path, len(self.rows), dict(dropped) or "none")
        if not self.rows:
            raise RuntimeError(f"{path} yielded no usable preference pairs")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        # The init pass guarantees a usable item exists; walk forward instead of
        # recursing so a mis-sized configuration fails loudly, not with a stack
        # overflow.
        for offset in range(len(self.rows)):
            row = self.rows[(index + offset) % len(self.rows)]
            try:
                prompt_ids = _encode_prompt(self.tokenizer, row["prompt"], self.args.enable_thinking)
            except Exception:  # noqa: BLE001 - init already filtered these
                continue
            if not prompt_ids:
                continue
            prepared = _prepare_pair(
                self.tokenizer,
                prompt_ids,
                row["chosen"],
                row["rejected"],
                self.args,
            )
            if prepared is not None:
                return prepared
        raise RuntimeError(
            "no preference pair fits --max_prompt_length/--max_total_length; "
            "raise --max_total_length or shorten the prompts"
        )


def _prepare_pair(tokenizer, prompt_ids, chosen, rejected, args) -> dict | None:
    prompt_ids = list(prompt_ids)
    if len(prompt_ids) > args.max_prompt_length:
        prompt_ids = prompt_ids[-args.max_prompt_length :]  # keep the most recent turns
    budget = args.max_total_length - len(prompt_ids)
    budget = min(args.max_response_length, budget)
    if budget < 2:
        return None

    def encode(text: str) -> list[int]:
        ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        ids = list(ids)[:budget]
        if args.append_eos and tokenizer.eos_token_id is not None:
            if not ids or ids[-1] != tokenizer.eos_token_id:
                ids = ids[: max(0, budget - 1)] + [tokenizer.eos_token_id]
        return ids

    chosen_ids = encode(chosen)
    rejected_ids = encode(rejected)
    if not chosen_ids or not rejected_ids:
        return None
    return {"prompt_ids": prompt_ids, "chosen_ids": chosen_ids, "rejected_ids": rejected_ids}


def _pack(items: list[dict], key: str, pad_token_id: int) -> dict:
    sequences = [(item["prompt_ids"] + item[key], len(item["prompt_ids"])) for item in items]
    max_len = max(len(ids) for ids, _ in sequences)
    batch_size = len(sequences)
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    response_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    response_lengths = torch.zeros(batch_size, dtype=torch.long)
    for index, (ids, start) in enumerate(sequences):
        length = len(ids)
        input_ids[index, :length] = torch.tensor(ids, dtype=torch.long)
        attention_mask[index, :length] = 1
        response_mask[index, start:length] = 1
        response_lengths[index] = length - start
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "response_mask": response_mask,
        "response_lengths": response_lengths,
    }


def dpo_collate(items: list[dict], pad_token_id: int) -> dict:
    return {
        "chosen": _pack(items, "chosen_ids", pad_token_id),
        "rejected": _pack(items, "rejected_ids", pad_token_id),
    }


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def _load_causal_lm(path: str, args: argparse.Namespace):
    from transformers import AutoModelForCausalLM

    kwargs = {
        "attn_implementation": args.attn_implementation,
        "trust_remote_code": args.trust_remote_code,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, **kwargs)
    except TypeError:  # transformers < 4.56 spells it torch_dtype
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, **kwargs)
    model.config.use_cache = False
    return model


def _enable_gradient_checkpointing(model) -> None:
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()


def _wrap_fsdp(module, args: argparse.Namespace, device_id: int, mixed_precision):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=args.fsdp_min_num_params
    )
    return FSDP(
        module,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device_id,
        use_orig_params=True,
        sync_module_states=True,
        limit_all_gathers=True,
    )


# ---------------------------------------------------------------------------
# DPO math
# ---------------------------------------------------------------------------


def _masked_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
    response_lengths: torch.Tensor,
    chunk_size: int,
    length_normalize: bool,
) -> torch.Tensor:
    """Sum of completion-token log-probs under ``logits``.

    The float32 softmax is computed one time-chunk at a time so the peak is
    ``batch * chunk * vocab`` rather than ``batch * seq * vocab``.
    """
    logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]
    mask = response_mask[:, 1:].float()
    totals = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.float32)
    total_tokens = logits.shape[1]
    for start in range(0, total_tokens, chunk_size):
        end = min(start + chunk_size, total_tokens)
        chunk = logits[:, start:end, :]
        target = targets[:, start:end]
        token_logprob = chunk.gather(-1, target.unsqueeze(-1)).squeeze(-1).float()
        token_logprob = token_logprob - torch.logsumexp(chunk.float(), dim=-1)
        totals = totals + (token_logprob * mask[:, start:end]).sum(dim=-1)
    if length_normalize:
        totals = totals / response_lengths.float().clamp_min(1.0)
    return totals


def _sequence_logprobs(model, batch: dict, args: argparse.Namespace) -> torch.Tensor:
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
    return _masked_logprobs(
        outputs.logits,
        batch["input_ids"],
        batch["response_mask"],
        batch["response_lengths"],
        args.logprob_chunk_size,
        args.length_normalize,
    )


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: {name: tensor.to(device, non_blocking=True) for name, tensor in value.items()}
        for key, value in batch.items()
    }


# ---------------------------------------------------------------------------
# checkpointing / logging
# ---------------------------------------------------------------------------


def _save_checkpoint(policy, tokenizer, output_dir: str, global_step: int, args, rank: int, is_fsdp: bool) -> None:
    from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType

    path = os.path.join(output_dir, f"global_step_{global_step}")
    if rank == 0:
        os.makedirs(path, exist_ok=True)
    if is_fsdp:
        with FSDP.state_dict_type(
            policy,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            state_dict = policy.state_dict()
        if rank == 0:
            policy.module.save_pretrained(path, state_dict=state_dict)
    elif rank == 0:
        policy.save_pretrained(path)
    if rank == 0:
        tokenizer.save_pretrained(path)
        with open(os.path.join(path, "trainer_state.json"), "w", encoding="utf-8") as handle:
            json.dump({"global_step": global_step, "args": vars(args)}, handle, ensure_ascii=False, indent=2)
        _prune_checkpoints(output_dir, args.save_total_limit)
        logger.info("[save] %s", path)
    dist.barrier()


def _prune_checkpoints(output_dir: str, keep: int) -> None:
    if keep <= 0:
        return
    entries = []
    for name in os.listdir(output_dir):
        if not name.startswith("global_step_"):
            continue
        try:
            step = int(name.split("_")[-1])
        except ValueError:
            continue
        entries.append((step, os.path.join(output_dir, name)))
    entries.sort()
    for _, path in entries[:-keep]:
        shutil.rmtree(path, ignore_errors=True)


def _init_wandb(args: argparse.Namespace, rank: int):
    if rank != 0 or not args.wandb_project:
        return None
    try:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=args.run_name or os.path.basename(os.path.abspath(args.output_dir)),
            config=vars(args),
        )
        return wandb
    except Exception as exc:  # noqa: BLE001 - logging must never kill training
        logger.warning("[wandb] disabled: %s", exc)
        return None


# ---------------------------------------------------------------------------
# train / eval
# ---------------------------------------------------------------------------


def _reduce_stats(values: list[float], device: torch.device) -> list[float]:
    # float32: HCCL does not reliably support float64 collectives.
    tensor = torch.tensor(values, dtype=torch.float32, device=device)
    dist.all_reduce(tensor)
    return tensor.tolist()


def _dpo_batch(
    policy,
    ref_model,
    batch: dict,
    args: argparse.Namespace,
    device: torch.device,
):
    pi_chosen = _sequence_logprobs(policy, batch["chosen"], args)
    pi_rejected = _sequence_logprobs(policy, batch["rejected"], args)
    if args.ref_free:
        ref_chosen = torch.zeros_like(pi_chosen)
        ref_rejected = torch.zeros_like(pi_rejected)
    else:
        with torch.no_grad():
            ref_chosen = _sequence_logprobs(ref_model, batch["chosen"], args)
            ref_rejected = _sequence_logprobs(ref_model, batch["rejected"], args)
    margin = args.beta * ((pi_chosen - pi_rejected) - (ref_chosen - ref_rejected))
    loss = -F.logsigmoid(margin).mean()
    with torch.no_grad():
        accuracy = (margin > 0).float().mean()
        stats = [
            float(loss.detach()) * len(pi_chosen),
            float(margin.detach().sum()),
            float(accuracy) * len(pi_chosen),
            float(pi_chosen.detach().sum()),
            float(pi_rejected.detach().sum()),
            float(len(pi_chosen)),
        ]
    return loss, stats


@torch.no_grad()
def evaluate(policy, ref_model, loader, args, device) -> dict:
    policy.eval()
    totals = [0.0] * 6
    for batch in loader:
        batch = _to_device(batch, device)
        _, stats = _dpo_batch(policy, ref_model, batch, args, device)
        totals = [left + right for left, right in zip(totals, stats, strict=True)]
    totals = _reduce_stats(totals, device)
    policy.train()
    count = max(totals[5], 1.0)
    return {
        "val/loss": totals[0] / count,
        "val/margin": totals[1] / count,
        "val/accuracy": totals[2] / count,
        "val/chosen_logp": totals[3] / count,
        "val/rejected_logp": totals[4] / count,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rank, world_size, local_rank, device, device_type = _init_distributed()
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _set_seed(args.seed + rank)
    if args.max_total_length <= 0:
        args.max_total_length = args.max_prompt_length + args.max_response_length
    if world_size > 1 and not args.fsdp:
        raise SystemExit("--no_fsdp only supports a single process; use torchrun + FSDP for multi-device training")
    if args.save_steps <= 0 and args.eval_steps <= 0:
        logger.warning("both --save_steps and --eval_steps are 0; only the final checkpoint will be written")

    os.makedirs(args.output_dir, exist_ok=True)
    ref_model_path = args.ref_model or args.model
    tokenizer_path = args.tokenizer or args.model
    logger.info("policy=%s ref=%s ref_free=%s", args.model, ref_model_path, args.ref_free)

    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- data -----------------------------------------------------------
    train_dataset = PreferenceDataset(os.path.expanduser(args.train_files), tokenizer, args)
    val_dataset = (
        PreferenceDataset(os.path.expanduser(args.val_files), tokenizer, args) if args.val_files else None
    )
    collate_fn = functools.partial(dpo_collate, pad_token_id=tokenizer.pad_token_id)
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True, seed=args.seed
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = None
    if val_dataset is not None:
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
            seed=args.seed,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=max(1, args.per_device_batch_size),
            sampler=val_sampler,
            collate_fn=collate_fn,
            num_workers=0,
            drop_last=False,
        )

    steps_per_epoch = len(train_loader) // args.grad_accum
    if steps_per_epoch <= 0:
        raise SystemExit(
            f"not enough preference pairs: {len(train_loader)} micro-batches cannot fill "
            f"--grad_accum={args.grad_accum}"
        )
    total_steps = max(1, int(math.ceil(steps_per_epoch * args.epochs)))
    warmup_steps = int(total_steps * args.warmup_ratio)
    logger.info(
        "pairs=%d steps/epoch=%d total_steps=%d warmup=%d world_size=%d",
        len(train_dataset),
        steps_per_epoch,
        total_steps,
        warmup_steps,
        world_size,
    )

    # ---- models ---------------------------------------------------------
    policy_module = _load_causal_lm(args.model, args)
    if args.gradient_checkpointing:
        _enable_gradient_checkpointing(policy_module)
    if args.fsdp:
        from torch.distributed.fsdp import MixedPrecision

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            # verl's FSDP recipe keeps buffers (e.g. RoPE inv_freq) in fp32;
            # casting them to bf16 costs precision for no memory win.
            buffer_dtype=torch.float32,
        )
        policy = _wrap_fsdp(policy_module, args, local_rank, mixed_precision)
    else:
        policy = policy_module.to(device)

    ref_model = None
    if not args.ref_free:
        ref_module = _load_causal_lm(ref_model_path, args)
        ref_module.eval()
        ref_module.requires_grad_(False)
        if args.fsdp:
            ref_model = _wrap_fsdp(ref_module, args, local_rank, None)
        else:
            ref_model = ref_module.to(device)
        ref_model.eval()

    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    if args.lr_scheduler == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    else:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.5 if args.lr_scheduler == "cosine" else 1.0,
        )

    wandb_run = _init_wandb(args, rank)
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")

    def log_metrics(metrics: dict, step: int) -> None:
        if rank != 0:
            return
        metrics = {"step": step, **metrics}
        printable = {
            key: round(value, 6) if isinstance(value, float) else value
            for key, value in metrics.items()
        }
        logger.info("[metrics] %s", json.dumps(printable))
        with open(metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        if wandb_run is not None:
            wandb_run.log(metrics, step=step)

    # ---- loop -----------------------------------------------------------
    policy.train()
    global_step = 0
    micro_step = 0
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    running = [0.0] * 6
    stop = False

    for epoch in range(int(math.ceil(args.epochs))):
        train_sampler.set_epoch(epoch)
        for batch in train_loader:
            step_in_accum = (micro_step % args.grad_accum) + 1
            should_sync = step_in_accum == args.grad_accum or not args.fsdp
            batch = _to_device(batch, device)
            context = contextlib.nullcontext() if should_sync else policy.no_sync()
            with context:
                loss, stats = _dpo_batch(policy, ref_model, batch, args, device)
                (loss / args.grad_accum).backward()
            running = [left + right for left, right in zip(running, stats, strict=True)]
            micro_step += 1

            if not should_sync:
                continue

            grad_norm = policy.clip_grad_norm_(args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % max(1, args.log_steps) == 0:
                reduced = _reduce_stats(running, device)
                count = max(reduced[5], 1.0)
                log_metrics(
                    {
                        "train/loss": reduced[0] / count,
                        "train/margin": reduced[1] / count,
                        "train/accuracy": reduced[2] / count,
                        "train/chosen_logp": reduced[3] / count,
                        "train/rejected_logp": reduced[4] / count,
                        "train/grad_norm": float(grad_norm),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/pairs": reduced[5],
                        "train/elapsed_min": (time.time() - started) / 60.0,
                    },
                    global_step,
                )
            running = [0.0] * 6

            if val_loader is not None and args.eval_steps > 0 and global_step % args.eval_steps == 0:
                log_metrics(evaluate(policy, ref_model, val_loader, args, device), global_step)
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                _save_checkpoint(policy, tokenizer, args.output_dir, global_step, args, rank, args.fsdp)
            if global_step >= total_steps:  # fractional --epochs stops mid-epoch
                stop = True
                break
        if stop:
            break

    if val_loader is not None:
        log_metrics(evaluate(policy, ref_model, val_loader, args, device), global_step)
    _save_checkpoint(policy, tokenizer, args.output_dir, max(global_step, 1), args, rank, args.fsdp)

    if wandb_run is not None:
        wandb_run.finish()
    logger.info("[done] %d optimizer steps in %.1f min", global_step, (time.time() - started) / 60.0)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
