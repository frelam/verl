# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Full-vocab KL distillation entry for the Megatron backend (vocab-parallel TP).

Each TP rank holds the student's ``[N, V/tp]`` logit shard and the teacher
lm_head shard ``[V/tp, H]`` sliced with the student's vocab layout, so the KL is
computed shard-locally with no cross-rank logit exchange. The only collectives
are exact online-softmax merges of per-token scalars (``O(N * tp)``):

- student logsumexp: ``all_reduce(MAX)`` + ``all_reduce(SUM)``;
- teacher online-softmax accumulators: all-gather + max/alpha-rescaled sum;
- reverse KL's additive moments (``E[p_S z_s]``, ``E[p_S z_t]``): ``all_reduce(SUM)``.

Backward needs no communication: every rank saved the globally-merged
log-partitions and differentiates only its local student logit shard.

Both forward KL (``KL(p_T||p_S)``) and reverse KL (``KL(p_S||p_T)``) are
implemented. CP splits reuse ``preprocess_thd_engine`` / ``preprocess_bshd_engine``
on the teacher hidden states so token alignment with the student logits is
guaranteed by construction.
"""

from typing import Optional

import torch

from verl.models.mcore.util import preprocess_bshd_engine, preprocess_thd_engine
from verl.trainer.distillation import full_vocab_kl as fvkl
from verl.trainer.distillation.full_vocab_kl import (
    _fwd_kl_bwd_chunk,
    _fwd_local_accumulators,
    _iter_weight_chunks,
    _rev_kl_bwd_chunk,
    _rev_local_accumulators,
    _scale_temperature,
    _student_floor,
)
from verl.workers.config import DistillationConfig


def _vocab_parallel_lse(s_f: torch.Tensor, tp_group) -> torch.Tensor:
    """Global logsumexp of vocab-parallel logits ``[N, V/tp]`` (fp32 in/out)."""
    local_max = s_f.max(dim=-1).values
    global_max = local_max.clone()
    torch.distributed.all_reduce(global_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)
    local_sumexp = (s_f - local_max.unsqueeze(-1)).exp().sum(-1)
    sumexp = local_sumexp * (local_max - global_max).exp()
    torch.distributed.all_reduce(sumexp, op=torch.distributed.ReduceOp.SUM, group=tp_group)
    return global_max + sumexp.log()


def _merge_tp_online(mt: torch.Tensor, accs: list[torch.Tensor], tp_group) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Exact merge of online-softmax accumulators across TP ranks.

    ``mt`` (running max) merges via max; every accumulator in ``accs`` merges via
    an alpha-rescaled sum, ``sum_r acc_r * exp(mt_r - max)``.
    """
    world_size = torch.distributed.get_world_size(tp_group)
    if world_size == 1:
        return mt, accs
    local = torch.stack([mt, *accs], dim=1)  # [N, 1 + K]
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, local, group=tp_group)
    stacked = torch.stack(gathered, dim=1)  # [N, tp, 1 + K]
    global_mt = stacked[:, :, 0].max(dim=1).values
    alpha = (stacked[:, :, 0] - global_mt.unsqueeze(1)).exp()  # [N, tp]
    merged = [(stacked[:, :, k + 1] * alpha).sum(dim=1) for k in range(len(accs))]
    return global_mt, merged


def _all_reduce_sum(x: torch.Tensor, tp_group) -> torch.Tensor:
    out = x.clone()
    torch.distributed.all_reduce(out, op=torch.distributed.ReduceOp.SUM, group=tp_group)
    return out


class _VocabParallelFullVocabForwardKL(torch.autograd.Function):
    """KL(p_T || p_S) over TP-sharded vocab: chunked local accumulators + exact TP merge."""

    @staticmethod
    def forward(
        ctx,
        z_cpu: torch.Tensor,
        weight_fwd: torch.Tensor,
        weight_bwd: torch.Tensor,
        vp_student_logits: torch.Tensor,
        chunk_vocab: int,
        temperature: float,
        log_prob_min_clamp: Optional[float],
    ) -> torch.Tensor:
        from megatron.core.parallel_state import get_tensor_model_parallel_group

        tp_group = get_tensor_model_parallel_group()
        device = vp_student_logits.device
        s_f = vp_student_logits.float()
        z_f = z_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
        z_f, s_f = _scale_temperature(z_f, s_f, temperature)
        s_lse = _vocab_parallel_lse(s_f, tp_group)
        s_min = _student_floor(s_lse, log_prob_min_clamp, device)

        mt, st, tt, ut = _fwd_local_accumulators(z_f, weight_fwd, s_f, chunk_vocab, s_min, device)
        mt, (st, tt, ut) = _merge_tp_online(mt, [st, tt, ut], tp_group)
        t_lse = mt + st.log()
        kl = tt / st - t_lse - ut / st + s_lse

        ctx.save_for_backward(z_cpu, weight_bwd, vp_student_logits, t_lse, s_lse, s_min)
        ctx.chunk_vocab = chunk_vocab
        ctx.temperature = temperature
        return kl

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        z_cpu, weight, vp_student_logits, t_lse, s_lse, s_min = ctx.saved_tensors
        device = vp_student_logits.device
        n, vocab = vp_student_logits.shape
        s_f = vp_student_logits.float()
        z_f = z_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
        z_f, s_f = _scale_temperature(z_f, s_f, ctx.temperature)
        grad = grad_output.float()

        grad_s = torch.empty(n, vocab, dtype=vp_student_logits.dtype, device=device)
        for v0, w_chunk in _iter_weight_chunks(weight, ctx.chunk_vocab, device):
            c = w_chunk.shape[0]
            grad_chunk = _fwd_kl_bwd_chunk(z_f, w_chunk, s_f[:, v0 : v0 + c], t_lse, s_lse, s_min, grad)
            grad_s[:, v0 : v0 + c] = grad_chunk.to(vp_student_logits.dtype)
        if ctx.temperature != 1.0:
            grad_s = grad_s / ctx.temperature
        return None, None, None, grad_s, None, None, None


class _VocabParallelFullVocabReverseKL(torch.autograd.Function):
    """KL(p_S || p_T) over TP-sharded vocab: merged teacher lse + all-reduced p_S moments."""

    @staticmethod
    def forward(
        ctx,
        z_cpu: torch.Tensor,
        weight_fwd: torch.Tensor,
        weight_bwd: torch.Tensor,
        vp_student_logits: torch.Tensor,
        chunk_vocab: int,
        temperature: float,
    ) -> torch.Tensor:
        from megatron.core.parallel_state import get_tensor_model_parallel_group

        tp_group = get_tensor_model_parallel_group()
        device = vp_student_logits.device
        s_f = vp_student_logits.float()
        z_f = z_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
        z_f, s_f = _scale_temperature(z_f, s_f, temperature)
        s_lse = _vocab_parallel_lse(s_f, tp_group)

        mt, st, ut, et = _rev_local_accumulators(z_f, weight_fwd, s_f, s_lse, chunk_vocab, device)
        mt, (st,) = _merge_tp_online(mt, [st], tp_group)
        # E[p_S * z_s] and E[p_S * z_t] are plain additive moments over the vocab.
        ut = _all_reduce_sum(ut, tp_group)
        et = _all_reduce_sum(et, tp_group)
        t_lse = mt + st.log()
        kl = et - ut - s_lse + t_lse

        ctx.save_for_backward(z_cpu, weight_bwd, vp_student_logits, t_lse, s_lse, kl)
        ctx.chunk_vocab = chunk_vocab
        ctx.temperature = temperature
        return kl

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        z_cpu, weight, vp_student_logits, t_lse, s_lse, kl = ctx.saved_tensors
        device = vp_student_logits.device
        n, vocab = vp_student_logits.shape
        s_f = vp_student_logits.float()
        z_f = z_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
        z_f, s_f = _scale_temperature(z_f, s_f, ctx.temperature)
        grad = grad_output.float()

        grad_s = torch.empty(n, vocab, dtype=vp_student_logits.dtype, device=device)
        for v0, w_chunk in _iter_weight_chunks(weight, ctx.chunk_vocab, device):
            c = w_chunk.shape[0]
            grad_chunk = _rev_kl_bwd_chunk(z_f, w_chunk, s_f[:, v0 : v0 + c], t_lse, s_lse, kl, grad)
            grad_s[:, v0 : v0 + c] = grad_chunk.to(vp_student_logits.dtype)
        if ctx.temperature != 1.0:
            grad_s = grad_s / ctx.temperature
        return None, None, None, grad_s, None, None


def compute_full_vocab_kl(
    student_logits: torch.Tensor,
    data,
    config,
    distillation_config: DistillationConfig,
    data_format: str,
    reverse: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute per-token full-vocab KL for one Megatron micro batch.

    Args:
        student_logits: ``(1, total/cp, V/tp)`` (thd) or ``(B, S/cp, V/tp)`` (bshd).
        data: micro batch TensorDict with ``input_ids`` (nested) and the
            ``teacher_full_vocab_artifact`` non-tensor column.
        data_format: ``"thd"`` or ``"bshd"``; selects the CP split rule so the
            teacher hidden states are sharded exactly like the student logits.
        reverse: compute ``KL(p_S||p_T)`` instead of ``KL(p_T||p_S)``.

    Returns:
        ``distillation_losses`` / ``teacher_hidden_norm`` / ``teacher_hidden_coverage``,
        each with shape ``student_logits.shape[:2]``.
    """
    del config  # uniform signature with the FSDP entry
    from megatron.core.parallel_state import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    loss_config = distillation_config.distillation_loss
    device = student_logits.device

    input_ids = data["input_ids"]
    assert input_ids.is_nested, "input_ids must be nested for full-vocab distillation"
    batch_size = input_ids.shape[0]
    artifacts = fvkl.normalize_artifacts(data["teacher_full_vocab_artifact"], batch_size)
    seq_lens = [int(t.shape[0]) for t in input_ids.unbind()]

    rows, teacher_index, teacher_keys = fvkl.fetch_hidden_rows(artifacts, seq_lens)

    hidden_nested = torch.nested.nested_tensor(rows)  # [B, j, H] CPU
    idx_nested = torch.nested.nested_tensor(
        [torch.full((seq_len,), g, dtype=torch.int64) for seq_len, g in zip(seq_lens, teacher_index, strict=True)]
    )  # [B, j] CPU

    # Shard the teacher tensors across CP with the exact same rule as input_ids;
    # CP-pad positions get zero hidden and are masked downstream by response_mask.
    if data_format == "thd":
        hidden_split, *_ = preprocess_thd_engine(hidden_nested, pre_process=True)
        idx_split, *_ = preprocess_thd_engine(idx_nested, pre_process=True)
    else:
        hidden_split, *_ = preprocess_bshd_engine(hidden_nested, pre_process=True)
        idx_split, *_ = preprocess_bshd_engine(idx_nested, pre_process=True)
    assert hidden_split.shape[:2] == student_logits.shape[:2], (
        f"teacher hidden {tuple(hidden_split.shape[:2])} does not match student logits "
        f"{tuple(student_logits.shape[:2])} after CP split ({data_format=})"
    )

    n = int(torch.tensor(hidden_split.shape[:2]).prod())
    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()
    store = fvkl.get_teacher_lm_head_store(
        distillation_config,
        vocab_size_padded=student_logits.shape[-1] * tp_size,
        tp_rank=tp_rank,
        tp_size=tp_size,
    )
    kl, norm = fvkl.run_teacher_passes(
        hidden_split.reshape(n, hidden_split.shape[-1]),
        idx_split.reshape(n),
        teacher_keys,
        student_logits.reshape(n, student_logits.shape[-1]),
        store=store,
        loss_config=loss_config,
        reverse=reverse,
        kernel_fwd=_VocabParallelFullVocabForwardKL.apply,
        kernel_rev=_VocabParallelFullVocabReverseKL.apply,
    )
    coverage = (idx_split.reshape(n) >= 0).to(torch.float32).to(device)
    out_shape = student_logits.shape[:2]
    return {
        "distillation_losses": kl.view(out_shape),
        "teacher_hidden_norm": norm.view(out_shape),
        "teacher_hidden_coverage": coverage.view(out_shape),
    }
