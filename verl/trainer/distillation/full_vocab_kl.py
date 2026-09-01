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

"""Full-vocab KL kernels and teacher lm_head management (strategy-agnostic).

This module implements the shared pieces of full-vocab OPD/MOPD distillation:

- Chunked online-softmax forward/reverse KL over the vocabulary axis: teacher
  logits are reconstructed on the fly as ``hidden @ W[v:v+C].T`` in vocab chunks
  of size ``C``, so the ``[N, V]`` teacher logit tensor is never materialized.
  Peak extra memory is ``O(N_chunk * C)`` instead of ``O(N * V)``.
- ``torch.autograd.Function`` wrappers whose backward **recomputes** the teacher
  logits per vocab chunk (second pass). Only student logits receive gradients;
  the teacher hidden states and the lm_head weight are stop-gradient. The
  autograd context only saves the (small) CPU hidden block, the CPU lm_head
  shard reference, and per-token ``[N]`` scalars — no ``[N, V]`` intermediates.
- Teacher lm_head loading directly from safetensors (no HF model construction),
  vocab-parallel sharding aligned with the student's TP layout (zero-row padding
  up to the student's padded vocab size), and :class:`TeacherLmHeadStore`, the
  per-engine-process registry implementing the MOPD residency policy:
  all teacher shards live on CPU (pinned); device copies are made on demand and
  bounded by an LRU of ``max_resident_teachers``.

Hardware agnosticism: no ``torch.cuda`` references; cache clearing goes through
``verl.utils.device.get_torch_device()`` and device context is always taken from
the student logits tensor.
"""

from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import torch

from verl.trainer.distillation import full_vocab_tq
from verl.utils.device import get_torch_device
from verl.utils.fs import copy_to_local

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_VOCAB = 8192
DEFAULT_CHUNK_TOKENS = 4096


def _maybe_compile(fn):
    """``torch.compile`` with graceful eager fallback.

    Disabled when ``TORCH_COMPILE_DISABLE``/``TORCHINDUCTOR_DISABLE`` is set
    (CPU test runners) or when the installed torch has no compile support.
    """
    if os.environ.get("TORCH_COMPILE_DISABLE", "0") == "1" or os.environ.get("TORCHINDUCTOR_DISABLE", "0") == "1":
        return fn
    if not hasattr(torch, "compile"):
        return fn
    try:
        return torch.compile(fn, dynamic=True)
    except Exception:  # pragma: no cover - compile is best effort
        return fn


# ---------------------------------------------------------------------------
# Chunk-level math (pure tensor in/out; fusible by torch.compile / Triton later)
# ---------------------------------------------------------------------------


@_maybe_compile
def _fwd_kl_chunk_update(
    z_f: torch.Tensor,  # (N, H) float32, temperature-scaled
    w_chunk: torch.Tensor,  # (C, H) float32, on device
    zs_chunk: torch.Tensor,  # (N, C) float32, temperature-scaled
    mt: torch.Tensor,  # (N,) running teacher max
    st: torch.Tensor,  # (N,) sum exp(zt - mt)
    tt: torch.Tensor,  # (N,) sum exp(zt - mt) * zt
    ut: torch.Tensor,  # (N,) sum exp(zt - mt) * clamp(zs)
    s_min: torch.Tensor,  # (N, 1) floor for student logits (-inf when clamp off)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Online-softmax update of the forward-KL accumulators for one vocab chunk."""
    zt = z_f @ w_chunk.T
    tile_mt = zt.max(dim=1).values
    new_mt = torch.maximum(mt, tile_mt)
    alpha = (mt - new_mt).exp()
    pt = (zt - new_mt.unsqueeze(1)).exp()
    st = st * alpha + pt.sum(dim=1)
    tt = tt * alpha + (pt * zt).sum(dim=1)
    zs_clamped = torch.maximum(zs_chunk, s_min)
    ut = ut * alpha + (pt * zs_clamped).sum(dim=1)
    return new_mt, st, tt, ut


@_maybe_compile
def _rev_kl_chunk_update(
    z_f: torch.Tensor,  # (N, H) float32, temperature-scaled
    w_chunk: torch.Tensor,  # (C, H) float32, on device
    zs_chunk: torch.Tensor,  # (N, C) float32, temperature-scaled
    s_lse: torch.Tensor,  # (N,) student logsumexp (global)
    mt: torch.Tensor,  # (N,) running teacher max
    st: torch.Tensor,  # (N,) sum exp(zt - mt)
    ut: torch.Tensor,  # (N,) sum p_S * zt
    et: torch.Tensor,  # (N,) sum p_S * zs
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Online-softmax update of the reverse-KL accumulators for one vocab chunk."""
    zt = z_f @ w_chunk.T
    tile_mt = zt.max(dim=1).values
    new_mt = torch.maximum(mt, tile_mt)
    alpha = (mt - new_mt).exp()
    st = st * alpha + (zt - new_mt.unsqueeze(1)).exp().sum(dim=1)
    ps = (zs_chunk - s_lse.unsqueeze(1)).exp()
    ut = ut + (ps * zt).sum(dim=1)
    et = et + (ps * zs_chunk).sum(dim=1)
    return new_mt, st, ut, et


@_maybe_compile
def _fwd_kl_bwd_chunk(
    z_f: torch.Tensor,  # (N, H) float32, temperature-scaled
    w_chunk: torch.Tensor,  # (C, H) float32, on device
    zs_chunk: torch.Tensor,  # (N, C) float32, temperature-scaled
    t_lse: torch.Tensor,  # (N,)
    s_lse: torch.Tensor,  # (N,)
    s_min: torch.Tensor,  # (N, 1) student-logit floor used by the forward clamp
    grad: torch.Tensor,  # (N,)
) -> torch.Tensor:
    """d(KL(p_T||p_S))/d(zs) for one vocab chunk: p_S - p_T * 1[zs not clamped]."""
    zt = z_f @ w_chunk.T
    p_T = (zt - t_lse.unsqueeze(1)).exp()
    p_S = (zs_chunk - s_lse.unsqueeze(1)).exp()
    active = (zs_chunk > s_min).to(p_T.dtype)
    return (p_S - p_T * active) * grad.unsqueeze(1)


@_maybe_compile
def _rev_kl_bwd_chunk(
    z_f: torch.Tensor,  # (N, H) float32, temperature-scaled
    w_chunk: torch.Tensor,  # (C, H) float32, on device
    zs_chunk: torch.Tensor,  # (N, C) float32, temperature-scaled
    t_lse: torch.Tensor,  # (N,)
    s_lse: torch.Tensor,  # (N,)
    kl: torch.Tensor,  # (N,) per-token reverse KL from the forward pass
    grad: torch.Tensor,  # (N,)
) -> torch.Tensor:
    """d(KL(p_S||p_T))/d(zs) = p_S * (log p_S - log p_T - KL) for one vocab chunk."""
    zt = z_f @ w_chunk.T
    log_ps = zs_chunk - s_lse.unsqueeze(1)
    log_pt = zt - t_lse.unsqueeze(1)
    ps = log_ps.exp()
    return ps * (log_ps - log_pt - kl.unsqueeze(1)) * grad.unsqueeze(1)


# ---------------------------------------------------------------------------
# Local (single-shard) accumulator drivers shared by FSDP and Megatron kernels
# ---------------------------------------------------------------------------


def _iter_weight_chunks(
    weight: torch.Tensor, chunk_vocab: int, device: torch.device
) -> Iterator[tuple[int, torch.Tensor]]:
    """Yield ``(v0, fp32 [C, H] chunk)`` on ``device``, streamed from wherever ``weight`` lives."""
    vocab = weight.shape[0]
    for v0 in range(0, vocab, chunk_vocab):
        yield v0, weight[v0 : v0 + chunk_vocab].to(device=device, dtype=torch.float32, non_blocking=True)


def _student_floor(s_lse: torch.Tensor, log_prob_min_clamp: Optional[float], device: torch.device) -> torch.Tensor:
    """Floor on raw student logits so that ``log p_S >= log_prob_min_clamp`` after softmax."""
    if log_prob_min_clamp is not None:
        return (s_lse + log_prob_min_clamp).unsqueeze(1)
    return torch.full((1, 1), float("-inf"), dtype=torch.float32, device=device)


def _fwd_local_accumulators(
    z_f: torch.Tensor,  # (N, H) float32, on device, temperature-scaled
    weight: torch.Tensor,  # (V_shard, H) CPU or device, any dtype
    s_f: torch.Tensor,  # (N, V_shard) float32, on device, temperature-scaled
    chunk_vocab: int,
    s_min: torch.Tensor,  # (N, 1)
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chunked single-pass online-softmax accumulators for forward KL on the local vocab shard."""
    n = z_f.shape[0]
    mt = torch.full((n,), float("-inf"), dtype=torch.float32, device=device)
    st = torch.zeros(n, dtype=torch.float32, device=device)
    tt = torch.zeros(n, dtype=torch.float32, device=device)
    ut = torch.zeros(n, dtype=torch.float32, device=device)
    for v0, w_chunk in _iter_weight_chunks(weight, chunk_vocab, device):
        c = w_chunk.shape[0]
        mt, st, tt, ut = _fwd_kl_chunk_update(z_f, w_chunk, s_f[:, v0 : v0 + c], mt, st, tt, ut, s_min)
    return mt, st, tt, ut


def _rev_local_accumulators(
    z_f: torch.Tensor,  # (N, H) float32, on device, temperature-scaled
    weight: torch.Tensor,  # (V_shard, H) CPU or device, any dtype
    s_f: torch.Tensor,  # (N, V_shard) float32, on device, temperature-scaled
    s_lse: torch.Tensor,  # (N,) global student logsumexp
    chunk_vocab: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chunked single-pass online-softmax accumulators for reverse KL on the local vocab shard."""
    n = z_f.shape[0]
    mt = torch.full((n,), float("-inf"), dtype=torch.float32, device=device)
    st = torch.zeros(n, dtype=torch.float32, device=device)
    ut = torch.zeros(n, dtype=torch.float32, device=device)
    et = torch.zeros(n, dtype=torch.float32, device=device)
    for v0, w_chunk in _iter_weight_chunks(weight, chunk_vocab, device):
        c = w_chunk.shape[0]
        mt, st, ut, et = _rev_kl_chunk_update(z_f, w_chunk, s_f[:, v0 : v0 + c], s_lse, mt, st, ut, et)
    return mt, st, ut, et


def _scale_temperature(z: torch.Tensor, s: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    if temperature != 1.0:
        inv_t = 1.0 / temperature
        return z * inv_t, s * inv_t
    return z, s


# ---------------------------------------------------------------------------
# autograd Functions (single vocab shard; the Megatron TP variants live in
# verl/trainer/distillation/megatron/full_vocab_kl.py and reuse the chunk math)
# ---------------------------------------------------------------------------


class FullVocabForwardKL(torch.autograd.Function):
    """KL(p_T || p_S) over the full vocabulary without materializing teacher logits.

    Args:
        z_cpu: (N, H) teacher pre-lm_head hidden states, on CPU (streamed on-device).
        weight_fwd: (V, H) teacher lm_head used by the forward pass; may live on
            device (GPU-resident policy) or on CPU (streamed per vocab chunk).
        weight_bwd: (V, H) CPU lm_head shard re-streamed during backward.
        student_logits: (N, V) student logits on device (the only input that
            receives gradients).
    """

    @staticmethod
    def forward(
        ctx,
        z_cpu: torch.Tensor,
        weight_fwd: torch.Tensor,
        weight_bwd: torch.Tensor,
        student_logits: torch.Tensor,
        chunk_vocab: int,
        temperature: float,
        log_prob_min_clamp: Optional[float],
    ) -> torch.Tensor:
        device = student_logits.device
        s_f = student_logits.float()
        z_f = z_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
        z_f, s_f = _scale_temperature(z_f, s_f, temperature)
        s_lse = s_f.logsumexp(dim=-1)
        s_min = _student_floor(s_lse, log_prob_min_clamp, device)

        mt, st, tt, ut = _fwd_local_accumulators(z_f, weight_fwd, s_f, chunk_vocab, s_min, device)
        t_lse = mt + st.log()
        kl = tt / st - t_lse - ut / st + s_lse

        ctx.save_for_backward(z_cpu, weight_bwd, student_logits, t_lse, s_lse, s_min)
        ctx.chunk_vocab = chunk_vocab
        ctx.temperature = temperature
        return kl

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        z_cpu, weight, student_logits, t_lse, s_lse, s_min = ctx.saved_tensors
        device = student_logits.device
        n, vocab = student_logits.shape
        s_f = student_logits.float()
        z_f = z_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
        z_f, s_f = _scale_temperature(z_f, s_f, ctx.temperature)
        grad = grad_output.float()

        grad_s = torch.empty(n, vocab, dtype=student_logits.dtype, device=device)
        for v0, w_chunk in _iter_weight_chunks(weight, ctx.chunk_vocab, device):
            c = w_chunk.shape[0]
            grad_chunk = _fwd_kl_bwd_chunk(z_f, w_chunk, s_f[:, v0 : v0 + c], t_lse, s_lse, s_min, grad)
            grad_s[:, v0 : v0 + c] = grad_chunk.to(student_logits.dtype)
        if ctx.temperature != 1.0:
            grad_s = grad_s / ctx.temperature
        return None, None, None, grad_s, None, None, None


class FullVocabReverseKL(torch.autograd.Function):
    """KL(p_S || p_T) over the full vocabulary; same memory layout as the forward KL."""

    @staticmethod
    def forward(
        ctx,
        z_cpu: torch.Tensor,
        weight_fwd: torch.Tensor,
        weight_bwd: torch.Tensor,
        student_logits: torch.Tensor,
        chunk_vocab: int,
        temperature: float,
    ) -> torch.Tensor:
        device = student_logits.device
        s_f = student_logits.float()
        z_f = z_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
        z_f, s_f = _scale_temperature(z_f, s_f, temperature)
        s_lse = s_f.logsumexp(dim=-1)

        mt, st, ut, et = _rev_local_accumulators(z_f, weight_fwd, s_f, s_lse, chunk_vocab, device)
        t_lse = mt + st.log()
        kl = et - ut - s_lse + t_lse

        ctx.save_for_backward(z_cpu, weight_bwd, student_logits, t_lse, s_lse, kl)
        ctx.chunk_vocab = chunk_vocab
        ctx.temperature = temperature
        return kl

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        z_cpu, weight, student_logits, t_lse, s_lse, kl = ctx.saved_tensors
        device = student_logits.device
        n, vocab = student_logits.shape
        s_f = student_logits.float()
        z_f = z_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
        z_f, s_f = _scale_temperature(z_f, s_f, ctx.temperature)
        grad = grad_output.float()

        grad_s = torch.empty(n, vocab, dtype=student_logits.dtype, device=device)
        for v0, w_chunk in _iter_weight_chunks(weight, ctx.chunk_vocab, device):
            c = w_chunk.shape[0]
            grad_chunk = _rev_kl_bwd_chunk(z_f, w_chunk, s_f[:, v0 : v0 + c], t_lse, s_lse, kl, grad)
            grad_s[:, v0 : v0 + c] = grad_chunk.to(student_logits.dtype)
        if ctx.temperature != 1.0:
            grad_s = grad_s / ctx.temperature
        return None, None, None, grad_s, None, None


# ---------------------------------------------------------------------------
# Teacher lm_head loading and sharding
# ---------------------------------------------------------------------------


def load_teacher_lm_head(
    checkpoint_path: str, layer: str = "auto", dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Read a teacher's lm_head (or tied embed_tokens) weight from safetensors.

    Returns a CPU tensor of shape ``[V, H]`` in the checkpoint's native dtype
    (unless ``dtype`` overrides). Avoids constructing a full HF model just to
    read a single matrix. ``checkpoint_path`` may be a local dir, an HDFS path
    (resolved via ``copy_to_local``), or a Hugging Face repo id.
    """
    from safetensors.torch import load_file
    from transformers import AutoConfig

    resolved = copy_to_local(checkpoint_path)
    if os.path.isdir(resolved):
        folder = resolved
    else:
        from huggingface_hub import snapshot_download

        folder = snapshot_download(resolved, allow_patterns=["*.safetensors*", "*.json"])

    if layer == "auto":
        cfg = AutoConfig.from_pretrained(folder, trust_remote_code=True)
        target_keys = (
            ["model.embed_tokens.weight", "embed_tokens.weight"]
            if getattr(cfg, "tie_word_embeddings", False)
            else ["lm_head.weight", "model.lm_head.weight"]
        )
    else:
        target_keys = [layer]

    index_path = os.path.join(folder, "model.safetensors.index.json")
    shard_for_key: dict[str, str] = {}
    if os.path.exists(index_path):
        with open(index_path) as f:
            shard_for_key = json.load(f)["weight_map"]

    candidate_files: list[str] = []
    if shard_for_key:
        for key in target_keys:
            shard = shard_for_key.get(key)
            if shard is not None:
                candidate_files.append(os.path.join(folder, shard))
    if not candidate_files:
        single = os.path.join(folder, "model.safetensors")
        if os.path.exists(single):
            candidate_files.append(single)

    for path in candidate_files:
        tensors = load_file(path)
        for key in target_keys:
            if key in tensors:
                weight = tensors[key]
                return weight.to(dtype) if dtype is not None else weight

    raise ValueError(
        f"Could not find lm_head/embed_tokens weight (keys={target_keys}) in {checkpoint_path}. "
        "Set distillation_loss.full_vocab_lm_head_layer to the exact weight name if 'auto' fails."
    )


def shard_lm_head(
    weight: torch.Tensor,
    vocab_size_padded: int,
    tp_rank: int = 0,
    tp_size: int = 1,
) -> torch.Tensor:
    """Zero-pad ``[V, H]`` to ``vocab_size_padded`` rows and take this TP rank's shard.

    The shard boundaries match the student's Megatron vocab-parallel layout
    (contiguous ``VocabUtility.vocab_range_from_per_partition_vocab_size``), so the
    KL is computed shard-locally with no cross-rank logit exchange. Padded rows are
    zeros, i.e. teacher logit 0 — finite, keeping both forward and reverse KL bounded.
    FSDP passes ``tp_size=1`` and receives the full padded matrix.
    """
    vocab, hidden = weight.shape
    if vocab_size_padded % tp_size != 0:
        raise ValueError(f"vocab_size_padded ({vocab_size_padded}) must be divisible by tp_size ({tp_size}).")
    if vocab > vocab_size_padded:
        raise ValueError(
            f"teacher vocab ({vocab}) exceeds the student's padded vocab ({vocab_size_padded}); "
            "full-vocab distillation requires teacher and student to share the tokenizer."
        )
    if vocab < vocab_size_padded:
        pad = weight.new_zeros(vocab_size_padded - vocab, hidden)
        weight = torch.cat([weight, pad], dim=0)
    per_partition = vocab_size_padded // tp_size
    return weight[tp_rank * per_partition : (tp_rank + 1) * per_partition].contiguous()


def resolve_teacher_lm_head_checkpoints(distillation_config) -> dict[str, str]:
    """Map each teacher key to the checkpoint its lm_head shard is loaded from."""
    loss_config = distillation_config.distillation_loss
    override = getattr(loss_config, "full_vocab_lm_head_checkpoint", None)
    teachers = distillation_config.teacher_models
    if override and len(teachers) > 1:
        raise ValueError(
            "full_vocab_lm_head_checkpoint is a single global override and is ambiguous with "
            f"multiple teachers ({sorted(teachers)}); leave it null so each teacher's own "
            "model_path is used."
        )
    return {key: (override or teacher.model_path) for key, teacher in teachers.items()}


# ---------------------------------------------------------------------------
# TeacherLmHeadStore: per-process lm_head residency manager (MOPD)
# ---------------------------------------------------------------------------


class TeacherLmHeadStore:
    """Registry of teacher lm_head vocab shards for one engine process.

    Residency policies (``full_vocab_lm_head_residency``):

    - ``"cpu"`` (default, recommended for MOPD): shards live on CPU (pinned when
      supported); forward/backward stream fp32 vocab chunks on-device. Device peak
      per teacher is ``O(chunk_vocab * H)``.
    - ``"gpu"``: shards are cached on device under an LRU of
      ``max_resident_teachers``; eviction happens **before** inserting a new shard,
      so at most that many teacher shards are ever resident. Backward still streams
      from the CPU copy so evicted shards never linger via autograd references.
    """

    def __init__(
        self,
        *,
        teacher_checkpoints: dict[str, str],
        vocab_size_padded: int,
        tp_rank: int = 0,
        tp_size: int = 1,
        layer: str = "auto",
        residency: str = "cpu",
        max_resident_teachers: int = 1,
        dtype: Optional[torch.dtype] = None,
        pin_memory: bool = True,
    ):
        if residency not in ("cpu", "gpu"):
            raise ValueError(f"full_vocab_lm_head_residency must be 'cpu' or 'gpu', got {residency!r}.")
        if max_resident_teachers < 1:
            raise ValueError(f"full_vocab_max_resident_teachers must be >= 1, got {max_resident_teachers}.")
        if not teacher_checkpoints:
            raise ValueError("TeacherLmHeadStore requires at least one teacher checkpoint.")
        self._teacher_checkpoints = dict(teacher_checkpoints)
        self._vocab_size_padded = vocab_size_padded
        self._tp_rank = tp_rank
        self._tp_size = tp_size
        self._layer = layer
        self._residency = residency
        self._max_resident = max_resident_teachers
        self._dtype = dtype
        self._pin_memory = pin_memory
        self._cpu_shards: dict[str, torch.Tensor] = {}
        self._gpu_cache: OrderedDict[str, torch.Tensor] = OrderedDict()

    @property
    def teacher_keys(self) -> list[str]:
        return sorted(self._teacher_checkpoints)

    @property
    def vocab_shard_size(self) -> int:
        return self._vocab_size_padded // self._tp_size

    def cpu_shard(self, teacher_key: str) -> torch.Tensor:
        """Return the teacher's CPU (pinned) vocab shard, loading it lazily."""
        shard = self._cpu_shards.get(teacher_key)
        if shard is None:
            if teacher_key not in self._teacher_checkpoints:
                raise KeyError(
                    f"unknown teacher key {teacher_key!r}; configured teachers: {sorted(self._teacher_checkpoints)}"
                )
            logger.info(
                "[FVKL] loading teacher lm_head for %r from %s (tp shard %d/%d)",
                teacher_key,
                self._teacher_checkpoints[teacher_key],
                self._tp_rank,
                self._tp_size,
            )
            weight = load_teacher_lm_head(self._teacher_checkpoints[teacher_key], self._layer, self._dtype)
            shard = shard_lm_head(weight, self._vocab_size_padded, self._tp_rank, self._tp_size)
            del weight
            if self._pin_memory:
                try:
                    shard = shard.pin_memory()
                except (RuntimeError, TypeError):
                    # pin_memory is unsupported on some backends (e.g. certain NPU
                    # builds); fall back to a regular CPU tensor (slower H2D only).
                    pass
            self._cpu_shards[teacher_key] = shard
        return shard

    @contextmanager
    def acquire(self, teacher_key: str, device: torch.device) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield ``(weight_fwd, weight_bwd)`` for one teacher pass.

        ``weight_fwd`` is the shard used by the forward pass (device-resident under
        the ``"gpu"`` policy, CPU under ``"cpu"``); ``weight_bwd`` is always the CPU
        shard re-streamed during backward. With the ``"gpu"`` policy, inserting a new
        shard first evicts down to ``max_resident_teachers`` so the resident budget
        (default 1) is never exceeded.
        """
        cpu_shard = self.cpu_shard(teacher_key)
        if self._residency == "cpu":
            yield cpu_shard, cpu_shard
            return

        dev_shard = self._gpu_cache.get(teacher_key)
        if dev_shard is None:
            while len(self._gpu_cache) >= self._max_resident:
                evicted_key, evicted = self._gpu_cache.popitem(last=False)
                del evicted
                logger.info("[FVKL] evicted teacher lm_head shard %r from device (LRU)", evicted_key)
                get_torch_device().empty_cache()
            dev_shard = cpu_shard.to(device, non_blocking=True)
            self._gpu_cache[teacher_key] = dev_shard
        else:
            self._gpu_cache.move_to_end(teacher_key)
        yield dev_shard, cpu_shard

    def release_all(self) -> None:
        """Drop every device-resident shard (CPU shards are kept)."""
        if self._gpu_cache:
            self._gpu_cache.clear()
            get_torch_device().empty_cache()


_PROCESS_STORE: Optional[TeacherLmHeadStore] = None


def get_teacher_lm_head_store(
    distillation_config,
    *,
    vocab_size_padded: int,
    tp_rank: int = 0,
    tp_size: int = 1,
) -> TeacherLmHeadStore:
    """Return the process-global :class:`TeacherLmHeadStore`, building it lazily.

    Engine processes call this from the full-vocab loss entry with their student
    vocab layout; the store is created once per process and reused across steps.
    """
    global _PROCESS_STORE
    loss_config = distillation_config.distillation_loss
    if _PROCESS_STORE is None:
        _PROCESS_STORE = TeacherLmHeadStore(
            teacher_checkpoints=resolve_teacher_lm_head_checkpoints(distillation_config),
            vocab_size_padded=vocab_size_padded,
            tp_rank=tp_rank,
            tp_size=tp_size,
            layer=getattr(loss_config, "full_vocab_lm_head_layer", "auto"),
            residency=getattr(loss_config, "full_vocab_lm_head_residency", "cpu"),
            max_resident_teachers=getattr(loss_config, "full_vocab_max_resident_teachers", 1),
        )
    else:
        if (
            _PROCESS_STORE._vocab_size_padded != vocab_size_padded
            or _PROCESS_STORE._tp_rank != tp_rank
            or _PROCESS_STORE._tp_size != tp_size
        ):
            raise RuntimeError(
                "TeacherLmHeadStore was already built with a different vocab/TP layout "
                f"(vocab={_PROCESS_STORE._vocab_size_padded}, tp={_PROCESS_STORE._tp_rank}/"
                f"{_PROCESS_STORE._tp_size}) than requested (vocab={vocab_size_padded}, "
                f"tp={tp_rank}/{tp_size})."
            )
    return _PROCESS_STORE


# ---------------------------------------------------------------------------
# Artifact grouping and TQ fetch helpers (shared MOPD scheduling logic)
# ---------------------------------------------------------------------------


def normalize_artifacts(raw: Any, batch_size: int) -> list[Optional[dict[str, Any]]]:
    """Normalize the per-sample artifact column of a micro batch to a list of dicts/None.

    The column arrives as a tensordict ``NonTensorStack`` (micro-batch slice), a numpy
    object array, or a plain list of per-sample artifact dicts; rows without a
    full-vocab artifact (e.g. validate samples) are None.
    """
    if raw is None:
        raise KeyError(
            "micro batch has no 'teacher_full_vocab_artifact' column: full-vocab distillation "
            "requires the rollout phase to export teacher hidden states (loss_mode "
            "forward_kl_full_vocab/reverse_kl_full_vocab)."
        )
    from tensordict import NonTensorData

    items = [item.data if isinstance(item, NonTensorData) else item for item in list(raw)]
    if len(items) != batch_size:
        raise RuntimeError(
            f"teacher_full_vocab_artifact column has {len(items)} entries but the micro batch "
            f"has {batch_size} rows."
        )
    out: list[Optional[dict[str, Any]]] = []
    for item in items:
        out.append(item if isinstance(item, dict) else None)
    return out


def group_artifacts_by_teacher(artifacts: list[Optional[dict[str, Any]]]) -> dict[str, list[int]]:
    """Group row indices by teacher name; keys are sorted so every rank iterates teachers in the same order."""
    groups: dict[str, list[int]] = {}
    for i, artifact in enumerate(artifacts):
        if artifact is None:
            continue
        groups.setdefault(artifact["teacher_name"], []).append(i)
    return {key: groups[key] for key in sorted(groups)}


def fetch_hidden_rows(
    artifacts: list[Optional[dict[str, Any]]],
    seq_lens: list[int],
    hidden_size: Optional[int] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[list[torch.Tensor], list[int], list[str]]:
    """Fetch every covered row's hidden states from TQ; zero-fill uncovered rows.

    Returns:
        rows: list of ``[S_i, H]`` CPU tensors (zeros for rows without artifact).
        teacher_index: per-row group id (``-1`` for uncovered rows).
        teacher_keys: sorted teacher names; group id indexes into this list.
    """
    groups = group_artifacts_by_teacher(artifacts)
    teacher_keys = sorted(groups)
    gid_of = {key: g for g, key in enumerate(teacher_keys)}

    rows: list[Optional[torch.Tensor]] = [None] * len(artifacts)
    teacher_index = [-1] * len(artifacts)
    if hidden_size is None:
        for key in teacher_keys:
            hidden_size = artifacts[groups[key][0]]["hidden_size"]
            break
    for key in teacher_keys:
        idx = groups[key]
        fetched = full_vocab_tq.fetch_hidden_batch([artifacts[i] for i in idx])
        for i, hidden in zip(idx, fetched, strict=True):
            if hidden.shape[0] != seq_lens[i]:
                raise RuntimeError(
                    f"row {i}: teacher hidden length {hidden.shape[0]} != sample token length "
                    f"{seq_lens[i]} (teacher={key!r}, uid={artifacts[i].get('uid')!r}). The teacher "
                    "prefill must cover exactly prompt+response tokens."
                )
            rows[i] = hidden
            teacher_index[i] = gid_of[key]
    if any(row is None for row in rows):
        if hidden_size is None:
            # No artifact in the whole micro batch (defensive case): every row is
            # uncovered, so no teacher pass will run and the hidden values are never
            # read. A dummy width keeps the nested-tensor construction well-formed.
            hidden_size = 1
        for i in range(len(rows)):
            if rows[i] is None:
                rows[i] = torch.zeros(seq_lens[i], hidden_size, dtype=dtype)
    return rows, teacher_index, teacher_keys


# ---------------------------------------------------------------------------
# Shared per-teacher pass scheduler (MOPD core)
# ---------------------------------------------------------------------------


def run_teacher_passes(
    z_cpu: torch.Tensor,
    teacher_idx: torch.Tensor,
    teacher_keys: list[str],
    student_logits: torch.Tensor,
    *,
    store: TeacherLmHeadStore,
    loss_config,
    reverse: bool,
    kernel_fwd,
    kernel_rev,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token full-vocab KL for one sequence-split micro batch, teacher by teacher.

    Args:
        z_cpu: ``[N, H]`` teacher hidden states on CPU, already split to this
            rank's token shard (CP/SP), with zero rows for unsupervised positions.
        teacher_idx: ``[N]`` int64 on CPU; group id into ``teacher_keys``, ``-1``
            for positions without a teacher artifact (kept at loss 0).
        teacher_keys: group id -> teacher name (sorted; identical iteration order
            on every rank, which the TP collectives inside the kernels rely on).
        student_logits: ``[N, V_shard]`` on device (Megatron: vocab shard; FSDP: full V).
        store: the process-local :class:`TeacherLmHeadStore`.
        loss_config: ``DistillationLossConfig`` (chunk sizes, temperature, clamp).
        reverse: compute ``KL(p_S||p_T)`` instead of ``KL(p_T||p_S)``.
        kernel_fwd/kernel_rev: autograd.Function ``apply``-ables with the common
            signature ``(z, w_fwd, w_bwd, s, chunk_vocab, temperature[, clamp])``.

    Returns:
        kl: ``[N]`` fp32 on device (gradients flow to ``student_logits``).
        hidden_norm: ``[N]`` fp32 on device, detached (alignment diagnostics).

    Memory behaviour: at most ``max_resident_teachers`` lm_head shards are on
    device at any moment; each teacher's tokens are processed in blocks of
    ``full_vocab_chunk_tokens`` (capped by ``full_vocab_max_tokens_per_pass``),
    and each block streams the lm_head in ``full_vocab_chunk_vocab`` chunks, so
    peak memory is decoupled from both the batch length and the vocab size.
    """
    device = student_logits.device
    n = student_logits.shape[0]
    chunk_vocab = int(getattr(loss_config, "full_vocab_chunk_vocab", DEFAULT_CHUNK_VOCAB))
    chunk_tokens = int(getattr(loss_config, "full_vocab_chunk_tokens", DEFAULT_CHUNK_TOKENS))
    max_per_pass = getattr(loss_config, "full_vocab_max_tokens_per_pass", None)
    if max_per_pass:
        chunk_tokens = min(chunk_tokens, int(max_per_pass))
    temperature = float(getattr(loss_config, "kd_temperature", 1.0))
    clamp = getattr(loss_config, "log_prob_min_clamp", None)

    kl_pos: list[torch.Tensor] = []
    kl_val: list[torch.Tensor] = []
    norm = torch.zeros(n, dtype=torch.float32)

    for gid, teacher_key in enumerate(teacher_keys):
        pos = (teacher_idx == gid).nonzero(as_tuple=True)[0]
        if pos.numel() == 0:
            continue
        with store.acquire(teacher_key, device) as (w_fwd, w_bwd):
            for s in range(0, pos.numel(), chunk_tokens):
                rows = pos[s : s + chunk_tokens]
                z_block = z_cpu.index_select(0, rows)  # [c, H] CPU (autograd-saved)
                s_block = student_logits.index_select(0, rows.to(device))  # [c, V_shard]
                if reverse:
                    kl_block = kernel_rev(z_block, w_fwd, w_bwd, s_block, chunk_vocab, temperature)
                else:
                    kl_block = kernel_fwd(z_block, w_fwd, w_bwd, s_block, chunk_vocab, temperature, clamp)
                kl_pos.append(rows)
                kl_val.append(kl_block)
                norm[rows] = z_block.float().norm(dim=-1)

    kl = torch.zeros(n, dtype=torch.float32, device=device)
    if kl_val:
        kl = kl.index_copy(0, torch.cat(kl_pos).to(device), torch.cat(kl_val))
    return kl, norm.to(device)
