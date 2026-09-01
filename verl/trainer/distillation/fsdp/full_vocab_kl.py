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

"""Full-vocab KL distillation entry for the FSDP/VeOmni backend.

FSDP keeps the full vocabulary on every rank (Ulysses SP splits the sequence
axis instead), so the plain :class:`FullVocabForwardKL` / :class:`FullVocabReverseKL`
kernels apply directly: the teacher lm_head (full ``[V, H]``) is streamed from
CPU-pinned memory in vocab chunks, and no cross-rank communication is needed.

Multi-teacher (MOPD) scheduling lives in
:func:`verl.trainer.distillation.full_vocab_kl.run_teacher_passes`.
"""

import torch

from verl.trainer.distillation import full_vocab_kl as fvkl
from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size, slice_input_tensor
from verl.workers.config import DistillationConfig


def compute_full_vocab_kl(
    student_logits: torch.Tensor,
    data,
    config,
    distillation_config: DistillationConfig,
    data_format: str,
    reverse: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute per-token full-vocab KL for one FSDP micro batch.

    Args:
        student_logits: ``(1, total/sp, V)`` remove-padded student logits.
        data: micro batch TensorDict with ``input_ids`` (nested) and the
            ``teacher_full_vocab_artifact`` non-tensor column.
        reverse: compute ``KL(p_S||p_T)`` instead of ``KL(p_T||p_S)``.

    Returns:
        ``distillation_losses`` / ``teacher_hidden_norm`` / ``teacher_hidden_coverage``,
        each ``(1, total/sp)``.
    """
    del config, data_format  # FSDP path needs neither; kept for a uniform signature.
    loss_config = distillation_config.distillation_loss
    device = student_logits.device

    input_ids = data["input_ids"]
    assert input_ids.is_nested, "input_ids must be nested (use_remove_padding) for full-vocab distillation"
    batch_size = input_ids.shape[0]
    artifacts = fvkl.normalize_artifacts(data["teacher_full_vocab_artifact"], batch_size)
    seq_lens = [int(t.shape[0]) for t in input_ids.unbind()]

    rows, teacher_index, teacher_keys = fvkl.fetch_hidden_rows(artifacts, seq_lens)

    # [1, total, H] CPU; pad positions (from the SP slice) carry teacher group 0
    # with zero hidden — their loss is masked out downstream by response_mask.
    hidden = torch.nested.nested_tensor(rows).values().unsqueeze(0)
    idx = torch.cat(
        [torch.full((seq_len,), g, dtype=torch.int64) for seq_len, g in zip(seq_lens, teacher_index, strict=True)]
    ).unsqueeze(0)
    if get_ulysses_sequence_parallel_world_size() > 1:
        hidden = slice_input_tensor(hidden, dim=1)
        idx = slice_input_tensor(idx, dim=1)
    assert hidden.shape[:2] == student_logits.shape[:2], (
        f"teacher hidden {tuple(hidden.shape[:2])} does not match student logits "
        f"{tuple(student_logits.shape[:2])} after SP split"
    )

    n = hidden.shape[1]
    store = fvkl.get_teacher_lm_head_store(
        distillation_config,
        vocab_size_padded=student_logits.shape[-1],
        tp_rank=0,
        tp_size=1,
    )
    kl, norm = fvkl.run_teacher_passes(
        hidden[0],
        idx[0],
        teacher_keys,
        student_logits.reshape(n, student_logits.shape[-1]),
        store=store,
        loss_config=loss_config,
        reverse=reverse,
        kernel_fwd=fvkl.FullVocabForwardKL.apply,
        kernel_rev=fvkl.FullVocabReverseKL.apply,
    )
    coverage = (idx[0] >= 0).to(torch.float32).to(device)
    return {
        "distillation_losses": kl.view(1, n),
        "teacher_hidden_norm": norm.view(1, n),
        "teacher_hidden_coverage": coverage.view(1, n),
    }
