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

"""TransferQueue transport for full-vocab OPD teacher hidden states.

Full-vocab KL distillation ships the teacher's pre-lm_head hidden states (not the
full logits) from the teacher rollout servers to the student training workers
through TransferQueue (TQ):

- One TQ **partition per (experiment prefix, teacher, global step)** so the driver
  can release a whole step's storage once the actor update for that step finished.
- One TQ **key per sample**: ``{teacher}/step={step}/sample={uid}``.
- Only a small **artifact** metadata dict rides with the batch (as a non-tensor
  column); the student fetches the tensor on demand inside the loss computation.

The ``transfer_queue`` package is an optional dependency at import time of this
module; every public function imports it lazily and raises a descriptive error
when full-vocab distillation is used without it.
"""

import logging
import os
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

HIDDEN_FIELD = "hidden_states"

PARTITION_NAME_TEMPLATE = "full_vocab_hidden_{prefix}_{teacher}_step_{step}"
SAMPLE_KEY_TEMPLATE = "{teacher}/step={step}/sample={uid}"


def resolve_partition_prefix(experiment_name: Optional[str]) -> str:
    """Resolve the TQ partition prefix isolating this run's hidden-state partitions.

    Priority: explicit ``full_vocab_experiment_name`` config, then the
    ``VERL_FULL_VOCAB_EXPERIMENT_NAME`` env var, then ``"default_exp"``.
    """
    if experiment_name:
        return str(experiment_name)
    return os.environ.get("VERL_FULL_VOCAB_EXPERIMENT_NAME", "default_exp")


def partition_name(prefix: str, teacher_name: str, step: int) -> str:
    return PARTITION_NAME_TEMPLATE.format(prefix=prefix, teacher=teacher_name, step=step)


def sample_key(teacher_name: str, step: int, uid: str) -> str:
    return SAMPLE_KEY_TEMPLATE.format(teacher=teacher_name, step=step, uid=uid)


def _tq():
    try:
        import transfer_queue as tq
    except ImportError as exc:
        raise RuntimeError(
            "Full-vocab distillation requires the `transfer_queue` package to transport teacher "
            "hidden states, but it is not importable in this process. Install TransferQueue and "
            "make sure `tq.init(...)` ran on this node before enabling "
            "distillation loss_mode='forward_kl_full_vocab'/'reverse_kl_full_vocab'."
        ) from exc
    return tq


def build_artifact(
    *,
    teacher_name: str,
    step: int,
    uid: str,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    prefix: str,
) -> dict[str, Any]:
    """Build the per-sample artifact metadata dict that travels with the batch."""
    return {
        "teacher_name": teacher_name,
        "step": int(step),
        "uid": str(uid),
        "key": sample_key(teacher_name, step, uid),
        "partition_id": partition_name(prefix, teacher_name, step),
        "seq_len": int(seq_len),
        "hidden_size": int(hidden_size),
        "dtype": str(dtype),
    }


def put_hidden(
    hidden: torch.Tensor,
    *,
    teacher_name: str,
    step: int,
    uid: str,
    prefix: str,
    seq_len: Optional[int] = None,
) -> dict[str, Any]:
    """Write one sample's pre-lm_head hidden states ``[S, H]`` to TQ and return the artifact.

    The tensor is moved to CPU before the put (TQ storage lives on host memory / disk,
    never on accelerator). ``seq_len`` pins the expected capture length: the prefill
    forward must produce exactly one hidden vector per prompt token.
    """
    if hidden.dim() != 2:
        raise ValueError(f"expected a 2-D [S, H] hidden-state tensor, got shape {tuple(hidden.shape)}")
    if seq_len is not None and hidden.shape[0] != seq_len:
        raise RuntimeError(
            f"captured hidden length {hidden.shape[0]} != expected seq_len {seq_len}: the teacher "
            "prefill was likely chunked (enable_chunked_prefill / max_num_batched_tokens < "
            "max_model_len), so only the tail chunk was captured. Refusing to export a truncated "
            "hidden state."
        )
    tq = _tq()
    from tensordict import TensorDict

    artifact = build_artifact(
        teacher_name=teacher_name,
        step=step,
        uid=uid,
        seq_len=hidden.shape[0],
        hidden_size=hidden.shape[1],
        dtype=hidden.dtype,
        prefix=prefix,
    )
    fields = TensorDict({HIDDEN_FIELD: hidden.detach().cpu().unsqueeze(0)}, batch_size=[1])
    tq.kv_batch_put(
        keys=[artifact["key"]],
        fields=fields,
        tags=[{"global_steps": int(step)}],
        partition_id=artifact["partition_id"],
    )
    return artifact


def fetch_hidden_batch(artifacts: list[dict[str, Any]]) -> list[torch.Tensor]:
    """Fetch many samples' hidden states from TQ, preserving the input order.

    Keys are grouped by ``partition_id`` so each partition is read with a single
    ``kv_batch_get`` call (one teacher+step group shares a partition). Every
    returned tensor stays on CPU; the caller decides which rows to move on-device.
    """
    if not artifacts:
        return []
    tq = _tq()
    by_partition: dict[str, list[int]] = {}
    for i, artifact in enumerate(artifacts):
        by_partition.setdefault(artifact["partition_id"], []).append(i)

    out: list[Optional[torch.Tensor]] = [None] * len(artifacts)
    for partition, indices in by_partition.items():
        keys = [artifacts[i]["key"] for i in indices]
        data = tq.kv_batch_get(keys=keys, partition_id=partition, select_fields=[HIDDEN_FIELD])
        hidden = data[HIDDEN_FIELD]
        for row, i in enumerate(indices):
            artifact = artifacts[i]
            h = hidden[row]
            if h.shape[0] != artifact["seq_len"] or h.shape[1] != artifact["hidden_size"]:
                raise RuntimeError(
                    f"fetched hidden shape {tuple(h.shape)} does not match artifact "
                    f"(seq_len={artifact['seq_len']}, hidden_size={artifact['hidden_size']}, "
                    f"key={artifact['key']!r}): the TQ entry was overwritten or corrupted."
                )
            out[i] = h
    return [h for h in out]


def fetch_hidden_single(artifact: dict[str, Any], device: Optional[torch.device] = None) -> torch.Tensor:
    """Fetch one sample's hidden states ``[S, H]`` from TQ by its artifact.

    ``device=None`` keeps the tensor on CPU; the caller decides when to pay the
    host-to-device copy (only the rows/chunks actually consumed are moved).
    """
    tq = _tq()
    data = tq.kv_batch_get(keys=[artifact["key"]], partition_id=artifact["partition_id"], select_fields=[HIDDEN_FIELD])
    hidden = data[HIDDEN_FIELD][0]
    if hidden.shape[0] != artifact["seq_len"] or hidden.shape[1] != artifact["hidden_size"]:
        raise RuntimeError(
            f"fetched hidden shape {tuple(hidden.shape)} does not match artifact "
            f"(seq_len={artifact['seq_len']}, hidden_size={artifact['hidden_size']}, "
            f"key={artifact['key']!r}): the TQ entry was overwritten or corrupted."
        )
    if device is not None:
        hidden = hidden.to(device)
    return hidden


def clear_step(prefix: str, teacher_name: str, step: int) -> int:
    """Best-effort cleanup of one step's hidden-state partition for one teacher.

    Returns the number of keys cleared. Failures only log a warning: a stale
    partition wastes TQ storage but must not abort the training loop.
    """
    tq = _tq()
    partition = partition_name(prefix, teacher_name, step)
    try:
        listing = tq.kv_list(partition)
        if not listing:
            return 0
        keys = list(listing.get(partition, {}).keys())
        if not keys:
            return 0
        tq.kv_clear(keys=keys, partition_id=partition)
        return len(keys)
    except Exception as exc:  # noqa: BLE001 - best effort cleanup
        logger.warning(
            "[FVKL-TQ] failed to clear full-vocab hidden partition %r (teacher=%r, step=%d): %s",
            partition,
            teacher_name,
            step,
            exc,
        )
        return 0
