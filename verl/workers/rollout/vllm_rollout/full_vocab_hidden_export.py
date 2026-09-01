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

"""Full-vocab OPD teacher hidden-state capture for vLLM rollout workers.

During full-vocab KL distillation the teacher's supervision signal is its
pre-lm_head hidden state ``[S, H]`` (one vector per prompt token), not the
materialized ``[S, V]`` logits. This module implements the teacher-rollout side
of that transport:

- :class:`FullVocabHiddenWorkerExtension` adds two RPCs to every vLLM worker of
  a *teacher* server (selected via ``worker_extension_cls`` when the server is
  started with ``full_vocab_export_config``):

  - ``start_hidden_capture()`` registers a forward pre-hook on the model's
    ``LogitsProcessor`` — the capture point is model-architecture agnostic
    because every causal-LM routes its pre-lm_head hidden through it.
  - ``fetch_captured_hidden()`` returns the captured buffer (moved to CPU) and
    removes the hook, so the GPU tensor is released as soon as the prefill
    finishes.

- :func:`unwrap_captured_hidden` reduces the ``collective_rpc`` fan-out result
  to a single tensor (all TP ranks of one engine capture the same full hidden;
  ``max_num_seqs=1`` guarantees one request in flight per engine).

- :func:`export_hidden_to_tq` validates the capture length against the prompt
  length (failing loudly on chunked prefills) and writes the tensor to
  TransferQueue, returning the per-sample artifact metadata dict.
"""

import logging
import os
from typing import Any, Optional

import torch

from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_CAPTURE_ATTR = "_fv_hidden_capture"
_HANDLE_ATTR = "_fv_hidden_capture_handle"


def _find_logits_processor(model: torch.nn.Module) -> torch.nn.Module:
    """Locate the model's ``LogitsProcessor`` module (the hidden-state capture point)."""
    proc = getattr(model, "logits_processor", None)
    if proc is not None:
        return proc
    # Some multimodal / custom architectures nest the LM head stack; fall back to
    # a module-tree scan matched by type name to stay vllm-version agnostic.
    for module in model.modules():
        if module is model:
            continue
        if type(module).__name__ == "LogitsProcessor":
            return module
    raise RuntimeError(
        f"full-vocab export: no LogitsProcessor found on model {type(model).__name__}; "
        "cannot capture pre-lm_head hidden states for this architecture."
    )


def _extract_hidden(args: tuple, kwargs: dict) -> Optional[torch.Tensor]:
    """Pull the ``hidden_states`` argument out of a LogitsProcessor call.

    Conventional signature is ``forward(lm_head, hidden_states, ...)``, but vLLM
    versions differ, so probe kwargs first, then the conventional position, then
    any floating tensor whose trailing dim matches the lm_head in-features.
    """
    if kwargs:
        hidden = kwargs.get("hidden_states")
        if isinstance(hidden, torch.Tensor):
            return hidden
    if len(args) >= 2 and isinstance(args[1], torch.Tensor) and args[1].is_floating_point():
        return args[1]
    lm_head = args[0] if args else None
    weight = getattr(lm_head, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.dim() == 2:
        hidden_size = weight.shape[1]
        for arg in args[1:]:
            if (
                isinstance(arg, torch.Tensor)
                and arg.is_floating_point()
                and arg.dim() >= 2
                and arg.shape[-1] == hidden_size
            ):
                return arg
    return None


class FullVocabHiddenWorkerExtension(vLLMColocateWorkerExtension):
    """vLLM worker extension adding the hidden-capture RPCs for full-vocab OPD teachers."""

    def start_hidden_capture(self) -> None:
        """Install the capture hook before a prefill-only teacher forward.

        Idempotent: a stale hook from an aborted request is removed first. The
        captured tensor is kept on the worker until :meth:`fetch_captured_hidden`.
        """
        model = self.model_runner.model
        logits_processor = _find_logits_processor(model)

        old_handle = getattr(self, _HANDLE_ATTR, None)
        if old_handle is not None:
            old_handle.remove()

        captured: dict[str, Any] = {"hidden": None, "calls": 0}

        def pre_hook(module, args, kwargs):
            hidden = _extract_hidden(args, kwargs)
            if hidden is None or hidden.dim() < 2:
                return
            hidden = hidden.detach()
            captured["calls"] += 1
            prev = captured["hidden"]
            # Keep the longest capture: the full-prefill call [S, H] must not be
            # overwritten by a 1-token decode step; a (mis-configured) chunked
            # prefill keeps its longest chunk so the seq_len check at export
            # fails loudly instead of exporting a silently truncated hidden.
            if prev is None or hidden.shape[0] > prev.shape[0]:
                captured["hidden"] = hidden

        handle = logits_processor.register_forward_pre_hook(pre_hook, with_kwargs=True)
        setattr(self, _CAPTURE_ATTR, captured)
        setattr(self, _HANDLE_ATTR, handle)

    def fetch_captured_hidden(self) -> Optional[torch.Tensor]:
        """Return the captured hidden ``[S, H]`` (on CPU) and remove the hook.

        Returns ``None`` when no capture happened (e.g. the forward never reached
        the LogitsProcessor); the caller treats that as a hard error.
        """
        captured = getattr(self, _CAPTURE_ATTR, None)
        handle = getattr(self, _HANDLE_ATTR, None)
        if handle is not None:
            handle.remove()
        setattr(self, _CAPTURE_ATTR, None)
        setattr(self, _HANDLE_ATTR, None)
        if captured is None:
            return None
        hidden = captured.get("hidden")
        if hidden is None:
            logger.warning(
                "full-vocab export: capture hook saw %d LogitsProcessor call(s) but no usable "
                "hidden_states argument; returning None.",
                captured.get("calls", 0),
            )
            return None
        if captured.get("calls", 0) > 1:
            logger.warning(
                "full-vocab export: %d LogitsProcessor calls during one request; kept the longest "
                "capture (%d rows). This usually means chunked prefill is enabled, which full-vocab "
                "export forbids — the seq_len check downstream will reject a truncated capture.",
                captured["calls"],
                hidden.shape[0],
            )
        # D2H inside the worker: the RPC return value is pickled back to the
        # driver process, and TQ storage is host-side anyway.
        return hidden.cpu()


def unwrap_captured_hidden(rpc_result: Any) -> Optional[torch.Tensor]:
    """Reduce a ``collective_rpc`` fan-out result to a single hidden tensor.

    ``collective_rpc`` returns one value per worker; TP ranks of one engine
    capture the same full hidden (``max_num_seqs=1`` ⇒ one request in flight),
    so the first non-None entry is authoritative.
    """
    if rpc_result is None:
        return None
    if isinstance(rpc_result, (list, tuple)):
        for item in rpc_result:
            if item is not None:
                return item
        return None
    return rpc_result


def export_hidden_to_tq(
    *,
    hidden: torch.Tensor,
    seq_len: int,
    teacher_name: str,
    step: int,
    uid: str,
    prefix: str,
) -> dict[str, Any]:
    """Validate the capture length and write the hidden state to TransferQueue.

    ``seq_len`` is the number of prompt tokens the engine prefilled; a mismatch
    means the prefill was chunked and the capture is truncated, which raises
    (see :func:`verl.trainer.distillation.full_vocab_tq.put_hidden`). Returns
    the per-sample artifact metadata dict that travels with the batch.
    """
    from verl.trainer.distillation import full_vocab_tq

    return full_vocab_tq.put_hidden(
        hidden,
        teacher_name=teacher_name,
        step=step,
        uid=uid,
        prefix=prefix,
        seq_len=seq_len,
    )
