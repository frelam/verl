import math
from typing import Iterable, Optional

import torch

from verl.utils.muon.newton_schulz import newton_schulz


class Muon(torch.optim.Optimizer):
    """Muon optimizer: MomentUm Orthogonalized by Newton-Schulz.

    Muon optimizes 2D weight matrices by:
    1. Accumulating gradient momentum (SGD with Nesterov momentum)
    2. Orthogonalizing the momentum via Newton-Schulz iteration
    3. Applying the orthogonalized update with consistent RMS scaling

    Reference: https://kellerjordan.github.io/posts/muon/
    MoonshotAI scaling: https://github.com/MoonshotAI/Moonlight

    Args:
        params: Iterable of 2D parameters to optimize.
        lr: Learning rate.
        momentum: Momentum factor (Nesterov-style).
        ns_steps: Number of Newton-Schulz iteration steps.
        weight_decay: Weight decay coefficient (decoupled, like AdamW).
        ns_eps: Epsilon for Newton-Schulz normalization.
        rms_scale: Scaling factor for consistent RMS updates.
            When > 0, the update is scaled by ``rms_scale * sqrt(max(m, n))``
            to match AdamW's update RMS across different matrix shapes.
            Default 0.2 (from MoonshotAI).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 2e-3,
        momentum: float = 0.95,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        ns_eps: float = 1e-7,
        rms_scale: float = 0.2,
        **kwargs,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            ns_eps=ns_eps,
            rms_scale=rms_scale,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            ns_eps = group["ns_eps"]
            rms_scale = group["rms_scale"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.ndim < 2:
                    continue

                g = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                update = self._orthogonalize(p, buf, ns_steps, ns_eps)

                if rms_scale > 0:
                    from torch.distributed.tensor import DTensor

                    if isinstance(p, DTensor):
                        full_shape = p._spec.shape
                    else:
                        full_shape = p.shape
                    max_dim = max(full_shape[0], full_shape[1])
                    update = update * (rms_scale * math.sqrt(max_dim))

                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)

                p.add_(update, alpha=-lr)
                state["step"] += 1

        return loss

    def _orthogonalize(self, param, momentum, ns_steps, ns_eps):
        from torch.distributed.tensor import DTensor

        is_dtensor = isinstance(momentum, DTensor)

        if is_dtensor:
            full_momentum = momentum.full_tensor()
        else:
            full_momentum = momentum

        full_update = newton_schulz(full_momentum.float(), steps=ns_steps, eps=ns_eps)
        full_update = full_update.to(full_momentum.dtype)

        if is_dtensor:
            local_update = _reshard_update(full_update, param)
        else:
            local_update = full_update

        return local_update


class MuonWithAdamW(torch.optim.Optimizer):
    """Mixed Muon + AdamW optimizer.

    Uses Muon for 2D hidden-layer parameters and AdamW for everything else
    (embeddings, output heads, biases, LayerNorm, etc.).

    This follows the recommended practice from the Muon paper and MoonshotAI:
    - 2D hidden-layer weights -> Muon (orthogonalized momentum)
    - Embeddings, lm_head, scalar params -> AdamW

    Args:
        muon_params: Parameters to optimize with Muon (2D hidden-layer weights).
        adamw_params: Parameters to optimize with AdamW (embeddings, heads, scalars).
        lr: Base learning rate (used by Muon).
        momentum: Momentum factor for Muon.
        ns_steps: Newton-Schulz iteration steps.
        weight_decay: Weight decay (applied to both Muon and AdamW).
        ns_eps: Epsilon for Newton-Schulz normalization.
        rms_scale: RMS scaling factor for Muon (default 0.2).
        adamw_lr: Learning rate for AdamW. If None, uses ``lr``.
        adamw_betas: Betas for AdamW.
        adamw_eps: Epsilon for AdamW.
    """

    def __init__(
        self,
        muon_params: Iterable[torch.nn.Parameter],
        adamw_params: Iterable[torch.nn.Parameter],
        lr: float = 2e-3,
        momentum: float = 0.95,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        ns_eps: float = 1e-7,
        rms_scale: float = 0.2,
        adamw_lr: Optional[float] = None,
        adamw_betas: tuple = (0.9, 0.999),
        adamw_eps: float = 1e-8,
        **kwargs,
    ):
        muon_param_list = list(muon_params)
        adamw_param_list = list(adamw_params)

        if not muon_param_list and not adamw_param_list:
            raise ValueError("At least one of muon_params or adamw_params must be non-empty")

        defaults = dict(lr=lr, weight_decay=weight_decay)

        param_groups = []
        if muon_param_list:
            param_groups.append({"params": muon_param_list, "muon_enabled": True})
        if adamw_param_list:
            param_groups.append({"params": adamw_param_list, "muon_enabled": False})

        if not param_groups:
            param_groups.append({"params": [], "muon_enabled": False})

        super().__init__(param_groups, defaults)

        self.muon = None
        if muon_param_list:
            self.muon = Muon(
                muon_param_list,
                lr=lr,
                momentum=momentum,
                ns_steps=ns_steps,
                weight_decay=weight_decay,
                ns_eps=ns_eps,
                rms_scale=rms_scale,
            )

        self.adamw = None
        if adamw_param_list:
            self.adamw = torch.optim.AdamW(
                adamw_param_list,
                lr=adamw_lr if adamw_lr is not None else lr,
                betas=adamw_betas,
                eps=adamw_eps,
                weight_decay=weight_decay,
            )

        self.param_groups = self._build_param_groups()

    def _build_param_groups(self):
        groups = []
        if self.muon is not None:
            for g in self.muon.param_groups:
                pg = dict(g)
                pg["muon_enabled"] = True
                groups.append(pg)
        if self.adamw is not None:
            for g in self.adamw.param_groups:
                pg = dict(g)
                pg["muon_enabled"] = False
                groups.append(pg)
        return groups

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.muon is not None:
            self.muon.step()
        if self.adamw is not None:
            self.adamw.step()
        return loss

    def zero_grad(self, set_to_none=True):
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=set_to_none)
        if self.adamw is not None:
            self.adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        result = {}
        if self.muon is not None:
            result["muon"] = self.muon.state_dict()
        if self.adamw is not None:
            result["adamw"] = self.adamw.state_dict()
        return result

    def load_state_dict(self, state_dict):
        if self.muon is not None and "muon" in state_dict:
            self.muon.load_state_dict(state_dict["muon"])
        if self.adamw is not None and "adamw" in state_dict:
            self.adamw.load_state_dict(state_dict["adamw"])


def _reshard_update(full_update: torch.Tensor, param) -> torch.Tensor:
    from torch.distributed.tensor import DTensor, distribute_tensor

    if not isinstance(param, DTensor):
        return full_update

    device_mesh = param.device_mesh
    placements = param.placements

    distributed_update = distribute_tensor(full_update, device_mesh, placements)
    return distributed_update.to_local()
