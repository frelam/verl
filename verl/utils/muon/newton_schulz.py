import torch


def newton_schulz(M: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    if M.ndim != 2:
        raise ValueError(f"Newton-Schulz requires 2D input, got {M.ndim}D")

    a, b, c = (3.4445, -4.7750, 2.0315)

    original_dtype = M.dtype
    X = M.bfloat16()
    X = X / (X.norm() + eps)

    transpose = M.size(0) > M.size(1)
    if transpose:
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transpose:
        X = X.T

    return X.to(original_dtype)
