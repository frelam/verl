"""Environment package for ``agentic_rl_data``.

Only the dependency-free base layer is imported eagerly; concrete
environments (``mock_fc`` / ``search`` / ``python_env`` / ``code_tests``) are
imported lazily by :func:`build_env` so importing this package never pulls in
fastapi / faiss / torch.
"""

from .base import (
    BaseEnv,
    EnvError,
    EnvProtocol,
    NullEnv,
    build_env,
    make_env_state,
    make_tool_result,
    normalise_checks,
    normalize_action,
    truncate_text,
    validate_env_state,
)

__all__ = [
    "BaseEnv",
    "EnvError",
    "EnvProtocol",
    "NullEnv",
    "build_env",
    "make_env_state",
    "make_tool_result",
    "normalise_checks",
    "normalize_action",
    "truncate_text",
    "validate_env_state",
]
