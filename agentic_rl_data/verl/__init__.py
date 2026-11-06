"""verl integration for ``agentic_rl_data`` (async agent-loop adapter).

The verl-facing agent loop lives in
:mod:`agentic_rl_data.verl.agent_loop_agentic` and is registered as
``agentic_agent``.  Importing it pulls in verl / torch, so this ``__init__``
stays import-light (only the dependency-free trajectory helpers are eager) and
resolves the loop class lazily::

    from agentic_rl_data.verl import AgenticAgentLoop      # lazy, verl required
    from agentic_rl_data.verl import trajectory             # stdlib only

Hydra loads the loop directly through ``agent_loop_config.yaml``
(``_target_: agentic_rl_data.verl.agent_loop_agentic.AgenticAgentLoop``), which
imports the module and runs the ``@register("agentic_agent")`` side effect.
"""

from __future__ import annotations

from typing import Any

from . import trajectory

__all__ = ["AgenticAgentLoop", "trajectory"]


def __getattr__(name: str) -> Any:
    """Resolve the verl-dependent loop class lazily (keeps this package import-light)."""
    if name == "AgenticAgentLoop":
        from .agent_loop_agentic import AgenticAgentLoop

        return AgenticAgentLoop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
