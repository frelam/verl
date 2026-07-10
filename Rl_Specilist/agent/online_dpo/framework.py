"""Agent framework for sandbox-based online DPO training.

No custom framework subclass is needed — the scoring is done inline by
the sandbox runners (``hermes_agent_runner`` / ``claude_code_runner``).
This module re-exports the default ``OpenAICompatibleAgentFramework`` so
configs that reference ``framework_class_fqn`` can still resolve.
"""

from uni_agent.framework.framework import OpenAICompatibleAgentFramework  # noqa: F401
