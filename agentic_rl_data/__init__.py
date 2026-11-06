"""agentic_rl_data — framework-agnostic Agentic RL data pipeline.

See ``README.md`` for layout, ownership and contracts; ``schema.py`` for the
single source of truth. Import ``agentic_rl_data.schema`` directly, e.g.::

    from agentic_rl_data.schema import validate_task, task_dedup_key

Modules are kept import-light (stdlib only at the schema level) so loaders,
rewards, envs and tests run without torch / verl installed.
"""

__all__ = ["schema"]
