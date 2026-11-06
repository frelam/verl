"""Fixture package for the ``agentic_rl_data`` reward tests.

The builders live in :mod:`fixtures.reward_fixtures`; tests add this tests
directory to ``sys.path`` and import ``fixtures.reward_fixtures`` directly (no
``__init__``-only tricks needed beyond this marker).
"""

from __future__ import annotations

__all__ = ["reward_fixtures"]
