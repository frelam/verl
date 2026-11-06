"""Procedural, self-verified task generators (no downloads, no services).

Each generator emits schema-valid ``mt_fc`` tasks for the stateful
``mock_fc`` env (``envs/mock_fc.py``): the initial state, goal assertions and
a ground-truth call plan are produced together, and every task is replayed
through the real env before it is yielded (``verify=True``), so a task is
never unsolvable or mis-labelled.  See ``generators/stateful_tasks.py``.
"""

from .stateful_tasks import (
    DOMAIN_GENERATORS,
    GeneratedTask,
    generate_tasks,
    verify_task,
)

__all__ = ["DOMAIN_GENERATORS", "GeneratedTask", "generate_tasks", "verify_task"]
