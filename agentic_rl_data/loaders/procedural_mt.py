"""Loader wrapper around the procedural stateful task generator.

``cfg``: ``n`` (default 4000), ``seed`` (0), ``domains`` (list, default all),
``level_weights`` ([0.3, 0.4, 0.3]), ``fault_prob`` (0.3), ``slack`` (3).
No raw input: tasks are generated and self-verified in-process
(``agentic_rl_data/generators/stateful_tasks.py``).
"""

from __future__ import annotations

from typing import Any, Iterator

from agentic_rl_data.generators.stateful_tasks import LICENSE, SOURCE, generate_tasks
from agentic_rl_data.loaders.common import get_stats


def load_tasks(cfg: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    cfg = dict(cfg or {})
    n = int(cfg.get("max_samples") or cfg.get("n") or 4000)
    weights = cfg.get("level_weights") or (0.3, 0.4, 0.3)
    stats: dict[str, Any] = {}
    source_stats = get_stats(cfg)
    for task in generate_tasks(
        n,
        seed=int(cfg.get("seed") or 0),
        domains=cfg.get("domains"),
        level_weights=tuple(float(w) for w in weights),
        fault_prob=float(cfg.get("fault_prob", 0.3)),
        slack=int(cfg.get("slack", 3)),
        verify=bool(cfg.get("verify", True)),
        stats=stats,
    ):
        if source_stats is not None:
            source_stats.row()
        yield task
    if source_stats is not None:
        for reason, count in stats.get("rejected", {}).items():
            for _ in range(count):
                source_stats.row()
                source_stats.skip(f"verify:{reason}")


__all__ = ["SOURCE", "LICENSE", "load_tasks"]
