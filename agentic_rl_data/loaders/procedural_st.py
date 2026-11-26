"""Thin loader for deterministic single-turn procedural function-calling tasks.

``cfg`` accepts ``n`` / ``max_samples`` (default 1000), ``seed`` (default 0),
``scenarios``, ``scenario_weights`` and ``verify`` (default true).  No raw input
is read; tasks are generated and reward-checked in-process.
"""

from __future__ import annotations

from typing import Any, Iterator

from agentic_rl_data.generators.procedural_st_tasks import LICENSE, SOURCE, generate_tasks
from agentic_rl_data.loaders.common import get_stats


def load_tasks(cfg: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    cfg = dict(cfg or {})
    n = int(cfg.get("max_samples") or cfg.get("n") or 1000)
    stats: dict[str, Any] = {}
    source_stats = get_stats(cfg)
    for task in generate_tasks(
        n,
        seed=int(cfg.get("seed") or 0),
        scenarios=cfg.get("scenarios"),
        scenario_weights=cfg.get("scenario_weights"),
        verify=bool(cfg.get("verify", True)),
        stats=stats,
    ):
        if source_stats is not None:
            source_stats.row()
        yield task


__all__ = ["SOURCE", "LICENSE", "load_tasks"]
