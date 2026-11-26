"""Thin loader for stateful clarification/abstention dialogues.

``cfg``: ``n``/``max_samples`` (default 1200), ``seed``, ``domains``, ``kinds``
(``miss_func``/``miss_param``/``inferable``), ``fault_prob`` (0.25),
``schema_prob`` (0.75), ``slack`` (3), and ``verify`` (true).
"""

from __future__ import annotations

from typing import Any, Iterator

from agentic_rl_data.generators.abstain_tasks import LICENSE, SOURCE, generate_tasks
from agentic_rl_data.loaders.common import get_stats


def load_tasks(cfg: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    cfg = dict(cfg or {})
    n = int(cfg.get("max_samples") or cfg.get("n") or 1200)
    stats: dict[str, Any] = {}
    source_stats = get_stats(cfg)
    for task in generate_tasks(
        n,
        seed=int(cfg.get("seed") or 0),
        domains=cfg.get("domains"),
        kinds=cfg.get("kinds") or cfg.get("scenarios"),
        fault_prob=float(cfg.get("fault_prob", 0.25)),
        schema_prob=float(cfg.get("schema_prob", 0.75)),
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
