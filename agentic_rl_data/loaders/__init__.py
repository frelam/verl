"""Loader registry: ``SOURCE`` -> ``load_tasks(cfg)`` per raw source.

Contract (``schema.py``): every ``loaders/<source>.py`` module exposes
``SOURCE: str`` and ``load_tasks(cfg: dict | None = None) -> Iterator[dict]``
yielding schema-valid tasks.  This package adds one small convenience layer:

    from agentic_rl_data.loaders import load_source
    tasks = list(load_source("funreason_mt", {"path": "data/funreason_mt.jsonl"}))

Modules are imported lazily so an optional dependency failure inside one
loader (``datasets`` / ``pyarrow``) does not break importing the package.

Download policy: a loader with an empty cfg falls back to its documented
``hf_dataset`` id and will only touch the network if ``datasets`` can fetch it
(``LoaderError`` otherwise).  Pass explicit local paths to stay offline.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Iterator, Mapping

logger = logging.getLogger(__name__)

#: Registry of source name -> module name under ``agentic_rl_data.loaders``.
SOURCE_REGISTRY: dict[str, str] = {
    # Existing single-turn sources (reuse the legacy prepare_data parsers).
    "apigen": "apigen",
    "sealtools": "sealtools",
    "toolace": "toolace",
    "hammer": "hammer",
    "apibank": "apibank",
    # Search / multi-hop QA.
    "search_r1": "search_r1",
    "asearcher": "asearcher",
    "a2search": "a2search",
    # Multi-turn function calling (SFT traces -> replay env).
    "funreason_mt": "funreason_mt",
    "apigen_mt": "apigen_mt",
    "toolmind": "toolmind",
    "toucan": "toucan",
    # Other task types.
    "when2call": "when2call",
    "retool": "retool",
    "code_exec": "code_exec",
}

#: Task type produced by each source (used by build_raw / reports).
SOURCE_TASK_TYPE: dict[str, str] = {
    "apigen": "st_fc",
    "sealtools": "st_fc",
    "toolace": "st_fc",
    "hammer": "irrelevance|st_fc",
    "apibank": "st_fc",
    "search_r1": "search_qa",
    "asearcher": "search_qa",
    "a2search": "search_qa",
    "funreason_mt": "mt_fc",
    "apigen_mt": "mt_fc",
    "toolmind": "mt_fc",
    "toucan": "mt_fc",
    "when2call": "st_fc|abstain",
    "retool": "math_ci",
    "code_exec": "code_exec",
}


def available_sources() -> list[str]:
    """Return the registered source names."""
    return sorted(SOURCE_REGISTRY)


def get_loader_module(source: str):
    if source not in SOURCE_REGISTRY:
        raise KeyError(f"unknown source {source!r}; available: {available_sources()}")
    return importlib.import_module(f"agentic_rl_data.loaders.{SOURCE_REGISTRY[source]}")


def load_source(source: str, cfg: Mapping[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Load one source with ``cfg``; yields schema-valid tasks."""
    module = get_loader_module(source)
    loader = getattr(module, "load_tasks", None)
    if not callable(loader):
        raise TypeError(f"loader module {module.__name__} does not expose load_tasks()")
    return loader(dict(cfg or {}))


def iter_sources(
    cfg_map: Mapping[str, Mapping[str, Any]] | None = None, sources: list[str] | None = None
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(source, task)`` pairs for all (or the given) registered sources.

    Loader failures (missing raw path, optional dependency) are logged and
    skipped so a partial local checkout still builds what it can.
    """
    cfg_map = dict(cfg_map or {})
    for source in sources or available_sources():
        try:
            for task in load_source(source, cfg_map.get(source)):
                yield source, task
        except Exception as exc:  # noqa: BLE001 - one source must not kill the run
            logger.warning("source %s skipped: %s", source, exc)


__all__ = [
    "SOURCE_REGISTRY",
    "SOURCE_TASK_TYPE",
    "available_sources",
    "get_loader_module",
    "load_source",
    "iter_sources",
]
