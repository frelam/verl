"""Toucan-1.5M loader (real-world MCP tool returns, plan §1.1).

Verified: official repo ``TheAgentArk/Toucan`` (MIT) describes "Synthesizing
1.5M Tool-Agentic Data from Real-World MCP Environments"; the HF id
``Agent-Ark/Toucan-1.5M`` is ``[UNVERIFIED]`` for this sandbox (hub unreachable,
org naming differs: GitHub ``TheAgentArk`` vs HF ``Agent-Ark``).

This is the **only multi-turn source expected to ship genuine tool outputs**,
hence the canonical real-return provider for replay envs.  The loader keeps
``reference.tool_returns`` when the record carries tool/observation turns and
marks ``meta.gt_returns_source == "reference"``; otherwise it explicitly
falls back to placeholder.

The plan says "only subsample": use ``cfg["max_samples"]`` (default 50k cap is
a safety limit in the report; pass 0 for all).
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

from agentic_rl_data.loaders.common import iter_source_records, load_mt_tasks_from_records

logger = logging.getLogger(__name__)

SOURCE = "toucan"
LICENSE = "mit (repo verified 2026-11-05); dataset card unreachable[UNVERIFIED]"
_DEFAULT_HF_DATASET = "Agent-Ark/Toucan-1.5M"
_DEFAULT_MAX_SAMPLES = 50_000


def load_tasks(cfg: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    cfg = dict(cfg or {})
    if not any(key in cfg for key in ("records", "path", "raw_path", "hf_dataset", "dataset")):
        cfg["hf_dataset"] = _DEFAULT_HF_DATASET
    cfg.setdefault("max_samples", _DEFAULT_MAX_SAMPLES)
    records = iter_source_records(cfg, SOURCE)
    yield from load_mt_tasks_from_records(
        records, source=SOURCE, cfg=cfg, default_license=cfg.get("license", LICENSE)
    )


__all__ = ["SOURCE", "LICENSE", "load_tasks"]
