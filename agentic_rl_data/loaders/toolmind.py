"""ToolMind loader (~110k reasoning traces, plan §1.1).

Source id ``Nanbeige/ToolMind`` is ``[UNVERIFIED]`` here (HF hub unreachable;
arXiv 2511.15718 verified, no official GitHub repo found).  The plan explicitly
warns ToolMind overlaps xlam / ToolACE, so **exact + MinHash dedup against the
legacy five sources is mandatory** (``scripts/dedup.py``).

Parsing: generic ``messages`` / ``conversations`` + ``tools`` records.
Reasoning chains are dropped from the prompt (schema roles are system/user
only; the replay env re-derives turns from ``gt_calls``); only the user
questions and tool observations survive.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

from agentic_rl_data.loaders.common import iter_source_records, load_mt_tasks_from_records

logger = logging.getLogger(__name__)

SOURCE = "toolmind"
LICENSE = "unreachable[UNVERIFIED]"
_DEFAULT_HF_DATASET = "Nanbeige/ToolMind"


def load_tasks(cfg: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    cfg = dict(cfg or {})
    if not any(key in cfg for key in ("records", "path", "raw_path", "hf_dataset", "dataset")):
        cfg["hf_dataset"] = _DEFAULT_HF_DATASET
    records = iter_source_records(cfg, SOURCE)
    yield from load_mt_tasks_from_records(
        records, source=SOURCE, cfg=cfg, default_license=cfg.get("license", LICENSE)
    )


__all__ = ["SOURCE", "LICENSE", "load_tasks"]
