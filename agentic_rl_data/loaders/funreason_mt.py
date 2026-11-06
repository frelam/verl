"""FunReason-MT loader (multi-turn function calling, plan §1.1 headline source).

Source status (2026-11-05): arXiv 2510.24645 verified; the HF dataset card
(``Bingguang/FunReason-MT``) is **not reachable from this sandbox**, and no
official GitHub repo was found, so the dataset id / license / size
(~16k samples) remain ``[UNVERIFIED]`` -> see ``reports/data_sources.md``.

The paper describes BFCL-style stateful synthetic environments (Environment-API
graph, guided iterative CoT).  Whether the released SFT traces include the
environment observations is unknown, which matters a lot for replay:

* observations present -> ``reference.tool_returns`` is filled and
  ``meta.gt_returns_source == "reference"`` (real replay returns);
* observations missing -> ``reference.tool_returns = None`` and
  ``meta.gt_returns_source == "placeholder"`` (T2 filters / down-weights).

The loader is a thin wrapper over ``common.load_mt_tasks_from_records`` so it
never silently fabricates returns.  ``cfg`` supports ``path`` / ``hf_dataset``,
``max_samples``, ``max_turns``, ``impl``, ``license``.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

from agentic_rl_data.loaders.common import iter_source_records, load_mt_tasks_from_records

logger = logging.getLogger(__name__)

SOURCE = "funreason_mt"
LICENSE = "unreachable[UNVERIFIED]"
_DEFAULT_HF_DATASET = "Bingguang/FunReason-MT"


def load_tasks(cfg: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Yield ``mt_fc`` tasks from FunReason-MT SFT traces."""
    cfg = dict(cfg or {})
    if not any(key in cfg for key in ("records", "path", "raw_path", "hf_dataset", "dataset")):
        cfg["hf_dataset"] = _DEFAULT_HF_DATASET
    records = iter_source_records(cfg, SOURCE)
    yield from load_mt_tasks_from_records(
        records, source=SOURCE, cfg=cfg, default_license=cfg.get("license", LICENSE)
    )


__all__ = ["SOURCE", "LICENSE", "load_tasks"]
