"""APIGen-MT-5k loader (multi-turn trajectories with user interaction, plan §1.1).

Source id ``Salesforce/APIGen-MT-5k`` comes from the plan; it is
``[UNVERIFIED]`` for this sandbox because the HF hub is unreachable and no
official GitHub repo with that name was found.  Parsing is tolerant: any
record carrying a ``messages`` / ``conversations`` list + a ``tools`` field is
understood (OpenAI style and ``from/value`` style alike).

Returns follow the same rule as every multi-turn loader: real observations ->
``reference.tool_returns`` + ``meta.gt_returns_source == "reference"``;
otherwise explicit placeholder, never fabricated content.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

from agentic_rl_data.loaders.common import iter_source_records, load_mt_tasks_from_records

logger = logging.getLogger(__name__)

SOURCE = "apigen_mt"
LICENSE = "unreachable[UNVERIFIED]"
_DEFAULT_HF_DATASET = "Salesforce/APIGen-MT-5k"


def load_tasks(cfg: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    cfg = dict(cfg or {})
    if not any(key in cfg for key in ("records", "path", "raw_path", "hf_dataset", "dataset")):
        cfg["hf_dataset"] = _DEFAULT_HF_DATASET
    records = iter_source_records(cfg, SOURCE)
    yield from load_mt_tasks_from_records(
        records, source=SOURCE, cfg=cfg, default_license=cfg.get("license", LICENSE)
    )


__all__ = ["SOURCE", "LICENSE", "load_tasks"]
