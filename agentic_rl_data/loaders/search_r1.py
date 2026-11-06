"""Search-R1 training-data loader (search QA, plan §1.2).

Verified raw path (GitHub, 2026-11-05): the official
``PeterGriffinJin/Search-R1`` preprocessing script
(``scripts/data_process/nq_search.py``) loads
``RUC-NLPIR/FlashRAG_datasets`` with config ``nq`` and writes
``train.parquet`` / ``test.parquet`` locally.  The documentation's
``PeterJinGo/search-r1`` collection is the model/resource collection, not the
training-data id, so the loader accepts any of:

* ``cfg["path"]`` -- local ``train.parquet`` / JSONL exported from FlashRAG,
* ``cfg["hf_dataset"]`` -- e.g. ``RUC-NLPIR/FlashRAG_datasets`` +
  ``cfg["hf_config"] = "nq"`` (verified path), or ``PeterJinGo/...`` once
  confirmed by the user,
* ``cfg["records"]`` -- inline synthetic records (tests).

Search-R1 has 1-2 hop questions and is trained with a local Wikipedia + E5
retriever; the env config points at a local retriever, never at the internet.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

from agentic_rl_data.loaders.common import (
    SEARCH_SYSTEM,
    SEARCH_TOOLS,
    build_messages,
    build_reference,
    build_task,
    env_search,
    first_user_text,
    gold_answers,
    iter_source_records,
    record_skip,
    safe_task_id,
)

logger = logging.getLogger(__name__)

SOURCE = "search_r1"
LICENSE = "apache-2.0 (code, verified); dataset provenance to verify"

_DEFAULT_MAX_TURNS = 8


def load_tasks(cfg: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Yield ``search_qa`` tasks over a local Wikipedia/E5 retriever."""
    cfg = dict(cfg or {})
    max_samples = int(cfg.get("max_samples") or 0)
    max_turns = int(cfg.get("max_turns") or _DEFAULT_MAX_TURNS)
    license_name = cfg.get("license", LICENSE)
    emitted = 0
    for index, record in enumerate(iter_source_records(cfg, SOURCE)):
        if max_samples and emitted >= max_samples:
            return
        user_text = first_user_text(record)
        if not user_text:
            # FlashRAG-derived parquet stores the rendered prompt instead.
            prompt = record.get("prompt")
            if isinstance(prompt, list) and prompt:
                user_text = "\n".join(
                    str(entry.get("content", ""))
                    for entry in prompt
                    if isinstance(entry, dict) and entry.get("role") == "user"
                ).strip()
        answers = gold_answers(record)
        if not user_text or not answers:
            logger.debug("%s: record %d missing question/answers", SOURCE, index)
            record_skip(cfg, "missing_question_or_answers")
            continue
        messages = build_messages(SEARCH_SYSTEM, [user_text]) if record.get("question") else build_messages(
            None, [user_text]
        )
        task = build_task(
            source=SOURCE,
            task_id=safe_task_id(SOURCE, record.get("id") or record.get("qid"), index),
            task_type="search_qa",
            messages=messages,
            tools=SEARCH_TOOLS,
            env=env_search(max_turns=max_turns, retriever=cfg.get("retriever")),
            reference=build_reference(final_answer=answers[0], final_answers=answers, gt_calls=[]),
            max_turns=max_turns,
            expected_turns=cfg.get("expected_turns") or None,
            license=license_name,
            meta_extra={"raw_index": index, "split": cfg.get("split")},
        )
        emitted += 1
        yield task


__all__ = ["SOURCE", "LICENSE", "load_tasks"]
