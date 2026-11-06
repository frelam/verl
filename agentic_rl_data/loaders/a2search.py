"""A²Search loader (ambiguity-aware multi-answer search QA, plan §1.2 optional).

Verified raw path (GitHub README, 2026-11-05): ``zfj1998/A2Search`` (Apache-2.0)
ships ``dataset/a2search_musique_2wiki_nq.parquet`` with fields::

    {"data_source": str, "question": str,
     "reward_model": {"ground_truth": [{"aliases": [str], "answer": str}]},
     "extra_info": {"id": str}}

It reuses the Search-R1 Wikipedia index + E5 retriever, so the env config
matches Search-R1.  ``reference.final_answers`` carries every valid answer
(answer + aliases) for the Answer-F1 reward; ``final_answer`` is the first one
for display / fallback.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Iterator

from agentic_rl_data.loaders.common import (
    SEARCH_SYSTEM,
    SEARCH_TOOLS,
    as_mapping,
    build_messages,
    build_reference,
    build_task,
    env_search,
    first_non_empty,
    first_user_text,
    iter_source_records,
    record_skip,
    safe_task_id,
)

logger = logging.getLogger(__name__)

SOURCE = "a2search"
LICENSE = "apache-2.0"
_DEFAULT_MAX_TURNS = 15


def _answers(record: dict[str, Any]) -> list[str]:
    """Collect every valid answer (main answer first, then aliases)."""
    ground_truth = as_mapping(record.get("reward_model")).get("ground_truth")
    if ground_truth is None:
        ground_truth = record.get("ground_truth") or record.get("answers")
    if isinstance(ground_truth, Mapping):
        ground_truth = [ground_truth]
    answers: list[str] = []
    if isinstance(ground_truth, list):
        for entry in ground_truth:
            if isinstance(entry, Mapping):
                main = first_non_empty(entry.get("answer"), entry.get("text"))
                if main is not None:
                    answers.append(str(main).strip())
                for alias in entry.get("aliases") or []:
                    if alias is not None and str(alias).strip():
                        answers.append(str(alias).strip())
            elif entry is not None and str(entry).strip():
                answers.append(str(entry).strip())
    seen: set[str] = set()
    return [answer for answer in answers if answer and not (answer in seen or seen.add(answer))]


def load_tasks(cfg: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Yield ``search_qa`` tasks with multi-answer ground truth."""
    cfg = dict(cfg or {})
    max_samples = int(cfg.get("max_samples") or 0)
    max_turns = int(cfg.get("max_turns") or _DEFAULT_MAX_TURNS)
    license_name = cfg.get("license", LICENSE)
    emitted = 0
    for index, record in enumerate(iter_source_records(cfg, SOURCE)):
        if max_samples and emitted >= max_samples:
            return
        question = first_user_text(record)
        answers = _answers(record)
        if not question or not answers:
            logger.debug("%s: record %d missing question/answers", SOURCE, index)
            record_skip(cfg, "missing_question_or_answers")
            continue
        task = build_task(
            source=SOURCE,
            task_id=safe_task_id(
                SOURCE,
                first_non_empty(record.get("id"), as_mapping(record.get("extra_info")).get("id")),
                index,
            ),
            task_type="search_qa",
            messages=build_messages(SEARCH_SYSTEM, [question]) if record.get("question") else build_messages(
                None, [question]
            ),
            tools=SEARCH_TOOLS,
            env=env_search(max_turns=max_turns, retriever=cfg.get("retriever")),
            reference=build_reference(final_answer=answers[0], final_answers=answers, gt_calls=[]),
            max_turns=max_turns,
            expected_turns=cfg.get("expected_turns") or None,
            license=license_name,
            meta_extra={"raw_index": index, "ambiguous": len(answers) > 1},
        )
        emitted += 1
        yield task


__all__ = ["SOURCE", "LICENSE", "load_tasks"]
