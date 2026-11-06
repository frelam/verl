"""ASearcher training-data loader (hard multi-hop search QA, plan §1.2).

Verified raw path (GitHub README, 2026-11-05): the training data is released
as ``inclusionAI/ASearcher-train-data`` on HuggingFace.  ASearcher's RL runs
allow very long trajectories (README: tool calls exceeding 100 turns, 128 turns
per trajectory), so this loader is the main reason the global <=15 turn filter
exists (CONTRACTS C9 allows a separate ASearcher threshold).

The raw QA file is question/answer only; the loader does not invent search
trajectories.  ``meta["expected_turns"]`` is therefore only set when the raw
record explicitly carries a turn count (``extra_info.turns`` etc.); downstream
``filter_rounds.py`` drops tasks whose *effective* turn count exceeds the
configured ASearcher threshold.
"""

from __future__ import annotations

import logging
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
    gold_answers,
    iter_source_records,
    record_skip,
    safe_task_id,
)

logger = logging.getLogger(__name__)

SOURCE = "asearcher"
LICENSE = "dataset card unreachable[UNVERIFIED]; repo Apache-2.0 (verify)"

_DEFAULT_MAX_TURNS = 15


def _expected_turns(record: dict[str, Any]) -> int | None:
    extra = as_mapping(record.get("extra_info"))
    for container in (record, extra):
        for key in ("expected_turns", "num_turns", "turns", "tool_calls", "num_tool_calls"):
            value = container.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return int(value)
    return None


def load_tasks(cfg: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Yield ``search_qa`` tasks from ASearcher QA pairs."""
    cfg = dict(cfg or {})
    max_samples = int(cfg.get("max_samples") or 0)
    max_turns = int(cfg.get("max_turns") or _DEFAULT_MAX_TURNS)
    license_name = cfg.get("license", LICENSE)
    emitted = 0
    for index, record in enumerate(iter_source_records(cfg, SOURCE)):
        if max_samples and emitted >= max_samples:
            return
        user_text = first_user_text(record)
        prompt = record.get("prompt")
        if not user_text and isinstance(prompt, list):
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
            task_id=safe_task_id(SOURCE, first_non_empty(record.get("id"), record.get("qid")), index),
            task_type="search_qa",
            messages=messages,
            tools=SEARCH_TOOLS,
            env=env_search(max_turns=max_turns, retriever=cfg.get("retriever")),
            reference=build_reference(final_answer=answers[0], final_answers=answers, gt_calls=[]),
            max_turns=max_turns,
            expected_turns=_expected_turns(record),
            license=license_name,
            meta_extra={"raw_index": index, "needs_long_horizon_filter": True},
        )
        emitted += 1
        yield task


__all__ = ["SOURCE", "LICENSE", "load_tasks"]
