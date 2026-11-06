"""ReTool-SFT loader (math + code interpreter, plan §1.3 supplementary).

Source status (2026-11-05): ``JoeYing/ReTool-SFT`` is an HF-only dataset
(~2k samples); the HF hub is unreachable from this sandbox, so the id / license
/ exact field names are ``[UNVERIFIED]`` and no official GitHub repo was found.

The plan's first choice for ``math_ci`` is to reuse the existing reasoning-RL
math prompt pool and just attach a ``python`` tool; ReTool-SFT is only a cold
start / format reference and **not** the RL main source.  This loader therefore
keeps the final answer for the math verifier and deliberately does not treat
ReTool's tool calls as replay ground truth (``gt_calls = []``, env ``python``).

cfg keys: ``path`` / ``hf_dataset``, ``max_samples``, ``max_turns`` (default 8),
``license``.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

from agentic_rl_data.loaders.common import (
    PYTHON_TOOL,
    build_messages,
    build_reference,
    build_task,
    env_python,
    extract_answer,
    first_user_text,
    iter_source_records,
    record_skip,
    safe_task_id,
    unwrap_text,
)

logger = logging.getLogger(__name__)

SOURCE = "retool"
LICENSE = "unreachable[UNVERIFIED]"
_DEFAULT_HF_DATASET = "JoeYing/ReTool-SFT"


def _final_answer(record: dict[str, Any]) -> str | None:
    last_assistant = ""
    for key in ("messages", "conversations"):
        entries = record.get(key)
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict) and str(entry.get("role") or entry.get("from")) in (
                    "assistant",
                    "gpt",
                ):
                    last_assistant = unwrap_text(entry.get("content") if "content" in entry else entry.get("value"))
    answer = extract_answer(last_assistant)
    if answer is None:
        for key in ("final_answer", "answer", "ground_truth"):
            value = record.get(key)
            if value is not None and str(value).strip():
                answer = str(value).strip()
                break
    return answer


def load_tasks(cfg: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Yield ``math_ci`` tasks (final answer scored by the math verifier)."""
    cfg = dict(cfg or {})
    if not any(key in cfg for key in ("records", "path", "raw_path", "hf_dataset", "dataset")):
        cfg["hf_dataset"] = _DEFAULT_HF_DATASET
    max_samples = int(cfg.get("max_samples") or 0)
    max_turns = int(cfg.get("max_turns") or 8)
    license_name = cfg.get("license", LICENSE)
    emitted = 0
    for index, record in enumerate(iter_source_records(cfg, SOURCE)):
        if max_samples and emitted >= max_samples:
            return
        question = first_user_text(record)
        answer = _final_answer(record)
        if not question or not answer:
            record_skip(cfg, "missing_question_or_answer")
            continue
        task = build_task(
            source=SOURCE,
            task_id=safe_task_id(SOURCE, record.get("id"), index),
            task_type="math_ci",
            messages=build_messages(
                "You are a math reasoning assistant. You may use the python tool to compute; "
                "finish with the final answer.",
                [question],
            ),
            tools=[PYTHON_TOOL],
            env=env_python(max_turns=max_turns),
            reference=build_reference(final_answer=answer, final_answers=[answer], gt_calls=[]),
            max_turns=max_turns,
            expected_turns=cfg.get("expected_turns") or None,
            license=license_name,
            meta_extra={"raw_index": index},
        )
        emitted += 1
        yield task


__all__ = ["SOURCE", "LICENSE", "load_tasks"]
