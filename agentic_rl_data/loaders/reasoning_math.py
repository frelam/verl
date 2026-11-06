"""Math prompts + python tool (``math_ci``) from any verl-style math parquet.

Accepts the verl RL math layout (``prompt`` chat list + ``reward_model.
ground_truth``, e.g. DAPO-Math-17k / DeepScaleR / the repo's reasoning-RL
pool) and plain ``{problem|question, answer|solution}`` records.  Only short,
verifiable final answers are kept (``\\boxed`` content or a plain answer of at
most ``max_answer_chars`` characters).

``cfg``: ``path`` / ``hf_dataset`` / ``records``, ``max_samples``,
``max_turns`` (default 6), ``max_answer_chars`` (40).
"""

from __future__ import annotations

from typing import Any, Iterator

from agentic_rl_data.loaders.common import (
    PYTHON_TOOL,
    build_messages,
    build_reference,
    build_task,
    env_python,
    first_user_text,
    gold_answers,
    iter_source_records,
    record_skip,
    safe_task_id,
)
from agentic_rl_data.rewards.math_ci import last_boxed_only_string, remove_boxed

SOURCE = "reasoning_math"
LICENSE = "inherits the source math dataset license [CONFIRM]"

MATH_SYSTEM = (
    "Solve the math problem. You may call the python tool to compute or check intermediate results; "
    "if the code fails, read the error and fix it. Give the final answer as \\boxed{...}."
)


def _answer(record: dict[str, Any]) -> str | None:
    for candidate in gold_answers(record):
        boxed = last_boxed_only_string(candidate)
        value = remove_boxed(boxed) if boxed else candidate
        if value and str(value).strip():
            return str(value).strip()
    return None


def load_tasks(cfg: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    cfg = dict(cfg or {})
    max_samples = int(cfg.get("max_samples") or 0)
    max_turns = int(cfg.get("max_turns") or 6)
    max_chars = int(cfg.get("max_answer_chars") or 40)
    emitted = 0
    for index, record in enumerate(iter_source_records(cfg, SOURCE)):
        if max_samples and emitted >= max_samples:
            return
        question = first_user_text(record)
        answer = _answer(record)
        if not question or not answer:
            record_skip(cfg, "missing_question_or_answer")
            continue
        if len(answer) > max_chars:
            record_skip(cfg, "long_answer")
            continue
        yield build_task(
            source=SOURCE,
            task_id=safe_task_id(SOURCE, record.get("id") or record.get("index"), index),
            task_type="math_ci",
            messages=build_messages(MATH_SYSTEM, [question]),
            tools=[PYTHON_TOOL],
            env=env_python(max_turns=max_turns),
            reference=build_reference(final_answer=answer, final_answers=[answer], gt_calls=[]),
            max_turns=max_turns,
            license=cfg.get("license", LICENSE),
            meta_extra={"raw_index": index},
        )
        emitted += 1


__all__ = ["SOURCE", "LICENSE", "MATH_SYSTEM", "load_tasks"]
