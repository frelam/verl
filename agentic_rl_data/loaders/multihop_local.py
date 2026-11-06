"""Multi-hop QA with a **per-task local corpus** (HotpotQA / 2Wiki / MuSiQue).

Why: Search-R1 style training needs a Wikipedia E5 index (tens of GB, a GPU
or CPU faiss service, a retriever server under load).  HotpotQA-distractor,
2WikiMultihopQA and MuSiQue already ship 10-20 candidate paragraphs per
question (2-4 gold + hard distractors).  Putting them into
``env.config.docs`` lets the stateless ``InMemorySearchBackend`` (BM25-lite,
stdlib only) answer every ``search`` / ``fetch`` call inside the rollout
worker: zero services, zero downloads at train time, ~0.1 ms per query.

Trade-off (documented in ``reports/AGENTIC_RL_V2.md``): the corpus is small,
so a single broad query can surface all paragraphs.  Retrieval is therefore
deliberately noisier than "return everything": ``top_k`` defaults to 2 and
snippets are short (``snippet_chars=240``) so the policy still has to issue
per-hop queries and ``fetch`` the right page.  Gold answers are short
entities, scored by ``search_qa`` (EM, length-guarded F1).

Supported raw layouts (auto-detected per record):

* HotpotQA / 2Wiki: ``context = [[title, [sentences...]], ...]`` or
  ``context = {"title": [...], "sentences": [[...], ...]}`` (HF datasets),
  ``answer`` str, ``question`` str, ``type`` / ``level`` optional;
* MuSiQue: ``paragraphs = [{"title", "paragraph_text", "is_supporting"}]``,
  ``answer`` + ``answer_aliases``, ``answerable`` (unanswerable rows skipped).

``cfg``: ``path`` / ``hf_dataset`` / ``records`` (see ``common.read_records``),
``max_samples``, ``max_turns`` (default 10), ``top_k`` (2),
``snippet_chars`` (240), ``min_hops`` (2), ``dataset`` (hotpotqa | 2wiki |
musique, only used for ids / meta), ``max_answer_tokens`` (6, long answers
are skipped because EM on them is unreliable).
"""

from __future__ import annotations

import logging
from typing import Any, Iterator, Mapping

from agentic_rl_data.loaders.common import (
    SEARCH_TOOLS,
    build_messages,
    build_reference,
    build_task,
    iter_source_records,
    record_skip,
    safe_task_id,
)

logger = logging.getLogger(__name__)

SOURCE = "multihop_local"
LICENSE = "CC BY-SA 4.0 (HotpotQA / 2Wiki / MuSiQue paragraphs are Wikipedia text)"

MULTIHOP_SYSTEM = (
    "You answer questions by searching a small document collection. The question needs several facts: "
    "plan which entity to look up first, search for it, read the result, then search for the next hop. "
    "If a search returns nothing useful, rephrase the query instead of guessing. Use fetch to read a full "
    "page when the snippet is cut off. When you know the answer, reply without calling a tool and put only "
    "the short answer (an entity, date or number) inside <answer></answer>."
)

_DEFAULT_MAX_TURNS = 10


def _hotpot_paragraphs(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    context = record.get("context")
    supporting = record.get("supporting_facts")
    gold_titles: set[str] = set()
    if isinstance(supporting, Mapping):
        gold_titles = {str(t) for t in supporting.get("title") or []}
    elif isinstance(supporting, list):
        gold_titles = {str(item[0]) for item in supporting if isinstance(item, (list, tuple)) and item}
    paragraphs: list[dict[str, Any]] = []
    if isinstance(context, Mapping):
        titles = context.get("title") or []
        sentences = context.get("sentences") or []
        pairs = list(zip(titles, sentences))
    elif isinstance(context, list):
        pairs = [(item[0], item[1]) for item in context if isinstance(item, (list, tuple)) and len(item) == 2]
    else:
        pairs = []
    for title, sents in pairs:
        text = " ".join(str(s).strip() for s in (sents or []) if str(s).strip())
        if text:
            paragraphs.append({"title": str(title), "text": text, "gold": str(title) in gold_titles})
    return paragraphs


def _musique_paragraphs(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    paragraphs = []
    for item in record.get("paragraphs") or []:
        if not isinstance(item, Mapping):
            continue
        text = str(item.get("paragraph_text") or "").strip()
        if text:
            paragraphs.append(
                {"title": str(item.get("title") or ""), "text": text, "gold": bool(item.get("is_supporting"))}
            )
    return paragraphs


def _answers(record: Mapping[str, Any]) -> list[str]:
    answers: list[str] = []
    for key in ("answer", "answers"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            answers.append(value.strip())
        elif isinstance(value, list):
            answers += [str(v).strip() for v in value if str(v).strip()]
    for alias in record.get("answer_aliases") or []:
        if str(alias).strip():
            answers.append(str(alias).strip())
    seen: set[str] = set()
    unique = []
    for answer in answers:
        if answer.lower() not in seen:
            seen.add(answer.lower())
            unique.append(answer)
    return unique


def load_tasks(cfg: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Yield ``search_qa`` tasks whose search env is a per-task in-memory corpus."""
    cfg = dict(cfg or {})
    max_samples = int(cfg.get("max_samples") or 0)
    max_turns = int(cfg.get("max_turns") or _DEFAULT_MAX_TURNS)
    top_k = int(cfg.get("top_k") or 2)
    snippet_chars = int(cfg.get("snippet_chars") or 240)
    min_hops = int(cfg.get("min_hops") or 2)
    max_answer_tokens = int(cfg.get("max_answer_tokens") or 6)
    dataset = str(cfg.get("dataset") or "multihop")
    emitted = 0
    for index, record in enumerate(iter_source_records(cfg, SOURCE)):
        if max_samples and emitted >= max_samples:
            return
        if record.get("answerable") is False:
            record_skip(cfg, "unanswerable")
            continue
        question = str(record.get("question") or "").strip()
        answers = _answers(record)
        paragraphs = _musique_paragraphs(record) if record.get("paragraphs") else _hotpot_paragraphs(record)
        if not question or not answers or not paragraphs:
            record_skip(cfg, "missing_question_answer_or_context")
            continue
        if answers[0].lower() in ("yes", "no"):
            # comparison yes/no questions: 50% guessable without any search
            record_skip(cfg, "yes_no_answer")
            continue
        if len(answers[0].split()) > max_answer_tokens:
            record_skip(cfg, "long_answer")
            continue
        n_gold = sum(1 for p in paragraphs if p["gold"])
        if n_gold and n_gold < min_hops:
            record_skip(cfg, "too_few_hops")
            continue
        docs = [
            {"doc_id": f"d{i}", "title": p["title"], "text": p["text"]}
            for i, p in enumerate(paragraphs)
        ]
        hops = max(n_gold, min_hops)
        config = {
            "backend": "inmemory",
            "docs": docs,
            "top_k": top_k,
            "snippet_chars": snippet_chars,
            "max_rounds": max_turns,
            "task_deadline_s": 0,
        }
        task = build_task(
            source=SOURCE,
            task_id=safe_task_id(f"{SOURCE}-{dataset}", record.get("id") or record.get("_id"), index),
            task_type="search_qa",
            messages=build_messages(MULTIHOP_SYSTEM, [question]),
            tools=SEARCH_TOOLS,
            env={"type": "search", "config": config},
            reference=build_reference(final_answer=answers[0], final_answers=answers, gt_calls=[]),
            max_turns=max_turns,
            # hops searches + (optional) fetches + final reply
            expected_turns=min(max_turns, hops + 2),
            difficulty=float(hops),
            license=cfg.get("license", LICENSE),
            meta_extra={"dataset": dataset, "hops": hops, "n_docs": len(docs), "raw_index": index},
        )
        emitted += 1
        yield task


__all__ = ["SOURCE", "LICENSE", "MULTIHOP_SYSTEM", "load_tasks"]
