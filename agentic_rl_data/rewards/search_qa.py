"""``search_qa`` reward - Search-R1 / A²Search style QA (plan §5).

    score = 0.9 x answer + 0.1 x format

``answer`` = max over the acceptable answers of ``max(EM, token-F1)`` after
SQuAD-style normalisation (lowercase, punctuation removal, article removal,
whitespace collapse).  Multi-answer scoring uses
``reference["final_answers"]`` (:class:`list[str]`, CONTRACTS C7) first; when
that is null it falls back to the single ``reference["final_answer"]``.  The
multi-answer path is what makes A²Search-style samples scoreable (plan §1.2).

The format term is the small structural format weight required by the plan
(plan §5 "格式合法性小权重").  The prediction is read **only** from
``trajectory["final_answer"]`` (CONTRACTS C4); assistant text is never mined
for answers.  An empty prediction or a task without a reference answer scores
``0.0`` with an explicit ``info["reason"]``.

All scores are clamped into the frozen ``[0, 1]`` range (CONTRACTS C4).
"""

from __future__ import annotations

import re
import string
import unicodedata
from collections.abc import Mapping

from . import base

TASK_TYPE = "search_qa"
ANSWER_WEIGHT = 0.9
FORMAT_WEIGHT = 0.1
MAX_SCORE = 1.0

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
# Full-width punctuation that NFKC does not fold into ``string.punctuation``.
_EXTRA_PUNCT = frozenset("，。！？；：、（）【】《》「」『』“”‘’…—–·")


def normalize_answer(text) -> str:
    """SQuAD-style answer normalisation (lower, strip punctuation/articles/space)."""
    if text is None:
        return ""
    value = unicodedata.normalize("NFKC", str(text)).lower()
    value = "".join(
        ch for ch in value if ch not in string.punctuation and ch not in _EXTRA_PUNCT
    )
    value = _ARTICLES_RE.sub(" ", value)
    return " ".join(value.split())


def exact_match(prediction, gold) -> float:
    """1.0 when the normalised prediction equals the normalised gold answer."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def token_f1(prediction, gold) -> float:
    """Token-level F1 between prediction and one gold answer."""
    prediction_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not prediction_tokens or not gold_tokens:
        return 0.0
    gold_counts: dict[str, int] = {}
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1
    common = 0
    for token in prediction_tokens:
        remaining = gold_counts.get(token, 0)
        if remaining > 0:
            common += 1
            gold_counts[token] = remaining - 1
    if common == 0:
        return 0.0
    precision = common / len(prediction_tokens)
    recall = common / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def _prediction(trajectory) -> str | None:
    """Read the prediction from ``trajectory["final_answer"]`` only."""
    answer = base.final_answer(trajectory)
    if answer is None:
        return None
    tag_matches = _ANSWER_TAG_RE.findall(answer)
    if tag_matches:
        candidate = tag_matches[-1].strip()
        if candidate:
            return candidate
    stripped = answer.strip()
    return stripped if stripped else None


def _answer_source(task) -> str:
    ref = base.reference(task)
    answers = ref.get("final_answers")
    if isinstance(answers, (list, tuple)) and any(
        isinstance(item, str) and item.strip() for item in answers
    ):
        return "reference.final_answers"
    return "reference.final_answer"


def reward(task: dict, trajectory: dict) -> dict:
    """Pure reward for one ``search_qa`` task/trajectory pair."""
    golds = base.final_answers(task)
    prediction = _prediction(trajectory)
    format_score, format_info = base.format_report(task, trajectory)

    info = {
        "weights": {"answer": ANSWER_WEIGHT, "format": FORMAT_WEIGHT},
        "answer_source": _answer_source(task),
        "n_gold_answers": len(golds),
        "gold_answers": golds[:10],
        "prediction": prediction,
        "normalized_prediction": normalize_answer(prediction),
        "format_score": format_score,
        "format_detail": format_info,
    }
    if not golds:
        info["reason"] = "missing_reference_answer"
        return base.make_result(0.0, info, task_type=TASK_TYPE)
    if prediction is None:
        info["reason"] = "empty_prediction"
        return base.make_result(0.0, info, task_type=TASK_TYPE)

    candidates: list[dict] = []
    for index, gold in enumerate(golds):
        em = exact_match(prediction, gold)
        f1 = token_f1(prediction, gold)
        candidates.append(
            {
                "index": index,
                "gold": gold,
                "em": em,
                "f1": f1,
                "answer_score": max(em, f1),
            }
        )
    best = max(
        candidates,
        key=lambda candidate: (
            candidate["answer_score"],
            candidate["em"],
            candidate["f1"],
            -candidate["index"],
        ),
    )
    answer_score = best["answer_score"]
    score = base.weighted_sum(
        [(ANSWER_WEIGHT, answer_score), (FORMAT_WEIGHT, format_score)]
    )

    info.update(
        {
            "em": best["em"],
            "f1": best["f1"],
            "answer_score": answer_score,
            "best_gold": best["gold"],
            "candidates": candidates[:10],
        }
    )
    return base.make_result(score, info, task_type=TASK_TYPE)
