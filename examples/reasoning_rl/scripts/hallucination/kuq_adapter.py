# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""KUQ adapter: judgment rows (KUQ-known) and bare-refusal rows (KUQ-unknown).

``amayuelas/KUQ`` (MIT) is one JSONL file, ``knowns_unknowns.jsonl``, with a
*balanced* answerability annotation: 3,447 questions that have a correct answer
(``source`` in hotpotqa / triviaqa / squad) and 3,437 that do not (``source ==
"turk"``, plus a ``category`` naming the kind of unanswerability).  The file is
two branches in one:

============================  =====================================  ==========
KUQ half                      branch (schema.py table B)             template
============================  =====================================  ==========
``unknown == false`` (3,447)  ``solvable_judge``                     B_judge
``unknown == true``  (3,437)  ``unsolvable_bare``                    B_judge
============================  =====================================  ==========

Neither half carries a pointer gold, so neither can carry an options block
(recon report section 4d: ``knowns_unknowns.jsonl`` has no span/offset/highlight
field at all, and 0/6,884 questions contain ``<...>`` / ``[[...]]`` / ``**``
markup).  Both halves therefore ask the model for **a verdict only** -- the
known half expects ``\\boxed{SOLVABLE}``, the unknown half
``\\boxed{UNSOLVABLE}`` -- and rows are built with the same template constant so
that "which template am I looking at" is not a label cue.

The raw ``answer`` list is deliberately *not* a gold here.  On the unknown side
it holds "sources of uncertainty" (``"Since it's a theory in name, the answer is
conjecture."``), which is prose, not an answer; on the known side it is the QA
corpus' accepted-answer list, which the reward has no matcher for.  It is kept in
``extra_info`` for audit only (see ``kuq_answer_text``).

Certificate and fail-closed policy
----------------------------------

KUQ ships no external oracle: the known half is factual QA (nothing in the file
can recompute "California") and there is no span to re-locate.  The strongest
per-row certificate the source admits is therefore a **three-axis conjunction**
that never looks at the ``unknown`` flag itself:

1. **provenance** -- a known row's ``source`` must be one of the three closed QA
   corpus ids, an unknown row's must be ``turk``;
2. **annotation schema** -- ``category`` is an *absent key* on known rows and a
   documented six-value string on unknown rows (measured 100% both ways by the
   recon report, re-measured here);
3. **payload form** -- the row's own ``answer`` list must be *directionally*
   consistent with the verdict: a known row needs at least one determinate
   answer span (<= :data:`ANSWER_SPAN_MAX_WORDS` word tokens), an unknown row at
   least one uncertainty explanation (>= :data:`UNCERTAINTY_MIN_WORDS` tokens).

A row whose three axes do not all point at the same verdict is dropped and
counted in the funnel -- the adapter never invents a gold and never repairs a
label.  Rows are additionally dropped, with a funnel entry each, for a malformed
record, a question that appears under *both* verdicts (a self-contradicting
duplicate), a repeated question, an out-of-scope category and a task_id
collision.

DEVIATIONS FROM THE DESIGN DOC
------------------------------

Measured against ``HALLUCINATION_RL_DESIGN.md`` section 3/4.9.1/4.9.3 and the
recon report.  Where the two disagree the recon report wins; where the design
doc's own contract makes a number unachievable it is recorded here.

1. **Template is ``B_judge`` for BOTH halves, not template B.**  Doc section
   4.9.3 table B puts KUQ-known (row 4) and KUQ(FA+CF) (row 6) on template B.
   That is not implementable: template B asks for *an answer* and never offers a
   verdict, while section 9's own KUQ audit demands "断言两个 label 共用同一模板
   （D14 同构约束）", and ``schema.validate_row`` rejects ``judgment_only`` rows on
   template B.  Template A is unavailable (no pointer gold -> no option block).
   ``TEMPLATE_B_JUDGE`` is the only constant that expresses a verdict and is
   legal for both labels, so both halves use it.  The isomorphism requirement is
   the stronger clause and wins.
2. **``task_id`` is derived from the question, not from a source id.**  Doc
   section 3 maps ``id`` -> ``extra_info.task_id``.  KUQ has no id column:
   measured key sets are exactly ``(answer, question, source, unknown)`` (3,447
   rows) and ``(answer, category, question, source, unknown)`` (3,437 rows).
   The stable key is therefore ``kuq-<sha1(normalised question)[:16]>``, which
   is deterministic, replay-stable and unique after de-duplication.
3. **``extra_info.difficulty`` is a length proxy.**  Doc section 3 says
   "来源原生难度"; KUQ carries no difficulty field (same key-set measurement), so
   a ``short`` / ``medium`` / ``long`` bucket over the question's word count is
   recorded instead.
4. **The unsolvable half is restricted to two of the six categories.**
   Section 4.9.3's defect table gives KUQ exactly one slot -- 假前提（不可指认）
   三档 200 -- and D18 names the categories for it: ``false assumption`` (520
   rows) + ``counterfactual`` (568 rows).  The other four (``controversial``
   676 / ``future unknown`` 659 / ``ambiguous`` 577 / ``unsolved problem`` 437)
   are counted and dropped by the ``category_in_scope`` funnel stage rather
   than silently folded into a defect type the balance table has no room for.
5. **``unknown == true`` rows never carry ``answer`` in the payload** (the task
   contract) and the known half does not either: the raw answer list is audit
   metadata in ``extra_info``.  Doc section 3 keeps ``ground_truth_answer`` for
   ``solvable=true``; D14 makes the KUQ-known half a *judgment* row whose gold
   is the ``SOLVABLE`` marker and whose answer is audit-only, so writing it into
   the payload would only create a second, unscored gold.
6. **The L3 gate is measured here and FAILS / was 未测 in the doc.**  Recon
   section 4.9.1/table A records KUQ's L3 as "未测" and the recon headline says
   the doc's claims were verified -- but neither the doc nor the recon ever ran
   the surface-heuristic audit on KUQ.  Measured here with the task's protocol
   (bag-of-words multinomial NB, 5-fold, out-of-fold balanced accuracy,
   vocabulary limited to tokens with >= 5 training documents):

   * all 3,437 unknown vs all 3,447 known  -> **0.858**
   * the 1,088 in-scope FA+CF unknown vs known -> **0.893**
   * on the artifact this adapter actually emits (3,360 solvable vs 758
     unsolvable) -> **0.883**
   * on the mandated ``--limit 300`` sample (245 vs 55) -> **0.807**; the
     sample's class ratio is thinner, so this is the noisiest of the three, and
     it is the number the mandated ``verify_kuq.py`` run prints
   * stopword-only 0.855 / content-word-only 0.749 (so it is not a "function
     words" artefact); per sub-corpus squad 0.848 / triviaqa 0.885 /
     hotpotqa 0.938; the length-only cue is 0.501 over the raw pools, 0.533 on
     the artifact and 0.548 on the 300-row sample, i.e. length is *not* the cue
     -- vocabulary and topic are.

   The task's threshold is <= 0.60, so ``verify_kuq.py`` reports this check as
   FAIL on the real build (see its module docstring for why the adapter does not
   try to filter its way under the gate).  For calibration: the design doc's own
   judgment branch (FalseQA ``label=0`` vs ``label=1``, D14) measures H6 =
   0.765/0.833/0.807 and section 9/11 accepts that as an *interpretation
   ceiling* -- "主题记忆" -- rather than a gate.  KUQ's known half is the same
   kind of judgment row and measures higher; the number belongs in the build
   report next to that ceiling, not quietly under a green check.
7. **Recon-report details re-measured, two of them need updating.**
   * 6,884 rows / unknown 3,437 / known 3,447 / six non-empty categories /
     ``source == "turk"`` iff unknown iff ``category`` present: **confirmed**.
   * ``len(answer) == 1`` implies known: **confirmed** (2,647 rows, 0 unknown).
     The recon printed the known distribution as ``{1: 2647, 3: 62, ...}``; the
     full measured distribution is ``{1: 2647, 2: 111, >=3: 689}`` (max 126), so
     "``len(answer) >= 3``" covers 4,126 rows of which 3,437 = **83.3%** unknown
     -- the recon's percentage is right, its printed support is partial.
   * 966 duplicate questions / 965 duplicate ``(question, unknown)`` pairs:
     **confirmed**, and the difference is exactly one question
     (``"What is the population of the city?"``) that appears under *both*
     verdicts.  Both of its rows are dropped here.
   * Not in the recon: **171 rows have no ``?`` anywhere in the question** (112
     unknown / 59 known), e.g. truncated turk questions
     (``"What are the challenges each year in e-"``).  They are kept -- a missing
     question mark is not evidence about answerability and no rule in the design
     doc licenses dropping them -- but the count is recorded in the funnel note.

Funnel (full file, no ``--limit``) is printed by :func:`main` and reproduced in
the build report; ``build_rows`` returns it so the caller can audit row loss.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import schema

DEFAULT_RAW_DIR = "/home/charles/data/reasoning_rl/halluc/raw/kuq_crepe"
DEFAULT_OUT = "/home/charles/data/reasoning_rl/halluc/built/kuq.parquet"

# The doc-referenced file.  ``modified_knowns_unknowns.jsonl`` has the identical
# five-field schema but is not referenced by the design doc, and
# ``unknowns_all.jsonl`` has a different 12-field schema and a different category
# vocabulary -- joining them would double-count and mix taxonomies, so this
# adapter reads exactly one file (recon report section 5, KUQ hazard 8).
RAW_FILENAME = "knowns_unknowns.jsonl"

# Closed provenance vocabularies.  An unseen ``source`` value means the file
# drifted and the row cannot be certified; it is dropped, not guessed.
KNOWN_SOURCE_IDS = ("hotpotqa", "squad", "triviaqa")
UNKNOWN_SOURCE_ID = "turk"

# The six documented ``category`` values (all non-empty: 437..676 rows each).
KUQ_CATEGORIES = (
    "ambiguous",
    "controversial",
    "counterfactual",
    "false assumption",
    "future unknown",
    "unsolved problem",
)

# Section 4.9.3's defect table gives KUQ one slot: 假前提（不可指认）, three-tier.
# The other four categories belong to defect types that need a pointer (歧义 /
# 不现实 / 问题缺失 / 无关实体) or to a slot this source does not fill.
UNSOLVABLE_CATEGORIES_IN_SCOPE = frozenset({"false assumption", "counterfactual"})

# ``extra_info.error_type`` for the D18 balance table: the one defect slot the
# design assigns to this source.  Kept as a slug so all adapters agree.
UNSOLVABLE_ERROR_TYPE = "false_premise_unpointable"

# Payload-form margins.  Measured on the raw file (word tokens per element):
#   known   per-row min  p1=0 p5=1 p50=2 p99=10   -> 76/3,447 rows fail <= 12
#   unknown per-row max  p1=7 p5=9 p50=16         -> 56/3,437 rows fail >= 8
# The thresholds sit just outside the 99th/5th percentile of the opposite side,
# so they are conservative: they only reject payloads that look like the *other*
# label's prose.
ANSWER_SPAN_MAX_WORDS = 12
UNCERTAINTY_MIN_WORDS = 8

# A question must carry at least this many word tokens.  Measured minimum in the
# raw file is 3 on both sides, so this only rejects empty/degenerate records.
MIN_QUESTION_WORDS = 3

DIFFICULTY_SHORT_MAX_WORDS = 7
DIFFICULTY_MEDIUM_MAX_WORDS = 14

# ``make_row`` needs an explicit template constant; KUQ is option-less, so both
# halves take the verdict-only wording (see deviation 1).
TEMPLATE = schema.TEMPLATE_B_JUDGE


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def normalise_question(text: str) -> str:
    """Whitespace-normalised question, the de-duplication and hash key."""
    return " ".join(text.split())


def task_id_for(question: str) -> str:
    """Stable hard-replay key: ``kuq-<sha1(normalised question)[:16]>``."""
    digest = hashlib.sha1(normalise_question(question).encode("utf-8")).hexdigest()
    return f"kuq-{digest[:16]}"


def word_count(text: str) -> int:
    """Word tokens, using the shared tokenizer (``schema.words``)."""
    return len(schema.words(text))


def difficulty_for(question: str) -> str:
    """Length-proxy difficulty bucket (KUQ ships no native difficulty tag)."""
    count = word_count(question)
    if count <= DIFFICULTY_SHORT_MAX_WORDS:
        return "short"
    if count <= DIFFICULTY_MEDIUM_MAX_WORDS:
        return "medium"
    return "long"


def payload_form_problem(answers: list[str], solvable: bool) -> str | None:
    """Return why ``answers`` cannot corroborate the verdict, else ``None``.

    A solvable row needs one determinate answer span; an unsolvable row needs one
    uncertainty explanation.  A row that carries only the other kind of prose
    cannot be certified -- it is dropped rather than relabelled.
    """
    if not answers:
        return "empty answer list"
    if solvable:
        if not any(a.strip() and word_count(a) <= ANSWER_SPAN_MAX_WORDS for a in answers):
            return "no determinate answer span (every element reads as an explanation)"
        return None
    if not any(a.strip() and word_count(a) >= UNCERTAINTY_MIN_WORDS for a in answers):
        return "no uncertainty explanation (every element reads as an answer)"
    return None


def certify_record(raw: dict) -> str | None:
    """Certificate for one raw KUQ record; ``None`` means certified.

    The three axes are checked in a fixed order and the first failure is
    returned, so the funnel can also report *why* rows were dropped.
    """
    question = raw.get("question")
    if not isinstance(question, str) or not question.strip():
        return "empty question"
    if word_count(question) < MIN_QUESTION_WORDS:
        return f"question shorter than {MIN_QUESTION_WORDS} word tokens"

    unknown = raw.get("unknown")
    if not isinstance(unknown, bool):
        return "unknown flag is not a bool"

    answer = raw.get("answer")
    if not isinstance(answer, list) or not answer:
        return "missing answer list"
    if not all(isinstance(item, str) and item.strip() for item in answer):
        return "answer list holds a non-string or empty element"

    source = raw.get("source")
    if not isinstance(source, str) or not source:
        return "missing source"
    if unknown:
        if source != UNKNOWN_SOURCE_ID:
            return f"unknown row from {source!r}, expected {UNKNOWN_SOURCE_ID!r}"
    elif source not in KNOWN_SOURCE_IDS:
        return f"known row from {source!r}, not one of {KNOWN_SOURCE_IDS}"

    category = raw.get("category")
    has_category = isinstance(category, str) and bool(category.strip())
    if has_category != unknown:
        return "category presence disagrees with the unknown flag"
    if has_category and category not in KUQ_CATEGORIES:
        return f"category {category!r} is outside the documented six"

    return payload_form_problem(answer, solvable=not unknown)


def category_in_scope(raw: dict) -> bool:
    """Whether an unknown record's category is assigned to this source (D18).

    Known rows are always in scope -- the restriction is about which *defect
    types* section 4.9.3 budgets KUQ for, not about the answerable half.
    """
    if not raw["unknown"]:
        return True
    return raw.get("category") in UNSOLVABLE_CATEGORIES_IN_SCOPE


def make_kuq_row(raw: dict, line_no: int, seed: int, split: str = "train") -> dict:
    """Build one contract row for a certified raw record.

    ``line_no`` is the record's 0-based line in ``knowns_unknowns.jsonl`` and is
    stored as ``extra_info.index``; it is stable for a given file, which is what
    makes the build byte-reproducible.
    """
    question = raw["question"].strip()
    solvable = not raw["unknown"]
    category = raw["category"] if raw["unknown"] else ""
    perturbation = None if solvable else "contradictory_condition"
    extra_info = {
        "split": split,
        "index": line_no,
        "task_id": task_id_for(question),
        "difficulty": difficulty_for(question),
        "seed": seed,
        "solvable": solvable,
        "judgment_only": solvable,
        "has_diagnosis_label": False,
        "correct_option_id": "",
        "perturbation_type": perturbation or "",
        "error_type": "" if solvable else UNSOLVABLE_ERROR_TYPE,
        # Audit-only provenance: the reward never reads these, and they are NOT
        # rendered into the prompt (the recon report's leak warning is about the
        # prompt, and the prompt here is the question plus a constant suffix).
        "kuq_source": raw["source"],
        "kuq_category": category,
        "kuq_answer_text": " || ".join(raw["answer"]),
        "kuq_answer_count": len(raw["answer"]),
    }
    return schema.make_row(
        data_source=schema.SOURCE_KUQ,
        question=question,
        ground_truth=schema.build_ground_truth(
            solvable=solvable,
            answer=None,
            correct_option_id=None,
            has_diagnosis_label=False,
            perturbation_type=perturbation,
            judgment_only=solvable,
        ),
        template=TEMPLATE,
        branch=(
            schema.BRANCH_SOLVABLE_JUDGE if solvable else schema.BRANCH_UNSOLVABLE_BARE
        ),
        extra_info=extra_info,
    )


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def _load_lines(raw_dir: str) -> tuple[list[dict], int, Counter]:
    """Parse the JSONL file.  Returns (records, line count, drop-reason counts)."""
    path = Path(raw_dir) / RAW_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"KUQ raw file not found: {path}")
    records: list[dict] = []
    reasons: Counter = Counter()
    raw_lines = 0
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            if not line.strip():
                continue
            raw_lines += 1
            try:
                record = json.loads(line)
            except ValueError:
                reasons["malformed JSON"] += 1
                continue
            if not isinstance(record, dict):
                reasons["JSON line is not an object"] += 1
                continue
            record["_line"] = line_no
            records.append(record)
    return records, raw_lines, reasons


def _dedupe_questions(records: list[dict]) -> tuple[list[dict], int, int]:
    """Drop questions that appear under both verdicts, then repeat questions.

    Returns ``(kept, dropped_both_labels, dropped_repeats)``.  The first is a
    fail-closed rule: ``"What is the population of the city?"`` really does occur
    as both a known and an unknown row in the raw file, and neither copy can be
    certified as the unique verdict.  The second keeps the earliest line so the
    artifact does not depend on iteration order.
    """
    verdicts: dict[str, set[bool]] = {}
    for record in records:
        verdicts.setdefault(normalise_question(record["question"]), set()).add(
            bool(record["unknown"])
        )
    ambiguous = {key for key, sides in verdicts.items() if len(sides) > 1}

    kept: list[dict] = []
    seen: set[str] = set()
    dropped_both = 0
    dropped_repeats = 0
    for record in records:
        key = normalise_question(record["question"])
        if key in ambiguous:
            dropped_both += 1
            continue
        if key in seen:
            dropped_repeats += 1
            continue
        seen.add(key)
        kept.append(record)
    return kept, dropped_both, dropped_repeats


def _drop_by_reason(
    records: list[dict], predicate
) -> tuple[list[dict], Counter]:
    """Split ``records`` on ``predicate``; the rejected side keeps its reasons."""
    kept: list[dict] = []
    reasons: Counter = Counter()
    for record in records:
        reason = predicate(record)
        if reason is None:
            kept.append(record)
        else:
            reasons[reason] += 1
    return kept, reasons


def branch_quotas(n_solvable: int, n_unsolvable: int, limit: int) -> tuple[int, int]:
    """Split ``limit`` across the two branches, proportionally, floor 1 each.

    A plain proportional split with ``round()`` gives a rare branch a quota of
    zero at small limits (1 unsolvable row in 31 needs ``limit >= 32`` to get
    one), which would silently ship a single-branch artifact.  Instead each
    non-empty branch is guaranteed one row when two rows fit, the larger quota
    absorbs any overshoot, and unused quota is handed to the branch that still
    has rows to give.
    """
    if limit >= n_solvable + n_unsolvable:
        return n_solvable, n_unsolvable
    quotas = {
        "solvable": min(n_solvable, max(1, round(limit * n_solvable / (n_solvable + n_unsolvable))))
        if n_solvable
        else 0,
        "unsolvable": min(
            n_unsolvable, max(1, round(limit * n_unsolvable / (n_solvable + n_unsolvable)))
        )
        if n_unsolvable
        else 0,
    }
    while quotas["solvable"] + quotas["unsolvable"] > limit:
        larger = "solvable" if quotas["solvable"] >= quotas["unsolvable"] else "unsolvable"
        quotas[larger] -= 1
    for _ in range(2):
        shortfall = limit - quotas["solvable"] - quotas["unsolvable"]
        if shortfall <= 0:
            break
        for name, size in (("solvable", n_solvable), ("unsolvable", n_unsolvable)):
            room = min(shortfall, size - quotas[name])
            if room > 0:
                quotas[name] += room
                shortfall -= room
    return quotas["solvable"], quotas["unsolvable"]


def apply_limit(rows: list[dict], limit: int | None, rng: random.Random) -> list[dict]:
    """Deterministically cap the build to ``limit`` rows, keeping both branches.

    The per-branch quota is proportional to the certified pool
    (:func:`branch_quotas`), and any quota a side cannot fill is handed to the
    other side, so ``limit`` rows come out whenever the pool is large enough.
    ``limit=None`` (or a limit larger than the pool) is a no-op.  Sampling is
    drawn from file order, so the same seed always yields the same subset.

    Proportional is deliberate: ``--limit`` is a debugging cap that keeps the
    pool's own branch ratio (measured 3,360 solvable : 758 unsolvable, so
    ``--limit 300`` gives 245 : 55).  It is **not** the balance mechanism -- the
    two arms the design budgets for KUQ (D18: 200 KUQ-known judgment rows and 200
    KUQ ``false assumption`` + ``counterfactual`` bare rows) are drawn per branch
    by ``mix_halluc.py`` from the full pool.
    """
    if limit is None or limit >= len(rows):
        return list(rows)
    if limit <= 0:
        return []
    solvable = [row for row in rows if row["extra_info"]["solvable"]]
    unsolvable = [row for row in rows if not row["extra_info"]["solvable"]]
    n_solvable, n_unsolvable = branch_quotas(len(solvable), len(unsolvable), limit)
    picked = rng.sample(solvable, n_solvable) + rng.sample(unsolvable, n_unsolvable)
    picked.sort(key=lambda row: row["extra_info"]["index"])
    return picked


def build_rows(
    raw_dir: str, limit: int | None = None, seed: int = 0
) -> tuple[list[dict], dict]:
    """Build the KUQ rows.  Deterministic in ``(raw_dir, limit, seed)``.

    Args:
        raw_dir: directory holding ``knowns_unknowns.jsonl``.
        limit: optional cap on the number of rows returned (proportional across
            the two branches, sampled with ``random.Random(seed)``).
        seed: recorded in ``extra_info.seed`` and used only by ``limit``.

    Returns:
        ``(rows, funnel)`` where ``funnel`` is an ordered mapping of filter stage
        name -> rows remaining after that stage, starting at the raw line count.
    """
    records, raw_lines, parse_reasons = _load_lines(raw_dir)
    without_question_mark = sum(1 for record in records if "?" not in str(record.get("question")))
    funnel: dict[str, int] = {"raw_lines": raw_lines}
    funnel["parsed_rows"] = len(records)

    records, reasons = _drop_by_reason(records, certify_record)
    funnel["certified_records"] = len(records)
    records, dropped_both, dropped_repeats = _dedupe_questions(records)
    funnel["unambiguous_verdicts"] = len(records) + dropped_repeats
    funnel["deduplicated_questions"] = len(records)
    records, _ = _drop_by_reason(
        records, lambda record: None if category_in_scope(record) else "out of scope"
    )
    funnel["category_in_scope"] = len(records)

    rows = [
        make_kuq_row(record, record["_line"], seed)
        for record in records
    ]
    # Defensive: a task_id collision would break the hard-replay dedup key.
    seen_ids: set[str] = set()
    unique_rows = []
    for row in rows:
        task_id = row["extra_info"]["task_id"]
        if task_id in seen_ids:
            continue
        seen_ids.add(task_id)
        unique_rows.append(row)
    funnel["unique_task_ids"] = len(unique_rows)

    rng = random.Random(seed)
    limited = apply_limit(unique_rows, limit, rng)
    funnel["rows_after_limit"] = len(limited)

    # Drop counters are returned alongside the ordered funnel stages so a report
    # can explain every row that did not survive.
    diagnostics = {
        "raw_file": str(Path(raw_dir) / RAW_FILENAME),
        "missing_or_malformed_lines": sum(parse_reasons.values()),
        "certificate_drop_reasons": dict(sorted(reasons.items())),
        "questions_under_both_verdicts": dropped_both,
        "repeated_questions": dropped_repeats,
        # Not a filter: kept as evidence for the build report (deviation 7).
        "questions_without_question_mark": without_question_mark,
        "limit": limit,
        "seed": seed,
    }
    return limited, {"stages": funnel, "diagnostics": diagnostics}


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def _breakdown(rows: list[dict], key) -> Counter:
    return Counter(key(row) for row in rows)


def format_report(rows: list[dict], funnel: dict) -> str:
    """Human-readable funnel + per-branch / per-template / per-solvable report."""
    stages: dict = funnel["stages"]
    diagnostics: dict = funnel.get("diagnostics", {})
    lines = [f"KUQ adapter: {len(rows)} rows built", "", "funnel:"]
    previous: int | None = None
    for name, count in stages.items():
        if previous is None:
            lines.append(f"  {name:<26} {count:>6}")
        else:
            lines.append(f"  {name:<26} {count:>6}  ({count - previous:+d})")
        previous = count
    for name, count in diagnostics.get("certificate_drop_reasons", {}).items():
        lines.append(f"    certificate drop: {name} ({count})")
    both = diagnostics.get("questions_under_both_verdicts", 0)
    if both:
        lines.append(f"    contradictory duplicates (both verdicts): {both} rows dropped")
    repeats = diagnostics.get("repeated_questions", 0)
    if repeats:
        lines.append(f"    repeated questions (same verdict): {repeats} rows dropped")
    lines.append(
        "    note: questions without a '?' kept (not a filter): "
        f"{diagnostics.get('questions_without_question_mark', 0)}"
    )

    lines.append("")
    lines.append("by branch:")
    for name, count in sorted(_breakdown(rows, lambda r: r["extra_info"]["branch"]).items()):
        lines.append(f"  {name:<26} {count:>6}")
    lines.append("by template:")
    for name, count in sorted(_breakdown(rows, lambda r: r["extra_info"]["template"]).items()):
        lines.append(f"  {name:<26} {count:>6}")
    lines.append("by solvable:")
    for name, count in sorted(
        _breakdown(rows, lambda r: r["extra_info"]["solvable"]).items()
    ):
        lines.append(f"  {name!s:<26} {count:>6}")
    lines.append("by kuq category (audit only):")
    for name, count in sorted(_breakdown(rows, lambda r: r["extra_info"]["kuq_category"]).items()):
        lines.append(f"  {name or '<known>':<26} {count:>6}")
    lines.append("by difficulty:")
    for name, count in sorted(_breakdown(rows, lambda r: r["extra_info"]["difficulty"]).items()):
        lines.append(f"  {name:<26} {count:>6}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the KUQ hallucination-RL rows.")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR, help="dir with knowns_unknowns.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="cap the number of rows")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output parquet path")
    parser.add_argument("--seed", type=int, default=0, help="seed for the --limit sample")
    args = parser.parse_args()

    rows, funnel = build_rows(args.raw_dir, limit=args.limit, seed=args.seed)
    print(format_report(rows, funnel))

    schema.normalise_extra_info(rows)
    schema.validate_rows(rows)
    schema.write_rows_parquet(rows, args.out)
    print(f"\nwrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
