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
"""GSM-IC adapter -- the ``solvable_numeric`` branch of the hallucination domain.

Source
------
``github.com/google-research-datasets/GSM-IC``, the raw ``GSM-IC_2step.json`` +
``GSM-IC_mstep.json`` (a single top-level JSON **array** each -- not JSONL).
Measured: 34,220 + 23,832 = **58,052 rows**, matching the recon report exactly.

Each row is a GSM8K grade-school problem with **one irrelevant sentence spliced
in**; the gold answer is unchanged (design doc section 4.5, D17).  So every row
here is ``solvable=True`` on ``halluc_math_gsmic``, ``template`` B (no options
block), ``perturbation_type="distracting_condition"``, ``branch=
solvable_numeric``; the reward routes it through the ordinary math matcher.

The branch teaches "do not be dragged off course by an unrelated condition",
which is why it carries a distractor and no diagnosis label.

What this adapter certifies, and what it drops
---------------------------------------------

Every emitted row passes three independent checks before it is built; a row that
fails any of them is dropped and counted in the funnel (fail closed -- a gold is
never invented, and option text is never borrowed from another question):

1. **Replay.** ``sentence_template.replace("{role}", role).replace("{number}",
   number)`` -- whitespace-normalised -- occurs verbatim in ``new_question`` and
   does *not* occur in ``original_question``.  Measured 58,052 / 58,052 pass.
   This is the chain-of-custody certificate for the inserted sentence: what we
   call the distractor really is the spliced text.
2. **Gold certificate.** ``original_question`` joins by exact text to OpenAI's
   GSM8K ``train`` + ``test``; every ``<<expr=val>>`` calculator annotation in
   the joined solution is re-evaluated here with an independent whitelisted-AST
   arithmetic walker (no ``eval``); the solution carries exactly one ``#### N``
   marker; ``N`` equals the row's ``answer``; and ``N`` is *derived* -- it is a
   value of the annotation chain or a literal in the derivation text.  Measured
   58,052 / 58,052 pass, 0 annotation disagreements.
3. **No read-off shortcut.** The inserted ``number`` is never numerically equal
   to the gold answer.  Measured 1,258 rows (2.17%) are equal and are dropped:
   on those rows "copy the number from the sentence that does not belong" would
   be a second admissible answer.

Two further de-duplication stages follow:

4. **Presented-question uniqueness** -- 80 duplicate ``new_question`` strings in
   2step are collapsed to one (recon hazard 9).
5. **Per-base-question cap** -- see below.

The per-base-question cap (the one real design decision here)
------------------------------------------------------------

The 58,052 rows are **not 58,052 problems**.  They collapse to **100 distinct
``original_question`` values** (2step 60 + mstep 40), each replicated up to
**640 times** by varying the injected role/number; the headline count overstates
effective diversity by ~580x (recon hazard 5).  Sampling 2,000 rows uniformly --
which is what the design doc's D18 quota asks for -- would feed the model the
same 100 grade-school problems ~20 times each.

So the cap is made explicit instead of implicit: at most
:data:`MAX_PER_BASE_QUESTION` rows survive per base question, sampled flat from
that question's certified variants.  With 100 bases and a cap of 20 that lands on
**exactly 2,000 rows**, i.e. the same quota D18 asks for, reached with a flat
per-problem budget rather than a lopsided one.  The cap is the knob; 2,000 is a
consequence, not an input.  Why flat sampling and not a per-template round-robin
is a measured trade-off -- see :func:`_select_for_base`.

DEVIATIONS FROM THE DESIGN DOC
------------------------------

Each of these is a place the design doc (``HALLUCINATION_RL_DESIGN.md``) states
a number or shape that the raw bytes contradict.  Every measurement below was
reproduced by this adapter's own loader, not copied from the recon report.

1. **Template inventory is 394, not 242.**  Doc sections 4.5 and D17 say
   "58,052 行里共 242 个不同模板" / "规则复用 GSM-IC 的 242 条
   ``sentence_template``".  242 is the *2step-only* count.  Measured: 2step 242,
   mstep 161, **union 394** (9 shared).  Any consumer of the template engine
   (D17's distractor synthesis) built from 242 silently discards the 152
   mstep-only templates.
2. **Top-4 template coverage is 47.2%, not 61%.**  Doc D17: "前 4 个就覆盖 61%".
   Measured on the union: top-4 = 27,392 / 58,052 = **47.2%** (2step alone
   50.8%, mstep alone 52.0%).  No top-N with N <= 5 reaches 61%.
3. **Role / number inventories are 457 / 58, not 272 / 55.**  Doc D17: "全部只由
   ``{role}``（272 个不同取值）与 ``{number}``（55 个）参数化".  Measured union:
   **457 roles / 58 numbers** (272/55 is the 2step-only pair; mstep alone is
   197/33).
4. **Not every template is two-way parameterised.**  Doc D17 says the templates
   are "全部由 ``{role}``/``{number}`` 参数化".  Measured over the 394 union
   templates: 331 carry both placeholders, **57 carry only ``{number}``**, **6
   carry only ``{role}``** (6/480 rows respectively).  The undeclared field is
   then the literal ``"n/a"`` -- ``{number} == "n/a"`` in 60 rows and
   ``role == "n/a"`` in 480 rows, exactly matching the ``role_label``/
   ``number_label`` ``"n/a"`` counts.  A generator that always varies the role
   crashes or writes ``"... is n/a ..."`` on those rows.
5. **There is an unreported 10th field, ``n_steps`` (int).**  Doc section 4.5
   lists nine fields.  All 58,052 rows carry ``n_steps`` too (2step constant 2;
   mstep in {3,4,5,6,7}).  Only ``n_steps`` is projected into
   ``extra_info.difficulty``; the projection is explicit, because ``dict(row)``
   or a field loop would silently carry the extra key.
6. **``sentence_label`` is ``in_topic`` / ``out_topic``, not
   ``out_of_topic``.**  Doc section 4.5 says the labels are "``in_topic`` 还是
   ``out_of_topic``".  Measured: only ``in_topic`` (26,756) and ``out_topic``
   (31,296) ever occur -- a filter written against the doc's spelling matches
   0 rows.
7. **640 answers are comma-grouped and are normalised here.**  Doc section 4.5
   treats ``answer`` as a plain number.  Measured: 640 2step rows carry a
   thousands separator (``"845,640"``), on which ``float(answer)`` raises
   ``ValueError``.  The adapter strips the separator, so the stored gold is the
   numerically identical plain digit string (``"845640"``); the reward's matcher
   never sees a comma-formatted gold.
8. **The doc's own "2,000 GSM-IC rows" quota silently means 100 problems x 20.**
   Doc section 4.9.3 table B row 1 and section 4.5 cap GSM-IC at 2,000 rows and
   say nothing about the 100-distinct-base collapse (recon hazard 5) or the
   9.5x-640x per-problem replication.  This adapter keeps the 2,000 quota but
   makes the per-problem budget an explicit constant
   (:data:`MAX_PER_BASE_QUESTION`).

Where the design doc and the measurements *agree* (recorded so the agreement is
auditable): file sizes and row counts 34,220 / 23,832 / 58,052; the ``out_topic``
/ ``in_topic`` and ``overlapped`` / ``nonoverlapped`` / ``in_range`` /
``out_range`` distributions, exact; the "no generator, pure rule replay" claim,
58,052 / 58,052; and answer-vs-GSM8K agreement, 58,052 / 58,052 with zero
disagreements.

Not this adapter's job
----------------------

* **D17 distractor synthesis** (replaying these templates onto SUM/UMWP/K&K) is
  a separate deliverable; this file only emits native GSM-IC rows and exposes
  the template inventory as a by-product of the funnel.
* **GSM8K-test contamination: measured, and absent.**  All 100 base questions
  join to GSM8K **train**; **0 of 100** occur in GSM8K **test**.  No base
  question is dropped for contamination, and GSM-IC contributes no leak into the
  GSM8K eval split.
* **Deduplication against the stage-1 mixture** belongs to the mix
  (``decontaminate.py`` / ``dedup.py``, design doc section 7.2), not here.

Determinism
-----------

``build_rows(raw_dir, limit, seed)`` is a pure function of its arguments: the
only randomness is a single ``random.Random(seed)`` consumed in a fixed order
(sorted base questions, sorted templates), and ``limit`` truncates an
already-fixed ordering, so a smaller ``limit`` returns a prefix of the larger
build's rows.  Two calls with the same arguments produce byte-identical rows.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import sys
from typing import Any

try:
    import schema
except ImportError:  # pragma: no cover - exercised by running as a script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import schema

# ---------------------------------------------------------------------------
# source facts and contract constants
# ---------------------------------------------------------------------------

DEFAULT_RAW_DIR = "/home/charles/data/reasoning_rl/halluc/raw/gsmic"
DEFAULT_OUT = "/home/charles/data/reasoning_rl/halluc/built/gsmic.parquet"

# The two raw source files, in a fixed order (the ordinal in ``task_id`` counts
# within a file, so the order is part of the row identity and must not drift).
SOURCE_FILES = ("GSM-IC_2step.json", "GSM-IC_mstep.json")

# Third-party cross-check corpus used to certify the gold.  These JSONL files are
# *not* part of the source; the recon report downloaded them for exactly this
# purpose (report section 5) and their absence is a hard failure, not a warning.
CROSS_CHECK_FILES = ("gsm8k_train.jsonl", "gsm8k_test.jsonl")

# Contract placement (design doc sections 3, 4.5, 4.9.3): the distractor branch
# is solvable + numeric + option-less, so it lands on the numeric branch with
# template B.
BRANCH = schema.BRANCH_SOLVABLE_NUMERIC
TEMPLATE = schema.TEMPLATE_B
PERTURBATION_TYPE = "distracting_condition"

# At most this many rows per distinct ``original_question``.  100 base questions
# x 20 lands exactly on the 2,000-row quota the design doc's D18 assigns to
# GSM-IC, with a flat per-problem budget rather than a lopsided one.  See the
# module docstring, "The per-base-question cap".
MAX_PER_BASE_QUESTION = 20

# ``<<expr=val>>`` calculator annotations in a GSM8K solution (the OpenAI
# grade-school-math convention this source inherits).
_ANNOTATION_RE = re.compile(r"<<([^<>]+?)=([^<>]+?)>>")
_PLAIN_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_WHITESPACE_RE = re.compile(r"\s+")

# GSM8K writes its final answer as ``#### <n>``; the source never does (recon
# section 5), so this marker only ever comes from the cross-check corpus.
_FINAL_MARKER = "####"

# Rounding used when comparing a re-evaluated annotation against the value GSM8K
# recorded for it.  The corpus was produced with a two-decimal calculator, so
# ``40*.6`` is stored as ``24`` and ``3.75+2.4+11.85`` as ``18`` -- comparing
# exact binary floats against those would reject 5,376 annotations that are
# arithmetically correct.
_CALCULATOR_DECIMALS = 2

def normalise(text: str) -> str:
    """Collapse whitespace runs to one space and strip the ends.

    ``original_question`` carries irregular double spaces (``"5'6\\".  He grows
    6 inches."``) while ``new_question`` is single-spaced (recon hazard 7).  Any
    equality, dedup or join that skips this step silently misses.
    """
    return _WHITESPACE_RE.sub(" ", text or "").strip()


# ---------------------------------------------------------------------------
# arithmetic certificate
# ---------------------------------------------------------------------------


class AnnotationError(ValueError):
    """An annotation that is not in the measured grammar (never accept it)."""


# The annotation grammar, measured over all 153,760 annotations in the source's
# 100 base questions: ``+ - * /``, parentheses, unary ``+``, numeric constants.
# Nothing else ever occurs, so the walker below is a closed whitelist.


def _eval_node(node: ast.AST) -> float:
    """Evaluate one node of a GSM8K calculator annotation.

    A closed whitelist: numeric constants, unary ``+``/``-`` and binary
    ``+ - * /``.  There are no names, calls or attributes, so no user-controlled
    code path exists (``eval`` is deliberately not used).  Anything outside the
    whitelist raises :class:`AnnotationError`, which the caller turns into a
    dropped row.
    """
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise AnnotationError(f"non-numeric constant {node.value!r}")
        return float(node.value)
    if isinstance(node, ast.UnaryOp):
        value = _eval_node(node.operand)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
        raise AnnotationError(f"unsupported unary operator {type(node.op).__name__}")
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        op = node.op
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            if right == 0:
                raise AnnotationError("division by zero")
            return left / right
        raise AnnotationError(f"unsupported binary operator {type(op).__name__}")
    raise AnnotationError(f"unsupported expression node {type(node).__name__}")


def evaluate_expression(expression: str) -> float:
    """Evaluate ``<<expression=...>>`` with the whitelisted walker.

    Commas and ``$`` are stripped first: GSM8K writes operands as ``482,653``
    and ``$3.50``.
    """
    cleaned = (expression or "").replace(",", "").replace("$", "").strip()
    if not cleaned:
        raise AnnotationError("empty expression")
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as exc:
        raise AnnotationError(f"cannot parse {expression!r}: {exc}") from exc
    return _eval_node(tree)


def parse_number(text: str) -> float | None:
    """First numeric literal in ``text`` (commas and ``$`` stripped), else None."""
    match = _PLAIN_NUMBER_RE.search((text or "").replace(",", "").replace("$", ""))
    return float(match.group(0)) if match else None


def annotation_pairs(solution: str) -> list[tuple[str, str]]:
    """All ``(expression, stated_value)`` pairs of a GSM8K solution."""
    return [(expr, value) for expr, value in _ANNOTATION_RE.findall(solution or "")]


def _same_number(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return False
    return round(left, _CALCULATOR_DECIMALS) == round(right, _CALCULATOR_DECIMALS)


def _format_number(value: float) -> str:
    """Render a float the way it reads in prose (``5.0`` -> ``"5"``)."""
    if value == int(value):
        return str(int(value))
    return repr(value)


def gold_is_derived(solution: str, gold: float) -> bool:
    """Whether ``gold`` is a computed value or a literal in the derivation.

    The last annotation is *not* always the answer: ``Asia saved $210 / $350 =
    <<210/350=0.60>>0.60 or 60% off`` closes with ``#### 60`` -- a final
    percentage conversion that carries no annotation of its own.  Measured over
    the whole source, requiring "a chain value **or** a literal in the derivation
    text" covers 58,052 / 58,052 rows where "a chain value" alone covers 54,572
    (94.0%).
    """
    for _expr, ratio in annotation_pairs(solution):
        if _same_number(parse_number(ratio), gold):
            return True
    body = _ANNOTATION_RE.sub(" ", solution.split(_FINAL_MARKER)[0])
    literal = _format_number(gold)
    return re.search(rf"(?<![\d.]){re.escape(literal)}(?![\d])", body) is not None


def certify_gold(question: str, answer: str, gsm8k_index: dict[str, str]) -> str | None:
    """Independently certify one row's gold; return the reason it fails, or None.

    The certificate is *recomputation*, not comparison: the joined GSM8K
    solution's own calculator annotations are re-evaluated here and must
    reproduce the values the corpus recorded, and the ``####`` terminal value
    must be the row's ``answer`` and must appear in the derivation.  A row whose
    gold cannot be certified is dropped by the caller -- the answer is never
    taken on trust from the source file (design doc section 9, fail closed).
    """
    solution = gsm8k_index.get(normalise(question))
    if solution is None:
        return "original question is not in the GSM8K cross-check corpus"
    pairs = annotation_pairs(solution)
    if not pairs:
        return "joined GSM8K solution carries no <<expr=val>> annotation"
    for expression, stated in pairs:
        try:
            recomputed = evaluate_expression(expression)
        except AnnotationError as exc:
            return f"annotation <<{expression}={stated}>> is not evaluable: {exc}"
        if not _same_number(recomputed, parse_number(stated)):
            return (
                f"annotation <<{expression}={stated}>> recomputes to {recomputed} "
                f"(rounded to {_CALCULATOR_DECIMALS} decimals)"
            )
    if solution.count(_FINAL_MARKER) != 1:
        return f"joined GSM8K solution carries {solution.count(_FINAL_MARKER)} '####' markers"
    terminal = parse_number(solution.split(_FINAL_MARKER)[-1])
    gold = parse_number(answer)
    if gold is None:
        return f"answer {answer!r} is not a plain number"
    if not _same_number(terminal, gold):
        return f"GSM8K terminal answer {terminal} disagrees with source answer {gold}"
    if not gold_is_derived(solution, gold):
        return f"answer {gold} appears in neither the chain values nor the derivation text"
    return None


def load_gsm8k_index(raw_dir: str) -> dict[str, str]:
    """Whitespace-normalised GSM8K question -> full solution text (train + test).

    Missing files are a hard error: without the cross-check corpus nothing can be
    certified, and an empty index would silently turn the certificate into a
    "drop every row" rule instead of a loud failure.
    """
    index: dict[str, str] = {}
    for name in CROSS_CHECK_FILES:
        path = os.path.join(raw_dir, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"cross-check corpus {path} is missing; the gold certificate needs "
                f"{CROSS_CHECK_FILES} alongside the source files in {raw_dir}"
            )
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                index.setdefault(normalise(record["question"]), record["answer"])
    return index


def distractor_text(record: dict) -> str:
    """The inserted sentence, replayed from its own template (normalised).

    ``sentence_template.replace("{role}", role).replace("{number}", number)`` --
    measured to occur verbatim in ``new_question`` for 58,052 / 58,052 rows and
    to be absent from ``original_question`` for 58,052 / 58,052 rows.  A row
    where either fails cannot be traced back to its template and is dropped.
    """
    return normalise(
        record["sentence_template"]
        .replace("{role}", record["role"])
        .replace("{number}", record["number"])
    )


# ---------------------------------------------------------------------------
# loading and filtering
# ---------------------------------------------------------------------------


def load_source(raw_dir: str) -> list[dict[str, Any]]:
    """Read both source files into flat candidate records, with a stable ordinal.

    Each candidate is ``{"file", "ordinal", "record"}``; ``task_id`` is derived
    from ``(file, ordinal)`` -- the source's own identity, never a random value.
    """
    candidates: list[dict[str, Any]] = []
    for name in SOURCE_FILES:
        path = os.path.join(raw_dir, name)
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError(f"{path} is not a top-level JSON array (it is not JSONL)")
        for ordinal, record in enumerate(payload):
            candidates.append({"file": name, "ordinal": ordinal, "record": record})
    return candidates


def _passes_replay(candidate: dict) -> str | None:
    """Replay certificate: the spliced sentence is in the new question only."""
    record = candidate["record"]
    template = record["sentence_template"]
    if "{role}" not in template and "{number}" not in template:
        return f"template {template!r} carries no placeholder"
    inserted = distractor_text(record)
    presented = normalise(record["new_question"])
    original = normalise(record["original_question"])
    if not inserted:
        return "replayed distractor sentence is empty"
    if inserted not in presented:
        return f"replayed distractor {inserted!r} is absent from the presented question"
    if inserted in original:
        return f"replayed distractor {inserted!r} already occurs in the original question"
    return None


def _gold_answer(record: dict) -> str:
    """The certified gold as a plain digit string (comma separators removed)."""
    return record["answer"].replace(",", "").strip()


def _passes_no_readoff(candidate: dict) -> str | None:
    """Drop rows whose inserted number *is* the gold (a copy-the-distractor win)."""
    record = candidate["record"]
    inserted = parse_number(record["number"])
    gold = parse_number(_gold_answer(record))
    if inserted is not None and _same_number(inserted, gold):
        return f"inserted number {inserted} equals the gold answer"
    return None


def _select_for_base(candidates: list[dict], cap: int, rng: random.Random) -> list[dict]:
    """A flat ``rng.sample`` of at most ``cap`` rows for one base question.

    Flat, not stratified by template, and that is a measured trade-off.  A base
    question is served by 7-8 templates whose source frequencies differ by more
    than an order of magnitude (the top template holds 7,744 rows, the rarest
    80).  Round-robin over templates -- the obvious way to keep every template --
    covers all 394 templates but drags the D17 difficulty axes away from the
    source marginals the design's "三类天然配平 -> 无标签泄漏" argument rests
    on: measured, ``role_label`` becomes 48.2 / 43.1 / **8.6** (vs 50.3 / 48.3 /
    1.4 at source) because ``role_label == "n/a"`` rides on the rare
    single-placeholder templates.  A flat sample keeps every marginal: measured
    ``role_label`` 50.3 / 48.3 / 1.4, ``number_label`` 50.9 / 49.1 / 0.05,
    ``sentence_label`` 54.5 / 45.6 (source 53.9 / 46.1), while still using 327 of
    the 394 templates.  Full template coverage is not lost work: D17 replays the
    complete 394-template inventory onto the *other* pools, so the templates this
    sample skips still reach the mixture.

    ``candidates`` arrives in ``(source file, ordinal)`` order and ``rng`` is
    consumed in sorted-base order, so the selection is reproducible; ``limit``
    never reaches this function, which is why a smaller ``limit`` yields a prefix
    of the larger build.
    """
    return rng.sample(candidates, min(cap, len(candidates)))


def _build_row(candidate: dict, index: int, seed: int) -> dict:
    """Assemble one schema row from a certified candidate."""
    record = candidate["record"]
    template_text = record["sentence_template"]
    inserted_parts = []
    if "{role}" in template_text:
        inserted_parts.append(f"role={record['role']}")
    if "{number}" in template_text:
        inserted_parts.append(f"number={record['number']}")
    return schema.make_row(
        data_source=schema.SOURCE_GSMIC,
        question=normalise(record["new_question"]),
        ground_truth=schema.build_ground_truth(
            solvable=True,
            answer=_gold_answer(record),
            correct_option_id=None,
            has_diagnosis_label=False,
            perturbation_type=PERTURBATION_TYPE,
        ),
        template=TEMPLATE,
        branch=BRANCH,
        options=None,
        extra_info={
            "split": "train",  # GSM-IC ships two difficulty slices, not real splits
            "index": index,
            "task_id": f"gsmic:{candidate['file'].removesuffix('.json')}:{candidate['ordinal']}",
            "difficulty": f"n_steps={record['n_steps']}",
            "seed": seed,
            "solvable": True,
            "has_diagnosis_label": False,
            "judgment_only": False,
            "correct_option_id": "",
            "perturbation_type": PERTURBATION_TYPE,
            "paired_original_text": normalise(record["original_question"]),
            "perturbed_entity_text": "; ".join(inserted_parts),
            "distractor_text": distractor_text(record),
            "distractor_labels": {
                "role_label": record["role_label"],
                "number_label": record["number_label"],
                "sentence_label": record["sentence_label"],
            },
        },
    )


def build_rows(
    raw_dir: str,
    limit: int | None = None,
    seed: int = 0,
) -> tuple[list[dict], dict]:
    """Build the GSM-IC branch rows plus the funnel that produced them.

    Args:
        raw_dir: directory holding ``GSM-IC_2step.json``, ``GSM-IC_mstep.json``
            and the two GSM8K cross-check JSONL files.
        limit: keep at most this many rows; ``None`` keeps the whole capped set.
            Applied last, so a smaller limit is a prefix of a larger build's rows.
        seed: seeds the single ``random.Random`` used by the per-base selection.

    Returns:
        ``(rows, funnel)`` -- an ordered mapping of filter stage name -> rows
        remaining after that stage, starting at the raw row count.  Each stage
        name is also the reason rows were dropped; the drop count of a stage is
        the difference to the previous stage.
    """
    rng = random.Random(seed)
    gsm8k_index = load_gsm8k_index(raw_dir)

    funnel: dict[str, int] = {}
    candidates = load_source(raw_dir)
    funnel["raw_rows"] = len(candidates)

    replayed = []
    for candidate in candidates:
        if _passes_replay(candidate) is None:
            replayed.append(candidate)
    funnel["template_replay_verified"] = len(replayed)

    certified = []
    for candidate in replayed:
        record = candidate["record"]
        reason = certify_gold(
            record["original_question"], _gold_answer(record), gsm8k_index
        )
        if reason is None:
            certified.append(candidate)
    funnel["gold_certified"] = len(certified)

    no_readoff = [c for c in certified if _passes_no_readoff(c) is None]
    funnel["distractor_number_not_gold"] = len(no_readoff)

    # De-duplicate the *presented* text: the source repeats some new_question
    # strings verbatim (80 rows in 2step), which would enter a batch as literal
    # duplicates.  Keeping the first in (file, ordinal) order is deterministic.
    seen_presented: set[str] = set()
    unique: list[dict] = []
    for candidate in no_readoff:
        presented = normalise(candidate["record"]["new_question"])
        if presented in seen_presented:
            continue
        seen_presented.add(presented)
        unique.append(candidate)
    funnel["presented_question_unique"] = len(unique)

    grouped: dict[str, list[dict]] = {}
    for candidate in unique:
        grouped.setdefault(normalise(candidate["record"]["original_question"]), []).append(
            candidate
        )

    # Sorted base questions keep the rng consumption order independent of dict
    # insertion order, which is what makes the build reproducible.
    selected = {
        base: _select_for_base(grouped[base], MAX_PER_BASE_QUESTION, rng)
        for base in sorted(grouped)
    }
    bases = sorted(selected)
    width = max((len(rows) for rows in selected.values()), default=0)

    # Interleave the bases so a truncated build stays diverse across problems
    # instead of exhausting the first few bases.
    ordered: list[dict] = []
    for position in range(width):
        for base in bases:
            if position < len(selected[base]):
                ordered.append(selected[base][position])
    funnel["per_base_capped"] = len(ordered)

    if limit is not None:
        ordered = ordered[: max(limit, 0)]
    funnel["limit_applied"] = len(ordered)

    rows = [_build_row(candidate, index, seed) for index, candidate in enumerate(ordered)]
    return rows, funnel


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def _breakdown(rows: list[dict], key) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = key(row)
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def describe(rows: list[dict]) -> str:
    """Human-readable per-branch / per-template / per-solvable breakdown."""
    lines = []

    def block(title: str, counts: dict[str, int], width: int = 46) -> None:
        lines.append(f"{title}:")
        total = sum(counts.values()) or 1
        for name, count in counts.items():
            share = 100.0 * count / total
            lines.append(f"  {name[:width]:<{width}} {count:>7}  {share:5.1f}%")

    block("branch", _breakdown(rows, lambda row: row["extra_info"]["branch"]))
    block("template", _breakdown(rows, lambda row: row["extra_info"]["template"]))
    block(
        "solvable",
        _breakdown(rows, lambda row: str(_payload(row)["solvable"])),
    )
    block("data_source", _breakdown(rows, lambda row: row["data_source"]))
    block("ability", _breakdown(rows, lambda row: row["ability"]))
    block("difficulty", _breakdown(rows, lambda row: row["extra_info"]["difficulty"]))
    block(
        "distractor_labels.sentence_label",
        _breakdown(rows, lambda row: row["extra_info"]["distractor_labels"]["sentence_label"]),
    )
    block(
        "distractor_labels.role_label",
        _breakdown(rows, lambda row: row["extra_info"]["distractor_labels"]["role_label"]),
    )
    block(
        "distractor_labels.number_label",
        _breakdown(rows, lambda row: row["extra_info"]["distractor_labels"]["number_label"]),
    )
    distinct_bases = len({row["extra_info"]["paired_original_text"] for row in rows})
    lines.append(f"distinct base questions: {distinct_bases}")
    lines.append(f"distinct presented questions: {len({row['prompt'][0]['content'] for row in rows})}")
    lines.append(f"distinct task_id: {len({row['extra_info']['task_id'] for row in rows})}")
    return "\n".join(lines)


def _payload(row: dict) -> dict:
    return json.loads(row["reward_model"]["ground_truth"])


def _funnel_lines(funnel: dict) -> str:
    lines = ["funnel:"]
    previous = None
    for stage, count in funnel.items():
        dropped = "" if previous is None else f"  (-{previous - count})"
        lines.append(f"  {stage:<32} {count:>7}{dropped}")
        previous = count
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR, help="GSM-IC + GSM8K raw files")
    parser.add_argument("--limit", type=int, default=None, help="keep at most N rows")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output parquet path")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    args = parser.parse_args()

    rows, funnel = build_rows(args.raw_dir, limit=args.limit, seed=args.seed)
    if not rows:
        raise SystemExit("adapter produced no rows; refusing to write an empty parquet")

    schema.normalise_extra_info(rows)
    schema.validate_rows(rows)
    schema.write_rows_parquet(rows, args.out)

    print(_funnel_lines(funnel))
    print()
    print(describe(rows))
    print()
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
