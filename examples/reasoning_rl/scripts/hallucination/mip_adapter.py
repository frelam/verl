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
"""MiP adapter -- the three-tier *bare* unsolvable source (design doc section 4.3 / D12).

Source: ``github.com/tianyi-lab/MiP-Overthinking`` ``data/{gsm8k,svamp,math,formula}.json``
(984 rows; **not** a HF dataset).  Each paired row carries a solvable original
(``question`` + the source's own ``answer``/``solution``) and its truncated twin
(``insufficient_question``), so one source row supplies both sides of the
answerability contrast without a second file.

What this adapter emits
-----------------------

Two branches, both on template B (no option block, D12) and both under
``data_source=halluc_math_mip``:

=========================  ==========================================  =========
branch                     gold                                        template
=========================  ==========================================  =========
``unsolvable_bare``        ``\\boxed{UNSOLVABLE}`` (answer=None)        B
``solvable_numeric``       ``\\boxed{<the source's own answer>}``       B
=========================  ==========================================  =========

Design doc section 4.3 says both things about the solvable side -- "可解侧用原题
自带 answer 做 exact-match" / "solvable 两版各出一行" in the implementation
bullet, but "MiP 可解侧（原题 634 条）默认不进池" in the last bullet.  This
adapter emits both, because the pair *is* the contrast (§4.9.4 成对性: the same
base question with and without a necessary premise); the solvable half is
flagged ``error_type=""`` and can simply be dropped by ``mix_halluc.py`` if the
quota does not want it.  Solving it does not need the pair to be broken up:
every solvable row is certifiable on its own (see ``_certify_gsm8k``).

Why no options, and no ``distractor_mining`` import
---------------------------------------------------

D12: ``has_diagnosis_label=false``, ``correct_option_id=null``, **no option
block**.  The recon report section 4.2 measured that *any* option set built from
MiP deletions is compromised -- "pick the option that does not occur in the
question" scores 94.2% when restricted to the uniquely-absent option and
**626/626 = 100.0%** for the any-absent variant (the design doc's 95.6% is
close but not exactly reproduced).  This module therefore never imports
``distractor_mining``: there is no option block to mine for, and mining one
would re-open the leak the design closed.

Fail-closed admission
---------------------

Every row must pass every certificate below or it is dropped and counted in the
funnel (``build_rows`` returns it): no invented gold, no cross-question
fallback, no "probably fine".
1. the source row carries both ``question`` and ``insufficient_question``;
2. the two texts really differ (whitespace-normalised);
3. the word-level diff has **exactly one** changed region;
4. the deleted side holds **exactly one** distinct numeric value;
5. that value is **absent** from the presented (truncated) question;
6. the replace side introduces **no** numeric value (else the question may still
   be answerable with a different number);
7. the deleted value is **necessary**: it occurs in the source's own derivation
   chain (gsm8k ``<<expr=val>>`` annotations, math ``solution``), compared as
   Decimals with percent folding;
8. a solvable row additionally needs a gold the source itself recomputes (see
   the two ``_certify_*`` functions).

Two claims in the recon are conventions, not measurements, and this adapter
takes the fail-closed side of both: rows with **no** derivation chain at all (2
gsm8k rows) and placeholders whose deleted value is *not* used in the chain (3
rows) are dropped rather than credited.

``verify_mip.py`` re-checks all eight of these from the written artifact alone,
with its own implementations and in the *opposite* diff direction (it recovers
the deleted span by diffing ``insufficient -> original``), so nothing here is
taken on trust.

Reproducing the recon's funnel conventions
------------------------------------------

The funnel stages are computed with the *same* conventions as
``scratch/halluc_recon/probe_mip.py`` so the numbers are directly comparable
(recon report section 4.1): word-level ``difflib`` with ``autojunk=False``,
membership tests on string-normalised numerics (``4.00`` stays ``4.00``, so a
deleted ``$4.00`` counts as gone), necessity on ``Decimal`` with percent folding
(``10`` vs ``0.1``).  Mixing the two conventions moves ``gone`` between 298 and
299 -- the recon needed three attempts to get this right; see §4.1 note 3.

DEVIATIONS FROM THE DESIGN DOC (recon report wins)
--------------------------------------------------

1. **Strict pool is 270, not 276** (doc section 4.3: "严口径 276").  Measured
   with the doc's own stage definitions: 260 pure deletions pass the necessity
   check, 2 more have *no* ``<<expr=val>>`` chain at all (doc: "另 23 待复核、14
   占位词型无需链校验"; recon: "260 pass / 24 fail / 2 empty-chain"), and of the
   14 placeholder rows 10 pass necessity, 3 fail it and 1 is still visible in
   the truncated question.  260 + 10 = **270**.  The doc's 276 additionally
   (a) credits the 2 chainless rows, (b) exempts all 14 placeholders from the
   necessity check and (c) admits a placeholder whose deleted value is still
   visible.  Recon section 6 hazard 2 says an adapter "must handle a missing
   chain rather than assume the annotation exists"; this adapter drops instead
   of crediting, per the fail-closed rule.
2. **svamp (300) and formula (50) contribute 0 rows**, against doc section 4.3's
   implementation note "SVAMP 配对成功 → 文本 diff 定位被删条件 ... 写
   extra_info.deleted_condition_text".  Recon section 5: 300/300 svamp rows are
   cross-problem splices (Body of problem A + Question of problem B) with **no
   deletion at all**, and ``answer`` is problem B's answer (recon: "answer ==
   the Body-source problem's answer in only 6/300") -- unusable as a gold for
   the visible text; formula has neither ``answer`` nor ``question``.  Dropping
   both is why the funnel starts 984 → 634.
3. **No option block is built at all** (D12), which the recon makes mandatory
   rather than optional: H1 "the option not found in the question" is 100.0%
   for the any-absent variant (626/626), i.e. *worse* than the doc's 95.6%.
   Prompt length/template are identical on both sides, so no surface cue was
   added.  See ``verify_mip.py`` check L3 for the re-measurement.
4. **MiP does contribute to both the solvable and the unsolvable side** (doc
   section 4.3, last bullet, says the solvable side does not enter the pool).
   Both are emitted; the mix decides.  This is also what makes the L3
   balanced-accuracy check meaningful on this source's own rows.
5. **2 gsm8k solvable rows are dropped** because their answer key contradicts
   their own derivation (e.g. the chain ends ``24/240 = 0.10`` while ``answer``
   says ``#### 10``).  254 gsm8k unsolvable rows survive, 252 of them keep their
   solvable twin (math: 16/16).
6. **The claim that a support-≥5 vocabulary makes the L3 Naive Bayes sane does
   not hold on MiP**, and this adapter says so in its artifact rather than
   hiding it.  Measured by ``verify_mip.py``'s own implementation on the full
   538-row artifact, the 5-fold out-of-fold balanced accuracy is 0.2566 at
   train support ≥ 5 and never rises to chance as the vocabulary widens or
   narrows (support ≥ 1: 0.1192, ≥ 2: 0.1882, ≥ 3: 0.1823, ≥ 10: 0.3345) --
   *all below chance*, i.e. the estimator sits in the sign-flipped regime the
   UMWP recon documents (train-fold class separation +0.72 / -0.53 flips to
   -0.92 / +1.24 held out).  The un-inverted reading is 0.7434.  A length-only
   threshold already reaches 0.6667 (mean 41.3 question tokens solvable vs
   31.1 unsolvable): the deletion *is* a visible surface cue.  It is not an
   option-selection shortcut (there is no option block, D12) and the design
   accepts it, but it is real -- ``verify_mip.py`` prints the raw number, the
   inverted reading and the length baseline on every run, so nothing here is
   decided by the below-chance figure alone.

Field choices that the schema leaves free (audit-only fields)
-------------------------------------------------------------

``paired_original_text`` = the solvable original question; ``deleted_condition_text``
= the deleted span (word-diff text); ``perturbed_entity_text`` = **the source's
own derivation text** (the gsm8k ``answer`` field / the math ``solution``).
The last one repurposes a field the design doc uses for the deleted sentence
(``halluc_samples.md`` section 5.3) -- that sentence is already carried by
``deleted_condition_text`` -- so that ``verify_mip.py`` can re-derive both the
gold answer and the necessity of the deleted value **from the artifact alone**,
without going back to ``/home/charles/data/...``.  ``canonical_solution`` and
``role_words`` stay empty (K&K-only), ``distractor_labels`` keeps its fixed
three-key shape, ``split`` is always ``"train"`` (MiP ships no native splits).
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import random
import re
import sys
from decimal import Decimal, InvalidOperation

try:  # run as a script (its own dir is sys.path[0]) or via the conftest hook
    import schema
except ImportError:  # pragma: no cover - only when imported from elsewhere
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import schema

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

DEFAULT_RAW_DIR = "/home/charles/data/reasoning_rl/halluc/raw/mip"
DEFAULT_OUT = "/home/charles/data/reasoning_rl/halluc/built/mip.parquet"

# Every file the source ships; only the two with a paired original can certify a
# gold (see DEVIATIONS 2), but all four are counted so the funnel starts at the
# raw row count of 984.
SOURCE_FILES = ("gsm8k", "svamp", "math", "formula")
PAIRED_FILES = ("gsm8k", "math")

DATA_SOURCE = schema.SOURCE_MIP
TEMPLATE = schema.TEMPLATE_B
SOLVABLE_BRANCH = schema.BRANCH_SOLVABLE_NUMERIC
UNSOLVABLE_BRANCH = schema.BRANCH_UNSOLVABLE_BARE

PERTURBATION = "missing_condition"
# D18 defect class: "缺一条必要条件（题面不可见）" -> the three-tier bare bucket.
ERROR_TYPE = "missing_condition"
FAMILY_DELETION = "deletion"  # the premise was removed outright
FAMILY_PLACEHOLDER = "placeholder"  # a numeric premise became "many"/"some"/...

TASK_PREFIX = "mip"
SIDE_UNSOLVABLE = "unsolvable"
SIDE_SOLVABLE = "solvable"

DIFFICULTY_UNLABELLED = "unlabelled"  # gsm8k ships no native difficulty

# The funnel, in the order rows are lost.  ``build_rows`` increments every stage
# a row survives, so the returned mapping is monotone by construction.
PER_ROW_STAGES = (
    "raw_rows",
    "paired_original_present",
    "question_differs",
    "single_value_single_region",
    "deleted_value_absent",
    "no_inserted_numeric",
    "necessary_value_certified",
)
LIMIT_STAGE = "after_limit"
FUNNEL_STAGES = PER_ROW_STAGES + (LIMIT_STAGE,)


# ---------------------------------------------------------------------------
# text / numeric helpers (the recon's conventions, see the module docstring)
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_WS_RE = re.compile(r"\s+")
_ANN_RE = re.compile(r"<<([^>]*)>>")
_BOXED_RE = re.compile(r"\\boxed\s*\{")
_HASH_RE = re.compile(r"####\s*(.+?)\s*$", re.S)
# The last "= <number>" of the *prose* (annotations are blanked first).
_TRAILER_RE = re.compile(r"=\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)")


def _normalise_ws(text: str) -> str:
    return _WS_RE.sub(" ", text or "").strip()


def _numeric_spans(text: str) -> list[str]:
    """All numeric tokens, in order of appearance."""
    return [m.group(0) for m in _NUM_RE.finditer(text or "")]


def _normalise_number(token: str) -> str:
    """Membership-normal form: ``4.00`` and ``4.0`` stay distinct from ``4``.

    Deliberately *not* a value normalisation -- the recon's funnel counts a
    deleted ``$4.00`` as gone unless that literal string reappears (section 4.1
    note 3).  Value equality lives in :func:`_necessary`, which uses Decimals.
    """
    return token.replace(",", "").rstrip(".").removesuffix(".0")


def _numeric_values(text: str) -> set[str]:
    return {_normalise_number(t) for t in _numeric_spans(text)}


def _decimal(token: str) -> Decimal | None:
    try:
        return Decimal(str(token).strip().replace(",", ""))
    except (InvalidOperation, AttributeError, ValueError):
        return None


def _necessary(value: str, chain: str) -> bool | None:
    """Does ``value`` occur in the source's derivation ``chain``?

    ``None`` means there is nothing to check against (an empty chain); the
    caller drops such a row rather than crediting it.  Percent folding: a
    deleted ``10`` is accepted against a chain ``0.1`` and vice versa, which is
    what the doc's "折算百分数" means.
    """
    if not (chain or "").strip():
        return None
    chain_values = _numeric_values(chain)
    if value in chain_values:
        return True
    target = _decimal(value)
    if target is None:
        return False
    for candidate in chain_values:
        current = _decimal(candidate)
        if current is None or current == 0 or target == 0:
            continue
        if current == target or current * 100 == target or target * 100 == current:
            return True
    return False


def _chain_text(source: str, row: dict) -> str:
    """The derivation the source itself gives for this row's answer."""
    if source == "gsm8k":
        return " ".join(_ANN_RE.findall(row.get("answer") or ""))
    if source == "math":
        return row.get("solution") or ""
    return ""


def _derivation_text(source: str, row: dict) -> str:
    """The raw derivation text written into ``extra_info.perturbed_entity_text``."""
    if source == "gsm8k":
        return row.get("answer") or ""
    if source == "math":
        return row.get("solution") or ""
    return ""


# ---------------------------------------------------------------------------
# word-level diff (recon convention: difflib, autojunk=False)
# ---------------------------------------------------------------------------


def _opcodes(a: str, b: str):
    a_words, b_words = (a or "").split(), (b or "").split()
    matcher = difflib.SequenceMatcher(a=a_words, b=b_words, autojunk=False)
    return matcher.get_opcodes(), a_words, b_words


def _diff_regions(original: str, insufficient: str) -> dict | None:
    """Forward diff of ``original -> insufficient``, or ``None`` if equivalent.

    ``deleted`` is the old side of every delete/replace opcode, ``inserted`` the
    new side of every insert/replace.  ``n_regions`` counts the changed opcode
    runs: the doc's "single value" bucket needs exactly one of each.
    """
    if _normalise_ws(original) == _normalise_ws(insufficient):
        return None
    opcodes, a_words, b_words = _opcodes(original, insufficient)
    deleted: list[str] = []
    inserted: list[str] = []
    changed = False
    n_regions = 0
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        n_regions += 1
        if tag in ("delete", "replace"):
            deleted.extend(a_words[i1:i2])
        if tag in ("insert", "replace"):
            inserted.extend(b_words[j1:j2])
            changed = True
    return {
        "deleted": " ".join(deleted),
        "inserted": " ".join(inserted),
        "pure_deletion": not changed,
        "n_regions": n_regions,
        "deleted_values": _numeric_values(" ".join(deleted)),
        "inserted_values": _numeric_values(" ".join(inserted)),
    }


# ---------------------------------------------------------------------------
# gold certificates (fail closed: no certificate, no row)
# ---------------------------------------------------------------------------


def _safe_arithmetic(expression: str) -> float | None:
    """Evaluate a simple ``+ - * / ( )`` expression, or ``None``.

    Digits only, no names: ``eval`` sees a literal-only namespace so this can
    never do anything but arithmetic (and it is only ever fed the source's own
    ``<<expr=val>>`` annotation bodies).
    """
    cleaned = (expression or "").strip().replace("^", "**")
    if not cleaned or not re.fullmatch(r"[0-9+\-*/(). ]+", cleaned):
        return None
    try:
        return eval(cleaned, {"__builtins__": {}}, {})  # noqa: S307 - literal-only
    except Exception:  # noqa: BLE001 - any failure means "cannot certify"
        return None


def _annotation_pairs(answer_text: str) -> list[tuple[str, str]]:
    pairs = []
    for body in _ANN_RE.findall(answer_text or ""):
        if "=" in body:
            expression, value = body.rsplit("=", 1)
            pairs.append((expression.strip(), value.strip()))
    return pairs


def _last_boxed(text: str) -> str | None:
    """The content of the *last* balanced ``\\boxed{...}``, or ``None``."""
    result: str | None = None
    for match in _BOXED_RE.finditer(text or ""):
        start = match.end()
        depth = 1
        i = start
        while i < len(text) and depth:
            char = text[i]
            if char == "\\":
                i += 2
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            i += 1
        if depth == 0:
            result = text[start : i - 1]
    return result


def _certify_gsm8k(answer_text: str) -> tuple[str | None, str | None]:
    """Extract the gsm8k gold and certify it against the source's own arithmetic.

    Two tiers, both a real recomputation rather than a substring test:

    ``T1``  the last ``<<expr=val>>`` annotation evaluates to ``val`` and
            ``val`` is the ``####`` value;
    ``T2``  the final ``= <n>`` of the prose that follows the last annotation
            equals the ``####`` value (the source often does its last step in
            prose: ``... he has 25-2 = 23 jewels. #### 23``).

    Any row whose gold passes neither tier is dropped: 2 of the 254 gsm8k rows
    have an answer key that contradicts their own chain (``24/240 = 0.10`` vs
    ``#### 10``), and a wrong gold in the reward is worse than 2 lost rows.
    """
    if not answer_text:
        return None, None
    match = _HASH_RE.search(answer_text.strip())
    if match is None:
        return None, None
    gold = match.group(1).strip().replace(",", "").replace("$", "").rstrip(".")
    gold_value = _decimal(gold)
    if not gold or gold_value is None:
        return None, None

    for expression, value in _annotation_pairs(answer_text):
        evaluated = _safe_arithmetic(expression)
        if evaluated is None:
            continue
        annotated = _decimal(value)
        if annotated is not None and annotated == gold_value == _decimal(str(evaluated)):
            return gold, "T1"

    # T2 looks at the *prose* only, so the annotation bodies are blanked first --
    # otherwise "the number to the right of the last =" would just re-read the
    # annotation T1 already evaluated, and would not be an independent check.
    body = _ANN_RE.sub(" ", answer_text[: answer_text.rfind("####")])
    trailers = _TRAILER_RE.findall(body)
    if trailers and _decimal(trailers[-1]) == gold_value:
        return gold, "T2"
    return None, None


def _certify_math(row: dict) -> tuple[str | None, str | None]:
    """The math gold is the last ``\\boxed{}`` the source's own solution reaches."""
    answer = (row.get("answer") or "").strip()
    if not answer:
        return None, None
    boxed = _last_boxed(row.get("solution") or "")
    if boxed is not None and _normalise_ws(boxed) == _normalise_ws(answer):
        return answer, "solution_boxed"
    return None, None


# ---------------------------------------------------------------------------
# reading the raw files
# ---------------------------------------------------------------------------


def _load(raw_dir: str, name: str) -> list[dict]:
    path = os.path.join(raw_dir, f"{name}.json")
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list, got {type(data).__name__}")
    return data


def _math_slug(unique_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", unique_id or "").strip("_")


def _base_task_id(source: str, index: int, row: dict) -> str:
    """A stable id built from the source's own identifier (never random).

    ``math`` ships ``unique_id`` (``test/algebra/478.json``); the other files
    have no id at all, so the file-local row index is the stable key.  The
    ``-solvable`` / ``-unsolvable`` suffix is what keeps the pair distinct for
    hard replay's dedup key.
    """
    if source == "math" and row.get("unique_id"):
        slug = _math_slug(str(row["unique_id"]))
    else:
        slug = f"{index:04d}"
    return f"{TASK_PREFIX}-{source}-{slug}"


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def _advance(funnel: dict, reached: str) -> None:
    """Credit one row to the last stage it survived.

    Each funnel value is "rows remaining after this stage's filter", so a row
    increments exactly the final filter it passed -- the next ``_advance`` call
    for the same row means it also passed one more filter.  A row that fails a
    filter simply never advances again, which is what makes the sequence
    monotone and each stage-to-stage drop readable as "why was this dropped".
    """
    if reached not in PER_ROW_STAGES:  # pragma: no cover - programming error
        raise ValueError(f"unknown funnel stage {reached!r}")
    funnel[reached] += 1


def _classify(original: str, insufficient: str) -> dict | None:
    """The doc's L1/L2 bucket for one paired row (``None`` == not perturbed)."""
    diff = _diff_regions(original, insufficient)
    if diff is None:
        return None
    n_values = len(diff["deleted_values"])
    if diff["n_regions"] > 1:
        bucket = "multi_region"
    elif n_values == 1:
        bucket = "single"
    elif n_values == 0:
        bucket = "zero"
    else:
        bucket = "multi"
    diff["bucket"] = bucket
    diff["n_values"] = n_values
    return diff


def build_rows(
    raw_dir: str = DEFAULT_RAW_DIR,
    limit: int | None = None,
    seed: int = 0,
) -> tuple[list[dict], dict]:
    """Build the MiP rows and the admission funnel.

    Args:
        raw_dir: directory holding ``gsm8k.json`` / ``svamp.json`` /
            ``math.json`` / ``formula.json``.
        limit: cap on the number of **rows** written.  Pairs are admitted whole
            (an unsolvable row and its solvable twin), so the artifact stays
            balanced and the effective size is the largest even number
            ``<= limit``.  ``None`` writes everything the certificates admit.
        seed: seeds ``random.Random`` for the output ordering only -- the row
            *set* is fixed by the certificates, so the same seed always yields
            byte-identical rows.

    Returns:
        ``(rows, funnel)``.  ``funnel`` is an ordered mapping of stage name ->
        rows remaining after that stage, starting at the raw row count of 984.
    """
    if limit is not None and limit < 2:
        raise ValueError(f"limit must be at least 2 (one pair), got {limit}")

    funnel = {name: 0 for name in FUNNEL_STAGES}
    candidates: list[dict] = []

    for source in SOURCE_FILES:
        for index, row in enumerate(_load(raw_dir, source)):
            _advance(funnel, "raw_rows")
            original = row.get("question") or ""
            insufficient = row.get("insufficient_question") or ""
            if source not in PAIRED_FILES or not original or not insufficient:
                # svamp: no original, no deletion (DEVIATIONS 2).  formula: no
                # answer and no question at all.
                continue
            _advance(funnel, "paired_original_present")
            diff = _classify(original, insufficient)
            if diff is None:
                continue  # byte-identical / whitespace-only: no perturbation
            _advance(funnel, "question_differs")
            if diff["bucket"] != "single":
                # >=2 deleted values, 0 deleted values, or >1 changed region --
                # the recon's 209 / 55 / 37 buckets (section 4.1 note 1: do NOT
                # bucket by numeric value alone, that inflates the pool by 12).
                continue
            _advance(funnel, "single_value_single_region")
            value = next(iter(diff["deleted_values"]))
            if diff["deleted_values"] & _numeric_values(insufficient):
                continue  # the "deleted" value is still readable in the prompt
            _advance(funnel, "deleted_value_absent")
            if diff["inserted_values"]:
                continue  # the edit introduced a new number: still answerable
            _advance(funnel, "no_inserted_numeric")
            if _necessary(value, _chain_text(source, row)) is not True:
                # Includes the chainless rows (None) and the placeholders whose
                # value the source never used (False): DEVIATIONS 1/6.
                continue
            _advance(funnel, "necessary_value_certified")
            candidates.append(
                {
                    "source": source,
                    "index": index,
                    "row": row,
                    "diff": diff,
                    "value": value,
                    "family": FAMILY_DELETION if diff["pure_deletion"] else FAMILY_PLACEHOLDER,
                }
            )

    # Round-robin across the source files before the limit is applied, so a
    # bounded build still spans both (math is 16 of the 270 candidates but 100%
    # of the LaTeX-gold rows -- a plain "first N" cap would be gsm8k-only).
    by_source: dict[str, list[dict]] = {}
    for candidate in candidates:
        by_source.setdefault(candidate["source"], []).append(candidate)
    candidates = []
    while any(by_source.values()):
        for name in SOURCE_FILES:
            if by_source.get(name):
                candidates.append(by_source[name].pop(0))

    if limit is not None:
        candidates = candidates[: limit // 2]
    funnel[LIMIT_STAGE] = len(candidates)

    rng = random.Random(seed)
    rng.shuffle(candidates)

    rows: list[dict] = []
    for candidate in candidates:
        rows.extend(_build_pair(candidate, seed=seed, base_index=len(rows)))
    for position, row in enumerate(rows):
        row["extra_info"]["index"] = position

    return rows, funnel


def _build_pair(candidate: dict, *, seed: int, base_index: int) -> list[dict]:
    """The unsolvable row (+ its solvable twin when the gold certifies)."""
    source = candidate["source"]
    row = candidate["row"]
    base = _base_task_id(source, candidate["index"], row)
    derivation = _derivation_text(source, row)
    difficulty = (
        f"level-{row.get('level')}"
        if source == "math" and row.get("level") is not None
        else DIFFICULTY_UNLABELLED
    )

    unsolvable = schema.make_row(
        data_source=DATA_SOURCE,
        question=row["insufficient_question"],
        ground_truth=schema.build_ground_truth(
            solvable=False,
            answer=None,
            correct_option_id=None,
            has_diagnosis_label=False,
            perturbation_type=PERTURBATION,
        ),
        template=TEMPLATE,
        branch=UNSOLVABLE_BRANCH,
        extra_info={
            "split": "train",
            "index": base_index,
            "task_id": f"{base}-{SIDE_UNSOLVABLE}",
            "seed": seed,
            "difficulty": difficulty,
            "perturbation_type": PERTURBATION,
            "perturbation_family": candidate["family"],
            "solvable": False,
            "paired_original_text": row["question"],
            "deleted_condition_text": candidate["diff"]["deleted"],
            "perturbed_entity_text": derivation,
            "error_type": ERROR_TYPE,
        },
    )
    out = [unsolvable]

    if source == "gsm8k":
        gold, _tier = _certify_gsm8k(row.get("answer") or "")
    else:
        gold, _tier = _certify_math(row)
    if gold is None:
        return out

    solvable = schema.make_row(
        data_source=DATA_SOURCE,
        question=row["question"],
        ground_truth=schema.build_ground_truth(
            solvable=True,
            answer=gold,
            correct_option_id=None,
            has_diagnosis_label=False,
            perturbation_type=None,
        ),
        template=TEMPLATE,
        branch=SOLVABLE_BRANCH,
        extra_info={
            "split": "train",
            "index": base_index + 1,
            "task_id": f"{base}-{SIDE_SOLVABLE}",
            "seed": seed,
            "difficulty": difficulty,
            "solvable": True,
            "paired_original_text": row["question"],
            "error_type": "",  # the solvable twin is a well-posed problem
            "perturbed_entity_text": derivation,
        },
    )
    out.append(solvable)
    return out


def _summarise(rows: list[dict]) -> dict:
    """Per-branch / per-template / per-solvable / per-source row counts."""
    summary = {
        "by_branch": {},
        "by_template": {},
        "by_solvable": {},
        "by_source_file": {},
    }
    for row in rows:
        info = row["extra_info"]
        gt = json.loads(row["reward_model"]["ground_truth"])
        for key, value in (
            ("by_branch", info["branch"]),
            ("by_template", info["template"]),
            ("by_solvable", str(gt["solvable"])),
        ):
            summary[key][value] = summary[key].get(value, 0) + 1
        source_file = info["task_id"].split("-")[1]
        summary["by_source_file"][source_file] = (
            summary["by_source_file"].get(source_file, 0) + 1
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR, help="MiP raw JSON directory")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap on rows written; pairs are whole, so the size is the largest even number <= limit",
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help="output parquet path")
    parser.add_argument("--seed", type=int, default=0, help="ordering seed")
    args = parser.parse_args(argv)

    rows, funnel = build_rows(args.raw_dir, limit=args.limit, seed=args.seed)
    if not rows:
        print("no rows survived the certificates; nothing written", file=sys.stderr)
        return 1
    schema.normalise_extra_info(rows)
    schema.validate_rows(rows)
    schema.write_rows_parquet(rows, args.out)

    print(f"funnel ({args.raw_dir}):")
    for name, count in funnel.items():
        print(f"  {name:28s} {count}")
    summary = _summarise(rows)
    print(f"rows written: {len(rows)} -> {args.out}")
    unsolvable = summary["by_branch"].get(UNSOLVABLE_BRANCH, 0)
    solvable = summary["by_branch"].get(SOLVABLE_BRANCH, 0)
    print(
        f"  solvable twins dropped (no gold certificate): {unsolvable - solvable}"
        "  -- the unsolvable row is kept, only the twin is lost"
    )
    for key in ("by_branch", "by_template", "by_solvable", "by_source_file"):
        print(f"{key}:")
        for name, count in sorted(summary[key].items()):
            print(f"  {name:24s} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
