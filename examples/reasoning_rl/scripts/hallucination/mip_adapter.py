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
"""MiP adapter -- the three-tier *bare* unsolvable source (design doc section 4.2 / D12).

Source: ``github.com/tianyi-lab/MiP-Overthinking`` ``data/{gsm8k,svamp,math,formula}.json``
(984 rows; **not** a HF dataset).  Each paired row carries a solvable original
(``question`` + the source's own ``answer``/``solution``) and its truncated twin
(``insufficient_question``), so one source row supplies the deletion the
answerability signal is built on.

What this adapter emits
-----------------------

Exactly one branch, under ``data_source=halluc_math_mip``:

========================  =========================================  =========
branch                    gold                                       template
========================  =========================================  =========
``unsolvable_bare``       ``\\boxed{UNSOLVABLE}`` (``answer=None``)   B
========================  =========================================  =========

The solvable twin is **not emitted**: D12 and Q7 keep MiP's solvable side out of
the pool (it shares its base problems with the stage-1 math pool -- GSM8K/MATH --
it is small, and the cross-pool de-duplication cost outweighs the benefit).  The
artifact is therefore single-sided by construction: every row is template B, no
option block, no ``judgment_only``, and the row count *is* the strict pool size
(design doc section 4.8 table B row 8, quota 276; see DEVIATIONS 1 for the
measured 270).

The one pairing constraint MiP still participates in is design doc section
4.7's 成对性 rule -- the synthesised-distractor rows (D17) must share a base
distribution with the three-tier sources, or "which base pool is this?" becomes a
shortcut -- and it holds by construction: MiP's bases are GSM8K/SVAMP/MATH
questions, the same distribution the 100 main-pool math distractor rows are drawn
from.

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
   Decimals with percent folding.

Two claims in the recon are conventions, not measurements, and this adapter
takes the fail-closed side of both: rows with **no** derivation chain at all (2
gsm8k rows) and placeholders whose deleted value is *not* used in the chain (3
rows) are dropped rather than credited.

``verify_mip.py`` re-checks all seven of these from the written artifact alone,
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
299 -- the recon needed three attempts to get this right; see the recon's
section 4.1 note 3.

DEVIATIONS FROM THE DESIGN DOC (recon report wins)
--------------------------------------------------

1. **The strict pool is 270, not 276** (doc section 4.2: "严口径 276"; the doc's
   Q6 records the same 276 -> 299 option).  Measured with the doc's own stage
   definitions: 260 pure deletions pass the necessity check, 2 more have *no*
   ``<<expr=val>>`` chain at all (doc: "另 23 待复核、14 占位词型无需链校验";
   recon: "260 pass / 24 fail / 2 empty-chain"), and of the 14 placeholder rows
   10 pass necessity, 3 fail it and 1 is still visible in the truncated
   question.  260 + 10 = **270**.  The doc's 276 additionally (a) credits the 2
   chainless rows, (b) exempts all 14 placeholders from the necessity check and
   (c) admits a placeholder whose deleted value is still visible.  Recon section
   6 hazard 2 says an adapter "must handle a missing chain rather than assume the
   annotation exists"; this adapter drops instead of crediting, per the
   fail-closed rule -- i.e. **the code implements the strict口径 and yields 270**.
   Q6's default is not to backfill the 23 unflagged rows, so 270 is the delivered
   pool; the 276 in the design doc is the wide口径 upper bound that this adapter
   deliberately does not reach.
2. **svamp (300) and formula (50) contribute 0 rows**, against doc section 4.2's
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
4. **The artifact no longer carries a solvable twin.**  An earlier revision
   emitted both sides of every pair and left the solvable half to the mix; D12
   and Q7 settle the question the other way (correctly), so the twin, its gold
   certificates and its ``--limit // 2`` pair accounting are gone.  One
   consequence: the two-sided bag-of-words reading the earlier revision printed
   in ``verify_mip.py``'s L3c (the "support >= 5 vocabulary" Naive Bayes) is no
   longer defined on this artifact -- there is no negative class left -- and that
   check is now reported as not applicable rather than computed on one side.

Field choices that the schema leaves free (audit-only fields)
-------------------------------------------------------------

``paired_original_text`` = the solvable original question; ``deleted_condition_text``
= the deleted span (word-diff text); ``perturbed_entity_text`` = **the source's
own derivation text** (the gsm8k ``answer`` field / the math ``solution``),
which is what ``verify_mip.py`` re-reads to re-prove that the deleted value was
necessary.  ``canonical_solution`` and ``role_words`` stay empty (K&K-only),
``distractor_labels`` keeps its fixed three-key shape, ``split`` is always
``"train"`` (MiP ships no native splits).
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
# deletion (see DEVIATIONS 2), but all four are counted so the funnel starts at
# the raw row count of 984.
SOURCE_FILES = ("gsm8k", "svamp", "math", "formula")
PAIRED_FILES = ("gsm8k", "math")

DATA_SOURCE = schema.SOURCE_MIP
TEMPLATE = schema.TEMPLATE_B
UNSOLVABLE_BRANCH = schema.BRANCH_UNSOLVABLE_BARE

PERTURBATION = "missing_condition"
# Design doc section 4.8 defect class: "缺一条必要条件（题面不可见）" -> the
# three-tier bare bucket (table B row 8).
ERROR_TYPE = "missing_condition"
FAMILY_DELETION = "deletion"  # the premise was removed outright
FAMILY_PLACEHOLDER = "placeholder"  # a numeric premise became "many"/"some"/...

TASK_PREFIX = "mip"
SIDE_UNSOLVABLE = "unsolvable"

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
    ``-unsolvable`` suffix is kept from the revision that also emitted the
    solvable twin, so a hard-replay dedup key stays stable across the rebuild
    even though the twin is gone (D12/Q7); it also keeps the branch explicit in
    the id that ``verify_mip.py`` audits.
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
        limit: cap on the number of **rows** written (one row per certified
            deletion; there is no twin to keep whole).  ``None`` writes
            everything the certificates admit -- the strict pool, measured at
            270 rows on the current raw bundle (DEVIATIONS 1).
        seed: seeds ``random.Random`` for the output ordering only -- the row
            *set* is fixed by the certificates, so the same seed always yields
            byte-identical rows.

    Returns:
        ``(rows, funnel)``.  ``funnel`` is an ordered mapping of stage name ->
        rows remaining after that stage, starting at the raw row count of 984.
    """
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")

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
                # the recon's 209 / 55 / 37 buckets (recon section 4.1 note 1:
                # do NOT bucket by numeric value alone, that inflates the pool
                # by 12).
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
                # value the source never used (False): DEVIATIONS 1.
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
        candidates = candidates[:limit]
    funnel[LIMIT_STAGE] = len(candidates)

    rng = random.Random(seed)
    rng.shuffle(candidates)

    rows: list[dict] = []
    for candidate in candidates:
        rows.append(_build_row(candidate, seed=seed, index=len(rows)))

    return rows, funnel


def _build_row(candidate: dict, *, seed: int, index: int) -> dict:
    """The unsolvable three-tier row for one certified deletion."""
    source = candidate["source"]
    row = candidate["row"]
    base = _base_task_id(source, candidate["index"], row)
    derivation = _derivation_text(source, row)
    difficulty = (
        f"level-{row.get('level')}"
        if source == "math" and row.get("level") is not None
        else DIFFICULTY_UNLABELLED
    )

    return schema.make_row(
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
            "index": index,
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
        help="cap on rows written; one row per certified deletion, no pairing",
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
    print(
        "  every row is the unsolvable three-tier branch (D12/Q7: no solvable twin,"
        " no options)"
    )
    for key in ("by_branch", "by_template", "by_solvable", "by_source_file"):
        print(f"{key}:")
        for name, count in sorted(summary[key].items()):
            print(f"  {name:24s} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
