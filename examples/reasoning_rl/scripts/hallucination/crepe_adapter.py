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
"""CREPE adapter -- the judgment-only source (design doc section 4.6; table B rows 4 and 8).

The source asks one question: *is this question answerable at all?*  It carries
no pointer gold, so it cannot enter the four-tier diagnosis branch:
``presuppositions`` is a *paraphrase*, not a span -- measured 14 of 1,271
single-label train fragments (1.10%) occur verbatim in ``question``, so "point at
the false premise in the prompt" is impossible (design doc section 4.6, pitfall
2).

Output: solvable rows on branch ``solvable_judge`` (gold ``\\boxed{SOLVABLE}``,
``judgment_only=true``) and unsolvable rows on branch ``unsolvable_bare`` (gold
``\\boxed{UNSOLVABLE}``, three-tier).  Neither side carries an option block: the
three-tier branch is option-less by D12/D24, and the judgment branch has no
option semantics to expose, so template ``B_judge`` is the only admissible
template for both.  Design doc section 4.8 table B assigns this source
**CREPE-normal 250** (row 4) and **CREPE-FP 400** (row 8) = 650 rows;
:data:`QUOTAS` is that allocation and nothing else (D25 removed the second
judgment-only source that an earlier revision shared this file with).

Why the labels are trustworthy (the certificates this adapter enforces)
----------------------------------------------------------------------

Every row is dropped unless a *second, independent* source field agrees with the
label -- fail closed, counted in the funnel:

======================  ==========================  ==================================
source / gold           label evidence              independent corroboration
======================  ==========================  ==================================
CREPE solvable          `labels == ['normal']`      `presuppositions == []`
CREPE unsolvable        `labels == ['false       `presuppositions` non-empty (the
                        presupposition']`            source states the false premise)
======================  ==========================  ==================================

Measured on the raw bundle: the corroboration holds on 100% of the single-label
rows (8,446 / 8,446), so the certificate stage itself drops nothing today -- it
is a guard, and it is what lets ``verify_crepe.py`` re-prove the gold from the
raw bytes instead of trusting this file.

DEVIATIONS FROM THE DESIGN DOC
------------------------------

Each item below is a place where the design doc's number or allocation is
contradicted by the recon report / by a fresh measurement.  The measured number
is quoted.

1. **Rows use template ``B_judge``, not template ``B``** (design doc section 5.2
   puts the judgment branch on template B plus one added sentence).  Template B
   asks for the *answer*; the frozen ``schema.py`` spells the judgment variant
   out as ``B_judge`` because B's "否则…输出你的最终答案" line contradicts the
   verdict-only instruction.  **Both labels therefore use ``B_judge``**: using B
   for the unsolvable side and B_judge for the solvable side would make "which
   verdict wording the prompt offers" a 100% label shortcut, which is exactly
   what the D18 isomorphism constraint forbids.  The whole file is in the
   option-less template family, which keeps that family's ``solvable`` split
   non-degenerate (design doc sections 5.2 and 9).
2. **CREPE train false-presupposition count is 907 here, not the doc's 927.**
   The doc's 927 folds in the 20 train rows whose `labels` is
   ``['false presupposition', 'normal']`` (annotator disagreement, measured
   20/3,462 in train and 0 in validation/test).  A dual-labelled row has two
   admissible golds, so L2 forbids it: dropped, not assigned to either side.
   Re-derived from raw: the space-form string ``'false presupposition'`` matches
   927 train rows (the doc's number, verified) and the underscore form
   ``'false_presupposition'`` matches 0.
3. **The L3 Naive-Bayes reading exceeds the 0.60 reference gate on these rows**
   (measured by ``verify_crepe.py``, which prints it verbatim).  Out-of-fold
   balanced accuracy with the vocabulary restricted to tokens with train support
   >= 5: **0.6075** on the 650-row build this adapter produces (the pre-D25
   ``--limit 300`` artifact read 0.6267 CREPE-only; the seed and the quota
   composition both move the figure, so re-run the audit rather than quoting
   this one).  Design doc table A gives CREPE no L3 figure at all, so this is new
   information rather than a contradiction of a number; it is recorded here
   because it is the single most important caveat on these rows.  Diagnosis: the
   signal is not in an option block (there is none) -- it is question *style*.
   The doc's section 9 hard gate ("随机基线 + 5pt", fail-closed) is scoped to
   sources whose enabled **option set** is the only signal; CREPE carries no
   option block, so the reading is an explicit reported finding, not a gate that
   decides admission.  ``verify_crepe.py`` labels it as such and does not
   silently pass or silently fail the run on it.  Escalate; do not treat it as
   clean.
4. **CREPE's native three-way split is preserved** in ``extra_info.split``
   (train 3,462 / validation 2,000 / test 3,004) instead of collapsing the source
   to ``train``.  The table B quota of a (source, branch) group is filled across
   that source's native splits in proportion, so the group total is exactly the
   quota and every row keeps its own split.  The design doc does not say how to
   treat these splits.

Other measured facts worth having in the file
---------------------------------------------

* ``difficulty`` carries the source's *own* annotation slug rather than an
  invented tier: ``normal`` / ``false_presupposition`` (its label vocabulary).
  It is a bucketing key for design doc section 10 monitoring, not a graded
  difficulty.
* ``perturbation_type`` is ``contradictory_condition`` on the unsolvable rows:
  the premise contradicts the true state of the world, which is the closest of
  the schema's seven values to a false premise; the finer D18 defect class is
  carried in ``error_type = false_premise_unpointable`` (table B row 8's
  假前提（不可指认）bucket).
* CREPE keys are its own ``id`` column (``dup_ids == 0`` per split).
* Determinism: the only randomness is the seeded subsample that fills each
  quota, drawn from its own :class:`random.Random` stream, so
  ``(raw_dir, limit, seed)`` reproduces the rows byte for byte.

Usage::

    python crepe_adapter.py --limit 300 --out /tmp/halluc_crepe.parquet
"""

from __future__ import annotations

import argparse
import os
import random
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass

import schema

# ---------------------------------------------------------------------------
# source constants (all measured on the raw bundle, none guessed)
# ---------------------------------------------------------------------------

RAW_DIR_DEFAULT = "/home/charles/data/reasoning_rl/halluc/raw/crepe"
# Named DEFAULT_OUT like every other adapter, and named after *this* source.
DEFAULT_OUT = os.path.expanduser("~/data/reasoning_rl/halluc/built/crepe.parquet")

CREPE_SPLITS = ("train", "validation", "test")
# The space, not the underscore: the upstream README says 'false_presupposition'
# and that spelling matches 0 rows (design doc section 4.6, pitfall 1).
CREPE_NORMAL = "normal"
CREPE_FALSE_PRESUPPOSITION = "false presupposition"
CREPE_LABELS = (CREPE_NORMAL, CREPE_FALSE_PRESUPPOSITION)
CREPE_REQUIRED_COLUMNS = ("id", "question", "labels", "presuppositions")

# Design doc section 4.8 table B: row 4 takes CREPE-normal 250 (judgment-only,
# template B_judge) and row 8 takes CREPE-false-presupposition 400 (three-tier
# bare, template B_judge with no options).
QUOTAS: dict[tuple[str, str], int] = {
    (schema.SOURCE_CREPE, schema.BRANCH_SOLVABLE_JUDGE): 250,
    (schema.SOURCE_CREPE, schema.BRANCH_UNSOLVABLE_BARE): 400,
}

# Fixed order: it decides the row order of the artifact, the quota RNG streams,
# and therefore the exact rows an --limit keeps.
GROUP_ORDER: tuple[tuple[str, str], ...] = (
    (schema.SOURCE_CREPE, schema.BRANCH_SOLVABLE_JUDGE),
    (schema.SOURCE_CREPE, schema.BRANCH_UNSOLVABLE_BARE),
)

# All rows are option-less verdict rows, so one template serves both labels.
TEMPLATE = schema.TEMPLATE_B_JUDGE

# The D18 defect class this source fills (design doc section 4.8 table B row 8,
# 假前提（不可指认）: a false premise that cannot be pointed at in the prompt).
ERROR_TYPE = "false_premise_unpointable"
# A false premise contradicts the true state of the world; the finer class goes
# in error_type (see the module docstring).
UNSOLVABLE_PERTURBATION = "contradictory_condition"

SOURCE_NAMES = (schema.SOURCE_CREPE,)
# Single-source file now that D25 removed the other judgment-only source: the
# flag stays for symmetry with the other adapters and so a bundle subset can
# still be selected explicitly.
SOURCE_CHOICES = ("crepe",)


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One raw record that may become a row, with its certificate already checked.

    ``label_ok`` and ``certificate_ok`` are computed by the loader rather than
    inside :func:`build_rows` so that each filter stage in the funnel maps to
    exactly one predicate -- which is what makes the funnel auditable.
    """

    source: str
    split: str
    key: str
    question: str
    solvable: bool
    label_ok: bool
    certificate_ok: bool
    certificate: str
    difficulty: str

    @property
    def task_id(self) -> str:
        """Stable hard-replay dedup key, derived from the source's own identity."""
        return f"crepe:{self.split}:{self.key}"

    @property
    def branch(self) -> str:
        if self.solvable:
            return schema.BRANCH_SOLVABLE_JUDGE
        return schema.BRANCH_UNSOLVABLE_BARE

    @property
    def perturbation_type(self) -> str | None:
        return None if self.solvable else UNSOLVABLE_PERTURBATION

    @property
    def error_type(self) -> str:
        return "" if self.solvable else ERROR_TYPE

    def order_key(self) -> tuple:
        return (self.source, self.split, self.key)


def normalise_question(question: str) -> str:
    """Case- and whitespace-insensitive dedup key (questions are kept verbatim)."""
    return " ".join(question.casefold().split())


def _crepe_candidate(split: str, record: dict) -> Candidate:
    raw_labels = record.get("labels") or []
    labels = [str(label) for label in raw_labels]
    presuppositions = [str(text) for text in (record.get("presuppositions") or [])]
    question = record.get("question") or ""
    key = str(record.get("id") or "")

    # L2 gold uniqueness: a dual-labelled row has two admissible golds.
    label_ok = len(labels) == 1 and labels[0] in CREPE_LABELS and bool(key)
    solvable = label_ok and labels[0] == CREPE_NORMAL

    if not label_ok:
        certificate_ok = False
        certificate = f"labels must be exactly one of {CREPE_LABELS} and id must be set"
    elif solvable:
        # An independent column: the source lists no false presupposition here.
        certificate_ok = not presuppositions
        certificate = "source recorded no false presupposition for this question"
    else:
        certificate_ok = bool(presuppositions)
        certificate = "source states the false presupposition this question rests on"

    difficulty = labels[0].replace(" ", "_") if label_ok else ""
    return Candidate(
        source=schema.SOURCE_CREPE,
        split=split,
        key=key,
        question=question,
        solvable=solvable,
        label_ok=label_ok,
        certificate_ok=certificate_ok,
        certificate=certificate,
        difficulty=difficulty,
    )


def load_crepe(raw_dir: str) -> list[Candidate]:
    """Every CREPE row of the three splits as a :class:`Candidate`."""
    import pyarrow.parquet as pq

    candidates: list[Candidate] = []
    for split in CREPE_SPLITS:
        path = os.path.join(raw_dir, f"crepe_{split}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing CREPE split {split!r}: {path}")
        # All columns are read on purpose: ``pq.read_table(columns=...)`` raises
        # its own ``ArrowInvalid`` for a missing column, which would make the
        # explicit check below unreachable.
        table = pq.read_table(path)
        missing = [column for column in CREPE_REQUIRED_COLUMNS if column not in table.column_names]
        if missing:
            raise ValueError(f"{path} is missing required columns {missing}")
        candidates.extend(_crepe_candidate(split, record) for record in table.to_pylist())
    return candidates


# ---------------------------------------------------------------------------
# row assembly
# ---------------------------------------------------------------------------


def make_candidate_row(candidate: Candidate, index: int, seed: int) -> dict:
    """Turn one candidate into a verl parquet row (schema contract, section 3).

    An unsolvable row carries ``answer=None`` and no diagnosis label; a solvable
    judgment row carries ``judgment_only=True`` because its gold is the
    ``\\boxed{SOLVABLE}`` marker, not an answer string.  Neither side gets an
    options block, so template ``B_judge`` is the only admissible template.
    """
    ground_truth = schema.build_ground_truth(
        solvable=candidate.solvable,
        answer=None,
        correct_option_id=None,
        has_diagnosis_label=False,
        perturbation_type=candidate.perturbation_type,
        judgment_only=candidate.solvable,
    )
    extra_info = {
        "split": candidate.split,
        "index": index,
        "task_id": candidate.task_id,
        "seed": seed,
        "difficulty": candidate.difficulty,
        "branch": candidate.branch,
        "template": TEMPLATE,
        "perturbation_type": candidate.perturbation_type or "",
        "judgment_only": candidate.solvable,
        "has_diagnosis_label": False,
        "solvable": candidate.solvable,
        "error_type": candidate.error_type,
    }
    return schema.make_row(
        data_source=candidate.source,
        question=candidate.question,
        ground_truth=ground_truth,
        template=TEMPLATE,
        branch=candidate.branch,
        extra_info=extra_info,
    )


def _group_rng(seed: int, group_index: int) -> random.Random:
    """A private RNG stream per quota group, so groups cannot shift each other."""
    return random.Random(seed * 1_000_003 + group_index)


def _interleave(groups: list[list[Candidate]]) -> list[Candidate]:
    """Round-robin over the groups: one row per group per round."""
    out: list[Candidate] = []
    round_index = 0
    longest = max((len(group) for group in groups), default=0)
    while round_index < longest:
        for group in groups:
            if round_index < len(group):
                out.append(group[round_index])
        round_index += 1
    return out


def _round_robin_by_split(members: list[Candidate]) -> list[Candidate]:
    """Order a quota group round-robin across the source's native splits.

    Members would otherwise be ordered by ``(source, split, key)``, which puts
    every ``validation`` row behind every ``test`` row -- so a small ``--limit``
    (which truncates the final order) would silently drop whole native splits.
    Round-robin keeps every split represented at any limit, deterministically.
    """
    by_split: dict[str, list[Candidate]] = {}
    for member in sorted(members, key=Candidate.order_key):
        by_split.setdefault(member.split, []).append(member)
    return _interleave([by_split[split] for split in sorted(by_split)])


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def build_rows(
    raw_dir: str,
    limit: int | None = None,
    seed: int = 0,
    sources: tuple[str, ...] = SOURCE_NAMES,
) -> tuple[list[dict], OrderedDict[str, int]]:
    """Build the rows and the filter funnel (design doc sections 4.6 and 4.8).

    Args:
        raw_dir: directory holding ``crepe_{train,validation,test}.parquet``.
        limit: cap on the number of emitted rows, applied last; ``None`` emits
            the full quota (650 rows).
        seed: seed of the quota subsampling; ``(raw_dir, limit, seed)`` is
            reproducible byte for byte.
        sources: which sources to build (``SOURCE_NAMES`` by default).

    Returns:
        ``(rows, funnel)``.  ``funnel`` maps each filter stage to the number of
        rows still alive *after* that stage, starting with ``raw_rows``.

    Raises:
        ValueError: on an unknown source or a negative limit.
        FileNotFoundError: when the raw bundle is incomplete.
    """
    unknown_sources = [s for s in sources if s not in SOURCE_NAMES]
    if unknown_sources:
        raise ValueError(f"unknown source(s) {unknown_sources}; expected {SOURCE_NAMES}")
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be non-negative, got {limit}")

    funnel: OrderedDict[str, int] = OrderedDict()

    candidates: list[Candidate] = []
    if schema.SOURCE_CREPE in sources:
        candidates.extend(load_crepe(raw_dir))
    # A fixed order before any sampling is what makes the outcome reproducible.
    candidates.sort(key=Candidate.order_key)
    funnel["raw_rows"] = len(candidates)

    stage = [c for c in candidates if c.question.strip()]
    funnel["after_question_text"] = len(stage)

    stage = [c for c in stage if c.label_ok]
    funnel["after_label_uniqueness"] = len(stage)

    stage = [c for c in stage if c.certificate_ok]
    funnel["after_certificate"] = len(stage)

    # L2 across the pool: a question whose gold is not unique (it appears on both
    # sides) is admissible under two labels, so every one of its rows goes.
    labels_by_question: dict[str, set[bool]] = defaultdict(set)
    for candidate in stage:
        labels_by_question[normalise_question(candidate.question)].add(candidate.solvable)
    conflicted = {q for q, labels in labels_by_question.items() if len(labels) > 1}
    stage = [c for c in stage if normalise_question(c.question) not in conflicted]
    funnel["after_cross_label_conflict"] = len(stage)

    seen: set[str] = set()
    deduped: list[Candidate] = []
    for candidate in stage:
        key = normalise_question(candidate.question)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    stage = deduped
    funnel["after_dedup"] = len(stage)

    grouped: dict[tuple[str, str], list[Candidate]] = {}
    for candidate in stage:
        grouped.setdefault((candidate.source, candidate.branch), []).append(candidate)

    filled: list[list[Candidate]] = []
    for group_index, group_key in enumerate(GROUP_ORDER):
        if group_key not in QUOTAS:
            raise ValueError(f"no quota declared for group {group_key!r}")
        members = grouped.get(group_key, [])
        quota = QUOTAS[group_key]
        if len(members) > quota:
            members = _group_rng(seed, group_index).sample(members, quota)
        members = _round_robin_by_split(members)
        if group_key[0] in sources:
            filled.append(members)
    stage = _interleave(filled)
    funnel["after_quota"] = len(stage)

    if limit is not None:
        stage = stage[:limit]
    funnel["after_limit"] = len(stage)

    rows = [make_candidate_row(candidate, index, seed) for index, candidate in enumerate(stage)]
    return rows, funnel


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def _print_funnel(funnel: OrderedDict[str, int]) -> None:
    print("funnel (rows remaining after each stage):")
    previous: int | None = None
    for stage, count in funnel.items():
        delta = "" if previous is None else f"   ({count - previous:+d})"
        print(f"  {stage:<28} {count:>7}{delta}")
        previous = count


def _print_breakdown(rows: list[dict]) -> None:
    if not rows:
        print("no rows emitted")
        return
    for name in ("branch", "template", "solvable", "source", "difficulty", "split"):
        counts = Counter(str(row["extra_info"].get(name)) for row in rows)
        summary = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        print(f"rows by {name}: {summary}")
    counts = Counter(row["data_source"] for row in rows)
    summary = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
    print(f"rows by data_source: {summary}")
    optioned = sum(1 for row in rows if row["extra_info"]["options"])
    diagnosis = sum(1 for row in rows if row["extra_info"]["has_diagnosis_label"])
    print(f"rows with an options block: {optioned}; with a diagnosis label: {diagnosis}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", default=RAW_DIR_DEFAULT, help="raw CREPE bundle")
    parser.add_argument("--limit", type=int, default=None, help="cap on emitted rows")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output parquet path")
    parser.add_argument("--seed", type=int, default=0, help="quota subsampling seed")
    parser.add_argument(
        "--sources",
        choices=SOURCE_CHOICES,
        default=SOURCE_CHOICES[0],
        help="which source to build; this adapter builds CREPE only",
    )
    args = parser.parse_args()

    # Single-source file: argparse's ``choices`` is what rejects anything but
    # CREPE, and the flag itself is kept for symmetry with the other adapters.
    sources = SOURCE_NAMES

    rows, funnel = build_rows(args.raw_dir, limit=args.limit, seed=args.seed, sources=sources)
    if not rows:
        raise SystemExit("no rows survived the funnel; refusing to write an empty parquet")
    schema.normalise_extra_info(rows)
    schema.validate_rows(rows)
    schema.write_rows_parquet(rows, args.out)

    print(f"source bundle : {args.raw_dir}")
    print(f"sources       : {', '.join(sources)}")
    print(f"seed          : {args.seed}    limit: {args.limit}")
    print(f"wrote         : {args.out} ({len(rows)} rows)")
    print()
    _print_funnel(funnel)
    print()
    _print_breakdown(rows)
    print()
    quotas = ", ".join(f"{src}/{branch}={quota}" for (src, branch), quota in QUOTAS.items())
    print(f"D18 table B quotas: {quotas}")


if __name__ == "__main__":
    main()
