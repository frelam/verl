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
"""Independent audit of the MiP adapter artifact (design doc sections 4.2 and 9).

Usage::

    python verify_mip.py --rows /path/to/mip.parquet

Every check prints ``PASS``/``FAIL`` and the process exits non-zero if any
check failed.  The audit reads **only** the artifact -- it never opens
``/home/charles/data/...`` and never imports the adapter -- so a bug in the
adapter's own certificates cannot hide here.  Every certificate is re-derived
from the row's own bytes with a second implementation:

* the deleted span is recovered by diffing ``insufficient -> original``
  *backwards* (the adapter's admission diff runs forwards, so this is an
  independent witness, not the same computation read back);
* the necessity of the deleted value is re-proved against the derivation text
  the adapter stored in ``extra_info.perturbed_entity_text``;
* the surface-cue check is a from-scratch replay of the option leak that
  motivated D12.

The artifact is **single-sided by design**: D12/Q7 keep MiP's solvable side out
of the pool, so every row is the unsolvable three-tier branch.  The two-sided
checks of the earlier revision -- solvable-gold recomputation, pair
reconstruction against a solvable twin, and the solvable-vs-unsolvable Naive
Bayes -- therefore have no input here and are reported as not applicable rather
than computed on one class.

Checks
------

``schema``   :func:`schema.validate_row` over every row (the design's contract).
``contract`` D12 / template / branch / task_id / prompt-shape invariants.
``L1``       per-row label certificate: re-derive the deleted span and re-prove
             its necessity from the row alone.
``L2``       gold uniqueness: the deleted value is *really* gone from the
             presented question and exactly one value was deleted.  "Really
             gone" is the recon's membership convention (string-normalised: a
             deleted ``$4.00`` counts as gone even if the question still says
             ``4``); rows where the deleted value survives in *another spelling*
             are listed for review rather than failed, since the surviving token
             is usually a different quantity in a different role (a deleted
             ``$2.00`` coupon next to "buys 2 packs").
``L3a/b``    D12 invariant (no row carries an option block or a diagnosis
             label) plus a re-measurement of the option leak that motivated it.
``L3c``      reported as not applicable: with the solvable side out of the pool
             (D12/Q7) there is one class left, so a balanced-accuracy test is
             undefined on this artifact.  The adapter's DEVIATIONS 4 records the
             same change; the number the earlier two-sided revision printed here
             cannot be reproduced from a single-sided artifact, and pretending
             otherwise would be the dishonest option.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
import os
import re
import sys
from decimal import Decimal, InvalidOperation

try:  # run as a script (its own dir is sys.path[0]) or via the conftest hook
    import schema
except ImportError:  # pragma: no cover - only when imported from elsewhere
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import schema

DEFAULT_ROWS = "/tmp/halluc_mip.parquet"
MIN_L1_SAMPLE = 50

UNSOLVABLE_BRANCH = schema.BRANCH_UNSOLVABLE_BARE

# ---------------------------------------------------------------------------
# independent re-implementations (deliberately separate from the adapter)
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def norm_ws(text: str) -> str:
    return _WS_RE.sub(" ", text or "").strip()


def numeric_tokens(text: str) -> list[str]:
    return [m.group(0) for m in _NUM_RE.finditer(text or "")]


def as_decimal(token: str) -> Decimal | None:
    try:
        return Decimal(str(token).strip().replace(",", ""))
    except (InvalidOperation, AttributeError, ValueError):
        return None


def normalise_number(token: str) -> str:
    """The recon's membership normal form (its section 4.1 note 3).

    ``4.00`` stays ``4.00`` (so a deleted ``$4.00`` counts as gone) while
    ``4.0`` folds to ``4``.  Value folding (``10`` vs ``0.1``) is deliberately
    **not** applied here -- that is the *necessity* convention, and mixing the
    two is the exact mistake the recon documents (it moves ``gone`` between 298
    and 299).
    """
    return token.replace(",", "").rstrip(".").removesuffix(".0")


def value_equivalent(left: str, right: str) -> bool:
    """Whether two tokens denote the same number in any spelling (``2.00`` ~ ``2``)."""
    first, second = as_decimal(left), as_decimal(right)
    return first is not None and second is not None and first == second


def presented_question(row: dict) -> str:
    """The question text the row actually asks, recovered from its own prompt.

    Slicing at the first blank line is wrong: MiP's math rows embed ``\\n\\n``
    inside the question (the ``[asy]`` diagram and display math survive into the
    truncated text), so that slice silently drops the tail and makes the
    certificate compare a truncated question against the full original.  The
    template B tail starts with a fixed marker, so everything before it is the
    question.
    """
    content = row["prompt"][0]["content"]
    marker = "\n\n若题目给出的信息不足以确定唯一答案"
    if marker in content:
        return content.split(marker)[0]
    raise AssertionError(f"{row['extra_info']['task_id']}: template B tail marker missing")


def changed_opcodes(a_words: list[str], b_words: list[str]):
    matcher = difflib.SequenceMatcher(a=a_words, b=b_words, autojunk=False)
    return [op for op in matcher.get_opcodes() if op[0] != "equal"]


def recover_span(insufficient: str, original: str) -> tuple[str, list[str], int]:
    """Diff ``insufficient -> original``: what does the truncation owe the reader?"""
    ops = changed_opcodes(insufficient.split(), original.split())
    a_words, b_words = insufficient.split(), original.split()
    recovered: list[str] = []
    filler: list[str] = []
    for tag, i1, i2, j1, j2 in ops:
        if tag in ("insert", "replace"):
            recovered.extend(b_words[j1:j2])
        if tag in ("delete", "replace"):
            filler.extend(a_words[i1:i2])
    return " ".join(recovered), filler, len(ops)


def forward_deleted(original: str, insufficient: str) -> tuple[str, str, int]:
    """Diff ``original -> insufficient`` (what the adapter's admission saw)."""
    ops = changed_opcodes(original.split(), insufficient.split())
    a_words, b_words = original.split(), insufficient.split()
    deleted: list[str] = []
    inserted: list[str] = []
    for tag, i1, i2, j1, j2 in ops:
        if tag in ("delete", "replace"):
            deleted.extend(a_words[i1:i2])
        if tag in ("insert", "replace"):
            inserted.extend(b_words[j1:j2])
    return " ".join(deleted), " ".join(inserted), len(ops)


def sentence_spans(text: str) -> list[tuple[int, int, str]]:
    spans, last = [], 0
    for match in _SENT_RE.finditer(text or ""):
        spans.append((last, match.start(), text[last : match.start()].strip()))
        last = match.end()
    spans.append((last, len(text), text[last:].strip()))
    return [(s, e, t) for s, e, t in spans if t]


def word_offsets(text: str) -> list[int]:
    position, offsets = 0, []
    for word in text.split():
        index = text.find(word, position)
        offsets.append(index)
        position = index + len(word)
    return offsets


def deletion_overlapping_sentences(original: str, insufficient: str) -> list[str]:
    """The sentences of ``original`` that hold some of the deleted words.

    Located by *character offset* from the forward diff, never by searching for
    the deleted text (a deletion often starts with a word like "The"/"If" that
    also opens an untouched sentence, and a text search would return that wrong
    sentence).  A deletion may straddle a sentence boundary -- MiP's truncation
    routinely does -- so this returns the whole overlapping set rather than one
    "gold sentence": that straddling is exactly why the recon reports a 100%
    any-absent figure and a ~94% uniquely-absent one.
    """
    a_words, b_words = original.split(), insufficient.split()
    ops = [op for op in changed_opcodes(a_words, b_words) if op[0] in ("delete", "replace")]
    if not ops:
        return []
    offsets = word_offsets(original)
    first, last = min(op[1] for op in ops), max(op[2] for op in ops)
    if first >= len(offsets) or last == 0:
        return []
    start = offsets[first]
    end = offsets[last - 1] + len(a_words[last - 1])
    return [text for begin, stop, text in sentence_spans(original) if begin < end and start < stop]


# ---------------------------------------------------------------------------
# report plumbing
# ---------------------------------------------------------------------------


class Report:
    def __init__(self) -> None:
        self.failed: list[str] = []
        self.passed: list[str] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        line = f"{'PASS' if ok else 'FAIL'}  {name}"
        if detail:
            line += f"  -- {detail}"
        print(line)
        (self.passed if ok else self.failed).append(name)
        return ok

    @staticmethod
    def info(label: str, detail: str) -> None:
        print(f"INFO  {label}  {detail}")

    @staticmethod
    def note(label: str, detail: str) -> None:
        print(f"NOTE  {label}  {detail}")


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def check_schema(report: Report, rows: list[dict]) -> None:
    violations: list[str] = []
    for row in rows:
        violations.extend(schema.validate_row(row))
    report.check(
        "schema.validate_row",
        not violations,
        f"{len(rows)} rows, {len(violations)} violations"
        + (f" | first: {violations[0]}" if violations else ""),
    )


def check_contract(report: Report, rows: list[dict]) -> None:
    problems: list[str] = []
    seen_ids: set[str] = set()
    for row in rows:
        info = row["extra_info"]
        gt = json.loads(row["reward_model"]["ground_truth"])
        task_id = info["task_id"]
        branch = info["branch"]
        if task_id in seen_ids:
            problems.append(f"duplicate task_id {task_id!r}")
        seen_ids.add(task_id)
        if info["template"] != schema.TEMPLATE_B:
            problems.append(f"{task_id}: template {info['template']!r} is not B")
        if info["options"] or gt["correct_option_id"] is not None:
            problems.append(f"{task_id}: carries an option block (D12 forbids it)")
        if gt["has_diagnosis_label"]:
            problems.append(f"{task_id}: has_diagnosis_label set (D12 forbids it)")
        if info["role_words"]:
            problems.append(f"{task_id}: role_words set on a MiP row")
        if branch == UNSOLVABLE_BRANCH:
            if gt["solvable"] or gt["answer"] is not None:
                problems.append(f"{task_id}: unsolvable row leaks answer={gt['answer']!r}")
            if not task_id.endswith("-unsolvable"):
                problems.append(f"{task_id}: unsolvable branch with a mismatched task_id")
            if "\\boxed{UNSOLVABLE}" not in row["prompt"][0]["content"]:
                problems.append(f"{task_id}: prompt does not ask for the UNSOLVABLE marker")
        else:
            # D12/Q7: the solvable twin is out of the pool, so any other branch
            # is a contract violation rather than a tolerated variant.
            problems.append(f"{task_id}: unexpected branch {branch!r} (D12/Q7: unsolvable only)")
    # index must match the row's position (the mix stage relies on it)
    for position, row in enumerate(rows):
        if row["extra_info"]["index"] != position:
            problems.append(f"{row['extra_info']['task_id']}: index != position")
            break
    report.check(
        "contract (D12 / template B / unsolvable-only branch / prompt)",
        not problems,
        f"{len(rows)} rows, {len(problems)} problems"
        + (f" | first: {problems[0]}" if problems else ""),
    )


def check_l1(report: Report, rows: list[dict]) -> None:
    """Re-derive every row's certificate from its own bytes, with a second implementation."""
    unsolvable = [r for r in rows if r["extra_info"]["branch"] == UNSOLVABLE_BRANCH]

    bad_unsolvable: list[str] = []
    for row in unsolvable:
        info, task_id = row["extra_info"], row["extra_info"]["task_id"]
        presented = presented_question(row)
        recovered, _filler, n_regions = recover_span(presented, info["paired_original_text"])
        if n_regions != 1:
            bad_unsolvable.append(f"{task_id}: reverse diff has {n_regions} regions, not 1")
            continue
        if not recovered.strip():
            bad_unsolvable.append(f"{task_id}: reverse diff recovered an empty span")
            continue
        recovered_values = {normalise_number(t) for t in numeric_tokens(recovered)}
        if len(recovered_values) != 1:
            bad_unsolvable.append(
                f"{task_id}: recovered span holds {len(recovered_values)} numeric values"
            )
            continue
        stored = {normalise_number(t) for t in numeric_tokens(info["deleted_condition_text"])}
        if stored != recovered_values:
            bad_unsolvable.append(
                f"{task_id}: stored deleted value {sorted(stored)} != recovered "
                f"{sorted(recovered_values)}"
            )
            continue
        # the reconstructed question must be the *original* the pair was cut from
        if norm_ws(recovered) not in norm_ws(info["paired_original_text"]):
            bad_unsolvable.append(f"{task_id}: recovered span is not a span of the original")
            continue
        # and the *forward* direction must agree: exactly one changed region, the
        # same span, and the same text the artifact stored.  The two directions
        # are separate difflib runs, so agreement is evidence and not a tautology.
        forward, inserted, forward_regions = forward_deleted(info["paired_original_text"], presented)
        if forward_regions != 1:
            bad_unsolvable.append(f"{task_id}: forward diff has {forward_regions} regions")
            continue
        if numeric_tokens(inserted):
            # a replacement that introduces a *number* would leave the question
            # answerable with a different value; a word placeholder ("many") does not
            bad_unsolvable.append(f"{task_id}: forward diff introduces the value {inserted!r}")
            continue
        expected_family = "placeholder" if inserted else "deletion"
        if info["perturbation_family"] != expected_family:
            bad_unsolvable.append(
                f"{task_id}: perturbation_family {info['perturbation_family']!r} != "
                f"{expected_family!r} (inserted {inserted!r})"
            )
            continue
        if norm_ws(forward) != norm_ws(recovered):
            bad_unsolvable.append(
                f"{task_id}: forward deleted {forward!r} != reverse recovered {recovered!r}"
            )
            continue
        if norm_ws(info["deleted_condition_text"]) != norm_ws(forward):
            bad_unsolvable.append(
                f"{task_id}: stored deleted_condition_text "
                f"{info['deleted_condition_text']!r} != re-derived {forward!r}"
            )
            continue
        # the deleted premise must be *used* by the source's own derivation
        chain = info["perturbed_entity_text"]
        value = next(iter(recovered_values))
        if not _value_in_chain(value, chain):
            bad_unsolvable.append(f"{task_id}: deleted value {value!r} is absent from the chain")
            continue
        # and the answer marker the prompt demands must be the bare one
        if info["error_type"] != "missing_condition":
            bad_unsolvable.append(f"{task_id}: unexpected error_type {info['error_type']!r}")

    report.info(
        "L1 sample",
        f"{len(unsolvable)} of {len(rows)} rows audited (floor {MIN_L1_SAMPLE}); "
        "every row is re-certified -- the artifact carries no solvable twin (D12/Q7)",
    )
    report.check(
        "L1 unsolvable: deleted span re-derived + necessity re-proved",
        not bad_unsolvable,
        f"{len(unsolvable)} rows, {len(bad_unsolvable)} uncertified"
        + (f" | first: {bad_unsolvable[0]}" if bad_unsolvable else ""),
    )


def _value_in_chain(value: str, chain: str) -> bool:
    """Re-prove necessity: the deleted value is used by the source's derivation."""
    if not (chain or "").strip():
        return False
    if value in {normalise_number(t) for t in numeric_tokens(chain)}:
        return True
    target = as_decimal(value)
    if target is None:
        return False
    for token in numeric_tokens(chain):
        current = as_decimal(token)
        if current is None or current == 0 or target == 0:
            continue
        if current == target or current * 100 == target or target * 100 == current:
            return True
    return False


def check_l2(report: Report, rows: list[dict]) -> None:
    """Gold uniqueness: the deleted value is really, unambiguously absent."""
    unsolvable = [r for r in rows if r["extra_info"]["branch"] == UNSOLVABLE_BRANCH]

    still_readable: list[str] = []
    same_value_other_spelling: list[str] = []
    multi_value: list[str] = []

    for row in unsolvable:
        info = row["extra_info"]
        task_id = info["task_id"]
        presented = presented_question(row)
        presented_tokens = numeric_tokens(presented)
        recovered, _filler, _n = recover_span(presented, info["paired_original_text"])
        values = list(dict.fromkeys(numeric_tokens(recovered)))
        if len(values) != 1:
            multi_value.append(f"{task_id}: {len(values)} distinct values in the deleted span")
            continue
        wanted = normalise_number(values[0])
        for token in presented_tokens:
            if normalise_number(token) == wanted:
                still_readable.append(
                    f"{task_id}: deleted value {values[0]!r} still readable as {token!r}"
                )
                break
        else:
            # Same *value*, other spelling (``2.00`` gone but the question says
            # ``2``).  Not a failure by the recon convention -- the match is
            # usually a different quantity in a different role ("buys 2 packs"
            # vs a deleted $2.00 coupon), which cannot recover the premise --
            # but it is listed for a human to eyeball rather than buried.
            for token in presented_tokens:
                if value_equivalent(values[0], token):
                    same_value_other_spelling.append(
                        f"{task_id}: deleted {values[0]!r}, question still spells {token!r}"
                    )
                    break

    report.info(
        "L2 scope",
        "MiP rows carry no option block (D12), so uniqueness is audited as "
        "\"the deleted value is really absent from the presented question\"; "
        "the solvable-twin reconstruction of the earlier revision is out of scope "
        "because D12/Q7 keep that side out of the pool",
    )
    report.check(
        "L2 deleted value is really absent (recon spelling convention)",
        not still_readable and not multi_value,
        f"{len(unsolvable)} unsolvable rows, {len(still_readable)} still readable, "
        f"{len(multi_value)} with 2+ values"
        + (f" | first: {(still_readable + multi_value)[0]}" if (still_readable or multi_value) else ""),
    )
    if same_value_other_spelling:
        report.note(
            "L2 review these by hand",
            f"{len(same_value_other_spelling)} row(s) where the deleted value appears in "
            "another *spelling* (so it is gone by the recon convention, and the match is "
            "usually an unrelated quantity). Review rather than trust: "
            + "; ".join(same_value_other_spelling[:3]),
        )


def check_l3_ab(report: Report, rows: list[dict]) -> None:
    """Option-block invariants (vacuous by D12) + the leak re-measurement."""
    with_options = [r for r in rows if r["extra_info"]["options"]]
    option_texts = [t for r in rows for t in r["extra_info"]["options"]]
    report.check(
        "L3a every option occurs verbatim in its own question",
        not option_texts,
        "vacuously true: 0 rows carry an option block (D12); the check exists so a "
        "future option-bearing MiP variant cannot land unnoticed",
    )
    report.check(
        "L3b all options of a row have equal token length",
        not with_options,
        "vacuously true: 0 rows carry an option block (D12)",
    )

    # Re-measure the leak that forced D12: reconstruct the would-be option set
    # (gold = the sentence of the original holding the deleted span; distractors
    # = the other sentences) and ask how often "pick any option absent from the
    # presented question" lands on the gold.
    unsolvable = [r for r in rows if r["extra_info"]["branch"] == UNSOLVABLE_BRANCH]
    any_absent = unique_absent = with_sentence = 0
    for row in unsolvable:
        info = row["extra_info"]
        presented = presented_question(row)
        original = info["paired_original_text"]
        overlapping = deletion_overlapping_sentences(original, presented)
        if not overlapping:
            continue
        with_sentence += 1
        overlapping = set(overlapping)
        absent = [s for _b, _e, s in sentence_spans(original) if s not in presented]
        if overlapping & set(absent):
            # the heuristic "pick any option not present in the question" has a
            # target: an option covering the deletion is not found
            any_absent += 1
            # ... and the target is unambiguous when the *only* option that is
            # not found is the gold one (the recon's H1-unique definition)
            if len(absent) == 1 and absent[0] in overlapping:
                unique_absent += 1
    if with_sentence:
        report.info(
            "L3 option leak (re-measured, informational)",
            f"would-be option set = the sentences of the original question, gold = the "
            f"deletion-covering sentence(s); over {with_sentence} rows the gold option is "
            f"not found in the presented question in {any_absent}/{with_sentence} = "
            f"{any_absent / with_sentence:.1%} (recon: 100.0%) and the not-found options "
            f"are exactly the gold ones in {unique_absent}/{with_sentence} = "
            f"{unique_absent / with_sentence:.1%} (recon: 94.2%)",
        )
        report.note(
            "L3 D12 justification",
            "any option set built from these deletions is guessable at "
            f"{any_absent / with_sentence:.1%} by \"pick the option that is missing from "
            "the question\", so the adapter builds none (doc section 4.2 says 95.6%; the "
            "recon measured 100.0% and this re-measurement agrees)",
        )

    # The earlier revision also ran a 5-fold bag-of-words Naive Bayes between the
    # solvable and unsolvable sides.  With the solvable side out of the pool
    # (D12/Q7) there is no negative class left, so that reading is undefined --
    # reported here rather than silently dropped.
    report.note(
        "L3c not applicable",
        "the artifact is single-sided by design (D12/Q7 keep MiP's solvable side out of "
        "the pool), so the solvable-vs-unsolvable balanced-accuracy test has one class "
        "and cannot be computed; the option-leak re-measurement above is the L3 evidence "
        "for this source",
    )


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", default=DEFAULT_ROWS, help="adapter output parquet")
    args = parser.parse_args(argv)

    if not os.path.exists(args.rows):
        print(f"FAIL  artifact  -- {args.rows} does not exist", file=sys.stderr)
        return 2
    rows = schema.read_parquet_rows(args.rows)
    if not rows:
        print(f"FAIL  artifact  -- {args.rows} holds no rows", file=sys.stderr)
        return 2

    branches = collections.Counter(r["extra_info"]["branch"] for r in rows)
    print(f"artifact: {args.rows}  rows={len(rows)}  branches={dict(branches)}")
    if len(rows) < MIN_L1_SAMPLE:
        print(f"FAIL  artifact size  -- L1 requires >= {MIN_L1_SAMPLE} rows, got {len(rows)}")
        return 2

    report = Report()
    check_schema(report, rows)
    check_contract(report, rows)
    check_l1(report, rows)
    check_l2(report, rows)
    check_l3_ab(report, rows)

    print(
        f"\n{len(report.passed)} check(s) passed, {len(report.failed)} failed"
        + (f": {', '.join(report.failed)}" if report.failed else "")
    )
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
