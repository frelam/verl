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
"""Independent audit of the MiP adapter artifact (design doc section 4.9).

Usage::

    python verify_mip.py --rows /path/to/mip.parquet

Every check prints ``PASS``/``FAIL`` and the process exits non-zero if any
check failed.  The audit reads **only** the artifact -- it never opens
``/home/charles/data/...`` and never imports the adapter -- so a bug in the
adapter's own certificates cannot hide here.  Every gold is re-derived from the
row's own bytes with a second implementation:

* the deleted span is recovered by diffing ``insufficient -> original``
  *backwards* (the adapter's admission diff runs forwards, so this is an
  independent witness, not the same computation read back);
* the gsm8k gold is recomputed by evaluating the arithmetic the source wrote;
* the math gold is re-extracted as the last ``\\boxed{}`` the solution reaches;
* the surface-cue check is a from-scratch numpy multinomial Naive Bayes.

Checks
------

``schema``   :func:`schema.validate_row` over every row (the design's contract).
``contract`` D12 / template / branch / task_id / prompt-shape invariants.
``L1``       per-row label certificate: re-derive the gold from the row alone.
``L2``       gold uniqueness: the deleted value is *really* gone from the
             presented question, exactly one value was deleted, the pair's
             shared premise still matches, and the solvable gold is the single
             value of the derivation's final step.  "Really gone" is the
             recon's membership convention (string-normalised: a deleted
             ``$4.00`` counts as gone even if the question still says ``4``);
             rows where the deleted value survives in *another spelling* are
             listed for review rather than failed, since the surviving token is
             usually a different quantity in a different role (a deleted
             ``$2.00`` coupon next to "buys 2 packs").
``L3a/b``    D12 invariant (no row carries an option block or a diagnosis
             label) plus a re-measurement of the option leak that motivated it.
``L3c``      bag-of-words multinomial Naive Bayes, 5-fold out-of-fold balanced
             accuracy between solvable and unsolvable, vocabulary restricted to
             train support >= 5.  Raw number reported verbatim; the sign-flipped
             reading and a length-only baseline are printed next to it because
             the recon documents this estimator flipping sign out of fold.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
import math
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

DEFAULT_ROWS = "/tmp/halluc_mip.parquet"
MIN_L1_SAMPLE = 50
NB_FOLDS = 5
NB_MIN_SUPPORT = 5
NB_ALPHA = 1.0
NB_BA_LIMIT = 0.60

SOLVABLE_BRANCH = schema.BRANCH_SOLVABLE_NUMERIC
UNSOLVABLE_BRANCH = schema.BRANCH_UNSOLVABLE_BARE

# ---------------------------------------------------------------------------
# independent re-implementations (deliberately separate from the adapter)
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_ANN_RE = re.compile(r"<<([^>]*)>>")
_BOXED_RE = re.compile(r"\\boxed\s*\{")
_TRAILER_RE = re.compile(r"=\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)")
_TOKEN_RE = re.compile(r"[a-z0-9']+")
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


def last_boxed(text: str) -> str | None:
    result: str | None = None
    for match in _BOXED_RE.finditer(text or ""):
        start, depth, i = match.end(), 1, match.end()
        while i < len(text) and depth:
            if text[i] == "\\":
                i += 2
                continue
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            result = text[start : i - 1]
    return result


def eval_expression(expression: str) -> float | None:
    cleaned = (expression or "").strip().replace("^", "**")
    if not cleaned or not re.fullmatch(r"[0-9+\-*/(). ]+", cleaned):
        return None
    try:
        return eval(cleaned, {"__builtins__": {}}, {})  # noqa: S307 - digits only
    except Exception:  # noqa: BLE001
        return None


def last_equals_value(text: str) -> str | None:
    """The right-hand side of the *last* ``= <number>`` anywhere in ``text``."""
    matches = _TRAILER_RE.findall(text or "")
    return matches[-1] if matches else None


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
        elif branch == SOLVABLE_BRANCH:
            if not gt["solvable"] or not gt["answer"]:
                problems.append(f"{task_id}: solvable row without an answer")
            if not task_id.endswith("-solvable"):
                problems.append(f"{task_id}: solvable branch with a mismatched task_id")
            if not row["prompt"][0]["content"].startswith(info["paired_original_text"]):
                problems.append(f"{task_id}: solvable prompt is not the source question")
        else:
            problems.append(f"{task_id}: unexpected branch {branch!r}")
    # index must match the row's position (the mix stage relies on it)
    for position, row in enumerate(rows):
        if row["extra_info"]["index"] != position:
            problems.append(f"{row['extra_info']['task_id']}: index != position")
            break
    report.check(
        "contract (D12 / template B / branch / prompt)",
        not problems,
        f"{len(rows)} rows, {len(problems)} problems"
        + (f" | first: {problems[0]}" if problems else ""),
    )


def check_l1(report: Report, rows: list[dict]) -> None:
    """Re-derive every row's gold from its own bytes, with a second implementation."""
    unsolvable = [r for r in rows if r["extra_info"]["branch"] == UNSOLVABLE_BRANCH]
    solvable = [r for r in rows if r["extra_info"]["branch"] == SOLVABLE_BRANCH]

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

    bad_solvable: list[str] = []
    tiers: collections.Counter[str] = collections.Counter()
    for row in solvable:
        info = row["extra_info"]
        task_id = info["task_id"]
        derivation = info["perturbed_entity_text"]
        gold = json.loads(row["reward_model"]["ground_truth"])["answer"]
        if task_id.split("-")[1] == "math":
            boxed = last_boxed(derivation)
            if boxed is None or norm_ws(boxed) != norm_ws(gold):
                bad_solvable.append(f"{task_id}: last \\boxed{{{boxed}}} != gold {gold!r}")
            else:
                tiers["solution_boxed"] += 1
            continue
        recomputed, tier = _recompute_gsm8k(derivation)
        # Numeric, not string, equality: the source may state the same value as
        # "16.00" where the key says "16", and that is a spelling difference, not
        # a different gold (the recon's necessity convention: Decimals, with only
        # the percent fold -- which does not arise between two golds).
        if recomputed is None or as_decimal(recomputed) != as_decimal(gold):
            bad_solvable.append(f"{task_id}: recomputed {recomputed!r} != gold {gold!r}")
        else:
            tiers[tier] += 1

    report.info(
        "L1 sample",
        f"{len(unsolvable) + len(solvable)} of {len(rows)} rows audited "
        f"(floor {MIN_L1_SAMPLE}); unsolvable-by-reconstruction {len(unsolvable)}, "
        f"solvable-by-recomputation {len(solvable)}",
    )
    report.info(
        "L1 gsm8k recomputation tiers",
        "recomputed from the source's own arithmetic: "
        + ", ".join(f"{name}={count}" for name, count in sorted(tiers.items())),
    )
    report.check(
        "L1 unsolvable: deleted span re-derived + necessity re-proved",
        not bad_unsolvable,
        f"{len(unsolvable)} rows, {len(bad_unsolvable)} uncertified"
        + (f" | first: {bad_unsolvable[0]}" if bad_unsolvable else ""),
    )
    report.check(
        "L1 solvable: gold recomputed from the derivation",
        not bad_solvable,
        f"{len(solvable)} rows, {len(bad_solvable)} uncertified"
        + (f" | first: {bad_solvable[0]}" if bad_solvable else ""),
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


def _recompute_gsm8k(derivation: str) -> tuple[str | None, str]:
    """Recompute the gsm8k answer: the final stated step must equal the key.

    Two independent recomputations, both from the source's own text:
    ``annotation`` re-evaluates the last ``<<expr=val>>``; ``final_step`` reads
    the value to the right of the last ``=`` in the whole text.  The check
    passes when either agrees with the ``####`` key, which is exactly the
    adapter's own admission rule re-derived rather than trusted.
    """
    key_match = re.search(r"####\s*(.+?)\s*$", (derivation or "").strip(), re.S)
    if key_match is None:
        return None, "no_key"
    key = key_match.group(1).strip().replace(",", "").replace("$", "").rstrip(".")
    pairs = [body.rsplit("=", 1) for body in _ANN_RE.findall(derivation or "") if "=" in body]
    for expression, value in reversed(pairs):
        evaluated = eval_expression(expression)
        if evaluated is None:
            continue
        if as_decimal(value) == as_decimal(key) == as_decimal(str(evaluated)):
            return value.strip(), "annotation"
    trailer = last_equals_value(derivation)
    if trailer is not None and as_decimal(trailer) == as_decimal(key):
        return trailer, "final_step"
    return None, "uncertified"


def check_l2(report: Report, rows: list[dict]) -> None:
    """Gold uniqueness: the deleted value is really, unambiguously absent."""
    unsolvable = [r for r in rows if r["extra_info"]["branch"] == UNSOLVABLE_BRANCH]
    solvable_by_base: dict[str, dict] = {}
    for row in rows:
        if row["extra_info"]["branch"] == SOLVABLE_BRANCH:
            solvable_by_base[row["extra_info"]["task_id"][: -len("-solvable")]] = row

    still_readable: list[str] = []
    same_value_other_spelling: list[str] = []
    multi_value: list[str] = []
    broken_pair: list[str] = []
    conflicted_final: list[str] = []
    checked_pairs = 0

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
        # the paired contrast: the same premise must be present on the other side
        base = task_id[: -len("-unsolvable")]
        twin = solvable_by_base.get(base)
        if twin is not None:
            checked_pairs += 1
            if norm_ws(recovered) not in norm_ws(twin["extra_info"]["paired_original_text"]):
                broken_pair.append(f"{task_id}: recovered span missing from the solvable twin")

    for twin in solvable_by_base.values():
        info = twin["extra_info"]
        if info["task_id"].split("-")[1] != "gsm8k":
            continue
        gold = json.loads(twin["reward_model"]["ground_truth"])["answer"]
        # "the only admissible option": the derivation's final stated value is
        # the gold and nothing else is claimed as the answer.
        final = last_equals_value(info["perturbed_entity_text"])
        key = re.search(r"####\s*(.+?)\s*$", info["perturbed_entity_text"].strip(), re.S)
        key_value = key.group(1).strip() if key else None
        for candidate, label in ((final, "final step"), (key_value, "answer key")):
            if candidate is not None and as_decimal(candidate) != as_decimal(gold):
                conflicted_final.append(
                    f"{info['task_id']}: {label} claims {candidate!r} but gold is {gold!r}"
                )
                break

    report.info(
        "L2 scope",
        "MiP rows carry no option block (D12), so uniqueness is audited as "
        "\"the deleted value is really absent from the presented question\" "
        f"plus the pair reconstruction; {checked_pairs} pairs checked",
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
    report.check(
        "L2 pair reconstruction (deleted span present in the solvable twin)",
        not broken_pair,
        f"{checked_pairs} pairs, {len(broken_pair)} broken"
        + (f" | first: {broken_pair[0]}" if broken_pair else ""),
    )
    report.check(
        "L2 solvable gold is the unique final value of its derivation",
        not conflicted_final,
        f"{len(solvable_by_base)} solvable rows, {len(conflicted_final)} with a rival gold"
        + (f" | first: {conflicted_final[0]}" if conflicted_final else ""),
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
            "the question\", so the adapter builds none (doc section 4.3 says 95.6%; the "
            "recon measured 100.0% and this re-measurement agrees)",
        )


# ---------------------------------------------------------------------------
# L3c: from-scratch multinomial Naive Bayes (sklearn is not installed)
# ---------------------------------------------------------------------------


def _bag(text: str) -> list[str]:
    return [m.group(0) for m in _TOKEN_RE.finditer((text or "").lower())]


def _balanced_accuracy(y_true: list[int], y_pred: list[int]) -> float:
    recalls = []
    for label in (0, 1):
        idx = [i for i, y in enumerate(y_true) if y == label]
        if not idx:
            continue
        recalls.append(sum(y_pred[i] == label for i in idx) / len(idx))
    return sum(recalls) / len(recalls) if recalls else float("nan")


def _naive_bayes_oof(
    docs: list[list[str]],
    labels: list[int],
    *,
    folds: int = NB_FOLDS,
    seed: int = 0,
    min_support: int = NB_MIN_SUPPORT,
    alpha: float = NB_ALPHA,
) -> tuple[float, list[float], int]:
    """Out-of-fold balanced accuracy of a multinomial NB, plus diagnostics.

    Vocabulary is restricted to tokens appearing in at least ``min_support``
    *training* documents of the fold, per the audit contract.  The returned
    ``scores`` are the model's own class-separation margins on the held-out
    fold (mean log-odds for the positive class minus the negative class), which
    is what exposes the sign-flip: a healthy classifier separates the training
    fold the same way it separates the held-out fold.
    """
    rng = random.Random(seed)
    order = list(range(len(docs)))
    rng.shuffle(order)
    fold_of = {index: position % folds for position, index in enumerate(order)}

    predictions: list[int | None] = [None] * len(docs)
    margins: list[float] = []
    vocab_sizes: list[int] = []
    for fold in range(folds):
        test = [i for i in range(len(docs)) if fold_of[i] == fold]
        train = [i for i in range(len(docs)) if fold_of[i] != fold]
        support: collections.Counter[str] = collections.Counter()
        for index in train:
            support.update(set(docs[index]))
        vocab = {token for token, count in support.items() if count >= min_support}
        vocab_sizes.append(len(vocab))

        counts = {0: collections.Counter(), 1: collections.Counter()}
        totals = {0: 0, 1: 0}
        n_docs = {0: 0, 1: 0}
        for index in train:
            label = labels[index]
            n_docs[label] += 1
            for token in docs[index]:
                if token in vocab:
                    counts[label][token] += 1
                    totals[label] += 1
        log_prior = {
            label: math.log(max(n_docs[label], 1) / max(len(train), 1)) for label in (0, 1)
        }
        vocab_list = sorted(vocab)
        for index in test:
            scores = {}
            for label in (0, 1):
                score = log_prior[label]
                denominator = totals[label] + alpha * max(len(vocab_list), 1)
                for token in docs[index]:
                    if token in vocab:
                        score += math.log(
                            (counts[label][token] + alpha) / denominator
                        )
                scores[label] = score
            margins.append(scores[1] - scores[0])
            predictions[index] = 1 if scores[1] > scores[0] else 0

    balanced = _balanced_accuracy(labels, [p if p is not None else 0 for p in predictions])
    return balanced, margins, max(vocab_sizes) if vocab_sizes else 0


def check_l3c(report: Report, rows: list[dict]) -> None:
    solvable = [r for r in rows if r["extra_info"]["branch"] == SOLVABLE_BRANCH]
    unsolvable = [r for r in rows if r["extra_info"]["branch"] == UNSOLVABLE_BRANCH]
    if not solvable or not unsolvable:
        report.note(
            "L3c skipped",
            f"the artifact has one side only (solvable={len(solvable)}, "
            f"unsolvable={len(unsolvable)}); a balanced-accuracy test is undefined, so "
            "the option-presence/length cues above are the whole L3 evidence",
        )
        return

    # The bag is the *question*, not the whole prompt: the template B tail is
    # byte-identical on both sides, so including it would only dilute the signal
    # with a constant.
    docs, labels = [], []
    for row in solvable:
        docs.append(_bag(presented_question(row)))
        labels.append(1)
    for row in unsolvable:
        docs.append(_bag(presented_question(row)))
        labels.append(0)

    balanced, margins, vocab_size = _naive_bayes_oof(docs, labels)
    report.info(
        "L3c setup",
        f"n={len(docs)} (solvable {len(solvable)} / unsolvable {len(unsolvable)}), "
        f"{NB_FOLDS}-fold out-of-fold, vocab = train tokens with support >= {NB_MIN_SUPPORT} "
        f"(largest fold vocabulary {vocab_size} types), alpha={NB_ALPHA}, sklearn absent -> "
        "own numpy/python implementation",
    )
    report.check(
        f"L3c out-of-fold balanced accuracy <= {NB_BA_LIMIT:.2f}",
        balanced <= NB_BA_LIMIT,
        f"raw balanced accuracy = {balanced:.4f} (chance 0.5000, limit {NB_BA_LIMIT:.2f})",
    )

    inverted = 1.0 - balanced
    lengths = {1: [len(d) for d, y in zip(docs, labels, strict=False) if y == 1],
               0: [len(d) for d, y in zip(docs, labels, strict=False) if y == 0]}
    print(
        f"NOTE  L3c raw diagnostics  inverted reading = {inverted:.4f} "
        f"(a multinomial NB whose sign flips out of fold scores below chance, so the "
        f"raw {balanced:.4f} is not by itself evidence that the two sides are "
        f"indistinguishable); mean token count solvable {sum(lengths[1]) / len(lengths[1]):.1f} "
        f"vs unsolvable {sum(lengths[0]) / len(lengths[0]):.1f}"
    )
    # length-only baseline: the strongest single surface cue, measured not assumed
    best = 0.0
    best_threshold = 0
    for threshold in sorted(set(lengths[1] + lengths[0])):
        predicted = [1 if len(doc) >= threshold else 0 for doc in docs]
        score = _balanced_accuracy(labels, predicted)
        if score > best:
            best, best_threshold = score, threshold
    print(
        f"NOTE  L3c length-only baseline  best balanced accuracy = {best:.4f} at "
        f"token-count threshold {best_threshold} (a length cue is a surface cue, not an "
        "option-selection shortcut: D12 leaves no option block to select)"
    )
    if inverted > NB_BA_LIMIT or best > NB_BA_LIMIT:
        report.note(
            "L3c caveat",
            "the raw balanced accuracy passes, but the un-inverted reading "
            f"({inverted:.4f}) and/or the length-only baseline ({best:.4f}) exceed "
            f"{NB_BA_LIMIT:.2f}: this source's two sides are separable by length. Both "
            "numbers are printed on every run rather than hidden behind the raw figure.",
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
    check_l3c(report, rows)

    print(
        f"\n{len(report.passed)} check(s) passed, {len(report.failed)} failed"
        + (f": {', '.join(report.failed)}" if report.failed else "")
    )
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
