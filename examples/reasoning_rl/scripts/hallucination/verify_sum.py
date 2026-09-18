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
"""Audit the SUM rows produced by ``sum_adapter.py`` (design doc section 9, "SUM").

Usage::

    python verify_sum.py --rows /tmp/sum.parquet --raw-dir <sum raw dir>

Reads the artifact and, where the gold can only be proved against the source,
``train.parquet`` under ``--raw-dir``.  Every check re-derives its answer from
the two texts a row carries (``prompt`` question + ``extra_info`` pair fields)
with this file's **own** token diff, never by calling the adapter, so an adapter
bug cannot certify itself.  Prints ``PASS``/``FAIL`` per check, ``----`` for
measurements that are reported rather than asserted, and exits non-zero if any
hard check fails.

Layers
------

**L0 -- contract and pairing** (section 9 (1)).  ``schema.validate_row`` on every
row, the branch invariants (gold shape, option block, diagnosis label), unique
and source-derived ``task_id``, and the pairing non-empty assertion: both members
of the pair must be present, non-empty and different.  SUM's pair is one row of
``train.parquet``, so an empty member is never a legitimate row.

**Source anchor.**  Each row's ``extra_info.index`` re-reads the source row it
claims: the presented question must be that row's own member text, the paired
original must be the other member, the judgment row's audit answer must be that
row's own ``ground_truth`` string, and the defect label must be one of the five
the classifier can emit.  Without the raw file the pair cannot be proved, so a
missing ``--raw-dir`` is a FAIL, not a skip.

**L1 -- gold certificate** (>= 50 rows per branch, every row by default).  This
file re-diffs the two stored texts and demands that the recorded defect *is* that
diff:

* ``unsolvable_diag``: the pair's re-derived diff has exactly one inserted
  region, its text is the gold option, the recorded deleted/inserted fields
  reproduce the diff token for token, and -- the clause the first build was
  missing -- that inserted text is a defect one of the three visible rules can
  name (:func:`certifies_as_defect`: the residual class only when it introduces a
  content word the answerable member never carried).  A row whose gold points at
  paraphrase padding ("and the", "its", ``How`` appended to a statement that is
  still answerable) fails here even though every other clause holds.
* ``unsolvable_bare``: the re-derived diff inserts nothing, and a content word of
  the recorded deleted text is really absent from the presented question.
* ``solvable_judge``: the pair's diff is the recorded defect, the audit answer is
  non-empty, and the option block is present.  SUM ships **no derivation chain**
  (only the final ``ground_truth``), so the answer itself is proved by the source
  anchor above -- the only place it can be proved -- and this adapter does not
  claim the necessity of the removed premise was established.

**L2 -- gold uniqueness.**  ``unsolvable_diag``: the gold must be absent from the
answerable member (text that already existed in the original cannot be the defect
that made the variant unanswerable) and must be unambiguous in the question.
``unsolvable_bare``: the deleted value must really be gone.  ``solvable_judge``:
the row must carry no ``correct_option_id`` -- contract, not a missing check.

**L3 -- anti-cheat.**  (a) every option occurs verbatim in the row's own
question; (b) all of a row's options have equal token length; (c) no option is a
content-free filler (:mod:`distractor_mining` rule 3, applied to the whole block
because the option the *caller* contributes as the gold is appended without that
filter -- the only exemption is a diag gold that is itself the vague or
impossible defect, where the function word "some"/"few" *is* the defect); (d) a
bag-of-words multinomial Naive Bayes, 5-fold cross-validated, must not beat chance on the
solvable/unsolvable label by more than 0.05 (out-of-fold balanced accuracy
<= 0.55, the doc's hard gate).  The estimator is :func:`verify_umwp.bow_nb_oof`
-- one ruler for every source, and ``sklearn`` is not installed here.  The corpus
is the artifact drawn at the **D18 quota proportions** (judge 350 : diag 5,094 :
bare 2,000, scaled to ``--l3-rows``), because the gate is about the composition
the mix trains on, not about how large a pool a branch happens to emit; the
composition, the reading and a shuffled-label control are printed together.

**Yield recomputation** (section 9 (2)).  The four-tier / three-tier yields are
recomputed from ``train.parquet`` with the same word-level ``difflib`` opcode
projection the doc names.  The doc claims 9,832 / 6,050; the deviations are
reported as ``----`` lines with the measurement, and a deviation beyond the doc's
own 2% band is called out as a finding -- not as an audit failure, because a
number the data does not support is a documentation defect, not a bad row.

**Coverage.**  L1 and L2 are exhaustive: every row of the artifact is re-derived
from its raw pair, with no sampling, by default (``--per-branch 0``).
``--per-branch N`` caps the sample per branch for a faster pass and makes
:func:`main` print ``NOT exhaustive`` beside the coverage line, so a green
capped run can never be read as a green audit.

**Type-classifier spot check** (section 9 (4), Q26, N=100).  SUM ships no type
labels, so the five defect types come from the adapter's rule-based classifier,
which serves balancing and monitoring only.  Q26 asks for a human spot check; its
substitute here is a **second, independently written** implementation of the same
five rules, run over a stratified N=100 sample: each labelled row must satisfy
its type's defining predicate, and no higher-priority predicate may fire.  The
failure rate is reported and >10% fails the check (the doc downgrades the source
in that case).  This measures agreement between two implementations of one spec
and the residual class's own predicate -- it is not a human review and is not
presented as one.  The ``question_missing`` predicate is the one rule the two
implementations disagreed about twice: it requires the clause to overlap a
deleted run **and** to be gone from the presented member (:func:`clause_cut`).
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
import os
import random
import re
import sys

import numpy as np

try:
    import schema
    import verify_umwp
except ImportError:  # pragma: no cover - running as a plain script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import schema
    import verify_umwp

BRANCH_DIAG = schema.BRANCH_UNSOLVABLE_DIAG
BRANCH_BARE = schema.BRANCH_UNSOLVABLE_BARE
BRANCH_JUDGE = schema.BRANCH_SOLVABLE_JUDGE

DEFAULT_RAW_DIR = "/home/charles/data/reasoning_rl/halluc/raw/sum"
DATA_FILE = "train.parquet"

L3_MAX_BALANCED_ACCURACY = 0.55
L3_MIN_SUPPORT = 5
L1_MIN_SAMPLE = 50

#: The D18 quota composition of this source's branches (design doc section
#: 4.9.3 table B).  The L3 corpus is drawn at these proportions.
D18_QUOTA = {BRANCH_JUDGE: 350, BRANCH_DIAG: 5094, BRANCH_BARE: 2000}
L3_DEFAULT_ROWS = 4000

#: The doc's claimed yields (section 4.9.1 table A / section 9 (2)).
DOC_FOUR_TIER_YIELD = 9832
DOC_THREE_TIER_YIELD = 6050
DOC_YIELD_TOLERANCE = 0.02

#: The five defect types and the branch each one routes to.  Kept here rather
#: than imported so a change to the adapter's table shows up as an audit failure
#: instead of silently redefining what the audit checks.
DEFECT_TYPES = (
    "missing_necessary_condition",
    "ambiguous_key_information",
    "unrealistic_self_contradiction",
    "irrelevant_undefined_entity",
    "question_missing",
)
BARE_TYPES = ("missing_necessary_condition", "question_missing")
DIAG_TYPES = (
    "unrealistic_self_contradiction",
    "ambiguous_key_information",
    "irrelevant_undefined_entity",
)

#: The doc's own type proportions (section 4.9.3 balance table); reported, never
#: asserted -- the source carries no labels to assert them against.
DOC_TYPE_SHARE = {
    "ambiguous_key_information": 0.385,
    "unrealistic_self_contradiction": 0.230,
    "question_missing": 0.205,
    "irrelevant_undefined_entity": 0.180,
}


class Audit:
    """Collects ``PASS`` / ``FAIL`` lines plus the exit status."""

    def __init__(self) -> None:
        self.failures = 0
        self.lines: list[str] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        status = "PASS" if ok else "FAIL"
        if not ok:
            self.failures += 1
        suffix = f"  {detail}" if detail else ""
        self.lines.append(f"{status}  {name}{suffix}")
        return ok

    def note(self, name: str, detail: str) -> None:
        self.lines.append(f"----  {name}  {detail}")

    def flush(self) -> None:
        for line in self.lines:
            print(line)
        print()


# ---------------------------------------------------------------------------
# independent re-implementations (deliberately separate from the adapter)
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’\-]*")

# Loaded dice: a minus-signed number or an impossibility word.
_IMPOSSIBLE_RE = re.compile(
    r"(?:^|[\s(\[{])[-−]\s?\d"
    r"|\b(?:negative|impossible|undefined|imaginary|infinite|infinity"
    r"|cannot exist|does not exist|no solution)\b",
    re.I,
)
# A vague quantifier; same vocabulary as the adapter, written out again here.
_VAGUE_RE = re.compile(
    r"\b(?:some|several|few|many|most|either|certain|various|multiple|numerous"
    r"|couple|approximately|roughly|nearly|unknown|unclear|unspecified|arbitrary"
    r"|varies|varying|fluctuat\w*|depends|random|unstated|unpredictable|unfixed"
    r"|not\s+(?:specified|stated|fixed|defined|given)|isn[’']t\s+fixed)\b",
    re.I,
)


def normalise(text: str) -> str:
    """Collapse whitespace and strip -- the form the adapter renders."""
    return _WS_RE.sub(" ", (text or "")).strip()


def words(text: str) -> list[str]:
    """Word tokens in the same grammar as the adapter and the option miner."""
    return [m.group(0) for m in _TOKEN_RE.finditer(text or "")]


class Region:
    """One re-derived changed run: what left the original and what took its place."""

    __slots__ = ("tag", "deleted", "inserted", "a_start", "b_start")

    def __init__(self, tag: str, deleted: str, inserted: str, a_start: int, b_start: int) -> None:
        self.tag = tag
        self.deleted = deleted
        self.inserted = inserted
        self.a_start = a_start
        self.b_start = b_start


def diff_regions(original: str, rewritten: str) -> list[Region]:
    """Word-level ``difflib`` regions of ``original -> rewritten``, with offsets.

    The same projection the adapter uses, re-implemented here so the audit's
    view of "what changed" does not come from the adapter's own code.
    """
    ta = [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(original)]
    tb = [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(rewritten)]
    matcher = difflib.SequenceMatcher(
        None, [t[0] for t in ta], [t[0] for t in tb], autojunk=False
    )
    runs: list[tuple] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if runs and runs[-1][2] == i1 and runs[-1][4] == j1:
            prev_tag, oi1, _, oj1, _ = runs[-1]
            runs[-1] = (prev_tag, oi1, i2, oj1, j2)
        else:
            runs.append((tag, i1, i2, j1, j2))

    regions: list[Region] = []
    for tag, i1, i2, j1, j2 in runs:
        a_start = ta[i1][1] if i1 < len(ta) else len(original)
        a_end = ta[i2 - 1][2] if i2 > i1 else a_start
        b_start = tb[j1][1] if j1 < len(tb) else len(rewritten)
        b_end = tb[j2 - 1][2] if j2 > j1 else b_start
        regions.append(
            Region(tag, original[a_start:a_end], rewritten[b_start:b_end], a_start, b_start)
        )
    return regions


def emitting(regions: list[Region]) -> list[Region]:
    """Regions that put text into the rewritten member (the pointerable ones)."""
    return [region for region in regions if region.inserted.strip()]


def question_clause(question: str) -> tuple[str, int]:
    """The final interrogative sentence of ``question`` and its offset.

    ``("", -1)`` when there is no ``?``.  The offset comes from the last sentence
    break before that ``?`` -- a question can repeat its own clause verbatim
    (``sum-uns-25944`` does, at offsets 260 and 475), and ``str.find`` would
    report the earlier one for both, locating the clause somewhere it is not.
    """
    mark = question.rfind("?")
    if mark < 0:
        return "", -1
    start = question.rfind(".", 0, mark)
    return question[start + 1 : mark + 1], start + 1


def clause_extent(clause: str, offset: int) -> tuple[int, int]:
    """``(first, last_exclusive)`` character offsets of the clause's own text.

    Whitespace does not count as clause text: the clause as sliced starts with
    the blank run that separated it from the previous sentence, and a rule
    measured against the slice would read a deletion touching only that blank as
    a cut question.  No SUM row exercises the difference (both readings select
    the same 2,243 candidate rows), so this is a guard -- the condition that does
    move rows is :func:`clause_cut`'s requirement that the clause also be gone
    from the presented member.
    """
    stripped = clause.strip()
    if not stripped:
        return offset, offset
    lead = len(clause) - len(clause.lstrip())
    return offset + lead, offset + lead + len(stripped)


def clause_cut(original: str, presented: str, regions: list[Region]) -> bool:
    """Whether the question clause was deleted and is gone from ``presented``.

    Both halves of the adapter's rule: the deleted run must overlap the clause's
    own text (not the blank before it), and the clause must no longer occur in
    the presented member.  A clause that survives -- because the answerable
    member states it twice, or because a word-level alignment charged a surviving
    word to the deleted run -- leaves the question standing, so the pair is a
    missing condition, not a missing question.
    """
    clause, offset = question_clause(original)
    if not clause:
        return False
    stripped = clause.strip()
    if not stripped:
        return False
    first, last = clause_extent(clause, offset)
    overlaps = any(
        region.deleted.strip()
        and region.a_start < last
        and first < region.a_start + len(region.deleted)
        for region in regions
    )
    return overlaps and stripped not in presented


def info_gone(deleted: str, presented: str) -> bool:
    """A content word of the deleted text is absent from the presented question.

    Set-based, not substring-based: ``10m`` removed from ``100m sections`` is a
    substring coincidence, but ``10`` is not one of the question's tokens.
    """
    presented_tokens = {token.lower() for token in words(presented)}
    return any(
        schema.is_content_word(token) and token.lower() not in presented_tokens
        for token in words(deleted)
    )


def introduces_undefined_entity(inserted: str, original: str) -> bool:
    """The residual class's own predicate, written out again for the audit.

    ``irrelevant_undefined_entity`` means "this span names something the original
    problem never carried"; a span whose content words all already stand in the
    answerable member names nothing new and is not that defect.
    """
    original_tokens = {token.lower() for token in words(original)}
    return any(
        schema.is_content_word(token) and token.lower() not in original_tokens
        for token in words(inserted)
    )


def certifies_as_defect(inserted: str, original: str) -> bool:
    """Whether one of the three visible rules justifies pointing at ``inserted``.

    The rule the adapter's certificate implements, re-derived here: the two
    non-residual classes are justified by the rule that fires, and the residual
    class only by its defining predicate.  A span that satisfies neither is
    padding the paraphrase added -- "and the", "its", "these", the sentence-initial
    capital a replace left behind -- and a gold pointing at it is a gold nothing in
    the row can justify, however faithfully the row records it (deviation 13).
    """
    if _IMPOSSIBLE_RE.search(inserted) or _VAGUE_RE.search(inserted):
        return True
    return introduces_undefined_entity(inserted, original)


def multiset_after_edit(original: str, rewritten: str, deleted: str, inserted: str) -> bool:
    """``tokens(rewritten)`` must equal ``tokens(original) - deleted + inserted``."""
    expected = collections.Counter(token.lower() for token in words(original))
    expected.subtract(token.lower() for token in words(deleted))
    expected.update(token.lower() for token in words(inserted))
    expected = +expected
    return expected == collections.Counter(token.lower() for token in words(rewritten))


def recorded_defect_matches(original: str, rewritten: str, deleted: str, inserted: str) -> str:
    """Whether the recorded defect is exactly what this file's diff re-derives.

    Returns ``""`` on agreement or a one-line reason.  The comparison is on token
    sequences, so punctuation and whitespace differences between the two members
    cannot mask a genuine mismatch.
    """
    regions = diff_regions(original, rewritten)
    rederived_deleted = " ".join(region.deleted for region in regions if region.deleted.strip())
    rederived_inserted = " ".join(
        region.inserted.strip() for region in emitting(regions)
    )
    if words(deleted) != words(rederived_deleted):
        return f"recorded deleted text != re-derived deletion ({deleted[:40]!r})"
    if words(inserted) != words(rederived_inserted):
        return f"recorded inserted text != re-derived insertion ({inserted[:40]!r})"
    return ""


# ---------------------------------------------------------------------------
# reading the artifact
# ---------------------------------------------------------------------------


def question_of(row: dict) -> str:
    """The question the model sees: the prompt is ``question + "\\n\\n" + template``.

    The adapter collapses every newline inside a question, so the first blank
    line is the template boundary.  A prompt without one means the rendering
    changed and every text check below would read the wrong string, so it is a
    hard error rather than a silently truncated comparison.
    """
    content = row["prompt"][0]["content"]
    head, sep, _ = content.partition("\n\n")
    if not sep:
        raise ValueError(f"prompt of {row['extra_info'].get('task_id')} has no template separator")
    return head


def payload_of(row: dict) -> dict:
    return json.loads(row["reward_model"]["ground_truth"])


def option_text(row: dict, option_id: str | None) -> str | None:
    for option in row["extra_info"].get("options") or []:
        if option.get("id") == option_id:
            return option.get("text")
    return None


def orient(row: dict) -> tuple[str, str]:
    """``(answerable member text, unanswerable member text)`` for this row."""
    question = question_of(row)
    partner = row["extra_info"].get("paired_original_text", "")
    if row["extra_info"].get("solvable"):
        return question, partner
    return partner, question


def load_source(raw_dir: str) -> list[dict]:
    """``train.parquet`` as a list of rows (empty when unreadable)."""
    path = os.path.join(raw_dir, DATA_FILE)
    if not os.path.exists(path):
        return []
    return schema.read_parquet_rows(path)


# ---------------------------------------------------------------------------
# L0 -- contract and pairing
# ---------------------------------------------------------------------------


def check_contract(rows: list[dict], audit: Audit) -> None:
    violations: list[str] = []
    for row in rows:
        violations.extend(schema.validate_row(row))
    audit.check(
        "L0a schema.validate_row on every row",
        not violations,
        f"{len(rows)} rows, {len(violations)} violations"
        + (f"; first: {violations[:3]}" if violations else ""),
    )

    failures: list[str] = []
    branch_of: dict[str, str] = {}
    for row in rows:
        info = row["extra_info"]
        task_id = info["task_id"]
        branch = info["branch"]
        ground = payload_of(row)
        branch_of.setdefault(branch, info["template"])
        if not info.get("solvable") and ground.get("answer") is not None:
            failures.append(f"{task_id}: unsolvable row carries an answer")
        if ground.get("judgment_only") and not info.get("solvable"):
            failures.append(f"{task_id}: judgment_only on an unsolvable row")
        if branch == BRANCH_JUDGE and not ground.get("judgment_only"):
            failures.append(f"{task_id}: judgment row without judgment_only")
        if branch == BRANCH_DIAG:
            if not ground.get("has_diagnosis_label"):
                failures.append(f"{task_id}: diag row without a diagnosis label")
            if ground.get("correct_option_id") not in [
                option["id"] for option in info.get("options") or []
            ]:
                failures.append(f"{task_id}: diag gold is not one of its options")
            if info.get("error_type") not in DIAG_TYPES:
                failures.append(f"{task_id}: diag row labelled {info.get('error_type')!r}")
        if branch == BRANCH_BARE:
            if ground.get("has_diagnosis_label") or info.get("options"):
                failures.append(f"{task_id}: bare row must not offer options")
            if info.get("error_type") not in BARE_TYPES:
                failures.append(f"{task_id}: bare row labelled {info.get('error_type')!r}")
        if info.get("error_type") not in DEFECT_TYPES:
            failures.append(f"{task_id}: unknown error_type {info.get('error_type')!r}")
    audit.check(
        "L0b branch invariants (gold shape / options / label)",
        not failures,
        f"{len(rows)} rows, {len(failures)} violations"
        + (f"; first: {failures[:3]}" if failures else ""),
    )
    audit.note("branch -> template", ", ".join(f"{b}:{t}" for b, t in sorted(branch_of.items())))

    # section 9 (1): the pairing non-empty assertion.  Both members of the pair
    # must be present, non-empty and different -- a SUM row *is* one source row
    # of a native pair, so an empty member is never legitimate.
    pairing: list[str] = []
    for row in rows:
        info = row["extra_info"]
        task_id = info["task_id"]
        question = question_of(row)
        partner = info.get("paired_original_text", "")
        if not question.strip():
            pairing.append(f"{task_id}: the presented question is empty")
        if not isinstance(partner, str) or not partner.strip():
            pairing.append(f"{task_id}: the paired original is empty")
        elif partner == question:
            pairing.append(f"{task_id}: the pair's two members are identical")
    audit.check(
        "L0c pairing non-empty (both members present, non-empty, distinct)",
        not pairing,
        f"{len(rows)} rows, {len(pairing)} violations"
        + (f"; first: {pairing[:3]}" if pairing else ""),
    )

    ids = [row["extra_info"]["task_id"] for row in rows]
    bad = [task_id for task_id in ids if not task_id.startswith("sum-")] if len(set(ids)) != len(ids) else []
    audit.check(
        "L0d task_id is unique and source-derived",
        not bad and len(set(ids)) == len(ids),
        f"{len(rows)} rows, {len(ids) - len(set(ids))} duplicate ids",
    )


def check_source_anchor(rows: list[dict], audit: Audit, source: list[dict], raw_dir: str) -> None:
    """Prove every row's texts and gold against ``train.parquet`` itself."""
    path = os.path.join(raw_dir, DATA_FILE)
    if not source:
        audit.check(
            "L0e source anchor (texts and golds re-read from the raw file)",
            False,
            f"cannot read {path} -- SUM ships no derivation chain, so the pair and the "
            "audit answer can only be proved against the source; re-run with --raw-dir "
            "pointing at the downloaded SUM directory",
        )
        return

    failures: list[str] = []
    for row in rows:
        info = row["extra_info"]
        task_id = info["task_id"]
        index = info.get("index")
        if not isinstance(index, int) or not 0 <= index < len(source):
            failures.append(f"{task_id}: index {index!r} is not a source row")
            continue
        src = source[index]
        answerable = normalise(src["answerable_question"])
        unanswerable = normalise(src["unanswerable_question"])
        if info["solvable"]:
            presented, partner, side = answerable, unanswerable, "ans"
        else:
            presented, partner, side = unanswerable, answerable, "uns"
        if question_of(row) != presented:
            failures.append(f"{task_id}: presented question differs from the source's")
            continue
        if info.get("paired_original_text") != partner:
            failures.append(f"{task_id}: paired original is not the source's other member")
            continue
        if task_id != f"sum-{side}-{index}":
            failures.append(f"{task_id}: task_id is not the source position {index} ({side})")
            continue
        ground = payload_of(row)
        if info["solvable"]:
            if ground.get("answer") != (src["ground_truth"] or "").strip():
                failures.append(f"{task_id}: audit answer != the source's ground_truth")
        elif ground.get("answer") is not None:
            failures.append(f"{task_id}: unsolvable row carries an answer")
    audit.check(
        "L0e source anchor (texts and golds re-read from the raw file)",
        not failures,
        f"{len(rows)} rows anchored to {path}, {len(failures)} failures"
        + (f"; first: {failures[:3]}" if failures else ""),
    )


# ---------------------------------------------------------------------------
# L1 / L2
# ---------------------------------------------------------------------------


def sample_by_branch(
    rows: list[dict], *, per_branch: int, seed: int
) -> dict[str, list[dict]]:
    """The rows L1/L2 re-derive: **every** row of a branch unless ``per_branch`` caps it.

    The default is exhaustive.  A sampled L1/L2 is blind to a single tampered
    stored field outside the sample, and an audit that passes on corrupted data
    proves nothing; re-diffing the whole artifact costs a few seconds.  A
    positive ``per_branch`` caps each branch, and the cap is printed, so a capped
    run cannot be mistaken for a full one.
    """
    groups: dict[str, list[dict]] = collections.OrderedDict()
    for row in rows:
        groups.setdefault(row["extra_info"]["branch"], []).append(row)
    rng = random.Random(seed)
    sample: dict[str, list[dict]] = {}
    for branch, group in groups.items():
        take = (
            len(group)
            if per_branch <= 0
            else min(len(group), max(per_branch, L1_MIN_SAMPLE))
        )
        sample[branch] = sorted(rng.sample(group, take), key=lambda row: row["extra_info"]["task_id"])
    return sample


def check_l1(sample: dict[str, list[dict]], audit: Audit) -> None:
    for branch, picked in sample.items():
        failures: list[str] = []
        for row in picked:
            info = row["extra_info"]
            task_id = info["task_id"]
            qa, qu = orient(row)
            regions = diff_regions(qa, qu)
            rederived = recorded_defect_matches(
                qa, qu, info.get("deleted_condition_text", ""), info.get("perturbed_entity_text", "")
            )
            if rederived:
                failures.append(f"{task_id}: {rederived}")
                continue
            ground = payload_of(row)
            if not multiset_after_edit(
                qa, qu, info.get("deleted_condition_text", ""), info.get("perturbed_entity_text", "")
            ):
                failures.append(f"{task_id}: presented question is not the original after the defect")
                continue
            if branch == BRANCH_DIAG:
                gold = option_text(row, info.get("correct_option_id"))
                if gold is None:
                    failures.append(f"{task_id}: correct_option_id is not an option of this row")
                    continue
                if gold.strip() != info.get("perturbed_entity_text", "").strip():
                    failures.append(f"{task_id}: gold {gold!r} != recorded defect")
                    continue
                pointable = emitting(regions)
                if len(pointable) != 1:
                    failures.append(f"{task_id}: pair has {len(pointable)} inserted regions, need 1")
                    continue
                if gold.strip() != pointable[0].inserted.strip():
                    failures.append(
                        f"{task_id}: gold {gold!r} is not the pair's re-derived insertion "
                        f"{pointable[0].inserted!r}"
                    )
                    continue
                if qu.find(gold) != pointable[0].b_start:
                    failures.append(f"{task_id}: gold is not located at the region's own offset")
                    continue
                if gold not in qu:
                    failures.append(f"{task_id}: gold is not a span of the presented question")
                    continue
                if not certifies_as_defect(gold, qa):
                    failures.append(
                        f"{task_id}: the insertion {gold!r} is not a defect any of the three "
                        "rules names, so the gold points at paraphrase padding"
                    )
            elif branch == BRANCH_BARE:
                if emitting(regions):
                    failures.append(f"{task_id}: a bare row's pair inserts text")
                    continue
                if not info_gone(info.get("deleted_condition_text", ""), qu):
                    failures.append(f"{task_id}: nothing informative is missing from {qu[:50]!r}")
            else:
                if not (ground.get("answer") or "").strip():
                    failures.append(f"{task_id}: judgment row has an empty audit answer")
                    continue
                if not (
                    info.get("deleted_condition_text", "").strip()
                    or info.get("perturbed_entity_text", "").strip()
                ):
                    failures.append(f"{task_id}: pair records no defect")
                    continue
                if not (info.get("options") or []):
                    failures.append(f"{task_id}: judgment row has no option block")
        detail = f"{len(picked)} sampled rows re-derived, {len(failures)} failures"
        if failures:
            detail += "; first: " + " | ".join(failures[:3])
        if branch == BRANCH_JUDGE:
            detail += " (answer proved by the source anchor; SUM ships no derivation chain)"
        audit.check(f"L1 {branch}", not failures, detail)


def check_l2(sample: dict[str, list[dict]], audit: Audit) -> None:
    for branch, picked in sample.items():
        failures: list[str] = []
        for row in picked:
            info = row["extra_info"]
            task_id = info["task_id"]
            qa, qu = orient(row)
            deleted = info.get("deleted_condition_text", "")
            if branch == BRANCH_DIAG:
                gold = option_text(row, info.get("correct_option_id"))
                if gold is None:
                    failures.append(f"{task_id}: no gold option")
                    continue
                if gold in qa:
                    failures.append(f"{task_id}: gold {gold!r} also occurs in the answerable member")
                    continue
                if qu.count(gold) != 1:
                    failures.append(f"{task_id}: gold {gold!r} occurs {qu.count(gold)} times in the question")
            elif branch == BRANCH_BARE:
                if not info_gone(deleted, qu):
                    failures.append(f"{task_id}: deleted value {deleted[:40]!r} is still present")
            else:
                if info.get("correct_option_id") not in ("", None):
                    failures.append(f"{task_id}: solvable row carries correct_option_id")
        detail = f"{len(picked)} sampled rows, {len(failures)} uniqueness failures"
        if branch == BRANCH_JUDGE:
            detail += " (gold is a verdict; checked as a contract)"
        if failures:
            detail += "; first: " + " | ".join(failures[:3])
        audit.check(f"L2 {branch}", not failures, detail)


# ---------------------------------------------------------------------------
# L3
# ---------------------------------------------------------------------------


def check_l3_options(rows: list[dict], audit: Audit) -> None:
    missing: list[str] = []
    unequal: list[str] = []
    checked = 0
    for row in rows:
        options = row["extra_info"].get("options") or []
        if not options:
            continue
        checked += 1
        question = question_of(row)
        task_id = row["extra_info"]["task_id"]
        for option in options:
            if option.get("text", "") not in question:
                missing.append(f"{task_id}:{option.get('text')!r}")
        lengths = {len(words(option.get("text", ""))) for option in options}
        if len(lengths) != 1:
            unequal.append(f"{task_id}:{sorted(lengths)}")
    audit.check(
        "L3a every option is a span of its own question",
        not missing,
        f"{checked} option-bearing rows, {len(missing)} options not found"
        + (f"; first: {missing[:3]}" if missing else ""),
    )
    audit.check(
        "L3b all options of a row have equal token length",
        not unequal,
        f"{checked} option-bearing rows, {len(unequal)} rows with unequal lengths"
        + (f"; first: {unequal[:3]}" if unequal else ""),
    )


def check_l3_option_content(rows: list[dict], audit: Audit) -> None:
    """L3d -- no filler option (:mod:`distractor_mining` rule 3, on every option).

    The miner filters its own distractors for content words, so the one option
    that can come back without one is the span the *caller* handed it as the
    gold, which it appends to the block unconditionally.  The audit therefore
    checks the property on the whole block, not just on the mined part.  One
    exemption is by design rather than a hole: a diag gold may be a vague
    quantifier ("some", "few"), and then the defect itself is a function word --
    those are exempt, and only those.
    """
    fillers: list[str] = []
    exempt = 0
    checked = 0
    for row in rows:
        info = row["extra_info"]
        options = info.get("options") or []
        if not options:
            continue
        checked += 1
        qa, _ = orient(row)
        for option in options:
            text = option.get("text", "")
            if any(schema.is_content_word(token) for token in words(text)):
                continue
            if (
                info["branch"] == BRANCH_DIAG
                and option.get("id") == info.get("correct_option_id")
                and certifies_as_defect(text, qa)
            ):
                exempt += 1
                continue
            fillers.append(f"{info['task_id']}:{text!r}")
    audit.check(
        "L3d no option is a content-free filler",
        not fillers,
        f"{checked} option-bearing rows, {len(fillers)} filler options "
        f"({exempt} diag golds exempted as the vague/impossible defect itself)"
        + (f"; first: {fillers[:3]}" if fillers else ""),
    )


def quota_corpus(
    rows: list[dict], *, total_rows: int, seed: int
) -> tuple[list[str], list[int], dict[str, int]]:
    """Draw the L3 corpus at the D18 quota proportions of the artifact.

    Returns ``(texts, labels, composition)``.  The gate is about the composition
    the mix trains on (judge 350 : diag 5,094 : bare 2,000 -- table B row 3/5/6),
    not about how many rows a branch happens to emit, and the dense estimator
    below makes a bounded corpus a requirement rather than a convenience.
    """
    groups: dict[str, list[dict]] = collections.OrderedDict()
    for row in rows:
        groups.setdefault(row["extra_info"]["branch"], []).append(row)
    quota_total = sum(D18_QUOTA.values())
    scale = min(1.0, total_rows / quota_total)
    rng = random.Random(seed)
    texts: list[str] = []
    labels: list[int] = []
    composition: dict[str, int] = {}
    for branch, quota in D18_QUOTA.items():
        group = groups.get(branch, [])
        take = min(len(group), max(int(round(quota * scale)), 1))
        if take <= 0:
            composition[branch] = 0
            continue
        picked = sorted(rng.sample(group, take), key=lambda row: row["extra_info"]["task_id"])
        composition[branch] = len(picked)
        for row in picked:
            texts.append(question_of(row))
            labels.append(1 if row["extra_info"]["solvable"] else 0)
    return texts, labels, composition


def check_l3_nb(
    rows: list[dict], audit: Audit, *, folds: int, seed: int, min_support: int, total_rows: int
) -> None:
    texts, labels, composition = quota_corpus(rows, total_rows=total_rows, seed=seed)
    positives = sum(labels)
    audit.note(
        "L3c corpus",
        f"D18-quota sample n={len(texts)} of {len(rows)} artifact rows "
        f"(solvable={positives}, unsolvable={len(texts) - positives}) "
        f"from {composition}; folds={folds}, vocabulary support >= {min_support} docs, alpha=1.0",
    )
    if positives == 0 or positives == len(texts):
        audit.check(
            "L3c out-of-fold balanced accuracy <= 0.55",
            False,
            "only one label side present in the artifact -- the estimator is not identifiable; "
            "build with a --limit that keeps all three branches (the adapter interleaves them)",
        )
        return
    accuracy = verify_umwp.bow_nb_oof(
        texts, labels, folds=folds, seed=seed, min_support=min_support
    )
    control_rng = random.Random(seed + 1)
    control_labels = [control_rng.random() > 0.5 for _ in texts]
    control = verify_umwp.bow_nb_oof(
        texts, control_labels, folds=folds, seed=seed, min_support=min_support
    )
    audit.note("L3c raw out-of-fold balanced accuracy", f"{accuracy:.4f} (chance 0.5)")
    audit.note("L3c shuffled-label control", f"{control:.4f}")
    audit.check(
        "L3c out-of-fold balanced accuracy <= 0.55",
        accuracy <= L3_MAX_BALANCED_ACCURACY,
        f"balanced accuracy {accuracy:.4f} <= {L3_MAX_BALANCED_ACCURACY:.2f} on {len(texts)} rows "
        f"(control {control:.4f}; doc claims 0.502 / 0.503)",
    )


def check_l3_raw(
    source: list[dict], audit: Audit, *, sample_rows: int, seed: int, folds: int, min_support: int
) -> None:
    """Report the raw A-vs-U separability the doc's 0.502 claims to measure.

    Informational: the doc's number is only reachable as a small-class artifact,
    so the honest reading of the source pairs is reported next to its
    shuffled-label control instead of being asserted.
    """
    if not source or sample_rows <= 0:
        return
    rng = random.Random(seed)
    take = min(len(source), max(sample_rows // 2, 1))
    picked = sorted(rng.sample(source, take), key=lambda row: row["_index"] if "_index" in row else 0)
    texts = [normalise(row["answerable_question"]) for row in picked]
    texts += [normalise(row["unanswerable_question"]) for row in picked]
    labels = [1] * take + [0] * take
    accuracy = verify_umwp.bow_nb_oof(
        texts, labels, folds=folds, seed=seed, min_support=min_support
    )
    control_rng = random.Random(seed + 1)
    control = verify_umwp.bow_nb_oof(
        texts, [control_rng.random() > 0.5 for _ in texts], folds=folds, seed=seed, min_support=min_support
    )
    audit.note(
        "L3 raw-corpus reading (informational)",
        f"answerable vs unanswerable over a balanced {2 * take}-member sample: balanced "
        f"accuracy {accuracy:.4f}; shuffled-label control {control:.4f}.  The reading rises "
        f"with the sample (0.550 at 4,000 members, 0.603 at 12,000) because the vocabulary is "
        f"support-gated, and tops out near the survey's full-corpus 0.638; the doc's 0.502 is "
        f"the shuffled-label reading of this estimator, not a property of these pairs",
    )


# ---------------------------------------------------------------------------
# yield recomputation (section 9 (2))
# ---------------------------------------------------------------------------


def measure_pools(source: list[dict]) -> dict[str, int]:
    """The yield projection the doc names: word-level opcodes over the raw pairs."""
    counts = {
        "pairs": 0,
        "no_word_change": 0,
        "three_tier_pool": 0,  # pure deletion: nothing inserted, so nothing to point at
        "four_tier_pool": 0,  # exactly one inserted region: the defect is visible
        "multi_region": 0,
    }
    for row in source:
        counts["pairs"] += 1
        regions = diff_regions(
            normalise(row["answerable_question"]), normalise(row["unanswerable_question"])
        )
        if not regions:
            counts["no_word_change"] += 1
            continue
        count = len(emitting(regions))
        if count == 0:
            counts["three_tier_pool"] += 1
        elif count == 1:
            counts["four_tier_pool"] += 1
        else:
            counts["multi_region"] += 1
    return counts


def check_yields(source: list[dict], audit: Audit) -> None:
    if not source:
        audit.note("yield recomputation", "skipped: the raw file is unreadable")
        return
    counts = measure_pools(source)
    audit.note(
        "yield recomputation (opcode projection over the raw file)",
        f"{counts['pairs']} pairs: three-tier pool {counts['three_tier_pool']}, "
        f"four-tier pool {counts['four_tier_pool']}, multi-region {counts['multi_region']}, "
        f"no word change {counts['no_word_change']}",
    )
    for label, measured, claimed in (
        ("four-tier", counts["four_tier_pool"], DOC_FOUR_TIER_YIELD),
        ("three-tier", counts["three_tier_pool"], DOC_THREE_TIER_YIELD),
    ):
        deviation = (measured - claimed) / claimed
        within = abs(deviation) <= DOC_YIELD_TOLERANCE
        audit.note(
            f"yield {label} vs the doc's claim",
            f"measured {measured} vs doc {claimed} ({deviation:+.1%}) -- "
            + ("within the doc's 2% band" if within else "OUTSIDE the doc's 2% band (finding)"),
        )


# ---------------------------------------------------------------------------
# the type-classifier spot check (section 9 (4), Q26)
# ---------------------------------------------------------------------------


def type_predicate(row: dict, defect_type: str) -> str:
    """Whether the row satisfies its label's defining predicate ("" when it does).

    The second, independently written implementation of the five rules.  For the
    three visible types the priority order is rechecked as well: a row labelled
    with a lower-priority type while a higher-priority rule fires is a
    classification failure, which is the bug this check exists to catch.
    """
    info = row["extra_info"]
    qa, qu = orient(row)
    regions = diff_regions(qa, qu)
    inserted = info.get("perturbed_entity_text", "")
    deleted = info.get("deleted_condition_text", "")
    pointable = emitting(regions)
    if defect_type == "missing_necessary_condition":
        if pointable:
            return "the pair inserts text, so nothing is merely missing"
        if not info_gone(deleted, qu):
            return "no content word of the deleted text is gone from the question"
        if clause_cut(qa, qu, regions):
            return "the question clause was cut, which is question_missing"
        return ""
    if defect_type == "question_missing":
        if pointable:
            return "the pair inserts text, so nothing is merely missing"
        if not clause_cut(qa, qu, regions):
            return "the question clause was not cut"
        return ""
    if defect_type == "unrealistic_self_contradiction":
        if not _IMPOSSIBLE_RE.search(inserted):
            return "the inserted span carries no negative or impossible value"
        return ""
    if defect_type == "ambiguous_key_information":
        if not _VAGUE_RE.search(inserted):
            return "the inserted span carries no vague quantifier"
        if _IMPOSSIBLE_RE.search(inserted):
            return "the higher-priority unrealistic rule also fires"
        return ""
    # irrelevant_undefined_entity -- the residual class
    if _IMPOSSIBLE_RE.search(inserted) or _VAGUE_RE.search(inserted):
        return "a higher-priority rule fires"
    qa_tokens = {token.lower() for token in words(qa)}
    if not any(
        schema.is_content_word(token) and token.lower() not in qa_tokens for token in words(inserted)
    ):
        return "the inserted span introduces no entity absent from the answerable member"
    return ""


def check_type_spot_check(
    rows: list[dict], audit: Audit, *, per_type: int, seed: int
) -> None:
    groups: dict[str, list[dict]] = collections.OrderedDict((name, []) for name in DEFECT_TYPES)
    for row in rows:
        name = row["extra_info"].get("error_type", "")
        if name in groups:
            groups[name].append(row)
    rng = random.Random(seed)
    sampled = 0
    failed = 0
    details: list[str] = []
    shares: list[tuple[float, str]] = []
    for name, group in groups.items():
        take = min(len(group), per_type)
        picked = rng.sample(group, take) if take else []
        sampled += len(picked)
        misses = 0
        examples: list[str] = []
        for row in picked:
            reason = type_predicate(row, name)
            if reason:
                misses += 1
                if len(examples) < 2:
                    examples.append(f"{row['extra_info']['task_id']}: {reason}")
        failed += misses
        share = misses / len(picked) if picked else 0.0
        shares.append((share, name))
        details.append(f"{name}={len(picked)}/{misses} failed")
        if picked and share > 0.10:
            audit.note(
                f"spot check {name}; the doc downgrades the source above 10%",
                f"{misses}/{len(picked)} = {share:.0%} predicate failures: {examples}",
            )
    rate = failed / sampled if sampled else 0.0
    worst = max(shares, default=(0.0, "no type"))
    audit.note(
        "type spot check sample",
        f"N={sampled} over the five types ({', '.join(details)}); "
        + ("full N=100 available" if sampled >= 100 else "artifact too small for the full N=100"),
    )
    if sampled and sampled < 100:
        audit.note(
            "type spot check size",
            f"only {sampled} rows could be drawn (20 per type requested) -- the built "
            "artifact is too small; the full-sample run is the audit for a full build",
        )
    audit.check(
        "L4 type classifier spot check (Q26) failure rate <= 10%",
        rate <= 0.10,
        f"{failed}/{sampled} = {rate:.1%} of sampled rows fail their type's defining predicate "
        f"(doc: >10% downgrades the source; {worst[0] or 'no type'} carries the worst per-type share)",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the SUM rows built by sum_adapter.py.")
    parser.add_argument("--rows", required=True, help="parquet written by sum_adapter.py")
    parser.add_argument(
        "--raw-dir",
        default=DEFAULT_RAW_DIR,
        help="downloaded SUM directory; the pair and the audit answer are proved against "
        "train.parquet here",
    )
    parser.add_argument("--seed", type=int, default=0, help="sampling + fold seed")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-support", type=int, default=L3_MIN_SUPPORT)
    parser.add_argument(
        "--per-branch",
        type=int,
        default=0,
        help="cap the L1/L2 coverage per branch; 0 (the default) re-derives every row",
    )
    parser.add_argument("--spot-check", type=int, default=20, help="rows sampled per defect type")
    parser.add_argument(
        "--l3-rows",
        type=int,
        default=L3_DEFAULT_ROWS,
        help="size of the D18-quota-proportional L3 corpus",
    )
    parser.add_argument(
        "--raw-l3-rows",
        type=int,
        default=4000,
        help="members sampled from the raw corpus for the informational A-vs-U reading (0 disables)",
    )
    args = parser.parse_args()

    rows = schema.read_parquet_rows(args.rows)
    if not rows:
        print(f"FAIL  {args.rows} holds no rows")
        raise SystemExit(1)
    print(f"rows    : {args.rows} ({len(rows)} rows)")
    print(f"branches: {dict(collections.Counter(r['extra_info']['branch'] for r in rows))}")
    print(f"types   : {dict(collections.Counter(r['extra_info']['error_type'] for r in rows))}")
    print()

    audit = Audit()
    check_contract(rows, audit)
    source = load_source(args.raw_dir)
    check_source_anchor(rows, audit, source, args.raw_dir)

    sample = sample_by_branch(rows, per_branch=args.per_branch, seed=args.seed)
    audit.note(
        "L1/L2 coverage",
        ", ".join(f"{b}={len(v)}" for b, v in sample.items())
        + (" (capped by --per-branch, NOT exhaustive)" if args.per_branch > 0 else " (every row)"),
    )
    for branch, picked in sample.items():
        total = len([r for r in rows if r["extra_info"]["branch"] == branch])
        if len(picked) < L1_MIN_SAMPLE and len(picked) < total:
            audit.check(f"L1 {branch} sample size", False, f"only {len(picked)} rows, need {L1_MIN_SAMPLE}")
    check_l1(sample, audit)
    check_l2(sample, audit)

    check_l3_options(rows, audit)
    check_l3_option_content(rows, audit)
    check_l3_nb(
        rows,
        audit,
        folds=args.folds,
        seed=args.seed,
        min_support=args.min_support,
        total_rows=args.l3_rows,
    )
    check_l3_raw(
        source,
        audit,
        sample_rows=args.raw_l3_rows,
        seed=args.seed,
        folds=args.folds,
        min_support=args.min_support,
    )
    check_yields(source, audit)
    check_type_spot_check(rows, audit, per_type=args.spot_check, seed=args.seed)

    audit.flush()
    if audit.failures:
        print(f"RESULT: FAIL ({audit.failures} check(s) failed)")
        raise SystemExit(1)
    print("RESULT: PASS (all hard checks passed)")
    raise SystemExit(0)


if __name__ == "__main__":  # pragma: no cover
    main()
