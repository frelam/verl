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
"""Audit the SUM pair rows produced by ``sum_adapter.py`` (design doc sections 4.5, 4.8, 9, 12 Q10).

Usage::

    python verify_sum.py --rows <built sum.parquet> --raw-dir <sum raw dir>

Reads the artifact and, where a claim can only be proved against the source,
``train.parquet`` under ``--raw-dir``.  Every check re-derives its answer from the
row's own prompt and from its raw source row with this file's **own** copy of the
template delimiters and token diff, never by calling the adapter, so an adapter bug
cannot certify itself.  Prints ``PASS``/``FAIL`` per check, ``----`` for
measurements that are reported rather than asserted, and exits non-zero if any hard
check fails.

Layers
------

**L0 -- pair contract** (section 9, "reward 与 adapter 字段契约").  ``schema.validate_row``
on every row, then the pair invariants: ``ground_truth`` is a JSON **string**;
``pair_task=true``; ``answerable_id`` in {``A``, ``B``}; template C; no option
block; ``has_diagnosis_label=false``; ``perturbation_type=null``; a non-empty
``answer``.  Then the two questions are re-parsed out of the prompt with this
file's own copy of the template-C wording and asserted non-empty and distinct:
**both questions must be in the prompt**, and a prompt this file cannot parse is a
FAIL, not a skip.

**A/B balance** (sections 4.5 / 9).  ``answerable_id`` must be 50:50 +-2pt over the
artifact -- a three-sigma statement at the 6,000-row quota, which is where the doc
makes it.  A scratch build is smaller than that, so the bound widens to the same
three sigma when that is larger (a 300-row build cannot resolve +-2pt, and failing it
on sampling noise would say nothing about the adapter); the line prints which bound
applied, and any real bias (a fixed A, a 60/40 skew) fails at every size.

**Quota** (section 4.8 table B row 6).  The artifact holds at most
:data:`SUM_QUOTA` = 6,000 pair rows, and the raw file's usable supply
(well-formed, deduplicated -- recomputed here, not taken from the adapter's report)
must cover the quota.

**Source anchor.**  The row's ``extra_info.index`` re-reads the source row it
claims: the question named by ``answerable_id`` must be byte-identical to that
row's ``answerable_question`` and the other one to its ``unanswerable_question`` --
which is what makes "``answerable_id`` names the answerable member" a proof rather
than a promise (section 9).  ``answer`` must be the raw ``ground_truth`` verbatim,
``paired_original_text`` the unanswerable member, and ``task_id`` the source
position.  SUM ships **no derivation chain**, so this anchor is the only place the
pair and the gold can be proved against something outside the artifact -- a missing
``--raw-dir`` is a FAIL, not a skip.

**L3 -- no shortcut** (section 4.5).  The doc's evidence that "which one is
solvable" is a real task is a bag-of-words Naive Bayes reading at chance
(0.502 / 0.503).  This file re-measures it on the artifact: both members of every
sampled pair, labelled by which one the gold names, out-of-fold balanced accuracy
against the 0.5 chance line (:func:`verify_umwp.bow_nb_oof`, one ruler for every
source) with a shuffled-label control.  Section 9 gives SUM no L3 **gate**, so the
reading is printed as a measurement -- and the caveat it needs: the estimator is
support-gated, so its data reading grows with the corpus (0.555 at the 4,000-member
protocol, 0.599 over the full 12,000-member artifact, control ~0.50) while the doc's
0.502 / 0.503 is what the same ruler returns on shuffled labels.  The audit states
the finding rather than turning a number the doc never gated into a build blocker.
The raw corpus's own A-vs-U reading is printed too.

**Q10 -- unanswerable-label spot check** (section 4.5, section 12 Q10; N=100).
SUM's unanswerable side is o3-mini generated and expert reviewed and the file ships
no machine certificate, so the doc's substitute is a spot check: sample N=100 rows
and check that the unanswerable member really is a deficient rewrite.  With
``--label-file`` the verdicts are human and the check is exactly the doc's; without
one, :func:`label_is_plausible` is a **rule-based proxy** (the rewrite cut a content
word the question no longer carries, or introduced a vague / impossible value or an
unnamed entity, or cut the question clause) and the result is reported as a proxy,
never as a human review.  A failure rate above 10% fails the check -- the doc
downgrades or drops the source above that line.

**Yield recomputation.**  The raw pool is measured with the same word-level
``difflib`` projection the adapter family uses: deletion-only pairs, single-
insertion pairs, multi-region paraphrases, and the rows no word-level rule can see
(operator and punctuation flips).  Reported as ``----``: it explains why a
proxy-based label check has a floor, not an audit failure.
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

try:
    import schema
    import verify_umwp
except ImportError:  # pragma: no cover - running as a plain script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import schema
    import verify_umwp

DEFAULT_RAW_DIR = "/home/charles/data/reasoning_rl/halluc/raw/sum"
DATA_FILE = "train.parquet"

#: Design doc section 4.8 table B row 6: SUM supplies 6,000 pair rows.
SUM_QUOTA = 6_000
#: Section 4.5 / 9: the answerable question must land on A and on B equally often.
AB_TOLERANCE = 0.02
#: The doc's +-2pt is a three-sigma statement at the 6,000-row quota; a smaller
#: build is checked at the same sigma instead of at a bound it cannot resolve.
AB_SIGMA = 3.0
#: The section 4.5 no-shortcut line (random + 5pt; the doc measures 0.502 / 0.503).
L3_MAX_BALANCED_ACCURACY = 0.55
L3_MIN_SUPPORT = 5
#: Pair rows in the fixed-protocol L3 corpus (4,000 members).  The reading is
#: protocol-dependent -- see :func:`check_l3_pair` -- so the corpus is pinned.
L3_DEFAULT_ROWS = 2_000
#: Q10: sample size and the doc's downgrade line.
SPOT_CHECK_N = 100
SPOT_CHECK_MAX_FAILURE = 0.10
#: The anchor check is exhaustive by default; a cap keeps this floor, like the rest
#: of the family, so a capped run can never look exhaustive.
ANCHOR_MIN_SAMPLE = 50

#: The two question labels template C uses, in order.
PROMPT_LABELS = ("A", "B")

#: This file's own copy of the template-C delimiters.  Kept literal (rather than
#: imported from ``schema``) so a change to the rendered template shows up as a
#: parse failure here instead of silently redefining what the audit compares.
_PAIR_PROMPT_RE = re.compile(
    r"^下面给出两个问题，其中一个可以求解，另一个因缺少条件或条件矛盾而无法求解。\n\n"
    r"问题 A：(?P<a>.*?)\n\n"
    r"问题 B：(?P<b>.*?)\n\n"
    r"请先判断哪个问题可解，再解答该问题。最后一行输出：\n"
    r"\\boxed\{<可解问题的编号>: <最终答案>\}\n"
    r"例如 \\boxed\{A: 42\}。$",
    re.DOTALL,
)


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
# independent text plumbing (deliberately separate from the adapter)
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
# A vague quantifier; same vocabulary as the adapter family, written out again here.
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

    The projection the adapter family uses, re-implemented here so the audit's view
    of "what changed between the two members" does not come from the adapter.
    """
    ta = [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(original)]
    tb = [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(rewritten)]
    matcher = difflib.SequenceMatcher(None, [t[0] for t in ta], [t[0] for t in tb], autojunk=False)
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
        regions.append(Region(tag, original[a_start:a_end], rewritten[b_start:b_end], a_start, b_start))
    return regions


def emitting(regions: list[Region]) -> list[Region]:
    """Regions that put text into the rewritten member."""
    return [region for region in regions if region.inserted.strip()]


def question_clause(question: str) -> tuple[str, int]:
    """The final interrogative sentence of ``question`` and its offset.

    ``("", -1)`` when there is no ``?``.  The offset comes from the last sentence
    break before that ``?``: a question can repeat its own clause verbatim, and
    ``str.find`` would report the earlier copy for both.
    """
    mark = question.rfind("?")
    if mark < 0:
        return "", -1
    start = question.rfind(".", 0, mark)
    return question[start + 1 : mark + 1], start + 1


def clause_extent(clause: str, offset: int) -> tuple[int, int]:
    """``(first, last_exclusive)`` character offsets of the clause's own text.

    Whitespace does not count as clause text: the clause as sliced starts with the
    blank run that separated it from the previous sentence.
    """
    stripped = clause.strip()
    if not stripped:
        return offset, offset
    lead = len(clause) - len(clause.lstrip())
    return offset + lead, offset + lead + len(stripped)


def clause_cut(original: str, presented: str, regions: list[Region]) -> bool:
    """Whether the question clause was deleted and is gone from ``presented``.

    Both halves matter: the deleted run must overlap the clause's own text, and the
    clause must no longer occur in the rewritten member (a clause that survives
    leaves the question standing).
    """
    clause, offset = question_clause(original)
    if not clause:
        return False
    stripped = clause.strip()
    if not stripped:
        return False
    first, last = clause_extent(clause, offset)
    overlaps = any(
        region.deleted.strip() and region.a_start < last and first < region.a_start + len(region.deleted)
        for region in regions
    )
    return overlaps and stripped not in presented


def info_gone(deleted: str, presented: str) -> bool:
    """A content word of the deleted text is absent from the rewritten member."""
    presented_tokens = {token.lower() for token in words(presented)}
    return any(schema.is_content_word(token) and token.lower() not in presented_tokens for token in words(deleted))


def introduces_undefined_entity(inserted: str, original: str) -> bool:
    """The residual class's own predicate, written out again for the audit.

    ``irrelevant_undefined_entity`` means "this span names something the original
    problem never carried".
    """
    original_tokens = {token.lower() for token in words(original)}
    return any(schema.is_content_word(token) and token.lower() not in original_tokens for token in words(inserted))


def certifies_as_defect(inserted: str, original: str) -> bool:
    """Whether one of the three visible rules justifies pointing at ``inserted``.

    The two non-residual defect classes are justified by the rule that fires; the
    residual class only by its defining predicate.
    """
    if _IMPOSSIBLE_RE.search(inserted) or _VAGUE_RE.search(inserted):
        return True
    return introduces_undefined_entity(inserted, original)


def label_is_plausible(answerable: str, unanswerable: str) -> str:
    """Rule-based proxy for "the second member is the deficient rewrite" (Q10).

    Returns ``""`` when the pair carries textual evidence of the defect, else the
    reason no rule sees one.  This is the substitute for the human spot check the
    doc asks for, and it is deliberately loose in the direction that *accepts* a
    label: it fires on any of the three shapes the source's rewrites take -- a
    content word the rewrite no longer carries, an inserted vague / impossible
    value or unnamed entity, or a cut question clause.  Pairs whose only change is
    one a word tokenizer cannot see (``y != 0`` -> ``y > 0``, punctuation-only
    edits, and the three rows whose two members are identical) have no textual
    evidence at all and are reported as proxy failures -- measured over the frozen
    file that floor is 1,013 / 36,480 = 2.8%, well inside the doc's 10% line.
    """
    if not answerable.strip() or not unanswerable.strip():
        return "an empty member"
    if answerable == unanswerable:
        return "the two members are identical"
    regions = diff_regions(answerable, unanswerable)
    if not regions:
        return "no word-level difference at all"
    deleted = " ".join(region.deleted for region in regions if region.deleted.strip())
    if deleted and info_gone(deleted, unanswerable):
        return ""
    for region in emitting(regions):
        if certifies_as_defect(region.inserted.strip(), answerable):
            return ""
    if clause_cut(answerable, unanswerable, regions):
        return ""
    return "no rule sees a defect (a paraphrase only the model could judge)"


# ---------------------------------------------------------------------------
# reading the artifact
# ---------------------------------------------------------------------------


def prompt_questions(row: dict) -> tuple[str, str]:
    """``(question A, question B)`` parsed out of the row's own prompt.

    This file's own parser, so "both questions are in the prompt" is checked
    against the rendered text rather than against a claim about it.  A prompt the
    template does not match is an error, not a skip: every text check below would
    otherwise compare the wrong string.
    """
    content = row["prompt"][0]["content"]
    match = _PAIR_PROMPT_RE.match(content)
    if match is None:
        raise ValueError(f"prompt of {row['extra_info'].get('task_id')} is not the template-C pair prompt")
    return match.group("a"), match.group("b")


def payload_of(row: dict) -> dict:
    """The decoded ``ground_truth`` payload, or ``{}`` when it is unreadable.

    ``check_contract`` is the layer that reports a malformed payload; every other
    layer then sees an empty payload and fails on the missing fields instead of
    aborting the whole audit with a traceback.
    """
    try:
        payload = json.loads(row["reward_model"]["ground_truth"])
    except (KeyError, TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def payload_error(row: dict) -> str:
    """``""`` when the row's ``ground_truth`` is a JSON object string, else the reason."""
    raw = (row.get("reward_model") or {}).get("ground_truth")
    if not isinstance(raw, str):
        return "ground_truth must be a JSON string"
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        return f"ground_truth is not valid JSON: {exc}"
    if not isinstance(payload, dict):
        return "ground_truth must decode to an object"
    return ""


def load_source(raw_dir: str) -> list[dict]:
    """``train.parquet`` as a list of rows (empty when unreadable)."""
    path = os.path.join(raw_dir, DATA_FILE)
    if not os.path.exists(path):
        return []
    return schema.read_parquet_rows(path)


def usable_supply(source: list[dict]) -> tuple[int, int, int]:
    """``(usable pairs, conflicting rows, duplicate rows)`` recomputed from the raw file.

    The adapter's own filter, re-derived here with this file's ``normalise``: rows
    whose three texts are nonempty strings survive; a group of identical normalised
    pairs with conflicting golds is dropped entirely (fail closed), a group with one
    distinct gold keeps its lowest index.
    """
    stage = [
        (index, row)
        for index, row in enumerate(source)
        if all(
            isinstance(row.get(key), str) and row[key].strip()
            for key in ("answerable_question", "unanswerable_question", "ground_truth")
        )
    ]
    groups: dict[tuple[str, str], list[tuple[int, str]]] = collections.defaultdict(list)
    for index, row in stage:
        groups[(normalise(row["answerable_question"]), normalise(row["unanswerable_question"]))].append(
            (index, row["ground_truth"].strip())
        )
    conflicting: set[int] = set()
    duplicate: set[int] = set()
    for members in groups.values():
        if len(members) < 2:
            continue
        if len({gold for _, gold in members}) > 1:
            conflicting.update(index for index, _ in members)
            continue
        duplicate.update(index for index, _ in sorted(members)[1:])
    return len(stage) - len(conflicting) - len(duplicate), len(conflicting), len(duplicate)


# ---------------------------------------------------------------------------
# L0 -- pair contract
# ---------------------------------------------------------------------------


def check_contract(rows: list[dict], audit: Audit) -> None:
    """Contract, gold shape and "both questions are in the prompt" (section 9)."""
    violations: list[str] = []
    for row in rows:
        violations.extend(schema.validate_row(row))
    audit.check(
        "L0a schema.validate_row on every row",
        not violations,
        f"{len(rows)} rows, {len(violations)} violations" + (f"; first: {violations[:3]}" if violations else ""),
    )

    contract: list[str] = []
    for row in rows:
        info = row["extra_info"]
        task_id = info["task_id"]
        error = payload_error(row)
        if error:
            contract.append(f"{task_id}: {error}")
            continue
        payload = payload_of(row)
        if payload.get("pair_task") is not True:
            contract.append(f"{task_id}: ground_truth is missing pair_task=true")
        if payload.get("answerable_id") not in PROMPT_LABELS:
            contract.append(f"{task_id}: answerable_id={payload.get('answerable_id')!r} is not A/B")
        if payload.get("solvable") is not True:
            contract.append(f"{task_id}: a pair row is a solvable row, got {payload.get('solvable')!r}")
        answer = payload.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            contract.append(f"{task_id}: the pair row has no answer text")
        if payload.get("has_diagnosis_label") is not False:
            contract.append(f"{task_id}: pair row must not carry has_diagnosis_label")
        if payload.get("perturbation_type") is not None:
            contract.append(f"{task_id}: pair row must not carry a perturbation_type")
        if info.get("template") != schema.TEMPLATE_C:
            contract.append(f"{task_id}: template {info.get('template')!r} is not C")
        if info.get("branch") != schema.BRANCH_SOLVABLE_PAIR:
            contract.append(f"{task_id}: branch {info.get('branch')!r} is not the pair branch")
        if info.get("options"):
            contract.append(f"{task_id}: a pair row must not carry an options block")
        if info.get("has_diagnosis_label"):
            contract.append(f"{task_id}: pair row must not carry a diagnosis label")
    audit.check(
        "L0b pair contract (pair_task / answerable_id / template C / no options)",
        not contract,
        f"{len(rows)} rows, {len(contract)} violations" + (f"; first: {contract[:3]}" if contract else ""),
    )

    prompts: list[str] = []
    for row in rows:
        task_id = row["extra_info"]["task_id"]
        try:
            question_a, question_b = prompt_questions(row)
        except ValueError as exc:
            prompts.append(str(exc))
            continue
        if not question_a.strip() or not question_b.strip():
            prompts.append(f"{task_id}: a prompt question is empty")
        elif question_a == question_b:
            prompts.append(f"{task_id}: the two prompt questions are identical")
    audit.check(
        "L0c both questions present in the prompt, non-empty and distinct",
        not prompts,
        f"{len(rows)} prompts re-parsed, {len(prompts)} violations" + (f"; first: {prompts[:3]}" if prompts else ""),
    )

    ids = [row["extra_info"]["task_id"] for row in rows]
    audit.check(
        "L0d task_id is unique and source-derived",
        len(set(ids)) == len(ids),
        f"{len(rows)} rows, {len(ids) - len(set(ids))} duplicate ids",
    )


def check_ab_balance(rows: list[dict], audit: Audit) -> None:
    """Section 4.5 / 9: the answerable question must land on A and B 50:50 +-2pt.

    The doc's assertion is about the 6,000-row artifact, where +-2pt is a three-sigma
    statement (sigma = 0.5/sqrt(6000) = 0.65pt).  A scratch build is smaller and its
    split simply cannot be resolved to +-2pt by any sample, so the bound widens to
    ``AB_SIGMA`` sigma when that is larger -- the check still catches a biased draw
    (a fixed A, a 60/40 skew) at every size, and the doc's literal +-2pt is what is
    applied to the artifact it is about.  The line says which bound was used.
    """
    counts = collections.Counter(payload_of(row).get("answerable_id") for row in rows)
    total = len(rows)
    share_a = counts.get("A", 0) / total if total else 0.0
    sigma = 0.5 / total**0.5 if total else 1.0
    tolerance = max(AB_TOLERANCE, AB_SIGMA * sigma)
    detail = (
        f"A={counts.get('A', 0)} B={counts.get('B', 0)} of {total} rows "
        f"(A share {share_a:.1%}, tolerance {tolerance:.2%}"
        + (", the doc's +-2pt" if tolerance <= AB_TOLERANCE else f", {AB_SIGMA:g} sigma at this size")
        + ")"
    )
    audit.check(
        "L0e answerable_id is 50:50 +-2pt",
        total > 0 and abs(share_a - 0.5) <= tolerance,
        detail,
    )
    if total and tolerance > AB_TOLERANCE:
        audit.note(
            "L0e A/B balance resolution",
            f"{total} rows: +-2pt is a three-sigma statement at the 6,000-row quota; below it the bound "
            f"widens to {tolerance:.2%} so a scratch build is not failed by sampling noise alone",
        )


def check_quota(rows: list[dict], audit: Audit, source: list[dict]) -> None:
    """Section 4.8 table B row 6: at most 6,000 pair rows, and enough raw supply."""
    audit.check(
        "L0f artifact holds at most the 6,000-pair quota",
        len(rows) <= SUM_QUOTA,
        f"{len(rows)} rows vs quota {SUM_QUOTA} (each row is one pair, not two)",
    )
    if not source:
        audit.note("L0g raw supply", "skipped: the raw file is unreadable")
        return
    usable, conflicting, duplicate = usable_supply(source)
    audit.check(
        "L0g raw supply covers the quota",
        usable >= SUM_QUOTA,
        f"{usable} usable pairs after dropping {conflicting} conflicting-gold rows and "
        f"{duplicate} same-gold duplicates, vs quota {SUM_QUOTA}",
    )


def check_source_anchor(
    rows: list[dict], audit: Audit, source: list[dict], raw_dir: str, *, per_row: int = 0, seed: int = 0
) -> int:
    """Prove every row's texts, answerability and gold against ``train.parquet``.

    Returns the number of rows re-derived (0 when the raw file is unreadable), which is
    what ``main`` prints as the coverage line: a capped run must never read as a full
    one.
    """
    path = os.path.join(raw_dir, DATA_FILE)
    if not source:
        audit.check(
            "L0h source anchor (both questions and the gold re-read from the raw file)",
            False,
            f"cannot read {path} -- SUM ships no derivation chain, so which member is "
            "answerable and what the answer is can only be proved against the source; "
            "re-run with --raw-dir pointing at the downloaded SUM directory",
        )
        return 0

    picked = sample_rows(rows, per_row=per_row, seed=seed)
    failures: list[str] = []
    for row in picked:
        info = row["extra_info"]
        task_id = info["task_id"]
        index = info.get("index")
        if not isinstance(index, int) or not 0 <= index < len(source):
            failures.append(f"{task_id}: index {index!r} is not a source row")
            continue
        src = source[index]
        answerable = normalise(src["answerable_question"])
        unanswerable = normalise(src["unanswerable_question"])
        try:
            question_a, question_b = prompt_questions(row)
        except ValueError as exc:
            failures.append(str(exc))
            continue
        payload = payload_of(row)
        answerable_id = payload.get("answerable_id")
        named, other = (question_a, question_b) if answerable_id == "A" else (question_b, question_a)
        if named != answerable:
            failures.append(f"{task_id}: the question named {answerable_id} is not the source's answerable_question")
            continue
        if other != unanswerable:
            failures.append(f"{task_id}: the other prompt question is not the source's unanswerable_question")
            continue
        if payload.get("answer") != src["ground_truth"]:
            failures.append(f"{task_id}: gold is not the source's ground_truth verbatim")
            continue
        if info.get("paired_original_text") != unanswerable:
            failures.append(f"{task_id}: paired_original_text is not the source's unanswerable member")
            continue
        if task_id != f"sum-pair-{index}":
            failures.append(f"{task_id}: task_id is not the source position {index}")
    detail = f"{len(picked)} rows anchored to {path}, {len(failures)} failures" + (
        f"; first: {failures[:3]}" if failures else ""
    )
    audit.check("L0h source anchor (both questions and the gold re-read from the raw file)", not failures, detail)
    return len(picked)


def sample_rows(rows: list[dict], *, per_row: int, seed: int) -> list[dict]:
    """The rows the anchor re-derives: **every** row unless ``per_row`` caps it.

    Exhaustive by default: a sampled anchor is blind to a single tampered row outside
    the sample, and re-reading a few thousand source rows costs a second.  A positive
    ``per_row`` caps the sample and keeps :data:`ANCHOR_MIN_SAMPLE`, so a capped run
    cannot be mistaken for a full one (``main`` prints the coverage line).
    """
    if per_row <= 0 or len(rows) <= per_row:
        return sorted(rows, key=lambda row: row["extra_info"]["task_id"])
    take = min(len(rows), max(per_row, ANCHOR_MIN_SAMPLE))
    picked = random.Random(seed).sample(rows, take)
    return sorted(picked, key=lambda row: row["extra_info"]["task_id"])


# ---------------------------------------------------------------------------
# L3 -- the section 4.5 no-shortcut reading
# ---------------------------------------------------------------------------


def pair_corpus(rows: list[dict], *, total_rows: int, seed: int) -> tuple[list[str], list[int]]:
    """Both members of every sampled pair, labelled by which one the gold names.

    This is the judgement layer of the pair task exactly as the model sees it: one
    text per question, ``1`` for the member ``answerable_id`` points at and ``0`` for
    the other.  A bag-of-words model that separates them is a shortcut the task does
    not want, and section 4.5's evidence that it does not is precisely this reading.
    """
    take = len(rows) if total_rows <= 0 else min(len(rows), total_rows)
    picked = random.Random(seed).sample(rows, take) if take < len(rows) else list(rows)
    texts: list[str] = []
    labels: list[int] = []
    for row in picked:
        question_a, question_b = prompt_questions(row)
        answerable_id = payload_of(row).get("answerable_id")
        texts.append(question_a)
        labels.append(1 if answerable_id == "A" else 0)
        texts.append(question_b)
        labels.append(1 if answerable_id == "B" else 0)
    return texts, labels


def _balanced_accuracy(
    texts: list[str], labels: list[int], *, folds: int, seed: int, min_support: int
) -> tuple[float, float]:
    """``(data reading, shuffled-label control)`` -- the control calibrates the ruler."""
    control_rng = random.Random(seed + 1)
    control_labels = [control_rng.random() > 0.5 for _ in texts]
    accuracy = verify_umwp.bow_nb_oof(texts, labels, folds=folds, seed=seed, min_support=min_support)
    control = verify_umwp.bow_nb_oof(texts, control_labels, folds=folds, seed=seed, min_support=min_support)
    return accuracy, control


def check_l3_pair(rows: list[dict], audit: Audit, *, folds: int, seed: int, min_support: int, total_rows: int) -> None:
    """Section 4.5's no-shortcut reading, reported with its control (not gated).

    Section 9 gives SUM no L3 threshold -- its three checks are the non-empty
    questions, the A/B split and the Q10 label spot check -- so the reading is
    printed as a measurement with the control that calibrates the ruler.  (Every pair
    contributes one member of each label by construction, so an artifact that passed
    the contract cannot make the estimator unidentifiable; the A/B balance check is
    what would catch a degenerate draw.)

    The caveat the numbers demand: this estimator is support-gated, so its data
    reading grows with the corpus while the shuffled-label control stays at chance.
    Measured on the frozen file: 0.555 at the fixed protocol here (4,000 members,
    ``min_support=5``), 0.599 over the full 12,000-member artifact (controls
    0.50 / 0.49), and across protocols the reading spans 0.46 (``min_support=1``) to
    0.63 (``min_support=20``).  The doc's 0.502 / 0.503 sits inside that band, so it
    is one point of a protocol-dependent curve rather than a property of the pairs:
    at the artifact's own size the judgement layer does carry a weak surface signal.
    That is a finding about the source for the build report, and the audit states it
    instead of hiding it behind a gate the doc never set.
    """
    texts, labels = pair_corpus(rows, total_rows=total_rows, seed=seed)
    positives = sum(labels)
    audit.note(
        "L3a corpus",
        f"{len(texts)} pair members from {len(texts) // 2} of {len(rows)} artifact rows "
        f"(answerable={positives}, unanswerable={len(texts) - positives}); folds={folds}, "
        f"vocabulary support >= {min_support} docs, alpha=1.0",
    )
    if not texts:
        audit.note("L3a corpus", "empty artifact: there is nothing to measure")
        return
    accuracy, control = _balanced_accuracy(texts, labels, folds=folds, seed=seed, min_support=min_support)
    audit.note(
        "L3b data reading (fixed protocol)",
        f"{accuracy:.4f} on {len(texts)} members (chance 0.5; doc claims 0.502 / 0.503)",
    )
    audit.note("L3b shuffled-label control", f"{control:.4f}")
    if len(rows) > total_rows:
        full_texts, full_labels = pair_corpus(rows, total_rows=0, seed=seed)
        full_accuracy, full_control = _balanced_accuracy(
            full_texts, full_labels, folds=folds, seed=seed, min_support=min_support
        )
        audit.note(
            "L3b full-artifact reading (informational, larger corpus)",
            f"{full_accuracy:.4f} on {len(full_texts)} members (control {full_control:.4f}) -- "
            "not comparable with the line above: every extra document admits more support-gated "
            "vocabulary, so this number grows with n",
        )
        accuracy, control = full_accuracy, full_control
    line = L3_MAX_BALANCED_ACCURACY
    verdict = (
        "at or below random + 5pt"
        if accuracy <= line
        else f"ABOVE random + 5pt ({line:.2f}); still below the Q8 default line of random + 10pt (0.60)"
        if accuracy <= 0.60
        else "ABOVE the Q8 default line of random + 10pt (0.60)"
    )
    audit.note(
        "L3b finding",
        f"balanced accuracy {accuracy:.4f} vs chance 0.5 -> {verdict}; the control ({control:.4f}) is "
        "at chance, so the ruler is calibrated and the deviation is a property of the question pairs -- "
        "weak surface separability of the answerable member, which section 4.5's 0.502 / 0.503 does not "
        "show at this corpus size and support gate",
    )


def check_l3_raw(
    source: list[dict], audit: Audit, *, sample_rows_n: int, seed: int, folds: int, min_support: int
) -> None:
    """Report the raw A-vs-U separability the doc's 0.502 claims to measure.

    Informational: the raw pairs are the judgement task's population, so this is the
    reading the doc's own number is about; it is reported with its sample size and a
    shuffled-label control rather than asserted, because the estimator's reading is a
    function of the corpus size (see :func:`check_l3_pair`).
    """
    if not source or sample_rows_n <= 0:
        return
    take = min(len(source), max(sample_rows_n // 2, 1))
    picked = random.Random(seed).sample(source, take)
    texts = [normalise(row["answerable_question"]) for row in picked]
    texts += [normalise(row["unanswerable_question"]) for row in picked]
    labels = [1] * take + [0] * take
    accuracy, control = _balanced_accuracy(texts, labels, folds=folds, seed=seed, min_support=min_support)
    audit.note(
        "L3c raw-corpus reading (informational)",
        f"answerable vs unanswerable over a balanced {2 * take}-member sample: {accuracy:.4f} "
        f"(shuffled-label control {control:.4f}); the doc reports 0.502 / 0.503 for this reading",
    )


# ---------------------------------------------------------------------------
# Q10 -- the unanswerable-label spot check
# ---------------------------------------------------------------------------


def check_spot_check(rows: list[dict], audit: Audit, *, n: int, seed: int, label_file: str | None = None) -> None:
    """Sample N rows and check the unanswerable label (section 4.5, section 12 Q10).

    ``--label-file`` carries the doc's human verdicts -- either a JSON object
    ``{task_id: false}`` for the rows judged wrong, or a JSON list of wrong
    task_ids.  A sampled row the file does not cover fails the check: a partial
    review has not cleared the source, so it must not read as a pass.  Without the
    file the rule-based proxy :func:`label_is_plausible` runs and the line says so.
    Either way a failure rate above 10% fails the check, the doc's downgrade line.
    """
    take = min(len(rows), max(n, 0))
    if take <= 0:
        audit.note("L4 Q10 spot check", "skipped: the artifact holds no rows")
        return
    picked = random.Random(seed).sample(rows, take)
    task_ids = [row["extra_info"]["task_id"] for row in picked]
    uncovered: list[str] = []
    if label_file:
        verdicts = _load_label_file(label_file)
        uncovered = [task_id for task_id in task_ids if task_id not in verdicts]
        failures = [task_id for task_id in task_ids if verdicts.get(task_id) is False]
        source_note = f"human verdicts from {label_file}"
    else:
        failures = []
        reasons: collections.Counter = collections.Counter()
        for row in picked:
            question_a, question_b = prompt_questions(row)
            answerable = question_a if payload_of(row).get("answerable_id") == "A" else question_b
            unanswerable = question_b if payload_of(row).get("answerable_id") == "A" else question_a
            reason = label_is_plausible(answerable, unanswerable)
            if reason:
                failures.append(row["extra_info"]["task_id"])
                reasons[reason] += 1
        source_note = "rule-based proxy, NOT a human review" + (f" (reasons: {dict(reasons)})" if reasons else "")
    rate = len(failures) / take
    if take < n:
        audit.note(
            "L4 Q10 spot check size",
            f"only {take} rows could be drawn (N={n} requested) -- the full-sample run is the audit for a full build",
        )
    audit.check(
        f"L4 Q10 unanswerable-label spot check (N={take}) failure rate <= 10%",
        not uncovered and rate <= SPOT_CHECK_MAX_FAILURE,
        f"{len(failures)}/{take} = {rate:.1%} fail ({source_note})"
        + (f"; first: {failures[:3]}" if failures else "")
        + (
            f"; {len(uncovered)} sampled rows have no verdict -- an incomplete review cannot clear "
            f"the source (first: {uncovered[:3]})"
            if uncovered
            else ""
        ),
    )


def _load_label_file(path: str) -> dict[str, bool]:
    """Read a Q10 verdict file: ``{task_id: bool}`` or a list of wrong task_ids."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return {str(task_id): False for task_id in data}
    if isinstance(data, dict):
        return {str(task_id): bool(verdict) for task_id, verdict in data.items()}
    raise ValueError(f"{path}: expected a JSON object or list of task_ids")


# ---------------------------------------------------------------------------
# yield recomputation
# ---------------------------------------------------------------------------


def measure_pools(source: list[dict]) -> dict[str, int]:
    """The pool projection: word-level opcodes over the raw pairs."""
    counts = {
        "pairs": 0,
        "no_word_change": 0,
        "three_tier_pool": 0,  # pure deletion: nothing inserted, so nothing to point at
        "four_tier_pool": 0,  # exactly one inserted region: the change is visible text
        "multi_region": 0,
    }
    for row in source:
        counts["pairs"] += 1
        regions = diff_regions(normalise(row["answerable_question"]), normalise(row["unanswerable_question"]))
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
    """Report the raw pool shape (informational; it explains the proxy's floor).

    The pair contract consumes every usable row regardless of the diff's shape, so
    this is no longer a yield projection for a branch -- it is the measurement of how
    many pairs carry a change a word-level rule can name, which is the ceiling of the
    Q10 proxy above.
    """
    if not source:
        audit.note("yield recomputation", "skipped: the raw file is unreadable")
        return
    counts = measure_pools(source)
    nameable = counts["three_tier_pool"] + counts["four_tier_pool"]
    audit.note(
        "yield recomputation (opcode projection over the raw file)",
        f"{counts['pairs']} pairs: deletion-only {counts['three_tier_pool']}, "
        f"single-insertion {counts['four_tier_pool']}, multi-region {counts['multi_region']}, "
        f"no word change {counts['no_word_change']} -- a word-level rule can name a change in "
        f"{nameable} of them ({nameable / counts['pairs']:.1%})",
    )
    usable, conflicting, duplicate = usable_supply(source)
    audit.note(
        "usable supply",
        f"{usable} pairs after the well-formedness filter and dedup "
        f"({conflicting} conflicting-gold rows, {duplicate} same-gold duplicates); "
        f"the {SUM_QUOTA}-pair quota is {SUM_QUOTA / usable:.0%} of it",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the SUM pair rows built by sum_adapter.py.")
    parser.add_argument("--rows", required=True, help="parquet written by sum_adapter.py")
    parser.add_argument(
        "--raw-dir",
        default=DEFAULT_RAW_DIR,
        help="downloaded SUM directory; the pair, the answerable side and the gold are proved "
        "against train.parquet here",
    )
    parser.add_argument("--seed", type=int, default=0, help="sampling + fold seed")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-support", type=int, default=L3_MIN_SUPPORT)
    parser.add_argument(
        "--per-row",
        "--per-branch",
        dest="per_row",
        type=int,
        default=0,
        help="cap the source-anchor coverage; 0 (the default) re-reads the raw row of every row",
    )
    parser.add_argument(
        "--spot-check",
        type=int,
        default=SPOT_CHECK_N,
        help="rows sampled for the section 12 Q10 unanswerable-label check",
    )
    parser.add_argument(
        "--label-file",
        default=None,
        help="JSON human verdicts for the Q10 check ({task_id: false} for the rows judged wrong, "
        "or a list of wrong task_ids); every sampled row must carry a verdict, and without the "
        "file the check runs its rule-based proxy",
    )
    parser.add_argument(
        "--l3-rows",
        type=int,
        default=L3_DEFAULT_ROWS,
        help="pair rows in the fixed-protocol L3 corpus (0 = the whole artifact)",
    )
    parser.add_argument(
        "--raw-l3-rows",
        type=int,
        default=4_000,
        help="members sampled from the raw corpus for the informational A-vs-U reading (0 disables)",
    )
    args = parser.parse_args()

    rows = schema.read_parquet_rows(args.rows)
    if not rows:
        print(f"FAIL  {args.rows} holds no rows")
        raise SystemExit(1)
    print(f"rows    : {args.rows} ({len(rows)} pair rows)")
    print(f"answers : {dict(collections.Counter(payload_of(r).get('answerable_id') for r in rows))}")
    print()

    audit = Audit()
    check_contract(rows, audit)
    check_ab_balance(rows, audit)
    source = load_source(args.raw_dir)
    check_quota(rows, audit, source)
    checked = check_source_anchor(rows, audit, source, args.raw_dir, per_row=args.per_row, seed=args.seed)
    audit.note(
        "source-anchor coverage",
        f"{checked}/{len(rows)} rows"
        + (" (capped by --per-row, NOT exhaustive)" if checked < len(rows) else " (every row)"),
    )
    check_l3_pair(
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
        sample_rows_n=args.raw_l3_rows,
        seed=args.seed,
        folds=args.folds,
        min_support=args.min_support,
    )
    check_spot_check(rows, audit, n=args.spot_check, seed=args.seed, label_file=args.label_file)
    check_yields(source, audit)

    audit.flush()
    if audit.failures:
        print(f"RESULT: FAIL ({audit.failures} check(s) failed)")
        raise SystemExit(1)
    print("RESULT: PASS (all hard checks passed)")
    raise SystemExit(0)


if __name__ == "__main__":  # pragma: no cover
    main()
