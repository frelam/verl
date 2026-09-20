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
"""Audit the FalseQA rows produced by ``falseqa_adapter.py``.

Usage::

    python verify_falseqa.py --rows /tmp/falseqa.parquet \\
        --raw-dir /home/charles/data/reasoning_rl/halluc/raw/falseqa

Reads the artifact and re-derives every certificate **from the row's own two
texts** (``prompt`` question + ``extra_info.paired_original_text``) plus the raw
CSV; it never calls the adapter and never trusts the adapter's verdict.  Prints
``PASS``/``FAIL`` per check, unlabelled informational lines for the measurements
that are *not* gates, and exits non-zero if any check fails.

Layers
------

**L1 -- index alignment and the defect certificate** (**every** row by default; a
positive ``--per-branch`` subsamples, which is a fast smoke mode and explicitly not
a certificate -- a gold flipped on a row outside the sample is invisible to a
sampled audit, so the default audits all rows).  The raw CSV is re-read and the
source's blocked layout is explicitly undone: for a row recorded as
``(split, side, k)``, the presented question must be row ``k`` of the source's
``label`` block for that side, and ``paired_original_text`` must be the *same*
index ``k`` of the *other* block.  That is the "row k of label=1 is a rewrite of
row k of label=0" claim, proved against the file rather than restated.  The
``(split, k)`` pair is read out of ``task_id`` rather than ``extra_info``, so the
audit still anchors after ``mix_halluc`` has re-indexed the rows.

The defect is then re-derived by diffing the two texts word-by-word
(``autojunk=False``): there must be exactly one changed region visible on the fake
side and the recorded deleted/inserted pair must replay one question into the
other as a token multiset.  The audit does not import the adapter's diff -- it
re-runs it on the artifact.  On ``unsolvable_diag`` rows the region also has to be
the recorded **gold replacement pair**: the gold option's left item equals the
region's fake-side text and its right item the region's real-side text, with the
left item verbatim in the question, carrying a content word, locatable by its
first token, and the right item non-empty and absent from the question.

**L2 -- the pair certificate.**  ``unsolvable_diag``: the gold pair is the
*recorded replacement between the twins* -- the left item occurs in the presented
question but not in the paired real question, and the right item occurs in the
paired real question (it is the fragment that repairs the premise).
``solvable_two_layer``: the row is a
placeholder row, so the check is that it carries no correct option and no
diagnosis label, while its gold is the source's own answer.

**L3 -- replacement-pair structure** (all rows, hard).  Every option must be a
``left -> right`` pair; all left items of a row must be identical and verbatim
spans of the row's own question; the right items must be pairwise distinct, all
of the same word count and the same surface type, and none of them may occur in
the question (section 4.3: all three right items out-of-passage is what makes the
"pick the right item not in the passage" heuristic undefined).  Both branches must
offer the same option count (D15/D18).

**L4 -- heuristic audit** (diagnosis rows, hard: each within the random baseline
plus 10 points).  "Pick the pair whose right item is absent from the question"
must be **0** -- the D21 rule makes it undefined, not merely weak -- and the
longest-option, unique-capital and unique-digit rules must stay inside
``1/k + 0.10``.  The longest-option rule is measured on the whole option text
(whose only varying part is the right item).

**L5 -- mirror control.**  The same "out-of-passage right item" rule that finds
nothing in this artifact is re-run on a *synthetic* block built the way D21 bans
(both distractor right items taken from the presented question): it then
identifies the gold every time.  The control is the evidence that the
out-of-passage rule is load-bearing rather than decorative.

**Pair atomicity.**  Every row's ``extra_info.pair_id`` must be
``"{source split}:{index}"`` from its own ``task_id``, and no two rows sharing a
``pair_id`` may land on opposite sides of the train/val boundary (D27).  The mixer
enforces this through ``mix_halluc.enforce_pair_atomicity``; the audit re-derives
it from the artifact.

**Informational (not gates).**  The L1 corroboration flag (what share of the
dataset's own rebuttals name the gold fragment -- the sample doc's human-review
hook), the H6 bag-of-words Naive Bayes reading of the label task, the gold-position
distribution the section 10 monitoring expects to be ~1/k, the per-branch option
character-length spread the D18 isomorphism depends on, the distractor-quality
spot-check sample (N=50, section 9 / Q9) and the corpus-frequency cue the
distractor pool leaves behind (printed rather than hidden: it is a property of the
source's vocabulary, not of the option block, and section 9's gate list does not
include it).

The H6 numbers are printed under **both** foldings, because the doc's two H6
columns are not reproducible together: its AUC row (0.172 train) is an ungrouped
reading at a 1-document vocabulary, where the model memorises each test question's
near-identical twin (which carries the opposite label) and the out-of-fold score
goes anti-correlated; its balanced-accuracy column (0.765/0.569/0.757) is not
reproduced under any folding or vocabulary threshold tried.  FalseQA's two members
are near-duplicates with opposite labels, so only the pair-aware folding measures
the topic-memory baseline the doc is describing -- and it puts that baseline at
0.55-0.59, not 0.765.
"""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import difflib
import json
import os
import random
import statistics
import sys

import numpy as np

try:
    import schema
    from verify_umwp import Reporter, bow_nb_oof
except ImportError:  # pragma: no cover - running as a plain script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import schema
    from verify_umwp import Reporter, bow_nb_oof

# Re-declared here rather than imported from the adapter: a check that reads its
# expectations out of the code under test is not a check.  K_OPTIONS is also
# measured off the artifact below, and the budget is derived from the measured k.
K_OPTIONS = 3
MAX_CHAR_RATIO = 2.0
#: The D21 replacement-pair arrow (``man -> women``).  Re-declared, for the same
#: reason as ``K_OPTIONS``.
PAIR_ARROW = " -> "
#: The two verdicts template A offers (design doc section 5.2, the wording every
#: option-block row shares).  The *marker* strings are what the reward parses, so
#: the audit checks that the prompt offers both bullets instead of trusting the
#: option block to imply them.  ``\boxed{SOLVABLE}`` is deliberately absent: the
#: reward accepts it on the ``solvable_answer`` branch, but template A never asks
#: for it, and a prompt that offered it would be a different scaffold.
MARKER_ANSWER = "\\boxed{<答案>}"
MARKER_UNSOLVABLE = "\\boxed{UNSOLVABLE"
MARKER_UNSOLVABLE_ID = "\\boxed{UNSOLVABLE: <选项ID>}"
MARKER_VERDICTS = (MARKER_ANSWER, MARKER_UNSOLVABLE_ID)
#: The whole wording of template A that is *not* the row's own text: the verdict
#: instruction between the question and the block, and the header that opens the
#: block.
OPTION_HEADER = "选项："
#: The D18 defect slot a diagnosis row fills and the section 4.3 perturbation class
#: it must carry.  These are the keys the stage-2 balance table groups on, so a row
#: whose metadata says otherwise is a bookkeeping corruption the artifact audit --
#: not only the adapter's unit test -- has to catch.
ERROR_TYPE_POINTABLE = "false_premise_pointable"
PERTURBATION_DIAG = "contradictory_condition"
LABEL_FALSE = "1"
LABEL_TRUE = "0"

BRANCH_DIAG = schema.BRANCH_UNSOLVABLE_DIAG
BRANCH_ANSWERABLE = schema.BRANCH_SOLVABLE_TWO_LAYER

L1_MIN_SAMPLE = 50
L1_CORROBORATION_SAMPLE = 50  # the design doc's N=50 human-review sample
DISTRACTOR_SAMPLE = 50  # section 9: "干扰右项也对" manual sample
L3_MARGIN = 0.10
H6_FOLDS = 5
H6_MIN_SUPPORT = 5
#: The vocabulary support at which the doc's H6 *AUC* row (0.172 train) reproduces:
#: a 1-document vocabulary memorises each test question's near-identical twin.
H6_DOC_AUC_SUPPORT = 1

#: The design doc's H6 reading (section 4.3 / section 9), all-words / function-only
#: / content-only, over the full split.  Printed next to the measurement; the
#: interpretation baseline is not a gate.
H6_DOC = {"all": 0.765, "function": 0.569, "content": 0.757}

#: Suffix classes for the independent re-derivation of the adapter's surface
#: "same type" rule (deviation 5 of the adapter).  Same table, separate copy.
_SUFFIX_CLASSES = (
    ("ing", 5),
    ("ed", 4),
    ("ly", 4),
    ("tion", 6),
    ("sion", 6),
    ("ness", 6),
    ("ity", 5),
    ("ment", 6),
    ("ance", 6),
    ("ence", 6),
    ("ous", 5),
    ("ive", 5),
    ("able", 6),
    ("ible", 6),
    ("ful", 5),
    ("less", 6),
    ("ist", 5),
    ("ism", 5),
    ("er", 5),
    ("or", 5),
    ("s", 4),
)


def _suffix_class(token: str) -> str:
    folded = token.casefold()
    for suffix, minimum in _SUFFIX_CLASSES:
        if len(folded) >= minimum and folded.endswith(suffix):
            return suffix
    return "plain"


def item_signature(text: str) -> tuple[int, bool, str, str]:
    """The audit's own copy of the adapter's surface type signature."""
    tokens = _tokens(text)
    caps = [token[0].isupper() for token in tokens if token]
    if not caps:
        capitalisation = "lower"
    elif all(caps):
        capitalisation = "title"
    else:
        capitalisation = "mixed" if any(caps) else "lower"
    return (
        len(tokens),
        any(ch.isdigit() for ch in text),
        capitalisation,
        _suffix_class(tokens[-1]) if tokens else "plain",
    )


# ---------------------------------------------------------------------------
# reading the artifact
# ---------------------------------------------------------------------------


def question_of(row: dict) -> str:
    """The question the model sees: the prompt is ``question + "\\n\\n" + template``.

    Questions are whitespace-normalised by the adapter, so the first blank line is
    the template boundary.  A prompt without one means the adapter changed the
    rendering and every text check below would silently read the wrong string, so
    it is a hard error.
    """
    content = row["prompt"][0]["content"]
    head, sep, _ = content.partition("\n\n")
    if not sep:
        raise ValueError(f"prompt of {row['extra_info'].get('task_id')} has no template separator")
    return head


def _normalise(text: str) -> str:
    return " ".join((text or "").split())


def _tokens(text: str) -> list[str]:
    return schema.words(text)


def _orient(row: dict) -> tuple[str, str]:
    """``(real question, fake question)`` for this row.

    The diff is always oriented ``real -> fake`` regardless of which side the row
    presents: the answerable row presents the real question and points at the fake
    one through ``paired_original_text``, the diagnosis row the other way round.
    """
    question = question_of(row)
    partner = row["extra_info"].get("paired_original_text", "")
    if row["extra_info"].get("solvable"):
        return question, partner
    return partner, question


def _option_text(row: dict, option_id: str | None) -> str | None:
    for option in row["extra_info"].get("options") or []:
        if option.get("id") == option_id:
            return option.get("text")
    return None


def split_option(text: str) -> tuple[str, str] | None:
    """``"men -> women"`` -> ``("men", "women")``, or ``None`` when malformed.

    A replacement pair is exactly one arrow between two non-empty items.  An
    option without an arrow, with two arrows, or with an empty side is a violation
    of the D21 contract rather than a pair the audit should try to read.
    """
    if not isinstance(text, str) or text.count(PAIR_ARROW) != 1:
        return None
    left, _, right = text.partition(PAIR_ARROW)
    left, right = left.strip(), right.strip()
    if not left or not right:
        return None
    return left, right


def _diag_rows(rows: list[dict]) -> list[dict]:
    """The diagnosis rows -- the only ones that carry a pointer gold."""
    return [
        row
        for row in rows
        if row["extra_info"]["branch"] == BRANCH_DIAG
        and row["extra_info"].get("correct_option_id")
    ]


def _answerable_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row["extra_info"]["branch"] == BRANCH_ANSWERABLE]


# ---------------------------------------------------------------------------
# the diff and the pair identity, re-derived from the artifact
# ---------------------------------------------------------------------------


def _offset_tokens(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in schema._WORD_RE.finditer(text)]


def rederive_regions(real_q: str, fake_q: str) -> list[dict]:
    """The single-region certificate, recomputed from the two question texts.

    Word-level ``difflib`` with ``autojunk=False``; opcodes whose fake-side token
    range is empty are dropped *before* the merge, because a pure deletion is not
    visible in the presented question and cannot be the item the model picks.
    Deliberately a standalone copy of the recipe: the audit must not be able to
    pass by calling the code it is auditing.
    """
    real_tokens, fake_tokens = _offset_tokens(real_q), _offset_tokens(fake_q)
    matcher = difflib.SequenceMatcher(
        None, [t[0] for t in real_tokens], [t[0] for t in fake_tokens], autojunk=False
    )
    runs: list[list[int]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal" or j2 <= j1:
            continue
        if runs and runs[-1][1] == i1 and runs[-1][3] == j1:
            runs[-1][1], runs[-1][3] = i2, j2
        else:
            runs.append([i1, i2, j1, j2])

    regions: list[dict] = []
    for i1, i2, j1, j2 in runs:
        a_start = real_tokens[i1][1] if i2 > i1 else real_tokens[i1 - 1][2] if i1 else 0
        regions.append(
            {
                "real_text": real_q[a_start : real_tokens[i2 - 1][2]] if i2 > i1 else "",
                "fake_text": fake_q[fake_tokens[j1][1] : fake_tokens[j2 - 1][2]],
                "n_real": i2 - i1,
                "n_fake": j2 - j1,
            }
        )
    return regions


def parse_task_id(task_id: str) -> tuple[str, str, int] | None:
    """``falseqa-{side}-{split}-{index}`` -> ``(side, split, index)``, or ``None``.

    The audit reads ``(split, index)`` out of the id rather than out of
    ``extra_info``: ``mix_halluc`` re-indexes its rows and marks val rows with
    ``split="val"``, so ``extra_info.index``/``split`` no longer name the source
    row once the artifact has been mixed.
    """
    if not isinstance(task_id, str):
        return None
    parts = task_id.split("-")
    if len(parts) != 4 or parts[0] != "falseqa" or parts[1] not in ("fake", "real"):
        return None
    try:
        return parts[1], parts[2], int(parts[3])
    except ValueError:
        return None


def source_split_of(row: dict) -> str:
    """The source CSV this row came from, or the artifact's own ``split`` value."""
    parsed = parse_task_id(row["extra_info"].get("task_id", ""))
    return parsed[1] if parsed else str(row["extra_info"].get("split", ""))


def source_index_of(row: dict) -> int | None:
    """The pair index ``k`` this row was built from, or ``None``."""
    parsed = parse_task_id(row["extra_info"].get("task_id", ""))
    return parsed[2] if parsed else None


def contains_token_window(text: str, item: str) -> bool:
    """Whether ``item``'s token sequence occurs as a contiguous window of ``text``.

    Token-level, not substring-level: the gold left item ``men`` must not count as
    "occurring" in the paired real question ``... women ...`` just because the
    characters are a substring of a longer word.  The *passage* test of L1/L3 stays
    a substring test on purpose -- it re-derives the adapter's conservative gate
    (the mirror heuristic a solver could run is a string match) -- while this
    helper answers the semantic question "did this fragment survive the rewrite?".
    """
    haystack = [token.casefold() for token in _tokens(text)]
    needle = [token.casefold() for token in _tokens(item)]
    if not needle or len(needle) > len(haystack):
        return False
    return any(
        haystack[start : start + len(needle)] == needle
        for start in range(len(haystack) - len(needle) + 1)
    )


def _multiset_after_edit(real_q: str, fake_q: str, deleted: str, inserted: str) -> bool:
    """``tokens(fake)`` must equal ``tokens(real) - deleted + inserted`` as a multiset.

    Proves the recorded defect pair really is the edit between the two questions,
    without trusting the adapter's region extraction: the record has to reproduce
    the text.  It is the full defect (invisible deletions included), which is why
    it is checked as a multiset and not against the gold.
    """
    expected = collections.Counter(token.casefold() for token in _tokens(real_q))
    expected.subtract(collections.Counter(token.casefold() for token in _tokens(deleted)))
    expected.update(collections.Counter(token.casefold() for token in _tokens(inserted)))
    return +expected == collections.Counter(token.casefold() for token in _tokens(fake_q))


# ---------------------------------------------------------------------------
# the source file
# ---------------------------------------------------------------------------


#: The source's three CSVs.  Checked rather than trusted because the split is not
#: only a CLI choice: it is read out of the artifact under audit, and this file's
#: whole purpose is to assume the artifact is hostile.  An unvalidated join would
#: turn it into a path component, so a crafted artifact carrying
#: ``"../../elsewhere/secret"`` would have the audit read an arbitrary file as CSV.
SPLITS = ("train", "valid", "test")
#: ``extra_info.split`` values the artifact may carry: the source's own three
#: splits (an adapter build) plus ``val`` (a mixed artifact).
ARTIFACT_SPLITS = SPLITS + ("val",)


def load_source_split(raw_dir: str, split: str) -> list[dict]:
    """Read one of the source's CSVs (never cached across splits on purpose)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    path = os.path.join(raw_dir, f"{split}.csv")
    with open(path, encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def source_blocks(raw_dir: str, split: str) -> dict[str, list[dict]]:
    """``{"1": fake block, "0": real block}`` in file order."""
    rows = load_source_split(raw_dir, split)
    return {
        LABEL_FALSE: [r for r in rows if r.get("label") == LABEL_FALSE],
        LABEL_TRUE: [r for r in rows if r.get("label") == LABEL_TRUE],
    }


def source_answer(row: dict) -> str:
    """The source's own answer text, parsing ``test``'s list-repr form.

    ``test``'s ``label=1`` answers are the repr of a 3-element Python list, so the
    list form is parsed and the first element taken -- the sample doc requires it.
    """
    text = (row.get("answer") or "").strip()
    if text.startswith("["):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text
        if isinstance(parsed, list | tuple) and parsed:
            return str(parsed[0]).strip()
    return text


def source_corpus(raw_dir: str, splits: list[str]) -> list[str]:
    """Every question of the named source splits, whitespace-normalised."""
    questions: list[str] = []
    for split in splits:
        for row in load_source_split(raw_dir, split):
            question = _normalise(row.get("question", ""))
            if question:
                questions.append(question)
    return questions


def item_document_frequency(questions: list[str], max_tokens: int = 10) -> collections.Counter:
    """How many *distinct* questions of the corpus contain each 1..n-token window.

    Used for the informational distractor-frequency reading: a model's word prior
    is a real cue on any option block built from a corpus, and this measures how
    much of the gold's identity it alone would recover.
    """
    frequency: collections.Counter = collections.Counter()
    for question in questions:
        folded = question.casefold()
        seen: set[str] = set()
        for size in range(1, max_tokens + 1):
            for start in range(len(_tokens(folded)) - size + 1):
                window = " ".join(_tokens(folded)[start : start + size])
                if window:
                    seen.add(window)
        frequency.update(seen)
    return frequency


def check_split_vocabulary(rows: list[dict], reporter: Reporter) -> bool:
    """Every row's source split is one of the source's CSVs.  Returns whether it is.

    A gate rather than a note: the source split is read out of the artifact and is
    used to build a file path, so a value the source cannot have is a boundary
    violation, not a formatting quirk.  The second half checks the *artifact*
    split vocabulary (source splits plus the mixer's ``val``).
    """
    unparsable = [
        row["extra_info"].get("task_id")
        for row in rows
        if parse_task_id(row["extra_info"].get("task_id", "")) is None
    ]
    seen = sorted({source_split_of(row) for row in rows}, key=str)
    unknown = [split for split in seen if split not in SPLITS]
    ok = reporter.check(
        f"contract: task_id encodes a source split in {SPLITS}",
        not unknown and not unparsable,
        f"seen {seen}, {len(unparsable)} unparsable task_id(s)"
        if (unknown or unparsable)
        else f"seen {seen} over {len(rows)} rows",
    )
    artifact_seen = sorted({str(row["extra_info"].get("split")) for row in rows})
    artifact_unknown = [split for split in artifact_seen if split not in ARTIFACT_SPLITS]
    ok = reporter.check(
        f"contract: extra_info.split is one of {ARTIFACT_SPLITS}",
        not artifact_unknown,
        f"seen {artifact_seen}" if artifact_seen else "no rows",
    ) and ok
    return ok


def check_source_anchor(rows: list[dict], reporter: Reporter, raw_dir: str) -> None:
    """Prove every row against the raw CSV: question, partner and answer.

    A missing or unreadable raw directory is a FAIL, not a skip: without the file
    the index alignment is unprovable and the L1 corroboration flag has no input.
    """
    cache: dict[str, dict[str, list[dict]]] = {}
    try:
        for row in rows:
            split = source_split_of(row)
            if split not in cache:
                cache[split] = source_blocks(raw_dir, split)
    except ValueError as exc:  # a split the source does not have -- artifact-controlled
        reporter.check("source anchor: raw CSV readable", False, str(exc))
        return
    except (OSError, csv.Error) as exc:
        reporter.check("source anchor: raw CSV readable", False, f"{raw_dir}: {exc}")
        return
    sizes = ", ".join(
        f"{split}={len(blk[LABEL_FALSE])}+{len(blk[LABEL_TRUE])}" for split, blk in sorted(cache.items())
    )
    reporter.check("source anchor: raw CSV readable", True, "splits " + sizes)

    question_failures: list[str] = []
    pair_failures: list[str] = []
    answer_failures: list[str] = []
    for row in rows:
        info = row["extra_info"]
        task_id = info["task_id"]
        blocks = cache[source_split_of(row)]
        index = source_index_of(row)
        if index is None:  # pragma: no cover - the vocabulary gate already failed
            continue
        side_label = LABEL_TRUE if info.get("solvable") else LABEL_FALSE
        other_label = LABEL_FALSE if side_label == LABEL_TRUE else LABEL_TRUE
        if not 0 <= index < len(blocks[side_label]):
            question_failures.append(f"{task_id}: index {index} outside the {side_label}-block")
            continue
        if _normalise(blocks[side_label][index]["question"]) != question_of(row):
            question_failures.append(f"{task_id}: question is not {side_label}-block row {index}")
        if index >= len(blocks[other_label]):
            pair_failures.append(f"{task_id}: index {index} outside the {other_label}-block")
        elif _normalise(blocks[other_label][index]["question"]) != info["paired_original_text"]:
            pair_failures.append(f"{task_id}: partner is not {other_label}-block row {index}")
        if not info.get("solvable"):
            if json.loads(row["reward_model"]["ground_truth"]).get("answer") is not None:
                answer_failures.append(f"{task_id}: unsolvable row carries an answer")
        else:
            expected = source_answer(blocks[side_label][index])
            stored = json.loads(row["reward_model"]["ground_truth"]).get("answer") or ""
            if expected != stored:
                answer_failures.append(f"{task_id}: gold answer differs from the source's label=0 answer")

    reporter.check(
        "source anchor: presented question matches the raw CSV",
        not question_failures,
        f"{len(rows)} rows, {len(question_failures)} mismatches"
        + (f"; first: {question_failures[:3]}" if question_failures else ""),
    )
    reporter.check(
        "L1 index alignment: label=1 row k pairs with label=0 row k",
        not pair_failures,
        f"{len(rows)} rows, {len(pair_failures)} misaligned partners"
        + (f"; first: {pair_failures[:3]}" if pair_failures else ""),
    )
    reporter.check(
        "source anchor: gold answer matches the raw CSV",
        not answer_failures,
        f"{len(rows)} rows, {len(answer_failures)} mismatches"
        + (f"; first: {answer_failures[:3]}" if answer_failures else ""),
    )


def check_pair_atomicity(rows: list[dict], reporter: Reporter) -> None:
    """D27: the twins of one index-aligned pair share ``pair_id`` and one side.

    ``mix_halluc.enforce_pair_atomicity`` returns a val row whose twin was not
    carved into val back to the train pool, so a pair's two rows can never be on
    opposite sides of the boundary.  The audit re-derives the property from the
    artifact: the ``pair_id`` has to be the ``"{split}:{index}"`` of the row's own
    ``task_id``, and no pair may appear on both sides.
    """
    wrong: list[str] = []
    sides: dict[str, set[str]] = collections.defaultdict(set)
    twins: collections.Counter = collections.Counter()
    for row in rows:
        info = row["extra_info"]
        parsed = parse_task_id(info.get("task_id", ""))
        pair_id = str(info.get("pair_id", "") or "")
        if parsed is None or not pair_id:
            wrong.append(f"{info.get('task_id')}: missing pair_id")
            continue
        expected = f"{parsed[1]}:{parsed[2]}"
        if pair_id != expected:
            wrong.append(f"{info['task_id']}: pair_id {pair_id!r} != {expected!r}")
        twins[pair_id] += 1
        sides[pair_id].add("val" if info.get("split") == "val" else "train")
    straddling = sorted(pair_id for pair_id, found in sides.items() if len(found) > 1)
    reporter.check(
        "D27 pair identity: extra_info.pair_id is the task_id's source split:index",
        not wrong,
        f"{len(rows)} rows, {len(wrong)} violations" + (f"; first: {wrong[:3]}" if wrong else ""),
    )
    reporter.check(
        "D27 pair atomicity: no pair is split across train/val",
        not straddling,
        f"{len(twins)} pairs, {len(straddling)} straddling sides"
        + (f"; first: {straddling[:3]}" if straddling else "")
        + f", {sum(1 for count in twins.values() if count == 2)} pair(s) with both twins present",
    )


# ---------------------------------------------------------------------------
# L1 -- the defect and the gold replacement pair
# ---------------------------------------------------------------------------


def check_l1(sample: dict[str, list[dict]], reporter: Reporter) -> None:
    for branch, picked in sample.items():
        region_failures: list[str] = []
        gold_failures: list[str] = []
        replay_failures: list[str] = []
        for row in picked:
            info = row["extra_info"]
            task_id = info["task_id"]
            real_q, fake_q = _orient(row)
            regions = rederive_regions(real_q, fake_q)
            if len(regions) != 1:
                region_failures.append(f"{task_id}: {len(regions)} visible regions")
                continue
            region = regions[0]
            if not region["fake_text"].strip():
                region_failures.append(f"{task_id}: empty fake-side region")
                continue
            deleted = info.get("deleted_condition_text", "")
            inserted = info.get("perturbed_entity_text", "")
            if not _multiset_after_edit(real_q, fake_q, deleted, inserted):
                replay_failures.append(f"{task_id}: recorded defect does not replay the pair")

            if branch != BRANCH_DIAG:
                # The answerable row is a placeholder row: it has no pointer gold,
                # so its certificate is the pair itself (region + replay), and the
                # block is audited by ``check_l3_options``/``check_answerable_side``.
                continue
            gold = _option_text(row, info.get("correct_option_id"))
            if gold is None:
                gold_failures.append(f"{task_id}: no gold option")
                continue
            pair = split_option(gold)
            if pair is None:
                gold_failures.append(f"{task_id}: gold option {gold!r} is not a replacement pair")
                continue
            left, right = pair
            if left != region["fake_text"].strip():
                gold_failures.append(f"{task_id}: gold left {left!r} is not the region {region['fake_text']!r}")
                continue
            if right != region["real_text"].strip():
                gold_failures.append(f"{task_id}: gold right {right!r} is not the repair {region['real_text']!r}")
                continue
            if left not in fake_q:
                gold_failures.append(f"{task_id}: gold left item is not verbatim in the question")
                continue
            tokens = _tokens(left)
            if not any(schema.is_content_word(token) for token in tokens):
                gold_failures.append(f"{task_id}: gold left item {left!r} has no content word")
                continue
            first = tokens[0].casefold()
            if sum(1 for token in _tokens(fake_q) if token.casefold() == first) != 1:
                gold_failures.append(f"{task_id}: first token {first!r} is not unique in the question")
                continue
            if right.casefold() in fake_q.casefold():
                gold_failures.append(f"{task_id}: gold right item {right!r} already occurs in the question")
        detail = f"{len(picked)} rows"
        reporter.check(
            f"L1 {branch}: exactly one visible defect region",
            not region_failures,
            detail + f", {len(region_failures)} failures"
            + (f"; first: {region_failures[:3]}" if region_failures else ""),
        )
        reporter.check(
            f"L1 {branch}: recorded defect replays the pair",
            not replay_failures,
            detail + f", {len(replay_failures)} failures"
            + (f"; first: {replay_failures[:3]}" if replay_failures else ""),
        )
        label = (
            "pointer gold is the region's replacement pair"
            if branch == BRANCH_DIAG
            else "placeholder row carries no pointer gold (region certificate only)"
        )
        reporter.check(
            f"L1 {branch}: {label}",
            not gold_failures,
            detail + f", {len(gold_failures)} failures"
            + (f"; first: {gold_failures[:3]}" if gold_failures else ""),
        )


def check_l2(sample: dict[str, list[dict]], reporter: Reporter) -> None:
    for branch, picked in sample.items():
        failures: list[str] = []
        for row in picked:
            info = row["extra_info"]
            task_id = info["task_id"]
            real_q, _ = _orient(row)
            if branch == BRANCH_DIAG:
                gold = _option_text(row, info.get("correct_option_id"))
                pair = split_option(gold or "")
                if pair is None:
                    failures.append(f"{task_id}: no gold replacement pair")
                    continue
                left, right = pair
                if contains_token_window(real_q, left):
                    # Text that survived the rewrite cannot be the fragment that
                    # made the premise false; the pointer must be absent from the
                    # repaired twin.
                    failures.append(f"{task_id}: gold left {left!r} also occurs in the paired real question")
                    continue
                if not contains_token_window(real_q, right):
                    # ... and the right item must be exactly what the twin says
                    # there, i.e. the replacement that repairs the premise.
                    failures.append(f"{task_id}: gold right {right!r} is not in the paired real question")
            else:
                if info.get("correct_option_id") not in ("", None):
                    failures.append(f"{task_id}: solvable row carries an extra_info correct_option_id")
                ground_truth = json.loads(row["reward_model"]["ground_truth"])
                if ground_truth.get("correct_option_id") is not None:
                    failures.append(f"{task_id}: placeholder block carries a correct option")
                if not ground_truth.get("solvable_answer"):
                    failures.append(f"{task_id}: answerable row is not on the solvable_answer branch")
                if not ground_truth.get("answer"):
                    failures.append(f"{task_id}: answerable row has an empty gold answer")
        detail = f"{len(picked)} rows, {len(failures)} failures"
        if branch == BRANCH_ANSWERABLE:
            detail += " (a placeholder block must have no correct option)"
        reporter.check(
            f"L2 {branch}",
            not failures,
            detail + (f"; first: {failures[:3]}" if failures else ""),
        )


# ---------------------------------------------------------------------------
# L3 -- the replacement-pair structure (D21/D27)
# ---------------------------------------------------------------------------


def _option_shape(rows: list[dict]) -> dict:
    """``{branch: {"count", "n", "median", "p90", "max"}}`` for the option blocks.

    The spread is max/min *character* length inside one block.  Every option
    starts with the same left item, so the spread is the right items' spread; both
    branches must show the same distribution, because a block whose lengths are
    wildly unequal is a branch-identifying shortcut no matter how well the option
    *count* matches.
    """
    raw: dict[str, tuple[int, list[float]]] = {}
    for row in rows:
        info = row["extra_info"]
        options = info.get("options") or []
        if not options:
            continue
        lengths = [len(o.get("text", "")) for o in options]
        count, spreads = raw.setdefault(info["branch"], (len(options), []))
        del count
        spreads.append(max(lengths) / max(min(lengths), 1))
    out: dict = {}
    for branch, (count, spreads) in raw.items():
        spreads.sort()
        out[branch] = {
            "count": count,
            "n": len(spreads),
            "median": statistics.median(spreads),
            "p90": spreads[int(0.9 * (len(spreads) - 1))],
            "max": max(spreads),
        }
    return out


def check_l3_options(rows: list[dict], reporter: Reporter) -> None:
    malformed: list[str] = []
    left_failures: list[str] = []
    right_failures: list[str] = []
    counts: collections.Counter = collections.Counter()
    for row in rows:
        info = row["extra_info"]
        options = info.get("options") or []
        if not options:
            continue
        counts[(info["branch"], len(options))] += 1
        question = question_of(row)
        folded = question.casefold()
        task_id = info["task_id"]
        pairs: list[tuple[str, str]] = []
        for option in options:
            pair = split_option(option.get("text", ""))
            if pair is None:
                malformed.append(f"{task_id}:{option.get('text')!r}")
            else:
                pairs.append(pair)
        if len(pairs) != len(options):
            continue
        lefts = {left for left, _ in pairs}
        if len(lefts) != 1:
            left_failures.append(f"{task_id}: {len(lefts)} distinct left items")
            continue
        left = pairs[0][0]
        if left not in question:
            left_failures.append(f"{task_id}: left item {left!r} is not verbatim in the question")
            continue
        if not any(schema.is_content_word(token) for token in _tokens(left)):
            left_failures.append(f"{task_id}: left item {left!r} has no content word")
        rights = [right for _, right in pairs]
        if len({right.casefold() for right in rights}) != len(rights):
            right_failures.append(f"{task_id}: right items are not pairwise distinct")
            continue
        signatures = {item_signature(right) for right in rights}
        if len(signatures) != 1:
            right_failures.append(f"{task_id}: right items have {len(signatures)} distinct types {sorted(signatures)}")
            continue
        present = [right for right in rights if right.casefold() in folded]
        if present:
            right_failures.append(f"{task_id}: right item(s) {present!r} occur in the question")

    reporter.check(
        "L3a every option is a `left -> right` replacement pair",
        not malformed,
        f"{len(rows)} rows, {len(malformed)} malformed option(s)"
        + (f"; first: {malformed[:3]}" if malformed else ""),
    )
    reporter.check(
        "L3b every block has one identical in-passage left item with a content word",
        not left_failures,
        f"{len(rows)} rows, {len(left_failures)} failures"
        + (f"; first: {left_failures[:3]}" if left_failures else ""),
    )
    reporter.check(
        "L3c right items are pairwise distinct, same type/length, and out-of-passage",
        not right_failures,
        f"{len(rows)} rows, {len(right_failures)} failures"
        + (f"; first: {right_failures[:3]}" if right_failures else ""),
    )
    per_branch = sorted({branch for branch, _ in counts})
    per_count = {len({count for branch, count in counts if branch == b}) for b in per_branch}
    reporter.check(
        "L3d both branches offer the same option count (D15/D18)",
        set(counts) and len(per_count) == 1 and per_count == {1} and {count for _, count in counts} == {K_OPTIONS},
        "branch -> count: "
        + ", ".join(f"{branch}:{sorted({count for b, count in counts if b == branch})}" for branch in per_branch),
    )


def check_answerable_side(rows: list[dict], reporter: Reporter) -> None:
    """The D27 contract of the answerable rows, re-derived from the artifact.

    A ``solvable_two_layer`` row is gold-answerable, so it must carry the source's
    answer verbatim, the ``solvable_answer`` flag, no correct option anywhere
    (``extra_info`` or payload) and the *placeholder* block whose only job is the
    D18 isomorphism.  ``check_l3_options`` audits the block's shape; this audits
    the contract that makes it a placeholder.
    """
    answerable = _answerable_rows(rows)
    failures: list[str] = []
    for row in answerable:
        info = row["extra_info"]
        task_id = info["task_id"]
        ground_truth = json.loads(row["reward_model"]["ground_truth"])
        if not info.get("solvable") or not ground_truth.get("solvable"):
            failures.append(f"{task_id}: not marked solvable")
        if not ground_truth.get("solvable_answer"):
            failures.append(f"{task_id}: missing the solvable_answer flag")
        if ground_truth.get("two_layer") or ground_truth.get("pair_task") or ground_truth.get("judgment_only"):
            failures.append(f"{task_id}: carries a second answer-contract flag")
        if ground_truth.get("has_diagnosis_label") or info.get("has_diagnosis_label"):
            failures.append(f"{task_id}: answerable row carries a diagnosis label")
        if ground_truth.get("correct_option_id") is not None:
            failures.append(f"{task_id}: payload correct_option_id is not null")
        if info.get("correct_option_id") not in ("", None):
            failures.append(f"{task_id}: extra_info correct_option_id is not empty")
        if info.get("error_type") or info.get("perturbation_type"):
            failures.append(f"{task_id}: answerable row carries defect metadata")
        if info.get("template") != schema.TEMPLATE_A:
            failures.append(f"{task_id}: answerable row is not on template A")
        if len(info.get("options") or []) != K_OPTIONS:
            failures.append(f"{task_id}: placeholder block is not k={K_OPTIONS}")
        if not str(ground_truth.get("answer") or "").strip():
            failures.append(f"{task_id}: gold answer is empty")
        prompt = row["prompt"][0]["content"]
        if MARKER_ANSWER not in prompt or MARKER_UNSOLVABLE_ID not in prompt:
            failures.append(f"{task_id}: template A prompt does not offer both verdicts")
    reporter.check(
        "D27 answerable contract (solvable_answer / null option / placeholder block)",
        not failures and bool(answerable),
        f"{len(answerable)} answerable rows, {len(failures)} violations"
        + (f"; first: {failures[:3]}" if failures else ""),
    )


# ---------------------------------------------------------------------------
# L4 -- the anti-cheat heuristics
# ---------------------------------------------------------------------------


def _right_item(option: dict) -> str:
    pair = split_option(option.get("text", ""))
    return pair[1] if pair else option.get("text", "")


def _pick_absent_right(options: list[dict], question: str) -> str | None:
    """The mirror shortcut of section 4.3: the right item that is *not* in the passage."""
    folded = question.casefold()
    absent = [o for o in options if _right_item(o).casefold() not in folded]
    return absent[0]["id"] if len(absent) == 1 else None


def _pick_longest(options: list[dict], question: str) -> str | None:
    lengths = [len(o.get("text", "")) for o in options]
    return options[lengths.index(max(lengths))]["id"]


def _pick_unique_capital(options: list[dict], question: str) -> str | None:
    hits = [o for o in options if any(ch.isupper() for ch in _right_item(o))]
    return hits[0]["id"] if len(hits) == 1 else None


def _pick_unique_digit(options: list[dict], question: str) -> str | None:
    hits = [o for o in options if any(ch.isdigit() for ch in _right_item(o))]
    return hits[0]["id"] if len(hits) == 1 else None


def _heuristic_rate(rows: list[dict], pick) -> tuple[float, int, int]:
    """``(accuracy, hits, rows with a defined pick)`` for one option heuristic."""
    hits = 0
    defined = 0
    for row in rows:
        gold = row["extra_info"].get("correct_option_id")
        options = row["extra_info"].get("options") or []
        choice = pick(options, question_of(row))
        if choice is None:
            continue
        defined += 1
        hits += choice == gold
    return (hits / len(rows) if rows else 0.0), hits, defined


def check_l4_heuristics(rows: list[dict], reporter: Reporter) -> None:
    diag = _diag_rows(rows)
    if not diag:
        reporter.check("L4 heuristic budget", False, "no diagnosis rows with a gold in the artifact")
        return
    k = len(diag[0]["extra_info"]["options"])
    baseline = 1 / k
    budget = baseline + L3_MARGIN

    rate, hits, defined = _heuristic_rate(diag, _pick_absent_right)
    reporter.check(
        "L4a 'pick the pair whose right item is absent from the question' is undefined, not weak",
        hits == 0,
        f"{rate:.4f} ({hits}/{len(diag)}), {defined} rows had a unique absent right item; "
        f"the D21 rule makes this cue empty",
    )

    rate, hits, _ = _heuristic_rate(diag, _pick_longest)
    reporter.check(
        f"L4b 'pick the longest option' <= {budget:.3f}",
        rate <= budget,
        f"{rate:.4f} ({hits}/{len(diag)}) vs baseline {baseline:.4f} + {L3_MARGIN:.2f}",
    )

    rate, hits, defined = _heuristic_rate(diag, _pick_unique_capital)
    reporter.check(
        f"L4c 'pick the unique option with a capital letter' <= {budget:.3f}",
        rate <= budget,
        f"{rate:.4f} ({hits}/{len(diag)}), defined on {defined} rows",
    )
    rate, hits, defined = _heuristic_rate(diag, _pick_unique_digit)
    reporter.check(
        f"L4d 'pick the unique option with a digit' <= {budget:.3f}",
        rate <= budget,
        f"{rate:.4f} ({hits}/{len(diag)}), defined on {defined} rows",
    )


def check_l5_cross_question_control(rows: list[dict], reporter: Reporter) -> None:
    """Rebuild the block D21 bans and measure what the out-of-passage rule buys.

    For each diagnosis row the control replaces *both* distractor right items with
    same-word-count spans of the presented question -- the mirror shortcut section
    4.3 warns about ("若干扰右项取自本题面，则 gold 右项是唯一的题外词") -- and re-runs
    "the right item absent from the question".  On the artifact's own blocks that
    rule has no candidate at all (L4a: 0 rows); on the minimal violation it
    identifies the gold every time.  The gate is on the control being
    *constructible* from the artifact's own text; both numbers are printed so the
    contrast is explicit.
    """
    diag = _diag_rows(rows)
    if not diag:
        reporter.check("L5 mirror control", False, "no diagnosis rows in the artifact")
        return
    usable = 0
    leaked = 0
    skipped = 0
    for row in diag:
        info = row["extra_info"]
        question = question_of(row)
        gold = _option_text(row, info.get("correct_option_id"))
        pair = split_option(gold or "")
        if pair is None:
            skipped += 1
            continue
        _, gold_right = pair
        n_right = len(_tokens(gold_right))
        borrowed = _in_passage_spans(question, n_right, 2)
        if borrowed is None:
            skipped += 1
            continue
        usable += 1
        banned = [gold] + [f"{pair[0]}{PAIR_ARROW}{span}" for span in borrowed]
        absent = [text for text in banned if _right_item({"text": text}).casefold() not in question.casefold()]
        leaked += len(absent) == 1 and absent[0] == gold
    leak_rate = leaked / usable if usable else 0.0
    reporter.note(
        f"L5 control: {usable} of {len(diag)} diagnosis rows can be rebuilt with in-passage "
        f"distractor rights, {skipped} skipped (no same-length span); mirror-shortcut leak {leak_rate:.4f}"
    )
    reporter.check(
        "L5 mirror control: an in-passage distractor identifies the gold by its absence",
        usable >= L1_MIN_SAMPLE and leak_rate >= 0.99,
        f"{usable}/{len(diag)} rows constructible, {leaked}/{usable} leaks "
        f"(the gate is >= {L1_MIN_SAMPLE} *constructible* rows -- a small artifact fails it "
        f"whatever the leak rate reads; the artifact's own blocks offer 0 such candidates, L4a)",
    )


def _in_passage_spans(question: str, n_tokens: int, k: int) -> list[str] | None:
    """``k`` distinct ``n_tokens``-token spans of ``question``, or ``None``."""
    if n_tokens <= 0:
        return None
    unique: dict[str, str] = {}
    for match in _span_windows(question, n_tokens):
        unique.setdefault(match.casefold(), match)
    if len(unique) < k:
        return None
    return [unique[key] for key in sorted(unique)[:k]]


def _span_windows(question: str, n_tokens: int) -> list[str]:
    """Character slices of every ``n_tokens``-token window of ``question``."""
    offsets = _offset_tokens(question)
    return [
        question[offsets[start][1] : offsets[start + n_tokens - 1][2]]
        for start in range(len(offsets) - n_tokens + 1)
    ]


# ---------------------------------------------------------------------------
# informational measurements
# ---------------------------------------------------------------------------


def check_l1_corroboration(rows: list[dict], reporter: Reporter, raw_dir: str, seed: int) -> None:
    """What share of the dataset's own rebuttals name the gold fragment.

    The design doc's L1 hook: a *sampling* flag for the N=50 human review, never a
    gate -- a rebuttal that paraphrases has no obligation to repeat the fragment.
    The rule here is the strictest available to a script: every content word of
    the gold's left item must appear in the rebuttal.
    """
    diag = _diag_rows(rows)
    if not diag:
        reporter.note("L1 corroboration: no diagnosis rows in the artifact")
        return
    picked = random.Random(seed).sample(diag, min(len(diag), L1_CORROBORATION_SAMPLE))
    tally: dict[str, list[bool]] = collections.defaultdict(list)
    for row in picked:
        info = row["extra_info"]
        index = source_index_of(row)
        try:
            answer = source_answer(source_blocks(raw_dir, source_split_of(row))[LABEL_FALSE][index])
        except (OSError, IndexError, TypeError):  # pragma: no cover - the anchor check already failed
            continue
        pair = split_option(_option_text(row, info.get("correct_option_id")) or "")
        if pair is None:
            continue
        left = pair[0]
        words = [t.casefold() for t in _tokens(left) if schema.is_content_word(t)]
        tally[source_split_of(row)].append(all(word in answer.casefold() for word in words))
    for split in sorted(tally):
        values = tally[split]
        reporter.note(
            f"L1 corroboration flag {split}: {sum(values)}/{len(values)} = {sum(values) / len(values):.3f} "
            f"of the dataset's own rebuttals name every content word of the gold fragment "
            f"(sample, not a gate)"
        )


def check_distractor_quality(rows: list[dict], reporter: Reporter, raw_dir: str, seed: int) -> None:
    """The section 9 N=50 spot-check hook plus the corpus-frequency reading.

    Two things are printed, neither of them a gate:

    * a reproducible 50-row sample of ``question | gold pair | distractor pairs``
      for the human judgement section 9 asks for ("干扰右项替换后前提是否也成立",
      Q9).  ``--spot-check-out`` writes the same sample as JSONL.
    * how much of the gold's identity the *corpus frequency* of the right item
      alone recovers.  The distractor pool is drawn from a corpus, so a model's
      word prior is a real cue on any block built this way; section 9's gate list
      does not include it, so it is measured and printed rather than hidden.
    """
    diag = _diag_rows(rows)
    if not diag:
        reporter.note("distractor spot check: no diagnosis rows in the artifact")
        return
    picked = random.Random(seed).sample(diag, min(len(diag), DISTRACTOR_SAMPLE))
    for row in sorted(picked, key=lambda r: r["extra_info"]["task_id"]):
        gold = _option_text(row, row["extra_info"]["correct_option_id"]) or ""
        others = [
            option["text"]
            for option in row["extra_info"]["options"]
            if option["id"] != row["extra_info"]["correct_option_id"]
        ]
        reporter.note(f"spot-check {row['extra_info']['task_id']}: gold [{gold}] vs {others}")

    try:
        corpus = source_corpus(raw_dir, sorted({source_split_of(row) for row in rows}))
    except (OSError, ValueError):  # pragma: no cover - the anchor check already failed
        return
    frequency = item_document_frequency(corpus)
    hits = 0
    defined = 0
    for row in diag:
        options = row["extra_info"]["options"]
        gold_id = row["extra_info"]["correct_option_id"]
        counts = [frequency.get(_right_item(option).casefold(), 0) for option in options]
        best = options[max(range(len(counts)), key=lambda j: (counts[j], -j))]
        defined += 1
        hits += best["id"] == gold_id
    reporter.note(
        f"distractor frequency reading: 'pick the most frequent right item' scores {hits / defined:.4f} "
        f"({len(corpus)} corpus questions, baseline {1 / len(diag[0]['extra_info']['options']):.4f}) "
        f"-- a residual corpus-vocabulary cue, printed not gated (section 9 / Q9)"
    )
    reporter.note(
        "distractor quality hook: use the spot-check sample above for the section 9 N=50 manual "
        "judgement ('does a distractor right item also repair the premise?'); >10% failures pause the source"
    )


def _count_matrix(
    indices: np.ndarray, documents: list[list[str]], vocabulary: dict[str, int]
) -> np.ndarray:
    """Bag-of-words counts for ``indices`` under one fold's fixed vocabulary."""
    matrix = np.zeros((len(indices), len(vocabulary)), dtype=np.float64)
    for row_index, doc_index in enumerate(indices):
        for token in documents[doc_index]:
            column = vocabulary.get(token)
            if column is not None:
                matrix[row_index, column] += 1.0
    return matrix


def _nb_oof_scores(
    texts: list[str],
    labels: list[int],
    *,
    folds: int,
    seed: int,
    min_support: int,
    alpha: float = 1.0,
    groups: list | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold log-odds of a multinomial Naive Bayes, mirroring ``bow_nb_oof``.

    Same ``min_support`` vocabulary rule (a token must appear in at least that many
    *training* documents), same Laplace smoothing, and -- with ``groups is None`` --
    the same fold assignment (``Random(seed).shuffle``), which is why the caller can
    cross-check the argmax decision against ``bow_nb_oof``.

    ``groups`` is the pair-aware folding the H6 reading needs.  FalseQA's two
    members are near-identical questions carrying *opposite* labels, so a random
    fold split puts a test question's own twin in the training set; a vocabulary
    rich enough to memorise it then predicts the twin's label and the out-of-fold
    score comes out *anti*-correlated with the truth (measured AUC 0.17 at
    ``min_support=1``, 0.41 at 5).  Grouping both members of a pair into the same
    fold is what makes the number mean "topic memory across pairs" instead of
    "how well the model memorised the twin".
    """
    documents = [[token.lower() for token in _tokens(text)] for text in texts]
    labels_array = np.asarray(labels, dtype=np.int64)
    count = len(texts)
    if groups is None:
        order = list(range(count))
        random.Random(seed).shuffle(order)
        fold_of = np.empty(count, dtype=np.int64)
        for rank, index in enumerate(order):
            fold_of[index] = rank % folds
    else:
        unique_groups = sorted({group for group in groups}, key=str)
        rng = random.Random(seed)
        rng.shuffle(unique_groups)
        group_fold = {group: rank % folds for rank, group in enumerate(unique_groups)}
        fold_of = np.array([group_fold[group] for group in groups], dtype=np.int64)

    margin = np.zeros(count, dtype=np.float64)
    for fold in range(folds):
        train = np.where(fold_of != fold)[0]
        test = np.where(fold_of == fold)[0]
        support: collections.Counter = collections.Counter()
        for index in train:
            support.update(set(documents[index]))
        vocabulary = {
            token: j
            for j, token in enumerate(sorted(t for t, c in support.items() if c >= min_support))
        }
        if not vocabulary:
            continue

        train_matrix = _count_matrix(train, documents, vocabulary)
        train_labels = labels_array[train]
        test_matrix = _count_matrix(test, documents, vocabulary)
        prior = np.array([np.sum(train_labels == c) for c in (0, 1)], dtype=np.float64) + alpha
        log_prior = np.log(prior / prior.sum())
        log_likelihood = np.zeros((2, len(vocabulary)), dtype=np.float64)
        for klass in (0, 1):
            totals = train_matrix[train_labels == klass].sum(axis=0) + alpha
            log_likelihood[klass] = np.log(totals / totals.sum())
        scores = test_matrix @ log_likelihood.T + log_prior
        margin[test] = scores[:, 1] - scores[:, 0]
    return margin, labels_array


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank AUC of ``scores`` against ``labels`` (ties averaged, numpy only)."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # Average the ranks of tied scores so the statistic does not depend on order.
    sorted_scores = scores[order]
    start = 0
    for index in range(1, len(scores) + 1):
        if index == len(scores) or sorted_scores[index] != sorted_scores[start]:
            if index - start > 1:
                ranks[order[start:index]] = (start + 1 + index) / 2
            start = index
    positives = labels == 1
    n_pos = int(positives.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _best_threshold_accuracy(scores: np.ndarray, labels: np.ndarray) -> float:
    """Max over thresholds of the balanced accuracy of ``scores > t``.

    This is the doc's H6 reading ("最优阈值 balanced-acc"): the ceiling a topic-memory
    baseline reaches, independent of where the argmax happens to sit.
    """
    thresholds = sorted({float(t) for t in scores})
    best = 0.0
    positives = labels == 1
    negatives = ~positives
    for threshold in thresholds:
        predicted = scores > threshold
        recall_pos = np.mean(predicted[positives]) if positives.any() else 0.0
        recall_neg = np.mean(~predicted[negatives]) if negatives.any() else 0.0
        best = max(best, 0.5 * (float(recall_pos) + float(recall_neg)))
    return best


H6_VARIANTS = {
    "all": lambda question: " ".join(_tokens(question)),
    "function": lambda question: " ".join(t for t in _tokens(question) if not schema.is_content_word(t)),
    "content": lambda question: " ".join(t for t in _tokens(question) if schema.is_content_word(t)),
}


def _argmax_accuracy(scores: np.ndarray, labels: np.ndarray) -> float:
    """Balanced accuracy of the sign decision ``score > 0``."""
    positives = labels == 1
    negatives = ~positives
    if not positives.any() or not negatives.any():
        return float("nan")
    return float(0.5 * (np.mean(scores[positives] > 0) + np.mean(scores[negatives] <= 0)))


def _h6_table(
    texts: list[str],
    labels: list[int],
    groups: list,
    *,
    folds: int,
    seed: int,
    min_support: int,
) -> dict:
    """Per-variant H6 numbers under both foldings.

    ``ungrouped_argmax`` is ``verify_umwp.bow_nb_oof`` itself, so the audit reports
    the repo's own estimator next to its score-based readings instead of only its
    own arithmetic.
    """
    out: dict = {}
    for name, transform in H6_VARIANTS.items():
        variant_texts = [transform(text) for text in texts]
        grouped, y = _nb_oof_scores(
            variant_texts, labels, folds=folds, seed=seed, min_support=min_support, groups=groups
        )
        ungrouped, _ = _nb_oof_scores(variant_texts, labels, folds=folds, seed=seed, min_support=min_support)
        out[name] = {
            "grouped_argmax": _argmax_accuracy(grouped, y),
            "grouped_best": _best_threshold_accuracy(grouped, y),
            "grouped_auc": _auc(grouped, y),
            "ungrouped_argmax": bow_nb_oof(variant_texts, labels, folds=folds, seed=seed, min_support=min_support),
            "ungrouped_best": _best_threshold_accuracy(ungrouped, y),
            "ungrouped_auc": _auc(ungrouped, y),
            "local_argmax": _argmax_accuracy(ungrouped, y),
            "doc": H6_DOC[name],
        }
    return out


def _print_h6_table(table: dict, reporter: Reporter, corpus: str) -> None:
    for name, entry in table.items():
        reporter.note(
            f"H6 {corpus:8s} {name:8s}: grouped argmax {entry['grouped_argmax']:.3f} "
            f"best {entry['grouped_best']:.3f} AUC {entry['grouped_auc']:.3f} | "
            f"ungrouped argmax {entry['ungrouped_argmax']:.3f} best {entry['ungrouped_best']:.3f} "
            f"AUC {entry['ungrouped_auc']:.3f} | doc {entry['doc']:.3f}"
        )


def _raw_h6_corpus(raw_dir: str, splits: list[str]) -> tuple[list[str], list[int], list]:
    """Every pair member of the named splits, grouped by (split, index)."""
    texts: list[str] = []
    labels: list[int] = []
    groups: list = []
    for split in splits:
        blocks = source_blocks(raw_dir, split)
        for index in range(min(len(blocks[LABEL_FALSE]), len(blocks[LABEL_TRUE]))):
            for label in (LABEL_FALSE, LABEL_TRUE):
                texts.append(_normalise(blocks[label][index]["question"]))
                labels.append(1 if label == LABEL_TRUE else 0)
                groups.append((split, index))
    return texts, labels, groups


def check_h6(rows: list[dict], reporter: Reporter, raw_dir: str, *, folds: int, seed: int) -> None:
    """The H6 interpretation baseline: topic memory on the label task (not a gate).

    Reported under both foldings because the doc's own two numbers are only
    reproducible one way each: its AUC row (0.172 train) is an *ungrouped* reading
    that a richer vocabulary drives below chance, while its balanced-accuracy row
    (0.765/0.569/0.757) needs the pair-aware split.  Neither is a gate -- H6 decides
    how the judgment metric may be read, not whether the artifact is valid.
    """
    artifact_texts = [question_of(row) for row in rows]
    artifact_labels = [1 if row["extra_info"]["solvable"] else 0 for row in rows]
    if len(set(artifact_labels)) < 2:
        reporter.note("H6: the artifact carries one label side only; skipping the interpretation baseline")
        return
    artifact_groups = [(source_split_of(row), source_index_of(row)) for row in rows]
    # ``bow_nb_oof`` refuses an artifact every fold of which is too thin to build
    # a vocabulary for; H6 is an interpretation baseline, so a small artifact is
    # reported rather than crashed on.
    try:
        table = _h6_table(
            artifact_texts, artifact_labels, artifact_groups, folds=folds, seed=seed, min_support=H6_MIN_SUPPORT
        )
    except ValueError as exc:
        reporter.note(f"H6: the artifact is too small for a {folds}-fold reading ({exc}); skipped")
        return
    agreement = all(abs(entry["local_argmax"] - entry["ungrouped_argmax"]) < 1e-9 for entry in table.values())
    reporter.check(
        "H6 local out-of-fold scorer reproduces verify_umwp.bow_nb_oof",
        agreement,
        "ungrouped argmax decisions agree on all three word subsets",
    )
    reporter.note(
        f"H6 corpus artifact: n={len(rows)} (solvable={sum(artifact_labels)}, "
        f"unsolvable={len(rows) - sum(artifact_labels)}), folds={folds}, support>={H6_MIN_SUPPORT}"
    )
    _print_h6_table(table, reporter, "artifact")

    try:
        raw_texts, raw_labels, raw_groups = _raw_h6_corpus(
            raw_dir, sorted({source_split_of(row) for row in rows})
        )
        raw_table = _h6_table(
            raw_texts, raw_labels, raw_groups, folds=folds, seed=seed, min_support=H6_MIN_SUPPORT
        )
    except (OSError, IndexError, ValueError):  # pragma: no cover - the anchor check already failed
        return
    reporter.note(f"H6 corpus raw split: n={len(raw_texts)} (the doc's own corpus), one group per pair")
    _print_h6_table(raw_table, reporter, "raw")
    # The doc's AUC row (0.172 train) reproduces only at the richest vocabulary
    # under the ungrouped split, i.e. exactly in the regime where the fold leak
    # dominates.  Reported so the deviation note rests on a measurement.
    for folding, groups_arg in (("ungrouped", None), ("grouped", raw_groups)):
        scores, y = _nb_oof_scores(
            raw_texts,
            raw_labels,
            folds=folds,
            seed=seed,
            min_support=H6_DOC_AUC_SUPPORT,
            groups=groups_arg,
        )
        reporter.note(
            f"H6 raw support={H6_DOC_AUC_SUPPORT} {folding:9s}: argmax {_argmax_accuracy(scores, y):.3f} "
            f"AUC(label=1) {_auc(scores, y):.3f} (the doc's AUC row: 0.172)"
        )


def check_monitoring(rows: list[dict], reporter: Reporter) -> None:
    """Section 10's monitors: gold position, per-branch shapes, branch balance."""
    diag = _diag_rows(rows)
    if diag:
        positions = collections.Counter(row["extra_info"]["correct_option_id"] for row in diag)
        k = len(diag[0]["extra_info"]["options"])
        reporter.note(
            f"monitor gold position: {dict(sorted(positions.items()))} (uniform expectation {len(diag) / k:.0f} per id)"
        )
        gold_lengths = collections.Counter(
            len(_tokens(split_option(_option_text(row, row["extra_info"]["correct_option_id"]) or "")[0]))
            for row in diag
            if split_option(_option_text(row, row["extra_info"]["correct_option_id"]) or "")
        )
        reporter.note(f"monitor gold left-item token length: {dict(sorted(gold_lengths.items()))}")
    shape = _option_shape(rows)
    for branch, entry in sorted(shape.items()):
        reporter.note(
            f"monitor option spread {branch}: count={entry['count']} median={entry['median']:.2f} "
            f"p90={entry['p90']:.2f} max={entry['max']:.2f} over {entry['n']} rows "
            f"(target <= {MAX_CHAR_RATIO ** 2:.1f}: the D18 isomorphism)"
        )
    answerable = len(_answerable_rows(rows))
    reporter.note(
        f"monitor branch balance: diag={len(diag)} answerable={answerable} "
        f"({answerable / len(rows):.1%} of rows are the D27 two-layer side; the two are twins and "
        f"share pair_id)"
    )


# ---------------------------------------------------------------------------
# contract invariants
# ---------------------------------------------------------------------------


def check_contract(rows: list[dict], reporter: Reporter) -> None:
    violations: list[str] = []
    for row in rows:
        violations.extend(schema.validate_row(row))
    reporter.check(
        "schema.validate_row on every row",
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
        ground_truth = json.loads(row["reward_model"]["ground_truth"])
        branch_of[branch] = info["template"]
        if not info.get("solvable") and ground_truth.get("answer") is not None:
            failures.append(f"{task_id}: unsolvable row carries an answer")
        if branch == BRANCH_ANSWERABLE:
            if not ground_truth.get("solvable_answer"):
                failures.append(f"{task_id}: answerable row without the solvable_answer flag")
            if ground_truth.get("correct_option_id") is not None:
                failures.append(f"{task_id}: answerable row carries a correct option")
            if ground_truth.get("has_diagnosis_label"):
                failures.append(f"{task_id}: answerable row carries a diagnosis label")
            if not info.get("options") or info["template"] != schema.TEMPLATE_A:
                failures.append(f"{task_id}: answerable row must carry a template A placeholder block (D18)")
            if info.get("error_type") or info.get("perturbation_type"):
                failures.append(
                    f"{task_id}: answerable row carries defect metadata "
                    f"(error_type={info.get('error_type')!r}, "
                    f"perturbation_type={info.get('perturbation_type')!r})"
                )
        if branch == BRANCH_DIAG:
            if not ground_truth.get("has_diagnosis_label"):
                failures.append(f"{task_id}: diag row without a diagnosis label")
            if info.get("error_type") != ERROR_TYPE_POINTABLE:
                failures.append(
                    f"{task_id}: diag error_type {info.get('error_type')!r} != {ERROR_TYPE_POINTABLE!r}"
                )
            if info.get("perturbation_type") != PERTURBATION_DIAG:
                failures.append(
                    f"{task_id}: diag extra_info.perturbation_type "
                    f"{info.get('perturbation_type')!r} != {PERTURBATION_DIAG!r}"
                )
            option_ids = [o["id"] for o in info.get("options") or []]
            if ground_truth.get("correct_option_id") not in option_ids:
                failures.append(
                    f"{task_id}: diag gold {ground_truth.get('correct_option_id')!r} is not one of {option_ids}"
                )
            if not ground_truth.get("perturbation_type"):
                failures.append(f"{task_id}: diag row without a perturbation_type")
            if MARKER_UNSOLVABLE_ID not in row["prompt"][0]["content"]:
                failures.append(f"{task_id}: prompt does not ask for the UNSOLVABLE:<id> marker")
        # The reward reads the *status the model emits*, so the expected marker is
        # a property of the prompt, not of a stored gold string: every template A
        # row must offer both verdicts, because the two sides share one wording.
        prompt = row["prompt"][0]["content"]
        if info["template"] == schema.TEMPLATE_A and not all(
            marker in prompt for marker in MARKER_VERDICTS
        ):
            failures.append(f"{task_id}: template A prompt does not offer both verdict markers")
        if not info.get("paired_original_text") or info["paired_original_text"] == question_of(row):
            failures.append(f"{task_id}: missing or identical paired original")
    reporter.check(
        "branch invariants (template A / verdicts / placeholder / pairing)",
        not failures,
        f"{len(rows)} rows, {len(failures)} violations" + (f"; first: {failures[:3]}" if failures else ""),
    )
    reporter.note("branch -> template: " + ", ".join(f"{b}:{t}" for b, t in sorted(branch_of.items())))

    ids = [row["extra_info"]["task_id"] for row in rows]
    bad_ids = [task_id for task_id in ids if not task_id.startswith("falseqa-")]
    reporter.check(
        "task_id is a unique, source-derived string",
        len(set(ids)) == len(ids) and not bad_ids,
        f"{len(rows)} rows, {len(set(ids))} unique, {len(bad_ids)} not source-derived"
        + (f"; first: {bad_ids[:3]}" if bad_ids else ""),
    )


def prompt_scaffold(prompt: str) -> tuple[str, str] | None:
    """The template-A wording around a row's own text, or ``None`` when it is absent.

    A template A prompt is exactly ``question + "\\n\\n" + instruction + header +
    option block``, so ``(instruction, header)`` is every part of the rendering that
    does *not* depend on the row's question or option texts.
    """
    _, separator, body = prompt.partition("\n\n")
    if not separator:
        return None
    instruction, header, _ = body.partition(OPTION_HEADER)
    if not header:
        return None
    return instruction, header


def check_template_isomorphism(rows: list[dict], reporter: Reporter) -> None:
    """The D18 defence, asserted on the artifact: one scaffold for both sides.

    Sections 5.1/5.2 and 9 make this the hard constraint of the source: the two
    sides must differ *only* in their own question and option texts, because any
    other difference is a branch-identifying shortcut that hands the model the
    verdict.  A differing option *count* is caught by :func:`check_l3_options` and
    the verdict markers by :func:`check_contract`, but neither notices a scaffold
    whose wording changed on one side: an answerable prompt that dropped the
    ``\\boxed{UNSOLVABLE: <id>}`` bullet, or renamed the option header, passes every
    other check in this file.  This check exists to catch exactly those two
    mutations.
    """
    scaffolds: dict[str, set[tuple[str, str]]] = collections.defaultdict(set)
    unparsable: list[str] = []
    unmarked: list[str] = []
    for row in rows:
        prompt = row["prompt"][0]["content"]
        scaffold = prompt_scaffold(prompt)
        if scaffold is None:
            unparsable.append(row["extra_info"]["task_id"])
            continue
        scaffolds[row["extra_info"]["branch"]].add(scaffold)
        if not all(marker in prompt for marker in MARKER_VERDICTS):
            unmarked.append(row["extra_info"]["task_id"])
    distinct = {scaffold for branch in scaffolds.values() for scaffold in branch}
    per_branch = ", ".join(f"{branch}:{len(found)}" for branch, found in sorted(scaffolds.items()))
    reporter.check(
        "D18 template isomorphism: one scaffold for both sides",
        bool(scaffolds) and len(distinct) == 1 and not unparsable and not unmarked,
        f"scaffolds per branch {{{per_branch}}}: {len(distinct)} distinct, "
        f"{len(unparsable)} prompt(s) without a template boundary/option header, "
        f"{len(unmarked)} without both verdict markers"
        + (f"; first: {(unparsable + unmarked)[:3]}" if (unparsable or unmarked) else ""),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def sample_by_branch(rows: list[dict], *, per_branch: int, seed: int) -> dict[str, list[dict]]:
    """``{branch: rows}`` for the L1/L2 certificate checks -- **all** rows by default.

    ``per_branch <= 0`` (the default) returns every row, because L1 and L2 re-derive
    the pointer gold and the pair certificate row by row: a sampled audit cannot
    certify an artifact, and a gold flipped on a row outside the sample passes it.
    A positive ``per_branch`` subsamples -- fast, but a smoke test rather than a
    certificate, which is why ``main`` labels that run explicitly.
    """
    groups: dict[str, list[dict]] = collections.OrderedDict()
    for row in rows:
        groups.setdefault(row["extra_info"]["branch"], []).append(row)
    rng = random.Random(seed)
    sample: dict[str, list[dict]] = {}
    for branch, group in groups.items():
        if per_branch <= 0:
            picked = list(group)
        else:
            picked = rng.sample(group, min(len(group), max(per_branch, L1_MIN_SAMPLE)))
        sample[branch] = sorted(picked, key=lambda row: row["extra_info"]["task_id"])
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the FalseQA rows built by falseqa_adapter.py.")
    parser.add_argument("--rows", required=True, help="parquet written by falseqa_adapter.py")
    parser.add_argument(
        "--raw-dir",
        default="/home/charles/data/reasoning_rl/halluc/raw/falseqa",
        help="downloaded FalseQA directory; the index alignment is proved against {train,valid,test}.csv here",
    )
    parser.add_argument("--seed", type=int, default=0, help="sampling + fold seed")
    parser.add_argument("--folds", type=int, default=H6_FOLDS)
    parser.add_argument("--min-support", type=int, default=H6_MIN_SUPPORT)
    parser.add_argument(
        "--per-branch",
        type=int,
        default=0,
        help="L1/L2 rows to audit per branch; 0 (default) audits every row, a positive "
        "value is a fast subsample that does not certify the artifact",
    )
    parser.add_argument(
        "--spot-check-out",
        default=None,
        help="write the section 9 N=50 distractor spot-check sample to this JSONL path",
    )
    args = parser.parse_args()

    rows = schema.read_parquet_rows(args.rows)
    if not rows:
        print(f"[FAIL] {args.rows} holds no rows")
        raise SystemExit(1)
    print(f"rows    : {args.rows} ({len(rows)} rows)")
    print(f"branches: {dict(collections.Counter(r['extra_info']['branch'] for r in rows))}")

    reporter = Reporter()
    if not check_split_vocabulary(rows, reporter):
        # Every stage below reads the source split -- as a cache key, as the name
        # of a CSV, and as a path component.  An artifact naming a split the source
        # does not have is not auditable; stop here rather than traceback inside a
        # later check with a message that reads like a verifier bug.
        print(f"RESULT: FAIL ({len(reporter.failed)}/{len(reporter.results)} checks failed: {reporter.failed})")
        raise SystemExit(1)
    check_contract(rows, reporter)
    check_template_isomorphism(rows, reporter)
    check_source_anchor(rows, reporter, args.raw_dir)
    check_pair_atomicity(rows, reporter)

    sample = sample_by_branch(rows, per_branch=args.per_branch, seed=args.seed)
    mode = "every" if args.per_branch <= 0 else f"at most {args.per_branch} per"
    reporter.note(f"L1/L2 audited {mode} branch: " + ", ".join(f"{b}={len(v)}" for b, v in sample.items()))
    for branch, picked in sample.items():
        total = len([r for r in rows if r["extra_info"]["branch"] == branch])
        if len(picked) < L1_MIN_SAMPLE and len(picked) < total:
            reporter.check(f"L1 {branch} sample size", False, f"only {len(picked)} rows to audit, need {L1_MIN_SAMPLE}")
    check_l1(sample, reporter)
    check_l2(sample, reporter)

    check_l3_options(rows, reporter)
    check_answerable_side(rows, reporter)
    check_l4_heuristics(rows, reporter)
    check_l5_cross_question_control(rows, reporter)

    print()
    check_l1_corroboration(rows, reporter, args.raw_dir, args.seed)
    check_distractor_quality(rows, reporter, args.raw_dir, args.seed)
    if args.spot_check_out:
        write_spot_check(rows, args.spot_check_out, args.seed)
        reporter.note(f"spot-check sample written to {args.spot_check_out}")
    check_h6(rows, reporter, args.raw_dir, folds=args.folds, seed=args.seed)
    check_monitoring(rows, reporter)

    print()
    if reporter.failed:
        print(f"RESULT: FAIL ({len(reporter.failed)}/{len(reporter.results)} checks failed: {reporter.failed})")
        raise SystemExit(1)
    print(f"RESULT: PASS ({len(reporter.results)}/{len(reporter.results)} checks passed)")
    raise SystemExit(0)


def write_spot_check(rows: list[dict], path: str, seed: int) -> None:
    """Write the section 9 N=50 distractor spot-check sample as JSONL."""
    diag = _diag_rows(rows)
    picked = random.Random(seed).sample(diag, min(len(diag), DISTRACTOR_SAMPLE))
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in sorted(picked, key=lambda r: r["extra_info"]["task_id"]):
            gold_id = row["extra_info"]["correct_option_id"]
            handle.write(
                json.dumps(
                    {
                        "task_id": row["extra_info"]["task_id"],
                        "split": source_split_of(row),
                        "index": source_index_of(row),
                        "question": question_of(row),
                        "gold_pair": _option_text(row, gold_id),
                        "distractor_pairs": [
                            option["text"] for option in row["extra_info"]["options"] if option["id"] != gold_id
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
