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
"""FalseQA adapter -- false presuppositions with a *pointer* gold (D13/D14, section 4.4).

Source: ``github.com/thunlp/FalseQA`` ``dataset/{train,valid,test}.csv``, three
columns ``question, answer, label`` (2,374 / 982 / 1,374 rows; each split strictly
50:50).  ``label=1`` is a question whose presupposition is false ("What should men
pay attention to when breastfeeding their child?"), ``label=0`` is the *same
template* with the presupposition repaired ("... women ...").  The two label
blocks are **index-aligned**: within a split, row ``k`` of the ``label=1`` block is
a local rewrite of row ``k`` of the ``label=0`` block.  The file is *blocked*, not
interleaved -- the first N file rows are all ``label=1`` -- so the pairing is
``zip(fake_block, real_block)`` and never ``zip(file, file[:N])``.

Why this source keeps the four-tier diagnosis (design doc section 4.4, D13)
----------------------------------------------------------------------------

MiP's defect is a *deletion*: the removed necessary condition is gone from the
prompt, so there is nothing to point at (section 4.3, D12).  FalseQA's defect is a
*replacement or insertion*: the false fragment is sitting in the question, and the
paired real question says exactly which fragment it is.  So the pointer gold is
known by construction -- a word-level ``difflib`` diff between the pair's two
questions -- and FalseQA is the only four-tier (``has_diagnosis_label=true``)
source whose gold does not have to be invented.

Contract produced by this adapter (design doc section 4.9.3 table B, rows 3/5):

===========================  ========  ===========================================
branch                       template  gold
===========================  ========  ===========================================
``unsolvable_diag``          A         ``\\boxed{UNSOLVABLE: <option id>}``
``solvable_judge`` (D14)     A         ``\\boxed{SOLVABLE}`` (``judgment_only=true``)
===========================  ========  ===========================================

The D14 isomorphism (the hard constraint of sections 4.4 and 9)
---------------------------------------------------------------

Both labels get **template A with the same option count (k=3, D15) and the same
wording**.  A row's option block is three equal-*token*-length spans of *its own*
question.  If only ``label=1`` carried an option block, "sees an option block =>
emit UNSOLVABLE" would be a 100% shortcut and the judgment half of the mix would
be free; if the two sides offered different option counts, the count itself would
be the shortcut.  ``verify_falseqa.py`` asserts both.

The judgment side has **no correct option** -- its gold is the verdict, and the
option block exists only so the prompt shape is identical to the diagnosis side.
Its options are therefore built by :func:`mine_anchor_option_spans`, which applies
the same rules as ``distractor_mining.mine_option_spans`` (same question, same
token length, at least one content word, character length within
``MAX_CHAR_RATIO`` of an anchor span) but needs no gold: the anchor is the
median-length candidate span, which plays the role the gold plays on the other
side.  See deviation 4 for why the character-length discipline is mirrored rather
than dropped.

Certificates (fail closed; each is re-derived by ``verify_falseqa.py``)
----------------------------------------------------------------------

A row is written only if every certificate below passes; otherwise the row -- or
the whole pair -- is dropped and counted.  No gold is invented and no option text
ever comes from another question.

* **Pair (both branches)** -- the split's ``label=1`` and ``label=0`` blocks have
  the same length, the two questions of a pair differ after whitespace
  normalisation, and the word-level diff ``real -> fake`` has exactly one changed
  region *visible on the fake side* (``autojunk=False``; opcodes whose fake-side
  token range is empty changed nothing the model can see and are excluded before
  the regions are merged -- including them turns 928 certified pairs into 885).
* **Pointer (diag)** -- the region's fake-side text is the gold; it must occur
  verbatim in the question, carry a content word, and its first token must occur
  exactly once in the question (so the span is locatable by its first word).
  ``distractor_mining.mine_option_spans`` then has to find two equal-length
  distractors in the same question.
* **Uniqueness (diag)** -- the gold must be the **only** option that does *not*
  occur in the paired real question, and every distractor must occur there.  Text
  that survived the rewrite cannot be the fragment that made the presupposition
  false.  This is the exact analogue of ``umwp_adapter``'s L2 certificate and it
  is a gate here, not a report.
* **Judgment (judge)** -- the pair's region certificate above, plus three
  equal-length spans of the real question at the *fake-side gold's* token length.

Determinism
-----------

The same ``(raw_dir, split, limit, seed)`` always produces byte-identical rows:
every random choice is made by a per-row :class:`random.Random` seeded from
``f"{seed}:{branch}:{split}:{index}"``, never by a process-wide RNG, so inserting
a drop anywhere cannot shift the option blocks of later rows.

DEVIATIONS FROM THE DESIGN DOC
------------------------------

Measured against ``HALLUCINATION_RL_DESIGN.md`` section 4.4 and the stage-2 recon
brief, on all three splits: 1,187 / 491 / 687 pairs, 928 / 377 / 535 certified.

1. **The k=3 funnel is 700, not the doc's 657.**  Section 4.4 says
   ``1,187 -> 928 -> 657`` ("55.3% raw / 70.8% of the pointer gold").  Measured
   with the repo's own ``distractor_mining.mine_option_spans``: **928 -> 700**
   (75.4% of the certified golds), and 288/400 on ``valid``/``test``.  The doc's
   657 is not reachable by this recipe: the two L1 gold gates leave 887 pairs and
   the k=3 pool gate 709, both above 657, so no ordering of the gates this adapter
   applies lands on it.  This adapter uses the repo miner, which is the shared
   contract, and reports the real number.

2. **The ``label=0`` side is 836, and the doc's 851 is only reproducible without
   the option-shape discipline.**  Section 4.4 allocates 851 judgment rows (71.7%).
   The recipe that reaches it is: take every pair with a region certificate (928),
   use the **fake-side gold's token count** as the anchor, and mine three
   equal-length spans from the **real** question -- measured **850**, one row from
   the doc's number.  The design doc describes the anchor as the *paired real-side*
   diff fragment's length/shape; that reading gives **816**, because 40 of the 928
   pairs have an empty real-side region and no fragment to measure.  The 850 drops
   to **836** once the judgment block is held to the same character-length
   discipline as the diagnosis side (deviation 4); the written artifact carries
   836 rows (345 valid, 483 test).

3. **The doc's per-gold-length distribution is right to the row; its total is
   not.**  Section 4.4 reports 620/201/52 certified golds of 1/2/3 tokens -- both
   measured (over the 928 certified pairs) and reproduced: **620/200/52**, with
   **{1: 608, 2: 184, 3: 45, 4: 26, 5: 13, 6: 7, 7: 1, 8: 2, 14: 1}** over the 887
   golds that clear the L1 gates.  So the doc's probe and this adapter agree on the
   data and disagree only on what the pool threshold leaves behind.  Its pool table
   (median 3/1/0, P(pool >= 3) 73.5%/9.0%) is *not* reproduced under any of the
   four pool definitions the recon tried (exact vs ratio-band character length,
   overlapping vs greedy spans): the repo miner's own rule gives median 4/4/4 and
   P(pool >= 3) 0.82/0.85/0.83 for 1/2/3-token golds, while exact character-length
   equality gives median 0 at every length -- the latter is where a 657-row funnel
   could come from, but it is not what the shared miner does.

4. **The judgment option block needs a character-length guard the doc does not
   mention, and the guard is what makes the two labels isomorphic.**  Without it
   the judgment blocks' char-length spread (max option / min option) reaches 13.00
   with p90 2.67 on ``train`` (9.00 and 10.00 on ``valid`` / ``test``), against
   4.00 / p90 2.00 on the diagnosis side -- "this block's option lengths are wildly
   unequal" would be a branch-identifying shortcut, exactly the D14 failure the
   option block exists to prevent.  With ``MAX_CHAR_RATIO=2.0`` mirrored around the
   anchor span the judgment spread becomes median 1.60 / p90 2.33 / max 3.67,
   against 1.50 / 2.00 / 4.00 for the diagnosis side, at a cost of 14 rows
   (850 -> 836 train, 349 -> 345 valid, 493 -> 483 test).

5. **The doc's L1 gates are reproduced, and one of them is now a gate.**  Section
   4.4 says the gold carries a content word on 98.9% of certified pairs (measured
   **917/928 = 98.8%**) and that its first word is unique on 96.3% (measured
   **898/928 = 96.8%**, case-insensitively).  Both are applied in order, so 887
   golds pass both (95.6% of the certified pairs, 700 written after the L2 gate).

6. **The doc's L2 as written is a pool check, not a uniqueness proof.**  Section
   4.4 defines L2 as "can k=3 equal-length spans be built from the same question",
   which certifies nothing about *which* span is the gold.  This adapter adds the
   ``umwp_adapter`` uniqueness gate of the certificates section: the gold must be
   the only option absent from the paired real question and every distractor must
   be present there.  Measured: 9 further rows (train) fail it, 3 on ``valid`` and
   2 on ``test``; the gate is what makes the pointer gold the only admissible
   answer, so it is a gate rather than a report.

7. **The doc's L3 "pick the longest" figure measures a degenerate cue, and the
   character-length cue it misses is worth 39-43%.**  Section 4.4 / section 9
   report 30.9% for the longest-option heuristic against a 33.3% baseline.  Every
   option in a block has the same *token* count, so a token-count reading of "the
   longest" ties everywhere and collapses to first-index ``1/k`` -- which is what
   30.9% is.  Measured on character length, with the option positions actually
   shuffled: **39.1% train / 39.6% valid / 42.8% test** against the 43.3% budget,
   and **43.7% / 41.8% / 46.9%** on the subset whose maximum is unique.  The cue is
   real: the pointed-at fragment is a rewrite or an insertion, so on **204 of the
   887** train golds that clear the L1 gates it is *strictly* longer than every
   other same-token-length window of its own question (23.0%; 25.5% valid, 27.3%
   test) and longest-or-tied on 298 (33.6%).  Inside the block the adapter writes,
   the gold is the unique longest option on 547 of 700 diagnosis rows (78%), and
   that is exactly the 43.7% reading above.  No in-question mining rule removes the
   cue without dropping those rows.
   Choosing tie-free blocks (``mine_distinct_block``) is the mitigation that does
   work -- it removes the first-index tie-break's free wins, 0.4600 -> 0.4275 on
   ``test`` at seed 0 (mean over seeds 0-3: 0.4371 -> 0.4258) -- and 8 attempts
   already saturate it (the stated rate is unchanged at 32 and 128 attempts).

   **The audit gates on the stated reading, which clears the budget, and the
   tie-free reading does not: 0.4369 > 0.4333 on train and 0.4693 > 0.4333 on
   ``test`` (valid is under, 0.4184).**  The gate is section 9's rule as written --
   "pick the longest option", first-index tie-break, defined on every row -- and the
   tie-free reading is a *stricter* rule over a subset (547 of 700 train rows, 309
   of 400 test).  Gating on the stricter rule would be inventing a check the doc
   does not state and would fail the artifact on a cue the stated rule cannot
   express, so the audit prints the tie-free number as a measurement instead of
   hiding it, and the excess is left visible rather than engineered away by
   dropping the 23% of pairs whose gold is the strictly longest window (or the 78%
   of rows whose block has a unique longest option).  This is the one place where
   the artifact is over a section 9 budget under a defensible reading of the same
   heuristic; it is recorded here and printed by the audit.  Seed spread on
   ``test``: the stated
   reading is 0.4196-0.4325 over seeds 0-3 -- 0.1 points of headroom at the worst
   seed, 0.6 at the default seed 0 -- and the audit prints all three readings.

8. **The test split's ``answer`` really is a 3-element Python list, and it is
   still unusable.**  Sample doc section 3.3 says ``test.csv``'s answer is a list
   repr; measured 687/687 on ``test``'s ``label=1`` block (0/687 on ``label=0``,
   and 0 lists in train/valid on either side).  The rebuttal is *audit material*
   only: it is not a span of the question, so it cannot be a gold, and the design
   doc's D13 ban on cross-question options stands.  The audit's corroboration flag
   (share of rebuttals naming every content word of the gold) measures 0.720 /
   0.620 / 0.560 against the doc's 56.8% / 56.2% / 85.3% -- the ``test`` gap is the
   list parse, where the doc's own rule and this one take different elements.

9. **The D14 judgment branch has no quota cell in ``mix_halluc.py``.**  Section
   4.9.3 table B row 3 (solvable judgment, with option block) lists UMWP and SUM
   only, so ``DEFAULT_QUOTA`` has no ``(solvable_judge, halluc_commonsense_falseqa)``
   entry and the 836 judgment rows this adapter writes are built but never mixed.
   That is the doc's own table, not a bug in the mix; it is recorded here because
   the coverage loss is easy to miss.  The judgment rows remain in the artifact so
   the D14 contract is testable and so a quota can be added without a rebuild.

10. **The doc's H6 numbers are not reproducible as a pair; the AUC row is a fold
    artefact.**  Section 4.4 claims bag-of-words Naive Bayes balanced accuracy
    0.765/0.569/0.757 (all / function / content words) with an AUC row of 0.172
    train.  Measured over the doc's own corpus (train, n=2374) with the repo's
    estimator: the AUC row reproduces at a 1-document vocabulary under a *random*
    fold split (**0.152**), where each test question's near-identical twin sits in
    the training set carrying the opposite label and the score goes
    anti-correlated.  Under pair-aware folds -- the only reading that measures
    topic memory rather than twin memorisation -- the same numbers are argmax
    0.552 / best-threshold 0.559 / AUC 0.580 for all words (and 0.52-0.55 for the
    two word subsets), and no folding or vocabulary threshold tried reaches the
    doc's 0.765.  So the judgment metric is *less* learnable by topic memory than
    the doc warns, not more; ``verify_falseqa.py`` prints both foldings and is
    explicit that H6 is an interpretation baseline, not a gate.

Module note on L1 and H6
------------------------

FalseQA's labels are human-constructed (the recon's H6 probe), and the certificates
above prove that *the recorded region is the only difference between the pair* and
that the pointer gold is the unique option absent from the original question.  They
do not prove that every ``label=1`` question really has a false presupposition --
that is the source's own annotation, and the L1 corroboration flag in
``verify_falseqa.py`` (what share of the dataset's own rebuttals name the gold
span) is the audit hook for it, not a gate.  Likewise, the judgment half is
measurably learnable by topic memory rather than by reasoning (the doc's H6);
``verify_falseqa.py`` reports the bag-of-words Naive Bayes number so the D14
metric is never read as a capability number.
"""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import difflib
import functools
import json
import os
import random
import sys

try:
    import schema
    from distractor_mining import mine_option_spans, token_spans
except ImportError:  # pragma: no cover - running as a plain script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import schema
    from distractor_mining import mine_option_spans, token_spans

# ---------------------------------------------------------------------------
# source constants
# ---------------------------------------------------------------------------

DATA_SOURCE = schema.SOURCE_FALSEQA
DEFAULT_RAW_DIR = "/home/charles/data/reasoning_rl/halluc/raw/falseqa"
DEFAULT_OUT = os.path.expanduser("~/data/reasoning_rl/halluc/built/falseqa.parquet")


def report_path_for(out: str) -> str:
    """The report that belongs to ``--out``: same stem, ``_report.json``.

    Derived from ``--out`` instead of pinned to the build directory, so that a
    scratch build (``--out /tmp/falseqa.parquet``) writes a scratch report rather
    than overwriting the canonical one the build report cites.  Same convention in
    all four adapters.
    """
    return os.path.splitext(out)[0] + "_report.json"


DATA_FILE_TEMPLATE = "{split}.csv"
SPLITS = ("train", "valid", "test")
DEFAULT_SPLIT = "train"

BRANCH_DIAG = schema.BRANCH_UNSOLVABLE_DIAG
BRANCH_JUDGE = schema.BRANCH_SOLVABLE_JUDGE

K_OPTIONS = 3  # design decision D15
MAX_CHAR_RATIO = 2.0  # the character-length discipline distractor_mining enforces
TIE_RETRIES = 8  # option-sampling attempts spent avoiding a character-length tie

#: ``label`` column values: ``"1"`` is the false presupposition (unsolvable),
#: ``"0"`` the paired true one.  The column is an unquoted single character.
LABEL_FALSE = "1"
LABEL_TRUE = "0"

#: A false presupposition that *can* be pointed at in the prompt -- the D18 defect
#: slot FalseQA fills (section 4.9.3 table B, 假前提（题面可指认）).  The slug is
#: deliberately distinct from ``false_premise_unpointable``, which CREPE and KUQ
#: carry: the balance table keys on these strings.
ERROR_TYPE = "false_premise_pointable"

#: Section 4.4's delivery constraint: the false fragment contradicts the question's
#: own premise (it is a rewrite of the paired real question, not an unrelated
#: inserted sentence).
PERTURBATION_TYPE = "contradictory_condition"

SIDE_FAKE = "fake"
SIDE_REAL = "real"


# ---------------------------------------------------------------------------
# text plumbing
# ---------------------------------------------------------------------------


def normalise_question(text: str) -> str:
    """Collapse whitespace and strip.

    One ``test`` question is not equal to its own ``.strip()`` (recon section 5.2
    item 10).  Every offset in this module is computed on the *normalised* text,
    which is also the text rendered into the prompt -- the two must never be
    mixed, or the gold offsets and the visible question drift apart.
    """
    return " ".join((text or "").split())


class _Region:
    """One contiguous run of changed word tokens between the two questions.

    ``a_*`` is the real (``label=0``) side, ``b_*`` the fake (``label=1``) side,
    so the diff is always ``real -> fake`` and the gold is always the fake-side
    text.  ``n_a`` / ``n_b`` are the token counts of the two sides, which the
    judgment branch reads as its mining anchor.
    """

    __slots__ = ("tag", "a_text", "b_text", "a_start", "b_start", "n_a", "n_b")

    def __init__(self, **kwargs) -> None:
        for key in self.__slots__:
            setattr(self, key, kwargs.get(key, ""))

    @property
    def visible(self) -> bool:
        """Whether the region changed anything on the fake side."""
        return bool(self.b_text)


def _token_offsets(text: str) -> list[tuple[str, int, int]]:
    """``[(token, start, end)]`` -- the same token grammar as distractor_mining."""
    return [(m.group(0), m.start(), m.end()) for m in schema._WORD_RE.finditer(text)]


def _merge(opcodes: list[tuple], *, require_visible: bool) -> list[tuple]:
    """Join adjacent changed opcodes into runs.

    ``difflib`` already reports a two-sided change as one ``replace``, so the join
    normally only drops the ``equal`` opcodes; it is kept because a hand-built or
    future opcode list must not turn one defect into two regions.  Two opcodes are
    adjacent when they are contiguous on **both** token sequences.

    ``require_visible`` drops the opcodes whose fake-side range is empty *before*
    grouping.  A pure deletion changed the real question but leaves the presented
    (fake) question untouched, so it cannot be pointed at; leaving it in the
    grouping merges it with a neighbouring replace and turns 928 certified pairs
    into 885 (recon section 5 item 3).
    """
    runs: list[tuple] = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        if require_visible and j2 <= j1:
            continue
        if runs and runs[-1][2] == i1 and runs[-1][4] == j1:
            previous = runs[-1]
            runs[-1] = (previous[0], previous[1], i2, previous[3], j2)
        else:
            runs.append((tag, i1, i2, j1, j2))
    return runs


def word_regions(real_q: str, fake_q: str) -> list[_Region]:
    """The changed regions of ``real_q -> fake_q`` that are visible on the fake side.

    Diffing *token sequences* (``autojunk=False``) rather than characters is what
    keeps every gold a whole word span.  The returned regions are already merged,
    and the caller's certificate requires exactly one of them.  When a pair also
    carries a pure deletion elsewhere, that deletion is *not* part of the region
    list -- it cannot be pointed at -- but it is still part of the full defect
    recorded in ``extra_info.deleted_condition_text`` (see :func:`defect_texts`).
    """
    ta, tb = _token_offsets(real_q), _token_offsets(fake_q)
    matcher = difflib.SequenceMatcher(
        None, [t[0] for t in ta], [t[0] for t in tb], autojunk=False
    )
    regions: list[_Region] = []
    for tag, i1, i2, j1, j2 in _merge(matcher.get_opcodes(), require_visible=True):
        a_start = ta[i1][1] if i2 > i1 else (ta[i1 - 1][2] if i1 > 0 else 0)
        a_end = ta[i2 - 1][2] if i2 > i1 else a_start
        b_start = tb[j1][1] if j2 > j1 else len(fake_q)
        b_end = tb[j2 - 1][2] if j2 > j1 else b_start
        regions.append(
            _Region(
                tag=tag,
                a_text=real_q[a_start:a_end],
                b_text=fake_q[b_start:b_end],
                a_start=a_start,
                b_start=b_start,
                n_a=i2 - i1,
                n_b=j2 - j1,
            )
        )
    return regions


def defect_texts(real_q: str, fake_q: str) -> tuple[str, str]:
    """``(deleted, inserted)`` over *every* changed opcode (audit fields only).

    Unlike :func:`word_regions` this keeps the invisible (pure deletion) opcodes,
    so re-applying the recorded edit to the real question rebuilds the fake one
    token for token -- which is what the audit's multiset check re-derives.  The
    gold is read off :func:`word_regions`, never off this pair of strings: on the
    43 train pairs that carry an extra invisible deletion the two disagree, and the
    pointer must be the fragment the model can see.
    """
    matcher = difflib.SequenceMatcher(
        None, schema.words(real_q), schema.words(fake_q), autojunk=False
    )
    ta, tb = _token_offsets(real_q), _token_offsets(fake_q)
    deleted: list[str] = []
    inserted: list[str] = []
    for tag, i1, i2, j1, j2 in _merge(matcher.get_opcodes(), require_visible=False):
        if i2 > i1:
            deleted.append(real_q[ta[i1][1] : ta[i2 - 1][2]])
        if j2 > j1:
            inserted.append(fake_q[tb[j1][1] : tb[j2 - 1][2]])
    return " ".join(deleted), " ".join(inserted)


# ---------------------------------------------------------------------------
# source loading and pairing
# ---------------------------------------------------------------------------


def load_source(raw_dir: str, split: str = DEFAULT_SPLIT) -> list[dict]:
    """Parse ``{split}.csv`` into a list of ``{question, answer, label}`` dicts.

    A row whose ``label`` is neither ``"0"`` nor ``"1"`` (or whose question is
    empty/not a string) is returned as-is so the caller can count and drop it
    rather than crash the build.
    """
    path = os.path.join(raw_dir, DATA_FILE_TEMPLATE.format(split=split))
    with open(path, encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _is_well_formed(row: object) -> bool:
    if not isinstance(row, dict):
        return False
    if not isinstance(row.get("question"), str) or not row["question"].strip():
        return False
    return row.get("label") in (LABEL_FALSE, LABEL_TRUE)


def _pair_blocks(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """``(fake block, real block)``, each in file order.

    The file is blocked by label, so the pairing is positional *within the two
    blocks*: file row ``k`` pairs with file row ``N + k``, never with row ``k + 1``.
    """
    fake = [row for row in rows if row["label"] == LABEL_FALSE]
    real = [row for row in rows if row["label"] == LABEL_TRUE]
    return fake, real


# ---------------------------------------------------------------------------
# certificates
# ---------------------------------------------------------------------------


def certify_gold(real_q: str, fake_q: str, regions: list[_Region]) -> str | None:
    """The pointer gold for a certified pair, or ``None`` when it cannot be proved.

    The gold is the fake-side text of the pair's single visible region.  It must
    occur verbatim in the presented question (it is a slice of it), carry at least
    one content word (a bare ``"the"`` makes an option block that is worse than no
    block at all), and its first token must occur exactly once in the question, so
    that the span is locatable by its first word rather than by an ambiguous
    occurrence.
    """
    if len(regions) != 1:
        return None
    gold = regions[0].b_text.strip()
    if not gold or gold not in fake_q:
        return None
    tokens = schema.words(gold)
    if not tokens:
        return None
    if not any(schema.is_content_word(token) for token in tokens):
        return None
    first = tokens[0].casefold()
    if sum(1 for token in schema.words(fake_q) if token.casefold() == first) != 1:
        return None
    return gold


def unique_absent_option(real_q: str, texts: list[str], gold: str) -> bool:
    """Whether the gold is the only option that did *not* survive the rewrite.

    The L2 uniqueness proof, and the exact analogue of ``umwp_adapter``'s rule:
    text that still occurs in the paired real question cannot be the fragment that
    made this question's presupposition false, so a block whose gold is the only
    absent option has exactly one admissible answer.
    """
    if gold in real_q:
        return False
    return all(text in real_q for text in texts if text != gold)


def mine_anchor_option_spans(
    question: str,
    n_tokens: int,
    k: int = K_OPTIONS,
    rng: random.Random | None = None,
    max_char_ratio: float = MAX_CHAR_RATIO,
) -> list[str] | None:
    """Mine ``k`` equal-token-length spans of ``question`` with no designated gold.

    The judgment branch has no correct option, so ``mine_option_spans`` cannot be
    called: it needs a gold substring to rank candidates against.  This function
    keeps the same three rules -- spans of the same question, the same token
    length, at least one content word -- and uses the **median-length candidate**
    as the anchor in the role the gold plays on the diagnosis side: candidates
    whose character length is more than ``max_char_ratio`` away from the anchor are
    rejected, and the ones closest to it are preferred.  See deviation 4 of the
    module docstring for why the character-length discipline is mirrored instead of
    dropped.

    Args:
        question: the exact question text that will be rendered into the prompt.
        n_tokens: the anchor token count (the fake-side gold's length).
        k: total option count (D15 fixes the default at 3).
        rng: used to sample among equally close candidates; a fixed seed makes the
            build reproducible.  ``None`` means the process-wide RNG.
        max_char_ratio: the character-length band around the anchor, in the same
            units ``distractor_mining.mine_option_spans`` uses.

    Returns:
        ``k`` distinct span texts, or ``None`` when the question does not contain
        ``k`` admissible spans at that token length.
    """
    rng = rng or random
    if n_tokens <= 0:
        return None
    unique: dict[str, object] = {}
    for span in token_spans(question, n_tokens):
        if not any(schema.is_content_word(token) for token in schema._WORD_RE.findall(span.text)):
            continue
        unique.setdefault(span.text.strip().casefold(), span)
    candidates = sorted(unique.values(), key=lambda span: (span.char_len, span.start))
    if len(candidates) < k:
        return None
    anchor = candidates[len(candidates) // 2]
    anchor_len = max(anchor.char_len, 1)
    admissible = [
        span
        for span in candidates
        if 1 / max_char_ratio <= span.char_len / anchor_len <= max_char_ratio
    ]
    if len(admissible) < k:
        return None
    admissible.sort(key=lambda span: (abs(span.char_len - anchor_len), span.start))
    best_delta = abs(admissible[0].char_len - anchor_len)
    best_tier = [span for span in admissible if abs(span.char_len - anchor_len) == best_delta]
    pool = best_tier if len(best_tier) >= k else admissible
    return [span.text.strip() for span in rng.sample(pool, k)]


def _mine_diag_options(question: str, gold: str, attempt_seed: str) -> list[str] | None:
    """One sampling attempt of the diagnosis block: the repo miner, texts only."""
    mined = mine_option_spans(question, gold, k=K_OPTIONS, rng=random.Random(attempt_seed))
    return None if mined is None else list(mined[0])


def _mine_judge_options(question: str, n_tokens: int, attempt_seed: str) -> list[str] | None:
    """One sampling attempt of the judgment block: the anchored miner, seeded."""
    return mine_anchor_option_spans(
        question, n_tokens, k=K_OPTIONS, rng=random.Random(attempt_seed)
    )


def mine_distinct_block(
    mine,
    seed_prefix: str,
    k: int,
    retries: int = TIE_RETRIES,
) -> list[str] | None:
    """Re-roll an option sampler until the block's character lengths are distinct.

    Both labels must offer a block whose option lengths carry as little about the
    answer as the source text allows.  ``mine_option_spans`` prefers candidates
    closest in character length to the gold -- deliberately, so the block stays
    tight -- but that makes a length *tie* common, and a strategy of "pick the
    first longest option" then wins on ties without knowing anything about the
    answer: with the tie left in place that rule scores 0.4600 on ``test`` at seed
    0 (mean 0.4371 over seeds 0-3), over the 0.4333 budget of the section 9 audit,
    and re-rolling brings it down to 0.4275 (mean 0.4258), while on the rows whose
    maximum is unique in the first place -- where the cue is a pure length cue --
    it still scores 0.4693.  The excess is the tie-break rule, not information, so
    it is sampled away here rather than left for the audit to excuse.  Re-rolling
    is bounded (``retries`` seeds derived from the row, so the outcome stays
    deterministic) and a row whose question offers no tie-free block keeps the tie
    rather than being dropped.

    Args:
        mine: ``seed -> list[str] | None``, one sampling attempt.
        seed_prefix: row-identifying string the attempt seeds are derived from.
        k: total option count, used for the distinctness test.
        retries: how many attempts before the tie is accepted.

    Returns:
        The first tie-free block, the first block sampled if none is tie-free, or
        ``None`` when the sampler itself reports an insufficient pool.
    """
    fallback: list[str] | None = None
    for attempt in range(retries):
        texts = mine(f"{seed_prefix}:{attempt}")
        if texts is None:
            return None
        if fallback is None:
            fallback = texts
        if len({len(text) for text in texts}) == k:
            return texts
    return fallback


def audit_answer(row: dict) -> str:
    """The source's own ``answer`` text, for ``ground_truth.answer`` (audit only).

    ``label=0``'s answer is free text ("Because cats are much larger than mice.")
    about 67.8% of the time and never a usable gold; the judgment branch stores it
    purely so the artifact is self-describing and the audit can anchor the row to
    the raw file.  ``test``'s ``label=1`` answers are the repr of a 3-element
    Python list (measured 687/687), so the list form is parsed and the first
    element taken -- the sample doc requires it -- though this adapter discards the
    ``label=1`` answer anyway (an unsolvable row must not carry one).
    """
    text = (row.get("answer") or "").strip()
    if text.startswith("["):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):  # pragma: no cover - defensive
            return text
        if isinstance(parsed, list | tuple) and parsed:
            return str(parsed[0]).strip()
    return text


# ---------------------------------------------------------------------------
# row construction
# ---------------------------------------------------------------------------


def _task_id(side: str, split: str, index: int) -> str:
    """Stable hard-replay key; carries the side because both sides may be written."""
    return f"falseqa-{side}-{split}-{index}"


def _base_extra_info(split: str, index: int, seed: int, partner_question: str) -> dict:
    return {
        "split": split,
        "index": index,
        "seed": seed,
        "paired_original_text": partner_question,
        "perturbation_family": "",
        "difficulty": "",  # the source carries no difficulty axis (recon section 4)
    }


def _build_diag(
    *,
    split: str,
    index: int,
    seed: int,
    question: str,
    partner_question: str,
    gold: str,
    texts: list[str],
    correct: str,
    deleted: str,
    inserted: str,
) -> dict:
    extra = _base_extra_info(split, index, seed, partner_question)
    extra.update(
        {
            "task_id": _task_id(SIDE_FAKE, split, index),
            "error_type": ERROR_TYPE,
            "perturbation_type": PERTURBATION_TYPE,
            "perturbed_entity_text": inserted,
            "deleted_condition_text": deleted,
            "solvable": False,
            "correct_option_id": correct,
            "has_diagnosis_label": True,
        }
    )
    options = [{"id": chr(ord("A") + i), "text": text} for i, text in enumerate(texts)]
    return schema.make_row(
        data_source=DATA_SOURCE,
        question=question,
        ground_truth=schema.build_ground_truth(
            solvable=False,
            answer=None,
            correct_option_id=correct,
            has_diagnosis_label=True,
            perturbation_type=PERTURBATION_TYPE,
        ),
        template=schema.TEMPLATE_A,
        branch=BRANCH_DIAG,
        extra_info=extra,
        options=options,
    )


def _build_judge(
    *,
    split: str,
    index: int,
    seed: int,
    question: str,
    partner_question: str,
    texts: list[str],
    deleted: str,
    inserted: str,
    audit: str,
) -> dict:
    extra = _base_extra_info(split, index, seed, partner_question)
    extra.update(
        {
            "task_id": _task_id(SIDE_REAL, split, index),
            "error_type": "",
            "perturbation_type": "",
            "perturbed_entity_text": inserted,
            "deleted_condition_text": deleted,
            "solvable": True,
            "correct_option_id": "",
            "judgment_only": True,
        }
    )
    options = [{"id": chr(ord("A") + i), "text": text} for i, text in enumerate(texts)]
    return schema.make_row(
        data_source=DATA_SOURCE,
        question=question,
        ground_truth=schema.build_ground_truth(
            solvable=True,
            answer=audit,
            judgment_only=True,
            has_diagnosis_label=False,
            perturbation_type=None,
        ),
        template=schema.TEMPLATE_A,
        branch=BRANCH_JUDGE,
        extra_info=extra,
        options=options,
    )


def build_rows(
    raw_dir: str,
    limit: int | None = None,
    seed: int = 0,
    split: str = DEFAULT_SPLIT,
) -> tuple[list[dict], dict]:
    """Build the FalseQA rows and the report that produced them.

    Args:
        raw_dir: directory holding ``train.csv`` / ``valid.csv`` / ``test.csv``.
        limit: cap on the number of **pairs** (source index ``k``), not on the
            number of rows.  A pair can yield one row (either branch) or two, so
            the written count does not equal ``limit``; the pairing is positional
            inside the two label blocks, so a pair cap is well defined and
            independent of the file's blocked layout.  ``None`` writes everything
            the certificates admit.
        seed: seeds the per-row :class:`random.Random` for the option sampling; the
            admitted *set* is fixed by the certificates, so the same seed always
            yields byte-identical rows.
        split: which of the source's own splits to read (``train`` by default).

    Returns:
        ``(rows, report)``.  ``rows`` are ready for
        :func:`schema.normalise_extra_info` / :func:`schema.validate_rows` /
        :func:`schema.write_rows_parquet`; ``report`` carries the funnel, an
        explicit drop-reason table, and the per-branch breakdown.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")

    raw = load_source(raw_dir, split)
    funnel: collections.OrderedDict[str, int] = collections.OrderedDict()
    drops: collections.OrderedDict[str, int] = collections.OrderedDict(
        (name, 0) for name in DROP_REASONS
    )
    funnel["raw_rows"] = len(raw)

    well_formed = [row for row in raw if _is_well_formed(row)]
    drops["malformed_row"] = len(raw) - len(well_formed)
    funnel["after_malformed_drop"] = len(well_formed)

    fake_block, real_block = _pair_blocks(well_formed)
    if len(fake_block) != len(real_block):
        # Fail closed on the whole split rather than guess a pairing: the index
        # alignment is the source's certificate, and a ragged block means it
        # cannot be trusted for any pair.
        drops["unpaired_label_block"] = abs(len(fake_block) - len(real_block))
        pairs: list[tuple[dict, dict]] = []
    else:
        pairs = list(zip(fake_block, real_block, strict=True))
    funnel["pairs_available"] = len(pairs)
    if limit is not None:
        pairs = pairs[: max(limit, 0)]
    funnel["pairs_after_limit"] = len(pairs)

    certified = 0
    built: list[dict] = []
    for index, (fake_row, real_row) in enumerate(pairs):
        real_q = normalise_question(real_row["question"])
        fake_q = normalise_question(fake_row["question"])
        if real_q == fake_q:
            drops["identical_question_pair"] += 1
            continue
        regions = word_regions(real_q, fake_q)
        if len(regions) != 1:
            drops["multi_region_defect"] += 1
            continue
        certified += 1
        deleted, inserted = defect_texts(real_q, fake_q)

        gold = certify_gold(real_q, fake_q, regions)
        if gold is not None:
            if not any(schema.is_content_word(token) for token in schema.words(gold)):
                drops["gold_has_no_content_word"] += 1
            else:
                texts = mine_distinct_block(
                    functools.partial(_mine_diag_options, fake_q, gold),
                    f"{seed}:{BRANCH_DIAG}:{split}:{index}",
                    K_OPTIONS,
                )
                if texts is None:
                    drops["diag_pool_below_k"] += 1
                else:
                    correct = chr(ord("A") + texts.index(gold))
                    if not unique_absent_option(real_q, texts, gold):
                        drops["diag_not_unique_absent_option"] += 1
                    else:
                        built.append(
                            _build_diag(
                                split=split,
                                index=index,
                                seed=seed,
                                question=fake_q,
                                partner_question=real_q,
                                gold=gold,
                                texts=texts,
                                correct=correct,
                                deleted=deleted,
                                inserted=inserted,
                            )
                        )
        elif len(regions) == 1:
            # The single region failed a gold gate; count which one, in the same
            # order certify_gold applies them.
            tokens = schema.words(regions[0].b_text.strip())
            if not tokens or not any(schema.is_content_word(token) for token in tokens):
                drops["gold_has_no_content_word"] += 1
            elif regions[0].b_text.strip() not in fake_q:
                drops["gold_not_verbatim_in_question"] += 1
            else:
                drops["gold_first_token_not_unique"] += 1

        anchor_tokens = regions[0].n_b
        anchor_texts = mine_distinct_block(
            functools.partial(_mine_judge_options, real_q, anchor_tokens),
            f"{seed}:{BRANCH_JUDGE}:{split}:{index}",
            K_OPTIONS,
        )
        if anchor_texts is None:
            drops["judge_pool_below_k"] += 1
        else:
            built.append(
                _build_judge(
                    split=split,
                    index=index,
                    seed=seed,
                    question=real_q,
                    partner_question=fake_q,
                    texts=anchor_texts,
                    deleted=deleted,
                    inserted=inserted,
                    audit=audit_answer(real_row),
                )
            )

    funnel["pairs_with_region_certificate"] = certified
    funnel["rows_built"] = len(built)

    report = collections.OrderedDict()
    report["raw_dir"] = raw_dir
    report["split"] = split
    report["limit"] = limit
    report["seed"] = seed
    report["funnel"] = dict(funnel)
    report["drops"] = dict(drops)
    report["by_branch"] = _breakdown(built, lambda row: row["extra_info"]["branch"])
    report["by_template"] = _breakdown(built, lambda row: row["extra_info"]["template"])
    report["by_solvable"] = _breakdown(built, lambda row: row["extra_info"]["solvable"])
    report["by_error_type"] = _breakdown(built, lambda row: row["extra_info"]["error_type"])
    return built, report


#: Every way a row can be dropped.  Kept as a module constant so the funnel prints
#: a zero for a reason that did not fire instead of silently omitting it.
DROP_REASONS = (
    "malformed_row",
    "unpaired_label_block",
    "identical_question_pair",
    "multi_region_defect",
    "gold_not_verbatim_in_question",
    "gold_has_no_content_word",
    "gold_first_token_not_unique",
    "diag_pool_below_k",
    "diag_not_unique_absent_option",
    "judge_pool_below_k",
)


def _breakdown(rows: list[dict], key) -> dict:
    counter: dict = collections.Counter(key(row) for row in rows)
    return dict(sorted(counter.items(), key=lambda item: str(item[0])))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR, help="directory with train/valid/test.csv")
    parser.add_argument("--split", default=DEFAULT_SPLIT, choices=SPLITS, help="source split to build")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap the number of *pairs* processed (a pair can yield two rows)",
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help="output parquet path")
    parser.add_argument("--report", default=None, help="JSON funnel report (default: alongside --out)")
    parser.add_argument("--seed", type=int, default=0, help="option-sampling seed")
    args = parser.parse_args(argv)
    report_path = args.report or report_path_for(args.out)

    rows, report = build_rows(args.raw_dir, limit=args.limit, seed=args.seed, split=args.split)
    if not rows:
        print("no rows survived the certificates; nothing written", file=sys.stderr)
        return 1
    schema.normalise_extra_info(rows)
    schema.validate_rows(rows)
    schema.write_rows_parquet(rows, args.out)

    print(f"raw dir : {args.raw_dir}  (split={args.split}, seed={args.seed}, limit={args.limit})")
    print(f"wrote   : {args.out}  ({len(rows)} rows)")
    funnel = report["funnel"]
    # The stages are not all in the same unit -- the first three count source
    # rows, the next three count pairs (source index k), the last counts rows
    # again -- so each line is labelled instead of printing a mixed-unit delta.
    unit_of = {
        "raw_rows": "rows",
        "after_malformed_drop": "rows",
        "pairs_available": "pairs",
        "pairs_after_limit": "pairs",
        "pairs_with_region_certificate": "pairs",
        "rows_built": "rows",
    }
    print("\nfunnel (source rows -> index-aligned pairs -> written rows):")
    for stage, count in funnel.items():
        print(f"  {stage:32s} {count:6d}  {unit_of.get(stage, '')}")
    print("\ndrop reasons (a row can only be dropped once):")
    for reason in DROP_REASONS:
        print(f"  {reason:32s} {report['drops'][reason]:6d}")
    print("\nper branch:")
    for branch, count in report["by_branch"].items():
        print(f"  {branch:22s} {count}")
    print("per template:")
    for template, count in report["by_template"].items():
        print(f"  {template:22s} {count}")
    print("per solvable:")
    for solvable, count in report["by_solvable"].items():
        print(f"  {str(solvable):22s} {count}")
    print("per error_type:")
    for error_type, count in report["by_error_type"].items():
        print(f"  {error_type or '(none)':30s} {count}")

    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")
    print(f"\nreport  : {report_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main() in tests
    raise SystemExit(main())
