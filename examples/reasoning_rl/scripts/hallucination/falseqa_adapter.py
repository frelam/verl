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
"""FalseQA adapter -- replacement-pair diagnosis (D21) and a two-layer answerable twin (D27).

Source: ``github.com/thunlp/FalseQA`` ``dataset/{train,valid,test}.csv``, three
columns ``question, answer, label`` (2,374 / 982 / 1,374 rows; each split strictly
50:50).  ``label=1`` is a question whose presupposition is false ("What should men
pay attention to when breastfeeding their child?"), ``label=0`` is the *same
template* with the presupposition repaired ("... women ...").  The two label
blocks are **index-aligned**: within a split, row ``k`` of the ``label=1`` block is
a local rewrite of row ``k`` of the ``label=0`` block.  The file is *blocked*, not
interleaved -- the first N file rows are all ``label=1`` -- so the pairing is
``zip(fake_block, real_block)`` and never ``zip(file, file[:N])``.

Both sides enter the pool (design doc section 4.3, D27, which supersedes D22):
829 ``unsolvable_diag`` rows (table B row 7) and 928 ``solvable_two_layer`` rows
(table B row 3) on the source's ``train`` split -- see deviation 1 for the funnel
and the reason the two sides are not the same size.  The two members of one pair
are twins -- their prompts differ by a single fragment -- so they carry the same
``extra_info.pair_id`` and the mixer keeps them on the same side of the train/val
boundary (``mix_halluc.enforce_pair_atomicity``).

The adapter applies **no quota**: it emits the whole eligible pool.  Section 4.8's
928-per-side figures are the mixer's cells, not a cap here -- ``mix_halluc``
selects, and a build that stopped at 928 would silently discard the rows the
source's other splits contribute to the same artifact.

The diagnosis side: the replacement-pair option contract (D21)
--------------------------------------------------------------

Why this source keeps a four-tier diagnosis (section 4.3): MiP's defect is a
*deletion*, so there is nothing in the prompt to point at (D12); FalseQA's is a
*replacement*, the false fragment is sitting in the question, and the paired real
question says exactly which fragment it is.  The pointer gold is therefore known
by construction -- a word-level ``difflib`` diff (``autojunk=False``) between the
pair's two questions -- and the option block is a set of **replacement pairs**::

    A. men -> women
    B. men -> child
    C. men -> teacher

* **Gold pair** = ``假前提片段 -> 配对真前提片段``: the single visible diff
  region's fake-side text (the fragment the model can point at) and its real-side
  text (the fragment that repairs the premise).  Section 4.3's L1 gates: exactly
  one contiguous visible region, the fake-side fragment occurs verbatim in the
  presented question, it carries a content word, its first token locates it
  unambiguously, the real-side fragment is non-empty and does **not** itself occur
  in the presented question.
* **Distractors (k-1 = 2, D15)** share the gold's *left* item, so the left item
  carries no information, and their right items are **out-of-passage** items of
  the same word count and the same surface type (capitalisation, digit, suffix
  class) as the gold right item -- sampled by rule from a bank of word windows
  taken from the source split's own questions (``build_item_bank``; the absence
  test excludes the row's passage, so a candidate is never taken from the
  question it fills).  The candidates closest in *character* length to the gold
  right item are preferred, so the block's option lengths stay as uninformative as
  the source allows.
* **Anti-shortcut argument (section 4.3).**  All three left items are identical
  -- no information.  All three right items are out-of-passage, so "pick the right
  item that is not in the passage" hits each option equally (it is undefined, not
  merely weak: the audit's L4a measures 0).  Had the distractors' right items come
  from the passage, the gold right item would be the only out-of-passage word and
  that mirror heuristic would win outright -- which is exactly why the pool is
  built out-of-passage and gated (``gold_pair_right_in_passage``).
* **Fail closed.**  A pair whose diff region is not unique splits into no rows; a
  pair whose gold pair fails an L1 gate (including a pure insertion, whose
  real-side fragment is empty) drops its diagnosis row; a pair that cannot field
  ``k-1`` qualifying distractor right items drops its diagnosis row.  Nothing is
  ever padded and no option text ever comes from another *question's* passage
  position: distractors are whole word windows of the corpus, which is what the
  section 4.3 argument requires.

The answerable side: a two-layer reward with an isomorphic placeholder block (D27)
----------------------------------------------------------------------------------

``label=0`` is the repaired twin.  It is a *solvable* row whose gold is the
source's own ``answer`` field, verbatim, scored by the two-layer
``solvable_answer`` branch of section 6: ``\\boxed{<answer>}`` = 0.5 for judging
the question answerable plus 0.5 for a normalised exact match, ``\\boxed{SOLVABLE}``
= 0.5, a refusal = 0.  ``correct_option_id`` is ``None`` -- there is no correct
option -- but the row still carries a **placeholder replacement-pair block** so
that template A looks identical on both sides (section 5.1/5.2, the D18
isomorphism hard constraint): same wording, same k=3, left item an in-passage
content phrase, three right items out-of-passage items of one type and one word
count.  Without it, "the prompt has an option block" would itself give the label
away.

The placeholder's *shape* is mirrored from the pair's own diagnosis side: each
answerable row samples a ``(left word count, right word count, right type, right
character length)`` reference from the region the two members of *its own* pair
differ by, so the two sides' blocks are not merely the same species but drawn from
the same shape distribution, pair by pair.  The block is dropped (never padded)
when the reference cannot be satisfied and the deterministic one-token fallback
shapes cannot either.

Certificates (fail closed; each is re-derived by ``verify_falseqa.py``)
----------------------------------------------------------------------

* **Pair (both sides)** -- the split's ``label=1`` and ``label=0`` blocks have the
  same length, the two questions of a pair differ after whitespace normalisation,
  and the word-level diff ``real -> fake`` has exactly one changed region *visible
  on the fake side* (``autojunk=False``; opcodes whose fake-side token range is
  empty changed nothing the model can see and are excluded before the regions are
  merged -- including them turns 928 certified pairs into 885).
* **Gold pair (diag)** -- the region's fake-side text is the left item and its
  real-side text the right item; the left item must occur verbatim in the
  question, carry a content word, have a first token that occurs exactly once in
  the question, and the right item must be non-empty and absent from the question
  (all three rights out-of-passage is what defuses the absence heuristic).
* **Distractor pool (diag)** -- at least ``k-1`` distinct out-of-passage items
  with the gold right item's word count and type signature; the closest
  character-length tier is preferred and the sample is drawn with the row's seed.
* **Placeholder (answerable)** -- the source's own answer is non-empty, an
  in-passage content span can be mined at the sampled left word count, and ``k``
  distinct out-of-passage items exist at the sampled right type and length.

Determinism
-----------

The same ``(raw_dir, split, limit, seed)`` always produces byte-identical rows:
every random choice is made by a per-row :class:`random.Random` seeded from
``f"{seed}:{branch}:{split}:{index}"``, never by a process-wide RNG, so inserting
a drop anywhere cannot shift the option blocks of later rows.

DEVIATIONS FROM THE DESIGN DOC
------------------------------

Measured on all three splits with ``--seed 0`` (numbers printed by the build and
by ``verify_falseqa.py``; 1,187 / 491 / 687 pairs).

1. **The diagnosis yield is 829 / 336 / 476, not 928 / 377 / 535.**  Section
   4.3's 928 is the number of pairs with a *region* certificate (reproduced
   exactly: 928 / 377 / 535 certified).  The D21 option contract needs a usable
   gold *pair*, and 99 train pairs fail that: 33 are pure insertions on the fake
   side (the real-side fragment is empty, so there is no replacement to offer),
   30 have a non-unique first token, 24 have a real-side fragment that already
   occurs in the presented question (which would make "the right item that *is* in
   the passage" a defined heuristic), 11 have a stop-word-only fake fragment, and
   1 cannot field two qualifying distractor items.  The answerable side needs the
   region certificate and a non-empty source answer, nothing else, so it keeps
   928 / 377 / 535 rows -- exactly the region-certificate count on all three
   splits, since no eligible answer was empty and no placeholder pool came up
   short.  The two pools are deliberately not forced to the same size; the twins
   stay linked by ``pair_id``.

2. **The ``label=0`` answer is now the *gold*, not an audit field.**  D22's
   judgment-only reading stored it in ``ground_truth.answer`` for provenance but
   scored only the verdict.  D27 puts the free text on the two-layer
   ``solvable_answer`` branch, so the row is dropped when the field is empty
   (0 of the eligible train pairs) and the stored value is the raw field with
   surrounding whitespace stripped.  67.8% of the source's answerable answers are
   free text, which section 6 scores with ``norm_match``; the paraphrase noise
   (a correct answer that fails normalised exact match loses 0.5 of 1.0) is the
   documented trade-off of Q17 and the reason the answer layer is only worth half.

3. **The placeholder block mirrors the pair's own diagnosis-side shape.**  D27
   fixes the text ("in-passage left, out-of-passage same-type same-length rights")
   but not the word-count relation between the left and right items.  On the
   diagnosis side that relation is data (the fake fragment is a rewrite of the
   real one; measured 69.7% same word count), so hard-coding "rights match the
   left" on the answerable side would make that relation a format cue.  Each
   answerable row instead mirrors the shape of its own twin's region, which
   removes the difference pair by pair and -- unlike a pool-wide reference --
   keeps a ``--limit`` slice byte-identical to the full build's prefix.

4. **The distractor right items are corpus word windows, not a hand-written word
   list.**  Section 4.3 says "从题外词表规则采样"; the "word list" here is every
   1..10-token window of the same source split, filtered to items that do not
   occur in the presented question and match the gold right item's word count and
   surface type.  The rule is therefore reproducible, has no external dependency,
   and -- because the bank is the source's own vocabulary -- keeps the distractors
   in the same register as the gold.  Section 9's N=50 manual sample is what
   checks whether a distractor also repairs the premise (risk 6).

5. **The type signature is a surface proxy.**  The doc asks for the distractors to
   be "同类" (same part of speech / entity type) as the gold right item.  No POS
   tagger is available offline, so ``item_signature`` uses word count,
   digit-bearing, capitalisation class and a coarse suffix class
   (``-ing``/``-ed``/``-ly``/``-tion``/``-er``/plural ``-s``/…).  It over-splits
   rather than under-splits: a candidate that fails the signature is rejected even
   when a human would call it the same kind of word, so the residual risk is
   distractor *quality* (risk 6), never gate leakage.

6. **The character-length tier is what keeps the length cue at chance, and a
   corpus-frequency cue remains.**  Drawing the two distractors from the whole
   candidate pool would make the gold right item the unique longest option on 82%
   of rows; restricting the draw to the candidates closest in character length to
   the gold brings the "pick the longest option" heuristic to 0.31 (chance) with a
   within-block length spread of 1.00 (median).  The pool is still drawn from a
   corpus, so the gold right item's *word frequency* is not neutralised: the audit
   measures "pick the most frequent right item" at 0.46 train / 0.42 test against
   the 1/3 baseline and prints it as an informational reading rather than a gate
   (section 9's gate list does not include it; §12 Q9's N=50 manual sample is the
   hook the doc provides for distractor quality).  Neutralising it would mean a
   further document-frequency band, which is deliberately not applied: it moves
   the cue to "pick the rarest item" rather than removing it (measured 0.48-0.56
   when the band is added) and costs ~4% of the diagnosis yield.
"""

from __future__ import annotations

import argparse
import collections
import csv
import difflib
import json
import os
import random
import sys

try:
    import schema
    from distractor_mining import token_spans
except ImportError:  # pragma: no cover - running as a plain script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import schema
    from distractor_mining import token_spans

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

#: Table B row 7 (unsolvable, four-tier diagnosis) and row 3 (solvable, two-layer
#: reward) -- see the branch registry of ``schema.py``.
BRANCH_DIAG = schema.BRANCH_UNSOLVABLE_DIAG
BRANCH_ANSWERABLE = schema.BRANCH_SOLVABLE_TWO_LAYER

K_OPTIONS = 3  # design decision D15
#: How many source rows are scanned for out-of-passage items.  The observed gold
#: right items run 1-9 tokens (measured), so 10 is a safety margin, not a knob.
MAX_ITEM_TOKENS = 10
#: Placeholder attempts spent sampling a reference shape before the deterministic
#: fallback list is walked (see ``mine_placeholder_options``).
PLACEHOLDER_ATTEMPTS = 4

#: ``label`` column values: ``"1"`` is the false presupposition (unsolvable),
#: ``"0"`` the paired true one.  The column is an unquoted single character.
LABEL_FALSE = "1"
LABEL_TRUE = "0"

#: A false presupposition that *can* be pointed at in the prompt -- the D18 defect
#: slot FalseQA fills (section 4.8 table B, 假前提（可指认，替换对）).  The slug is
#: deliberately distinct from ``false_premise_unpointable``, which CREPE carries:
#: the balance table keys on these strings.
ERROR_TYPE = "false_premise_pointable"

#: Section 4.3's delivery constraint: the false fragment contradicts the question's
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

    One ``test`` question is not equal to its own ``.strip()``, and every offset
    in this module is computed on the *normalised* text, which is also the text
    rendered into the prompt -- the two must never be mixed, or the gold offsets
    and the visible question drift apart.
    """
    return " ".join((text or "").split())


class _Region:
    """One contiguous run of changed word tokens between the two questions.

    ``a_*`` is the real (``label=0``) side, ``b_*`` the fake (``label=1``) side,
    so the diff is always ``real -> fake``: the left item of the gold pair is the
    fake-side text, the right item the real-side text that repairs it.  ``n_a`` /
    ``n_b`` are the two sides' token counts.
    """

    __slots__ = ("tag", "a_text", "b_text", "a_start", "b_start", "n_a", "n_b")

    def __init__(self, **kwargs) -> None:
        for key in self.__slots__:
            setattr(self, key, kwargs.get(key, ""))


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
    into 885.
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

    Diffing *token sequences* (``autojunk=False``) rather than characters keeps
    every fragment a whole word span.  The returned regions are already merged,
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
    option block is read off :func:`word_regions`, never off this pair of strings:
    on the train pairs that carry an extra invisible deletion the two disagree,
    and the pointer must be the fragment the model can see.
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
# the out-of-passage item bank and the surface type signature
# ---------------------------------------------------------------------------

#: Suffix classes, longest suffix first.  A coarse, offline stand-in for the part
#: of speech the design doc's "同类" asks for (deviation 5): ``-ing`` is a
#: participle, ``-er``/``-or`` an agent noun, ``-s`` a plural, and so on.  The
#: table is paired with the same table re-declared in ``verify_falseqa.py`` --
#: deliberately, so the audit's notion of "same type" is not read out of the code
#: it audits.
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


def _capitalisation_class(tokens: list[str]) -> str:
    """``lower`` / ``mixed`` / ``title`` -- a proper-noun proxy for the item."""
    caps = [token[0].isupper() for token in tokens if token]
    if not caps:
        return "lower"
    if all(caps):
        return "title"
    return "mixed" if any(caps) else "lower"


def item_signature(text: str) -> tuple[int, bool, str, str]:
    """The surface type of an item: ``(words, digit-bearing, caps, suffix)``.

    Two items with the same signature are the adapter's notion of "same type": a
    gold right item ``women`` and a distractor ``adults`` agree, ``rainy days``
    and ``Academy of`` do not.  Word count is part of the signature, so the
    ``same word count`` rule of section 4.3 is the first coordinate of the type
    rule rather than a separate check.
    """
    tokens = schema.words(text)
    return (
        len(tokens),
        any(ch.isdigit() for ch in text),
        _capitalisation_class(tokens),
        _suffix_class(tokens[-1]) if tokens else "plain",
    )


def build_item_bank(questions: list[str], max_tokens: int = MAX_ITEM_TOKENS) -> dict:
    """``signature -> {casefolded text: text}`` over every window of the corpus.

    The bank is the source split's own vocabulary of 1..``max_tokens``-token word
    windows (whitespace-normalised, verbatim slices, so the item keeps the
    corpus's spelling).  Sampling from a bank rather than from a hand-written list
    is what makes "out-of-passage" a *rule*: any window that does not occur in the
    presented question is admissible, so the distractor set is never a small fixed
    vocabulary the model could memorise.
    """
    bank: dict[tuple, dict[str, str]] = collections.defaultdict(dict)
    for question in questions:
        for size in range(1, max_tokens + 1):
            for span in token_spans(question, size):
                text = span.text.strip()
                if len(schema.words(text)) != size:
                    continue
                bank[item_signature(text)].setdefault(text.casefold(), text)
    return bank


def out_of_passage_right_pool(
    bank: dict,
    *,
    signature: tuple,
    char_len: int,
    passage: str,
    exclude: str,
    minimum: int,
) -> list[str] | None:
    """Items matching ``signature`` that do not occur in ``passage``, best first.

    ``exclude`` is the gold right item (it lives in the bank -- it is a window of
    the paired real question -- and must not be offered as its own distractor).
    ``passage`` is tested case-insensitively as a substring: a candidate that only
    occurs inside a longer word counts as present, which is the conservative
    reading of "题面外".

    The returned pool is the *closest character-length tier* when it is large
    enough, and the whole candidate list otherwise.  Tiering is what keeps the
    block's option lengths uninformative: with the tier, the gold right item is
    the unique longest option on 3% of rows instead of 82%, so "pick the longest"
    collapses to the first-index tie-break and measures at chance.
    """
    items = bank.get(signature) or {}
    folded_passage = passage.casefold()
    folded_exclude = exclude.strip().casefold()
    candidates = [
        text
        for folded, text in items.items()
        if folded not in folded_passage and folded != folded_exclude
    ]
    if len(candidates) < minimum:
        return None
    candidates.sort(key=lambda text: (abs(len(text) - char_len), text.casefold()))
    best_delta = abs(len(candidates[0]) - char_len)
    tier = [text for text in candidates if abs(len(text) - char_len) == best_delta]
    return tier if len(tier) >= minimum else candidates


def mine_content_spans(question: str, n_tokens: int) -> list[str]:
    """Distinct ``n_tokens``-token spans of ``question`` carrying a content word.

    The answerable side's left item is not a defect: it is an ordinary in-passage
    phrase, so the only requirements are that it is a verbatim slice of the
    question (it is, by construction) and that it is not stop-word filler -- the
    same content-word rule the diagnosis side's fake fragment has to pass.
    """
    unique: dict[str, str] = {}
    if n_tokens <= 0:
        return []
    for span in token_spans(question, n_tokens):
        text = span.text.strip()
        if not any(schema.is_content_word(token) for token in schema.words(text)):
            continue
        unique.setdefault(text.casefold(), text)
    return [unique[key] for key in sorted(unique)]


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


def gold_pair_problem(real_q: str, fake_q: str, regions: list[_Region]) -> str | None:
    """The first D21 gold-pair certificate this pair fails, or ``None``.

    Returning the reason (rather than a boolean) is what lets ``build_rows`` print
    a drop table that says *why* a pair lost its diagnosis row.  The names are
    the ``DROP_REASONS`` entries.  The gold pair is the single visible region:
    left = the fake fragment the model can point at, right = the real fragment
    that repairs it.
    """
    if len(regions) != 1:
        return "multi_region_defect"
    region = regions[0]
    left, right = region.b_text.strip(), region.a_text.strip()
    if not left or left not in fake_q:
        return "gold_not_verbatim_in_question"
    tokens = schema.words(left)
    if not tokens or not any(schema.is_content_word(token) for token in tokens):
        return "gold_has_no_content_word"
    first = tokens[0].casefold()
    if sum(1 for token in schema.words(fake_q) if token.casefold() == first) != 1:
        return "gold_first_token_not_unique"
    if not right:
        # A pure insertion: the fake question gained a fragment, so there is no
        # replacement to offer and the pair cannot carry a replacement-pair gold.
        return "gold_pair_right_empty"
    if right.casefold() in fake_q.casefold():
        # The repairing fragment already occurs in the presented question.  All
        # three right items must be out-of-passage (section 4.3), otherwise "pick
        # the right item that *is* in the passage" identifies the gold outright.
        return "gold_pair_right_in_passage"
    return None


def certify_gold_pair(real_q: str, fake_q: str, regions: list[_Region]) -> tuple[str, str] | None:
    """``(fake fragment, real fragment)`` for a certified pair, or ``None``."""
    if gold_pair_problem(real_q, fake_q, regions) is not None:
        return None
    region = regions[0]
    return region.b_text.strip(), region.a_text.strip()


# ---------------------------------------------------------------------------
# row construction
# ---------------------------------------------------------------------------


def _task_id(side: str, split: str, index: int) -> str:
    """Stable hard-replay key; carries the side because both sides are written."""
    return f"falseqa-{side}-{split}-{index}"


def pair_id(split: str, index: int) -> str:
    """The D27 pair identity shared by a pair's two rows.

    ``mix_halluc.enforce_pair_atomicity`` groups val rows by this value, so the
    answerable and unanswerable twins of one index-aligned pair never land on
    opposite sides of the train/val boundary (design doc section 4.3/9).  The
    source split is part of the id because index ``k`` of two different CSVs names
    two different pairs.
    """
    return f"{split}:{index}"


def _base_extra_info(split: str, index: int, seed: int, partner_question: str) -> dict:
    return {
        "split": split,
        "index": index,
        "seed": seed,
        "pair_id": pair_id(split, index),
        "paired_original_text": partner_question,
        "perturbation_family": "",
        "difficulty": "",  # the source carries no difficulty axis
    }


def _build_diag(
    *,
    split: str,
    index: int,
    seed: int,
    question: str,
    partner_question: str,
    options: list[dict],
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


def _build_answerable(
    *,
    split: str,
    index: int,
    seed: int,
    question: str,
    partner_question: str,
    options: list[dict],
    answer: str,
    deleted: str,
    inserted: str,
) -> dict:
    """One ``solvable_two_layer`` row: gold is the source's own label=0 answer.

    The option block is a *placeholder*: no option is correct
    (``correct_option_id`` is ``None``), and the reward never reads it.  It exists
    so template A has one appearance on both sides of the source (D18).
    """
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
            "has_diagnosis_label": False,
        }
    )
    return schema.make_row(
        data_source=DATA_SOURCE,
        question=question,
        ground_truth=schema.build_ground_truth(
            solvable=True,
            answer=answer,
            correct_option_id=None,
            has_diagnosis_label=False,
            perturbation_type=None,
            solvable_answer=True,
        ),
        template=schema.TEMPLATE_A,
        branch=BRANCH_ANSWERABLE,
        extra_info=extra,
        options=options,
    )


def _reference_of(left_tokens: int, right: str) -> tuple[int, int, tuple, int] | None:
    """The diagnosis-side shape an answerable placeholder block mirrors.

    ``(left words, right words, right type, right character length)`` -- exactly
    the four coordinates a diagnosis block's shape is made of.  The reference is
    taken from the pair's **own** twin (the region the two questions differ by), so
    the two rows of one pair share a shape and the answerable side's block
    distribution matches the diagnosis side's by construction.  Coupling it to the
    rest of the build instead would make a ``--limit`` slice differ from the full
    build's prefix -- the property the slice builds rely on -- which is why the
    mirroring is pairwise rather than pool-wide (deviation 3 of the module
    docstring).

    ``None`` means the pair has no usable reference (an empty repairing fragment):
    the caller then walks the deterministic one-token fallback shapes.
    """
    right = right.strip()
    if not right or left_tokens <= 0:
        return None
    signature = item_signature(right)
    return left_tokens, signature[0], signature, len(right)


def _fallback_references(bank: dict, minimum: int) -> list[tuple[int, int, tuple, int]]:
    """One-token reference shapes for a question whose sampled shapes fail.

    Ordered by descending pool size and then by signature, so the choice is
    deterministic and the first entry is the type class the corpus offers most of.
    """
    references: list[tuple[int, int, tuple, int]] = []
    groups = [(len(items), signature) for signature, items in bank.items() if signature[0] == 1]
    for size, signature in sorted(groups, key=lambda entry: (-entry[0], entry[1])):
        if size < minimum:
            continue
        lengths = sorted(len(text) for text in bank[signature].values())
        references.append((1, 1, signature, lengths[len(lengths) // 2]))
    return references


def _placeholder_from_reference(
    question: str,
    bank: dict,
    reference: tuple[int, int, tuple, int],
    rng: random.Random,
    k: int,
) -> tuple[list[dict], str, list[str]] | None:
    """Mine one placeholder block for one reference shape, or ``None``."""
    n_left, n_right, signature, char_len = reference
    if signature[0] != n_right:
        return None
    lefts = mine_content_spans(question, n_left)
    if not lefts:
        return None
    pool = out_of_passage_right_pool(
        bank, signature=signature, char_len=char_len, passage=question, exclude="", minimum=k
    )
    if pool is None:
        return None
    left = rng.choice(lefts)
    rights = rng.sample(pool, k)
    texts = [schema.pair_option_text(left, right) for right in rights]
    return schema.shuffle_options(texts, rng), left, rights


def mine_placeholder_options(
    question: str,
    bank: dict,
    references: list[tuple[int, int, tuple, int]],
    seed_prefix: str,
    k: int = K_OPTIONS,
    attempts: int = PLACEHOLDER_ATTEMPTS,
) -> tuple[list[dict], str, list[str]] | None:
    """The answerable side's placeholder replacement-pair block, or ``None``.

    ``references`` holds the shape the pair's own region implies (at most one
    entry, and empty when the pair has no usable region).  The first ``attempts``
    tries re-sample the left span and the distractors with the row's seed; if the
    shape cannot be satisfied at all the deterministic one-token fallback list is
    walked, and a row that still cannot field a block is dropped rather than
    padded.

    Returns ``(options, left, rights)``.
    """
    for attempt in range(attempts):
        rng = random.Random(f"{seed_prefix}:{attempt}")
        if not references:
            break
        mined = _placeholder_from_reference(question, bank, rng.choice(references), rng, k)
        if mined is not None:
            return mined
    rng = random.Random(f"{seed_prefix}:fallback")
    for reference in _fallback_references(bank, k):
        mined = _placeholder_from_reference(question, bank, reference, rng, k)
        if mined is not None:
            return mined
    return None


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
            number of rows.  A pair can yield one row (either side) or two, so
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

    # The item bank is the split's own vocabulary; the passage test excludes the
    # presented question, so a candidate is never taken from the row it fills.
    # It is built from every well-formed row, *before* ``limit`` is applied to the
    # pairs, so a limited build is byte-identical to the full build's prefix.
    bank = build_item_bank([normalise_question(row["question"]) for row in well_formed])

    built: list[dict] = []
    certified = 0
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
        real_answer = (real_row.get("answer") or "").strip()
        region = regions[0]
        left, right = region.b_text.strip(), region.a_text.strip()
        # The reference the answerable side's placeholder mirrors: the pair's own
        # shape, whether or not the diagnosis side can use it as a gold.
        reference = _reference_of(region.n_b, right)

        problem = gold_pair_problem(real_q, fake_q, regions)
        if problem is not None:
            # The pair still has a region certificate, so the answerable side can
            # be built; only the diagnosis side loses its pointer gold.
            drops[problem] += 1
        else:
            pool = out_of_passage_right_pool(
                bank,
                signature=item_signature(right),
                char_len=len(right),
                passage=fake_q,
                exclude=right,
                minimum=K_OPTIONS - 1,
            )
            if pool is None:
                drops["pair_distractors_below_k"] += 1
            else:
                rng = random.Random(f"{seed}:{BRANCH_DIAG}:{split}:{index}")
                mined = schema.build_pair_options(left, right, pool, K_OPTIONS, rng)
                if mined is None:  # pragma: no cover - the pool gate already holds
                    drops["pair_distractors_below_k"] += 1
                else:
                    options, correct = mined
                    built.append(
                        _build_diag(
                            split=split,
                            index=index,
                            seed=seed,
                            question=fake_q,
                            partner_question=real_q,
                            options=options,
                            correct=correct,
                            deleted=deleted,
                            inserted=inserted,
                        )
                    )

        if not real_answer:
            drops["answer_empty"] += 1
        else:
            rng_prefix = f"{seed}:{BRANCH_ANSWERABLE}:{split}:{index}"
            mined = mine_placeholder_options(
                real_q,
                bank,
                [reference] if reference is not None else [],
                rng_prefix,
                K_OPTIONS,
            )
            if mined is None:
                drops["placeholder_pool_below_k"] += 1
            else:
                options, _left, _rights = mined
                built.append(
                    _build_answerable(
                        split=split,
                        index=index,
                        seed=seed,
                        question=real_q,
                        partner_question=fake_q,
                        options=options,
                        answer=real_answer,
                        deleted=deleted,
                        inserted=inserted,
                    )
                )

    built.sort(key=lambda row: (row["extra_info"]["index"], row["extra_info"]["task_id"]))
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
#: a zero for a reason that did not fire instead of silently omitting it.  The
#: first four are pair-level (both rows go), the rest drop one side.
DROP_REASONS = (
    "malformed_row",
    "unpaired_label_block",
    "identical_question_pair",
    "multi_region_defect",
    "gold_not_verbatim_in_question",
    "gold_has_no_content_word",
    "gold_first_token_not_unique",
    "gold_pair_right_empty",
    "gold_pair_right_in_passage",
    "pair_distractors_below_k",
    "answer_empty",
    "placeholder_pool_below_k",
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
