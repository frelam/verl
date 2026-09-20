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
"""UMWP adapter -- native answerable/unanswerable pairs (design doc D24, section 4.6).

Source: ``github.com/Yuki-Asuuna/UMWP``, ``data/StandardDataset.jsonl`` (one file,
5,200 rows, CC-BY-SA-4.0).  Every row is one of two members of a *native* pair:
positions 0-2599 are ``answerable=true`` with a 1-element float ``answer``, and
each unanswerable row carries ``relevant_ids=[<the answerable row's id>]``.  That
link -- not ``id - 2600``, which the recon measured to fail on 2600/2600 rows --
is how the partner is recovered here.

Contract produced by this adapter (design doc section 4.8 table B, rows 3 and 8):

=====================  ========  ================================================
branch                 template  gold
=====================  ========  ================================================
``solvable_two_layer`` A         ``\\boxed{<answer>}`` (two-layer reward, D24)
``unsolvable_bare``    B         ``\\boxed{UNSOLVABLE}``
=====================  ========  ================================================

* **Answerable side (table B row 3, 550 selected by the mix).**  ``solvable=true``,
  ``two_layer=true``, ``answer`` = the source's own number,
  ``correct_option_id=null``, ``has_diagnosis_label=false``.  The prompt is
  template A with a k=3 **placeholder** option block: no option is correct and
  the reward never reads the block, but section 5.2 / D18 require every template-A
  row to look alike, so the block is still k=3 equal-token-length spans of *this*
  question (cross-question text is the 99% "the option that is not in the
  question" shortcut of section 4.4).
* **Unanswerable side (table B row 8, 2,489 selected by the mix).**  Every defect
  class is three-tier bare: ``solvable=false``, ``has_diagnosis_label=false``,
  ``correct_option_id=null``, template B, **no option block at all**
  (section 5.1).  D24 retired the visible-defect four-tier rows ("可见缺陷类的同题
  等长跨度选项整体废弃"), so a pair's class is used only for the section 4.8
  defect-type table and the section 10 monitoring -- it no longer decides the
  contract branch.

``category`` is therefore *not* a filter: a pair the source leaves unlabelled
(``category is None``; section 4.8's 200 "无类别标注" rows) is admitted with no
``perturbation_type`` instead of being dropped.  A category code *outside* the
frozen table is still a hard drop -- that means the file changed shape, not that
the row is unlabelled.

Certificates (fail closed; each is re-derived by ``verify_umwp.py``)
-------------------------------------------------------------------

* **Pairing / residual content** -- the two members are the same problem modulo
  the recorded edit: drop every token inside the changed regions from both
  questions and the remaining token sequences are identical
  (:func:`_residual_tokens`).  This is the only hard gate left.  It admits the
  native pairs whose defect is character-level and therefore invisible to a token
  diff -- category 3's ``"23" -> "-23"`` sign flip, 187 pairs per side -- which
  the bare refusal contract does not need to point at.
* **Refusal shape** -- :func:`_certify_refusal` is kept as the *classifier* of an
  unlabelled pair: a removed quantity (its category-1 rule) or a cut question
  clause (its category-5 rule) names the defect class when the source did not.
  It is no longer a drop rule: every class ships bare (D24).
* **Placeholder block** -- the answerable side ships only when its question yields
  ``k`` distinct equal-token-length content spans of its own; a question that
  cannot is dropped rather than padded with cross-question text.

Everything is derived from the row's own pair by a **word-level** ``difflib``
diff (with a character-level fallback for the token-invisible pairs), and every
gold is re-derivable from the artifact alone (``verify_umwp.py`` does exactly
that).  A row whose question cannot supply a placeholder block is dropped and
counted in the funnel -- nothing is invented and no option text ever comes from
another question.

Measured funnel (full ``StandardDataset.jsonl``, seed 0; the mix then selects the
section 4.8 quotas from this pool)
------------------------------------------------------------------------------------------------

=========================  ======
raw rows                   5,200
after malformed drop       5,200
after stray-answer drop    5,198
after duplicate drop       5,183
after pair resolution      5,176  (2,588 native pairs; both members ship)
after defect-class drop    5,176  (every unanswerable row is labelled 1..5)
after pair certificate     5,176  (the 187 token-invisible pairs ship bare)
after placeholder mining   5,176  (every answerable question yields a block)
=========================  ======

So the eligible pool is **2,588 answerable / 2,588 unanswerable**, against the
mix's 550 / 2,489 quota.  Defect-class split of the unanswerable side:
cat1 834 / cat2 1,259 / cat3 273 / cat4 103 / cat5 119 / unlabelled 0.  The
section 4.8 table's per-class numbers (840 / 1,040 / 226 / 85 / 98 plus 200
unlabelled) are the recon's *scaled* estimate, not this file's counts; the class
cells matter for monitoring, not for the contract.

DEVIATIONS FROM THE RECON / DESIGN DOC
--------------------------------------

1. **The diff must be word-level.**  The recon's UMWP shapes were measured with
   character-level ``get_opcodes``.  On normalized text a char-level diff splits
   words (measured gold spans such as ``'Som'``, ``'ome'``, ``'-1'`` -- mid-word
   fragments) because UMWP repeats words inside one question ("...46 magazines
   ... How many magazines..."), which lets ``difflib`` align a repeated word to
   the wrong occurrence.  This adapter diffs **token sequences** and maps the
   resulting opcodes back to character offsets, so every recorded span is a whole
   word span; :func:`character_defect` is the fallback for the pairs whose tokens
   are identical.
2. **The source has no ``difficulty``, ``split`` or ``index`` field.**  The recon
   (section 1.3) found one implicit split and no split column, so ``split`` is
   ``"train"`` for every row and ``index`` is the source's own ``id``.  UMWP's
   native difficulty axis does not exist; ``extra_info.difficulty`` records the
   *base problem's* origin (GSM8K / SVAMP / MultiArith / ASDiv), which is the only
   native per-row stratum the file carries.  It is a provenance label, not a
   measured difficulty.
3. **The section 4.8 / D24 quota arithmetic does not match the file.**  The doc's
   2,489 answers = 2,289 labelled + 200 unlabelled; this file labels **all** 2,600
   unanswerable rows with an integer 1..5, so a real build measures 0 unlabelled
   rows and 2,588 labelled ones after the pair drops.  The adapter still admits an
   unlabelled row (with ``perturbation_type=None``) because that is the contract
   the section 4.8 note asks for; it is exercised by the unit tests, not by this
   snapshot of the source.

Module note on L1
-----------------

UMWP's labels are human-constructed and the file ships no derivation chain -- the
recon flagged the source as "⚠️ 人工构造" (``scratch/halluc_recon/umwp_report.md``).
The pairing certificate proves that the recorded defect is the only difference
between the two members, and the source anchor in ``verify_umwp.py`` proves the
answerable gold is the source's own number.  Neither proves that a category-2
replacement really makes the problem unanswerable -- no mechanical witness for
that exists in this source, and the adapter does not pretend otherwise.
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

try:
    import schema
except ImportError:  # pragma: no cover - running as a plain script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import schema

# ---------------------------------------------------------------------------
# source constants
# ---------------------------------------------------------------------------

DATA_SOURCE = schema.SOURCE_UMWP
DEFAULT_RAW_DIR = "/home/charles/data/reasoning_rl/halluc/raw/umwp"
DEFAULT_OUT = os.path.expanduser("~/data/reasoning_rl/halluc/built/umwp.parquet")
DATA_FILE = "StandardDataset.jsonl"

BRANCH_TWO_LAYER = schema.BRANCH_SOLVABLE_TWO_LAYER
BRANCH_BARE = schema.BRANCH_UNSOLVABLE_BARE

K_OPTIONS = 3  # design decision D15

# category code -> (error_type for the section 4.8 defect-type table, perturbation_type).
# D24: the class is *statistics only* -- every class ships on the bare branch, so
# this table no longer routes anything.
CATEGORIES: dict[int, tuple[str, str]] = {
    1: ("key_information_missing", "missing_condition"),
    2: ("ambiguous_key_information", "ambiguous_condition"),
    3: ("unrealistic_conditions", "unrealistic_condition"),
    4: ("unrelated_object", "unrelated_entity"),
    5: ("question_missing", "question_missing"),
}

#: Defect record of a pair the source left unlabelled: outside the section 4.8
#: defect-type table, so it carries no ``error_type`` and no ``perturbation_type``.
UNLABELLED_DEFECT: tuple[str, str | None] = ("", None)

_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’\-]*")


# ---------------------------------------------------------------------------
# text plumbing
# ---------------------------------------------------------------------------


def normalise_question(text: str) -> str:
    """Collapse whitespace and strip.

    UMWP's questions carry leading/trailing spaces, doubled spaces and doubled
    periods from the way the problem and the question sentence were concatenated
    (recon section 5.2 item 6).  Every offset in this module is computed on the
    *normalised* text, which is also the text that is rendered into the prompt --
    the recon's warning is that offsets computed on the raw string break once the
    string is stripped, and the fix is to never mix the two.
    """
    return _WHITESPACE_RE.sub(" ", (text or "")).strip()


def _tokens(text: str) -> list[tuple[str, int, int]]:
    """``[(token, start, end)]`` -- the same token grammar as distractor_mining."""
    return [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


class _Region:
    """One contiguous run of changed word tokens between two questions."""

    __slots__ = ("a_text", "b_text", "a_start", "b_start", "a_expanded", "b_expanded", "tag")

    def __init__(self, **kwargs) -> None:
        for key in self.__slots__:
            setattr(self, key, kwargs.get(key, ""))

    @property
    def inserted(self) -> str:
        """Text present in the unanswerable member and not in the answerable one."""
        return self.b_text


def _merge_opcodes(opcodes: list[tuple]) -> list[tuple]:
    """Join adjacent non-equal opcodes into one changed run.

    ``difflib`` already reports a two-sided change as a single ``replace``, so in
    practice this only ever drops the ``equal`` opcodes; the join is kept because
    a hand-built or future opcode list (a ``delete`` immediately followed by an
    ``insert`` at the same index) must not turn one defect into two regions.
    """
    runs: list[tuple] = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        if runs and runs[-1][2] == i1 and runs[-1][4] == j1:
            prev_tag, oi1, _, oj1, _ = runs[-1]
            runs[-1] = (prev_tag, oi1, i2, oj1, j2)
        else:
            runs.append((tag, i1, i2, j1, j2))
    return runs


def word_regions(qa: str, qu: str) -> list[_Region]:
    """Merged runs of adjacent, non-equal *token* opcodes of ``qa -> qu``.

    Diffing token sequences (rather than characters) is what keeps every recorded
    span a whole word span; ``a_start`` / ``b_start`` are character offsets into
    the two strings, and ``a_expanded`` additionally covers the punctuation and
    whitespace between the surrounding equal tokens, which is what the "was the
    question clause touched?" classification needs.
    """
    ta, tb = _tokens(qa), _tokens(qu)
    matcher = difflib.SequenceMatcher(None, [t[0] for t in ta], [t[0] for t in tb], autojunk=False)
    runs = _merge_opcodes(matcher.get_opcodes())

    regions: list[_Region] = []
    for tag, i1, i2, j1, j2 in runs:
        a_start = ta[i1][1] if i1 < len(ta) else len(qa)
        a_end = ta[i2 - 1][2] if i2 > i1 else a_start
        b_start = tb[j1][1] if j1 < len(tb) else len(qu)
        b_end = tb[j2 - 1][2] if j2 > j1 else b_start
        expanded_a_start = ta[i1 - 1][2] if i1 > 0 else 0
        expanded_a_end = ta[i2][1] if i2 < len(ta) else len(qa)
        expanded_b_start = tb[j1 - 1][2] if j1 > 0 else 0
        expanded_b_end = tb[j2][1] if j2 < len(tb) else len(qu)
        regions.append(
            _Region(
                tag=tag,
                a_text=qa[a_start:a_end],
                b_text=qu[b_start:b_end],
                a_start=a_start,
                b_start=b_start,
                a_expanded=qa[expanded_a_start:expanded_a_end],
                b_expanded=qu[expanded_b_start:expanded_b_end],
            )
        )
    return regions


def _residual_tokens(qa: str, qu: str, regions: list[_Region]) -> bool:
    """Whether the two questions agree outside the changed regions.

    The token-level restatement of "the regions are the whole difference": drop
    every token inside a changed region from both sides and the remaining token
    sequences must be identical.  With no regions at all this says the two token
    sequences are equal, which is exactly the character-level-defect case
    (``"23" -> "-23"``) that the bare side admits.  Re-derivable by the verifier
    from the artifact.
    """

    def residual(text: str, spans: list[tuple[int, int]]) -> list[str]:
        keep = []
        for token, start, end in _tokens(text):
            if any(lo <= start and end <= hi for lo, hi in spans):
                continue
            keep.append(token)
        return keep

    spans_a = [(r.a_start, r.a_start + len(r.a_text)) for r in regions]
    spans_b = [(r.b_start, r.b_start + len(r.b_text)) for r in regions]
    return residual(qa, spans_a) == residual(qu, spans_b)


def character_defect(qa: str, qu: str) -> tuple[str, str]:
    """Character-level ``(deleted, inserted)`` for a pair whose tokens are identical.

    The source records defects that a token diff cannot see: category 3 flips a
    sign (``"23" -> "-23"``) or swaps a one-character number, which leaves the
    token sequences equal.  Such a pair is still a native pair and the bare
    contract only needs the source's refusal label, so it ships -- this keeps the
    artifact's audit fields honest instead of empty.  Note that a character-level
    edit never changes the token multiset (an inserted character that started a
    token would have changed the token sequence and produced a word region), so
    the token-level audit in ``verify_umwp.py`` still applies.
    """
    matcher = difflib.SequenceMatcher(None, qa, qu, autojunk=False)
    deleted, inserted = [], []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        deleted.append(qa[i1:i2])
        inserted.append(qu[j1:j2])
    return "".join(deleted).strip(), "".join(inserted).strip()


def defect_texts(qa: str, qu: str, regions: list[_Region]) -> tuple[str, str]:
    """The pair's recorded ``(deleted, inserted)`` audit text (section 3 keys)."""
    deleted = " ".join(region.a_text for region in regions if region.a_text.strip())
    inserted = " ".join(region.b_text.strip() for region in regions if region.b_text.strip())
    if deleted or inserted:
        return deleted, inserted
    return character_defect(qa, qu)


def _question_clause(question: str) -> str:
    """The final interrogative sentence of ``question`` ("" when there is none)."""
    mark = question.rfind("?")
    if mark < 0:
        return ""
    start = question.rfind(".", 0, mark)
    return question[start + 1 : mark + 1]


# ---------------------------------------------------------------------------
# source loading and pairing
# ---------------------------------------------------------------------------


def load_source(raw_dir: str) -> list[dict]:
    """Parse ``StandardDataset.jsonl``; malformed lines fail closed.

    A line that is not a JSON object is returned as-is so the caller can count and
    drop it, rather than crashing the build.
    """
    path = os.path.join(raw_dir, DATA_FILE)
    with open(path, encoding="utf-8") as handle:
        rows = []
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                rows.append({"_unparseable": line[:200]})
        return rows


def _is_well_formed(row: object) -> bool:
    if not isinstance(row, dict):
        return False
    if not isinstance(row.get("id"), int):
        return False
    if not isinstance(row.get("question"), str) or not row["question"].strip():
        return False
    if not isinstance(row.get("answerable"), bool):
        return False
    return True


def _partner_map(rows: list[dict]) -> tuple[dict[int, dict], dict[int, dict]]:
    """``id -> row`` plus ``answerable id -> its unanswerable partner``.

    The link is the unanswerable row's own ``relevant_ids`` (a 1-element list);
    the arithmetic shortcut ``id - 2600`` is not used because the recon measured it
    to fail on every row (the id space is permuted).  An ambiguous partner
    (multiplicity > 1) is dropped from the map, which makes the caller drop both
    members -- fail closed rather than pick a side.
    """
    by_id = {row["id"]: row for row in rows if _is_well_formed(row)}
    partners: dict[int, dict] = {}
    ambiguous: set[int] = set()
    for row in by_id.values():
        if row["answerable"]:
            continue
        linked = row.get("relevant_ids")
        if not (isinstance(linked, list) and len(linked) == 1):
            continue
        target = linked[0]
        if target not in by_id or not by_id[target]["answerable"]:
            continue
        if target in partners:
            ambiguous.add(target)
            continue
        partners[target] = row
    for target in ambiguous:
        partners.pop(target, None)
    return by_id, partners


# ---------------------------------------------------------------------------
# certificates
# ---------------------------------------------------------------------------


def _certify_refusal(qa: str, qu: str, regions: list[_Region], category: int) -> bool:
    """Whether "this question lost what it needs" is provable from the text.

    D24 removed this from the *drop* path (every class ships bare); it survives as
    the classifier of an unlabelled pair -- see :func:`_derive_unlabelled_defect`.
    """
    removed = " ".join(r.a_expanded for r in regions if r.a_text.strip())
    if not removed:
        return False
    if category == 1:
        # A quantity the question supplied is gone from the question.
        lost = set(schema.numbers_in(removed)) - set(schema.numbers_in(qu))
        return bool(lost)
    # cat5: the removal is inside the question clause and the clause is not the
    # same question any more.
    clause = _question_clause(qa)
    if not clause:
        return False
    offset = qa.find(clause)
    for region in regions:
        end = region.a_start + len(region.a_text)
        if region.a_start >= offset and end <= offset + len(clause):
            return True
    return False


def mine_placeholder_spans(
    question: str,
    k: int = K_OPTIONS,
    rng: random.Random | None = None,
    exclude: tuple[str, ...] = (),
) -> list[str] | None:
    """Mine ``k`` distinct equal-token-length spans of ``question``, with no anchor.

    Template A's solvable rows (the D24 two-layer side) carry an option block only
    to satisfy the D18/section 5.2 isomorphism rule: no option is correct and the
    reward never reads the block.  The block must still *look* like the one on the
    four-tier rows of the other sources -- k=3, equal token length, and every span
    a real span of this question, because "the option that does not occur in the
    question" is a measured 99-100% shortcut (section 4.4) -- so the mining
    constraints are the same as :func:`distractor_mining.mine_option_spans`; only
    the gold anchor is gone.

    The window length is chosen by the number of distinct admissible spans
    (content-carrying, not the excluded gold answer), ties broken towards the
    longer window; a question that cannot field ``k`` such spans at any length
    returns ``None`` and the caller drops the row.

    Args:
        question: the exact question text rendered into the prompt.
        k: total option count (D15 fixes the default at 3).
        rng: seeded per row by the caller so the build is reproducible.
        exclude: texts that must not become options -- the row's own gold answer,
            so the placeholder block can never contain the number the model is
            asked for.
    """
    rng = rng or random
    tokens = _tokens(question)
    excluded = {text.strip().casefold() for text in exclude if text and text.strip()}
    by_length: dict[int, list[str]] = {}
    for n_tokens in range(1, len(tokens) + 1):
        seen: set[str] = set()
        spans: list[str] = []
        for start in range(len(tokens) - n_tokens + 1):
            window = tokens[start : start + n_tokens]
            if not any(schema.is_content_word(token) for token, _, _ in window):
                continue
            text = question[window[0][1] : window[-1][2]].strip()
            key = text.casefold()
            if key in seen or key in excluded:
                continue
            seen.add(key)
            spans.append(text)
        if len(spans) >= k:
            by_length[n_tokens] = spans
    if not by_length:
        return None
    best = max(by_length, key=lambda n_tokens: (len(by_length[n_tokens]), n_tokens))
    return rng.sample(by_length[best], k)


# ---------------------------------------------------------------------------
# row construction
# ---------------------------------------------------------------------------


def _pair_category(row: dict, partner: dict) -> int | None:
    """The pair's defect class, or ``None`` when the source left it unlabelled.

    ``category`` lives on the unanswerable member only; the answerable member (and
    any other row without one) inherits its partner's.  ``None`` is the source's
    own "no label" value and is admitted -- section 4.8's category-less rows --
    while an integer outside the frozen table raises, because that is the file
    changing shape under us rather than a row without a class.
    """
    for member in (row, partner):
        category = member.get("category")
        if category is None:
            continue
        if isinstance(category, bool) or not isinstance(category, int):
            raise ValueError(f"UMWP category {category!r} is not an integer")
        if category not in CATEGORIES:
            raise ValueError(f"unknown UMWP category {category!r}")
        return category
    return None


def _derive_unlabelled_defect(qa: str, qu: str, regions: list[_Region]) -> tuple[str, str | None]:
    """Best-effort class of a pair the source left unlabelled.

    :func:`_certify_refusal` proves two of the five classes from the text (a
    removed quantity, a cut question clause).  When neither holds, the pair is
    still a valid bare row but stays outside the section 4.8 defect-type table:
    ``("", None)``, never a guessed label.
    """
    for category in (1, 5):
        if _certify_refusal(qa, qu, regions, category):
            return CATEGORIES[category]
    return UNLABELLED_DEFECT


def _pair_defect(category: int | None, qa: str, qu: str, regions: list[_Region]) -> tuple[str, str | None]:
    """``(error_type, perturbation_type)`` for the pair -- statistics, not routing."""
    if category is not None:
        return CATEGORIES[category]
    return _derive_unlabelled_defect(qa, qu, regions)


def _answer_text(row: dict) -> str:
    """The source's numeric answer as the gold string ("" when it is unusable).

    ``answer`` is polymorphic in this file: a 1-element ``list[float]`` on the
    2,600 answerable rows, ``null`` on the unanswerable ones and a bare ``int`` on
    two stray unanswerable rows (recon section 5.1).  The gold must be a real
    finite number, so anything else fails closed and the row is dropped.
    """
    answer = row.get("answer")
    if not isinstance(answer, list) or not answer:
        return ""
    value = answer[0]
    if isinstance(value, bool) or not isinstance(value, int | float):
        return ""
    if not math.isfinite(float(value)):
        return ""
    return str(value)


def _base_extra_info(row: dict, partner: dict, seed: int, defect: tuple[str, str | None]) -> dict:
    """The source-derived ``extra_info`` keys shared by both sides of the pair.

    ``defect`` is *this row's* defect: the pair's class on the unanswerable side,
    and :data:`UNLABELLED_DEFECT` on the answerable side.  A solvable row has no
    defect of its own, and copying the partner's class onto it would double-count
    every pair in the section 4.8 defect-type table (the shared audit text --
    ``deleted_condition_text`` / ``perturbed_entity_text`` -- stays, because it
    documents the pair, not the row's own class).
    """
    error_type, perturbation = defect
    return {
        "split": "train",  # the source has one implicit split (recon section 1.3)
        "index": row["id"],
        "seed": seed,
        "difficulty": row.get("source", "") or "",
        "perturbation_type": perturbation or "",
        "perturbation_family": "",
        "paired_original_text": normalise_question(partner["question"]),
        "error_type": error_type,
    }


def _task_id(row: dict) -> str:
    """Stable hard-replay key derived from the source's own id (never random)."""
    side = "ans" if row["answerable"] else "uns"
    return f"umwp-{side}-{row['id']}"


def build_rows(raw_dir: str, limit: int | None = None, seed: int = 0) -> tuple[list[dict], dict]:
    """Build the UMWP parquet rows and the funnel that produced them.

    Returns ``(rows, funnel)``: ``rows`` are ready for
    :func:`schema.normalise_extra_info` / :func:`schema.validate_rows` /
    :func:`schema.write_rows_parquet`, and ``funnel`` is an ordered mapping of
    filter stage -> rows remaining.  ``limit=None`` (the CLI default) emits the
    **full eligible pool**; the mix selects the section 4.8 quotas (550 answerable
    / 2,489 unanswerable) from it.  ``limit`` exists for fast local runs and is
    applied after the branch interleave, so a small limit keeps both sides.

    The same ``(raw_dir, limit, seed)`` always produces byte-identical rows: every
    random choice is made by a per-row :class:`random.Random` seeded from
    ``(seed, task_id)``.
    """
    raw = load_source(raw_dir)
    funnel: dict[str, int] = collections.OrderedDict()
    funnel["raw_rows"] = len(raw)

    well_formed = [row for row in raw if _is_well_formed(row)]
    funnel["after_malformed_drop"] = len(well_formed)

    # 1. the 2 stray rows whose unanswerable ``answer`` is a bare int rather than
    #    null (recon section 5.1).  They are dropped, not coerced: the field is the
    #    source's own statement that the row is unanswerable, and a number there
    #    contradicts the label.
    stray = [row for row in well_formed if not row["answerable"] and row.get("answer") is not None]
    stray_ids = {row["id"] for row in stray}
    stage = [row for row in well_formed if row["id"] not in stray_ids]
    funnel["after_stray_answer_drop"] = len(stage)

    # 2. duplicate questions.  Normalised-question groups with conflicting labels
    #    are contradictory input (recon section 5.2 item 5 measured 5 such pairs);
    #    groups of same-label duplicates are the same row twice.
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for row in stage:
        groups[normalise_question(row["question"])].append(row)
    conflicting: set[int] = set()
    duplicate_of: dict[int, int] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        labels = {member["answerable"] for member in members}
        if len(labels) > 1:
            conflicting.update(member["id"] for member in members)
            continue
        ordered = sorted(members, key=lambda item: item["id"])
        for keep, drop in zip(ordered, ordered[1:], strict=False):
            duplicate_of[drop["id"]] = keep["id"]
    stage = [row for row in stage if row["id"] not in conflicting and row["id"] not in duplicate_of]
    funnel["after_duplicate_question_drop"] = len(stage)

    by_id, partners = _partner_map(stage)

    # 3. pair resolution.  Every row is kept only together with a partner that the
    #    source's own ``relevant_ids`` links it to.
    paired: list[tuple[dict, dict]] = []
    with_partner = 0
    for row in stage:
        if row["answerable"]:
            partner = partners.get(row["id"])
        else:
            linked = row.get("relevant_ids")
            target = linked[0] if isinstance(linked, list) and len(linked) == 1 else None
            partner = by_id.get(target) if target is not None else None
            if partner is not None and not partner["answerable"]:
                partner = None
        if partner is None:
            continue
        with_partner += 1
        paired.append((row, partner))
    funnel["after_pair_resolution_drop"] = with_partner

    # 4. defect class.  The class is statistics only (D24), so an *absent* label is
    #    admitted (section 4.8's category-less rows); an unknown code is not.
    classified: list[tuple[dict, dict, int | None]] = []
    for row, partner in paired:
        try:
            category = _pair_category(row, partner)
        except ValueError:
            continue
        classified.append((row, partner, category))
    funnel["after_defect_class_drop"] = len(classified)

    # 5. the pair certificate: the changed regions are the whole difference between
    #    the two questions, and the two questions really differ.  This is the only
    #    content gate -- it admits the token-invisible (character-level) defects,
    #    which the bare contract does not need to point at.
    certified: list[tuple[dict, dict, int | None, str, str, list[_Region]]] = []
    for row, partner, category in classified:
        answerable = row["answerable"]
        qa = normalise_question(row["question"] if answerable else partner["question"])
        qu = normalise_question(partner["question"] if answerable else row["question"])
        regions = word_regions(qa, qu)
        if qa == qu or not _residual_tokens(qa, qu, regions):
            continue
        certified.append((row, partner, category, qa, qu, regions))
    funnel["after_defect_certificate_drop"] = len(certified)

    # 6. row construction.  Both sides ship; the answerable side additionally needs
    #    a placeholder option block (template A) and a real numeric answer.
    built: list[dict] = []
    for row, partner, category, qa, qu, regions in certified:
        defect = _pair_defect(category, qa, qu, regions)
        extra = _base_extra_info(row, partner, seed, UNLABELLED_DEFECT if row["answerable"] else defect)
        extra["task_id"] = _task_id(row)
        deleted, inserted = defect_texts(qa, qu, regions)
        extra["deleted_condition_text"] = deleted
        extra["perturbed_entity_text"] = inserted
        if row["answerable"]:
            answer = _answer_text(row)
            texts = (
                mine_placeholder_spans(
                    qa,
                    K_OPTIONS,
                    rng=random.Random(f"{seed}:placeholder:{row['id']}"),
                    exclude=(answer,),
                )
                if answer
                else None
            )
            if texts is None:
                continue  # no placeholder block and no answer -> drop this side only
            options = schema.shuffle_options(texts, random.Random(f"{seed}:options:{row['id']}"))
            ground_truth = schema.build_ground_truth(
                solvable=True,
                answer=answer,
                correct_option_id=None,
                has_diagnosis_label=False,
                perturbation_type=None,
                two_layer=True,
            )
            extra.update({"solvable": True, "correct_option_id": ""})
            built.append(
                schema.make_row(
                    data_source=DATA_SOURCE,
                    question=qa,
                    ground_truth=ground_truth,
                    template=schema.TEMPLATE_A,
                    branch=BRANCH_TWO_LAYER,
                    extra_info=extra,
                    options=options,
                )
            )
            continue
        ground_truth = schema.build_ground_truth(
            solvable=False,
            answer=None,
            correct_option_id=None,
            has_diagnosis_label=False,
            perturbation_type=defect[1],
        )
        extra.update({"solvable": False, "correct_option_id": ""})
        built.append(
            schema.make_row(
                data_source=DATA_SOURCE,
                question=qu,
                ground_truth=ground_truth,
                template=schema.TEMPLATE_B,
                branch=BRANCH_BARE,
                extra_info=extra,
            )
        )
    funnel["after_option_mining_drop"] = len(built)

    rows = _interleave_by_branch(built)
    if limit is not None:
        rows = rows[: max(limit, 0)]
    funnel["after_limit"] = len(rows)
    return rows, funnel


def _interleave_by_branch(rows: list[dict]) -> list[dict]:
    """Round-robin the two branches so a ``limit`` keeps both sides represented.

    Without this a small ``--limit`` would return only answerable rows, because
    the source file stores them first -- and any audit that needs both the
    answerable and the unanswerable side would have nothing to run on.
    """
    groups: dict[str, list[dict]] = collections.OrderedDict()
    for row in rows:
        groups.setdefault(row["extra_info"]["branch"], []).append(row)
    out: list[dict] = []
    index = 0
    while True:
        progressed = False
        for group in groups.values():
            if index < len(group):
                out.append(group[index])
                progressed = True
        if not progressed:
            return out
        index += 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _breakdown(rows: list[dict], key) -> dict:
    counter: dict = collections.Counter(key(row) for row in rows)
    return dict(sorted(counter.items(), key=lambda item: str(item[0])))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the UMWP hallucination-domain rows.")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="truncate the built rows (after the branch interleave); the default emits the full "
        "eligible pool, from which mix_halluc.py selects the section 4.8 quotas",
    )
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows, funnel = build_rows(args.raw_dir, limit=args.limit, seed=args.seed)
    schema.normalise_extra_info(rows)
    schema.validate_rows(rows)
    schema.write_rows_parquet(rows, args.out)

    print(f"raw dir : {args.raw_dir}")
    print(f"wrote   : {args.out}  ({len(rows)} rows, seed={args.seed})")
    print("\nfunnel (rows remaining after each stage, and what that stage cost):")
    previous = None
    for stage, count in funnel.items():
        cost = "" if previous is None else f"   -{previous - count}"
        print(f"  {stage:34s} {count:6d}{cost}")
        previous = count
    print("\nper branch:")
    for branch, count in _breakdown(rows, lambda r: r["extra_info"]["branch"]).items():
        print(f"  {branch:22s} {count}")
    print("\nper template:")
    for template, count in _breakdown(rows, lambda r: r["extra_info"]["template"]).items():
        print(f"  {template:22s} {count}")
    print("\nper solvable:")
    for solvable, count in _breakdown(rows, lambda r: r["extra_info"]["solvable"]).items():
        print(f"  {str(solvable):22s} {count}")
    print("\nper error_type:")
    for error_type, count in _breakdown(rows, lambda r: r["extra_info"]["error_type"]).items():
        print(f"  {error_type:26s} {count}")
    print("\nper perturbation_type:")
    for perturbation, count in _breakdown(rows, lambda r: r["extra_info"]["perturbation_type"]).items():
        print(f"  {perturbation or '<unlabelled>':26s} {count}")


if __name__ == "__main__":  # pragma: no cover - exercised through main() in tests
    main()
