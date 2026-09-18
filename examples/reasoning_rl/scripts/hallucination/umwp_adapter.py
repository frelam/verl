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
"""UMWP adapter -- native answerable/unanswerable pairs (design doc D16, section 4.7).

Source: ``github.com/Yuki-Asuuna/UMWP``, ``data/StandardDataset.jsonl`` (one file,
5,200 rows, CC-BY-SA-4.0).  Every row is one of two members of a *native* pair:
positions 0-2599 are ``answerable=true`` with a 1-element float ``answer``, and
each unanswerable row carries ``relevant_ids=[<the answerable row's id>]``.  That
link -- not ``id - 2600``, which the recon measured to fail on 2600/2600 rows --
is how the partner is recovered here.

Contract produced by this adapter (design doc section 4.9.3 table B, rows 3/5/6):

===================  ========  ================================================
branch               template  gold
===================  ========  ================================================
``solvable_judge``   A         ``\\boxed{SOLVABLE}`` (``judgment_only=true``)
``unsolvable_diag``  A         ``\\boxed{UNSOLVABLE: <option id>}``
``unsolvable_bare``  B         ``\\boxed{UNSOLVABLE}``
===================  ========  ================================================

cat2/cat3/cat4 (ambiguous key information / unrealistic conditions / unrelated
object) are the pointer-gold categories: their defect is text *inserted into* the
question, so the model can be asked to point at it.  cat1/cat5 (key information
missing / question missing) are not: their defect is text that is *gone*, so the
only honest target is a bare refusal -- exactly the design doc's own split
(section 4.7: "cat1 ... 只能走三档 bare"), extended to cat5 on the recon's
measurement (section 4.9.3 table B sends cat5 to four-tier; see deviation 3).

Everything is derived from the row's own pair by a **word-level** ``difflib``
diff, and every gold is re-derivable from the artifact alone (the verifier does
exactly that, see ``verify_umwp.py``).  A row whose gold cannot be certified is
dropped and counted in the funnel -- nothing is invented and no option text ever
comes from another question.

Certificates (fail closed; each is re-derived by ``verify_umwp.py``)
-------------------------------------------------------------------

* **Pointer (diag)** -- exactly one contiguous changed region; its inserted text
  is nonempty, is absent from the answerable member, and every distractor occurs
  *verbatim in the answerable member*.  That last property is the uniqueness
  proof: text that survived in the original problem cannot be the defect that
  made this variant unanswerable, so the gold is the only admissible option.
* **Refusal (bare)** -- the question lost something it needs.  cat1: a number
  from the removed region is absent from the presented question.  cat5: the
  removed region lies inside the original question clause and the presented
  question no longer carries it.
* **Judgment (solvable)** -- the row is the answerable member, its audit answer
  parses as a finite float, and applying the pair's recorded defect to it
  reproduces the unanswerable member.  UMWP ships **no derivation chain**, so this
  certifies the *pair* (the two members are the same problem modulo the recorded
  edit), not the arithmetic of the number; see the module note on L1 below.

DEVIATIONS FROM THE DESIGN DOC
------------------------------

Measured against ``scratch/halluc_recon/umwp_report.md`` (ground truth) and
``HALLUCINATION_RL_DESIGN.md``.  Every number below is measured by this adapter;
the funnel printed by ``main()`` shows where each row is lost.

1. **cat1+cat5 size: the doc's 840 (cat1 bare) / 98 (cat5 four-tier) do not hold.**
   The doc (D18 table B row 6, section 4.9.3) puts cat1 at 840 bare and cat5 at 98
   *four-tier*; the recon refuted the cat5 half ("cat5 is a pure deletion on
   116/119 rows ... no span in the question to point at"), so cat5 goes to the bare
   branch here.  Measured: 641 cat1 + 114 cat5 = 755 bare rows.  The cat1
   shortfall against the doc's 840 (the recon's own cat1 count) is dominated by
   the refusal certificate: a cat1 row is kept only if the removed text carries a
   number the presented question no longer has, and rows whose "missing" text
   leaves the question answerable -- the label contradicts the text -- are dropped
   rather than given a gold the text does not support.

2. **cat1 is not uniformly "invisible in the question": 271 of the 755 bare rows
   carry inserted (pointerable) text.**  The doc's verification list for UMWP
   (section 9) asks the audit to "assert cat1's visible region = 0"; that
   assertion is false on the data -- the recon measured 413/840 cat1 rows with
   nonempty inserted text and this adapter measures 271/755 over cat1+cat5.  The
   refusal branch does not need an invisible defect (the model is asked for a
   verdict, not a pointer), so those rows are kept and the counter-example is
   reported instead of assert-failed.

3. **Four-tier yield is 839 rows, not the doc's 1,449; solvable-judge is 1,945,
   not the doc's 550.**  The doc (D16, section 4.9.1 table A) allocates "UMWP ...
   四档 1,449 / 三档 440", and section 4.9.3 table B row 3 gives the solvable
   judgment rows an allocation of 550.  The recon could not reproduce 1,449 under
   any of 14 rules and flagged it as a uniform ~0.8233 rescaling of the raw
   counts.  Measured here after the one-region + uniqueness + k=3 certificates:
   839 pointer-gold rows (cat2 738 / cat3 28 / cat4 73) and 1,945 solvable-judge
   rows (once the answerable member can supply a 3-option equal-length block,
   which it fails to do for 643 of the certified pairs).  The doc's 440 for
   three-tier is contradicted by the doc itself (section 4.9.3 says 840); neither
   survives, the measured bare pool is 755.

4. **L3 is not 0.500.**  The doc reports UMWP as "0.500 / 0.500".  The recon
   measured 0.257 (raw multinomial NB) and ~0.58 with a support-filtered
   vocabulary.  ``verify_umwp.py`` re-runs it on the built rows and prints the raw
   number; the report's diagnosis (an unfiltered multinomial NB is numerically
   broken here, its class score flips sign out of fold) is why the vocabulary is
   restricted to tokens with train support >= 5 documents.

5. **The diff must be word-level.**  The recon's UMWP shapes were measured with
   character-level ``get_opcodes``.  On normalized text a char-level diff splits
   words (measured gold spans such as ``'Som'``, ``'ome'``, ``'-1'`` -- mid-word
   fragments) because UMWP repeats words inside one question ("...46 magazines
   ... How many magazines..."), which lets ``difflib`` align a repeated word to
   the wrong occurrence.  This adapter diffs **token sequences** and maps the
   resulting opcodes back to character offsets, so every gold span is a whole word
   span.  Consequence: cat3's sign-flip defects ("26" -> "-26", 188 of 275 rows)
   tokenize identically and are therefore *invisible* at word level; they are
   dropped rather than pointed at with a one-character option.

6. **The source has no ``difficulty``, ``split`` or ``index`` field.**  The recon
   (section 1.3) found one implicit split and no split column, so ``split`` is
   ``"train"`` for every row and ``index`` is the source's own ``id``.  UMWP's
   native difficulty axis does not exist; ``extra_info.difficulty`` records the
   *base problem's* origin (GSM8K / SVAMP / MultiArith / ASDiv), which is the only
   native per-row stratum the file carries.  It is a provenance label, not a
   measured difficulty.

Module note on L1
-----------------

UMWP's labels are human-constructed and the file ships no derivation chain (recon
section 4.9.1 table A: "⚠️ 人工构造").  The certificates above prove that the
*recorded defect is the only difference between the two members* and that the
pointer/refusal gold is unique among the options.  They do not prove that a
category-2 replacement really makes the problem unanswerable -- no mechanical
witness for that exists in this source, and the adapter does not pretend
otherwise.
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
import unicodedata

try:
    import schema
    from distractor_mining import mine_option_spans
except ImportError:  # pragma: no cover - running as a plain script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import schema
    from distractor_mining import mine_option_spans

# ---------------------------------------------------------------------------
# source constants
# ---------------------------------------------------------------------------

DATA_SOURCE = schema.SOURCE_UMWP
DEFAULT_RAW_DIR = "/home/charles/data/reasoning_rl/halluc/raw/umwp"
DEFAULT_OUT = os.path.expanduser("~/data/reasoning_rl/halluc/built/umwp.parquet")
DATA_FILE = "StandardDataset.jsonl"

BRANCH_DIAG = schema.BRANCH_UNSOLVABLE_DIAG
BRANCH_BARE = schema.BRANCH_UNSOLVABLE_BARE
BRANCH_JUDGE = schema.BRANCH_SOLVABLE_JUDGE

K_OPTIONS = 3  # design decision D15

# category code -> (error_type for the D18 balance table, perturbation_type)
CATEGORIES: dict[int, tuple[str, str]] = {
    1: ("key_information_missing", "missing_condition"),
    2: ("ambiguous_key_information", "ambiguous_condition"),
    3: ("unrealistic_conditions", "unrealistic_condition"),
    4: ("unrelated_object", "unrelated_entity"),
    5: ("question_missing", "question_missing"),
}
DIAG_CATEGORIES = (2, 3, 4)  # defect is visible in the question -> pointer gold
BARE_CATEGORIES = (1, 5)  # defect is an absence -> bare refusal

# The defect must be pointerable with a word that means something: a one-character
# gold ("-", ".") makes an option block that is worse than no option block at all.
MIN_GOLD_CHARS = 2

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
    ``insert`` at the same index) must not turn one defect into two regions --
    the pointer certificate requires exactly one.
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

    Diffing token sequences (rather than characters) is what keeps every gold a
    whole word span; ``a_start`` / ``b_start`` are character offsets into the two
    strings, and ``a_expanded`` additionally covers the punctuation and whitespace
    between the surrounding equal tokens, which is what the "was the question
    clause touched?" certificate needs.
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
    sequences must be identical.  Re-derivable by the verifier from the artifact.
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


def _certify_pointer(qa: str, qu: str, regions: list[_Region], rng: random.Random):
    """Pointer gold for cat2/cat3/cat4, or ``None`` when it cannot be certified."""
    if len(regions) != 1:
        return None
    region = regions[0]
    gold = region.inserted.strip()
    if not gold or len(gold) < MIN_GOLD_CHARS:
        return None
    # The gold must be introduced by the defect: if the same text already sits in
    # the answerable member then it cannot be what made this variant unanswerable.
    if gold in qa:
        return None
    # ... and it must be locatable at the region's own offset, not at an earlier
    # accidental occurrence of the same string.
    if qu.find(gold) != region.b_start:
        return None
    mined = mine_option_spans(qu, gold, k=K_OPTIONS, rng=rng)
    if mined is None:
        return None
    texts, correct = mined
    # Every distractor must be text that survived from the original problem.  This
    # is the L2 uniqueness proof, not a style preference.
    if not all(text in qa for text in texts if text != gold):
        return None
    return {"gold": gold, "texts": texts, "correct": correct}


def _certify_refusal(qa: str, qu: str, regions: list[_Region], category: int) -> bool:
    """Whether "this question lost what it needs" is provable from the text."""
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


# ---------------------------------------------------------------------------
# row construction
# ---------------------------------------------------------------------------


def _base_extra_info(row: dict, partner: dict, seed: int) -> dict:
    # ``category`` lives on the unanswerable member only; on the answerable side the
    # pair's defect type is the partner's.
    category = int(row["category"] if row["category"] is not None else partner["category"])
    error_type, perturbation = CATEGORIES[category]
    return {
        "split": "train",  # the source has one implicit split (recon section 1.3)
        "index": row["id"],
        "seed": seed,
        "difficulty": row.get("source", "") or "",
        "perturbation_type": perturbation,
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
    filter stage -> rows remaining.  The same ``(raw_dir, limit, seed)`` always
    produces byte-identical rows: every random choice is made by a per-row
    :class:`random.Random` seeded from ``(seed, task_id)``.
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
        for keep, drop in zip(ordered, ordered[1:]):
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

    # 4. certificates.  An unanswerable row is kept only if its defect is either
    #    pointerable or provably an absence; an answerable row is kept only if the
    #    pair leaves it a usable option anchor (template A needs an equal-length
    #    option block on both sides -- design decision D18's isomorphism rule).
    certified: list[tuple[dict, dict, dict]] = []
    for row, partner in paired:
        # The defect type lives on the unanswerable member; the answerable member
        # inherits it.  An unlabelled or unknown category has no certificate and no
        # perturbation_type to record, so the whole pair fails closed -- including
        # the answerable side, whose extra_info would otherwise name a defect that
        # does not exist.
        category = row["category"] if row["category"] is not None else partner["category"]
        if category is None or int(category) not in CATEGORIES:
            continue
        category = int(category)
        qa = normalise_question(partner["question"] if not row["answerable"] else row["question"])
        qu = normalise_question(row["question"] if not row["answerable"] else partner["question"])
        regions = word_regions(qa, qu)
        plan: dict = {"regions": regions, "category": category}
        if row["answerable"]:
            anchors = [r.a_text.strip() for r in regions if r.a_text.strip()]
            plan["anchor"] = max(anchors, key=len) if anchors else ""
        else:
            rng = random.Random(f"{seed}:pointer:{row['id']}")
            if category in DIAG_CATEGORIES:
                pointer = _certify_pointer(qa, qu, regions, rng)
                if pointer is None:
                    continue
                plan["pointer"] = pointer
            else:  # category in BARE_CATEGORIES -- the guard above admits nothing else
                if not _certify_refusal(qa, qu, regions, category):
                    continue
        if not _residual_tokens(qa, qu, regions):  # pragma: no cover - holds by construction
            continue
        plan["qa"], plan["qu"] = qa, qu
        certified.append((row, partner, plan))
    funnel["after_defect_certificate_drop"] = len(certified)

    # 5. option mining.  Template A rows (the pointer rows and the judgment rows)
    #    need k=3 equal-length spans of their own question; a row that cannot get
    #    them is dropped rather than given cross-question text.
    built: list[dict] = []
    for row, partner, plan in certified:
        category = plan["category"]
        extra = _base_extra_info(row, partner, seed)
        extra["task_id"] = _task_id(row)
        deleted = " ".join(r.a_text for r in plan["regions"] if r.a_text.strip())
        inserted = " ".join(r.b_text.strip() for r in plan["regions"] if r.b_text.strip())
        extra["deleted_condition_text"] = deleted
        extra["perturbed_entity_text"] = inserted
        if row["answerable"]:
            anchor = plan["anchor"]
            if not anchor:
                continue
            mined = mine_option_spans(plan["qa"], anchor, k=K_OPTIONS, rng=random.Random(f"{seed}:anchor:{row['id']}"))
            if mined is None:
                continue
            texts, _ = mined
            options = [{"id": chr(ord("A") + i), "text": text} for i, text in enumerate(texts)]
            answer = row["answer"]
            audit = str(answer[0]) if isinstance(answer, list) and answer else ""
            ground_truth = schema.build_ground_truth(
                solvable=True, answer=audit, judgment_only=True, perturbation_type=None
            )
            extra.update({"solvable": True, "judgment_only": True, "correct_option_id": ""})
            built.append(
                schema.make_row(
                    data_source=DATA_SOURCE,
                    question=plan["qa"],
                    ground_truth=ground_truth,
                    template=schema.TEMPLATE_A,
                    branch=BRANCH_JUDGE,
                    extra_info=extra,
                    options=options,
                )
            )
            continue
        if category in DIAG_CATEGORIES:
            pointer = plan["pointer"]
            options = [
                {"id": chr(ord("A") + i), "text": text} for i, text in enumerate(pointer["texts"])
            ]
            ground_truth = schema.build_ground_truth(
                solvable=False,
                answer=None,
                correct_option_id=pointer["correct"],
                has_diagnosis_label=True,
                perturbation_type=CATEGORIES[category][1],
            )
            extra.update({"solvable": False, "correct_option_id": pointer["correct"]})
            built.append(
                schema.make_row(
                    data_source=DATA_SOURCE,
                    question=plan["qu"],
                    ground_truth=ground_truth,
                    template=schema.TEMPLATE_A,
                    branch=BRANCH_DIAG,
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
            perturbation_type=CATEGORIES[category][1],
        )
        extra.update({"solvable": False, "correct_option_id": ""})
        built.append(
            schema.make_row(
                data_source=DATA_SOURCE,
                question=plan["qu"],
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
    """Round-robin the branches so a ``limit`` keeps every branch represented.

    Without this a small ``--limit`` would return only answerable rows, because
    the source file stores them first -- and the anti-cheat audit, which needs both
    the solvable and the unsolvable side, would have nothing to run on.
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
    parser.add_argument("--limit", type=int, default=None)
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


if __name__ == "__main__":  # pragma: no cover - exercised through main() in tests
    main()
