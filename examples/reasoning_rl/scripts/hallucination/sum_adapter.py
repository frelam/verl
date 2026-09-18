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
"""SUM adapter -- Synthetic Unanswerable Math native pairs (design doc D16, section 4.9.1).

Source: HF ``lime-nlp/Synthetic_Unanswerable_Math``, the converted-parquet repo
(``refs/convert/parquet``), file ``train.parquet`` -- 36,480 rows, exactly three
columns: ``answerable_question``, ``unanswerable_question``, ``ground_truth``.
Each row is already a *pair*: the answerable member and the o3-mini rewrite that
made it unanswerable, with ``ground_truth`` the answer to the **answerable**
member.  There is no ``id``, no split column and no defect label.

Contract produced by this adapter (design doc section 4.9.3 table B, rows 1/3/5/6):

===================  ========  ================================================
branch               template  gold
===================  ========  ================================================
``solvable_judge``   A         ``\\boxed{SOLVABLE}`` (``judgment_only=true``)
``unsolvable_diag``  A         ``\\boxed{UNSOLVABLE: <option id>}``
``unsolvable_bare``  B         ``\\boxed{UNSOLVABLE}``
===================  ========  ================================================

The three branches come from the *shape* of the word-level diff between the two
members of the pair, not from a label the source does not carry:

* **no inserted text at all** (the rewrite only deletes) -- the removed premise is
  *gone* from the presented text, so no option can point at it and the only honest
  target is a bare refusal.  Design doc section 4.9.3 table B row 6 ("SUM-del").
* **exactly one contiguous inserted/replaced region, and that text is a defect one
  of the three visible rules can name** -- the defect *is* text in the question, so
  the model can be asked to point at it.  Table B row 5 ("SUM-visible").  The
  second half of the condition is what makes the pointer a gold rather than a
  guess: a rewrite that pads the sentence while it drops a premise leaves an
  insertion no rule names, and the model would be told to point at filler
  (1,557 rows; deviation 13).
* **more than one inserted region** -- the model would have to pick one of several
  defects; the row is dropped and counted (19,422 rows, 53.2% of the file: the
  o3-mini rewrites are paraphrases, not minimal edits).
* **no word-level change** -- nothing to certify; dropped (116 rows: 3 are the same
  text twice and 113 differ only in characters the word tokenizer cannot see, such
  as ``y ≠ 0`` -> ``y > 0``; see deviation 10).

Everything is derived from the row's own pair by a **word-level** ``difflib`` diff,
and every gold is re-derivable from the built artifact alone: each row carries the
pair's *other* member in ``paired_original_text``, so the audit can re-read the
source row without trusting the rendered question, and it re-diffs the pair with
its own implementation (``verify_sum.py``).  A row whose gold cannot be
certified is dropped and counted in the funnel -- nothing is invented, and no
option text ever comes from another question.

Certificates (fail closed; each is re-derived by ``verify_sum.py``)
-------------------------------------------------------------------

* **Pointer (diag)** -- the pair has exactly one contiguous inserted region; its
  text is at least :data:`MIN_GOLD_CHARS` long, is **absent from the answerable
  member** (text that already existed in the original problem cannot be what made
  the variant unanswerable), is located at the region's own character offset (so
  the pointer is unambiguous), is **a defect one of the three visible rules can
  name** (:func:`certifies_as_visible_defect` -- the residual class only when it
  introduces a content word the original lacked, the other two by the rule that
  fires), and the question can supply k=3 equal-length same-question option spans
  including it.  Note the SUM-specific weakening: the
  UMWP pointer certificate additionally demands that every *distractor* occur
  verbatim in the answerable member.  That property is not sound here -- the
  unanswerable member is a paraphrase, so a distractor absent from the original
  is normally just reworded content, not a defect (requiring it costs 576 rows
  and buys no proof; see deviation 11).
* **Refusal (bare)** -- the pair records only deletions, the deleted text carries
  at least one content word, and at least one of those content words is **absent
  from the presented question**.  Without that last clause a "deletion" could be
  a word that still stands elsewhere in the question (789 rows), which is not
  evidence that anything was lost.
* **Judgment (solvable)** -- the pair is certified as the same problem modulo the
  recorded edit (token-level residual check), the answerable member carries a
  nonempty deleted anchor that itself carries a content word (that anchor is what
  :func:`mine_option_spans` appends to the block as its gold, so it is subject to
  the same rule 3 as the distractors; deviation 14), and that anchor can host a
  k=3 equal-length option block, which is what keeps template A isomorphic with
  the diag rows (D14/D18).  A judgment row is emitted only for a pair whose
  unanswerable member is itself emitted (a diag row), so one pair supplies
  **both** label sides of template A.

DEVIATIONS FROM THE DESIGN DOC, AND DEFECTS FOUND BY THE AUDIT
--------------------------------------------------------------

Every number below is measured by this adapter on the frozen ``train.parquet``;
the funnel printed by ``main()`` shows where each row is lost.  Items 1-12 are
places where the measurement departs from what the design doc reports; items
13-15 are defects in this adapter's *first* build, found by an adversarial audit
that re-derived every claim from the built artifact plus the raw file, and fixed
here.  Each fix has a failing case behind it and neither weakens a check.

1. **``test.parquet`` is not data -- it supplies 0 rows.**  Doc section 4.9.1
   table A: *"train 36,480 (+test 284)"*.  Measured: ``test.prompt`` is
   ``train.unanswerable_question[k]`` + the project's own "think step by step"
   suffix for k = 0..283, position by position, and ``test.ground_truth`` is the
   single constant string ``"I don't know."`` on all 284 rows.  The test split
   carries no ``answerable_question``, so it cannot form a pair and its gold is
   not the project contract.  The adapter reads ``train.parquet`` only and the
   "+284" is counted as 0 usable rows.

2. **The four-tier / three-tier yields 9,832 / 6,050 are not reproducible.**
   Doc section 4.9.1 table A and section 9 (2).  Measured under the house
   tokenizer with the projection the doc names (word-level ``difflib`` opcodes):
   pure deletion (three-tier / bare pool) = **8,224**; exactly one inserted
   region (four-tier / diag pool) = **8,718**; more than one inserted region =
   19,422; no word-level change = 116.  Neither 9,832 nor 6,050 is reachable: the
   measured deviation is -11.3% on the four-tier pool and +35.9% on the
   three-tier pool, both far outside the doc's own 2% warning band.
   ``verify_sum.py`` recomputes both numbers from the raw file and WARNs.

3. **The emitted yields after the certificates are 4,229 diag / 7,308 bare /
   3,195 judge.**  (The funnel's ``after_defect_certificate_drop`` counts pairs,
   so its 11,537 is diag + bare; the judgment rows are emitted *on top of* the diag
   rows, one per diag pair that can host one.)  Against the doc's quotas
   (SUM-visible 5,094, SUM-del 2,000, solvable_judge 350 from table B row 5/6/3)
   that is **0.83x the four-tier quota** -- the SUM-visible branch cannot supply
   what table B asks for, which is a finding about the source, not a gap to paper
   over -- 3.65x the deletion quota and 9.1x the judgment quota.  The rejections,
   by the clause that rejected them (``main()`` prints the same table, and
   ``funnel["certificate_drops"]`` carries the counts into the JSON report):
   1,606 inserted spans shorter than :data:`MIN_GOLD_CHARS`, 1,557 insertions that
   are not a defect, 819 golds that also occur in the answerable member, 484
   questions that cannot host a k=3 equal-length option block, 804 deletion-only
   pairs whose lost text proves nothing; on the judgment side 684 pairs with
   nothing deleted, 225 whose answerable member cannot host an option block and
   125 whose anchor is a bare function word.  The five pointer/refusal clauses
   sum to the 5,270 pairs the funnel stage actually lost
   (16,807 - 11,537); the three judgment clauses are a **different unit** -- the
   pair still emits its diag row, only the judgment row is withheld -- so their
   1,034 are printed apart by ``main()`` and do not subtract from the funnel.
   All counts are post-dedup, the units the funnel counts -- the pre-dedup
   projection is a few rows larger for the clauses dedup can touch (1,617
   inserted-too-short / 1,560 not-a-defect / 823 gold-in-member /
   806 deletion-lost-nothing), while the option-host and judgment clauses are
   unchanged (484 / 225 / 684 / 125).

4. **L3 is not 0.502 / 0.503.**  Doc section 4.9.1 and section 9 (3) report the
   BoW Naive Bayes reading as 0.502 / 0.503.  Measured with the house estimator
   (5-fold out-of-fold balanced accuracy) on the *raw pair corpus*, A vs U: 0.638
   over all 36,480 pairs (survey), 0.550 over a balanced 4,000-member sample,
   0.603 over 12,000 -- the reading rises with the sample because the vocabulary
   is support-gated, so the number is only meaningful next to the corpus size.
   The doc's 0.502 is what this estimator returns on **shuffled labels** (the
   audit's control reads 0.5025 over all 36,480 pairs and 0.5062 over the balanced
   4,000-member sample -- i.e. the doc's own number, re-measured as noise).  On
   the built artifact
   at the D18 quota composition the gate reads 0.4517 (limit-400 build, control
   0.4997) and 0.5100 (full pool, n=4,000, control 0.5084), both below the 0.55
   hard gate.  ``verify_sum.py`` prints the composition, the raw reading, its
   sample size and the shuffled-label control next to each other.

5. ``question_missing`` is a **bare** type, not a four-tier one.  Doc section
   4.9.3 balance table assigns *"SUM-visible ≈1,042"* missing-question rows to
   the four-tier branch.  A deleted question clause is invisible in the presented
   text -- there is no span to point at -- so those rows route to
   ``unsolvable_bare`` exactly like the other deletions, which is also what
   ``umwp_adapter.py`` does for UMWP cat5.  Measured: 2,052 rows whose question
   clause was deleted **and is gone from the prompt** are counted under
   ``question_missing`` on the bare branch (2,054 before deviation 15's third
   condition).

6. **The doc's per-type table contradicts its own yield claim.**  Section 4.9.3's
   balance table sums to ≈1,960 + ≈1,174 + ≈1,042 + ≈915 = 5,091 ≈ SUM-visible
   5,094, yet section 4.9.1 claims the four-tier yield is 9,832.  The per-type
   numbers were sized against the 5,094 quota, not against 9,832; both cannot be
   the same measurement.

7. **The type proportions of that table are not measurable.**  SUM ships no type
   labels (deviation-free observation: three columns, no ``category``), so the
   five types come from the rule-based classifier below, which serves balancing
   and monitoring only and never appears in reward.  Measured over the emitted
   rows the split is dominated by ``irrelevant_undefined_entity`` (the residual
   class, 5,798 rows = 39.4%) instead of the doc's ≈38.5% ambiguity, with
   ``missing_necessary_condition`` 5,256 (35.7%), ``question_missing`` 2,052
   (13.9%), ``ambiguous_key_information`` 1,456 (9.9%) and
   ``unrealistic_self_contradiction`` 170 (1.2%); see the "per error_type" block
   ``main()`` prints.  ``verify_sum.py`` runs the N=100 stratified spot check the
   doc asks for (Q26) and reports the failure rate; measured over the full
   artifact it is **0/100 = 0.0%**, under the doc's 10% downgrade line.  The
   first build read 7/100, and all 7 were in the residual class (7/20 = 35%: the
   "no content word absent from the answerable member" predicate is the
   classifier's weakest rule) -- the same rows the defect-class clause of
   deviation 13 rejects, so the fix and the spot check agree on which rows were
   wrong.

8. **License.**  Doc section 4.9.1 table A says SUM is MIT and the HF dataset
   card agrees, but ``raw_manifest.json`` records ``"license": "Apache-2.0"``
   for both files (the parquet-conversion repo default).  The artifacts are MIT;
   only the manifest string disagrees.

9. **The README documents two columns; the parquet has three.**  The dataset card
   lists ``answerable_question`` / ``unanswerable_question`` only.  The converted
   file adds ``ground_truth``, which is the column this adapter needs, so the
   adapter does not assume the published schema.

10. **116 rows carry no word-level diff, and only 3 of them are identical.**
   113 of the 116 differ only in characters the tokenizer drops -- ``y ≠ 0`` ->
    ``y > 0``, ``AC=2`` -> ``AC = -2``, ``x>10^10`` -> ``x < 10^10``.  Those are
   real defects (a sign or inequality flip) but a word-level diff cannot see
   them, so the gold is uncertifiable and the rows are dropped as one funnel
   bucket (``after_no_word_diff_drop``), the same call UMWP's adapter makes for
   its invisible sign flips.

11. **The pointer certificate drops the UMWP uniqueness clause.**  Measured on
   the emitted diag rows: requiring every distractor to occur verbatim in the
   answerable member (the UMWP rule) leaves 3,738 of 4,229, a cost of 491 rows,
   and the property is not evidence here -- the two members are paraphrases, so a
   distractor that is absent from the original is usually reworded content (on
   the first build, whose pointer pool was larger, the same rule cost 576).  The
   certificate keeps the part that *is* evidence (the gold is absent from the
   answerable member, is a defect, and is unambiguous in the question).

12. **The judgment pool is pair-tied, not the whole answerable corpus.**  Every
   pair carrying a word diff whose answerable member has a mineable,
   content-carrying deletion anchor could supply a judgment row (31,644 rows,
   86.7% of the file), which would make the artifact 73% solvable (31,644 vs
   11,537) and turn the L3 anti-cheat reading into an A-vs-U measurement of the
   raw corpus.  Emitting a judgment row only for a pair whose unanswerable member
   is itself emitted (3,195 pairs after dedup) keeps template A carrying both
   label sides of the *same* pair (D14/D18 isomorphism) and keeps the artifact's
   composition representative of the mix's own quotas.

13. **DEFECT: the pointer certificate did not check that the insertion is a
   defect.**  Found by the adversarial audit on the first build.  The certificate
   proved that the inserted text was unique, absent from the original and
   pointable -- but never that any of the three visible rules could *name* it as a
   defect, so the classifier's residual label was applied to whatever survived.
   Measured on that build: 1,522 of 5,751 diag rows (26.5%) shipped a gold that no
   rule justifies.  The rejected class is not marginal -- the rewrite recasts the
   request instead of breaking the problem: the inserted span is ``how`` (387),
   ``what is`` (237), ``and`` (87), ``what`` (85), ``find`` (71), ``what is the``
   (48), ``its`` (35), ``with`` (30), ``that`` (19); 1,173 of them carry no
   content word at all and the other 349 introduce no content word the answerable
   member lacks (``sum-uns-143`` pointed at ``these`` while the defect was the deleted
   "each with a common base width of 2"; row 35212 pointed at ``How`` in
   "2w+4w+6w+8w+10w+12. How?", a question that is still answerable to a reader).
   The fix is :func:`certifies_as_visible_defect`: the gold must satisfy the rule
   that labelled it, and the residual class must satisfy its own predicate (a
   content word the answerable member never carried).  The pair is then **dropped
   and counted** under ``pointer_insertion_is_not_a_defect`` (1,557 post-dedup)
   rather than rebuilt as another row shape: table B maps an inserted region to
   "SUM-visible", so the shape, not the certificate, decides the branch, and a
   bare row whose pair inserts text would contradict the mapping the design
   documents -- although 1,427 of the 1,557 would also have satisfied the refusal
   certificate, so the choice does cost rows.  ``verify_sum.py`` applies
   the same clause independently in its L1 diag check, so a build that re-admits
   these rows fails the audit -- the first build does, with exactly 1,522
   failures.
14. **DEFECT: the judgment anchor could be a bare function word.**  The anchor is
   the largest deleted span of the pair, and it is handed to
   :func:`mine_option_spans` as its ``gold`` -- which that function appends to the
   option block unconditionally, because rule 3 ("every span carries at least one
   content word") is only applied to the distractors it mines.  So a pair whose
   largest deletion is a function word produced a block with a filler in it: the
   first build shipped 159 such judgment rows (its audit counted 159 filler
   options on the judgment side and 1,173 on the diag side).  Under this clause
   the current artifact loses 125 pairs, out of 855 whose largest deletion is
   content-free -- the rest never reach the judgment certificate, the pointer
   certificate having rejected them first.  No gold was ever wrong
   (a judgment row's gold is SOLVABLE and its block is padding to keep template A
   isomorphic), but the block violated the miner's own rule.  Fixed by the same
   principle as 13, applied by the caller that chose the anchor; the pair then
   emits its diag row with no judgment row, and the audit's L3d checks rule 3 on
   every option of every block (the only exemption is a diag gold that *is* the
   vague or impossible defect -- "some", "few" -- where the function word is the
   defect itself; 104 rows).
15. **DEFECT: the question-clause rule counted text that survived.**  Two bugs on
   one rule, found by the adversarially written copy in ``verify_sum.py``, plus a
   third condition the same row exposed.  (i) The audit located the clause with
   ``str.find`` (first occurrence) where this module reasoned about the last one,
   so the two implementations disagreed on ``sum-uns-25944`` -- a question stated
   twice in the answerable member: the audit found the clause intact at the first
   offset while this module tested the copy the diff actually deleted.  The audit
   now takes its offset from the last sentence break before the final ``?``.
   (ii) Both copies sliced the clause from the sentence break, so it carried the
   separating blank; overlap is now measured against the clause's non-whitespace
   extent.  No SUM row exercises that refinement -- both readings select the same
   2,243 candidate rows -- so it is a guard, not a fix.  (iii) The rule also had
   to require the clause to be **absent** from the presented member:
   ``sum-uns-25944`` quotes the problem and then restates it, so deleting the
   second copy leaves "How much does the gold rod weigh?" standing in the prompt,
   and ``sum-uns-1240`` is the alignment variant, where the word-level diff
   charges the surviving "In" to the deleted run while the prompt keeps the whole
   clause.  Both rows now route to ``missing_necessary_condition`` (2,054 ->
   2,052 ``question_missing``), which is what the design doc's "delete the
   question sentence" definition asks for.

Module note on L1
-----------------

SUM's labels were produced by o3-mini and reviewed by experts, and the file ships
**no derivation chain**: only the final ``ground_truth`` string, nothing that
proves the deleted premise was necessary or that the rewritten question really
became unanswerable (section 4.9.1 table A marks the source "no machine
certificate").  This adapter therefore certifies the *pair* -- the recorded defect
is the only difference between the two members, the pointer gold is absent from
the original and unique in the question, and a refusal row provably lost a content
word -- and it does **not** pretend the necessity of the removed premise was
proven.  Q26's substitute for the missing machine certificate is the human/rule
spot check in ``verify_sum.py`` (N=100, stratified over the five types), and its
result is part of the audit, not an assumption of the adapter.  Measured: 0/100
rows fail their type's defining predicate (the first build read 7/100, all seven
in the residual class, which is the class deviation 13's clause now rejects).

Read-only cross-check: ``verify_sum.py`` re-derives every claim above from the
built parquet plus ``train.parquet`` and fails closed on each one, including the
defect-class certificate of deviation 13 and the option-content rule of
deviation 14.  Two of its checks are deliberately *stronger* than this adapter's
own: L1/L2 cover every emitted row (not a sample), and the source anchor re-reads
the pair, the presented question and the audit answer from the raw file, because
SUM ships no derivation chain that could prove them from the artifact alone.
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
    from distractor_mining import mine_option_spans
except ImportError:  # pragma: no cover - running as a plain script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import schema
    from distractor_mining import mine_option_spans

# ---------------------------------------------------------------------------
# source constants
# ---------------------------------------------------------------------------

DATA_SOURCE = schema.SOURCE_SUM
DEFAULT_RAW_DIR = "/home/charles/data/reasoning_rl/halluc/raw/sum"
DEFAULT_OUT = os.path.expanduser("~/data/reasoning_rl/halluc/built/sum.parquet")


def report_path_for(out: str) -> str:
    """The report that belongs to ``--out``: same stem, ``_report.json``.

    Derived from ``--out`` instead of pinned to the build directory, so that a
    scratch build (``--out /tmp/sum.parquet``) writes a scratch report rather than
    overwriting the canonical one the build report cites.  Same convention in all
    four adapters.
    """
    return os.path.splitext(out)[0] + "_report.json"


DATA_FILE = "train.parquet"
TEST_FILE = "test.parquet"  # read for the report only; it supplies no pair

BRANCH_DIAG = schema.BRANCH_UNSOLVABLE_DIAG
BRANCH_BARE = schema.BRANCH_UNSOLVABLE_BARE
BRANCH_JUDGE = schema.BRANCH_SOLVABLE_JUDGE

K_OPTIONS = 3  # design decision D15

# The five defect types of design doc section 4.9.3's balance table, in the
# names the shared context uses.  They are *inferred* -- the source carries no
# label -- and the classifier is monitoring-only: reward reads ``solvable`` and
# the gold, never this field.
MISSING_NECESSARY_CONDITION = "missing_necessary_condition"
AMBIGUOUS_KEY_INFORMATION = "ambiguous_key_information"
UNREALISTIC_SELF_CONTRADICTION = "unrealistic_self_contradiction"
IRRELEVANT_UNDEFINED_ENTITY = "irrelevant_undefined_entity"
QUESTION_MISSING = "question_missing"

DEFECT_TYPES = (
    MISSING_NECESSARY_CONDITION,
    AMBIGUOUS_KEY_INFORMATION,
    UNREALISTIC_SELF_CONTRADICTION,
    IRRELEVANT_UNDEFINED_ENTITY,
    QUESTION_MISSING,
)

# defect type -> (perturbation_type from schema.PERTURBATION_TYPES, branch).
# All five land on the two unsolvable branches: the visible defects are pointed
# at (diag), the absences can only be refused (bare).
DEFECT_SPECS: dict[str, tuple[str, str]] = {
    MISSING_NECESSARY_CONDITION: ("missing_condition", BRANCH_BARE),
    QUESTION_MISSING: ("question_missing", BRANCH_BARE),
    UNREALISTIC_SELF_CONTRADICTION: ("unrealistic_condition", BRANCH_DIAG),
    AMBIGUOUS_KEY_INFORMATION: ("ambiguous_condition", BRANCH_DIAG),
    IRRELEVANT_UNDEFINED_ENTITY: ("unrelated_entity", BRANCH_DIAG),
}

# The visible-defect types, in priority order: the first rule that fires labels
# the inserted span.  ``unrelated_entity`` is the residual class -- any inserted
# content that is neither a vague quantifier nor an impossible value introduces
# an entity the original problem never mentioned.  The order is what makes the
# classifier a function (one label per row) and it is rechecked by the audit.
DIAG_PRIORITY = (
    UNREALISTIC_SELF_CONTRADICTION,
    AMBIGUOUS_KEY_INFORMATION,
    IRRELEVANT_UNDEFINED_ENTITY,
)

# A minus-signed number (whitespace/paren preceded, so that "x-1" -- a
# subtraction -- does not fire) or an impossibility word.  Loaded dice such as
# "a negative prime number" or "assume tan φ is negative" are the archetype.
_IMPOSSIBLE_RE = re.compile(
    r"(?:^|[\s(\[{])[-−]\s?\d"
    r"|\b(?:negative|impossible|undefined|imaginary|infinite|infinity"
    r"|cannot exist|does not exist|no solution)\b",
    re.I,
)

# A vague quantifier: the inserted span refuses to name the value the problem
# needs.  Deliberately excludes "about"/"around"/"different", which are ordinary
# English ("a circle about the origin") rather than vagueness markers.
_VAGUE_RE = re.compile(
    r"\b(?:some|several|few|many|most|either|certain|various|multiple|numerous"
    r"|couple|approximately|roughly|nearly|unknown|unclear|unspecified|arbitrary"
    r"|varies|varying|fluctuat\w*|depends|random|unstated|unpredictable|unfixed"
    r"|not\s+(?:specified|stated|fixed|defined|given)|isn[’']t\s+fixed)\b",
    re.I,
)

# The defect must be pointerable with a word that means something: a one-character
# gold ("x", "-") makes an option block worse than no option block at all.
MIN_GOLD_CHARS = 2

_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’\-]*")


# ---------------------------------------------------------------------------
# text plumbing
# ---------------------------------------------------------------------------


def normalise_question(text: str) -> str:
    """Collapse whitespace and strip.

    The source carries embedded newlines (6,106 answerable questions), tabs,
    leading/trailing spaces and non-breaking-ish control characters.  Every
    offset in this module is computed on the *normalised* text, which is also
    what is rendered into the prompt -- mixing offsets from the raw and the
    normalised string is the bug this rule exists to prevent.
    """
    return _WHITESPACE_RE.sub(" ", (text or "")).strip()


def _tokens(text: str) -> list[tuple[str, int, int]]:
    """``[(token, start, end)]`` -- the same token grammar as distractor_mining."""
    return [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


class _Region:
    """One contiguous run of changed word tokens between two questions."""

    __slots__ = ("a_text", "b_text", "a_start", "b_start", "tag")

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

    Diffing token sequences (rather than characters) keeps every gold a whole
    word span.  ``a_start`` / ``b_start`` are character offsets into the two
    strings, so a gold taken from ``b_text`` can be located in the rendered
    question at the region's own offset.
    """
    ta, tb = _tokens(qa), _tokens(qu)
    matcher = difflib.SequenceMatcher(
        None, [t[0] for t in ta], [t[0] for t in tb], autojunk=False
    )
    runs = _merge_opcodes(matcher.get_opcodes())

    regions: list[_Region] = []
    for tag, i1, i2, j1, j2 in runs:
        a_start = ta[i1][1] if i1 < len(ta) else len(qa)
        a_end = ta[i2 - 1][2] if i2 > i1 else a_start
        b_start = tb[j1][1] if j1 < len(tb) else len(qu)
        b_end = tb[j2 - 1][2] if j2 > j1 else b_start
        regions.append(
            _Region(
                tag=tag,
                a_text=qa[a_start:a_end],
                b_text=qu[b_start:b_end],
                a_start=a_start,
                b_start=b_start,
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


def _question_clause(question: str) -> tuple[str, int]:
    """The final interrogative sentence of ``question`` and its offset.

    ``("", -1)`` when there is no ``?`` at all.  The offset is what the "was the
    question clause cut?" rule needs; the text is what the audit re-checks.
    """
    mark = question.rfind("?")
    if mark < 0:
        return "", -1
    start = question.rfind(".", 0, mark)
    return question[start + 1 : mark + 1], start + 1


def _clause_cut(qa: str, qu: str, regions: list[_Region]) -> bool:
    """Whether the question clause was deleted and is gone from the question.

    A cut question clause is invisible in the presented text -- there is nothing
    to point at -- so this is the rule that separates ``question_missing`` from
    the other bare deletions.  Re-derived independently by ``verify_sum.py``.

    Two conditions, both necessary.  The overlap is measured against the clause's
    own text, not the whitespace that separated it from the sentence before: the
    clause as sliced always begins with that blank run, so a rule measured
    against the slice could read a deletion that merely touches the blank as a
    cut question (no SUM row exercises the difference -- the refinement is a
    guard, deviation 15).  And the clause must be **absent** from the presented
    member: the clause can occur twice in the answerable member, or a word-level
    alignment can charge a surviving word to the deleted run, and in either case
    the prompt still asks the question, so the pair lost a condition rather than
    its question (measured: ``sum-uns-25944`` and ``sum-uns-1240``, the two rows
    of 2,054 this second condition moved off ``question_missing``).
    """
    clause, offset = _question_clause(qa)
    if not clause:
        return False
    stripped = clause.strip()
    if not stripped:
        return False
    lead = len(clause) - len(clause.lstrip())
    first, last = offset + lead, offset + lead + len(stripped)
    overlaps = any(
        region.a_text.strip()
        and region.a_start < last
        and first < region.a_start + len(region.a_text)
        for region in regions
    )
    return overlaps and stripped not in qu


# ---------------------------------------------------------------------------
# the rule-based defect classifier (monitoring only -- never in reward)
# ---------------------------------------------------------------------------


def classify_visible_defect(inserted: str) -> str:
    """Label one *inserted* defect span with one of the three visible types.

    First match wins in :data:`DIAG_PRIORITY` order: an impossible value beats a
    vague quantifier (a span may contain both -- "a negative unknown number"),
    and anything else is the residual class ``irrelevant_undefined_entity``,
    because the span introduces content the original problem never carried.

    This classifier serves the D18 balance table and drift monitoring only.  The
    reward never reads the field, so a mislabel costs a monitoring cell, not a
    training signal -- which is also why the audit only reports its failure rate.
    """
    if _IMPOSSIBLE_RE.search(inserted):
        return UNREALISTIC_SELF_CONTRADICTION
    if _VAGUE_RE.search(inserted):
        return AMBIGUOUS_KEY_INFORMATION
    return IRRELEVANT_UNDEFINED_ENTITY


def classify_bare_defect(qa: str, qu: str, regions: list[_Region]) -> str:
    """Label a deletion-only pair as ``question_missing`` or ``missing_condition``."""
    return QUESTION_MISSING if _clause_cut(qa, qu, regions) else MISSING_NECESSARY_CONDITION


def introduces_undefined_entity(inserted: str, qa: str) -> bool:
    """Whether the inserted span carries a content token absent from the original.

    The defining predicate of ``irrelevant_undefined_entity``: the audit applies
    the same rule independently, and the rows where the residual label does not
    satisfy it are the classifier's reported failure rate.
    """
    qa_tokens = {token.lower() for token in schema.words(qa)}
    return any(
        schema.is_content_word(token) and token.lower() not in qa_tokens
        for token in schema.words(inserted)
    )


def certifies_as_visible_defect(inserted: str, qa: str) -> bool:
    """Whether one of the three visible rules actually justifies this insertion.

    The rule is the classifier's own, read as a *certificate* rather than as a
    label: a span labelled ``unrealistic_self_contradiction`` or
    ``ambiguous_key_information`` is justified by the rule that fired, and the
    residual ``irrelevant_undefined_entity`` is justified only when it satisfies
    its defining predicate -- it must introduce a content word the answerable
    member never carried.  Anything else (``"and the"``, ``"its"``, ``"these"``,
    a sentence-initial capital that a replace left behind) is not a defect at
    all: it is padding the paraphrase happened to add, and a gold pointing at it
    cannot be justified by anything the row contains.  See deviation 13.
    """
    if _IMPOSSIBLE_RE.search(inserted) or _VAGUE_RE.search(inserted):
        return True
    return introduces_undefined_entity(inserted, qa)


# ---------------------------------------------------------------------------
# source loading
# ---------------------------------------------------------------------------


def load_source(raw_dir: str) -> list[dict]:
    """Read ``train.parquet`` and stamp each row with its positional index.

    The file has no id column, so ``_index`` (the row's position in the frozen
    file) is the row identity: it is what makes the artifact's ``task_id``
    reproducible and what lets the audit re-read the exact source row.  The test
    split is deliberately not read -- it has no ``answerable_question`` (see
    deviation 1).
    """
    path = os.path.join(raw_dir, DATA_FILE)
    rows = schema.read_parquet_rows(path)
    for index, row in enumerate(rows):
        row["_index"] = index
    return rows


def _is_well_formed(row: object) -> bool:
    """A row is usable when all three texts are nonempty strings."""
    if not isinstance(row, dict):
        return False
    for key in ("answerable_question", "unanswerable_question", "ground_truth"):
        value = row.get(key)
        if not isinstance(value, str) or not value.strip():
            return False
    return True


def _pair_key(row: dict) -> tuple[str, str]:
    """The dedup key: the normalised question pair (never the question alone).

    144 answerable questions carry more than one ``ground_truth`` in this file,
    so a key built from one member would silently graft another row's answer onto
    this row's question.
    """
    return (
        normalise_question(row["answerable_question"]),
        normalise_question(row["unanswerable_question"]),
    )


# ---------------------------------------------------------------------------
# certificates
# ---------------------------------------------------------------------------


def _certify_pointer(
    qa: str, qu: str, emitting: list[_Region], rng: random.Random
) -> tuple[dict | None, str]:
    """Pointer gold for a visible-defect pair, or ``(None, reason)``.

    Five clauses, and the caller **drops** the pair on the first one that fails,
    counting ``reason``.  A pair rejected here does not become a row of another
    shape: table B maps an inserted region to "SUM-visible", so a pair that
    cannot certify as visible is not a deletion pair either (the branch is a
    function of the pair's shape, not of what the certificates managed).
    """
    if len(emitting) != 1:
        return None, "pointer_not_one_inserted_region"
    region = emitting[0]
    gold = region.inserted.strip()
    if len(gold) < MIN_GOLD_CHARS:
        return None, "pointer_gold_too_short"
    # The gold must be introduced by the defect: if the same text already sits in
    # the answerable member it cannot be what made this variant unanswerable.
    # (Checked before the defect-class clause below, which would otherwise
    # absorb it: a gold that occurs in the answerable member has no content word
    # the original lacks, so it fails that clause too and would be counted there.)
    if gold in qa:
        return None, "pointer_gold_in_answerable_member"
    # ... and it must be locatable at the region's own offset, not at an earlier
    # accidental occurrence of the same string.
    if qu.find(gold) != region.b_start:
        return None, "pointer_gold_off_its_region_offset"
    # The inserted text has to *be* a defect of one of the three visible types;
    # a paraphrase that pads the sentence while it drops a premise leaves an
    # insertion no rule can name, and pointing the model at that padding is a
    # gold nothing in the row justifies (measured: 1,522 pairs, deviation 13).
    if not certifies_as_visible_defect(gold, qa):
        return None, "pointer_insertion_is_not_a_defect"
    mined = mine_option_spans(qu, gold, k=K_OPTIONS, rng=rng)
    if mined is None:
        return None, "pointer_question_cannot_host_options"
    texts, correct = mined
    return {"gold": gold, "texts": texts, "correct": correct}, ""


def _certify_refusal(qa: str, qu: str, regions: list[_Region]) -> bool:
    """Whether "this question lost what it needs" is provable from the text.

    The deleted text must carry a content word, and that word must no longer
    appear anywhere in the presented question.  A deleted word that still stands
    elsewhere in the question is not evidence that information was lost.
    """
    deleted = " ".join(r.a_text for r in regions if r.a_text.strip())
    if not deleted:
        return False
    qu_tokens = {token.lower() for token in schema.words(qu)}
    return any(
        schema.is_content_word(token) and token.lower() not in qu_tokens
        for token in schema.words(deleted)
    )


def _certify_judge(
    qa: str, regions: list[_Region], rng: random.Random
) -> tuple[dict | None, str]:
    """Option anchor for the answerable member of a certified pair.

    The anchor is the largest deleted span of the pair -- the same edit that made
    the partner unanswerable is what the option block is built around, so the
    judgment row and the diag row of one pair are isomorphic (same question pair,
    same option count, same length rule) with opposite golds.

    Returns ``(None, reason)`` when the pair cannot host a judgment row.  The
    anchor is what :func:`mine_option_spans` appends to its own distractors as
    the "gold", and that function's rule 3 promises every option of the block it
    returns carries a content word -- a promise it cannot keep for a caller that
    hands it a bare function word, which is why the clause is here (deviation 14).
    """
    anchors = [r.a_text.strip() for r in regions if r.a_text.strip()]
    if not anchors:
        return None, "judge_no_deleted_anchor"
    anchor = max(anchors, key=len)
    if not any(schema.is_content_word(token) for token in schema.words(anchor)):
        return None, "judge_anchor_has_no_content_word"
    mined = mine_option_spans(qa, anchor, k=K_OPTIONS, rng=rng)
    if mined is None:
        return None, "judge_question_cannot_host_options"
    texts, _ = mined
    return {"anchor": anchor, "texts": texts}, ""


# ---------------------------------------------------------------------------
# row construction
# ---------------------------------------------------------------------------


def _task_id(index: int, side: str) -> str:
    """Stable hard-replay key: the source row's position plus which member.

    One pair yields at most two rows (the answerable member's judgment row and
    the unanswerable member's diag/bare row), so the side suffix is what keeps
    the two apart.
    """
    return f"sum-{side}-{index}"


def _base_extra_info(row: dict, defect_type: str, seed: int) -> dict:
    perturbation, _ = DEFECT_SPECS[defect_type]
    return {
        "split": "train",  # the source ships one file and no split column
        "index": row["_index"],
        "seed": seed,
        "difficulty": "",  # SUM has no difficulty axis at all
        "perturbation_type": perturbation,
        "perturbation_family": "",
        "error_type": defect_type,
    }


def build_rows(
    raw_dir: str, limit: int | None = None, seed: int = 0
) -> tuple[list[dict], dict]:
    """Build the SUM parquet rows and the funnel that produced them.

    Returns ``(rows, funnel)``: ``rows`` are ready for
    :func:`schema.normalise_extra_info` / :func:`schema.validate_rows` /
    :func:`schema.write_rows_parquet`, and ``funnel`` is an ordered mapping of
    filter stage -> count.  The first six stages count **pairs**; the last two
    count **rows**, because one certified pair can supply both a judgment row
    (the answerable member) and a diag row (the unanswerable member).  The last
    key, ``certificate_drops``, is not a stage but a nested mapping: for every
    certificate rejection, the clause that rejected it (fail closed -- a rejected
    pair is counted, never patched into some other row shape).  It holds two
    units, and ``main()`` prints them apart: a ``pointer_*`` / refusal clause
    rejects the whole pair, while a ``judge_*`` clause only withholds the
    judgment row, the pair still emitting its diag row -- so the ``judge_*``
    counts do not subtract from ``after_defect_certificate_drop``.

    The same ``(raw_dir, limit, seed)`` always produces byte-identical rows:
    every random choice is made by a per-row :class:`random.Random` seeded from
    ``(seed, index)``.
    """
    raw = load_source(raw_dir)
    funnel: dict[str, int] = collections.OrderedDict()
    funnel["raw_rows"] = len(raw)

    stage = [row for row in raw if _is_well_formed(row)]
    funnel["after_malformed_drop"] = len(stage)

    # 1. duplicate pairs.  Groups with conflicting golds are contradictory input
    #    (rows 436/2528 are the same text pair with golds '2' and '3'); groups of
    #    same-gold duplicates are the same row twice.  Fail closed on the first,
    #    keep the lowest index on the second.
    groups: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for row in stage:
        groups[_pair_key(row)].append(row)
    conflicting: set[int] = set()
    duplicate_of: set[int] = set()
    for members in groups.values():
        if len(members) < 2:
            continue
        golds = {member["ground_truth"].strip() for member in members}
        if len(golds) > 1:
            conflicting.update(member["_index"] for member in members)
            continue
        ordered = sorted(members, key=lambda item: item["_index"])
        duplicate_of.update(member["_index"] for member in ordered[1:])
    stage = [
        row for row in stage if row["_index"] not in conflicting and row["_index"] not in duplicate_of
    ]
    funnel["after_duplicate_pair_drop"] = len(stage)

    # 2. the word-level diff.  A pair with no word-level change carries no
    #    certifiable defect (deviation 10: 114 of the 116 are operator-only
    #    flips the tokenizer cannot see).
    diffed: list[tuple[dict, str, str, list[_Region]]] = []
    no_diff = 0
    residual_fail = 0
    for row in stage:
        qa = normalise_question(row["answerable_question"])
        qu = normalise_question(row["unanswerable_question"])
        regions = word_regions(qa, qu)
        if not regions:
            no_diff += 1
            continue
        if not _residual_tokens(qa, qu, regions):  # pragma: no cover - by construction
            residual_fail += 1
            continue
        diffed.append((row, qa, qu, regions))
    funnel["after_no_word_diff_drop"] = len(diffed)

    # 3. branch routing by the diff's shape.  More than one inserted region means
    #    the model would have to guess which defect the option block points at.
    single: list[tuple[dict, str, str, list[_Region], list[_Region]]] = []
    multi = 0
    for row, qa, qu, regions in diffed:
        emitting = [region for region in regions if region.b_text.strip()]
        if len(emitting) > 1:
            multi += 1
            continue
        single.append((row, qa, qu, regions, emitting))
    funnel["after_multi_region_drop"] = len(single)

    # 4. certificates.  A deletion-only pair must prove information was lost; a
    #    pair with one inserted region must yield a pointer gold that a visible
    #    rule can name.  A pair that certifies as a pointer pair additionally
    #    yields the answerable member's judgment row when that member can host an
    #    equal-length option block.  Every rejection is counted under the clause
    #    that caused it.
    built: list[dict] = []
    certified_pairs = 0
    drops: dict[str, int] = collections.OrderedDict()

    def drop(reason: str) -> None:
        """Count one rejected pair under the certificate clause that rejected it."""
        drops[reason] = drops.get(reason, 0) + 1

    for row, qa, qu, regions, emitting in single:
        index = row["_index"]
        deleted = " ".join(r.a_text for r in regions if r.a_text.strip())
        inserted = " ".join(r.b_text.strip() for r in emitting if r.b_text.strip())
        if not emitting:
            if not _certify_refusal(qa, qu, regions):
                drop("refusal_lost_no_content_word")
                continue
            defect_type = classify_bare_defect(qa, qu, regions)
            plan: dict = {"kind": "bare", "defect_type": defect_type}
        else:
            pointer, reason = _certify_pointer(
                qa, qu, emitting, random.Random(f"{seed}:ptr:{index}")
            )
            if pointer is None:
                drop(reason)
                continue
            defect_type = classify_visible_defect(pointer["gold"])
            plan = {"kind": "diag", "defect_type": defect_type, "pointer": pointer}
            judge, reason = _certify_judge(qa, regions, random.Random(f"{seed}:judge:{index}"))
            if judge is not None:
                plan["judge"] = judge
            else:
                drop(reason)
        certified_pairs += 1
        extra = _base_extra_info(row, plan["defect_type"], seed)
        extra["deleted_condition_text"] = deleted
        extra["perturbed_entity_text"] = inserted
        plan["qa"], plan["qu"], plan["extra"] = qa, qu, extra
        plan["ground_truth"] = row["ground_truth"].strip()
        if plan["kind"] == "bare":
            built.append(_bare_row(plan))
        else:
            built.append(_diag_row(plan))
            if "judge" in plan:
                built.append(_judge_row(plan))
    funnel["after_defect_certificate_drop"] = certified_pairs
    funnel["certificate_drops"] = drops

    rows = _interleave_by_branch(built)
    if limit is not None:
        rows = rows[: max(limit, 0)]
    funnel["rows_before_limit"] = len(built)
    funnel["after_limit"] = len(rows)
    return rows, funnel


# ---------------------------------------------------------------------------
# the three emitted row shapes
# ---------------------------------------------------------------------------


def _option_block(texts: list[str]) -> list[dict]:
    """Render mined option spans as ``[{"id": "A", "text": ...}]``."""
    return [{"id": chr(ord("A") + i), "text": text} for i, text in enumerate(texts)]


def _bare_row(plan: dict) -> dict:
    perturbation, _ = DEFECT_SPECS[plan["defect_type"]]
    extra = dict(plan["extra"])
    extra.update(
        {
            "task_id": _task_id(plan["extra"]["index"], "uns"),
            "solvable": False,
            # the other member of the pair, so the audit can re-read the source
            # row's own text instead of trusting the rendered question
            "paired_original_text": plan["qa"],
        }
    )
    ground_truth = schema.build_ground_truth(
        solvable=False,
        answer=None,
        correct_option_id=None,
        has_diagnosis_label=False,
        perturbation_type=perturbation,
    )
    return schema.make_row(
        data_source=DATA_SOURCE,
        question=plan["qu"],
        ground_truth=ground_truth,
        template=schema.TEMPLATE_B,
        branch=BRANCH_BARE,
        extra_info=extra,
    )


def _diag_row(plan: dict) -> dict:
    perturbation, _ = DEFECT_SPECS[plan["defect_type"]]
    pointer = plan["pointer"]
    extra = dict(plan["extra"])
    extra.update(
        {
            "task_id": _task_id(plan["extra"]["index"], "uns"),
            "solvable": False,
            "paired_original_text": plan["qa"],
            "correct_option_id": pointer["correct"],
        }
    )
    ground_truth = schema.build_ground_truth(
        solvable=False,
        answer=None,
        correct_option_id=pointer["correct"],
        has_diagnosis_label=True,
        perturbation_type=perturbation,
    )
    return schema.make_row(
        data_source=DATA_SOURCE,
        question=plan["qu"],
        ground_truth=ground_truth,
        template=schema.TEMPLATE_A,
        branch=BRANCH_DIAG,
        extra_info=extra,
        options=_option_block(pointer["texts"]),
    )


def _judge_row(plan: dict) -> dict:
    perturbation, _ = DEFECT_SPECS[plan["defect_type"]]
    extra = dict(plan["extra"])
    extra.update(
        {
            "task_id": _task_id(plan["extra"]["index"], "ans"),
            "solvable": True,
            "judgment_only": True,
            "paired_original_text": plan["qu"],
            "correct_option_id": "",
        }
    )
    ground_truth = schema.build_ground_truth(
        solvable=True,
        answer=plan["ground_truth"],
        judgment_only=True,
        perturbation_type=None,
    )
    return schema.make_row(
        data_source=DATA_SOURCE,
        question=plan["qa"],
        ground_truth=ground_truth,
        template=schema.TEMPLATE_A,
        branch=BRANCH_JUDGE,
        extra_info=extra,
        options=_option_block(plan["judge"]["texts"]),
    )


def _interleave_by_branch(rows: list[dict]) -> list[dict]:
    """Round-robin the branches so a ``limit`` keeps every branch represented.

    Without this a small ``--limit`` would return only judgment rows, and the
    anti-cheat audit -- which needs both the solvable and the unsolvable side --
    would have nothing to measure.
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


def test_split_note(raw_dir: str) -> str:
    """One line about ``test.parquet`` for the report (it supplies no pair)."""
    path = os.path.join(raw_dir, TEST_FILE)
    if not os.path.exists(path):
        return f"{TEST_FILE}: absent"
    rows = schema.read_parquet_rows(path)
    columns = sorted(rows[0]) if rows else []
    golds = {row.get("ground_truth", "") for row in rows}
    return (
        f"{TEST_FILE}: {len(rows)} rows, columns={columns}, "
        f"distinct ground_truth={len(golds)} -- no answerable_question, so it pairs nothing"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the SUM hallucination-domain rows.")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--report", default=None, help="JSON report path (default: alongside --out)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report_path = args.report or report_path_for(args.out)

    rows, funnel = build_rows(args.raw_dir, limit=args.limit, seed=args.seed)
    schema.normalise_extra_info(rows)
    schema.validate_rows(rows)
    schema.write_rows_parquet(rows, args.out)

    branches = _breakdown(rows, lambda row: row["extra_info"]["branch"])
    templates = _breakdown(rows, lambda row: row["extra_info"]["template"])
    solvable = _breakdown(rows, lambda row: row["extra_info"]["solvable"])
    error_types = _breakdown(rows, lambda row: row["extra_info"]["error_type"])

    print(f"raw dir : {args.raw_dir}")
    print(f"wrote   : {args.out}  ({len(rows)} rows, seed={args.seed})")
    print(f"---- {test_split_note(args.raw_dir)}")
    print("\nfunnel (rows remaining after each stage, and what that stage cost):")
    previous = None
    for stage, count in funnel.items():
        if not isinstance(count, int):  # certificate_drops is a nested mapping
            continue
        cost = "" if previous is None else f"   {previous - count:+d}"
        print(f"  {stage:34s} {count:6d}{cost}")
        previous = count
    # Two units in one mapping, so print them apart: a pointer or refusal clause
    # rejects the whole pair (no row at all), while a ``judge_*`` clause only
    # withholds the judgment row -- the pair is still certified and still emits
    # its diag row, which is why those counts do not subtract from the funnel.
    drops = funnel["certificate_drops"]
    pair_clauses = sorted(
        (reason for reason in drops if not reason.startswith("judge_")),
        key=lambda reason: (-drops[reason], reason),
    )
    judge_clauses = sorted(
        (reason for reason in drops if reason.startswith("judge_")),
        key=lambda reason: (-drops[reason], reason),
    )
    print("\npairs dropped by the clause that rejected them (no row emitted, fail closed):")
    for reason in pair_clauses:
        print(f"  {reason:38s} {drops[reason]:6d}")
    print(f"  {'= pairs lost':38s} {sum(drops[r] for r in pair_clauses):6d}")
    print("\njudgment rows not emitted (the pair still emits its diag row):")
    for reason in judge_clauses:
        print(f"  {reason:38s} {drops[reason]:6d}")
    print("\nper branch:")
    for branch, count in branches.items():
        print(f"  {branch:22s} {count}")
    print("\nper template:")
    for template, count in templates.items():
        print(f"  {template:22s} {count}")
    print("\nper solvable:")
    for value, count in solvable.items():
        print(f"  {str(value):22s} {count}")
    print("\nper error_type:")
    for error_type, count in error_types.items():
        print(f"  {error_type:26s} {count}")

    report = {
        "raw_dir": args.raw_dir,
        "out": args.out,
        "rows": len(rows),
        "seed": args.seed,
        "limit": args.limit,
        "funnel": funnel,
        "per_branch": branches,
        "per_template": templates,
        "per_solvable": {str(key): value for key, value in solvable.items()},
        "per_error_type": error_types,
        "test_split": test_split_note(args.raw_dir),
    }
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"\nreport  : {report_path}")


if __name__ == "__main__":  # pragma: no cover - exercised through main() in tests
    main()
