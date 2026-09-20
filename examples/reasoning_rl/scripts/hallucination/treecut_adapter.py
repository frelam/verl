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
"""TreeCut adapter -- generated math pairs shipped on both sides of the contract.

Source: ``github.com/j-bagel/treecut-math`` (Apache-2.0, pure Python, no
dependencies; the HF mirror ``jouyang/treecut-math`` holds 21,000 samples of the
same generator).  Unlike every other source in this pool TreeCut is a *generator*,
not a file: ``treecut/gen_data.py`` samples a dependency tree over
``numVars`` variables, renders one sentence per tree edge, and -- when the row is
meant to be unanswerable -- deletes one whole sentence.  The adapter runs the
pinned generator (``<raw_dir>/repo/treecut``, read out of the downloaded repo, not
vendored) and records the seam the design doc asks for.

Contract produced by this adapter (design doc D26; sections 4.6, 4.8 table B rows
5 and 7, 5.1, 5.2, 9):

=================  ====================  ========  ============================================
side               branch                template  gold
=================  ====================  ========  ============================================
negative (4,907)   ``unsolvable_diag``   A         ``\\boxed{UNSOLVABLE: <选项ID>}`` (four-tier)
positive (500)     ``solvable_numeric``  A         ``\\boxed{<答案>}`` (numeric match)
=================  ====================  ========  ============================================

Table B row 5 (TreeCut positives) shares ``solvable_numeric`` with row 1 (GSM-IC
and the synthesised distractors); the mix tells the two apart by ``data_source``
(schema.py's branch registry).

Both sides carry a template-A option block of exactly k=3 (D15) and the two
blocks are structurally identical -- that isomorphism is the D18 hard constraint
D26 re-states: without it, "the prompt carries an option block" alone would
predict UNSOLVABLE.

* **negative (table B row 7, four-tier)**: the options are *candidate missing
  conditions*.  The gold is the edge the generator cut
  (``cut = ans_upstream[cutDepth - 1]``, the answer's own parent -- its sentence
  is the row's ``deleted_condition_text``); the two distractors are cut sentences
  of sibling scenarios **in the same grid cell** (``render_scenario`` yields
  ``sentence_dict`` and ``member_of`` extracts a cut's sentence, so no new
  generator is involved).  All three options are of the same family and all three
  are absent from the shipped passage *verbatim*, which keeps MiP's "pick the
  option that is not in the passage" shortcut (95.6% on MiP, section 4.2) at
  chance ~1/k here -- section 9's hard gate is ``<= 1/k + 5pt = 38.3%`` (D26,
  section 11 risk 1).
* **positive (table B row 5, solvable numeric)**: the options are *placeholders*
  -- k=3 passage conditions, each with one variable name **or** one variable value
  swapped (Q16: 50/50 random, required only to be absent from the passage;
  semantic plausibility is explicitly not required), with no correct item and
  ``correct_option_id=null``.  The gold is the generator's answer
  (``render["answer"]``) and section 6 routes the row through the ordinary
  solvable branch: ``\\boxed{<答案>}`` matching gold = +1, a misrefusal
  (``UNSOLVABLE`` in any form) = 0, never -1 (D26).  The placeholder block never
  enters the reward.

TreeCut's own generation-time cut removes an edge that is *provably on the
root-to-answer path*, which is what makes the unanswerability a construction fact
rather than a human label: after the cut, the answer variable sits in a component
that has N variables and only N-1 equations (``gen_disproof``'s certificate).
Section 4.6's pitfall and section 11 risk 1 record the cost of using the stock
generator as-is: its own measurement puts the two classes 48 characters apart on
average (351.6 vs 303.6), the length heuristic reads 0.726 and the BoW NB 0.755 --
a structural leak (this adapter re-measures both figures on the released files;
see DEVIATIONS item 5).  The fix this adapter implements is the recon's
``FIX matched paired``:

* the shipped member cuts a **necessary** edge (the parent of the answer node),
  so the pair's own disproof certificate proves it unsolvable;
* the counterpart cuts one **matched non-necessary** edge -- same parent kind
  (ROOT iff the shipped cut is at the root, i.e. ``ansDepth == 2``), and its child
  must be a non-leaf so no variable vanishes -- so both members are the *same
  scenario* with the same variable set, the same sentence count and, measured,
  the same length distribution;
* the counterpart keeps the answer reachable from the root fact, which is the
  "removed edge is provably irrelevant" half of the certificate.

Certificates (fail closed; each is re-derived by ``verify_treecut.py``)
---------------------------------------------------------------------

* **Construction (proof shape)** -- ``extra_info.proof`` must carry the
  generator's disproof certificate: "There are N variables but only M linear
  formula(s), so we cannot calculate the price of <asked variable>", with
  ``M < N`` and ``M == N - 1``.  Re-derived from the text: N must equal the number
  of variables in the asked variable's own component and M the number of
  sentences inside it, and every clause the certificate cites must be one of the
  sentences the shipped question actually carries.
* **Graph (solvability)** -- the row's own text is turned back into a graph
  (variables are nodes, each body sentence is an edge between the one or two
  variables it names, and the single-variable sentences are the root facts).  The
  *unsolvable* member (the four-tier negative) must have **no path** from any fact
  to the asked variable; the *solvable* member (the positive, which is the
  counterpart of the matched pair) must have one, of exactly ``ans_depth - 1``
  hops.
* **Pairing** -- the two members must be the same scenario: same asked variable,
  same variable set (``num_vars`` of them), ``num_vars - 1`` sentences each, and a
  body-sentence multiset symmetric difference of exactly one sentence on each
  side, each side's being its own ``deleted_condition_text``.  The solvable
  member's own derivation must exist at ``ans_depth - 1`` hops and must **not** use
  the sentence it itself lost -- of the pair's two cuts, only the unsolvable
  member's touches the root-to-answer chain -- and adding the unsolvable member's
  recorded deleted sentence back to its text must make the answer reachable again
  at ``ans_depth - 1`` hops.  Together those two halves are "this sentence is the
  pivot": removing it breaks the chain, restoring it rebuilds the chain.
* **Answer (arithmetic)** -- the positive's gold is the generator's own
  ``render["answer"]``, and the certificate re-derives it from the passage's text
  alone: walk the certified fact-to-answer chain, decode each sentence by
  enumerating the formulas the pinned renderer could have produced for it
  (``x, y in {1,2,3} x {-3..3}`` and the sentence's own numbers as candidate
  results), and propagate the value down the chain.  The propagated value must
  equal the recorded gold.  This is what makes shipping a *number* fail-closed:
  the value is not trusted because the generator computed it, it is re-derived
  from the sentence set the row ships.  Because only a shipped positive needs its
  number proved, the certificate runs on the pairs that can reach the positive
  quota; a pair that fails it keeps its four-tier negative and loses only the
  positive.

A row whose certificate cannot be re-derived from its own two texts is dropped
and counted in the funnel; nothing is invented, no counterpart text is reused
across questions, and there is no fallback.

Which rows ship (deterministic two-sided interleave)
----------------------------------------------------

The negative stream is the 12-cell grid round-robined (``_interleave_by_cell``),
so a small ``--limit`` and the L3 estimators both see every configuration instead
of reading a configuration difference as a label difference.  **Positives are the
positive members of the first ``POS_TARGET`` pairs of that stream whose answer
certificate and placeholder block can both be built** (4,907 pairs, 500
positives; a pair that fails either gate loses only its positive, never the
negative, and the next pair takes the slot).  The emitted row stream is a
deterministic two-sided merge (``_merge_two_sided``): after the k-th negative it
carries ``ceil(k * n_pos / n_neg)`` positives, so ``--limit 2`` already returns
both sides and ``--limit`` cannot silently produce a single-sided artifact.
``--limit`` therefore means "the first N rows of the merged stream", and the same
``(raw_dir, limit, seed)`` always produces byte-identical rows.

DEVIATIONS FROM THE DESIGN DOC
------------------------------

Measured against ``HALLUCINATION_RL_DESIGN.md`` (final revision D1-D27: sections
4.2, 4.6, 4.8 table B rows 5 and 7, 5.1, 5.2, 6, 9, 11 risk 1, and Q8 / Q11 /
Q16), ``scratch/halluc_recon/`` (the recon's ground truth: ``final_summary.json``,
``hf_stats.json``, ``probe_treecut.py``) and the released HF files under
``<raw_dir>/hf``.  Every number below is measured by this adapter; the funnel
printed by ``main()`` shows where each row is lost.

1. **The doc's fix sentence reads inverted; the reading used here is the recon's
   measured one.**  Section 4.6 says "负类不取'完整题'，改取'剪掉一条非必要边'的版本
   （句子数、变量数、长度分布全对齐，**可解性不变**）".  Taken literally, the
   unanswerable member would be a non-necessary-edge cut, i.e. still solvable, and
   the row could not carry ``\\boxed{UNSOLVABLE}`` -- and it would contradict the
   same section's own L1 argument ("剪掉的边必在根→答案路径上").  The reading that
   survives both the argument and the measurement is: unsolvable member =
   necessary-edge cut, solvable member = matched non-necessary-edge cut.
   Measured on the 12-cell grid below (4,907 pairs, 0 drops; the L3 corpus is the
   two *shipped* sides): the numbers are printed by ``verify_treecut.py`` on every
   build, against the recon's *recorded* ``FIX matched paired`` 0.4964 / 0.4695
   and ``FIX naive paired`` 0.5221 / 0.5052 (``final_summary.json``).  The recon's
   recorded artifact is the only citable reading: ``probe_treecut.py`` seeds its
   configuration rng but its generator draws formulas with the global unseeded
   one, so re-running it reproduces neither its own recorded values nor a stable
   number across runs.

2. **The doc's ``proof`` string does not exist verbatim in the generator.**
   Section 9's TreeCut check ① says to assert that ``proof`` contains
   "N variables but M formulas".  The generator emits "There are {N} variables but
   only {M} linear formula(s), so we cannot calculate the price of {ans}", and
   uses the singular for M == 1 -- measured over the whole HF release: the literal
   "but M formulas" occurs 0 times.  Both this adapter and ``verify_treecut.py``
   parse the real string (accepting the singular and the plural) and additionally
   require ``M == N - 1``, which is the tree identity the certificate rests on.

3. **Q11's grid has 12 usable cells, not 18, and the ">= 100 per cell" floor is
   met by the negative side only.**  Q11 asks for the negative/positive split to be
   stratified over ``numVars`` x ``ansDepth`` x ``theme`` with "at least 100 rows
   per cell".  Of the 9 combinations the doc's grid spans, only 6 can hold a pair:
   ``ansDepth == numVars`` at (4,4) and (6,6) leaves no non-necessary edge for the
   counterpart to cut, and ``ansDepth > numVars`` at (4,6) is rejected by the
   generator itself.  The 12 cells therefore get 408-409 negatives each (>= 100,
   satisfied) and 41-42 positives each (below 100 by construction: the positive
   quota is 500, D27's concession).  Both shortfalls are reported rather than
   padded -- padding would change the doc's own 4,907 / 500 (table B rows 5 and 7).

4. **The positive option block is generated by this adapter, not by the
   generator.**  D26 says the placeholder options come from "题面条件中随机抽 k=3 条
   换变量名/变量值"; the pinned generator has no such mode (its only defect mode is
   the necessary-edge cut), so the block is built here from the shipped passage:
   one variable spelling may be swapped for another spelling of the same theme, or
   one numeric value changed, 50/50 per option (Q16).  Each option is required to
   be absent from the passage verbatim and the block is dropped (never padded) if
   three distinct ones cannot be built.  This is a *rendering* step over the
   generator's own passage, not a second data source.

5. **The doc's pre-fix L3 figures are not reproducible as pooled release numbers,
   but the doc's conclusion is.**  Section 4.6 and section 11 risk 1 quote TreeCut
   as "0.755 (length 0.726)".  Measured here on the released HF files
   (``<raw_dir>/hf``, 21,000 rows, 5,500 answerable / 15,500 cut): the *pooled*
   readings are length BA 0.5670 and BoW NB 0.5772, not 0.726/0.755; the recon's
   own pooled figures are 0.5666 / 0.5843 (``hf_stats.json``) and 0.5809 / 0.5533
   (repo-generated cutDepth=1), and the doc's 0.726 sits next to the release's
   worst per-config length BA (0.724 at ``nv10_ad6``; the per-config worst is
   0.922 at ``nv4_ad4`` by the recon's estimator and 0.9190 by
   ``verify_treecut.py``'s, and the max per-config BoW NB is 0.999 at
   ``nv2_ad2``).
   The doc's length gap has the opposite sign on the release, too: it records the
   cut member as the *shorter* one (351.6 vs 303.6 characters), while the released
   files measure 588.28 answerable vs 609.05 cut (mean char delta +20.77, i.e. the
   cut member is longer) -- the doc's pair describes one configuration, not the
   pooled release.
   Whichever pooled number is used, all of them exceed section 9's hard gate
   (random + 5pt = 0.55), so the doc's "must fix before pooling" holds; only the
   exact figures drift.  ``verify_treecut.py`` prints this pre-fix line as
   information and gates on the *built* rows.

6. **``paired_original_text`` is the matched counterpart, not the uncut
   problem.**  For UMWP the key holds the other member of a native pair, and this
   adapter keeps that meaning: for TreeCut the pair is (necessary-edge cut,
   matched non-necessary-edge cut) and each shipped side records the *other*
   member -- both are one-sentence-short versions of the scenario, not the
   generator's full problem.  The full problem is not recoverable from the pair
   alone; it is not needed, because the certificate is about the pair, and
   ``deleted_condition_text`` names the sentence the row itself lost, which its
   counterpart still carries.

7. **The positive's number is certified only along its derivation chain.**
   Section 6 routes the positive through the ordinary numeric branch, so the row
   must ship the generator's answer as gold (the old adapter refused to record it
   at all).  The chain certificate above re-derives that value from the passage's
   own sentences, using the pinned renderer as the decoder, so the gold is anchored
   rather than trusted.  What is *not* checked is the consistency of sentences
   **off** the chain with the same value assignment: doing that would need the
   full linear system and the generator's per-row formula record, which the source
   does not ship.  A wrong off-chain formula would therefore still add an
   unprovable (but unused) sentence to the passage -- it cannot affect the gold,
   which is computed from the chain only.

Module note on L1
-----------------

TreeCut's certificate is the strongest structural one in the pool: ``cut`` is
taken from ``ans_upstream``, so the deleted sentence is provably necessary, and
the deletion is a real syntactic operation on the text (the sentence is gone
verbatim, not paraphrased).  Since D26 the four-tier gold is the cut edge itself
(``correct_option_id`` points at the sentence the generator removed, re-derived
from ``deleted_condition_text``), so the refusal's diagnosis is as provable as the
refusal.  The L3 alignment and the option-leak heuristic are what the design doc
asks to be measured on the built artifact (section 9; section 11 risk 1), and
``verify_treecut.py`` measures both on the shipped rows.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import deque

try:
    import schema
except ImportError:  # pragma: no cover - running as a plain script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import schema

# ---------------------------------------------------------------------------
# source constants
# ---------------------------------------------------------------------------

DATA_SOURCE = schema.SOURCE_TREECUT
DEFAULT_RAW_DIR = "/home/charles/data/reasoning_rl/halluc/raw/treecut"
DEFAULT_OUT = os.path.expanduser("~/data/reasoning_rl/halluc/built/treecut.parquet")


def report_path_for(out: str) -> str:
    """The report that belongs to ``--out``: same stem, ``_report.json``.

    Derived from ``--out`` instead of pinned to the build directory, so that a
    scratch build (``--out /tmp/treecut.parquet``) writes a scratch report rather
    than overwriting the canonical one the build report cites.  Same convention in
    all four adapters.
    """
    return os.path.splitext(out)[0] + "_report.json"


REPO_SUBDIR = os.path.join("repo", "treecut")
ENTITIES_FILE = "entities_items.py"

#: Table B row 7 (four-tier diagnosis, D26) and row 5 (solvable numeric with a
#: placeholder block, D26/D27).  Row 5 shares ``solvable_numeric`` with row 1 --
#: the mix separates them by ``data_source`` (schema.py's branch registry).
BRANCH_DIAG = schema.BRANCH_UNSOLVABLE_DIAG
BRANCH_POSITIVE = schema.BRANCH_SOLVABLE_NUMERIC
TEMPLATE_A = schema.TEMPLATE_A

#: Table B rows 5 and 7 (D26/D27): the four-tier negative quota and the solvable
#: positive quota.  The source is a generator, so these are generation targets,
#: not source-side caps.
NEG_TARGET = 4907
POS_TARGET = 500

#: D15: every option block is exactly k = 3 (1 correct + 2 distractors, A/B/C).
K_OPTIONS = 3

#: Stratification (design doc Q11): theme x numVars x ansDepth.  ``ansDepth ==
#: numVars`` is excluded because it leaves no off-path edge to cut, and
#: ``ansDepth > numVars`` is rejected by the generator itself.
GRID_THEMES = ("food", "outfit")
GRID_CONFIGS = ((4, 2), (6, 2), (6, 4), (8, 2), (8, 4), (8, 6))
ORDER = "random"  # every config the HF release ships uses order=random

#: Retries per generation slot before the slot is dropped (fail closed).
MAX_TRIES = 200

ERROR_TYPE = "key_information_missing"
PERTURBATION_TYPE = "missing_condition"

#: The two option-block families (D26).  The four-tier negative's options are
#: candidate missing conditions (its gold is the generator's cut edge, the
#: distractors are sibling cuts); the solvable positive's options are placeholders
#: built from its own passage.
OPTION_KIND_CUT = "missing_condition"
OPTION_KIND_PLACEHOLDER = "placeholder"

#: Q16: the two placeholder modes, 50/50 per option.
PLACEHOLDER_MODES = ("variable_name", "variable_value")

_QUESTION_MARK = "Question: "
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_PROOF_NUMBERS_RE = re.compile(r"There are (\d+) variables but only (\d+) linear formula")
_NUMBER_SPAN_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


# ---------------------------------------------------------------------------
# source loading
# ---------------------------------------------------------------------------


def load_source(raw_dir: str) -> dict:
    """Read the pinned TreeCut generator and its entity tables.

    The generator is not vendored: ``<raw_dir>/repo/treecut`` is put on
    ``sys.path`` and its own modules are imported, so the rows are produced by
    the code the repository pins (revision recorded in the recon's notes) rather
    than by a re-implementation.  Raises ``FileNotFoundError`` when the directory
    is missing -- a generator source with no generator cannot fail closed softly.
    """
    repo = os.path.join(raw_dir, REPO_SUBDIR)
    if not os.path.isdir(repo):
        raise FileNotFoundError(f"TreeCut generator not found under {repo}")
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import dependency_tree  # noqa: PLC0415 - imported from the downloaded repo
    import entities_items  # noqa: PLC0415
    import gen_questions  # noqa: PLC0415
    import structure_graph  # noqa: PLC0415

    vocabulary = {
        "food": entities_items.food_entity_item,
        "outfit": entities_items.outfit_entity_item,
    }
    return {
        "repo": repo,
        "tree_node": dependency_tree.TreeNode,
        "structure_graph": structure_graph.StructureGraph,
        "gen_question": gen_questions.gen_question,
        "gen_disproof": gen_questions.gen_disproof,
        "vocabulary": vocabulary,
        # The answer certificate's decoder (DEVIATIONS item 7): the generator's own
        # sentence renderer plus its formula record, so a step of the derivation
        # chain can be decoded by asking "which formula would have rendered this
        # exact sentence?" instead of by parsing the prose with a second grammar.
        "formula": gen_questions.Formula,
        "render_sentence": gen_questions.edge_and_formula_to_sentence,
    }


def vocabulary_pairs(entities: list, item_dict: dict) -> dict:
    """``{canonical variable: (entity, item)}`` -- the generator's ``node2var`` values.

    ``build_scenario`` samples its variables from the (entity, item) edges of a
    structure graph, so this is the inverse map the answer certificate needs to
    hand the pinned renderer a synthetic edge.
    """
    return {f"{item} at {entity}": (entity, item) for entity in entities for item in item_dict}


def theme_vocabulary(source: dict, theme: str) -> dict:
    """``(matcher, pairs, item_dict)`` for one theme: spellings, node2var, plurals."""
    table = source["vocabulary"][theme]
    matcher = compile_matcher(vocabulary_forms(table["entities"], table["item_dict"]))
    pairs = vocabulary_pairs(table["entities"], table["item_dict"])
    return matcher, pairs, table["item_dict"]


def vocabulary_forms(entities: list, item_dict: dict) -> dict:
    """``{canonical variable: [surface forms]}`` for the text-level certificate.

    A composite variable is a (entity, item) pair, so the sentence that mentions
    it spells the item in the singular or in its listed plural (``plural_form``)
    followed by " at <entity>".  Both spellings are one variable, which is what
    lets the certificate segment a sentence back into the variables it names.
    """
    forms: dict[str, list[str]] = {}
    for entity in entities:
        for item, plural in item_dict.items():
            key = f"{item} at {entity}"
            forms[key] = [key, f"{plural} at {entity}"]
    return forms


def compile_matcher(forms: dict) -> tuple:
    """``(regex, surface -> canonical)`` for one theme's variable forms."""
    surface = [variant for variants in forms.values() for variant in variants]
    surface.sort(key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(s) for s in surface))
    lookup = {variant: key for key, variants in forms.items() for variant in variants}
    return pattern, lookup


# ---------------------------------------------------------------------------
# text plumbing (shared shape with verify_treecut.py, which re-implements it)
# ---------------------------------------------------------------------------


def split_body(text: str) -> tuple[str, str]:
    """``(body, question sentence)`` -- the generator always ends with "Question: ..."."""
    mark = text.rfind(_QUESTION_MARK)
    if mark < 0:
        return (text or "").strip(), ""
    return text[:mark].strip(), text[mark:].strip()


def sentences_of(text: str) -> list[str]:
    """The body sentences of a problem text (the question sentence removed)."""
    body, _ = split_body(text)
    return [s for s in _SENTENCE_SPLIT_RE.split(body) if s.strip()]


def clause_of(sentence: str) -> str:
    """A body sentence as ``gen_disproof`` quotes it (first letter down, no stop).

    The generator upper-cases a sentence's first letter when it renders it
    (``u0``) and lower-cases it again when it quotes it in a certificate
    (``l0(sentence[:-1])``), so this is the exact inverse for the sentence shapes
    this source emits (a leading digit is left alone by both).
    """
    return sentence[:1].lower() + sentence[1:-1]


def _variables_in(text: str, matcher: tuple) -> list:
    pattern, lookup = matcher
    return [lookup[m.group(0)] for m in pattern.finditer(text)]


def text_graph(text: str, matcher: tuple) -> dict:
    """Turn a problem text back into the tree it came from.

    Returns ``{"facts", "links", "asked", "malformed"}``: ``links`` are
    ``(sentence, variable, variable)`` for every body sentence that names two
    variables, ``facts`` are ``(sentence, variable)`` for the single-variable
    sentences (the root facts), and ``asked`` is the variable named by the
    question sentence.  A sentence naming anything other than one or two known
    variables means the vocabulary does not cover the row, which is a hard
    ``malformed`` flag rather than a silently skipped edge.
    """
    body, question = split_body(text)
    facts: list = []
    links: list = []
    malformed: list = []
    for sentence in sentences_of(body):
        named = _variables_in(sentence, matcher)
        unique = list(dict.fromkeys(named))
        if len(unique) == 1:
            facts.append((sentence, unique[0]))
        elif len(unique) == 2:
            links.append((sentence, unique[0], unique[1]))
        else:
            malformed.append(sentence)
    asked_vars = _variables_in(question, matcher)
    asked = asked_vars[0] if asked_vars else None
    return {"facts": facts, "links": links, "asked": asked, "malformed": malformed}


def _variables(graph: dict) -> set:
    """Every variable the text names, whether in a link or in a root fact."""
    named = {variable for _, variable in graph["facts"]}
    for _, left, right in graph["links"]:
        named.add(left)
        named.add(right)
    return named


def _component(graph: dict) -> tuple:
    """``(variables, links)`` of the asked variable's own component."""
    asked = graph["asked"]
    adjacency: dict = collections.defaultdict(list)
    for _, left, right in graph["links"]:
        adjacency[left].append(right)
        adjacency[right].append(left)
    seen = {asked}
    queue = deque([asked])
    while queue:
        node = queue.popleft()
        for neighbour in adjacency[node]:
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    inside = [link for link in graph["links"] if link[1] in seen and link[2] in seen]
    return seen, inside


def _reachable(graph: dict) -> set:
    """Every variable connected to a root fact by the retained sentences."""
    start = {variable for _, variable in graph["facts"]}
    adjacency: dict = collections.defaultdict(list)
    for _, left, right in graph["links"]:
        adjacency[left].append(right)
        adjacency[right].append(left)
    seen = set(start)
    queue = deque(start)
    while queue:
        node = queue.popleft()
        for neighbour in adjacency[node]:
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    return seen


def _hops_to(graph: dict, target: str) -> int | None:
    """Shortest fact-to-``target`` hop count, or ``None`` when unreachable."""
    adjacency: dict = collections.defaultdict(list)
    for _, left, right in graph["links"]:
        adjacency[left].append(right)
        adjacency[right].append(left)
    queue = deque((variable, 0) for _, variable in graph["facts"])
    seen = {variable for _, variable in graph["facts"]}
    while queue:
        node, distance = queue.popleft()
        if node == target:
            return distance
        for neighbour in adjacency[node]:
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append((neighbour, distance + 1))
    return None


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


def cells() -> list:
    """The ``(theme, num_vars, ans_depth)`` grid, in generation order."""
    return [(theme, nv, ad) for theme in GRID_THEMES for nv, ad in GRID_CONFIGS]


def quota_for(n_cells: int, total: int) -> list:
    """Split ``total`` rows over ``n_cells`` cells, remainder to the first cells."""
    base, extra = divmod(total, n_cells)
    return [base + (1 if index < extra else 0) for index in range(n_cells)]


def pair_seed(seed: int, theme: str, num_vars: int, ans_depth: int, ordinal: int) -> int:
    """Deterministic per-slot scenario seed (``hash()`` is salted per process)."""
    key = f"{seed}|{theme}|{num_vars}|{ans_depth}|{ordinal}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def build_scenario(source: dict, theme: str, num_vars: int, ans_depth: int, order: str,
                   rng) -> dict:
    """Mirror of ``gen_data.generate_qa``'s steps 1-4 with the rng injected.

    Byte-for-byte faithfulness to the pinned generator is asserted by
    ``test_treecut_adapter.py``: seeding the global ``random`` module, running
    this and then :func:`render_member` reproduces ``generate_qa``'s ``problem``
    exactly, for every theme / config / order combination tested.
    """
    entities = source["vocabulary"][theme]["entities"]
    item_dict = source["vocabulary"][theme]["item_dict"]
    num_entities = 2
    num_items = math.ceil(num_vars / num_entities)
    sampled_entities = rng.sample(entities, num_entities)
    sampled_items = rng.sample(list(item_dict.keys()), num_items)
    structure = source["structure_graph"]()
    for entity in sampled_entities:
        for item in sampled_items:
            structure.add_edge(entity, item)
    all_item_edges = structure.get_all_edges()
    sampled_variable_names = rng.sample(all_item_edges, num_vars)
    value_dict = {}
    for name in sampled_variable_names:
        value_dict[name] = (rng.randint(10, 20) * 5 if theme == "outfit"
                            else rng.randint(5, 15))
    node_names = [str(index) for index in range(1, num_vars + 1)]
    node2var = {str(index + 1): sampled_variable_names[index] for index in range(num_vars)}
    node2var["ROOT"] = "ROOT"
    tree_node = source["tree_node"]
    root = tree_node("ROOT")
    nodes = [root]
    node = root
    for index in range(1, ans_depth + 1):
        child = tree_node(node_names[index - 1])
        node.add_child(child)
        nodes.append(child)
        node = child
    for index in range(ans_depth + 1, len(node_names) + 1):
        child = tree_node(node_names[index - 1])
        rng.choice(nodes).add_child(child)
        nodes.append(child)
    ans_upstream = nodes[ans_depth].get_ancestors()
    ans_upstream.pop()  # drop 'ROOT'
    return {
        "theme": theme,
        "item_dict": item_dict,
        "num_vars": num_vars,
        "ans_depth": ans_depth,
        "node2var": node2var,
        "value_dict": value_dict,
        "nodes": nodes,
        "edges": root.get_all_edges(),
        "ans_node": node_names[ans_depth - 1],
        "ans_upstream": ans_upstream,
    }


def necessary_heads(scenario: dict) -> set:
    """Nodes whose incoming edge the answer's derivation needs."""
    return set(scenario["ans_upstream"]) | {scenario["ans_node"]}


def render_scenario(source: dict, scenario: dict, order: str, shuffle_rng) -> dict:
    """Render the whole scenario once -- the shared render both members come from.

    Mirrors ``gen_data.generate_qa``'s rendering step (``gen_question`` plus the
    ``order`` transform), with the formulas drawn **once**: both members of a pair
    then carry the same numbers and the same question sentence, and differ by
    exactly the one sentence their cut edge owns.  Drawing the formulas per member
    instead is what re-opens the length leak the pair is built to close.

    Consequence for byte-faithfulness: a member is *not* byte-identical to
    ``generate_qa(..., hallu=True, cutDepth=1)`` for the same scenario seed,
    because the stock generator never draws the removed edge's formula (it filters
    the edges before calling ``gen_question``).  ``test_treecut_adapter.py``
    asserts byte-equality on the uncut render instead, which is where the two code
    paths are the same computation.
    """
    sentences, question, answer, sentence_dict = source["gen_question"](
        scenario["edges"],
        scenario["ans_node"],
        scenario["node2var"],
        scenario["value_dict"],
        scenario["item_dict"],
    )
    ordered = list(sentences)
    if order == "backward":
        ordered.reverse()
    elif order == "random":
        shuffle_rng.shuffle(ordered)
    return {
        "edges": list(scenario["edges"]),
        "sentences": list(sentences),
        "ordered": ordered,
        "question": question,
        "answer": answer,
        "sentence_dict": sentence_dict,
        "problem": " ".join(ordered + [question]),
    }


def member_of(render: dict, cut_edge: tuple, order: str, shuffle_rng) -> dict:
    """One member of the pair: the shared render minus ``cut_edge``'s sentence."""
    keep = [s for edge, s in zip(render["edges"], render["sentences"], strict=False) if edge != cut_edge]
    if order == "backward":
        keep.reverse()
    elif order == "random":
        shuffle_rng.shuffle(keep)
    return {
        "problem": " ".join(keep + [render["question"]]),
        "sentences": keep,
        "deleted_sentence": render["sentence_dict"][cut_edge],
    }


def build_pair(source: dict, seed: int, theme: str, num_vars: int, ans_depth: int,
               order: str, ordinal: int, max_tries: int = MAX_TRIES) -> tuple:
    """Build one matched pair; ``(negative, positive, tries)``, both ``None`` on failure.

    The unsolvable member (the shipped four-tier negative) cuts
    ``ans_upstream[0]``, the answer's own parent edge, so its disproof certificate
    is the generator's own.  The solvable counterpart (the positive) cuts one
    non-necessary edge -- one whose child is not an ancestor of the answer and is
    itself a non-leaf, so no variable vanishes -- matched on whether that child
    hangs off ROOT (the two cut sentences then have the same template, which is
    what equalises the lengths).  The positive also carries the generator's own
    ``answer`` for the shared render, which is the row 5 gold that
    :func:`certify_answer` then re-derives from the positive's text.
    """
    base = pair_seed(seed, theme, num_vars, ans_depth, ordinal)
    random.seed(base)
    scenario = None
    positive_cut = None
    for tries in range(1, max_tries + 1):
        scenario = build_scenario(source, theme, num_vars, ans_depth, order, random)
        heads = necessary_heads(scenario)
        non_necessary = [edge for edge in scenario["edges"] if edge[1] not in heads]
        if not non_necessary:
            return None, None, tries  # structurally impossible configuration
        parents = {edge[0] for edge in scenario["edges"]}
        root_parented = ans_depth == 2
        candidates = [
            edge for edge in non_necessary
            if ((edge[0] == "ROOT") == root_parented) and (edge[1] in parents)
        ]
        if not candidates:
            continue
        positive_cut = random.choice(candidates)
        break
    else:
        return None, None, max_tries

    negative_cut = scenario["ans_upstream"][0]
    negative_edge = [e for e in scenario["edges"] if e[1] == negative_cut][0]
    render = render_scenario(source, scenario, order, random.Random(f"{base}:render"))
    negative = member_of(render, negative_edge, order, random.Random(f"{base}:negative"))
    positive = member_of(render, positive_cut, order, random.Random(f"{base}:positive"))
    positive["answer"] = render["answer"]

    cut_node = next(node for node in scenario["nodes"] if node.name == negative_cut)
    negative["proof"] = source["gen_disproof"](
        cut_node.get_all_edges(),
        scenario["node2var"],
        render["sentence_dict"],
        scenario["ans_node"],
    )
    return negative, positive, tries


# ---------------------------------------------------------------------------
# certificates
# ---------------------------------------------------------------------------


def certify_pair(negative: dict, positive: dict, config: dict, matcher: tuple) -> list:
    """Re-derive the pair's certificate from its own two texts; ``[]`` == certified.

    Every clause below is checked against the artifact's own strings, so the
    verifier can repeat the derivation without touching this module's internals.
    """
    problems: list = []
    num_vars = config["num_vars"]
    ans_depth = config["ans_depth"]

    negative_sentences = sentences_of(negative["problem"])
    positive_sentences = sentences_of(positive["problem"])
    if len(negative_sentences) != num_vars - 1 or len(positive_sentences) != num_vars - 1:
        problems.append(
            f"sentence counts {len(negative_sentences)}/{len(positive_sentences)} "
            f"!= num_vars - 1 ({num_vars - 1})"
        )
    negative_counts = collections.Counter(negative_sentences)
    positive_counts = collections.Counter(positive_sentences)
    lost_by_negative = list((positive_counts - negative_counts).elements())
    lost_by_positive = list((negative_counts - positive_counts).elements())
    if len(lost_by_negative) != 1 or len(lost_by_positive) != 1:
        problems.append(
            f"pair differs by {len(lost_by_negative)}/{len(lost_by_positive)} sentences, not 1/1"
        )
    else:
        if lost_by_negative[0] != negative["deleted_sentence"]:
            problems.append("deleted_condition_text is not the sentence the question lost")
        if lost_by_positive[0] != positive["deleted_sentence"]:
            problems.append("the counterpart's lost sentence is not the recorded one")
    if negative["deleted_sentence"] in negative_counts:
        problems.append("the deleted sentence is still present in the question")

    negative_graph = text_graph(negative["problem"], matcher)
    positive_graph = text_graph(positive["problem"], matcher)
    for label, graph in (("negative", negative_graph), ("positive", positive_graph)):
        if graph["malformed"]:
            problems.append(f"{label}: {len(graph['malformed'])} sentence(s) name "
                            f"neither one nor two known variables: {graph['malformed'][0][:60]!r}")
    if not problems:
        negative_vars = _variables(negative_graph)
        positive_vars = _variables(positive_graph)
        if negative_graph["asked"] is None or positive_graph["asked"] is None:
            problems.append("the question sentence names no known variable")
        elif negative_graph["asked"] != positive_graph["asked"]:
            problems.append("the two members ask about different variables")
        elif negative_vars != positive_vars or len(negative_vars) != num_vars:
            problems.append(
                f"variable sets differ ({len(negative_vars)} vs {len(positive_vars)}), "
                f"expected {num_vars} each"
            )
        elif negative_graph["asked"] in _reachable(negative_graph):
            problems.append("the asked variable is reachable from a root fact: solvable")
        else:
            hops = _hops_to(positive_graph, positive_graph["asked"])
            if hops != ans_depth - 1:
                problems.append(f"the counterpart's derivation is {hops} hops, "
                                f"expected ans_depth - 1 ({ans_depth - 1})")

    proof = negative.get("proof", "")
    match = _PROOF_NUMBERS_RE.search(proof)
    if not match:
        problems.append("proof carries no 'N variables but only M linear formula' certificate")
    else:
        proof_vars, proof_formulas = int(match.group(1)), int(match.group(2))
        if not proof_formulas < proof_vars:
            problems.append(f"proof certificate is not deficient: M={proof_formulas} >= N={proof_vars}")
        if proof_formulas != proof_vars - 1:
            problems.append(f"proof certificate breaks the tree identity: M={proof_formulas} "
                            f"!= N - 1 ({proof_vars - 1})")
        if proof_vars > num_vars:
            problems.append(f"proof certificate claims {proof_vars} variables > num_vars {num_vars}")
        if negative_graph["facts"] or negative_graph["links"]:
            component_vars, component_links = _component(negative_graph)
            if len(component_vars) != proof_vars:
                problems.append(f"proof says {proof_vars} variables, the asked variable's "
                                f"component has {len(component_vars)}")
            if len(component_links) != proof_formulas:
                problems.append(f"proof says {proof_formulas} formulas, the component has "
                                f"{len(component_links)} sentences")
            for sentence, _, _ in component_links:
                if clause_of(sentence) not in proof:
                    problems.append(f"proof cites a clause that is not a sentence of this "
                                    f"question: {sentence[:60]!r}")
                    break
        asked = negative_graph["asked"]
        if asked is not None and not proof.rstrip().endswith(f"the price of {asked}."):
            problems.append("the proof certificate does not name the asked variable")
    return problems


# ---------------------------------------------------------------------------
# the answer certificate: the positive's gold, re-derived from its own passage
# ---------------------------------------------------------------------------


def _chain_to_ask(graph: dict) -> list | None:
    """The shortest fact-to-answer chain as ``[(variable, sentence), ...]``.

    The first element is the root fact the chain starts from (its sentence is the
    one that gives the price), every later element is the variable the chain
    reaches and the sentence that links it to the previous one.  ``None`` when the
    asked variable is unreachable -- for a positive member that is a certificate
    failure, not a fallback.
    """
    asked = graph["asked"]
    if asked is None:
        return None
    neighbours: dict = collections.defaultdict(list)
    for sentence, left, right in graph["links"]:
        neighbours[left].append((right, sentence))
        neighbours[right].append((left, sentence))
    parent: dict = {}
    seen: set = set()
    queue: deque = deque()
    for sentence, variable in graph["facts"]:
        parent[variable] = (None, sentence)
        seen.add(variable)
        queue.append(variable)
    while queue:
        node = queue.popleft()
        if node == asked:
            chain: list = []
            current = node
            while current is not None:
                previous, sentence = parent[current]
                chain.append((current, sentence))
                current = previous
            chain.reverse()
            return chain
        for other, sentence in neighbours[node]:
            if other not in seen:
                seen.add(other)
                parent[other] = (node, sentence)
                queue.append(other)
    return None


def _candidate_results(sentence: str) -> list:
    """The values a sentence could encode: its own numbers, their negatives, +/-1, 0.

    The renderer writes ``res`` as ``{res} dollars`` in the "and ... cost" family,
    as the *absolute* gap in the "more/less than" family (with the sign folded into
    the comparison), as the word "a dollar" when ``|res| == 1`` -- which is why
    ``+/-1`` is always a candidate even though no digit appears -- and as no number
    at all when ``res == 0`` ("is the same as that of").
    """
    values: list = [1, -1]
    for token in schema.numbers_in(sentence):
        if token.isdigit():
            values.extend([int(token), -int(token)])
    values.append(0)
    return list(dict.fromkeys(values))


def _decode_fact(sentence: str, pair: tuple, config: dict) -> list:
    """Which root-fact values could have rendered this sentence?"""
    node2var = {"ROOT": "ROOT", "C": pair}
    return [
        res for res in _candidate_results(sentence)
        if config["render_sentence"](("ROOT", "C"), config["formula"]((1,), res),
                                     node2var, config["item_dict"]) == sentence
    ]


def _decode_link(sentence: str, known_pair: tuple, unknown_pair: tuple,
                 known_value: int, config: dict) -> list:
    """Which ``(x, y, value of the unknown variable)`` triples could have rendered it?

    The edge is handed to the renderer as ``(known, unknown)`` -- the edge order of
    the tree, not the word order of the sentence: ``gen_formulas`` draws
    ``para`` from ``(x, y)``, ``(x, -y)`` or ``(-x, y)``, and the last form renders
    the *child* first.  Enumerating ``x`` and ``y`` over the generator's own
    parameter space (``+/-1..3``) plus the sentence's own numbers as candidate
    results is what lets this file decode a step without a second prose grammar.

    The wording narrows the space before rendering (``_sign_space``); if that
    narrowing ever found nothing, the full space is tried, so a mis-read wording
    costs time and never a decode.
    """
    signs, results = _sign_space(sentence)
    decoded = _decode_space(sentence, known_pair, unknown_pair, known_value, config, signs, results)
    if not decoded and signs is not _ALL_SIGNS:
        decoded = _decode_space(sentence, known_pair, unknown_pair, known_value, config,
                                _ALL_SIGNS, None)
    return decoded


_ALL_SIGNS = tuple((x, y) for x in (1, -1, 2, -2, 3, -3) for y in (1, -1, 2, -2, 3, -3))
_PRODUCT_SIGNS = tuple((x, y) for x in (1, 2, 3) for y in (1, 2, 3))
_MIXED_SIGNS = (tuple((x, y) for x in (1, 2, 3) for y in (-1, -2, -3))
                + tuple((x, y) for x in (-1, -2, -3) for y in (1, 2, 3)))


def _sign_space(sentence: str) -> tuple:
    """The ``(x, y)`` sign families, and the result candidates, a wording implies.

    The rendering is injective in the sign family: only ``x, y > 0`` writes the
    "A and B cost N" form, only opposite signs write "more/less than", and only
    ``res == 0`` writes "is the same as that of".  So this is a filter on the
    *wording*, with :func:`_decode_link` falling back to the full space if a
    sentence ever fails to decode under it.
    """
    if " is the same as that of " in sentence:
        return _MIXED_SIGNS, [0]
    if " and " in sentence:
        return _PRODUCT_SIGNS, None
    if " more than " in sentence or " less than " in sentence:
        return _MIXED_SIGNS, None
    return _ALL_SIGNS, None


def _decode_space(sentence: str, known_pair: tuple, unknown_pair: tuple, known_value: int,
                  config: dict, signs: tuple, results: list | None) -> list:
    """One enumeration pass over ``signs`` (see :func:`_decode_link`)."""
    node2var = {"ROOT": "ROOT", "P": known_pair, "C": unknown_pair}
    candidates = _candidate_results(sentence) if results is None else results
    decoded: list = []
    for x, y in signs:
        for res in candidates:
            rendered = config["render_sentence"](("P", "C"), config["formula"]((x, y), res),
                                                 node2var, config["item_dict"])
            if rendered != sentence:
                continue
            numerator = res - x * known_value
            if numerator % y:
                continue
            value = numerator // y
            if value > 0:
                decoded.append((x, y, res, value))
    return decoded


def certify_answer(positive: dict, config: dict) -> list:
    """Re-derive the positive's gold from its own text; ``[]`` == certified.

    The row-5 gold is a *number*, and a number is only shippable if the artifact
    can prove it (DEVIATIONS item 7): this walks the certified fact-to-answer chain
    of the positive's own passage and decodes each step with the pinned
    generator's own renderer (``edge_and_formula_to_sentence`` + ``Formula``), so
    the value is re-derived from the sentence set the row ships rather than trusted
    from the generation-time record.
    """
    matcher = config["matcher"]
    pairs = config["pairs"]
    graph = text_graph(positive["problem"], matcher)
    if graph["malformed"]:
        return [f"the passage has {len(graph['malformed'])} unparsable sentence(s): "
                f"{graph['malformed'][0][:60]!r}"]
    asked = graph["asked"]
    if asked is None:
        return ["the question sentence names no known variable"]
    if asked not in pairs:
        return [f"the asked variable {asked!r} has no (entity, item) pair in the theme vocabulary"]
    chain = _chain_to_ask(graph)
    if chain is None:
        return ["the asked variable is unreachable from a root fact: nothing to certify"]

    start_variable, fact_sentence = chain[0]
    if start_variable not in pairs:
        return [f"the fact variable {start_variable!r} has no pair in the theme vocabulary"]
    facts = _decode_fact(fact_sentence, pairs[start_variable], config)
    if len(facts) != 1:
        return [f"the root fact decodes to {len(facts)} values, not 1: {fact_sentence[:60]!r}"]

    value = facts[0]
    previous_variable = start_variable
    for variable, sentence in chain[1:]:
        if variable not in pairs:
            return [f"the chain variable {variable!r} has no pair in the theme vocabulary"]
        steps = _decode_link(sentence, pairs[previous_variable], pairs[variable], value, config)
        if len(steps) != 1:
            return [f"the chain step decodes to {len(steps)} formulas, not 1: {sentence[:60]!r}"]
        value = steps[0][3]
        previous_variable = variable

    gold = str(positive.get("answer", "")).strip()
    if not gold:
        return ["the pair records no generator answer for the solvable member"]
    if not gold.isdigit():
        return [f"the generator answer {gold!r} is not an integer"]
    if int(gold) != value:
        return [f"the chain derives {value}, the recorded gold is {gold}"]
    return []


# ---------------------------------------------------------------------------
# options (D26): candidate missing conditions (negative) + placeholders (positive)
# ---------------------------------------------------------------------------


def options_absent_from(options: list, text: str) -> bool:
    """Whether every option text is verbatim absent from ``text`` (D26 / section 9)."""
    return bool(options) and all(opt["text"] and opt["text"] not in text for opt in options)


def options_pairwise_distinct(options: list) -> bool:
    texts = [opt["text"] for opt in options]
    return len(set(texts)) == len(texts) and all(texts)


def negative_options(passage: str, gold_sentence: str, family: list, rng) -> tuple | None:
    """The four-tier option block: k=3 candidate missing conditions (D26).

    ``family`` is the grid cell's candidate cut sentences -- every planned pair in
    the cell contributes the two sentences its members cut.  The gold is this row's
    own cut edge; the distractors are sibling cuts.  Every option must be absent
    from the passage verbatim: that is the leak defence (section 4.6 / 9 / 11 risk
    1).  If the gold were the only option *missing* from the passage, "pick the
    sentence you cannot find" would be a 100% shortcut -- the MiP measurement is
    95.6% (section 4.2) -- so this returns ``None`` (drop, never pad) unless the
    finished block has k options, all out of passage and pairwise distinct.
    """
    candidates = [s for s in family if s.strip() and s != gold_sentence and s not in passage]
    built = schema.build_options(candidates, gold_sentence, K_OPTIONS, rng)
    if built is None:
        return None
    options, correct = built
    if not (options_absent_from(options, passage) and options_pairwise_distinct(options)):
        return None
    return options, correct


def placeholder_variant(sentence: str, matcher: tuple, rng, mode: str) -> str | None:
    """One placeholder option: a passage condition with one name *or* value swapped.

    ``variable_name`` replaces one variable spelling with another spelling of the
    same theme (chosen outside the sentence, so the result differs); the replacement
    is only required to be out of the passage -- Q16 explicitly does not require it
    to be semantically plausible.  ``variable_value`` changes one numeric literal.
    """
    if mode == PLACEHOLDER_MODES[0]:
        matches = list(matcher[0].finditer(sentence))
        outside = sorted(spelling for spelling in matcher[1] if spelling not in sentence)
        if not matches or not outside:
            return None
        match = rng.choice(matches)
        return sentence[: match.start()] + rng.choice(outside) + sentence[match.end():]
    if mode == PLACEHOLDER_MODES[1]:
        spans = list(_NUMBER_SPAN_RE.finditer(sentence))
        if not spans:
            return None
        span = rng.choice(spans)
        value = int(span.group(0).replace(",", ""))
        delta = rng.choice((-7, -5, -3, 3, 5, 7))
        if value + delta <= 0:
            return None
        return sentence[: span.start()] + str(value + delta) + sentence[span.end():]
    raise ValueError(f"unknown placeholder mode {mode!r}")


def placeholder_options(passage: str, matcher: tuple, rng, k: int = K_OPTIONS) -> tuple | None:
    """The solvable side's placeholder block: k=3, no correct item (D26 / Q16).

    Every option is built from one of the passage's own body sentences, and every
    one must be absent from the passage verbatim -- that is what keeps the block
    from being mistaken for a real (or a missing) condition, and it is asserted
    again by ``verify_treecut.py`` on the artifact.  Returns ``(options, modes)``
    where ``modes`` is the multiset of placeholder modes used (sorted, *not*
    aligned with the shuffled options) for the Q16 50/50 monitoring.
    """
    sentences = sentences_of(passage)
    order = list(sentences)
    rng.shuffle(order)
    chosen: list = []
    modes: list = []
    for sentence in order:
        if len(chosen) == k:
            break
        mode = PLACEHOLDER_MODES[0] if rng.random() < 0.5 else PLACEHOLDER_MODES[1]
        variant = placeholder_variant(sentence, matcher, rng, mode)
        if not variant or variant == sentence or variant in passage or variant in chosen:
            continue
        chosen.append(variant)
        modes.append(mode)
    if len(chosen) < k:
        return None
    options = schema.shuffle_options(chosen, rng)
    if not (options_absent_from(options, passage) and options_pairwise_distinct(options)):
        return None
    return options, sorted(modes)


# ---------------------------------------------------------------------------
# row construction
# ---------------------------------------------------------------------------


def task_id_for(config: dict, side: str = "") -> str:
    """The row's ``task_id``; the positive side carries a ``-pos`` suffix.

    The suffix is what makes the pair recoverable from the artifact:
    ``verify_treecut.py`` groups a positive row with its negative by stripping it,
    so the L3 alignment and the pair certificate can be checked without trusting
    the row order.
    """
    base = (f"treecut-{config['theme']}-nv{config['num_vars']}"
            f"-ad{config['ans_depth']}-{config['ordinal']:06d}")
    return f"{base}{side}"


def _shared_extra_info(config: dict, seed: int, task_id: str) -> dict:
    return {
        "split": "train",  # a generator has one implicit split
        "seed": seed,
        "task_id": task_id,
        # TreeCut extensions (the HF release carries no per-row column for any of
        # these; normalise_extra_info backfills them with "" on other sources).
        "num_vars": config["num_vars"],
        "ans_depth": config["ans_depth"],
        "theme": config["theme"],
        "order": config["order"],
    }


def row_for_negative(negative: dict, positive: dict, config: dict, options: list,
                     correct: str, seed: int) -> dict:
    """One four-tier negative row (table B row 7, D26): refusal + diagnosis."""
    extra_info = _shared_extra_info(config, seed, task_id_for(config))
    extra_info.update(
        {
            "error_type": ERROR_TYPE,
            "perturbation_type": PERTURBATION_TYPE,
            "paired_original_text": positive["problem"],
            "deleted_condition_text": negative["deleted_sentence"],
            "solvable": False,
            "correct_option_id": correct,
            "has_diagnosis_label": True,
            "option_kind": OPTION_KIND_CUT,
            "proof": negative["proof"],
            "asked_variable": text_graph(negative["problem"], config["matcher"])["asked"] or "",
        }
    )
    return schema.make_row(
        data_source=DATA_SOURCE,
        question=negative["problem"],
        ground_truth=schema.build_ground_truth(
            solvable=False,
            answer=None,
            correct_option_id=correct,
            has_diagnosis_label=True,
            perturbation_type=PERTURBATION_TYPE,
        ),
        template=TEMPLATE_A,
        branch=BRANCH_DIAG,
        extra_info=extra_info,
        options=options,
    )


def row_for_positive(negative: dict, positive: dict, config: dict, options: list,
                     modes: list, seed: int) -> dict:
    """One solvable positive row (table B row 5, D26/D27): answer + placeholder block.

    The gold is the generator's own answer for the shared render (certified by
    :func:`certify_answer`), the block is a placeholder (no correct item,
    ``correct_option_id=null``) and section 6 scores the row on the ordinary
    numeric branch: ``\\boxed{<答案>}`` matching gold = +1, a misrefusal = 0.
    """
    extra_info = _shared_extra_info(config, seed, task_id_for(config, "-pos"))
    extra_info.update(
        {
            "error_type": "",
            "perturbation_type": "",
            "paired_original_text": negative["problem"],
            "deleted_condition_text": positive["deleted_sentence"],
            "solvable": True,
            "correct_option_id": "",
            "has_diagnosis_label": False,
            "option_kind": OPTION_KIND_PLACEHOLDER,
            "placeholder_modes": list(modes),
            "answer": positive["answer"],
            "asked_variable": text_graph(positive["problem"], config["matcher"])["asked"] or "",
            # The counterpart's own disproof certificate, kept for audit only: the
            # positive's row is solvable, so it carries no proof of its own.
            "paired_proof": negative["proof"],
        }
    )
    return schema.make_row(
        data_source=DATA_SOURCE,
        question=positive["problem"],
        ground_truth=schema.build_ground_truth(
            solvable=True,
            answer=positive["answer"],
            correct_option_id=None,
            has_diagnosis_label=False,
            perturbation_type=None,
        ),
        template=TEMPLATE_A,
        branch=BRANCH_POSITIVE,
        extra_info=extra_info,
        options=options,
    )


def build_rows(raw_dir: str, limit: int | None = None, seed: int = 0) -> tuple:
    """Build the TreeCut parquet rows and the funnel that produced them.

    Returns ``(rows, funnel)``: ``rows`` are ready for
    :func:`schema.normalise_extra_info` / :func:`schema.validate_rows` /
    :func:`schema.write_rows_parquet`, and ``funnel`` is an ordered mapping of
    filter stage -> negative rows remaining (plus the emitted total).  The same
    ``(raw_dir, limit, seed)`` always produces byte-identical rows: each slot seeds
    the global ``random`` module from ``sha256(seed, theme, num_vars, ans_depth,
    ordinal)`` before the generator runs, and every shuffle (both members'
    sentences, the option block, the placeholder pick) comes from a
    ``random.Random`` derived from the same key.

    The positive side ships for the first ``POS_TARGET`` pairs of the
    cell-interleaved order whose answer certificate and placeholder block can both
    be built (see the module docstring): a pair that fails either gate loses only
    its positive, never its negative, and its slot is taken by the next pair so the
    quota is met.
    """
    source = load_source(raw_dir)
    grid = cells()
    quotas = quota_for(len(grid), NEG_TARGET)
    vocabularies = {theme: theme_vocabulary(source, theme) for theme in GRID_THEMES}

    funnel: dict = collections.OrderedDict()
    funnel["raw_rows"] = sum(quotas)
    funnel["after_scenario_reject_drop"] = 0
    funnel["after_certificate_drop"] = 0

    planned: list = []
    scenario_drops: dict = collections.Counter()
    certificate_drops: dict = collections.Counter()
    attempts = 0
    retried = 0
    for cell_index, (theme, num_vars, ans_depth) in enumerate(grid):
        matcher, pairs, item_dict = vocabularies[theme]
        for ordinal in range(quotas[cell_index]):
            negative, positive, tries = build_pair(
                source, seed, theme, num_vars, ans_depth, ORDER, ordinal
            )
            attempts += tries
            if tries > 1:
                retried += 1
            if negative is None:
                scenario_drops[f"{theme}-nv{num_vars}-ad{ans_depth}"] += 1
                continue
            funnel["after_scenario_reject_drop"] += 1
            config = {
                "theme": theme,
                "num_vars": num_vars,
                "ans_depth": ans_depth,
                "order": ORDER,
                "ordinal": ordinal,
                "matcher": matcher,
                "pairs": pairs,
                "item_dict": item_dict,
                "render_sentence": source["render_sentence"],
                "formula": source["formula"],
            }
            problems = certify_pair(negative, positive, config, matcher)
            if problems:
                certificate_drops[f"{theme}-nv{num_vars}-ad{ans_depth}: {problems[0]}"] += 1
                continue
            funnel["after_certificate_drop"] += 1
            planned.append((cell_index, ordinal, negative, positive, config))

    # The distractor family of each cell (D26): every planned pair contributes the
    # two sentences its members cut, and a row's block is drawn from its own cell.
    families: dict = collections.defaultdict(list)
    for cell_index, _, negative, positive, _ in planned:
        families[cell_index].extend([negative["deleted_sentence"], positive["deleted_sentence"]])

    interleaved = _interleave_by_cell(planned)
    option_drops: dict = collections.Counter()
    shipped_negatives: list = []
    for cell_index, ordinal, negative, positive, config in interleaved:
        base = pair_seed(seed, config["theme"], config["num_vars"], config["ans_depth"], ordinal)
        built = negative_options(
            negative["problem"], negative["deleted_sentence"],
            families[cell_index], random.Random(f"{base}:options"),
        )
        if built is None:
            option_drops[f"{config['theme']}-nv{config['num_vars']}-ad{config['ans_depth']}"] += 1
            continue
        options, correct = built
        shipped_negatives.append((negative, positive, config,
                                  row_for_negative(negative, positive, config, options, correct, seed)))
    funnel["after_option_drop"] = len(shipped_negatives)

    positive_drops = 0
    answer_drops: dict = collections.Counter()
    positive_rows: list = []
    for negative, positive, config, _ in shipped_negatives:
        if len(positive_rows) == POS_TARGET:
            break
        # The answer certificate is a positive-side gate: only the row that ships a
        # number needs its number proved, so it runs here (on the pairs that can
        # actually reach the quota) rather than on all 4,907 planned pairs.
        problems = certify_answer(positive, config)
        if problems:
            answer_drops[f"{config['theme']}-nv{config['num_vars']}-ad{config['ans_depth']}: "
                         f"{problems[0]}"] += 1
            continue
        base = pair_seed(seed, config["theme"], config["num_vars"], config["ans_depth"],
                         config["ordinal"])
        built = placeholder_options(positive["problem"], config["matcher"],
                                    random.Random(f"{base}:placeholder"))
        if built is None:
            positive_drops += 1
            continue
        options, modes = built
        positive_rows.append(row_for_positive(negative, positive, config, options, modes, seed))

    rows = _merge_two_sided([row for _, _, _, row in shipped_negatives], positive_rows)
    for index, row in enumerate(rows):
        row["extra_info"]["index"] = index
    if limit is not None:
        rows = rows[: max(limit, 0)]
    funnel["emitted_rows"] = len(rows)
    funnel["plan"] = {
        "grid": [
            {
                "theme": theme,
                "num_vars": num_vars,
                "ans_depth": ans_depth,
                "quota": quotas[cell_index],
            }
            for cell_index, (theme, num_vars, ans_depth) in enumerate(grid)
        ],
        "targets": {"negatives": NEG_TARGET, "positives": POS_TARGET},
        "attempts": attempts,
        "retried_slots": retried,
        "scenario_drops": dict(scenario_drops),
        "certificate_drops": dict(certificate_drops),
        # Positive-side counters: a pair dropped here keeps its negative, because a
        # positive that cannot prove its number (or field a placeholder block) is
        # simply not shipped while its four-tier negative is unaffected.
        "answer_certificate_drops": dict(answer_drops),
        "negative_option_drops": dict(option_drops),
        "positive_placeholder_drops": positive_drops,
        "positives_selected": len(positive_rows),
        "positives_by_cell": {
            f"{theme}-nv{num_vars}-ad{ans_depth}": count
            for (theme, num_vars, ans_depth), count in sorted(
                collections.Counter(_cell_of(row) for row in positive_rows).items(),
                key=lambda item: str(item[0]),
            )
        },
        "positive_rule": (
            "the positive members of the first POS_TARGET pairs of the "
            "cell-interleaved negative stream whose gold the answer certificate "
            "re-derives and whose placeholder block can be built; the emitted stream "
            "merges ceil(k * n_pos / n_neg) positives after the k-th negative "
            "(deterministic, both sides from --limit 2)"
        ),
    }
    return rows, funnel


def _interleave_by_cell(planned: list) -> list:
    """Round-robin the cells so a ``limit`` keeps every config represented.

    Without this a small ``--limit`` would return only the first cell's rows, and
    the L3 estimators would then be reading a *configuration* difference as a
    label difference -- the recon measured exactly that confound (two halves of
    one class came out at 0.956 balanced accuracy when the halves were different
    configs).
    """
    groups: dict = collections.OrderedDict()
    for item in planned:
        theme, num_vars, ans_depth = item[4]["theme"], item[4]["num_vars"], item[4]["ans_depth"]
        groups.setdefault((theme, num_vars, ans_depth), []).append(item)
    out: list = []
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


def _merge_two_sided(negatives: list, positives: list) -> list:
    """Merge the two shipped sides deterministically, both represented early.

    After the k-th negative the stream carries ``ceil(k * n_pos / n_neg)``
    positives (integer ceiling, so the ratio is exact and the first positive lands
    at position 2).  This is what makes ``--limit`` safe: a small artifact always
    carries both labels, which the L3 estimators and the option checks need, and
    the merged order is a pure function of the two lists.
    """
    out: list = []
    n_neg, n_pos = len(negatives), len(positives)
    taken = 0
    for k in range(1, n_neg + 1):
        out.append(negatives[k - 1])
        target = -((-k * n_pos) // n_neg) if n_pos else 0
        while taken < target and taken < n_pos:
            out.append(positives[taken])
            taken += 1
    while taken < n_pos:
        out.append(positives[taken])
        taken += 1
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _breakdown(rows: list[dict], key) -> dict:
    counter: dict = collections.Counter(key(row) for row in rows)
    return dict(sorted(counter.items(), key=lambda item: str(item[0])))


def branch_breakdown(rows: list[dict]) -> dict:
    """Every contract branch, including the five this source cannot fill."""
    counts = collections.Counter(row["extra_info"]["branch"] for row in rows)
    return {branch: counts.get(branch, 0) for branch in schema.BRANCHES}


def _cell_of(row: dict) -> tuple:
    """The ``(theme, num_vars, ans_depth)`` grid cell a row belongs to."""
    return (row["extra_info"]["theme"], row["extra_info"]["num_vars"],
            row["extra_info"]["ans_depth"])


def _body_chars(row: dict) -> int:
    """Body characters of a row's own passage (the question sentence excluded)."""
    return len(split_body(_question_text(row))[0])


def _length_summary(rows: list[dict]) -> dict:
    """Body-character alignment of the two shipped sides (section 4.6's fix).

    The pair construction matches the two members' sentence templates, so the
    shipped sides must not separate on length; this is the same quantity the L3
    estimator measures as a classifier.
    """
    negative = [row for row in rows if not row["extra_info"]["solvable"]]
    positive = [row for row in rows if row["extra_info"]["solvable"]]
    if not negative or not positive:
        return {"negative_rows": len(negative), "positive_rows": len(positive)}
    negative_chars = [_body_chars(row) for row in negative]
    positive_chars = [_body_chars(row) for row in positive]
    return {
        "negative_rows": len(negative),
        "positive_rows": len(positive),
        "negative_body_chars_mean": round(sum(negative_chars) / len(negative_chars), 2),
        "positive_body_chars_mean": round(sum(positive_chars) / len(positive_chars), 2),
        "delta_mean": round(
            (sum(positive_chars) - sum(negative_chars) * len(positive_chars) / len(negative_chars))
            / len(positive_chars), 2
        ),
        "negative_deleted_chars_mean": round(
            sum(len(row["extra_info"]["deleted_condition_text"]) for row in negative) / len(negative), 2
        ),
        "positive_deleted_chars_mean": round(
            sum(len(row["extra_info"]["deleted_condition_text"]) for row in positive) / len(positive), 2
        ),
    }


def _question_text(row: dict) -> str:
    content = row["prompt"][0]["content"]
    head, sep, _ = content.partition("\n\n")
    return head if sep else content


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the TreeCut hallucination-domain rows.")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--report", default=None, help="JSON funnel/cell plan (default: alongside --out)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report_path = args.report or report_path_for(args.out)

    rows, funnel = build_rows(args.raw_dir, limit=args.limit, seed=args.seed)
    schema.normalise_extra_info(rows)
    schema.validate_rows(rows)
    schema.write_rows_parquet(rows, args.out)
    plan = funnel["plan"]

    print(f"raw dir : {args.raw_dir}")
    print(f"wrote   : {args.out}  ({len(rows)} rows, seed={args.seed}; "
          f"targets: negatives {NEG_TARGET}, positives {POS_TARGET})")
    print("\nfunnel (negative rows remaining after each stage, and what that stage cost):")
    previous = None
    for stage, count in funnel.items():
        if not isinstance(count, int):
            continue
        if stage == "emitted_rows":
            print(f"  {stage:34s} {count:6d}   (+{len(rows) - funnel['after_option_drop']} positives)")
            continue
        cost = "" if previous is None else f"   -{previous - count}"
        print(f"  {stage:34s} {count:6d}{cost}")
        previous = count
    print(f"\n  generation attempts {plan['attempts']}, slots that needed a retry "
          f"{plan['retried_slots']}, slots dropped with no scenario "
          f"{sum(plan['scenario_drops'].values())}, pairs dropped by the pair certificate "
          f"{sum(plan['certificate_drops'].values())}, negatives dropped by the option "
          f"contract {sum(plan['negative_option_drops'].values())}, positives skipped (gold "
          f"not certifiable {sum(plan['answer_certificate_drops'].values())}, placeholder "
          f"block not buildable {plan['positive_placeholder_drops']})")
    for label, drops in (("certificate", plan["certificate_drops"]),
                         ("answer certificate", plan["answer_certificate_drops"]),
                         ("option", plan["negative_option_drops"])):
        for reason, count in drops.items():
            print(f"    {label} drop x{count}: {reason}")
    print("\nper branch (rows 5 and 7 of table B; the others are shown as zero):")
    for branch, count in branch_breakdown(rows).items():
        print(f"  {branch:22s} {count}")
    print("\nper template:")
    for template, count in _breakdown(rows, lambda r: r["extra_info"]["template"]).items():
        print(f"  {template:22s} {count}")
    print("\nper option_kind:")
    for kind, count in _breakdown(rows, lambda r: r["extra_info"]["option_kind"]).items():
        print(f"  {kind:22s} {count}")
    print("\nper placeholder mode (Q16: variable name / value, 50/50):")
    modes: collections.Counter = collections.Counter()
    for row in rows:
        modes.update(row["extra_info"].get("placeholder_modes") or [])
    for mode, count in sorted(modes.items()):
        print(f"  {mode:22s} {count}")
    print("\nper cell (theme/numVars/ansDepth):")
    cells: dict = collections.OrderedDict()
    for row in rows:
        entry = cells.setdefault(_cell_of(row), {"negatives": 0, "positives": 0})
        entry["positives" if row["extra_info"]["solvable"] else "negatives"] += 1
    for cell, counts in sorted(cells.items(), key=lambda item: str(item[0])):
        print(f"  {cell[0]:8s} nv{cell[1]} ad{cell[2]}      negatives {counts['negatives']:4d}"
              f"   positives {counts['positives']:3d}")
    if rows:
        print("\nlength alignment of the two shipped sides (body characters, question excluded):")
        for key, value in _length_summary(rows).items():
            print(f"  {key:32s} {value}")
    report = {
        "data_source": DATA_SOURCE,
        "raw_dir": args.raw_dir,
        "out": args.out,
        "seed": args.seed,
        "limit": args.limit,
        "targets": {"negatives": NEG_TARGET, "positives": POS_TARGET},
        "rows": len(rows),
        "negatives": sum(1 for row in rows if not row["extra_info"]["solvable"]),
        "positives": sum(1 for row in rows if row["extra_info"]["solvable"]),
        "funnel": dict(funnel),
        "plan": plan,
        "branch_breakdown": branch_breakdown(rows),
        "template_breakdown": _breakdown(rows, lambda r: r["extra_info"]["template"]),
        "option_kind_breakdown": _breakdown(rows, lambda r: r["extra_info"]["option_kind"]),
        "placeholder_modes": dict(sorted(modes.items())),
        "cell_breakdown": {
            f"{cell[0]}-nv{cell[1]}-ad{cell[2]}": counts
            for cell, counts in sorted(cells.items(), key=lambda item: str(item[0]))
        },
        "length": _length_summary(rows) if rows else {},
        "notes": [
            "Q11's grid: only 6 of the 9 numVars x ansDepth combinations can hold a pair "
            "(ansDepth == numVars leaves no non-necessary edge, ansDepth > numVars is rejected "
            "by the generator), so there are 12 cells; the 408-409 negatives per cell clear the "
            "'>= 100 per cell' floor and the 41-42 positives per cell cannot (D27 caps them at "
            "500). See DEVIATIONS item 3.",
            "The positive placeholder block is rendered by this adapter from the passage (the "
            "pinned generator has no placeholder mode). See DEVIATIONS item 4.",
            "order is a recorded value only: both members come from one shuffled render, but a "
            "shuffled order cannot be proved from the text.",
            "The answer certificate re-derives the gold along the certified fact-to-answer chain; "
            "sentences off that chain are not checked against the same value assignment. "
            "See DEVIATIONS item 7.",
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"\nreport  : {report_path}")


if __name__ == "__main__":  # pragma: no cover - exercised through main() in tests
    main()
