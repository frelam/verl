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
"""TreeCut adapter -- generated math pairs whose defect is provable by construction.

Source: ``github.com/j-bagel/treecut-math`` (Apache-2.0, pure Python, no
dependencies; the HF mirror ``jouyang/treecut-math`` holds 21,000 samples of the
same generator).  Unlike every other source in this pool TreeCut is a *generator*,
not a file: ``treecut/gen_data.py`` samples a dependency tree over
``numVars`` variables, renders one sentence per tree edge, and -- when the row is
meant to be unanswerable -- deletes one whole sentence.  The adapter runs the
pinned generator (``<raw_dir>/repo/treecut``, read out of the downloaded repo, not
vendored) and records the seam the design doc asks for.

Contract produced by this adapter (design doc section 4.9.3 table B row 6):

===================  ========  ================================================
branch               template  gold
===================  ========  ================================================
``unsolvable_bare``  B         ``\\boxed{UNSOLVABLE}`` (template B's refusal line)
===================  ========  ================================================

TreeCut's own generation-time cut (``cut = ans_upstream[cutDepth - 1]``) removes
an edge that is *provably on the root-to-answer path*, which is what makes the
unanswerability a construction fact rather than a human label: after the cut, the
answer variable sits in a component that has N variables and only N-1 equations
(``gen_disproof``'s certificate).  Section 4.9.2 pit 3 recorded the cost of using
the stock generator as-is: its own measurement puts the two classes 48 characters
apart on average (351.6 vs 303.6), the length heuristic reads 0.726 and the BoW NB
0.755 -- a structural leak (this adapter re-measures both figures on the released
files; see DEVIATIONS item 5).  The fix this adapter implements is the recon's
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
  shipped member must have **no path** from any fact to the asked variable; the
  counterpart must have one, of exactly ``ans_depth - 1`` hops.
* **Pairing** -- the two members must be the same scenario: same asked variable,
  same variable set (``num_vars`` of them), ``num_vars - 1`` sentences each, and a
  body-sentence multiset symmetric difference of exactly one sentence on each
  side, the shipped side's being ``deleted_condition_text``.  The counterpart's
  own derivation must exist at ``ans_depth - 1`` hops and must **not** use the
  sentence the counterpart itself lost -- of the pair's two cuts, only the shipped
  member's touches the root-to-answer chain -- and adding the recorded deleted
  sentence back to the shipped text must make the answer reachable again at
  ``ans_depth - 1`` hops.  Together those two halves are "this sentence is the
  pivot": removing it breaks the chain, restoring it rebuilds the chain.

A row whose certificate cannot be re-derived from its own two texts is dropped
and counted in the funnel; nothing is invented, no counterpart text is reused
across questions, and there is no fallback.

DEVIATIONS FROM THE DESIGN DOC
------------------------------

Measured against ``HALLUCINATION_RL_DESIGN.md`` (sections 4.9.1-4.9.3, 9, Q27),
``scratch/halluc_recon/`` (the recon's ground truth: ``final_summary.json``,
``hf_stats.json``, ``probe_treecut.py``) and the released HF files under
``<raw_dir>/hf``.  Every number below is measured by this adapter; the funnel
printed by ``main()`` shows where each row is lost.

1. **The doc's fix sentence reads inverted; the reading used here is the recon's
   measured one.**  Section 4.9.2 pit 3 says "负类不取'完整题'，改取'剪掉一条非必
   要边'的版本（句子数、变量数、长度分布全对齐，**可解性不变**）".  Taken
   literally, the unanswerable member would be a non-necessary-edge cut, i.e.
   still solvable, and the row could not carry ``\\boxed{UNSOLVABLE}`` -- and it
   would contradict the same section's own L1 argument ("剪掉的边必在根→答案路径
   上").  The reading that survives both the argument and the measurement is:
   shipped member = necessary-edge cut, counterpart = non-necessary-edge cut.
   Measured on the 12-cell grid below (1,084 pairs, 0 drops): length BA 0.4755 and
   BoW NB 0.3561 by ``verify_treecut.py`` on the built rows, against the recon's
   *recorded* ``FIX matched paired`` 0.4964 / 0.4695 and ``FIX naive paired``
   0.5221 / 0.5052 (``final_summary.json``).  The recon's recorded artifact is the
   only citable reading: ``probe_treecut.py`` seeds its configuration rng but its
   generator draws formulas with the global unseeded one, so re-running it
   reproduces neither its own recorded values nor a stable number across runs.

2. **The doc's ``proof`` string does not exist verbatim in the generator.**
   Section 9's TreeCut check ① says to assert that ``proof`` contains
   "N variables but M formulas".  The generator emits "There are {N} variables but
   only {M} linear formula(s), so we cannot calculate the price of {ans}", and
   uses the singular for M == 1 -- measured over the whole HF release: the literal
   "but M formulas" occurs 0 times.  Both this adapter and ``verify_treecut.py``
   parse the real string (accepting the singular and the plural) and additionally
   require ``M == N - 1``, which is the tree identity the certificate rests on.

3. **Q27's "≥100 per cell" is not satisfiable at the doc's own 1,084 rows.**
   Q27 asks for 1,084 rows stratified over ``numVars`` x ``ansDepth`` x ``theme``
   with "at least 100 rows per cell".  The doc's grid ``{4,6,8} x {2,4,6}`` gives
   9 combinations x 2 themes = 18 cells, and 1,084 / 18 = 60.2 -- below 100 before
   any validity argument.  Measured: of the 9 combinations only 6 can hold a
   *pair* (``ansDepth == numVars`` at (4,4) and (6,6) leaves no non-necessary edge
   to cut, so the counterpart cannot be built; ``ansDepth > numVars`` at (4,6) is
   rejected by the generator itself), so the grid this adapter uses is 6 configs x
   2 themes = 12 cells and 1,084 / 12 = 90.3, still below 100.  The 90-91 rows per
   cell are shipped and the shortfall is reported rather than padded, because
   padding would change the doc's own 1,084 (table B row 6) and
   ``mix_halluc.py``'s quota for this cell.

4. **Q27's 125 "unrealistic" rows are unsupported.**  The cell is not built.  The
   generator has exactly one defect mode (delete the sentence of an edge that lies
   on the root-to-answer path); no argument or flag makes it emit an unrealistic
   premise, and ``mix_halluc.py`` has no ``(unsolvable_diag, halluc_math_treecut)``
   quota cell, so a pointer-gold TreeCut row would have nowhere to go.  The 125
   rows are reported as unsupported instead of being fabricated.

5. **The doc's pre-fix L3 figures are not reproducible as pooled release numbers,
   but the doc's conclusion is.**  Section 4.9.1/4.9.2 and the version log quote
   TreeCut as "0.755 (length 0.726)".  Measured here on the released HF files
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
   matched non-necessary-edge cut) and the counterpart recorded here is the
   *solvable* member, i.e. also a one-sentence-short version of the scenario, not
   the generator's full problem.  The full problem is not recoverable from the
   pair alone; it is not needed, because the certificate is about the pair, and
   ``deleted_condition_text`` names the sentence the shipped question lost, which
   the counterpart still carries.

7. **The counterpart's numeric answer is deliberately not recorded.**  The
   generator knows it, but nothing in this source can certify the arithmetic of a
   number (the counterpart's own chain would have to be re-derived from the
   sentence set, and TreeCut ships no per-row source record to anchor against).
   Storing an unprovable number is exactly what fail-closed forbids, so the
   counterpart is text-only, and the shipped row's gold carries no number at all.

Module note on L1
-----------------

TreeCut's certificate is the strongest structural one in the pool: ``cut`` is
taken from ``ans_upstream``, so the deleted sentence is provably necessary, and
the deletion is a real syntactic operation on the text (the sentence is gone
verbatim, not paraphrased).  What the certificate does *not* do is compute the
answer: the shipped row's gold is the refusal, the number never enters the reward,
and the counterpart is audit-only.  The L3 alignment is what the design doc asks
to be measured (section 9), and ``verify_treecut.py`` measures it on the built
rows.
"""

from __future__ import annotations

import argparse
import ast
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

BRANCH_BARE = schema.BRANCH_UNSOLVABLE_BARE
TEMPLATE_B = schema.TEMPLATE_B

#: Table B row 6's number: the three-tier quota for TreeCut.  The source is a
#: generator, so this is a generation target, not a source-side cap.
ROW_TARGET = 1084

#: Stratification (design doc Q27): theme x numVars x ansDepth.  ``ansDepth ==
#: numVars`` is excluded because it leaves no off-path edge to cut, and
#: ``ansDepth > numVars`` is rejected by the generator itself.
GRID_THEMES = ("food", "outfit")
GRID_CONFIGS = ((4, 2), (6, 2), (6, 4), (8, 2), (8, 4), (8, 6))
ORDER = "random"  # every config the HF release ships uses order=random

#: Retries per generation slot before the slot is dropped (fail closed).
MAX_TRIES = 200

ERROR_TYPE = "key_information_missing"
PERTURBATION_TYPE = "missing_condition"

_QUESTION_MARK = "Question: "
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_PROOF_NUMBERS_RE = re.compile(r"There are (\d+) variables but only (\d+) linear formula")


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
    }


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
    key = f"{seed}|{theme}|{num_vars}|{ans_depth}|{ordinal}".encode("utf-8")
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
    keep = [s for edge, s in zip(render["edges"], render["sentences"]) if edge != cut_edge]
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

    The shipped member cuts ``ans_upstream[0]``, the answer's own parent edge, so
    its disproof certificate is the generator's own.  The counterpart cuts one
    non-necessary edge -- one whose child is not an ancestor of the answer and is
    itself a non-leaf, so no variable vanishes -- matched on whether that child
    hangs off ROOT (the two cut sentences then have the same template, which is
    what equalises the lengths).
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

    cut_node = next(node for node in scenario["nodes"] if node.name == negative_cut)
    negative["proof"] = source["gen_disproof"](
        cut_node.get_all_edges(),
        scenario["node2var"],
        render["sentence_dict"],
        scenario["ans_node"],
    )
    config = {
        "theme": theme,
        "num_vars": num_vars,
        "ans_depth": ans_depth,
        "order": order,
        "negative_cut": negative_cut,
        "positive_cut": positive_cut,
    }
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
# row construction
# ---------------------------------------------------------------------------


def row_for_pair(negative: dict, positive: dict, config: dict, index: int, seed: int) -> dict:
    """Assemble the shipped (negative) member of one pair as a contract row."""
    task_id = (f"treecut-{config['theme']}-nv{config['num_vars']}"
               f"-ad{config['ans_depth']}-{config['ordinal']:06d}")
    ground_truth = schema.build_ground_truth(
        solvable=False,
        answer=None,
        correct_option_id=None,
        has_diagnosis_label=False,
        perturbation_type=PERTURBATION_TYPE,
    )
    extra_info = {
        "split": "train",  # a generator has one implicit split
        "index": index,
        "seed": seed,
        "task_id": task_id,
        "error_type": ERROR_TYPE,
        "perturbation_type": PERTURBATION_TYPE,
        "paired_original_text": positive["problem"],
        "deleted_condition_text": negative["deleted_sentence"],
        "solvable": False,
        "correct_option_id": "",
        # TreeCut extensions (the HF release carries no per-row column for any of
        # these; normalise_extra_info backfills them with "" on other sources).
        "proof": negative["proof"],
        "num_vars": config["num_vars"],
        "ans_depth": config["ans_depth"],
        "theme": config["theme"],
        "order": config["order"],
        "asked_variable": text_graph(negative["problem"], config["matcher"])["asked"] or "",
    }
    return schema.make_row(
        data_source=DATA_SOURCE,
        question=negative["problem"],
        ground_truth=ground_truth,
        template=TEMPLATE_B,
        branch=BRANCH_BARE,
        extra_info=extra_info,
    )


def build_rows(raw_dir: str, limit: int | None = None, seed: int = 0) -> tuple:
    """Build the TreeCut parquet rows and the funnel that produced them.

    Returns ``(rows, funnel)``: ``rows`` are ready for
    :func:`schema.normalise_extra_info` / :func:`schema.validate_rows` /
    :func:`schema.write_rows_parquet`, and ``funnel`` is an ordered mapping of
    filter stage -> rows remaining.  The same ``(raw_dir, limit, seed)`` always
    produces byte-identical rows: each slot seeds the global ``random`` module
    from ``sha256(seed, theme, num_vars, ans_depth, ordinal)`` before the
    generator runs, and both members' sentence shuffles come from
    ``random.Random`` instances derived from the same key.
    """
    source = load_source(raw_dir)
    grid = cells()
    quotas = quota_for(len(grid), ROW_TARGET)
    matchers = {
        theme: compile_matcher(vocabulary_forms(source["vocabulary"][theme]["entities"],
                                               source["vocabulary"][theme]["item_dict"]))
        for theme in GRID_THEMES
    }

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
                "matcher": matchers[theme],
            }
            problems = certify_pair(negative, positive, config, matchers[theme])
            if problems:
                certificate_drops[f"{theme}-nv{num_vars}-ad{ans_depth}: {problems[0]}"] += 1
                continue
            funnel["after_certificate_drop"] += 1
            planned.append((cell_index, ordinal, negative, positive, config))

    interleaved = _interleave_by_cell(planned)
    rows: list = []
    for index, (_, _, negative, positive, config) in enumerate(interleaved):
        rows.append(row_for_pair(negative, positive, config, index, seed))
    if limit is not None:
        rows = rows[: max(limit, 0)]
    funnel["after_limit"] = len(rows)
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
        "attempts": attempts,
        "retried_slots": retried,
        "scenario_drops": dict(scenario_drops),
        "certificate_drops": dict(certificate_drops),
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _breakdown(rows: list[dict], key) -> dict:
    counter: dict = collections.Counter(key(row) for row in rows)
    return dict(sorted(counter.items(), key=lambda item: str(item[0])))


def branch_breakdown(rows: list[dict]) -> dict:
    """Every contract branch, including the four this source cannot fill."""
    counts = collections.Counter(row["extra_info"]["branch"] for row in rows)
    return {
        branch: counts.get(branch, 0)
        for branch in (
            schema.BRANCH_SOLVABLE_NUMERIC,
            schema.BRANCH_SOLVABLE_ROLES,
            schema.BRANCH_SOLVABLE_JUDGE,
            schema.BRANCH_UNSOLVABLE_DIAG,
            BRANCH_BARE,
        )
    }


def _length_summary(rows: list[dict]) -> dict:
    negative = [len(row["extra_info"]["deleted_condition_text"]) for row in rows]
    shipped = [len(split_body(_question_text(row))[0]) for row in rows]
    counterpart = [len(split_body(row["extra_info"]["paired_original_text"])[0]) for row in rows]
    return {
        "shipped_body_chars_mean": round(sum(shipped) / len(shipped), 2),
        "counterpart_body_chars_mean": round(sum(counterpart) / len(counterpart), 2),
        "delta_mean": round((sum(counterpart) - sum(shipped)) / len(shipped), 2),
        "deleted_sentence_chars_mean": round(sum(negative) / len(negative), 2),
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
    print(f"wrote   : {args.out}  ({len(rows)} rows, seed={args.seed})")
    print("\nfunnel (rows remaining after each stage, and what that stage cost):")
    previous = None
    for stage, count in funnel.items():
        if not isinstance(count, int):
            continue
        cost = "" if previous is None else f"   -{previous - count}"
        print(f"  {stage:34s} {count:6d}{cost}")
        previous = count
    print(f"\n  generation attempts {plan['attempts']}, slots that needed a retry "
          f"{plan['retried_slots']}, slots dropped with no scenario "
          f"{sum(plan['scenario_drops'].values())}, pairs dropped by the certificate "
          f"{sum(plan['certificate_drops'].values())}")
    for reason, count in plan["certificate_drops"].items():
        print(f"    certificate drop x{count}: {reason}")
    print("\nper branch (this source can only fill one; the others are shown as zero):")
    for branch, count in branch_breakdown(rows).items():
        print(f"  {branch:22s} {count}")
    print("\nper template:")
    for template, count in _breakdown(rows, lambda r: r["extra_info"]["template"]).items():
        print(f"  {template:22s} {count}")
    print("\nper solvable:")
    for solvable, count in _breakdown(rows, lambda r: r["extra_info"]["solvable"]).items():
        print(f"  {str(solvable):22s} {count}")
    print("\nper cell (theme/numVars/ansDepth):")
    for cell, count in _breakdown(
            rows, lambda r: (r["extra_info"]["theme"], r["extra_info"]["num_vars"],
                             r["extra_info"]["ans_depth"])).items():
        print(f"  {cell[0]:8s} nv{cell[1]} ad{cell[2]}      {count}")
    if rows:
        print("\nlength alignment of the pair (body characters, question excluded):")
        for key, value in _length_summary(rows).items():
            print(f"  {key:32s} {value}")
    report = {
        "data_source": DATA_SOURCE,
        "raw_dir": args.raw_dir,
        "out": args.out,
        "seed": args.seed,
        "limit": args.limit,
        "row_target": ROW_TARGET,
        "rows": len(rows),
        "funnel": dict(funnel),
        "plan": plan,
        "branch_breakdown": branch_breakdown(rows),
        "template_breakdown": _breakdown(rows, lambda r: r["extra_info"]["template"]),
        "cell_breakdown": {
            f"{cell[0]}-nv{cell[1]}-ad{cell[2]}": count
            for cell, count in _breakdown(
                rows, lambda r: (r["extra_info"]["theme"], r["extra_info"]["num_vars"],
                                 r["extra_info"]["ans_depth"])).items()
        },
        "length": _length_summary(rows) if rows else {},
        "unsupported": [
            "Q27's 125 'unrealistic condition' rows: the generator has one defect mode "
            "(a necessary-edge cut) and mix_halluc.py has no quota cell for a pointer-gold "
            "TreeCut row.",
            "Q27's 'at least 100 rows per cell': the doc's own grid gives 18 cells "
            "(1,084 / 18 = 60.2) and only 12 of them can hold a pair (1,084 / 12 = 90.3), "
            "both below 100.",
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"\nreport  : {report_path}")


if __name__ == "__main__":  # pragma: no cover - exercised through main() in tests
    main()
