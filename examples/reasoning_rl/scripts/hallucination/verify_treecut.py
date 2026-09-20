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
"""Audit the TreeCut rows produced by ``treecut_adapter.py``.

Usage::

    python verify_treecut.py --rows /tmp/treecut.parquet --raw-dir <raw subdir>

Reads only the artifact: every check re-derives its answer from the text the row
itself carries (``prompt`` question + ``extra_info.paired_original_text``, the
recorded ``deleted_condition_text``, ``proof``, ``options`` and the gold fields),
and never trusts the adapter's own verdict.  Prints ``PASS``/``FAIL`` per check and
exits non-zero if any check fails.

A TreeCut row is one side of a *pair* (D26 ships both sides, table B rows 5 and 7):

* the **negative** (branch ``unsolvable_diag``) is the member whose cut took an
  edge the answer's derivation needs -- nothing grounds the answer, so its gold is
  ``\\boxed{UNSOLVABLE: <选项ID>}`` and the option block holds candidate missing
  conditions;
* the **positive** (branch ``solvable_numeric``) is the counterpart, whose cut took
  an edge the derivation does not use, so it stays solvable and ships the
  generator's answer as a numeric gold with a *placeholder* option block.

Both members come from one shared render, which is why the pair is a fair
length-matched control and why the certificate can be re-derived from either side.

Layers
------

**Prompt binding.**  The stored prompt must be exactly ``schema.render_prompt``'s
output for the row's own question under the row's recorded ``template`` and
``options`` -- the prompt is frozen at write time, so the wording the model sees
can be re-derived from the row.  This is what ties the gold to the question: a
prompt rewritten to another template while the gold stayed the four-tier refusal
is a row that never asks for its own answer, and it fails here.  Both sides must
render template A with exactly k=3 option lines and the same instruction preamble
(the D18 isomorphism D26 re-states).

**Source anchor.**  The row's texts are proved against the source itself, and the
anchor fails closed: if ``<raw_dir>/repo/treecut/entities_items.py`` cannot be
read, the check is a ``FAIL``, not a skip.  Two parts:

* *vocabulary* -- the entity/item tables are parsed out of the file's literals
  (by :mod:`ast`, without importing it), turned into the surface forms a variable
  can be spelled with, and every row's two texts must then parse as a graph of
  known variables under **exactly one** theme's vocabulary -- the recorded one.
  This proves the variable names are the source's and that ``theme`` is not
  mislabelled.
* *generator* -- the pinned generator is imported and run on each configuration
  the artifact uses (both the uncut and the stock cut member), and the stock
  output must satisfy the same certificate this file asserts on the built rows:
  the uncut member's answer is reachable at ``ansDepth - 1`` hops, the stock cut
  member's answer is not reachable at all, and its ``proof``'s certificate agrees
  with the asked variable's own component.  That anchors the *reading* of the
  source -- text-to-graph parsing, reachability, the N/M tree identity -- against
  ground truth the adapter did not produce.

**L1 -- proof certificate** (design doc section 9, D26).  Every *negative* row's
``proof`` must carry the generator's disproof certificate: "There are N variables
but only M linear formula(s), so we cannot calculate the price of <asked>", with
``M == N - 1 < N``.  Re-derived from the row's own text: the certificate's
variable list must be exactly the asked variable's component (the variables the
shipped text still connects to the answer), N must be that component's variable
count and M its sentence count, and every clause the certificate cites -- it
quotes the sentences as ``l0(sentence[:-1])`` -- must actually **appear in the
recorded ``proof`` string**.  That last comparison is deliberately against the
proof and not against the graph the clauses were derived from: matching a clause
to the graph it came from is a tautology, so it could never catch a certificate
whose quoted justification had been fabricated or copied from another row.  The
positive side carries no proof of its own (it is solvable) and must say so.

**L2 -- pair certificate** (design doc section 9, D26).  Re-derived from the two
texts alone, and symmetric in the two sides (a row's own side is whichever member
is reachable from a root fact):

* the symmetric difference of the two bodies is exactly one sentence each way;
  a row's own lost sentence is its recorded ``deleted_condition_text`` and is
  absent from its own body;
* exactly one member is solvable (the two texts must not agree on reachability),
  the unsolvable member has **no** fact-to-answer chain and the solvable one has
  one of exactly ``ansDepth - 1`` hops;
* the solvable member's derivation does not use the sentence the *solvable* member
  lost ("the removed edge is provably irrelevant": the only edge whose removal
  changes solvability is the unsolvable member's);
* adding the *unsolvable* member's recorded lost sentence back to its own text
  makes the answer reachable at ``ansDepth - 1`` hops -- the pivot proves itself,
  using nothing but the row's own fields;
* the pair is one scenario: same asked variable, same variable set, same sentence
  count (``numVars - 1`` per member), and the same configuration.  ``theme`` is
  checked from the text, ``numVars`` from the variable count, ``ansDepth`` from
  the solvable member's hop count; ``order`` is recorded and cannot be recovered
  from a shuffled body, so it is only checked to be a legal value (see the note in
  ``main()``).

**L2b -- answer certificate** (design doc section 6 / 9, D26).  The positive's
gold is a number, so the artifact must prove it.  This layer re-derives it from
the positive's own passage: walk the certified fact-to-answer chain, decode each
step by enumerating the formulas the pinned renderer could have produced for that
exact sentence (``x, y`` over the generator's ``+/-1..3`` space, the sentence's own
numbers plus ``+/-1``/0 as candidate results), and require the propagated value to
equal the recorded gold.  The decoder is the source's *own renderer*, so this is
not a second prose grammar; it is the same computation the adapter performs,
repeated from the artifact.

**L3 -- anti-cheat gate** (design doc section 9, check 2; section 11 risk 1).  The
two shipped sides are the corpus: negative rows (label 0) against positive rows
(label 1).  Section 4.6 measured the stock generator's pair as leaking (the cut
member loses a whole sentence, so the two classes separate on length and on a bag
of words); this file re-measures the gate on the **built** rows, as the doc
requires:

* a 1-D length threshold, 5-fold out of fold, must read balanced accuracy
  ``<= 0.55`` (random + 5 points);
* a Bernoulli bag-of-words Naive Bayes, same folds, must read ``<= 0.55``;
* the shipped pair must actually be length-matched: mean ``|word-count delta| <= 2``.

Both raw numbers, a pair-aligned shuffled-label null control (the labels are
flipped *within* each shipped pair, so the estimator sees the same pairs with
random sides) and the length-alignment summary are printed pass or fail.  The
Bernoulli reading is reported with its known pathology: on near-identical
corpora it can read *below* chance (as it does here and as UMWP's verifier
records for that corpus), so a sub-0.5 reading is not evidence of anything by
itself -- the control is what makes it interpretable.

**L4 -- the D26 option contract and the leak gate** (design doc section 9, D26,
section 11 risk 1).  Every row carries template A with k=3 options (D15), and the
two sides look structurally identical (D18).  Per side:

* *negative*: exactly three options, ids A/B/C, pairwise distinct, **every option
  absent from the shipped passage verbatim**, each one a same-family condition
  (it parses under the row's own theme vocabulary and names a variable), and
  ``correct_option_id`` pointing at the option whose text is the recorded
  ``deleted_condition_text`` -- i.e. at the generator's own cut edge;
* *positive*: exactly three options, pairwise distinct, every one absent from the
  passage, ``correct_option_id`` null, and every one re-derivable from a passage
  sentence by **exactly one** variable-name swap or one variable-value swap (Q16;
  the mode balance is reported);
* *leak heuristic*: choosing uniformly among the options that are not in the
  passage must hit the gold at most ``1/k + 5pt = 38.3%``.  Because all three
  options of a negative are out of passage, the heuristic has no signal and reads
  ~k^-1; if an option were a passage condition the heuristic would collapse onto
  the gold (MiP's measured 95.6%, section 4.2).
* *option-shuffle invariance*: permuting a row's options and moving
  ``correct_option_id`` to the option that still holds the gold text must leave the
  **reward** unchanged -- checked against the frozen dispatcher, not a copy of its
  table.

**L5 -- the branch-dependent bare refusal** (design doc sections 5.1 and 6, D26).
``\\boxed{UNSOLVABLE}`` scores +1 on a three-tier source and **0** on TreeCut's
four-tier negatives -- the easiest cell in the matrix to get wrong.  This layer
runs the frozen dispatcher on a real TreeCut negative and on a three-tier payload,
and prints both data sources in the message.

**L5b -- the TreeCut positive's reward cells** (design doc section 9 matrix, D26).
The solvable side is an ordinary numeric row behind a placeholder block: the gold
answer scores +1, while a misrefusal (``\\boxed{UNSOLVABLE}``, with or without an
option id) and a placeholder pick (``\\boxed{B}``) score **0** -- never -1 -- and an
empty response scores 0.  The placeholder block is never read by the reward, which
is what this check pins down.

What is *not* re-derivable, stated plainly
-----------------------------------------

* ``order``: only the shuffle's output survives in the text, so a ``random``
  label cannot be proved from the artifact.  The pair shares it by construction
  (both members are filtered out of one render before the same transform).
* the negative's distractors are sibling cuts: that they come from the *same grid
  cell* is a build-time property of the generator's configuration, not recoverable
  from a single row.  What is checked instead is the property the leak defence
  rests on (every option out of passage, same-family sentence shapes).
* sentences **off** the positive's derivation chain are not checked against the
  same value assignment (that would need the generator's per-row formula record,
  which the source does not ship); the gold is derived from the chain only.
* the released HF files under ``<raw_dir>/hf`` are printed as a pre-fix
  reference for the section 9 gate.  They measure the *source*, not this build,
  and are informational: the doc quotes the stock leak as length 0.726 / BoW NB
  0.755 (section 4.6 / section 11 risk 1), and the pooled release reads 0.5670 on
  length (see the adapter's DEVIATIONS item 5).  Both exceed the gate, which is
  why the pair construction exists.
"""

from __future__ import annotations

import argparse
import ast
import collections
import glob
import json
import os
import random
import re
import sys
from collections import deque

import numpy as np

try:
    import schema
except ImportError:  # pragma: no cover - running as a plain script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import schema

BRANCH_DIAG = schema.BRANCH_UNSOLVABLE_DIAG
BRANCH_POSITIVE = schema.BRANCH_SOLVABLE_NUMERIC
TEMPLATE_A = schema.TEMPLATE_A

DEFAULT_RAW_DIR = "/home/charles/data/reasoning_rl/halluc/raw/treecut"
REPO_SUBDIR = os.path.join("repo", "treecut")
ENTITIES_FILE = "entities_items.py"
REWARD_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "reward", "hallucination_compute_score.py",
)

#: D15: every option block is exactly k = 3.
K_OPTIONS = 3
#: Section 9's hard gate on the two shipped sides: random + 5 points.
L3_MAX_BALANCED_ACCURACY = 0.55
L3_MIN_SUPPORT = 5
#: Section 9 / D26 / section 11 risk 1's option-leak gate: 1/k + 5pt = 38.3%.
LEAK_MAX_HIT_RATE = 1.0 / K_OPTIONS + 0.05
#: Q16's two placeholder modes.
PLACEHOLDER_MODES = ("variable_name", "variable_value")
#: The pair must be length-matched in words, not merely similar in characters.
ALIGNMENT_MAX_WORD_DELTA = 2.0

#: The generator's own certificate sentence; "formula" is singular when M == 1.
PROOF_NUMBERS_RE = re.compile(
    r"There are (\d+) variables but only (\d+) linear formula(?:s)?, so we cannot "
)
PROOF_VARIABLES_RE = re.compile(r"All we know about the prices of (.+?) are: ")
PROOF_ASKED_RE = re.compile(r"the price of (.+?)\.\s*$")
#: "numVars-4_ansDepth-2" and "_hallu-True" out of a released file name.
HF_CONFIG_RE = re.compile(r"_numVars-(\d+)_ansDepth-(\d+)")
HF_CUT_RE = re.compile(r"_hallu-(True|False)")
#: Numeric literals, with their spans, for the placeholder-value re-derivation.
NUMBER_SPAN_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")

ORDER_VALUES = ("forward", "backward", "random")


class Reporter:
    """Collects PASS/FAIL lines; the exit code is ``all(ok)``."""

    def __init__(self) -> None:
        self.results: list = []

    def check(self, name: str, ok: bool, detail: str) -> bool:
        self.results.append((name, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        return bool(ok)

    def note(self, message: str) -> None:
        print(f"       {message}")

    @property
    def failed(self) -> list:
        return [name for name, ok, _ in self.results if not ok]


# ---------------------------------------------------------------------------
# reading the artifact
# ---------------------------------------------------------------------------


def question_of(row: dict) -> str:
    """The question the model sees: the prompt is ``question + "\\n\\n" + template``.

    Raises:
        ValueError: the prompt carries no template separator, so it cannot be split.
    """
    content = row["prompt"][0]["content"]
    head, sep, _ = content.partition("\n\n")
    if not sep:
        raise ValueError(
            f"prompt of {row['extra_info'].get('task_id')} has no template separator"
        )
    return head


def question_or_empty(row: dict) -> str:
    """``question_of``, but a prompt with no seam yields ``""`` instead of raising.

    Checks use this one: a prompt whose template tail was stripped is a corrupt
    row to be *reported* by every layer that reads the text, not an exception that
    aborts the audit before any verdict is printed.
    """
    try:
        return question_of(row)
    except ValueError:
        return ""


def _text_of(value) -> str:
    return value if isinstance(value, str) else ""


def ground_truth_of(row: dict) -> dict:
    """The row's parsed ``ground_truth`` payload (``{}`` when unreadable).

    Every layer that reads a gold field goes through this, so a malformed payload
    is reported by the contract check and then treated as "no gold" rather than
    crashing the audit.
    """
    payload = row.get("reward_model", {}).get("ground_truth")
    if isinstance(payload, dict):
        return payload
    if not isinstance(payload, str):
        return {}
    try:
        parsed = json.loads(payload)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def is_positive(row: dict) -> bool:
    """Whether the row is the solvable side of its pair (D26's table B row 5)."""
    return bool(ground_truth_of(row).get("solvable"))


def task_base(task_id: str) -> str:
    """A row's pair key: the positive's ``task_id`` minus its ``-pos`` suffix."""
    return task_id[:-4] if task_id.endswith("-pos") else task_id


def shipped_pairs(rows: list) -> dict:
    """``{task_base: (negative_row, positive_row)}`` for the pairs that shipped both sides."""
    pairs: dict = {}
    for row in rows:
        base = task_base(row["extra_info"].get("task_id", ""))
        entry = pairs.setdefault(base, {"negative": None, "positive": None})
        entry["positive" if is_positive(row) else "negative"] = row
    return {base: (entry["negative"], entry["positive"])
            for base, entry in pairs.items() if entry["negative"] and entry["positive"]}


# ---------------------------------------------------------------------------
# the frozen reward dispatcher (the L4 / L5 assertions run the real table)
# ---------------------------------------------------------------------------


def load_reward_module():
    """Import ``reward/hallucination_compute_score.py`` (the frozen dispatcher).

    The option-shuffle invariance and the branch-dependent bare refusal are
    statements about the *reward*, so they are checked against the shipped
    dispatcher instead of against a copy of its decision table.  The module is
    loaded by path because this file lives under ``scripts/hallucination`` while
    the dispatcher lives under ``reward``; its own import fallback puts its
    directory on ``sys.path``, so it loads either way.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("hallucination_compute_score", REWARD_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# text plumbing -- this file's own re-derivation, not the adapter's
# ---------------------------------------------------------------------------


def split_body(text: str) -> tuple:
    """``(body, question sentence)``; the generator always ends with "Question: "."""
    mark = text.rfind("Question: ")
    if mark < 0:
        return text.strip(), ""
    return text[:mark].strip(), text[mark:].strip()


def sentences_of(text: str) -> list:
    """The body sentences of a problem (the question sentence removed)."""
    body, _ = split_body(text)
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", body)
            if sentence.strip()]


def clause_of(sentence: str) -> str:
    """The quote form of a sentence: the generator's ``l0(sentence[:-1])``.

    ``gen_disproof`` upper-cases a sentence when it renders it and lower-cases the
    first letter when it quotes it, so the quoted clause is the rendered sentence
    with the first character folded down and the full stop dropped.
    """
    return sentence[:1].lower() + sentence[1:-1]


def _variables_in(text: str, matcher: tuple) -> list:
    pattern, lookup = matcher
    return [lookup[match.group(0)] for match in pattern.finditer(text)]


def text_graph(text: str, matcher: tuple) -> dict:
    """Turn a problem text back into the graph it came from.

    ``links`` are ``(sentence, variable, variable)`` for every body sentence that
    names two variables, ``facts`` are ``(sentence, variable)`` for the sentences
    that name one (a root price), and ``asked`` is the variable the question asks
    about.  A sentence naming any other number of known variables means the
    vocabulary does not cover this text, which is a hard ``malformed`` flag.
    """
    body, question = split_body(text)
    facts: list = []
    links: list = []
    malformed: list = []
    for sentence in sentences_of(body):
        named = list(dict.fromkeys(_variables_in(sentence, matcher)))
        if len(named) == 1:
            facts.append((sentence, named[0]))
        elif len(named) == 2:
            links.append((sentence, named[0], named[1]))
        else:
            malformed.append(sentence)
    asked = _variables_in(question, matcher)
    return {
        "facts": facts,
        "links": links,
        "asked": asked[0] if asked else None,
        "malformed": malformed,
    }


def _variables(graph: dict) -> set:
    named = {variable for _, variable in graph["facts"]}
    for _, left, right in graph["links"]:
        named.add(left)
        named.add(right)
    return named


def _adjacency(graph: dict) -> dict:
    neighbours: dict = collections.defaultdict(list)
    for sentence, left, right in graph["links"]:
        neighbours[left].append((right, sentence))
        neighbours[right].append((left, sentence))
    return neighbours


def _component(graph: dict) -> tuple:
    """``(variables, links)`` of the asked variable's own component.

    A text whose question names no variable has no component; returning an empty
    one keeps the certificate comparison a *failure report* instead of a
    ``None.lower()`` crash on a corrupt row.
    """
    asked = graph["asked"]
    if asked is None:
        return set(), []
    neighbours = _adjacency(graph)
    seen = {asked}
    queue = deque([asked])
    while queue:
        node = queue.popleft()
        for other, _ in neighbours[node]:
            if other not in seen:
                seen.add(other)
                queue.append(other)
    inside = [link for link in graph["links"] if link[1] in seen and link[2] in seen]
    return seen, inside


def _derivation(graph: dict, target: str) -> tuple:
    """``(hops, sentences)`` of a shortest fact-to-``target`` chain.

    The chain starts at a root fact (whose own sentence is quoted first, since
    that is where the price is given) and then follows retained sentences; a
    returned hop count of ``None`` means the target is not reachable at all, which
    is the shipped member's required state.
    """
    neighbours = _adjacency(graph)
    queue = deque((variable, 0, (sentence,)) for sentence, variable in graph["facts"])
    seen = {variable for _, variable in graph["facts"]}
    while queue:
        node, distance, trail = queue.popleft()
        if node == target:
            return distance, list(trail)
        for other, sentence in neighbours[node]:
            if other not in seen:
                seen.add(other)
                queue.append((other, distance + 1, trail + (sentence,)))
    return None, []


def _reachable(graph: dict, target: str) -> bool:
    return target in _reachable_set(graph)


def _reachable_set(graph: dict) -> set:
    neighbours = _adjacency(graph)
    seen = {variable for _, variable in graph["facts"]}
    queue = deque(seen)
    while queue:
        node = queue.popleft()
        for other, _ in neighbours[node]:
            if other not in seen:
                seen.add(other)
                queue.append(other)
    return seen


# ---------------------------------------------------------------------------
# the source anchor: vocabulary
# ---------------------------------------------------------------------------


def _literal_value(node, namespace: dict):
    """Evaluate a literal-only AST node against earlier assignments."""
    if isinstance(node, ast.Str):  # Python 3.7 parses string literals as Str
        return node.s
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in namespace:
            raise ValueError(f"unknown name {node.id!r} in {ENTITIES_FILE}")
        return namespace[node.id]
    if isinstance(node, ast.List):
        return [_literal_value(element, namespace) for element in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_literal_value(element, namespace) for element in node.elts)
    if isinstance(node, ast.Dict):
        return {
            _literal_value(key, namespace): _literal_value(value, namespace)
            for key, value in zip(node.keys, node.values, strict=False)
        }
    raise ValueError(f"unsupported literal node {type(node).__name__} in {ENTITIES_FILE}")


def load_vocabulary(raw_dir: str) -> dict:
    """``{theme: {"forms", "pairs", "item_dict"}}``, parsed with :mod:`ast`.

    The file is never imported: the adapter imports it (executing the module), so
    reading it as literals is this file's independent route to the same names.
    ``forms`` is what the text parser matches on, ``pairs`` is the generator's
    ``node2var`` value for each canonical variable and ``item_dict`` its plural
    table -- the two extra pieces the answer certificate's decoder needs to hand
    the pinned renderer a synthetic edge.
    """
    path = os.path.join(raw_dir, REPO_SUBDIR, ENTITIES_FILE)
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    namespace: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                namespace[target.id] = _literal_value(node.value, namespace)
    tables = {
        "food": namespace.get("food_entity_item"),
        "outfit": namespace.get("outfit_entity_item"),
    }
    vocabulary: dict = {}
    for theme, table in tables.items():
        if not isinstance(table, dict):
            raise ValueError(f"{ENTITIES_FILE} has no {theme} table")
        forms: dict = {}
        pairs: dict = {}
        for entity in table["entities"]:
            for item, plural in table["item_dict"].items():
                key = f"{item} at {entity}"
                forms[key] = [key, f"{plural} at {entity}"]
                pairs[key] = (entity, item)
        vocabulary[theme] = {"forms": forms, "pairs": pairs, "item_dict": table["item_dict"]}
    return vocabulary


def compile_matcher(forms: dict) -> tuple:
    """``(regex, surface -> canonical)`` for one theme's variable spellings."""
    surface = [variant for variants in forms.values() for variant in variants]
    surface.sort(key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(spelling) for spelling in surface))
    lookup = {variant: key for key, variants in forms.items() for variant in variants}
    return pattern, lookup


def matchers_for(vocabulary: dict) -> dict:
    """``{theme: matcher}`` out of :func:`load_vocabulary`'s rich entries."""
    return {theme: compile_matcher(entry["forms"]) for theme, entry in vocabulary.items()}


def themes_of(text: str, matchers: dict) -> set:
    """The themes whose vocabulary parses ``text`` with no malformed sentence."""
    return {
        theme for theme, matcher in matchers.items()
        if not text_graph(text, matcher)["malformed"]
    }


def check_source_vocabulary(rows: list, reporter: Reporter, raw_dir: str,
                            matchers: dict) -> None:
    """Prove the texts are the source's own variable names, and ``theme`` is right."""
    path = os.path.join(raw_dir, REPO_SUBDIR, ENTITIES_FILE)
    if not matchers:
        reporter.check(
            "source anchor: vocabulary (entities_items.py parsed)",
            False,
            f"cannot read {path} -- without the entity tables the variable names "
            "cannot be proved; point --raw-dir at the downloaded treecut directory",
        )
        return
    reporter.note("source anchor: vocabularies " + ", ".join(
        f"{theme}={len(matcher[1])} spellings, "
        f"{len({key for key in matcher[1].values()})} variables"
        for theme, matcher in matchers.items()
    ))
    failures: list = []
    for row in rows:
        info = row["extra_info"]
        task_id = info["task_id"]
        theme = info.get("theme")
        if theme not in matchers:
            failures.append(f"{task_id}: recorded theme {theme!r} is not a source theme")
            continue
        for label, text in (("shipped", question_or_empty(row)),
                            ("counterpart", _text_of(info.get("paired_original_text")))):
            if not text:
                failures.append(f"{task_id}: {label} text is empty")
                continue
            found = themes_of(text, matchers)
            if found != {theme}:
                failures.append(
                    f"{task_id}: {label} text parses under {sorted(found) or 'no theme'}, "
                    f"recorded {theme!r}"
                )
    reporter.check(
        "source anchor: every row's text parses under exactly its recorded theme",
        not failures,
        f"{len(rows)} rows against {os.path.join(raw_dir, REPO_SUBDIR, ENTITIES_FILE)}, "
        f"{len(failures)} failures" + (f"; first: {failures[:3]}" if failures else ""),
    )


# ---------------------------------------------------------------------------
# the source anchor: the generator itself
# ---------------------------------------------------------------------------


def load_generator(raw_dir: str) -> dict:
    """Import the pinned generator directly (the adapter is not in this path).

    .. warning::
       This **executes** ``gen_data.py`` and ``gen_questions.py`` from the
       downloaded repository at import time, with this process's privileges.
       Importing them is the point -- the check below holds the *source*
       generator's own output to this file's certificate, and a local
       reimplementation would certify nothing -- but it does mean running this
       audit runs third-party code, so point ``--raw-dir`` at the repository
       ``fetch_raw.py`` downloaded (its bytes and sha256 are recorded in
       ``raw_manifest.json``) and at nothing else.  Same import, same trust, in
       ``treecut_adapter.py``: pip-installing the generator instead would be the
       way to get a pinned hash, and this repo is not packaged.
    """
    repo = os.path.join(raw_dir, REPO_SUBDIR)
    if not os.path.isdir(repo):
        raise FileNotFoundError(f"TreeCut generator not found under {repo}")
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import gen_data  # noqa: PLC0415 - imported from the downloaded repo
    import gen_questions  # noqa: PLC0415

    return {
        "generate_qa": gen_data.generate_qa,
        "gen_disproof": gen_questions.gen_disproof,
        # The answer certificate's decoder: the source's own sentence renderer and
        # formula record, so a derivation step is decoded by re-rendering it rather
        # than by a second prose grammar implemented here.
        "formula": gen_questions.Formula,
        "render_sentence": gen_questions.edge_and_formula_to_sentence,
    }


def check_source_generator(rows: list, reporter: Reporter, raw_dir: str,
                           matchers: dict) -> None:
    """Run the source generator and hold its *own* output to this file's certificate.

    The built rows are a fixed reconstruction (one shared render, two cuts), so
    they cannot be reproduced byte for byte by the stock generator.  What can be
    anchored is the reading: on each configuration the artifact uses, the
    generator's uncut member must be solvable at ``ansDepth - 1`` hops and its
    stock cut member (``cutDepth=1``) must be unsolvable with a deficient
    certificate that matches the asked variable's component -- computed here by
    :func:`text_graph`, :func:`_derivation` and friends, from the generator's text
    alone.  If this file's graph reading were wrong, this check would fail on
    ground truth the adapter never touched.
    """
    configs = sorted({
        (row["extra_info"].get("theme"), row["extra_info"].get("num_vars"),
         row["extra_info"].get("ans_depth")) for row in rows
    })
    if not configs:
        reporter.check(
            "source anchor: generator certificate on its own output",
            False,
            "no rows to take configurations from",
        )
        return
    try:
        generator = load_generator(raw_dir)
    except (FileNotFoundError, ImportError, ValueError) as error:
        reporter.check(
            "source anchor: generator certificate on its own output",
            False,
            f"cannot run the pinned generator ({error}) -- the certificate shape cannot "
            "be anchored; point --raw-dir at the downloaded treecut directory",
        )
        return

    failures: list = []
    for theme, num_vars, ans_depth in configs:
        if theme not in matchers or not isinstance(num_vars, int) or not isinstance(ans_depth, int):
            failures.append(f"{theme}/nv{num_vars}/ad{ans_depth}: unusable configuration")
            continue
        matcher = matchers[theme]
        for hallu, cut_depth in ((False, 0), (True, 1)):
            random.seed(f"verify-treecut:{theme}:{num_vars}:{ans_depth}:{hallu}")
            output = generator["generate_qa"](
                theme, True, num_vars, ans_depth, "random", hallu, cut_depth
            )
            label = f"{theme}/nv{num_vars}/ad{ans_depth}/{'cut' if hallu else 'uncut'}"
            graph = text_graph(output["problem"], matcher)
            if graph["malformed"]:
                failures.append(f"{label}: {len(graph['malformed'])} unparsable sentence(s)")
                continue
            if graph["asked"] is None:
                failures.append(f"{label}: the question names no known variable")
                continue
            if len(_variables(graph)) != num_vars:
                failures.append(f"{label}: {len(_variables(graph))} variables, expected {num_vars}")
                continue
            expected_sentences = num_vars - 1 if hallu else num_vars
            if len(sentences_of(output["problem"])) != expected_sentences:
                failures.append(
                    f"{label}: {len(sentences_of(output['problem']))} sentences, "
                    f"expected {expected_sentences}"
                )
                continue
            hops, _ = _derivation(graph, graph["asked"])
            if hallu:
                if hops is not None:
                    failures.append(f"{label}: the stock cut member is still solvable at {hops} hops")
                    continue
                failures.extend(_proof_failures(output["proof"], graph, label))
                if output["answer"] != "unknown":
                    failures.append(f"{label}: the stock cut member's answer is {output['answer']!r}")
            else:
                if hops != ans_depth - 1:
                    failures.append(f"{label}: uncut member is {hops} hops, expected {ans_depth - 1}")
    reporter.check(
        "source anchor: generator certificate on its own output",
        not failures,
        f"{len(configs)} configurations x (uncut, stock cut), {len(failures)} failures"
        + (f"; first: {failures[:3]}" if failures else ""),
    )


# ---------------------------------------------------------------------------
# L1 -- the proof certificate (section 9, check 1)
# ---------------------------------------------------------------------------


def _proof_failures(proof: str, graph: dict, label: str) -> list:
    """Re-derive the disproof certificate's shape from the text it is paired with."""
    problems: list = []
    numbers = PROOF_NUMBERS_RE.search(proof or "")
    variables = PROOF_VARIABLES_RE.search(proof or "")
    asked_in_proof = PROOF_ASKED_RE.search(proof or "")
    if not numbers:
        return [f"{label}: proof carries no 'There are N variables but only M linear "
                "formula(s)' certificate"]
    if not variables or not asked_in_proof:
        return [f"{label}: proof is not in the generator's certificate form"]
    declared = [name.strip() for name in variables.group(1).split(",") if name.strip()]
    proof_vars, proof_formulas = int(numbers.group(1)), int(numbers.group(2))
    asked = graph["asked"]
    if declared != list(dict.fromkeys(declared)):
        problems.append(f"{label}: the certificate lists a variable twice")
    if len(declared) != proof_vars:
        problems.append(f"{label}: the certificate lists {len(declared)} variables but says "
                        f"{proof_vars}")
    if not proof_formulas < proof_vars:
        problems.append(f"{label}: the certificate is not deficient: M={proof_formulas} "
                        f">= N={proof_vars}")
    elif proof_formulas != proof_vars - 1:
        problems.append(f"{label}: the certificate breaks the tree identity: "
                        f"M={proof_formulas} != N - 1")
    if asked is not None and asked_in_proof.group(1).strip() != asked:
        problems.append(f"{label}: the certificate concludes about "
                        f"{asked_in_proof.group(1).strip()!r}, not the asked variable {asked!r}")
    component_vars, component_links = _component(graph)
    if len(component_vars) != proof_vars:
        problems.append(f"{label}: the certificate says {proof_vars} variables, the asked "
                        f"variable's component has {len(component_vars)}")
    if len(component_links) != proof_formulas:
        problems.append(f"{label}: the certificate says {proof_formulas} formulas, the "
                        f"component has {len(component_links)} sentences")
    declared_folded = sorted(name.lower() for name in declared)
    component_folded = sorted(variable.lower() for variable in component_vars)
    if declared_folded != component_folded:
        problems.append(f"{label}: the certificate's variable list is not the asked "
                        "variable's component")
    # The citation is matched against the *recorded proof text* in the generator's
    # ``l0(sentence[:-1])`` quote form (first letter folded, full stop dropped).
    # Comparing it against ``graph`` instead would compare a derived value with
    # the graph it was derived from -- true by construction and unable to catch a
    # fabricated or transplanted clause, so it is intentionally not done here.
    quoted = proof or ""
    for sentence, _, _ in component_links:
        if clause_of(sentence) not in quoted:
            problems.append(f"{label}: the certificate does not quote the clause its own "
                            f"component rests on: {sentence[:60]!r}")
            break
    return problems


def check_l1_proof(rows: list, reporter: Reporter, matchers: dict) -> None:
    """The four-tier negatives' proof certificate; the positives carry none."""
    failures: list = []
    negatives = 0
    for row in rows:
        info = row["extra_info"]
        task_id = info["task_id"]
        theme = info.get("theme")
        if theme not in matchers:
            failures.append(f"{task_id}: recorded theme {theme!r} has no vocabulary")
            continue
        graph = text_graph(question_or_empty(row), matchers[theme])
        if graph["malformed"]:
            failures.append(f"{task_id}: shipped text does not parse")
            continue
        if info.get("asked_variable") != graph["asked"]:
            failures.append(f"{task_id}: recorded asked_variable "
                            f"{info.get('asked_variable')!r} is not the variable the "
                            f"question asks about ({graph['asked']!r})")
        proof = _text_of(info.get("proof"))
        if is_positive(row):
            if proof:
                failures.append(f"{task_id}: the solvable side carries a disproof certificate")
            continue
        negatives += 1
        if not proof:
            failures.append(f"{task_id}: the four-tier negative carries no proof certificate")
            continue
        failures.extend(_proof_failures(proof, graph, task_id))
    reporter.check(
        "L1 proof certificate (N variables, M == N - 1 formulas, component, clauses)",
        not failures,
        f"{negatives} negatives re-derived from their own proof, text and asked variable "
        f"({len(rows) - negatives} positives carry none), {len(failures)} failures"
        + (f"; first: {failures[:3]}" if failures else ""),
    )


# ---------------------------------------------------------------------------
# L2 -- the pair certificate (section 9, check 3)
# ---------------------------------------------------------------------------


def _pair_failures(row: dict, matcher: tuple) -> list:
    """Re-derive the certificate of one row's pair, from the row's own text.

    Symmetric in the two sides (D26 ships both): the row's own member is whichever
    text is reachable from a root fact, and the checks then apply to that member
    and to its counterpart by role rather than by which one happens to be shipped.
    """
    info = row["extra_info"]
    task_id = info["task_id"]
    shipped_text = question_or_empty(row)
    counterpart_text = _text_of(info.get("paired_original_text"))
    deleted = _text_of(info.get("deleted_condition_text"))
    num_vars = info.get("num_vars")
    ans_depth = info.get("ans_depth")
    problems: list = []

    if not counterpart_text or not deleted:
        return [f"{task_id}: the row records no counterpart text or no deleted sentence"]
    shipped_sentences = sentences_of(shipped_text)
    counterpart_sentences = sentences_of(counterpart_text)
    shipped_counts = collections.Counter(shipped_sentences)
    counterpart_counts = collections.Counter(counterpart_sentences)
    lost_by_shipped = list((counterpart_counts - shipped_counts).elements())
    lost_by_counterpart = list((shipped_counts - counterpart_counts).elements())
    if len(lost_by_shipped) != 1 or len(lost_by_counterpart) != 1:
        problems.append(
            f"{task_id}: the pair differs by {len(lost_by_shipped)}/"
            f"{len(lost_by_counterpart)} sentences, not 1/1"
        )
    else:
        if lost_by_shipped[0] != deleted:
            problems.append(f"{task_id}: deleted_condition_text is not the sentence this "
                            "row lost")
        if deleted in shipped_counts:
            problems.append(f"{task_id}: the deleted sentence is still present in this row's "
                            "own text")

    shipped_graph = text_graph(shipped_text, matcher)
    counterpart_graph = text_graph(counterpart_text, matcher)
    for label, graph in (("this row", shipped_graph), ("counterpart", counterpart_graph)):
        if graph["malformed"]:
            problems.append(f"{task_id}: {label} has {len(graph['malformed'])} unparsable "
                            f"sentence(s): {graph['malformed'][0][:60]!r}")
    if problems:
        return problems

    shipped_vars = _variables(shipped_graph)
    counterpart_vars = _variables(counterpart_graph)
    if shipped_graph["asked"] is None:
        return [f"{task_id}: the question names no known variable"]
    if shipped_graph["asked"] != counterpart_graph["asked"]:
        problems.append(f"{task_id}: the two members ask about different variables")
    if shipped_vars != counterpart_vars:
        problems.append(f"{task_id}: the two members name different variable sets "
                        f"({len(shipped_vars)} vs {len(counterpart_vars)})")
    if len(shipped_vars) != num_vars:
        problems.append(f"{task_id}: {len(shipped_vars)} variables, expected num_vars {num_vars}")
    if len(shipped_sentences) != num_vars - 1 or len(counterpart_sentences) != num_vars - 1:
        problems.append(f"{task_id}: sentence counts {len(shipped_sentences)}/"
                        f"{len(counterpart_sentences)}, expected num_vars - 1 ({num_vars - 1})")
    if problems:
        return problems

    asked = shipped_graph["asked"]
    row_solvable = _reachable(shipped_graph, asked)
    counterpart_solvable = _reachable(counterpart_graph, asked)
    if row_solvable == counterpart_solvable:
        problems.append(f"{task_id}: both members are "
                        f"{'solvable' if row_solvable else 'unsolvable'} -- a pair is one of each")
        return problems
    # The solvable member must derive the answer at ansDepth - 1 hops from a fact
    # without leaning on the sentence it itself lost; the unsolvable member must
    # become solvable again when its own lost sentence is put back (the pivot).
    if row_solvable:
        solvable_graph, solvable_lost = shipped_graph, lost_by_shipped
        unsolvable_text, unsolvable_lost = counterpart_text, lost_by_counterpart
    else:
        solvable_graph, solvable_lost = counterpart_graph, lost_by_counterpart
        unsolvable_text, unsolvable_lost = shipped_text, lost_by_shipped
    hops, trail = _derivation(solvable_graph, asked)
    if hops != ans_depth - 1:
        problems.append(f"{task_id}: the solvable member's derivation is {hops} hops, "
                        f"expected ans_depth - 1 ({ans_depth - 1})")
    elif solvable_lost and solvable_lost[0] in trail:
        problems.append(f"{task_id}: the solvable member's derivation uses the sentence the "
                        "solvable member itself lost")
    body, question = split_body(unsolvable_text)
    repaired = text_graph(f"{body} {unsolvable_lost[0]} {question}", matcher)
    repaired_hops, _ = _derivation(repaired, asked)
    if repaired_hops != ans_depth - 1:
        problems.append(f"{task_id}: adding the unsolvable member's lost sentence back gives "
                        f"{repaired_hops} hops, expected {ans_depth - 1}")
    return problems


def check_l2_pair(rows: list, reporter: Reporter, matchers: dict) -> None:
    failures: list = []
    for row in rows:
        info = row["extra_info"]
        theme = info.get("theme")
        if theme not in matchers:
            failures.append(f"{info['task_id']}: recorded theme {theme!r} has no vocabulary")
            continue
        failures.extend(_pair_failures(row, matchers[theme]))
    pairs = len(shipped_pairs(rows))
    reporter.check(
        "L2 pair certificate (1/1 sentence swap, one solvable side, derivation at ad - 1, pivot)",
        not failures,
        f"{len(rows)} rows ({pairs} with both sides shipped) re-derived from their own two "
        f"texts, {len(failures)} failures"
        + (f"; first: {failures[:3]}" if failures else ""),
    )

    illegal = sorted({
        row["extra_info"].get("order") for row in rows
        if row["extra_info"].get("order") not in ORDER_VALUES
    })
    reporter.check(
        "L2 configuration: the pair is one scenario (theme, numVars, ansDepth, order)",
        not illegal,
        f"{len(rows)} rows; theme, numVars and ansDepth are re-derived from the text and "
        f"agreed per pair, {len(illegal)} illegal order values"
        + (f": {illegal}" if illegal else ""),
    )
    reporter.note("L2 configuration: order is a recorded value only -- both members come "
                  "from one shuffled render, but a shuffled order cannot be proved from it")


# ---------------------------------------------------------------------------
# L2b -- the answer certificate (section 6 / 9, D26)
# ---------------------------------------------------------------------------


def _candidate_results(sentence: str) -> list:
    """The values a sentence could encode: its numbers, their negatives, +/-1, 0.

    ``+/-1`` is always a candidate because the renderer spells it "a dollar"
    instead of "1 dollar"; 0 is the "is the same as that of" form, which carries no
    number at all.
    """
    values: list = [1, -1]
    for token in schema.numbers_in(sentence):
        if token.isdigit():
            values.extend([int(token), -int(token)])
    values.append(0)
    return list(dict.fromkeys(values))


def _chain_to_ask(graph: dict) -> list:
    """The shortest fact-to-answer chain as ``[(variable, sentence), ...]``, or ``[]``."""
    asked = graph["asked"]
    if asked is None:
        return []
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
    return []


def answer_failures(row: dict, entry: dict, generator: dict) -> list:
    """Re-derive the positive's numeric gold from its own passage; ``[]`` == certified.

    The decoder is the pinned generator's own ``edge_and_formula_to_sentence``: a
    derivation step is decoded by asking which formula would have rendered that
    exact sentence (``x, y`` over the generator's ``+/-1..3`` space, the sentence's
    own numbers plus ``+/-1``/0 as candidate results), and the value is propagated
    down the chain.  Nothing here trusts the adapter's certificate -- the ground
    truth is the renderer.
    """
    info = row["extra_info"]
    task_id = info["task_id"]
    matcher = compile_matcher(entry["forms"])
    graph = text_graph(question_or_empty(row), matcher)
    if graph["malformed"]:
        return [f"{task_id}: the passage has {len(graph['malformed'])} unparsable sentence(s)"]
    if graph["asked"] is None:
        return [f"{task_id}: the question sentence names no known variable"]
    asked = graph["asked"]
    if asked not in entry["pairs"]:
        return [f"{task_id}: asked variable {asked!r} is not in the theme's vocabulary"]
    chain = _chain_to_ask(graph)
    if not chain:
        return [f"{task_id}: the solvable member has no fact-to-answer chain"]
    fact_variable, fact_sentence = chain[0]
    if fact_variable not in entry["pairs"]:
        return [f"{task_id}: fact variable {fact_variable!r} is not in the theme's vocabulary"]

    facts = [
        res for res in _candidate_results(fact_sentence)
        if generator["render_sentence"](
            ("ROOT", "C"), generator["formula"]((1,), res),
            {"ROOT": "ROOT", "C": entry["pairs"][fact_variable]}, entry["item_dict"]
        ) == fact_sentence
    ]
    if len(facts) != 1:
        return [f"{task_id}: the root fact decodes to {len(facts)} values, not 1: "
                f"{fact_sentence[:60]!r}"]
    value = facts[0]
    previous = fact_variable
    for variable, sentence in chain[1:]:
        if variable not in entry["pairs"]:
            return [f"{task_id}: chain variable {variable!r} is not in the theme's vocabulary"]
        node2var = {"ROOT": "ROOT", "P": entry["pairs"][previous], "C": entry["pairs"][variable]}
        steps = []
        for x in (1, -1, 2, -2, 3, -3):
            for y in (1, -1, 2, -2, 3, -3):
                for res in _candidate_results(sentence):
                    rendered = generator["render_sentence"](
                        ("P", "C"), generator["formula"]((x, y), res), node2var, entry["item_dict"]
                    )
                    if rendered != sentence:
                        continue
                    numerator = res - x * value
                    if numerator % y:
                        continue
                    candidate = numerator // y
                    if candidate > 0:
                        steps.append(candidate)
        steps = list(dict.fromkeys(steps))
        if len(steps) != 1:
            return [f"{task_id}: the chain step decodes to {len(steps)} values, not 1: "
                    f"{sentence[:60]!r}"]
        value = steps[0]
        previous = variable

    gold = ground_truth_of(row).get("answer")
    if str(gold).strip() != str(value):
        return [f"{task_id}: the chain derives {value}, the recorded gold is {gold!r}"]
    return []


def check_answer_gold(rows: list, reporter: Reporter, vocabulary: dict,
                      generator: dict | None) -> None:
    """Every positive's number must be re-derivable from its own text."""
    positives = [row for row in rows if is_positive(row)]
    if not positives:
        reporter.check("L2b answer certificate (gold re-derived from the passage)", False,
                       "no positive rows in the artifact")
        return
    if generator is None:
        reporter.check(
            "L2b answer certificate (gold re-derived from the passage)",
            False,
            "the pinned generator could not be imported, so the renderer that decodes a "
            "chain step is unavailable; point --raw-dir at the downloaded treecut directory",
        )
        return
    failures: list = []
    for row in positives:
        entry = vocabulary.get(row["extra_info"].get("theme"))
        if not entry:
            failures.append(f"{row['extra_info']['task_id']}: recorded theme has no vocabulary")
            continue
        failures.extend(answer_failures(row, entry, generator))
    reporter.check(
        "L2b answer certificate (gold re-derived from the passage)",
        not failures,
        f"{len(positives)} positives decoded step by step with the pinned renderer, "
        f"{len(failures)} failures" + (f"; first: {failures[:3]}" if failures else ""),
    )


# ---------------------------------------------------------------------------
# L4 -- the D26 option contract and the leak gate (section 9, D26, section 11)
# ---------------------------------------------------------------------------


def _mask_numbers(text: str) -> str:
    return NUMBER_SPAN_RE.sub("\x00", text)


def _mask_variables(text: str, matcher: tuple) -> str:
    return matcher[0].sub("\x00", text)


def _number_tokens(text: str) -> list:
    return NUMBER_SPAN_RE.findall(text)


def _variable_spellings(text: str, matcher: tuple) -> list:
    return [match.group(0) for match in matcher[0].finditer(text)]


def placeholder_mode_of(option: str, passage: str, matcher: tuple) -> str | None:
    """Which Q16 swap turns a passage sentence into ``option``?  ``None`` if neither.

    The reading is exact rather than masked-only: a *name* swap keeps the sentence's
    numbers (in order) and changes a variable spelling while leaving everything else
    byte-identical; a *value* swap keeps the variable spellings (in order) and
    changes a number.  Anything else is not the placeholder shape D26 asks for.
    """
    for sentence in sentences_of(passage):
        if sentence == option:
            continue
        if (_number_tokens(sentence) == _number_tokens(option)
                and _variable_spellings(sentence, matcher) != _variable_spellings(option, matcher)
                and _mask_variables(sentence, matcher) == _mask_variables(option, matcher)):
            return PLACEHOLDER_MODES[0]
        if (_variable_spellings(sentence, matcher) == _variable_spellings(option, matcher)
                and _number_tokens(sentence) != _number_tokens(option)
                and _mask_numbers(sentence) == _mask_numbers(option)):
            return PLACEHOLDER_MODES[1]
    return None


def _negative_option_failures(row: dict, matcher: tuple) -> list:
    info = row["extra_info"]
    task_id = info["task_id"]
    passage = question_or_empty(row)
    options = info.get("options") or []
    gt = ground_truth_of(row)
    problems: list = []
    if len(options) != K_OPTIONS:
        return [f"{task_id}: {len(options)} options, expected k={K_OPTIONS}"]
    ids = [opt.get("id") for opt in options]
    if ids != list("ABCDEFG"[:K_OPTIONS]):
        problems.append(f"{task_id}: option ids {ids} are not A/B/C")
    texts = [opt.get("text") or "" for opt in options]
    if len(set(texts)) != len(texts):
        problems.append(f"{task_id}: option texts are not pairwise distinct")
    inside = [text[:40] for text in texts if text and text in passage]
    if inside:
        problems.append(f"{task_id}: {len(inside)} option(s) appear in the passage verbatim: "
                        f"{inside}")
    for text in texts:
        graph = text_graph(text, matcher)
        named = list(dict.fromkeys(_variables_in(text, matcher)))
        if graph["malformed"] or not named or len(named) > 2:
            problems.append(f"{task_id}: option {text[:50]!r} is not a same-family condition")
            break
    correct = gt.get("correct_option_id")
    if correct not in ids:
        problems.append(f"{task_id}: correct_option_id {correct!r} is not among {ids}")
    else:
        gold_text = next(opt["text"] for opt in options if opt["id"] == correct)
        if gold_text != _text_of(info.get("deleted_condition_text")):
            problems.append(f"{task_id}: correct_option_id points at {gold_text[:50]!r}, "
                            "not at the generator's cut edge")
    return problems


def _positive_option_failures(row: dict, matcher: tuple) -> list:
    info = row["extra_info"]
    task_id = info["task_id"]
    passage = question_or_empty(row)
    options = info.get("options") or []
    gt = ground_truth_of(row)
    problems: list = []
    if len(options) != K_OPTIONS:
        return [f"{task_id}: {len(options)} options, expected k={K_OPTIONS}"]
    ids = [opt.get("id") for opt in options]
    if ids != list("ABCDEFG"[:K_OPTIONS]):
        problems.append(f"{task_id}: option ids {ids} are not A/B/C")
    texts = [opt.get("text") or "" for opt in options]
    if len(set(texts)) != len(texts):
        problems.append(f"{task_id}: option texts are not pairwise distinct")
    inside = [text[:40] for text in texts if text and text in passage]
    if inside:
        problems.append(f"{task_id}: {len(inside)} placeholder(s) appear in the passage "
                        f"verbatim: {inside}")
    if gt.get("correct_option_id") is not None or info.get("correct_option_id") not in ("", None):
        problems.append(f"{task_id}: a placeholder block must carry no correct option "
                        f"(gold {gt.get('correct_option_id')!r}, extra_info "
                        f"{info.get('correct_option_id')!r})")
    modes = [placeholder_mode_of(text, passage, matcher) for text in texts]
    if any(mode is None for mode in modes):
        problems.append(f"{task_id}: an option is not one name/value swap away from any "
                        f"passage sentence: {[t[:40] for t, m in zip(texts, modes, strict=False) if m is None]}")
    return problems


def check_option_structure(rows: list, reporter: Reporter, matchers: dict) -> None:
    """(a)/(c) per-side option contract: k=3, distinct, out of passage, gold pointer."""
    negatives = [row for row in rows if not is_positive(row)]
    positives = [row for row in rows if is_positive(row)]
    failures: list = []
    for row in negatives:
        matcher = matchers.get(row["extra_info"].get("theme"))
        if matcher is None:
            failures.append(f"{row['extra_info']['task_id']}: no vocabulary for the theme")
            continue
        failures.extend(_negative_option_failures(row, matcher))
    reporter.check(
        "L4a negatives: k=3 candidate missing conditions, all out of passage, gold = cut edge",
        not failures and bool(negatives),
        f"{len(negatives)} negative rows, {len(failures)} failures"
        + (f"; first: {failures[:3]}" if failures else ""),
    )

    positives = [row for row in rows if is_positive(row)]
    failures = []
    for row in positives:
        matcher = matchers.get(row["extra_info"].get("theme"))
        if matcher is None:
            failures.append(f"{row['extra_info']['task_id']}: no vocabulary for the theme")
            continue
        failures.extend(_positive_option_failures(row, matcher))
    reporter.check(
        "L4c positives: k=3 placeholders, all out of passage, no correct option",
        not failures and bool(positives),
        f"{len(positives)} positive rows, {len(failures)} failures"
        + (f"; first: {failures[:3]}" if failures else ""),
    )

    modes: collections.Counter = collections.Counter()
    for row in positives:
        matcher = matchers.get(row["extra_info"].get("theme"))
        if matcher is None:
            continue
        passage = question_or_empty(row)
        for opt in row["extra_info"].get("options") or []:
            mode = placeholder_mode_of(opt.get("text") or "", passage, matcher)
            if mode:
                modes[mode] += 1
    reporter.note(
        "L4c placeholder modes (Q16, 50/50 per option): "
        + ", ".join(f"{mode}={modes.get(mode, 0)}" for mode in PLACEHOLDER_MODES)
    )


def leak_heuristic_hit_rate(rows: list, seed: int = 0) -> tuple:
    """``(hit rate, negative rows, rows with a unique absent option)``.

    The heuristic is MiP's "pick the sentence you cannot find in the passage"
    (section 4.2), applied to TreeCut's option blocks: guess uniformly among the
    options that are absent from the passage.  All three of a negative's options
    are out of passage by construction, so the guess carries no signal and the rate
    sits at ~1/k; a row with exactly one absent option would hand the heuristic the
    gold, which is what the section 9 gate (``<= 1/k + 5pt``) exists to catch.
    """
    rng = random.Random(seed)
    negatives = [row for row in rows if not is_positive(row)]
    if not negatives:
        return 0.0, 0, 0
    hits = 0
    unique = 0
    for row in negatives:
        passage = question_or_empty(row)
        options = row["extra_info"].get("options") or []
        absent = [opt["id"] for opt in options if (opt.get("text") or "") not in passage]
        if len(absent) == 1:
            unique += 1
        if not absent:
            continue
        guess = rng.choice(absent)
        hits += int(guess == ground_truth_of(row).get("correct_option_id"))
    return hits / len(negatives), len(negatives), unique


def check_option_leak(rows: list, reporter: Reporter, *, seed: int) -> None:
    """(b) the leak heuristic must read at most 1/k + 5pt = 38.3%."""
    rate, count, unique = leak_heuristic_hit_rate(rows, seed=seed)
    if not count:
        reporter.check("L4b option leak heuristic <= 1/k + 5pt", False,
                       "no negative rows in the artifact")
        return
    reporter.note(f"L4b analytic chance level is 1/k = {1.0 / K_OPTIONS:.4f}; rows with a "
                  f"unique out-of-passage option: {unique}")
    reporter.check(
        f"L4b option leak heuristic ('pick the option not in the passage') <= "
        f"{LEAK_MAX_HIT_RATE:.4f}",
        rate <= LEAK_MAX_HIT_RATE and unique == 0,
        f"hit rate {rate:.4f} over {count} negative rows ({unique} rows where the heuristic "
        f"is forced onto the gold); section 4.2 measured MiP's version at 0.956",
    )


def option_shuffle_failures(row: dict, reward, rng: random.Random) -> list:
    """One row's option-shuffle invariance against the frozen dispatcher.

    Permuting the option block while moving ``correct_option_id`` to the option that
    still holds the gold text must not change the reward: that is the adapter-side
    companion of section 9's "reward only compares the option id" rule, and it
    catches a positional-index bug that a text-only audit would miss.
    """
    info = row["extra_info"]
    task_id = info["task_id"]
    options = info.get("options") or []
    gt = ground_truth_of(row)
    ids = [opt.get("id") for opt in options]
    correct = gt.get("correct_option_id")
    if len(options) < 2 or correct not in ids:
        return [f"{task_id}: no scorable option block to permute ({ids}, gold {correct!r})"]
    texts = [opt["text"] for opt in options]
    gold_text = next(opt["text"] for opt in options if opt["id"] == correct)
    rng.shuffle(texts)
    problems: list = []
    if sorted(texts) != sorted(opt["text"] for opt in options):
        problems.append(f"{task_id}: the permutation lost or duplicated an option text")
    moved = dict(gt)
    moved["correct_option_id"] = next(ids[index] for index, text in enumerate(texts)
                                      if text == gold_text)
    before = reward.score_from_status(gt, reward.STATUS_UNSOLVABLE_OPTION, None, correct)
    after = reward.score_from_status(moved, reward.STATUS_UNSOLVABLE_OPTION, None,
                                     moved["correct_option_id"])
    if before != 1.0:
        problems.append(f"{task_id}: the recorded gold scores {before}, expected +1")
    if after != 1.0:
        problems.append(f"{task_id}: after permuting the block the gold text scores {after}, "
                        "expected +1 (the id moved with the text)")
    wrong = next((option_id for option_id in ids if option_id != moved["correct_option_id"]), None)
    if wrong is not None:
        wrong_score = reward.score_from_status(moved, reward.STATUS_UNSOLVABLE_OPTION, None, wrong)
        if wrong_score != 0.0:
            problems.append(f"{task_id}: a wrong option id scores {wrong_score}, expected 0")
    return problems


def check_option_shuffle(rows: list, reporter: Reporter, reward, *, seed: int) -> None:
    """(d) the option-shuffle invariance check on every four-tier negative."""
    negatives = [row for row in rows if not is_positive(row)]
    if not negatives:
        reporter.check("L4d option-shuffle invariance (reward unchanged)", False,
                       "no negative rows in the artifact")
        return
    failures: list = []
    for index, row in enumerate(negatives):
        failures.extend(option_shuffle_failures(row, reward, random.Random(f"{seed}:{index}")))
    reporter.check(
        "L4d option-shuffle invariance (reward unchanged when the block is permuted)",
        not failures,
        f"{len(negatives)} negative rows permuted against "
        f"{os.path.basename(REWARD_FILE)}, {len(failures)} failures"
        + (f"; first: {failures[:3]}" if failures else ""),
    )


def check_bare_unsolvable(rows: list, reporter: Reporter, reward) -> None:
    """(e) the branch-dependent bare refusal, both directions, with data_source.

    ``\\boxed{UNSOLVABLE}`` is +1 on a three-tier source (MiP, UMWP-unanswerable,
    CREPE-FP) and **0** on TreeCut's four-tier negatives (D26, sections 5.1 / 6).
    The same string, two scores: the check runs the frozen dispatcher on a real
    TreeCut negative and on a three-tier payload, and names both data sources.
    """
    negatives = [row for row in rows if not is_positive(row)]
    if not negatives:
        reporter.check("L5 bare UNSOLVABLE is branch-dependent", False,
                       "no negative rows in the artifact")
        return
    reply = "<think>\nThe conditions given are not sufficient.\n</think>\n\\boxed{UNSOLVABLE}"
    failures: list = []
    for row in negatives:
        source = row.get("data_source", "")
        score = reward.score_halluc_row(source, reply, row["reward_model"]["ground_truth"])
        if score != 0.0:
            failures.append(f"{source}: bare \\boxed{{UNSOLVABLE}} on the four-tier negative "
                            f"{row['extra_info']['task_id']} scored {score}, expected 0")
            break
    three_tier = schema.build_ground_truth(
        solvable=False, answer=None, correct_option_id=None, has_diagnosis_label=False,
        perturbation_type="missing_condition",
    )
    three_score = reward.score_halluc_row(schema.SOURCE_MIP, reply, three_tier)
    if three_score != 1.0:
        failures.append(f"{schema.SOURCE_MIP}: bare \\boxed{{UNSOLVABLE}} on the three-tier "
                        f"payload scored {three_score}, expected +1")
    correctness: list = []
    sample = negatives[0]
    gold = ground_truth_of(sample).get("correct_option_id")
    option_reply = (f"<think>\nThe conditions given are not sufficient.\n</think>\n"
                    f"\\boxed{{UNSOLVABLE: {gold}}}")
    option_score = reward.score_halluc_row(sample["data_source"], option_reply,
                                           sample["reward_model"]["ground_truth"])
    if option_score != 1.0:
        correctness.append(f"{sample['data_source']}: \\boxed{{UNSOLVABLE: {gold}}} scored "
                           f"{option_score}, expected +1")
    answer_score = reward.score_halluc_row(
        sample["data_source"], "<think>\nForty-two.\n</think>\n\\boxed{42}",
        sample["reward_model"]["ground_truth"],
    )
    if answer_score != -1.0:
        correctness.append(f"{sample['data_source']}: fabricating an answer scored "
                           f"{answer_score}, expected -1")
    reporter.check(
        "L5 bare UNSOLVABLE is branch-dependent (four-tier 0, three-tier +1)",
        not failures and not correctness,
        f"TreeCut negatives ({sample['data_source']}) -> 0, {schema.SOURCE_MIP} (three-tier) "
        f"-> +1; the diagnosed refusal -> +1 and a fabricated answer -> -1 on the same "
        f"four-tier row, {len(failures) + len(correctness)} failures"
        + (f"; first: {(failures + correctness)[:3]}" if failures or correctness else ""),
    )


def check_positive_reward(rows: list, reporter: Reporter, reward) -> None:
    """The TreeCut positive's reward cells (design doc sections 6 and 9, D26).

    The solvable side is an ordinary numeric row with a placeholder block: the gold
    answer scores +1, and a misrefusal -- ``\\boxed{UNSOLVABLE}`` or a bare option id
    -- scores **0**, never -1.  The block is never read by the reward, which is why
    picking a placeholder earns nothing rather than earning a wrong-answer penalty.
    """
    positives = [row for row in rows if is_positive(row)]
    if not positives:
        reporter.check("L5b TreeCut positives score on the numeric branch", False,
                       "no positive rows in the artifact")
        return
    row = positives[0]
    source = row.get("data_source", "")
    gold = ground_truth_of(row).get("answer")
    cases = [
        (f"\\boxed{{{gold}}}", 1.0, "the gold answer"),
        ("\\boxed{UNSOLVABLE}", 0.0, "a misrefusal"),
        ("\\boxed{UNSOLVABLE: B}", 0.0, "a diagnosed misrefusal"),
        ("\\boxed{B}", 0.0, "a placeholder pick"),
        ("no box at all", 0.0, "an empty response"),
    ]
    failures: list = []
    for reply, expected, label in cases:
        solution = f"<think>\nWorking.\n</think>\n{reply}"
        score = reward.score_halluc_row(source, solution, row["reward_model"]["ground_truth"])
        if score != expected:
            failures.append(f"{row['extra_info']['task_id']}: {label} scored {score}, "
                            f"expected {expected}")
    reporter.check(
        "L5b TreeCut positives score on the numeric branch (gold +1, misrefusal 0, "
        "placeholder pick 0)",
        not failures,
        f"{source} ({len(positives)} positives; checked {row['extra_info']['task_id']}), "
        f"{len(failures)} failures" + (f"; first: {failures[:3]}" if failures else ""),
    )


# ---------------------------------------------------------------------------
# L3 -- the anti-cheat gate (section 9, check 2)
# ---------------------------------------------------------------------------


def _fold_of(count: int, folds: int, rng: random.Random) -> np.ndarray:
    order = list(range(count))
    rng.shuffle(order)
    fold = np.empty(count, dtype=np.int64)
    for rank, index in enumerate(order):
        fold[index] = rank % folds
    return fold


def length_only_oof(lengths: list, labels: list, *, folds: int = 5, seed: int = 0) -> float:
    """Out-of-fold balanced accuracy of a 1-D length threshold, numpy only.

    Each fold learns the threshold and the direction (longer means positive /
    shorter means positive) that maximise balanced accuracy on its training half,
    then scores its held-out half.  Folds are cut inside each class, so every test
    fold holds both labels in the same proportion -- an unbalanced fold would make
    a balanced-accuracy score noisy rather than informative.
    """
    lengths = np.asarray(lengths, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    rng = random.Random(seed)
    positive = np.where(labels == 1)[0]
    negative = np.where(labels == 0)[0]
    positive_fold = _fold_of(len(positive), folds, rng)
    negative_fold = _fold_of(len(negative), folds, rng)
    scores: list = []
    for fold in range(folds):
        train_positive = lengths[positive[positive_fold != fold]]
        test_positive = lengths[positive[positive_fold == fold]]
        train_negative = lengths[negative[negative_fold != fold]]
        test_negative = lengths[negative[negative_fold == fold]]
        if not len(test_positive) or not len(test_negative):
            continue
        best, threshold, positive_is_longer = -1.0, float(train_positive[0]), True
        for candidate in np.unique(np.concatenate([train_positive, train_negative])):
            longer = 0.5 * (np.mean(train_positive >= candidate)
                            + np.mean(train_negative < candidate))
            if longer > best:
                best, threshold, positive_is_longer = longer, float(candidate), True
            shorter = 0.5 * (np.mean(train_positive < candidate)
                             + np.mean(train_negative >= candidate))
            if shorter > best:
                best, threshold, positive_is_longer = shorter, float(candidate), False
        if positive_is_longer:
            scores.append(0.5 * (np.mean(test_positive >= threshold)
                                 + np.mean(test_negative < threshold)))
        else:
            scores.append(0.5 * (np.mean(test_positive < threshold)
                                 + np.mean(test_negative >= threshold)))
    return float(np.mean(scores))


def _bow(text: str) -> list:
    return [token.lower() for token in schema.words(text)]


def bernoulli_nb_oof(texts: list, labels: list, *, folds: int = 5, seed: int = 0,
                     min_support: int = L3_MIN_SUPPORT, alpha: float = 1.0) -> float:
    """Out-of-fold balanced accuracy of a Bernoulli bag-of-words Naive Bayes.

    The vocabulary is every token seen in at least ``min_support`` *training*
    documents of the fold (once per document); unseen tokens are ignored at test
    time.  ``sklearn`` is deliberately not used (it is not installed here).  The
    estimator is the recon's, so its reading is comparable with the recorded
    pre-fix numbers; its known pathology is documented in the module docstring.
    """
    documents = [_bow(text) for text in texts]
    labels = np.asarray(labels, dtype=np.int64)
    rng = random.Random(seed)
    positive = np.where(labels == 1)[0]
    negative = np.where(labels == 0)[0]
    positive_fold = _fold_of(len(positive), folds, rng)
    negative_fold = _fold_of(len(negative), folds, rng)
    fold_of = np.empty(len(texts), dtype=np.int64)
    fold_of[positive] = positive_fold
    fold_of[negative] = negative_fold

    out_of_fold = np.full(len(texts), -1, dtype=np.int64)
    for fold in range(folds):
        train = np.where(fold_of != fold)[0]
        test = np.where(fold_of == fold)[0]
        support: collections.Counter = collections.Counter()
        for index in train:
            support.update(set(documents[index]))
        vocabulary = {
            token: column for column, token in enumerate(
                sorted(token for token, count in support.items() if count >= min_support)
            )
        }
        if not vocabulary:
            continue

        def presence(indices, vocabulary=vocabulary) -> np.ndarray:
            matrix = np.zeros((len(indices), len(vocabulary)), dtype=np.float64)
            for row_index, doc_index in enumerate(indices):
                for token in set(documents[doc_index]):
                    column = vocabulary.get(token)
                    if column is not None:
                        matrix[row_index, column] = 1.0
            return matrix

        train_matrix = presence(train)
        train_labels = labels[train]
        documents_per_class = np.array(
            [np.sum(train_labels == klass) for klass in (0, 1)], dtype=np.float64
        )
        counts = np.zeros((2, len(vocabulary)), dtype=np.float64)
        for klass in (0, 1):
            counts[klass] = train_matrix[train_labels == klass].sum(axis=0)
        probability = (counts + alpha) / (documents_per_class[:, None] + 2.0 * alpha)
        weights = np.log(probability) - np.log1p(-probability)
        bias = np.log1p(-probability).sum(axis=1) + np.log(
            documents_per_class / documents_per_class.sum()
        )
        scores = presence(test) @ weights.T + bias
        out_of_fold[test] = scores.argmax(axis=1)

    if np.any(out_of_fold < 0):
        raise ValueError("some rows were never scored out of fold")
    positives = labels == 1
    negatives = ~positives
    return float(0.5 * (np.mean(out_of_fold[positives] == 1)
                        + np.mean(out_of_fold[negatives] == 0)))


def _l3_corpus(rows: list) -> tuple:
    """``(texts, labels)`` of the two **shipped** sides: labels 1 are the positives.

    The corpus is the artifact as the model would see it -- TreeCut's negatives
    (label 0) against the TreeCut positives that shipped (label 1) -- not the
    build-time pair, because the leak the gate measures is the one the pool carries.
    """
    texts: list = []
    labels: list = []
    for row in rows:
        texts.append(question_or_empty(row))
        labels.append(1 if is_positive(row) else 0)
    return texts, labels


def check_l3_alignment(rows: list, reporter: Reporter) -> None:
    """The 500 shipped pairs must be length-matched (mean |word delta| <= 2)."""
    pairs = shipped_pairs(rows)
    if not pairs:
        reporter.check("L3c the shipped pair is length-matched", False,
                       "the artifact ships no complete pair (both sides)")
        return
    word_delta = []
    char_delta = []
    for negative, positive in pairs.values():
        negative_words = len(schema.words(question_or_empty(negative)))
        positive_words = len(schema.words(question_or_empty(positive)))
        word_delta.append(abs(positive_words - negative_words))
        char_delta.append(len(question_or_empty(positive)) - len(question_or_empty(negative)))
    mean_words = float(np.mean(word_delta))
    mean_chars = float(np.mean(char_delta))
    reporter.note(
        f"L3 alignment: {len(pairs)} shipped pairs, mean |word delta| {mean_words:.3f}, mean "
        f"char delta {mean_chars:+.2f}, negative deleted sentence mean "
        f"{np.mean([len(r['extra_info'].get('deleted_condition_text') or '') for r, _ in pairs.values()]):.2f} "
        f"chars, positive deleted sentence mean "
        f"{np.mean([len(p['extra_info'].get('deleted_condition_text') or '') for _, p in pairs.values()]):.2f} "
        f"chars"
    )
    reporter.check(
        "L3c the shipped pair is length-matched (mean |word-count delta| <= "
        f"{ALIGNMENT_MAX_WORD_DELTA:.0f})",
        mean_words <= ALIGNMENT_MAX_WORD_DELTA,
        f"mean |word delta| {mean_words:.3f} over {len(pairs)} shipped pairs",
    )


def check_l3_gate(rows: list, reporter: Reporter, *, folds: int, seed: int,
                  min_support: int) -> None:
    """L3a/L3b: the two shipped sides must not separate on length or on a bag of words."""
    texts, labels = _l3_corpus(rows)
    positives = sum(labels)
    reporter.note(
        f"L3 corpus (the shipped sides): n={len(texts)} (negatives={len(texts) - positives}, "
        f"positives={positives}), folds={folds}, vocabulary support >= {min_support} docs, "
        f"alpha=1.0"
    )
    if positives == 0 or positives == len(texts):
        reporter.check(
            f"L3a length-only out-of-fold balanced accuracy <= {L3_MAX_BALANCED_ACCURACY:.2f}",
            False,
            "only one label side shipped -- the two-sided interleave is broken, so the "
            "gate cannot be measured",
        )
        return

    lengths = [len(text) for text in texts]
    length_accuracy = length_only_oof(lengths, labels, folds=folds, seed=seed)
    reporter.note(f"L3a length-only out-of-fold balanced accuracy = {length_accuracy:.4f} "
                  f"(chance 0.5)")
    reporter.check(
        f"L3a length-only out-of-fold balanced accuracy <= {L3_MAX_BALANCED_ACCURACY:.2f}",
        length_accuracy <= L3_MAX_BALANCED_ACCURACY,
        f"balanced accuracy {length_accuracy:.4f} on the {len(rows)} shipped rows "
        f"(section 4.6 measured the stock pair at 0.726; section 11 risk 1)",
    )

    accuracy = bernoulli_nb_oof(texts, labels, folds=folds, seed=seed, min_support=min_support)
    control_rng = random.Random(seed + 1)
    control_labels = []
    for label in labels:
        control_labels.append(1 - label if control_rng.random() < 0.5 else label)
    control = bernoulli_nb_oof(texts, control_labels, folds=folds, seed=seed,
                               min_support=min_support)
    reporter.note(f"L3b raw out-of-fold balanced accuracy = {accuracy:.4f} (chance 0.5; "
                  "a sub-0.5 reading is this estimator's known pathology on near-identical "
                  "corpora and is not evidence of anything by itself)")
    reporter.note(f"L3b pair-aligned shuffled-label control = {control:.4f}")
    reporter.check(
        f"L3b Bernoulli bag-of-words out-of-fold balanced accuracy <= "
        f"{L3_MAX_BALANCED_ACCURACY:.2f}",
        accuracy <= L3_MAX_BALANCED_ACCURACY,
        f"balanced accuracy {accuracy:.4f} on the {len(rows)} shipped rows "
        f"(control {control:.4f}; section 4.6 measured the stock pair at 0.755; "
        "section 11 risk 1)",
    )


def pre_fix_reference(raw_dir: str, reporter: Reporter, *, folds: int, seed: int) -> None:
    """Print the released files' own leak as information -- not a gate on this build.

    The released HF mirror is the *stock* generator's output, which is the thing
    section 4.6 measures; the gate above is on the built rows.  Reading it here
    is what keeps the doc's 0.726/0.755 honest: the release's pooled length
    reading is far below those figures, but still above the 0.55 gate, so the doc's
    conclusion (fix before pooling) holds while its exact numbers do not.
    """
    directory = os.path.join(raw_dir, "hf")
    paths = sorted(glob.glob(os.path.join(directory, "*.jsonl")))
    if not paths:
        reporter.note(f"pre-fix reference: no released files under {directory} -- skipped")
        return
    positive: list = []
    negative: list = []
    per_config: dict = collections.defaultdict(lambda: {"pos": [], "neg": []})
    for path in paths:
        name = os.path.basename(path)
        cut_marker = HF_CUT_RE.search(name)
        if not cut_marker:
            continue
        is_cut = cut_marker.group(1) == "True"
        config = HF_CONFIG_RE.search(name)
        key = f"nv{config.group(1)}_ad{config.group(2)}" if config else name
        lengths: list = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                lengths.append(len(json.loads(line).get("problem") or ""))
        if is_cut:
            negative.extend(lengths)
            per_config[key]["neg"].extend(lengths)
        else:
            positive.extend(lengths)
            per_config[key]["pos"].extend(lengths)
    if not positive or not negative:
        reporter.note(f"pre-fix reference: {directory} holds no usable pair -- skipped")
        return
    pooled = length_only_oof(positive + negative, [1] * len(positive) + [0] * len(negative),
                             folds=folds, seed=seed)
    worst = sorted(
        (
            length_only_oof(entry["pos"] + entry["neg"],
                            [1] * len(entry["pos"]) + [0] * len(entry["neg"]),
                            folds=folds, seed=seed),
            key,
        )
        for key, entry in per_config.items() if entry["pos"] and entry["neg"]
    )
    reporter.note(
        f"pre-fix reference (informational, the released stock generator, not this build): "
        f"{len(paths)} files, {len(positive)} answerable / {len(negative)} cut, mean length "
        f"{np.mean(positive):.2f} / {np.mean(negative):.2f}"
    )
    reporter.note(
        f"pre-fix reference: pooled length-only OOF balanced accuracy = {pooled:.4f} "
        f"(section 4.6 / section 11 risk 1 quotes 0.726; the worst per-config reading is {worst[-1][0]:.4f} "
        f"at {worst[-1][1]})"
    )
    recorded = os.path.join(raw_dir, "hf_stats.json")
    if os.path.exists(recorded):
        try:
            with open(recorded, encoding="utf-8") as handle:
                stats = json.load(handle)
            pooled_bow = stats.get("HF_all_rows", {}).get("bow_nb_ba")
            per_config = stats.get("per_config", {})
            worst_bow = max(
                ((entry.get("bow_nb_ba"), key) for key, entry in per_config.items()
                 if entry.get("bow_nb_ba") is not None), default=(None, None)
            )
            reporter.note(
                "pre-fix reference: the recon's recorded pooled BoW NB = "
                f"{pooled_bow} (section 4.6 / section 11 risk 1 quotes 0.755; its multinomial variant "
                f"reads {stats.get('HF_all_rows', {}).get('bow_mnb_ba')}; the worst "
                f"per-config reading it recorded is {worst_bow[0]} at {worst_bow[1]})"
            )
        except (ValueError, OSError) as error:  # pragma: no cover - corrupt recon file
            reporter.note(f"pre-fix reference: {recorded} is unreadable ({error})")


# ---------------------------------------------------------------------------
# contract invariants
# ---------------------------------------------------------------------------


def check_contract(rows: list, reporter: Reporter) -> None:
    """Schema validity plus the per-side contract fields (table B rows 5 and 7)."""
    violations: list = []
    for row in rows:
        violations.extend(schema.validate_row(row))
    reporter.check(
        "schema.validate_row on every row",
        not violations,
        f"{len(rows)} rows, {len(violations)} violations"
        + (f"; first: {violations[:3]}" if violations else ""),
    )

    failures: list = []
    ability, source_label = schema.SOURCES[schema.SOURCE_TREECUT]
    for row in rows:
        info = row["extra_info"]
        task_id = info["task_id"]
        truth = ground_truth_of(row)
        positive = is_positive(row)
        if row.get("data_source") != schema.SOURCE_TREECUT:
            failures.append(f"{task_id}: data_source {row.get('data_source')!r}")
        if info.get("template") != TEMPLATE_A:
            failures.append(f"{task_id}: template {info.get('template')!r}, expected A")
        if info.get("options") is None or len(info.get("options") or []) != K_OPTIONS:
            failures.append(f"{task_id}: {len(info.get('options') or [])} options, expected "
                            f"k={K_OPTIONS}")
        if not _text_of(info.get("paired_original_text")):
            failures.append(f"{task_id}: no counterpart text")
        if not _text_of(info.get("deleted_condition_text")):
            failures.append(f"{task_id}: no deleted sentence")
        if info.get("paired_original_text") == question_or_empty(row):
            failures.append(f"{task_id}: the pair's two members are the same text")
        if info.get("source") != source_label:
            failures.append(f"{task_id}: extra_info.source {info.get('source')!r} "
                            f"(expected {source_label!r})")
        if info.get("domain") != ability:
            failures.append(f"{task_id}: extra_info.domain {info.get('domain')!r} "
                            f"(expected {ability!r})")
        if row.get("ability") != ability:
            failures.append(f"{task_id}: ability {row.get('ability')!r} (expected {ability!r})")

        if positive:
            # Table B row 5: a solvable numeric row with a placeholder block.  The
            # placeholders exist only for the D18 isomorphism, so the gold must be
            # an answer, the diagnosis fields must be empty, and the row must be on
            # the numeric branch.
            if info.get("branch") != BRANCH_POSITIVE or truth.get("solvable") is not True:
                failures.append(f"{task_id}: positive branch/solvable "
                                f"{info.get('branch')!r}/{truth.get('solvable')!r}")
            if not truth.get("answer"):
                failures.append(f"{task_id}: the solvable row carries no answer")
            if truth.get("correct_option_id") is not None or info.get("correct_option_id") not in ("", None):
                failures.append(f"{task_id}: a placeholder row carries a correct option")
            if truth.get("has_diagnosis_label") or info.get("has_diagnosis_label"):
                failures.append(f"{task_id}: a solvable row carries a diagnosis label")
            if truth.get("perturbation_type") is not None or info.get("perturbation_type"):
                failures.append(f"{task_id}: a solvable row carries a perturbation type")
            if info.get("error_type") != "":
                failures.append(f"{task_id}: a solvable row carries error_type "
                                f"{info.get('error_type')!r}")
            if info.get("option_kind") != "placeholder":
                failures.append(f"{task_id}: option_kind {info.get('option_kind')!r}")
            if _text_of(info.get("proof")):
                failures.append(f"{task_id}: the solvable side carries a disproof certificate")
            if str(info.get("answer", "")) != str(truth.get("answer")):
                failures.append(f"{task_id}: extra_info.answer {info.get('answer')!r} is not "
                                f"the gold {truth.get('answer')!r}")
        else:
            # Table B row 7: the four-tier negative (D26).
            if info.get("branch") != BRANCH_DIAG or truth.get("solvable") is not False:
                failures.append(f"{task_id}: negative branch/solvable "
                                f"{info.get('branch')!r}/{truth.get('solvable')!r}")
            if truth.get("answer") is not None:
                failures.append(f"{task_id}: an unsolvable row carries an answer")
            if truth.get("correct_option_id") is None:
                failures.append(f"{task_id}: a four-tier row carries no correct option")
            if not truth.get("has_diagnosis_label") or not info.get("has_diagnosis_label"):
                failures.append(f"{task_id}: a four-tier row carries no diagnosis label")
            if truth.get("perturbation_type") != "missing_condition":
                failures.append(f"{task_id}: perturbation_type {truth.get('perturbation_type')!r}")
            # The gold's perturbation_type is checked above; the *extra_info* copy is
            # what the D18 balance table reads, so a row whose two copies disagree
            # would silently mis-tabulate the defect mix while passing everything else.
            if info.get("perturbation_type") != "missing_condition":
                failures.append(f"{task_id}: extra_info.perturbation_type "
                                f"{info.get('perturbation_type')!r}")
            if info.get("error_type") != "key_information_missing":
                failures.append(f"{task_id}: error_type {info.get('error_type')!r}")
            if info.get("option_kind") != "missing_condition":
                failures.append(f"{task_id}: option_kind {info.get('option_kind')!r}")
            if not _text_of(info.get("proof")):
                failures.append(f"{task_id}: the four-tier negative carries no proof")
    reporter.check(
        "contract: table B rows 5/7 fields, source metadata, pairing fields, option kind",
        not failures,
        f"{len(rows)} rows, {len(failures)} violations"
        + (f"; first: {failures[:3]}" if failures else ""),
    )

    ids = [row["extra_info"]["task_id"] for row in rows]
    bad = [task_id for task_id in ids if not task_id.startswith("treecut-")]
    if len(set(ids)) != len(ids):
        bad.append(f"{len(ids) - len(set(ids))} duplicate task_id(s)")
    reporter.check(
        "contract: task_id is a unique, source-derived string",
        not bad,
        f"{len(ids)} rows, {len(bad)} bad ids" + (f"; first: {bad[:3]}" if bad else ""),
    )

    pairs = shipped_pairs(rows)
    positives = [row for row in rows if is_positive(row)]
    orphans = sorted({task_base(row["extra_info"]["task_id"]) for row in positives} - set(pairs))
    reporter.check(
        "contract: every shipped positive has its negative partner in the artifact",
        not orphans,
        f"{len(pairs)} complete pairs, {len(positives)} positive rows, {len(orphans)} orphan(s)"
        + (f"; first: {orphans[:3]}" if orphans else ""),
    )

    branches = dict(sorted(collections.Counter(
        row["extra_info"]["branch"] for row in rows
    ).items()))
    reporter.note(f"branch -> rows: {branches} (this source fills rows 5 and 7)")


def check_prompt_coherence(rows: list, reporter: Reporter) -> None:
    """The stored prompt must be the recorded template's rendering of the question.

    Every gold in this pool is only meaningful if the prompt asks for it, and the
    prompt is frozen at write time by ``schema.make_row``.  Rebuilding it here with
    the contract's own ``render_prompt`` -- **including the row's own options**, which
    is what template A needs -- binds the wording to the row: a prompt whose tail was
    rewritten (to another template, or with someone else's option block) while the
    gold stayed put, or a ``template`` field that no longer describes the text the
    model actually sees, both fail here and nowhere else.
    """
    failures: list = []
    for row in rows:
        info = row["extra_info"]
        task_id = info.get("task_id", "<missing>")
        content = row["prompt"][0]["content"]
        template = info.get("template")
        try:
            question = question_of(row)
        except ValueError as error:
            failures.append(f"{task_id}: the prompt has no question/template seam "
                            f"({error})")
            continue
        try:
            expected = schema.render_prompt(question, template, options=info.get("options"),
                                            role_words=info.get("role_words"))
        except ValueError as error:
            failures.append(f"{task_id}: recorded template {template!r} cannot "
                            f"render this question ({error})")
            continue
        if content != expected:
            failures.append(f"{task_id}: the prompt is not template {template!r}'s "
                            f"rendering of the row's own question and options")
    reporter.check(
        "prompt: the stored prompt is the recorded template's rendering of question + options",
        not failures,
        f"{len(rows)} prompts rebuilt with schema.render_prompt, {len(failures)} failures"
        + (f"; first: {failures[:3]}" if failures else ""),
    )

    # Per-branch wording: the four-tier negative's prompt must ask for the diagnosis
    # and the solvable row's for an answer, while the two sides share one instruction
    # preamble and one k=3 block shape (the D18/D26 isomorphism).
    failures = []
    preamble = None
    for row in rows:
        info = row["extra_info"]
        task_id = info.get("task_id", "<missing>")
        content = row["prompt"][0]["content"]
        options = info.get("options") or []
        _, sep, tail = content.partition("\n\n")
        if not sep:
            failures.append(f"{task_id}: the prompt has no question/template seam")
            continue
        lines = tail.split("\n")
        block_start = next((index for index, line in enumerate(lines) if line == "选项："), None)
        if block_start is None:
            failures.append(f"{task_id}: the prompt has no 选项： block")
            continue
        head = "\n".join(lines[: block_start + 1])
        if preamble is None:
            preamble = head
        elif head != preamble:
            failures.append(f"{task_id}: the template-A preamble differs from the first row's")
        block = lines[block_start + 1:]
        if len(block) != K_OPTIONS or any(
                not line.startswith(f"{opt['id']}. ") for line, opt in zip(block, options, strict=False)):
            failures.append(f"{task_id}: the rendered option block is not the row's k="
                            f"{K_OPTIONS} options")
        if is_positive(row):
            if "\\boxed{<答案>}" not in content:
                failures.append(f"{task_id}: the solvable prompt does not ask for an answer")
        else:
            if "\\boxed{UNSOLVABLE: <选项ID>}" not in content:
                failures.append(f"{task_id}: the four-tier prompt does not ask for the diagnosis")
    reporter.check(
        "prompt: template A is isomorphic across the two sides (same preamble, k=3 block) "
        "and asks each side for its own gold",
        not failures,
        f"{len(rows)} prompts checked, {len(failures)} failures"
        + (f"; first: {failures[:3]}" if failures else ""),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the TreeCut rows built by treecut_adapter.py.")
    parser.add_argument("--rows", required=True, help="parquet written by treecut_adapter.py")
    parser.add_argument(
        "--raw-dir",
        default=DEFAULT_RAW_DIR,
        help="downloaded TreeCut directory; the vocabulary and the generator are proved here",
    )
    parser.add_argument("--seed", type=int, default=0, help="fold + control + heuristic seed")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-support", type=int, default=L3_MIN_SUPPORT)
    args = parser.parse_args()

    rows = schema.read_parquet_rows(args.rows)
    if not rows:
        print(f"[FAIL] {args.rows} holds no rows")
        raise SystemExit(1)
    negatives = sum(1 for row in rows if not is_positive(row))
    print(f"rows    : {args.rows} ({len(rows)} rows: {negatives} negatives, "
          f"{len(rows) - negatives} positives)")
    print(f"raw dir : {args.raw_dir}")

    try:
        vocabulary = load_vocabulary(args.raw_dir)
        matchers = matchers_for(vocabulary)
    except (OSError, SyntaxError, ValueError, KeyError) as error:
        print(f"cannot read the source's entity tables ({error})")
        vocabulary, matchers = {}, {}

    generator = None
    try:
        generator = load_generator(args.raw_dir)
    except (FileNotFoundError, ImportError, ValueError) as error:
        print(f"cannot import the pinned generator ({error})")

    reward = None
    try:
        reward = load_reward_module()
    except Exception as error:  # noqa: BLE001 - reported as a FAIL below
        print(f"cannot import the frozen reward dispatcher from {REWARD_FILE} ({error})")

    reporter = Reporter()
    check_contract(rows, reporter)
    check_prompt_coherence(rows, reporter)
    check_source_vocabulary(rows, reporter, args.raw_dir, matchers)
    check_source_generator(rows, reporter, args.raw_dir, matchers)
    check_l1_proof(rows, reporter, matchers)
    check_l2_pair(rows, reporter, matchers)
    check_answer_gold(rows, reporter, vocabulary, generator)
    check_option_structure(rows, reporter, matchers)
    check_option_leak(rows, reporter, seed=args.seed)
    if reward is None:
        reporter.check(
            "L4d option-shuffle invariance (reward unchanged when the block is permuted)",
            False, f"cannot import the frozen dispatcher at {REWARD_FILE}",
        )
        reporter.check(
            "L5 bare UNSOLVABLE is branch-dependent (four-tier 0, three-tier +1)",
            False, f"cannot import the frozen dispatcher at {REWARD_FILE}",
        )
        reporter.check(
            "L5b TreeCut positives score on the numeric branch (gold +1, misrefusal 0, "
            "placeholder pick 0)",
            False, f"cannot import the frozen dispatcher at {REWARD_FILE}",
        )
    else:
        check_option_shuffle(rows, reporter, reward, seed=args.seed)
        check_bare_unsolvable(rows, reporter, reward)
        check_positive_reward(rows, reporter, reward)
    check_l3_alignment(rows, reporter)
    check_l3_gate(rows, reporter, folds=args.folds, seed=args.seed,
                  min_support=args.min_support)
    pre_fix_reference(args.raw_dir, reporter, folds=args.folds, seed=args.seed)

    print()
    if reporter.failed:
        print(f"RESULT: FAIL ({len(reporter.failed)}/{len(reporter.results)} checks failed: "
              f"{reporter.failed})")
        raise SystemExit(1)
    print(f"RESULT: PASS ({len(reporter.results)}/{len(reporter.results)} checks passed)")
    raise SystemExit(0)


if __name__ == "__main__":  # pragma: no cover - exercised through main() in tests
    main()
