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
recorded ``deleted_condition_text`` and ``proof``) and from the gold fields, and
never trusts the adapter's own verdict.  Prints ``PASS``/``FAIL`` per check and
exits non-zero if any check fails.

A TreeCut row is a *pair*: the shipped member (an edge the answer's derivation
needs was deleted, so nothing grounds the answer) and the counterpart
(``paired_original_text``: one edge the derivation does **not** use was deleted,
so it stays solvable).  The two members come from one shared render, which is why
the pair is a fair length-matched control and why the certificate below can be
re-derived from either side.

Layers
------

**Prompt binding.**  The stored prompt must be exactly ``schema.render_prompt``'s
output for the row's own question under the row's recorded ``template`` -- the
prompt is frozen at write time, so the wording the model sees can be re-derived
from the row.  This is what ties the gold to the question: a prompt rewritten to
another template while the gold stayed the bare refusal is a row that never asks
for its own answer, and it fails here.

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

**L1 -- proof certificate** (design doc section 9, check 1).  Every row's
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
whose quoted justification had been fabricated or copied from another row.  A
proof that does not quote the clauses its own component rests on is the adapter
inventing its own justification, which is what this check exists to catch.

**L2 -- pair certificate** (design doc section 9, check 3).  Re-derived from the
two texts alone:

* the symmetric difference of the two bodies is exactly one sentence each way;
  the shipped side's is ``deleted_condition_text``, it is absent from the shipped
  body, and the counterpart's own lost sentence is absent from the counterpart;
* the shipped member has **no** fact-to-answer chain (unsolvable), and the
  counterpart has one of exactly ``ansDepth - 1`` hops;
* the counterpart's derivation does not use the sentence the *counterpart* lost
  ("the removed edge is provably irrelevant": the only edge whose removal changes
  solvability is the shipped member's);
* adding the recorded ``deleted_condition_text`` back to the shipped text makes
  the answer reachable at ``ansDepth - 1`` hops -- the pivot proves itself, using
  nothing but the shipped row's own fields;
* the pair is one scenario: same asked variable, same variable set, same sentence
  count (``numVars - 1`` per member), and the same configuration.  ``theme`` is
  checked from the text, ``numVars`` from the variable count, ``ansDepth`` from
  the counterpart's hop count; ``order`` is recorded and cannot be recovered from
  a shuffled body, so it is only checked to be a legal value (see the note in
  ``main()``).

**L3 -- anti-cheat gate** (design doc section 9, check 2).  Section 4.9.2 pit 3
measured the stock generator's pair as leaking: the cut member loses a whole
sentence, so the two classes separate on length and on a bag of words.  This file
re-measures the gate on the **built** rows, as the doc requires:

* a 1-D length threshold, 5-fold out of fold, must read balanced accuracy
  ``<= 0.55`` (random + 5 points);
* a Bernoulli bag-of-words Naive Bayes, same folds, must read ``<= 0.55``;
* the pair must actually be length-matched: mean ``|word-count delta| <= 2``.

Both raw numbers, a pair-aligned shuffled-label null control (the labels are
flipped *within* each pair, so the estimator sees the same pairs with random
sides) and the length-alignment summary are printed pass or fail.  The
Bernoulli reading is reported with its known pathology: on near-identical
corpora it can read *below* chance (as it does here and as UMWP's verifier
records for that corpus), so a sub-0.5 reading is not evidence of anything by
itself -- the control is what makes it interpretable.

What is *not* re-derivable, stated plainly
------------------------------------------

* ``order``: only the shuffle's output survives in the text, so a ``random``
  label cannot be proved from the artifact.  The pair shares it by construction
  (both members are filtered out of one render before the same transform).
* ``proof``'s numeric values are *not* checked against arithmetic: the shipped
  row's gold is the refusal, the counterpart is audit-only, and no per-row source
  record exists to anchor a number.  What is checked is the *shape* (N/M/asked
  variable/clauses), which is this source's whole certificate.
* the released HF files under ``<raw_dir>/hf`` are printed as a pre-fix
  reference for the section 9 gate.  They measure the *source*, not this build,
  and are informational: the doc quotes the stock leak as length 0.726 / BoW NB
  0.755, and the pooled release reads 0.5670 on length (see the adapter's
  DEVIATIONS item 5).  Both exceed the gate, which is why the pair construction
  exists.
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

BRANCH_BARE = schema.BRANCH_UNSOLVABLE_BARE
TEMPLATE_B = schema.TEMPLATE_B

DEFAULT_RAW_DIR = "/home/charles/data/reasoning_rl/halluc/raw/treecut"
REPO_SUBDIR = os.path.join("repo", "treecut")
ENTITIES_FILE = "entities_items.py"

#: Section 9's hard gate: random + 5 points on balanced accuracy.
L3_MAX_BALANCED_ACCURACY = 0.55
L3_MIN_SUPPORT = 5
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
            for key, value in zip(node.keys, node.values)
        }
    raise ValueError(f"unsupported literal node {type(node).__name__} in {ENTITIES_FILE}")


def load_vocabulary(raw_dir: str) -> dict:
    """``{theme: {canonical variable: [surface spellings]}}``, parsed with :mod:`ast`.

    The file is never imported: the adapter imports it (executing the module), so
    reading it as literals is this file's independent route to the same names.
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
        for entity in table["entities"]:
            for item, plural in table["item_dict"].items():
                key = f"{item} at {entity}"
                forms[key] = [key, f"{plural} at {entity}"]
        vocabulary[theme] = forms
    return vocabulary


def compile_matcher(forms: dict) -> tuple:
    """``(regex, surface -> canonical)`` for one theme's variable spellings."""
    surface = [variant for variants in forms.values() for variant in variants]
    surface.sort(key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(spelling) for spelling in surface))
    lookup = {variant: key for key, variants in forms.items() for variant in variants}
    return pattern, lookup


def matchers_for(vocabulary: dict) -> dict:
    return {theme: compile_matcher(forms) for theme, forms in vocabulary.items()}


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

    return {"generate_qa": gen_data.generate_qa, "gen_disproof": gen_questions.gen_disproof}


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
    failures: list = []
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
        failures.extend(_proof_failures(info.get("proof", ""), graph, task_id))
    reporter.check(
        "L1 proof certificate (N variables, M == N - 1 formulas, component, clauses)",
        not failures,
        f"{len(rows)} rows re-derived from their own proof, text and asked variable, "
        f"{len(failures)} failures" + (f"; first: {failures[:3]}" if failures else ""),
    )


# ---------------------------------------------------------------------------
# L2 -- the pair certificate (section 9, check 3)
# ---------------------------------------------------------------------------


def _pair_failures(row: dict, matcher: tuple) -> list:
    """Re-derive the certificate of one row's pair, from the row's own text."""
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
            problems.append(f"{task_id}: deleted_condition_text is not the sentence the "
                            "question lost")
        if deleted in shipped_counts:
            problems.append(f"{task_id}: the deleted sentence is still present in the question")

    shipped_graph = text_graph(shipped_text, matcher)
    counterpart_graph = text_graph(counterpart_text, matcher)
    for label, graph in (("shipped", shipped_graph), ("counterpart", counterpart_graph)):
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
    if _reachable(shipped_graph, asked):
        problems.append(f"{task_id}: the shipped member is solvable -- the answer is reachable "
                        "from a root fact")
    hops, trail = _derivation(counterpart_graph, asked)
    if hops != ans_depth - 1:
        problems.append(f"{task_id}: the counterpart's derivation is {hops} hops, expected "
                        f"ans_depth - 1 ({ans_depth - 1})")
    elif lost_by_counterpart and lost_by_counterpart[0] in trail:
        problems.append(f"{task_id}: the counterpart's derivation uses the sentence the "
                        "counterpart itself lost")
    body, question = split_body(shipped_text)
    repaired = text_graph(f"{body} {deleted} {question}", matcher)
    repaired_hops, _ = _derivation(repaired, asked)
    if repaired_hops != ans_depth - 1:
        problems.append(f"{task_id}: adding the recorded deleted sentence back gives "
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
    reporter.check(
        "L2 pair certificate (1/1 sentence swap, unsolvable, counterpart solvable at ad - 1)",
        not failures,
        f"{len(rows)} pairs re-derived from their own two texts, {len(failures)} failures"
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

        def presence(indices) -> np.ndarray:
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
    """``(texts, labels)`` of the pair corpus: counterparts are the solvable side."""
    texts: list = []
    labels: list = []
    for row in rows:
        texts.append(question_or_empty(row))
        labels.append(0)
        texts.append(_text_of(row["extra_info"].get("paired_original_text")))
        labels.append(1)
    return texts, labels


def check_l3_alignment(rows: list, reporter: Reporter) -> None:
    if not rows:
        reporter.check("L3c the pair is length-matched", False, "no rows to measure")
        return
    word_delta = [
        abs(len(schema.words(row["extra_info"].get("paired_original_text") or ""))
            - len(schema.words(question_or_empty(row))))
        for row in rows
    ]
    char_delta = [
        len(row["extra_info"].get("paired_original_text") or "") - len(question_or_empty(row))
        for row in rows
    ]
    mean_words = float(np.mean(word_delta)) if word_delta else float("nan")
    mean_chars = float(np.mean(char_delta)) if char_delta else float("nan")
    reporter.note(
        f"L3 alignment: {len(rows)} pairs, mean |word delta| {mean_words:.3f}, mean char "
        f"delta {mean_chars:+.2f}, deleted sentence mean "
        f"{np.mean([len(row['extra_info'].get('deleted_condition_text') or '') for row in rows]):.2f} chars"
    )
    reporter.check(
        "L3c the pair is length-matched (mean |word-count delta| <= "
        f"{ALIGNMENT_MAX_WORD_DELTA:.0f})",
        mean_words <= ALIGNMENT_MAX_WORD_DELTA,
        f"mean |word delta| {mean_words:.3f} over {len(rows)} pairs",
    )


def check_l3_gate(rows: list, reporter: Reporter, *, folds: int, seed: int,
                  min_support: int) -> None:
    texts, labels = _l3_corpus(rows)
    positives = sum(labels)
    reporter.note(
        f"L3 corpus: n={len(texts)} (shipped={len(texts) - positives}, counterpart={positives}), "
        f"folds={folds}, vocabulary support >= {min_support} docs, alpha=1.0"
    )
    if positives == 0 or positives == len(texts):
        reporter.check(
            "L3a length-only out-of-fold balanced accuracy <= 0.55",
            False,
            "only one label side present in the artifact -- every row carries both "
            "members, so the artifact is incomplete",
        )
        return

    lengths = [len(text) for text in texts]
    length_accuracy = length_only_oof(lengths, labels, folds=folds, seed=seed)
    reporter.note(f"L3a raw out-of-fold balanced accuracy = {length_accuracy:.4f} (chance 0.5)")
    reporter.check(
        f"L3a length-only out-of-fold balanced accuracy <= {L3_MAX_BALANCED_ACCURACY:.2f}",
        length_accuracy <= L3_MAX_BALANCED_ACCURACY,
        f"balanced accuracy {length_accuracy:.4f} on {len(rows)} pairs "
        f"(section 4.9.2 measured the stock pair at 0.726)",
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
        f"balanced accuracy {accuracy:.4f} on {len(rows)} pairs (control {control:.4f}; "
        "section 4.9.2 measured the stock pair at 0.755)",
    )


def pre_fix_reference(raw_dir: str, reporter: Reporter, *, folds: int, seed: int) -> None:
    """Print the released files' own leak as information -- not a gate on this build.

    The released HF mirror is the *stock* generator's output, which is the thing
    section 4.9.2 measures; the gate above is on the built rows.  Reading it here
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
        f"(section 4.9.1 quotes 0.726; the worst per-config reading is {worst[-1][0]:.4f} "
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
                f"{pooled_bow} (section 4.9.2 quotes 0.755; its multinomial variant "
                f"reads {stats.get('HF_all_rows', {}).get('bow_mnb_ba')}; the worst "
                f"per-config reading it recorded is {worst_bow[0]} at {worst_bow[1]})"
            )
        except (ValueError, OSError) as error:  # pragma: no cover - corrupt recon file
            reporter.note(f"pre-fix reference: {recorded} is unreadable ({error})")


# ---------------------------------------------------------------------------
# contract invariants
# ---------------------------------------------------------------------------


def check_contract(rows: list, reporter: Reporter) -> None:
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
    for row in rows:
        info = row["extra_info"]
        task_id = info["task_id"]
        truth = json.loads(row["reward_model"]["ground_truth"])
        if row.get("data_source") != schema.SOURCE_TREECUT:
            failures.append(f"{task_id}: data_source {row.get('data_source')!r}")
        if info.get("branch") != BRANCH_BARE or info.get("template") != TEMPLATE_B:
            failures.append(f"{task_id}: branch/template {info.get('branch')!r}/"
                            f"{info.get('template')!r}")
        if info.get("solvable") or truth.get("solvable"):
            failures.append(f"{task_id}: this source only ships the unanswerable member")
        if truth.get("answer") is not None:
            failures.append(f"{task_id}: an unsolvable row carries an answer")
        if truth.get("correct_option_id") is not None or info.get("correct_option_id") not in ("", None):
            failures.append(f"{task_id}: a bare row carries a correct option")
        if truth.get("has_diagnosis_label") or info.get("options"):
            failures.append(f"{task_id}: a bare row offers options")
        if truth.get("perturbation_type") != "missing_condition":
            failures.append(f"{task_id}: perturbation_type {truth.get('perturbation_type')!r}")
        if info.get("error_type") != "key_information_missing":
            failures.append(f"{task_id}: error_type {info.get('error_type')!r}")
        # The gold's perturbation_type is checked below; the *extra_info* copy is
        # what the D18 balance table reads, so a row whose two copies disagree
        # would silently mis-tabulate the defect mix while passing everything else.
        if info.get("perturbation_type") != "missing_condition":
            failures.append(f"{task_id}: extra_info.perturbation_type "
                            f"{info.get('perturbation_type')!r}")
        ability, source_label = schema.SOURCES[schema.SOURCE_TREECUT]
        if info.get("source") != source_label:
            failures.append(f"{task_id}: extra_info.source {info.get('source')!r} "
                            f"(expected {source_label!r})")
        if info.get("domain") != ability:
            failures.append(f"{task_id}: extra_info.domain {info.get('domain')!r} "
                            f"(expected {ability!r})")
        if row.get("ability") != ability:
            failures.append(f"{task_id}: ability {row.get('ability')!r} "
                            f"(expected {ability!r})")
        if not _text_of(info.get("paired_original_text")):
            failures.append(f"{task_id}: no counterpart text")
        if not _text_of(info.get("proof")) or not _text_of(info.get("deleted_condition_text")):
            failures.append(f"{task_id}: no proof or no deleted sentence")
        if info.get("paired_original_text") == question_or_empty(row):
            failures.append(f"{task_id}: the pair's two members are the same text")
    reporter.check(
        "contract: bare branch, template B, refusal gold, source metadata, pairing fields",
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

    branches = dict(sorted(collections.Counter(
        row["extra_info"]["branch"] for row in rows
    ).items()))
    reporter.note(f"branch -> rows: {branches} (this source cannot fill the other four)")


def check_prompt_coherence(rows: list, reporter: Reporter) -> None:
    """The stored prompt must be the recorded template's rendering of the question.

    Every gold in this pool is only meaningful if the prompt asks for it, and the
    prompt is frozen at write time by ``schema.make_row``.  Rebuilding it here with
    the contract's own ``render_prompt`` is what binds the wording to the row: a
    prompt whose tail was rewritten (to template A's option block, say) while the
    gold stayed the bare refusal, or a ``template`` field that no longer describes
    the text the model actually sees, both fail here and nowhere else.
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
            expected = schema.render_prompt(question, template)
        except ValueError as error:
            failures.append(f"{task_id}: recorded template {template!r} cannot "
                            f"render this question ({error})")
            continue
        if content != expected:
            failures.append(f"{task_id}: the prompt is not template {template!r}'s "
                            f"rendering of the row's own question")
    reporter.check(
        "prompt: the stored prompt is the recorded template's rendering of the question",
        not failures,
        f"{len(rows)} prompts rebuilt with schema.render_prompt, {len(failures)} failures"
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
    parser.add_argument("--seed", type=int, default=0, help="fold + control seed")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-support", type=int, default=L3_MIN_SUPPORT)
    args = parser.parse_args()

    rows = schema.read_parquet_rows(args.rows)
    if not rows:
        print(f"[FAIL] {args.rows} holds no rows")
        raise SystemExit(1)
    print(f"rows    : {args.rows} ({len(rows)} rows)")
    print(f"raw dir : {args.raw_dir}")

    try:
        vocabulary = load_vocabulary(args.raw_dir)
        matchers = matchers_for(vocabulary)
    except (OSError, SyntaxError, ValueError, KeyError) as error:
        print(f"cannot read the source's entity tables ({error})")
        matchers = {}

    reporter = Reporter()
    check_contract(rows, reporter)
    check_prompt_coherence(rows, reporter)
    check_source_vocabulary(rows, reporter, args.raw_dir, matchers)
    check_source_generator(rows, reporter, args.raw_dir, matchers)
    check_l1_proof(rows, reporter, matchers)
    check_l2_pair(rows, reporter, matchers)
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
