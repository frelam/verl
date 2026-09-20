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
"""Audit the K&K rows produced by ``kk_adapter.py``.

Usage::

    python verify_kk.py --rows /tmp/kk.parquet --raw-dir ~/data/reasoning_rl/halluc/raw/kk

Reads the artifact plus the raw parquet files the artifact's ``task_id`` points at,
and re-derives every gold with the verifier's **own** enumerator, statement parser
and sentence builders -- no certificate below is computed by ``kk_adapter`` code.
The adapter symbols still imported are the ``task_id`` / file-name grammar and the
family vocabulary, i.e. how to *locate* a raw row, never how to certify one; a
mistake there makes the audit fail loudly, not pass quietly.  Prints
``PASS``/``FAIL`` per check and exits non-zero if any hard check fails.

Layers
------

**Prompt contract** (every row).  The row is only usable if the model is asked
what the gold answers, so the stored prompt must be *exactly* the contract's own
render -- ``schema.render_prompt(question, B, role_words)`` -- with the question
being the source's own ``quiz``.  ``extra_info.template == "B"`` alone proves
nothing: deleting the K&K role instruction, swapping in the ``B_judge`` wording or
breaking the ``\\boxed`` markers all survive a label check and all destroy the row.

**L1 -- gold certificate** (a sample of >= 50 rows).  The artifact carries neither
``statements`` nor ``solution_text``, so the gold can only be proved against the
raw row: the verifier re-parses the row's own ``statements`` (a Python tuple, not
JSON), brute-forces all ``2**N`` assignments with its own evaluator, and demands

* **exactly one** solution -- 0 or >= 2 is a fail, not a guess;
* that solution is the source's own ``solution`` field;
* that solution is the artifact's own ``extra_info.canonical_solution``;
* the artifact's ``ground_truth.answer`` is the **name -> surface role word
  mapping** of D19, built from the source's ``names`` + that solution + the
  artifact's own ``role_words`` -- per inhabitant, keys and values both.

**Sentence oracles** (every row).  The two sentences the dataset ships --
``solution_text`` and ``solution_text_format`` -- are rebuilt from the artifact's
own ``canonical_solution`` + the raw row's ``names``/``knight_knave`` and must come
out byte-identical.  They are what proves ``knight_knave['knight']`` is the
truth-teller word on ``flip_role``, whose statements are byte-identical to the
clean row but whose sentence is flipped, and they run over **every** row rather
than the L1 sample.

**Source anchor.**  Beyond the sample, every row is re-read against the raw files:
the question in the prompt must be the source's own ``quiz`` for that
``(family, len(names), index)``, ``role_words`` must be that row's
``knight_knave`` pair, ``extra_info.canonical_solution`` must be the source's own
``solution``, the source's own ``statements`` must enumerate to exactly that one
solution, the stored gold must be that D19 mapping, and -- D20 -- the row must be a
*perturbed* variant whose ``paired_original_text`` is the clean member of the same
group.  Without the raw files nothing here can be proved, so a missing ``--raw-dir``
is a FAIL, not a skip.

**L2 -- gold uniqueness.**  ``(len(names), index)`` is the group key of design doc
section 4.1: the enumerator must find one solution (the puzzle is uniquely
determined), and the artifact must hold **at most one** row per group -- the
variants of one abstract problem are sevenfold restatements, and D20 admits only
the perturbed ones, so two of them in the pool is a leak even before they can
straddle a split.  The key is re-derived from ``task_id`` alone, because
``mix_halluc`` overwrites ``extra_info.index`` downstream.

**L3 -- anti-cheat.**  The design doc defines no L3 for K&K (no options block, no
binary label, so the BoW-NB yardstick of the other sources does not exist).  The
gold is a per-inhabitant mapping (D19), so the analogous floor is scored **per
person**:

* **a. no inverted gold** -- every row's mapping is re-derived from the source's
  *formula* under the artifact's own ``role_words`` and compared inhabitant by
  inhabitant.  This is design risk 5 as a check: a canonical-code gold, one built
  from the wrong side of ``flip_role``, a bare sequence or a wrong name set lands
  here.  It runs over **every** row, not a sample, and enumerates the puzzle rather
  than trusting the source's stored ``solution`` field.
* **b. constant per-person label assignments** -- "every inhabitant is the truth
  word" and "every inhabitant is the lie word", scored against the gold mapping.
  Their whole-mapping hit rate must stay at/below the ``2**-L`` floor bound, so a
  prompt whose answer is always "everyone tells the truth" cannot pass.
* **c. per-position majority** -- the strongest roster-position guess inside each
  size tier, again scored per person.  Its whole-mapping rate must stay at/below
  the same bound, and its per-person rate must not beat the better constant
  assignment by more than ``L3_CONSTANT_MARGIN`` (5pt): roster order may not leak
  more than the class prior.

The measured numbers are printed pass or fail, so the report carries the actual
rate rather than a verdict.
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import os
import random
import sys
from pathlib import Path

try:
    import schema
except ImportError:  # pragma: no cover - running as a plain script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import schema

try:
    import kk_adapter
except ImportError:  # pragma: no cover - running as a plain script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import kk_adapter

BRANCH_ROLES = schema.BRANCH_SOLVABLE_ROLES
DEFAULT_RAW_DIR = kk_adapter.DEFAULT_RAW_DIR
CANONICAL_ROLE_WORDS = kk_adapter.CANONICAL_ROLE_WORDS

L1_MIN_SAMPLE = 50
L3_MAX_HEURISTIC_ACCURACY = 0.25
#: How much a position-wise guess may beat the better constant (class-prior)
#: assignment per person before it counts as a leak.  Design doc Q8 fixes the
#: analogous L3 budget for the sources that have one at "random + 10pt"; 5pt is the
#: stricter reading of section 9's "near the floor" and still clears the measured
#: corpus (best constant 0.5448, per-position majority 0.5484).
L3_CONSTANT_MARGIN = 0.05
#: A majority sequence over a handful of rows is noise, not a heuristic a model
#: could exploit; a tier is only measured once it has this many rows.  Measured:
#: every tier in the shipped pool has 1,000 rows, so this bound never bites there.
L3_MIN_TIER_ROWS = 20


class Reporter:
    """Collects PASS/FAIL lines; the exit code is ``all(ok)``."""

    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str) -> bool:
        self.results.append((name, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        return bool(ok)

    def note(self, message: str) -> None:
        print(f"       {message}")

    @property
    def failed(self) -> list[str]:
        return [name for name, ok, _ in self.results if not ok]


# ---------------------------------------------------------------------------
# reading the artifact
# ---------------------------------------------------------------------------


def question_of(row: dict) -> str:
    """The question the model sees: the prompt is ``question + "\\n\\n" + template``.

    A prompt with no blank line means the adapter stopped rendering the template
    the way ``schema.render_prompt`` does, and every text check below would
    silently read the wrong string, so it is a hard error.
    """
    content = row["prompt"][0]["content"]
    head, sep, _ = content.partition("\n\n")
    if not sep:
        raise ValueError(f"prompt of {row['extra_info'].get('task_id')} has no template separator")
    return head


def payload_of(row: dict) -> dict:
    return json.loads(row["reward_model"]["ground_truth"])


def normalise(text: str) -> str:
    return " ".join((text or "").split())


# ---------------------------------------------------------------------------
# the verifier's own certificate machinery
# ---------------------------------------------------------------------------
# None of this calls ``kk_adapter``.  A verifier that re-derives the gold with the
# code that produced it proves only that the code is self-consistent -- and a
# mirrored bug is precisely design risk 5 (the surface/canonical role-word trap).
# Everything the audit asserts about a gold is computed below, from the raw row and
# the artifact, by this file.

_LEAF_KINDS = ("telling-truth", "lying")
_OPERATORS = ("and", "or", "->", "<=>")
_MAX_DEPTH = 64


class BadStatement(ValueError):
    """A ``statements`` payload that is not a legal K&K formula tree."""


def check_node(node: object, count: int, depth: int = 0) -> None:
    """Reject anything that is not a legal K&K formula over ``count`` people."""
    if depth > _MAX_DEPTH:
        raise BadStatement(f"statement tree deeper than {_MAX_DEPTH}")
    if not isinstance(node, tuple) or not node:
        raise BadStatement(f"statement node must be a non-empty tuple, got {node!r}")
    kind = node[0]
    if kind in _LEAF_KINDS:
        if len(node) != 2:
            raise BadStatement(f"{kind!r} takes one person index, got {node!r}")
        index = node[1]
        if isinstance(index, bool) or not isinstance(index, int):
            raise BadStatement(f"{kind!r} index must be an int, got {index!r}")
        if not 0 <= index < count:
            raise BadStatement(f"{kind!r} index {index} outside 0..{count - 1}")
        return
    if kind == "not":
        if len(node) != 2:
            raise BadStatement(f"'not' takes one operand, got {node!r}")
        check_node(node[1], count, depth + 1)
        return
    if kind in _OPERATORS:
        if len(node) != 3:
            raise BadStatement(f"{kind!r} takes two operands, got {node!r}")
        check_node(node[1], count, depth + 1)
        check_node(node[2], count, depth + 1)
        return
    raise BadStatement(f"unknown statement operator {kind!r}")


def parse_own_statements(text: str, count: int) -> tuple:
    """``statements`` (the repr of a Python tuple, not JSON) -> validated formulas."""
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError) as exc:
        raise BadStatement(f"statements is not a Python literal: {exc}") from exc
    if not isinstance(parsed, tuple) or not parsed:
        raise BadStatement("statements must be a non-empty tuple of formulas")
    if len(parsed) != count:
        raise BadStatement(f"{len(parsed)} statements for {count} inhabitants")
    for node in parsed:
        check_node(node, count)
    return parsed


def evaluate_own(node: tuple, truth: tuple[bool, ...]) -> bool:
    """Truth value of one (already validated) formula under ``truth``."""
    kind = node[0]
    if kind == "telling-truth":
        return bool(truth[node[1]])
    if kind == "lying":
        return not truth[node[1]]
    if kind == "not":
        return not evaluate_own(node[1], truth)
    if kind == "and":
        return evaluate_own(node[1], truth) and evaluate_own(node[2], truth)
    if kind == "or":
        return evaluate_own(node[1], truth) or evaluate_own(node[2], truth)
    if kind == "->":
        return (not evaluate_own(node[1], truth)) or evaluate_own(node[2], truth)
    if kind == "<=>":
        return evaluate_own(node[1], truth) == evaluate_own(node[2], truth)
    raise BadStatement(f"unknown statement operator {kind!r}")


def solutions_own(statements: tuple, count: int) -> list[tuple[bool, ...]]:
    """Every assignment satisfying the puzzle, by brute force over ``2**count``.

    Person ``i`` asserts their own formula, so the constraint is
    ``truth[i] == evaluate(statements[i], truth)`` for every ``i``.
    """
    found: list[tuple[bool, ...]] = []
    for mask in range(1 << count):
        truth = tuple(bool(mask >> i & 1) for i in range(count))
        if all(evaluate_own(stmt, truth) == truth[i] for i, stmt in enumerate(statements)):
            found.append(truth)
    return found


def role_words_own(row: dict) -> tuple[str, str] | None:
    """``(truth-teller word, liar word)`` as the raw row spells them, or ``None``."""
    roles = row.get("knight_knave")
    if not isinstance(roles, dict):
        return None
    truth_word = roles.get("knight")
    lie_word = roles.get("knave")
    if not isinstance(truth_word, str) or not isinstance(lie_word, str):
        return None
    truth_word, lie_word = truth_word.strip(), lie_word.strip()
    if not truth_word or not lie_word:
        return None
    return truth_word, lie_word


def answer_mapping_own(names: list[str], solution: list[bool], role_words: list[str]) -> dict[str, str]:
    """The D19 gold: ``{name: surface role word}``, one entry per inhabitant.

    Index ``i`` pairs ``names[i]`` with the truth-teller word when ``solution[i]``
    is True and with the liar word otherwise.  Recomputed here from the raw row, so
    it disagrees with the artifact whenever the artifact used canonical words, the
    wrong side of a flipped mapping, or a name set that is not the source's.
    """
    truth_word, lie_word = role_words
    return {name: (truth_word if flag else lie_word) for name, flag in zip(names, solution, strict=False)}


def mapping_diff(actual: object, expected: dict[str, str]) -> list[str]:
    """The inhabitants on which an answer mapping disagrees with the expected one."""
    if not isinstance(actual, dict):
        return ["<not a name -> role mapping>"]
    shared = sorted(set(actual) & set(expected))
    diff = [f"{name}:{actual[name]!r} != {expected[name]!r}" for name in shared if actual[name] != expected[name]]
    diff += [f"missing {name!r}" for name in sorted(set(expected) - set(actual))]
    diff += [f"unexpected {name!r}" for name in sorted(set(actual) - set(expected))]
    return diff


def solution_text_own(row: dict, witness: list[bool]) -> str:
    """Rebuild the source's own ``solution_text`` sentence.

    This is the independent half of the surface-word oracle: it reads
    ``knight_knave['a_knight']``/``['a_knave']`` straight out of the raw row, so it
    disagrees with the artifact whenever the gold was built from the wrong side of
    the mapping -- exactly what ``flip_role`` does to a canonical-word gold.
    """
    roles = row["knight_knave"]
    parts = [
        f"{name} is {roles['a_knight'] if flag else roles['a_knave']}"
        for name, flag in zip(row["names"], witness)
    ]
    if len(parts) == 1:
        return parts[0] + "."
    return ", and ".join([", ".join(parts[:-1]), parts[-1]]) + "."


def solution_text_format_own(row: dict, witness: list[bool]) -> str:
    """Rebuild the source's own ``solution_text_format`` block (one line per person)."""
    roles = row["knight_knave"]
    return "\n".join(
        f"({rank}) {name} is {roles['a_knight'] if flag else roles['a_knave']}"
        for rank, (name, flag) in enumerate(zip(row["names"], witness), start=1)
    )


# ---------------------------------------------------------------------------
# the raw index
# ---------------------------------------------------------------------------


def load_raw_index(raw_dir: str) -> dict[tuple[str, int, int], dict]:
    """``(family, len(names), index) -> raw row`` over every ``*.parquet``.

    The adapter's own ``load_source`` is deliberately not reused: the audit must
    re-read the files itself, and it needs no ``_``-prefixed plumbing.  An empty
    index means the raw files are unreachable, which the caller turns into a FAIL.
    """
    import pyarrow.parquet as pq

    root = Path(raw_dir)
    if not root.is_dir():
        return {}
    index: dict[tuple[str, int, int], dict] = {}
    for path in sorted(root.glob("*.parquet")):
        parsed = kk_adapter.parse_file_stem(path.stem)
        if parsed is None:
            continue
        _, _, family = parsed
        for row in pq.read_table(path).to_pylist():
            names = row.get("names")
            if not isinstance(names, list) or not names:
                continue
            index[(family, len(names), int(row["index"]))] = row
    return index


def clean_quiz(index: dict[tuple[str, int, int], dict], count: int, raw_index: int) -> str | None:
    """The clean member's quiz for the group ``(count, raw_index)``, or ``None``."""
    row = index.get((kk_adapter.FAMILY_CLEAN, count, raw_index))
    return None if row is None else normalise(row["quiz"])


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
    for row in rows:
        info = row["extra_info"]
        task_id = info["task_id"]
        parsed = kk_adapter.parse_task_id(task_id)
        if parsed is None:
            failures.append(f"{task_id}: task_id is not kk:{{family}}:{{N}}ppl:{{index}}")
            continue
        family, count, raw_index = parsed
        payload = payload_of(row)
        if info.get("branch") != BRANCH_ROLES:
            failures.append(f"{task_id}: branch {info.get('branch')!r}")
        if info.get("template") != schema.TEMPLATE_B:
            failures.append(f"{task_id}: template {info.get('template')!r} (K&K answers, it never judges)")
        if info.get("solvable") is not True or payload.get("solvable") is not True:
            failures.append(f"{task_id}: not solvable")
        if info.get("options"):
            failures.append(f"{task_id}: carries an options block")
        if info.get("has_diagnosis_label") or payload.get("has_diagnosis_label"):
            failures.append(f"{task_id}: carries a diagnosis label")
        if payload.get("correct_option_id") is not None:
            failures.append(f"{task_id}: carries correct_option_id")
        if payload.get("perturbation_type") is not None:
            failures.append(f"{task_id}: perturbation_type must be None for K&K")
        if payload.get("judgment_only"):
            failures.append(f"{task_id}: judgment_only on an answer branch")
        if info.get("perturbation_family") != family:
            failures.append(f"{task_id}: perturbation_family {info.get('perturbation_family')!r} != task_id")
        if family == kk_adapter.FAMILY_CLEAN:
            failures.append(f"{task_id}: a clean row is in the pool (D20 ships perturbed rows only)")
        if info.get("difficulty") != f"{count}ppl":
            failures.append(f"{task_id}: difficulty {info.get('difficulty')!r} != {count}ppl")
        if info.get("index") != raw_index and info.get("index") is not None:
            failures.append(f"{task_id}: index {info.get('index')!r} != task_id index {raw_index}")
        words = info.get("role_words") or []
        if len(words) != 2 or not all(isinstance(word, str) and word for word in words):
            failures.append(f"{task_id}: role_words {words!r}")
        elif words[0].casefold() == words[1].casefold():
            failures.append(f"{task_id}: role_words are the same word")
        if payload.get("role_words") != words:
            failures.append(f"{task_id}: ground_truth.role_words != extra_info.role_words")
        answer = payload.get("answer")
        if not isinstance(answer, dict) or not answer:
            # D19: the gold is a per-inhabitant mapping; a bare sequence or a coded
            # string is the superseded format and must fail here.
            failures.append(f"{task_id}: answer is not a non-empty name -> role mapping")
        elif any(not isinstance(name, str) or not name.strip() for name in answer):
            failures.append(f"{task_id}: answer mapping has an empty name")
        elif len(answer) != count:
            failures.append(f"{task_id}: answer mapping has {len(answer)} entries, expected {count}")
        elif set(answer.values()) - set(words):
            failures.append(f"{task_id}: answer uses words outside role_words")
        else:
            try:
                roster = names_in_question(question_of(row))
            except ValueError as exc:
                failures.append(f"{task_id}: {exc}")
            else:
                if set(answer) != set(roster):
                    failures.append(f"{task_id}: answer names {sorted(answer)} != the question's roster {roster}")
        prompt = row["prompt"][0]["content"]
        for word in words:
            if word not in prompt:
                failures.append(f"{task_id}: role word {word!r} does not appear in the prompt")
    reporter.check(
        "branch invariants (branch/template/solvable/answer/role_words)",
        not failures,
        f"{len(rows)} rows, {len(failures)} violations" + (f"; first: {failures[:3]}" if failures else ""),
    )

    ids = [row["extra_info"]["task_id"] for row in rows]
    duplicates = [task_id for task_id, n in collections.Counter(ids).items() if n > 1]
    reporter.check(
        "task_id is unique",
        not duplicates,
        f"{len(rows)} rows, {len(duplicates)} duplicated ids" + (f"; first: {duplicates[:3]}" if duplicates else ""),
    )


# ---------------------------------------------------------------------------
# the prompt contract
# ---------------------------------------------------------------------------


def check_prompt_contract(rows: list[dict], reporter: Reporter) -> None:
    """The whole prompt equals ``schema.render_prompt(question, B, role_words)``.

    Every other check reads ``question_of``, which stops at the blank line -- so a
    template the adapter dropped, swapped for the judging variant, or appended a
    contradictory role instruction to, is invisible to all of them.  This compares
    the full string the model will actually read.
    """
    failures: list[str] = []
    for row in rows:
        info = row["extra_info"]
        task_id = info["task_id"]
        try:
            question = question_of(row)
        except ValueError as exc:
            failures.append(str(exc))
            continue
        try:
            expected = schema.render_prompt(question, schema.TEMPLATE_B, role_words=info["role_words"])
        except Exception as exc:  # noqa: BLE001 - report any render failure as a failure
            failures.append(f"{task_id}: schema.render_prompt raised {exc!r}")
            continue
        if row["prompt"][0]["content"] != expected:
            failures.append(f"{task_id}: prompt is not schema.render_prompt(question, B, role_words)")
    reporter.check(
        "prompt contract (full prompt == schema.render_prompt(question, B, role_words))",
        not failures,
        f"{len(rows)} rows, {len(failures)} violations" + (f"; first: {failures[:3]}" if failures else ""),
    )


# ---------------------------------------------------------------------------
# the source anchor
# ---------------------------------------------------------------------------


def check_source_anchor(rows: list[dict], index: dict, raw_dir: str, reporter: Reporter) -> None:
    """Prove every row's text, role words, gold and pairing against the raw files.

    The artifact-local checks cannot tell an honest adapter from one that invented
    a question or a gold.  This one re-reads the row ``task_id`` names and demands
    that the question, the two role words, the family, the tier and the group's
    clean sibling are all the source's own.
    """
    if not index:
        reporter.check(
            "source anchor (texts and golds re-read from the raw files)",
            False,
            f"cannot read any parquet under {raw_dir} -- the gold cannot be proved without the "
            "source; re-run with --raw-dir pointing at the K&K directory",
        )
        return

    failures: list[str] = []
    for row in rows:
        info = row["extra_info"]
        task_id = info["task_id"]
        parsed = kk_adapter.parse_task_id(task_id)
        if parsed is None:
            failures.append(f"{task_id}: unparseable task_id")
            continue
        family, count, raw_index = parsed
        source = index.get((family, count, raw_index))
        if source is None:
            failures.append(f"{task_id}: ({family}, {count}, {raw_index}) is not a source row")
            continue
        if question_of(row) != normalise(source["quiz"]):
            failures.append(f"{task_id}: presented question differs from the source's")
            continue
        words = role_words_own(source)
        if words is None or list(words) != list(info["role_words"]):
            failures.append(f"{task_id}: role_words {info['role_words']!r} != source {words!r}")
            continue
        expected = answer_mapping_own(
            list(source["names"]), [bool(f) for f in source["solution"]], list(words)
        )
        if payload_of(row).get("answer") != expected:
            diff = mapping_diff(payload_of(row).get("answer"), expected)[:3]
            failures.append(f"{task_id}: gold differs from the D19 mapping on {diff}")
            continue
        if [bool(f) for f in source["solution"]] != [bool(f) for f in info["canonical_solution"]]:
            failures.append(f"{task_id}: canonical_solution differs from the source's solution")
            continue
        # The gold must also follow from the puzzle itself, not merely agree with the
        # row's own stored ``solution`` field -- otherwise a source whose formula and
        # solution disagree would anchor "correctly" to a puzzle no one can solve.
        try:
            statements = parse_own_statements(source["statements"], count)
        except (BadStatement, KeyError) as exc:
            failures.append(f"{task_id}: source statements are not a legal formula: {exc}")
            continue
        solutions = solutions_own(statements, count)
        if len(solutions) != 1:
            failures.append(f"{task_id}: the source puzzle has {len(solutions)} solutions, not 1")
            continue
        if list(solutions[0]) != [bool(f) for f in source["solution"]]:
            failures.append(f"{task_id}: enumerating the source formula contradicts its stored solution")
            continue
        if family == kk_adapter.FAMILY_CLEAN:
            failures.append(f"{task_id}: a clean row is in the pool (D20 emits perturbed rows only)")
            continue
        sibling = clean_quiz(index, count, raw_index)
        if sibling is None:
            failures.append(f"{task_id}: no clean member for this group in the source")
        elif normalise(info.get("paired_original_text", "")) != sibling:
            failures.append(f"{task_id}: paired_original_text is not the group's clean quiz")
    reporter.check(
        "source anchor (texts, role words, gold and pairing re-read from the raw files)",
        not failures,
        f"{len(rows)} rows anchored, {len(failures)} failures" + (f"; first: {failures[:3]}" if failures else ""),
    )


# ---------------------------------------------------------------------------
# L1 / L2
# ---------------------------------------------------------------------------


def check_l1(sampled: list[dict], index: dict, reporter: Reporter) -> None:
    """Re-derive every sampled gold with the verifier's own enumerator."""
    failures: list[str] = []
    zero = multi = mismatch = 0
    for row in sampled:
        info = row["extra_info"]
        task_id = info["task_id"]
        parsed = kk_adapter.parse_task_id(task_id)
        if parsed is None:
            failures.append(f"{task_id}: unparseable task_id")
            continue
        family, count, raw_index = parsed
        source = index.get((family, count, raw_index))
        if source is None:
            failures.append(f"{task_id}: not found in the raw files")
            continue
        try:
            statements = parse_own_statements(source["statements"], count)
        except BadStatement as exc:
            failures.append(f"{task_id}: {exc}")
            continue
        solutions = solutions_own(statements, count)
        if len(solutions) == 0:
            zero += 1
            failures.append(f"{task_id}: the source puzzle has no solution")
            continue
        if len(solutions) > 1:
            multi += 1
            failures.append(f"{task_id}: the source puzzle has {len(solutions)} solutions")
            continue
        witness = [bool(flag) for flag in solutions[0]]
        if witness != [bool(flag) for flag in source["solution"]]:
            mismatch += 1
            failures.append(f"{task_id}: enumeration != the source's own solution field")
            continue
        if witness != [bool(flag) for flag in info["canonical_solution"]]:
            failures.append(f"{task_id}: artifact canonical_solution != the enumerated solution")
            continue
        expected = answer_mapping_own(list(source["names"]), witness, list(info["role_words"]))
        if payload_of(row).get("answer") != expected:
            diff = mapping_diff(payload_of(row).get("answer"), expected)[:3]
            failures.append(f"{task_id}: gold differs from the D19 mapping on {diff}")
    detail = (
        f"{len(sampled)} sampled rows re-derived (0-solution {zero}, multi-solution {multi}, "
        f"field mismatch {mismatch}), {len(failures)} failures"
    )
    if failures:
        detail += "; first: " + " | ".join(failures[:3])
    reporter.check(
        "L1 gold certificate (independent enumerator, source solution, D19 mapping)", not failures, detail
    )


def names_in_question(question: str) -> list[str]:
    """The roster the question names, in the order the question names it.

    Both the gold mapping and the sentence oracles are ordered by the roster, so a
    question that lists the inhabitants differently would make every one of them read
    against the wrong person.
    """
    head, marker, roster = question.partition("inhabitants:")
    if not marker:
        raise ValueError("question does not name a roster")
    roster = roster.split(".", 1)[0]
    return [name.strip() for name in roster.replace(", and ", ",").replace(" and ", ",").split(",") if name.strip()]


def check_sentence_oracles(rows: list[dict], index: dict, reporter: Reporter) -> None:
    """Re-derive the source's two solution sentences from the artifact, every row.

    These are the strings the reward matches the model's answer against, so they must
    agree with the source's own ``solution_text`` / ``solution_text_format`` for the
    canonical_solution the artifact stores -- if the artifact wrote a canonical-word gold under
    a ``flip_role`` mapping, the sentences built from the *source's* words will not
    match the ones built from the artifact's.
    """
    if not index:
        reporter.check(
            "sentence oracles (both source sentences rebuilt for every row)",
            False,
            "the raw files are unreadable, so the sentences cannot be rebuilt",
        )
        return
    failures: list[str] = []
    matches = 0
    for row in rows:
        info = row["extra_info"]
        task_id = info["task_id"]
        parsed = kk_adapter.parse_task_id(task_id)
        if parsed is None:
            failures.append(f"{task_id}: unparseable task_id")
            continue
        family, count, raw_index = parsed
        source = index.get((family, count, raw_index))
        if source is None:
            failures.append(f"{task_id}: not found in the raw files")
            continue
        if role_words_own(source) != tuple(info["role_words"]):
            failures.append(f"{task_id}: the artifact's role words are not the source's")
            continue
        # Both sentences and the gold are ordered by the roster, so a question that
        # lists the inhabitants in another order silently grades every one of them
        # against the wrong person.
        try:
            roster = names_in_question(question_of(row))
        except ValueError as exc:
            failures.append(f"{task_id}: {exc}")
            continue
        if roster != list(source["names"]):
            failures.append(f"{task_id}: the question's roster {roster} is not the source's {list(source['names'])}")
            continue
        try:
            text = solution_text_own(source, info["canonical_solution"])
            text_format = solution_text_format_own(source, info["canonical_solution"])
        except (KeyError, TypeError) as exc:
            failures.append(f"{task_id}: cannot rebuild the source sentences: {exc!r}")
            continue
        if normalise(text) != normalise(source.get("solution_text", "")):
            failures.append(
                f"{task_id}: gold sentence {normalise(text)!r} != source {normalise(source.get('solution_text', ''))!r}"
            )
            continue
        if text_format != source.get("solution_text_format", ""):
            failures.append(f"{task_id}: solution_text_format rebuilt from the artifact differs from the source's")
            continue
        matches += 1
    detail = f"{matches}/{len(rows)} rows reproduce both source sentences, {len(failures)} failures"
    if failures:
        detail += "; first: " + " | ".join(failures[:3])
    reporter.check("sentence oracles (both source sentences rebuilt for every row)", not failures, detail)


def check_l2(rows: list[dict], sampled: list[dict], index: dict, reporter: Reporter) -> None:
    """Uniqueness: one solution per puzzle, one row per abstract problem."""
    failures: list[str] = []
    if not index:
        reporter.check(
            "L2 gold uniqueness (unique solution per puzzle, <= 1 variant per group)",
            False,
            "the raw files are unreadable, so the per-puzzle uniqueness half cannot be proved",
        )
        return
    for row in sampled:
        info = row["extra_info"]
        task_id = info["task_id"]
        parsed = kk_adapter.parse_task_id(task_id)
        if parsed is None:
            continue
        family, count, raw_index = parsed
        source = index.get((family, count, raw_index))
        if source is None:
            continue
        try:
            statements = parse_own_statements(source["statements"], count)
        except BadStatement:
            continue
        if len(solutions_own(statements, count)) != 1:
            failures.append(f"{task_id}: the source puzzle is not uniquely determined")

    groups: dict[tuple[int, int], list[str]] = collections.defaultdict(list)
    for row in rows:
        parsed = kk_adapter.parse_task_id(row["extra_info"]["task_id"])
        if parsed is None:
            continue
        _, count, raw_index = parsed
        groups[(count, raw_index)].append(row["extra_info"]["perturbation_family"])
    duplicated = {key: families for key, families in groups.items() if len(families) > 1}
    failures.extend(
        f"group {count}ppl/{raw_index} holds {sorted(families)}"
        for (count, raw_index), families in list(sorted(duplicated.items()))[:3]
    )
    reporter.check(
        "L2 gold uniqueness (unique solution per puzzle, <= 1 variant per group)",
        not failures,
        f"{len(sampled)} sampled puzzles enumerated, {len(groups)} groups, "
        f"{len(duplicated)} groups with more than one variant, {len(failures)} failures"
        + (f"; first: {failures[:3]}" if failures else ""),
    )


# ---------------------------------------------------------------------------
# L3
# ---------------------------------------------------------------------------


def _role_flag(role: object, role_words: list[str]) -> bool | None:
    """One surface role word -> canonical truth-teller bool for that row, or None."""
    value = " ".join(str(role).casefold().split())
    truth_word = " ".join(str(role_words[0]).casefold().split())
    lie_word = " ".join(str(role_words[1]).casefold().split())
    if value == truth_word:
        return True
    if value == lie_word:
        return False
    return None


def gold_vector(row: dict) -> list[bool] | None:
    """The artifact's gold mapping as canonical booleans in the question's roster order.

    D19's mapping is keyed by name, so every heuristic below has to read it *per
    person*: this resolves each roster name through the row's own ``role_words`` and
    returns ``None`` (counted as a miss by the caller) when the mapping is not a
    readable per-inhabitant answer.
    """
    answer = payload_of(row).get("answer")
    if not isinstance(answer, dict) or not answer:
        return None
    words = list(row["extra_info"].get("role_words") or [])
    if len(words) != 2:
        return None
    try:
        roster = names_in_question(question_of(row))
    except ValueError:
        return None
    vector: list[bool] = []
    for name in roster:
        if name not in answer:
            return None
        flag = _role_flag(answer[name], words)
        if flag is None:
            return None
        vector.append(flag)
    return vector


def check_l3(rows: list[dict], index: dict, reporter: Reporter) -> None:
    """The anti-cheat layer the design doc does not define for K&K.

    Every row is re-derived against the source formula (L3a, design risk 5), and
    the prompt-independent guess baselines (L3b/L3c) are scored **per person**
    against the D19 gold mapping -- the mapping is keyed by name, so a
    position-wise guess has to be evaluated person by person rather than as a
    role-word sequence.
    """
    # a. no inverted gold.  Every row, not a sample: this is design risk 5.
    inverted: list[str] = []
    non_canonical = 0
    for row in rows:
        info = row["extra_info"]
        task_id = info["task_id"]
        words = list(info["role_words"])
        if tuple(words) != CANONICAL_ROLE_WORDS:
            non_canonical += 1
        parsed = kk_adapter.parse_task_id(task_id)
        if parsed is None:
            inverted.append(f"{task_id}: unparseable task_id")
            continue
        family, count, raw_index = parsed
        source = index.get((family, count, raw_index))
        if source is None:
            inverted.append(f"{task_id}: not in the raw files")
            continue
        # Re-derive from the puzzle, not from the row's stored solution field: a gold
        # taken from the canonical side of a flipped mapping survives every
        # artifact-local check and only shows up when the formula is enumerated.
        try:
            statements = parse_own_statements(source["statements"], count)
        except BadStatement as exc:
            inverted.append(f"{task_id}: {exc}")
            continue
        solutions = solutions_own(statements, count)
        if len(solutions) != 1:
            inverted.append(f"{task_id}: the source puzzle has {len(solutions)} solutions")
            continue
        expected = answer_mapping_own(list(source["names"]), [bool(flag) for flag in solutions[0]], words)
        answer = payload_of(row).get("answer")
        if answer != expected:
            inverted.append(f"{task_id}: " + ", ".join(mapping_diff(answer, expected)[:3]))
    reporter.check(
        "L3a no inverted gold (every row re-derived per inhabitant from the source formula)",
        not inverted,
        f"{len(rows)} rows re-derived against the source ({non_canonical} with non-canonical "
        f"role words), {len(inverted)} inverted" + (f"; first: {inverted[:3]}" if inverted else ""),
    )

    # b/c. prompt-independent guess baselines, scored per person against the mapping.
    vectors: list[tuple[int, list[bool]]] = []
    unreadable = 0
    for row in rows:
        vector = gold_vector(row)
        if vector is None:
            unreadable += 1
            continue
        vectors.append((len(vector), vector))
    if unreadable:
        reporter.note(f"L3b/c: {unreadable} rows without a readable mapping gold are counted as misses")

    total = max(len(vectors) + unreadable, 1)
    person_total = max(sum(width for width, _ in vectors), 1)
    constant_rows: dict[str, int] = {}
    constant_people: dict[str, float] = {}
    for label, value in (("truth word", True), ("lie word", False)):
        rows_hit = sum(1 for _, vector in vectors if all(flag == value for flag in vector))
        people_hit = sum(sum(1 for flag in vector if flag == value) for _, vector in vectors)
        constant_rows[label] = rows_hit
        constant_people[label] = people_hit / person_total
        reporter.note(
            f"L3b everyone-the-{label}: {rows_hit}/{len(vectors)} whole mappings = "
            f"{rows_hit / total:.4f}, per person {constant_people[label]:.4f}"
        )
    best_constant_rows = max(constant_rows.values())
    best_constant_people = max(constant_people.values())
    reporter.check(
        "L3b constant label assignments stay near the 2**-L floor",
        best_constant_rows / total <= L3_MAX_HEURISTIC_ACCURACY,
        f"best constant assignment matches {best_constant_rows}/{total} = "
        f"{best_constant_rows / total:.4f} whole mappings (per person {best_constant_people:.4f}, "
        f"threshold {L3_MAX_HEURISTIC_ACCURACY:.2f})",
    )

    # The pool mixes the five size tiers (L = 4..8), and a position-wise guess only
    # makes sense inside one tier.  ``hits`` counts rows whose own tier's majority
    # mapping they reproduce (the whole task), ``person_hits`` the individual
    # inhabitant decisions, so the rate is comparable with the 2**-L floor.
    by_length: dict[int, list[list[bool]]] = collections.defaultdict(list)
    for width, vector in vectors:
        by_length[width].append(vector)
    hits = person_hits = counted = person_counted = 0
    floors: dict[int, float] = {}
    for width, group in sorted(by_length.items()):
        if len(group) < L3_MIN_TIER_ROWS:
            reporter.note(
                f"L3c L={width}: {len(group)} rows, too few to measure a heuristic rate "
                f"(needs {L3_MIN_TIER_ROWS}) -- not counted"
            )
            continue
        majority = [
            collections.Counter(vector[position] for vector in group).most_common(1)[0][0]
            for position in range(width)
        ]
        matched = sum(1 for vector in group if all(vector[p] == majority[p] for p in range(width)))
        matched_people = sum(sum(1 for p in range(width) if vector[p] == majority[p]) for vector in group)
        hits += matched
        person_hits += matched_people
        counted += len(group)
        person_counted += len(group) * width
        floors[width] = 2.0**-width
        reporter.note(
            f"L3c L={width}: majority per-person {matched_people / (len(group) * width):.4f}, "
            f"whole mappings {matched}/{len(group)} = {matched / len(group):.4f} "
            f"(chance {floors[width]:.4f})"
        )
    # A row whose gold is unreadable cannot be reproduced by any position-wise
    # guess, so it is counted as a miss rather than dropped from the denominator.
    denominator = max(counted + unreadable, 1)
    exact_rate = (hits + unreadable) / denominator
    position_people = person_hits / max(person_counted, 1)
    reporter.check(
        "L3c per-position majority stays near the 2**-L floor",
        exact_rate <= L3_MAX_HEURISTIC_ACCURACY
        and position_people <= best_constant_people + L3_CONSTANT_MARGIN,
        f"per-position majority {hits + unreadable}/{denominator} whole mappings = {exact_rate:.4f} "
        f"<= {L3_MAX_HEURISTIC_ACCURACY:.2f} (2**-L floor for the easiest tier "
        f"{max(floors.values()) if floors else float('nan'):.4f}); per person {position_people:.4f} "
        f"vs best constant {best_constant_people:.4f} + {L3_CONSTANT_MARGIN:.2f}",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def sample_rows(rows: list[dict], *, size: int, seed: int) -> list[dict]:
    """A deterministic sample, sorted by ``task_id`` for a readable failure list."""
    rng = random.Random(seed)
    take = min(len(rows), max(size, L1_MIN_SAMPLE))
    return sorted(rng.sample(rows, take), key=lambda row: row["extra_info"]["task_id"])


def main(argv: list[str] | None = None) -> None:
    """Audit an artifact; exits non-zero unless every check passes.

    Args:
        argv: Command line without ``argv[0]``; defaults to ``sys.argv[1:]``.

    Raises:
        SystemExit: 0 when every check passes, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description="Audit the K&K rows built by kk_adapter.py.")
    parser.add_argument("--rows", required=True, help="parquet written by kk_adapter.py")
    parser.add_argument(
        "--raw-dir",
        default=DEFAULT_RAW_DIR,
        help="the K&K raw directory; the gold is proved against the parquet files there",
    )
    parser.add_argument("--seed", type=int, default=0, help="sampling seed")
    parser.add_argument("--sample", type=int, default=200, help="L1/L2 sample size")
    args = parser.parse_args(argv)

    rows = schema.read_parquet_rows(args.rows)
    if not rows:
        print(f"[FAIL] {args.rows} holds no rows")
        raise SystemExit(1)
    print(f"rows    : {args.rows} ({len(rows)} rows)")
    print(f"families: {dict(collections.Counter(r['extra_info']['perturbation_family'] for r in rows))}")
    print(f"tiers   : {dict(collections.Counter(r['extra_info']['difficulty'] for r in rows))}")

    index = load_raw_index(args.raw_dir)

    reporter = Reporter()
    check_contract(rows, reporter)
    check_prompt_contract(rows, reporter)
    check_source_anchor(rows, index, args.raw_dir, reporter)
    check_sentence_oracles(rows, index, reporter)

    sampled = sample_rows(rows, size=args.sample, seed=args.seed)
    reporter.note(f"L1/L2 sample: {len(sampled)} rows of {len(rows)}")
    if len(sampled) < L1_MIN_SAMPLE:
        reporter.check("L1/L2 sample size", False, f"only {len(sampled)} rows to audit, need {L1_MIN_SAMPLE}")
    check_l1(sampled, index, reporter)
    check_l2(rows, sampled, index, reporter)
    check_l3(rows, index, reporter)

    print()
    if reporter.failed:
        print(f"RESULT: FAIL ({len(reporter.failed)}/{len(reporter.results)} checks failed: {reporter.failed})")
        raise SystemExit(1)
    print(f"RESULT: PASS ({len(reporter.results)}/{len(reporter.results)} checks passed)")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
