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
"""K&K adapter -- knights-and-knaves rows whose gold is a role-word sequence.

Source: the ``K-and-K/knights-and-knaves`` (clean) and
``K-and-K/perturbed-knights-and-knaves`` (perturbed) HF datasets, downloaded as 26
parquet files, 48,076 rows (CC-BY-NC-SA-4.0).  Design doc section 4.1 records the
verification of this source and section 4.2 specifies this adapter.

Every row is a *solvable* logical puzzle: a constraint system over N inhabitants,
each of whom is a truth-teller or a liar.  Sections 4.1/4.2 are unambiguous that
K&K supplies **no** unanswerable / refusal / diagnosis data (measured: 0 rows with
0 or >= 2 solutions over all 48,076), so this adapter fills exactly one contract
cell (design doc section 4.9.3 table B row 2):

========================  ========  ===============================================
branch                    template  gold
========================  ========  ===============================================
``solvable_roles``        B         the **surface** role words, one per inhabitant,
                                    in ``names`` order (``angel devil devil``)
========================  ========  ===============================================

The gold is the *surface* sequence, never the canonical ``knight``/``knave``
words: 13,800 rows (28.70%) -- the ``flip_role`` and ``random_pair`` families --
spell the two roles differently, and ``role_words`` carries the row's own pair so
the reward can map both sides into canonical space.  Skipping that read is design
risk 9 and makes the reward run *backwards* on those rows (design doc sections 5.1
and 12 Q17; the sample document's ``"KNAK"`` payload is an instance of the bug).

Certificates (fail closed; each is re-derived by ``verify_kk.py``)
-----------------------------------------------------------------

* **Enumerator** -- ``statements`` is the puzzle's constraint system as a Python
  tuple (``ast.literal_eval``, *not* JSON): one formula per inhabitant, in
  ``names`` order, over four operators and two leaf forms.  Brute-forcing the
  ``2**N`` assignments must yield **exactly one** solution and that solution must
  equal the row's own ``solution`` field.  A row with 0 or >= 2 solutions is
  dropped, never guessed.  This is the free, always-recomputable ground truth
  design doc section 9 asks for.
* **Surface words** -- ``role_words = (knight_knave['knight'], knight_knave['knave'])``
  read from the row itself, and the two words must be distinct.  A row whose
  ``knight_knave`` is missing, blank or degenerate is dropped.
* **Group** -- ``(len(names), index)`` identifies one abstract problem and its
  (up to 7) wording variants; at most one variant enters the pool, so no group can
  straddle a train/val boundary (design doc section 4.2; the split is by group,
  and with one member per group the rule holds by construction).
* **Shape** -- names are distinct, ``len(solution) == len(names)``, every entry a
  bool, ``index`` an int (fail closed on anything else).

Two independent oracles corroborate the gold from the raw row, and
``verify_kk.py`` re-runs both: the dataset's own ``solution_text`` and
``solution_text_format`` sentences are rebuilt from ``names`` + ``solution`` +
``knight_knave`` and must match the source byte for byte (measured 48,076/48,076
for each).  They are what proves that ``knight_knave['knight']`` is the
truth-teller word on every family, ``flip_role`` included.

DEVIATIONS FROM THE DESIGN DOC
------------------------------

Every number below is measured by this adapter; ``main()`` prints the funnel that
shows where each row is lost.

1. **Output size: the adapter ships the full 5,000-row group-deduplicated pool,
   not the 2,000 of table B row 2.**  The 2,000 is the *mix* quota, not the
   adapter's: ``mix_halluc.DEFAULT_QUOTA[(solvable_roles, halluc_logic_kk)]`` is
   2,400 = these 2,000 plus the 400 synthesised-distractor rows of
   ``distractor_synth.py``, and ``mix_halluc`` carves its val split out of the
   cell *before* filling that quota.  The house convention is explicit
   (``kuq_adapter.py``: "It is not the balance mechanism -- the two arms ... are
   drawn per branch by ``mix_halluc.py`` from the full pool"), and
   ``umwp_adapter`` ships 1,945 solvable-judge rows against a 550 quota.
   ``--limit 2000`` reproduces the doc's number exactly (1,000 clean + 1,000
   perturbed); the default ships all 5,000.

2. **``difficulty`` is the string ``"4ppl"``, not the int 4.**  Section 4.2 says
   ``difficulty_tag = len(names)``, but ``schema.py``'s ``validate_row`` rejects a
   non-string ``difficulty`` ("stage 1 stores a string").  The only precedent in
   the repo is the sample document's ``"difficulty": "4ppl"``, which is also how
   the source filenames spell the tier, so the tier is rendered ``f"{n}ppl"``.

3. **The source has no ``split`` and no ``family`` column.**  Section 4.2 assumes
   the perturbed files "carry a family name"; measured, all 26 files share one
   Arrow schema of 11 columns and neither field is in it.  Both are parsed from
   the filename -- and the third field of a *clean* file is the tier
   (``clean__train__2ppl`` -> ``"2ppl"``), not the family, so ``family`` is forced
   to ``"clean"`` whenever the first field is ``clean``.

4. **Section 4.1 verification 4: "all 48,076 prompts end in 'So who is a knight
   and who is a knave?'" is false.**  Measured, there are **8** distinct tails;
   only **34,276/48,076 (71.3%)** carry the quoted one -- ``flip_role`` asks
   "a knave and who is a knight?" (6,900) and ``random_pair`` asks in its own
   role words (6,900 over 6 pairs: hero/villain, saint/sinner, sage/fool,
   angel/devil, pioneer/laggard, altruist/egoist).  The doc's *next* clause ("role-word regex
   48,076/48,076, 0 unmatched") is exactly true and is the one this adapter
   relies on; the two sentences contradict each other one line apart.

5. **The pool is train-only.**  Sections 4.1/4.8 measure the corpus as "about
   6,900 abstract problems x 7 variants" and put the pool unit at "5,000 groups",
   which is the ``train`` side after ``N >= 4``.  The source ``test`` split
   (500 eligible groups / 3,492 rows) is held out and reported in the funnel as
   ``after_source_test_drop``.  Never mixing the two source splits is also what
   makes the section 4.2 group-leak rule automatic: measured, **0 of 6,900
   groups** straddle the source boundary.

6. **The doc defines no L3 for K&K** (there is no binary label and no options
   block, so the BoW-NB yardstick used for SUM/UMWP/TreeCut/CREPE/KUQ does not
   exist here).  ``verify_kk.py`` substitutes the analogous anti-cheat floor for a
   *sequence* answer: the per-position-majority baseline and the single-word
   heuristics must stay near the ``2**-L`` chance floor.  Measured maxima are
   printed, and the hard threshold (<= 0.25) is set well above them.

Departures from ``halluc_samples.md`` 5.1/5.2 (the user's own sample file)
--------------------------------------------------------------------------

The sample document's K&K payloads contradict the design doc, the schema and the
reward on four points; this adapter follows the design doc.  Recorded here because
the sample document is an input to the design, not because it is authoritative.

* ``answer`` is the **surface** sequence (``angel devil``), not a canonical
  ``KNAK``/``NKKN`` code.  Section 4.2, section 3's row example and section 12 Q17
  all say surface words; ``schema.py``'s ``_KK_ROLE_INSTRUCTION`` tells the model
  to answer in the prompt's own words; and the reward maps the answer through
  ``_role_map``, so ``K``/``N`` are out of vocabulary and a coded gold would score
  **0 forever** (``reward/hallucination_compute_score.py``).
* ``ground_truth.role_words`` is present on every row.  Both sample payloads omit
  it; without it the reward logs "K&K row without a usable role_words list; scoring
  0" -- the sample document is itself an instance of design risk 9.
* ``answer`` is not derived by stripping the English ``solution_text``; it is built
  from ``solution`` and ``knight_knave`` directly.
* ``perturbation_type`` stays ``None``.  The sample payload sets
  ``"perturbed_statement"``, which is not in ``schema.PERTURBATION_TYPES`` and is
  rejected by ``validate_row``; the family belongs in
  ``extra_info.perturbation_family``, where it is a monitoring bucket and never a
  reward term.
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import os
import sys
from pathlib import Path

try:
    import schema
except ImportError:  # pragma: no cover - running as a plain script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import schema

# ---------------------------------------------------------------------------
# source constants
# ---------------------------------------------------------------------------

DATA_SOURCE = schema.SOURCE_KK
DEFAULT_RAW_DIR = "/home/charles/data/reasoning_rl/halluc/raw/kk"
DEFAULT_OUT = os.path.expanduser("~/data/reasoning_rl/halluc/built/kk.parquet")


def report_path_for(out: str) -> str:
    """The report that belongs to ``--out``: same stem, ``_report.json``.

    Derived from ``--out`` instead of pinned to the build directory, so that a
    scratch build (``--out /tmp/kk.parquet``) cannot overwrite the canonical
    ``kk_report.json`` -- the file the build report cites -- while the rows it
    does not touch stay in place.  Same convention in all four adapters.
    """
    return os.path.splitext(out)[0] + "_report.json"

BRANCH_ROLES = schema.BRANCH_SOLVABLE_ROLES
TEMPLATE = schema.TEMPLATE_B

#: The seven wording families design doc section 4.2 names.  Six are the source's
#: own perturbation files; ``clean`` is the unperturbed member of a group.
FAMILY_CLEAN = "clean"
FAMILIES = (
    "clean",
    "perturbed_statement",
    "perturbed_leaf",
    "reorder_statement",
    "random_pair",
    "uncommon_name",
    "flip_role",
)
PERTURBATION_FAMILIES = tuple(family for family in FAMILIES if family != FAMILY_CLEAN)

#: The pair a row must spell to be "canonical"; anything else is a flipped or
#: renamed pair and is the 28.7% the reward must not invert on.
CANONICAL_ROLE_WORDS = ("knight", "knave")

#: Design doc section 4.1: N = 2 and N = 3 tiers are guesswork (25% / 12.5%).
MIN_INHABITANTS = 4

#: The enumerator is a 2**N brute force.  The corpus max is 8; anything past this
#: bound is dropped rather than silently skipped (fail closed).
MAX_ENUMERATED = 16

#: Statement-tree depth bound (the corpus max is a handful).
MAX_STATEMENT_DEPTH = 64

_FILE_PARTS = ("clean", "perturbed")
_FILE_SPLITS = ("train", "test")


# ---------------------------------------------------------------------------
# the enumerator -- the free, recomputable oracle of design doc section 9
# ---------------------------------------------------------------------------


class StatementError(ValueError):
    """A ``statements`` payload that is not a well-formed K&K formula tree."""


_OPERATORS = ("and", "or", "->", "<=>")
_LEAF_KINDS = ("telling-truth", "lying")


def _check_node(node: object, count: int, depth: int = 0) -> None:
    """Reject anything that is not a legal K&K formula over ``count`` people."""
    if depth > MAX_STATEMENT_DEPTH:
        raise StatementError(f"statement tree deeper than {MAX_STATEMENT_DEPTH}")
    if not isinstance(node, tuple) or not node:
        raise StatementError(f"statement node must be a non-empty tuple, got {node!r}")
    kind = node[0]
    if kind in _LEAF_KINDS:
        if len(node) != 2:
            raise StatementError(f"{kind!r} takes one person index, got {node!r}")
        index = node[1]
        if isinstance(index, bool) or not isinstance(index, int):
            raise StatementError(f"{kind!r} index must be an int, got {index!r}")
        if not 0 <= index < count:
            raise StatementError(f"{kind!r} index {index} outside 0..{count - 1}")
        return
    if kind == "not":
        if len(node) != 2:
            raise StatementError(f"'not' takes one operand, got {node!r}")
        _check_node(node[1], count, depth + 1)
        return
    if kind in _OPERATORS:
        if len(node) != 3:
            raise StatementError(f"{kind!r} takes two operands, got {node!r}")
        _check_node(node[1], count, depth + 1)
        _check_node(node[2], count, depth + 1)
        return
    raise StatementError(f"unknown statement operator {kind!r}")


def parse_statements(text: str, count: int) -> tuple:
    """``statements`` -> a validated tuple of ``count`` formula trees.

    The column is the ``repr`` of a Python tuple, so ``ast.literal_eval`` is the
    parser (``json.loads`` fails on it).  Anything unparseable or out of shape
    raises :class:`StatementError`, which the caller turns into a drop.
    """
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError) as exc:
        raise StatementError(f"statements is not a Python literal: {exc}") from exc
    if not isinstance(parsed, tuple) or not parsed:
        raise StatementError("statements must be a non-empty tuple of formulas")
    if len(parsed) != count:
        raise StatementError(f"{len(parsed)} statements for {count} inhabitants")
    for node in parsed:
        _check_node(node, count)
    return parsed


def evaluate(node: tuple, truth: tuple[bool, ...]) -> bool:
    """Truth value of one formula under ``truth`` (``truth[i]`` == person i tells truth).

    ``telling-truth i`` is true exactly when person ``i`` is a truth-teller and
    ``lying i`` exactly when they are not; there is no third leaf form in the
    corpus (measured over all 48,076 rows).

    The tree is assumed already validated by :func:`parse_statements`, which is
    where a malformed shape is rejected: this function is the ``2**N`` inner loop
    and re-checking every node there would cost more than the enumeration.
    """
    kind = node[0]
    if kind == "telling-truth":
        return bool(truth[node[1]])
    if kind == "lying":
        return not truth[node[1]]
    if kind == "not":
        return not evaluate(node[1], truth)
    if kind == "and":
        return evaluate(node[1], truth) and evaluate(node[2], truth)
    if kind == "or":
        return evaluate(node[1], truth) or evaluate(node[2], truth)
    if kind == "->":
        return (not evaluate(node[1], truth)) or evaluate(node[2], truth)
    if kind == "<=>":
        return evaluate(node[1], truth) == evaluate(node[2], truth)
    raise StatementError(f"unknown statement operator {kind!r}")


def enumerate_solutions(statements: tuple, count: int) -> list[tuple[bool, ...]]:
    """Every assignment satisfying the puzzle, by brute force over ``2**count``.

    Person ``i``'s assertion asserts their own statement, so the constraint is
    ``truth[i] == evaluate(statements[i], truth)`` for every ``i``.
    """
    solutions: list[tuple[bool, ...]] = []
    for mask in range(1 << count):
        truth = tuple(bool(mask >> i & 1) for i in range(count))
        if all(evaluate(stmt, truth) == truth[i] for i, stmt in enumerate(statements)):
            solutions.append(truth)
    return solutions


def unique_solution(statements: tuple, count: int) -> tuple[bool, ...] | None:
    """The one satisfying assignment, or ``None`` when there is none or several.

    Stops at the second solution, so the happy path costs one sweep.  A caller
    that needs to know *why* it was rejected can re-run
    :func:`enumerate_solutions` on the rejected rows only.
    """
    found: tuple[bool, ...] | None = None
    for mask in range(1 << count):
        truth = tuple(bool(mask >> i & 1) for i in range(count))
        if all(evaluate(stmt, truth) == truth[i] for i, stmt in enumerate(statements)):
            if found is not None:
                return None
            found = truth
    return found


# ---------------------------------------------------------------------------
# text plumbing
# ---------------------------------------------------------------------------


def normalise_text(text: str) -> str:
    """Collapse whitespace and strip.

    A no-op on this corpus (measured: no leading whitespace, no newline, no
    non-ASCII in ``quiz``), but the prompt is rendered from the result and the
    verifier compares against it, so both sides use the same function.
    """
    return " ".join((text or "").split())


def canonical_gold(names: list[str], solution: list[bool]) -> str:
    """The canonical knight/knave rendering -- an *audit* string, never the gold."""
    return " ".join(CANONICAL_ROLE_WORDS[0] if flag else CANONICAL_ROLE_WORDS[1] for flag in solution)


def surface_gold(solution: list[bool], role_words: list[str]) -> str:
    """The gold: the row's own two words, one per inhabitant, in ``names`` order.

    Kept byte-identical to ``distractor_synth.kk_answer`` (whose docstring says it
    "must stay identical to whatever ``kk_adapter.py`` emits") -- the reward maps
    both this and the model's answer through ``role_words``, so a divergence
    silently mis-scores the 400 synthesised rows that share this branch.
    """
    truth_word, lie_word = role_words
    return " ".join(truth_word if flag else lie_word for flag in solution)


# ---------------------------------------------------------------------------
# source loading
# ---------------------------------------------------------------------------


def parse_file_stem(stem: str) -> tuple[str, str, str] | None:
    """``"{clean|perturbed}__{train|test}__{tier|family}"`` -> ``(part, split, family)``.

    Returns ``None`` for any name that is not this contract, so the caller can
    count and drop the file rather than guess a family.  The third field of a clean
    file is the tier (``clean__train__2ppl``), not the family, so ``family`` is
    forced to ``"clean"`` on that side.
    """
    parts = stem.split("__")
    if len(parts) != 3:
        return None
    part, split, third = parts
    if part not in _FILE_PARTS or split not in _FILE_SPLITS:
        return None
    family = FAMILY_CLEAN if part == "clean" else third
    if family not in FAMILIES:
        return None
    return part, split, family


def load_source(raw_dir: str) -> list[dict]:
    """Read every ``*.parquet`` under ``raw_dir``, tagging each row with its file.

    Every row gains three private keys -- ``_family``, ``_split`` and ``_file`` --
    carrying what the Arrow schema does not: the wording family and the source
    split.  A file whose name is not the dataset's own contract yields rows with
    ``_family == ""``, which the build drops and counts (fail closed rather than
    inventing a family).
    """
    import pyarrow.parquet as pq

    root = Path(raw_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"--raw-dir {raw_dir} is not a directory")
    files = sorted(root.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no *.parquet files under {raw_dir}")

    rows: list[dict] = []
    for path in files:
        parsed = parse_file_stem(path.stem)
        part, split, family = parsed if parsed else ("", "", "")
        for row in pq.read_table(path).to_pylist():
            row["_part"], row["_split"], row["_family"] = part, split, family
            row["_file"] = path.name
            rows.append(row)
    return rows


def is_well_formed(row: object) -> bool:
    """Whether one raw row carries a usable puzzle (fail closed on anything else)."""
    if not isinstance(row, dict):
        return False
    if not isinstance(row.get("_family"), str) or row["_family"] not in FAMILIES:
        return False
    if not isinstance(row.get("quiz"), str) or not row["quiz"].strip():
        return False
    names = row.get("names")
    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(name, str) and name.strip() for name in names)
        or len(set(names)) != len(names)
    ):
        return False
    solution = row.get("solution")
    if not isinstance(solution, list) or len(solution) != len(names):
        return False
    if not all(isinstance(flag, bool) for flag in solution):
        return False
    roles = row.get("knight_knave")
    if not isinstance(roles, dict):
        return False
    truth_word, lie_word = role_words_of(row) or ("", "")
    if not truth_word or not lie_word or truth_word.casefold() == lie_word.casefold():
        return False
    if not isinstance(row.get("statements"), str) or not row["statements"].strip():
        return False
    if isinstance(row.get("index"), bool) or not isinstance(row.get("index"), int):
        return False
    return True


def role_words_of(row: dict) -> tuple[str, str] | None:
    """``(truth-teller word, liar word)`` as this row spells them, or ``None``."""
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


# ---------------------------------------------------------------------------
# certificates
# ---------------------------------------------------------------------------


def certify_row(row: dict) -> dict | None:
    """The row's gold and canonical witness, or ``None`` when it cannot be proved.

    Three things must hold: the row passes :func:`is_well_formed`, the enumerator
    yields exactly one solution, that solution is the row's own ``solution`` field,
    and the row's two role words are readable and distinct.  Anything else is a
    drop -- no gold is ever invented.
    """
    if not is_well_formed(row):
        return None
    words = role_words_of(row)
    if words is None:
        return None
    truth_word, lie_word = words
    if truth_word.casefold() == lie_word.casefold():
        return None
    names = list(row["names"])
    solution = [bool(flag) for flag in row["solution"]]
    try:
        statements = parse_statements(row["statements"], len(names))
    except StatementError:
        return None
    witness = unique_solution(statements, len(names))
    if witness is None or witness != tuple(solution):
        return None
    return {
        "names": names,
        "solution": solution,
        "role_words": [truth_word, lie_word],
        "gold": surface_gold(solution, [truth_word, lie_word]),
        "canonical_gold": canonical_gold(names, solution),
    }


def solution_text_of(row: dict, witness: list[bool]) -> str:
    """Rebuild the source's own ``solution_text`` sentence.

    Independent of the adapter: it is rebuilt from ``names``, the enumerated
    solution and the row's ``knight_knave``, and must match the file's sentence.
    Used by the verifier as the second oracle for the surface-word convention.
    """
    words = role_words_of(row)
    if words is None:
        return ""
    roles = row["knight_knave"]
    parts = [
        f"{name} is {roles['a_knight'] if flag else roles['a_knave']}"
        for name, flag in zip(row["names"], witness)
    ]
    if len(parts) == 1:
        return parts[0] + "."
    return ", and ".join([", ".join(parts[:-1]), parts[-1]]) + "."


def solution_text_format_of(row: dict, witness: list[bool]) -> str:
    """Rebuild the source's own ``solution_text_format`` block (one line per person)."""
    roles = row["knight_knave"]
    return "\n".join(
        f"({rank}) {name} is {roles['a_knight'] if flag else roles['a_knave']}"
        for rank, (name, flag) in enumerate(zip(row["names"], witness), start=1)
    )


# ---------------------------------------------------------------------------
# group selection (design doc section 4.2)
# ---------------------------------------------------------------------------


def group_key(row: dict) -> tuple[int, int]:
    """``(len(names), index)`` -- one abstract problem and all its wording variants.

    ``index`` is unique only inside a size tier (each ``{N}ppl`` file restarts at
    0 on the test side), so the tier is part of the key; measured, this key is
    unique over all 48,076 rows.
    """
    return (len(row["names"]), int(row["index"]))


def task_id_of(family: str, count: int, index: int) -> str:
    """Stable hard-replay key.  Distinct prefix from ``distractor_synth``'s ``d17:``.

    Deliberately *not* ``kk:{stem}:{index}``: ``distractor_synth`` fills 400 rows
    into this same ``(solvable_roles, halluc_logic_kk)`` cell from the same files,
    and its ``uid`` is built from the stem.  The key here is also the group key, so
    ``verify_kk.py`` can re-derive the group from the artifact alone (which matters
    because ``mix_halluc`` overwrites ``extra_info.index`` downstream).
    """
    return f"kk:{family}:{count}ppl:{index}"


def parse_task_id(task_id: str) -> tuple[str, int, int] | None:
    """``kk:{family}:{N}ppl:{index}`` -> ``(family, N, index)``, or ``None``."""
    parts = task_id.split(":")
    if len(parts) != 4 or parts[0] != "kk":
        return None
    family, tier, raw_index = parts[1], parts[2], parts[3]
    if family not in FAMILIES or not tier.endswith("ppl"):
        return None
    try:
        return family, int(tier[:-3]), int(raw_index)
    except ValueError:
        return None


def group_key_for_task_id(task_id: str) -> tuple[int, int] | None:
    """The group key ``(len(names), index)`` encoded in a ``task_id``, or ``None``.

    ``verify_kk.py`` audits a *mixed* artifact, where ``mix_halluc`` has already
    overwritten ``extra_info.index`` and ``split``; the group is only recoverable
    from ``task_id``, which is why the id is built from the group in the first
    place.
    """
    parsed = parse_task_id(task_id)
    if parsed is None:
        return None
    _, count, index = parsed
    return count, index


def _families_present(members: list[dict]) -> dict[str, dict]:
    """``family -> member`` for one group, first wins under a deterministic order."""
    present: dict[str, dict] = {}
    for member in sorted(members, key=lambda item: (item["_file"], int(item["index"]))):
        present.setdefault(member["_family"], member)
    return present


def _pick_perturbed(present: dict[str, dict], offset: int) -> dict | None:
    """The group's perturbed member at ``offset`` in a rotating family order.

    The rotation is what keeps the six perturbation families roughly even: a
    group that lacks the family the rotation lands on (measured: ``perturbed_leaf``
    is absent from 218 clean problems and ``perturbed_statement`` from 6) shifts to
    the next one rather than collapsing the balance onto one family.
    """
    total = len(PERTURBATION_FAMILIES)
    for step in range(total):
        family = PERTURBATION_FAMILIES[(offset + step) % total]
        member = present.get(family)
        if member is not None:
            return member
    return None


def select_group_members(groups: dict[tuple[int, int], list[dict]]) -> tuple[list[dict], dict]:
    """Pick at most one member per group, aiming for clean : perturbed = 1:1.

    Groups are visited in ``(len(names), index)`` order and alternate clean /
    perturbed, which is exact rather than approximate: measured, **2,500 clean +
    2,500 perturbed**, 500/500 inside every size tier.  The alternative -- a per
    group coin flip -- would leave the ratio to chance for no benefit, and the
    design doc only asks for "roughly 1:1".
    """
    chosen: list[dict] = []
    stats = collections.Counter()
    perturbed_rank = 0
    for rank, key in enumerate(sorted(groups)):
        present = _families_present(groups[key])
        member = None
        if rank % 2 == 0:
            member = present.get(FAMILY_CLEAN)
            if member is None:  # unreachable on this corpus; fail open to perturbed
                stats["group_without_clean"] += 1
        if member is None:
            member = _pick_perturbed(present, perturbed_rank)
            if member is not None:
                perturbed_rank += 1
        if member is None:
            stats["group_without_member"] += 1
            continue
        chosen.append(member)
    return chosen, dict(stats)


def _interleave_by_tier(chosen: list[dict]) -> list[dict]:
    """Round-robin the size tiers so a ``limit`` keeps every tier represented.

    :func:`select_group_members` returns its members in ``(len(names), index)``
    order, so a plain ``chosen[:limit]`` is *tier-ordered*, not merely
    tier-sorted: measured, ``--limit 400`` returned 400 rows all at ``4ppl``,
    whose answer strings are the shortest in the corpus.  The slice then reads
    as a corpus with one difficulty, and every length-sensitive statistic
    (answer form, prompt size, the L3 floors) is measured on the easiest tier
    only.  The same defect, with the same fix, is guarded in the ``sum``
    (by branch) and ``treecut`` (by cell) adapters.
    """
    tiers: dict[int, list[dict]] = collections.OrderedDict()
    for member in chosen:
        tiers.setdefault(len(member["names"]), []).append(member)
    out: list[dict] = []
    index = 0
    while True:
        progressed = False
        for tier in tiers.values():
            if index < len(tier):
                out.append(tier[index])
                progressed = True
        if not progressed:
            return out
        index += 1


# ---------------------------------------------------------------------------
# row construction
# ---------------------------------------------------------------------------


def _build_row(member: dict, clean_sibling: dict | None, seed: int) -> dict:
    """One artifact row.  ``clean_sibling`` supplies the audit-only pairing text."""
    count = len(member["names"])
    index = int(member["index"])
    family = member["_family"]
    certificate = certify_row(member)
    if certificate is None:  # pragma: no cover - selection only admits certified rows
        raise ValueError(f"uncertified row reached the builder: {member.get('_file')}")
    role_words = certificate["role_words"]
    paired = ""
    if family != FAMILY_CLEAN and clean_sibling is not None:
        paired = normalise_text(clean_sibling["quiz"])
    extra_info = {
        "split": "train",
        "index": index,
        "task_id": task_id_of(family, count, index),
        "seed": seed,
        "difficulty": f"{count}ppl",
        "perturbation_family": family,
        "solvable": True,
        "role_words": list(role_words),
        "canonical_solution": list(certificate["solution"]),
        "paired_original_text": paired,
    }
    ground_truth = schema.build_ground_truth(
        solvable=True,
        answer=certificate["gold"],
        correct_option_id=None,
        has_diagnosis_label=False,
        perturbation_type=None,
        role_words=list(role_words),
    )
    return schema.make_row(
        data_source=DATA_SOURCE,
        question=normalise_text(member["quiz"]),
        ground_truth=ground_truth,
        template=TEMPLATE,
        branch=BRANCH_ROLES,
        extra_info=extra_info,
        role_words=list(role_words),
    )


def build_rows(raw_dir: str, limit: int | None = None, seed: int = 0) -> tuple[list[dict], dict]:
    """Build the K&K parquet rows and the report that produced them.

    Returns ``(rows, report)``: ``rows`` are ready for
    :func:`schema.normalise_extra_info` / :func:`schema.validate_rows` /
    :func:`schema.write_rows_parquet`, and ``report`` carries the ordered drop
    chain under ``"funnel"`` plus the measured group / family / enumerator
    counters.  Selection is deterministic -- ``seed`` is recorded but never
    chooses a row -- so the same ``raw_dir`` and ``limit`` always produce
    byte-identical rows.
    """
    raw = load_source(raw_dir)
    funnel: dict[str, int] = collections.OrderedDict()
    funnel["raw_rows"] = len(raw)

    # 1. file names.  A file that is not named to the dataset's own contract has
    #    no family and no split, so its rows cannot be placed in the pool.
    named = [row for row in raw if row.get("_family") in FAMILIES]
    funnel["after_filename_drop"] = len(named)

    # 2. row shape.
    shaped = [row for row in named if is_well_formed(row)]
    funnel["after_malformed_drop"] = len(shaped)

    # 3. the enumerator: exactly one solution, equal to the row's own field.
    enumerator = collections.Counter()
    certified: list[dict] = []
    for row in shaped:
        count = len(row["names"])
        if count > MAX_ENUMERATED:
            enumerator["too_many_inhabitants"] += 1
            continue
        try:
            statements = parse_statements(row["statements"], count)
        except StatementError:
            enumerator["unparseable_statements"] += 1
            continue
        witness = unique_solution(statements, count)
        if witness is None:
            reason = "zero_solution" if not enumerate_solutions(statements, count) else "multi_solution"
            enumerator[reason] += 1
            continue
        if witness != tuple(bool(flag) for flag in row["solution"]):
            enumerator["solution_mismatch"] += 1
            continue
        enumerator["checked"] += 1
        certified.append(row)
    funnel["after_enumerator_drop"] = len(certified)

    # 4. the N >= 4 tiers only (design doc section 4.1: 2ppl/3ppl are guesswork).
    big_enough = [row for row in certified if len(row["names"]) >= MIN_INHABITANTS]
    funnel["after_n_ge_4_drop"] = len(big_enough)

    # 5. the train side only.  Never mixing the two source splits is what makes
    #    the group-leak rule automatic (measured: 0 groups straddle the boundary).
    train = [row for row in big_enough if row["_split"] == "train"]
    funnel["after_source_test_drop"] = len(train)

    # 6. one variant per abstract problem.
    groups: dict[tuple[int, int], list[dict]] = collections.defaultdict(list)
    for row in train:
        groups[group_key(row)].append(row)
    selected, selection_stats = select_group_members(groups)
    funnel["after_group_dedup"] = len(selected)
    # Interleave *before* the limit, so the slice spans the tiers rather than
    # being the first tiers.  Membership is unchanged: the full build is the
    # same list in a different order.
    selected = _interleave_by_tier(selected)

    rows = [_build_row(member, _clean_sibling(groups, member), seed) for member in selected]
    if limit is not None:
        rows = rows[: max(limit, 0)]
    funnel["after_limit"] = len(rows)

    report = {
        "raw_dir": str(raw_dir),
        "seed": seed,
        "limit": limit,
        "funnel": dict(funnel),
        "groups": {
            "total": len(groups),
            "size_histogram": _histogram(len(members) for members in groups.values()),
            "skipped": selection_stats,
        },
        "selected": {
            "per_tier": _breakdown(rows, lambda row: row["extra_info"]["difficulty"]),
            "per_family": _breakdown(rows, lambda row: row["extra_info"]["perturbation_family"]),
            "clean": sum(1 for row in rows if row["extra_info"]["perturbation_family"] == FAMILY_CLEAN),
            "perturbed": sum(1 for row in rows if row["extra_info"]["perturbation_family"] != FAMILY_CLEAN),
            "non_canonical_role_words": sum(
                1 for row in rows if tuple(row["extra_info"]["role_words"]) != CANONICAL_ROLE_WORDS
            ),
            "distinct_role_word_pairs": len({tuple(row["extra_info"]["role_words"]) for row in rows}),
        },
        "enumerator": {
            "checked": enumerator["checked"],
            "zero_solution": enumerator["zero_solution"],
            "multi_solution": enumerator["multi_solution"],
            "solution_mismatch": enumerator["solution_mismatch"],
            "unparseable_statements": enumerator["unparseable_statements"],
            "too_many_inhabitants": enumerator["too_many_inhabitants"],
            "solution_text_oracle": oracle_counts(selected),
            "solution_text_format_oracle": oracle_counts(selected, formatted=True),
        },
    }
    return rows, report


def _clean_sibling(groups: dict, member: dict) -> dict | None:
    """The clean member of ``member``'s own group, or ``None``."""
    for candidate in groups[group_key(member)]:
        if candidate["_family"] == FAMILY_CLEAN:
            return candidate
    return None


def oracle_counts(members: list[dict], *, formatted: bool = False) -> dict:
    """How many selected members the dataset's own sentence corroborates.

    The two sentences the source ships (``solution_text`` and
    ``solution_text_format``) are rebuilt from ``names`` + the enumerated solution
    + ``knight_knave`` and compared verbatim.  They are a second, independent
    witness that ``knight_knave['knight']`` is the truth-teller word on every
    family; the count is measured here so the report can never claim an oracle that
    was not run.  A row whose ``solution`` disagrees with the enumeration simply
    does not match -- it is a counter, not a filter (the enumerator already
    certified the row).
    """
    matches = 0
    for row in members:
        witness = [bool(flag) for flag in row["solution"]]
        builder = solution_text_format_of if formatted else solution_text_of
        if builder(row, witness) == row.get(
            "solution_text_format" if formatted else "solution_text"
        ):
            matches += 1
    return {"checked": len(members), "matches": matches}


def _histogram(values) -> dict[str, int]:
    counter = collections.Counter(values)
    return {str(key): counter[key] for key in sorted(counter)}


def _breakdown(rows: list[dict], key) -> dict:
    counter: dict = collections.Counter(key(row) for row in rows)
    return {str(name): counter[name] for name in sorted(counter, key=str)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the K&K hallucination-domain rows.")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--report", default=None, help="JSON report path (default: alongside --out)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report_path = args.report or report_path_for(args.out)

    rows, report = build_rows(args.raw_dir, limit=args.limit, seed=args.seed)
    schema.normalise_extra_info(rows)
    schema.validate_rows(rows)
    schema.write_rows_parquet(rows, args.out)

    print(f"raw dir : {args.raw_dir}")
    print(f"wrote   : {args.out}  ({len(rows)} rows, seed={args.seed})")
    print("\nfunnel (rows remaining after each stage, and what that stage cost):")
    previous = None
    for stage, count in report["funnel"].items():
        cost = "" if previous is None else f"   -{previous - count}"
        print(f"  {stage:34s} {count:6d}{cost}")
        previous = count
    print("\ngroups (before in-group dedup):")
    print(f"  total {report['groups']['total']}, sizes {report['groups']['size_histogram']}")
    print("\nper perturbation_family:")
    for family, count in report["selected"]["per_family"].items():
        print(f"  {family:22s} {count}")
    print(
        f"\nclean/perturbed: {report['selected']['clean']}/{report['selected']['perturbed']}"
        f"   non-canonical role words: {report['selected']['non_canonical_role_words']}"
        f"   distinct word pairs: {report['selected']['distinct_role_word_pairs']}"
    )
    print("\nper difficulty:")
    for tier, count in report["selected"]["per_tier"].items():
        print(f"  {tier:22s} {count}")
    print("\nenumerator (over the pre-N>=4 pool):")
    for key, count in report["enumerator"].items():
        print(f"  {key:30s} {count}")

    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"\nreport  : {report_path}")


if __name__ == "__main__":  # pragma: no cover - exercised through main() in tests
    main()
