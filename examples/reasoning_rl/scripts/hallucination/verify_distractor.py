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
"""Audit the D17 distractor branch parquet produced by ``distractor_synth.py``.

Three layers, printed as ``PASS`` / ``FAIL`` per check; the process exits
non-zero if any check fails.  This file re-reads the **raw upstream files** with
its own readers and re-derives every per-row claim from the bytes that reach the
model; it does not call ``distractor_synth.choose_distractor`` or its verifier, and
it does not reuse the module's loaders.  What it does share with the module under
audit is the *data* the algorithm is defined in terms of -- the tokeniser
(``schema.words`` / ``is_content_word``), the generic-topic vocabulary, the three
axis vocabularies and targets, and the design quota.  Those are the recon-derived
premises, not the claims; the claims are re-derived here.

L1 -- the answer-invariance certificate
---------------------------------------

D17 asserts a narrow, checkable thing: the synthesised row is a *real* solvable
problem from a real upstream source, with one extra sentence that no correct
answer depends on.  For **every** row (not a sample -- the check is cheap):

* **the join to upstream is by key, not by claim.**  ``extra_info.task_id``
  carries the pool-unique upstream key, and the base question is looked up in the
  raw file by re-deriving that key here; a row whose key does not resolve in the
  raw file is a FAIL, as is a row whose ``paired_original_text`` is not the raw
  file's own question text.
* **the gold is upstream's gold.**  The gold is re-read from the raw row with this
  file's own value handling (SUM's ``ground_truth`` string, UMWP's ``answer``
  array, and K&K's knight/knave mapping re-implemented from ``names`` order,
  ``solution`` flags and the row's own ``knight_knave`` word pair).  It must equal
  the row's ``ground_truth`` answer byte for byte.  This is the check that makes
  the branch a *distractor* branch: the gold was never recomputed, and the
  sentence was never allowed to move it.
* **the sentence is an insertion, not a rewrite.**  The stored ``distractor_text``
  must occur verbatim in the prompt, and deleting it must restore the base
  question.  Note the direction that matters: the base is *not* a contiguous span
  of the prompt, because the sentence lands second-to-last by design, so a
  containment test against the whole prompt would be wrong (it is what an earlier
  revision of the module's self-check got wrong).
* **the sentence is a render, and its fill is recovered.**  Every distractor is a
  ``(template, role, number)`` substitution, so the audit recovers the triple by
  matching the sentence against the branch's own safe templates and requiring the
  re-render to reproduce the sentence byte for byte.  A sentence that no safe
  template reproduces, or that two distinct triples both reproduce, is reported and
  never guessed at.  The recovery is what makes the two label axes below exact
  rather than one-directional: the actor is part of the *fill*, so a reading that
  counted it among the sentence's own words called 93% of this branch ``in_topic``
  when it stores 44%, every overlapped row being one where the actor is by
  construction a word of the base.
* **the red herring is inert.**  The sentence must not occur in the base, and for a
  row that reuses an actor the base already names ("overlapped") the template's
  *own* words -- the skeleton, placeholders blanked -- must share no content word
  with the base at all: restating a property the problem has already fixed is how a
  distractor turns into a contradiction.  This is also the premise the design's
  structural law rests on (an overlapped fill is necessarily out-of-topic).

L2 -- distribution and determinism
----------------------------------

* the three ``extra_info.distractor_labels`` axes must be *balanced*: each axis's
  two labels within +-5pt of the design targets, recomputed from the stored labels
  **and** independently from the bytes (see below), because the balance claim is
  about the branch, and a stored label that disagrees with its own text would make
  the balance meaningless.
* the byte-level re-derivation of each axis, from the recovered fill: the
  ``role_label`` exactly (an overlapped actor is a word of the base, a nonoverlapped
  one is not), the ``number_label`` exactly (the substituted number inside or
  outside the base's numeric span; undecidable only when the base has no numbers),
  and the ``sentence_label`` exactly (the skeleton's topic words intersected with
  the base's content words).  Rows whose fill could not be recovered are counted and
  reported, never silently passed.
* ``task_id`` and ``prompt`` must be unique, and no base may be used twice.  This
  is not ceremonial: K&K's five per-size parquet files each number their rows from
  0, so ``index`` alone collides five-fold within the pool, and a ``task_id`` built
  from it silently maps 61 of the 2,000 rows onto the wrong base.
* re-running the synthesis with the observed per-pool counts must reproduce the
  same rows.  A branch whose balance depends on a random walk that cannot be
  replayed is not auditable, and the D17 margins are close enough to the +-5pt
  boundary that "it balanced when I ran it" is not a checkable statement.

L3 -- anti-cheat
----------------

Every D17 row is ``solvable=true`` (measured: the branch is inserted into
answerable rows only), so the design's 5-fold bag-of-words Naive Bayes between the
solvable and unsolvable label has no negative class and is **inapplicable** --
which is *reported*, not hidden.  The estimator is still run, on the only binary
labels this branch carries: each of the three axes in turn.  Those readings are
*reference numbers*, printed and never gated (a distractor axis that were linearly
decodable from the question text would be a giveaway, so the interesting reading is
that it is *not* decodable).  The estimator itself is gated by a positive control
on synthetic separable data, so a broken estimator cannot masquerade as a clean
reading -- the failure mode the UMWP recon report documents for an unfiltered
multinomial NB (class score flips sign out of fold).

Also reported, and **not** gated, because it is a property of this machine rather
than of D17: the fourth pool (the stage-1 math slice, 400 rows of the design's
2,400) has no source file here, so it is unbuilt.  The summary line states the
built fraction against the design quota so the gap cannot be mistaken for a
complete branch.

The L0 layer asserts :func:`schema.validate_row` on every row.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import os
import random
import re
import sys

try:
    import schema
except ImportError:  # pragma: no cover - exercised by running as a script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import schema

try:
    import distractor_synth as ds
except ImportError:  # pragma: no cover - exercised by running as a script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import distractor_synth as ds

from verify_gsmic import bow_nb_out_of_fold  # the design-specified L3 estimator

DEFAULT_ROWS = "/home/charles/data/reasoning_rl/halluc/d17/d17_rows.parquet"
DEFAULT_RAW_DIR = ds.DEFAULT_RAW_DIR

TASK_ID_PREFIX = "d17:"
# The pool keys `distractor_synth` builds, re-derived independently below.  They
# are a *contract* between the builder and this audit, so they are written out
# rather than imported: an audit that took the key format from the code under
# audit could not catch a change to it.
POOL_KEYS = {
    ds.POOL_SUM: "sum:{split}:{index}",
    ds.POOL_UMWP: "umwp:{line}",
    ds.POOL_KK: "kk:{stem}:{index}",
    ds.POOL_MAIN: "main:{row}",
}
# The pool titles, also part of that contract (the branch name is reported by
# `data_source`, which is what the mixer and the reward see).
POOL_TITLES = {
    ds.POOL_SUM: ds.POOL_SUM,
    ds.POOL_UMWP: ds.POOL_UMWP,
    ds.POOL_KK: ds.POOL_KK,
    ds.POOL_MAIN: ds.POOL_MAIN,
}
DATA_SOURCE_OF_POOL = {
    ds.POOL_SUM: "halluc_math_sum",
    ds.POOL_UMWP: "halluc_math_umwp",
    ds.POOL_KK: "halluc_logic_kk",
    ds.POOL_MAIN: "halluc_math_main",
}
POOL_OF_DATA_SOURCE = {source: pool for pool, source in DATA_SOURCE_OF_POOL.items()}

_APOSTROPHE_RE = re.compile(r"[‘’ʼ`]")
_NB_CHANCE = 0.50
_NB_MIN_DOC_SUPPORT = 5
_NB_POSITIVE_CONTROL_MIN = 0.90


def normalise(text: str) -> str:
    """Collapse whitespace and fold typographic apostrophes to ASCII."""
    return _APOSTROPHE_RE.sub("'", " ".join((text or "").split()))


# ---------------------------------------------------------------------------
# independent readers of the raw upstream files
# ---------------------------------------------------------------------------


def _literal(value):
    """A list/dict that upstream stores either typed or as a Python repr."""
    import ast

    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def read_sum_bases(raw_dir: str) -> dict[str, dict]:
    """SUM's answerable rows, keyed by this audit's own uid."""
    import pyarrow.parquet as pq

    rows = pq.read_table(
        os.path.join(raw_dir, "sum", "train.parquet"),
        columns=["answerable_question", "ground_truth"],
    ).to_pylist()
    bases: dict[str, dict] = {}
    for index, row in enumerate(rows):
        question = (row["answerable_question"] or "").strip()
        if not question:
            continue
        bases[POOL_KEYS[ds.POOL_SUM].format(split="train", index=index)] = {
            "question": question,
            "answer": (row["ground_truth"] or "").strip(),
        }
    return bases


def read_umwp_bases(raw_dir: str) -> dict[str, dict]:
    """UMWP's answerable half, keyed by physical line number."""
    bases: dict[str, dict] = {}
    path = os.path.join(raw_dir, "umwp", "StandardDataset.jsonl")
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if str(row.get("answerable", "")).strip().lower() != "true":
                continue
            question = (row.get("question") or "").strip()
            values = row.get("answer")
            if values is None or not question:
                continue
            if isinstance(values, str):
                try:
                    values = json.loads(values)
                except ValueError:
                    values = [values]
            if not isinstance(values, list) or not values or values[0] is None:
                continue
            value = values[0]
            answer = str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
            bases[POOL_KEYS[ds.POOL_UMWP].format(line=line_number)] = {
                "question": question,
                "answer": answer,
            }
    return bases


def read_kk_bases(raw_dir: str) -> dict[str, dict]:
    """K&K's clean 4+-inhabitant puzzles, with the answer mapped here from scratch."""
    import glob

    import pyarrow.parquet as pq

    bases: dict[str, dict] = {}
    pattern = os.path.join(raw_dir, "kk", "clean__train__*ppl.parquet")
    for path in sorted(glob.glob(pattern)):
        stem = os.path.splitext(os.path.basename(path))[0]
        table = pq.read_table(
            path, columns=["quiz", "names", "knight_knave", "solution", "index"]
        )
        for row in table.to_pylist():
            names = _literal(row["names"])
            solution = _literal(row["solution"])
            roles = _literal(row["knight_knave"])
            if not isinstance(names, list) or not isinstance(solution, list):
                continue
            if not isinstance(roles, dict):
                continue
            if len(names) < 4 or len(names) != len(solution):
                continue
            quiz = (row["quiz"] or "").strip()
            if not quiz:
                continue
            truth = roles.get("knight") or "knight"
            lie = roles.get("knave") or "knave"
            bases[POOL_KEYS[ds.POOL_KK].format(stem=stem, index=row["index"])] = {
                "question": quiz,
                "answer": " ".join(truth if flag else lie for flag in solution),
            }
    return bases


def read_raw_bases(raw_dir: str) -> dict[str, dict]:
    """All three present pools under one ``uid -> {question, answer}`` map."""
    bases: dict[str, dict] = {}
    for reader in (read_sum_bases, read_umwp_bases, read_kk_bases):
        bases.update(reader(raw_dir))
    return bases


# ---------------------------------------------------------------------------
# byte-level label re-derivation
# ---------------------------------------------------------------------------


def content_words(text: str) -> set[str]:
    """The module's ``_base_content_words``, written out: content words, no digits."""
    return {
        token.casefold()
        for token in schema.words(text)
        if schema.is_content_word(token) and not token.isdigit()
    }


def topic_words_of(text: str) -> set[str]:
    """Content words minus the generic vocabulary -- the topicality axis' side."""
    return content_words(text) - ds.GENERIC_TOPIC_WORDS


def mentions(text: str, phrase: str) -> bool:
    """Whole-phrase occurrence of ``phrase`` in ``text``, punctuation-tolerant."""
    haystack = normalise(text).casefold()
    needle = normalise(phrase).casefold()
    if not needle:
        return False
    start = 0
    while True:
        found = haystack.find(needle, start)
        if found < 0:
            return False
        before = haystack[found - 1] if found else " "
        end = found + len(needle)
        after = haystack[end] if end < len(haystack) else " "
        if not before.isalnum() and not after.isalnum():
            return True
        start = found + 1


_PLACEHOLDER_RE = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")
# The number fills are drawn from a pool the builder filters to plain decimals
# (`distractor_synth._NUMBER_RE`), so the recovery can require exactly that shape
# and let the role group take everything else.
_NUMBER_FILL_RE = r"\d+(?:\.\d+)?"


@dataclasses.dataclass(frozen=True)
class RecoveredFill:
    """A template and the two fills this audit read back out of a sentence.

    ``sentence`` is ``template.replace("{role}", role).replace("{number}", number)``
    with the result stripped -- that equality, not the regex, is what makes the
    recovery a check rather than a guess.
    """

    template: str
    skeleton: str
    role: str
    number: str | None


class TemplateMatcher:
    """Match a rendered sentence back to its ``(template, role, number)``, or refuse.

    The builder makes every distractor by substituting into a template, so the fill
    can be recovered exactly -- and recovering it is worth doing, because the
    actor is part of the *substitution* and not part of the sentence.  A topicality
    reading that counted the actor as one of the sentence's own words called 93% of
    this branch ``in_topic`` when it stores 44%, every overlapped row being one
    where the actor is by construction a word of the base.

    Recovery is also the check: a sentence that is not the render of any safe
    template, or that two distinct ``(template, role, number)`` triples both
    reproduce, yields no fill and is reported (L1i) rather than guessed at.
    """

    def __init__(self, templates: list) -> None:
        self.role_pool = role_pool_of(templates)
        self._entries: list[tuple[str, list[str], list[str], list[re.Pattern]]] = []
        for template in templates:
            text = template.sentence_template
            if not text or any(entry[0] == text for entry in self._entries):
                continue
            names = _PLACEHOLDER_RE.findall(text)
            literals = _PLACEHOLDER_RE.split(text)
            self._entries.append((text, names, literals, self._patterns(literals, names)))

    @property
    def size(self) -> int:
        """How many distinct safe templates a sentence may be a render of."""
        return len(self._entries)

    @staticmethod
    def _patterns(literals: list[str], names: list[str]) -> list[re.Pattern]:
        """Two readings of each boundary: the shortest role, and the longest.

        A wrong split cannot survive the re-render equality in :meth:`recover`, so
        the greedy variant exists only for the sentences the non-greedy one cannot
        reach (a fill that itself contains a literal part).
        """
        patterns = []
        for greedy in (False, True):
            pattern = ""
            for index, literal in enumerate(literals):
                pattern += re.escape(literal)
                if index < len(names):
                    name = names[index]
                    if name.startswith("{number"):
                        pattern += f"({_NUMBER_FILL_RE})"
                    else:
                        pattern += "(.+)" if greedy else "(.+?)"
            patterns.append(re.compile(r"\A" + pattern + r"\Z", re.DOTALL))
        return patterns

    @staticmethod
    def render(text: str, role: str, number: str | None) -> str:
        """The builder's own substitution, replayed: ``str.replace``, then strip."""
        return text.replace("{role}", role).replace("{number}", number or "").strip()

    def candidates(self, sentence: str) -> list[RecoveredFill]:
        """Every ``(template, role, number)`` triple whose re-render is ``sentence``.

        Usually there is exactly one.  Ambiguity is real and benign, though: two
        templates may differ only by a literal that the other's role group can
        swallow, so ``"The salary of Katja's mother, a professor, is $9 per
        month."`` is a render both of ``The salary of {role}, a professor, is
        ${number} per month.`` and of ``The salary of {role} is ${number} per
        month.``  Which one the builder drew is not recoverable from the bytes, so
        the audit keeps both and requires their *labels* to agree (see
        :func:`derive_labels`) rather than picking one.
        """
        text = (sentence or "").strip()
        if not text:
            return []
        found: dict[tuple[str, str, str | None], RecoveredFill] = {}
        for template, names, literals, patterns in self._entries:
            if not all(literal in text for literal in literals):
                continue
            for pattern in patterns:
                match = pattern.match(text)
                if match is None:
                    continue
                role: str | None = None
                number: str | None = None
                consistent = True
                for name, value in zip(names, match.groups()):
                    if name.startswith("{number"):
                        consistent = consistent and (number is None or number == value)
                        number = number or value
                    else:
                        consistent = consistent and (role is None or role == value)
                        role = role or value
                if not consistent or not role:
                    continue
                if self.render(template, role, number) != text:
                    continue
                found[(template, role, number)] = RecoveredFill(
                    template=template,
                    skeleton=_PLACEHOLDER_RE.sub(" ", template),
                    role=role,
                    number=number,
                )
        return list(found.values())

    def recover(self, sentence: str) -> RecoveredFill | None:
        """The unique fill that reproduces ``sentence``, or ``None`` if not exactly one."""
        found = self.candidates(sentence)
        return found[0] if len(found) == 1 else None


def derive_sentence_label(question: str, fill: RecoveredFill) -> str:
    """Exact: in-topic iff the sentence's *own* topic words meet the base's.

    The skeleton -- the template with both placeholders blanked -- is the sentence
    minus its substitution, which is what "this sentence is about a familiar topic"
    has to mean.  Reading the rendered sentence instead counts the actor, and an
    overlapped actor is a base word by construction.
    """
    return "in_topic" if topic_words_of(fill.skeleton) & content_words(question) else "out_topic"


def derive_role_label(question: str, fill: RecoveredFill) -> str:
    """Exact: an overlapped fill is an actor the base itself names.

    ``mentions`` is the word-boundary reading of "the base carries this name"; the
    builder's pool construction uses a plain substring test, which is strictly
    weaker (``"Alex"`` is a substring of ``"Alexandra"``).  So a disagreement found
    here is a real one and not an artefact of spelling the same rule two ways.
    """
    return "overlapped" if mentions(question, fill.role) else "nonoverlapped"


def derive_number_label(question: str, fill: RecoveredFill) -> str | None:
    """Exact on the recovered fill; ``None`` only when the base has no numbers.

    The recovery is what makes this exact rather than a necessary condition: with
    the fill in hand, a template that also hardcodes a number no longer muddies the
    reading, because only the substituted one is classified.
    """
    if not fill.number:
        return None
    values = [float(token) for token in _plain_numbers(question)]
    if not values:
        return None
    low, high = min(values), max(values)
    try:
        value = float(fill.number)
    except ValueError:
        return None
    return "in_range" if low <= value <= high else "out_range"


def derive_labels(question: str, fills: list[RecoveredFill]) -> dict[str, str | None]:
    """The three labels from the recovered fill(s), or ``None`` where undecidable.

    When the sentence is a render of several templates the labels are taken from
    the candidates that agree: an axis is decided only if every candidate reading
    gives it the same label, so an ambiguity can never be resolved in whichever
    direction happens to flatter the branch.
    """
    if not fills:
        return {axis: None for axis in ("role_label", "number_label", "sentence_label")}
    derived: dict[str, str | None] = {}
    for axis, function in (
        ("role_label", derive_role_label),
        ("number_label", derive_number_label),
        ("sentence_label", derive_sentence_label),
    ):
        values = {function(question, fill) for fill in fills}
        derived[axis] = values.pop() if len(values) == 1 else None
    return derived


def _plain_numbers(text: str) -> list[str]:
    """The numbers this audit re-reads, through the shared tokeniser."""
    return list(schema.numbers_in(text))


def role_pool_of(templates: list) -> list[str]:
    """The substitutable actors the branch draws from: the templates' own roles."""
    return sorted({template.role for template in templates if template.role})


def positive_control(seed: int = 0) -> float:
    """A separable synthetic task the NB must solve, else the estimator is broken."""
    rng = random.Random(seed)
    texts: list[str] = []
    labels: list[int] = []
    left = [f"lword{i}" for i in range(12)]
    right = [f"rword{i}" for i in range(12)]
    for index in range(200):
        label = index % 2
        pool = left if label == 0 else right
        texts.append(" ".join(rng.sample(pool, 6)))
        labels.append(label)
    accuracy, _vocab = bow_nb_out_of_fold(texts, labels, seed=seed)
    return accuracy


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


class Audit:
    """Collects ``PASS`` / ``FAIL`` lines and the exit status."""

    def __init__(self) -> None:
        self.failures = 0
        self.lines: list[str] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        status = "PASS" if ok else "FAIL"
        if not ok:
            self.failures += 1
        suffix = f"  {detail}" if detail else ""
        self.lines.append(f"{status}  {name}{suffix}")
        return ok

    def note(self, name: str, detail: str) -> None:
        self.lines.append(f"----  {name}  {detail}")

    def flush(self) -> None:
        for line in self.lines:
            print(line)


def _prompt(row: dict) -> str:
    """The prompt as the model sees it, whitespace-normalised."""
    return normalise(row["prompt"][0]["content"])


def _payload(row: dict) -> dict:
    return json.loads(row["reward_model"]["ground_truth"])


def _uid(row: dict) -> str:
    task_id = row["extra_info"].get("task_id") or ""
    return task_id[len(TASK_ID_PREFIX) :] if task_id.startswith(TASK_ID_PREFIX) else ""


def audit(
    rows: list[dict],
    raw_bases: dict[str, dict],
    matcher: TemplateMatcher,
    *,
    seed: int,
    replay: bool,
    sample: int,
) -> Audit:
    audit = Audit()
    total = len(rows)

    # -- L0: schema contract -------------------------------------------------
    violations: list[str] = []
    for row in rows:
        violations.extend(schema.validate_row(row))
        if len(violations) >= 10:
            break
    audit.check(
        "L0 schema.validate_row on every row",
        not violations,
        violations[0] if violations else f"{total} rows",
    )

    # -- L1: answer-invariance certificate -----------------------------------
    unreachable: list[str] = []
    not_solvable: list[str] = []
    gold_mismatch: list[str] = []
    unjoined: list[str] = []
    question_mismatch: list[str] = []
    sentence_missing: list[str] = []
    not_restored: list[str] = []
    sentence_in_base: list[str] = []
    overlap_contradiction: list[str] = []
    unrecovered: list[str] = []
    recovered_by_uid: dict[str, list[RecoveredFill]] = {}

    for row in rows:
        info = row["extra_info"]
        uid = _uid(row)
        base = raw_bases.get(uid)
        if base is None:
            unjoined.append(uid or repr(info.get("task_id")))
            continue
        recorded = info.get("paired_original_text") or ""
        if normalise(recorded) != normalise(base["question"]):
            question_mismatch.append(uid)
        prompt = _prompt(row)
        sentence = normalise(info.get("distractor_text") or "")
        if not sentence or sentence not in prompt:
            sentence_missing.append(uid)
        else:
            residual = normalise(prompt.replace(sentence, " "))
            if normalise(base["question"]) not in residual:
                not_restored.append(uid)
        if sentence and sentence in normalise(base["question"]):
            sentence_in_base.append(uid)

        payload = _payload(row)
        if not payload.get("solvable"):
            not_solvable.append(uid)
        if payload.get("answer") != base["answer"]:
            gold_mismatch.append(uid)

        # Recover the substitution before reading any label off the text: the
        # actor belongs to the fill, and a reading that counts it as one of the
        # sentence's own words is wrong in the direction that hides an overlap.
        fills = matcher.candidates(info.get("distractor_text") or "")
        if not fills:
            unrecovered.append(uid)
        else:
            recovered_by_uid[uid] = fills
            labels = info.get("distractor_labels") or {}
            if labels.get("role_label") == "overlapped":
                # Restating a fixed property is the one way this sentence could
                # stop being inert, so the template's *own* words must be
                # lexically disjoint from the base -- which is precisely what the
                # builder's guard tests, and what the design's structural law
                # (overlapped implies out_topic) follows from.  Every candidate
                # reading is required to be disjoint, because a guard is the wrong
                # place to be permissive about an ambiguity.
                shared = set()
                for fill in fills:
                    shared |= content_words(fill.skeleton) & content_words(base["question"])
                if shared:
                    overlap_contradiction.append(f"{uid}: {sorted(shared)[:3]}")

    def summarise(problems: list[str], limit: int = 3) -> str:
        return f"{len(problems)}/{total}" + (f"; e.g. {problems[:limit]}" if problems else "")

    audit.check("L1a every row joins to a raw upstream row by key", not unjoined, summarise(unjoined))
    audit.check(
        "L1b paired_original_text is the raw row's own question",
        not question_mismatch,
        summarise(question_mismatch),
    )
    audit.check(
        "L1c gold answer equals the raw row's own gold",
        not gold_mismatch,
        summarise(gold_mismatch),
    )
    audit.check("L1d every row is marked solvable", not not_solvable, summarise(not_solvable))
    audit.check(
        "L1e distractor sentence occurs verbatim in the prompt",
        not sentence_missing,
        summarise(sentence_missing),
    )
    audit.check(
        "L1f deleting the sentence restores the base question",
        not not_restored,
        summarise(not_restored),
    )
    audit.check(
        "L1g the sentence is an insertion, not part of the base",
        not sentence_in_base,
        summarise(sentence_in_base),
    )
    audit.check(
        "L1h an overlapped actor shares no content word with the base",
        not overlap_contradiction,
        summarise(overlap_contradiction),
    )
    audit.check(
        "L1i every sentence is a render of a safe template, fill recoverable",
        not unrecovered,
        summarise(unrecovered),
    )
    template_roles = {role.casefold() for role in matcher.role_pool}
    actors = {fill.role for fills in recovered_by_uid.values() for fill in fills}
    ambiguous = sum(1 for fills in recovered_by_uid.values() if len(fills) > 1)
    audit.note(
        "L1i recovered fills",
        f"{len(actors)} distinct actors over {len(recovered_by_uid)} rows; "
        f"{sum(1 for actor in actors if actor.casefold() in template_roles)} of the actors are in "
        f"the templates' own {len(template_roles)}-name pool, the rest are the base's own names; "
        f"{ambiguous} rows are ambiguous (more than one template reproduces the sentence)",
    )

    # -- L2: distribution ----------------------------------------------------
    task_ids = [row["extra_info"].get("task_id") for row in rows]
    prompts = [_prompt(row) for row in rows]
    duplicate_ids = [key for key, count in collections.Counter(task_ids).items() if count > 1]
    duplicate_prompts = [key for key, count in collections.Counter(prompts).items() if count > 1]
    audit.check(
        "L2a task_id is unique",
        not duplicate_ids,
        f"{len(set(task_ids))} unique of {total}"
        + (f"; e.g. {duplicate_ids[:3]}" if duplicate_ids else ""),
    )
    audit.check(
        "L2b prompt is unique",
        not duplicate_prompts,
        f"{len(set(prompts))} unique of {total}",
    )
    used_bases = collections.Counter(_uid(row) for row in rows)
    reused = [uid for uid, count in used_bases.items() if count > 1]
    audit.check(
        "L2c no base is used twice",
        not reused,
        f"{len(used_bases)} distinct bases" + (f"; e.g. {reused[:3]}" if reused else ""),
    )

    by_pool = collections.defaultdict(list)
    for row in rows:
        by_pool[row["data_source"]].append(row)
    for data_source in sorted(by_pool):
        for axis in ds.AXES:
            present = sorted(
                {row["extra_info"]["distractor_labels"][axis] for row in by_pool[data_source]}
            )
            audit.note(f"L2d {data_source} {axis}", f"labels present: {present}")

    per_axis = {}
    for axis in ds.AXES:
        counts = collections.Counter(
            row["extra_info"]["distractor_labels"][axis] for row in rows
        )
        achieved = {label: counts.get(label, 0) / total for label in ds.AXIS_TARGETS[axis]}
        per_axis[axis] = achieved
        detail = " ".join(
            f"{label}={achieved[label]:.3f} (target {ds.AXIS_TARGETS[axis][label]:.2f})"
            for label in ds.AXIS_TARGETS[axis]
        )
        audit.check(
            f"L2e stored {axis} within +-{ds.AXIS_TOLERANCE:.2f} of target",
            all(
                abs(achieved[label] - fraction) <= ds.AXIS_TOLERANCE
                for label, fraction in ds.AXIS_TARGETS[axis].items()
            ),
            detail,
        )

    # The same marginals, recomputed from the bytes rather than the stored labels.
    derived_counts: dict[str, collections.Counter] = {
        axis: collections.Counter() for axis in ds.AXES
    }
    decisive: dict[str, int] = {axis: 0 for axis in ds.AXES}
    disagreements: dict[str, list[str]] = {axis: [] for axis in ds.AXES}
    for row in rows:
        info = row["extra_info"]
        labels = info.get("distractor_labels") or {}
        uid = _uid(row)
        question = info.get("paired_original_text") or ""
        derived = derive_labels(question, recovered_by_uid.get(uid) or [])
        for axis, label in derived.items():
            if label is None:
                continue
            decisive[axis] += 1
            derived_counts[axis][label] += 1
            if label != labels.get(axis):
                disagreements[axis].append(f"{uid}: stored {labels.get(axis)!r} vs text {label!r}")
    for axis in ds.AXES:
        counts = derived_counts[axis]
        measured = decisive[axis]
        shares = " ".join(
            f"{label}={counts.get(label, 0) / measured:.3f}" for label in ds.AXIS_TARGETS[axis]
        )
        audit.note(
            f"L2f {axis} re-derived from the bytes",
            f"decisive on {measured}/{total} rows; {shares}" if measured else "no decisive row",
        )
        audit.check(
            f"L2g {axis} stored label agrees with the text where decisive",
            not disagreements[axis],
            f"{measured - len(disagreements[axis])}/{measured} agree"
            + (f"; e.g. {disagreements[axis][:2]}" if disagreements[axis] else ""),
        )
        gate = measured >= total // 2
        audit.check(
            f"L2h {axis} re-derived marginal within +-{ds.AXIS_TOLERANCE:.2f} of target",
            (not gate)
            or all(
                abs(counts.get(label, 0) / measured - fraction) <= ds.AXIS_TOLERANCE
                for label, fraction in ds.AXIS_TARGETS[axis].items()
            ),
            f"decisive on {measured}/{total} rows"
            + ("" if gate else " (too few decisive rows to gate)")
            if measured
            else "not decidable",
        )

    # -- L2i: replay determinism --------------------------------------------
    if replay:
        observed = observed_by_pool(rows)
        quota = {pool: int(observed.get(pool, 0)) for pool in ds.POOLS}
        pools, _notes = ds.load_pools(ds.DEFAULT_RAW_DIR, None)
        templates, _report = ds.load_safe_templates()
        replayed, _replay_report = ds.synthesise(pools, templates=templates, quota=quota, seed=seed)
        replay_map = {row["extra_info"]["task_id"]: row for row in replayed}
        mismatched = [
            row["extra_info"]["task_id"]
            for row in rows
            if replay_map.get(row["extra_info"]["task_id"], {}).get("extra_info", {}).get(
                "distractor_text"
            )
            != row["extra_info"]["distractor_text"]
        ]
        audit.check(
            "L2i the branch replays byte-identically from its seed",
            len(replayed) == total and not mismatched,
            f"replayed {len(replayed)} rows with seed {seed} and quota {quota}"
            + (f"; {len(mismatched)} rows differ" if mismatched else ""),
        )
    else:
        audit.note("L2i replay determinism", "skipped (--skip-replay)")

    # -- L3: anti-cheat ------------------------------------------------------
    audit.note(
        "L3a solvable/unsolvable Naive Bayes",
        "inapplicable: the branch is solvable-only, so the design's binary label "
        "has no negative class",
    )
    control = positive_control(seed=seed)
    audit.check(
        "L3b the estimator passes a positive control",
        control >= _NB_POSITIVE_CONTROL_MIN,
        f"separable synthetic task reads {control:.3f} (chance {_NB_CHANCE:.2f})",
    )
    for axis in ds.AXES:
        values = [row["extra_info"]["distractor_labels"][axis] for row in rows]
        present = sorted(set(values))
        if len(present) != 2:
            audit.note(f"L3c {axis}", f"reference number skipped: labels {present} are not binary")
            continue
        low, high = present
        labels = [0 if value == low else 1 for value in values]
        accuracy, vocab_size = bow_nb_out_of_fold(prompts, labels, seed=seed)
        audit.note(
            f"L3c {axis} (reference, not gated)",
            f"labels {low}/{high} = {labels.count(0)}/{labels.count(1)}; 5-fold "
            f"out-of-fold balanced accuracy = {accuracy:.3f} (chance {_NB_CHANCE:.2f}); "
            f"vocab = {vocab_size} tokens with train doc support >= "
            f"{_NB_MIN_DOC_SUPPORT}",
        )

    # -- reporting only: the unbuilt fourth pool -----------------------------
    quota_total = sum(ds.DEFAULT_POOL_QUOTA.values())
    audit.note(
        "summary",
        f"built {total}/{quota_total} rows of the design quota "
        f"({total / quota_total:.0%})",
    )
    for pool in ds.POOLS:
        want = ds.DEFAULT_POOL_QUOTA.get(pool, 0)
        got = observed_by_pool(rows).get(pool, 0)
        if got < want:
            audit.note(
                f"summary {pool}",
                f"{got}/{want} rows -- short by {want - got}"
                + (
                    " (no stage-1 pool on this machine: --stage1-path was never given)"
                    if pool == ds.POOL_MAIN
                    else ""
                ),
            )
    return audit


def observed_by_pool(rows: list[dict]) -> dict[str, int]:
    """Rows per synthesis pool, read back from ``data_source``."""
    counts = collections.Counter(
        POOL_OF_DATA_SOURCE.get(row["data_source"]) for row in rows
    )
    return {pool: int(count) for pool, count in counts.items() if pool}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", default=DEFAULT_ROWS, help="distractor_synth.py output parquet")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR, help="raw upstream files")
    parser.add_argument("--seed", type=int, default=42, help="the seed the branch was built with")
    parser.add_argument("--skip-replay", action="store_true", help="skip the determinism replay")
    parser.add_argument("--sample", type=int, default=50, help="unused; kept for CLI parity")
    args = parser.parse_args()

    rows = schema.read_parquet_rows(args.rows)
    if not rows:
        raise SystemExit(f"no rows in {args.rows}")
    raw_bases = read_raw_bases(args.raw_dir)
    templates, _report = ds.load_safe_templates(args.raw_dir)
    matcher = TemplateMatcher(templates)

    print(f"auditing {len(rows)} rows from {args.rows}")
    print(f"raw rows available to join against: {len(raw_bases)}")
    print(f"safe templates the fills are recovered against: {matcher.size}")
    result = audit(
        rows,
        raw_bases,
        matcher,
        seed=args.seed,
        replay=not args.skip_replay,
        sample=args.sample,
    )
    result.flush()

    if result.failures:
        raise SystemExit(f"{result.failures} check(s) FAILED")
    print("all checks passed")


if __name__ == "__main__":
    main()
