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
"""Audit the GSM-IC branch parquet produced by ``gsm_ic_adapter.py``.

Three layers, printed as ``PASS`` / ``FAIL`` per check; the process exits
non-zero if any check fails.  Nothing here trusts the adapter: the gold is
re-derived from the raw GSM8K cross-check corpus with this file's own arithmetic
walker and its own corpus loader (the two do not import each other), and every
per-row cue is recomputed from the bytes that will actually reach the model.

L1 -- label certificate
-----------------------

For a sample of at least 50 rows (``--sample``, default 50), independently prove
the gold:

* **recompute the answer.**  The row's ``extra_info.paired_original_text`` is
  joined by exact text to the GSM8K corpus; every ``<<expr=val>>`` calculator
  annotation in the joined solution is re-evaluated here and must reproduce the
  recorded value; the solution must carry exactly one ``#### N``; and ``N`` must
  equal the row's ``ground_truth`` answer and appear in the derivation.  A row
  whose gold cannot be recomputed is a FAIL, not a warning.
* **re-locate the pointed-at span.**  The row's stored
  ``extra_info.distractor_text`` must occur verbatim in the row's own rendered
  question, and must not occur in ``paired_original_text``.

L2 -- gold uniqueness
---------------------

The certificate form depends on the source's shape, and this source's shape
makes two of the three usual forms vacuous -- which is *reported*, not hidden:

* "the gold is the only admissible option" is vacuous: the source is fully
  solvable and option-less, so the admissible set is not an option set.  What is
  asserted instead is the invariant that replaces it: the gold is derived from
  the **paired original** question, i.e. from the problem with the distractor
  removed, so the inserted sentence is not needed for the answer and cannot
  change it.
* "the deleted value is really absent" is vacuous: this source deletes nothing.
* The uniqueness question that *does* bite is answered loudly: no row's inserted
  number may be numerically equal to its gold, because "the answer is the number
  from the sentence that does not belong" would then be a second admissible
  value.  The build drops those rows; this check fails the run if one survives.

L3 -- anti-cheat
----------------

GSM-IC contributes to **one** side only (measured: every row is
``solvable=true``, zero unsolvable rows), so the specified 5-fold bag-of-words
Naive Bayes between the *solvable* and *unsolvable* label has no negative class
and is reported as inapplicable.  Two consequences, both handled explicitly:

* the "every option occurs verbatim in the question" and "all options have equal
  token length" cues are also vacuous -- there is no option block -- so the run
  asserts the no-option invariant and then runs the cues that *do* exist for a
  distractor branch: the inserted sentence is verbatim in the row's own question
  (nothing in the prompt comes from anywhere else, which is what kills the
  cross-question leaks the MiP/FalseQA audits measured at 88.5-100%), and no
  single inserted number can be read off as the answer.
* the Naive Bayes is still implemented as specified (numpy only -- sklearn is not
  installed) and run on the only binary labels this source carries: the three D17
  difficulty axes in ``extra_info.distractor_labels``.  Those readings are
  *reference numbers*, printed and never gated, because they are not the
  contract check.  The estimator itself is gated by a positive control on
  synthetic separable data, so a broken estimator cannot masquerade as a clean
  reading -- the failure mode the UMWP recon report documents for an unfiltered
  multinomial NB (class score flips sign out of fold).

Vocabulary is restricted to tokens with training-fold document support >= 5, as
specified.

The L0 layer asserts :func:`schema.validate_row` on every row.
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import os
import random
import re
import sys

import numpy as np

try:
    import schema
except ImportError:  # pragma: no cover - exercised by running as a script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import schema

DEFAULT_ROWS = "/home/charles/data/reasoning_rl/halluc/built/gsmic.parquet"
DEFAULT_RAW_DIR = "/home/charles/data/reasoning_rl/halluc/raw/gsmic"

CROSS_CHECK_FILES = ("gsm8k_train.jsonl", "gsm8k_test.jsonl")

_ANNOTATION_RE = re.compile(r"<<([^<>]+?)=([^<>]+?)>>")
_PLAIN_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_WHITESPACE_RE = re.compile(r"\s+")
_FINAL_MARKER = "####"
_CALCULATOR_DECIMALS = 2

# Naive Bayes settings (design doc section 9: 5 folds, chance 0.50, gate 0.60).
NB_FOLDS = 5
NB_MIN_DOC_SUPPORT = 5
NB_SMOOTHING = 1.0
NB_CHANCE = 0.50
NB_POSITIVE_CONTROL_MIN = 0.90

#: Section 4.7's three label axes are "naturally balanced" and its verify bullet
#: promises the balance is asserted (section 8 lists it too), so each labelled
#: share must sit within this distance of its documented target, and the share
#: the generator leaves as ``n/a`` must stay small enough for the reading to mean
#: anything.
LABEL_BALANCE_TOLERANCE = 0.05
LABEL_BALANCE_UNLABELLED_MAX = 0.05
#: One row of a small fixture is worth more than the tolerance (a 12-row build
#: moves a share by 8pt per row), so the balance gate only fires once the artifact
#: is big enough for a 5pt band to mean something.  The real pool is 800 rows.
LABEL_BALANCE_MIN_ROWS = 100
LABEL_BALANCE_TARGETS = {
    "sentence_label": {"in_topic": 0.45, "out_topic": 0.55},
    "role_label": {"overlapped": 0.50, "nonoverlapped": 0.50},
    "number_label": {"in_range": 0.50, "out_range": 0.50},
}


# ---------------------------------------------------------------------------
# independent arithmetic certificate (deliberately not imported from the adapter)
# ---------------------------------------------------------------------------


class AnnotationError(ValueError):
    """An annotation outside the measured ``+ - * /`` grammar."""


def _walk(node: ast.AST) -> float:
    """Re-evaluate one annotation node with a closed whitelist (no ``eval``)."""
    if isinstance(node, ast.Expression):
        return _walk(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise AnnotationError(f"non-numeric constant {node.value!r}")
        return float(node.value)
    if isinstance(node, ast.UnaryOp):
        value = _walk(node.operand)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
        raise AnnotationError(f"unsupported unary operator {type(node.op).__name__}")
    if isinstance(node, ast.BinOp):
        left = _walk(node.left)
        right = _walk(node.right)
        op = node.op
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            if right == 0:
                raise AnnotationError("division by zero")
            return left / right
        raise AnnotationError(f"unsupported binary operator {type(op).__name__}")
    raise AnnotationError(f"unsupported expression node {type(node).__name__}")


def recompute_annotation(expression: str) -> float:
    """Re-evaluate ``<<expression=...>>`` (commas and ``$`` stripped first)."""
    cleaned = (expression or "").replace(",", "").replace("$", "").strip()
    if not cleaned:
        raise AnnotationError("empty expression")
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as exc:
        raise AnnotationError(f"cannot parse {expression!r}: {exc}") from exc
    return _walk(tree)


def plain_number(text: str) -> float | None:
    """First numeric literal in ``text`` (commas and ``$`` stripped), else None."""
    match = _PLAIN_NUMBER_RE.search((text or "").replace(",", "").replace("$", ""))
    return float(match.group(0)) if match else None


def _same(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return False
    return round(left, _CALCULATOR_DECIMALS) == round(right, _CALCULATOR_DECIMALS)


def load_gsm8k(raw_dir: str) -> dict[str, str]:
    """Whitespace-normalised GSM8K question -> full solution text (train + test)."""
    corpus: dict[str, str] = {}
    for name in CROSS_CHECK_FILES:
        path = os.path.join(raw_dir, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"cross-check corpus {path} is missing; L1 cannot re-derive any gold "
                f"without {CROSS_CHECK_FILES} in {raw_dir}"
            )
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                corpus.setdefault(normalise(record["question"]), record["answer"])
    return corpus


def normalise(text: str) -> str:
    """Collapse whitespace runs to one space and strip the ends."""
    return _WHITESPACE_RE.sub(" ", text or "").strip()


def theorem_gold(solution: str, gold: float) -> tuple[bool, str]:
    """Re-derive ``gold`` from one joined GSM8K solution.

    Returns ``(ok, detail)``.  Every ``<<expr=val>>`` must recompute; there must
    be exactly one ``####`` marker; its value must be ``gold``; and ``gold`` must
    be visible in the derivation as a computed value or a literal.
    """
    pairs = _ANNOTATION_RE.findall(solution)
    if not pairs:
        return False, "joined solution carries no <<expr=val>> annotation"
    for expression, stated in pairs:
        try:
            recomputed = recompute_annotation(expression)
        except AnnotationError as exc:
            return False, f"<<{expression}={stated}>> not evaluable: {exc}"
        if not _same(recomputed, plain_number(stated)):
            return False, f"<<{expression}={stated}>> recomputes to {recomputed}"
    if solution.count(_FINAL_MARKER) != 1:
        return False, f"{solution.count(_FINAL_MARKER)} '####' markers"
    terminal = plain_number(solution.split(_FINAL_MARKER)[-1])
    if not _same(terminal, gold):
        return False, f"terminal {terminal} != gold {gold}"
    for _expression, stated in pairs:
        if _same(plain_number(stated), gold):
            return True, "gold is a computed chain value"
    body = _ANNOTATION_RE.sub(" ", solution.split(_FINAL_MARKER)[0])
    literal = str(int(gold)) if gold == int(gold) else repr(gold)
    if re.search(rf"(?<![\d.]){re.escape(literal)}(?![\d])", body):
        return True, "gold is a literal in the derivation text"
    return False, f"gold {gold} is in neither the chain values nor the derivation text"


# ---------------------------------------------------------------------------
# numpy multinomial Naive Bayes (sklearn is not installed)
# ---------------------------------------------------------------------------


def balanced_accuracy(y_true: list[int], y_pred: list[int]) -> float:
    """Mean of the per-class recalls (chance is 0.50 for a balanced binary task)."""
    classes = sorted(set(y_true))
    if len(classes) < 2:
        return float("nan")
    recalls = []
    for label in classes:
        total = sum(1 for value in y_true if value == label)
        hit = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        recalls.append(hit / total if total else 0.0)
    return sum(recalls) / len(recalls)


def _bow_matrix(
    texts: list[str], vocab: dict[str, int]
) -> np.ndarray:
    """Document x vocab token-count matrix, restricted to ``vocab``."""
    matrix = np.zeros((len(texts), len(vocab)), dtype=np.float64)
    for row, text in enumerate(texts):
        for token in schema.words(text):
            key = token.casefold()
            column = vocab.get(key)
            if column is not None:
                matrix[row, column] += 1.0
    return matrix


def _fit_predict(
    train_matrix: np.ndarray, train_labels: list[int], test_matrix: np.ndarray
) -> list[int]:
    """Multinomial NB in log space with Laplace smoothing, fit here and only here.

    The vocabulary is fixed by the caller from the training fold, so the features
    are counted before this function runs; this only fits the class-conditional
    token distributions.
    """
    classes = sorted(set(train_labels))
    counts = np.zeros((len(classes), train_matrix.shape[1]), dtype=np.float64)
    for index, label in enumerate(classes):
        mask = np.array([value == label for value in train_labels])
        counts[index] = train_matrix[mask].sum(axis=0)
    totals = counts.sum(axis=1, keepdims=True)
    vocab_size = train_matrix.shape[1]
    log_likelihood = np.log(
        (counts + NB_SMOOTHING) / (totals + NB_SMOOTHING * vocab_size)
    )
    priors = np.array(
        [np.log(max((np.array(train_labels) == label).sum(), 1) / len(train_labels)) for label in classes]
    )
    scores = test_matrix @ log_likelihood.T + priors
    return [classes[int(index)] for index in scores.argmax(axis=1)]


def bow_nb_out_of_fold(
    texts: list[str],
    labels: list[int],
    *,
    folds: int = NB_FOLDS,
    seed: int = 0,
    min_doc_support: int = NB_MIN_DOC_SUPPORT,
) -> tuple[float, int]:
    """5-fold out-of-fold balanced accuracy of a BoW multinomial NB.

    The vocabulary is rebuilt inside every fold from the *training* documents
    only, keeping tokens with training document support >= ``min_doc_support``.
    That filter is not cosmetic: the UMWP recon report documents that an
    unfiltered multinomial NB on this shape of data is a numerically broken
    estimator whose class score flips sign out of fold, so an unfiltered
    sub-0.50 reading would not be evidence of anything.

    Returns ``(balanced_accuracy, vocabulary_size)`` where the vocabulary size is
    the mean over folds.
    """
    indices = list(range(len(texts)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    fold_of = {index: position % folds for position, index in enumerate(indices)}
    predictions: list[int | None] = [None] * len(texts)
    vocab_sizes: list[int] = []

    tokenised = [[token.casefold() for token in schema.words(text)] for text in texts]

    for fold in range(folds):
        train_indices = [i for i in range(len(texts)) if fold_of[i] != fold]
        test_indices = [i for i in range(len(texts)) if fold_of[i] == fold]
        if not train_indices or not test_indices:
            continue
        support = collections.Counter()
        for index in train_indices:
            support.update(set(tokenised[index]))
        vocab = {
            token: column
            for column, token in enumerate(
                sorted(token for token, count in support.items() if count >= min_doc_support)
            )
        }
        vocab_sizes.append(len(vocab))
        if not vocab:
            predictions_fallback = [collections.Counter(
                labels[i] for i in train_indices
            ).most_common(1)[0][0]] * len(test_indices)
            for index, prediction in zip(test_indices, predictions_fallback):
                predictions[index] = prediction
            continue
        train_matrix = _bow_matrix([texts[i] for i in train_indices], vocab)
        test_matrix = _bow_matrix([texts[i] for i in test_indices], vocab)
        fold_predictions = _fit_predict(
            train_matrix, [labels[i] for i in train_indices], test_matrix
        )
        for index, prediction in zip(test_indices, fold_predictions):
            predictions[index] = prediction

    filled_true = []
    filled_pred = []
    for index, prediction in enumerate(predictions):
        if prediction is None:
            continue
        filled_true.append(labels[index])
        filled_pred.append(prediction)
    mean_vocab = int(round(sum(vocab_sizes) / len(vocab_sizes))) if vocab_sizes else 0
    return balanced_accuracy(filled_true, filled_pred), mean_vocab


def _positive_control(seed: int = 0) -> float:
    """A separable synthetic task the NB must solve, else the estimator is broken.

    Two classes whose documents draw words from two disjoint vocabularies: a
    working estimator reads ~1.0.  This is the guard against the failure mode the
    UMWP report documents (an estimator that silently flips sign out of fold),
    without which a low surrogate reading below would be uninterpretable.
    """
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


def _payload(row: dict) -> dict:
    return json.loads(row["reward_model"]["ground_truth"])


def _inserted_number(info: dict) -> float | None:
    """The ``{number}`` value the adapter recorded in ``perturbed_entity_text``.

    The field is ``"role=<r>; number=<n>"`` with either half omitted when the
    row's template carries only the other placeholder, so the number is looked up
    by name rather than by position -- a role name containing a digit must not be
    mistaken for the inserted number.
    """
    for part in (info.get("perturbed_entity_text") or "").split(";"):
        part = part.strip()
        if part.startswith("number="):
            return plain_number(part[len("number=") :])
    return None


def _prompt_question(row: dict) -> str:
    """The question as the model sees it, with the trailing instruction removed.

    Recovered by taking the prompt up to its first blank line -- the templates in
    :mod:`schema` all separate the question from the instruction that way.
    """
    content = row["prompt"][0]["content"]
    return normalise(content.split("\n\n")[0])


def audit(rows: list[dict], corpus: dict[str, str], sample: int, seed: int) -> Audit:
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
        f"{total} rows" if not violations else f"{len(violations)}+ violations: {violations[:3]}",
    )

    # -- L1 / L2 / L3 sample ------------------------------------------------
    sampled = random.Random(seed).sample(rows, min(sample, total)) if total else []
    audit.note(
        "sample",
        f"{len(sampled)} of {total} rows (seed={seed}), the rest audited in bulk below",
    )

    proved = 0
    l1_failures: list[str] = []
    l2_failures: list[str] = []
    solved_bases: list[str] = []
    for row in sampled:
        info = row["extra_info"]
        gold = _payload(row).get("answer")
        gold_value = plain_number(gold or "")
        original = info.get("paired_original_text") or ""
        presented = _prompt_question(row)
        distractor = info.get("distractor_text") or ""

        if gold_value is None:
            l1_failures.append(f"[{info.get('task_id')}] stored gold {gold!r} is not numeric")
            continue

        # L1a: re-derive the answer from the paired original question.
        solution = corpus.get(normalise(original))
        if solution is None:
            l1_failures.append(f"[{info.get('task_id')}] paired original not in GSM8K corpus")
            continue
        ok, detail = theorem_gold(solution, gold_value)
        if not ok:
            l1_failures.append(f"[{info.get('task_id')}] {detail}")
            continue

        # L1b: re-locate the pointed-at span in the question the model receives.
        if not distractor:
            l1_failures.append(f"[{info.get('task_id')}] no distractor_text stored")
            continue
        if distractor not in presented:
            l1_failures.append(
                f"[{info.get('task_id')}] distractor {distractor!r} absent from the prompt"
            )
            continue
        if distractor in normalise(original):
            l1_failures.append(
                f"[{info.get('task_id')}] distractor {distractor!r} also occurs in the original"
            )
            continue
        proved += 1
        solved_bases.append(normalise(original))

        # L2: the gold is invariant to deleting the distractor, and the inserted
        # number is never itself an admissible answer.
        inserted = _inserted_number(info)
        if inserted is not None and _same(inserted, gold_value):
            l2_failures.append(
                f"[{info.get('task_id')}] inserted number {inserted} equals the gold"
            )
        if solution.count(_FINAL_MARKER) != 1:
            l2_failures.append(
                f"[{info.get('task_id')}] joined solution has "
                f"{solution.count(_FINAL_MARKER)} '####' markers (answer not unique)"
            )

    audit.check(
        "L1 label certificate (recomputed answer + re-located span)",
        not l1_failures and proved >= min(sample, total),
        f"proved {proved}/{len(sampled)} sampled rows"
        + ("" if not l1_failures else f"; first: {l1_failures[0]}"),
    )

    # -- L2: uniqueness, with the vacuous forms reported ---------------------
    no_options = all(not row["extra_info"].get("options") for row in rows)
    no_diagnosis = all(not _payload(row).get("has_diagnosis_label") for row in rows)
    all_solvable_here = all(_payload(row).get("solvable") is True for row in rows)
    audit.note(
        "L2 option-uniqueness form",
        "vacuous: this branch is option-less by contract, so the admissible set is "
        "not an option set",
    )
    audit.note(
        "L2 deleted-value form",
        "vacuous: this source deletes nothing, it inserts a distractor",
    )
    audit.check(
        "L2 no admissible second answer (inserted number != gold), all rows",
        no_options
        and no_diagnosis
        and all_solvable_here
        and all(
            not _same(
                _inserted_number(row["extra_info"]),
                plain_number(_payload(row).get("answer") or ""),
            )
            for row in rows
        ),
        f"{total} rows; option-less={no_options} no-diagnosis={no_diagnosis} "
        f"all-solvable={all_solvable_here}",
    )
    audit.check(
        "L2 gold invariant to deleting the distractor (answer derived from the paired original)",
        not l2_failures,
        "all sampled rows" if not l2_failures else l2_failures[0],
    )
    distinct_bases = len({normalise(row["extra_info"].get("paired_original_text") or "") for row in rows})
    repeats = collections.Counter(
        normalise(row["extra_info"].get("paired_original_text") or "") for row in rows
    )
    audit.note(
        "L2 base-question collapse",
        f"{total} rows over {distinct_bases} distinct base questions, "
        f"max {max(repeats.values()) if repeats else 0} rows per base",
    )

    # -- L3: anti-cheat ------------------------------------------------------
    unsolvable_rows = [row for row in rows if _payload(row).get("solvable") is False]
    audit.note(
        "L3 solvable-vs-unsolvable Naive Bayes",
        f"INAPPLICABLE as specified: this source is single-sided "
        f"({total - len(unsolvable_rows)}/{total} solvable, {len(unsolvable_rows)} unsolvable), "
        f"so there is no negative class to cross-validate against",
    )

    option_cue_failures = [row for row in rows if row["extra_info"].get("options")]
    audit.check(
        "L3a option text verbatim in the question",
        not option_cue_failures,
        "vacuous (no option block); instead asserted: every row is option-less "
        f"({total} rows)",
    )

    distractor_failures = [
        row
        for row in rows
        if (row["extra_info"].get("distractor_text") or "") not in _prompt_question(row)
    ]
    audit.check(
        "L3a' distractor sentence verbatim in the row's own presented question",
        not distractor_failures,
        f"{total} rows",
    )

    readoff_failures = []
    for row in rows:
        inserted = _inserted_number(row["extra_info"])
        if _same(inserted, plain_number(_payload(row).get("answer") or "")):
            readoff_failures.append(row["extra_info"].get("task_id"))
    audit.check(
        "L3b' no single-number read-off: gold != inserted number",
        not readoff_failures,
        f"{total} rows" if not readoff_failures else f"{len(readoff_failures)} rows: {readoff_failures[:3]}",
    )

    control = _positive_control(seed)
    audit.check(
        "L3c' Naive Bayes positive control (separable synthetic data)",
        control >= NB_POSITIVE_CONTROL_MIN,
        f"out-of-fold balanced accuracy = {control:.3f} (needs >= {NB_POSITIVE_CONTROL_MIN})",
    )

    texts = [_prompt_question(row) for row in rows]
    for axis in ("sentence_label", "role_label", "number_label"):
        values = [row["extra_info"]["distractor_labels"][axis] for row in rows]
        present = sorted({value for value in values if value != "n/a"})
        if len(present) != 2:
            audit.note(f"L3c {axis}", f"reference number skipped: labels {present} are not binary")
            continue
        low, high = present
        labels = [0 if value == low else 1 for value in values]
        accuracy, vocab_size = bow_nb_out_of_fold(texts, labels, seed=seed)
        audit.note(
            f"L3c {axis} (reference, not gated)",
            f"labels {low}/{high} = {labels.count(0)}/{labels.count(1)}; "
            f"5-fold out-of-fold balanced accuracy = {accuracy:.3f} "
            f"(chance {NB_CHANCE:.2f}); vocab = {vocab_size} tokens with "
            f"train doc support >= {NB_MIN_DOC_SUPPORT}",
        )

    # Section 4.7's balance claim, asserted instead of only printed: the three
    # axes are what makes the synthesised-distractor pool a *graded* control
    # rather than a single easy bucket, and section 8 lists "three-way label
    # balance" as one of the things this audit proves.
    for axis, targets in LABEL_BALANCE_TARGETS.items():
        value_counts = collections.Counter(row["extra_info"]["distractor_labels"][axis] for row in rows)
        unlabelled = sum(count for label, count in value_counts.items() if label not in targets)
        labelled = total - unlabelled
        shares = {label: (value_counts.get(label, 0) / labelled if labelled else 0.0) for label in targets}
        worst = max(abs(shares[label] - target) for label, target in targets.items())
        detail = (
            f"{ {label: round(share, 3) for label, share in shares.items()} } "
            f"(targets {targets}), {unlabelled} n/a of {total} rows"
        )
        if total < LABEL_BALANCE_MIN_ROWS:
            audit.note(
                f"L3d {axis} balance",
                f"not gated on {total} rows (needs >= {LABEL_BALANCE_MIN_ROWS} for a "
                f"{LABEL_BALANCE_TOLERANCE:.2f} band to be meaningful): {detail}",
            )
            continue
        audit.check(
            f"L3d {axis} balance within {LABEL_BALANCE_TOLERANCE:.2f} of section 4.7",
            worst <= LABEL_BALANCE_TOLERANCE and unlabelled / total <= LABEL_BALANCE_UNLABELLED_MAX,
            detail,
        )

    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", default=DEFAULT_ROWS, help="adapter output parquet")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR, help="GSM8K cross-check corpus")
    parser.add_argument("--sample", type=int, default=50, help="rows to prove individually")
    parser.add_argument("--seed", type=int, default=0, help="sampling / CV seed")
    args = parser.parse_args()

    rows = schema.read_parquet_rows(args.rows)
    if not rows:
        raise SystemExit(f"no rows in {args.rows}")
    corpus = load_gsm8k(args.raw_dir)

    print(f"auditing {len(rows)} rows from {args.rows}")
    audit_result = audit(rows, corpus, args.sample, args.seed)
    audit_result.flush()

    if audit_result.failures:
        raise SystemExit(f"{audit_result.failures} check(s) FAILED")
    print("all checks passed")


if __name__ == "__main__":
    main()
