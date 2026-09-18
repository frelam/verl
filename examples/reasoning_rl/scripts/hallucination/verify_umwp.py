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
"""Audit the UMWP rows produced by ``umwp_adapter.py``.

Usage::

    python verify_umwp.py --rows /tmp/halluc_umwp.parquet

Reads only the artifact: every check re-derives its answer from the two texts the
row itself carries (``prompt`` question + ``extra_info.paired_original_text``)
plus the gold fields, and never trusts the adapter's own verdict.  Prints
``PASS``/``FAIL`` per check and exits non-zero if any check fails.

Layers
------

**L1 -- gold certificate** (a sample of >= 50 rows per branch).  Each branch's
gold is re-derived from the row alone:

* ``unsolvable_diag``: the pointed-at span is re-located in the presented
  question at the recorded offset, and swapping the recorded deleted text back in
  for it reproduces the paired answerable question **token for token**.  Proof
  that the recorded option really is the defect that separates the two questions.
* ``unsolvable_bare``: the question really did lose material -- the token
  multiset of the presented question must equal the original's minus the recorded
  deleted text plus the recorded inserted text, and the lost material must be
  gone: a number (``key_information_missing``) or a word sequence
  (``question_missing``).
* ``solvable_judge``: the row's audit answer parses as a finite number and the
  recorded defect is exactly the edit that separates this question from its
  unanswerable partner.  UMWP ships no derivation chain, so the *number* cannot be
  recomputed from the text; it is proved instead against the source file by the
  source anchor below (the source's own ``answer[0]`` for that id), which is the
  only place it can be proved.

**Source anchor.**  Beyond the sample, every row is re-read against
``StandardDataset.jsonl``: the presented question must be the source's own
question for ``extra_info.index``, ``paired_original_text`` must be the source's
question for the partner that the source's own ``relevant_ids`` names, the label
must match the source (an unanswerable row must have ``answer is None`` there, an
answerable row must carry the source's number), and the defect label must be the
category's own name.  Without the raw file the gold cannot be proved, so a missing
``--raw-dir`` is a FAIL, not a skip.

**L2 -- gold uniqueness.**  ``unsolvable_diag``: every distractor must be text
that survived in the paired answerable question -- text that survived cannot be
the defect that made the variant unanswerable, so the gold is the only admissible
option; a second option absent from the original fails loudly.  ``unsolvable_bare``:
the deleted value must really be absent from the presented question.

**L3 -- anti-cheat.**  (a) every option occurs verbatim in the row's own question;
(b) all of a row's options have equal token length; (c) a bag-of-words
multinomial Naive Bayes, 5-fold cross-validated, must not beat chance on the
solvable/unsolvable label by more than 0.10 (out-of-fold balanced accuracy
<= 0.60).  The vocabulary is restricted to tokens with train support >= 5
documents because the recon measured the unfiltered estimator to be numerically
broken on this corpus (its class score flips sign out of fold, so a sub-0.5
reading from it is evidence of nothing).  The raw number is always printed, pass
or fail, together with a shuffled-label control.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import sys

import numpy as np

try:
    import schema
except ImportError:  # pragma: no cover - running as a plain script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import schema

BRANCH_DIAG = schema.BRANCH_UNSOLVABLE_DIAG
BRANCH_BARE = schema.BRANCH_UNSOLVABLE_BARE
BRANCH_JUDGE = schema.BRANCH_SOLVABLE_JUDGE

L3_MAX_BALANCED_ACCURACY = 0.60
L3_MIN_SUPPORT = 5
L1_MIN_SAMPLE = 50


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

    Questions are whitespace-normalised by the adapter, so the first blank line is
    the template boundary.  A prompt without one means the adapter changed the
    rendering and every text check below would silently read the wrong string, so
    it is a hard error.
    """
    content = row["prompt"][0]["content"]
    head, sep, _ = content.partition("\n\n")
    if not sep:
        raise ValueError(f"prompt of {row['extra_info'].get('task_id')} has no template separator")
    return head


def _tokens(text: str) -> list[str]:
    return schema.words(text)


def _counts(tokens: list[str]) -> collections.Counter:
    return collections.Counter(token.lower() for token in tokens)


def _orient(row: dict) -> tuple[str, str]:
    """``(answerable member text, unanswerable member text)`` for this row."""
    question = question_of(row)
    partner = row["extra_info"].get("paired_original_text", "")
    if row["extra_info"].get("solvable"):
        return question, partner
    return partner, question


def _reconstruct(qa: str, qu: str, deleted: str, inserted: str) -> bool:
    """Swap ``inserted`` back for ``deleted`` in ``qu``; must give ``qa``'s tokens.

    Independent of the adapter's diff -- it uses only the two stored texts and the
    stored defect pair, locates the defect itself, and rebuilds.  The comparison is
    at the **token** level, not character level, because UMWP re-punctuates the two
    members of a pair independently: the answerable member of id 2502 ends
    ``them..How`` and its unanswerable partner ``them. How``, so a character-exact
    reconstruction is impossible for a reason that has nothing to do with the gold
    (this is why the adapter certificates are token-level too).
    """
    inserted_tokens = [token.lower() for token in _tokens(inserted)]
    if not inserted_tokens:
        return False
    haystack = [token.lower() for token in _tokens(qu)]
    start = None
    for index in range(len(haystack) - len(inserted_tokens) + 1):
        if haystack[index : index + len(inserted_tokens)] == inserted_tokens:
            start = index
            break
    if start is None:
        return False
    deleted_tokens = [token.lower() for token in _tokens(deleted)]
    rebuilt = haystack[:start] + deleted_tokens + haystack[start + len(inserted_tokens) :]
    return rebuilt == [token.lower() for token in _tokens(qa)]


def _multiset_after_edit(qa: str, qu: str, deleted: str, inserted: str) -> bool:
    """``tokens(qu)`` must equal ``tokens(qa) - deleted + inserted`` as a multiset."""
    expected = _counts(_tokens(qa))
    expected.subtract(_counts(_tokens(deleted)))
    expected.update(_counts(_tokens(inserted)))
    expected = +expected
    return expected == _counts(_tokens(qu))


def _number_lost(deleted: str, qu: str) -> bool:
    """A number the original carried is no longer among the question's numbers.

    Set-based (not substring-based) so that ``10m`` removed from ``100m sections``
    counts as lost -- ``"10m" in "100m"`` is a substring coincidence, ``10`` is not
    one of the question's numbers.
    """
    return bool(set(schema.numbers_in(deleted)) - set(schema.numbers_in(qu)))


def _phrase_absent(deleted: str, qu: str) -> bool:
    """The deleted words no longer appear together anywhere in the question.

    Token-sequence containment, not substring containment: dropping
    ``objects can she juggle`` from a question that still says "she can juggle 2
    more objects" loses the phrase even though it keeps every word.
    """
    needle = [token.lower() for token in _tokens(deleted)]
    if not needle:
        return False
    haystack = [token.lower() for token in _tokens(qu)]
    for index in range(len(haystack) - len(needle) + 1):
        if haystack[index : index + len(needle)] == needle:
            return False
    return True


def _info_lost(deleted: str, qu: str, error_type: str) -> bool:
    """Mirror of the adapter's refusal certificate, re-derived from the artifact.

    ``key_information_missing`` rows claim a quantity vanished, so a number must be
    gone; ``question_missing`` rows claim the question itself was cut, so the
    removed words must no longer stand together.  Anything else fails closed.
    """
    if error_type == "key_information_missing":
        return _number_lost(deleted, qu)
    if error_type == "question_missing":
        return _phrase_absent(deleted, qu) or _number_lost(deleted, qu)
    return False


# ---------------------------------------------------------------------------
# the source anchor
# ---------------------------------------------------------------------------

#: category code -> (error_type, perturbation_type); the design doc's own names.
CATEGORY_NAMES = {
    1: ("key_information_missing", "missing_condition"),
    2: ("ambiguous_key_information", "ambiguous_condition"),
    3: ("unrealistic_conditions", "unrealistic_condition"),
    4: ("unrelated_object", "unrelated_entity"),
    5: ("question_missing", "question_missing"),
}


def load_source_index(raw_dir: str) -> dict[int, dict]:
    """``id -> source row`` from ``StandardDataset.jsonl`` (empty if unreadable)."""
    path = os.path.join(raw_dir, "StandardDataset.jsonl")
    if not os.path.exists(path):
        return {}
    index: dict[int, dict] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and isinstance(row.get("id"), int):
                index[row["id"]] = row
    return index


def check_source_anchor(rows: list[dict], reporter: Reporter, index: dict[int, dict], raw_dir: str) -> None:
    """Prove every row's text and gold against the source file itself.

    The artifact-local checks cannot tell an honest adapter from one that invented
    a question or a number.  This check can: it re-reads the source row the artifact
    points at and demands that

    * the presented question is the source's own question for that id, verbatim;
    * ``paired_original_text`` is the source's own question for the partner that the
      source's own ``relevant_ids`` names (never ``id - 2600``);
    * the label agrees with the source (answerable rows carry the source's numeric
      answer as their audit value; unanswerable rows carry none);
    * the defect label is the category's own name.
    """
    if not index:
        reporter.check(
            "source anchor (texts and golds re-read from the raw file)",
            False,
            f"cannot read {os.path.join(raw_dir, 'StandardDataset.jsonl')} -- the gold cannot be proved "
            "without the source; re-run with --raw-dir pointing at the downloaded UMWP directory",
        )
        return
    partners: dict[int, dict] = {}
    ambiguous: set[int] = set()
    for source in index.values():
        if source["answerable"]:
            continue
        linked = source.get("relevant_ids")
        target = linked[0] if isinstance(linked, list) and len(linked) == 1 else None
        if target is None:
            continue
        if target in partners:
            ambiguous.add(target)
            continue
        partners[target] = source
    for target in ambiguous:
        partners.pop(target, None)

    failures: list[str] = []
    for row in rows:
        info = row["extra_info"]
        task_id = info["task_id"]
        source = index.get(info["index"])
        if source is None:
            failures.append(f"{task_id}: index {info['index']} is not a source id")
            continue
        question = question_of(row)
        if question != _normalise(source["question"]):
            failures.append(f"{task_id}: presented question differs from the source's")
            continue
        if info["solvable"]:
            if not source["answerable"] or not source["answer"]:
                failures.append(f"{task_id}: source row is not answerable")
                continue
            partner = partners.get(source["id"])
            if partner is None or info["paired_original_text"] != _normalise(partner["question"]):
                failures.append(f"{task_id}: paired original is not the source's partner question")
                continue
            audit = json.loads(row["reward_model"]["ground_truth"]).get("answer")
            if audit != str(source["answer"][0]):
                failures.append(f"{task_id}: audit answer {audit!r} != source answer {source['answer'][0]!r}")
        else:
            if source["answerable"] or source["answer"] is not None:
                failures.append(f"{task_id}: source row is not a clean unanswerable row")
                continue
            linked = source.get("relevant_ids")
            target = linked[0] if isinstance(linked, list) and len(linked) == 1 else None
            partner = index.get(target) if target is not None else None
            if partner is None or not partner["answerable"]:
                failures.append(f"{task_id}: source relevant_ids does not name an answerable row")
                continue
            if info["paired_original_text"] != _normalise(partner["question"]):
                failures.append(f"{task_id}: paired original is not the source's relevant_ids target")
                continue
            names = CATEGORY_NAMES.get(source["category"])
            if names is None:
                failures.append(f"{task_id}: source category {source['category']!r} is unknown")
                continue
            if (info.get("error_type"), info.get("perturbation_type")) != names:
                failures.append(f"{task_id}: defect label {info.get('perturbation_type')!r} != category {source['category']}")
    reporter.check(
        "source anchor (texts and golds re-read from the raw file)",
        not failures,
        f"{len(rows)} rows anchored to {os.path.join(raw_dir, 'StandardDataset.jsonl')}, {len(failures)} failures"
        + (f"; first: {failures[:3]}" if failures else ""),
    )


def _normalise(text: str) -> str:
    return " ".join((text or "").split())




# ---------------------------------------------------------------------------
# L1 / L2
# ---------------------------------------------------------------------------


def check_l1(rows: list[dict], sample: dict[str, list[dict]], reporter: Reporter) -> None:
    for branch, picked in sample.items():
        failures: list[str] = []
        for row in picked:
            info = row["extra_info"]
            task_id = info["task_id"]
            question = question_of(row)
            qa, qu = _orient(row)
            deleted = info.get("deleted_condition_text", "")
            inserted = info.get("perturbed_entity_text", "")
            if branch == BRANCH_DIAG:
                option = _option_text(row, info.get("correct_option_id"))
                if option is None:
                    failures.append(f"{task_id}: correct_option_id is not an option of this row")
                    continue
                offset = qu.find(option)
                if offset < 0:
                    failures.append(f"{task_id}: gold {option!r} is not a span of its own question")
                    continue
                if not _reconstruct(qa, qu, deleted, inserted):
                    failures.append(f"{task_id}: re-inserting {deleted!r} does not rebuild the original")
                elif option.strip() != inserted.strip():
                    failures.append(f"{task_id}: gold {option!r} != recorded defect {inserted!r}")
            elif branch == BRANCH_BARE:
                if not _multiset_after_edit(qa, qu, deleted, inserted):
                    failures.append(f"{task_id}: question is not the original minus the recorded defect")
                elif not _info_lost(deleted, qu, info.get("error_type", "")):
                    failures.append(f"{task_id}: nothing informative is missing from {question[:60]!r}")
            else:
                # The gold here is the SOLVABLE verdict, and the number is audit
                # only, so the certificate is (a) the number is a real finite
                # number and (b) the recorded defect is exactly the edit that
                # separates this question from its unanswerable partner.  The
                # number itself is proved against the source file by the anchor
                # check above, which is the only place it can be proved.
                audit = json.loads(row["reward_model"]["ground_truth"]).get("answer")
                try:
                    value = float(audit)
                except (TypeError, ValueError):
                    failures.append(f"{task_id}: audit answer {audit!r} is not a number")
                    continue
                if not math.isfinite(value):
                    failures.append(f"{task_id}: audit answer {audit!r} is not finite")
                elif not (inserted.strip() or deleted.strip()):
                    failures.append(f"{task_id}: pair records no defect (nothing separates the two questions)")
                elif not _multiset_after_edit(qa, qu, deleted, inserted):
                    failures.append(f"{task_id}: question is not the original rewritten by the recorded defect")
        detail = f"{len(picked)} sampled rows re-derived, {len(failures)} failures"
        if failures:
            detail += "; first: " + " | ".join(failures[:3])
        reporter.check(f"L1 {branch}", not failures, detail)


def _option_text(row: dict, option_id: str | None) -> str | None:
    for option in row["extra_info"].get("options") or []:
        if option.get("id") == option_id:
            return option.get("text")
    return None


def check_l2(sample: dict[str, list[dict]], reporter: Reporter) -> None:
    for branch, picked in sample.items():
        failures: list[str] = []
        for row in picked:
            info = row["extra_info"]
            task_id = info["task_id"]
            qa, qu = _orient(row)
            deleted = info.get("deleted_condition_text", "")
            if branch == BRANCH_DIAG:
                gold = _option_text(row, info.get("correct_option_id"))
                if gold is None:
                    failures.append(f"{task_id}: no gold option")
                    continue
                if gold in qa:
                    failures.append(f"{task_id}: gold {gold!r} also occurs in the answerable question")
                    continue
                for option in row["extra_info"].get("options") or []:
                    text = option.get("text")
                    if text == gold:
                        continue
                    if text not in qa:
                        failures.append(f"{task_id}: second admissible option {text!r} (not in the original)")
            elif branch == BRANCH_BARE:
                if not _info_lost(deleted, qu, info.get("error_type", "")):
                    failures.append(f"{task_id}: deleted value {deleted[:40]!r} is still present")
            else:
                # The gold is the SOLVABLE marker, so there is deliberately no
                # correct option: contract, not a missing check.
                if info.get("correct_option_id") not in ("", None):
                    failures.append(f"{task_id}: solvable row carries correct_option_id")
        detail = f"{len(picked)} sampled rows, {len(failures)} uniqueness failures"
        if branch == BRANCH_JUDGE:
            detail += " (gold is a verdict, checked as a contract)"
        if failures:
            detail += "; first: " + " | ".join(failures[:3])
        reporter.check(f"L2 {branch}", not failures, detail)


# ---------------------------------------------------------------------------
# L3
# ---------------------------------------------------------------------------


def check_l3_options(rows: list[dict], reporter: Reporter) -> None:
    missing: list[str] = []
    unequal: list[str] = []
    checked = 0
    for row in rows:
        options = row["extra_info"].get("options") or []
        if not options:
            continue
        checked += 1
        question = question_of(row)
        task_id = row["extra_info"]["task_id"]
        for option in options:
            if option.get("text", "") not in question:
                missing.append(f"{task_id}:{option.get('text')!r}")
        lengths = {len(_tokens(option.get("text", ""))) for option in options}
        if len(lengths) != 1:
            unequal.append(f"{task_id}:{sorted(lengths)}")
    reporter.check(
        "L3a options are spans of their own question",
        not missing,
        f"{checked} option-bearing rows, {len(missing)} options not found"
        + (f"; first: {missing[:3]}" if missing else ""),
    )
    reporter.check(
        "L3b options equal token length",
        not unequal,
        f"{checked} option-bearing rows, {len(unequal)} rows with unequal lengths"
        + (f"; first: {unequal[:3]}" if unequal else ""),
    )


def bow_nb_oof(
    texts: list[str],
    labels: list[int],
    *,
    folds: int = 5,
    seed: int = 0,
    min_support: int = L3_MIN_SUPPORT,
    alpha: float = 1.0,
) -> float:
    """Out-of-fold balanced accuracy of a multinomial Naive Bayes, numpy only.

    Vocabulary is the tokens seen in at least ``min_support`` *training* documents
    of the fold (counting a token once per document); everything else -- including
    tokens unseen in training -- is ignored at test time.  Laplace smoothing and a
    smoothed class prior, both ``alpha``.  ``sklearn`` is deliberately not used
    (it is not installed in this environment).
    """
    documents = [[token.lower() for token in _tokens(text)] for text in texts]
    labels = np.asarray(labels, dtype=np.int64)
    count = len(texts)
    order = list(range(count))
    random.Random(seed).shuffle(order)
    fold_of = np.empty(count, dtype=np.int64)
    for rank, index in enumerate(order):
        fold_of[index] = rank % folds

    out_of_fold = np.full(count, -1, dtype=np.int64)
    for fold in range(folds):
        train = np.where(fold_of != fold)[0]
        test = np.where(fold_of == fold)[0]
        support: collections.Counter = collections.Counter()
        for index in train:
            support.update(set(documents[index]))
        vocabulary = {token: j for j, token in enumerate(sorted(t for t, c in support.items() if c >= min_support))}
        if not vocabulary:
            continue

        def design(indices) -> np.ndarray:
            matrix = np.zeros((len(indices), len(vocabulary)), dtype=np.float64)
            for row_index, doc_index in enumerate(indices):
                for token in documents[doc_index]:
                    column = vocabulary.get(token)
                    if column is not None:
                        matrix[row_index, column] += 1.0
            return matrix

        train_matrix = design(train)
        train_labels = labels[train]
        test_matrix = design(test)
        prior = np.array([np.sum(train_labels == c) for c in (0, 1)], dtype=np.float64) + alpha
        log_prior = np.log(prior / prior.sum())
        log_likelihood = np.zeros((2, len(vocabulary)), dtype=np.float64)
        for klass in (0, 1):
            totals = train_matrix[train_labels == klass].sum(axis=0) + alpha
            log_likelihood[klass] = np.log(totals / totals.sum())
        scores = test_matrix @ log_likelihood.T + log_prior
        out_of_fold[test] = scores.argmax(axis=1)

    if np.any(out_of_fold < 0):
        raise ValueError("some rows were never scored out of fold")
    positives = labels == 1
    negatives = ~positives
    recall_pos = np.mean(out_of_fold[positives] == 1) if positives.any() else float("nan")
    recall_neg = np.mean(out_of_fold[negatives] == 0) if negatives.any() else float("nan")
    return float(0.5 * (recall_pos + recall_neg))


def check_l3_nb(rows: list[dict], reporter: Reporter, *, folds: int, seed: int, min_support: int) -> None:
    texts = [question_of(row) for row in rows]
    labels = [1 if row["extra_info"]["solvable"] else 0 for row in rows]
    positives = sum(labels)
    reporter.note(
        f"L3c corpus: n={len(rows)} (solvable={positives}, unsolvable={len(rows) - positives}), "
        f"folds={folds}, vocabulary support >= {min_support} docs, alpha=1.0"
    )
    if positives == 0 or positives == len(rows):
        reporter.check(
            "L3c out-of-fold balanced accuracy <= 0.60",
            False,
            "only one label side present in the artifact -- the estimator is not identifiable; "
            "build with a --limit that keeps both branches (the adapter interleaves them)",
        )
        return
    accuracy = bow_nb_oof(texts, labels, folds=folds, seed=seed, min_support=min_support)
    control_rng = random.Random(seed + 1)
    control_labels = [control_rng.random() > 0.5 for _ in rows]
    control = bow_nb_oof(texts, control_labels, folds=folds, seed=seed, min_support=min_support)
    reporter.note(f"L3c raw out-of-fold balanced accuracy = {accuracy:.4f} (chance 0.5)")
    reporter.note(f"L3c shuffled-label control = {control:.4f}")
    reporter.check(
        "L3c out-of-fold balanced accuracy <= 0.60",
        accuracy <= L3_MAX_BALANCED_ACCURACY,
        f"balanced accuracy {accuracy:.4f} <= {L3_MAX_BALANCED_ACCURACY:.2f} on {len(rows)} rows "
        f"(control {control:.4f})",
    )


# ---------------------------------------------------------------------------
# contract invariants
# ---------------------------------------------------------------------------


def check_contract(rows: list[dict], reporter: Reporter) -> None:
    violations: list[str] = []
    for row in rows:
        for problem in schema.validate_row(row):
            violations.append(problem)
    reporter.check(
        "schema.validate_row on every row",
        not violations,
        f"{len(rows)} rows, {len(violations)} violations"
        + (f"; first: {violations[:3]}" if violations else ""),
    )

    failures: list[str] = []
    branch_of: dict[str, str] = {}
    for row in rows:
        info = row["extra_info"]
        task_id = info["task_id"]
        branch = info["branch"]
        ground_truth = json.loads(row["reward_model"]["ground_truth"])
        branch_of[branch] = info["template"]
        if not info.get("solvable") and ground_truth.get("answer") is not None:
            failures.append(f"{task_id}: unsolvable row carries an answer")
        if ground_truth.get("judgment_only") and not info.get("solvable"):
            failures.append(f"{task_id}: judgment_only on an unsolvable row")
        if branch == BRANCH_JUDGE and not ground_truth.get("judgment_only"):
            failures.append(f"{task_id}: solvable judge row without judgment_only")
        if branch == BRANCH_DIAG:
            if not ground_truth.get("has_diagnosis_label"):
                failures.append(f"{task_id}: diag row without a diagnosis label")
            if ground_truth.get("correct_option_id") not in [o["id"] for o in info.get("options") or []]:
                failures.append(f"{task_id}: diag gold is not one of its options")
        if branch == BRANCH_BARE and (ground_truth.get("has_diagnosis_label") or info.get("options")):
            failures.append(f"{task_id}: bare row must not offer options")
        if not info.get("paired_original_text") or info["paired_original_text"] == question_of(row):
            failures.append(f"{task_id}: missing or identical paired original")
    reporter.check(
        "branch invariants (answer/judgment_only/options/pairing)",
        not failures,
        f"{len(rows)} rows, {len(failures)} violations" + (f"; first: {failures[:3]}" if failures else ""),
    )
    reporter.note("branch -> template: " + ", ".join(f"{b}:{t}" for b, t in sorted(branch_of.items())))

    bad_ids = [
        row["extra_info"]["task_id"]
        for row in rows
        if not row["extra_info"]["task_id"].startswith("umwp-")
        or len({row["extra_info"]["task_id"] for row in rows}) != len(rows)
    ]
    reporter.check(
        "task_id is a unique, source-derived string",
        not bad_ids,
        f"{len(rows)} rows, {len(bad_ids)} bad ids" + (f"; first: {bad_ids[:3]}" if bad_ids else ""),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def sample_by_branch(rows: list[dict], *, per_branch: int, seed: int) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = collections.OrderedDict()
    for row in rows:
        groups.setdefault(row["extra_info"]["branch"], []).append(row)
    rng = random.Random(seed)
    sample: dict[str, list[dict]] = {}
    for branch, group in groups.items():
        take = min(len(group), max(per_branch, L1_MIN_SAMPLE))
        sample[branch] = sorted(rng.sample(group, take), key=lambda row: row["extra_info"]["task_id"])
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the UMWP rows built by umwp_adapter.py.")
    parser.add_argument("--rows", required=True, help="parquet written by umwp_adapter.py")
    parser.add_argument(
        "--raw-dir",
        default="/home/charles/data/reasoning_rl/halluc/raw/umwp",
        help="downloaded UMWP directory; the gold is proved against StandardDataset.jsonl here",
    )
    parser.add_argument("--seed", type=int, default=0, help="sampling + fold seed")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-support", type=int, default=L3_MIN_SUPPORT)
    parser.add_argument("--per-branch", type=int, default=200, help="L1/L2 sample ceiling per branch")
    args = parser.parse_args()

    rows = schema.read_parquet_rows(args.rows)
    if not rows:
        print(f"[FAIL] {args.rows} holds no rows")
        raise SystemExit(1)
    print(f"rows    : {args.rows} ({len(rows)} rows)")
    print(f"branches: {dict(collections.Counter(r['extra_info']['branch'] for r in rows))}")

    reporter = Reporter()
    check_contract(rows, reporter)
    check_source_anchor(rows, reporter, load_source_index(args.raw_dir), args.raw_dir)

    sample = sample_by_branch(rows, per_branch=args.per_branch, seed=args.seed)
    reporter.note("L1/L2 sample: " + ", ".join(f"{b}={len(v)}" for b, v in sample.items()))
    for branch, picked in sample.items():
        if len(picked) < L1_MIN_SAMPLE and len(picked) < len([r for r in rows if r["extra_info"]["branch"] == branch]):
            reporter.check(f"L1 {branch} sample size", False, f"only {len(picked)} rows to audit, need {L1_MIN_SAMPLE}")
    check_l1(rows, sample, reporter)
    check_l2(sample, reporter)

    check_l3_options(rows, reporter)
    check_l3_nb(rows, reporter, folds=args.folds, seed=args.seed, min_support=args.min_support)

    print()
    if reporter.failed:
        print(f"RESULT: FAIL ({len(reporter.failed)}/{len(reporter.results)} checks failed: {reporter.failed})")
        raise SystemExit(1)
    print(f"RESULT: PASS ({len(reporter.results)}/{len(reporter.results)} checks passed)")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
