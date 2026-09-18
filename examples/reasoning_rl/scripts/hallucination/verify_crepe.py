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
"""Audit for the KUQ + CREPE built parquet (design doc sections 4.9.1/4.9.2/9).

Reads the artifact ``crepe_adapter.py`` writes and re-proves its golds from the
**raw bytes**, so a bug in the adapter's parsing cannot hide behind the adapter's
own opinion:

* **L1 label certificate** -- for a stratified sample of >= 50 rows, the gold is
  re-derived from the raw record (CREPE: `labels`; KUQ: `unknown`), the question
  is compared byte for byte, and a *second, independent* source column is
  required to agree: ``presuppositions`` empty/non-empty for CREPE, a non-empty
  `answer` list plus the absence of a `category` key for KUQ-known, a `category`
  in {false assumption, counterfactual} plus `source == 'turk'` for KUQ-unknown.
* **L1b pointer-gold refusal** -- `presuppositions` must not be usable as a span:
  the verbatim-in-question ratio is recomputed from raw and required to stay
  under 10% (design doc section 9, `verify_crepe.py` item 2), and every row must
  carry ``has_diagnosis_label=false`` and no option block.
* **L2 gold uniqueness** -- every emitted row's raw record must admit exactly one
  gold (no CREPE dual labels), no question may be emitted under both labels, and
  no question or task_id may be emitted twice.
* **L2b label-string assertion** -- `'false presupposition'` with a space must
  match 927 train / 544 validation / 751 test rows and the README's
  `'false_presupposition'` must match 0 in every split (design doc section 9,
  item 1).
* **L3a/L3b anti-cheat, structural** -- option text verbatim in the question,
  equal option token length, and (the check that actually matters here) both
  labels must present an identical template and an identical option-presence
  rate, so "which verdict wording the prompt offers" is not a label cue.
* **L3c anti-cheat, statistical** -- a bag-of-words multinomial Naive Bayes, 5
  folds, out-of-fold balanced accuracy, vocabulary restricted to tokens with
  train support >= 5 documents, must be <= 0.60.

On L3c, two implementation notes that matter for reading the number:

1. The vocabulary filter is not an optimisation.  The UMWP recon report shows
   the *unfiltered* multinomial NB on this kind of short-question data is a
   numerically broken estimator whose class-conditional score changes sign out of
   fold, so a sub-0.5 reading from it is not evidence of anything.  The filtered
   number is the trustworthy one, and it is reported alongside the unfiltered
   number and the in-fold / out-of-fold sign diagnostic.
2. The decision rule is equal priors (the balanced-accuracy rule), which is the
   *most generous* threshold to the classifier and therefore the conservative
   choice for a gate.  The empirical-prior variant is printed for reference.

**Expected outcome on the current raw bundle: L3c FAILS.**  Measured on the
``--limit 300`` artifact: pooled 0.6800, KUQ-only 0.9067, CREPE-only 0.6267
against a 0.60 gate; on the full 1,300-row artifact pooled 0.6708, KUQ-only
0.8650, CREPE-only 0.5978.  The design doc's table A lists KUQ's L3 as 未测 and
gives CREPE no L3 figure at all.  The check is left at
0.60 on purpose -- lowering it would hide the one finding a reader needs.  See the
``crepe_adapter`` module docstring, deviation 4, for the diagnosis: the signal is
question *style*, the residue of the recon report's documented
``source == 'turk' ⟺ unknown`` metadata leak, not an option-block shortcut.

Exit status is non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from collections import Counter, defaultdict

import numpy as np
import schema

RAW_DIR_DEFAULT = "/home/charles/data/reasoning_rl/halluc/raw/kuq_crepe"

CREPE_SPLITS = ("train", "validation", "test")
CREPE_NORMAL = "normal"
CREPE_FALSE_PRESUPPOSITION = "false presupposition"
CREPE_LABELS = (CREPE_NORMAL, CREPE_FALSE_PRESUPPOSITION)
KUQ_FILE = "knowns_unknowns.jsonl"
KUQ_UNKNOWN_CATEGORIES = ("false assumption", "counterfactual")
KUQ_SOURCE_MARKER = "turk"

# Measured on the raw bundle; the design doc quotes the train one (927).
EXPECTED_CREPE_LABEL_HITS = {"train": 927, "validation": 544, "test": 751}
UNDERSCORE_LABEL_FORM = "false_presupposition"
VERBATIM_SPAN_LIMIT = 0.10

NB_FOLDS = 5
NB_MIN_SUPPORT = 5
NB_GATE = 0.60
NB_TOKEN_RE = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------------------
# check harness
# ---------------------------------------------------------------------------


class Audit:
    """Collects PASS/FAIL lines and remembers whether anything failed."""

    def __init__(self) -> None:
        self.checks = 0
        self.failures = 0

    def record(self, code: str, ok: bool, headline: str, details: list[str] | None = None) -> None:
        self.checks += 1
        if not ok:
            self.failures += 1
        print(f"{'PASS' if ok else 'FAIL'}  {code:<4} {headline}")
        for line in details or []:
            print(f"          {line}")

    @property
    def exit_code(self) -> int:
        return 1 if self.failures else 0


# ---------------------------------------------------------------------------
# independent raw readers (deliberately not imported from the adapter)
# ---------------------------------------------------------------------------


def read_crepe_records(raw_dir: str) -> dict[str, dict[str, dict]]:
    """``{split: {source_id: record}}`` with only the four audited columns."""
    import pyarrow.parquet as pq

    records: dict[str, dict[str, dict]] = {}
    for split in CREPE_SPLITS:
        path = os.path.join(raw_dir, f"crepe_{split}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing CREPE split: {path}")
        table = pq.read_table(path, columns=["id", "question", "labels", "presuppositions"])
        split_records: dict[str, dict] = {}
        for record in table.to_pylist():
            split_records[str(record["id"])] = {
                "question": record["question"] or "",
                "labels": [str(label) for label in (record["labels"] or [])],
                "presuppositions": [str(text) for text in (record["presuppositions"] or [])],
            }
        records[split] = split_records
    return records


def read_kuq_records(raw_dir: str) -> list[dict]:
    """``knowns_unknowns.jsonl`` as a list indexed by physical line number."""
    path = os.path.join(raw_dir, KUQ_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing KUQ label file: {path}")
    records: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            records.append(json.loads(line) if line else {})
    return records


def decode_task_id(task_id: str) -> tuple[str, str]:
    """``crepe:<split>:<id>`` / ``kuq:<index>`` -> (source, locator)."""
    parts = task_id.split(":")
    if parts[0] == "crepe" and len(parts) >= 3:
        return "crepe", f"{parts[1]}:{':'.join(parts[2:])}"
    if parts[0] == "kuq" and len(parts) == 2:
        return "kuq", parts[1]
    raise ValueError(f"unrecognised task_id {task_id!r}")


# ---------------------------------------------------------------------------
# L1 / L2: gold certificates re-derived from raw
# ---------------------------------------------------------------------------


def stratified_sample(rows: list[dict], size: int, rng: random.Random) -> list[dict]:
    """Round-robin across (data_source, solvable) so every group is covered."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        info = row["extra_info"]
        groups[(row["data_source"], bool(info.get("solvable")))].append(row)
    ordered = [groups[key] for key in sorted(groups, key=str)]
    for group in ordered:
        rng.shuffle(group)
    sample: list[dict] = []
    index = 0
    while len(sample) < size and any(index < len(group) for group in ordered):
        for group in ordered:
            if len(sample) >= size:
                break
            if index < len(group):
                sample.append(group[index])
        index += 1
    return sample


def certify_crepe(row: dict, record: dict) -> tuple[bool, str, bool]:
    """Return ``(ok, description, fragment_is_verbatim)`` for one CREPE row."""
    labels = record["labels"]
    question = record["question"]
    presuppositions = record["presuppositions"]
    payload = json.loads(row["reward_model"]["ground_truth"])
    solvable = bool(payload["solvable"])

    if labels not in ([CREPE_NORMAL], [CREPE_FALSE_PRESUPPOSITION]):
        return False, f"raw labels {labels!r} is not a single admissible label", False
    if (labels[0] == CREPE_FALSE_PRESUPPOSITION) == solvable:
        return False, "gold disagrees with the raw label", False
    if solvable:
        if presuppositions:
            return False, "raw presuppositions is not empty on a solvable row", False
        return True, "labels == ['normal'] and presuppositions empty", False
    if not presuppositions:
        return False, "raw presuppositions is empty on an unsolvable row", False
    verbatim = any(text.casefold() in question.casefold() for text in presuppositions)
    note = "presupposition is a verbatim span" if verbatim else "presupposition is a paraphrase"
    return True, f"labels == ['false presupposition'] and {note}", verbatim


def certify_kuq(row: dict, record: dict) -> tuple[bool, str]:
    """Return ``(ok, description)`` for one KUQ row."""
    payload = json.loads(row["reward_model"]["ground_truth"])
    solvable = bool(payload["solvable"])
    if not isinstance(record.get("unknown"), bool):
        return False, "'unknown' is not a JSON bool"
    if record["unknown"] == solvable:
        return False, "gold disagrees with the raw 'unknown' flag"
    if solvable:
        answers = record.get("answer")
        if (
            not isinstance(answers, list)
            or not answers
            or not all(isinstance(a, str) and a.strip() for a in answers)
        ):
            return False, "solvable row without a non-empty list of string answers"
        if "category" in record:
            return False, "solvable row still carries a category key"
        return True, "unknown == False, non-empty answer list, no category key"
    category = record.get("category")
    if category not in KUQ_UNKNOWN_CATEGORIES:
        return False, f"category {category!r} is outside the D18 allocation"
    if record.get("source") != KUQ_SOURCE_MARKER:
        return False, f"unknown row from source {record.get('source')!r}, not crowd-written"
    return True, f"unknown == True, category={category!r}, source='turk'"


def check_l1(
    rows: list[dict],
    crepe: dict,
    kuq: list[dict],
    sample_size: int,
    rng: random.Random,
    audit: Audit,
) -> None:
    sample = stratified_sample(rows, sample_size, rng)
    proven = Counter()
    verbatim_note = 0
    whitespace_note = 0
    problems: list[str] = []
    for row in sample:
        info = row["extra_info"]
        source, locator = decode_task_id(info["task_id"])
        if source == "crepe":
            split, source_id = locator.split(":", 1)
            record = crepe.get(split, {}).get(source_id)
            if record is None:
                problems.append(f"[{info['task_id']}] absent from raw crepe_{split}.parquet")
                continue
            raw_question = record["question"]
        else:
            index = int(locator)
            if index >= len(kuq) or not kuq[index]:
                problems.append(f"[{info['task_id']}] absent from {KUQ_FILE}")
                continue
            record = kuq[index]
            raw_question = record.get("question", "")

        # ``schema.render_prompt`` strips surrounding whitespace, so the recovered
        # question is compared against the stripped raw text and the rows that
        # needed the strip are counted rather than silently tolerated.
        whitespace_note += int(raw_question != raw_question.strip())
        if raw_question.strip() != info["_question"]:
            problems.append(
                f"[{info['task_id']}] question differs from the raw record beyond surrounding whitespace"
            )
            continue

        if source == "crepe":
            ok, description, verbatim = certify_crepe(row, record)
            verbatim_note += int(verbatim)
        else:
            ok, description = certify_kuq(row, record)
        if ok:
            proven[f"{source}/{'solvable' if info['solvable'] else 'unsolvable'}"] += 1
        else:
            problems.append(f"[{info['task_id']}] {description}")

    breakdown = ", ".join(f"{key}={value}" for key, value in sorted(proven.items()))
    audit.record(
        "L1",
        not problems and sum(proven.values()) == len(sample) and bool(sample),
        f"label certificate re-derived from raw for {len(sample)} rows "
        f"({breakdown}); {verbatim_note} sampled false presuppositions are verbatim spans",
        [
            f"{whitespace_note} sampled questions needed schema.render_prompt's .strip() "
            "(94 of 6884 KUQ raw questions carry surrounding whitespace; 0 CREPE)",
        ]
        + problems[:5]
        + ([f"... and {len(problems) - 5} more"] if len(problems) > 5 else []),
    )


def check_l1b_pointer_refusal(
    rows: list[dict], crepe: dict, audit: Audit
) -> None:
    diagnosis = [r for r in rows if r["extra_info"]["has_diagnosis_label"]]
    optioned = [r for r in rows if r["extra_info"]["options"]]
    problems = []
    if diagnosis:
        problems.append(f"{len(diagnosis)} rows carry has_diagnosis_label=True")
    if optioned:
        problems.append(f"{len(optioned)} rows carry an options block")

    ratios: list[str] = []
    worst = 0.0
    for split in CREPE_SPLITS:
        records = crepe[split]
        fragments = verbatim = 0
        for record in records.values():
            if CREPE_FALSE_PRESUPPOSITION not in record["labels"]:
                continue
            for text in record["presuppositions"]:
                fragments += 1
                verbatim += int(text.casefold() in record["question"].casefold())
        ratio = verbatim / max(fragments, 1)
        worst = max(worst, ratio)
        ratios.append(f"{split}: {verbatim}/{fragments} = {ratio:.4f}")
    if worst >= VERBATIM_SPAN_LIMIT:
        problems.append(f"verbatim presupposition ratio {worst:.4f} >= {VERBATIM_SPAN_LIMIT}")

    audit.record(
        "L1b",
        not problems,
        f"pointer gold refused: 0 rows with an options block or a diagnosis label; "
        f"verbatim presupposition ratio < {VERBATIM_SPAN_LIMIT:.0%} in every split",
        ratios + problems,
    )


def check_l2(rows: list[dict], crepe: dict, kuq: list[dict], audit: Audit) -> None:
    problems: list[str] = []
    questions: dict[str, set[bool]] = defaultdict(set)
    duplicates: list[str] = []
    seen_questions: set[str] = set()
    seen_task_ids: set[str] = set()
    duals = 0

    for row in rows:
        info = row["extra_info"]
        payload = json.loads(row["reward_model"]["ground_truth"])
        task_id = info["task_id"]
        if task_id in seen_task_ids:
            duplicates.append(task_id)
        seen_task_ids.add(task_id)
        source, locator = decode_task_id(task_id)
        if source == "crepe":
            split, source_id = locator.split(":", 1)
            labels = crepe[split][source_id]["labels"]
            if len(labels) != 1:
                duals += 1
                problems.append(f"[{task_id}] raw labels {labels!r} admits two golds")
        else:
            record = kuq[int(locator)]
            if not isinstance(record.get("unknown"), bool):
                problems.append(f"[{task_id}] raw 'unknown' is not a bool")
        key = " ".join(info["_question"].casefold().split())
        questions[key].add(bool(payload["solvable"]))
        if key in seen_questions:
            duplicates.append(key[:40])
        seen_questions.add(key)

    both = [key for key, labels in questions.items() if len(labels) > 1]
    if both:
        problems.append(f"{len(both)} questions are emitted under both labels: {both[:2]}")
    if duplicates:
        problems.append(f"{len(duplicates)} duplicated questions/task_ids: {duplicates[:2]}")

    audit.record(
        "L2",
        not problems,
        f"gold unique for every row: {len(rows)} rows, {len(seen_task_ids)} task_ids, "
        f"{len(seen_questions)} questions, {duals} dual-labelled rows",
        problems[:5],
    )


def check_l2b_label_string(crepe: dict, audit: Audit) -> None:
    hits = {}
    underscore = {}
    for split in CREPE_SPLITS:
        records = crepe[split].values()
        hits[split] = sum(1 for r in records if CREPE_FALSE_PRESUPPOSITION in r["labels"])
        underscore[split] = sum(1 for r in records if UNDERSCORE_LABEL_FORM in r["labels"])
    problems = [
        f"{split}: expected {EXPECTED_CREPE_LABEL_HITS[split]} space-form hits, measured {hits[split]}"
        for split in CREPE_SPLITS
        if hits[split] != EXPECTED_CREPE_LABEL_HITS[split]
    ]
    problems += [
        f"{split}: the README's underscore form matched {underscore[split]} rows"
        for split in CREPE_SPLITS
        if underscore[split]
    ]
    audit.record(
        "L2b",
        not problems,
        "label vocabulary: 'false presupposition' (space) matches "
        + ", ".join(f"{split}={hits[split]}" for split in CREPE_SPLITS)
        + "; 'false_presupposition' (underscore) matches 0 rows",
        problems,
    )


# ---------------------------------------------------------------------------
# L3: anti-cheat
# ---------------------------------------------------------------------------


def check_l3_structural(rows: list[dict], audit: Audit) -> None:
    """Option verbatim / equal length, plus the template-leak guard."""
    problems: list[str] = []
    option_rows = 0
    unequal_length = 0
    missing_verbatim: list[str] = []
    templates: dict[bool, set[str]] = defaultdict(set)
    total: Counter = Counter()
    option_presence: Counter = Counter()
    for row in rows:
        info = row["extra_info"]
        payload = json.loads(row["reward_model"]["ground_truth"])
        solvable = bool(payload["solvable"])
        templates[solvable].add(info["template"])
        total[solvable] += 1
        options = info["options"] or []
        option_presence[solvable] += 1 if options else 0
        if not options:
            continue
        option_rows += 1
        for option in options:
            if option["text"] not in info["_question"]:
                missing_verbatim.append(f"[{info['task_id']}] {option['text']!r} not in question")
        lengths = {len(option["text"].split()) for option in options}
        if len(lengths) != 1:
            unequal_length += 1

    if missing_verbatim:
        problems.append(f"{len(missing_verbatim)} options are not verbatim question spans")
    if unequal_length:
        problems.append(f"{unequal_length} rows have options of unequal token length")
    if len(templates[True]) != 1 or len(templates[False]) != 1 or templates[True] != templates[False]:
        problems.append(
            f"template differs by label: solvable={sorted(templates[True])} unsolvable={sorted(templates[False])}"
        )
    rate = {label: option_presence[label] / max(total[label], 1) for label in (True, False)}
    if abs(rate[True] - rate[False]) > 1e-9:
        problems.append(f"option-presence rate differs by label: {rate}")

    audit.record(
        "L3a",
        not problems,
        f"option cue absent: {option_rows} rows carry an options block, so 'verbatim in the "
        f"question' and 'equal option length' hold vacuously; both labels use "
        f"{sorted(templates[True])} with option-presence {rate[True]:.3f} vs {rate[False]:.3f}",
        problems[:5],
    )


def _tokens(text: str) -> list[str]:
    return NB_TOKEN_RE.findall(text.casefold())


def _score_documents(
    documents: list[list[str]],
    indices: np.ndarray,
    token_index: dict[str, int],
    size: int,
    difference: np.ndarray,
) -> np.ndarray:
    """Bag-of-words counts of ``indices`` dotted with the class-score vector."""
    matrix = np.zeros((len(indices), size))
    for position, index in enumerate(indices):
        for token in documents[index]:
            column = token_index.get(token)
            if column is not None:
                matrix[position, column] += 1
    return matrix @ difference


def nb_out_of_fold(
    texts: list[str],
    labels: np.ndarray,
    *,
    folds: int = NB_FOLDS,
    min_support: int = NB_MIN_SUPPORT,
    seed: int = 0,
) -> tuple[float, list[float], dict]:
    """5-fold out-of-fold multinomial NB, numpy only (sklearn is not installed).

    ``labels`` is 0 for the solvable side and 1 for the unsolvable side.
    ``min_support`` prunes the vocabulary to tokens seen in at least that many
    *training-fold documents*; ``min_support=0`` keeps everything, which is the
    variant the UMWP recon report shows is numerically unstable.

    The decision rule is equal priors (``score > 0``), i.e. the balanced-accuracy
    rule; that is the most generous threshold for the classifier and therefore
    the conservative choice for a gate.  ``diagnostics`` also carries the
    empirical-prior reading and the in-fold / out-of-fold mean class score, which
    is what distinguishes a weak-but-real signal from a broken estimator.
    """
    documents = [_tokens(text) for text in texts]
    count = len(documents)
    class_counts = np.array([int((labels == cls).sum()) for cls in (0, 1)])
    folds = min(folds, int(class_counts.min()))
    if folds < 2:
        raise ValueError(f"cannot cross-validate {class_counts.tolist()} rows in 2 folds")

    rng = np.random.RandomState(seed)
    assignment = np.empty(count, dtype=int)
    for cls in (0, 1):
        members = np.where(labels == cls)[0]
        members = members[rng.permutation(len(members))]
        assignment[members] = np.arange(len(members)) % folds

    predictions = np.zeros(count, dtype=int)
    per_fold: list[float] = []
    in_fold_d: dict[int, list[float]] = {0: [], 1: []}
    out_fold_score: dict[int, list[float]] = {0: [], 1: []}
    prior_fold_score: dict[int, list[float]] = {0: [], 1: []}
    for fold in range(folds):
        train = np.where(assignment != fold)[0]
        test = np.where(assignment == fold)[0]
        document_frequency: Counter = Counter()
        for index in train:
            document_frequency.update(set(documents[index]))
        vocabulary = {token: df for token, df in document_frequency.items() if df >= min_support}
        vocabulary = dict(sorted(vocabulary.items()))
        token_index = {token: column for column, token in enumerate(vocabulary)}
        size = len(token_index)

        counts = np.zeros((2, size))
        for cls in (0, 1):
            for index in train[labels[train] == cls]:
                for token in documents[index]:
                    column = token_index.get(token)
                    if column is not None:
                        counts[cls, column] += 1
        log_prob = np.log((counts + 1.0) / (counts.sum(axis=1, keepdims=True) + size))
        difference = log_prob[1] - log_prob[0]

        test_scores = _score_documents(documents, test, token_index, size, difference)
        predictions[test] = (test_scores > 0).astype(int)
        true_positive = int(((predictions[test] == 1) & (labels[test] == 1)).sum())
        true_negative = int(((predictions[test] == 0) & (labels[test] == 0)).sum())
        false_positive = int(((predictions[test] == 1) & (labels[test] == 0)).sum())
        false_negative = int(((predictions[test] == 0) & (labels[test] == 1)).sum())
        per_fold.append(
            0.5
            * (
                true_positive / max(true_positive + false_negative, 1)
                + true_negative / max(true_negative + false_positive, 1)
            )
        )
        prior_threshold = math.log(
            max((labels[train] == 1).sum(), 1) / max((labels[train] == 0).sum(), 1)
        )
        train_scores = _score_documents(documents, train, token_index, size, difference)
        for cls in (0, 1):
            in_fold_d[cls].append(float(train_scores[labels[train] == cls].mean()))
            if (labels[test] == cls).any():
                out_fold_score[cls].append(float(test_scores[labels[test] == cls].mean()))
                prior_fold_score[cls].append(
                    float(((test_scores > prior_threshold) == (cls == 1))[labels[test] == cls].mean())
                )

    true_positive = int(((predictions == 1) & (labels == 1)).sum())
    true_negative = int(((predictions == 0) & (labels == 0)).sum())
    false_positive = int(((predictions == 1) & (labels == 0)).sum())
    false_negative = int(((predictions == 0) & (labels == 1)).sum())
    balanced = 0.5 * (
        true_positive / max(true_positive + false_negative, 1)
        + true_negative / max(true_negative + false_positive, 1)
    )
    diagnostics = {
        "in_fold_mean_score": {cls: float(np.mean(v)) for cls, v in in_fold_d.items()},
        "out_of_fold_mean_score": {cls: float(np.mean(v)) for cls, v in out_fold_score.items()},
        "empirical_prior_balanced_accuracy": 0.5
        * (float(np.mean(prior_fold_score[1])) + float(np.mean(prior_fold_score[0]))),
    }
    return float(balanced), per_fold, diagnostics


def check_l3_statistical(rows: list[dict], audit: Audit, seed: int) -> None:
    texts = [r["extra_info"]["_question"] for r in rows]
    labels = np.array(
        [0 if json.loads(r["reward_model"]["ground_truth"])["solvable"] else 1 for r in rows]
    )
    sources = [r["data_source"] for r in rows]

    filtered, folds, diagnostics = nb_out_of_fold(texts, labels, min_support=NB_MIN_SUPPORT, seed=seed)
    unfiltered, _, unfiltered_diagnostics = nb_out_of_fold(texts, labels, min_support=0, seed=seed)

    control = np.array(labels)
    random.Random(seed).shuffle(control)
    control_score, _, _ = nb_out_of_fold(texts, control, min_support=NB_MIN_SUPPORT, seed=seed)

    per_source: list[str] = []
    for source in sorted(set(sources)):
        subset = [i for i, value in enumerate(sources) if value == source]
        if len(subset) < 100:
            per_source.append(f"{source}: only {len(subset)} rows, skipped")
            continue
        subset_score, _, _ = nb_out_of_fold(
            [texts[i] for i in subset], labels[subset], min_support=NB_MIN_SUPPORT, seed=seed
        )
        per_source.append(f"{source}: {subset_score:.4f} (n={len(subset)})")

    details = [
        f"per source (support >= {NB_MIN_SUPPORT}): " + "; ".join(per_source),
        "per fold (pooled, equal priors): " + " / ".join(f"{value:.3f}" for value in folds),
        f"unfiltered vocabulary (support >= 0): {unfiltered:.4f}"
        f"  [OOF mean score solvable={unfiltered_diagnostics['out_of_fold_mean_score'][0]:+.3f},"
        f" unsolvable={unfiltered_diagnostics['out_of_fold_mean_score'][1]:+.3f}]",
        "sign diagnostic (support >= 5): in-fold mean score "
        f"solvable={diagnostics['in_fold_mean_score'][0]:+.3f} / "
        f"unsolvable={diagnostics['in_fold_mean_score'][1]:+.3f}; out-of-fold "
        f"solvable={diagnostics['out_of_fold_mean_score'][0]:+.3f} / "
        f"unsolvable={diagnostics['out_of_fold_mean_score'][1]:+.3f}",
        f"empirical-prior variant (reference only): {diagnostics['empirical_prior_balanced_accuracy']:.4f}",
        f"random-label sanity control (same pipeline): {control_score:.4f}",
    ]
    ok = filtered <= NB_GATE and 0.40 <= control_score <= 0.60
    audit.record(
        "L3c",
        ok,
        f"BoW multinomial NB, {NB_FOLDS}-fold out-of-fold balanced accuracy "
        f"(verbatim questions, vocabulary = tokens with train support >= {NB_MIN_SUPPORT}): "
        f"{filtered:.4f} (gate <= {NB_GATE})",
        details,
    )


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


# The template text is appended to the question with a blank line, so the question
# is everything before the first marker.  ``run_checks`` proves the split is exact
# by re-rendering the prompt from the recovered question and the row's own
# template / options / role words and requiring byte equality.
TAIL_MARKERS = (
    "\n\n请先判断该题的前提是否成立",
    "\n\n请判断该题的前提是否成立",
    "\n\n若题目给出的信息不足",
    "\n\n注意：请用题目中出现的角色名称作答",
)


def extract_question(content: str) -> str:
    for marker in TAIL_MARKERS:
        if marker in content:
            return content.split(marker)[0]
    return content


def run_checks(rows: list[dict], raw_dir: str, sample_size: int, seed: int) -> Audit:
    audit = Audit()
    print(f"rows          : {len(rows)}")
    print(f"raw bundle    : {raw_dir}")
    print(f"data_sources  : {dict(Counter(r['data_source'] for r in rows))}")
    print()

    violations: list[str] = []
    for row in rows:
        info = row["extra_info"]
        info["_question"] = extract_question(row["prompt"][0]["content"]).strip()
        violations.extend(schema.validate_row(row))
        rerendered = schema.render_prompt(
            info["_question"], info["template"], options=info["options"], role_words=info["role_words"]
        )
        if rerendered != row["prompt"][0]["content"]:
            violations.append(f"[{info['task_id']}] prompt is not render_prompt(recovered question)")
    audit.record(
        "L0",
        not violations,
        f"schema.validate_row passes and prompt == render_prompt(recovered question) "
        f"for all {len(rows)} rows",
        violations[:5],
    )

    crepe = read_crepe_records(raw_dir)
    kuq = read_kuq_records(raw_dir)
    rng = random.Random(seed)

    check_l1(rows, crepe, kuq, sample_size, rng, audit)
    check_l1b_pointer_refusal(rows, crepe, audit)
    check_l2(rows, crepe, kuq, audit)
    check_l2b_label_string(crepe, audit)
    check_l3_structural(rows, audit)
    check_l3_statistical(rows, audit, seed)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", required=True, help="built parquet from crepe_adapter.py")
    parser.add_argument("--raw-dir", default=RAW_DIR_DEFAULT, help="raw kuq_crepe bundle")
    parser.add_argument("--sample", type=int, default=60, help="L1 certificate sample (>= 50)")
    parser.add_argument("--seed", type=int, default=0, help="sampling / CV seed")
    args = parser.parse_args()

    if args.sample < 50:
        raise SystemExit(f"--sample must be >= 50 (the L1 certificate floor), got {args.sample}")

    rows = schema.read_parquet_rows(args.rows)
    audit = run_checks(rows, args.raw_dir, args.sample, args.seed)

    print()
    print(f"{audit.checks - audit.failures}/{audit.checks} checks passed")
    raise SystemExit(audit.exit_code)


if __name__ == "__main__":
    main()
