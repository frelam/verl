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
"""Audit the built KUQ parquet: L1 label certificate, L2 verdict uniqueness, L3 anti-cheat.

Run it on the artifact alone::

    python examples/reasoning_rl/scripts/hallucination/verify_kuq.py \\
        --rows /tmp/halluc_kuq.parquet

Every check prints ``PASS``/``FAIL`` (or ``SKIP`` for the raw-file group when the
raw download is not on this machine) and the process exits non-zero if any check
failed.  The numbers are printed verbatim because this is the evidence a build
report quotes.

What the checks can and cannot be
---------------------------------

**L1 -- label certificate.**  KUQ ships no external oracle: the known half is
factual QA ("California"), so nothing in the file can recompute the answer, and
the unknown half carries no span or offset of any kind.  The verifier therefore
re-derives each sampled row's verdict from **three axes that never read the
stored label**, and fails if any axis dissents:

1. ``kuq_source`` -- the closed provenance vocabulary (``turk`` is the
   crowdsourced unanswerable half; hotpotqa / squad / triviaqa are answerable QA
   corpora);
2. ``kuq_category`` -- presence *and* membership of the documented six-value
   taxonomy;
3. the row's own answer payload, re-split from ``kuq_answer_text`` and re-tested
   against the adapter's form margins (a solvable row needs a determinate answer
   span, an unsolvable row an uncertainty explanation).

Axes 1 and 2 decide; axis 3 can only corroborate or contradict, and that is
measured rather than assumed -- 145 known and 3,191 unknown raw rows satisfy
*both* forms, because a turk row routinely includes a plausible-looking answer
element as part of documenting its unanswerability.  A contradiction fails the
check; an abstention is counted and printed.  Claiming the form axis "re-derived"
those rows would be the kind of green check this report exists to avoid.

The check also proves the payload carries no second gold (``answer`` must be
``None``), that the row kept its template, and that the stored prompt is exactly
``schema.render_prompt(question, template)``.

**L2 -- verdict uniqueness.**  There is no option set here, so the analogous
question is "could a second verdict be argued?".  The verifier fails loudly if
any row's axes disagree, if the same question occurs under both verdicts, if two
rows share a ``task_id``, or if any row carries a diagnosis payload (options /
``correct_option_id`` / ``has_diagnosis_label``) it must not have.

**L3 -- anti-cheat.**  (a) option presence: KUQ is option-less by construction,
and an option block would *reintroduce* the "pick the option that is not in the
question" leak the recon measured at 88-100% on MiP -- so the check asserts no
row carries one.  (b) equal-length cue: with no options the analogous cue is the
*shape of the prompt*; the verifier asserts the two verdict classes share one
byte-identical template suffix (design doc section 9, D14 isomorphism) and
reports the question-length cue's out-of-fold balanced accuracy.  (c) bag-of-words
multinomial Naive Bayes (implemented here with numpy; sklearn is not installed),
5-fold, out-of-fold balanced accuracy with the vocabulary restricted to tokens
that appear in >= 5 *training* documents, against the <= 0.60 gate.

**L3(c) FAILS on the real build, and this report says so.**  Measured
out-of-fold balanced accuracy: **0.858** over the whole raw source, **0.893** over
the raw in-scope ``false assumption`` / ``counterfactual`` pool, **0.883** on the
full built artifact, and **0.807** on the ``--limit 300`` sample the mandated run
uses.  The known and unknown halves are separable from the question's vocabulary
alone -- on the full artifact, stopwords only 0.855, content words only 0.749,
question length 0.533 -- so it is a vocabulary/topic cue, not a stylistic or
length artefact, and the sample's dip to 0.807 follows from its thinner class
ratio rather than from any filtering.

The gate is not weakened and the adapter does not filter its way under it:
dropping the rows a classifier gets right would be selecting on the label with the
very classifier being audited, i.e. manufacturing a pass rather than earning one.
The finding is recorded as a build-report item (``kuq_adapter``'s deviation 6)
together with the design doc's own precedent -- its FalseQA judgment branch
measures H6 0.765/0.833/0.807 and section 9/11 treats that as an *interpretation
ceiling*.  Both numbers are printed so the two judgment branches can be read side
by side.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import schema
from kuq_adapter import (
    DEFAULT_RAW_DIR,
    KNOWN_SOURCE_IDS,
    KUQ_CATEGORIES,
    RAW_FILENAME,
    payload_form_problem,
    word_count,
)

# The task's gate.  Deliberately not a CLI flag: a threshold that can be raised
# from the command line is a threshold that will be raised.
NB_BALANCED_ACCURACY_GATE = 0.60
NB_MIN_SUPPORT = 5
NB_FOLDS = 5
NB_ALPHA = 1.0
L1_MIN_SAMPLE = 50
ANSWER_JOIN = " || "

TEMPLATE_SUFFIX = schema.render_prompt("X", schema.TEMPLATE_B_JUDGE)[len("X") :]


# ---------------------------------------------------------------------------
# check plumbing
# ---------------------------------------------------------------------------


def result(name: str, ok: bool | None, detail: str = "") -> dict:
    """One check outcome; ``ok=None`` means SKIP."""
    return {"name": name, "ok": ok, "detail": detail}


def _payload(row: dict) -> dict:
    return json.loads(row["reward_model"]["ground_truth"])


def _split_answer_text(row: dict) -> list[str]:
    text = row["extra_info"].get("kuq_answer_text", "")
    return [part for part in text.split(ANSWER_JOIN)]


def _prompt_question(row: dict) -> str:
    """Recover the question from the rendered prompt (prompt = question + suffix)."""
    content = row["prompt"][0]["content"]
    return content[: len(content) - len(TEMPLATE_SUFFIX)]


# ---------------------------------------------------------------------------
# L1
# ---------------------------------------------------------------------------


def l1_axis_verdicts(row: dict) -> dict[str, bool | None]:
    """Re-derive the verdict from each independent axis (``None`` == abstain).

    Axis values are "the row is solvable" booleans; an axis that cannot decide
    (unknown source token, category outside the taxonomy, ambiguous join) returns
    ``None`` so the caller can report it as a disagreement instead of silently
    passing.
    """
    info = row["extra_info"]

    source = info.get("kuq_source", "")
    if source == "turk":
        provenance: bool | None = False
    elif source in KNOWN_SOURCE_IDS:
        provenance = True
    else:
        provenance = None

    category = info.get("kuq_category", "")
    if not category:
        annotation: bool | None = True  # absent key -> the answerable half
    elif category in KUQ_CATEGORIES:
        annotation = False
    else:
        annotation = None

    answers = _split_answer_text(row)
    if len(answers) != info.get("kuq_answer_count"):
        # The stored join is ambiguous: the payload cannot be re-split, so the
        # axis abstains rather than guessing which element was which.
        payload_axis: bool | None = None
    else:
        # The axis reads to the solvable side when the payload looks like a
        # determinate answer and to the unsolvable side when it reads as an
        # explanation; both at once (or neither) is an abstention.
        looks_solvable = payload_form_problem(answers, solvable=True) is None
        looks_unsolvable = payload_form_problem(answers, solvable=False) is None
        payload_axis = (
            True
            if looks_solvable and not looks_unsolvable
            else False
            if looks_unsolvable and not looks_solvable
            else None
        )
    return {"provenance": provenance, "annotation": annotation, "payload": payload_axis}


DECISIVE_AXES = ("provenance", "annotation")


def check_l1_certificate(rows: list[dict], sample_size: int) -> dict:
    """Re-prove the gold for a >= 50 row sample from three independent axes.

    ``provenance`` and ``annotation`` decide and must agree with the stored label
    on every sampled row.  ``payload`` is a *one-sided necessary condition* on
    this source and cannot decide: 145 known rows and 3,191 unknown rows satisfy
    both the answer-span and the uncertainty-explanation form (an unsolvable turk
    row routinely carries a plausible-looking answer element *as part of* its
    explanation), so the axis is scored as corroboration, abstention or
    contradiction -- and only a contradiction fails the check.  The rates are
    printed so the reader can see how much of the gold this axis actually
    re-derives instead of having it counted as proof.
    """
    ordered = sorted(rows, key=lambda row: row["extra_info"]["task_id"])
    if not ordered:
        return result("L1 label certificate", False, "no rows")
    size = max(L1_MIN_SAMPLE, sample_size)
    stride = max(1, len(ordered) // size)
    sample = ordered[::stride][: max(size, 1)]

    disagreements: Counter = Counter()
    corroborated = 0
    undecided = 0
    unsolvable_with_answer = 0
    judgment_not_marked = 0
    prompt_mismatch = 0
    for row in sample:
        payload = _payload(row)
        verdict = bool(payload["solvable"])
        axes = l1_axis_verdicts(row)
        for axis, value in axes.items():
            if axis in DECISIVE_AXES:
                if value is None or value != verdict:
                    disagreements[f"{axis}={value}"] += 1
            elif value is None:
                undecided += 1
            elif value != verdict:
                disagreements["payload contradicts"] += 1
            else:
                corroborated += 1
        if not verdict and payload.get("answer") is not None:
            unsolvable_with_answer += 1
        if verdict and not payload.get("judgment_only"):
            judgment_not_marked += 1
        question = _prompt_question(row)
        if row["prompt"][0]["content"] != schema.render_prompt(
            question, row["extra_info"]["template"]
        ):
            prompt_mismatch += 1

    ok = not disagreements and not unsolvable_with_answer and not judgment_not_marked
    ok = ok and not prompt_mismatch
    detail = (
        f"sample={len(sample)}/{len(rows)} rows; decisive axes "
        f"({'/'.join(DECISIVE_AXES)}) re-derived the stored verdict for "
        f"{len(sample) - sum(disagreements.values())} rows; "
        f"axis disagreements={dict(disagreements) or 'none'}; "
        f"payload axis: corroborates={corroborated}, undecided={undecided} "
        f"(both forms satisfiable -- see docstring), contradicts="
        f"{disagreements.get('payload contradicts', 0)}; "
        f"unsolvable rows carrying an answer={unsolvable_with_answer}; "
        f"solvable rows without judgment_only={judgment_not_marked}; "
        f"prompts not equal to render_prompt(question, template)={prompt_mismatch}"
    )
    return result("L1 label certificate", ok, detail)


# ---------------------------------------------------------------------------
# L2
# ---------------------------------------------------------------------------


def check_l2_uniqueness(rows: list[dict]) -> dict:
    """Fail loudly when a second verdict (or a second gold) can be argued."""
    axis_dissent = 0
    payload_undecided = 0
    questions_per_verdict: dict[bool, set[str]] = {True: set(), False: set()}
    task_ids: Counter = Counter()
    option_rows = 0
    diagnosis_rows = 0
    for row in rows:
        payload = _payload(row)
        verdict = bool(payload["solvable"])
        axes = l1_axis_verdicts(row)
        for axis in DECISIVE_AXES:
            if axes[axis] is None or axes[axis] != verdict:
                axis_dissent += 1
                break
        if axes["payload"] is None:
            payload_undecided += 1
        elif axes["payload"] != verdict:
            axis_dissent += 1
        questions_per_verdict[verdict].add(_prompt_question(row))
        task_ids[row["extra_info"]["task_id"]] += 1
        info = row["extra_info"]
        if info.get("options"):
            option_rows += 1
        if payload.get("has_diagnosis_label") or payload.get("correct_option_id"):
            diagnosis_rows += 1

    shared = questions_per_verdict[True] & questions_per_verdict[False]
    duplicates = {tid: n for tid, n in task_ids.items() if n > 1}
    ok = not axis_dissent and not shared and not duplicates
    ok = ok and not option_rows and not diagnosis_rows
    detail = (
        f"rows with a dissenting axis={axis_dissent} "
        f"(payload axis undecided on {payload_undecided} rows, which is not a "
        f"second verdict); "
        f"questions under both verdicts={len(shared)}; "
        f"duplicate task_ids={len(duplicates)}; "
        f"rows carrying options={option_rows}; "
        f"rows carrying a diagnosis payload={diagnosis_rows}"
    )
    return result("L2 verdict uniqueness", ok, detail)


# ---------------------------------------------------------------------------
# L3
# ---------------------------------------------------------------------------


def check_l3_option_cues(rows: list[dict]) -> dict:
    """(a) no option block, and no orphan option text anywhere."""
    option_rows = 0
    block_rows = 0
    orphan = 0
    for row in rows:
        info = row["extra_info"]
        options = info.get("options") or []
        if options:
            option_rows += 1
            question = _prompt_question(row)
            if not all(opt["text"] in question for opt in options):
                orphan += 1
        if "选项：" in row["prompt"][0]["content"]:
            block_rows += 1
    ok = option_rows == 0 and block_rows == 0 and orphan == 0
    detail = (
        "option-less source: "
        f"rows with an options list={option_rows}; "
        f"rows whose prompt renders an option block={block_rows}; "
        f"rows with option text absent from the question={orphan} "
        "(the check is the presence cue itself: an option block here would "
        "re-open the 'pick what is not in the question' shortcut)"
    )
    return result("L3a option-presence cue", ok, detail)


def check_l3_template_isomorphism(rows: list[dict]) -> dict:
    """(b) both verdicts share one template, and the length cue is at chance."""
    suffixes: Counter = Counter()
    for row in rows:
        content = row["prompt"][0]["content"]
        suffixes[content[len(_prompt_question(row)) :]] += 1
    templates = Counter(row["extra_info"]["template"] for row in rows)

    texts = [_prompt_question(row) for row in rows]
    labels = [1 if _payload(row)["solvable"] else 0 for row in rows]
    length_cue = length_cue_balanced_accuracy(texts, labels)

    ok = len(suffixes) == 1 and len(templates) == 1 and length_cue <= 0.60
    detail = (
        f"distinct prompt suffixes={len(suffixes)}; distinct template constants="
        f"{dict(templates)}; question-length cue out-of-fold balanced accuracy="
        f"{length_cue:.3f} (chance 0.5)"
    )
    return result("L3b template isomorphism / length cue", ok, detail)


def _token_lists(texts: list[str]) -> tuple[list[np.ndarray], int]:
    """Per-document token ids into one shared vocabulary, plus its size."""
    index: dict[str, int] = {}
    docs: list[np.ndarray] = []
    for text in texts:
        ids = [index.setdefault(token.casefold(), len(index)) for token in schema.words(text)]
        docs.append(np.asarray(ids, dtype=np.int64))
    return docs, len(index)


def nb_out_of_fold(
    texts: list[str],
    labels: list[int],
    folds: int = NB_FOLDS,
    seed: int = 0,
    min_support: int = NB_MIN_SUPPORT,
    alpha: float = NB_ALPHA,
) -> tuple[list[int], np.ndarray]:
    """Out-of-fold predictions of a multinomial Naive Bayes, plus fold assignment.

    ``min_support`` restricts the vocabulary to tokens appearing in at least that
    many *training* documents.  The restriction matters: an unfiltered multinomial
    NB over a raw vocabulary has no probability mass for unseen tokens and its
    class score can flip sign out of fold, which makes a sub-chance reading
    meaningless (the UMWP recon report documents exactly that).  Here the
    vocabulary filter is a parameter so the caller can show the number both ways.
    """
    docs, vocab_size = _token_lists(texts)
    doc_ids = [np.unique(doc) for doc in docs]
    doc_counts = [
        np.bincount(doc, minlength=vocab_size) for doc in docs
    ]
    y = np.asarray(labels, dtype=np.int64)
    n = len(texts)
    order = np.arange(n)
    np.random.RandomState(seed).shuffle(order)
    fold_of = np.empty(n, dtype=np.int64)
    fold_of[order] = np.arange(n) % folds

    predictions = np.full(n, -1, dtype=np.int64)
    for fold in range(folds):
        train = np.nonzero(fold_of != fold)[0]
        test = np.nonzero(fold_of == fold)[0]
        document_frequency = np.zeros(vocab_size)
        for i in train:
            document_frequency[doc_ids[i]] += 1
        keep = document_frequency >= min_support
        counts = np.zeros((2, vocab_size))
        for cls in (0, 1):
            for i in train[y[train] == cls]:
                counts[cls] += doc_counts[i]
            counts[cls] = counts[cls] + alpha
        log_likelihood = np.zeros((2, vocab_size))
        for cls in (0, 1):
            denominator = counts[cls][keep].sum()
            if denominator > 0:
                log_likelihood[cls][keep] = np.log(counts[cls][keep] / denominator)
        priors = np.log(
            np.array(
                [
                    max(1, int((y[train] == cls).sum())) / max(1, len(train))
                    for cls in (0, 1)
                ]
            )
        )
        for i in test:
            scores = priors + log_likelihood[:, doc_ids[i]].sum(axis=1)
            predictions[i] = int(scores[1] > scores[0])
    return predictions.tolist(), fold_of


def balanced_accuracy(predictions: list[int], labels: list[int]) -> float:
    tp = fp = tn = fn = 0
    for pred, true in zip(predictions, labels, strict=True):
        if true == 1 and pred == 1:
            tp += 1
        elif true == 1 and pred == 0:
            fn += 1
        elif true == 0 and pred == 0:
            tn += 1
        else:
            fp += 1
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return (sensitivity + specificity) / 2


def length_cue_balanced_accuracy(
    texts: list[str], labels: list[int], folds: int = NB_FOLDS, seed: int = 0
) -> float:
    """Out-of-fold balanced accuracy of "the longer question is the solvable one".

    The threshold is chosen on the training fold only.  This is the cheap
    surface cue the equal-length rule exists to kill for option sources; for an
    option-less source it is the one cue left to check.
    """
    lengths = np.array([word_count(text) for text in texts])
    y = np.asarray(labels, dtype=np.int64)
    n = len(texts)
    order = np.arange(n)
    np.random.RandomState(seed).shuffle(order)
    fold_of = np.empty(n, dtype=np.int64)
    fold_of[order] = np.arange(n) % folds
    predictions = np.zeros(n, dtype=np.int64)
    for fold in range(folds):
        train = np.nonzero(fold_of != fold)[0]
        test = np.nonzero(fold_of == fold)[0]
        best_threshold, best_score = 0, -1.0
        for threshold in range(0, int(lengths[train].max()) + 2):
            above = np.where(lengths[train] > threshold, 1, 0)
            score = balanced_accuracy(above.tolist(), y[train].tolist())
            if score > best_score:
                best_threshold, best_score = threshold, score
        predictions[test] = np.where(lengths[test] > best_threshold, 1, 0)
    return balanced_accuracy(predictions.tolist(), y.tolist())


def check_l3_nb(rows: list[dict], gate: float = NB_BALANCED_ACCURACY_GATE) -> dict:
    """(c) the bag-of-words NB gate, reported verbatim whether it passes or not."""
    texts = [_prompt_question(row) for row in rows]
    labels = [1 if _payload(row)["solvable"] else 0 for row in rows]
    if len(set(labels)) < 2:
        return result(
            "L3c bag-of-words NB",
            None,
            "single-class artifact: the cross-label check does not apply",
        )
    predictions, _ = nb_out_of_fold(texts, labels)
    balanced = balanced_accuracy(predictions, labels)
    plain = sum(p == y for p, y in zip(predictions, labels, strict=True)) / len(labels)

    # Vocabulary-size ablations are printed for interpretation only, never gated.
    stopword_only = balanced_accuracy(
        *(
            nb_out_of_fold(
                [" ".join(t for t in schema.words(t) if t.casefold() in schema.STOPWORDS) for t in texts],
                labels,
            )[0],
            labels,
        )
    )
    content_only = balanced_accuracy(
        *(
            nb_out_of_fold(
                [" ".join(t for t in schema.words(t) if t.casefold() not in schema.STOPWORDS) for t in texts],
                labels,
            )[0],
            labels,
        )
    )
    ok = balanced <= gate
    detail = (
        f"out-of-fold balanced accuracy={balanced:.3f} (gate <= {gate:.2f}, chance 0.50), "
        f"plain accuracy={plain:.3f}, min_support={NB_MIN_SUPPORT}, folds={NB_FOLDS}; "
        f"ablations: stopwords-only={stopword_only:.3f}, content-only={content_only:.3f}, "
        f"length-only={length_cue_balanced_accuracy(texts, labels):.3f}; "
        f"n_solvable={labels.count(1)} n_unsolvable={labels.count(0)}"
    )
    return result("L3c bag-of-words NB", ok, detail)


# ---------------------------------------------------------------------------
# schema + raw-source checks
# ---------------------------------------------------------------------------


def check_schema(rows: list[dict]) -> dict:
    violations: list[str] = []
    for row in rows:
        violations.extend(schema.validate_row(row))
    ok = not violations
    detail = f"{len(rows)} rows, {len(violations)} violations"
    if violations:
        detail += ": " + "; ".join(violations[:5])
    return result("schema.validate_row", ok, detail)


def check_raw_source_counts(raw_dir: str | None) -> dict:
    """Re-measure the raw file's own numbers (design doc section 9, KUQ line)."""
    if not raw_dir:
        return result("raw-source counts", None, "no --raw-dir given")
    path = Path(raw_dir) / RAW_FILENAME
    if not path.is_file():
        return result("raw-source counts", None, f"{path} not available on this machine")
    unknown = 0
    known = 0
    malformed_lines = 0
    unknown_with_category = 0
    known_with_category = 0
    categories: Counter = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                malformed_lines += 1
                continue
            if not isinstance(record, dict) or "unknown" not in record:
                malformed_lines += 1
                continue
            has_category = isinstance(record.get("category"), str) and bool(
                record["category"].strip()
            )
            if record["unknown"]:
                unknown += 1
                if has_category:
                    unknown_with_category += 1
                    categories[record["category"]] += 1
            else:
                known += 1
                if has_category:
                    known_with_category += 1
    total = known + unknown
    six = [category for category in categories if category in KUQ_CATEGORIES]
    ok = (
        total == 6884
        and unknown == 3437
        and known == 3447
        and malformed_lines == 0
        and len(six) == 6
        and unknown_with_category == unknown
        and known_with_category == 0
    )
    detail = (
        f"rows={total} (expect 6884), unknown={unknown} (expect 3437), "
        f"known={known} (expect 3447), unparseable lines={malformed_lines} "
        f"(expect 0), unknown rows carrying a category="
        f"{unknown_with_category}/{unknown}, known rows carrying a category="
        f"{known_with_category} (expect 0), categories present={len(six)}/6 "
        f"{dict(sorted(categories.items()))}"
    )
    return result("raw-source counts", ok, detail)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def audit(
    rows: list[dict],
    raw_dir: str | None = None,
    l1_sample_size: int = L1_MIN_SAMPLE,
    nb_gate: float = NB_BALANCED_ACCURACY_GATE,
) -> list[dict]:
    """Run every check and return the outcomes in report order."""
    if not rows:
        return [result("artifact is non-empty", False, "no rows read")]
    return [
        check_schema(rows),
        check_l1_certificate(rows, l1_sample_size),
        check_l2_uniqueness(rows),
        check_l3_option_cues(rows),
        check_l3_template_isomorphism(rows),
        check_l3_nb(rows, gate=nb_gate),
        check_raw_source_counts(raw_dir),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a built KUQ parquet.")
    parser.add_argument("--rows", required=True, help="parquet produced by kuq_adapter.py")
    parser.add_argument(
        "--raw-dir",
        default=DEFAULT_RAW_DIR,
        help="optional: re-measure knowns_unknowns.jsonl's own counts",
    )
    parser.add_argument("--l1-sample", type=int, default=L1_MIN_SAMPLE)
    args = parser.parse_args()

    rows = schema.read_parquet_rows(args.rows)
    outcomes = audit(rows, raw_dir=args.raw_dir, l1_sample_size=args.l1_sample)
    for outcome in outcomes:
        status = {True: "PASS", False: "FAIL", None: "SKIP"}[outcome["ok"]]
        print(f"[{status}] {outcome['name']}")
        print(f"        {outcome['detail']}")

    failed = [outcome for outcome in outcomes if outcome["ok"] is False]
    print(
        f"\n{len(outcomes) - len(failed)}/{len(outcomes)} checks passed "
        f"({sum(1 for o in outcomes if o['ok'] is None)} skipped)"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
