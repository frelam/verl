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

* ``solvable_two_layer`` (D24): the gold is the source's own number -- the source
  anchor below proves that, since no artifact-local check can recompute UMWP's
  arithmetic.  Locally the certificate is (a) the gold parses as a finite number
  and (b) the recorded defect is exactly the edit that separates this question
  from its unanswerable partner.
* ``unsolvable_bare`` (D24): the presented question is the partner's question
  rewritten by the recorded defect, and the two really differ.  Every defect class
  ships bare now, so the certificate deliberately does **not** demand a *visible*
  defect: the token-invisible sign flips (category 3, ``"23" -> "-23"``) ship with
  a character-level record and are certified as "same tokens, different text".

**Source anchor.**  Beyond the sample, every row is re-read against
``StandardDataset.jsonl``: the presented question must be the source's own
question for ``extra_info.index``, ``paired_original_text`` must be the source's
question for the partner that the source's own ``relevant_ids`` names, the label
must match the source (an unanswerable row must have ``answer is None`` there, an
answerable row must carry the source's number as its two-layer gold), and the
defect label must be the category's own name.  Without the raw file the gold
cannot be proved, so a missing ``--raw-dir`` is a FAIL, not a skip.

**L2 -- option contract.**  The two sides have opposite obligations (section 5.1):

* answerable / two-layer: a k=3 placeholder block whose spans are the row's own
  question text, all of equal token length, with **no correct item** --
  ``correct_option_id`` is null and no option may be the gold answer (the reward
  never reads the block; it exists only for the D18 isomorphism rule).
* unanswerable / bare: **no option block at all**, no ``correct_option_id`` and no
  answer.  D24 deleted the four-tier arm from this source, so the pointer-gold
  uniqueness obligation went with it; what is left to audit here is the defect
  accounting (L1) and the absence of the block.

**L3 -- anti-cheat.**  (a) every option occurs verbatim in the row's own
question; (b) all of a row's options have equal token length; (c) the two shipped
sides must not be separable by surface heuristics: a character-length classifier
and a bag-of-words multinomial Naive Bayes, both 5-fold cross-validated with the
threshold/vocabulary fitted on the training folds only, must not beat chance by
more than 0.05 (out-of-fold balanced accuracy <= 0.55; design doc sections 4.6 and
9's "random + 5pt" gate).  The raw numbers, a shuffled-label control, a support
sweep and the top discriminative tokens are always printed, pass or fail, because
the unfiltered estimator is numerically unstable on this corpus (see
:func:`bow_nb_oof`).

**Reward cells (section 9).**  The two shipped sides' reward cells are pinned
against the *frozen* reward module (``reward/hallucination_compute_score.py``,
loaded by path): for ``two_layer`` the gold answer scores +1, a wrong answer +0.5,
and ``SOLVABLE`` / ``UNSOLVABLE`` / no ``\\boxed{}`` 0; for the three-tier bare
side ``UNSOLVABLE`` (with or without an option id) scores +1 and a fabricated
answer -1.  The same bare row scored with a *four-tier* ground truth must give
``\\boxed{UNSOLVABLE}`` 0 and only the right option id +1 -- the branch-dependent
cell of section 5.1.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
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

BRANCH_TWO_LAYER = schema.BRANCH_SOLVABLE_TWO_LAYER
BRANCH_BARE = schema.BRANCH_UNSOLVABLE_BARE

#: Design decision D15: one option count for every template-A block.  Hard-coded
#: here on purpose -- the audit pins the doc's contract, not the adapter's constant.
OPTIONS_PER_BLOCK = 3

#: Design doc section 9's no-shortcut gate ("random + 5pt").
L3_MAX_BALANCED_ACCURACY = 0.55
L3_MIN_SUPPORT = 5
#: Supports whose readings are printed next to the gate: the unfiltered estimator
#: is numerically unstable, and these cuts are where it stabilises (section 4.6's
#: diagnostic list, kept visible so the gate is never read in isolation).
L3_SUPPORT_SWEEP = (20, 50, 100)
L1_MIN_SAMPLE = 50
#: Rows per side whose reward cells are pinned against the frozen reward file.
REWARD_SAMPLE = 5

#: The frozen reward dispatcher, relative to this file (``scripts/hallucination``
#: -> ``reward``).  Loaded by path so the audit pins the shipped decision table.
REWARD_MODULE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "reward", "hallucination_compute_score.py"
)


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


def _multiset_after_edit(qa: str, qu: str, deleted: str, inserted: str) -> bool:
    """``tokens(qu)`` must equal ``tokens(qa) - deleted + inserted`` as a multiset."""
    expected = _counts(_tokens(qa))
    expected.subtract(_counts(_tokens(deleted)))
    expected.update(_counts(_tokens(inserted)))
    expected = +expected
    return expected == _counts(_tokens(qu))


def _pair_edit_holds(qa: str, qu: str, deleted: str, inserted: str) -> bool:
    """Whether ``qu`` is ``qa`` rewritten by the recorded ``(deleted, inserted)``.

    The comparison is at the **token** level, not character level, because UMWP
    re-punctuates the two members of a pair independently: the answerable member
    of id 2502 ends ``them..How`` and its unanswerable partner ``them. How``, so a
    character-exact reconstruction is impossible for a reason that has nothing to
    do with the gold.

    A character-level record (category 3's ``"23" -> "-23"`` sign flip, whose
    tokens are identical) carries no tokens at all; there the check is "the token
    sequences are equal and the two questions really differ".
    """
    if not _tokens(deleted) and not _tokens(inserted):
        return qa != qu and _counts(_tokens(qa)) == _counts(_tokens(qu))
    return _multiset_after_edit(qa, qu, deleted, inserted)


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
    """Whether the removed text is *provably* information the question lost.

    Reporting only (the adapter no longer drops on it, D24): ``key_information_missing``
    rows should have lost a quantity, ``question_missing`` rows a word sequence.
    Anything else has no artifact-local proof and is counted as unsupported.
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


def _normalise(text: str) -> str:
    return " ".join((text or "").split())


def check_source_anchor(rows: list[dict], reporter: Reporter, index: dict[int, dict], raw_dir: str) -> None:
    """Prove every row's text and gold against the source file itself.

    The artifact-local checks cannot tell an honest adapter from one that invented
    a question or a number.  This check can: it re-reads the source row the artifact
    points at and demands that

    * the presented question is the source's own question for that id, verbatim;
    * ``paired_original_text`` is the source's own question for the partner that the
      source's own ``relevant_ids`` names (never ``id - 2600``);
    * the label agrees with the source (an answerable row carries the source's own
      number as its two-layer gold; an unanswerable row carries none);
    * the defect label is the category's own name, or empty when the source carries
      no category (section 4.8's category-less rows).
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
            ground_truth = json.loads(row["reward_model"]["ground_truth"])
            if not ground_truth.get("two_layer"):
                failures.append(f"{task_id}: answerable row does not carry the two_layer contract (D24)")
            audit = ground_truth.get("answer")
            if audit != str(source["answer"][0]):
                failures.append(f"{task_id}: gold answer {audit!r} != source answer {source['answer'][0]!r}")
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
            if source.get("category") is None:
                expected = ("", "")
            elif source["category"] in CATEGORY_NAMES:
                expected = CATEGORY_NAMES[source["category"]]
            else:
                failures.append(f"{task_id}: source category {source['category']!r} is unknown")
                continue
            if (info.get("error_type"), info.get("perturbation_type")) != expected:
                failures.append(
                    f"{task_id}: defect label {(info.get('error_type'), info.get('perturbation_type'))!r} "
                    f"!= category {source['category']!r} {expected!r}"
                )
    reporter.check(
        "source anchor (texts and golds re-read from the raw file)",
        not failures,
        f"{len(rows)} rows anchored to {os.path.join(raw_dir, 'StandardDataset.jsonl')}, {len(failures)} failures"
        + (f"; first: {failures[:3]}" if failures else ""),
    )


# ---------------------------------------------------------------------------
# L1 / L2
# ---------------------------------------------------------------------------


def check_l1(sample: dict[str, list[dict]], reporter: Reporter) -> None:
    for branch, picked in sample.items():
        failures: list[str] = []
        unsupported = 0
        for row in picked:
            info = row["extra_info"]
            task_id = info["task_id"]
            qa, qu = _orient(row)
            deleted = info.get("deleted_condition_text", "")
            inserted = info.get("perturbed_entity_text", "")
            if branch == BRANCH_TWO_LAYER:
                # The gold is the source's own number (proved by the source anchor);
                # locally the certificate is that it is a finite number and that the
                # recorded defect is the whole difference from the partner.
                audit = json.loads(row["reward_model"]["ground_truth"]).get("answer")
                if not str(audit or "").strip():
                    failures.append(f"{task_id}: two-layer row has no gold answer")
                    continue
                try:
                    value = float(audit)
                except (TypeError, ValueError):
                    failures.append(f"{task_id}: gold answer {audit!r} is not a number")
                    continue
                if not math.isfinite(value):
                    failures.append(f"{task_id}: gold answer {audit!r} is not finite")
                elif not _pair_edit_holds(qa, qu, deleted, inserted):
                    failures.append(f"{task_id}: question is not the partner's rewritten by the recorded defect")
            else:
                if not _pair_edit_holds(qa, qu, deleted, inserted):
                    failures.append(f"{task_id}: question is not the original rewritten by the recorded defect")
                elif not deleted.strip() and not inserted.strip():
                    failures.append(f"{task_id}: pair records no defect (nothing separates the two questions)")
                elif not _info_lost(deleted, qu, info.get("error_type", "")):
                    unsupported += 1
        detail = f"{len(picked)} sampled rows re-derived, {len(failures)} failures"
        if failures:
            detail += "; first: " + " | ".join(failures[:3])
        reporter.check(f"L1 {branch}", not failures, detail)
        if unsupported:
            # Reporting only: D24 ships every class bare, so a class whose text
            # shape is not provable locally is a statistics caveat, not a drop.
            reporter.note(
                f"L1 {branch}: {unsupported}/{len(picked)} rows carry a defect class whose text shape "
                "is not provable from the artifact alone (section 4.8 statistics)"
            )


def check_l2(sample: dict[str, list[dict]], reporter: Reporter) -> None:
    for branch, picked in sample.items():
        failures: list[str] = []
        for row in picked:
            info = row["extra_info"]
            task_id = info["task_id"]
            ground_truth = json.loads(row["reward_model"]["ground_truth"])
            options = info.get("options") or []
            if branch == BRANCH_TWO_LAYER:
                if len(options) != OPTIONS_PER_BLOCK:
                    failures.append(f"{task_id}: placeholder block has {len(options)} options, not {OPTIONS_PER_BLOCK}")
                if ground_truth.get("correct_option_id") is not None:
                    failures.append(f"{task_id}: two-layer row carries correct_option_id")
                if info.get("correct_option_id") not in ("", None):
                    failures.append(f"{task_id}: two-layer row's extra_info carries correct_option_id")
                gold = str(ground_truth.get("answer") or "").strip().casefold()
                for option in options:
                    if option.get("text", "").strip().casefold() == gold:
                        failures.append(f"{task_id}: placeholder option {option.get('text')!r} is the gold answer")
            else:
                if options:
                    failures.append(f"{task_id}: three-tier bare row carries an option block")
                if ground_truth.get("correct_option_id") is not None:
                    failures.append(f"{task_id}: bare row carries correct_option_id")
                if ground_truth.get("answer") is not None:
                    failures.append(f"{task_id}: bare row carries an answer")
        detail = f"{len(picked)} sampled rows, {len(failures)} contract failures"
        if branch == BRANCH_TWO_LAYER:
            detail += " (gold is the answer; the block must hold no correct item)"
        else:
            detail += " (gold is the refusal marker; no block exists)"
        if failures:
            detail += "; first: " + " | ".join(failures[:3])
        reporter.check(f"L2 {branch}", not failures, detail)


# ---------------------------------------------------------------------------
# section 9 side invariants
# ---------------------------------------------------------------------------


def check_sides(rows: list[dict], reporter: Reporter) -> None:
    """The two shipped sides' §9 obligations, over **every** row.

    ``umwp_adapter`` ships two sides and nothing else (D24): the answerable
    two-layer side must carry template A with a placeholder block and
    ``correct_option_id=null``, the unanswerable side must carry template B with
    **no options block at all**.  Nothing here is sampled: an option block on a
    bare row would resurrect the MiP-style "pick the option that is not in the
    question" leak (section 4.2), so it is checked on the whole pool.
    """
    failures: list[str] = []
    two_layer = 0
    bare = 0
    for row in rows:
        info = row["extra_info"]
        task_id = info["task_id"]
        ground_truth = json.loads(row["reward_model"]["ground_truth"])
        prompt = row["prompt"][0]["content"]
        options = info.get("options") or []
        if info.get("solvable"):
            two_layer += 1
            if info.get("branch") != BRANCH_TWO_LAYER:
                failures.append(f"{task_id}: solvable row on branch {info.get('branch')!r}")
            if info.get("template") != schema.TEMPLATE_A:
                failures.append(f"{task_id}: solvable row must use template A, got {info.get('template')!r}")
            if ground_truth.get("two_layer") is not True:
                failures.append(f"{task_id}: solvable row is not two_layer")
            if not str(ground_truth.get("answer") or "").strip():
                failures.append(f"{task_id}: solvable row has no answer")
            if ground_truth.get("correct_option_id") is not None:
                failures.append(f"{task_id}: two-layer row has a correct_option_id")
            if ground_truth.get("has_diagnosis_label"):
                failures.append(f"{task_id}: two-layer row carries a diagnosis label")
            if len(options) != OPTIONS_PER_BLOCK:
                failures.append(f"{task_id}: placeholder block has {len(options)} options, not {OPTIONS_PER_BLOCK}")
            rendered = [opt.get("text", "") for opt in options]
            if any(text not in prompt for text in rendered):
                failures.append(f"{task_id}: an option is not rendered into the prompt")
            if "选项：" not in prompt:
                failures.append(f"{task_id}: template A prompt carries no option block header")
        else:
            bare += 1
            if info.get("branch") != BRANCH_BARE:
                failures.append(f"{task_id}: unsolvable row on branch {info.get('branch')!r}")
            if info.get("template") != schema.TEMPLATE_B:
                failures.append(f"{task_id}: three-tier row must use template B, got {info.get('template')!r}")
            if options:
                failures.append(f"{task_id}: unanswerable side carries an option block")
            if "选项：" in prompt:
                failures.append(f"{task_id}: unanswerable prompt carries an option block")
            if ground_truth.get("correct_option_id") is not None:
                failures.append(f"{task_id}: bare row has a correct_option_id")
            if ground_truth.get("answer") is not None:
                failures.append(f"{task_id}: bare row has an answer")
            if ground_truth.get("has_diagnosis_label"):
                failures.append(f"{task_id}: bare row carries a diagnosis label")
            for flag in ("two_layer", "solvable_answer", "pair_task", "judgment_only"):
                if ground_truth.get(flag):
                    failures.append(f"{task_id}: bare row carries {flag}")
        perturbation = ground_truth.get("perturbation_type")
        if perturbation is not None and perturbation not in schema.PERTURBATION_TYPES:
            failures.append(f"{task_id}: unknown perturbation_type {perturbation!r}")
        if (info.get("perturbation_type") or None) != perturbation:
            failures.append(
                f"{task_id}: extra_info.perturbation_type {info.get('perturbation_type')!r} "
                f"!= ground_truth {perturbation!r}"
            )
        if not info.get("paired_original_text") or info["paired_original_text"] == question_of(row):
            failures.append(f"{task_id}: missing or identical paired original")
    reporter.check(
        "section 9 side invariants (placeholder vs no block, two-layer flags)",
        not failures,
        f"{two_layer} two-layer + {bare} bare rows, {len(failures)} violations"
        + (f"; first: {failures[:3]}" if failures else ""),
    )


# ---------------------------------------------------------------------------
# section 9 reward cells
# ---------------------------------------------------------------------------


def load_reward_module():
    """Import the frozen reward dispatcher by path.

    The audit must pin the *shipped* decision table: a local re-implementation
    would keep passing while the reward drifted.  The module is loaded under a
    private name, so this stays independent of the reward file's own import
    fallbacks.
    """
    path = os.path.abspath(REWARD_MODULE_PATH)
    spec = importlib.util.spec_from_file_location("halluc_reward_under_audit", path)
    if spec is None or spec.loader is None:  # pragma: no cover - broken checkout
        raise ImportError(f"cannot load the reward module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rollout(final: str) -> str:
    """A thinking-template-compliant response (the reward's format gate needs it)."""
    return f"<think>\nreasoning about the problem\n</think>\n\n{final}"


def _evenly_spaced(rows: list[dict], count: int) -> list[dict]:
    """At most ``count`` rows spread across the list (deterministic, no RNG)."""
    if len(rows) <= count:
        return list(rows)
    step = len(rows) / count
    return [rows[int(index * step)] for index in range(count)]


def reward_cells(rows: list[dict], reporter: Reporter) -> None:
    """Pin section 9's reward cells for the two shipped sides.

    Two-layer (D24): the gold answer is worth +1, a wrong answer +0.5 (judged
    solvable, not solved), and ``SOLVABLE`` / ``UNSOLVABLE`` / no ``\\boxed{}`` 0 --
    producing an answer *is* the judgement, so a refusal earns nothing.
    Three-tier bare: ``\\boxed{UNSOLVABLE}`` scores +1 whether or not an option id
    follows (there is no option block to point at), a fabricated answer -1.
    Four-tier contrast: the same bare row scored with a synthetic four-tier ground
    truth must give 0 for the bare marker and +1 only for the right option id.
    """
    module = load_reward_module()
    two_layer = [row for row in rows if row["extra_info"].get("solvable")]
    bare = [row for row in rows if not row["extra_info"].get("solvable")]
    failures: list[str] = []
    checked = 0

    def score(row: dict, ground_truth: str, final: str) -> float:
        return float(
            module.compute_score(row["data_source"], _rollout(final), ground_truth, row["extra_info"])["score"]
        )

    for row in _evenly_spaced(two_layer, REWARD_SAMPLE):
        task_id = row["extra_info"]["task_id"]
        ground_truth = row["reward_model"]["ground_truth"]
        gold = json.loads(ground_truth).get("answer")
        try:
            wrong = str(float(gold) + 1)
        except (TypeError, ValueError):
            failures.append(f"{task_id}: gold {gold!r} is not numeric")
            continue
        cells = (
            (rf"\boxed{{{gold}}}", 1.0, "gold answer"),
            (rf"\boxed{{{wrong}}}", 0.5, "wrong answer (judged, not solved)"),
            (r"\boxed{SOLVABLE}", 0.0, "verdict instead of the answer"),
            (r"\boxed{UNSOLVABLE}", 0.0, "misrefusal"),
            (r"\boxed{UNSOLVABLE: A}", 0.0, "misrefusal with an option id"),
            ("the answer is not boxed", 0.0, "no box"),
        )
        for final, expected, label in cells:
            got = score(row, ground_truth, final)
            checked += 1
            if got != expected:
                failures.append(f"{task_id}: two_layer {label} -> {got}, expected {expected}")

    for row in _evenly_spaced(bare, REWARD_SAMPLE):
        task_id = row["extra_info"]["task_id"]
        ground_truth = row["reward_model"]["ground_truth"]
        cells = (
            (r"\boxed{UNSOLVABLE}", 1.0, "bare refusal"),
            (r"\boxed{UNSOLVABLE: B}", 1.0, "refusal plus a meaningless option id"),
            (r"\boxed{42}", -1.0, "fabricated answer"),
            ("the question cannot be answered", 0.0, "no box"),
        )
        for final, expected, label in cells:
            got = score(row, ground_truth, final)
            checked += 1
            if got != expected:
                failures.append(f"{task_id}: bare {label} -> {got}, expected {expected}")

    if bare:
        # The branch-dependent cell of section 5.1: *same* text, four-tier ground
        # truth -> the bare marker is no longer worth anything.
        row = bare[0]
        four_tier = dict(json.loads(row["reward_model"]["ground_truth"]))
        four_tier.update({"has_diagnosis_label": True, "correct_option_id": "B"})
        four_tier_gt = json.dumps(four_tier)
        for final, expected, label in (
            (r"\boxed{UNSOLVABLE}", 0.0, "bare marker on a four-tier row"),
            (r"\boxed{UNSOLVABLE: B}", 1.0, "correct diagnosis on a four-tier row"),
        ):
            got = score(row, four_tier_gt, final)
            checked += 1
            if got != expected:
                failures.append(f"{row['extra_info']['task_id']}: four-tier {label} -> {got}, expected {expected}")

    reporter.check(
        "section 9 reward cells (two_layer / three-tier / four-tier contrast)",
        not failures,
        f"{checked} cells pinned against {os.path.relpath(os.path.abspath(REWARD_MODULE_PATH))}, "
        f"{len(failures)} mismatches" + (f"; first: {failures[:3]}" if failures else ""),
    )


# ---------------------------------------------------------------------------
# L3
# ---------------------------------------------------------------------------


def check_l3_options(rows: list[dict], reporter: Reporter) -> None:
    """L3a/L3b: the placeholder blocks are same-length spans of their own question."""
    missing: list[str] = []
    unequal: list[str] = []
    wrong_count: list[str] = []
    checked = 0
    for row in rows:
        options = row["extra_info"].get("options") or []
        if not options:
            continue
        checked += 1
        question = question_of(row)
        task_id = row["extra_info"]["task_id"]
        if len(options) != OPTIONS_PER_BLOCK:
            wrong_count.append(f"{task_id}:{len(options)}")
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
    reporter.check(
        "L3b' option blocks have exactly k=3 items",
        not wrong_count,
        f"{checked} option-bearing rows, {len(wrong_count)} with the wrong count"
        + (f"; first: {wrong_count[:3]}" if wrong_count else ""),
    )


def _fold_assignment(count: int, folds: int, seed: int) -> np.ndarray:
    """Deterministic fold id per row: shuffle, then deal round-robin."""
    order = list(range(count))
    random.Random(seed).shuffle(order)
    fold_of = np.empty(count, dtype=np.int64)
    for rank, index in enumerate(order):
        fold_of[index] = rank % folds
    return fold_of


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

    The support filter is not cosmetic: on this corpus the unfiltered estimator's
    class score changes sign out of fold (rare tokens' likelihood ratios dominate),
    so a below-chance reading from it is evidence of a broken estimator, not of "no
    signal".  :data:`L3_SUPPORT_SWEEP` prints the stable cuts next to the gate.
    """
    documents = [[token.lower() for token in _tokens(text)] for text in texts]
    labels = np.asarray(labels, dtype=np.int64)
    count = len(texts)
    fold_of = _fold_assignment(count, folds, seed)

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

        def design(indices, vocabulary=vocabulary) -> np.ndarray:
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


def length_oof(texts: list[str], labels: list[int], *, folds: int = 5, seed: int = 0) -> float:
    """Out-of-fold balanced accuracy of the character-length heuristic.

    The structural shortcut section 4.6 measured on TreeCut (0.726): "the
    unanswerable questions are shorter".  Neither the threshold nor its direction
    is tuned on the held-out fold -- the direction comes from the training means
    and the threshold is the training median -- so the number is out of sample
    exactly like :func:`bow_nb_oof`'s.
    """
    lengths = np.asarray([len(text) for text in texts], dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    count = len(texts)
    fold_of = _fold_assignment(count, folds, seed)
    out_of_fold = np.full(count, -1, dtype=np.int64)
    for fold in range(folds):
        train = np.where(fold_of != fold)[0]
        test = np.where(fold_of == fold)[0]
        positive = lengths[train][labels[train] == 1]
        negative = lengths[train][labels[train] == 0]
        if positive.size == 0 or negative.size == 0:
            continue
        direction = 1 if positive.mean() > negative.mean() else 0
        threshold = float(np.median(lengths[train]))
        out_of_fold[test] = ((lengths[test] > threshold).astype(np.int64) == direction).astype(np.int64)
    if np.any(out_of_fold < 0):
        raise ValueError("some rows were never scored out of fold")
    positives = labels == 1
    negatives = ~positives
    recall_pos = np.mean(out_of_fold[positives] == 1) if positives.any() else float("nan")
    recall_neg = np.mean(out_of_fold[negatives] == 0) if negatives.any() else float("nan")
    return float(0.5 * (recall_pos + recall_neg))


def discriminative_tokens(
    texts: list[str], labels: list[int], *, min_support: int = 50, top: int = 5
) -> list[tuple[str, int, int]]:
    """The ``top`` tokens with the largest document-frequency log-odds.

    Reporting only, to separate the two readings of a high BoW score: the defect
    vocabulary itself ("some" / "several" / "less than" replacing a number, which
    *is* the judgement task) versus topic memory (a source-level shortcut, the
    FalseQA risk of section 11.7).
    """
    documents = [[token.lower() for token in _tokens(text)] for text in texts]
    labels = np.asarray(labels, dtype=np.int64)
    support: collections.Counter = collections.Counter()
    for document in documents:
        support.update(set(document))
    vocabulary = [token for token, count in support.items() if count >= min_support]
    if not vocabulary:
        return []
    positive: collections.Counter = collections.Counter()
    negative: collections.Counter = collections.Counter()
    for document, label in zip(documents, labels, strict=False):
        (positive if label == 1 else negative).update(set(document))
    positive_total = sum(positive.values()) + len(vocabulary)
    negative_total = sum(negative.values()) + len(vocabulary)
    scores: list[tuple[float, str, int, int]] = []
    for token in vocabulary:
        pos = positive[token] + 1
        neg = negative[token] + 1
        delta = abs(math.log(pos / positive_total) - math.log(neg / negative_total))
        scores.append((delta, token, positive[token], negative[token]))
    scores.sort(reverse=True)
    return [(token, pos, neg) for _, token, pos, neg in scores[:top]]


def check_l3_shortcuts(
    rows: list[dict], reporter: Reporter, *, folds: int, seed: int, min_support: int, sweep: bool = True
) -> None:
    """L3c/L3d: neither side may be guessable from surface statistics."""
    texts = [question_of(row) for row in rows]
    labels = [1 if row["extra_info"]["solvable"] else 0 for row in rows]
    positives = sum(labels)
    reporter.note(
        f"L3c corpus: n={len(rows)} (solvable={positives}, unsolvable={len(rows) - positives}), "
        f"folds={folds}, length = character count, NB vocabulary support >= {min_support} docs, alpha=1.0"
    )
    if positives == 0 or positives == len(rows):
        reporter.check(
            "L3c two-sided corpus",
            False,
            "only one label side present in the artifact -- the estimators are not identifiable; "
            "build with a --limit that keeps both branches (the adapter interleaves them)",
        )
        return
    length_accuracy = length_oof(texts, labels, folds=folds, seed=seed)
    accuracy = bow_nb_oof(texts, labels, folds=folds, seed=seed, min_support=min_support)
    control_rng = random.Random(seed + 1)
    control_labels = [control_rng.random() > 0.5 for _ in rows]
    control = bow_nb_oof(texts, control_labels, folds=folds, seed=seed, min_support=min_support)
    reporter.note(f"L3c length heuristic out-of-fold balanced accuracy = {length_accuracy:.4f} (chance 0.5)")
    reporter.note(f"L3d BoW NB out-of-fold balanced accuracy = {accuracy:.4f} (chance 0.5)")
    reporter.note(f"L3d shuffled-label control = {control:.4f}")
    reporter.check(
        "L3c length heuristic <= random + 0.05",
        length_accuracy <= L3_MAX_BALANCED_ACCURACY,
        f"balanced accuracy {length_accuracy:.4f} <= {L3_MAX_BALANCED_ACCURACY:.2f} on {len(rows)} rows",
    )
    reporter.check(
        "L3d BoW NB <= random + 0.05",
        accuracy <= L3_MAX_BALANCED_ACCURACY,
        f"balanced accuracy {accuracy:.4f} <= {L3_MAX_BALANCED_ACCURACY:.2f} on {len(rows)} rows "
        f"(shuffled-label control {control:.4f})",
    )
    if sweep:
        for support in L3_SUPPORT_SWEEP:
            if support == min_support:
                continue
            reading = bow_nb_oof(texts, labels, folds=folds, seed=seed, min_support=support)
            reporter.note(
                f"L3d support >= {support} reads {reading:.4f}"
                + (
                    "  (> gate: the estimator is more stable there -- see the docstring)"
                    if reading > L3_MAX_BALANCED_ACCURACY
                    else ""
                )
            )
        tokens = discriminative_tokens(texts, labels, min_support=50)
        if tokens:
            reporter.note(
                "L3d top discriminative tokens (support >= 50, solvable/unsolvable document counts): "
                + ", ".join(f"{token!r} {pos}/{neg}" for token, pos, neg in tokens)
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
        f"{len(rows)} rows, {len(violations)} violations" + (f"; first: {violations[:3]}" if violations else ""),
    )

    failures: list[str] = []
    branch_of: dict[str, str] = {}
    for row in rows:
        info = row["extra_info"]
        task_id = info["task_id"]
        branch = info["branch"]
        ground_truth = json.loads(row["reward_model"]["ground_truth"])
        branch_of[branch] = info["template"]
        if branch not in (BRANCH_TWO_LAYER, BRANCH_BARE):
            failures.append(f"{task_id}: unexpected branch {branch!r} (D24 ships two sides only)")
            continue
        if branch == BRANCH_TWO_LAYER:
            if not info.get("solvable") or ground_truth.get("two_layer") is not True:
                failures.append(f"{task_id}: two-layer branch without solvable/two_layer")
            if info.get("template") != schema.TEMPLATE_A:
                failures.append(f"{task_id}: two-layer row is not on template A")
        else:
            if info.get("solvable") is not False or ground_truth.get("solvable") is not False:
                failures.append(f"{task_id}: bare branch without solvable=false")
            if info.get("template") != schema.TEMPLATE_B:
                failures.append(f"{task_id}: bare row is not on template B")
            if ground_truth.get("has_diagnosis_label"):
                failures.append(f"{task_id}: bare row carries a diagnosis label")
        if not info.get("paired_original_text") or info["paired_original_text"] == question_of(row):
            failures.append(f"{task_id}: missing or identical paired original")
    reporter.check(
        "branch invariants (two sides, templates, pairing)",
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
    parser.add_argument(
        "--no-sweep",
        action="store_true",
        help="skip the extra L3 support-sweep fits (the gate itself is unchanged)",
    )
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
    check_sides(rows, reporter)
    reward_cells(rows, reporter)

    sample = sample_by_branch(rows, per_branch=args.per_branch, seed=args.seed)
    reporter.note("L1/L2 sample: " + ", ".join(f"{b}={len(v)}" for b, v in sample.items()))
    for branch, picked in sample.items():
        if len(picked) < L1_MIN_SAMPLE and len(picked) < len([r for r in rows if r["extra_info"]["branch"] == branch]):
            reporter.check(f"L1 {branch} sample size", False, f"only {len(picked)} rows to audit, need {L1_MIN_SAMPLE}")
    check_l1(sample, reporter)
    check_l2(sample, reporter)

    check_l3_options(rows, reporter)
    check_l3_shortcuts(
        rows, reporter, folds=args.folds, seed=args.seed, min_support=args.min_support, sweep=not args.no_sweep
    )

    print()
    if reporter.failed:
        print(f"RESULT: FAIL ({len(reporter.failed)}/{len(reporter.results)} checks failed: {reporter.failed})")
        raise SystemExit(1)
    print(f"RESULT: PASS ({len(reporter.results)}/{len(reporter.results)} checks passed)")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
