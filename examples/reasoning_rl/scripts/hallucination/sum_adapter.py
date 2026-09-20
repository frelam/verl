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
"""SUM adapter -- the pair judgement task (design doc D23; sections 4.5, 5.1, 5.2, 4.8, 9, 10).

Source
------

HF ``lime-nlp/Synthetic_Unanswerable_Math`` (MIT), the converted-parquet repo
(``refs/convert/parquet``), file ``train.parquet``: 36,480 rows, exactly three
columns -- ``answerable_question``, ``unanswerable_question`` and ``ground_truth``
(the answer to the **answerable** member).  Every row is already a pair, so this
adapter never has to match anything: no diff, no certificate, no branch routing.
The unanswerable labels are o3-mini generated and expert reviewed and the source
ships **no derivation chain** (section 4.5: "无机器证书"), so the label's quality
is the section 12 Q10 spot check in ``verify_sum.py``, not something this adapter
can prove.

``test.parquet`` supplies no pair and is read for the report only: its 284 rows
carry no ``answerable_question`` and one constant ``ground_truth`` string, so it
cannot form a row of this contract.

Contract (section 4.8 table B row 6; sections 5.1 / 5.2 template C)
-------------------------------------------------------------------

One **pair row per usable raw row**.  D23 retires the superseded diff-driven
three-branch design (``unsolvable_diag`` / ``unsolvable_bare`` /
``solvable_judge``): SUM is not shattered into four-tier / three-tier rows, it
gets one judgement-then-solve row that carries both sides by construction.

* prompt = template C with **both** questions, labelled ``问题 A`` / ``问题 B``;
* the A/B order is a **per-row seeded 50:50 draw**,
  ``random.Random(f"{seed}:pair:{index}")``.  Section 4.5 is explicit about why it
  cannot be fixed: A = answerable would let the model guess by position, and the
  measured bag-of-words reading (0.502 / 0.503) says there is no surface shortcut
  to replace it with;
* gold = ``pair_task=true``, ``answerable_id`` = the label the answerable question
  landed on, ``answer`` = the raw ``ground_truth`` column (stored verbatim, so the
  audit can compare it byte for byte), ``has_diagnosis_label=false``,
  ``perturbation_type=null`` -- section 3's SUM row example;
* reward (section 6): correct id + ``math_match`` answer -> 1.0; correct id +
  wrong answer -> 0.5; wrong or unparseable id -> 0, with **no answer-layer credit
  even when the answer text equals the gold**; a bare number, ``UNSOLVABLE`` or no
  ``\\boxed{}`` -> 0.

``answerable_id`` is by construction the label of the question whose text is the
raw ``answerable_question`` -- the section 9 contract assertion.

Quota and selection
-------------------

6,000 pairs, section 4.8 table B row 6.  A pair row is one row *and* one pair:
6,000 rows, not 12,000, and in the 60/40 accounting of section 4.8 it counts
0.5/0.5 each side -- 3,000 solvable-equivalent + 3,000 unsolvable-equivalent.
``--limit`` overrides the quota for scratch and fixture builds.

The selection is a deterministic **file-wide shuffle**
(``random.Random(f"{seed}:order")``), never "the first N": the raw file is ordered
by generation batch, so a prefix would confine the artifact to one corner of the
source.  The shuffle is seeded independently of the cap, so a small build is a
subset of a larger one and every row's A/B draw depends only on its own index.

Monitoring (section 10)
-----------------------

The row carries everything the doc's three SUM metrics need: ``answerable_id`` is
the judgement-layer gold -- its distribution *is* the position bias section 10 says
must stay near 50/50, and the report prints it -- and ``answer`` is the
answer-layer gold the reward's ``math_match`` compares against.

What ``verify_sum.py`` re-derives (section 9)
---------------------------------------------

Both questions non-empty and present in the prompt; the question named by
``answerable_id`` is byte-identical to the raw ``answerable_question`` and the
other to the raw ``unanswerable_question``; ``answer`` is the raw ``ground_truth``
verbatim; ``ground_truth`` is a JSON string carrying ``pair_task=true``; the A/B
distribution is 50:50 +-2pt over the artifact; the artifact holds at most 6,000
rows while the raw file supplies far more; the section 12 Q10 N=100 label spot
check; and the section 4.5 no-shortcut reading (bag-of-words Naive Bayes on the
two members of a pair against the 0.5 chance line).

Measured over the frozen file (seed 0, no ``--limit``): 36,480 raw rows, 0
malformed, 94 rows dropped as conflicting-gold duplicates and 44 as same-gold
duplicates -> **36,342 usable pairs**, i.e. 6.06x the 6,000 quota, so the cap is
what shapes the artifact rather than the source's supply.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import sys

try:
    import schema
except ImportError:  # pragma: no cover - running as a plain script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import schema

# ---------------------------------------------------------------------------
# source constants
# ---------------------------------------------------------------------------

DATA_SOURCE = schema.SOURCE_SUM
DEFAULT_RAW_DIR = "/home/charles/data/reasoning_rl/halluc/raw/sum"
DEFAULT_OUT = os.path.expanduser("~/data/reasoning_rl/halluc/built/sum.parquet")

DATA_FILE = "train.parquet"
TEST_FILE = "test.parquet"  # read for the report only; it supplies no pair

BRANCH_PAIR = schema.BRANCH_SOLVABLE_PAIR

#: Design doc section 4.8 table B row 6.  Counted in rows *and* pairs (one row is
#: one pair); the 60/40 accounting splits it 3,000 / 3,000 across the two sides.
PAIR_QUOTA = 6_000

#: ``extra_info.error_type`` for every row of this source.  Section 4.8's defect
#: table gives SUM the "混合" (mixed) cell -- the pair task counts a whole pair as
#: one mixed entry and never asks for a per-row defect type -- so this is an
#: audit/statistics field only, exactly like D24's "cat 分类只用于缺陷类型统计,
#: 不再决定契约分支".  The reward never reads it.
MIXED_ERROR_TYPE = "mixed"

_WHITESPACE_RE = re.compile(r"\s+")
#: A plain decimal answer (reporting only: how much of the answer layer is numeric
#: rather than an expression such as ``\frac{1}{3}``).
_NUMERIC_ANSWER_RE = re.compile(r"^[+-]?\d[\d,]*(?:\.\d+)?$")


def report_path_for(out: str) -> str:
    """The report that belongs to ``--out``: same stem, ``_report.json``.

    Derived from ``--out`` instead of pinned to the build directory, so that a
    scratch build (``--out /tmp/sum.parquet``) writes a scratch report rather than
    overwriting the canonical one the build report cites.
    """
    return os.path.splitext(out)[0] + "_report.json"


# ---------------------------------------------------------------------------
# text plumbing
# ---------------------------------------------------------------------------


def normalise_question(text: str) -> str:
    """Collapse whitespace and strip.

    The source carries embedded newlines, tabs and stray spacing.  Everything this
    adapter emits is the normalised form, which is also the form ``verify_sum.py``
    re-derives before its byte-identity checks -- mixing raw and normalised text is
    the bug this rule exists to prevent.
    """
    return _WHITESPACE_RE.sub(" ", (text or "")).strip()


def looks_numeric(text: str) -> bool:
    """Whether an answer is a plain decimal number (reporting only)."""
    return bool(_NUMERIC_ANSWER_RE.match((text or "").strip()))


# ---------------------------------------------------------------------------
# source loading and filtering
# ---------------------------------------------------------------------------


def load_source(raw_dir: str) -> list[dict]:
    """Read ``train.parquet`` and stamp each row with its positional index.

    The file has no id column, so ``_index`` (the row's position in the frozen
    file) is the row identity: it seeds the A/B draw, it is what makes the
    artifact's ``task_id`` reproducible, and it is what lets the audit re-read the
    exact source row.  ``test.parquet`` is deliberately not read (no pair, see the
    module docstring).
    """
    path = os.path.join(raw_dir, DATA_FILE)
    rows = schema.read_parquet_rows(path)
    for index, row in enumerate(rows):
        row["_index"] = index
    return rows


def _is_well_formed(row: object) -> bool:
    """A row is usable when all three texts are nonempty strings.

    Section 4.5's adapter contract: both questions non-empty and the gold
    parseable.  SUM's ``ground_truth`` is the answer *text* (never JSON), so
    "parseable" here means a non-empty string.
    """
    if not isinstance(row, dict):
        return False
    for key in ("answerable_question", "unanswerable_question", "ground_truth"):
        value = row.get(key)
        if not isinstance(value, str) or not value.strip():
            return False
    return True


def _pair_key(row: dict) -> tuple[str, str]:
    """The dedup key: the normalised question pair (never the question alone).

    144 answerable questions carry more than one ``ground_truth`` in this file, so
    a key built from one member would silently graft another row's answer onto
    this row's question.
    """
    return (
        normalise_question(row["answerable_question"]),
        normalise_question(row["unanswerable_question"]),
    )


def _dedup(rows: list[dict]) -> tuple[list[dict], int, int]:
    """Drop duplicate and contradictory pairs; return ``(kept, conflicting, duplicates)``.

    Groups with conflicting golds are contradictory input (rows 436/2528 are the
    same text pair with golds ``2`` and ``3``): every member is dropped, fail
    closed.  Groups of same-gold duplicates are the same row twice: keep the lowest
    index.  Golds are compared stripped -- the *stored* gold stays verbatim, but
    ``"2"`` and ``" 2"`` are not two different answers.
    """
    groups: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for row in rows:
        groups[_pair_key(row)].append(row)
    conflicting: set[int] = set()
    duplicate_of: set[int] = set()
    for members in groups.values():
        if len(members) < 2:
            continue
        golds = {member["ground_truth"].strip() for member in members}
        if len(golds) > 1:
            conflicting.update(member["_index"] for member in members)
            continue
        ordered = sorted(members, key=lambda member: member["_index"])
        duplicate_of.update(member["_index"] for member in ordered[1:])
    kept = [row for row in rows if row["_index"] not in conflicting and row["_index"] not in duplicate_of]
    return kept, len(conflicting), len(duplicate_of)


# ---------------------------------------------------------------------------
# A/B randomisation and the deterministic quota selection
# ---------------------------------------------------------------------------


def _answerable_label(index: int, seed: int) -> str:
    """Which label the answerable question lands on: a per-row seeded 50:50 draw.

    Section 4.5: the order must be random per row and reproducible; the seed is the
    row's own source index, so the draw is independent of how many rows the build
    keeps and of every other row.
    """
    return "A" if random.Random(f"{seed}:pair:{index}").random() < 0.5 else "B"


def _select_pairs(pairs: list[dict], cap: int, seed: int) -> list[dict]:
    """Pick ``cap`` of ``pairs`` by a deterministic, file-wide shuffle.

    Not "the first N": the raw file is ordered by generation batch, so a prefix
    would confine the artifact to one corner of the source.  The shuffle is seeded
    by ``seed`` alone (not by ``cap``), so a build with a smaller cap is a subset
    of a build with a larger one, and the returned rows are sorted back into source
    order so the artifact is stable and readable.
    """
    ordered = sorted(pairs, key=lambda row: row["_index"])
    if cap <= 0:
        return []
    if len(ordered) <= cap:
        return ordered
    shuffled = list(ordered)
    random.Random(f"{seed}:order").shuffle(shuffled)
    return sorted(shuffled[:cap], key=lambda row: row["_index"])


def _task_id(index: int) -> str:
    """Stable hard-replay key: one pair row per source position."""
    return f"sum-pair-{index}"


# ---------------------------------------------------------------------------
# row construction
# ---------------------------------------------------------------------------


def _pair_row(row: dict, seed: int) -> dict:
    """One SUM pair row: both questions in template C, gold = the pair contract."""
    index = row["_index"]
    answerable = normalise_question(row["answerable_question"])
    unanswerable = normalise_question(row["unanswerable_question"])
    answerable_id = _answerable_label(index, seed)
    question_a, question_b = (answerable, unanswerable) if answerable_id == "A" else (unanswerable, answerable)
    ground_truth = schema.build_ground_truth(
        solvable=True,
        answer=row["ground_truth"],  # the raw column, verbatim (section 4.5)
        has_diagnosis_label=False,
        perturbation_type=None,
        pair_task=True,
        answerable_id=answerable_id,
    )
    extra_info = {
        "split": "train",  # the source ships one file and no split column
        "index": index,
        "task_id": _task_id(index),
        "seed": seed,
        "difficulty": "",  # SUM has no difficulty axis at all
        "solvable": True,
        "has_diagnosis_label": False,
        "error_type": MIXED_ERROR_TYPE,
        # The pair's other member, so the audit can check both prompt questions
        # against the raw row without trusting the rendered text.
        "paired_original_text": unanswerable,
    }
    return schema.make_row(
        data_source=DATA_SOURCE,
        question=question_a,
        question_b=question_b,
        ground_truth=ground_truth,
        template=schema.TEMPLATE_C,
        branch=BRANCH_PAIR,
        extra_info=extra_info,
    )


def build_rows(raw_dir: str, limit: int | None = None, seed: int = 0) -> tuple[list[dict], dict]:
    """Build the SUM parquet rows and the funnel that produced them.

    Returns ``(rows, funnel)``: ``rows`` are ready for
    :func:`schema.normalise_extra_info` / :func:`schema.validate_rows` /
    :func:`schema.write_rows_parquet`, and ``funnel`` is an ordered mapping of
    stage -> count, every stage counting **pairs** (one pair, one row).  The nested
    ``duplicate_drops`` mapping says which clause rejected each dropped row, so no
    loss is anonymous.

    ``limit`` overrides :data:`PAIR_QUOTA` (a scratch/fixture build); ``None`` means
    the 6,000 of section 4.8.  The same ``(raw_dir, limit, seed)`` always produces
    byte-identical rows: the A/B draw is seeded per row and the selection is a
    seeded shuffle.
    """
    raw = load_source(raw_dir)
    funnel: dict = collections.OrderedDict()
    funnel["raw_rows"] = len(raw)

    stage = [row for row in raw if _is_well_formed(row)]
    funnel["after_malformed_drop"] = len(stage)

    stage, conflicting, duplicates = _dedup(stage)
    funnel["after_duplicate_pair_drop"] = len(stage)

    cap = PAIR_QUOTA if limit is None else max(limit, 0)
    selected = _select_pairs(stage, cap, seed)
    funnel["after_quota_cap"] = len(selected)
    funnel["duplicate_drops"] = collections.OrderedDict(conflicting_gold=conflicting, same_gold_duplicate=duplicates)

    return [_pair_row(row, seed) for row in selected], funnel


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def _breakdown(rows: list[dict], key) -> dict:
    counter: dict = collections.Counter(key(row) for row in rows)
    return dict(sorted(counter.items(), key=lambda item: str(item[0])))


def test_split_note(raw_dir: str) -> str:
    """One line about ``test.parquet`` for the report (it supplies no pair)."""
    path = os.path.join(raw_dir, TEST_FILE)
    if not os.path.exists(path):
        return f"{TEST_FILE}: absent"
    rows = schema.read_parquet_rows(path)
    columns = sorted(rows[0]) if rows else []
    golds = {row.get("ground_truth", "") for row in rows}
    return (
        f"{TEST_FILE}: {len(rows)} rows, columns={columns}, "
        f"distinct ground_truth={len(golds)} -- no answerable_question, so it pairs nothing"
    )


def _gold_kind(row: dict) -> str:
    answer = json.loads(row["reward_model"]["ground_truth"]).get("answer")
    return "numeric" if looks_numeric(str(answer)) else "expression"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the SUM hallucination-domain pair rows.")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=f"pairs to emit; default {PAIR_QUOTA} (design doc section 4.8 table B row 6). "
        "An explicit value overrides the quota, for scratch/fixture builds.",
    )
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--report", default=None, help="JSON report path (default: alongside --out)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report_path = args.report or report_path_for(args.out)

    rows, funnel = build_rows(args.raw_dir, limit=args.limit, seed=args.seed)
    schema.normalise_extra_info(rows)
    schema.validate_rows(rows)
    schema.write_rows_parquet(rows, args.out)

    branches = _breakdown(rows, lambda row: row["extra_info"]["branch"])
    templates = _breakdown(rows, lambda row: row["extra_info"]["template"])
    answerable_ids = _breakdown(
        [json.loads(row["reward_model"]["ground_truth"]) for row in rows],
        lambda payload: payload["answerable_id"],
    )
    gold_kinds = _breakdown(rows, _gold_kind)

    print(f"raw dir : {args.raw_dir}")
    print(f"wrote   : {args.out}  ({len(rows)} pair rows, seed={args.seed})")
    print(f"---- {test_split_note(args.raw_dir)}")
    print("\nfunnel (pairs remaining after each stage, and what that stage cost):")
    previous = None
    for stage, count in funnel.items():
        if not isinstance(count, int):  # duplicate_drops is a nested mapping
            continue
        cost = "" if previous is None else f"   {previous - count:+d}"
        print(f"  {stage:30s} {count:6d}{cost}")
        previous = count
    drops = funnel["duplicate_drops"]
    print("\nrows dropped by the clause that rejected them (fail closed):")
    for reason, count in drops.items():
        print(f"  {reason:38s} {count:6d}")
    print("\nper branch:")
    for branch, count in branches.items():
        print(f"  {branch:22s} {count}")
    print("\nper template:")
    for template, count in templates.items():
        print(f"  {template:22s} {count}")
    print("\nper answerable_id (the section 4.5 50:50 draw):")
    for label, count in answerable_ids.items():
        share = count / len(rows) if rows else 0.0
        print(f"  {label:22s} {count:6d}  {share:.1%}")
    print("\nper gold kind:")
    for kind, count in gold_kinds.items():
        print(f"  {kind:22s} {count}")

    report = {
        "raw_dir": args.raw_dir,
        "out": args.out,
        "rows": len(rows),
        "quota": PAIR_QUOTA,
        "seed": args.seed,
        "limit": args.limit,
        "funnel": funnel,
        "per_branch": branches,
        "per_template": templates,
        "per_answerable_id": answerable_ids,
        "per_gold_kind": gold_kinds,
        "test_split": test_split_note(args.raw_dir),
    }
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"\nreport  : {report_path}")


if __name__ == "__main__":  # pragma: no cover - exercised through main() in tests
    main()
