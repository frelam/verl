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
"""Unified schema, prompt templates and parquet I/O for the hallucination domain.

This module is the single contract every ``*_adapter.py`` writes against and the
reward file reads (HALLUCINATION_RL_DESIGN.md sections 3 and 5).  Nothing here
downloads or transforms source data -- it only fixes *shapes*: the verl parquet
row, the ``ground_truth`` JSON payload, the ``extra_info`` key set, and the three
prompt templates.

Why ``ground_truth`` is a JSON string
-------------------------------------

The parquet column stores a JSON **string**, not a struct, so adding a field
never requires a schema migration and every row of a mixed parquet keeps one
Arrow schema (design doc section 3).  ``parse_ground_truth`` in the reward file
reads it back; a malformed payload scores 0 and logs.

Branch registry
---------------

Table B of design doc section 4.9.3 fixes six contract branches.  The reward does
not need them (it routes off the ground-truth fields), but the mix does: quotas,
the two hard balance constraints (template A and template B must each contain
solvable *and* unsolvable rows) and the per-branch monitoring of section 10 are
all expressed in branch terms.

====================  ==========================================  =============
branch                example source                              template
====================  ==========================================  =============
``solvable_numeric``  GSM-IC, synthesised distractors               B
``solvable_roles``    K&K                                           B
``solvable_judge``    SUM/UMWP answerable, CREPE-normal, KUQ-known  A or B_judge
``unsolvable_diag``   SUM/UMWP visible defect, FalseQA-fake         A
``unsolvable_bare``   MiP, SUM-deletion, TreeCut, UMWP cat1,        B or B_judge
                      CREPE-false-presupposition, KUQ-unknown
====================  ==========================================  =============
"""

from __future__ import annotations

import json
import os
import random
import re
from typing import Any

# ---------------------------------------------------------------------------
# data sources (the reward's routing key)
# ---------------------------------------------------------------------------

SOURCE_KK = "halluc_logic_kk"
SOURCE_GSMIC = "halluc_math_gsmic"
SOURCE_SUM = "halluc_math_sum"
SOURCE_UMWP = "halluc_math_umwp"
SOURCE_MIP = "halluc_math_mip"
SOURCE_TREECUT = "halluc_math_treecut"
# The stage-1 math pool, used only as a question source for the fourth
# synthesised-distractor slice of table B row 1 (400 rows).  Rows carrying this
# data_source are produced by distractor_synth.py, not by a source adapter.
SOURCE_MAIN = "halluc_math_main"
SOURCE_FALSEQA = "halluc_commonsense_falseqa"
SOURCE_CREPE = "halluc_commonsense_crepe"
SOURCE_KUQ = "halluc_commonsense_kuq"

# data_source -> (ability, human-readable source label for extra_info.source)
SOURCES: dict[str, tuple[str, str]] = {
    SOURCE_KK: ("logic", "K-and-K"),
    SOURCE_GSMIC: ("math", "GSM-IC"),
    SOURCE_SUM: ("math", "SUM"),
    SOURCE_UMWP: ("math", "UMWP"),
    SOURCE_MIP: ("math", "MiP"),
    SOURCE_TREECUT: ("math", "TreeCut"),
    SOURCE_MAIN: ("math", "stage-1 math pool"),
    SOURCE_FALSEQA: ("commonsense", "FalseQA"),
    SOURCE_CREPE: ("commonsense", "CREPE"),
    SOURCE_KUQ: ("commonsense", "KUQ"),
}

# The six contract branches of design doc section 4.9.3 table B.
BRANCH_SOLVABLE_NUMERIC = "solvable_numeric"
BRANCH_SOLVABLE_ROLES = "solvable_roles"
BRANCH_SOLVABLE_JUDGE = "solvable_judge"
BRANCH_UNSOLVABLE_DIAG = "unsolvable_diag"
BRANCH_UNSOLVABLE_BARE = "unsolvable_bare"

BRANCHES = (
    BRANCH_SOLVABLE_NUMERIC,
    BRANCH_SOLVABLE_ROLES,
    BRANCH_SOLVABLE_JUDGE,
    BRANCH_UNSOLVABLE_DIAG,
    BRANCH_UNSOLVABLE_BARE,
)

# perturbation_type vocabulary (design doc section 3).
PERTURBATION_TYPES = (
    "missing_condition",
    "contradictory_condition",
    "distracting_condition",
    "ambiguous_condition",
    "unrealistic_condition",
    "unrelated_entity",
    "question_missing",
)

# ---------------------------------------------------------------------------
# extra_info: the canonical key set
# ---------------------------------------------------------------------------

# The first block is stage 1's key set (scripts/to_parquet_*.py's
# make_extra_info plus mix.py's pass_rate).  It must stay here verbatim: the
# stage-2 parquet mixes old-domain rows with hallucination rows into ONE Arrow
# struct column, so a key present on only one side makes the write fail.
# ``normalise_extra_info`` fills whichever of these a given row lacks.
STAGE1_EXTRA_INFO_KEYS = (
    "split",
    "index",
    "task_id",
    "domain",
    "source",
    "difficulty",
    "prior_solve_rate",
    "seed",
    "pass_rate",
)

# Stage-2 additions (design doc sections 3 and 4).
STAGE2_EXTRA_INFO_KEYS = (
    "branch",  # table B branch, for quota bookkeeping and per-branch monitoring
    "template",  # A / B / B_judge -- see below
    "perturbation_type",
    "perturbation_family",  # K&K: clean / perturbed_statement / ...
    "options",  # [{"id": "A", "text": ...}], empty for option-less rows
    "correct_option_id",
    "judgment_only",
    "has_diagnosis_label",
    "solvable",
    "paired_original_text",  # audit only (D4 dropped the consistency term)
    "perturbed_entity_text",  # audit only
    "deleted_condition_text",  # MiP/SUM audit only
    "distractor_text",  # GSM-IC / synthesised distractor (D17)
    "distractor_labels",  # its three GSM-IC labels
    "error_type",  # defect class used for the D18 balance table
    "canonical_solution",  # K&K: the enum solution as [bool, ...], audit only
    "role_words",  # K&K: [truth-teller word, liar word] as spelled in the prompt
)

EXTRA_INFO_KEYS = STAGE1_EXTRA_INFO_KEYS + STAGE2_EXTRA_INFO_KEYS

# Default per key, used when a row does not set one.  Types matter: they decide
# the Arrow field type of the union schema, so "" vs 0 vs None is not cosmetic.
EXTRA_INFO_DEFAULTS: dict[str, Any] = {
    "split": "train",
    "index": -1,
    "task_id": "",
    "domain": "",
    "source": "",
    "difficulty": "",
    "prior_solve_rate": -1.0,
    "seed": -1,
    "pass_rate": -1.0,
    "branch": "",
    "template": "",
    "perturbation_type": "",
    "perturbation_family": "",
    "options": [],
    "correct_option_id": "",
    "judgment_only": False,
    "has_diagnosis_label": False,
    "solvable": False,
    "paired_original_text": "",
    "perturbed_entity_text": "",
    "deleted_condition_text": "",
    "distractor_text": "",
    # Fixed-shape dict, never {} .  pyarrow refuses to write a struct with no
    # child field ("Cannot write struct type 'distractor_labels' with no child
    # field to Parquet"), and the three label axes are the D17 balance keys, so
    # every row carries them whether or not it came from GSM-IC.
    "distractor_labels": {"role_label": "", "number_label": "", "sentence_label": ""},
    "error_type": "",
    "canonical_solution": [],
    "role_words": [],
}


# ---------------------------------------------------------------------------
# templates (design doc section 5.2)
# ---------------------------------------------------------------------------

TEMPLATE_A = "A"  # options block + explicit verdict  (judgment + diagnosis)
TEMPLATE_B = "B"  # no options block, answer or refuse
TEMPLATE_B_JUDGE = "B_judge"  # no options block, verdict only

_TEMPLATE_A = """{question}

请先判断该题的前提是否成立、是否可以作答，并只在最后一行给出结论：

- 若前提成立、题目可以作答，最后一行输出：\\boxed{{SOLVABLE}}
- 若前提不成立 / 条件相互矛盾，最后一行输出：\\boxed{{UNSOLVABLE: <选项ID>}}
  其中 <选项ID> 是下面选项中"让题目前提为假"的那一项。

选项：
{options}"""

_TEMPLATE_B = """{question}

若题目给出的信息不足以确定唯一答案，或条件相互矛盾，请在最后一行输出：
\\boxed{{UNSOLVABLE}}

否则，请 step by step 推理，并在最后一行输出 \\boxed{{你的最终答案}}。"""

# The judgement branch needs its own option-less wording: template B asks for the
# *answer*, this one asks only for the verdict.  Design doc section 4.9.3 puts
# CREPE-normal and KUQ-known on template B with gold \boxed{SOLVABLE}, which
# template B cannot express -- see the DESIGN GAP note in
# ``test_schema.py::TestTemplateIsomorphism``.
_TEMPLATE_B_JUDGE = """{question}

请判断该题的前提是否成立、是否可以作答，并只在最后一行给出结论：

- 若前提成立、题目可以作答，最后一行输出：\\boxed{{SOLVABLE}}
- 若前提不成立 / 条件相互矛盾，最后一行输出：\\boxed{{UNSOLVABLE}}"""

# K&K answers with the row's *own* role words, in the order the inhabitants are
# listed.  Naming the words as knight/knave would be wrong on the 28.7% of rows
# whose prompt says angel/devil or knave/knight (design doc section 5.1).
_KK_ROLE_INSTRUCTION = (
    "\n\n注意：请用题目中出现的角色名称作答（例如题目若称其为 angel/devil 或 knave/knight，"
    "就用该题使用的这两个词），顺序与题面列出的居民一致，"
    "多个词之间用空格分隔，例如 \\boxed{<第1人> <第2人> ...}。"
)


def render_options_block(options: list[dict]) -> str:
    """Render ``[{"id": "A", "text": ...}]`` as the ``A. …`` block of template A."""
    return "\n".join(f"{opt['id']}. {opt['text']}" for opt in options)


def render_prompt(
    question: str,
    template: str,
    options: list[dict] | None = None,
    role_words: list[str] | None = None,
) -> str:
    """Build the final user prompt for one row.

    ``role_words`` only applies to K&K rows under template B; it appends the
    instruction that names the answer format without hard-coding knight/knave.
    """
    question = question.strip()
    if template == TEMPLATE_A:
        if not options:
            raise ValueError("template A requires an options block")
        return _TEMPLATE_A.format(question=question, options=render_options_block(options))
    if template == TEMPLATE_B:
        prompt = _TEMPLATE_B.format(question=question)
        if role_words:
            prompt += _KK_ROLE_INSTRUCTION
        return prompt
    if template == TEMPLATE_B_JUDGE:
        return _TEMPLATE_B_JUDGE.format(question=question)
    raise ValueError(f"unknown template {template!r}")


# ---------------------------------------------------------------------------
# ground truth
# ---------------------------------------------------------------------------


def ground_truth_payload(
    solvable: bool,
    *,
    answer: str | None = None,
    correct_option_id: str | None = None,
    has_diagnosis_label: bool = False,
    perturbation_type: str | None = None,
    role_words: list[str] | None = None,
    judgment_only: bool = False,
) -> dict:
    """Build the ``reward_model.ground_truth`` payload (design doc section 3).

    ``role_words`` is K&K-only: the row's own ``[truth-teller, liar]`` surface
    words.  The reward maps both the model's answer and the stored ``answer``
    through it before comparing, which is what stops ``flip_role`` /
    ``random_pair`` rows (28.7% of K&K) from scoring inverted.

    ``judgment_only`` is the D14 branch: the row asks "is this answerable?" and
    ``answer`` is audit-only, never compared.
    """
    payload: dict[str, Any] = {
        "solvable": bool(solvable),
        "answer": answer,
        "correct_option_id": correct_option_id,
        "has_diagnosis_label": bool(has_diagnosis_label),
        "perturbation_type": perturbation_type,
    }
    if role_words:
        payload["role_words"] = list(role_words)
    if judgment_only:
        payload["judgment_only"] = True
    return payload


def ground_truth_json(payload: dict) -> str:
    """Serialise a payload for the parquet column (stable key order)."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def build_ground_truth(**kwargs) -> str:
    """``ground_truth_payload`` + ``ground_truth_json`` in one call."""
    return ground_truth_json(ground_truth_payload(**kwargs))


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------


def make_row(
    *,
    data_source: str,
    question: str,
    ground_truth: str,
    template: str,
    branch: str,
    extra_info: dict,
    options: list[dict] | None = None,
    role_words: list[str] | None = None,
) -> dict:
    """Assemble one verl parquet row, with the prompt already rendered.

    The prompt is fixed here (design doc section 5.2: "adapter 落盘时就固定进
    prompt，reward 侧不再判模板"), so the reward never has to infer a template.

    Adapters may attach private ``_``-prefixed keys for their own pipeline
    (dedup text, source of a synthesised row, ...); :func:`validate_rows` rejects
    any that survive into the written artifact.
    """
    ability, source_label = SOURCES[data_source]
    prompt = render_prompt(question, template, options=options, role_words=role_words)
    info = dict(EXTRA_INFO_DEFAULTS)
    info.update(
        {
            "domain": ability,
            "source": source_label,
            "branch": branch,
            "template": template,
            "options": options or [],
            "role_words": list(role_words) if role_words else [],
        }
    )
    info.update(extra_info)
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": prompt}],
        "ability": ability,
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": info,
    }


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

_OPTION_ID_ALPHABET = "ABCDEFGH"


def validate_row(row: dict) -> list[str]:
    """Return the list of contract violations of one row (empty == valid)."""
    problems: list[str] = []
    task_id = row.get("extra_info", {}).get("task_id", "<missing>")

    def problem(msg: str) -> None:
        problems.append(f"[{task_id}] {msg}")

    data_source = row.get("data_source")
    if data_source not in SOURCES:
        problem(f"unknown data_source {data_source!r}")
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or not prompt or not prompt[0].get("content", "").strip():
        problem("empty prompt")
    payload = row.get("reward_model", {}).get("ground_truth")
    if not isinstance(payload, str):
        problem("ground_truth must be a JSON string")
        return problems
    try:
        gt = json.loads(payload)
    except ValueError as exc:
        problem(f"ground_truth is not valid JSON: {exc}")
        return problems
    if not isinstance(gt, dict):
        problem("ground_truth must decode to an object")
        return problems

    solvable = gt.get("solvable")
    if not isinstance(solvable, bool):
        problem("ground_truth.solvable must be a bool")

    info = row.get("extra_info", {})
    for key in EXTRA_INFO_KEYS:
        if key not in info:
            problem(f"extra_info is missing the canonical key {key!r}")
    if info.get("task_id") in (None, ""):
        problem("extra_info.task_id is empty (hard-replay dedup key)")
    if info.get("difficulty", "") != "" and not isinstance(info.get("difficulty"), str):
        problem("extra_info.difficulty must be a string (stage 1 stores a string)")

    options = info.get("options") or []
    ids = [opt.get("id") for opt in options]
    if len(set(ids)) != len(ids):
        problem(f"duplicate option ids: {ids}")
    if ids and ids != list(_OPTION_ID_ALPHABET[: len(ids)]):
        problem(f"option ids must be a prefix of A,B,C...: {ids}")

    correct = gt.get("correct_option_id")
    if solvable:
        if gt.get("judgment_only"):
            pass  # answer is audit-only here
        elif gt.get("role_words"):
            if not gt.get("answer"):
                problem("role-word row has no answer sequence")
            if len(gt.get("role_words") or []) != 2:
                problem("role_words must be [truth-teller word, liar word]")
        elif not gt.get("answer"):
            problem("solvable row has no answer")
        if gt.get("has_diagnosis_label"):
            problem("a solvable row must not carry has_diagnosis_label")
    else:
        if gt.get("answer") is not None:
            problem("unsolvable row must not carry an answer")
        if correct is not None and correct not in ids:
            # Catches the adapter's option-shuffle / index off-by-one bug that
            # design doc section 9 calls out explicitly.
            problem(f"correct_option_id {correct!r} is not among the row's options {ids}")

    if gt.get("has_diagnosis_label"):
        if correct is None:
            problem("has_diagnosis_label row needs correct_option_id")
        if not options:
            problem("has_diagnosis_label row needs an options block")
        if gt.get("perturbation_type") is None:
            problem("has_diagnosis_label row needs a perturbation_type")
    if solvable and correct is not None:
        problem("correct_option_id set on a solvable row")

    # Template invariants.  ``judgment_only`` is orthogonal to template A: A is
    # shared by the solvable judgment rows (D14, gold \boxed{SOLVABLE}) and by
    # the four-tier unsolvable rows (gold \boxed{UNSOLVABLE: <id>}).  What ties
    # them together is that only a *judgment* row can be ``judgment_only``, and
    # its prompt must therefore ask for a verdict -- A or B_judge, never plain B.
    template = info.get("template")
    judgment_only = bool(gt.get("judgment_only"))
    has_options_block = bool(options)
    if template == TEMPLATE_A and not has_options_block:
        problem("template A without an options block")
    if template in (TEMPLATE_B, TEMPLATE_B_JUDGE) and has_options_block:
        problem(f"template {template} must not carry an options block")
    if judgment_only and template == TEMPLATE_B:
        problem("judgment_only row cannot use template B (it never asks for a verdict)")
    if isinstance(solvable, bool) and judgment_only and not solvable:
        problem("judgment_only implies solvable: its gold answer is the SOLVABLE marker")
    # A solvable row whose gold is an *answer* must be asked for that answer:
    # neither A nor B_judge ever prompts for one.
    if isinstance(solvable, bool) and solvable and not judgment_only and template != TEMPLATE_B:
        problem(f"solvable non-judgment row must use template B, got {template!r}")

    if gt.get("perturbation_type") is not None and gt["perturbation_type"] not in PERTURBATION_TYPES:
        problem(f"unknown perturbation_type {gt['perturbation_type']!r}")

    for key in row:
        if key.startswith("_"):
            problem(f"private key {key!r} leaked into the artifact")
    return problems


def validate_rows(rows: list[dict], *, limit: int = 10) -> None:
    """Raise on the first batch of contract violations (fail closed)."""
    failures: list[str] = []
    for row in rows:
        failures.extend(validate_row(row))
        if len(failures) >= limit:
            break
    if failures:
        head = "\n".join(f"  - {f}" for f in failures[:limit])
        raise ValueError(f"{len(failures)}+ schema violations (showing up to {limit}):\n{head}")


# ---------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------


def build_options(
    candidates: list[str], gold_text: str, k: int, rng: random.Random
) -> tuple[list[dict], str] | None:
    """Assemble a shuffled A/B/C block, or None when the pool is too small.

    ``candidates`` are the distractor spans drawn from **the same question** and
    forced to the same token length as ``gold_text`` (design doc section 4.6:
    equal length is the anti-shortcut constraint, not an optimisation).  Having
    ``k - 1`` distinct candidates is a hard requirement -- an adapter must drop
    the row rather than fall back to cross-question text, which the L3 audit
    showed is 99-100% guessable (section 4.4).
    """
    unique: list[str] = []
    seen = {gold_text.strip().casefold()}
    for candidate in candidates:
        key = candidate.strip().casefold()
        if key and key not in seen:
            seen.add(key)
            unique.append(candidate.strip())
    if len(unique) < k - 1:
        return None
    chosen = rng.sample(unique, k - 1) + [gold_text.strip()]
    rng.shuffle(chosen)
    options = [{"id": _OPTION_ID_ALPHABET[i], "text": text} for i, text in enumerate(chosen)]
    correct = next(opt["id"] for opt in options if opt["text"] == gold_text.strip())
    return options, correct  # type: ignore[return-value]


def spans_equal_length(spans: list[str]) -> bool:
    """Whether every span has the same word count (the D15 anti-shortcut rule)."""
    lengths = {len(span.split()) for span in spans}
    return len(lengths) == 1


# ---------------------------------------------------------------------------
# parquet I/O
# ---------------------------------------------------------------------------


def normalise_extra_info(rows: list[dict]) -> None:
    """Fill every row's ``extra_info`` up to the canonical key set, in place.

    A mixed parquet is one Arrow struct column: a key that exists on only some
    rows would make ``Dataset.from_list`` fail (or, worse on an older pyarrow,
    silently drop it).  Rows coming from a stage-1 parquet carry extra keys of
    their own; those are preserved, and any key seen on *any* row is backfilled
    onto the others with a type-appropriate default.
    """
    observed: dict[str, Any] = {}
    for row in rows:
        info = row.setdefault("extra_info", {})
        for key, value in info.items():
            if value is not None and key not in observed:
                observed[key] = value
    for row in rows:
        info = row["extra_info"]
        for key, sample in observed.items():
            if key not in info or info[key] is None:
                info[key] = _default_like(sample)
        for key, default in EXTRA_INFO_DEFAULTS.items():
            info.setdefault(key, default)


def _default_like(sample: Any) -> Any:
    """A blank value of the same Arrow-compatible type as ``sample``.

    A ``dict`` recurses rather than collapsing to ``{}``: an empty struct has no
    child field and pyarrow rejects it at write time, so the blank must keep the
    sample's key set.
    """
    if isinstance(sample, bool):
        return False
    if isinstance(sample, int):
        return -1
    if isinstance(sample, float):
        return -1.0
    if isinstance(sample, list):
        return []
    if isinstance(sample, dict):
        return {key: _default_like(value) for key, value in sample.items()}
    return ""


def write_rows_parquet(rows: list[dict], path: str, chunk_size: int = 10_000) -> None:
    """Write rows as several row groups (mirrors ``mix.py``'s overflow workaround).

    ``datasets.Dataset.from_list(rows).to_parquet(path)`` builds one Arrow string
    array for the whole split, whose 32-bit offsets overflow past ~2 GB of text.
    Writing chunk by chunk with ``ParquetWriter`` keeps the offsets local.

    The writer schema is inferred from a **stratified** probe, not from chunk 1.
    A field populated by only one source (K&K's ``canonical_solution``, GSM-IC's
    ``distractor_labels``) is an empty ``[]``/``{}`` on every other row, and
    chunk 1 can easily contain only those -- pyarrow would then lock the column
    to ``list<null>`` and every later chunk carrying a real value would fail to
    cast.  One probe row per ``data_source`` makes the inferred schema cover all
    of them by construction.
    """
    import datasets
    import pyarrow.parquet as pq

    if not rows:
        raise ValueError("refusing to write an empty parquet")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    probe: list[dict] = []
    seen_sources: dict[str, int] = {}
    for row in rows:
        source = row.get("data_source", "")
        if seen_sources.get(source, 0) < 3:
            seen_sources[source] = seen_sources.get(source, 0) + 1
            probe.append(row)
    schema = datasets.Dataset.from_list(probe).data.table.schema

    writer = pq.ParquetWriter(path, schema)
    try:
        for start in range(0, len(rows), chunk_size):
            table = datasets.Dataset.from_list(rows[start : start + chunk_size]).data.table
            try:
                writer.write_table(table.cast(schema))
            except Exception as exc:  # noqa: BLE001 - re-raised with the fix
                raise ValueError(
                    f"chunk {start}:{start + chunk_size} does not match the schema inferred "
                    f"from the source probe ({exc}). A row type differs from every row of its "
                    f"data_source -- check that its extra_info went through "
                    f"normalise_extra_info and that its ground_truth is a JSON string."
                ) from exc
    finally:
        writer.close()


def read_parquet_rows(path: str) -> list[dict]:
    """Read a parquet file / glob / directory into Python dicts, batch by batch."""
    import glob

    import pyarrow.parquet as pq

    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))
    else:
        files = sorted(glob.glob(path))
    if not files:
        raise FileNotFoundError(f"no parquet files matched: {path}")
    rows: list[dict] = []
    for file in files:
        for batch in pq.ParquetFile(file).iter_batches(batch_size=8192):
            rows.extend(batch.to_pylist())
    return rows


# ---------------------------------------------------------------------------
# text helpers shared by the diff-based adapters (MiP / FalseQA / SUM / UMWP)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’\-]*")
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

# A small stop list, enough for "does the diff region carry content?" and for
# the content-word restriction on distractor spans (design doc sections 4.4/4.6).
STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those of in on at to for from by with without
    is are was were be been being am do does did doing have has had having will would shall should
    can could may might must not no nor so as it its he she they them his her their you your we our
    what which who whom whose when where why how many much more most other others some any all both
    each few own same such only also very just about into over under again further once here there
    """.split()
)


def words(text: str) -> list[str]:
    """Word tokens, keeping the original spelling (numbers included)."""
    return _WORD_RE.findall(text or "")


def is_content_word(token: str) -> bool:
    return token.casefold() not in STOPWORDS


def numbers_in(text: str) -> list[str]:
    """Numeric literals in a text, normalised (no commas)."""
    return [n.replace(",", "") for n in _NUMBER_RE.findall(text or "")]
