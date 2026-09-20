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

Table B of design doc section 4.8 fixes eight contract rows.  The reward does not
need the branch labels (it routes off the ground-truth fields), but the mix does:
quotas, the two hard balance constraints (template A and template B must each
contain solvable *and* unsolvable rows) and the per-branch monitoring of section
10 are all expressed in branch terms.

=====================  ==============================================  =========
branch                 example source                                  template
=====================  ==============================================  =========
``solvable_numeric``   GSM-IC, synthesised distractors (row 1)          B
``solvable_roles``     K&K (row 2)                                      B
``solvable_two_layer`` UMWP-answerable, FalseQA-answerable (row 3)       A
``solvable_judge``     CREPE-normal (row 4)                             B_judge
``solvable_pair``      SUM pair task (row 6)                            C
``unsolvable_diag``    FalseQA-fake, TreeCut negatives (row 7)          A
``unsolvable_bare``    UMWP-unanswerable, CREPE-FP, MiP (row 8)         B
=====================  ==============================================  =========

Row 5 of table B (TreeCut positives) shares ``solvable_numeric`` with row 1 --
both are template-A numeric rows whose only difference is the *placeholder*
option block, which the reward ignores.
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
SOURCE_SUM = "halluc_math_sumpair"
SOURCE_UMWP = "halluc_math_umwp"
SOURCE_MIP = "halluc_math_mip"
SOURCE_TREECUT = "halluc_math_treecut"
SOURCE_FALSEQA = "halluc_commonsense_falseqa"
SOURCE_CREPE = "halluc_commonsense_crepe"
# The stage-1 math pool, used only as a question source for the synthesised
# distractor rows of D17 (design doc section 4.7: UMWP-answerable 200 + K&K 100 +
# main-pool math 100).  Rows carrying this data_source are produced by
# ``distractor_synth.py``, not by a source adapter.
SOURCE_MAIN = "halluc_math_main"
# KUQ is deliberately absent: D25 removes it from the pool entirely (no
# verifiable answer, so the judgement signal has no answer-layer constraint).

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
}

# The contract branches of design doc section 4.8 table B.  Rows 1 (GSM-IC +
# synthesised distractors) and 5 (TreeCut positives) share
# ``BRANCH_SOLVABLE_NUMERIC``; the mix tells them apart by data_source.
BRANCH_SOLVABLE_NUMERIC = "solvable_numeric"
BRANCH_SOLVABLE_ROLES = "solvable_roles"
BRANCH_SOLVABLE_TWO_LAYER = "solvable_two_layer"
BRANCH_SOLVABLE_JUDGE = "solvable_judge"
BRANCH_SOLVABLE_PAIR = "solvable_pair"
BRANCH_UNSOLVABLE_DIAG = "unsolvable_diag"
BRANCH_UNSOLVABLE_BARE = "unsolvable_bare"

BRANCHES = (
    BRANCH_SOLVABLE_NUMERIC,
    BRANCH_SOLVABLE_ROLES,
    BRANCH_SOLVABLE_TWO_LAYER,
    BRANCH_SOLVABLE_JUDGE,
    BRANCH_SOLVABLE_PAIR,
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
    "pair_id",  # FalseQA (D27): stable id shared by the answerable/unanswerable twins
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
    # Only the D27 FalseQA rows fill this; the mixer uses it to keep a pair's two
    # members on the same side of the train/val boundary (design doc section 4.3).
    "pair_id": "",
}


# ---------------------------------------------------------------------------
# templates (design doc section 5.2)
# ---------------------------------------------------------------------------

TEMPLATE_A = "A"  # options block + explicit verdict  (judgment + diagnosis)
TEMPLATE_B = "B"  # no options block, answer or refuse
TEMPLATE_B_JUDGE = "B_judge"  # no options block, verdict only
TEMPLATE_C = "C"  # two questions labelled A/B, judge-then-solve (SUM, D23)

TEMPLATES = (TEMPLATE_A, TEMPLATE_B, TEMPLATE_B_JUDGE, TEMPLATE_C)

#: D15 pins k = 3 (1 correct + 2 distractors, ids A/B/C) and requires the same k
#: across the subsets of one template.  ``validate_row`` enforces it on the
#: artifact, not just inside the option builders: a block of another size means
#: an adapter bypassed ``build_options``/``build_pair_options``.
OPTION_COUNT = 3

# Template A is shared by *every* option-block row -- the four-tier unsolvable
# rows (FalseQA-fake D21, TreeCut negatives D26) and the solvable rows that
# carry a placeholder block only to satisfy the D18 isomorphism constraint
# (UMWP-answerable D24, FalseQA-answerable D27, TreeCut positives D26).  One
# wording for both sides is the point: any difference in the option-block
# sentence would itself become the label.
_TEMPLATE_A = """{question}

请判断该问题是否可解，并在最后一行给出结论：

- 若前提成立、题目可以作答，最后一行输出：\\boxed{{<答案>}}
- 若前提不成立 / 条件相互矛盾，最后一行输出：\\boxed{{UNSOLVABLE: <选项ID>}}
  其中 <选项ID> 是下面候选中，能让该问题变为可解的那一项。

选项：
{options}"""

_TEMPLATE_B = """{question}

若题目给出的信息不足以确定唯一答案，或条件相互矛盾，请在最后一行输出：
\\boxed{{UNSOLVABLE}}

否则，请 step by step 推理，并在最后一行输出 \\boxed{{你的最终答案}}。"""

# The judgement branch needs its own option-less wording: template B asks for the
# *answer*, this one asks only for the verdict.  Design doc section 5.2 puts
# CREPE-normal on template B plus one added sentence ("若该题可以作答，最后一行
# 输出 \boxed{SOLVABLE}"); that sentence replaces B's "否则…输出你的最终答案"
# line, which would contradict it, so the variant is spelled out here.  Keeping it
# a distinct constant also lets the schema reject a `judgment_only` row on plain B.
_TEMPLATE_B_JUDGE = """{question}

若题目给出的信息不足以确定唯一答案，或条件相互矛盾，请在最后一行输出：
\\boxed{{UNSOLVABLE}}

若该题可以作答，最后一行输出：\\boxed{{SOLVABLE}}"""

# SUM's pair task (D23): both questions in one prompt, A/B order randomised per
# row by the adapter, and the model must first say which one is solvable and then
# solve it.  Template C carries both sides by construction, which is why the mix
# balance check exempts it (design doc section 4.8).
_TEMPLATE_C = """下面给出两个问题，其中一个可以求解，另一个因缺少条件或条件矛盾而无法求解。

问题 A：{question_a}

问题 B：{question_b}

请先判断哪个问题可解，再解答该问题。最后一行输出：
\\boxed{{<可解问题的编号>: <最终答案>}}
例如 \\boxed{{A: 42}}。"""

# K&K answers with the row's *own* role words, one entry per inhabitant, as a
# name -> role pair (design doc D19).  Naming the words as knight/knave would be
# wrong on the 28.7% of rows whose prompt says angel/devil or knave/knight, and a
# bare role-word sequence carries no name to check the answer against
# (design doc section 5.1/5.2).
_KK_ROLE_INSTRUCTION = (
    "\n\n请逐人给出结论，最后一行输出 "
    "\\boxed{人名: 角色词, 人名: 角色词, …}（例如 \\boxed{Oliver: angel, Ethan: devil}）。"
    "人名须与题面拼写一致，角色词用题面原词（若题目称其为 saint/sinner，就用这两个词），"
    "顺序不限。"
)


def render_options_block(options: list[dict]) -> str:
    """Render ``[{"id": "A", "text": ...}]`` as the ``A. …`` block of template A."""
    return "\n".join(f"{opt['id']}. {opt['text']}" for opt in options)


def render_prompt(
    question: str,
    template: str,
    options: list[dict] | None = None,
    role_words: list[str] | None = None,
    question_b: str | None = None,
) -> str:
    """Build the final user prompt for one row.

    ``role_words`` only applies to K&K rows under template B; it appends the
    instruction that names the answer format without hard-coding knight/knave.
    ``question_b`` is the second question of a SUM pair (template C) and must be
    given exactly when the template is C.
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
    if template == TEMPLATE_C:
        if not (question_b or "").strip():
            raise ValueError("template C requires the second question")
        return _TEMPLATE_C.format(question_a=question, question_b=question_b.strip())
    raise ValueError(f"unknown template {template!r}")


# ---------------------------------------------------------------------------
# ground truth
# ---------------------------------------------------------------------------


def ground_truth_payload(
    solvable: bool,
    *,
    answer: str | dict | None = None,
    correct_option_id: str | None = None,
    has_diagnosis_label: bool = False,
    perturbation_type: str | None = None,
    role_words: list[str] | None = None,
    judgment_only: bool = False,
    two_layer: bool = False,
    solvable_answer: bool = False,
    pair_task: bool = False,
    answerable_id: str | None = None,
) -> dict:
    """Build the ``reward_model.ground_truth`` payload (design doc section 3).

    ``role_words`` is K&K-only and ``answer`` is then a ``name -> surface role
    word`` mapping (D19): the reward maps both the model's answer and the stored
    mapping back to canonical truth-teller booleans before comparing, which is
    what stops ``flip_role`` / ``random_pair`` rows (28.7% of K&K) from scoring
    inverted.

    The three D23/D24/D27 flags each switch the reward onto a two-layer branch:

    ``two_layer``        UMWP-answerable: ``\\boxed{<answer>}`` is worth 0.5 for
                         judging the question solvable and another 0.5 for the
                         answer (numeric match).
    ``solvable_answer``  FalseQA-answerable: same shape, but the answer layer is
                         free text compared with a normalised exact match.
    ``pair_task``        SUM: the gold is ``answerable_id`` + ``answer`` and the
                         model writes ``\\boxed{<A|B>: <answer>}``.

    ``judgment_only`` is the CREPE-normal branch: the row asks "is this
    answerable?" and ``answer`` is audit-only, never compared.
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
    if two_layer:
        payload["two_layer"] = True
    if solvable_answer:
        payload["solvable_answer"] = True
    if pair_task:
        payload["pair_task"] = True
        payload["answerable_id"] = answerable_id
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


#: Where a template's instruction block starts, i.e. where the question ends.
#: Templates A / B / B_judge render ``{question}\n\n<instruction>``; template C
#: puts its instructions first and has no terminator here (its whole prompt is
#: returned by :func:`question_of`).
_QUESTION_TERMINATORS = (
    "\n\n请判断该问题是否可解",
    "\n\n若题目给出的信息不足以确定唯一答案",
)


def question_of(row: dict) -> str:
    """The row's problem text, with the template's instruction block removed.

    Section 7.2's dedup compares *problems*: every row of a template shares the
    same instruction block, so keeping it would inflate the similarity of two
    unrelated rows and dilute that of a genuine near-duplicate.  The K&K role
    instruction and the SUM pair prompt are returned whole -- they are part of
    what the model is asked.
    """
    content = row["prompt"][0]["content"]
    cut = len(content)
    for marker in _QUESTION_TERMINATORS:
        position = content.find(marker)
        if position != -1:
            cut = min(cut, position)
    return content[:cut].strip()


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
    question_b: str | None = None,
) -> dict:
    """Assemble one verl parquet row, with the prompt already rendered.

    The prompt is fixed here (design doc section 5.2: "adapter 落盘时就固定进
    prompt，reward 侧不再判模板"), so the reward never has to infer a template.

    Adapters may attach private ``_``-prefixed keys for their own pipeline
    (dedup text, source of a synthesised row, ...); :func:`validate_rows` rejects
    any that survive into the written artifact.
    """
    ability, source_label = SOURCES[data_source]
    prompt = render_prompt(question, template, options=options, role_words=role_words, question_b=question_b)
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
    # The bookkeeping keys are part of the contract, not free-form metadata: the
    # quota/monitoring code re-reads them, so an unregistered value would silently
    # drop the row out of every table-B cell (section 4.8) instead of failing.
    branch = info.get("branch")
    if branch not in BRANCHES:
        problem(f"unknown branch {branch!r} (must be a section 4.8 contract branch)")
    template = info.get("template")
    if template not in TEMPLATES:
        problem(f"unknown template {template!r}")
    domain = info.get("domain")
    if domain and row.get("ability") != domain:
        # Section 3 maps `domain` onto BOTH `ability` and `extra_info.domain`.
        problem(f"ability {row.get('ability')!r} must mirror extra_info.domain {domain!r}")

    options = info.get("options") or []
    ids = [opt.get("id") for opt in options]
    if len(set(ids)) != len(ids):
        problem(f"duplicate option ids: {ids}")
    if ids and ids != list(_OPTION_ID_ALPHABET[: len(ids)]):
        problem(f"option ids must be a prefix of A,B,C...: {ids}")
    if options and len(options) != OPTION_COUNT:
        # D15: k is a coverage parameter and every subset of a template must share
        # it, so a block of another size means an adapter bypassed the builders.
        problem(f"options block has {len(options)} items; D15 pins k={OPTION_COUNT}")

    correct = gt.get("correct_option_id")
    role_words = gt.get("role_words")
    two_layer = bool(gt.get("two_layer"))
    solvable_answer = bool(gt.get("solvable_answer"))
    pair_task = bool(gt.get("pair_task"))
    if solvable:
        if gt.get("judgment_only"):
            pass  # answer is audit-only here
        elif role_words:
            # D19: a K&K answer is a name -> *surface* role word mapping, never a
            # bare role sequence (a sequence carries no name to check against).
            if not isinstance(gt.get("answer"), dict) or not gt.get("answer"):
                problem("role-word row needs an answer mapping {name: role word}")
            if len(role_words) != 2:
                problem("role_words must be [truth-teller word, liar word]")
        elif not gt.get("answer"):
            problem("solvable row has no answer")
        if pair_task and gt.get("answerable_id") not in ("A", "B"):
            problem(f"pair_task row needs answerable_id in A/B, got {gt.get('answerable_id')!r}")
        if gt.get("has_diagnosis_label"):
            problem("a solvable row must not carry has_diagnosis_label")
    else:
        if gt.get("answer") is not None:
            problem("unsolvable row must not carry an answer")
        if correct is not None and correct not in ids:
            # Catches the adapter's option-shuffle / index off-by-one bug that
            # design doc section 9 calls out explicitly.
            problem(f"correct_option_id {correct!r} is not among the row's options {ids}")
        for flag in ("two_layer", "solvable_answer", "pair_task", "judgment_only"):
            if gt.get(flag):
                problem(f"unsolvable row must not carry {flag}")

    if gt.get("has_diagnosis_label"):
        if correct is None:
            problem("has_diagnosis_label row needs correct_option_id")
        if not options:
            problem("has_diagnosis_label row needs an options block")
        if gt.get("perturbation_type") is None:
            problem("has_diagnosis_label row needs a perturbation_type")
    if solvable and correct is not None:
        problem("correct_option_id set on a solvable row")
    if sum((role_words is not None, two_layer, solvable_answer, pair_task, bool(gt.get("judgment_only")))) > 1:
        problem("a row must not combine two answer-contract flags")

    # Template invariants.  Template A is shared by the four-tier unsolvable rows
    # (gold \boxed{UNSOLVABLE: <id>}) and by the solvable rows that carry a
    # *placeholder* block to satisfy the D18 isomorphism constraint (UMWP /
    # FalseQA answerable, TreeCut positives); what ties them together is that A
    # always has an options block and that a solvable A row never carries a
    # correct option.  Template C carries both sides of SUM's pair task by
    # construction (design doc section 4.8).
    judgment_only = bool(gt.get("judgment_only"))
    has_options_block = bool(options)
    if template == TEMPLATE_A and not has_options_block:
        problem("template A without an options block")
    if template in (TEMPLATE_B, TEMPLATE_B_JUDGE, TEMPLATE_C) and has_options_block:
        problem(f"template {template} must not carry an options block")
    if judgment_only and template != TEMPLATE_B_JUDGE:
        problem(f"judgment_only row must use template {TEMPLATE_B_JUDGE}, got {template!r}")
    if isinstance(solvable, bool) and judgment_only and not solvable:
        problem("judgment_only implies solvable: its gold answer is the SOLVABLE marker")
    if pair_task != (template == TEMPLATE_C):
        problem(f"pair_task={pair_task} but template={template!r} (template C is SUM's pair task)")
    if (two_layer or solvable_answer) and template != TEMPLATE_A:
        problem(f"two-layer row must use template {TEMPLATE_A} (placeholder option block), got {template!r}")
    if (two_layer or solvable_answer) and not has_options_block:
        problem("two-layer row needs a placeholder options block (D18 isomorphism)")
    if isinstance(solvable, bool) and not solvable and not gt.get("has_diagnosis_label") and template == TEMPLATE_A:
        problem("a three-tier unsolvable row must not carry an options block (D12/D24)")
    solvable_with_answer = isinstance(solvable, bool) and solvable and not judgment_only
    if solvable_with_answer and template not in (TEMPLATE_A, TEMPLATE_B, TEMPLATE_C):
        problem(f"solvable row must use template A, B or C, got {template!r}")

    if gt.get("perturbation_type") is not None and gt["perturbation_type"] not in PERTURBATION_TYPES:
        problem(f"unknown perturbation_type {gt['perturbation_type']!r}")

    for key in row:
        if key.startswith("_"):
            problem(f"private key {key!r} leaked into the artifact")
    return problems


def _duplicate_task_ids(rows: list[dict]) -> list[str]:
    """Task ids used by more than one row, in first-seen order.

    Section 3 declares ``extra_info.task_id`` globally unique -- it is hard
    replay's dedup key -- so a collision is a build bug, not a cosmetic issue.
    """
    seen: set[str] = set()
    duplicates: list[str] = []
    for row in rows:
        task_id = row.get("extra_info", {}).get("task_id")
        if not isinstance(task_id, str) or not task_id:
            continue
        if task_id in seen and task_id not in duplicates:
            duplicates.append(task_id)
        seen.add(task_id)
    return duplicates


def validate_rows(rows: list[dict], *, limit: int = 10) -> None:
    """Raise on the first batch of contract violations (fail closed)."""
    failures: list[str] = []
    for row in rows:
        failures.extend(validate_row(row))
        if len(failures) >= limit:
            break
    duplicates = _duplicate_task_ids(rows)
    if duplicates:
        shown = ", ".join(repr(task_id) for task_id in duplicates[:limit])
        failures.append(f"duplicate extra_info.task_id values ({len(duplicates)}): {shown}")
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
    forced to the same token length as ``gold_text`` (design doc section 4.3:
    equal length is the anti-shortcut constraint, not an optimisation).  Having
    ``k - 1`` distinct candidates is a hard requirement -- an adapter must drop
    the row rather than fall back to cross-question text, which the L3 audit
    showed is 99-100% guessable (section 5.1/11).
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


#: How a replacement pair is written inside an option (D21): ``man -> women``.
PAIR_ARROW = " -> "


def pair_option_text(left: str, right: str) -> str:
    """Render one replacement-pair option (design doc section 4.3 / D21)."""
    return f"{left.strip()}{PAIR_ARROW}{right.strip()}"


def shuffle_options(texts: list[str], rng: random.Random) -> list[dict]:
    """Shuffle arbitrary option texts into an A/B/C… block with no correct item.

    Used by the *placeholder* blocks of the D18 isomorphism constraint
    (UMWP-answerable D24, FalseQA-answerable D27, TreeCut positives D26): the row
    is solvable and its gold is an answer, so no option is correct and the reward
    never reads the block.
    """
    chosen = [text.strip() for text in texts]
    rng.shuffle(chosen)
    return [{"id": _OPTION_ID_ALPHABET[i], "text": text} for i, text in enumerate(chosen)]


def build_pair_options(
    gold_left: str,
    gold_right: str,
    distractor_rights: list[str],
    k: int,
    rng: random.Random,
) -> tuple[list[dict], str] | None:
    """Assemble a shuffled replacement-*pair* block, or None when short (D21).

    Every option shares ``gold_left`` (design doc section 4.3: the left item is
    identical across the three options, so it carries no information) and differs
    only in the right item; the distractors are out-of-passage same-type
    same-length words supplied by the adapter.  Returning None rather than
    padding is deliberate: a row that cannot field ``k - 1`` distinct distractors
    is dropped (fail closed).
    """
    unique: list[str] = []
    seen = {gold_right.strip().casefold()}
    for candidate in distractor_rights:
        key = candidate.strip().casefold()
        if key and key not in seen:
            seen.add(key)
            unique.append(candidate.strip())
    if len(unique) < k - 1:
        return None
    rights = rng.sample(unique, k - 1) + [gold_right.strip()]
    rng.shuffle(rights)
    options = [
        {"id": _OPTION_ID_ALPHABET[i], "text": pair_option_text(gold_left, right)} for i, right in enumerate(rights)
    ]
    correct = next(opt["id"] for opt in options if opt["text"] == pair_option_text(gold_left, gold_right))
    return options, correct


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


def _populated(value: Any) -> bool:
    """Whether a value carries type information (``[]``/``{}``/``""``/None do not)."""
    if value is None:
        return False
    if isinstance(value, list | dict | str):
        return len(value) > 0
    return True


def probe_rows(rows: list[dict]) -> list[dict]:
    """A row subset whose Arrow union schema covers every populated field.

    Stratifying by ``data_source`` alone is not enough.  Two producers can share a
    data_source (``halluc_logic_kk`` is written by both ``kk_adapter`` and
    ``distractor_synth``), and an ``extra_info`` list that happens to be empty on
    the sampled rows infers as ``list<null>`` -- every later row carrying a real
    value then fails to cast ("Unsupported cast from string to null").  So after
    the per-source sample the probe is extended, one row at a time, with a row that
    populates a key no probe row has populated yet.  The loop terminates because
    each pass covers at least one fresh key.
    """
    probe: list[dict] = []
    seen_sources: dict[str, int] = {}
    for row in rows:
        source = row.get("data_source", "")
        if seen_sources.get(source, 0) < 3:
            seen_sources[source] = seen_sources.get(source, 0) + 1
            probe.append(row)

    def covered_keys() -> set[str]:
        return {key for row in probe for key, value in row.get("extra_info", {}).items() if _populated(value)}

    while True:
        covered = covered_keys()
        for row in rows:
            fresh = {
                key for key, value in row.get("extra_info", {}).items() if _populated(value) and key not in covered
            }
            if fresh:
                probe.append(row)
                break
        else:
            return probe


def write_rows_parquet(rows: list[dict], path: str, chunk_size: int = 10_000) -> None:
    """Write rows as several row groups (mirrors ``mix.py``'s overflow workaround).

    ``datasets.Dataset.from_list(rows).to_parquet(path)`` builds one Arrow string
    array for the whole split, whose 32-bit offsets overflow past ~2 GB of text.
    Writing chunk by chunk with ``ParquetWriter`` keeps the offsets local.

    The writer schema is inferred from :func:`probe_rows`, a stratified probe
    rather than chunk 1: a field populated by only some rows (K&K's
    ``canonical_solution``, TreeCut's ``placeholder_modes``) is an empty ``[]`` on
    every other row, and pyarrow would otherwise lock the column to ``list<null>``
    and fail to cast every later chunk carrying a real value.
    """
    import datasets
    import pyarrow.parquet as pq

    if not rows:
        raise ValueError("refusing to write an empty parquet")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    probe = probe_rows(rows)
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
