# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Stage-4 hard-math adapter (HALLUCINATION_RL_DESIGN.md D34).

Stage 4 is a new round whose job is to *fill in* and *raise* capabilities, and
the benchmark report shows difficult math is the weakest one
(``BBH_Benchmark_评测分析报告.md``).  Stage 1's math replay was dominated by
Big-Math/DAPO, i.e. mostly below AIME difficulty; this adapter builds the
**hard-math additive domain** that is mixed next to the (now smaller) old-domain
replay:

* ``nvidia/AceReason-Math`` -- RLVR-curated, roughly AMC..AIME;
* ``agentica-org/DeepScaleR-Preview-Dataset`` -- AIME 1984-2023 + AMC +
  Omni-MATH + Still, competition level;
* ``zwhe99/DeepMath-103K`` -- levels 5-9, the subset ``--deepmath-min-difficulty``
  keeps (default 6) is close to the second half of an AIME paper.

``BytedTsinghua-SIA/DAPO-Math-17k`` is supported as a fourth, *optional* source
(``--sources ...,dapo``) but is off by default: it already forms the old-domain
math pool (``scripts/to_parquet_math.py``), so feeding it again would double-count
it unless the old-pool exclusion removed it.

The pipeline, in order (every stage is counted in the funnel report):

1. **load** -- a local file under ``--raw-dir`` (``<source>.parquet|json|jsonl``)
   wins; otherwise ``datasets.load_dataset`` pulls the HF repo.  Loading is
   idempotent through the HF cache, so a re-run does not re-download.
2. **normalise** -- prompt = problem + the stage-1 MATH instruction; answer =
   the source's answer field with ``$...$`` / ``\\boxed{...}`` wrappers removed
   (the ground truth is *not* rewritten beyond that, so the reward's own
   matchers see the source's answer).
3. **answer-format filter** -- keep only answers that are a plain integer, a
   simple fraction, or a short decimal (optionally ``X\\%``).  The user-facing
   rule from the stage-4 data plan: "优先保留整数或简单分数答案，复杂符号答案
   容易让规则匹配误判".  ``--keep-symbolic`` disables this filter for experiments
   that add a judge fallback to the reward.  Problems that depend on an image
   (``<image>`` / Asymptote) are always dropped -- the policy cannot see them.
4. **decontamination** -- AIME 2024/2025 (and any other held-out set passed
   through ``--decontaminate-path``) is compared by word n-gram overlap plus an
   optional MinHash pass.  The build **fails closed** when no decontamination
   file is given: a contaminated AIME 2024/2025 result is worse than no data.
5. **dedup** -- exact then MinHash-LSH *inside* the merged pool (the three
   sources overlap heavily), an optional **embedding-level** pass for true
   paraphrases (``--embedding-base-url`` pointing at a served
   ``/v1/embeddings``; random-hyperplane LSH keeps it ~O(N)), then an optional
   MinHash pass against ``--anchor-path`` (the raw old-domain math pool) so a
   hard-math row does not re-train a problem the old domain already covers.
   Previously trained artifacts (``--exclude-path``) are removed by exact
   prompt key.
6. **candidate quota** -- draw at most ``--candidate-quota`` rows, apportioned
   over the sources by ``--weights``, *before* the expensive difficulty filter.
7. **difficulty filter** -- sample ``--difficulty-k`` (8-16) completions per
   problem from the stage-4 starting checkpoint served through vLLM's
   OpenAI-compatible API, score them with the training reward, and keep only
   rows whose pass rate lies in ``[--min-pass, --max-pass]`` (0.1-0.9 by
   default).  All-wrong rows are dropped as well: for AIME-level problems the
   4B model starts with a very high all-wrong rate, and those rows carry no
   gradient.  The measured rate is written to ``extra_info.pass_rate``, which is
   the same field ``scripts/difficulty_tag.py`` uses.

Usage::

    # 1. serve the stage-4 starting checkpoint (reasoning parser REQUIRED)
    vllm serve <stage3_or_dpo_ckpt> --served-model-name stage4_start \\
        --reasoning-parser qwen3 --max-model-len 32768

    # 2. build (decontamination is mandatory unless explicitly waived)
    python hard_math_adapter.py \\
        --out ~/data/reasoning_rl/halluc/built_stage4/hard_math.parquet \\
        --decontaminate-path ~/data/reasoning_rl/val/aime_2024_2025.parquet \\
        --exclude-path ~/data/reasoning_rl/final \\
        --anchor-path ~/data/reasoning_rl/math/train_math.parquet \\
        --embedding-base-url http://127.0.0.1:8001/v1 \\
        --difficulty-base-url http://127.0.0.1:8000/v1 \\
        --difficulty-model stage4_start

The run is resumable: the pass-rate cache (``<out>_difficulty.jsonl`` plus
``<out>_difficulty.done``) is keyed by ``extra_info.task_id``, so a crash only
loses the in-flight batch.  ``--num-shards/--shard-index`` split the generation
across processes; every filter pass merges all shard caches next to ``--out``,
so after the shards finish run the same command once more with
``--num-shards 1`` to assemble the complete artifact from the caches (no
generation is repeated).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dedup import exact_dedup, minhash_near_dedup, normalize_text  # noqa: E402

import schema  # noqa: E402

# The HTTP helpers and the resume/shard conventions are shared with the
# reasoning-efficiency DPO pipeline; importing them keeps the two collectors from
# drifting (see scripts/distill/common.py).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "distill"))
from common import (  # noqa: E402
    append_done,
    append_jsonl,
    chat_completion,
    read_done,
    read_jsonl,
    resolve_model_id,
    response_from_choice,
    stable_shard,
)

#: Same instruction stage 1 appends to every math problem
#: (scripts/to_parquet_math.py), so the hard-math rows and the old-domain math
#: rows present the identical answer contract to the policy.
MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."

DEFAULT_RAW_DIR = os.path.expanduser("~/data/reasoning_rl/halluc/raw/hard_math")
DEFAULT_OUT = os.path.expanduser("~/data/reasoning_rl/halluc/built/hard_math.parquet")
REPORT_SUFFIX = "_report.json"


class SourceSpec:
    """One hard-math upstream dataset and its column mapping."""

    def __init__(
        self,
        key: str,
        data_source: str,
        repo: str,
        label: str,
        problem_fields: tuple[str, ...],
        answer_fields: tuple[str, ...],
        difficulty_field: str | None = None,
        verl_native: bool = False,
    ):
        self.key = key
        self.data_source = data_source
        self.repo = repo
        self.label = label
        self.problem_fields = problem_fields
        self.answer_fields = answer_fields
        self.difficulty_field = difficulty_field
        #: DAPO ships in the verl schema (prompt + reward_model) instead of
        #: problem/answer columns.
        self.verl_native = verl_native


SOURCES: dict[str, SourceSpec] = {
    "ace_reason": SourceSpec(
        "ace_reason",
        "math_hard_ace_reason",
        "nvidia/AceReason-Math",
        "AceReason-Math",
        ("problem", "question"),
        ("answer", "final_answer"),
    ),
    "deepscaler": SourceSpec(
        "deepscaler",
        "math_hard_deepscaler",
        "agentica-org/DeepScaleR-Preview-Dataset",
        "DeepScaleR-Preview",
        ("problem", "question"),
        ("answer", "final_answer"),
    ),
    "deepmath": SourceSpec(
        "deepmath",
        "math_hard_deepmath",
        "zwhe99/DeepMath-103K",
        "DeepMath-103K",
        ("question", "problem"),
        ("final_answer", "answer"),
        difficulty_field="difficulty",
    ),
    "dapo": SourceSpec(
        "dapo",
        "math_hard_dapo",
        "BytedTsinghua-SIA/DAPO-Math-17k",
        "DAPO-Math-17k",
        ("prompt",),
        ("reward_model",),
        verl_native=True,
    ),
}

#: Default per-source share of the hard-math domain.  DAPO keeps a zero weight
#: on purpose (it is the old-domain math pool); enable it explicitly with
#: ``--weights dapo=0.2`` after the old-pool exclusion has proven it is fresh.
DEFAULT_WEIGHTS: dict[str, float] = {
    "ace_reason": 0.40,
    "deepscaler": 0.35,
    "deepmath": 0.25,
    "dapo": 0.0,
}

WEIGHT_ALIASES: dict[str, str] = {
    "ace": "ace_reason",
    "ace-reason": "ace_reason",
    "acereason": "ace_reason",
    "deepscale": "deepscaler",
    "deepscaler-preview": "deepscaler",
    "deepmath-103k": "deepmath",
    **{key: key for key in SOURCES},
}


# ---------------------------------------------------------------------------
# answer normalisation / the rule-friendly filter
# ---------------------------------------------------------------------------

_BOXED_PREFIX = "\\boxed"
_FRAC_SHORT_RE = re.compile(r"\\[dt]?frac\s*(\d)\s*(\d)")
_FRAC_RE = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_PERCENT_RE = re.compile(r"(?:\\%|%)\s*$")
_GROUPED_INT_RE = re.compile(r"^[+-]?\d{1,3}(?:,\d{3})+$")
_SIMPLE_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
_TEXT_WRAP_RE = re.compile(r"\\(?:text|mathrm|textrm)\s*\{([^{}]*)\}")
# A problem that depends on a picture cannot be answered by a text-only policy.
_IMAGE_MARKERS = ("<image>", "[asy]", "\\begin{asy}", "\\includegraphics")


def unwrap_boxed(text: str) -> str:
    """``\\boxed{...}`` -> the inner text (brace-balanced, one layer)."""
    s = str(text or "").strip()
    if not s.startswith(_BOXED_PREFIX):
        return s
    rest = s[len(_BOXED_PREFIX) :].lstrip()
    if not rest.startswith("{"):
        return s
    depth = 0
    for index, char in enumerate(rest):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                if index == len(rest) - 1:
                    return rest[1:index].strip()
                break
    return s


def normalise_answer(raw) -> str:
    """Strip presentation wrappers, keep the source's own answer value.

    ``$\\boxed{28}$`` -> ``28``; ``$-4$`` -> ``-4``; ``42.86\\%`` stays as-is.
    The value itself is never rewritten: re-formatting a ground truth is how
    false negatives are introduced, and the reward already normalises both sides.
    """
    s = str(raw if raw is not None else "").strip()
    if s.startswith("$") and s.endswith("$") and len(s) > 1:
        s = s[1:-1].strip()
    s = unwrap_boxed(s)
    if s.startswith("$") and s.endswith("$") and len(s) > 1:
        s = s[1:-1].strip()
    return s


def _fraction_of(token: str) -> Fraction | None:
    """One unsigned decimal/integer token -> exact Fraction."""
    token = token.strip()
    if not _SIMPLE_NUMBER_RE.match(token):
        return None
    try:
        if "." in token:
            return Fraction(Decimal(token))
        return Fraction(token)
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def numeric_literal(text) -> tuple[Fraction, str] | None:
    """Canonical ``(value, unit)`` of a plain numeric answer, or None.

    Accepted shapes are exactly the ones the stage-4 data plan calls "clean":
    integers (with optional thousands separators), decimals, simple fractions
    (``a/b`` or ``\\frac{a}{b}``) and the same with a trailing percent sign.
    ``unit`` is ``""`` or ``"%"`` so ``42`` and ``42\\%`` never compare equal.

    Rejecting everything else is deliberate -- roots, ``\\pi``, intervals,
    tuples and symbolic expressions are the shapes where a rule matcher is most
    likely to zero a correct answer, and the build drops them unless
    ``--keep-symbolic`` is passed.
    """
    s = unwrap_boxed(str(text if text is not None else "")).strip()
    if not s:
        return None
    s = s.replace("$", "")
    for token in ("\\left", "\\right", "\\,", "\\!", "\\;", "\\ ", "\\quad", "\\qquad"):
        s = s.replace(token, "")
    s = _TEXT_WRAP_RE.sub(r"\1", s).strip()
    if not s:
        return None

    unit = ""
    if _PERCENT_RE.search(s):
        unit = "%"
        s = _PERCENT_RE.sub("", s).strip()
    if not s:
        return None

    # At most a couple of nesting-free fraction rewrites; a nested \\frac is not
    # a "simple fraction" and falls through to the reject path below.
    for _ in range(3):
        rewritten = _FRAC_SHORT_RE.sub(r"\1/\2", s)
        rewritten = _FRAC_RE.sub(r"\1/\2", rewritten)
        rewritten = rewritten.replace(" ", "")
        if rewritten == s:
            break
        s = rewritten
    s = s.replace(" ", "")

    sign = 1
    if s.startswith("+"):
        s = s[1:]
    elif s.startswith("-"):
        sign = -1
        s = s[1:]
    if not s:
        return None

    if _GROUPED_INT_RE.match(s):
        s = s.replace(",", "")
    if s.count("/") > 1:
        return None
    if "/" in s:
        numerator, _, denominator = s.partition("/")
        if not numerator or not denominator:
            return None
        top, bottom = _fraction_of(numerator), _fraction_of(denominator)
        if top is None or bottom is None or bottom == 0:
            return None
        value = top / bottom
    else:
        value = _fraction_of(s)
        if value is None:
            return None
    return sign * value, unit


def is_rule_friendly_answer(answer: str) -> bool:
    """Whether the answer is an integer / simple fraction / short decimal."""
    return numeric_literal(answer) is not None


# ---------------------------------------------------------------------------
# raw loading
# ---------------------------------------------------------------------------

_LOCAL_SUFFIXES = (".parquet", ".json", ".jsonl")


def _local_raw_file(raw_dir: str, source: str) -> str | None:
    for suffix in _LOCAL_SUFFIXES:
        candidate = os.path.join(raw_dir, f"{source}{suffix}")
        if os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
            return candidate
    return None


def _load_local(path: str):
    import datasets

    if path.endswith(".parquet"):
        return datasets.load_dataset("parquet", data_files=path, split="train")
    return datasets.load_dataset("json", data_files=path, split="train")


def _first_field(example: dict, fields: tuple[str, ...]) -> str:
    for field in fields:
        value = example.get(field)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def load_raw_rows(source: SourceSpec, raw_dir: str) -> list[dict]:
    """Raw rows of one source (local file first, HF repo otherwise)."""
    local = _local_raw_file(raw_dir, source.key)
    dataset = _load_local(local) if local else None
    if dataset is None:
        import datasets

        dataset = datasets.load_dataset(source.repo, split="train")
    return list(dataset)


def _problem_and_answer(source: SourceSpec, example: dict) -> tuple[str, str, str]:
    """``(problem, answer, difficulty)`` extracted from one raw row."""
    if source.verl_native:
        prompt = example.get("prompt")
        if isinstance(prompt, list) and prompt:
            problem = str(prompt[0].get("content", "")).strip()
        else:
            problem = str(prompt or "").strip()
        problem = problem.replace(MATH_INSTRUCTION, "").strip()
        reward_model = example.get("reward_model") or {}
        answer = str(reward_model.get("ground_truth", "")).strip()
    else:
        problem = _first_field(example, source.problem_fields)
        answer = _first_field(example, source.answer_fields)
    difficulty = ""
    if source.difficulty_field:
        raw_level = example.get(source.difficulty_field)
        if raw_level is not None:
            difficulty = str(raw_level).strip()
    return problem, answer, difficulty


def _append_instruction(problem: str) -> str:
    if "\\boxed{}" in problem:
        return problem
    return f"{problem} {MATH_INSTRUCTION}".strip()


def _usable_problem(problem: str) -> bool:
    if len(problem) < 20:
        return False
    lower = problem.lower()
    return not any(marker in lower for marker in _IMAGE_MARKERS)


# ---------------------------------------------------------------------------
# decontamination / dedup
# ---------------------------------------------------------------------------

_VAL_TEXT_FIELDS = ("problem", "question", "Problem", "Question", "content", "query", "prompt")


def _row_text(row: dict) -> str:
    """Problem text of a decontamination/old-domain row (verl rows included)."""
    prompt = row.get("prompt")
    if isinstance(prompt, list) and prompt:
        return str(prompt[0].get("content", "")).replace(MATH_INSTRUCTION, "").strip()
    if isinstance(prompt, str) and prompt.strip():
        return prompt.replace(MATH_INSTRUCTION, "").strip()
    return _first_field(row, _VAL_TEXT_FIELDS).replace(MATH_INSTRUCTION, "").strip()


def _load_text_file(path: str) -> list[str]:
    import datasets

    path = os.path.expanduser(path)
    if path.endswith(".parquet"):
        dataset = datasets.load_dataset("parquet", data_files=path, split="train")
    elif path.endswith(".jsonl") or path.endswith(".json"):
        dataset = datasets.load_dataset("json", data_files=path, split="train")
    else:
        raise ValueError(f"unsupported decontamination file type: {path}")
    rows = list(dataset)
    texts = [_row_text(row) for row in rows]
    return [text for text in texts if text]


def load_decontamination_texts(paths: list[str] | None) -> tuple[list[str], dict]:
    """Held-out problem texts (AIME 2024/2025 by convention)."""
    texts: list[str] = []
    report: dict[str, int] = {}
    for path in paths or []:
        loaded = _load_text_file(path)
        texts.extend(loaded)
        report[os.path.expanduser(path)] = len(loaded)
        print(f"[hard_math] decontamination source {path}: {len(loaded)} problem(s)")
    return texts, report


def word_ngrams(text: str, n: int) -> set[str]:
    words = normalize_text(text).split()
    if not words:
        return set()
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def decontaminate(
    rows: list[dict],
    held_out_texts: list[str],
    *,
    ngram_n: int,
    minhash_threshold: float,
) -> tuple[list[dict], dict]:
    """Drop rows that overlap a held-out problem (n-gram first, MinHash next)."""
    report = {"ngram_dropped": 0, "minhash_dropped": 0, "held_out": len(held_out_texts)}
    if not held_out_texts or not rows:
        return rows, report

    held_out_ngrams: set[str] = set()
    for text in held_out_texts:
        held_out_ngrams.update(word_ngrams(text, ngram_n))

    kept: list[dict] = []
    for row in rows:
        if word_ngrams(row["_problem"], ngram_n) & held_out_ngrams:
            report["ngram_dropped"] += 1
            continue
        kept.append(row)

    if minhash_threshold > 0 and kept:
        texts = [normalize_text(text) for text in held_out_texts] + [
            normalize_text(row["_problem"]) for row in kept
        ]
        kept_indices = set(
            minhash_near_dedup(texts, threshold=minhash_threshold, n_anchors=len(held_out_texts))
        )
        offset = len(held_out_texts)
        survivors = [row for index, row in enumerate(kept) if offset + index in kept_indices]
        report["minhash_dropped"] = len(kept) - len(survivors)
        kept = survivors
    return kept, report


def dedup_pool(rows: list[dict], threshold: float) -> tuple[list[dict], dict]:
    """Exact then MinHash dedup *inside* the merged hard-math pool."""
    report = {"exact_dropped": 0, "near_dropped": 0}
    if not rows:
        return rows, report
    keys = [row["_problem"] for row in rows]
    keep = exact_dedup(keys)
    report["exact_dropped"] = len(rows) - len(keep)
    rows = [rows[index] for index in keep]
    if threshold > 0 and rows:
        texts = [normalize_text(row["_problem"]) for row in rows]
        keep = minhash_near_dedup(texts, threshold=threshold)
        report["near_dropped"] = len(rows) - len(keep)
        rows = [rows[index] for index in keep]
    return rows, report


#: Default embedding model for the semantic dedup layer.  It is the model the
#: repository's decontamination pipeline already assumes (`decontaminate.py`),
#: so a build box usually has it served already.
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"


def load_embedding_encoder(base_url: str, model: str, batch_size: int):
    """``texts -> (N, D) float32`` encoder, or None when disabled.

    Reuses the embedding client of ``scripts/decontaminate.py`` (vLLM's
    OpenAI-compatible ``/v1/embeddings`` endpoint) instead of duplicating the
    retry/normalisation logic.  Kept injectable so the semantic dedup can be
    unit-tested without a served model.
    """
    if not base_url:
        return None
    from decontaminate import Embedder

    embedder = Embedder(base_url, model, batch_size=batch_size)
    return embedder.encode


def semantic_dedup(
    rows: list[dict],
    encode,
    *,
    threshold: float,
    bits: int = 64,
    bands: int = 8,
    seed: int = 0,
) -> tuple[list[dict], dict]:
    """Greedy embedding-level dedup of the merged hard-math pool.

    The n-gram MinHash pass catches reformatted copies; this layer catches the
    true paraphrases (the same AIME/AMC problem stated with different wording
    by two upstream corpora).  Embeddings are compared with cosine similarity;
    a row whose maximum cosine to an already-kept row is ``>= threshold`` is
    dropped (first occurrence wins, same convention as the MinHash pass).

    Candidates are found with random-hyperplane LSH (``bits``-bit signature
    split into ``bands`` byte bands), so the pass is ~O(N) rather than O(N^2).
    The kept vectors are compacted **in place** into the input matrix, which
    keeps memory at one ``(N, D)`` float32 array.

    Returns ``(kept_rows, report)``; ``encode=None`` disables the layer.
    """
    if encode is None or threshold <= 0 or len(rows) < 2:
        return rows, {"enabled": False, "threshold": threshold}

    import numpy as np

    texts = [row["_problem"] for row in rows]
    vectors = np.asarray(encode(texts), dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(rows):
        raise ValueError(
            f"embedding encoder returned shape {vectors.shape}, expected ({len(rows)}, D)"
        )
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors /= norms

    # Pack the random-hyperplane signs into byte bands.  ``bits`` and ``bands``
    # are rounded/clamped so every band is at least one byte.
    bits = max(8, bits - bits % 8)
    bands = max(1, min(bands, bits // 8))
    rows_per_band = (bits // 8) // bands
    bits = rows_per_band * bands * 8
    projection = np.random.default_rng(seed).standard_normal((vectors.shape[1], bits)).astype(np.float32)
    buckets = [defaultdict(list) for _ in range(bands)]

    kept_rows: list[dict] = []
    kept = 0
    for index in range(len(rows)):
        vector = vectors[index]
        signature = np.packbits(vector @ projection > 0)
        candidates: set[int] = set()
        for band in range(bands):
            key = signature[band * rows_per_band : (band + 1) * rows_per_band].tobytes()
            candidates.update(buckets[band].get(key, ()))
        if candidates:
            cand = np.fromiter(candidates, dtype=np.int64, count=len(candidates))
            cand = cand[cand < kept]
            if cand.size and float((vectors[cand] @ vector).max()) >= threshold:
                continue
        # Compact in place: positions < kept are already processed, so writing
        # the survivor there cannot clobber an unprocessed row.
        vectors[kept] = vector
        kept_rows.append(rows[index])
        for band in range(bands):
            key = signature[band * rows_per_band : (band + 1) * rows_per_band].tobytes()
            buckets[band][key].append(kept)
        kept += 1

    return kept_rows, {
        "enabled": True,
        "threshold": threshold,
        "embedded": len(rows),
        "dropped": len(rows) - len(kept_rows),
        "kept": len(kept_rows),
        "lsh_bits": bits,
        "lsh_bands": bands,
    }


def dedup_against_anchors(
    rows: list[dict], anchor_texts: list[str], *, threshold: float
) -> tuple[list[dict], int]:
    """Drop hard-math rows near-duplicating an old-domain math problem."""
    if not anchor_texts or not rows or threshold <= 0:
        return rows, 0
    texts = [normalize_text(text) for text in anchor_texts] + [
        normalize_text(row["_problem"]) for row in rows
    ]
    kept = set(minhash_near_dedup(texts, threshold=threshold, n_anchors=len(anchor_texts)))
    offset = len(anchor_texts)
    survivors = [row for index, row in enumerate(rows) if offset + index in kept]
    return survivors, len(rows) - len(survivors)


def read_trained_keys(paths: list[str] | None) -> set[str]:
    """Exact prompt keys of previously trained artifacts (``--exclude-path``)."""
    if not paths:
        return set()
    from mix import read_parquet_rows

    keys: set[str] = set()
    for path in paths:
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            print(f"[hard_math] WARNING: --exclude-path does not exist: {path}", file=sys.stderr)
            continue
        rows = read_parquet_rows(path)
        keys.update(_prompt_key(row) for row in rows)
        print(f"[hard_math] exclusion: {len(rows)} previously trained row(s) from {path}")
    return keys


def _prompt_key(row: dict) -> str:
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        text = " ".join(str(message.get("content", "")) for message in prompt)
    else:
        text = str(prompt or "")
    return " ".join(text.split()).casefold()


def read_anchor_texts(paths: list[str] | None, *, limit: int, seed: int) -> list[str]:
    """Problem texts of the raw old-domain math pool (``--anchor-path``)."""
    if not paths:
        return []
    from mix import read_parquet_rows

    texts: list[str] = []
    for path in paths:
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            print(f"[hard_math] WARNING: --anchor-path does not exist: {path}", file=sys.stderr)
            continue
        texts.extend(text for text in (_row_text(row) for row in read_parquet_rows(path)) if text)
        print(f"[hard_math] anchor pool {path}: {len(texts)} problem(s) so far")
    if limit and len(texts) > limit:
        texts = random.Random(seed).sample(texts, limit)
        print(f"[hard_math] anchor pool sampled down to {limit} problem(s) (--anchor-sample)")
    return texts


# ---------------------------------------------------------------------------
# candidate quota
# ---------------------------------------------------------------------------

def parse_weights(spec: str | None) -> dict[str, float]:
    """``ace=0.4,deepscaler=0.35,deepmath=0.25`` -> normalised source shares."""
    weights = dict(DEFAULT_WEIGHTS)
    if spec:
        for item in spec.split(","):
            name, separator, value = item.partition("=")
            if not separator:
                raise ValueError(f"--weights item {item!r} is not name=value")
            key = WEIGHT_ALIASES.get(name.strip())
            if key is None:
                raise ValueError(f"unknown --weights source {name.strip()!r}; expected one of {sorted(WEIGHT_ALIASES)}")
            weight = float(value)
            if weight < 0:
                raise ValueError(f"--weights source {key!r} has a negative weight")
            weights[key] = weight
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("--weights must have a positive total")
    return {source: weight / total for source, weight in weights.items()}


def _largest_remainder(shares: dict[str, int], total: int) -> dict[str, int]:
    weight = sum(shares.values())
    if weight <= 0:
        raise ValueError("cannot apportion a zero-weight quota")
    exact = {key: total * value / weight for key, value in shares.items()}
    floors = {key: int(value) for key, value in exact.items()}
    remainder = total - sum(floors.values())
    for key in sorted(exact, key=lambda k: (-(exact[k] - floors[k]), k))[:remainder]:
        floors[key] += 1
    return floors


def apply_candidate_quota(
    rows: list[dict], weights: dict[str, float], quota: int, seed: int
) -> tuple[list[dict], dict]:
    """Cap the pool at ``quota`` rows, weighted per source (before the filter)."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[row["_source"]].append(row)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    report = {"requested": quota, "supply": {key: len(value) for key, value in buckets.items()}}
    if not quota or quota >= len(rows):
        report["allocation"] = {key: len(value) for key, value in buckets.items()}
        return rows, report

    supply = {key: len(value) for key, value in buckets.items()}
    allocation: dict[str, int] = {key: 0 for key in supply}
    remaining = min(quota, sum(supply.values()))
    active = {key: weight for key, weight in weights.items() if weight > 0 and supply.get(key, 0) > 0}
    while remaining > 0 and active:
        shares = _largest_remainder(active, remaining)
        progress = 0
        for key, take in shares.items():
            add = min(take, supply.get(key, 0) - allocation[key])
            if add > 0:
                allocation[key] += add
                progress += add
        if progress == 0:
            break
        remaining = min(quota, sum(supply.values())) - sum(allocation.values())
        active = {
            key: weight
            for key, weight in active.items()
            if supply.get(key, 0) > allocation[key]
        }
    chosen: list[dict] = []
    for key, take in allocation.items():
        chosen.extend(buckets[key][:take])
    rng.shuffle(chosen)
    report["allocation"] = allocation
    return chosen, report


# ---------------------------------------------------------------------------
# difficulty filter (pass@k on the stage-4 starting checkpoint)
# ---------------------------------------------------------------------------


def load_reward():
    """The training reward dispatcher (``math_hard_*`` -> stage-1 math path)."""
    from examples.reasoning_rl.reward.compute_score import compute_score

    return compute_score


def _score_text(reward_fn, data_source: str, text: str, ground_truth: str) -> float:
    result = reward_fn(
        data_source=data_source,
        solution_str=text,
        ground_truth=ground_truth,
        extra_info={},
    )
    return float(result["score"] if isinstance(result, dict) else result)


def preflight_reward(reward_fn) -> None:
    """Abort if the math reward cannot be evaluated in this environment.

    A missing ``verl``/``math_verify`` makes ``compute_score`` log and return 0
    for every sample, which would silently classify the whole pool as all-wrong
    and drop it.  Two synthetic samples pin the two directions.
    """
    good = "<think>42</think>\n\n\\boxed{42}"
    bad = "<think>41</think>\n\n\\boxed{41}"
    try:
        good_score = _score_text(reward_fn, "math_hard_deepmath", good, "42")
        bad_score = _score_text(reward_fn, "math_hard_deepmath", bad, "42")
    except Exception as exc:  # noqa: BLE001 - surfaced as a build error below
        raise RuntimeError(f"the math reward raised during preflight: {exc}") from exc
    if good_score <= 0.5 or bad_score > 0.5:
        raise RuntimeError(
            "the math reward failed its preflight (correct -> "
            f"{good_score}, wrong -> {bad_score}); install verl + math_verify before "
            "running the difficulty filter, or pass --skip-difficulty-filter"
        )


def _difficulty_cache_paths(out: str) -> tuple[str, str]:
    stem = os.path.splitext(os.path.abspath(out))[0]
    return f"{stem}_difficulty.jsonl", f"{stem}_difficulty.done"


def _shard_cache_paths(out: str, shard_index: int, num_shards: int) -> tuple[str, str]:
    cache, done = _difficulty_cache_paths(out)
    if num_shards <= 1:
        return cache, done
    return f"{cache}.shard{shard_index}", f"{done}.shard{shard_index}"


def _read_all_difficulty_caches(out: str) -> dict[str, dict]:
    """Merge every ``<out>_difficulty.jsonl*`` shard cache next to the artifact.

    Sharded generation writes one cache per shard; the filter always reads all of
    them so the last finishing shard (or any later single-shard rerun) writes the
    complete artifact instead of only its own slice.
    """
    import glob

    cache, _ = _difficulty_cache_paths(out)
    merged: dict[str, dict] = {}
    for path in sorted(glob.glob(cache + "*")):
        if ".jsonl" not in os.path.basename(path):
            continue
        for row in read_jsonl(path):
            merged[str(row.get("task_id", ""))] = row
    merged.pop("", None)
    return merged


def filter_by_difficulty(
    rows: list[dict],
    *,
    base_url: str,
    api_key: str,
    model: str,
    k: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    enable_thinking: str,
    min_pass: float,
    max_pass: float,
    concurrency: int,
    out: str,
    num_shards: int,
    shard_index: int,
    resume: bool,
) -> tuple[list[dict], dict]:
    """Measure pass@k and keep only rows inside ``[min_pass, max_pass]``.

    The cache is keyed by task id and appended *before* a row is filtered, so a
    rerun (or a second shard) never regenerates a scored row.
    """
    from concurrent.futures import ThreadPoolExecutor

    if not base_url:
        raise ValueError("--difficulty-base-url is required unless --skip-difficulty-filter is passed")
    model_id = resolve_model_id(base_url, api_key, model)
    reward_fn = load_reward()
    preflight_reward(reward_fn)
    print(f"[hard_math] difficulty filter: model={model_id} k={k} keep=[{min_pass}, {max_pass}]")

    cache_path, done_path = _shard_cache_paths(out, shard_index, num_shards)
    if not resume:
        for path in (cache_path, done_path):
            if os.path.exists(path):
                os.remove(path)
    done = read_done(done_path) if resume else set()
    if done:
        print(f"[hard_math] resuming: {len(done)} row(s) already scored in {cache_path}")

    pending = [
        row
        for row in rows
        if stable_shard(str(row["task_id"]), num_shards) == shard_index
        and str(row["task_id"]) not in done
    ]
    print(f"[hard_math] shard {shard_index}/{num_shards}: {len(pending)} row(s) to generate")

    failure_budget = max(1, int(0.02 * max(1, len(pending))))

    def score_one(row: dict) -> dict:
        payload = {
            "model": model_id,
            "messages": row["prompt"],
            "n": k,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if enable_thinking != "default":
            payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking == "1"}
        data = chat_completion(base_url, api_key, payload, label="difficulty")
        choices = data.get("choices") or []
        texts = [response_from_choice(choice) for choice in choices]
        correct = sum(1 for text in texts if _score_text(reward_fn, row["data_source"], text, row["answer"]) > 0.5)
        return {
            "task_id": row["task_id"],
            "pass_rate": correct / len(texts) if texts else 0.0,
            "correct": correct,
            "n": len(texts),
        }

    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        for start in range(0, len(pending), max(1, concurrency)):
            batch = pending[start : start + max(1, concurrency)]
            futures = [pool.submit(score_one, row) for row in batch]
            results = []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001 - counted, then re-raised in bulk
                    failures += 1
                    if failures <= 5:
                        print(f"[hard_math] difficulty request failed: {exc}", file=sys.stderr)
            if results:
                append_jsonl(cache_path, results)
                append_done(done_path, [str(result["task_id"]) for result in results])
            if failures > failure_budget:
                raise RuntimeError(
                    f"{failures} difficulty request(s) failed (budget {failure_budget}); "
                    "fix the rollout service and rerun -- scored rows are cached"
                )

    scored = _read_all_difficulty_caches(out) if resume else {
        str(row["task_id"]): row for row in read_jsonl(cache_path)
    }
    kept: list[dict] = []
    histogram = Counter()
    all_wrong = all_right = 0
    for row in rows:
        result = scored.get(str(row["task_id"]))
        if result is None:
            continue
        rate = float(result["pass_rate"])
        histogram[f"{rate:.2f}"] += 1
        if rate <= 0.0:
            all_wrong += 1
        elif rate >= 1.0:
            all_right += 1
        if min_pass <= rate <= max_pass:
            row["pass_rate"] = rate
            kept.append(row)
    report = {
        "model": model_id,
        "k": k,
        "min_pass": min_pass,
        "max_pass": max_pass,
        "scored": len(scored),
        "all_wrong": all_wrong,
        "all_right": all_right,
        "kept": len(kept),
        "pass_rate_histogram": dict(sorted(histogram.items())),
        "failed_requests": failures,
    }
    return kept, report


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def collect_candidates(
    raw_dir: str,
    sources: list[str],
    *,
    deepmath_min_difficulty: int,
    keep_symbolic: bool,
) -> tuple[list[dict], dict]:
    """Load and normalise every source; returns ``(candidates, funnel)``."""
    funnel: dict[str, Counter] = {key: Counter() for key in sources}
    candidates: list[dict] = []
    for key in sources:
        spec = SOURCES[key]
        raw_rows = load_raw_rows(spec, raw_dir)
        for index, example in enumerate(raw_rows):
            funnel[key]["raw"] += 1
            problem, answer, difficulty = _problem_and_answer(spec, example)
            if difficulty:
                try:
                    if int(float(difficulty)) < deepmath_min_difficulty:
                        funnel[key]["below_min_difficulty"] += 1
                        continue
                except ValueError:
                    pass
            if not problem or not answer:
                funnel[key]["empty_row"] += 1
                continue
            if not _usable_problem(problem):
                funnel[key]["unusable_problem"] += 1
                continue
            answer = normalise_answer(answer)
            if not answer:
                funnel[key]["empty_answer"] += 1
                continue
            if not keep_symbolic and not is_rule_friendly_answer(answer):
                funnel[key]["non_rule_answer"] += 1
                continue
            candidates.append(
                {
                    "_source": key,
                    "_problem": problem,
                    "_answer": answer,
                    "_difficulty": difficulty,
                    "_raw_index": index,
                    "prompt": [{"role": "user", "content": _append_instruction(problem)}],
                    "answer": answer,
                    "data_source": spec.data_source,
                    "task_id": f"hardmath-{key}-{index}",
                }
            )
            funnel[key]["admissible"] += 1
    return candidates, {key: dict(value) for key, value in funnel.items()}


def build_rows(candidates: list[dict], *, seed: int) -> list[dict]:
    """Candidates -> verl parquet rows (stage-1 schema, ability math)."""
    rows = []
    for index, candidate in enumerate(candidates):
        spec = SOURCES[candidate["_source"]]
        info = dict(schema.EXTRA_INFO_DEFAULTS)
        info.update(
            {
                "split": "train",
                "index": index,
                "task_id": candidate["task_id"],
                "domain": "math",
                "source": spec.label,
                "difficulty": candidate["_difficulty"],
                "prior_solve_rate": -1.0,
                "seed": seed,
                "pass_rate": float(candidate.get("pass_rate", -1.0)),
            }
        )
        rows.append(
            {
                "data_source": candidate["data_source"],
                "prompt": candidate["prompt"],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": candidate["answer"]},
                "extra_info": info,
            }
        )
    return rows


def report_path_for(out: str) -> str:
    return os.path.splitext(out)[0] + REPORT_SUFFIX


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR, help="local raw dir (<source>.parquet/json/jsonl)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output parquet path")
    parser.add_argument(
        "--sources",
        default="ace_reason,deepscaler,deepmath",
        help="comma-separated hard-math sources (dapo is optional and shares the old math pool)",
    )
    parser.add_argument("--weights", default=None, help="per-source shares, e.g. ace_reason=0.4,deepscaler=0.35,deepmath=0.25")
    parser.add_argument("--deepmath-min-difficulty", type=int, default=6, help="DeepMath-103K difficulty floor")
    parser.add_argument("--keep-symbolic", action="store_true", help="keep answers the rule matcher cannot read")
    parser.add_argument("--candidate-quota", type=int, default=28000, help="cap before the difficulty filter (0 = all)")
    parser.add_argument("--dedup-threshold", type=float, default=0.6, help="in-pool MinHash Jaccard threshold (0 disables)")
    parser.add_argument(
        "--embedding-base-url",
        default="",
        help="OpenAI-compatible /v1/embeddings endpoint for the semantic dedup layer "
        "(empty disables it; e.g. http://127.0.0.1:8001/v1)",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--embedding-threshold",
        type=float,
        default=0.95,
        help="cosine similarity at or above which a hard-math row is a semantic duplicate",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--embedding-lsh-bits", type=int, default=64, help="random-hyperplane signature bits")
    parser.add_argument("--embedding-lsh-bands", type=int, default=8, help="LSH bands the signature is split into")
    parser.add_argument("--anchor-path", action="append", default=None, help="raw old math pool(s) to near-dedup against")
    parser.add_argument("--anchor-threshold", type=float, default=0.6, help="MinHash threshold against --anchor-path")
    parser.add_argument("--anchor-sample", type=int, default=0, help="random cap on anchor texts (0 = all)")
    parser.add_argument("--exclude-path", action="append", default=None, help="previously trained artifacts (repeatable)")
    parser.add_argument("--decontaminate-path", action="append", default=None, help="held-out set(s) to remove (repeatable)")
    parser.add_argument("--decontaminate-ngram", type=int, default=8, help="word n-gram size for decontamination")
    parser.add_argument("--decontaminate-minhash", type=float, default=0.8, help="near-verbatim paraphrase threshold (0 disables)")
    parser.add_argument(
        "--allow-no-decontamination",
        action="store_true",
        help="explicitly waive the fail-closed AIME 2024/2025 requirement",
    )
    parser.add_argument("--difficulty-base-url", default="", help="vLLM OpenAI-compatible base URL")
    parser.add_argument("--difficulty-api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--difficulty-model", default="", help="served model id (empty = GET /models)")
    parser.add_argument("--difficulty-k", type=int, default=8, help="samples per problem (8-16)")
    parser.add_argument("--difficulty-temperature", type=float, default=1.0)
    parser.add_argument("--difficulty-top-p", type=float, default=1.0)
    parser.add_argument("--difficulty-max-tokens", type=int, default=8192)
    parser.add_argument(
        "--difficulty-enable-thinking",
        choices=("1", "0", "default"),
        default="1",
        help="chat_template_kwargs.enable_thinking for the rollout service (default sends True)",
    )
    parser.add_argument("--difficulty-concurrency", type=int, default=64)
    parser.add_argument("--min-pass", type=float, default=0.1)
    parser.add_argument("--max-pass", type=float, default=0.9)
    parser.add_argument("--skip-difficulty-filter", action="store_true", help="emit the pool untagged (not recommended)")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--seed", type=int, default=44)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.min_pass <= args.max_pass <= 1.0:
        raise ValueError(f"--min-pass/--max-pass must satisfy 0 <= min <= max <= 1, got {args.min_pass}/{args.max_pass}")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    if args.difficulty_k < 1:
        raise ValueError("--difficulty-k must be at least 1")
    sources = [item.strip() for item in args.sources.split(",") if item.strip()]
    unknown = [key for key in sources if key not in SOURCES]
    if unknown:
        raise ValueError(f"unknown --sources {unknown}; expected one of {sorted(SOURCES)}")
    weights = parse_weights(args.weights)
    missing_weight = [key for key in sources if weights.get(key, 0) <= 0]
    if missing_weight:
        raise ValueError(f"--sources {missing_weight} have a zero --weights share")

    if not args.decontaminate_path and not args.allow_no_decontamination:
        raise ValueError(
            "no --decontaminate-path given. AIME 2024/2025 contamination invalidates the "
            "stage-4 evaluation; pass the held-out parquet(s), or explicitly waive with "
            "--allow-no-decontamination"
        )

    raw_dir = os.path.expanduser(args.raw_dir)
    os.makedirs(raw_dir, exist_ok=True)

    candidates, funnel = collect_candidates(
        raw_dir,
        sources,
        deepmath_min_difficulty=args.deepmath_min_difficulty,
        keep_symbolic=args.keep_symbolic,
    )
    print(f"[hard_math] admissible candidates: {len(candidates)}")
    for key, counts in funnel.items():
        print(f"[hard_math]   {key}: {counts}")

    trained = read_trained_keys(args.exclude_path)
    if trained:
        before = len(candidates)
        candidates = [row for row in candidates if _prompt_key(row) not in trained]
        print(f"[hard_math] -{before - len(candidates)} already trained row(s)")

    held_out_texts, decontam_report = load_decontamination_texts(args.decontaminate_path)
    candidates, contamination = decontaminate(
        candidates,
        held_out_texts,
        ngram_n=args.decontaminate_ngram,
        minhash_threshold=args.decontaminate_minhash,
    )
    decontam_report.update(contamination)
    print(f"[hard_math] decontamination: {contamination}")

    candidates, dedup_report = dedup_pool(candidates, args.dedup_threshold)
    print(f"[hard_math] dedup: {dedup_report}")

    if not 0.0 < args.embedding_threshold <= 1.0:
        raise ValueError(f"--embedding-threshold must be in (0, 1], got {args.embedding_threshold}")
    encoder = load_embedding_encoder(
        args.embedding_base_url, args.embedding_model, args.embedding_batch_size
    )
    candidates, semantic_report = semantic_dedup(
        candidates,
        encoder,
        threshold=args.embedding_threshold,
        bits=args.embedding_lsh_bits,
        bands=args.embedding_lsh_bands,
        seed=args.seed,
    )
    if encoder is None:
        print(
            "[hard_math] WARNING: no --embedding-base-url; semantic dedup is disabled "
            "(in-pool dedup is n-gram MinHash only)",
            file=sys.stderr,
        )
    else:
        print(f"[hard_math] semantic dedup: {semantic_report}")

    anchors = read_anchor_texts(args.anchor_path, limit=args.anchor_sample, seed=args.seed)
    candidates, anchor_dropped = dedup_against_anchors(candidates, anchors, threshold=args.anchor_threshold)
    print(f"[hard_math] -{anchor_dropped} near-duplicate(s) of the old math pool")

    candidates, quota_report = apply_candidate_quota(candidates, weights, args.candidate_quota, args.seed)
    print(f"[hard_math] candidate quota: {quota_report}")

    if args.skip_difficulty_filter:
        print("[hard_math] WARNING: --skip-difficulty-filter: pass_rate is left at -1", file=sys.stderr)
        difficulty_report = {"skipped": True}
    else:
        candidates, difficulty_report = filter_by_difficulty(
            candidates,
            base_url=args.difficulty_base_url,
            api_key=args.difficulty_api_key,
            model=args.difficulty_model,
            k=args.difficulty_k,
            temperature=args.difficulty_temperature,
            top_p=args.difficulty_top_p,
            max_tokens=args.difficulty_max_tokens,
            enable_thinking=args.difficulty_enable_thinking,
            min_pass=args.min_pass,
            max_pass=args.max_pass,
            concurrency=args.difficulty_concurrency,
            out=args.out,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            resume=not args.no_resume,
        )
        print(f"[hard_math] difficulty: {difficulty_report}")

    if not candidates:
        print("no rows survived filtering; nothing written", file=sys.stderr)
        return 1

    rows = build_rows(candidates, seed=args.seed)
    schema.normalise_extra_info(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    schema.write_rows_parquet(rows, args.out)

    by_source = Counter(row["data_source"] for row in rows)
    report = {
        "config": {
            "raw_dir": raw_dir,
            "sources": sources,
            "weights": weights,
            "deepmath_min_difficulty": args.deepmath_min_difficulty,
            "keep_symbolic": args.keep_symbolic,
            "candidate_quota": args.candidate_quota,
            "dedup_threshold": args.dedup_threshold,
            "embedding_base_url": args.embedding_base_url,
            "embedding_model": args.embedding_model if args.embedding_base_url else "",
            "embedding_threshold": args.embedding_threshold,
            "embedding_batch_size": args.embedding_batch_size,
            "embedding_lsh_bits": args.embedding_lsh_bits,
            "embedding_lsh_bands": args.embedding_lsh_bands,
            "anchor_paths": args.anchor_path or [],
            "anchor_threshold": args.anchor_threshold,
            "anchor_sample": args.anchor_sample,
            "exclude_paths": args.exclude_path or [],
            "decontaminate_paths": args.decontaminate_path or [],
            "decontaminate_ngram": args.decontaminate_ngram,
            "decontaminate_minhash": args.decontaminate_minhash,
            "difficulty_k": args.difficulty_k,
            "min_pass": args.min_pass,
            "max_pass": args.max_pass,
            "seed": args.seed,
        },
        "funnel": funnel,
        "already_trained_removed": len(trained),
        "decontamination": decontam_report,
        "dedup": dedup_report,
        "semantic_dedup": semantic_report,
        "anchor_dropped": anchor_dropped,
        "candidate_quota": quota_report,
        "difficulty": difficulty_report,
        "rows": len(rows),
        "by_source": dict(sorted(by_source.items())),
    }
    report_path = report_path_for(args.out)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"rows written: {len(rows)} -> {args.out}")
    print(f"  by source: {dict(sorted(by_source.items()))}")
    print(f"report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
