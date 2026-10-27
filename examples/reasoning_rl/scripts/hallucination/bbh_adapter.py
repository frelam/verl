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
"""Stage-4 BBH-gap adapter (HALLUCINATION_RL_DESIGN.md D32).

Stage 3 trains abstention / calibration; stage 4 already adds the
instruction-following domain.  This adapter adds a **targeted skill-repair**
domain for the five BIG-Bench-Hard subtasks the stage-3 checkpoint is weakest on
(``BBH_Benchmark_评测分析报告.md``):

=====================  ======  ================================================
subtask                score   what the rows train
=====================  ======  ================================================
``dyck_languages``     40%     stack-like bracket completion
``disambiguation_qa``  64%     pronoun / referent ambiguity resolution
``causal_judgement``   64.7%   causal attribution, confounders, preemption
``movie_recommendation`` 66.4% preference constraints + candidate comparison
``geometric_shapes``   74%     SVG path parsing / vertex-counting geometry
=====================  ======  ================================================

Data sources
------------

The first four are **fresh generators**, not the benchmark rows.  The original
BIG-Bench tasks are hand-written or one-off test sets (BBH ships only 250 rows
per subtask), so training on them would be direct contamination; the generators
re-create the *skill* instead:

* Dyck: PCFG-generated Dyck-1..4 words (the BIG-Bench task's own generator
  parameters), prefix + closing suffix split, lengths bounded to [4, 100].
* Geometric shapes: shapes rendered as SVG ``M``/``L``/``A`` paths and
  re-classified by an independent geometry parser before being written, so a
  mislabelled row cannot enter the pool.
* Disambiguation QA: ten template families covering expertise cues, speaker /
  listener knowledge, possessive ambiguity, "they" number ambiguity, and
  genuinely ambiguous two-referent sentences.
* Causal judgement: twelve vignette families covering direct causation,
  common-cause confounders, preemption, omission with / without a duty,
  counterfactual chains, correlation without mechanism, and norm-violating
  actions that do / do not cause the outcome.

``movie_recommendation`` is built from the **same public source the BIG-Bench
task used** -- MovieLens latest-small -- because the skill needs real user
preference data.  The loader downloads ``ml-latest-small.zip`` from GroupLens
when the CSV files are not present (or use ``--movielens-dir`` to point at a
local copy).  Queries are built the benchmark's way: four liked seed movies,
one held-out liked movie as the gold, three disliked movies as distractors, with
an optional genre-consistency filter so the gold is not simply the most popular
title.  MovieLens is released by GroupLens for research use; the build reads the
CSVs locally and does not redistribute them.

Every row uses the stage-1 parquet shape (``ability="bbh"``) plus a
``bbh_task`` audit key, so ``mix_halluc.py`` can mix it with the hallucination
and old domains without a schema cast.  The ground truth is a JSON payload::

    {"task": "dyck_languages", "answer": "] )"}

and the reward is ``reward/bbh_verifier.py`` (routed by the ``bbh_`` data-source
prefix in ``reward/compute_score.py``).

Usage::

    python bbh_adapter.py --out ~/data/reasoning_rl/halluc/built_stage4/bbh.parquet
    python bbh_adapter.py --tasks dyck_languages,geometric_shapes --seed 44
    python bbh_adapter.py --movielens-dir /data/ml-latest-small
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import schema  # noqa: E402

TASK_DYCK = "dyck_languages"
TASK_GEOMETRIC = "geometric_shapes"
TASK_DISAMBIGUATION = "disambiguation_qa"
TASK_CAUSAL = "causal_judgement"
TASK_MOVIE = "movie_recommendation"

TASKS = (TASK_DYCK, TASK_GEOMETRIC, TASK_DISAMBIGUATION, TASK_CAUSAL, TASK_MOVIE)
DATA_SOURCES = {
    TASK_DYCK: "bbh_dyck",
    TASK_GEOMETRIC: "bbh_geometric_shapes",
    TASK_DISAMBIGUATION: "bbh_disambiguation_qa",
    TASK_CAUSAL: "bbh_causal_judgement",
    TASK_MOVIE: "bbh_movie_recommendation",
}
ABILITY = "bbh"
DOMAIN = "bbh"

DEFAULT_RAW_DIR = os.path.expanduser("~/data/reasoning_rl/halluc/raw/bbh")
# Same convention as every other adapter: the default output lives in the
# directory test_mix_halluc asserts against; the stage-4 build overrides it with
# built_stage4/bbh.parquet.
DEFAULT_OUT = os.path.expanduser("~/data/reasoning_rl/halluc/built/bbh.parquet")

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
MOVIELENS_DIRNAME = "ml-latest-small"

#: Pool sizes.  The stage-4 mix asks for ~4.4k BBH rows (10% of the final train
#: set); the pools are 1.5-2x larger so val carving and dedup cannot starve the
#: quota.  The real build caps each task at its supply.
DEFAULT_QUOTA = {
    TASK_DYCK: 3000,
    TASK_GEOMETRIC: 2000,
    TASK_DISAMBIGUATION: 2000,
    TASK_CAUSAL: 2000,
    TASK_MOVIE: 1500,
}


@dataclass(frozen=True)
class Generated:
    """One rendered problem: prompt, gold answer and audit metadata."""

    prompt: str
    answer: str
    difficulty: str = ""
    options: list[dict] = field(default_factory=list)
    correct_option_id: str = ""


# ---------------------------------------------------------------------------
# shared row assembly
# ---------------------------------------------------------------------------


def prompt_key(prompt: list[dict]) -> str:
    """Whitespace/case-folded key used for exact dedup and ``--exclude-path``.

    Deliberately *not* ``dedup.exact_key``: that helper strips non-alphanumeric
    characters, which would collapse every Dyck prompt (they differ only in
    brackets) onto one key.
    """
    text = " ".join(str(message.get("content", "")) for message in prompt or [])
    return " ".join(text.split()).casefold()


def make_row(task: str, generated: Generated, index: int, seed: int) -> dict:
    """Assemble one verl parquet row in the stage-1 shape."""
    info = dict(schema.EXTRA_INFO_DEFAULTS)
    info.update(
        {
            "split": "train",
            "index": index,
            "task_id": f"bbh:{task}:{seed}:{index:06d}",
            "domain": DOMAIN,
            "source": f"BBH-gap {task}",
            "difficulty": generated.difficulty,
            "prior_solve_rate": -1.0,
            "seed": seed,
            "bbh_task": task,
            "options": [dict(option) for option in generated.options],
            "correct_option_id": generated.correct_option_id,
        }
    )
    return {
        "data_source": DATA_SOURCES[task],
        "prompt": [{"role": "user", "content": generated.prompt}],
        "ability": ABILITY,
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps({"task": task, "answer": generated.answer}, ensure_ascii=False),
        },
        "extra_info": info,
    }


def read_excluded_keys(paths: list[str] | None) -> set[str]:
    """Prompt keys of previously trained artifacts (``--exclude-path``)."""
    if not paths:
        return set()
    from mix import read_parquet_rows

    keys: set[str] = set()
    for path in paths:
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            print(f"[bbh_adapter] WARNING: --exclude-path does not exist: {path}", file=sys.stderr)
            continue
        rows = read_parquet_rows(path)
        keys.update(prompt_key(row.get("prompt", [])) for row in rows)
        print(f"[bbh_adapter] excluded {len(rows)} previously trained rows from {path}", file=sys.stderr)
    return keys


# ---------------------------------------------------------------------------
# Dyck languages
# ---------------------------------------------------------------------------

_DYCK_PAIRS = (("(", ")"), ("[", "]"), ("{", "}"), ("<", ">"))
_DYCK_CLOSERS = frozenset(close for _, close in _DYCK_PAIRS)
_DYCK_CLOSER_FOR = {open_: close for open_, close in _DYCK_PAIRS}


def _dyck_expand(rng: random.Random, n_types: int, depth: int, max_depth: int) -> list[str]:
    """PCFG expansion: nest (p), sequence (q), or stop (1-p-q)."""
    if depth >= max_depth:
        return []
    roll = rng.random()
    if roll < 0.30:
        return []
    if roll < 0.62:
        open_, close = _DYCK_PAIRS[rng.randrange(n_types)]
        return [open_, *_dyck_expand(rng, n_types, depth + 1, max_depth), close]
    return [
        *_dyck_expand(rng, n_types, depth + 1, max_depth),
        *_dyck_expand(rng, n_types, depth + 1, max_depth),
    ]


def _valid_prefix(tokens: list[str]) -> bool:
    stack: list[str] = []
    for token in tokens:
        if token in _DYCK_CLOSERS:
            if not stack or stack[-1] != token:
                return False
            stack.pop()
        elif token in _DYCK_CLOSER_FOR:
            stack.append(_DYCK_CLOSER_FOR[token])
        else:
            return False
    return True


def _is_balanced(tokens: list[str]) -> bool:
    """Whether every opening bracket is closed (the stack ends empty)."""
    stack: list[str] = []
    for token in tokens:
        if token in _DYCK_CLOSERS:
            if not stack or stack[-1] != token:
                return False
            stack.pop()
        elif token in _DYCK_CLOSER_FOR:
            stack.append(_DYCK_CLOSER_FOR[token])
        else:
            return False
    return not stack


def _max_depth(tokens: list[str]) -> int:
    depth = peak = 0
    for token in tokens:
        if token in _DYCK_CLOSER_FOR:
            depth += 1
            peak = max(peak, depth)
        elif token in _DYCK_CLOSERS:
            depth -= 1
    return peak


def _dyck_cut(tokens: list[str], rng: random.Random) -> tuple[list[str], list[str]] | None:
    """Split a Dyck word into a prefix and the last 1-4 closing brackets."""
    n = len(tokens)
    candidates: list[int] = []
    for suffix_len in range(1, min(4, n - 1) + 1):
        cut = n - suffix_len
        suffix = tokens[cut:]
        if all(token in _DYCK_CLOSERS for token in suffix) and _valid_prefix(tokens[:cut]):
            candidates.append(cut)
    if not candidates:
        return None
    cut = rng.choice(candidates)
    return tokens[:cut], tokens[cut:]


def generate_dyck(rng: random.Random, index: int) -> Generated:
    """A Dyck-1..4 prefix whose missing closing suffix is the gold answer."""
    del index
    for _ in range(500):
        n_types = rng.choices((1, 2, 3, 4), weights=(0.05, 0.15, 0.35, 0.45), k=1)[0]
        full = _dyck_expand(rng, n_types, 0, 8)
        if not 4 <= len(full) <= 100:
            continue
        cut = _dyck_cut(full, rng)
        if cut is None:
            continue
        prefix, suffix = cut
        # Require a non-trivial prefix: the benchmark's shortest rows (e.g.
        # ``[ [``) live in a tiny sequence space, and a fresh generator could
        # reproduce one by chance.  Longer prefixes are both collision-free in
        # practice and the part of the distribution that actually matters.
        if not 6 <= len(prefix) <= 95:
            continue
        prompt = (
            "Complete the rest of the sequence, making sure that the parentheses are closed properly. "
            f"Input: {' '.join(prefix)}"
        )
        return Generated(
            prompt=prompt,
            answer=" ".join(suffix),
            difficulty=f"types{n_types}_depth{_max_depth(prefix)}_len{len(prefix)}",
        )
    raise RuntimeError("could not generate a Dyck row within the length bounds")


# ---------------------------------------------------------------------------
# Geometric shapes (SVG paths)
# ---------------------------------------------------------------------------

#: The label set of the BIG-Bench task (alphabetical, as its options are rendered).
_SHAPE_NAMES = (
    "circle",
    "heptagon",
    "hexagon",
    "kite",
    "line",
    "octagon",
    "pentagon",
    "rectangle",
    "sector",
    "triangle",
)
_ALL_SHAPE_NAMES = tuple(sorted(_SHAPE_NAMES + ("trapezoid",)))


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _rotate(points: list[tuple[float, float]], angle: float, center=(50.0, 50.0)) -> list[tuple[float, float]]:
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    cx, cy = center
    return [
        (cx + (x - cx) * cos_a - (y - cy) * sin_a, cy + (x - cx) * sin_a + (y - cy) * cos_a)
        for x, y in points
    ]


def _regular_polygon(count: int, radius: float, angle: float) -> list[tuple[float, float]]:
    return [
        (
            50.0 + radius * math.cos(angle + 2 * math.pi * i / count),
            50.0 + radius * math.sin(angle + 2 * math.pi * i / count),
        )
        for i in range(count)
    ]


def _render_polygon(points: list[tuple[float, float]], closed: bool = True) -> str:
    parts = [f"M {points[0][0]:.2f},{points[0][1]:.2f}"]
    parts.extend(f"L {x:.2f},{y:.2f}" for x, y in points[1:])
    if closed:
        parts.append(f"L {points[0][0]:.2f},{points[0][1]:.2f}")
    return " ".join(parts)


def _render_circle(radius: float) -> str:
    x1, y1 = 50.0 - radius, 50.0
    x2, y2 = 50.0 + radius, 50.0
    return (
        f"M {x1:.2f},{y1:.2f} A {radius:.2f},{radius:.2f} 0 1,0 {x2:.2f},{y2:.2f} "
        f"A {radius:.2f},{radius:.2f} 0 1,0 {x1:.2f},{y1:.2f}"
    )


def _render_sector(radius: float, start: float, span: float) -> str:
    end = start + span
    x1, y1 = 50.0 + radius * math.cos(start), 50.0 + radius * math.sin(start)
    x2, y2 = 50.0 + radius * math.cos(end), 50.0 + radius * math.sin(end)
    large = 1 if span > math.pi else 0
    return (
        f"M 50.00,50.00 L {x1:.2f},{y1:.2f} "
        f"A {radius:.2f},{radius:.2f} 0 {large},1 {x2:.2f},{y2:.2f} Z"
    )


def _cross(left: tuple[float, float], right: tuple[float, float]) -> float:
    return left[0] * right[1] - left[1] * right[0]


def _sub(left: tuple[float, float], right: tuple[float, float]) -> tuple[float, float]:
    return (left[0] - right[0], left[1] - right[1])


def _parallel(a, b, c, d) -> bool:
    u, v = _sub(b, a), _sub(d, c)
    nu, nv = math.hypot(*u), math.hypot(*v)
    if nu == 0 or nv == 0:
        return False
    return abs(_cross(u, v)) / (nu * nv) < 1e-3


def _cos_angle(u: tuple[float, float], v: tuple[float, float]) -> float:
    nu, nv = math.hypot(*u), math.hypot(*v)
    if nu == 0 or nv == 0:
        return 0.0
    return (u[0] * v[0] + u[1] * v[1]) / (nu * nv)


def _is_rectangle(points: list[tuple[float, float]]) -> bool:
    sides = [_sub(points[(i + 1) % 4], points[i]) for i in range(4)]
    return all(abs(_cos_angle(sides[i], sides[(i + 1) % 4])) < 1e-3 for i in range(4))


def _is_kite(points: list[tuple[float, float]]) -> bool:
    sides = [_distance(points[i], points[(i + 1) % 4]) for i in range(4)]
    tolerance = 1e-3 * max(sides)
    first_pair = abs(sides[0] - sides[1]) < tolerance and abs(sides[2] - sides[3]) < tolerance
    second_pair = abs(sides[1] - sides[2]) < tolerance and abs(sides[3] - sides[0]) < tolerance
    return (first_pair or second_pair) and not _is_rectangle(points)


def _is_trapezoid(points: list[tuple[float, float]]) -> bool:
    first = _parallel(points[0], points[1], points[2], points[3])
    second = _parallel(points[1], points[2], points[3], points[0])
    return first != second


def classify_path(path: str) -> str | None:
    """Independent geometry classifier used to audit every generated SVG path."""
    if "A" in path or "a" in path:
        return "sector" if "L" in path or "l" in path else "circle"
    points = [
        (float(x), float(y))
        for x, y in re.findall(r"[ML]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)", path)
    ]
    if len(points) > 1 and _distance(points[0], points[-1]) < 1e-6:
        points = points[:-1]
    count = len(points)
    if count == 2:
        return "line"
    if count == 3:
        return "triangle"
    if count == 4:
        if _is_rectangle(points):
            return "rectangle"
        if _is_kite(points):
            return "kite"
        if _is_trapezoid(points):
            return "trapezoid"
        return None
    return {5: "pentagon", 6: "hexagon", 7: "heptagon", 8: "octagon"}.get(count)


def _random_shape(rng: random.Random) -> tuple[str, str, int]:
    """A ``(label, svg_path, vertex_count)`` triple from the benchmark's families."""
    kind = rng.choices(
        ("line", "triangle", "rectangle", "kite", "trapezoid", "polygon", "circle", "sector"),
        weights=(0.09, 0.13, 0.13, 0.11, 0.11, 0.28, 0.08, 0.07),
        k=1,
    )[0]
    if kind == "line":
        for _ in range(50):
            p1 = (rng.uniform(5, 95), rng.uniform(5, 95))
            p2 = (rng.uniform(5, 95), rng.uniform(5, 95))
            if _distance(p1, p2) >= 20:
                return "line", _render_polygon([p1, p2], closed=False), 2
        return "line", _render_polygon([(10.0, 10.0), (80.0, 80.0)], closed=False), 2

    if kind == "triangle":
        for _ in range(50):
            points = [(rng.uniform(8, 92), rng.uniform(8, 92)) for _ in range(3)]
            area = abs(_cross(_sub(points[1], points[0]), _sub(points[2], points[0]))) / 2
            if area >= 200 and min(_distance(points[i], points[(i + 1) % 3]) for i in range(3)) >= 15:
                return "triangle", _render_polygon(points), 3
        return "triangle", _render_polygon([(10.0, 10.0), (80.0, 20.0), (30.0, 85.0)]), 3

    if kind == "rectangle":
        width = rng.uniform(25, 55)
        height = rng.uniform(10, width / 1.5)
        angle = rng.uniform(0, math.pi / 2)
        points = [
            (-width / 2, -height / 2),
            (width / 2, -height / 2),
            (width / 2, height / 2),
            (-width / 2, height / 2),
        ]
        return "rectangle", _render_polygon(_rotate(points, angle)), 4

    if kind == "kite":
        long_axis = rng.uniform(25, 45)
        short_axis = long_axis * rng.uniform(0.35, 0.6)
        width = rng.uniform(15, 35)
        points = [(0.0, long_axis), (width, 0.0), (0.0, -short_axis), (-width, 0.0)]
        return "kite", _render_polygon(_rotate(points, rng.uniform(0, 2 * math.pi))), 4

    if kind == "trapezoid":
        bottom = rng.uniform(35, 60)
        top = bottom * rng.uniform(0.3, 0.7)
        height = rng.uniform(15, 35)
        points = [(-bottom / 2, height / 2), (bottom / 2, height / 2), (top / 2, -height / 2), (-top / 2, -height / 2)]
        return "trapezoid", _render_polygon(_rotate(points, rng.uniform(0, 2 * math.pi))), 4

    if kind == "polygon":
        count = rng.choice((5, 6, 7, 8))
        radius = rng.uniform(22, 44)
        points = _regular_polygon(count, radius, rng.uniform(0, 2 * math.pi))
        name = {5: "pentagon", 6: "hexagon", 7: "heptagon", 8: "octagon"}[count]
        return name, _render_polygon(points), count

    if kind == "circle":
        return "circle", _render_circle(rng.uniform(15, 40)), 0

    radius = rng.uniform(20, 40)
    start = rng.uniform(0, 2 * math.pi)
    span = rng.uniform(math.radians(40), math.radians(160))
    return "sector", _render_sector(radius, start, span), 0


def generate_geometric(rng: random.Random, index: int) -> Generated:
    """One SVG path plus the ten/eleven-option shape block of the benchmark."""
    del index
    for _ in range(300):
        label, path, _ = _random_shape(rng)
        if classify_path(path) != label:
            continue
        names = list(_ALL_SHAPE_NAMES) if label == "trapezoid" or rng.random() < 0.25 else list(_SHAPE_NAMES)
        options = [{"id": chr(ord("A") + i), "text": name} for i, name in enumerate(names)]
        correct = next(option["id"] for option in options if option["text"] == label)
        prompt = (
            f'This SVG path element <path d="{path}"/> draws a\nOptions:\n'
            + "\n".join(f"({option['id']}) {option['text']}" for option in options)
        )
        return Generated(
            prompt=prompt,
            answer=correct,
            difficulty=f"{label}_options{len(options)}",
            options=options,
            correct_option_id=correct,
        )
    raise RuntimeError("could not generate a self-consistent geometric-shape row")


# ---------------------------------------------------------------------------
# Disambiguation QA
# ---------------------------------------------------------------------------

_UNISEX_NAMES = (
    "Avery",
    "Bailey",
    "Casey",
    "Dakota",
    "Jordan",
    "Morgan",
    "Parker",
    "Quinn",
    "Reese",
    "Riley",
    "Rowan",
    "Taylor",
)
_FEMALE_NAMES = ("Alice", "Bianca", "Clara", "Diana", "Elena", "Fiona", "Grace", "Hannah", "Irene", "Julia")
_MALE_NAMES = ("Andrew", "Brian", "Carlos", "David", "Ethan", "Frank", "George", "Henry", "Ivan", "Jack")

#: (expert role, other role, skill that unambiguously belongs to the expert).
_ROLE_SKILLS = (
    ("doctor", "patient", "treat the infection"),
    ("mechanic", "customer", "repair the engine"),
    ("teacher", "student", "explain the lesson"),
    ("plumber", "homeowner", "fix the leaking pipe"),
    ("chef", "waiter", "prepare the sauce"),
    ("pilot", "passenger", "fly the plane"),
    ("programmer", "designer", "debug the code"),
    ("lawyer", "witness", "question the witness"),
    ("engineer", "technician", "inspect the machine"),
    ("nurse", "patient", "change the bandage"),
    ("librarian", "reader", "catalog the books"),
    ("photographer", "model", "adjust the lighting"),
    ("electrician", "tenant", "repair the wiring"),
    ("dentist", "patient", "fill the cavity"),
    ("architect", "client", "draw the floor plan"),
    ("translator", "author", "translate the article"),
    ("accountant", "client", "audit the books"),
    ("veterinarian", "pet owner", "treat the animal"),
    ("carpenter", "customer", "build the cabinet"),
    ("florist", "customer", "arrange the flowers"),
    ("baker", "customer", "decorate the cake"),
    ("barber", "customer", "cut the hair"),
    ("coach", "player", "plan the training"),
    ("judge", "defendant", "rule on the case"),
)
_ROLE_TASKS = {
    "mechanic": "repair the engine",
    "plumber": "fix the leaking pipe",
    "teacher": "explain the lesson",
    "chef": "prepare the sauce",
    "pilot": "fly the plane",
    "programmer": "debug the code",
    "electrician": "repair the wiring",
    "dentist": "fill the cavity",
    "librarian": "catalog the books",
    "photographer": "adjust the lighting",
    "translator": "translate the article",
    "architect": "draw the floor plan",
    "accountant": "audit the books",
    "veterinarian": "treat the animal",
    "carpenter": "build the cabinet",
    "florist": "arrange the flowers",
    "baker": "decorate the cake",
    "barber": "cut the hair",
    "coach": "plan the training",
    "judge": "rule on the case",
}
_ACTIONS = (
    "liked the proposal",
    "shared an anecdote",
    "finished the report",
    "bought the coffee",
    "found the keys",
    "missed the train",
    "read the memo",
    "planned the trip",
    "chose the restaurant",
    "reviewed the budget",
)
_OBJECTS = ("report", "design", "memo", "proposal", "invoice", "schedule", "contract", "budget")
_EXPERIENCES = (
    "visited the museum",
    "read the report",
    "met the client",
    "finished the assignment",
    "seen the movie",
    "attended the workshop",
    "reviewed the application",
)


def _plural(role: str) -> str:
    if role.endswith("y"):
        return role[:-1] + "ies"
    if role.endswith(("s", "x", "ch", "sh")):
        return role + "es"
    return role + "s"


def _referent_mcq(
    rng: random.Random,
    sentence: str,
    first: str,
    second: str,
    gold_side: str,
    family: str,
) -> Generated:
    """Render the three-option disambiguation block and track the gold side.

    ``gold_side`` is ``A`` / ``B`` (the first / second referent) or ``C``
    (ambiguous).  The two referents are shuffled across options A/B, so the
    correct letter is recomputed after the shuffle.
    """
    referents = [("A", first), ("B", second)]
    rng.shuffle(referents)
    options = [
        {"id": "A", "text": referents[0][1]},
        {"id": "B", "text": referents[1][1]},
        {"id": "C", "text": "Ambiguous"},
    ]
    if gold_side == "C":
        correct = "C"
    else:
        correct = "A" if referents[0][0] == gold_side else "B"
    prompt = (
        "In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), "
        "or state that it is ambiguous.\n"
        f"Sentence: {sentence}\nOptions:\n"
        f"(A) {options[0]['text']}\n(B) {options[1]['text']}\n(C) Ambiguous"
    )
    return Generated(
        prompt=prompt,
        answer=correct,
        difficulty=family,
        options=options,
        correct_option_id=correct,
    )


def _disambig_expertise(rng: random.Random) -> Generated:
    expert, other, skill = rng.choice(_ROLE_SKILLS)
    pronoun = rng.choice(("he", "she", "they"))
    sentence = rng.choice(
        (
            f"The {expert} helped the {other} because {pronoun} knew how to {skill}.",
            f"Because {pronoun} knew how to {skill}, the {expert} helped the {other}.",
            f"The {other} asked the {expert} for help, since {pronoun} knew how to {skill}.",
        )
    )
    return _referent_mcq(
        rng,
        sentence,
        f"The {expert} knew how to {skill}",
        f"The {other} knew how to {skill}",
        "A",
        "expertise",
    )


def _disambig_ask_experience(rng: random.Random) -> Generated:
    first, second = rng.sample(_UNISEX_NAMES, 2)
    experience = rng.choice(_EXPERIENCES)
    pronoun = rng.choice(("he", "she", "they"))
    sentence = f"{first} asked {second} if {pronoun} had {experience}."
    return _referent_mcq(
        rng,
        sentence,
        f"{first} had {experience}",
        f"{second} had {experience}",
        "B",
        "addressee_knowledge",
    )


def _disambig_possessive(rng: random.Random) -> Generated:
    name = rng.choice(_UNISEX_NAMES)
    role = rng.choice(_ROLE_SKILLS)[0]
    pronoun = rng.choice(("his", "her", "their"))
    sentence = f"{name} planned to meet the {role} at {pronoun} office."
    return _referent_mcq(
        rng,
        sentence,
        f"The office was {name}'s office",
        f"The office was the {role}'s office",
        "C",
        "possessive",
    )


def _disambig_two_referent_action(rng: random.Random) -> Generated:
    first, second = rng.sample([role for role, _, _ in _ROLE_SKILLS], 2)
    action = rng.choice(_ACTIONS)
    pronoun = rng.choice(("he", "she", "they"))
    verb = rng.choice(("disagreed with", "spoke with", "worked with"))
    sentence = f"The {first} {verb} the {second} because {pronoun} {action}."
    return _referent_mcq(
        rng,
        sentence,
        f"The {first} {action}",
        f"The {second} {action}",
        "C",
        "two_referent_action",
    )


def _disambig_role_task(rng: random.Random) -> Generated:
    worker = rng.choice(list(_ROLE_TASKS))
    task = _ROLE_TASKS[worker]
    other = rng.choice([role for role, _, _ in _ROLE_SKILLS if role != worker])
    pronoun = rng.choice(("he", "she", "they"))
    sentence = f"The {worker} showed the {other} how {pronoun} would {task}."
    return _referent_mcq(
        rng,
        sentence,
        f"The {worker} would {task}",
        f"The {other} would {task}",
        "A",
        "role_cue",
    )


def _disambig_plural(rng: random.Random) -> Generated:
    worker = rng.choice(list(_ROLE_TASKS))
    task = _ROLE_TASKS[worker]
    other = rng.choice([role for role, _, _ in _ROLE_SKILLS if role != worker])
    sentence = f"The {_plural(worker)} told the {other} that they would {task}."
    return _referent_mcq(
        rng,
        sentence,
        f"The {_plural(worker)} would {task}",
        f"The {other} would {task}",
        "A",
        "plural_agreement",
    )


def _disambig_gender(rng: random.Random) -> Generated:
    female, male = rng.choice(_FEMALE_NAMES), rng.choice(_MALE_NAMES)
    action = rng.choice(_ACTIONS)
    if rng.random() < 0.5:
        sentence = f"{female} met {male} at the cafe because she {action}."
        return _referent_mcq(rng, sentence, f"{female} {action}", f"{male} {action}", "A", "gender_agreement")
    sentence = f"{male} met {female} at the cafe because he {action}."
    return _referent_mcq(rng, sentence, f"{male} {action}", f"{female} {action}", "A", "gender_agreement")


def _disambig_speaker(rng: random.Random) -> Generated:
    first, second = rng.sample(_UNISEX_NAMES, 2)
    obj = rng.choice(_OBJECTS)
    pronoun = rng.choice(("he", "she", "they"))
    sentence = f"{first} told {second} that {pronoun} liked the {obj}."
    return _referent_mcq(
        rng,
        sentence,
        f"{first} liked the {obj}",
        f"{second} liked the {obj}",
        "A",
        "speaker_attitude",
    )


def _disambig_name_role(rng: random.Random) -> Generated:
    name = rng.choice(_UNISEX_NAMES)
    role = rng.choice([role for role, _, _ in _ROLE_SKILLS])
    obj = rng.choice(_OBJECTS)
    pronoun = rng.choice(("he", "she", "they"))
    sentence = f"{name} emailed the {role} because {pronoun} needed the {obj}."
    return _referent_mcq(
        rng,
        sentence,
        f"{name} needed the {obj}",
        f"The {role} needed the {obj}",
        "C",
        "name_role",
    )


def _disambig_they(rng: random.Random) -> Generated:
    first, second = rng.sample([role for role, _, _ in _ROLE_SKILLS], 2)
    action = rng.choice(_ACTIONS)
    sentence = f"The {first} and the {second} talked while they {action}."
    return _referent_mcq(
        rng,
        sentence,
        f"The {first} {action}",
        f"The {second} {action}",
        "C",
        "they_number",
    )


_DISAMBIG_FAMILIES: tuple[Callable[[random.Random], Generated], ...] = (
    _disambig_expertise,
    _disambig_ask_experience,
    _disambig_possessive,
    _disambig_two_referent_action,
    _disambig_role_task,
    _disambig_plural,
    _disambig_gender,
    _disambig_speaker,
    _disambig_name_role,
    _disambig_they,
)


def generate_disambiguation(rng: random.Random, index: int) -> Generated:
    """Cycle the ten families in order, then randomise inside the family."""
    return _DISAMBIG_FAMILIES[index % len(_DISAMBIG_FAMILIES)](rng)


# ---------------------------------------------------------------------------
# Causal judgement
# ---------------------------------------------------------------------------

_CAUSAL_AGENTS = (
    "Maya",
    "Daniel",
    "the gardener",
    "the electrician",
    "the neighbor",
    "the shop owner",
    "the intern",
    "the technician",
    "the cook",
    "the homeowner",
    "the driver",
    "the teacher",
)
_HEAT_SOURCES = ("space heater", "candle", "stove", "soldering iron", "grill", "lamp")
_FLAMMABLES = ("curtains", "paper", "cardboard box", "towel", "wooden shelf", "newspaper")
_ROOMS = ("workshop", "kitchen", "garage", "office", "shed", "living room")
_WATER_SOURCES = ("bathtub", "sink", "hose", "washing machine", "radiator")
_CASUAL_ACTIONS = (
    ("play loud music", "played loud music"),
    ("paint the fence", "painted the fence"),
    ("wash the car", "washed the car"),
    ("host a party", "hosted a party"),
)
_NORM_VIOLATIONS = (
    ("used the manager's coffee mug", "use the manager's coffee mug"),
    ("parked in the reserved space", "park in the reserved space"),
    ("took an extra lunch break", "take an extra lunch break"),
)
_COMMON_CAUSE_EVENTS = (
    (
        "the water pipes froze",
        "the freezing of the water pipes",
        "the bakery closed",
        "the bakery's closure",
        "the temperature dropped to minus twenty degrees",
    ),
    (
        "the road iced over",
        "the road icing over",
        "the school closed",
        "the school's closure",
        "a severe ice storm hit the region",
    ),
    (
        "the river rose",
        "the rise of the river",
        "the town canceled its fireworks display",
        "the cancellation of the fireworks display",
        "three days of heavy rain fell upstream",
    ),
    (
        "a tree fell on Main Street",
        "a tree falling on Main Street",
        "the outdoor concert was canceled",
        "the cancellation of the outdoor concert",
        "a severe thunderstorm hit the town",
    ),
)


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _causal_generated(story: str, question: str, answer: str, family: str) -> Generated:
    prompt = (
        "How would a typical person answer each of the following questions about causation?\n"
        f"{story}\n{question}\nOptions:\n- Yes\n- No"
    )
    return Generated(prompt=prompt, answer=answer, difficulty=family)


def _causal_direct_fire(rng: random.Random) -> Generated:
    agent = rng.choice(_CAUSAL_AGENTS)
    heat = rng.choice(_HEAT_SOURCES)
    flammable = rng.choice(_FLAMMABLES)
    room = rng.choice(_ROOMS)
    story = (
        f"{_cap(agent)} left the {heat} running next to the {flammable} in the {room}. "
        f"A little later, the {flammable} caught fire and the {room} filled with smoke."
    )
    return _causal_generated(
        story,
        f"Did {agent}'s decision to leave the {heat} running cause the fire?",
        "Yes",
        "direct_cause",
    )


def _causal_direct_flood(rng: random.Random) -> Generated:
    agent = rng.choice(_CAUSAL_AGENTS)
    source = rng.choice(_WATER_SOURCES)
    room = rng.choice(_ROOMS)
    story = (
        f"{_cap(agent)} left the {source} running in the {room}. "
        "By the time anyone noticed, water had spread across the floor and damaged the floorboards."
    )
    return _causal_generated(
        story,
        f"Did {agent}'s decision to leave the {source} running cause the water damage?",
        "Yes",
        "direct_cause",
    )


def _causal_common_cause(rng: random.Random) -> Generated:
    story_a, question_a, story_b, question_b, common = rng.choice(_COMMON_CAUSE_EVENTS)
    story = f"{_cap(common)}. Because of this, {story_a}, and also {story_b}."
    return _causal_generated(
        story,
        f"Did {question_a} cause {question_b}?",
        "No",
        "common_cause",
    )


def _causal_preemption(rng: random.Random) -> Generated:
    agent = rng.choice(_CAUSAL_AGENTS)
    story = (
        f"{_cap(agent)} left a candle burning next to the curtains. "
        "The room had an automatic sprinkler, but it had been switched off for maintenance. "
        "The curtains caught fire before anyone could react."
    )
    return _causal_generated(
        story,
        f"Did {agent}'s decision to leave the candle burning cause the curtains to catch fire?",
        "Yes",
        "preemption",
    )


def _causal_omission_duty(rng: random.Random) -> Generated:
    agent = rng.choice(("the building manager", "the safety officer", "the landlord"))
    story = (
        f"{_cap(agent)} was responsible for checking the fire alarm every week. "
        f"{_cap(agent)} skipped the check for three months. "
        "When a small fire started in the storage room, the alarm did not sound and the fire spread unnoticed."
    )
    return _causal_generated(
        story,
        f"Did {agent}'s failure to check the fire alarm cause the fire to spread unnoticed?",
        "Yes",
        "omission_duty",
    )


def _causal_omission_no_duty(rng: random.Random) -> Generated:
    bystander = rng.choice(("a customer", "a passerby", "a visitor"))
    duty_holder = rng.choice(("the electrician", "the building manager", "the maintenance crew"))
    outcome, outcome_infinitive = rng.choice(
        (
            ("the lights went out", "the lights to go out"),
            ("the heating failed", "the heating to fail"),
            ("the elevator stopped", "the elevator to stop"),
        )
    )
    story = (
        f"{_cap(bystander)} noticed that the wiring in the store looked old, but {bystander} was not responsible "
        f"for maintenance. {_cap(duty_holder)} had inspected the wiring the week before and found it safe. "
        f"That evening, a city-wide power failure caused {outcome_infinitive}."
    )
    return _causal_generated(
        story,
        f"Did {bystander}'s failure to report the wiring cause {outcome_infinitive}?",
        "No",
        "omission_no_duty",
    )


def _causal_counterfactual_chain(rng: random.Random) -> Generated:
    agent = rng.choice(_CAUSAL_AGENTS)
    story = (
        f"{_cap(agent)} poured the used oil down the kitchen sink. "
        "Over the next few days, the oil solidified and the drain clogged. "
        "When the dishwasher ran, the water had nowhere to go and the kitchen flooded."
    )
    return _causal_generated(
        story,
        f"Did {agent}'s decision to pour oil down the sink cause the kitchen to flood?",
        "Yes",
        "counterfactual_chain",
    )


def _causal_correlation(rng: random.Random) -> Generated:
    agent = rng.choice(("the neighbor", "the shop owner", "the driver"))
    action, past_action = rng.choice(_CASUAL_ACTIONS)
    outcome, outcome_infinitive = rng.choice(
        (
            ("the street lights failed", "the street lights to fail"),
            ("the internet went down", "the internet to go down"),
            ("the elevator stopped", "the elevator to stop"),
        )
    )
    other_cause = rng.choice(
        (
            "a transformer exploded",
            "a cable was cut by construction workers",
            "a scheduled maintenance outage began",
        )
    )
    story = (
        f"{_cap(agent)} {past_action} on Saturday afternoon. At about the same time, {other_cause} and "
        f"{outcome}. There was no connection between the two events."
    )
    return _causal_generated(
        story,
        f"Did {agent}'s decision to {action} cause {outcome_infinitive}?",
        "No",
        "correlation_no_mechanism",
    )


def _causal_norm_no_cause(rng: random.Random) -> Generated:
    agent = rng.choice(("the intern", "the new employee", "the volunteer"))
    past_violation, base_violation = rng.choice(_NORM_VIOLATIONS)
    outcome, outcome_infinitive = rng.choice(
        (
            ("the meeting was canceled", "the meeting to be canceled"),
            ("the presentation was postponed", "the presentation to be postponed"),
            ("the client canceled the order", "the client to cancel the order"),
        )
    )
    other_cause = rng.choice(
        (
            "the client's flight was delayed",
            "a key presenter called in sick",
            "a severe storm closed the airport",
        )
    )
    story = (
        f"{_cap(agent)} {past_violation} without asking. That same morning, {other_cause}, and {outcome}."
    )
    return _causal_generated(
        story,
        f"Did {agent}'s decision to {base_violation} cause {outcome_infinitive}?",
        "No",
        "norm_violation_no_cause",
    )


def _causal_norm_cause(rng: random.Random) -> Generated:
    agent = rng.choice(("the technician", "the operator", "the supervisor"))
    story = (
        f"{_cap(agent)} skipped the required safety check on the machine. "
        "The machine's guard was loose, and later that day it came off while a worker was operating it, "
        "injuring the worker's hand."
    )
    return _causal_generated(
        story,
        f"Did {agent}'s decision to skip the safety check cause the worker's injury?",
        "Yes",
        "norm_violation_cause",
    )


def _causal_double_prevention(rng: random.Random) -> Generated:
    agent = rng.choice(("the cook", "the kitchen assistant", "the restaurant owner"))
    story = (
        f"{_cap(agent)} left the fryer unattended. The fryer had an automatic shut-off, but the sensor was "
        "covered with grease and did not trigger. The oil overheated and the kitchen caught fire."
    )
    return _causal_generated(
        story,
        f"Did {agent}'s decision to leave the fryer unattended cause the kitchen fire?",
        "Yes",
        "failed_prevention",
    )


def _causal_two_candidates(rng: random.Random) -> Generated:
    agent_one = rng.choice(("the plumber", "the electrician", "the contractor"))
    agent_two = rng.choice(("the homeowner", "the tenant", "the neighbor"))
    action = rng.choice(("tightened the valve", "replaced the switch", "sealed the window"))
    outcome, outcome_infinitive = rng.choice(
        (
            ("the pipe burst", "the pipe to burst"),
            ("the lights flickered", "the lights to flicker"),
            ("the window leaked", "the window to leak"),
        )
    )
    cause = rng.choice(
        (
            "a pressure surge from the city main",
            "a faulty breaker",
            "a severe rainstorm",
        )
    )
    story = (
        f"{_cap(agent_one)} {action}. Later the same day, {_cap(agent_two)} painted the wall nearby. "
        f"The paint was cosmetic and did not affect the system. That night, {outcome} because of {cause}."
    )
    return _causal_generated(
        story,
        f"Did {agent_one}'s work cause {outcome_infinitive}?",
        "No",
        "alternative_cause",
    )


_CAUSAL_FAMILIES: tuple[Callable[[random.Random], Generated], ...] = (
    _causal_direct_fire,
    _causal_direct_flood,
    _causal_common_cause,
    _causal_preemption,
    _causal_omission_duty,
    _causal_omission_no_duty,
    _causal_counterfactual_chain,
    _causal_correlation,
    _causal_norm_no_cause,
    _causal_norm_cause,
    _causal_double_prevention,
    _causal_two_candidates,
)


def generate_causal(rng: random.Random, index: int) -> Generated:
    """Cycle the twelve vignette families in order."""
    return _CAUSAL_FAMILIES[index % len(_CAUSAL_FAMILIES)](rng)


# ---------------------------------------------------------------------------
# Movie recommendation (MovieLens latest-small)
# ---------------------------------------------------------------------------


@dataclass
class MovieLens:
    """The slice of MovieLens the generator needs."""

    titles: dict[int, str]
    genres: dict[int, set[str]]
    counts: dict[int, int]
    liked: dict[int, list[int]]
    disliked: dict[int, list[int]]


def _find_movielens_files(raw_dir: str) -> tuple[str, str] | None:
    for base in (raw_dir, os.path.join(raw_dir, MOVIELENS_DIRNAME), os.path.join(raw_dir, "movielens")):
        movies = os.path.join(base, "movies.csv")
        ratings = os.path.join(base, "ratings.csv")
        if os.path.isfile(movies) and os.path.isfile(ratings):
            return movies, ratings
    return None


def _download_movielens(raw_dir: str) -> None:
    os.makedirs(raw_dir, exist_ok=True)
    zip_path = os.path.join(raw_dir, "ml-latest-small.zip")
    if not os.path.exists(zip_path):
        print(f"[bbh_adapter] downloading {MOVIELENS_URL} -> {zip_path}", file=sys.stderr)
        try:
            urllib.request.urlretrieve(MOVIELENS_URL, zip_path)
        except Exception as exc:  # noqa: BLE001 - re-raised with the fix
            raise RuntimeError(
                f"failed to download MovieLens from {MOVIELENS_URL} ({exc}). "
                "Pass --movielens-dir pointing at an existing ml-latest-small directory "
                "(movies.csv + ratings.csv), or place ml-latest-small.zip under the raw dir."
            ) from exc
    print(f"[bbh_adapter] extracting {zip_path}", file=sys.stderr)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(raw_dir)


def _clean_title(title: str) -> str:
    """Drop the release year (the BIG-Bench prompts use bare titles)."""
    return re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()


def load_movielens(
    raw_dir: str,
    *,
    download: bool = True,
    min_movie_ratings: int = 20,
) -> MovieLens:
    """Load latest-small and keep only the users/movies the generator can use.

    Ratings are binarised the way the BIG-Bench task describes: ``>= 4`` is
    liked, ``<= 2`` is disliked (the task itself uses 3/2; 4/2 gives a cleaner
    preference signal).  Movies with fewer than ``min_movie_ratings`` ratings are
    dropped so a query cannot hinge on an obscure title, and duplicate titles
    (remakes) keep the most-rated movie.
    """
    files = _find_movielens_files(raw_dir)
    if files is None and download:
        _download_movielens(raw_dir)
        files = _find_movielens_files(raw_dir)
    if files is None:
        raise FileNotFoundError(
            f"MovieLens latest-small not found under {raw_dir}; pass --movielens-dir or allow the download"
        )
    movies_path, ratings_path = files

    titles: dict[int, str] = {}
    genres: dict[int, set[str]] = {}
    with open(movies_path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                movie_id = int(row["movieId"])
            except (KeyError, TypeError, ValueError):
                continue
            title = _clean_title(row.get("title", ""))
            genre_set = {genre.strip() for genre in row.get("genres", "").split("|") if genre.strip()}
            if not title or "," in title or genre_set == {"(no genres listed)"}:
                continue
            titles[movie_id] = title
            genres[movie_id] = genre_set

    counts: Counter[int] = Counter()
    per_user: dict[int, dict[str, list[int]]] = defaultdict(lambda: {"liked": [], "disliked": []})
    with open(ratings_path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                movie_id = int(row["movieId"])
                user_id = int(row["userId"])
                rating = float(row["rating"])
            except (KeyError, TypeError, ValueError):
                continue
            if movie_id not in titles:
                continue
            counts[movie_id] += 1
            if rating >= 4.0:
                per_user[user_id]["liked"].append(movie_id)
            elif rating <= 2.0:
                per_user[user_id]["disliked"].append(movie_id)

    kept = {movie_id for movie_id, count in counts.items() if count >= min_movie_ratings}
    by_title: dict[str, int] = {}
    for movie_id in kept:
        title = titles[movie_id]
        if title not in by_title or counts[movie_id] > counts[by_title[title]]:
            by_title[title] = movie_id
    kept = set(by_title.values())

    liked = {
        user: [movie_id for movie_id in data["liked"] if movie_id in kept]
        for user, data in per_user.items()
    }
    disliked = {
        user: [movie_id for movie_id in data["disliked"] if movie_id in kept]
        for user, data in per_user.items()
    }
    liked = {user: movies for user, movies in liked.items() if len(movies) >= 8}
    disliked = {user: movies for user, movies in disliked.items() if len(movies) >= 4}
    users = set(liked) & set(disliked)
    liked = {user: liked[user] for user in users}
    disliked = {user: disliked[user] for user in users}
    if not users:
        raise ValueError(
            "MovieLens slice has no user with at least 8 liked and 4 disliked movies; "
            "lower --min-movie-ratings or check the raw files"
        )

    return MovieLens(
        titles={movie_id: titles[movie_id] for movie_id in kept},
        genres={movie_id: genres[movie_id] for movie_id in kept},
        counts={movie_id: counts[movie_id] for movie_id in kept},
        liked=liked,
        disliked=disliked,
    )


def generate_movie(rng: random.Random, index: int, movielens: MovieLens | None) -> Generated:
    """A four-option recommendation query built from one user's ratings."""
    del index
    if movielens is None:
        raise RuntimeError("movie_recommendation requires MovieLens data")
    users = sorted(movielens.liked)
    for _ in range(200):
        user = rng.choice(users)
        liked = movielens.liked[user]
        disliked = movielens.disliked[user]
        seeds = rng.sample(liked, 4)
        gold_candidates = [movie_id for movie_id in liked if movie_id not in seeds]
        if not gold_candidates:
            continue
        gold = rng.choice(gold_candidates)
        seed_genres = set().union(*(movielens.genres[movie_id] for movie_id in seeds))
        gold_genres = movielens.genres[gold]
        shared = sum(1 for movie_id in seeds if movielens.genres[movie_id] & gold_genres)
        strict = rng.random() < 0.7
        if strict and shared < 2:
            continue

        gold_count = movielens.counts[gold]
        band = [
            movie_id
            for movie_id in disliked
            if 0.2 * gold_count <= movielens.counts[movie_id] <= 5.0 * gold_count
        ]
        if len(band) < 3:
            band = list(disliked)
        distractors = rng.sample(band, 3)
        if strict:
            thin_overlap = sum(1 for movie_id in distractors if len(movielens.genres[movie_id] & seed_genres) <= 1)
            if thin_overlap < 2:
                continue

        choices = [gold, *distractors]
        if len({movielens.titles[movie_id] for movie_id in choices}) < 4:
            continue

        rng.shuffle(choices)
        options = [
            {"id": chr(ord("A") + i), "text": movielens.titles[movie_id]}
            for i, movie_id in enumerate(choices)
        ]
        gold_title = movielens.titles[gold]
        correct = next(option["id"] for option in options if option["text"] == gold_title)
        seed_titles = [movielens.titles[movie_id] for movie_id in seeds]
        prompt = (
            "Find a movie similar to " + ", ".join(seed_titles) + ":\nOptions:\n"
            + "\n".join(f"({option['id']}) {option['text']}" for option in options)
        )
        return Generated(
            prompt=prompt,
            answer=correct,
            difficulty=f"shared_genres{shared}",
            options=options,
            correct_option_id=correct,
        )
    raise RuntimeError("could not generate a movie query with four distinct options")


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def validate_generated(task: str, generated: Generated) -> str | None:
    """Return a human-readable problem for a malformed row, or None.

    The geometry check is the important one: it re-derives the label from the
    rendered SVG path with ``classify_path`` instead of trusting the generator's
    own choice.
    """
    if not generated.prompt.strip():
        return "empty prompt"
    if task == TASK_DYCK:
        if "Input: " not in generated.prompt:
            return "dyck prompt has no Input block"
        prefix = generated.prompt.split("Input: ", 1)[1].split()
        suffix = generated.answer.split()
        if not suffix or any(token not in _DYCK_CLOSERS for token in suffix):
            return "dyck answer is not a closing-bracket sequence"
        if not _valid_prefix(prefix) or not _is_balanced(prefix + suffix):
            return "dyck prefix+answer is not a balanced Dyck word"
        if _is_balanced(prefix):
            return "dyck prefix is already balanced"
        return None
    if task == TASK_GEOMETRIC:
        match = re.search(r'<path d="([^"]+)"/>', generated.prompt)
        if match is None:
            return "geometric prompt has no SVG path"
        if classify_path(match.group(1)) is None:
            return "SVG path does not classify to a known shape"
        if generated.correct_option_id not in {option["id"] for option in generated.options}:
            return "correct_option_id is not among the options"
        label = next(option["text"] for option in generated.options if option["id"] == generated.correct_option_id)
        if classify_path(match.group(1)) != label:
            return f"SVG path classifies as {classify_path(match.group(1))}, options say {label}"
        return None
    if task == TASK_DISAMBIGUATION:
        if len(generated.options) != 3 or generated.options[-1]["text"] != "Ambiguous":
            return "disambiguation row must have A/B + Ambiguous options"
        if generated.correct_option_id not in {"A", "B", "C"}:
            return "disambiguation gold must be A/B/C"
        return None
    if task == TASK_CAUSAL:
        if generated.answer not in {"Yes", "No"}:
            return "causal gold must be Yes/No"
        return None
    if task == TASK_MOVIE:
        if len(generated.options) != 4:
            return "movie row must have four options"
        if len({option["text"] for option in generated.options}) != 4:
            return "movie options are not distinct"
        if generated.correct_option_id not in {option["id"] for option in generated.options}:
            return "correct_option_id is not among the options"
        return None
    return f"unknown task {task!r}"


# ---------------------------------------------------------------------------
# build + CLI
# ---------------------------------------------------------------------------


def _parse_csv_list(spec: str, allowed: tuple[str, ...], label: str) -> list[str]:
    items = [item.strip() for item in spec.split(",") if item.strip()]
    unknown = [item for item in items if item not in allowed]
    if unknown:
        raise ValueError(f"unknown {label}: {unknown}; expected a subset of {list(allowed)}")
    return items


def _parse_quota(spec: str, tasks: list[str]) -> dict[str, int]:
    if not spec:
        return {task: DEFAULT_QUOTA[task] for task in tasks}
    quota: dict[str, int] = {}
    for item in spec.split(","):
        name, _, value = item.partition("=")
        name = name.strip()
        if name not in TASKS:
            raise ValueError(f"unknown --quota task {name!r}")
        quota[name] = int(value)
    for task in tasks:
        quota.setdefault(task, DEFAULT_QUOTA[task])
    return quota


def build_task_rows(
    task: str,
    quota: int,
    seed: int,
    *,
    movielens: MovieLens | None,
    excluded: set[str],
    seen: set[str],
    skip_invalid: bool = False,
) -> tuple[list[dict], dict]:
    """Generate up to ``quota`` unique, valid rows for one task."""
    rng = random.Random(f"bbh:{seed}:{task}")
    rows: list[dict] = []
    invalid = 0
    duplicates = 0
    already_trained = 0
    attempts = 0
    limit = max(quota * 40, 1000)
    while len(rows) < quota and attempts < limit:
        attempts += 1
        try:
            if task == TASK_MOVIE:
                generated = generate_movie(rng, attempts, movielens)
            elif task == TASK_DYCK:
                generated = generate_dyck(rng, attempts)
            elif task == TASK_GEOMETRIC:
                generated = generate_geometric(rng, attempts)
            elif task == TASK_DISAMBIGUATION:
                generated = generate_disambiguation(rng, attempts)
            elif task == TASK_CAUSAL:
                generated = generate_causal(rng, attempts)
            else:  # pragma: no cover - guarded by the CLI
                raise ValueError(f"unknown task {task!r}")
        except RuntimeError:
            break
        problem = validate_generated(task, generated)
        if problem is not None:
            invalid += 1
            if skip_invalid:
                continue
            raise ValueError(f"{task}: generated an invalid row: {problem}")
        row = make_row(task, generated, len(rows), seed)
        key = prompt_key(row["prompt"])
        if key in excluded:
            already_trained += 1
            continue
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        rows.append(row)

    report = {
        "requested": quota,
        "generated": len(rows),
        "attempts": attempts,
        "invalid": invalid,
        "duplicates": duplicates,
        "already_trained": already_trained,
        "difficulty": dict(Counter(row["extra_info"]["difficulty"] for row in rows)),
    }
    return rows, report


def summarise(rows: list[dict]) -> dict:
    by_task = Counter(row["extra_info"]["bbh_task"] for row in rows)
    options = Counter(len(row["extra_info"].get("options", [])) for row in rows)
    return {
        "rows": len(rows),
        "by_task": dict(sorted(by_task.items())),
        "options_per_row": dict(sorted(options.items())),
        "prompt_chars": {
            "min": min((len(row["prompt"][0]["content"]) for row in rows), default=0),
            "max": max((len(row["prompt"][0]["content"]) for row in rows), default=0),
        },
    }


def report_path_for(out: str) -> str:
    return os.path.splitext(out)[0] + "_report.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR, help="raw dir; MovieLens is downloaded/extracted here")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output parquet path")
    parser.add_argument(
        "--tasks",
        default=",".join(TASKS),
        help="comma-separated subset of " + ", ".join(TASKS),
    )
    parser.add_argument(
        "--quota",
        default="",
        help="comma-separated task=count overrides (e.g. dyck_languages=6000); omitted tasks use DEFAULT_QUOTA",
    )
    parser.add_argument("--seed", type=int, default=44, help="generator seed (stage 4 uses 44)")
    parser.add_argument(
        "--exclude-path",
        action="append",
        default=None,
        help="previously trained parquet/file/dir; prompts already seen there are dropped (repeatable)",
    )
    parser.add_argument(
        "--movielens-dir",
        default=None,
        help="local MovieLens latest-small directory (containing movies.csv and ratings.csv); "
        "defaults to --raw-dir and downloads the zip when missing",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="never download MovieLens; fail if the CSVs are missing",
    )
    parser.add_argument(
        "--min-movie-ratings",
        type=int,
        default=20,
        help="drop movies with fewer ratings than this (default 20)",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="drop a row that fails validate_generated instead of aborting the build",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    tasks = _parse_csv_list(args.tasks, TASKS, "--tasks")
    if not tasks:
        parser.error("--tasks is empty")
    quota = _parse_quota(args.quota, tasks)

    movielens: MovieLens | None = None
    if TASK_MOVIE in tasks:
        movielens_dir = os.path.expanduser(args.movielens_dir or args.raw_dir)
        movielens = load_movielens(
            movielens_dir,
            download=not args.no_download,
            min_movie_ratings=args.min_movie_ratings,
        )
        print(
            f"[bbh_adapter] MovieLens: {len(movielens.titles)} movies, "
            f"{len(movielens.liked)} users with >=8 liked / >=4 disliked",
            file=sys.stderr,
        )

    excluded = read_excluded_keys(args.exclude_path)
    rows: list[dict] = []
    task_reports: dict[str, dict] = {}
    seen: set[str] = set()
    for task in tasks:
        task_rows, report = build_task_rows(
            task,
            quota.get(task, DEFAULT_QUOTA[task]),
            args.seed,
            movielens=movielens,
            excluded=excluded,
            seen=seen,
            skip_invalid=args.skip_invalid,
        )
        rows.extend(task_rows)
        task_reports[task] = report
        print(f"[bbh_adapter] {task}: {report['generated']}/{report['requested']} rows", file=sys.stderr)
        if report["generated"] < report["requested"]:
            print(
                f"[bbh_adapter] WARNING: {task} supplied {report['generated']} of "
                f"{report['requested']} requested rows",
                file=sys.stderr,
            )

    if not rows:
        print("no rows survived generation; nothing written", file=sys.stderr)
        return 1

    schema.normalise_extra_info(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    schema.write_rows_parquet(rows, args.out)

    report = {
        "config": {
            "tasks": tasks,
            "quota": quota,
            "seed": args.seed,
            "raw_dir": os.path.expanduser(args.raw_dir),
            "movielens_dir": os.path.expanduser(args.movielens_dir) if args.movielens_dir else None,
            "min_movie_ratings": args.min_movie_ratings,
            "exclude_paths": args.exclude_path or [],
        },
        "tasks": task_reports,
        "summary": summarise(rows),
    }
    report_path = report_path_for(args.out)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"rows written: {len(rows)} -> {args.out}")
    print(f"  by task: {report['summary']['by_task']}")
    print(f"report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
